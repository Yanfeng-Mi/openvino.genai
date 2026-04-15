// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <openvino/openvino.hpp>
#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/float16.hpp>

#include "imwrite.hpp"
#include "openvino/genai/chat_history.hpp"
#include "openvino/genai/json_container.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "safetensors_utils/quantization_utils.hpp"
#include "safetensors_utils/safetensors_loader.hpp"
#include "safetensors_utils/safetensors_weight_finalizer.hpp"
#include "safetensors_utils/safetensors_weight_source.hpp"

#include "modeling/builder_context.hpp"
#include "modeling/models/qwen3/modeling_qwen3.hpp"
#include "modeling/models/zimage/modeling_zimage_dit.hpp"
#include "modeling/models/zimage/modeling_zimage_vae.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace {

constexpr int32_t kSeqMultiOf = 32;
constexpr size_t kDefaultMaxSeqLen = 512;

struct DiTRopeConfig {
    std::vector<int32_t> axes_dims = {32, 48, 48};
    std::vector<int32_t> axes_lens = {1536, 512, 512};
    float rope_theta = 256.0f;
};

struct SchedulerConfig {
    int32_t num_train_timesteps = 1000;
    float shift = 1.0f;
    bool use_dynamic_shifting = false;
    int32_t base_image_seq_len = 256;
    int32_t max_image_seq_len = 4096;
    float base_shift = 0.5f;
    float max_shift = 1.15f;
};

struct ImageTokensMeta {
    int32_t channels = 0;
    int32_t frames = 1;
    int32_t height_latent = 0;
    int32_t width_latent = 0;
    int32_t patch_size = 1;
    int32_t f_patch_size = 1;
    int32_t patch_dim = 0;
    int32_t f_tokens = 1;
    int32_t h_tokens = 0;
    int32_t w_tokens = 0;
    size_t image_len = 0;
    size_t pad_len = 0;
    size_t padded_len = 0;
};

struct PromptContext {
    ov::Tensor cap_feats;
    ov::Tensor cap_mask;
    ov::Tensor cap_rope_cos;
    ov::Tensor cap_rope_sin;
    ov::Tensor x_rope_cos;
    ov::Tensor x_rope_sin;
    size_t cap_len = 0;
    size_t cap_len_padded = 0;
};

struct DumpContext {
    bool enabled = false;
    std::filesystem::path dir;
};

nlohmann::json load_json(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open JSON file: " + path.string());
    }
    return nlohmann::json::parse(file);
}

template <typename T>
void read_json_value(const nlohmann::json& data, const char* key, T& out) {
    if (data.contains(key)) {
        out = data.at(key).get<T>();
    }
}

std::vector<int32_t> read_int_array(const nlohmann::json& data, const char* key) {
    std::vector<int32_t> values;
    if (!data.contains(key)) {
        return values;
    }
    for (const auto& item : data.at(key)) {
        values.push_back(item.get<int32_t>());
    }
    return values;
}

ov::genai::modeling::models::Qwen3DenseConfig read_qwen3_config(const std::filesystem::path& config_path) {
    auto data = load_json(config_path);
    ov::genai::modeling::models::Qwen3DenseConfig cfg;
    read_json_value(data, "hidden_size", cfg.hidden_size);
    read_json_value(data, "num_attention_heads", cfg.num_attention_heads);
    read_json_value(data, "num_key_value_heads", cfg.num_key_value_heads);
    read_json_value(data, "head_dim", cfg.head_dim);
    read_json_value(data, "intermediate_size", cfg.intermediate_size);
    read_json_value(data, "num_hidden_layers", cfg.num_hidden_layers);
    read_json_value(data, "rms_norm_eps", cfg.rms_norm_eps);
    read_json_value(data, "rope_theta", cfg.rope_theta);
    read_json_value(data, "hidden_act", cfg.hidden_act);
    read_json_value(data, "attention_bias", cfg.attention_bias);
    read_json_value(data, "tie_word_embeddings", cfg.tie_word_embeddings);
    return cfg;
}

ov::genai::modeling::models::ZImageDiTConfig read_dit_config(const std::filesystem::path& config_path,
                                                             DiTRopeConfig& rope_cfg) {
    auto data = load_json(config_path);
    ov::genai::modeling::models::ZImageDiTConfig cfg;
    read_json_value(data, "dim", cfg.dim);
    read_json_value(data, "n_layers", cfg.n_layers);
    read_json_value(data, "n_refiner_layers", cfg.n_refiner_layers);
    read_json_value(data, "n_heads", cfg.n_heads);
    read_json_value(data, "n_kv_heads", cfg.n_kv_heads);
    read_json_value(data, "in_channels", cfg.in_channels);
    read_json_value(data, "cap_feat_dim", cfg.cap_feat_dim);
    read_json_value(data, "norm_eps", cfg.norm_eps);
    read_json_value(data, "qk_norm", cfg.qk_norm);
    read_json_value(data, "t_scale", cfg.t_scale);

    if (data.contains("all_patch_size") && data["all_patch_size"].is_array() && !data["all_patch_size"].empty()) {
        cfg.patch_size = data["all_patch_size"][0].get<int32_t>();
    }
    if (data.contains("all_f_patch_size") && data["all_f_patch_size"].is_array() && !data["all_f_patch_size"].empty()) {
        cfg.f_patch_size = data["all_f_patch_size"][0].get<int32_t>();
    }

    auto axes_dims = read_int_array(data, "axes_dims");
    auto axes_lens = read_int_array(data, "axes_lens");
    if (!axes_dims.empty()) {
        rope_cfg.axes_dims = std::move(axes_dims);
    }
    if (!axes_lens.empty()) {
        rope_cfg.axes_lens = std::move(axes_lens);
    }
    read_json_value(data, "rope_theta", rope_cfg.rope_theta);
    return cfg;
}

ov::genai::modeling::models::ZImageVAEConfig read_vae_config(const std::filesystem::path& config_path) {
    auto data = load_json(config_path);
    ov::genai::modeling::models::ZImageVAEConfig cfg;
    read_json_value(data, "in_channels", cfg.in_channels);
    read_json_value(data, "out_channels", cfg.out_channels);
    read_json_value(data, "latent_channels", cfg.latent_channels);
    read_json_value(data, "layers_per_block", cfg.layers_per_block);
    read_json_value(data, "norm_num_groups", cfg.norm_num_groups);
    read_json_value(data, "mid_block_add_attention", cfg.mid_block_add_attention);
    read_json_value(data, "scaling_factor", cfg.scaling_factor);
    read_json_value(data, "shift_factor", cfg.shift_factor);

    if (data.contains("block_out_channels")) {
        cfg.block_out_channels = read_int_array(data, "block_out_channels");
    }
    return cfg;
}

SchedulerConfig read_scheduler_config(const std::filesystem::path& config_path) {
    SchedulerConfig cfg;
    if (!std::filesystem::exists(config_path)) {
        return cfg;
    }
    auto data = load_json(config_path);
    read_json_value(data, "num_train_timesteps", cfg.num_train_timesteps);
    read_json_value(data, "shift", cfg.shift);
    read_json_value(data, "use_dynamic_shifting", cfg.use_dynamic_shifting);
    return cfg;
}

float calculate_shift(size_t image_seq_len, const SchedulerConfig& cfg) {
    const float m = (cfg.max_shift - cfg.base_shift) /
                    static_cast<float>(cfg.max_image_seq_len - cfg.base_image_seq_len);
    const float b = cfg.base_shift - m * static_cast<float>(cfg.base_image_seq_len);
    return m * static_cast<float>(image_seq_len) + b;
}

class FlowMatchEulerDiscreteScheduler {
public:
    explicit FlowMatchEulerDiscreteScheduler(const SchedulerConfig& cfg)
        : cfg_(cfg) {
        init_training_sigmas();
    }

    void set_timesteps(int32_t num_inference_steps, float mu) {
        if (num_inference_steps <= 0) {
            throw std::runtime_error("num_inference_steps must be > 0");
        }
        num_inference_steps_ = num_inference_steps;

        const float start_t = sigma_max_ * static_cast<float>(cfg_.num_train_timesteps);
        const float end_t = sigma_min_ * static_cast<float>(cfg_.num_train_timesteps);

        timesteps_.resize(static_cast<size_t>(num_inference_steps_));
        for (int32_t i = 0; i < num_inference_steps_; ++i) {
            const float frac = static_cast<float>(i) / static_cast<float>(num_inference_steps_);
            timesteps_[static_cast<size_t>(i)] = start_t + (end_t - start_t) * frac;
        }

        sigmas_.resize(static_cast<size_t>(num_inference_steps_) + 1);
        for (int32_t i = 0; i < num_inference_steps_; ++i) {
            float sigma = timesteps_[static_cast<size_t>(i)] / static_cast<float>(cfg_.num_train_timesteps);
            if (cfg_.use_dynamic_shifting) {
                sigma = time_shift(mu, 1.0f, sigma);
            } else {
                sigma = cfg_.shift * sigma / (1.0f + (cfg_.shift - 1.0f) * sigma);
            }
            sigmas_[static_cast<size_t>(i)] = sigma;
        }
        sigmas_.back() = 0.0f;

        for (int32_t i = 0; i < num_inference_steps_; ++i) {
            timesteps_[static_cast<size_t>(i)] = sigmas_[static_cast<size_t>(i)] *
                                                 static_cast<float>(cfg_.num_train_timesteps);
        }
    }

    const std::vector<float>& timesteps() const {
        return timesteps_;
    }

    const std::vector<float>& sigmas() const {
        return sigmas_;
    }

private:
    void init_training_sigmas() {
        const int32_t n = cfg_.num_train_timesteps;
        std::vector<float> sigmas;
        sigmas.reserve(static_cast<size_t>(n));
        for (int32_t i = 0; i < n; ++i) {
            const float t = static_cast<float>(n - i);
            float sigma = t / static_cast<float>(n);
            if (!cfg_.use_dynamic_shifting) {
                sigma = cfg_.shift * sigma / (1.0f + (cfg_.shift - 1.0f) * sigma);
            }
            sigmas.push_back(sigma);
        }
        sigma_max_ = sigmas.front();
        sigma_min_ = sigmas.back();
    }

    static float time_shift(float mu, float sigma, float t) {
        const float exp_mu = std::exp(mu);
        const float base = std::pow((1.0f / t - 1.0f), sigma);
        return exp_mu / (exp_mu + base);
    }

    SchedulerConfig cfg_;
    int32_t num_inference_steps_ = 0;
    float sigma_min_ = 0.0f;
    float sigma_max_ = 1.0f;
    std::vector<float> timesteps_;
    std::vector<float> sigmas_;
};
ImageTokensMeta build_image_meta(int32_t channels,
                                 int32_t frames,
                                 int32_t height_latent,
                                 int32_t width_latent,
                                 int32_t patch_size,
                                 int32_t f_patch_size) {
    ImageTokensMeta meta;
    meta.channels = channels;
    meta.frames = frames;
    meta.height_latent = height_latent;
    meta.width_latent = width_latent;
    meta.patch_size = patch_size;
    meta.f_patch_size = f_patch_size;
    meta.patch_dim = patch_size * patch_size * f_patch_size * channels;

    if (frames % f_patch_size != 0 || height_latent % patch_size != 0 || width_latent % patch_size != 0) {
        throw std::runtime_error("Latent dimensions are not divisible by patch sizes");
    }

    meta.f_tokens = frames / f_patch_size;
    meta.h_tokens = height_latent / patch_size;
    meta.w_tokens = width_latent / patch_size;

    meta.image_len = static_cast<size_t>(meta.f_tokens) * meta.h_tokens * meta.w_tokens;
    meta.pad_len = (kSeqMultiOf - (meta.image_len % kSeqMultiOf)) % kSeqMultiOf;
    meta.padded_len = meta.image_len + meta.pad_len;
    return meta;
}

std::vector<size_t> mask_indices(const ov::Tensor& mask) {
    const auto shape = mask.get_shape();
    if (shape.size() != 2 || shape[0] != 1) {
        throw std::runtime_error("attention_mask must have shape [1, seq_len]");
    }
    const size_t seq_len = shape[1];
    std::vector<size_t> indices;
    indices.reserve(seq_len);

    if (mask.get_element_type() == ov::element::i64) {
        const auto* data = mask.data<const int64_t>();
        for (size_t i = 0; i < seq_len; ++i) {
            if (data[i] != 0) {
                indices.push_back(i);
            }
        }
        return indices;
    }
    if (mask.get_element_type() == ov::element::i32) {
        const auto* data = mask.data<const int32_t>();
        for (size_t i = 0; i < seq_len; ++i) {
            if (data[i] != 0) {
                indices.push_back(i);
            }
        }
        return indices;
    }
    if (mask.get_element_type() == ov::element::boolean) {
        const auto* data = mask.data<const char>();
        for (size_t i = 0; i < seq_len; ++i) {
            if (data[i] != 0) {
                indices.push_back(i);
            }
        }
        return indices;
    }
    throw std::runtime_error("Unsupported attention_mask dtype");
}

template <typename SrcT>
void copy_rows_to_float_impl(const SrcT* src,
                             const std::vector<size_t>& indices,
                             size_t row_size,
                             std::vector<float>& out) {
    out.resize(indices.size() * row_size);
    for (size_t i = 0; i < indices.size(); ++i) {
        const SrcT* row = src + indices[i] * row_size;
        float* dst = out.data() + i * row_size;
        for (size_t j = 0; j < row_size; ++j) {
            dst[j] = static_cast<float>(row[j]);
        }
    }
}

std::vector<float> copy_rows_to_float(const ov::Tensor& src,
                                      const std::vector<size_t>& indices,
                                      size_t row_size) {
    if (src.get_element_type() == ov::element::f32) {
        std::vector<float> out(indices.size() * row_size);
        const auto* data = src.data<const float>();
        for (size_t i = 0; i < indices.size(); ++i) {
            const float* row = data + indices[i] * row_size;
            std::memcpy(out.data() + i * row_size, row, row_size * sizeof(float));
        }
        return out;
    }
    if (src.get_element_type() == ov::element::f16) {
        std::vector<float> out;
        copy_rows_to_float_impl(src.data<const ov::float16>(), indices, row_size, out);
        return out;
    }
    if (src.get_element_type() == ov::element::bf16) {
        std::vector<float> out;
        copy_rows_to_float_impl(src.data<const ov::bfloat16>(), indices, row_size, out);
        return out;
    }
    throw std::runtime_error("Unsupported hidden_states dtype");
}

std::vector<float> tensor_to_float_vector(const ov::Tensor& src) {
    const size_t count = src.get_size();
    std::vector<float> out(count, 0.0f);
    if (count == 0) {
        return out;
    }
    const auto type = src.get_element_type();
    if (type == ov::element::f32) {
        std::memcpy(out.data(), src.data<const float>(), count * sizeof(float));
        return out;
    }
    if (type == ov::element::f16) {
        const auto* data = src.data<const ov::float16>();
        for (size_t i = 0; i < count; ++i) {
            out[i] = static_cast<float>(data[i]);
        }
        return out;
    }
    if (type == ov::element::bf16) {
        const auto* data = src.data<const ov::bfloat16>();
        for (size_t i = 0; i < count; ++i) {
            out[i] = static_cast<float>(data[i]);
        }
        return out;
    }
    throw std::runtime_error("Unsupported tensor dtype for conversion");
}

std::vector<int64_t> tensor_to_int64_vector(const ov::Tensor& src) {
    const size_t count = src.get_size();
    std::vector<int64_t> out(count, 0);
    if (count == 0) {
        return out;
    }
    const auto type = src.get_element_type();
    if (type == ov::element::i64) {
        std::memcpy(out.data(), src.data<const int64_t>(), count * sizeof(int64_t));
        return out;
    }
    if (type == ov::element::i32) {
        const auto* data = src.data<const int32_t>();
        for (size_t i = 0; i < count; ++i) {
            out[i] = static_cast<int64_t>(data[i]);
        }
        return out;
    }
    if (type == ov::element::boolean) {
        const auto* data = src.data<const char>();
        for (size_t i = 0; i < count; ++i) {
            out[i] = data[i] ? 1 : 0;
        }
        return out;
    }
    throw std::runtime_error("Unsupported tensor dtype for int64 conversion");
}

std::string element_type_to_string(const ov::element::Type& type) {
    if (type == ov::element::f32) {
        return "f32";
    }
    if (type == ov::element::f16) {
        return "f16";
    }
    if (type == ov::element::bf16) {
        return "bf16";
    }
    if (type == ov::element::i64) {
        return "i64";
    }
    if (type == ov::element::i32) {
        return "i32";
    }
    if (type == ov::element::boolean) {
        return "bool";
    }
    return "unknown";
}

void write_json_file(const std::filesystem::path& path, const nlohmann::json& data) {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to write JSON file: " + path.string());
    }
    file << data.dump(2);
}

template <typename T>
void write_binary_file(const std::filesystem::path& path, const std::vector<T>& data) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to write binary file: " + path.string());
    }
    if (!data.empty()) {
        file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(T)));
    }
}

template <typename T>
nlohmann::json compute_stats(const std::vector<T>& data) {
    nlohmann::json stats;
    if (data.empty()) {
        stats["count"] = 0;
        return stats;
    }
    T min_val = data.front();
    T max_val = data.front();
    long double sum = 0.0;
    for (const auto& v : data) {
        if (v < min_val) {
            min_val = v;
        }
        if (v > max_val) {
            max_val = v;
        }
        sum += static_cast<long double>(v);
    }
    stats["count"] = data.size();
    stats["min"] = min_val;
    stats["max"] = max_val;
    stats["mean"] = static_cast<double>(sum / static_cast<long double>(data.size()));
    return stats;
}

nlohmann::json compute_bool_stats(const std::vector<int64_t>& data) {
    nlohmann::json stats;
    size_t count_true = 0;
    for (auto v : data) {
        if (v != 0) {
            ++count_true;
        }
    }
    stats["count"] = data.size();
    stats["count_true"] = count_true;
    stats["count_false"] = data.size() - count_true;
    return stats;
}

std::filesystem::path ensure_dump_dir(const DumpContext& dump, const std::string& name) {
    auto dir = dump.dir / name;
    std::filesystem::create_directories(dir);
    return dir;
}

void dump_tensor(const DumpContext& dump, const std::string& name, const ov::Tensor& tensor) {
    if (!dump.enabled) {
        return;
    }
    nlohmann::json meta;
    meta["name"] = name;
    meta["shape"] = tensor.get_shape();
    meta["dtype"] = element_type_to_string(tensor.get_element_type());
    meta["element_count"] = tensor.get_size();

    const auto type = tensor.get_element_type();
    if (type == ov::element::f32 || type == ov::element::f16 || type == ov::element::bf16) {
        auto data = tensor_to_float_vector(tensor);
        meta["data_dtype"] = "f32";
        meta["stats"] = compute_stats(data);
        write_binary_file(dump.dir / (name + ".bin"), data);
    } else if (type == ov::element::i64 || type == ov::element::i32) {
        auto data = tensor_to_int64_vector(tensor);
        meta["data_dtype"] = "i64";
        meta["stats"] = compute_stats(data);
        write_binary_file(dump.dir / (name + ".bin"), data);
    } else if (type == ov::element::boolean) {
        auto data = tensor_to_int64_vector(tensor);
        meta["data_dtype"] = "i64";
        meta["stats"] = compute_bool_stats(data);
        write_binary_file(dump.dir / (name + ".bin"), data);
    } else {
        meta["data_dtype"] = "unsupported";
    }
    write_json_file(dump.dir / (name + ".json"), meta);
}

void dump_f32_vector(const DumpContext& dump,
                     const std::string& name,
                     const std::vector<float>& data,
                     const ov::Shape& shape) {
    if (!dump.enabled) {
        return;
    }
    nlohmann::json meta;
    meta["name"] = name;
    meta["shape"] = shape;
    meta["dtype"] = "f32";
    meta["element_count"] = data.size();
    meta["stats"] = compute_stats(data);
    write_binary_file(dump.dir / (name + ".bin"), data);
    write_json_file(dump.dir / (name + ".json"), meta);
}

void dump_text_file(const DumpContext& dump, const std::string& name, const std::string& text) {
    if (!dump.enabled) {
        return;
    }
    std::ofstream file(dump.dir / (name + ".txt"));
    if (!file.is_open()) {
        throw std::runtime_error("Failed to write text file: " + name);
    }
    file << text;
}

ov::Tensor make_f32_tensor(const std::vector<float>& data, const ov::Shape& shape) {
    ov::Tensor tensor(ov::element::f32, shape);
    if (!data.empty()) {
        std::memcpy(tensor.data(), data.data(), data.size() * sizeof(float));
    }
    return tensor;
}

ov::Tensor make_bool_tensor(const std::vector<char>& data, const ov::Shape& shape) {
    ov::Tensor tensor(ov::element::boolean, shape);
    if (!data.empty()) {
        std::memcpy(tensor.data(), data.data(), data.size() * sizeof(char));
    }
    return tensor;
}

ov::Tensor make_position_ids(size_t batch, size_t seq) {
    ov::Tensor ids(ov::element::i64, {batch, seq});
    auto* data = ids.data<int64_t>();
    for (size_t b = 0; b < batch; ++b) {
        for (size_t i = 0; i < seq; ++i) {
            data[b * seq + i] = static_cast<int64_t>(i);
        }
    }
    return ids;
}

std::vector<int32_t> build_cap_pos_ids(size_t cap_len_padded) {
    std::vector<int32_t> pos_ids;
    pos_ids.reserve(cap_len_padded * 3);
    for (size_t i = 0; i < cap_len_padded; ++i) {
        pos_ids.push_back(static_cast<int32_t>(i + 1));
        pos_ids.push_back(0);
        pos_ids.push_back(0);
    }
    return pos_ids;
}

std::vector<int32_t> build_image_pos_ids(const ImageTokensMeta& meta, size_t cap_len_padded) {
    std::vector<int32_t> pos_ids;
    pos_ids.reserve(meta.padded_len * 3);
    const int32_t base = static_cast<int32_t>(cap_len_padded) + 1;
    for (int32_t f = 0; f < meta.f_tokens; ++f) {
        const int32_t f_pos = base + f;
        for (int32_t h = 0; h < meta.h_tokens; ++h) {
            for (int32_t w = 0; w < meta.w_tokens; ++w) {
                pos_ids.push_back(f_pos);
                pos_ids.push_back(h);
                pos_ids.push_back(w);
            }
        }
    }
    for (size_t i = 0; i < meta.pad_len; ++i) {
        pos_ids.push_back(0);
        pos_ids.push_back(0);
        pos_ids.push_back(0);
    }
    return pos_ids;
}

void validate_pos_ids(const std::vector<int32_t>& pos_ids, const std::vector<int32_t>& axes_lens) {
    const size_t axes = axes_lens.size();
    if (axes == 0 || pos_ids.size() % axes != 0) {
        throw std::runtime_error("Invalid pos_ids shape");
    }
    for (size_t i = 0; i < pos_ids.size(); ++i) {
        const size_t axis = i % axes;
        const int32_t pos = pos_ids[i];
        if (pos < 0 || pos >= axes_lens[axis]) {
            throw std::runtime_error("Position id out of range for axis " + std::to_string(axis));
        }
    }
}

std::pair<std::vector<float>, std::vector<float>> compute_rope_cos_sin(
    const std::vector<int32_t>& pos_ids,
    const std::vector<int32_t>& axes_dims,
    float rope_theta) {
    const size_t axes = axes_dims.size();
    if (axes == 0 || pos_ids.size() % axes != 0) {
        throw std::runtime_error("Invalid pos_ids shape for RoPE");
    }
    size_t seq_len = pos_ids.size() / axes;
    size_t half_dim_total = 0;
    std::vector<std::vector<float>> inv_freqs;
    inv_freqs.reserve(axes);
    for (size_t i = 0; i < axes; ++i) {
        const int32_t dim = axes_dims[i];
        if (dim % 2 != 0) {
            throw std::runtime_error("RoPE axis dim must be even");
        }
        const size_t half = static_cast<size_t>(dim / 2);
        std::vector<float> inv_freq(half);
        for (size_t j = 0; j < half; ++j) {
            const float exponent = static_cast<float>(2.0 * j) / static_cast<float>(dim);
            inv_freq[j] = 1.0f / std::pow(rope_theta, exponent);
        }
        half_dim_total += half;
        inv_freqs.push_back(std::move(inv_freq));
    }

    std::vector<float> cos_vals(seq_len * half_dim_total);
    std::vector<float> sin_vals(seq_len * half_dim_total);
    for (size_t s = 0; s < seq_len; ++s) {
        size_t offset = 0;
        for (size_t axis = 0; axis < axes; ++axis) {
            const int32_t pos = pos_ids[s * axes + axis];
            const auto& inv = inv_freqs[axis];
            for (size_t j = 0; j < inv.size(); ++j) {
                const float angle = static_cast<float>(pos) * inv[j];
                cos_vals[s * half_dim_total + offset + j] = std::cos(angle);
                sin_vals[s * half_dim_total + offset + j] = std::sin(angle);
            }
            offset += inv.size();
        }
    }
    return {cos_vals, sin_vals};
}

void patchify_latents(const std::vector<float>& latents,
                      const ImageTokensMeta& meta,
                      std::vector<float>& tokens_out) {
    const size_t expected = static_cast<size_t>(meta.channels) * meta.frames *
                            meta.height_latent * meta.width_latent;
    if (latents.size() != expected) {
        throw std::runtime_error("Latents buffer size mismatch");
    }
    tokens_out.assign(meta.padded_len * static_cast<size_t>(meta.patch_dim), 0.0f);
    size_t token_idx = 0;
    for (int32_t f = 0; f < meta.f_tokens; ++f) {
        for (int32_t h = 0; h < meta.h_tokens; ++h) {
            for (int32_t w = 0; w < meta.w_tokens; ++w) {
                const size_t base = token_idx * static_cast<size_t>(meta.patch_dim);
                size_t offset = 0;
                for (int32_t pf = 0; pf < meta.f_patch_size; ++pf) {
                    const int32_t f_idx = f * meta.f_patch_size + pf;
                    for (int32_t ph = 0; ph < meta.patch_size; ++ph) {
                        const int32_t h_idx = h * meta.patch_size + ph;
                        for (int32_t pw = 0; pw < meta.patch_size; ++pw) {
                            const int32_t w_idx = w * meta.patch_size + pw;
                            for (int32_t c = 0; c < meta.channels; ++c) {
                                const size_t lat_idx =
                                    static_cast<size_t>(((c * meta.frames + f_idx) * meta.height_latent + h_idx) *
                                                       meta.width_latent + w_idx);
                                tokens_out[base + offset] = latents[lat_idx];
                                ++offset;
                            }
                        }
                    }
                }
                ++token_idx;
            }
        }
    }
}

void unpatchify_tokens(const float* tokens,
                       const ImageTokensMeta& meta,
                       std::vector<float>& latents_out) {
    latents_out.assign(static_cast<size_t>(meta.channels) * meta.frames *
                           meta.height_latent * meta.width_latent,
                       0.0f);
    size_t token_idx = 0;
    for (int32_t f = 0; f < meta.f_tokens; ++f) {
        for (int32_t h = 0; h < meta.h_tokens; ++h) {
            for (int32_t w = 0; w < meta.w_tokens; ++w) {
                const float* token = tokens + token_idx * static_cast<size_t>(meta.patch_dim);
                size_t offset = 0;
                for (int32_t pf = 0; pf < meta.f_patch_size; ++pf) {
                    const int32_t f_idx = f * meta.f_patch_size + pf;
                    for (int32_t ph = 0; ph < meta.patch_size; ++ph) {
                        const int32_t h_idx = h * meta.patch_size + ph;
                        for (int32_t pw = 0; pw < meta.patch_size; ++pw) {
                            const int32_t w_idx = w * meta.patch_size + pw;
                            for (int32_t c = 0; c < meta.channels; ++c) {
                                const size_t lat_idx =
                                    static_cast<size_t>(((c * meta.frames + f_idx) * meta.height_latent + h_idx) *
                                                       meta.width_latent + w_idx);
                                latents_out[lat_idx] = token[offset];
                                ++offset;
                            }
                        }
                    }
                }
                ++token_idx;
            }
        }
    }
}

std::shared_ptr<ov::Model> create_zimage_dit_model(
    const ov::genai::modeling::models::ZImageDiTConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    ov::genai::modeling::BuilderContext ctx;
    ov::genai::modeling::models::ZImageDiTModel model(ctx, cfg);

    ov::genai::modeling::weights::LoadOptions options;
    options.allow_unmatched = true;
    options.allow_missing = false;
    options.report_missing = true;
    options.report_unmatched = true;
    auto report = ov::genai::modeling::weights::load_model(model, source, finalizer, options);
    if (!report.unmatched.empty()) {
        std::cout << "[ZImageDiT] Unmatched weights: " << report.unmatched.size() << std::endl;
    }

    auto x_tokens = ctx.parameter("x_tokens", ov::element::f32, ov::PartialShape{-1, -1, cfg.patch_dim()});
    auto x_mask = ctx.parameter("x_mask", ov::element::boolean, ov::PartialShape{-1, -1});
    auto cap_feats = ctx.parameter("cap_feats", ov::element::f32, ov::PartialShape{-1, -1, cfg.cap_feat_dim});
    auto cap_mask = ctx.parameter("cap_mask", ov::element::boolean, ov::PartialShape{-1, -1});
    auto timesteps = ctx.parameter("timesteps", ov::element::f32, ov::PartialShape{-1});
    auto x_rope_cos = ctx.parameter("x_rope_cos", ov::element::f32, ov::PartialShape{-1, -1, cfg.head_dim() / 2});
    auto x_rope_sin = ctx.parameter("x_rope_sin", ov::element::f32, ov::PartialShape{-1, -1, cfg.head_dim() / 2});
    auto cap_rope_cos =
        ctx.parameter("cap_rope_cos", ov::element::f32, ov::PartialShape{-1, -1, cfg.head_dim() / 2});
    auto cap_rope_sin =
        ctx.parameter("cap_rope_sin", ov::element::f32, ov::PartialShape{-1, -1, cfg.head_dim() / 2});

    auto output = model.forward(x_tokens,
                                x_mask,
                                cap_feats,
                                cap_mask,
                                timesteps,
                                x_rope_cos,
                                x_rope_sin,
                                cap_rope_cos,
                                cap_rope_sin);
    auto result = std::make_shared<ov::op::v0::Result>(output.output());
    result->output(0).set_names({"noise_pred"});
    result->set_friendly_name("noise_pred");
    return ctx.build_model({result->output(0)});
}
PromptContext build_prompt_context(const std::string& prompt,
                                   ov::genai::Tokenizer& tokenizer,
                                   ov::CompiledModel& text_compiled,
                                   int32_t cap_feat_dim,
                                   const DiTRopeConfig& rope_cfg,
                                   const ImageTokensMeta& image_meta,
                                   size_t max_seq_len,
                                   const DumpContext* dump_ctx,
                                   const std::string& tag) {
    ov::genai::ChatHistory history({{{"role", "user"}, {"content", prompt}}});
    ov::genai::JsonContainer extra({{"enable_thinking", true}});
    const std::string formatted = tokenizer.apply_chat_template(history, true, {}, std::nullopt, extra);

    ov::AnyMap encode_opts;
    encode_opts["add_special_tokens"] = false;
    encode_opts["max_length"] = static_cast<int32_t>(max_seq_len);
    auto tokenized = tokenizer.encode(formatted, encode_opts);
    if (dump_ctx && dump_ctx->enabled) {
        auto dir = ensure_dump_dir(*dump_ctx, "prompt_" + tag);
        DumpContext local{true, dir};
        dump_text_file(local, "formatted_prompt", formatted);
        dump_tensor(local, "input_ids", tokenized.input_ids);
        dump_tensor(local, "attention_mask", tokenized.attention_mask);
    }

    const auto input_shape = tokenized.input_ids.get_shape();
    if (input_shape.size() != 2 || input_shape[0] != 1) {
        throw std::runtime_error("Tokenizer output must have shape [1, seq_len]");
    }
    const size_t seq_len = input_shape[1];
    ov::Tensor position_ids = make_position_ids(1, seq_len);

    // Ensure attention_mask has shape [1, seq_len] to match input_ids/position_ids for text encoder
    ov::Tensor attention_mask_for_text(ov::element::i64, {1, seq_len});
    {
        const auto am_shape = tokenized.attention_mask.get_shape();
        const size_t am_len = (am_shape.size() == 2 && am_shape[0] == 1) ? am_shape[1] : 0;
        int64_t* out = attention_mask_for_text.data<int64_t>();
        if (am_len == seq_len && tokenized.attention_mask.get_element_type() == ov::element::i64) {
            std::memcpy(out, tokenized.attention_mask.data<const int64_t>(), seq_len * sizeof(int64_t));
        } else if (am_len == seq_len && tokenized.attention_mask.get_element_type() == ov::element::i32) {
            const int32_t* src = tokenized.attention_mask.data<const int32_t>();
            for (size_t i = 0; i < seq_len; ++i)
                out[i] = static_cast<int64_t>(src[i]);
        } else {
            std::fill(out, out + seq_len, 1);
        }
    }

    auto request = text_compiled.create_infer_request();
    request.set_tensor("input_ids", tokenized.input_ids);
    request.set_tensor("attention_mask", attention_mask_for_text);
    request.set_tensor("position_ids", position_ids);
    request.infer();

    ov::Tensor hidden_states = request.get_tensor("hidden_states");
    if (dump_ctx && dump_ctx->enabled) {
        auto dir = ensure_dump_dir(*dump_ctx, "text_encoder_" + tag);
        DumpContext local{true, dir};
        dump_tensor(local, "hidden_states", hidden_states);
        dump_tensor(local, "position_ids", position_ids);
    }
    const auto hs_shape = hidden_states.get_shape();
    if (hs_shape.size() != 3 || hs_shape[0] != 1) {
        throw std::runtime_error("hidden_states must have shape [1, seq, dim]");
    }
    const size_t hidden = hs_shape[2];
    if (hidden != static_cast<size_t>(cap_feat_dim)) {
        throw std::runtime_error("hidden_states dim does not match cap_feat_dim");
    }

    auto indices = mask_indices(tokenized.attention_mask);
    if (indices.empty()) {
        throw std::runtime_error("No valid tokens in attention_mask");
    }
    auto prompt_feats = copy_rows_to_float(hidden_states, indices, hidden);

    const size_t cap_len = indices.size();
    const size_t cap_pad_len = (kSeqMultiOf - (cap_len % kSeqMultiOf)) % kSeqMultiOf;
    const size_t cap_len_padded = cap_len + cap_pad_len;

    std::vector<float> cap_feats_padded(cap_len_padded * hidden, 0.0f);
    std::memcpy(cap_feats_padded.data(), prompt_feats.data(), prompt_feats.size() * sizeof(float));

    std::vector<char> cap_mask_vec(cap_len_padded, 0);
    for (size_t i = 0; i < cap_len; ++i) {
        cap_mask_vec[i] = 1;
    }

    auto cap_pos_ids = build_cap_pos_ids(cap_len_padded);
    auto x_pos_ids = build_image_pos_ids(image_meta, cap_len_padded);

    validate_pos_ids(cap_pos_ids, rope_cfg.axes_lens);
    validate_pos_ids(x_pos_ids, rope_cfg.axes_lens);

    auto cap_cos_sin = compute_rope_cos_sin(cap_pos_ids, rope_cfg.axes_dims, rope_cfg.rope_theta);
    auto x_cos_sin = compute_rope_cos_sin(x_pos_ids, rope_cfg.axes_dims, rope_cfg.rope_theta);

    const size_t head_dim_half =
        static_cast<size_t>(std::accumulate(rope_cfg.axes_dims.begin(), rope_cfg.axes_dims.end(), 0)) / 2;

    PromptContext ctx;
    ctx.cap_len = cap_len;
    ctx.cap_len_padded = cap_len_padded;
    ctx.cap_feats = make_f32_tensor(cap_feats_padded, {1, cap_len_padded, hidden});
    ctx.cap_mask = make_bool_tensor(cap_mask_vec, {1, cap_len_padded});
    ctx.cap_rope_cos = make_f32_tensor(cap_cos_sin.first, {1, cap_len_padded, head_dim_half});
    ctx.cap_rope_sin = make_f32_tensor(cap_cos_sin.second, {1, cap_len_padded, head_dim_half});
    ctx.x_rope_cos = make_f32_tensor(x_cos_sin.first, {1, image_meta.padded_len, head_dim_half});
    ctx.x_rope_sin = make_f32_tensor(x_cos_sin.second, {1, image_meta.padded_len, head_dim_half});
    if (dump_ctx && dump_ctx->enabled) {
        auto dir = ensure_dump_dir(*dump_ctx, "prompt_context_" + tag);
        DumpContext local{true, dir};
        dump_f32_vector(local, "cap_feats_padded", cap_feats_padded, {1, cap_len_padded, hidden});
        dump_tensor(local, "cap_mask", ctx.cap_mask);
        dump_tensor(local, "cap_rope_cos", ctx.cap_rope_cos);
        dump_tensor(local, "cap_rope_sin", ctx.cap_rope_sin);
        dump_tensor(local, "x_rope_cos", ctx.x_rope_cos);
        dump_tensor(local, "x_rope_sin", ctx.x_rope_sin);
    }
    return ctx;
}

double elapsed_ms(const std::chrono::steady_clock::time_point& start,
                  const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

int main(int argc, char* argv[]) try {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <MODEL_DIR> <PROMPT> [OUTPUT_BMP] [DEVICE] [HEIGHT] [WIDTH] [STEPS] [SEED] [GUIDANCE]"
                  << " [DUMP_DIR|none] [TEXT_QUANT] [TEXT_GS] [TEXT_BACKUP] [DIT_QUANT] [DIT_GS] [DIT_BACKUP] [VAE_QUANT] [VAE_GS] [VAE_BACKUP]\n";
        return 1;
    }

    const std::filesystem::path model_dir = argv[1];
    const std::string prompt = argv[2];
    const std::string output_path = (argc > 3) ? argv[3] : "zimage_out.bmp";
    const std::string device = (argc > 4) ? argv[4] : "GPU";
    const int32_t height = (argc > 5) ? std::stoi(argv[5]) : 512;
    const int32_t width = (argc > 6) ? std::stoi(argv[6]) : 512;
    const int32_t steps = (argc > 7) ? std::stoi(argv[7]) : 8;
    const int32_t seed_arg = (argc > 8) ? std::stoi(argv[8]) : 0;
    const float guidance_scale = (argc > 9) ? std::stof(argv[9]) : 0.0f;
    const std::string dump_dir_arg = (argc > 10) ? argv[10] : "";
    
    // Text Encoder Quantization args
    std::string text_quant_mode = (argc > 11) ? argv[11] : "";
    int text_group_size = (argc > 12) ? std::stoi(argv[12]) : 128;
    std::string text_backup_mode = (argc > 13) ? argv[13] : "";

    // DiT Quantization args
    std::string dit_quant_mode = (argc > 14) ? argv[14] : "";
    int dit_group_size = (argc > 15) ? std::stoi(argv[15]) : 128;
    std::string dit_backup_mode = (argc > 16) ? argv[16] : "";

    // VAE Quantization args
    std::string vae_quant_mode = (argc > 17) ? argv[17] : "";
    int vae_group_size = (argc > 18) ? std::stoi(argv[18]) : 128;
    std::string vae_backup_mode = (argc > 19) ? argv[19] : "";

    const float cfg_truncation = 1.0f;

    DumpContext dump;
    dump.enabled = !dump_dir_arg.empty() && dump_dir_arg != "none";
    if (dump.enabled) {
        dump.dir = std::filesystem::path(dump_dir_arg);
        std::filesystem::create_directories(dump.dir);
    }

    // setup huggingface safetensors model and config files path
    const std::filesystem::path text_dir = model_dir / "text_encoder";
    const std::filesystem::path dit_dir = model_dir / "transformer";
    const std::filesystem::path vae_dir = model_dir / "vae";
    const std::filesystem::path tokenizer_dir = model_dir / "tokenizer";

    // read config.json files to get model config parameters
    auto text_cfg = read_qwen3_config(text_dir / "config.json");
    DiTRopeConfig rope_cfg;
    auto dit_cfg = read_dit_config(dit_dir / "config.json", rope_cfg);
    auto vae_cfg = read_vae_config(vae_dir / "config.json");
    auto sched_cfg = read_scheduler_config(model_dir / "scheduler" / "scheduler_config.json");

    const int32_t head_dim = dit_cfg.head_dim();
    const int32_t rope_dim_sum = std::accumulate(rope_cfg.axes_dims.begin(), rope_cfg.axes_dims.end(), 0);
    if (head_dim != rope_dim_sum) {
        throw std::runtime_error("RoPE axes_dims sum does not match head_dim");
    }
    if (rope_cfg.axes_dims.size() != rope_cfg.axes_lens.size()) {
        throw std::runtime_error("axes_dims and axes_lens size mismatch");
    }
    if (text_cfg.hidden_size != dit_cfg.cap_feat_dim) {
        std::cout << "[ZImage] Warning: text hidden_size != cap_feat_dim" << std::endl;
    }
    if (dit_cfg.in_channels != vae_cfg.latent_channels) {
        std::cout << "[ZImage] Warning: DiT in_channels != VAE latent_channels" << std::endl;
    }

    const int32_t vae_scale_factor = 1 << (static_cast<int32_t>(vae_cfg.block_out_channels.size()) - 1);
    const int32_t vae_scale = vae_scale_factor * 2;
    if (height % vae_scale != 0 || width % vae_scale != 0) {
        throw std::runtime_error("Height/width must be divisible by " + std::to_string(vae_scale));
    }
    const int32_t height_latent = 2 * (height / vae_scale);
    const int32_t width_latent = 2 * (width / vae_scale);

    ImageTokensMeta image_meta =
        build_image_meta(dit_cfg.in_channels, 1, height_latent, width_latent, dit_cfg.patch_size, dit_cfg.f_patch_size);

    const size_t image_seq_len = image_meta.image_len;
    const float mu = calculate_shift(image_seq_len, sched_cfg);
    FlowMatchEulerDiscreteScheduler scheduler(sched_cfg);
    scheduler.set_timesteps(steps, mu);

    // initialize tokenizer (convert json to xml at first time if no xml exists)
    const bool has_ov_tokenizer = std::filesystem::exists(tokenizer_dir / "openvino_tokenizer.xml");
    const bool has_hf_tokenizer = std::filesystem::exists(tokenizer_dir / "tokenizer.json");
    if (!has_ov_tokenizer && !has_hf_tokenizer) {
        throw std::runtime_error("Missing tokenizer.json or openvino_tokenizer.xml in tokenizer folder");
    }

    ov::genai::Tokenizer tokenizer(tokenizer_dir);

    // parse quantization config from args
    auto text_quant_config = create_quantization_config(text_quant_mode, text_group_size, text_backup_mode);
    auto dit_quant_config = create_quantization_config(dit_quant_mode, dit_group_size, dit_backup_mode);
    auto vae_quant_config = create_quantization_config(vae_quant_mode, vae_group_size, vae_backup_mode);

    std::shared_ptr<ov::Model> text_model;
    {
        auto text_data = ov::genai::safetensors::load_safetensors(text_dir);
        ov::genai::safetensors::SafetensorsWeightSource text_source(std::move(text_data));
        ov::genai::safetensors::SafetensorsWeightFinalizer text_finalizer(text_quant_config);
        text_model = ov::genai::modeling::models::create_qwen3_text_encoder_model(text_cfg, text_source, text_finalizer);
    }

    std::shared_ptr<ov::Model> dit_model;
    {
        auto dit_data = ov::genai::safetensors::load_safetensors(dit_dir);
        ov::genai::safetensors::SafetensorsWeightSource dit_source(std::move(dit_data));
        ov::genai::safetensors::SafetensorsWeightFinalizer dit_finalizer(dit_quant_config);
        dit_model = create_zimage_dit_model(dit_cfg, dit_source, dit_finalizer);
    }

    std::shared_ptr<ov::Model> vae_model;
    {
        auto vae_data = ov::genai::safetensors::load_safetensors(vae_dir);
        ov::genai::safetensors::SafetensorsWeightSource vae_source(std::move(vae_data));
        ov::genai::safetensors::SafetensorsWeightFinalizer vae_finalizer(vae_quant_config);
        vae_model = ov::genai::modeling::models::create_zimage_vae_decoder_model(vae_cfg, vae_source, vae_finalizer);
    }

    // compile the 3 models
    ov::Core core;
    ov::AnyMap compile_props;
    if (device.find("GPU") != std::string::npos) {
        compile_props[ov::hint::inference_precision.name()] = ov::element::f32;
    }
    auto compiled_text = core.compile_model(text_model, device);
    auto compiled_dit = compile_props.empty()
                            ? core.compile_model(dit_model, device)
                            : core.compile_model(dit_model, device, compile_props);
    auto compiled_vae = core.compile_model(vae_model, device);

    // build chat template, tokenize, text encoder model inference
    auto prompt_start = std::chrono::steady_clock::now();
    PromptContext pos_ctx =
        build_prompt_context(prompt,
                             tokenizer,
                             compiled_text,
                             dit_cfg.cap_feat_dim,
                             rope_cfg,
                             image_meta,
                             kDefaultMaxSeqLen,
                             &dump,
                             "pos");
    PromptContext neg_ctx;
    const bool do_cfg = guidance_scale > 1.0f;
    if (do_cfg) {
        neg_ctx = build_prompt_context("",
                                       tokenizer,
                                       compiled_text,
                                       dit_cfg.cap_feat_dim,
                                       rope_cfg,
                                       image_meta,
                                       kDefaultMaxSeqLen,
                                       &dump,
                                       "neg");
    }
    auto prompt_end = std::chrono::steady_clock::now();

    std::vector<char> x_mask_vec(image_meta.padded_len, 0);
    for (size_t i = 0; i < image_meta.image_len; ++i) {
        x_mask_vec[i] = 1;
    }
    ov::Tensor x_mask = make_bool_tensor(x_mask_vec, {1, image_meta.padded_len});
    if (dump.enabled) {
        dump_tensor(dump, "x_mask", x_mask);
    }

    std::mt19937 rng(seed_arg < 0 ? std::random_device{}() : static_cast<uint32_t>(seed_arg));
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> latents(
        static_cast<size_t>(dit_cfg.in_channels) * image_meta.frames * height_latent * width_latent);
    for (auto& v : latents) {
        v = dist(rng);
    }
    if (dump.enabled) {
        dump_f32_vector(dump,
                        "latents_init",
                        latents,
                        {1,
                         static_cast<size_t>(dit_cfg.in_channels),
                         static_cast<size_t>(height_latent),
                         static_cast<size_t>(width_latent)});
    }

    ov::Tensor x_tokens_tensor(ov::element::f32, {1, image_meta.padded_len, static_cast<size_t>(image_meta.patch_dim)});
    ov::Tensor t_tensor(ov::element::f32, {1});
    std::vector<float> x_tokens;
    std::vector<float> model_out_latents;
    std::vector<float> noise_pred;
    std::vector<float> noise_pred_neg;


    // DiT denoise loop
    auto dit_request = compiled_dit.create_infer_request();
    dit_request.set_tensor("x_mask", x_mask);

    auto denoise_start = std::chrono::steady_clock::now();
    const auto& timesteps = scheduler.timesteps();
    const auto& sigmas = scheduler.sigmas();
    for (size_t i = 0; i < timesteps.size(); ++i) {
        const float t = timesteps[i];
        if (t == 0.0f && i + 1 == timesteps.size()) {
            continue;
        }

        // patchify latents: latents[16, 1, 64, 64] (C, F, H, W) -> tokens[4096, 16] (seq, patch_dim)
        patchify_latents(latents, image_meta, x_tokens);
        std::memcpy(x_tokens_tensor.data(), x_tokens.data(), x_tokens.size() * sizeof(float));

        const float t_norm = (1000.0f - t) / 1000.0f;
        t_tensor.data<float>()[0] = t_norm;

        // DiT inference
        dit_request.set_tensor("x_tokens", x_tokens_tensor);
        dit_request.set_tensor("timesteps", t_tensor);
        dit_request.set_tensor("cap_feats", pos_ctx.cap_feats);
        dit_request.set_tensor("cap_mask", pos_ctx.cap_mask);
        dit_request.set_tensor("x_rope_cos", pos_ctx.x_rope_cos);
        dit_request.set_tensor("x_rope_sin", pos_ctx.x_rope_sin);
        dit_request.set_tensor("cap_rope_cos", pos_ctx.cap_rope_cos);
        dit_request.set_tensor("cap_rope_sin", pos_ctx.cap_rope_sin);
        dit_request.infer();

        // unpatchify tokens: tokens tokens[4096, 16] (seq, patch_dim) -> latents[16, 1, 64, 64] (C, F, H, W)
        ov::Tensor output = dit_request.get_output_tensor();
        auto out_data = tensor_to_float_vector(output);
        unpatchify_tokens(out_data.data(), image_meta, model_out_latents);

        // noise prediction processing
        noise_pred.resize(model_out_latents.size());
        for (size_t j = 0; j < model_out_latents.size(); ++j) {
            noise_pred[j] = -model_out_latents[j];
        }

        float current_guidance = guidance_scale;
        if (do_cfg && cfg_truncation <= 1.0f && t_norm > cfg_truncation) {
            current_guidance = 0.0f;
        }

        // Classifier - Free Guidance(CFG)
        if (do_cfg && current_guidance > 0.0f) {
            dit_request.set_tensor("cap_feats", neg_ctx.cap_feats);
            dit_request.set_tensor("cap_mask", neg_ctx.cap_mask);
            dit_request.set_tensor("x_rope_cos", neg_ctx.x_rope_cos);
            dit_request.set_tensor("x_rope_sin", neg_ctx.x_rope_sin);
            dit_request.set_tensor("cap_rope_cos", neg_ctx.cap_rope_cos);
            dit_request.set_tensor("cap_rope_sin", neg_ctx.cap_rope_sin);
            dit_request.infer();
            ov::Tensor neg_output = dit_request.get_output_tensor();
            auto neg_data = tensor_to_float_vector(neg_output);
            unpatchify_tokens(neg_data.data(), image_meta, model_out_latents);
            noise_pred_neg.resize(model_out_latents.size());
            for (size_t j = 0; j < model_out_latents.size(); ++j) {
                noise_pred_neg[j] = -model_out_latents[j];
            }

            for (size_t j = 0; j < noise_pred.size(); ++j) {
                noise_pred[j] = noise_pred[j] + current_guidance * (noise_pred[j] - noise_pred_neg[j]);
            }
        }

        // Euler step update latents
        const float sigma = sigmas[i];
        const float sigma_next = sigmas[i + 1];
        const float dt = sigma_next - sigma;
        for (size_t j = 0; j < latents.size(); ++j) {
            latents[j] += dt * noise_pred[j];
        }

        if (dump.enabled) {
            std::ostringstream step_name;
            step_name << "denoise_step_" << std::setw(2) << std::setfill('0') << i;
            auto step_dir = ensure_dump_dir(dump, step_name.str());
            DumpContext step_dump{true, step_dir};

            nlohmann::json step_meta;
            step_meta["step"] = i;
            step_meta["t"] = t;
            step_meta["t_norm"] = t_norm;
            step_meta["sigma"] = sigma;
            step_meta["sigma_next"] = sigma_next;
            step_meta["dt"] = dt;
            write_json_file(step_dir / "meta.json", step_meta);

            dump_f32_vector(step_dump,
                            "latents",
                            latents,
                            {1,
                             static_cast<size_t>(dit_cfg.in_channels),
                             static_cast<size_t>(height_latent),
                             static_cast<size_t>(width_latent)});
            dump_f32_vector(step_dump,
                            "x_tokens",
                            x_tokens,
                            {1, image_meta.padded_len, static_cast<size_t>(image_meta.patch_dim)});
            dump_f32_vector(step_dump,
                            "model_out_latents",
                            model_out_latents,
                            {1,
                             static_cast<size_t>(dit_cfg.in_channels),
                             static_cast<size_t>(height_latent),
                             static_cast<size_t>(width_latent)});
            dump_f32_vector(step_dump,
                            "noise_pred",
                            noise_pred,
                            {1,
                             static_cast<size_t>(dit_cfg.in_channels),
                             static_cast<size_t>(height_latent),
                             static_cast<size_t>(width_latent)});
            if (do_cfg && current_guidance > 0.0f) {
                dump_f32_vector(step_dump,
                                "noise_pred_neg",
                                noise_pred_neg,
                                {1,
                                 static_cast<size_t>(dit_cfg.in_channels),
                                 static_cast<size_t>(height_latent),
                                 static_cast<size_t>(width_latent)});
            }
        }
    }
    auto denoise_end = std::chrono::steady_clock::now();

    std::vector<float> scaled_latents(latents.size());
    for (size_t i = 0; i < latents.size(); ++i) {
        scaled_latents[i] = latents[i] / vae_cfg.scaling_factor + vae_cfg.shift_factor;
    }
    if (dump.enabled) {
        dump_f32_vector(dump,
                        "scaled_latents",
                        scaled_latents,
                        {1,
                         static_cast<size_t>(vae_cfg.latent_channels),
                         static_cast<size_t>(height_latent),
                         static_cast<size_t>(width_latent)});
    }

    ov::Tensor latents_tensor(ov::element::f32,
                              {1,
                               static_cast<size_t>(vae_cfg.latent_channels),
                               static_cast<size_t>(height_latent),
                               static_cast<size_t>(width_latent)});
    std::memcpy(latents_tensor.data(), scaled_latents.data(), scaled_latents.size() * sizeof(float));

    // VAE inference
    auto vae_request = compiled_vae.create_infer_request();
    vae_request.set_tensor("latents", latents_tensor);
    auto vae_start = std::chrono::steady_clock::now();
    vae_request.infer();
    auto vae_end = std::chrono::steady_clock::now();

    ov::Tensor image = vae_request.get_tensor("sample");
    if (dump.enabled) {
        dump_tensor(dump, "vae_output", image);
    }
    const auto image_shape = image.get_shape();
    if (image_shape.size() != 4 || image_shape[1] != 3) {
        throw std::runtime_error("VAE output must have shape [B, 3, H, W]");
    }
    const size_t batch = image_shape[0];
    const size_t out_h = image_shape[2];
    const size_t out_w = image_shape[3];

    // convert to RGB image
    ov::Tensor image_u8(ov::element::u8, {batch, out_h, out_w, 3});
    auto* dst = image_u8.data<uint8_t>();
    auto image_data = tensor_to_float_vector(image);
    const float* src = image_data.data();
    const size_t hw = out_h * out_w;
    for (size_t b = 0; b < batch; ++b) {
        const float* base = src + b * 3 * hw;
        for (size_t h = 0; h < out_h; ++h) {
            for (size_t w = 0; w < out_w; ++w) {
                const size_t idx = h * out_w + w;
                for (size_t c = 0; c < 3; ++c) {
                    float val = base[c * hw + idx];
                    val = val / 2.0f + 0.5f;
                    val = std::min(std::max(val, 0.0f), 1.0f);
                    dst[((b * out_h + h) * out_w + w) * 3 + c] =
                        static_cast<uint8_t>(std::round(val * 255.0f));
                }
            }
        }
    }
    if (dump.enabled) {
        dump_tensor(dump, "image_u8", image_u8);
    }

    imwrite(output_path, image_u8, false);

    const double prompt_ms = elapsed_ms(prompt_start, prompt_end);
    const double denoise_ms = elapsed_ms(denoise_start, denoise_end);
    const double vae_ms = elapsed_ms(vae_start, vae_end);
    std::cout << "Prompt encode time: " << prompt_ms << " ms" << std::endl;
    std::cout << "Denoise time: " << denoise_ms << " ms" << std::endl;
    std::cout << "VAE decode time: " << vae_ms << " ms" << std::endl;
    std::cout << "Output saved to " << output_path << std::endl;
    return 0;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {
    }
    return 1;
}
