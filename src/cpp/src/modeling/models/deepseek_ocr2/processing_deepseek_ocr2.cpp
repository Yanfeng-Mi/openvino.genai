// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/deepseek_ocr2/processing_deepseek_ocr2.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

#include <openvino/core/except.hpp>
#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/float16.hpp>

#include "json_utils.hpp"

namespace {

void read_json_file(const std::filesystem::path& path, nlohmann::json& data) {
    std::ifstream file(path);
    if (!file.is_open()) {
        OPENVINO_THROW("Failed to open config file: ", path.string());
    }
    file >> data;
}

std::filesystem::path resolve_config_path(const std::filesystem::path& path, const std::string& file_name) {
    if (std::filesystem::is_directory(path)) {
        return path / file_name;
    }
    return path;
}

void parse_candidate_resolutions(const nlohmann::json& data,
                                 const std::string& key,
                                 std::vector<std::array<int32_t, 2>>& out) {
    if (!data.contains(key)) {
        return;
    }
    const auto& arr = data.at(key);
    if (!arr.is_array()) {
        OPENVINO_THROW("candidate_resolutions must be an array");
    }
    out.clear();
    for (const auto& entry : arr) {
        if (!entry.is_array() || entry.size() != 2) {
            OPENVINO_THROW("candidate_resolutions entries must be [width, height]");
        }
        std::array<int32_t, 2> value{entry.at(0).get<int32_t>(), entry.at(1).get<int32_t>()};
        out.push_back(value);
    }
}

void parse_language_config(const nlohmann::json& data,
                           ov::genai::modeling::models::DeepseekOCR2LanguageConfig& cfg) {
    using ov::genai::utils::read_json_param;
    read_json_param(data, "vocab_size", cfg.vocab_size);
    read_json_param(data, "hidden_size", cfg.hidden_size);
    read_json_param(data, "intermediate_size", cfg.intermediate_size);
    read_json_param(data, "moe_intermediate_size", cfg.moe_intermediate_size);
    read_json_param(data, "num_hidden_layers", cfg.num_hidden_layers);
    read_json_param(data, "num_attention_heads", cfg.num_attention_heads);
    read_json_param(data, "num_key_value_heads", cfg.num_key_value_heads);
    read_json_param(data, "n_routed_experts", cfg.n_routed_experts);
    read_json_param(data, "n_shared_experts", cfg.n_shared_experts);
    read_json_param(data, "num_experts_per_tok", cfg.num_experts_per_tok);
    read_json_param(data, "moe_layer_freq", cfg.moe_layer_freq);
    read_json_param(data, "first_k_dense_replace", cfg.first_k_dense_replace);
    read_json_param(data, "rms_norm_eps", cfg.rms_norm_eps);
    read_json_param(data, "rope_theta", cfg.rope_theta);
    read_json_param(data, "hidden_act", cfg.hidden_act);
    read_json_param(data, "attention_bias", cfg.attention_bias);
    read_json_param(data, "attention_dropout", cfg.attention_dropout);
    read_json_param(data, "use_mla", cfg.use_mla);
    read_json_param(data, "qk_rope_head_dim", cfg.qk_rope_head_dim);
    read_json_param(data, "qk_nope_head_dim", cfg.qk_nope_head_dim);
    read_json_param(data, "v_head_dim", cfg.v_head_dim);
    read_json_param(data, "kv_lora_rank", cfg.kv_lora_rank);
    read_json_param(data, "q_lora_rank", cfg.q_lora_rank);
    read_json_param(data, "bos_token_id", cfg.bos_token_id);
    read_json_param(data, "eos_token_id", cfg.eos_token_id);
    read_json_param(data, "max_position_embeddings", cfg.max_position_embeddings);

    cfg.finalize();
}

void parse_vision_config(const nlohmann::json& data,
                         ov::genai::modeling::models::DeepseekOCR2VisionConfig& cfg) {
    using ov::genai::utils::read_json_param;
    read_json_param(data, "image_size", cfg.image_size);
    read_json_param(data, "mlp_ratio", cfg.mlp_ratio);
    read_json_param(data, "model_name", cfg.model_name);
    read_json_param(data, "model_type", cfg.model_type);

    if (data.contains("width") && data.at("width").is_object()) {
        const auto& width = data.at("width");
        if (width.contains("sam_vit_b")) {
            const auto& sam = width.at("sam_vit_b");
            read_json_param(sam, "width", cfg.sam_vit_b.width);
            read_json_param(sam, "layers", cfg.sam_vit_b.layers);
            read_json_param(sam, "heads", cfg.sam_vit_b.heads);
            read_json_param(sam, "global_attn_indexes", cfg.sam_vit_b.global_attn_indexes);
            read_json_param(sam, "downsample_channels", cfg.sam_vit_b.downsample_channels);
        }
        if (width.contains("qwen2-0-5b")) {
            const auto& qwen2 = width.at("qwen2-0-5b");
            read_json_param(qwen2, "dim", cfg.qwen2_0_5b.dim);
        }
    }

    cfg.finalize();
}

void parse_projector_config(const nlohmann::json& data,
                            ov::genai::modeling::models::DeepseekOCR2ProjectorConfig& cfg) {
    using ov::genai::utils::read_json_param;
    read_json_param(data, "input_dim", cfg.input_dim);
    read_json_param(data, "n_embed", cfg.n_embed);
    read_json_param(data, "projector_type", cfg.projector_type);
    read_json_param(data, "model_type", cfg.model_type);

    cfg.finalize();
}

void parse_processor_config(const nlohmann::json& data,
                            ov::genai::modeling::models::DeepseekOCR2ProcessorConfig& cfg) {
    using ov::genai::utils::read_json_param;
    read_json_param(data, "patch_size", cfg.patch_size);
    read_json_param(data, "downsample_ratio", cfg.downsample_ratio);
    read_json_param(data, "image_mean", cfg.image_mean);
    read_json_param(data, "image_std", cfg.image_std);
    read_json_param(data, "normalize", cfg.normalize);
    read_json_param(data, "add_special_token", cfg.add_special_token);
    read_json_param(data, "mask_prompt", cfg.mask_prompt);
    read_json_param(data, "image_token", cfg.image_token);
    read_json_param(data, "pad_token", cfg.pad_token);
    read_json_param(data, "sft_format", cfg.sft_format);
    read_json_param(data, "ignore_id", cfg.ignore_id);

    parse_candidate_resolutions(data, "candidate_resolutions", cfg.candidate_resolutions);
    cfg.finalize();
}

std::vector<int64_t> tensor_to_i64(const ov::Tensor& tensor) {
    const auto type = tensor.get_element_type();
    if (type != ov::element::i64 && type != ov::element::i32) {
        OPENVINO_THROW("Expected int32 or int64 tensor for tokens");
    }
    const size_t count = tensor.get_size();
    std::vector<int64_t> out(count);
    if (type == ov::element::i64) {
        const auto* src = tensor.data<const int64_t>();
        std::copy(src, src + count, out.begin());
    } else {
        const auto* src = tensor.data<const int32_t>();
        for (size_t i = 0; i < count; ++i) {
            out[i] = static_cast<int64_t>(src[i]);
        }
    }
    return out;
}

ov::Tensor make_i64_tensor(const std::vector<int64_t>& data, const ov::Shape& shape) {
    ov::Tensor tensor(ov::element::i64, shape);
    std::memcpy(tensor.data(), data.data(), data.size() * sizeof(int64_t));
    return tensor;
}

ov::Tensor make_bool_tensor(const std::vector<char>& data, const ov::Shape& shape) {
    ov::Tensor tensor(ov::element::boolean, shape);
    std::memcpy(tensor.data(), data.data(), data.size() * sizeof(char));
    return tensor;
}

struct ResizedImage {
    std::vector<float> data;
    size_t height = 0;
    size_t width = 0;
};

ResizedImage resize_bilinear_hwc(const uint8_t* src,
                                 size_t src_h,
                                 size_t src_w,
                                 size_t channels,
                                 size_t dst_h,
                                 size_t dst_w) {
    ResizedImage out;
    out.height = dst_h;
    out.width = dst_w;
    out.data.assign(dst_h * dst_w * channels, 0.0f);

    const float scale_y = static_cast<float>(src_h) / static_cast<float>(dst_h);
    const float scale_x = static_cast<float>(src_w) / static_cast<float>(dst_w);

    auto fetch = [&](size_t y, size_t x, size_t c) -> float {
        const size_t idx = (y * src_w + x) * channels + c;
        return static_cast<float>(src[idx]);
    };

    for (size_t y = 0; y < dst_h; ++y) {
        float in_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        in_y = std::max(0.0f, std::min(in_y, static_cast<float>(src_h - 1)));
        size_t y0 = static_cast<size_t>(std::floor(in_y));
        size_t y1 = std::min(y0 + 1, src_h - 1);
        float wy = in_y - static_cast<float>(y0);
        for (size_t x = 0; x < dst_w; ++x) {
            float in_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            in_x = std::max(0.0f, std::min(in_x, static_cast<float>(src_w - 1)));
            size_t x0 = static_cast<size_t>(std::floor(in_x));
            size_t x1 = std::min(x0 + 1, src_w - 1);
            float wx = in_x - static_cast<float>(x0);

            const float w00 = (1.0f - wy) * (1.0f - wx);
            const float w01 = (1.0f - wy) * wx;
            const float w10 = wy * (1.0f - wx);
            const float w11 = wy * wx;

            const size_t out_base = (y * dst_w + x) * channels;
            for (size_t c = 0; c < channels; ++c) {
                float v = 0.0f;
                v += w00 * fetch(y0, x0, c);
                v += w01 * fetch(y0, x1, c);
                v += w10 * fetch(y1, x0, c);
                v += w11 * fetch(y1, x1, c);
                out.data[out_base + c] = v;
            }
        }
    }

    return out;
}

void hwc_to_chw_normalized(const std::vector<float>& src_hwc,
                           size_t height,
                           size_t width,
                           size_t channels,
                           const std::array<float, 3>& mean,
                           const std::array<float, 3>& std,
                           float* dst_chw) {
    const float inv255 = 1.0f / 255.0f;
    for (size_t c = 0; c < channels; ++c) {
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                const size_t idx = (y * width + x) * channels + c;
                float v = src_hwc[idx];
                float norm = (v * inv255 - mean[c]) / std[c];
                dst_chw[(c * height + y) * width + x] = norm;
            }
        }
    }
}

void crop_hwc_to_chw_normalized(const std::vector<float>& src_hwc,
                                size_t src_h,
                                size_t src_w,
                                size_t channels,
                                size_t crop_x,
                                size_t crop_y,
                                size_t crop_h,
                                size_t crop_w,
                                const std::array<float, 3>& mean,
                                const std::array<float, 3>& std,
                                float* dst_chw) {
    const float inv255 = 1.0f / 255.0f;
    for (size_t c = 0; c < channels; ++c) {
        for (size_t y = 0; y < crop_h; ++y) {
            for (size_t x = 0; x < crop_w; ++x) {
                const size_t src_idx = ((crop_y + y) * src_w + (crop_x + x)) * channels + c;
                float v = src_hwc[src_idx];
                float norm = (v * inv255 - mean[c]) / std[c];
                dst_chw[(c * crop_h + y) * crop_w + x] = norm;
            }
        }
    }
}

std::vector<float> pad_and_resize_to_chw(const uint8_t* src,
                                         size_t src_h,
                                         size_t src_w,
                                         size_t channels,
                                         size_t target_size,
                                         const std::array<float, 3>& mean,
                                         const std::array<float, 3>& std) {
    if (channels != 3) {
        OPENVINO_THROW("Expected 3 channels for pad_and_resize");
    }
    const float scale = std::min(static_cast<float>(target_size) / static_cast<float>(src_w),
                                 static_cast<float>(target_size) / static_cast<float>(src_h));
    const size_t new_w = static_cast<size_t>(std::round(static_cast<float>(src_w) * scale));
    const size_t new_h = static_cast<size_t>(std::round(static_cast<float>(src_h) * scale));

    auto resized = resize_bilinear_hwc(src, src_h, src_w, channels, new_h, new_w);

    const std::array<float, 3> pad_color = {mean[0] * 255.0f, mean[1] * 255.0f, mean[2] * 255.0f};
    std::vector<float> padded(target_size * target_size * channels, 0.0f);
    for (size_t y = 0; y < target_size; ++y) {
        for (size_t x = 0; x < target_size; ++x) {
            size_t base = (y * target_size + x) * channels;
            padded[base + 0] = pad_color[0];
            padded[base + 1] = pad_color[1];
            padded[base + 2] = pad_color[2];
        }
    }

    const size_t offset_x = (target_size - new_w) / 2;
    const size_t offset_y = (target_size - new_h) / 2;

    for (size_t y = 0; y < new_h; ++y) {
        for (size_t x = 0; x < new_w; ++x) {
            const size_t src_base = (y * new_w + x) * channels;
            const size_t dst_base = ((offset_y + y) * target_size + (offset_x + x)) * channels;
            for (size_t c = 0; c < channels; ++c) {
                padded[dst_base + c] = resized.data[src_base + c];
            }
        }
    }

    std::vector<float> out(target_size * target_size * channels);
    hwc_to_chw_normalized(padded, target_size, target_size, channels, mean, std, out.data());
    return out;
}

std::vector<std::pair<int32_t, int32_t>> build_target_ratios(int32_t min_num, int32_t max_num) {
    std::vector<std::pair<int32_t, int32_t>> ratios;
    for (int32_t n = min_num; n <= max_num; ++n) {
        for (int32_t i = 1; i <= n; ++i) {
            for (int32_t j = 1; j <= n; ++j) {
                const int32_t prod = i * j;
                if (prod >= min_num && prod <= max_num) {
                    ratios.emplace_back(i, j);
                }
            }
        }
    }
    std::sort(ratios.begin(), ratios.end(), [](const auto& a, const auto& b) {
        return (a.first * a.second) < (b.first * b.second);
    });
    ratios.erase(std::unique(ratios.begin(), ratios.end()), ratios.end());
    return ratios;
}

std::pair<int32_t, int32_t> find_closest_aspect_ratio(double aspect_ratio,
                                                      const std::vector<std::pair<int32_t, int32_t>>& target_ratios,
                                                      size_t width,
                                                      size_t height,
                                                      int32_t image_size) {
    double best_diff = std::numeric_limits<double>::max();
    std::pair<int32_t, int32_t> best_ratio{1, 1};
    const double area = static_cast<double>(width) * static_cast<double>(height);

    for (const auto& ratio : target_ratios) {
        const double target_aspect = static_cast<double>(ratio.first) / static_cast<double>(ratio.second);
        const double diff = std::abs(aspect_ratio - target_aspect);
        if (diff < best_diff) {
            best_diff = diff;
            best_ratio = ratio;
        } else if (diff == best_diff) {
            const double threshold = 0.5 * static_cast<double>(image_size) * static_cast<double>(image_size) *
                                     static_cast<double>(ratio.first) * static_cast<double>(ratio.second);
            if (area > threshold) {
                best_ratio = ratio;
            }
        }
    }
    return best_ratio;
}

float read_as_f32(const void* data, size_t idx, const ov::element::Type& type) {
    if (type == ov::element::f32) {
        return static_cast<const float*>(data)[idx];
    }
    if (type == ov::element::f16) {
        return static_cast<const ov::float16*>(data)[idx];
    }
    if (type == ov::element::bf16) {
        return static_cast<const ov::bfloat16*>(data)[idx];
    }
    OPENVINO_THROW("Unsupported view separator dtype");
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

int32_t DeepseekOCR2LanguageConfig::resolved_kv_heads() const {
    return num_key_value_heads > 0 ? num_key_value_heads : num_attention_heads;
}

int32_t DeepseekOCR2LanguageConfig::resolved_q_head_dim() const {
    if (qk_nope_head_dim > 0 || qk_rope_head_dim > 0) {
        return qk_nope_head_dim + qk_rope_head_dim;
    }
    if (num_attention_heads <= 0) {
        return 0;
    }
    return hidden_size / num_attention_heads;
}

int32_t DeepseekOCR2LanguageConfig::resolved_v_head_dim() const {
    if (v_head_dim > 0) {
        return v_head_dim;
    }
    return resolved_q_head_dim();
}

void DeepseekOCR2LanguageConfig::finalize() {
    if (num_key_value_heads <= 0) {
        num_key_value_heads = num_attention_heads;
    }
    if (moe_layer_freq <= 0) {
        moe_layer_freq = 1;
    }
}

void DeepseekOCR2LanguageConfig::validate() const {
    if (hidden_size <= 0 || num_hidden_layers <= 0 || num_attention_heads <= 0) {
        OPENVINO_THROW("Invalid DeepseekOCR2 language config");
    }
    if (!use_mla && hidden_size % num_attention_heads != 0) {
        OPENVINO_THROW("hidden_size must be divisible by num_attention_heads for MHA");
    }
    if (resolved_kv_heads() <= 0) {
        OPENVINO_THROW("num_key_value_heads must be > 0");
    }
}

void DeepseekOCR2VisionConfig::finalize() {
    if (sam_vit_b.width <= 0) {
        sam_vit_b.width = 768;
    }
    if (sam_vit_b.layers <= 0) {
        sam_vit_b.layers = 12;
    }
    if (sam_vit_b.heads <= 0) {
        sam_vit_b.heads = 12;
    }
    if (qwen2_0_5b.dim <= 0) {
        qwen2_0_5b.dim = 896;
    }
}

void DeepseekOCR2VisionConfig::validate() const {
    if (image_size <= 0 || sam_vit_b.width <= 0 || sam_vit_b.layers <= 0 || sam_vit_b.heads <= 0) {
        OPENVINO_THROW("Invalid DeepseekOCR2 vision config");
    }
}

void DeepseekOCR2ProjectorConfig::finalize() {
    if (projector_type.empty()) {
        projector_type = "linear";
    }
}

void DeepseekOCR2ProjectorConfig::validate() const {
    if (input_dim <= 0 || n_embed <= 0) {
        OPENVINO_THROW("Invalid DeepseekOCR2 projector config");
    }
}

void DeepseekOCR2Config::finalize() {
    language.finalize();
    vision.finalize();
    projector.finalize();
    bos_token_id = language.bos_token_id;
    eos_token_id = language.eos_token_id;
}

void DeepseekOCR2Config::validate() const {
    language.validate();
    vision.validate();
    projector.validate();
}

DeepseekOCR2Config DeepseekOCR2Config::from_json(const nlohmann::json& data) {
    DeepseekOCR2Config cfg;
    using ov::genai::utils::read_json_param;
    read_json_param(data, "model_type", cfg.model_type);
    read_json_param(data, "tile_tag", cfg.tile_tag);
    read_json_param(data, "global_view_pos", cfg.global_view_pos);
    read_json_param(data, "torch_dtype", cfg.torch_dtype);
    read_json_param(data, "transformers_version", cfg.transformers_version);
    read_json_param(data, "architectures", cfg.architectures);
    parse_candidate_resolutions(data, "candidate_resolutions", cfg.candidate_resolutions);

    if (data.contains("language_config")) {
        parse_language_config(data.at("language_config"), cfg.language);
    } else {
        parse_language_config(data, cfg.language);
    }

    if (data.contains("vision_config")) {
        parse_vision_config(data.at("vision_config"), cfg.vision);
    }

    if (data.contains("projector_config")) {
        parse_projector_config(data.at("projector_config"), cfg.projector);
    }

    cfg.finalize();
    cfg.validate();
    return cfg;
}

DeepseekOCR2Config DeepseekOCR2Config::from_json_file(const std::filesystem::path& config_path) {
    nlohmann::json data;
    auto path = resolve_config_path(config_path, "config.json");
    read_json_file(path, data);
    return DeepseekOCR2Config::from_json(data);
}

void DeepseekOCR2ProcessorConfig::finalize() {
    if (patch_size <= 0) {
        patch_size = 16;
    }
    if (downsample_ratio <= 0) {
        downsample_ratio = 4;
    }
}

void DeepseekOCR2ProcessorConfig::validate() const {
    if (patch_size <= 0 || downsample_ratio <= 0) {
        OPENVINO_THROW("Invalid DeepseekOCR2 processor config");
    }
    if (image_token.empty()) {
        OPENVINO_THROW("DeepseekOCR2 processor image_token must be set");
    }
}

int64_t DeepseekOCR2ProcessorConfig::resolve_image_token_id(const ov::genai::Tokenizer& tokenizer) const {
    const auto vocab = tokenizer.get_vocab();
    auto it = vocab.find(image_token);
    if (it == vocab.end()) {
        OPENVINO_THROW("Image token not found in tokenizer vocab: ", image_token);
    }
    return it->second;
}

DeepseekOCR2ProcessorConfig DeepseekOCR2ProcessorConfig::from_json(const nlohmann::json& data) {
    DeepseekOCR2ProcessorConfig cfg;
    parse_processor_config(data, cfg);
    cfg.validate();
    return cfg;
}

DeepseekOCR2ProcessorConfig DeepseekOCR2ProcessorConfig::from_json_file(const std::filesystem::path& path) {
    nlohmann::json data;
    auto resolved = resolve_config_path(path, "processor_config.json");
    read_json_file(resolved, data);
    return DeepseekOCR2ProcessorConfig::from_json(data);
}

void DeepseekOCR2PreprocessConfig::validate() const {
    if (base_size <= 0 || image_size <= 0 || patch_size <= 0 || downsample_ratio <= 0) {
        OPENVINO_THROW("Invalid DeepseekOCR2 preprocess config");
    }
}

DeepseekOCR2ImagePreprocessor::DeepseekOCR2ImagePreprocessor(const DeepseekOCR2PreprocessConfig& cfg)
    : cfg_(cfg) {
    cfg_.validate();
}

int64_t DeepseekOCR2ImagePreprocessor::num_queries(int32_t image_size,
                                                   int32_t patch_size,
                                                   int32_t downsample_ratio) {
    if (patch_size <= 0 || downsample_ratio <= 0) {
        return 0;
    }
    const double tokens = static_cast<double>(image_size) / static_cast<double>(patch_size);
    return static_cast<int64_t>(std::ceil(tokens / static_cast<double>(downsample_ratio)));
}

int64_t DeepseekOCR2ImagePreprocessor::base_tokens(int32_t base_size,
                                                   int32_t patch_size,
                                                   int32_t downsample_ratio) {
    const int64_t q = num_queries(base_size, patch_size, downsample_ratio);
    return q * q;
}

int64_t DeepseekOCR2ImagePreprocessor::local_tokens(int32_t image_size,
                                                    int32_t patch_size,
                                                    int32_t downsample_ratio,
                                                    int32_t width_crop_num,
                                                    int32_t height_crop_num) {
    const int64_t q = num_queries(image_size, patch_size, downsample_ratio);
    return q * q * static_cast<int64_t>(width_crop_num) * static_cast<int64_t>(height_crop_num);
}

DeepseekOCR2ImageInputs DeepseekOCR2ImagePreprocessor::preprocess(const ov::Tensor& images) const {
    const auto shape = images.get_shape();
    if (shape.size() != 3 && shape.size() != 4) {
        OPENVINO_THROW("images must have shape [H, W, C] or [B, H, W, C]");
    }
    if (images.get_element_type() != ov::element::u8) {
        OPENVINO_THROW("images must be u8 for DeepseekOCR2 preprocessing");
    }

    const bool has_batch = shape.size() == 4;
    const size_t batch = has_batch ? shape[0] : 1;
    const size_t in_h = has_batch ? shape[1] : shape[0];
    const size_t in_w = has_batch ? shape[2] : shape[1];
    const size_t channels = has_batch ? shape[3] : shape[2];
    if (channels != 3) {
        OPENVINO_THROW("images must have 3 channels");
    }

    const uint8_t* src = images.data<const uint8_t>();

    const int32_t base_size = cfg_.crop_mode ? cfg_.base_size : cfg_.image_size;
    const size_t base_img_size = static_cast<size_t>(base_size);

    std::vector<float> global_data;
    global_data.reserve(batch * 3 * base_img_size * base_img_size);

    std::vector<float> local_data;
    std::vector<std::array<int32_t, 2>> spatial_crop;
    spatial_crop.reserve(batch);

    std::vector<DeepseekOCR2ImageTokens> image_tokens;
    image_tokens.reserve(batch);

    const auto ratios = build_target_ratios(cfg_.min_num, cfg_.max_num);

    for (size_t b = 0; b < batch; ++b) {
        const uint8_t* src_img = src + b * in_h * in_w * channels;
        size_t width = in_w;
        size_t height = in_h;

        bool use_crop = cfg_.crop_mode && (width > static_cast<size_t>(cfg_.image_size) ||
                                           height > static_cast<size_t>(cfg_.image_size));

        int32_t crop_w_num = 1;
        int32_t crop_h_num = 1;

        if (use_crop) {
            const double aspect = static_cast<double>(width) / static_cast<double>(height);
            auto ratio = find_closest_aspect_ratio(aspect, ratios, width, height, cfg_.image_size);
            crop_w_num = ratio.first;
            crop_h_num = ratio.second;

            const size_t target_w = static_cast<size_t>(cfg_.image_size) * static_cast<size_t>(crop_w_num);
            const size_t target_h = static_cast<size_t>(cfg_.image_size) * static_cast<size_t>(crop_h_num);

            auto resized = resize_bilinear_hwc(src_img, height, width, channels, target_h, target_w);
            const size_t crop_size = static_cast<size_t>(cfg_.image_size);

            const size_t crop_count = static_cast<size_t>(crop_w_num) * static_cast<size_t>(crop_h_num);
            local_data.reserve(local_data.size() + crop_count * channels * crop_size * crop_size);

            for (size_t i = 0; i < crop_count; ++i) {
                const size_t crop_x = (i % static_cast<size_t>(crop_w_num)) * crop_size;
                const size_t crop_y = (i / static_cast<size_t>(crop_w_num)) * crop_size;
                std::vector<float> crop(channels * crop_size * crop_size);
                crop_hwc_to_chw_normalized(resized.data,
                                           resized.height,
                                           resized.width,
                                           channels,
                                           crop_x,
                                           crop_y,
                                           crop_size,
                                           crop_size,
                                           cfg_.image_mean,
                                           cfg_.image_std,
                                           crop.data());
                local_data.insert(local_data.end(), crop.begin(), crop.end());
            }

            if (cfg_.use_thumbnail && crop_count != 1) {
                auto thumb = resize_bilinear_hwc(src_img, height, width, channels, crop_size, crop_size);
                std::vector<float> crop(channels * crop_size * crop_size);
                hwc_to_chw_normalized(thumb.data, crop_size, crop_size, channels, cfg_.image_mean, cfg_.image_std, crop.data());
                local_data.insert(local_data.end(), crop.begin(), crop.end());
            }
        }

        auto global_chw = pad_and_resize_to_chw(src_img,
                                                height,
                                                width,
                                                channels,
                                                base_img_size,
                                                cfg_.image_mean,
                                                cfg_.image_std);
        global_data.insert(global_data.end(), global_chw.begin(), global_chw.end());

        spatial_crop.push_back({crop_w_num, crop_h_num});

        DeepseekOCR2ImageTokens tokens;
        tokens.base_tokens = base_tokens(base_size, cfg_.patch_size, cfg_.downsample_ratio);
        if (use_crop && (crop_w_num > 1 || crop_h_num > 1)) {
            tokens.local_tokens = local_tokens(cfg_.image_size,
                                              cfg_.patch_size,
                                              cfg_.downsample_ratio,
                                              crop_w_num,
                                              crop_h_num);
        }
        image_tokens.push_back(tokens);
    }

    ov::Tensor global_tensor(ov::element::f32, {batch, 3, base_img_size, base_img_size});
    std::memcpy(global_tensor.data(), global_data.data(), global_data.size() * sizeof(float));

    ov::Tensor local_tensor;
    if (local_data.empty()) {
        local_tensor = ov::Tensor(ov::element::f32, {1, 3, base_img_size, base_img_size});
        std::memset(local_tensor.data(), 0, local_tensor.get_byte_size());
    } else {
        const size_t crop_size = static_cast<size_t>(cfg_.image_size);
        const size_t crop_count = local_data.size() / (3 * crop_size * crop_size);
        local_tensor = ov::Tensor(ov::element::f32, {crop_count, 3, crop_size, crop_size});
        std::memcpy(local_tensor.data(), local_data.data(), local_data.size() * sizeof(float));
    }

    ov::Tensor crop_tensor(ov::element::i64, {batch, 2});
    auto* crop_ptr = crop_tensor.data<int64_t>();
    for (size_t i = 0; i < spatial_crop.size(); ++i) {
        crop_ptr[i * 2 + 0] = spatial_crop[i][0];
        crop_ptr[i * 2 + 1] = spatial_crop[i][1];
    }

    return {global_tensor, local_tensor, crop_tensor, image_tokens};
}

DeepseekOCR2PromptPlan build_prompt_plan_from_tokens(const std::vector<std::vector<int64_t>>& text_segments,
                                                     const std::vector<DeepseekOCR2ImageTokens>& image_tokens,
                                                     int64_t image_token_id,
                                                     bool add_bos,
                                                     int64_t bos_id,
                                                     bool add_eos,
                                                     int64_t eos_id) {
    if (text_segments.size() != image_tokens.size() + 1) {
        OPENVINO_THROW("text_segments size must equal image_tokens + 1");
    }

    std::vector<int64_t> ids;
    std::vector<char> mask;

    if (add_bos) {
        ids.push_back(bos_id);
        mask.push_back(0);
    }

    for (size_t i = 0; i < image_tokens.size(); ++i) {
        const auto& seg = text_segments[i];
        ids.insert(ids.end(), seg.begin(), seg.end());
        mask.insert(mask.end(), seg.size(), 0);

        const int64_t count = image_tokens[i].total_tokens();
        ids.insert(ids.end(), static_cast<size_t>(count), image_token_id);
        mask.insert(mask.end(), static_cast<size_t>(count), 1);
    }

    const auto& last = text_segments.back();
    ids.insert(ids.end(), last.begin(), last.end());
    mask.insert(mask.end(), last.size(), 0);

    if (add_eos) {
        ids.push_back(eos_id);
        mask.push_back(0);
    }

    std::vector<int64_t> attention(ids.size(), 1);

    auto input_ids = make_i64_tensor(ids, {1, ids.size()});
    auto attention_mask = make_i64_tensor(attention, {1, attention.size()});
    auto images_seq_mask = make_bool_tensor(mask, {1, mask.size()});

    return {input_ids, attention_mask, images_seq_mask};
}

std::vector<std::string> split_prompt_by_image_token(const std::string& prompt,
                                                     const std::string& image_token) {
    std::vector<std::string> parts;
    if (image_token.empty()) {
        parts.push_back(prompt);
        return parts;
    }
    size_t start = 0;
    while (true) {
        size_t pos = prompt.find(image_token, start);
        if (pos == std::string::npos) {
            break;
        }
        parts.push_back(prompt.substr(start, pos - start));
        start = pos + image_token.size();
    }
    parts.push_back(prompt.substr(start));
    return parts;
}

DeepseekOCR2PromptPlan build_prompt_plan(ov::genai::Tokenizer& tokenizer,
                                         const std::string& prompt,
                                         const std::vector<DeepseekOCR2ImageTokens>& image_tokens,
                                         const DeepseekOCR2ProcessorConfig& processor_cfg,
                                         bool add_bos,
                                         std::optional<int64_t> bos_id,
                                         bool add_eos,
                                         std::optional<int64_t> eos_id) {
    const auto segments = split_prompt_by_image_token(prompt, processor_cfg.image_token);
    if (segments.size() != image_tokens.size() + 1) {
        OPENVINO_THROW("Prompt image tokens do not match image inputs");
    }

    std::vector<std::vector<int64_t>> token_segments;
    token_segments.reserve(segments.size());

    for (const auto& segment : segments) {
        auto encoded = tokenizer.encode(segment, ov::genai::add_special_tokens(false));
        auto ids = tensor_to_i64(encoded.input_ids);
        token_segments.push_back(std::move(ids));
    }

    const int64_t image_token_id = processor_cfg.resolve_image_token_id(tokenizer);
    const int64_t bos = bos_id.has_value() ? *bos_id : tokenizer.get_bos_token_id();
    const int64_t eos = eos_id.has_value() ? *eos_id : tokenizer.get_eos_token_id();

    return build_prompt_plan_from_tokens(token_segments,
                                         image_tokens,
                                         image_token_id,
                                         add_bos,
                                         bos,
                                         add_eos,
                                         eos);
}

DeepseekOCR2VisionPackager::DeepseekOCR2VisionPackager(const ov::Tensor& view_separator)
    : view_separator_(view_separator) {
}

const ov::Tensor& DeepseekOCR2VisionPackager::view_separator() const {
    return view_separator_;
}

std::vector<ov::Tensor> DeepseekOCR2VisionPackager::pack(
    const ov::Tensor& global_embeds,
    const ov::Tensor* local_embeds,
    const std::vector<DeepseekOCR2ImageTokens>& image_tokens) const {
    const auto global_shape = global_embeds.get_shape();
    if (global_shape.size() != 3) {
        OPENVINO_THROW("global_embeds must have shape [B, T, H]");
    }
    const size_t batch = global_shape[0];
    const size_t base_tokens = global_shape[1];
    const size_t hidden = global_shape[2];
    if (image_tokens.size() != batch) {
        OPENVINO_THROW("image_tokens size does not match global_embeds batch");
    }

    size_t total_local_tokens = 0;
    for (const auto& tokens : image_tokens) {
        if (tokens.base_tokens > 0 && static_cast<size_t>(tokens.base_tokens) != base_tokens) {
            OPENVINO_THROW("base_tokens mismatch between image_tokens and global_embeds");
        }
        if (tokens.local_tokens < 0) {
            OPENVINO_THROW("local_tokens must be non-negative");
        }
        total_local_tokens += static_cast<size_t>(tokens.local_tokens);
    }

    const ov::Tensor* local_ptr = local_embeds;
    if (total_local_tokens == 0) {
        local_ptr = nullptr;
    }
    if (!local_ptr && total_local_tokens > 0) {
        OPENVINO_THROW("local_embeds is required when local_tokens are present");
    }

    size_t tokens_per_crop = 0;
    if (local_ptr) {
        const auto local_shape = local_ptr->get_shape();
        if (local_shape.size() != 3) {
            OPENVINO_THROW("local_embeds must have shape [N, T, H]");
        }
        if (local_shape[2] != hidden) {
            OPENVINO_THROW("local_embeds hidden size mismatch");
        }
        tokens_per_crop = local_shape[1];
        if (tokens_per_crop == 0) {
            OPENVINO_THROW("local_embeds tokens_per_crop must be > 0");
        }
        const size_t local_total = local_shape[0] * local_shape[1];
        if (local_total != total_local_tokens) {
            OPENVINO_THROW("local_embeds length does not match image_tokens.local_tokens");
        }
    }

    const auto out_type = global_embeds.get_element_type();
    const size_t row_bytes = hidden * out_type.size();

    if (view_separator_.get_size() != hidden) {
        OPENVINO_THROW("view_separator must have size equal to hidden");
    }

    const void* view_ptr = view_separator_.data();
    std::vector<float> view_f32;
    std::vector<ov::float16> view_f16;
    std::vector<ov::bfloat16> view_bf16;
    if (view_separator_.get_element_type() != out_type) {
        if (out_type == ov::element::f32) {
            view_f32.resize(hidden);
            for (size_t i = 0; i < hidden; ++i) {
                view_f32[i] = read_as_f32(view_separator_.data(), i, view_separator_.get_element_type());
            }
            view_ptr = view_f32.data();
        } else if (out_type == ov::element::f16) {
            view_f16.resize(hidden);
            for (size_t i = 0; i < hidden; ++i) {
                float value = read_as_f32(view_separator_.data(), i, view_separator_.get_element_type());
                view_f16[i] = ov::float16(value);
            }
            view_ptr = view_f16.data();
        } else if (out_type == ov::element::bf16) {
            view_bf16.resize(hidden);
            for (size_t i = 0; i < hidden; ++i) {
                float value = read_as_f32(view_separator_.data(), i, view_separator_.get_element_type());
                view_bf16[i] = ov::bfloat16(value);
            }
            view_ptr = view_bf16.data();
        } else {
            OPENVINO_THROW("Unsupported output dtype for view_separator cast");
        }
    }

    const char* global_data = static_cast<const char*>(global_embeds.data());
    const char* local_data = local_ptr ? static_cast<const char*>(local_ptr->data()) : nullptr;

    std::vector<ov::Tensor> outputs;
    outputs.reserve(batch);

    size_t local_row_offset = 0;
    for (size_t b = 0; b < batch; ++b) {
        const size_t local_tokens = static_cast<size_t>(image_tokens[b].local_tokens);
        if (local_ptr && local_tokens > 0 && tokens_per_crop > 0 && (local_tokens % tokens_per_crop != 0)) {
            OPENVINO_THROW("local_tokens must be divisible by tokens_per_crop");
        }
        const size_t total_rows = local_tokens + base_tokens + 1;

        ov::Tensor packed(out_type, {total_rows, hidden});
        char* dst = static_cast<char*>(packed.data());
        size_t offset_bytes = 0;

        if (local_data && local_tokens > 0) {
            const size_t local_bytes = local_tokens * row_bytes;
            std::memcpy(dst + offset_bytes, local_data + local_row_offset * row_bytes, local_bytes);
            offset_bytes += local_bytes;
            local_row_offset += local_tokens;
        }

        const size_t global_bytes = base_tokens * row_bytes;
        const size_t global_offset = b * base_tokens * row_bytes;
        std::memcpy(dst + offset_bytes, global_data + global_offset, global_bytes);
        offset_bytes += global_bytes;

        std::memcpy(dst + offset_bytes, view_ptr, row_bytes);

        outputs.push_back(std::move(packed));
    }

    return outputs;
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
