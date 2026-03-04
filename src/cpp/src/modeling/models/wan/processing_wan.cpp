// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/wan/processing_wan.hpp"

#include <algorithm>
#include <fstream>
#include <random>
#include <regex>

#include <openvino/core/except.hpp>

#include "json_utils.hpp"

namespace {

void read_config_json_file(const std::filesystem::path& path, nlohmann::json& data) {
    std::ifstream file(path);
    if (!file.is_open()) {
        OPENVINO_THROW("Failed to open config file: ", path.string());
    }
    file >> data;
}

std::filesystem::path resolve_config_path(const std::filesystem::path& path) {
    if (std::filesystem::is_directory(path)) {
        return path / "config.json";
    }
    return path;
}

void replace_all(std::string& text, const std::string& from, const std::string& to) {
    if (from.empty()) {
        return;
    }
    size_t start = 0;
    while ((start = text.find(from, start)) != std::string::npos) {
        text.replace(start, from.size(), to);
        start += to.size();
    }
}

std::string html_unescape(std::string text) {
    replace_all(text, "&lt;", "<");
    replace_all(text, "&gt;", ">");
    replace_all(text, "&quot;", "\"");
    replace_all(text, "&#39;", "'");
    replace_all(text, "&nbsp;", " ");
    replace_all(text, "&amp;", "&");
    return text;
}

std::string trim(const std::string& text) {
    auto start = text.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return std::string();
    }
    auto end = text.find_last_not_of(" \t\r\n");
    return text.substr(start, end - start + 1);
}

template <typename T>
void fill_row(ov::Tensor& tensor, size_t row, T pad_value, const T* src, size_t count) {
    auto* data = tensor.data<T>();
    size_t stride = tensor.get_shape().at(1);
    T* row_ptr = data + row * stride;
    std::fill_n(row_ptr, stride, pad_value);
    if (src && count > 0) {
        std::copy_n(src, count, row_ptr);
    }
}

void parse_transformer_config(const nlohmann::json& data, ov::genai::modeling::models::WanTransformer3DConfig& cfg) {
    using ov::genai::utils::read_json_param;
    read_json_param(data, "_class_name", cfg.class_name);
    read_json_param(data, "architectures", cfg.architectures);
    read_json_param(data, "patch_size", cfg.patch_size);
    read_json_param(data, "num_attention_heads", cfg.num_attention_heads);
    read_json_param(data, "attention_head_dim", cfg.attention_head_dim);
    read_json_param(data, "in_channels", cfg.in_channels);
    read_json_param(data, "out_channels", cfg.out_channels);
    read_json_param(data, "text_dim", cfg.text_dim);
    read_json_param(data, "freq_dim", cfg.freq_dim);
    read_json_param(data, "ffn_dim", cfg.ffn_dim);
    read_json_param(data, "num_layers", cfg.num_layers);
    read_json_param(data, "cross_attn_norm", cfg.cross_attn_norm);
    read_json_param(data, "qk_norm", cfg.qk_norm);
    read_json_param(data, "eps", cfg.eps);
    read_json_param(data, "image_dim", cfg.image_dim);
    read_json_param(data, "added_kv_proj_dim", cfg.added_kv_proj_dim);
    read_json_param(data, "rope_max_seq_len", cfg.rope_max_seq_len);
    read_json_param(data, "pos_embed_seq_len", cfg.pos_embed_seq_len);

    cfg.finalize();
    cfg.validate();
}

void parse_vae_config(const nlohmann::json& data, ov::genai::modeling::models::WanVAEConfig& cfg) {
    using ov::genai::utils::read_json_param;
    read_json_param(data, "_class_name", cfg.class_name);
    read_json_param(data, "architectures", cfg.architectures);
    read_json_param(data, "base_dim", cfg.base_dim);
    read_json_param(data, "decoder_base_dim", cfg.decoder_base_dim);
    read_json_param(data, "z_dim", cfg.z_dim);
    read_json_param(data, "dim_mult", cfg.dim_mult);
    read_json_param(data, "num_res_blocks", cfg.num_res_blocks);
    read_json_param(data, "attn_scales", cfg.attn_scales);
    read_json_param(data, "temperal_downsample", cfg.temperal_downsample);
    read_json_param(data, "dropout", cfg.dropout);
    read_json_param(data, "latents_mean", cfg.latents_mean);
    read_json_param(data, "latents_std", cfg.latents_std);
    read_json_param(data, "is_residual", cfg.is_residual);
    read_json_param(data, "in_channels", cfg.in_channels);
    read_json_param(data, "out_channels", cfg.out_channels);
    read_json_param(data, "patch_size", cfg.patch_size);
    read_json_param(data, "scale_factor_temporal", cfg.scale_factor_temporal);
    read_json_param(data, "scale_factor_spatial", cfg.scale_factor_spatial);

    cfg.finalize();
    cfg.validate();
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

int32_t WanTransformer3DConfig::inner_dim() const {
    return num_attention_heads > 0 ? num_attention_heads * attention_head_dim : 0;
}

int32_t WanTransformer3DConfig::patch_volume() const {
    if (patch_size.empty()) {
        return 0;
    }
    int32_t volume = 1;
    for (int32_t v : patch_size) {
        volume *= v;
    }
    return volume;
}

bool WanTransformer3DConfig::use_image_condition() const {
    return image_dim.has_value();
}

void WanTransformer3DConfig::finalize() {
    if (out_channels <= 0) {
        out_channels = in_channels;
    }
    if (patch_size.empty()) {
        patch_size = {1, 2, 2};
    }
}

void WanTransformer3DConfig::validate() const {
    if (patch_size.size() != 3) {
        OPENVINO_THROW("WanTransformer3DConfig.patch_size must have 3 elements");
    }
    for (int32_t v : patch_size) {
        if (v <= 0) {
            OPENVINO_THROW("WanTransformer3DConfig.patch_size values must be > 0");
        }
    }
    if (num_attention_heads <= 0) {
        OPENVINO_THROW("WanTransformer3DConfig.num_attention_heads must be > 0");
    }
    if (attention_head_dim <= 0) {
        OPENVINO_THROW("WanTransformer3DConfig.attention_head_dim must be > 0");
    }
    if (num_layers <= 0) {
        OPENVINO_THROW("WanTransformer3DConfig.num_layers must be > 0");
    }
    if (ffn_dim <= 0) {
        OPENVINO_THROW("WanTransformer3DConfig.ffn_dim must be > 0");
    }
    if (in_channels <= 0 || out_channels <= 0) {
        OPENVINO_THROW("WanTransformer3DConfig.in_channels/out_channels must be > 0");
    }
    if (text_dim <= 0) {
        OPENVINO_THROW("WanTransformer3DConfig.text_dim must be > 0");
    }
    if (freq_dim <= 0) {
        OPENVINO_THROW("WanTransformer3DConfig.freq_dim must be > 0");
    }
    if (rope_max_seq_len <= 0) {
        OPENVINO_THROW("WanTransformer3DConfig.rope_max_seq_len must be > 0");
    }
    if (inner_dim() <= 0) {
        OPENVINO_THROW("WanTransformer3DConfig.inner_dim must be > 0");
    }
}

WanTransformer3DConfig WanTransformer3DConfig::from_json(const nlohmann::json& data) {
    WanTransformer3DConfig cfg;
    parse_transformer_config(data, cfg);
    return cfg;
}

WanTransformer3DConfig WanTransformer3DConfig::from_json_file(const std::filesystem::path& config_path) {
    nlohmann::json data;
    read_config_json_file(resolve_config_path(config_path), data);
    return from_json(data);
}

void WanVAEConfig::finalize() {
    if (decoder_base_dim <= 0) {
        decoder_base_dim = base_dim;
    }
}

void WanVAEConfig::validate() const {
    if (base_dim <= 0 || decoder_base_dim <= 0) {
        OPENVINO_THROW("WanVAEConfig.base_dim/decoder_base_dim must be > 0");
    }
    if (z_dim <= 0) {
        OPENVINO_THROW("WanVAEConfig.z_dim must be > 0");
    }
    if (dim_mult.empty()) {
        OPENVINO_THROW("WanVAEConfig.dim_mult must not be empty");
    }
    if (num_res_blocks <= 0) {
        OPENVINO_THROW("WanVAEConfig.num_res_blocks must be > 0");
    }
    if (!latents_mean.empty() && static_cast<int32_t>(latents_mean.size()) != z_dim) {
        OPENVINO_THROW("WanVAEConfig.latents_mean size must match z_dim");
    }
    if (!latents_std.empty() && static_cast<int32_t>(latents_std.size()) != z_dim) {
        OPENVINO_THROW("WanVAEConfig.latents_std size must match z_dim");
    }
    if (!temperal_downsample.empty() && dim_mult.size() > 1 &&
        temperal_downsample.size() != dim_mult.size() - 1) {
        OPENVINO_THROW("WanVAEConfig.temperal_downsample size must match dim_mult.size() - 1");
    }
    if (in_channels <= 0 || out_channels <= 0) {
        OPENVINO_THROW("WanVAEConfig.in_channels/out_channels must be > 0");
    }
    if (patch_size.has_value() && patch_size.value() <= 0) {
        OPENVINO_THROW("WanVAEConfig.patch_size must be > 0");
    }
    if (scale_factor_temporal <= 0 || scale_factor_spatial <= 0) {
        OPENVINO_THROW("WanVAEConfig.scale_factor_temporal/spatial must be > 0");
    }
}

WanVAEConfig WanVAEConfig::from_json(const nlohmann::json& data) {
    WanVAEConfig cfg;
    parse_vae_config(data, cfg);
    return cfg;
}

WanVAEConfig WanVAEConfig::from_json_file(const std::filesystem::path& config_path) {
    nlohmann::json data;
    read_config_json_file(resolve_config_path(config_path), data);
    return from_json(data);
}

std::string prompt_clean(const std::string& text) {
    std::string cleaned = html_unescape(html_unescape(text));
    cleaned = trim(cleaned);
    cleaned = std::regex_replace(cleaned, std::regex(R"(\s+)"), " ");
    return cleaned;
}

std::vector<std::string> prompt_clean(const std::vector<std::string>& texts) {
    std::vector<std::string> cleaned;
    cleaned.reserve(texts.size());
    for (const auto& text : texts) {
        cleaned.push_back(prompt_clean(text));
    }
    return cleaned;
}

WanTextInputs tokenize_prompts(ov::genai::Tokenizer& tokenizer,
                               const std::vector<std::string>& prompts,
                               int32_t max_sequence_length,
                               bool add_special_tokens) {
    if (max_sequence_length <= 0) {
        OPENVINO_THROW("max_sequence_length must be > 0");
    }
    auto cleaned_prompts = prompt_clean(prompts);
    auto tokenization = tokenizer.encode(cleaned_prompts, ov::genai::add_special_tokens(add_special_tokens));

    const auto& ids = tokenization.input_ids;
    const auto& mask = tokenization.attention_mask;
    const auto ids_type = ids.get_element_type();
    const auto mask_type = mask.get_element_type();

    size_t batch = cleaned_prompts.size();
    size_t max_len = static_cast<size_t>(max_sequence_length);

    ov::Tensor out_ids(ids_type, {batch, max_len});
    ov::Tensor out_mask(mask_type, {batch, max_len});

    int64_t pad_id = tokenizer.get_pad_token_id();

    for (size_t i = 0; i < batch; ++i) {
        auto ids_row = ov::Tensor(ids, {i, 0}, {i + 1, ids.get_shape().at(1)});
        auto mask_row = ov::Tensor(mask, {i, 0}, {i + 1, mask.get_shape().at(1)});
        size_t seq_len = ids_row.get_shape().at(1);
        size_t copy_len = std::min(seq_len, max_len);

        if (ids_type == ov::element::i32) {
            fill_row<int32_t>(out_ids, i, static_cast<int32_t>(pad_id),
                              ids_row.data<int32_t>(), copy_len);
        } else {
            fill_row<int64_t>(out_ids, i, static_cast<int64_t>(pad_id),
                              ids_row.data<int64_t>(), copy_len);
        }

        if (mask_type == ov::element::i32) {
            fill_row<int32_t>(out_mask, i, 0, mask_row.data<int32_t>(), copy_len);
        } else {
            fill_row<int64_t>(out_mask, i, 0, mask_row.data<int64_t>(), copy_len);
        }
    }

    return WanTextInputs{out_ids, out_mask};
}

ov::Tensor prepare_latents(size_t batch,
                           size_t channels,
                           size_t frames,
                           size_t height,
                           size_t width,
                           int32_t seed) {
    ov::Tensor latents(ov::element::f32, {batch, channels, frames, height, width});
    auto* data = latents.data<float>();

    std::mt19937 rng(seed < 0 ? std::random_device{}() : static_cast<uint32_t>(seed));
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < latents.get_size(); ++i) {
        data[i] = dist(rng);
    }

    return latents;
}

std::vector<float> apply_cfg(const std::vector<float>& noise_pred,
                             const std::vector<float>& noise_pred_uncond,
                             float guidance_scale) {
    if (noise_pred.size() != noise_pred_uncond.size()) {
        OPENVINO_THROW("CFG tensors size mismatch");
    }
    std::vector<float> out(noise_pred.size(), 0.0f);
    for (size_t i = 0; i < noise_pred.size(); ++i) {
        out[i] = noise_pred_uncond[i] + guidance_scale * (noise_pred[i] - noise_pred_uncond[i]);
    }
    return out;
}

void WanWeightMapping::apply_transformer_packed_mapping(ov::genai::modeling::Module& model) {
    auto& rules = model.packed_mapping().rules;
    rules.push_back({"transformer.", "", 0});
    rules.push_back({"model.", "", 0});
}

void WanWeightMapping::apply_vae_packed_mapping(ov::genai::modeling::Module& model) {
    auto& rules = model.packed_mapping().rules;
    rules.push_back({"vae.", "", 0});
    rules.push_back({"model.", "", 0});
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
