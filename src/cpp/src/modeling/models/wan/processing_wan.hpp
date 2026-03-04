// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <openvino/runtime/tensor.hpp>

#include "openvino/genai/tokenizer.hpp"
#include "modeling/module.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

struct WanTransformer3DConfig {
    std::string class_name = "WanTransformer3DModel";
    std::vector<std::string> architectures;
    std::vector<int32_t> patch_size = {1, 2, 2};
    int32_t num_attention_heads = 0;
    int32_t attention_head_dim = 0;
    int32_t in_channels = 16;
    int32_t out_channels = 16;
    int32_t text_dim = 4096;
    int32_t freq_dim = 256;
    int32_t ffn_dim = 0;
    int32_t num_layers = 0;
    bool cross_attn_norm = true;
    std::string qk_norm = "rms_norm_across_heads";
    float eps = 1e-6f;
    std::optional<int32_t> image_dim;
    std::optional<int32_t> added_kv_proj_dim;
    int32_t rope_max_seq_len = 1024;
    std::optional<int32_t> pos_embed_seq_len;

    int32_t inner_dim() const;
    int32_t patch_volume() const;
    bool use_image_condition() const;
    void finalize();
    void validate() const;

    static WanTransformer3DConfig from_json(const nlohmann::json& data);
    static WanTransformer3DConfig from_json_file(const std::filesystem::path& config_path);
};

struct WanVAEConfig {
    std::string class_name = "AutoencoderKLWan";
    std::vector<std::string> architectures;
    int32_t base_dim = 96;
    int32_t decoder_base_dim = 0;
    int32_t z_dim = 16;
    std::vector<int32_t> dim_mult = {1, 2, 4, 4};
    int32_t num_res_blocks = 2;
    std::vector<float> attn_scales;
    std::vector<bool> temperal_downsample = {false, true, true};
    float dropout = 0.0f;
    std::vector<float> latents_mean;
    std::vector<float> latents_std;
    bool is_residual = false;
    int32_t in_channels = 3;
    int32_t out_channels = 3;
    std::optional<int32_t> patch_size;
    int32_t scale_factor_temporal = 4;
    int32_t scale_factor_spatial = 8;

    void finalize();
    void validate() const;

    static WanVAEConfig from_json(const nlohmann::json& data);
    static WanVAEConfig from_json_file(const std::filesystem::path& config_path);
};

struct WanTextInputs {
    ov::Tensor input_ids;
    ov::Tensor attention_mask;
};

std::string prompt_clean(const std::string& text);
std::vector<std::string> prompt_clean(const std::vector<std::string>& texts);

WanTextInputs tokenize_prompts(ov::genai::Tokenizer& tokenizer,
                               const std::vector<std::string>& prompts,
                               int32_t max_sequence_length,
                               bool add_special_tokens = true);

ov::Tensor prepare_latents(size_t batch,
                           size_t channels,
                           size_t frames,
                           size_t height,
                           size_t width,
                           int32_t seed);

std::vector<float> apply_cfg(const std::vector<float>& noise_pred,
                             const std::vector<float>& noise_pred_uncond,
                             float guidance_scale);

struct WanWeightMapping {
    static void apply_transformer_packed_mapping(ov::genai::modeling::Module& model);
    static void apply_vae_packed_mapping(ov::genai::modeling::Module& model);
};

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
