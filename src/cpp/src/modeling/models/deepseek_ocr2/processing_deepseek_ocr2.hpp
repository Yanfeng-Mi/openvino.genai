// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <openvino/runtime/tensor.hpp>

#include "openvino/genai/tokenizer.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

struct DeepseekOCR2LanguageConfig {
    int32_t vocab_size = 0;
    int32_t hidden_size = 0;
    int32_t intermediate_size = 0;
    int32_t moe_intermediate_size = 0;
    int32_t num_hidden_layers = 0;
    int32_t num_attention_heads = 0;
    int32_t num_key_value_heads = 0;
    int32_t n_routed_experts = 0;
    int32_t n_shared_experts = 0;
    int32_t num_experts_per_tok = 0;
    int32_t moe_layer_freq = 1;
    int32_t first_k_dense_replace = 0;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    std::string hidden_act = "silu";
    bool attention_bias = false;
    float attention_dropout = 0.0f;
    bool use_mla = false;
    int32_t qk_rope_head_dim = 0;
    int32_t qk_nope_head_dim = 0;
    int32_t v_head_dim = 0;
    int32_t kv_lora_rank = 0;
    int32_t q_lora_rank = 0;
    int64_t bos_token_id = 0;
    int64_t eos_token_id = 1;
    int64_t max_position_embeddings = 0;

    int32_t resolved_kv_heads() const;
    int32_t resolved_q_head_dim() const;
    int32_t resolved_v_head_dim() const;
    void finalize();
    void validate() const;
};

struct DeepseekOCR2VisionSamConfig {
    int32_t width = 768;
    int32_t layers = 12;
    int32_t heads = 12;
    std::vector<int32_t> global_attn_indexes;
    std::vector<int32_t> downsample_channels;
};

struct DeepseekOCR2VisionQwen2Config {
    int32_t dim = 896;
};

struct DeepseekOCR2VisionConfig {
    int32_t image_size = 1024;
    float mlp_ratio = 4.0f;
    std::string model_name = "deepencoderv2";
    std::string model_type = "vision";
    DeepseekOCR2VisionSamConfig sam_vit_b;
    DeepseekOCR2VisionQwen2Config qwen2_0_5b;

    void finalize();
    void validate() const;
};

struct DeepseekOCR2ProjectorConfig {
    int32_t input_dim = 0;
    int32_t n_embed = 0;
    std::string projector_type = "linear";
    std::string model_type = "mlp_projector";

    void finalize();
    void validate() const;
};

struct DeepseekOCR2Config {
    std::string model_type = "deepseek_vl_v2";
    std::string tile_tag = "2D";
    std::string global_view_pos = "head";
    std::string torch_dtype = "bfloat16";
    std::string transformers_version;
    std::vector<std::string> architectures;
    std::vector<std::array<int32_t, 2>> candidate_resolutions;

    DeepseekOCR2LanguageConfig language;
    DeepseekOCR2VisionConfig vision;
    DeepseekOCR2ProjectorConfig projector;

    int64_t bos_token_id = 0;
    int64_t eos_token_id = 1;

    void finalize();
    void validate() const;

    static DeepseekOCR2Config from_json(const nlohmann::json& data);
    static DeepseekOCR2Config from_json_file(const std::filesystem::path& config_path);
};

struct DeepseekOCR2ProcessorConfig {
    std::vector<std::array<int32_t, 2>> candidate_resolutions;
    int32_t patch_size = 16;
    int32_t downsample_ratio = 4;
    std::array<float, 3> image_mean = {0.5f, 0.5f, 0.5f};
    std::array<float, 3> image_std = {0.5f, 0.5f, 0.5f};
    bool normalize = true;
    bool add_special_token = false;
    bool mask_prompt = false;
    std::string image_token = "<image>";
    std::string pad_token;
    std::string sft_format = "deepseek";
    int32_t ignore_id = -100;

    void finalize();
    void validate() const;

    int64_t resolve_image_token_id(const ov::genai::Tokenizer& tokenizer) const;

    static DeepseekOCR2ProcessorConfig from_json(const nlohmann::json& data);
    static DeepseekOCR2ProcessorConfig from_json_file(const std::filesystem::path& path);
};

struct DeepseekOCR2ImageTokens {
    int64_t base_tokens = 0;
    int64_t local_tokens = 0;
    int64_t total_tokens() const {
        return base_tokens + 1 + local_tokens;
    }
};

struct DeepseekOCR2PreprocessConfig {
    int32_t base_size = 1024;
    int32_t image_size = 768;
    int32_t min_num = 2;
    int32_t max_num = 6;
    bool crop_mode = true;
    bool use_thumbnail = false;
    int32_t patch_size = 16;
    int32_t downsample_ratio = 4;
    std::array<float, 3> image_mean = {0.5f, 0.5f, 0.5f};
    std::array<float, 3> image_std = {0.5f, 0.5f, 0.5f};

    void validate() const;
};

struct DeepseekOCR2ImageInputs {
    ov::Tensor global_images;       // [B, 3, base_size, base_size]
    ov::Tensor local_images;        // [N, 3, image_size, image_size] or placeholder
    ov::Tensor images_spatial_crop; // [B, 2] (width, height)
    std::vector<DeepseekOCR2ImageTokens> image_tokens;
};

class DeepseekOCR2ImagePreprocessor {
public:
    explicit DeepseekOCR2ImagePreprocessor(const DeepseekOCR2PreprocessConfig& cfg);

    DeepseekOCR2ImageInputs preprocess(const ov::Tensor& images) const;

    static int64_t num_queries(int32_t image_size, int32_t patch_size, int32_t downsample_ratio);
    static int64_t base_tokens(int32_t base_size, int32_t patch_size, int32_t downsample_ratio);
    static int64_t local_tokens(int32_t image_size,
                                int32_t patch_size,
                                int32_t downsample_ratio,
                                int32_t width_crop_num,
                                int32_t height_crop_num);

private:
    DeepseekOCR2PreprocessConfig cfg_;
};

struct DeepseekOCR2PromptPlan {
    ov::Tensor input_ids;
    ov::Tensor attention_mask;
    ov::Tensor images_seq_mask;
};

DeepseekOCR2PromptPlan build_prompt_plan_from_tokens(const std::vector<std::vector<int64_t>>& text_segments,
                                                     const std::vector<DeepseekOCR2ImageTokens>& image_tokens,
                                                     int64_t image_token_id,
                                                     bool add_bos,
                                                     int64_t bos_id,
                                                     bool add_eos = false,
                                                     int64_t eos_id = 1);

DeepseekOCR2PromptPlan build_prompt_plan(ov::genai::Tokenizer& tokenizer,
                                         const std::string& prompt,
                                         const std::vector<DeepseekOCR2ImageTokens>& image_tokens,
                                         const DeepseekOCR2ProcessorConfig& processor_cfg,
                                         bool add_bos = true,
                                         std::optional<int64_t> bos_id = std::nullopt,
                                         bool add_eos = false,
                                         std::optional<int64_t> eos_id = std::nullopt);

std::vector<std::string> split_prompt_by_image_token(const std::string& prompt,
                                                     const std::string& image_token);

class DeepseekOCR2VisionPackager {
public:
    explicit DeepseekOCR2VisionPackager(const ov::Tensor& view_separator);

    const ov::Tensor& view_separator() const;

    std::vector<ov::Tensor> pack(const ov::Tensor& global_embeds,
                                 const ov::Tensor* local_embeds,
                                 const std::vector<DeepseekOCR2ImageTokens>& image_tokens) const;

private:
    ov::Tensor view_separator_;
};

struct DeepseekOCR2WeightNames {
    static constexpr const char* kSamPrefix = "model.sam_model.";
    static constexpr const char* kQwen2Prefix = "model.qwen2_model.model.model.";
    static constexpr const char* kProjectorPrefix = "model.projector.";
    static constexpr const char* kQuery768 = "model.qwen2_model.query_768.weight";
    static constexpr const char* kQuery1024 = "model.qwen2_model.query_1024.weight";
    static constexpr const char* kProjectorWeight = "model.projector.layers.weight";
    static constexpr const char* kProjectorBias = "model.projector.layers.bias";
    static constexpr const char* kViewSeparator = "model.view_seperator";
    static constexpr const char* kEmbedTokens = "model.embed_tokens.weight";
    static constexpr const char* kNormWeight = "model.norm.weight";
    static constexpr const char* kLmHeadWeight = "lm_head.weight";
};

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov

