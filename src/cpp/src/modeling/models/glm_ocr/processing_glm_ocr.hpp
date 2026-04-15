// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <openvino/openvino.hpp>
#include <nlohmann/json.hpp>

namespace ov {
namespace genai {
namespace modeling {
namespace models {

struct GlmOcrRopeConfig {
    std::vector<int32_t> mrope_section = {16, 24, 24};
    std::string rope_type = "default";
    float rope_theta = 10000.0f;
};

struct GlmOcrTextConfig {
    std::string model_type = "glm_ocr_text";
    int32_t vocab_size = 59392;
    int32_t hidden_size = 1536;
    int32_t intermediate_size = 4608;
    int32_t num_hidden_layers = 16;
    int32_t num_attention_heads = 16;
    int32_t num_key_value_heads = 8;
    int32_t head_dim = 128;
    int32_t max_position_embeddings = 131072;
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    std::string hidden_act = "silu";
    bool attention_bias = false;
    bool tie_word_embeddings = false;
    std::string dtype = "bfloat16";
    GlmOcrRopeConfig rope;

    int32_t kv_heads() const;
    int32_t resolved_head_dim() const;
    void finalize();
    void validate() const;
};

struct GlmOcrVisionConfig {
    std::string model_type = "glm_ocr_vision";
    int32_t depth = 24;
    int32_t hidden_size = 1024;
    std::string hidden_act = "silu";
    int32_t intermediate_size = 4096;
    int32_t num_heads = 16;
    int32_t in_channels = 3;
    int32_t patch_size = 14;
    int32_t spatial_merge_size = 2;
    int32_t temporal_patch_size = 2;
    int32_t out_hidden_size = 1536;
    float rms_norm_eps = 1e-5f;
    bool attention_bias = true;

    int32_t head_dim() const;
    void finalize();
    void validate() const;
};

struct GlmOcrConfig {
    std::string model_type = "glm_ocr";
    std::vector<std::string> architectures;
    GlmOcrTextConfig text;
    GlmOcrVisionConfig vision;
    int32_t image_token_id = 59280;
    int32_t image_start_token_id = 59256;
    int32_t image_end_token_id = 59257;
    bool tie_word_embeddings = false;
    std::vector<int32_t> eos_token_ids = {59246, 59253};

    void finalize();
    void validate() const;

    static GlmOcrConfig from_json(const nlohmann::json& data);
    static GlmOcrConfig from_json_file(const std::filesystem::path& config_path);
};

struct GlmOcrModuleNames {
    static constexpr const char* kRoot = "model";
    static constexpr const char* kVision = "visual";
    static constexpr const char* kText = "language_model";
    static constexpr const char* kLmHead = "lm_head";
    static std::string vision_block(int32_t index);
    static std::string text_layer(int32_t index);
};

struct GlmOcrVisionIO {
    static constexpr const char* kPixelValues = "pixel_values";
    static constexpr const char* kGridThw = "grid_thw";
    static constexpr const char* kRotaryCos = "rotary_cos";
    static constexpr const char* kRotarySin = "rotary_sin";
    static constexpr const char* kVisualEmbeds = "visual_embeds";
};

struct GlmOcrTextIO {
    static constexpr const char* kInputIds = "input_ids";
    static constexpr const char* kAttentionMask = "attention_mask";
    static constexpr const char* kPositionIds = "position_ids";
    static constexpr const char* kBeamIdx = "beam_idx";
    static constexpr const char* kVisualEmbeds = "visual_embeds";
    static constexpr const char* kVisualPosMask = "visual_pos_mask";
    static constexpr const char* kLogits = "logits";
};

struct GlmOcrInputPlan {
    ov::Tensor position_ids;
    ov::Tensor visual_pos_mask;
    ov::Tensor rope_deltas;
};

class GlmOcrInputPlanner {
public:
    explicit GlmOcrInputPlanner(const GlmOcrConfig& cfg);

    GlmOcrInputPlan build_plan(const ov::Tensor& input_ids,
                               const ov::Tensor* attention_mask = nullptr,
                               const ov::Tensor* image_grid_thw = nullptr) const;

    ov::Tensor build_visual_pos_mask(const ov::Tensor& input_ids,
                                     const ov::Tensor* attention_mask = nullptr) const;

    static ov::Tensor scatter_visual_embeds(const ov::Tensor& visual_embeds,
                                            const ov::Tensor& visual_pos_mask);

    static ov::Tensor build_decode_position_ids(const ov::Tensor& rope_deltas,
                                                int64_t past_length,
                                                int64_t seq_len);

private:
    int64_t image_token_id_ = 0;
    int32_t spatial_merge_size_ = 1;
};

struct GlmOcrVisionPreprocessConfig {
    int64_t min_pixels = 56 * 56;
    int64_t max_pixels = 28 * 28 * 1280;
    int32_t patch_size = 14;
    int32_t temporal_patch_size = 2;
    int32_t merge_size = 2;
    std::array<float, 3> image_mean = {0.48145466f, 0.4578275f, 0.40821073f};
    std::array<float, 3> image_std = {0.26862954f, 0.26130258f, 0.27577711f};
    bool do_resize = true;

    static GlmOcrVisionPreprocessConfig from_json_file(const std::filesystem::path& path);
};

struct GlmOcrVisionInputs {
    ov::Tensor pixel_values;
    ov::Tensor grid_thw;
    ov::Tensor rotary_cos;
    ov::Tensor rotary_sin;
};

class GlmOcrVisionPreprocessor {
public:
    GlmOcrVisionPreprocessor(const GlmOcrVisionConfig& vision_cfg,
                             const GlmOcrVisionPreprocessConfig& preprocess_cfg);

    GlmOcrVisionInputs preprocess(const ov::Tensor& images) const;

    static int64_t count_visual_tokens(const ov::Tensor& grid_thw,
                                       int32_t spatial_merge_size);

private:
    GlmOcrVisionConfig vision_cfg_;
    GlmOcrVisionPreprocessConfig preprocess_cfg_;
};

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
