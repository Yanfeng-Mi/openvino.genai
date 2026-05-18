// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ov {
class Model;
}  // namespace ov

namespace ov {
namespace genai {
namespace modeling {
namespace weights {
class WeightFinalizer;
class WeightSource;
}  // namespace weights
}  // namespace modeling
}  // namespace genai
}  // namespace ov

#include "modeling/builder_context.hpp"
#include "modeling/layers/rms_norm.hpp"
#include "modeling/models/deepseek_ocr2/processing_deepseek_ocr2.hpp"
#include "modeling/module.hpp"
#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

struct DeepseekQwen2VisionIO {
    static constexpr const char* kVisionFeats = "vision_feats";
    static constexpr const char* kQueryEmbeds = "query_embeds";
    static constexpr const char* kQueryFeats = "query_feats";
};

struct DeepseekQwen2VisionConfig {
    int32_t hidden_size = 896;
    int32_t num_hidden_layers = 24;
    int32_t num_attention_heads = 14;
    int32_t num_key_value_heads = 2;
    int32_t intermediate_size = 4864;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    std::string hidden_act = "silu";
    bool attention_bias = true;

    int32_t head_dim() const;
};

Tensor build_qwen2_attention_mask(const Tensor& token_type_ids, float masked_value = -65504.0f);

class DeepseekQwen2Attention : public Module {
public:
    DeepseekQwen2Attention(BuilderContext& ctx,
                           const std::string& name,
                           const DeepseekQwen2VisionConfig& cfg,
                           Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states,
                   const Tensor& rope_cos,
                   const Tensor& rope_sin,
                   const Tensor& attn_mask) const;

private:
    const Tensor& q_proj_weight() const;
    const Tensor& k_proj_weight() const;
    const Tensor& v_proj_weight() const;
    const Tensor& o_proj_weight() const;

    const Tensor* q_proj_bias() const;
    const Tensor* k_proj_bias() const;
    const Tensor* v_proj_bias() const;
    const Tensor* o_proj_bias() const;

    int32_t num_heads_ = 0;
    int32_t num_kv_heads_ = 0;
    int32_t head_dim_ = 0;
    int32_t hidden_size_ = 0;
    float scaling_ = 1.0f;

    WeightParameter* q_proj_param_ = nullptr;
    WeightParameter* k_proj_param_ = nullptr;
    WeightParameter* v_proj_param_ = nullptr;
    WeightParameter* o_proj_param_ = nullptr;

    WeightParameter* q_bias_param_ = nullptr;
    WeightParameter* k_bias_param_ = nullptr;
    WeightParameter* v_bias_param_ = nullptr;
    WeightParameter* o_bias_param_ = nullptr;
};

class DeepseekQwen2MLP : public Module {
public:
    DeepseekQwen2MLP(BuilderContext& ctx,
                     const std::string& name,
                     const DeepseekQwen2VisionConfig& cfg,
                     Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states) const;

private:
    const Tensor& gate_proj_weight() const;
    const Tensor& up_proj_weight() const;
    const Tensor& down_proj_weight() const;

    WeightParameter* gate_proj_param_ = nullptr;
    WeightParameter* up_proj_param_ = nullptr;
    WeightParameter* down_proj_param_ = nullptr;
};

class DeepseekQwen2DecoderLayer : public Module {
public:
    DeepseekQwen2DecoderLayer(BuilderContext& ctx,
                              const std::string& name,
                              const DeepseekQwen2VisionConfig& cfg,
                              Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states,
                   const Tensor& rope_cos,
                   const Tensor& rope_sin,
                   const Tensor& attn_mask) const;

private:
    DeepseekQwen2Attention self_attn_;
    DeepseekQwen2MLP mlp_;
    RMSNorm input_layernorm_;
    RMSNorm post_attention_layernorm_;
};

class DeepseekQwen2VisionModel : public Module {
public:
    DeepseekQwen2VisionModel(BuilderContext& ctx,
                             const DeepseekQwen2VisionConfig& cfg,
                             Module* parent = nullptr);

    Tensor forward(const Tensor& vision_feats, const Tensor& query_embeds);

private:
    Tensor build_positions(const Tensor& token_type_ids) const;
    Tensor slice_query(const Tensor& hidden_states, const Tensor& start, const Tensor& end) const;
    Tensor build_token_type_ids(const Tensor& vision_flat, const Tensor& query_embeds) const;

    DeepseekQwen2VisionConfig cfg_;
    std::vector<DeepseekQwen2DecoderLayer> layers_;
    RMSNorm norm_;
};

std::shared_ptr<ov::Model> create_deepseek_qwen2_encoder_model(
    const DeepseekOCR2VisionConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
