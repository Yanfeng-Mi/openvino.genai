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
#include "modeling/layers/lm_head.hpp"
#include "modeling/layers/rms_norm.hpp"
#include "modeling/layers/vocab_embedding.hpp"
#include "modeling/models/glm_ocr/processing_glm_ocr.hpp"
#include "modeling/module.hpp"
#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

// Reuse EmbeddingInjector from qwen3_vl (same masked_scatter pattern)
class GlmOcrEmbeddingInjector : public Module {
public:
    GlmOcrEmbeddingInjector(BuilderContext& ctx, const std::string& name, Module* parent = nullptr);

    Tensor forward(const Tensor& inputs_embeds,
                   const Tensor& visual_embeds,
                   const Tensor& visual_pos_mask) const;
};

class GlmOcrTextAttention : public Module {
public:
    GlmOcrTextAttention(BuilderContext& ctx,
                        const std::string& name,
                        const GlmOcrTextConfig& cfg,
                        Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states,
                   const Tensor& beam_idx,
                   const Tensor& rope_cos,
                   const Tensor& rope_sin) const;

private:
    const Tensor& q_proj_weight() const;
    const Tensor& k_proj_weight() const;
    const Tensor& v_proj_weight() const;
    const Tensor& o_proj_weight() const;

    std::pair<Tensor, Tensor> append_kv_cache(const Tensor& keys,
                                              const Tensor& values,
                                              const Tensor& beam_idx) const;

    int32_t num_heads_ = 0;
    int32_t num_kv_heads_ = 0;
    int32_t head_dim_ = 0;
    int32_t hidden_size_ = 0;
    float scaling_ = 1.0f;

    WeightParameter* q_proj_param_ = nullptr;
    WeightParameter* k_proj_param_ = nullptr;
    WeightParameter* v_proj_param_ = nullptr;
    WeightParameter* o_proj_param_ = nullptr;
};

// Combined gate_up_proj: [9216, 1536] -> slice -> silu(gate) * up -> down_proj
class GlmOcrTextMLP : public Module {
public:
    GlmOcrTextMLP(BuilderContext& ctx, const std::string& name, const GlmOcrTextConfig& cfg,
                  Module* parent = nullptr);

    Tensor forward(const Tensor& x) const;

private:
    const Tensor& gate_up_proj_weight() const;
    const Tensor& down_proj_weight() const;

    int32_t intermediate_size_ = 0;

    WeightParameter* gate_up_proj_param_ = nullptr;
    WeightParameter* down_proj_param_ = nullptr;
};

// 4 norms per layer: input_layernorm, post_self_attn_layernorm, post_attention_layernorm, post_mlp_layernorm
class GlmOcrTextDecoderLayer : public Module {
public:
    GlmOcrTextDecoderLayer(BuilderContext& ctx,
                           const std::string& name,
                           const GlmOcrTextConfig& cfg,
                           Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states,
                   const Tensor& beam_idx,
                   const Tensor& rope_cos,
                   const Tensor& rope_sin) const;

private:
    GlmOcrTextAttention self_attn_;
    GlmOcrTextMLP mlp_;
    RMSNorm input_layernorm_;
    RMSNorm post_self_attn_layernorm_;
    RMSNorm post_attention_layernorm_;
    RMSNorm post_mlp_layernorm_;
};

class GlmOcrTextModel : public Module {
public:
    GlmOcrTextModel(BuilderContext& ctx, const GlmOcrTextConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& input_ids,
                   const Tensor& position_ids,
                   const Tensor& beam_idx,
                   const Tensor* visual_embeds = nullptr,
                   const Tensor* visual_pos_mask = nullptr);

    Tensor forward_embeds(const Tensor& inputs_embeds,
                          const Tensor& position_ids,
                          const Tensor& beam_idx,
                          const Tensor* visual_embeds = nullptr,
                          const Tensor* visual_pos_mask = nullptr);

    VocabEmbedding& embed_tokens();
    RMSNorm& norm();

private:
    std::pair<Tensor, Tensor> build_mrope_cos_sin(const Tensor& position_ids) const;

    GlmOcrTextConfig cfg_;
    VocabEmbedding embed_tokens_;
    GlmOcrEmbeddingInjector embedding_injector_;
    std::vector<GlmOcrTextDecoderLayer> layers_;
    RMSNorm norm_;
    int32_t head_dim_ = 0;
};

class GlmOcrTextForCausalLM : public Module {
public:
    GlmOcrTextForCausalLM(BuilderContext& ctx, const GlmOcrTextConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& input_ids,
                   const Tensor& position_ids,
                   const Tensor& beam_idx,
                   const Tensor* visual_embeds = nullptr,
                   const Tensor* visual_pos_mask = nullptr);

    GlmOcrTextModel& model();
    LMHead& lm_head();

private:
    GlmOcrTextConfig cfg_;
    GlmOcrTextModel model_;
    LMHead lm_head_;
};

std::shared_ptr<ov::Model> create_glm_ocr_text_model(
    const GlmOcrConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer,
    bool use_inputs_embeds = false,
    bool enable_visual_inputs = true);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
