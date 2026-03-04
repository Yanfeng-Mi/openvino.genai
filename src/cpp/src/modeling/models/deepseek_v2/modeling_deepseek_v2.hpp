// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
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
#include "modeling/models/deepseek_ocr2/processing_deepseek_ocr2.hpp"
#include "modeling/module.hpp"
#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

struct DeepseekV2TextIO {
    static constexpr const char* kInputIds = "input_ids";
    static constexpr const char* kInputsEmbeds = "inputs_embeds";
    static constexpr const char* kAttentionMask = "attention_mask";
    static constexpr const char* kPositionIds = "position_ids";
    static constexpr const char* kBeamIdx = "beam_idx";
    static constexpr const char* kVisualEmbeds = "visual_embeds";
    static constexpr const char* kImagesSeqMask = "images_seq_mask";
    static constexpr const char* kLogits = "logits";
};

struct DeepseekV2TextConfig {
    std::string architecture = "deepseek_v2";
    int32_t hidden_size = 0;
    int32_t num_attention_heads = 0;
    int32_t num_key_value_heads = 0;
    int32_t head_dim = 0;
    int32_t intermediate_size = 0;
    int32_t moe_intermediate_size = 0;
    int32_t num_hidden_layers = 0;
    int32_t n_routed_experts = 0;
    int32_t n_shared_experts = 0;
    int32_t num_experts_per_tok = 0;
    int32_t moe_layer_freq = 1;
    int32_t first_k_dense_replace = 0;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    std::string hidden_act = "silu";
    bool attention_bias = false;
    bool tie_word_embeddings = false;
    bool use_mla = false;
    bool norm_topk_prob = false;
    float routed_scaling_factor = 1.0f;

    int32_t resolved_kv_heads() const;
    int32_t resolved_head_dim() const;
    bool is_moe_layer(int32_t layer_idx) const;
    void validate() const;
};

class DeepseekV2EmbeddingInjector : public Module {
public:
    DeepseekV2EmbeddingInjector(BuilderContext& ctx, const std::string& name, Module* parent = nullptr);

    Tensor forward(const Tensor& inputs_embeds,
                   const Tensor& visual_embeds,
                   const Tensor& images_seq_mask) const;
};

class DeepseekV2Attention : public Module {
public:
    DeepseekV2Attention(BuilderContext& ctx,
                        const std::string& name,
                        const DeepseekV2TextConfig& cfg,
                        Module* parent = nullptr);

    Tensor forward(const Tensor& positions, const Tensor& hidden_states, const Tensor& beam_idx) const;
    Tensor forward(const Tensor& hidden_states,
                   const Tensor& beam_idx,
                   const Tensor& rope_cos,
                   const Tensor& rope_sin,
                   const Tensor& causal_mask) const;

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
    float rope_theta_ = 10000.0f;

    WeightParameter* q_proj_param_ = nullptr;
    WeightParameter* k_proj_param_ = nullptr;
    WeightParameter* v_proj_param_ = nullptr;
    WeightParameter* o_proj_param_ = nullptr;

    WeightParameter* q_bias_param_ = nullptr;
    WeightParameter* k_bias_param_ = nullptr;
    WeightParameter* v_bias_param_ = nullptr;
    WeightParameter* o_bias_param_ = nullptr;
};

class DeepseekV2MLP : public Module {
public:
    DeepseekV2MLP(BuilderContext& ctx, const std::string& name, const DeepseekV2TextConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& x) const;

private:
    const Tensor& gate_proj_weight() const;
    const Tensor& up_proj_weight() const;
    const Tensor& down_proj_weight() const;

    WeightParameter* gate_proj_param_ = nullptr;
    WeightParameter* up_proj_param_ = nullptr;
    WeightParameter* down_proj_param_ = nullptr;
};

class DeepseekV2MoE : public Module {
public:
    DeepseekV2MoE(BuilderContext& ctx, const std::string& name, const DeepseekV2TextConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& x) const;

private:
    const Tensor& gate_weight() const;
    std::vector<Tensor> gate_expert_weights() const;
    std::vector<Tensor> up_expert_weights() const;
    std::vector<Tensor> down_expert_weights() const;

    const Tensor& shared_gate_weight() const;
    const Tensor& shared_up_weight() const;
    const Tensor& shared_down_weight() const;

    WeightParameter* gate_param_ = nullptr;
    std::vector<WeightParameter*> gate_experts_param_;
    std::vector<WeightParameter*> up_experts_param_;
    std::vector<WeightParameter*> down_experts_param_;

    WeightParameter* shared_gate_param_ = nullptr;
    WeightParameter* shared_up_param_ = nullptr;
    WeightParameter* shared_down_param_ = nullptr;

    int32_t hidden_size_ = 0;
    int32_t inter_size_ = 0;
    int32_t num_experts_ = 0;
    int32_t top_k_ = 1;
    int32_t shared_experts_ = 0;
    bool norm_topk_prob_ = false;
    float routed_scaling_factor_ = 1.0f;
};

class DeepseekV2DecoderLayer : public Module {
public:
    DeepseekV2DecoderLayer(BuilderContext& ctx,
                           const std::string& name,
                           const DeepseekV2TextConfig& cfg,
                           bool is_moe,
                           Module* parent = nullptr);

    std::pair<Tensor, Tensor> forward(const Tensor& hidden_states,
                                      const Tensor& beam_idx,
                                      const Tensor& rope_cos,
                                      const Tensor& rope_sin,
                                      const Tensor& causal_mask,
                                      const std::optional<Tensor>& residual) const;

private:
    DeepseekV2Attention self_attn_;
    std::unique_ptr<DeepseekV2MLP> mlp_;
    std::unique_ptr<DeepseekV2MoE> moe_;
    RMSNorm input_layernorm_;
    RMSNorm post_attention_layernorm_;
    bool is_moe_ = false;
};

class DeepseekV2Model : public Module {
public:
    DeepseekV2Model(BuilderContext& ctx, const DeepseekV2TextConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& input_ids,
                   const Tensor& position_ids,
                   const Tensor& beam_idx,
                   const Tensor* visual_embeds = nullptr,
                   const Tensor* images_seq_mask = nullptr);

    Tensor forward_embeds(const Tensor& inputs_embeds,
                          const Tensor& position_ids,
                          const Tensor& beam_idx,
                          const Tensor* visual_embeds = nullptr,
                          const Tensor* images_seq_mask = nullptr);

    VocabEmbedding& embed_tokens();
    RMSNorm& norm();

private:
    DeepseekV2TextConfig cfg_;
    VocabEmbedding embed_tokens_;
    DeepseekV2EmbeddingInjector embedding_injector_;
    std::vector<DeepseekV2DecoderLayer> layers_;
    RMSNorm norm_;
    int32_t head_dim_ = 0;
    float rope_theta_ = 10000.0f;
};

class DeepseekV2ForCausalLM : public Module {
public:
    DeepseekV2ForCausalLM(BuilderContext& ctx, const DeepseekV2TextConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& input_ids,
                   const Tensor& position_ids,
                   const Tensor& beam_idx,
                   const Tensor* visual_embeds = nullptr,
                   const Tensor* images_seq_mask = nullptr);

    Tensor forward_embeds(const Tensor& inputs_embeds,
                          const Tensor& position_ids,
                          const Tensor& beam_idx,
                          const Tensor* visual_embeds = nullptr,
                          const Tensor* images_seq_mask = nullptr);

    DeepseekV2Model& model();
    LMHead& lm_head();

private:
    DeepseekV2TextConfig cfg_;
    DeepseekV2Model model_;
    LMHead lm_head_;
};

std::shared_ptr<ov::Model> create_deepseek_v2_text_model(
    const DeepseekOCR2LanguageConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer,
    bool use_inputs_embeds = false,
    bool enable_visual_inputs = false);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov

