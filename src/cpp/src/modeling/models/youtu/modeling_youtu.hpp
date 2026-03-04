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
#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

struct YoutuConfig {
    std::string architecture = "youtu_llm";
    int32_t hidden_size = 0;
    int32_t intermediate_size = 0;
    int32_t num_attention_heads = 0;
    int32_t num_key_value_heads = 0;
    int32_t num_hidden_layers = 0;

    int32_t q_lora_rank = 0;
    int32_t kv_lora_rank = 0;
    int32_t qk_rope_head_dim = 0;
    int32_t qk_nope_head_dim = 0;
    int32_t qk_head_dim = 0;
    int32_t v_head_dim = 0;

    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    bool rope_interleave = false;

    std::string hidden_act = "silu";
    bool attention_bias = false;
    bool mlp_bias = false;
    bool tie_word_embeddings = true;
};

class YoutuMLAttention : public Module {
public:
    YoutuMLAttention(BuilderContext& ctx, const std::string& name, const YoutuConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states,
                   const Tensor& beam_idx,
                   const Tensor& rope_cos,
                   const Tensor& rope_sin) const;

private:
    const Tensor& q_a_proj_weight() const;
    const Tensor& q_b_proj_weight() const;
    const Tensor& kv_a_proj_weight() const;
    const Tensor& kv_b_proj_weight() const;
    const Tensor& o_proj_weight() const;

    const Tensor* q_a_proj_bias() const;
    const Tensor* kv_a_proj_bias() const;
    const Tensor* o_proj_bias() const;

    std::pair<Tensor, Tensor> append_kv_cache(const Tensor& keys,
                                              const Tensor& values,
                                              const Tensor& beam_idx) const;

    int32_t num_heads_ = 0;
    int32_t q_lora_rank_ = 0;
    int32_t kv_lora_rank_ = 0;
    int32_t qk_rope_head_dim_ = 0;
    int32_t qk_nope_head_dim_ = 0;
    int32_t qk_head_dim_ = 0;
    int32_t v_head_dim_ = 0;
    float scaling_ = 1.0f;
    bool rope_interleave_ = false;

    WeightParameter* q_a_proj_param_ = nullptr;
    WeightParameter* q_b_proj_param_ = nullptr;
    WeightParameter* kv_a_proj_param_ = nullptr;
    WeightParameter* kv_b_proj_param_ = nullptr;
    WeightParameter* o_proj_param_ = nullptr;

    WeightParameter* q_a_bias_param_ = nullptr;
    WeightParameter* kv_a_bias_param_ = nullptr;
    WeightParameter* o_bias_param_ = nullptr;

    RMSNorm q_a_layernorm_;
    RMSNorm kv_a_layernorm_;
};

class YoutuMLP : public Module {
public:
    YoutuMLP(BuilderContext& ctx, const std::string& name, const YoutuConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& x) const;

private:
    const Tensor& gate_proj_weight() const;
    const Tensor& up_proj_weight() const;
    const Tensor& down_proj_weight() const;

    const Tensor* gate_proj_bias() const;
    const Tensor* up_proj_bias() const;
    const Tensor* down_proj_bias() const;

    WeightParameter* gate_proj_param_ = nullptr;
    WeightParameter* up_proj_param_ = nullptr;
    WeightParameter* down_proj_param_ = nullptr;

    WeightParameter* gate_bias_param_ = nullptr;
    WeightParameter* up_bias_param_ = nullptr;
    WeightParameter* down_bias_param_ = nullptr;
};

class YoutuDecoderLayer : public Module {
public:
    YoutuDecoderLayer(BuilderContext& ctx, const std::string& name, const YoutuConfig& cfg, Module* parent = nullptr);

    std::pair<Tensor, Tensor> forward(const Tensor& hidden_states,
                                      const Tensor& beam_idx,
                                      const Tensor& rope_cos,
                                      const Tensor& rope_sin,
                                      const std::optional<Tensor>& residual) const;

private:
    YoutuMLAttention self_attn_;
    YoutuMLP mlp_;
    RMSNorm input_layernorm_;
    RMSNorm post_attention_layernorm_;
};

class YoutuModel : public Module {
public:
    YoutuModel(BuilderContext& ctx, const YoutuConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& input_ids,
                   const Tensor& position_ids,
                   const Tensor& beam_idx);

    VocabEmbedding& embed_tokens();
    RMSNorm& norm();

private:
    VocabEmbedding embed_tokens_;
    std::vector<YoutuDecoderLayer> layers_;
    RMSNorm norm_;
    int32_t rope_head_dim_ = 0;
    float rope_theta_ = 10000.0f;
};

class YoutuForCausalLM : public Module {
public:
    YoutuForCausalLM(BuilderContext& ctx, const YoutuConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& input_ids,
                   const Tensor& position_ids,
                   const Tensor& beam_idx);

    YoutuModel& model();
    LMHead& lm_head();

private:
    YoutuConfig cfg_;
    YoutuModel model_;
    LMHead lm_head_;
};

std::shared_ptr<ov::Model> create_youtu_llm_model(
    const YoutuConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
