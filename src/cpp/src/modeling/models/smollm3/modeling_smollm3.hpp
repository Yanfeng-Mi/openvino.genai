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

struct SmolLM3Config {
    std::string architecture = "smollm3";
    int32_t hidden_size = 0;
    int32_t num_attention_heads = 0;
    int32_t num_key_value_heads = 0;
    int32_t head_dim = 0;
    int32_t intermediate_size = 0;
    int32_t num_hidden_layers = 0;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    std::string hidden_act = "silu";
    bool attention_bias = false;
    bool mlp_bias = false;
    bool tie_word_embeddings = true;
    int32_t no_rope_layer_interval = 4;
    std::vector<int32_t> no_rope_layers;
};

class SmolLM3Attention : public Module {
public:
    SmolLM3Attention(BuilderContext& ctx,
                     const std::string& name,
                     const SmolLM3Config& cfg,
                     int32_t layer_idx,
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
    bool use_rope_ = true;

    WeightParameter* q_proj_param_ = nullptr;
    WeightParameter* k_proj_param_ = nullptr;
    WeightParameter* v_proj_param_ = nullptr;
    WeightParameter* o_proj_param_ = nullptr;

    WeightParameter* q_bias_param_ = nullptr;
    WeightParameter* k_bias_param_ = nullptr;
    WeightParameter* v_bias_param_ = nullptr;
    WeightParameter* o_bias_param_ = nullptr;
};

class SmolLM3MLP : public Module {
public:
    SmolLM3MLP(BuilderContext& ctx, const std::string& name, const SmolLM3Config& cfg, Module* parent = nullptr);

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

class SmolLM3DecoderLayer : public Module {
public:
    SmolLM3DecoderLayer(BuilderContext& ctx,
                        const std::string& name,
                        const SmolLM3Config& cfg,
                        int32_t layer_idx,
                        Module* parent = nullptr);

    std::pair<Tensor, Tensor> forward(const Tensor& hidden_states,
                                      const Tensor& beam_idx,
                                      const Tensor& rope_cos,
                                      const Tensor& rope_sin,
                                      const std::optional<Tensor>& residual) const;

private:
    SmolLM3Attention self_attn_;
    SmolLM3MLP mlp_;
    RMSNorm input_layernorm_;
    RMSNorm post_attention_layernorm_;
};

class SmolLM3Model : public Module {
public:
    SmolLM3Model(BuilderContext& ctx, const SmolLM3Config& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& input_ids,
                   const Tensor& position_ids,
                   const Tensor& beam_idx);

    VocabEmbedding& embed_tokens();
    RMSNorm& norm();

private:
    VocabEmbedding embed_tokens_;
    std::vector<SmolLM3DecoderLayer> layers_;
    RMSNorm norm_;
    int32_t head_dim_ = 0;
    float rope_theta_ = 10000.0f;
};

class SmolLM3ForCausalLM : public Module {
public:
    SmolLM3ForCausalLM(BuilderContext& ctx, const SmolLM3Config& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& input_ids,
                   const Tensor& position_ids,
                   const Tensor& beam_idx);

    SmolLM3Model& model();
    LMHead& lm_head();

private:
    SmolLM3Config cfg_;
    SmolLM3Model model_;
    LMHead lm_head_;
};

std::shared_ptr<ov::Model> create_smollm3_model(
    const SmolLM3Config& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
