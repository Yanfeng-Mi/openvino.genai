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
#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

struct DFlashDraftConfig {
    std::string architecture = "dflash";
    int32_t hidden_size = 0;
    int32_t intermediate_size = 0;
    int32_t num_hidden_layers = 0;
    int32_t num_target_layers = 0;
    int32_t num_attention_heads = 0;
    int32_t num_key_value_heads = 0;
    int32_t head_dim = 0;
    int32_t block_size = 0;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    std::string hidden_act = "silu";
    bool attention_bias = false;
};

class DFlashAttention : public Module {
public:
    DFlashAttention(BuilderContext& ctx, const std::string& name, const DFlashDraftConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& target_hidden,
                   const Tensor& hidden_states,
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
    bool attention_bias_ = false;

    WeightParameter* q_proj_param_ = nullptr;
    WeightParameter* k_proj_param_ = nullptr;
    WeightParameter* v_proj_param_ = nullptr;
    WeightParameter* o_proj_param_ = nullptr;

    WeightParameter* q_bias_param_ = nullptr;
    WeightParameter* k_bias_param_ = nullptr;
    WeightParameter* v_bias_param_ = nullptr;
    WeightParameter* o_bias_param_ = nullptr;

    RMSNorm q_norm_;
    RMSNorm k_norm_;
};

class DFlashMLP : public Module {
public:
    DFlashMLP(BuilderContext& ctx, const std::string& name, const DFlashDraftConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& x) const;

private:
    const Tensor& gate_proj_weight() const;
    const Tensor& up_proj_weight() const;
    const Tensor& down_proj_weight() const;

    WeightParameter* gate_proj_param_ = nullptr;
    WeightParameter* up_proj_param_ = nullptr;
    WeightParameter* down_proj_param_ = nullptr;
};

class DFlashDecoderLayer : public Module {
public:
    DFlashDecoderLayer(BuilderContext& ctx, const std::string& name, const DFlashDraftConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& target_hidden,
                   const Tensor& hidden_states,
                   const Tensor& rope_cos,
                   const Tensor& rope_sin) const;

private:
    DFlashAttention self_attn_;
    DFlashMLP mlp_;
    RMSNorm input_layernorm_;
    RMSNorm post_attention_layernorm_;
};

class DFlashDraftModel : public Module {
public:
    DFlashDraftModel(BuilderContext& ctx, const DFlashDraftConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& target_hidden,
                   const Tensor& noise_embedding,
                   const Tensor& position_ids) const;

private:
    const Tensor& fc_weight() const;

    DFlashDraftConfig cfg_;
    std::vector<DFlashDecoderLayer> layers_;
    RMSNorm norm_;
    RMSNorm hidden_norm_;
    WeightParameter* fc_weight_param_ = nullptr;
    int32_t head_dim_ = 0;
    float rope_theta_ = 10000.0f;
};

std::shared_ptr<ov::Model> create_dflash_draft_model(
    const DFlashDraftConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer,
    // NOTE: Draft is always constructed in FP32 to avoid BF16 instabilities; the
    // type argument is kept for API compatibility only.
    const ov::element::Type& input_type = ov::element::f32);

std::vector<int32_t> build_target_layer_ids(int32_t num_target_layers, int32_t num_draft_layers);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
