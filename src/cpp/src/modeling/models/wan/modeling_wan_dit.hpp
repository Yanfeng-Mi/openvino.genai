// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
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

#include "modeling/layers/layer_norm.hpp"
#include "modeling/layers/rms_norm.hpp"
#include "modeling/models/wan/processing_wan.hpp"
#include "modeling/module.hpp"
#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

struct WanConditionEmbeddings {
    Tensor temb;
    Tensor timestep_proj;
    Tensor text_embeds;
    std::optional<Tensor> image_embeds;
};

class WanRotaryPosEmbed : public Module {
public:
    WanRotaryPosEmbed(BuilderContext& ctx,
                      const std::string& name,
                      int32_t attention_head_dim,
                      const std::vector<int32_t>& patch_size,
                      int32_t max_seq_len,
                      float theta = 10000.0f,
                      Module* parent = nullptr);

    std::pair<Tensor, Tensor> forward(const Tensor& hidden_states) const;

private:
    std::vector<int32_t> patch_size_;
    int32_t head_dim_ = 0;
    int32_t t_half_ = 0;
    int32_t h_half_ = 0;
    int32_t w_half_ = 0;
    Tensor t_cos_;
    Tensor t_sin_;
    Tensor h_cos_;
    Tensor h_sin_;
    Tensor w_cos_;
    Tensor w_sin_;
};

class WanTimestepEmbedder : public Module {
public:
    WanTimestepEmbedder(BuilderContext& ctx,
                        const std::string& name,
                        int32_t freq_dim,
                        int32_t out_dim,
                        Module* parent = nullptr);

    Tensor forward(const Tensor& t) const;

private:
    Tensor timestep_embedding(const Tensor& t) const;

    int32_t freq_dim_ = 0;
    int32_t out_dim_ = 0;
    float max_period_ = 10000.0f;

    WeightParameter* fc1_weight_ = nullptr;
    WeightParameter* fc1_bias_ = nullptr;
    WeightParameter* fc2_weight_ = nullptr;
    WeightParameter* fc2_bias_ = nullptr;
};

class WanTextProjection : public Module {
public:
    WanTextProjection(BuilderContext& ctx,
                      const std::string& name,
                      int32_t in_features,
                      int32_t hidden_size,
                      int32_t out_features,
                      Module* parent = nullptr);

    Tensor forward(const Tensor& input) const;

private:
    WeightParameter* fc1_weight_ = nullptr;
    WeightParameter* fc1_bias_ = nullptr;
    WeightParameter* fc2_weight_ = nullptr;
    WeightParameter* fc2_bias_ = nullptr;
};

class WanFeedForward : public Module {
public:
    WanFeedForward(BuilderContext& ctx,
                   const std::string& name,
                   int32_t dim,
                   int32_t ffn_dim,
                   bool approximate,
                   Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states) const;

private:
    bool gelu_approximate_ = true;
    WeightParameter* proj_weight_ = nullptr;
    WeightParameter* proj_bias_ = nullptr;
    WeightParameter* out_weight_ = nullptr;
    WeightParameter* out_bias_ = nullptr;
};

class WanImageEmbedding : public Module {
public:
    WanImageEmbedding(BuilderContext& ctx,
                      const std::string& name,
                      int32_t in_features,
                      int32_t out_features,
                      std::optional<int32_t> pos_embed_seq_len,
                      Module* parent = nullptr);

    Tensor forward(const Tensor& encoder_hidden_states_image) const;

private:
    FP32LayerNorm norm1_;
    WanFeedForward ff_;
    FP32LayerNorm norm2_;
    WeightParameter* pos_embed_ = nullptr;
};

class WanTimeTextImageEmbedding : public Module {
public:
    WanTimeTextImageEmbedding(BuilderContext& ctx,
                              const std::string& name,
                              int32_t dim,
                              int32_t time_freq_dim,
                              int32_t time_proj_dim,
                              int32_t text_embed_dim,
                              std::optional<int32_t> image_embed_dim,
                              std::optional<int32_t> pos_embed_seq_len,
                              Module* parent = nullptr);

    WanConditionEmbeddings forward(const Tensor& timestep,
                                   const Tensor& encoder_hidden_states,
                                   const Tensor* encoder_hidden_states_image = nullptr) const;

private:
    int32_t dim_ = 0;
    int32_t time_proj_dim_ = 0;

    WanTimestepEmbedder time_embedder_;
    WanTextProjection text_embedder_;
    std::unique_ptr<WanImageEmbedding> image_embedder_;

    WeightParameter* time_proj_weight_ = nullptr;
    WeightParameter* time_proj_bias_ = nullptr;
};

class WanAttention : public Module {
public:
    WanAttention(BuilderContext& ctx,
                 const std::string& name,
                 int32_t dim,
                 int32_t num_heads,
                 int32_t head_dim,
                 float eps,
                 const std::string& qk_norm,
                 std::optional<int32_t> added_kv_proj_dim = std::nullopt,
                 Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states,
                   const Tensor* encoder_hidden_states,
                   const Tensor* encoder_hidden_states_image,
                   const Tensor* rotary_cos,
                   const Tensor* rotary_sin) const;

private:
    Tensor apply_linear(const Tensor& input, const Tensor& weight, const Tensor* bias) const;

    int32_t num_heads_ = 0;
    int32_t head_dim_ = 0;
    int32_t inner_dim_ = 0;
    float scaling_ = 1.0f;
    bool use_qk_norm_ = true;
    std::optional<int32_t> added_kv_proj_dim_;

    RMSNorm q_norm_;
    RMSNorm k_norm_;
    std::unique_ptr<RMSNorm> added_k_norm_;

    WeightParameter* q_weight_ = nullptr;
    WeightParameter* q_bias_ = nullptr;
    WeightParameter* k_weight_ = nullptr;
    WeightParameter* k_bias_ = nullptr;
    WeightParameter* v_weight_ = nullptr;
    WeightParameter* v_bias_ = nullptr;
    WeightParameter* o_weight_ = nullptr;
    WeightParameter* o_bias_ = nullptr;

    WeightParameter* add_k_weight_ = nullptr;
    WeightParameter* add_k_bias_ = nullptr;
    WeightParameter* add_v_weight_ = nullptr;
    WeightParameter* add_v_bias_ = nullptr;
};

class WanTransformerBlock : public Module {
public:
    WanTransformerBlock(BuilderContext& ctx,
                        const std::string& name,
                        const WanTransformer3DConfig& cfg,
                        Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states,
                   const Tensor& encoder_hidden_states,
                   const Tensor& temb,
                   const Tensor& rotary_cos,
                   const Tensor& rotary_sin,
                   const Tensor* encoder_hidden_states_image = nullptr) const;

private:
    int32_t inner_dim_ = 0;
    float eps_ = 1e-6f;
    bool cross_attn_norm_ = true;

    FP32LayerNorm norm1_;
    WanAttention attn1_;
    WanAttention attn2_;
    std::unique_ptr<FP32LayerNorm> norm2_;
    WanFeedForward ffn_;
    FP32LayerNorm norm3_;

    WeightParameter* scale_shift_table_ = nullptr;
};

class WanTransformer3DModel : public Module {
public:
    WanTransformer3DModel(BuilderContext& ctx,
                          const WanTransformer3DConfig& cfg,
                          Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states,
                   const Tensor& timestep,
                   const Tensor& encoder_hidden_states,
                   const Tensor* encoder_hidden_states_image = nullptr);

private:
    WanTransformer3DConfig cfg_;
    int32_t inner_dim_ = 0;

    WanRotaryPosEmbed rope_;
    WanTimeTextImageEmbedding condition_embedder_;
    std::vector<WanTransformerBlock> blocks_;
    FP32LayerNorm norm_out_;

    WeightParameter* patch_weight_ = nullptr;
    WeightParameter* patch_bias_ = nullptr;
    WeightParameter* proj_out_weight_ = nullptr;
    WeightParameter* proj_out_bias_ = nullptr;
    WeightParameter* scale_shift_table_ = nullptr;
};

std::shared_ptr<ov::Model> create_wan_dit_model(
    const WanTransformer3DConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
