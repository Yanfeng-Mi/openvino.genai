// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "modeling/layers/rms_norm.hpp"
#include "modeling/module.hpp"
#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

struct ZImageDiTConfig {
    int32_t dim = 3840;
    int32_t n_layers = 30;
    int32_t n_refiner_layers = 2;
    int32_t n_heads = 30;
    int32_t n_kv_heads = 30;
    int32_t in_channels = 16;
    int32_t cap_feat_dim = 2560;
    int32_t patch_size = 2;
    int32_t f_patch_size = 1;
    float norm_eps = 1e-5f;
    bool qk_norm = true;
    float t_scale = 1000.0f;
    int32_t adaln_embed_dim = 256;
    int32_t frequency_embedding_size = 256;
    int32_t t_mid_dim = 1024;
    float max_period = 10000.0f;

    int32_t head_dim() const {
        return n_heads > 0 ? dim / n_heads : 0;
    }

    int32_t patch_dim() const {
        return patch_size * patch_size * f_patch_size * in_channels;
    }

    int32_t ffn_hidden_dim() const {
        return static_cast<int32_t>(static_cast<float>(dim) / 3.0f * 8.0f);
    }
};

class ZImageTimestepEmbedder : public Module {
public:
    ZImageTimestepEmbedder(BuilderContext& ctx,
                           const std::string& name,
                           int32_t out_size,
                           int32_t mid_size,
                           int32_t frequency_embedding_size,
                           float max_period,
                           Module* parent = nullptr);

    Tensor forward(const Tensor& t) const;

private:
    Tensor timestep_embedding(const Tensor& t) const;

    int32_t out_size_ = 0;
    int32_t mid_size_ = 0;
    int32_t frequency_embedding_size_ = 0;
    float max_period_ = 10000.0f;

    WeightParameter* fc1_weight_ = nullptr;
    WeightParameter* fc1_bias_ = nullptr;
    WeightParameter* fc2_weight_ = nullptr;
    WeightParameter* fc2_bias_ = nullptr;
};

class ZImageLinear : public Module {
public:
    ZImageLinear(BuilderContext& ctx, const std::string& name, Module* parent = nullptr);

    Tensor forward(const Tensor& input) const;

private:
    WeightParameter* weight_ = nullptr;
    WeightParameter* bias_ = nullptr;
};

class ZImageAttention : public Module {
public:
    ZImageAttention(BuilderContext& ctx,
                    const std::string& name,
                    int32_t dim,
                    int32_t n_heads,
                    int32_t n_kv_heads,
                    float norm_eps,
                    bool qk_norm,
                    Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states,
                   const Tensor& attention_mask,
                   const Tensor& rope_cos,
                   const Tensor& rope_sin) const;

private:
    Tensor build_attention_bias(const Tensor& attention_mask) const;

    int32_t num_heads_ = 0;
    int32_t num_kv_heads_ = 0;
    int32_t head_dim_ = 0;
    float scaling_ = 1.0f;
    bool qk_norm_ = true;

    RMSNorm q_norm_;
    RMSNorm k_norm_;

    WeightParameter* q_weight_ = nullptr;
    WeightParameter* k_weight_ = nullptr;
    WeightParameter* v_weight_ = nullptr;
    WeightParameter* o_weight_ = nullptr;
};

class ZImageFeedForward : public Module {
public:
    ZImageFeedForward(BuilderContext& ctx, const std::string& name, int32_t dim, Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states) const;

private:
    WeightParameter* w1_weight_ = nullptr;
    WeightParameter* w2_weight_ = nullptr;
    WeightParameter* w3_weight_ = nullptr;
};

class ZImageTransformerBlock : public Module {
public:
    ZImageTransformerBlock(BuilderContext& ctx,
                           const std::string& name,
                           const ZImageDiTConfig& cfg,
                           bool modulation,
                           Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states,
                   const Tensor& attention_mask,
                   const Tensor& rope_cos,
                   const Tensor& rope_sin,
                   const Tensor* adaln_input = nullptr) const;

private:
    int32_t dim_ = 0;
    bool modulation_ = false;

    ZImageAttention attention_;
    ZImageFeedForward feed_forward_;
    RMSNorm attention_norm1_;
    RMSNorm ffn_norm1_;
    RMSNorm attention_norm2_;
    RMSNorm ffn_norm2_;

    WeightParameter* adaln_weight_ = nullptr;
    WeightParameter* adaln_bias_ = nullptr;
};

class ZImageFinalLayer : public Module {
public:
    ZImageFinalLayer(BuilderContext& ctx,
                     const std::string& name,
                     int32_t hidden_size,
                     int32_t out_features,
                     int32_t adaln_embed_dim,
                     Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states, const Tensor& adaln_input) const;

private:
    Tensor layer_norm_no_affine(const Tensor& x) const;

    int32_t hidden_size_ = 0;
    float eps_ = 1e-6f;

    WeightParameter* linear_weight_ = nullptr;
    WeightParameter* linear_bias_ = nullptr;
    WeightParameter* adaln_weight_ = nullptr;
    WeightParameter* adaln_bias_ = nullptr;
};

class ZImageDiTModel : public Module {
public:
    ZImageDiTModel(BuilderContext& ctx, const ZImageDiTConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& x_tokens,
                   const Tensor& x_mask,
                   const Tensor& cap_feats,
                   const Tensor& cap_mask,
                   const Tensor& timesteps,
                   const Tensor& x_rope_cos,
                   const Tensor& x_rope_sin,
                   const Tensor& cap_rope_cos,
                   const Tensor& cap_rope_sin);

private:
    Tensor apply_pad_token(const Tensor& hidden_states,
                           const Tensor& mask,
                           const WeightParameter& pad_token) const;

    ZImageDiTConfig cfg_;

    ZImageTimestepEmbedder t_embedder_;
    ZImageLinear x_embedder_;
    RMSNorm cap_norm_;
    ZImageLinear cap_linear_;
    WeightParameter* x_pad_token_ = nullptr;
    WeightParameter* cap_pad_token_ = nullptr;

    std::vector<ZImageTransformerBlock> noise_refiner_;
    std::vector<ZImageTransformerBlock> context_refiner_;
    std::vector<ZImageTransformerBlock> layers_;
    ZImageFinalLayer final_layer_;
};

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
