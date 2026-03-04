// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/zimage/modeling_zimage_dit.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <openvino/core/except.hpp>

#include "modeling/ops/llm.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/ops/shape.hpp"

namespace {

ov::genai::modeling::Tensor add_bias_if_present(const ov::genai::modeling::Tensor& x,
                                                const ov::genai::modeling::Tensor* bias) {
    if (!bias) {
        return x;
    }
    return x + *bias;
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

ZImageTimestepEmbedder::ZImageTimestepEmbedder(BuilderContext& ctx,
                                               const std::string& name,
                                               int32_t out_size,
                                               int32_t mid_size,
                                               int32_t frequency_embedding_size,
                                               float max_period,
                                               Module* parent)
    : Module(name, ctx, parent),
      out_size_(out_size),
      mid_size_(mid_size),
      frequency_embedding_size_(frequency_embedding_size),
      max_period_(max_period) {
    fc1_weight_ = &register_parameter("mlp.0.weight");
    fc1_bias_ = &register_parameter("mlp.0.bias");
    fc2_weight_ = &register_parameter("mlp.2.weight");
    fc2_bias_ = &register_parameter("mlp.2.bias");
}

Tensor ZImageTimestepEmbedder::timestep_embedding(const Tensor& t) const {
    auto* ctx = t.context();
    const int32_t half = frequency_embedding_size_ / 2;
    std::vector<float> freqs(static_cast<size_t>(half));
    const float log_max = std::log(max_period_);
    for (int32_t i = 0; i < half; ++i) {
        float exponent = static_cast<float>(i) / static_cast<float>(half);
        freqs[static_cast<size_t>(i)] = std::exp(-log_max * exponent);
    }

    Tensor freqs_tensor(ops::const_vec(ctx, freqs), ctx);
    auto freqs_row = freqs_tensor.reshape({1, -1});
    auto t_f = t.to(ov::element::f32).unsqueeze(1);
    auto args = t_f * freqs_row;
    auto emb = ops::concat({args.cos(), args.sin()}, 1);

    if (frequency_embedding_size_ % 2 != 0) {
        auto batch = shape::dim(t, 0);
        auto one = ops::const_vec(ctx, std::vector<int64_t>{1});
        auto target = shape::make({batch, one});
        auto zero = Tensor(ops::const_scalar(ctx, 0.0f), ctx);
        auto zeros = shape::broadcast_to(zero, target);
        emb = ops::concat({emb, zeros}, 1);
    }
    return emb;
}

Tensor ZImageTimestepEmbedder::forward(const Tensor& t) const {
    auto t_freq = timestep_embedding(t);
    auto weight_dtype = fc1_weight_->value().dtype();
    t_freq = t_freq.to(weight_dtype);
    auto fc1 = add_bias_if_present(ops::linear(t_freq, fc1_weight_->value()),
                                   fc1_bias_ ? &fc1_bias_->value() : nullptr);
    auto act = ops::silu(fc1);
    return add_bias_if_present(ops::linear(act, fc2_weight_->value()),
                               fc2_bias_ ? &fc2_bias_->value() : nullptr);
}

ZImageLinear::ZImageLinear(BuilderContext& ctx, const std::string& name, Module* parent)
    : Module(name, ctx, parent) {
    weight_ = &register_parameter("weight");
    bias_ = &register_parameter("bias");
}

Tensor ZImageLinear::forward(const Tensor& input) const {
    return add_bias_if_present(ops::linear(input, weight_->value()),
                               bias_ ? &bias_->value() : nullptr);
}

ZImageAttention::ZImageAttention(BuilderContext& ctx,
                                 const std::string& name,
                                 int32_t dim,
                                 int32_t n_heads,
                                 int32_t n_kv_heads,
                                 float norm_eps,
                                 bool qk_norm,
                                 Module* parent)
    : Module(name, ctx, parent),
      num_heads_(n_heads),
      num_kv_heads_(n_kv_heads),
      head_dim_(n_heads > 0 ? dim / n_heads : 0),
      scaling_(1.0f / std::sqrt(static_cast<float>(head_dim_))),
      qk_norm_(qk_norm),
      q_norm_(ctx, "norm_q", norm_eps, this),
      k_norm_(ctx, "norm_k", norm_eps, this) {
    if (num_heads_ <= 0 || head_dim_ <= 0) {
        OPENVINO_THROW("Invalid ZImageAttention head configuration");
    }
    register_module("norm_q", &q_norm_);
    register_module("norm_k", &k_norm_);

    q_weight_ = &register_parameter("to_q.weight");
    k_weight_ = &register_parameter("to_k.weight");
    v_weight_ = &register_parameter("to_v.weight");
    o_weight_ = &register_parameter("to_out.0.weight");

    if (!qk_norm_) {
        q_norm_.weight_param().set_optional(true);
        k_norm_.weight_param().set_optional(true);
    }
}

Tensor ZImageAttention::build_attention_bias(const Tensor& attention_mask) const {
    auto* ctx = attention_mask.context();
    auto mask_bool = attention_mask.to(ov::element::boolean);
    auto mask_shape = shape::of(mask_bool);
    auto zero = Tensor(ops::const_scalar(ctx, 0.0f), ctx);
    auto neg = Tensor(ops::const_scalar(ctx, -65504.0f), ctx);
    auto zeros = shape::broadcast_to(zero, mask_shape);
    auto negs = shape::broadcast_to(neg, mask_shape);
    auto mask = ops::where(mask_bool, zeros, negs);
    return mask.unsqueeze({1, 2});
}

Tensor ZImageAttention::forward(const Tensor& hidden_states,
                                const Tensor& attention_mask,
                                const Tensor& rope_cos,
                                const Tensor& rope_sin) const {
    auto q = ops::linear(hidden_states, q_weight_->value());
    auto k = ops::linear(hidden_states, k_weight_->value());
    auto v = ops::linear(hidden_states, v_weight_->value());

    auto q_heads = q.reshape({0, 0, num_heads_, head_dim_}).permute({0, 2, 1, 3});
    auto k_heads = k.reshape({0, 0, num_kv_heads_, head_dim_}).permute({0, 2, 1, 3});
    auto v_heads = v.reshape({0, 0, num_kv_heads_, head_dim_}).permute({0, 2, 1, 3});

    if (q_norm_.weight_param().is_bound()) {
        q_heads = q_norm_.forward(q_heads);
    }
    if (k_norm_.weight_param().is_bound()) {
        k_heads = k_norm_.forward(k_heads);
    }

    auto q_rot = ops::llm::apply_rope_interleave(q_heads, rope_cos, rope_sin, head_dim_);
    auto k_rot = ops::llm::apply_rope_interleave(k_heads, rope_cos, rope_sin, head_dim_);

    auto* policy = &ctx().op_policy();
    auto attn_bias = build_attention_bias(attention_mask).to(q_rot.dtype());
    auto context = ops::llm::sdpa(q_rot, k_rot, v_heads, scaling_, 3, &attn_bias, false, policy);

    auto merged = context.permute({0, 2, 1, 3}).reshape({0, 0, num_heads_ * head_dim_});
    return ops::linear(merged, o_weight_->value());
}

ZImageFeedForward::ZImageFeedForward(BuilderContext& ctx, const std::string& name, int32_t dim, Module* parent)
    : Module(name, ctx, parent) {
    (void)dim;
    w1_weight_ = &register_parameter("w1.weight");
    w2_weight_ = &register_parameter("w2.weight");
    w3_weight_ = &register_parameter("w3.weight");
}

Tensor ZImageFeedForward::forward(const Tensor& hidden_states) const {
    auto w1 = ops::linear(hidden_states, w1_weight_->value());
    auto w3 = ops::linear(hidden_states, w3_weight_->value());
    auto act = ops::silu(w1) * w3;
    return ops::linear(act, w2_weight_->value());
}

ZImageTransformerBlock::ZImageTransformerBlock(BuilderContext& ctx,
                                               const std::string& name,
                                               const ZImageDiTConfig& cfg,
                                               bool modulation,
                                               Module* parent)
    : Module(name, ctx, parent),
      dim_(cfg.dim),
      modulation_(modulation),
      attention_(ctx, "attention", cfg.dim, cfg.n_heads, cfg.n_kv_heads, cfg.norm_eps, cfg.qk_norm, this),
      feed_forward_(ctx, "feed_forward", cfg.dim, this),
      attention_norm1_(ctx, "attention_norm1", cfg.norm_eps, this),
      ffn_norm1_(ctx, "ffn_norm1", cfg.norm_eps, this),
      attention_norm2_(ctx, "attention_norm2", cfg.norm_eps, this),
      ffn_norm2_(ctx, "ffn_norm2", cfg.norm_eps, this) {
    register_module("attention", &attention_);
    register_module("feed_forward", &feed_forward_);
    register_module("attention_norm1", &attention_norm1_);
    register_module("ffn_norm1", &ffn_norm1_);
    register_module("attention_norm2", &attention_norm2_);
    register_module("ffn_norm2", &ffn_norm2_);

    if (modulation_) {
        adaln_weight_ = &register_parameter("adaLN_modulation.0.weight");
        adaln_bias_ = &register_parameter("adaLN_modulation.0.bias");
    }
}

Tensor ZImageTransformerBlock::forward(const Tensor& hidden_states,
                                       const Tensor& attention_mask,
                                       const Tensor& rope_cos,
                                       const Tensor& rope_sin,
                                       const Tensor* adaln_input) const {
    if (modulation_ && !adaln_input) {
        OPENVINO_THROW("ZImageTransformerBlock requires adaln_input when modulation is enabled");
    }

    auto x = hidden_states;
    if (modulation_) {
        auto mod = add_bias_if_present(ops::linear(*adaln_input, adaln_weight_->value()),
                                       adaln_bias_ ? &adaln_bias_->value() : nullptr)
                       .unsqueeze(1);
        auto scale_msa = ops::slice(mod, 0, dim_, 1, 2);
        auto gate_msa = ops::slice(mod, dim_, 2 * dim_, 1, 2).tanh();
        auto scale_mlp = ops::slice(mod, 2 * dim_, 3 * dim_, 1, 2);
        auto gate_mlp = ops::slice(mod, 3 * dim_, 4 * dim_, 1, 2).tanh();
        scale_msa = scale_msa + 1.0f;
        scale_mlp = scale_mlp + 1.0f;

        auto attn_in = attention_norm1_.forward(x) * scale_msa;
        auto attn_out = attention_.forward(attn_in, attention_mask, rope_cos, rope_sin);
        x = x + gate_msa * attention_norm2_.forward(attn_out);

        auto ffn_in = ffn_norm1_.forward(x) * scale_mlp;
        auto ffn_out = feed_forward_.forward(ffn_in);
        x = x + gate_mlp * ffn_norm2_.forward(ffn_out);
        return x;
    }

    auto attn_out = attention_.forward(attention_norm1_.forward(x), attention_mask, rope_cos, rope_sin);
    x = x + attention_norm2_.forward(attn_out);
    auto ffn_out = feed_forward_.forward(ffn_norm1_.forward(x));
    return x + ffn_norm2_.forward(ffn_out);
}

ZImageFinalLayer::ZImageFinalLayer(BuilderContext& ctx,
                                   const std::string& name,
                                   int32_t hidden_size,
                                   int32_t out_features,
                                   int32_t adaln_embed_dim,
                                   Module* parent)
    : Module(name, ctx, parent),
      hidden_size_(hidden_size) {
    (void)out_features;
    (void)adaln_embed_dim;
    linear_weight_ = &register_parameter("linear.weight");
    linear_bias_ = &register_parameter("linear.bias");
    adaln_weight_ = &register_parameter("adaLN_modulation.1.weight");
    adaln_bias_ = &register_parameter("adaLN_modulation.1.bias");
}

Tensor ZImageFinalLayer::layer_norm_no_affine(const Tensor& x) const {
    auto orig_dtype = x.dtype();
    auto xf = x.to(ov::element::f32);
    auto mean = xf.mean(-1, true);
    auto diff = xf - mean;
    auto var = diff.pow(2.0f).mean(-1, true);
    auto norm = diff * (var + eps_).rsqrt();
    return norm.to(orig_dtype);
}

Tensor ZImageFinalLayer::forward(const Tensor& hidden_states, const Tensor& adaln_input) const {
    auto normed = layer_norm_no_affine(hidden_states);
    auto mod = ops::silu(adaln_input);
    auto scale = add_bias_if_present(ops::linear(mod, adaln_weight_->value()),
                                     adaln_bias_ ? &adaln_bias_->value() : nullptr);
    scale = scale + 1.0f;
    auto scaled = normed * scale.unsqueeze(1);
    return add_bias_if_present(ops::linear(scaled, linear_weight_->value()),
                               linear_bias_ ? &linear_bias_->value() : nullptr);
}

ZImageDiTModel::ZImageDiTModel(BuilderContext& ctx, const ZImageDiTConfig& cfg, Module* parent)
    : Module("", ctx, parent),
      cfg_(cfg),
      t_embedder_(ctx,
                  "t_embedder",
                  std::min(cfg.dim, cfg.adaln_embed_dim),
                  cfg.t_mid_dim,
                  cfg.frequency_embedding_size,
                  cfg.max_period,
                  this),
      x_embedder_(ctx, "all_x_embedder.2-1", this),
      cap_norm_(ctx, "cap_embedder.0", cfg.norm_eps, this),
      cap_linear_(ctx, "cap_embedder.1", this),
      noise_refiner_(),
      context_refiner_(),
      layers_(),
      final_layer_(ctx,
                   "all_final_layer.2-1",
                   cfg.dim,
                   cfg.patch_dim(),
                   cfg.adaln_embed_dim,
                   this) {
    register_module("t_embedder", &t_embedder_);
    register_module("all_x_embedder.2-1", &x_embedder_);
    register_module("cap_embedder.0", &cap_norm_);
    register_module("cap_embedder.1", &cap_linear_);
    register_module("all_final_layer.2-1", &final_layer_);

    x_pad_token_ = &register_parameter("x_pad_token");
    cap_pad_token_ = &register_parameter("cap_pad_token");

    noise_refiner_.reserve(static_cast<size_t>(cfg.n_refiner_layers));
    for (int32_t i = 0; i < cfg.n_refiner_layers; ++i) {
        const std::string name = "noise_refiner." + std::to_string(i);
        noise_refiner_.emplace_back(ctx, name, cfg, true, this);
        register_module(name, &noise_refiner_.back());
    }

    context_refiner_.reserve(static_cast<size_t>(cfg.n_refiner_layers));
    for (int32_t i = 0; i < cfg.n_refiner_layers; ++i) {
        const std::string name = "context_refiner." + std::to_string(i);
        context_refiner_.emplace_back(ctx, name, cfg, false, this);
        register_module(name, &context_refiner_.back());
    }

    layers_.reserve(static_cast<size_t>(cfg.n_layers));
    for (int32_t i = 0; i < cfg.n_layers; ++i) {
        const std::string name = "layers." + std::to_string(i);
        layers_.emplace_back(ctx, name, cfg, true, this);
        register_module(name, &layers_.back());
    }
}

Tensor ZImageDiTModel::apply_pad_token(const Tensor& hidden_states,
                                       const Tensor& mask,
                                       const WeightParameter& pad_token) const {
    auto* ctx = hidden_states.context();
    auto mask_bool = mask.to(ov::element::boolean).unsqueeze(2);
    auto pad = pad_token.value().unsqueeze(1);
    auto pad_broadcast = shape::broadcast_to(pad, shape::of(hidden_states));
    return ops::where(mask_bool, hidden_states, pad_broadcast);
}

Tensor ZImageDiTModel::forward(const Tensor& x_tokens,
                               const Tensor& x_mask,
                               const Tensor& cap_feats,
                               const Tensor& cap_mask,
                               const Tensor& timesteps,
                               const Tensor& x_rope_cos,
                               const Tensor& x_rope_sin,
                               const Tensor& cap_rope_cos,
                               const Tensor& cap_rope_sin) {
    auto t_scaled = timesteps * cfg_.t_scale;
    auto adaln_input = t_embedder_.forward(t_scaled);

    auto x = x_embedder_.forward(x_tokens);
    x = apply_pad_token(x, x_mask, *x_pad_token_);

    for (auto& layer : noise_refiner_) {
        x = layer.forward(x, x_mask, x_rope_cos, x_rope_sin, &adaln_input);
    }

    auto cap = cap_linear_.forward(cap_norm_.forward(cap_feats));
    cap = apply_pad_token(cap, cap_mask, *cap_pad_token_);

    for (auto& layer : context_refiner_) {
        cap = layer.forward(cap, cap_mask, cap_rope_cos, cap_rope_sin, nullptr);
    }

    auto unified = ops::concat({x, cap}, 1);
    auto unified_mask = ops::concat({x_mask, cap_mask}, 1);
    auto unified_rope_cos = ops::concat({x_rope_cos, cap_rope_cos}, 1);
    auto unified_rope_sin = ops::concat({x_rope_sin, cap_rope_sin}, 1);

    for (auto& layer : layers_) {
        unified = layer.forward(unified, unified_mask, unified_rope_cos, unified_rope_sin, &adaln_input);
    }

    auto out = final_layer_.forward(unified, adaln_input);
    auto img_len = Tensor(shape::dim(x_tokens, 1), x_tokens.context()).squeeze(0);
    auto indices = ops::range(img_len, 0, 1, ov::element::i64);
    return ops::gather(out, indices, 1);
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
