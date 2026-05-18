// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/wan/modeling_wan_dit.hpp"

#include <cmath>
#include <vector>

#include <openvino/core/except.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/llm.hpp"
#include "modeling/ops/nn.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/ops/shape.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace {

ov::genai::modeling::Tensor add_bias_if_present(const ov::genai::modeling::Tensor& x,
                                                const ov::genai::modeling::Tensor* bias) {
    if (!bias) {
        return x;
    }
    return x + *bias;
}

auto set_name = [](const std::shared_ptr<ov::Node>& node, const std::string& name) {
    node->output(0).set_names({name});
    node->set_friendly_name(name);
};

ov::Output<ov::Node> div_dim(const ov::Output<ov::Node>& dim, int64_t divisor, ov::genai::modeling::OpContext* ctx) {
    auto denom = ov::genai::modeling::ops::const_scalar(ctx, static_cast<int64_t>(divisor));
    auto div = std::make_shared<ov::op::v1::Divide>(dim, denom, ov::op::AutoBroadcastType::NUMPY);
    return std::make_shared<ov::op::v0::Convert>(div, ov::element::i64);
}

ov::genai::modeling::Tensor slice_to(const ov::genai::modeling::Tensor& data,
                                     const ov::Output<ov::Node>& stop,
                                     int64_t axis) {
    auto* ctx = data.context();
    auto start = ov::genai::modeling::ops::const_vec(ctx, std::vector<int64_t>{0});
    auto step = ov::genai::modeling::ops::const_vec(ctx, std::vector<int64_t>{1});
    auto axes = ov::genai::modeling::ops::const_vec(ctx, std::vector<int64_t>{axis});
    auto stop_vec = ov::genai::modeling::Tensor(stop, ctx);
    auto node = std::make_shared<ov::opset13::Slice>(data.output(),
                                                     start,
                                                     stop_vec.output(),
                                                     step,
                                                     axes);
    return ov::genai::modeling::Tensor(node, ctx);
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

WanRotaryPosEmbed::WanRotaryPosEmbed(BuilderContext& ctx,
                                     const std::string& name,
                                     int32_t attention_head_dim,
                                     const std::vector<int32_t>& patch_size,
                                     int32_t max_seq_len,
                                     float theta,
                                     Module* parent)
    : Module(name, ctx, parent),
      patch_size_(patch_size),
      head_dim_(attention_head_dim) {
    if (patch_size_.size() != 3) {
        OPENVINO_THROW("WanRotaryPosEmbed expects patch_size with 3 values");
    }
    if (head_dim_ <= 0 || max_seq_len <= 0) {
        OPENVINO_THROW("WanRotaryPosEmbed requires positive head_dim and max_seq_len");
    }

    const int32_t h_dim = 2 * (head_dim_ / 6);
    const int32_t w_dim = h_dim;
    const int32_t t_dim = head_dim_ - h_dim - w_dim;
    t_half_ = t_dim / 2;
    h_half_ = h_dim / 2;
    w_half_ = w_dim / 2;

    auto build_freqs = [&](int32_t dim, Tensor& cos_out, Tensor& sin_out) {
        const int32_t half = dim / 2;
        std::vector<float> inv_freq(static_cast<size_t>(half));
        for (int32_t i = 0; i < half; ++i) {
            float exponent = static_cast<float>(2 * i) / static_cast<float>(dim);
            inv_freq[static_cast<size_t>(i)] = 1.0f / std::pow(theta, exponent);
        }

        std::vector<float> cos_vals(static_cast<size_t>(max_seq_len * half));
        std::vector<float> sin_vals(static_cast<size_t>(max_seq_len * half));
        for (int32_t pos = 0; pos < max_seq_len; ++pos) {
            for (int32_t i = 0; i < half; ++i) {
                float freq = static_cast<float>(pos) * inv_freq[static_cast<size_t>(i)];
                size_t idx = static_cast<size_t>(pos * half + i);
                cos_vals[idx] = std::cos(freq);
                sin_vals[idx] = std::sin(freq);
            }
        }

        auto* op_ctx = &ctx.op_context();
        Tensor cos_tensor(ops::const_vec(op_ctx, cos_vals), op_ctx);
        Tensor sin_tensor(ops::const_vec(op_ctx, sin_vals), op_ctx);
        cos_out = cos_tensor.reshape({max_seq_len, half}, false);
        sin_out = sin_tensor.reshape({max_seq_len, half}, false);
    };

    build_freqs(t_dim, t_cos_, t_sin_);
    build_freqs(h_dim, h_cos_, h_sin_);
    build_freqs(w_dim, w_cos_, w_sin_);
}

std::pair<Tensor, Tensor> WanRotaryPosEmbed::forward(const Tensor& hidden_states) const {
    auto* ctx = hidden_states.context();
    auto batch = shape::dim(hidden_states, 0);
    auto frames = shape::dim(hidden_states, 2);
    auto height = shape::dim(hidden_states, 3);
    auto width = shape::dim(hidden_states, 4);

    auto ppf = div_dim(frames, patch_size_[0], ctx);
    auto pph = div_dim(height, patch_size_[1], ctx);
    auto ppw = div_dim(width, patch_size_[2], ctx);

    auto one = ops::const_vec(ctx, std::vector<int64_t>{1});
    auto t_half = ops::const_vec(ctx, std::vector<int64_t>{static_cast<int64_t>(t_half_)});
    auto h_half = ops::const_vec(ctx, std::vector<int64_t>{static_cast<int64_t>(h_half_)});
    auto w_half = ops::const_vec(ctx, std::vector<int64_t>{static_cast<int64_t>(w_half_)});

    auto t_cos = slice_to(t_cos_, ppf, 0);
    auto h_cos = slice_to(h_cos_, pph, 0);
    auto w_cos = slice_to(w_cos_, ppw, 0);
    auto t_sin = slice_to(t_sin_, ppf, 0);
    auto h_sin = slice_to(h_sin_, pph, 0);
    auto w_sin = slice_to(w_sin_, ppw, 0);

    auto t_shape = shape::make({ppf, one, one, t_half});
    auto h_shape = shape::make({one, pph, one, h_half});
    auto w_shape = shape::make({one, one, ppw, w_half});

    auto t_target = shape::make({ppf, pph, ppw, t_half});
    auto h_target = shape::make({ppf, pph, ppw, h_half});
    auto w_target = shape::make({ppf, pph, ppw, w_half});

    auto t_cos_b = shape::broadcast_to(t_cos.reshape(t_shape), t_target);
    auto h_cos_b = shape::broadcast_to(h_cos.reshape(h_shape), h_target);
    auto w_cos_b = shape::broadcast_to(w_cos.reshape(w_shape), w_target);

    auto t_sin_b = shape::broadcast_to(t_sin.reshape(t_shape), t_target);
    auto h_sin_b = shape::broadcast_to(h_sin.reshape(h_shape), h_target);
    auto w_sin_b = shape::broadcast_to(w_sin.reshape(w_shape), w_target);

    auto cos = ops::concat({t_cos_b, h_cos_b, w_cos_b}, 3);
    auto sin = ops::concat({t_sin_b, h_sin_b, w_sin_b}, 3);

    auto seq = std::make_shared<ov::op::v1::Multiply>(ppf, pph);
    seq = std::make_shared<ov::op::v1::Multiply>(seq, ppw);
    auto total_half = ops::const_vec(ctx, std::vector<int64_t>{
                                             static_cast<int64_t>(t_half_ + h_half_ + w_half_)});
    auto flat_shape = shape::make({one, seq, total_half});
    auto cos_flat = cos.reshape(flat_shape);
    auto sin_flat = sin.reshape(flat_shape);

    auto broadcast_shape = shape::make({batch, seq, total_half});
    cos_flat = shape::broadcast_to(cos_flat, broadcast_shape);
    sin_flat = shape::broadcast_to(sin_flat, broadcast_shape);
    return {cos_flat, sin_flat};
}

WanTimestepEmbedder::WanTimestepEmbedder(BuilderContext& ctx,
                                         const std::string& name,
                                         int32_t freq_dim,
                                         int32_t out_dim,
                                         Module* parent)
    : Module(name, ctx, parent),
      freq_dim_(freq_dim),
      out_dim_(out_dim) {
    fc1_weight_ = &register_parameter("linear_1.weight");
    fc1_bias_ = &register_parameter("linear_1.bias");
    fc2_weight_ = &register_parameter("linear_2.weight");
    fc2_bias_ = &register_parameter("linear_2.bias");
}

Tensor WanTimestepEmbedder::timestep_embedding(const Tensor& t) const {
    auto* ctx = t.context();
    const int32_t half = freq_dim_ / 2;
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

    if (freq_dim_ % 2 != 0) {
        auto batch = shape::dim(t, 0);
        auto one = ops::const_vec(ctx, std::vector<int64_t>{1});
        auto target = shape::make({batch, one});
        auto zero = Tensor(ops::const_scalar(ctx, 0.0f), ctx);
        auto zeros = shape::broadcast_to(zero, target);
        emb = ops::concat({emb, zeros}, 1);
    }
    return emb;
}

Tensor WanTimestepEmbedder::forward(const Tensor& t) const {
    auto t_freq = timestep_embedding(t);
    auto weight_dtype = fc1_weight_->value().dtype();
    t_freq = t_freq.to(weight_dtype);
    auto fc1 = add_bias_if_present(ops::linear(t_freq, fc1_weight_->value()),
                                   fc1_bias_ ? &fc1_bias_->value() : nullptr);
    auto act = ops::silu(fc1);
    return add_bias_if_present(ops::linear(act, fc2_weight_->value()),
                               fc2_bias_ ? &fc2_bias_->value() : nullptr);
}

WanTextProjection::WanTextProjection(BuilderContext& ctx,
                                     const std::string& name,
                                     int32_t in_features,
                                     int32_t hidden_size,
                                     int32_t out_features,
                                     Module* parent)
    : Module(name, ctx, parent) {
    (void)in_features;
    (void)hidden_size;
    (void)out_features;
    fc1_weight_ = &register_parameter("linear_1.weight");
    fc1_bias_ = &register_parameter("linear_1.bias");
    fc2_weight_ = &register_parameter("linear_2.weight");
    fc2_bias_ = &register_parameter("linear_2.bias");
}

Tensor WanTextProjection::forward(const Tensor& input) const {
    auto x = input.to(fc1_weight_->value().dtype());
    x = add_bias_if_present(ops::linear(x, fc1_weight_->value()),
                            fc1_bias_ ? &fc1_bias_->value() : nullptr);
    x = ops::nn::gelu(x, true);
    x = add_bias_if_present(ops::linear(x, fc2_weight_->value()),
                            fc2_bias_ ? &fc2_bias_->value() : nullptr);
    return x;
}

WanFeedForward::WanFeedForward(BuilderContext& ctx,
                               const std::string& name,
                               int32_t dim,
                               int32_t ffn_dim,
                               bool approximate,
                               Module* parent)
    : Module(name, ctx, parent),
      gelu_approximate_(approximate) {
    (void)dim;
    (void)ffn_dim;
    proj_weight_ = &register_parameter("net.0.proj.weight");
    proj_bias_ = &register_parameter("net.0.proj.bias");
    out_weight_ = &register_parameter("net.2.weight");
    out_bias_ = &register_parameter("net.2.bias");
}

Tensor WanFeedForward::forward(const Tensor& hidden_states) const {
    auto x = hidden_states.to(proj_weight_->value().dtype());
    x = add_bias_if_present(ops::linear(x, proj_weight_->value()),
                            proj_bias_ ? &proj_bias_->value() : nullptr);
    x = ops::nn::gelu(x, gelu_approximate_);
    x = add_bias_if_present(ops::linear(x, out_weight_->value()),
                            out_bias_ ? &out_bias_->value() : nullptr);
    return x;
}

WanImageEmbedding::WanImageEmbedding(BuilderContext& ctx,
                                     const std::string& name,
                                     int32_t in_features,
                                     int32_t out_features,
                                     std::optional<int32_t> pos_embed_seq_len,
                                     Module* parent)
    : Module(name, ctx, parent),
      norm1_(ctx, "norm1", 1e-6f, true, true, this),
      ff_(ctx, "ff", in_features, out_features, false, this),
      norm2_(ctx, "norm2", 1e-6f, true, true, this) {
    register_module("norm1", &norm1_);
    register_module("ff", &ff_);
    register_module("norm2", &norm2_);

    if (pos_embed_seq_len) {
        pos_embed_ = &register_parameter("pos_embed");
    }
}

Tensor WanImageEmbedding::forward(const Tensor& encoder_hidden_states_image) const {
    auto x = encoder_hidden_states_image;
    if (pos_embed_) {
        auto* ctx = x.context();
        auto seq = shape::dim(x, 1);
        auto embed = shape::dim(x, 2);
        auto two = ops::const_scalar(ctx, static_cast<int64_t>(2));
        auto seq2 = std::make_shared<ov::op::v1::Multiply>(seq, two);
        auto neg_one = ops::const_vec(ctx, std::vector<int64_t>{-1});
        auto reshape_shape = shape::make({neg_one, seq2, embed});
        x = x.reshape(reshape_shape);
        x = x + pos_embed_->value();
    }

    x = norm1_.forward(x);
    x = ff_.forward(x);
    return norm2_.forward(x);
}

WanTimeTextImageEmbedding::WanTimeTextImageEmbedding(BuilderContext& ctx,
                                                     const std::string& name,
                                                     int32_t dim,
                                                     int32_t time_freq_dim,
                                                     int32_t time_proj_dim,
                                                     int32_t text_embed_dim,
                                                     std::optional<int32_t> image_embed_dim,
                                                     std::optional<int32_t> pos_embed_seq_len,
                                                     Module* parent)
    : Module(name, ctx, parent),
      dim_(dim),
      time_proj_dim_(time_proj_dim),
      time_embedder_(ctx, "time_embedder", time_freq_dim, dim, this),
      text_embedder_(ctx, "text_embedder", text_embed_dim, dim, dim, this) {
    register_module("time_embedder", &time_embedder_);
    register_module("text_embedder", &text_embedder_);

    time_proj_weight_ = &register_parameter("time_proj.weight");
    time_proj_bias_ = &register_parameter("time_proj.bias");

    if (image_embed_dim.has_value()) {
        image_embedder_ = std::make_unique<WanImageEmbedding>(ctx,
                                                              "image_embedder",
                                                              image_embed_dim.value(),
                                                              dim,
                                                              pos_embed_seq_len,
                                                              this);
        register_module("image_embedder", image_embedder_.get());
    }
}

WanConditionEmbeddings WanTimeTextImageEmbedding::forward(const Tensor& timestep,
                                                          const Tensor& encoder_hidden_states,
                                                          const Tensor* encoder_hidden_states_image) const {
    auto temb = time_embedder_.forward(timestep);
    temb = temb.to(encoder_hidden_states.dtype());

    auto temb_act = ops::silu(temb);
    temb_act = temb_act.to(time_proj_weight_->value().dtype());
    auto timestep_proj = add_bias_if_present(ops::linear(temb_act, time_proj_weight_->value()),
                                             time_proj_bias_ ? &time_proj_bias_->value() : nullptr);

    auto text_embeds = text_embedder_.forward(encoder_hidden_states);
    std::optional<Tensor> image_embeds;
    if (encoder_hidden_states_image && image_embedder_) {
        image_embeds = image_embedder_->forward(*encoder_hidden_states_image);
    }

    WanConditionEmbeddings out{temb, timestep_proj, text_embeds, image_embeds};
    return out;
}

WanAttention::WanAttention(BuilderContext& ctx,
                           const std::string& name,
                           int32_t dim,
                           int32_t num_heads,
                           int32_t head_dim,
                           float eps,
                           const std::string& qk_norm,
                           std::optional<int32_t> added_kv_proj_dim,
                           Module* parent)
    : Module(name, ctx, parent),
      num_heads_(num_heads),
      head_dim_(head_dim),
      inner_dim_(num_heads * head_dim),
      scaling_(1.0f / std::sqrt(static_cast<float>(head_dim))),
      use_qk_norm_(qk_norm == "rms_norm_across_heads" || qk_norm == "rms_norm"),
      added_kv_proj_dim_(added_kv_proj_dim),
      q_norm_(ctx, "norm_q", eps, this),
      k_norm_(ctx, "norm_k", eps, this) {
    if (num_heads_ <= 0 || head_dim_ <= 0) {
        OPENVINO_THROW("Invalid WanAttention head configuration");
    }

    register_module("norm_q", &q_norm_);
    register_module("norm_k", &k_norm_);

    q_weight_ = &register_parameter("to_q.weight");
    q_bias_ = &register_parameter("to_q.bias");
    k_weight_ = &register_parameter("to_k.weight");
    k_bias_ = &register_parameter("to_k.bias");
    v_weight_ = &register_parameter("to_v.weight");
    v_bias_ = &register_parameter("to_v.bias");
    o_weight_ = &register_parameter("to_out.0.weight");
    o_bias_ = &register_parameter("to_out.0.bias");

    if (!use_qk_norm_) {
        q_norm_.weight_param().set_optional(true);
        k_norm_.weight_param().set_optional(true);
    }

    if (added_kv_proj_dim_) {
        add_k_weight_ = &register_parameter("add_k_proj.weight");
        add_k_bias_ = &register_parameter("add_k_proj.bias");
        add_v_weight_ = &register_parameter("add_v_proj.weight");
        add_v_bias_ = &register_parameter("add_v_proj.bias");
        added_k_norm_ = std::make_unique<RMSNorm>(ctx, "norm_added_k", eps, this);
        register_module("norm_added_k", added_k_norm_.get());
    }
}

Tensor WanAttention::apply_linear(const Tensor& input, const Tensor& weight, const Tensor* bias) const {
    auto x = input.to(weight.dtype());
    auto out = ops::linear(x, weight);
    return add_bias_if_present(out, bias);
}

Tensor WanAttention::forward(const Tensor& hidden_states,
                             const Tensor* encoder_hidden_states,
                             const Tensor* encoder_hidden_states_image,
                             const Tensor* rotary_cos,
                             const Tensor* rotary_sin) const {
    const Tensor* enc = encoder_hidden_states ? encoder_hidden_states : &hidden_states;
    auto q = apply_linear(hidden_states, q_weight_->value(), q_bias_ ? &q_bias_->value() : nullptr);
    auto k = apply_linear(*enc, k_weight_->value(), k_bias_ ? &k_bias_->value() : nullptr);
    auto v = apply_linear(*enc, v_weight_->value(), v_bias_ ? &v_bias_->value() : nullptr);

    if (use_qk_norm_ && q_norm_.weight_param().is_bound()) {
        q = q_norm_.forward(q);
    }
    if (use_qk_norm_ && k_norm_.weight_param().is_bound()) {
        k = k_norm_.forward(k);
    }

    auto q_heads = q.reshape({0, 0, num_heads_, head_dim_}).permute({0, 2, 1, 3});
    auto k_heads = k.reshape({0, 0, num_heads_, head_dim_}).permute({0, 2, 1, 3});
    auto v_heads = v.reshape({0, 0, num_heads_, head_dim_}).permute({0, 2, 1, 3});

    auto* policy = &ctx().op_policy();

    if (rotary_cos || rotary_sin) {
        if (!rotary_cos || !rotary_sin) {
            OPENVINO_THROW("WanAttention requires both rotary_cos and rotary_sin");
        }
        auto cos = rotary_cos->to(q_heads.dtype());
        auto sin = rotary_sin->to(q_heads.dtype());
        q_heads = ops::llm::apply_rope_interleave(q_heads, cos, sin, head_dim_, policy);
        k_heads = ops::llm::apply_rope_interleave(k_heads, cos, sin, head_dim_, policy);
    }

    auto context = ops::llm::sdpa(q_heads, k_heads, v_heads, scaling_, 3, nullptr, false, policy);

    if (encoder_hidden_states_image && added_kv_proj_dim_) {
        auto k_img = apply_linear(*encoder_hidden_states_image,
                                  add_k_weight_->value(),
                                  add_k_bias_ ? &add_k_bias_->value() : nullptr);
        auto v_img = apply_linear(*encoder_hidden_states_image,
                                  add_v_weight_->value(),
                                  add_v_bias_ ? &add_v_bias_->value() : nullptr);
        if (added_k_norm_ && added_k_norm_->weight_param().is_bound()) {
            k_img = added_k_norm_->forward(k_img);
        }
        auto k_img_heads = k_img.reshape({0, 0, num_heads_, head_dim_}).permute({0, 2, 1, 3});
        auto v_img_heads = v_img.reshape({0, 0, num_heads_, head_dim_}).permute({0, 2, 1, 3});
        auto context_img = ops::llm::sdpa(q_heads, k_img_heads, v_img_heads, scaling_, 3, nullptr, false, policy);
        context = context + context_img;
    }

    auto merged = context.permute({0, 2, 1, 3}).reshape({0, 0, inner_dim_});
    return apply_linear(merged, o_weight_->value(), o_bias_ ? &o_bias_->value() : nullptr);
}

WanTransformerBlock::WanTransformerBlock(BuilderContext& ctx,
                                         const std::string& name,
                                         const WanTransformer3DConfig& cfg,
                                         Module* parent)
    : Module(name, ctx, parent),
      inner_dim_(cfg.inner_dim()),
      eps_(cfg.eps),
      cross_attn_norm_(cfg.cross_attn_norm),
      norm1_(ctx, "norm1", cfg.eps, false, false, this),
      attn1_(ctx,
             "attn1",
             inner_dim_,
             cfg.num_attention_heads,
             cfg.attention_head_dim,
             cfg.eps,
             cfg.qk_norm,
             std::nullopt,
             this),
      attn2_(ctx,
             "attn2",
             inner_dim_,
             cfg.num_attention_heads,
             cfg.attention_head_dim,
             cfg.eps,
             cfg.qk_norm,
             cfg.added_kv_proj_dim,
             this),
      ffn_(ctx, "ffn", inner_dim_, cfg.ffn_dim, true, this),
      norm3_(ctx, "norm3", cfg.eps, false, false, this) {
    register_module("norm1", &norm1_);
    register_module("attn1", &attn1_);
    register_module("attn2", &attn2_);
    register_module("ffn", &ffn_);
    register_module("norm3", &norm3_);

    if (cross_attn_norm_) {
        norm2_ = std::make_unique<FP32LayerNorm>(ctx, "norm2", cfg.eps, true, true, this);
        register_module("norm2", norm2_.get());
    }

    scale_shift_table_ = &register_parameter("scale_shift_table");
}

Tensor WanTransformerBlock::forward(const Tensor& hidden_states,
                                    const Tensor& encoder_hidden_states,
                                    const Tensor& temb,
                                    const Tensor& rotary_cos,
                                    const Tensor& rotary_sin,
                                    const Tensor* encoder_hidden_states_image) const {
    auto temb_f = temb.to(ov::element::f32);
    auto table = scale_shift_table_->value().to(ov::element::f32);
    auto mod = table + temb_f;

    auto shift_msa = ops::slice(mod, 0, 1, 1, 1);
    auto scale_msa = ops::slice(mod, 1, 2, 1, 1);
    auto gate_msa = ops::slice(mod, 2, 3, 1, 1);
    auto c_shift_msa = ops::slice(mod, 3, 4, 1, 1);
    auto c_scale_msa = ops::slice(mod, 4, 5, 1, 1);
    auto c_gate_msa = ops::slice(mod, 5, 6, 1, 1);

    auto norm_in = norm1_.forward(hidden_states.to(ov::element::f32));
    norm_in = (norm_in * (scale_msa + 1.0f) + shift_msa).to(hidden_states.dtype());
    auto attn_out = attn1_.forward(norm_in, nullptr, nullptr, &rotary_cos, &rotary_sin);

    auto hs_f = hidden_states.to(ov::element::f32);
    auto attn_f = attn_out.to(ov::element::f32);
    auto hs = (hs_f + attn_f * gate_msa).to(hidden_states.dtype());

    auto cross_in = hs;
    if (cross_attn_norm_ && norm2_) {
        cross_in = norm2_->forward(hs.to(ov::element::f32)).to(hs.dtype());
    }

    auto cross_out = attn2_.forward(cross_in,
                                    &encoder_hidden_states,
                                    encoder_hidden_states_image,
                                    nullptr,
                                    nullptr);
    if (cross_out.dtype() != hs.dtype()) {
        cross_out = cross_out.to(hs.dtype());
    }
    hs = hs + cross_out;

    auto ffn_in = norm3_.forward(hs.to(ov::element::f32));
    ffn_in = (ffn_in * (c_scale_msa + 1.0f) + c_shift_msa).to(hs.dtype());
    auto ffn_out = ffn_.forward(ffn_in);

    auto ffn_f = ffn_out.to(ov::element::f32);
    hs = (hs.to(ov::element::f32) + ffn_f * c_gate_msa).to(hs.dtype());
    return hs;
}

WanTransformer3DModel::WanTransformer3DModel(BuilderContext& ctx,
                                             const WanTransformer3DConfig& cfg,
                                             Module* parent)
    : Module("", ctx, parent),
      cfg_(cfg),
      inner_dim_(cfg.inner_dim()),
      rope_(ctx,
            "rope",
            cfg.attention_head_dim,
            cfg.patch_size,
            cfg.rope_max_seq_len,
            10000.0f,
            this),
      condition_embedder_(ctx,
                          "condition_embedder",
                          inner_dim_,
                          cfg.freq_dim,
                          inner_dim_ * 6,
                          cfg.text_dim,
                          cfg.image_dim,
                          cfg.pos_embed_seq_len,
                          this),
      norm_out_(ctx, "norm_out", cfg.eps, false, false, this) {
    register_module("rope", &rope_);
    register_module("condition_embedder", &condition_embedder_);
    register_module("norm_out", &norm_out_);

    patch_weight_ = &register_parameter("patch_embedding.weight");
    patch_bias_ = &register_parameter("patch_embedding.bias");
    proj_out_weight_ = &register_parameter("proj_out.weight");
    proj_out_bias_ = &register_parameter("proj_out.bias");
    scale_shift_table_ = &register_parameter("scale_shift_table");

    blocks_.reserve(static_cast<size_t>(cfg_.num_layers));
    for (int32_t i = 0; i < cfg_.num_layers; ++i) {
        std::string block_name = "blocks." + std::to_string(i);
        blocks_.emplace_back(ctx, block_name, cfg_, this);
        register_module(block_name, &blocks_.back());
    }
}

Tensor WanTransformer3DModel::forward(const Tensor& hidden_states,
                                      const Tensor& timestep,
                                      const Tensor& encoder_hidden_states,
                                      const Tensor* encoder_hidden_states_image) {
    auto* ctx = hidden_states.context();
    auto batch = shape::dim(hidden_states, 0);
    auto frames = shape::dim(hidden_states, 2);
    auto height = shape::dim(hidden_states, 3);
    auto width = shape::dim(hidden_states, 4);

    auto ppf = div_dim(frames, cfg_.patch_size[0], ctx);
    auto pph = div_dim(height, cfg_.patch_size[1], ctx);
    auto ppw = div_dim(width, cfg_.patch_size[2], ctx);

    auto rope = rope_.forward(hidden_states);

    auto x = hidden_states.to(patch_weight_->value().dtype());
    x = ops::nn::conv3d(x,
                        patch_weight_->value(),
                        patch_bias_->value(),
                        {cfg_.patch_size[0], cfg_.patch_size[1], cfg_.patch_size[2]},
                        {0, 0, 0},
                        {0, 0, 0});

    auto seq = std::make_shared<ov::op::v1::Multiply>(ppf, pph);
    seq = std::make_shared<ov::op::v1::Multiply>(seq, ppw);
    auto embed = shape::dim(x, 1);
    auto shape_tokens = shape::make({batch, embed, seq});
    auto tokens = x.reshape(shape_tokens).permute({0, 2, 1});

    auto cond = condition_embedder_.forward(timestep, encoder_hidden_states, encoder_hidden_states_image);
    auto temb = cond.temb;
    auto timestep_proj = cond.timestep_proj.reshape({0, 6, inner_dim_});

    const Tensor* image_embeds = cond.image_embeds ? &*cond.image_embeds : nullptr;

    auto hs = tokens;
    for (auto& block : blocks_) {
        hs = block.forward(hs,
                           cond.text_embeds,
                           timestep_proj,
                           rope.first,
                           rope.second,
                           image_embeds);
    }

    auto table = scale_shift_table_->value().to(ov::element::f32);
    auto temb_f = temb.to(ov::element::f32).unsqueeze(1);
    auto mod = table + temb_f;
    auto shift = ops::slice(mod, 0, 1, 1, 1);
    auto scale = ops::slice(mod, 1, 2, 1, 1);

    auto normed = norm_out_.forward(hs.to(ov::element::f32));
    normed = (normed * (scale + 1.0f) + shift).to(hs.dtype());

    auto proj_in = normed.to(proj_out_weight_->value().dtype());
    auto out = add_bias_if_present(ops::linear(proj_in, proj_out_weight_->value()),
                                   proj_out_bias_ ? &proj_out_bias_->value() : nullptr);

    auto p_t = ops::const_scalar(ctx, static_cast<int64_t>(cfg_.patch_size[0]));
    auto p_h = ops::const_scalar(ctx, static_cast<int64_t>(cfg_.patch_size[1]));
    auto p_w = ops::const_scalar(ctx, static_cast<int64_t>(cfg_.patch_size[2]));

    auto out_ch = ops::const_vec(ctx, std::vector<int64_t>{cfg_.out_channels});
    auto p_t_vec = ops::const_vec(ctx, std::vector<int64_t>{cfg_.patch_size[0]});
    auto p_h_vec = ops::const_vec(ctx, std::vector<int64_t>{cfg_.patch_size[1]});
    auto p_w_vec = ops::const_vec(ctx, std::vector<int64_t>{cfg_.patch_size[2]});

    auto reshape_shape = shape::make({batch, ppf, pph, ppw, p_t_vec, p_h_vec, p_w_vec, out_ch});
    auto unpatched = out.reshape(reshape_shape).permute({0, 7, 1, 4, 2, 5, 3, 6});

    auto out_frames = std::make_shared<ov::op::v1::Multiply>(ppf, p_t);
    auto out_height = std::make_shared<ov::op::v1::Multiply>(pph, p_h);
    auto out_width = std::make_shared<ov::op::v1::Multiply>(ppw, p_w);
    auto final_shape = shape::make({batch, out_ch, out_frames, out_height, out_width});
    return unpatched.reshape(final_shape);
}

std::shared_ptr<ov::Model> create_wan_dit_model(
    const WanTransformer3DConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    ctx.op_policy().use_internal_rope = false;
    WanTransformer3DModel model(ctx, cfg);

    WanWeightMapping::apply_transformer_packed_mapping(model);
    ov::genai::modeling::weights::load_model(model, source, finalizer);

    auto latents = ctx.parameter("hidden_states",
                                 ov::element::f32,
                                 ov::PartialShape{-1, cfg.in_channels, -1, -1, -1});
    auto timesteps = ctx.parameter("timestep", ov::element::f32, ov::PartialShape{-1});
    auto text = ctx.parameter("encoder_hidden_states",
                              ov::element::f32,
                              ov::PartialShape{-1, -1, cfg.text_dim});

    Tensor output;
    if (cfg.use_image_condition()) {
        auto image = ctx.parameter("encoder_hidden_states_image",
                                   ov::element::f32,
                                   ov::PartialShape{-1, -1, cfg.image_dim.value()});
        output = model.forward(latents, timesteps, text, &image);
    } else {
        output = model.forward(latents, timesteps, text, nullptr);
    }

    auto result = std::make_shared<ov::op::v0::Result>(output.output());
    set_name(result, "sample");
    return ctx.build_model({result->output(0)});
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
