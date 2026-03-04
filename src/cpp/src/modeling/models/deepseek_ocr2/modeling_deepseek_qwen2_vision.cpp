// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/deepseek_ocr2/modeling_deepseek_qwen2_vision.hpp"

#include <cmath>

#include <openvino/openvino.hpp>
#include <openvino/core/except.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/llm.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/ops/shape.hpp"
#include "modeling/ops/tensor_ops.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace {

ov::genai::modeling::Tensor add_bias_if_present(const ov::genai::modeling::Tensor& x,
                                                const ov::genai::modeling::Tensor* bias) {
    if (!bias) {
        return x;
    }
    return x + *bias;
}

auto set_name = [](auto node, const std::string& name) {
    node->output(0).set_names({name});
    node->set_friendly_name(name);
};

ov::Output<ov::Node> as_shape_dim(const ov::Output<ov::Node>& value, ov::genai::modeling::OpContext* ctx) {
    auto rank = value.get_partial_shape().rank();
    if (rank.is_static() && rank.get_length() == 0) {
        ov::genai::modeling::Tensor t(value, ctx);
        return t.unsqueeze(0).output();
    }
    return value;
}

ov::genai::modeling::Tensor logical_and(const ov::genai::modeling::Tensor& a,
                                        const ov::genai::modeling::Tensor& b) {
    auto* ctx = a.context() ? a.context() : b.context();
    auto node = std::make_shared<ov::op::v1::LogicalAnd>(a.output(), b.output());
    return ov::genai::modeling::Tensor(node, ctx);
}

ov::genai::modeling::Tensor logical_or(const ov::genai::modeling::Tensor& a,
                                       const ov::genai::modeling::Tensor& b) {
    auto* ctx = a.context() ? a.context() : b.context();
    auto node = std::make_shared<ov::op::v1::LogicalOr>(a.output(), b.output());
    return ov::genai::modeling::Tensor(node, ctx);
}

ov::genai::modeling::Tensor equal(const ov::genai::modeling::Tensor& a,
                                  const ov::genai::modeling::Tensor& b) {
    auto* ctx = a.context() ? a.context() : b.context();
    auto node = std::make_shared<ov::op::v1::Equal>(a.output(), b.output());
    return ov::genai::modeling::Tensor(node, ctx);
}

ov::genai::modeling::Tensor broadcast_positions(const ov::genai::modeling::Tensor& positions,
                                                const ov::Output<ov::Node>& batch,
                                                const ov::Output<ov::Node>& seq,
                                                ov::genai::modeling::OpContext* ctx) {
    auto shape = ov::genai::modeling::shape::make({batch, as_shape_dim(seq, ctx)});
    return ov::genai::modeling::shape::broadcast_to(positions, shape);
}

ov::genai::modeling::Tensor slice_sequence(const ov::genai::modeling::Tensor& x,
                                           const ov::genai::modeling::Tensor& start,
                                           const ov::genai::modeling::Tensor& end) {
    auto* ctx = x.context();
    auto batch = ov::genai::modeling::Tensor(ov::genai::modeling::shape::dim(x, 0), ctx);
    auto hidden = ov::genai::modeling::Tensor(ov::genai::modeling::shape::dim(x, 2), ctx);
    auto zero_vec = ov::genai::modeling::Tensor(ov::genai::modeling::ops::const_vec(ctx, std::vector<int64_t>{0}), ctx);
    auto one_vec = ov::genai::modeling::Tensor(ov::genai::modeling::ops::const_vec(ctx, std::vector<int64_t>{1}), ctx);

    auto begin = ov::genai::modeling::ops::concat({zero_vec, start.unsqueeze(0), zero_vec}, 0);
    auto end_vec = ov::genai::modeling::ops::concat({batch, end.unsqueeze(0), hidden}, 0);
    auto step = ov::genai::modeling::ops::concat({one_vec, one_vec, one_vec}, 0);

    auto node = std::make_shared<ov::op::v8::Slice>(x.output(), begin.output(), end_vec.output(), step.output());
    return ov::genai::modeling::Tensor(node, ctx);
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

int32_t DeepseekQwen2VisionConfig::head_dim() const {
    if (num_attention_heads <= 0) {
        return 0;
    }
    return hidden_size / num_attention_heads;
}

Tensor build_qwen2_attention_mask(const Tensor& token_type_ids, float masked_value) {
    auto* ctx = token_type_ids.context();
    auto zero = Tensor(ops::const_scalar(ctx, static_cast<int64_t>(0)), ctx);
    auto one = Tensor(ops::const_scalar(ctx, static_cast<int64_t>(1)), ctx);

    auto is_image = equal(token_type_ids, zero);
    auto is_text = equal(token_type_ids, one);

    auto row_image = is_image.unsqueeze(2);  // [B, L, 1]
    auto col_image = is_image.unsqueeze(1);  // [B, 1, L]
    auto row_text = is_text.unsqueeze(2);
    auto col_text = is_text.unsqueeze(1);

    auto image_image = logical_and(row_image, col_image);
    auto text_image = logical_and(row_text, col_image);

    auto seq_len = Tensor(shape::dim(token_type_ids, 1), ctx).squeeze(0);
    auto positions = ops::range(seq_len, 0, 1, ov::element::i64);
    auto row = positions.unsqueeze(1);
    auto col = positions.unsqueeze(0);
    auto causal = ops::less_equal(col, row);

    auto batch = shape::dim(token_type_ids, 0);
    auto seq_dim = as_shape_dim(seq_len.output(), ctx);
    auto causal_shape = shape::make({batch, seq_dim, seq_dim});
    auto causal_b = shape::broadcast_to(causal.unsqueeze(0), causal_shape);

    auto text_text = logical_and(row_text, col_text);
    text_text = logical_and(text_text, causal_b);

    auto allowed = logical_or(image_image, text_image);
    allowed = logical_or(allowed, text_text);

    auto zero_f = Tensor(ops::const_scalar(ctx, 0.0f), ctx);
    auto neg_f = Tensor(ops::const_scalar(ctx, masked_value), ctx);
    auto mask = ops::where(allowed, zero_f, neg_f);
    return mask.unsqueeze(1);
}

DeepseekQwen2Attention::DeepseekQwen2Attention(BuilderContext& ctx,
                                               const std::string& name,
                                               const DeepseekQwen2VisionConfig& cfg,
                                               Module* parent)
    : Module(name, ctx, parent),
      num_heads_(cfg.num_attention_heads),
      num_kv_heads_(cfg.num_key_value_heads > 0 ? cfg.num_key_value_heads : cfg.num_attention_heads),
      head_dim_(cfg.head_dim()),
      hidden_size_(cfg.hidden_size),
      scaling_(1.0f / std::sqrt(static_cast<float>(head_dim_))) {
    if (num_heads_ <= 0 || head_dim_ <= 0) {
        OPENVINO_THROW("Invalid DeepseekQwen2 attention configuration");
    }
    if (num_heads_ % num_kv_heads_ != 0) {
        OPENVINO_THROW("num_attention_heads must be divisible by num_key_value_heads");
    }
    q_proj_param_ = &register_parameter("q_proj.weight");
    k_proj_param_ = &register_parameter("k_proj.weight");
    v_proj_param_ = &register_parameter("v_proj.weight");
    o_proj_param_ = &register_parameter("o_proj.weight");

    q_bias_param_ = &register_parameter("q_proj.bias");
    k_bias_param_ = &register_parameter("k_proj.bias");
    v_bias_param_ = &register_parameter("v_proj.bias");
    o_bias_param_ = &register_parameter("o_proj.bias");

    // Qwen2 weights do not include o_proj bias; keep it optional.
    o_bias_param_->set_optional(true);

    if (!cfg.attention_bias) {
        q_bias_param_->set_optional(true);
        k_bias_param_->set_optional(true);
        v_bias_param_->set_optional(true);
    }
}

const Tensor& DeepseekQwen2Attention::q_proj_weight() const {
    if (!q_proj_param_) {
        OPENVINO_THROW("DeepseekQwen2Attention q_proj parameter not registered");
    }
    return q_proj_param_->value();
}

const Tensor& DeepseekQwen2Attention::k_proj_weight() const {
    if (!k_proj_param_) {
        OPENVINO_THROW("DeepseekQwen2Attention k_proj parameter not registered");
    }
    return k_proj_param_->value();
}

const Tensor& DeepseekQwen2Attention::v_proj_weight() const {
    if (!v_proj_param_) {
        OPENVINO_THROW("DeepseekQwen2Attention v_proj parameter not registered");
    }
    return v_proj_param_->value();
}

const Tensor& DeepseekQwen2Attention::o_proj_weight() const {
    if (!o_proj_param_) {
        OPENVINO_THROW("DeepseekQwen2Attention o_proj parameter not registered");
    }
    return o_proj_param_->value();
}

const Tensor* DeepseekQwen2Attention::q_proj_bias() const {
    return (q_bias_param_ && q_bias_param_->is_bound()) ? &q_bias_param_->value() : nullptr;
}

const Tensor* DeepseekQwen2Attention::k_proj_bias() const {
    return (k_bias_param_ && k_bias_param_->is_bound()) ? &k_bias_param_->value() : nullptr;
}

const Tensor* DeepseekQwen2Attention::v_proj_bias() const {
    return (v_bias_param_ && v_bias_param_->is_bound()) ? &v_bias_param_->value() : nullptr;
}

const Tensor* DeepseekQwen2Attention::o_proj_bias() const {
    return (o_bias_param_ && o_bias_param_->is_bound()) ? &o_bias_param_->value() : nullptr;
}

Tensor DeepseekQwen2Attention::forward(const Tensor& hidden_states,
                                       const Tensor& rope_cos,
                                       const Tensor& rope_sin,
                                       const Tensor& attn_mask) const {
    auto q = add_bias_if_present(ops::linear(hidden_states, q_proj_weight()), q_proj_bias());
    auto k = add_bias_if_present(ops::linear(hidden_states, k_proj_weight()), k_proj_bias());
    auto v = add_bias_if_present(ops::linear(hidden_states, v_proj_weight()), v_proj_bias());

    auto q_heads = q.reshape({0, 0, num_heads_, head_dim_}).permute({0, 2, 1, 3});
    auto k_heads = k.reshape({0, 0, num_kv_heads_, head_dim_}).permute({0, 2, 1, 3});
    auto v_heads = v.reshape({0, 0, num_kv_heads_, head_dim_}).permute({0, 2, 1, 3});

    auto* policy = &ctx().op_policy();
    auto q_rot = ops::llm::apply_rope(q_heads, rope_cos, rope_sin, head_dim_, policy);
    auto k_rot = ops::llm::apply_rope(k_heads, rope_cos, rope_sin, head_dim_, policy);

    auto k_expanded = ops::llm::repeat_kv(k_rot, num_heads_, num_kv_heads_, head_dim_);
    auto v_expanded = ops::llm::repeat_kv(v_heads, num_heads_, num_kv_heads_, head_dim_);

    auto context = ops::llm::sdpa(q_rot, k_expanded, v_expanded, scaling_, 3, &attn_mask, false, policy);
    const int64_t attn_out_dim = static_cast<int64_t>(num_heads_) * head_dim_;
    auto merged = context.permute({0, 2, 1, 3}).reshape({0, 0, attn_out_dim});
    return add_bias_if_present(ops::linear(merged, o_proj_weight()), o_proj_bias());
}

DeepseekQwen2MLP::DeepseekQwen2MLP(BuilderContext& ctx,
                                   const std::string& name,
                                   const DeepseekQwen2VisionConfig& cfg,
                                   Module* parent)
    : Module(name, ctx, parent) {
    if (!cfg.hidden_act.empty() && cfg.hidden_act != "silu") {
        OPENVINO_THROW("Unsupported DeepseekQwen2 MLP activation: ", cfg.hidden_act);
    }
    gate_proj_param_ = &register_parameter("gate_proj.weight");
    up_proj_param_ = &register_parameter("up_proj.weight");
    down_proj_param_ = &register_parameter("down_proj.weight");
}

const Tensor& DeepseekQwen2MLP::gate_proj_weight() const {
    if (!gate_proj_param_) {
        OPENVINO_THROW("DeepseekQwen2MLP gate projection parameter not registered");
    }
    return gate_proj_param_->value();
}

const Tensor& DeepseekQwen2MLP::up_proj_weight() const {
    if (!up_proj_param_) {
        OPENVINO_THROW("DeepseekQwen2MLP up projection parameter not registered");
    }
    return up_proj_param_->value();
}

const Tensor& DeepseekQwen2MLP::down_proj_weight() const {
    if (!down_proj_param_) {
        OPENVINO_THROW("DeepseekQwen2MLP down projection parameter not registered");
    }
    return down_proj_param_->value();
}

Tensor DeepseekQwen2MLP::forward(const Tensor& hidden_states) const {
    auto gate = ops::linear(hidden_states, gate_proj_weight());
    auto up = ops::linear(hidden_states, up_proj_weight());
    auto act = ops::silu(gate);
    auto prod = act * up;
    return ops::linear(prod, down_proj_weight());
}

DeepseekQwen2DecoderLayer::DeepseekQwen2DecoderLayer(BuilderContext& ctx,
                                                     const std::string& name,
                                                     const DeepseekQwen2VisionConfig& cfg,
                                                     Module* parent)
    : Module(name, ctx, parent),
      self_attn_(ctx, "self_attn", cfg, this),
      mlp_(ctx, "mlp", cfg, this),
      input_layernorm_(ctx, "input_layernorm", cfg.rms_norm_eps, this),
      post_attention_layernorm_(ctx, "post_attention_layernorm", cfg.rms_norm_eps, this) {
    register_module("self_attn", &self_attn_);
    register_module("mlp", &mlp_);
    register_module("input_layernorm", &input_layernorm_);
    register_module("post_attention_layernorm", &post_attention_layernorm_);
}

Tensor DeepseekQwen2DecoderLayer::forward(const Tensor& hidden_states,
                                          const Tensor& rope_cos,
                                          const Tensor& rope_sin,
                                          const Tensor& attn_mask) const {
    auto normed = input_layernorm_.forward(hidden_states);
    auto attn_out = self_attn_.forward(normed, rope_cos, rope_sin, attn_mask);
    auto resid = hidden_states + attn_out;
    auto normed2 = post_attention_layernorm_.forward(resid);
    auto mlp_out = mlp_.forward(normed2);
    return resid + mlp_out;
}

DeepseekQwen2VisionModel::DeepseekQwen2VisionModel(BuilderContext& ctx,
                                                   const DeepseekQwen2VisionConfig& cfg,
                                                   Module* parent)
    : Module("qwen2_model", ctx, parent),
      cfg_(cfg),
      layers_(),
      norm_(ctx, "norm", cfg.rms_norm_eps, this) {
    register_module("norm", &norm_);

    layers_.reserve(static_cast<size_t>(cfg.num_hidden_layers));
    for (int32_t i = 0; i < cfg.num_hidden_layers; ++i) {
        std::string name = std::string("layers[") + std::to_string(i) + "]";
        layers_.emplace_back(ctx, name, cfg, this);
        register_module(name, &layers_.back());
    }
}

Tensor DeepseekQwen2VisionModel::build_positions(const Tensor& token_type_ids) const {
    auto* ctx = token_type_ids.context();
    auto seq_len = Tensor(shape::dim(token_type_ids, 1), ctx).squeeze(0);
    auto positions = ops::range(seq_len, 0, 1, ov::element::i64);
    auto batch = shape::dim(token_type_ids, 0);
    return broadcast_positions(positions, batch, seq_len.output(), ctx);
}

Tensor DeepseekQwen2VisionModel::build_token_type_ids(const Tensor& vision_flat,
                                                      const Tensor& query_embeds) const {
    auto* ctx = vision_flat.context();
    auto batch = shape::dim(vision_flat, 0);
    auto n_query = Tensor(shape::dim(query_embeds, 1), ctx).squeeze(0);
    auto shape_bn = shape::make({batch, as_shape_dim(n_query.output(), ctx)});

    auto zero = Tensor(ops::const_scalar(ctx, static_cast<int64_t>(0)), ctx);
    auto one = Tensor(ops::const_scalar(ctx, static_cast<int64_t>(1)), ctx);
    auto image_type = shape::broadcast_to(zero, shape_bn);
    auto query_type = shape::broadcast_to(one, shape_bn);
    return ops::concat({image_type, query_type}, 1);
}

Tensor DeepseekQwen2VisionModel::slice_query(const Tensor& hidden_states,
                                             const Tensor& start,
                                             const Tensor& end) const {
    return slice_sequence(hidden_states, start, end);
}

Tensor DeepseekQwen2VisionModel::forward(const Tensor& vision_feats, const Tensor& query_embeds) {
    auto* ctx = vision_feats.context();
    auto batch = shape::dim(vision_feats, 0);
    auto channels = shape::dim(vision_feats, 1);
    auto height = shape::dim(vision_feats, 2);
    auto width = shape::dim(vision_feats, 3);
    auto hw = std::make_shared<ov::op::v1::Multiply>(height, width);

    auto vision_hw = vision_feats.permute({0, 2, 3, 1});
    auto flat_shape = shape::make({batch, as_shape_dim(hw, ctx), channels});
    auto vision_flat = vision_hw.reshape(flat_shape, false);

    auto vision_cast = vision_flat.to(query_embeds.dtype());
    auto query_cast = query_embeds.to(vision_cast.dtype());
    auto hidden_states = ops::concat({vision_cast, query_cast}, 1);

    auto token_type_ids = build_token_type_ids(vision_flat, query_embeds);
    auto attn_mask = build_qwen2_attention_mask(token_type_ids);
    auto positions = build_positions(token_type_ids);

    auto* policy = &this->ctx().op_policy();
    auto cos_sin = ops::llm::rope_cos_sin(positions, cfg_.head_dim(), cfg_.rope_theta, policy);

    for (auto& layer : layers_) {
        hidden_states = layer.forward(hidden_states, cos_sin.first, cos_sin.second, attn_mask);
    }

    hidden_states = norm_.forward(hidden_states);

    auto n_query = Tensor(shape::dim(query_embeds, 1), ctx).squeeze(0);
    auto two = Tensor(ops::const_scalar(ctx, static_cast<int64_t>(2)), ctx);
    auto end = Tensor(std::make_shared<ov::op::v1::Multiply>(n_query.output(), two.output()), ctx);
    return slice_query(hidden_states, n_query, end);
}

std::shared_ptr<ov::Model> create_deepseek_qwen2_encoder_model(
    const DeepseekOCR2VisionConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    DeepseekQwen2VisionConfig qcfg;
    qcfg.hidden_size = cfg.qwen2_0_5b.dim;

    BuilderContext ctx;
    DeepseekQwen2VisionModel model(ctx, qcfg);
    model.packed_mapping().rules.push_back({DeepseekOCR2WeightNames::kQwen2Prefix, "qwen2_model.", 0});

    ov::genai::modeling::weights::LoadOptions options;
    options.allow_unmatched = true;
    options.allow_missing = false;
    options.report_missing = true;
    options.report_unmatched = true;
    auto report = ov::genai::modeling::weights::load_model(model, source, finalizer, options);
    (void)report;

    auto vision_feats = ctx.parameter(DeepseekQwen2VisionIO::kVisionFeats,
                                      ov::element::f32,
                                      ov::PartialShape{-1, qcfg.hidden_size, -1, -1});
    auto query_embeds = ctx.parameter(DeepseekQwen2VisionIO::kQueryEmbeds,
                                      ov::element::f32,
                                      ov::PartialShape{-1, -1, qcfg.hidden_size});

    auto output = model.forward(vision_feats, query_embeds);
    auto result = std::make_shared<ov::op::v0::Result>(output.output());
    set_name(result, DeepseekQwen2VisionIO::kQueryFeats);
    return ctx.build_model({result->output(0)});
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
