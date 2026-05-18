// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/youtu/modeling_youtu.hpp"

#include <cmath>

#include <openvino/core/except.hpp>
#include <openvino/openvino.hpp>
#include <openvino/op/util/variable.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/llm.hpp"
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

auto set_name = [](auto node, const std::string& name) {
    node->output(0).set_names({name});
    node->set_friendly_name(name);
};

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

YoutuMLAttention::YoutuMLAttention(BuilderContext& ctx,
                                   const std::string& name,
                                   const YoutuConfig& cfg,
                                   Module* parent)
    : Module(name, ctx, parent),
      num_heads_(cfg.num_attention_heads),
      q_lora_rank_(cfg.q_lora_rank),
      kv_lora_rank_(cfg.kv_lora_rank),
      qk_rope_head_dim_(cfg.qk_rope_head_dim),
      qk_nope_head_dim_(cfg.qk_nope_head_dim),
      qk_head_dim_(cfg.qk_head_dim > 0 ? cfg.qk_head_dim : (cfg.qk_rope_head_dim + cfg.qk_nope_head_dim)),
      v_head_dim_(cfg.v_head_dim > 0 ? cfg.v_head_dim : qk_head_dim_),
      scaling_(1.0f / std::sqrt(static_cast<float>(qk_head_dim_))),
      rope_interleave_(cfg.rope_interleave),
      q_a_layernorm_(ctx, "q_a_layernorm", cfg.rms_norm_eps, this),
      kv_a_layernorm_(ctx, "kv_a_layernorm", cfg.rms_norm_eps, this) {
    if (num_heads_ <= 0) {
        OPENVINO_THROW("Invalid attention head configuration");
    }
    if (cfg.num_key_value_heads > 0 && cfg.num_key_value_heads != num_heads_) {
        OPENVINO_THROW("Youtu MLA expects num_key_value_heads == num_attention_heads");
    }
    if (q_lora_rank_ <= 0 || kv_lora_rank_ <= 0) {
        OPENVINO_THROW("Youtu MLA requires q_lora_rank and kv_lora_rank > 0");
    }
    if (qk_head_dim_ <= 0 || v_head_dim_ <= 0) {
        OPENVINO_THROW("Invalid head dimensions for Youtu MLA");
    }
    if (qk_head_dim_ != (qk_nope_head_dim_ + qk_rope_head_dim_)) {
        OPENVINO_THROW("qk_head_dim must equal qk_nope_head_dim + qk_rope_head_dim");
    }
    if (qk_rope_head_dim_ <= 0 || (qk_rope_head_dim_ % 2) != 0) {
        OPENVINO_THROW("qk_rope_head_dim must be positive and even");
    }

    q_a_proj_param_ = &register_parameter("q_a_proj.weight");
    q_b_proj_param_ = &register_parameter("q_b_proj.weight");
    kv_a_proj_param_ = &register_parameter("kv_a_proj_with_mqa.weight");
    kv_b_proj_param_ = &register_parameter("kv_b_proj.weight");
    o_proj_param_ = &register_parameter("o_proj.weight");

    if (cfg.attention_bias) {
        q_a_bias_param_ = &register_parameter("q_a_proj.bias");
        kv_a_bias_param_ = &register_parameter("kv_a_proj_with_mqa.bias");
        o_bias_param_ = &register_parameter("o_proj.bias");
    }
}

const Tensor& YoutuMLAttention::q_a_proj_weight() const {
    if (!q_a_proj_param_) {
        OPENVINO_THROW("YoutuMLAttention q_a_proj parameter not registered");
    }
    return q_a_proj_param_->value();
}

const Tensor& YoutuMLAttention::q_b_proj_weight() const {
    if (!q_b_proj_param_) {
        OPENVINO_THROW("YoutuMLAttention q_b_proj parameter not registered");
    }
    return q_b_proj_param_->value();
}

const Tensor& YoutuMLAttention::kv_a_proj_weight() const {
    if (!kv_a_proj_param_) {
        OPENVINO_THROW("YoutuMLAttention kv_a_proj parameter not registered");
    }
    return kv_a_proj_param_->value();
}

const Tensor& YoutuMLAttention::kv_b_proj_weight() const {
    if (!kv_b_proj_param_) {
        OPENVINO_THROW("YoutuMLAttention kv_b_proj parameter not registered");
    }
    return kv_b_proj_param_->value();
}

const Tensor& YoutuMLAttention::o_proj_weight() const {
    if (!o_proj_param_) {
        OPENVINO_THROW("YoutuMLAttention o_proj parameter not registered");
    }
    return o_proj_param_->value();
}

const Tensor* YoutuMLAttention::q_a_proj_bias() const {
    return (q_a_bias_param_ && q_a_bias_param_->is_bound()) ? &q_a_bias_param_->value() : nullptr;
}

const Tensor* YoutuMLAttention::kv_a_proj_bias() const {
    return (kv_a_bias_param_ && kv_a_bias_param_->is_bound()) ? &kv_a_bias_param_->value() : nullptr;
}

const Tensor* YoutuMLAttention::o_proj_bias() const {
    return (o_bias_param_ && o_bias_param_->is_bound()) ? &o_bias_param_->value() : nullptr;
}

std::pair<Tensor, Tensor> YoutuMLAttention::append_kv_cache(const Tensor& keys,
                                                            const Tensor& values,
                                                            const Tensor& beam_idx) const {
    auto* op_ctx = keys.context();
    auto batch = shape::dim(keys, 0);
    auto heads = ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(num_heads_)});
    auto zero_len = ops::const_vec(op_ctx, std::vector<int64_t>{0});
    auto key_dim = ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(qk_head_dim_)});
    auto val_dim = ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(v_head_dim_)});

    auto k_shape = shape::make({batch, heads, zero_len, key_dim});
    auto v_shape = shape::make({batch, heads, zero_len, val_dim});

    auto zero = Tensor(ops::const_scalar(op_ctx, 0.0f), op_ctx).to(keys.dtype());
    auto k_init = shape::broadcast_to(zero, k_shape);
    auto v_init = shape::broadcast_to(zero, v_shape);

    const std::string cache_prefix = full_path().empty() ? name() : full_path();
    const std::string k_name = cache_prefix + ".key_cache";
    const std::string v_name = cache_prefix + ".value_cache";

    ov::op::util::VariableInfo k_info{ov::PartialShape{-1, num_heads_, -1, qk_head_dim_},
                                      keys.dtype(),
                                      k_name};
    auto k_var = std::make_shared<ov::op::util::Variable>(k_info);
    auto k_read = std::make_shared<ov::op::v6::ReadValue>(k_init.output(), k_var);

    ov::op::util::VariableInfo v_info{ov::PartialShape{-1, num_heads_, -1, v_head_dim_},
                                      values.dtype(),
                                      v_name};
    auto v_var = std::make_shared<ov::op::util::Variable>(v_info);
    auto v_read = std::make_shared<ov::op::v6::ReadValue>(v_init.output(), v_var);

    auto k_cached = ops::gather(Tensor(k_read->output(0), op_ctx), beam_idx, 0);
    auto v_cached = ops::gather(Tensor(v_read->output(0), op_ctx), beam_idx, 0);

    auto k_combined = ops::concat({k_cached, keys}, 2);
    auto v_combined = ops::concat({v_cached, values}, 2);

    auto k_assign = std::make_shared<ov::opset13::Assign>(k_combined.output(), k_var);
    auto v_assign = std::make_shared<ov::opset13::Assign>(v_combined.output(), v_var);
    ctx().register_sink(k_assign);
    ctx().register_sink(v_assign);

    return {k_combined, v_combined};
}

Tensor YoutuMLAttention::forward(const Tensor& hidden_states,
                                 const Tensor& beam_idx,
                                 const Tensor& rope_cos,
                                 const Tensor& rope_sin) const {
    auto q_latent = add_bias_if_present(ops::linear(hidden_states, q_a_proj_weight()), q_a_proj_bias());
    auto q_norm = q_a_layernorm_.forward(q_latent);
    auto q_states = ops::linear(q_norm, q_b_proj_weight());

    auto q_heads = q_states.reshape({0, 0, num_heads_, qk_head_dim_}).permute({0, 2, 1, 3});
    auto q_pass = ops::slice(q_heads, 0, qk_nope_head_dim_, 1, 3);
    auto q_rot = ops::slice(q_heads, qk_nope_head_dim_, qk_head_dim_, 1, 3);

    auto kv_latent = add_bias_if_present(ops::linear(hidden_states, kv_a_proj_weight()), kv_a_proj_bias());
    auto k_pass_latent = ops::slice(kv_latent, 0, kv_lora_rank_, 1, 2);
    auto k_rot = ops::slice(kv_latent, kv_lora_rank_, kv_lora_rank_ + qk_rope_head_dim_, 1, 2);

    auto k_norm = kv_a_layernorm_.forward(k_pass_latent);
    auto kv_states = ops::linear(k_norm, kv_b_proj_weight());
    auto kv_heads = kv_states
                        .reshape({0, 0, num_heads_, qk_nope_head_dim_ + v_head_dim_})
                        .permute({0, 2, 1, 3});
    auto k_pass = ops::slice(kv_heads, 0, qk_nope_head_dim_, 1, 3);
    auto v_heads = ops::slice(kv_heads, qk_nope_head_dim_, qk_nope_head_dim_ + v_head_dim_, 1, 3);

    auto* op_ctx = hidden_states.context();
    auto batch = shape::dim(k_rot, 0);
    auto seq = shape::dim(k_rot, 1);
    auto one = ops::const_vec(op_ctx, std::vector<int64_t>{1});
    auto rope_dim = ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(qk_rope_head_dim_)});
    auto k_rot_shape = shape::make({batch, one, seq, rope_dim});
    auto k_rot_reshaped = k_rot.reshape(k_rot_shape, false);

    auto* policy = &ctx().op_policy();
    if (rope_interleave_) {
        q_rot = ops::llm::apply_rope_interleave(q_rot, rope_cos, rope_sin, qk_rope_head_dim_, policy);
        k_rot_reshaped = ops::llm::apply_rope_interleave(k_rot_reshaped, rope_cos, rope_sin, qk_rope_head_dim_, policy);
    } else {
        q_rot = ops::llm::apply_rope(q_rot, rope_cos, rope_sin, qk_rope_head_dim_, policy);
        k_rot_reshaped = ops::llm::apply_rope(k_rot_reshaped, rope_cos, rope_sin, qk_rope_head_dim_, policy);
    }

    auto heads = ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(num_heads_)});
    auto target_shape = shape::make({batch, heads, seq, rope_dim});
    auto k_rot_expanded = shape::broadcast_to(k_rot_reshaped, target_shape);

    auto query_states = ops::concat({q_pass, q_rot}, 3);
    auto key_states = ops::concat({k_pass, k_rot_expanded}, 3);

    auto cached = append_kv_cache(key_states, v_heads, beam_idx);
    auto v_for_sdpa = cached.second;
    if (qk_head_dim_ != v_head_dim_) {
        v_for_sdpa = ops::llm::pad_to_head_dim(v_for_sdpa, v_head_dim_, qk_head_dim_);
    }

    auto mask = ops::llm::build_kv_causal_mask(query_states, cached.first);
    auto context = ops::llm::sdpa(query_states, cached.first, v_for_sdpa, scaling_, 3, &mask, false, policy);

    if (qk_head_dim_ != v_head_dim_) {
        context = ops::llm::slice_to_head_dim(context, qk_head_dim_, v_head_dim_);
    }

    const int64_t attn_out_dim = static_cast<int64_t>(num_heads_) * v_head_dim_;
    auto merged = context.permute({0, 2, 1, 3}).reshape({0, 0, attn_out_dim});
    auto out = add_bias_if_present(ops::linear(merged, o_proj_weight()), o_proj_bias());
    return out;
}

YoutuMLP::YoutuMLP(BuilderContext& ctx, const std::string& name, const YoutuConfig& cfg, Module* parent)
    : Module(name, ctx, parent) {
    if (!cfg.hidden_act.empty() && cfg.hidden_act != "silu") {
        OPENVINO_THROW("Unsupported Youtu MLP activation: ", cfg.hidden_act);
    }
    gate_proj_param_ = &register_parameter("gate_proj.weight");
    up_proj_param_ = &register_parameter("up_proj.weight");
    down_proj_param_ = &register_parameter("down_proj.weight");

    if (cfg.mlp_bias) {
        gate_bias_param_ = &register_parameter("gate_proj.bias");
        up_bias_param_ = &register_parameter("up_proj.bias");
        down_bias_param_ = &register_parameter("down_proj.bias");
    }
}

const Tensor& YoutuMLP::gate_proj_weight() const {
    if (!gate_proj_param_) {
        OPENVINO_THROW("YoutuMLP gate projection parameter not registered");
    }
    return gate_proj_param_->value();
}

const Tensor& YoutuMLP::up_proj_weight() const {
    if (!up_proj_param_) {
        OPENVINO_THROW("YoutuMLP up projection parameter not registered");
    }
    return up_proj_param_->value();
}

const Tensor& YoutuMLP::down_proj_weight() const {
    if (!down_proj_param_) {
        OPENVINO_THROW("YoutuMLP down projection parameter not registered");
    }
    return down_proj_param_->value();
}

const Tensor* YoutuMLP::gate_proj_bias() const {
    return (gate_bias_param_ && gate_bias_param_->is_bound()) ? &gate_bias_param_->value() : nullptr;
}

const Tensor* YoutuMLP::up_proj_bias() const {
    return (up_bias_param_ && up_bias_param_->is_bound()) ? &up_bias_param_->value() : nullptr;
}

const Tensor* YoutuMLP::down_proj_bias() const {
    return (down_bias_param_ && down_bias_param_->is_bound()) ? &down_bias_param_->value() : nullptr;
}

Tensor YoutuMLP::forward(const Tensor& x) const {
    auto gate = add_bias_if_present(ops::linear(x, gate_proj_weight()), gate_proj_bias());
    auto up = add_bias_if_present(ops::linear(x, up_proj_weight()), up_proj_bias());
    auto gated = ops::silu(gate) * up;
    return add_bias_if_present(ops::linear(gated, down_proj_weight()), down_proj_bias());
}

YoutuDecoderLayer::YoutuDecoderLayer(BuilderContext& ctx,
                                     const std::string& name,
                                     const YoutuConfig& cfg,
                                     Module* parent)
    : Module(name, ctx, parent),
      self_attn_(ctx, "self_attn", cfg, this),
      mlp_(ctx, "mlp", cfg, this),
      input_layernorm_(ctx, "input_layernorm", cfg.rms_norm_eps, this),
      post_attention_layernorm_(ctx, "post_attention_layernorm", cfg.rms_norm_eps, this) {}

std::pair<Tensor, Tensor> YoutuDecoderLayer::forward(const Tensor& hidden_states,
                                                     const Tensor& beam_idx,
                                                     const Tensor& rope_cos,
                                                     const Tensor& rope_sin,
                                                     const std::optional<Tensor>& residual) const {
    Tensor normed;
    Tensor next_residual;
    if (residual) {
        auto norm_out = input_layernorm_.forward(hidden_states, *residual);
        normed = norm_out.first;
        next_residual = norm_out.second;
    } else {
        normed = input_layernorm_.forward(hidden_states);
        next_residual = hidden_states;
    }
    auto attn_out = self_attn_.forward(normed, beam_idx, rope_cos, rope_sin);
    auto post_norm = post_attention_layernorm_.forward(attn_out, next_residual);
    auto mlp_out = mlp_.forward(post_norm.first);
    return {mlp_out, post_norm.second};
}

YoutuModel::YoutuModel(BuilderContext& ctx, const YoutuConfig& cfg, Module* parent)
    : Module("model", ctx, parent),
      embed_tokens_(ctx, "embed_tokens", this),
      layers_(),
      norm_(ctx, "norm", cfg.rms_norm_eps, this),
      rope_head_dim_(cfg.qk_rope_head_dim),
      rope_theta_(cfg.rope_theta) {
    if (rope_head_dim_ <= 0) {
        OPENVINO_THROW("Invalid Youtu rope head dimension");
    }
    layers_.reserve(static_cast<size_t>(cfg.num_hidden_layers));
    for (int32_t i = 0; i < cfg.num_hidden_layers; ++i) {
        layers_.emplace_back(ctx, "layers[" + std::to_string(i) + "]", cfg, this);
    }
}

Tensor YoutuModel::forward(const Tensor& input_ids, const Tensor& position_ids, const Tensor& beam_idx) {
    auto hidden_states = embed_tokens_.forward(input_ids);
    auto* policy = &ctx().op_policy();
    auto cos_sin = ops::llm::rope_cos_sin(position_ids, rope_head_dim_, rope_theta_, policy);
    std::optional<Tensor> residual;
    for (auto& layer : layers_) {
        auto layer_out = layer.forward(hidden_states, beam_idx, cos_sin.first, cos_sin.second, residual);
        hidden_states = layer_out.first;
        residual = layer_out.second;
    }
    if (residual) {
        return norm_.forward(hidden_states, *residual).first;
    }
    return norm_.forward(hidden_states);
}

VocabEmbedding& YoutuModel::embed_tokens() {
    return embed_tokens_;
}

RMSNorm& YoutuModel::norm() {
    return norm_;
}

YoutuForCausalLM::YoutuForCausalLM(BuilderContext& ctx, const YoutuConfig& cfg, Module* parent)
    : Module("", ctx, parent),
      cfg_(cfg),
      model_(ctx, cfg, this),
      lm_head_(ctx, "lm_head", this) {
    if (cfg_.tie_word_embeddings) {
        lm_head_.tie_to(model_.embed_tokens().weight_param());
    }
}

Tensor YoutuForCausalLM::forward(const Tensor& input_ids,
                                 const Tensor& position_ids,
                                 const Tensor& beam_idx) {
    auto hidden = model_.forward(input_ids, position_ids, beam_idx);
    return lm_head_.forward(hidden);
}

YoutuModel& YoutuForCausalLM::model() {
    return model_;
}

LMHead& YoutuForCausalLM::lm_head() {
    return lm_head_;
}

std::shared_ptr<ov::Model> create_youtu_llm_model(
    const YoutuConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    YoutuForCausalLM model(ctx, cfg);

    ov::genai::modeling::weights::load_model(model, source, finalizer);

    auto input_ids = ctx.parameter("input_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto attention_mask = ctx.parameter("attention_mask", ov::element::i64, ov::PartialShape{-1, -1});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto beam_idx = ctx.parameter("beam_idx", ov::element::i32, ov::PartialShape{-1});

    (void)attention_mask;
    auto logits = model.forward(input_ids, position_ids, beam_idx);

    auto result = std::make_shared<ov::op::v0::Result>(logits.output());
    set_name(result, "logits");
    return ctx.build_model({result->output(0)});
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
