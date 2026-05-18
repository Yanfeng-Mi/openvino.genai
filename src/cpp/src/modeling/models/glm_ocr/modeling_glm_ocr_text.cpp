// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/glm_ocr/modeling_glm_ocr_text.hpp"

#include <cmath>

#include <openvino/openvino.hpp>
#include <openvino/core/except.hpp>
#include <openvino/op/util/variable.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/llm.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/ops/rope.hpp"
#include "modeling/ops/shape.hpp"
#include "modeling/ops/tensor_ops.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace {

auto set_name = [](auto node, const std::string& name) {
    node->output(0).set_names({name});
    node->set_friendly_name(name);
};

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

// ======================== EmbeddingInjector ========================

GlmOcrEmbeddingInjector::GlmOcrEmbeddingInjector(BuilderContext& ctx, const std::string& name, Module* parent)
    : Module(name, ctx, parent) {}

Tensor GlmOcrEmbeddingInjector::forward(const Tensor& inputs_embeds,
                                        const Tensor& visual_embeds,
                                        const Tensor& visual_pos_mask) const {
    auto mask = visual_pos_mask.unsqueeze(2);
    auto updates = visual_embeds.to(inputs_embeds.dtype());
    return ops::tensor::masked_scatter(inputs_embeds, mask, updates);
}

// ======================== TextAttention ========================

GlmOcrTextAttention::GlmOcrTextAttention(BuilderContext& ctx,
                                         const std::string& name,
                                         const GlmOcrTextConfig& cfg,
                                         Module* parent)
    : Module(name, ctx, parent),
      num_heads_(cfg.num_attention_heads),
      num_kv_heads_(cfg.kv_heads()),
      head_dim_(cfg.resolved_head_dim()),
      hidden_size_(cfg.hidden_size),
      scaling_(1.0f / std::sqrt(static_cast<float>(head_dim_))) {
    if (num_heads_ <= 0 || head_dim_ <= 0) {
        OPENVINO_THROW("Invalid GlmOcrTextAttention head configuration");
    }
    if (num_heads_ % num_kv_heads_ != 0) {
        OPENVINO_THROW("num_attention_heads must be divisible by num_key_value_heads");
    }

    // GLM-OCR text attention: separate Q/K/V/O projections, no bias
    q_proj_param_ = &register_parameter("q_proj.weight");
    k_proj_param_ = &register_parameter("k_proj.weight");
    v_proj_param_ = &register_parameter("v_proj.weight");
    o_proj_param_ = &register_parameter("o_proj.weight");
}

const Tensor& GlmOcrTextAttention::q_proj_weight() const {
    if (!q_proj_param_) OPENVINO_THROW("GlmOcrTextAttention q_proj parameter not registered");
    return q_proj_param_->value();
}

const Tensor& GlmOcrTextAttention::k_proj_weight() const {
    if (!k_proj_param_) OPENVINO_THROW("GlmOcrTextAttention k_proj parameter not registered");
    return k_proj_param_->value();
}

const Tensor& GlmOcrTextAttention::v_proj_weight() const {
    if (!v_proj_param_) OPENVINO_THROW("GlmOcrTextAttention v_proj parameter not registered");
    return v_proj_param_->value();
}

const Tensor& GlmOcrTextAttention::o_proj_weight() const {
    if (!o_proj_param_) OPENVINO_THROW("GlmOcrTextAttention o_proj parameter not registered");
    return o_proj_param_->value();
}

std::pair<Tensor, Tensor> GlmOcrTextAttention::append_kv_cache(const Tensor& keys,
                                                                const Tensor& values,
                                                                const Tensor& beam_idx) const {
    auto* op_ctx = keys.context();
    auto batch = shape::dim(keys, 0);
    auto kv_heads = ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(num_kv_heads_)});
    auto zero_len = ops::const_vec(op_ctx, std::vector<int64_t>{0});
    auto head_dim = ops::const_vec(op_ctx, std::vector<int64_t>{static_cast<int64_t>(head_dim_)});
    auto cache_shape = shape::make({batch, kv_heads, zero_len, head_dim});

    auto zero = Tensor(ops::const_scalar(op_ctx, 0.0f), op_ctx).to(keys.dtype());
    auto k_init = shape::broadcast_to(zero, cache_shape);
    auto v_init = shape::broadcast_to(zero, cache_shape);

    const std::string cache_prefix = full_path().empty() ? name() : full_path();
    const std::string k_name = cache_prefix + ".key_cache";
    const std::string v_name = cache_prefix + ".value_cache";

    ov::op::util::VariableInfo k_info{ov::PartialShape{-1, num_kv_heads_, -1, head_dim_},
                                      keys.dtype(),
                                      k_name};
    auto k_var = std::make_shared<ov::op::util::Variable>(k_info);
    auto k_read = std::make_shared<ov::op::v6::ReadValue>(k_init.output(), k_var);

    ov::op::util::VariableInfo v_info{ov::PartialShape{-1, num_kv_heads_, -1, head_dim_},
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

Tensor GlmOcrTextAttention::forward(const Tensor& hidden_states,
                                    const Tensor& beam_idx,
                                    const Tensor& rope_cos,
                                    const Tensor& rope_sin) const {
    auto q = ops::linear(hidden_states, q_proj_weight());
    auto k = ops::linear(hidden_states, k_proj_weight());
    auto v = ops::linear(hidden_states, v_proj_weight());

    auto q_heads = q.reshape({0, 0, num_heads_, head_dim_}).permute({0, 2, 1, 3});
    auto k_heads = k.reshape({0, 0, num_kv_heads_, head_dim_}).permute({0, 2, 1, 3});
    auto v_heads = v.reshape({0, 0, num_kv_heads_, head_dim_}).permute({0, 2, 1, 3});

    auto* policy = &ctx().op_policy();
    // GLM-OCR text uses interleaved RoPE (pairs adjacent elements 0,1 then 2,3 etc.)
    auto q_rot = ops::llm::apply_rope_interleave(q_heads, rope_cos, rope_sin, head_dim_, policy);
    auto k_rot = ops::llm::apply_rope_interleave(k_heads, rope_cos, rope_sin, head_dim_, policy);

    auto cached = append_kv_cache(k_rot, v_heads, beam_idx);
    auto k_expanded = ops::llm::repeat_kv(cached.first, num_heads_, num_kv_heads_, head_dim_);
    auto v_expanded = ops::llm::repeat_kv(cached.second, num_heads_, num_kv_heads_, head_dim_);

    auto mask = ops::llm::build_kv_causal_mask(q_rot, k_expanded);
    auto context = ops::llm::sdpa(q_rot, k_expanded, v_expanded, scaling_, 3, &mask, false, policy);
    const int64_t attn_out_dim = static_cast<int64_t>(num_heads_) * head_dim_;
    auto merged = context.permute({0, 2, 1, 3}).reshape({0, 0, attn_out_dim});
    auto out = ops::linear(merged, o_proj_weight());
    return out;
}

// ======================== TextMLP (combined gate_up_proj) ========================

GlmOcrTextMLP::GlmOcrTextMLP(BuilderContext& ctx,
                              const std::string& name,
                              const GlmOcrTextConfig& cfg,
                              Module* parent)
    : Module(name, ctx, parent),
      intermediate_size_(cfg.intermediate_size) {
    if (!cfg.hidden_act.empty() && cfg.hidden_act != "silu") {
        OPENVINO_THROW("Unsupported GlmOcrText MLP activation: ", cfg.hidden_act);
    }
    // Combined gate_up_proj: [2*intermediate_size, hidden_size]
    gate_up_proj_param_ = &register_parameter("gate_up_proj.weight");
    down_proj_param_ = &register_parameter("down_proj.weight");
}

const Tensor& GlmOcrTextMLP::gate_up_proj_weight() const {
    if (!gate_up_proj_param_) OPENVINO_THROW("GlmOcrTextMLP gate_up_proj parameter not registered");
    return gate_up_proj_param_->value();
}

const Tensor& GlmOcrTextMLP::down_proj_weight() const {
    if (!down_proj_param_) OPENVINO_THROW("GlmOcrTextMLP down_proj parameter not registered");
    return down_proj_param_->value();
}

Tensor GlmOcrTextMLP::forward(const Tensor& x) const {
    // gate_up = linear(x, gate_up_proj_weight)  -> [B, S, 2*intermediate_size]
    auto gate_up = ops::linear(x, gate_up_proj_weight());
    // Split into gate [0:intermediate_size] and up [intermediate_size:2*intermediate_size]
    auto gate = ops::slice(gate_up, 0, static_cast<int64_t>(intermediate_size_), 1, 2);
    auto up = ops::slice(gate_up, static_cast<int64_t>(intermediate_size_),
                         static_cast<int64_t>(2 * intermediate_size_), 1, 2);
    auto gated = ops::silu(gate) * up;
    return ops::linear(gated, down_proj_weight());
}

// ======================== TextDecoderLayer (4 norms) ========================

GlmOcrTextDecoderLayer::GlmOcrTextDecoderLayer(BuilderContext& ctx,
                                                const std::string& name,
                                                const GlmOcrTextConfig& cfg,
                                                Module* parent)
    : Module(name, ctx, parent),
      self_attn_(ctx, "self_attn", cfg, this),
      mlp_(ctx, "mlp", cfg, this),
      input_layernorm_(ctx, "input_layernorm", cfg.rms_norm_eps, this),
      post_self_attn_layernorm_(ctx, "post_self_attn_layernorm", cfg.rms_norm_eps, this),
      post_attention_layernorm_(ctx, "post_attention_layernorm", cfg.rms_norm_eps, this),
      post_mlp_layernorm_(ctx, "post_mlp_layernorm", cfg.rms_norm_eps, this) {
    register_module("self_attn", &self_attn_);
    register_module("mlp", &mlp_);
    register_module("input_layernorm", &input_layernorm_);
    register_module("post_self_attn_layernorm", &post_self_attn_layernorm_);
    register_module("post_attention_layernorm", &post_attention_layernorm_);
    register_module("post_mlp_layernorm", &post_mlp_layernorm_);
}

Tensor GlmOcrTextDecoderLayer::forward(const Tensor& hidden_states,
                                       const Tensor& beam_idx,
                                       const Tensor& rope_cos,
                                       const Tensor& rope_sin) const {
    // GLM-OCR residual pattern (from PyTorch):
    // residual = hidden_states
    // hidden_states = input_layernorm(hidden_states)
    // hidden_states = self_attn(hidden_states)
    // hidden_states = post_self_attn_layernorm(hidden_states)
    // hidden_states = residual + hidden_states
    //
    // residual = hidden_states
    // hidden_states = post_attention_layernorm(hidden_states)
    // hidden_states = mlp(hidden_states)
    // hidden_states = post_mlp_layernorm(hidden_states)
    // hidden_states = residual + hidden_states

    auto normed = input_layernorm_.forward(hidden_states);
    auto attn_out = self_attn_.forward(normed, beam_idx, rope_cos, rope_sin);
    auto attn_normed = post_self_attn_layernorm_.forward(attn_out);
    auto residual1 = hidden_states + attn_normed;

    auto mlp_normed = post_attention_layernorm_.forward(residual1);
    auto mlp_out = mlp_.forward(mlp_normed);
    auto mlp_post_normed = post_mlp_layernorm_.forward(mlp_out);
    return residual1 + mlp_post_normed;
}

// ======================== TextModel ========================

GlmOcrTextModel::GlmOcrTextModel(BuilderContext& ctx, const GlmOcrTextConfig& cfg, Module* parent)
    : Module(GlmOcrModuleNames::kText, ctx, parent),
      cfg_(cfg),
      embed_tokens_(ctx, "embed_tokens", this),
      embedding_injector_(ctx, "embedding_injector", this),
      layers_(),
      norm_(ctx, "norm", cfg.rms_norm_eps, this),
      head_dim_(cfg.resolved_head_dim()) {
    register_module("embed_tokens", &embed_tokens_);
    register_module("embedding_injector", &embedding_injector_);
    register_module("norm", &norm_);

    if (!cfg_.rope.rope_type.empty() && cfg_.rope.rope_type != "default") {
        OPENVINO_THROW("Unsupported GlmOcr rope_type: ", cfg_.rope.rope_type);
    }

    layers_.reserve(static_cast<size_t>(cfg.num_hidden_layers));
    for (int32_t i = 0; i < cfg.num_hidden_layers; ++i) {
        layers_.emplace_back(ctx, GlmOcrModuleNames::text_layer(i), cfg, this);
        register_module(GlmOcrModuleNames::text_layer(i), &layers_.back());
    }
}

std::pair<Tensor, Tensor> GlmOcrTextModel::build_mrope_cos_sin(const Tensor& position_ids) const {
    auto* ctx = position_ids.context();
    const int32_t half_dim = head_dim_ / 2;
    std::vector<float> inv_freq(static_cast<size_t>(half_dim));
    for (int32_t i = 0; i < half_dim; ++i) {
        float exponent = static_cast<float>(2 * i) / static_cast<float>(head_dim_);
        inv_freq[static_cast<size_t>(i)] = 1.0f / std::pow(cfg_.rope_theta, exponent);
    }

    auto inv_freq_const = ops::const_vec(ctx, inv_freq);
    Tensor inv_freq_tensor(inv_freq_const, ctx);
    auto inv_freq_reshaped =
        inv_freq_tensor.reshape({1, 1, static_cast<int64_t>(half_dim)}, false);

    auto pos_t = ops::slice(position_ids, 0, 1, 1, 0).squeeze(0).to(ov::element::f32);
    auto pos_h = ops::slice(position_ids, 1, 2, 1, 0).squeeze(0).to(ov::element::f32);
    auto pos_w = ops::slice(position_ids, 2, 3, 1, 0).squeeze(0).to(ov::element::f32);

    auto freqs_t = pos_t.unsqueeze(2) * inv_freq_reshaped;
    auto freqs_h = pos_h.unsqueeze(2) * inv_freq_reshaped;
    auto freqs_w = pos_w.unsqueeze(2) * inv_freq_reshaped;

    // GLM-OCR uses chunked mRoPE (contiguous blocks), not interleaved
    auto freqs_all = ops::tensor::stack({freqs_t, freqs_h, freqs_w}, 0);
    auto freqs = ops::rope::mrope_chunked(freqs_all, cfg_.rope.mrope_section);
    return {freqs.cos(), freqs.sin()};
}

Tensor GlmOcrTextModel::forward(const Tensor& input_ids,
                                const Tensor& position_ids,
                                const Tensor& beam_idx,
                                const Tensor* visual_embeds,
                                const Tensor* visual_pos_mask) {
    auto hidden_states = embed_tokens_.forward(input_ids);
    return forward_embeds(hidden_states, position_ids, beam_idx, visual_embeds, visual_pos_mask);
}

Tensor GlmOcrTextModel::forward_embeds(const Tensor& inputs_embeds,
                                       const Tensor& position_ids,
                                       const Tensor& beam_idx,
                                       const Tensor* visual_embeds,
                                       const Tensor* visual_pos_mask) {
    auto cos_sin = build_mrope_cos_sin(position_ids);
    Tensor hidden_states = inputs_embeds;
    if (visual_embeds && visual_pos_mask) {
        hidden_states = embedding_injector_.forward(hidden_states, *visual_embeds, *visual_pos_mask);
    }
    for (size_t layer_idx = 0; layer_idx < layers_.size(); ++layer_idx) {
        hidden_states = layers_[layer_idx].forward(hidden_states, beam_idx, cos_sin.first, cos_sin.second);
    }
    return norm_.forward(hidden_states);
}

VocabEmbedding& GlmOcrTextModel::embed_tokens() {
    return embed_tokens_;
}

RMSNorm& GlmOcrTextModel::norm() {
    return norm_;
}

// ======================== CausalLM ========================

GlmOcrTextForCausalLM::GlmOcrTextForCausalLM(BuilderContext& ctx,
                                              const GlmOcrTextConfig& cfg,
                                              Module* parent)
    : Module("", ctx, parent),
      cfg_(cfg),
      model_(ctx, cfg, this),
      lm_head_(ctx, GlmOcrModuleNames::kLmHead, this) {
    register_module(GlmOcrModuleNames::kText, &model_);
    register_module(GlmOcrModuleNames::kLmHead, &lm_head_);

    if (cfg_.tie_word_embeddings) {
        lm_head_.tie_to(model_.embed_tokens().weight_param());
    }
}

Tensor GlmOcrTextForCausalLM::forward(const Tensor& input_ids,
                                      const Tensor& position_ids,
                                      const Tensor& beam_idx,
                                      const Tensor* visual_embeds,
                                      const Tensor* visual_pos_mask) {
    auto hidden = model_.forward(input_ids, position_ids, beam_idx, visual_embeds, visual_pos_mask);
    return lm_head_.forward(hidden);
}

GlmOcrTextModel& GlmOcrTextForCausalLM::model() {
    return model_;
}

LMHead& GlmOcrTextForCausalLM::lm_head() {
    return lm_head_;
}

std::shared_ptr<ov::Model> create_glm_ocr_text_model(
    const GlmOcrConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer,
    bool use_inputs_embeds,
    bool enable_visual_inputs) {
    BuilderContext ctx;
    GlmOcrTextForCausalLM model(ctx, cfg.text);
    model.packed_mapping().rules.push_back({"model.", "", 0});

    ov::genai::modeling::weights::LoadOptions options;
    options.allow_unmatched = true;
    options.allow_missing = false;
    options.report_missing = true;
    options.report_unmatched = true;
    auto report = ov::genai::modeling::weights::load_model(model, source, finalizer, options);
    (void)report;

    const auto float_type = ov::element::f32;

    auto attention_mask = ctx.parameter(GlmOcrTextIO::kAttentionMask,
                                        ov::element::i64,
                                        ov::PartialShape{-1, -1});
    auto position_ids = ctx.parameter(GlmOcrTextIO::kPositionIds,
                                      ov::element::i64,
                                      ov::PartialShape{3, -1, -1});
    auto beam_idx = ctx.parameter(GlmOcrTextIO::kBeamIdx,
                                  ov::element::i32,
                                  ov::PartialShape{-1});

    (void)attention_mask;

    const Tensor* visual_embeds_ptr = nullptr;
    const Tensor* visual_pos_mask_ptr = nullptr;

    Tensor visual_embeds;
    Tensor visual_pos_mask;
    if (enable_visual_inputs) {
        visual_embeds = ctx.parameter(GlmOcrTextIO::kVisualEmbeds,
                                      float_type,
                                      ov::PartialShape{-1, -1, cfg.text.hidden_size});
        visual_pos_mask = ctx.parameter(GlmOcrTextIO::kVisualPosMask,
                                        ov::element::boolean,
                                        ov::PartialShape{-1, -1});
        visual_embeds_ptr = &visual_embeds;
        visual_pos_mask_ptr = &visual_pos_mask;
    }

    Tensor logits;
    if (use_inputs_embeds) {
        auto inputs_embeds = ctx.parameter(GlmOcrTextIO::kInputIds,
                                           float_type,
                                           ov::PartialShape{-1, -1, cfg.text.hidden_size});
        logits = model.forward(inputs_embeds,
                               position_ids,
                               beam_idx,
                               visual_embeds_ptr,
                               visual_pos_mask_ptr);
    } else {
        auto input_ids = ctx.parameter(GlmOcrTextIO::kInputIds,
                                       ov::element::i64,
                                       ov::PartialShape{-1, -1});
        logits = model.forward(input_ids,
                               position_ids,
                               beam_idx,
                               visual_embeds_ptr,
                               visual_pos_mask_ptr);
    }

    auto result = std::make_shared<ov::op::v0::Result>(logits.output());
    set_name(result, GlmOcrTextIO::kLogits);
    return ctx.build_model({result->output(0)});
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
