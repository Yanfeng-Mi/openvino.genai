// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/smollm3/modeling_smollm3.hpp"

#include <cmath>
#include <openvino/openvino.hpp>

#include <openvino/core/except.hpp>
#include <openvino/op/util/variable.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/llm.hpp"
#include "modeling/ops/kv_cache.hpp"
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

bool resolve_use_rope(const ov::genai::modeling::models::SmolLM3Config& cfg, int32_t layer_idx) {
    if (!cfg.no_rope_layers.empty()) {
        if (layer_idx < 0 || static_cast<size_t>(layer_idx) >= cfg.no_rope_layers.size()) {
            OPENVINO_THROW("SmolLM3Config.no_rope_layers size mismatch");
        }
        return cfg.no_rope_layers[static_cast<size_t>(layer_idx)] != 0;
    }
    if (cfg.no_rope_layer_interval <= 0) {
        return true;
    }
    return ((layer_idx + 1) % cfg.no_rope_layer_interval) != 0;
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

SmolLM3Attention::SmolLM3Attention(BuilderContext& ctx,
                                   const std::string& name,
                                   const SmolLM3Config& cfg,
                                   int32_t layer_idx,
                                   Module* parent)
    : Module(name, ctx, parent),
      num_heads_(cfg.num_attention_heads),
      num_kv_heads_(cfg.num_key_value_heads > 0 ? cfg.num_key_value_heads : cfg.num_attention_heads),
      head_dim_(cfg.head_dim > 0 ? cfg.head_dim : (cfg.hidden_size / cfg.num_attention_heads)),
      hidden_size_(cfg.hidden_size),
      scaling_(1.0f / std::sqrt(static_cast<float>(head_dim_))),
      rope_theta_(cfg.rope_theta),
      use_rope_(resolve_use_rope(cfg, layer_idx)) {
    if (num_heads_ <= 0 || head_dim_ <= 0) {
        OPENVINO_THROW("Invalid attention head configuration");
    }
    if (num_heads_ % num_kv_heads_ != 0) {
        OPENVINO_THROW("num_attention_heads must be divisible by num_key_value_heads");
    }

    q_proj_param_ = &register_parameter("q_proj.weight");
    k_proj_param_ = &register_parameter("k_proj.weight");
    v_proj_param_ = &register_parameter("v_proj.weight");
    o_proj_param_ = &register_parameter("o_proj.weight");

    if (cfg.attention_bias) {
        q_bias_param_ = &register_parameter("q_proj.bias");
        k_bias_param_ = &register_parameter("k_proj.bias");
        v_bias_param_ = &register_parameter("v_proj.bias");
        o_bias_param_ = &register_parameter("o_proj.bias");
    }
}

const Tensor& SmolLM3Attention::q_proj_weight() const {
    if (!q_proj_param_) {
        OPENVINO_THROW("SmolLM3Attention q_proj parameter not registered");
    }
    return q_proj_param_->value();
}

const Tensor& SmolLM3Attention::k_proj_weight() const {
    if (!k_proj_param_) {
        OPENVINO_THROW("SmolLM3Attention k_proj parameter not registered");
    }
    return k_proj_param_->value();
}

const Tensor& SmolLM3Attention::v_proj_weight() const {
    if (!v_proj_param_) {
        OPENVINO_THROW("SmolLM3Attention v_proj parameter not registered");
    }
    return v_proj_param_->value();
}

const Tensor& SmolLM3Attention::o_proj_weight() const {
    if (!o_proj_param_) {
        OPENVINO_THROW("SmolLM3Attention o_proj parameter not registered");
    }
    return o_proj_param_->value();
}

const Tensor* SmolLM3Attention::q_proj_bias() const {
    return (q_bias_param_ && q_bias_param_->is_bound()) ? &q_bias_param_->value() : nullptr;
}

const Tensor* SmolLM3Attention::k_proj_bias() const {
    return (k_bias_param_ && k_bias_param_->is_bound()) ? &k_bias_param_->value() : nullptr;
}

const Tensor* SmolLM3Attention::v_proj_bias() const {
    return (v_bias_param_ && v_bias_param_->is_bound()) ? &v_bias_param_->value() : nullptr;
}

const Tensor* SmolLM3Attention::o_proj_bias() const {
    return (o_bias_param_ && o_bias_param_->is_bound()) ? &o_bias_param_->value() : nullptr;
}

Tensor SmolLM3Attention::forward(const Tensor& hidden_states,
                                 const Tensor& beam_idx,
                                 const Tensor& rope_cos,
                                 const Tensor& rope_sin) const {
    auto q = add_bias_if_present(ops::linear(hidden_states, q_proj_weight()), q_proj_bias());
    auto k = add_bias_if_present(ops::linear(hidden_states, k_proj_weight()), k_proj_bias());
    auto v = add_bias_if_present(ops::linear(hidden_states, v_proj_weight()), v_proj_bias());

    auto q_heads = q.reshape({0, 0, num_heads_, head_dim_}).permute({0, 2, 1, 3});
    auto k_heads = k.reshape({0, 0, num_kv_heads_, head_dim_}).permute({0, 2, 1, 3});
    auto v_heads = v.reshape({0, 0, num_kv_heads_, head_dim_}).permute({0, 2, 1, 3});

    Tensor q_rot = q_heads;
    Tensor k_rot = k_heads;
    if (use_rope_) {
        auto* policy = &ctx().op_policy();
        q_rot = ops::llm::apply_rope(q_heads, rope_cos, rope_sin, head_dim_, policy);
        k_rot = ops::llm::apply_rope(k_heads, rope_cos, rope_sin, head_dim_, policy);
    }

    const std::string cache_prefix = full_path().empty() ? name() : full_path();
    auto cached = ops::append_kv_cache(k_rot, v_heads, beam_idx, num_kv_heads_, head_dim_, cache_prefix, ctx());
    auto k_expanded = ops::llm::repeat_kv(cached.first, num_heads_, num_kv_heads_, head_dim_);
    auto v_expanded = ops::llm::repeat_kv(cached.second, num_heads_, num_kv_heads_, head_dim_);

    auto mask = ops::llm::build_kv_causal_mask(q_rot, k_expanded);
    auto* policy = &ctx().op_policy();
    auto context = ops::llm::sdpa(q_rot, k_expanded, v_expanded, scaling_, 3, &mask, false, policy);
    const int64_t attn_out_dim = static_cast<int64_t>(num_heads_) * head_dim_;
    auto merged = context.permute({0, 2, 1, 3}).reshape({0, 0, attn_out_dim});
    auto out = add_bias_if_present(ops::linear(merged, o_proj_weight()), o_proj_bias());
    return out;
}

SmolLM3MLP::SmolLM3MLP(BuilderContext& ctx, const std::string& name, const SmolLM3Config& cfg, Module* parent)
    : Module(name, ctx, parent) {
    if (!cfg.hidden_act.empty() && cfg.hidden_act != "silu") {
        OPENVINO_THROW("Unsupported SmolLM3 MLP activation: ", cfg.hidden_act);
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

const Tensor& SmolLM3MLP::gate_proj_weight() const {
    if (!gate_proj_param_) {
        OPENVINO_THROW("SmolLM3MLP gate projection parameter not registered");
    }
    return gate_proj_param_->value();
}

const Tensor& SmolLM3MLP::up_proj_weight() const {
    if (!up_proj_param_) {
        OPENVINO_THROW("SmolLM3MLP up projection parameter not registered");
    }
    return up_proj_param_->value();
}

const Tensor& SmolLM3MLP::down_proj_weight() const {
    if (!down_proj_param_) {
        OPENVINO_THROW("SmolLM3MLP down projection parameter not registered");
    }
    return down_proj_param_->value();
}

const Tensor* SmolLM3MLP::gate_proj_bias() const {
    return (gate_bias_param_ && gate_bias_param_->is_bound()) ? &gate_bias_param_->value() : nullptr;
}

const Tensor* SmolLM3MLP::up_proj_bias() const {
    return (up_bias_param_ && up_bias_param_->is_bound()) ? &up_bias_param_->value() : nullptr;
}

const Tensor* SmolLM3MLP::down_proj_bias() const {
    return (down_bias_param_ && down_bias_param_->is_bound()) ? &down_bias_param_->value() : nullptr;
}

Tensor SmolLM3MLP::forward(const Tensor& x) const {
    auto gate = add_bias_if_present(ops::linear(x, gate_proj_weight()), gate_proj_bias());
    auto up = add_bias_if_present(ops::linear(x, up_proj_weight()), up_proj_bias());
    auto gated = ops::silu(gate) * up;
    return add_bias_if_present(ops::linear(gated, down_proj_weight()), down_proj_bias());
}

SmolLM3DecoderLayer::SmolLM3DecoderLayer(BuilderContext& ctx,
                                         const std::string& name,
                                         const SmolLM3Config& cfg,
                                         int32_t layer_idx,
                                         Module* parent)
    : Module(name, ctx, parent),
      self_attn_(ctx, "self_attn", cfg, layer_idx, this),
      mlp_(ctx, "mlp", cfg, this),
      input_layernorm_(ctx, "input_layernorm", cfg.rms_norm_eps, this),
      post_attention_layernorm_(ctx, "post_attention_layernorm", cfg.rms_norm_eps, this) {}

std::pair<Tensor, Tensor> SmolLM3DecoderLayer::forward(const Tensor& hidden_states,
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

SmolLM3Model::SmolLM3Model(BuilderContext& ctx, const SmolLM3Config& cfg, Module* parent)
    : Module("model", ctx, parent),
      embed_tokens_(ctx, "embed_tokens", this),
      layers_(),
      norm_(ctx, "norm", cfg.rms_norm_eps, this),
      head_dim_(cfg.head_dim > 0
                    ? cfg.head_dim
                    : (cfg.num_attention_heads > 0 ? (cfg.hidden_size / cfg.num_attention_heads) : 0)),
      rope_theta_(cfg.rope_theta) {
    if (!cfg.no_rope_layers.empty() && cfg.no_rope_layers.size() != static_cast<size_t>(cfg.num_hidden_layers)) {
        OPENVINO_THROW("SmolLM3Config.no_rope_layers size mismatch");
    }
    layers_.reserve(static_cast<size_t>(cfg.num_hidden_layers));
    for (int32_t i = 0; i < cfg.num_hidden_layers; ++i) {
        layers_.emplace_back(ctx, "layers[" + std::to_string(i) + "]", cfg, i, this);
    }
}

Tensor SmolLM3Model::forward(const Tensor& input_ids, const Tensor& position_ids, const Tensor& beam_idx) {
    auto hidden_states = embed_tokens_.forward(input_ids);
    auto* policy = &ctx().op_policy();
    auto cos_sin = ops::llm::rope_cos_sin(position_ids, head_dim_, rope_theta_, policy);
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

VocabEmbedding& SmolLM3Model::embed_tokens() {
    return embed_tokens_;
}

RMSNorm& SmolLM3Model::norm() {
    return norm_;
}

SmolLM3ForCausalLM::SmolLM3ForCausalLM(BuilderContext& ctx, const SmolLM3Config& cfg, Module* parent)
    : Module("", ctx, parent),
      cfg_(cfg),
      model_(ctx, cfg, this),
      lm_head_(ctx, "lm_head", this) {
    if (cfg_.tie_word_embeddings) {
        lm_head_.tie_to(model_.embed_tokens().weight_param());
    }
}

Tensor SmolLM3ForCausalLM::forward(const Tensor& input_ids,
                                   const Tensor& position_ids,
                                   const Tensor& beam_idx) {
    auto hidden = model_.forward(input_ids, position_ids, beam_idx);
    return lm_head_.forward(hidden);
}

SmolLM3Model& SmolLM3ForCausalLM::model() {
    return model_;
}

LMHead& SmolLM3ForCausalLM::lm_head() {
    return lm_head_;
}

std::shared_ptr<ov::Model> create_smollm3_model(
    const SmolLM3Config& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    SmolLM3ForCausalLM model(ctx, cfg);

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

// ============================================================================
// Model Builder Registration
// ============================================================================
// The build function and registration are placed here (in the model file)
// rather than in model_builder.cpp, following the pattern from vLLM where
// each model file is self-contained with its own registration.
// This allows adding new models without modifying model_builder.cpp.
// ============================================================================

#include "loaders/model_builder.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace {

/**
 * @brief Build SmolLM3 model using modeling API
 *
 * This function converts ModelConfig to SmolLM3Config, creates the model
 * structure, loads weights, and builds the final OpenVINO model.
 */
std::shared_ptr<ov::Model> build_smollm3_model(
    const ov::genai::loaders::ModelConfig& config,
    ov::genai::modeling::weights::WeightSource& weight_source,
    ov::genai::modeling::weights::WeightFinalizer& weight_finalizer) {

    using namespace ov::genai::modeling;
    using namespace ov::genai::modeling::models;

    BuilderContext ctx;

    // Convert ModelConfig to SmolLM3Config
    SmolLM3Config cfg;
    cfg.architecture = config.architecture;
    cfg.hidden_size = config.hidden_size;
    cfg.num_hidden_layers = config.num_hidden_layers;
    cfg.num_attention_heads = config.num_attention_heads;
    cfg.num_key_value_heads = config.num_key_value_heads > 0
        ? config.num_key_value_heads : config.num_attention_heads;
    cfg.head_dim = config.head_dim > 0
        ? config.head_dim : (config.hidden_size / config.num_attention_heads);
    cfg.intermediate_size = config.intermediate_size;
    cfg.rope_theta = config.rope_theta;
    cfg.attention_bias = config.attention_bias;
    cfg.mlp_bias = config.mlp_bias;
    cfg.rms_norm_eps = config.rms_norm_eps;
    cfg.tie_word_embeddings = config.tie_word_embeddings;
    cfg.hidden_act = config.hidden_act;
    
    // SmolLM3-specific: no_rope_layer_interval
    cfg.no_rope_layer_interval = config.no_rope_layer_interval;
    cfg.no_rope_layers = config.no_rope_layers;

    // Create model
    SmolLM3ForCausalLM model(ctx, cfg);

    // Load weights
    weights::load_model(model, weight_source, weight_finalizer);

    // Helper to set tensor name
    auto set_name = [](auto node, const std::string& name) {
        node->output(0).set_names({name});
    };

    // Create input parameters
    auto input_ids = ctx.parameter("input_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto attention_mask = ctx.parameter("attention_mask", ov::element::i64, ov::PartialShape{-1, -1});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto beam_idx = ctx.parameter("beam_idx", ov::element::i32, ov::PartialShape{-1});

    // Set tensor names for inputs
    input_ids.output().get_node()->output(0).set_names({"input_ids"});
    attention_mask.output().get_node()->output(0).set_names({"attention_mask"});
    position_ids.output().get_node()->output(0).set_names({"position_ids"});
    beam_idx.output().get_node()->output(0).set_names({"beam_idx"});

    (void)attention_mask;  // Attention mask is handled internally by SDPA

    // Forward pass
    auto logits = model.forward(input_ids, position_ids, beam_idx);

    // Build model with named output
    auto result = std::make_shared<ov::op::v0::Result>(logits.output());
    set_name(result, "logits");
    auto ov_model = ctx.build_model({result->output(0)});

    // Set runtime options for optimal performance
    ov_model->set_rt_info(ov::element::f16, {"runtime_options", ov::hint::kv_cache_precision.name()});
    ov_model->set_rt_info(8.0f, {"runtime_options", ov::hint::activations_scale_factor.name()});

    return ov_model;
}

// Self-registration: Register SmolLM3 builder at static initialization
static bool smollm3_registered = []() {
    ov::genai::loaders::ModelBuilder::instance().register_architecture("smollm3", build_smollm3_model);
    return true;
}();

}  // namespace
