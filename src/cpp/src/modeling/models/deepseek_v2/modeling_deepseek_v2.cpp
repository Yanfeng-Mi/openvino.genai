// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/deepseek_v2/modeling_deepseek_v2.hpp"

#include <cmath>

#include <openvino/openvino.hpp>
#include <openvino/core/except.hpp>
#include <openvino/op/util/variable.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/kv_cache.hpp"
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

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

int32_t DeepseekV2TextConfig::resolved_kv_heads() const {
    return num_key_value_heads > 0 ? num_key_value_heads : num_attention_heads;
}

int32_t DeepseekV2TextConfig::resolved_head_dim() const {
    if (head_dim > 0) {
        return head_dim;
    }
    if (num_attention_heads <= 0) {
        return 0;
    }
    return hidden_size / num_attention_heads;
}

bool DeepseekV2TextConfig::is_moe_layer(int32_t layer_idx) const {
    if (n_routed_experts <= 0 || num_experts_per_tok <= 0) {
        return false;
    }
    const int32_t freq = moe_layer_freq > 0 ? moe_layer_freq : 1;
    if (layer_idx < first_k_dense_replace) {
        return false;
    }
    return ((layer_idx - first_k_dense_replace) % freq) == 0;
}

void DeepseekV2TextConfig::validate() const {
    if (hidden_size <= 0 || num_hidden_layers <= 0 || num_attention_heads <= 0) {
        OPENVINO_THROW("Invalid DeepseekV2 text config");
    }
    if (resolved_head_dim() <= 0) {
        OPENVINO_THROW("Invalid DeepseekV2 head dimension");
    }
    if (resolved_kv_heads() <= 0) {
        OPENVINO_THROW("Invalid DeepseekV2 KV head configuration");
    }
    if (num_attention_heads % resolved_kv_heads() != 0) {
        OPENVINO_THROW("num_attention_heads must be divisible by num_key_value_heads");
    }
    if (use_mla) {
        OPENVINO_THROW("DeepseekV2 MLA attention is not supported in modeling API yet");
    }
    if (n_routed_experts > 0 && num_experts_per_tok <= 0) {
        OPENVINO_THROW("num_experts_per_tok must be > 0 for MoE layers");
    }
    if (num_experts_per_tok > n_routed_experts && n_routed_experts > 0) {
        OPENVINO_THROW("num_experts_per_tok must be <= n_routed_experts");
    }
}

DeepseekV2EmbeddingInjector::DeepseekV2EmbeddingInjector(BuilderContext& ctx,
                                                         const std::string& name,
                                                         Module* parent)
    : Module(name, ctx, parent) {}

Tensor DeepseekV2EmbeddingInjector::forward(const Tensor& inputs_embeds,
                                            const Tensor& visual_embeds,
                                            const Tensor& images_seq_mask) const {
    auto mask = images_seq_mask.unsqueeze(2);
    auto updates = visual_embeds.to(inputs_embeds.dtype());
    return ops::tensor::masked_scatter(inputs_embeds, mask, updates);
}

DeepseekV2Attention::DeepseekV2Attention(BuilderContext& ctx,
                                         const std::string& name,
                                         const DeepseekV2TextConfig& cfg,
                                         Module* parent)
    : Module(name, ctx, parent),
      num_heads_(cfg.num_attention_heads),
      num_kv_heads_(cfg.resolved_kv_heads()),
      head_dim_(cfg.resolved_head_dim()),
      hidden_size_(cfg.hidden_size),
      scaling_(1.0f / std::sqrt(static_cast<float>(head_dim_))),
      rope_theta_(cfg.rope_theta) {
    if (num_heads_ <= 0 || head_dim_ <= 0) {
        OPENVINO_THROW("Invalid DeepseekV2 attention head configuration");
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

    if (!cfg.attention_bias) {
        q_bias_param_->set_optional(true);
        k_bias_param_->set_optional(true);
        v_bias_param_->set_optional(true);
        o_bias_param_->set_optional(true);
    }
}

const Tensor& DeepseekV2Attention::q_proj_weight() const {
    if (!q_proj_param_) {
        OPENVINO_THROW("DeepseekV2Attention q_proj parameter not registered");
    }
    return q_proj_param_->value();
}

const Tensor& DeepseekV2Attention::k_proj_weight() const {
    if (!k_proj_param_) {
        OPENVINO_THROW("DeepseekV2Attention k_proj parameter not registered");
    }
    return k_proj_param_->value();
}

const Tensor& DeepseekV2Attention::v_proj_weight() const {
    if (!v_proj_param_) {
        OPENVINO_THROW("DeepseekV2Attention v_proj parameter not registered");
    }
    return v_proj_param_->value();
}

const Tensor& DeepseekV2Attention::o_proj_weight() const {
    if (!o_proj_param_) {
        OPENVINO_THROW("DeepseekV2Attention o_proj parameter not registered");
    }
    return o_proj_param_->value();
}

const Tensor* DeepseekV2Attention::q_proj_bias() const {
    return (q_bias_param_ && q_bias_param_->is_bound()) ? &q_bias_param_->value() : nullptr;
}

const Tensor* DeepseekV2Attention::k_proj_bias() const {
    return (k_bias_param_ && k_bias_param_->is_bound()) ? &k_bias_param_->value() : nullptr;
}

const Tensor* DeepseekV2Attention::v_proj_bias() const {
    return (v_bias_param_ && v_bias_param_->is_bound()) ? &v_bias_param_->value() : nullptr;
}

const Tensor* DeepseekV2Attention::o_proj_bias() const {
    return (o_bias_param_ && o_bias_param_->is_bound()) ? &o_bias_param_->value() : nullptr;
}

Tensor DeepseekV2Attention::forward(const Tensor& positions,
                                    const Tensor& hidden_states,
                                    const Tensor& beam_idx) const {
    auto* policy = &ctx().op_policy();
    auto cos_sin = ops::llm::rope_cos_sin(positions, head_dim_, rope_theta_, policy);
    auto seq_len = Tensor(shape::dim(positions, 1), positions.context()).squeeze(0);
    auto causal_mask = ops::llm::causal_mask_from_seq_len(seq_len);
    return forward(hidden_states, beam_idx, cos_sin.first, cos_sin.second, causal_mask);
}

Tensor DeepseekV2Attention::forward(const Tensor& hidden_states,
                                    const Tensor& beam_idx,
                                    const Tensor& rope_cos,
                                    const Tensor& rope_sin,
                                    const Tensor& causal_mask) const {
    auto q = add_bias_if_present(ops::linear(hidden_states, q_proj_weight()), q_proj_bias());
    auto k = add_bias_if_present(ops::linear(hidden_states, k_proj_weight()), k_proj_bias());
    auto v = add_bias_if_present(ops::linear(hidden_states, v_proj_weight()), v_proj_bias());

    auto q_heads = q.reshape({0, 0, num_heads_, head_dim_}).permute({0, 2, 1, 3});
    auto k_heads = k.reshape({0, 0, num_kv_heads_, head_dim_}).permute({0, 2, 1, 3});
    auto v_heads = v.reshape({0, 0, num_kv_heads_, head_dim_}).permute({0, 2, 1, 3});

    auto* policy = &ctx().op_policy();
    auto q_rot = ops::llm::apply_rope(q_heads, rope_cos, rope_sin, head_dim_, policy);
    auto k_rot = ops::llm::apply_rope(k_heads, rope_cos, rope_sin, head_dim_, policy);

    const std::string cache_prefix = full_path().empty() ? name() : full_path();
    auto cached = ops::append_kv_cache(k_rot, v_heads, beam_idx, num_kv_heads_, head_dim_, cache_prefix, ctx());
    auto k_expanded = ops::llm::repeat_kv(cached.first, num_heads_, num_kv_heads_, head_dim_);
    auto v_expanded = ops::llm::repeat_kv(cached.second, num_heads_, num_kv_heads_, head_dim_);

    auto mask = ops::llm::build_kv_causal_mask(q_rot, k_expanded);
    auto context = ops::llm::sdpa(q_rot, k_expanded, v_expanded, scaling_, 3, &mask, false, policy);
    const int64_t attn_out_dim = static_cast<int64_t>(num_heads_) * head_dim_;
    auto merged = context.permute({0, 2, 1, 3}).reshape({0, 0, attn_out_dim});
    return add_bias_if_present(ops::linear(merged, o_proj_weight()), o_proj_bias());
}

DeepseekV2MLP::DeepseekV2MLP(BuilderContext& ctx,
                             const std::string& name,
                             const DeepseekV2TextConfig& cfg,
                             Module* parent)
    : Module(name, ctx, parent) {
    if (!cfg.hidden_act.empty() && cfg.hidden_act != "silu") {
        OPENVINO_THROW("Unsupported DeepseekV2 MLP activation: ", cfg.hidden_act);
    }
    gate_proj_param_ = &register_parameter("gate_proj.weight");
    up_proj_param_ = &register_parameter("up_proj.weight");
    down_proj_param_ = &register_parameter("down_proj.weight");
}

const Tensor& DeepseekV2MLP::gate_proj_weight() const {
    if (!gate_proj_param_) {
        OPENVINO_THROW("DeepseekV2MLP gate projection parameter not registered");
    }
    return gate_proj_param_->value();
}

const Tensor& DeepseekV2MLP::up_proj_weight() const {
    if (!up_proj_param_) {
        OPENVINO_THROW("DeepseekV2MLP up projection parameter not registered");
    }
    return up_proj_param_->value();
}

const Tensor& DeepseekV2MLP::down_proj_weight() const {
    if (!down_proj_param_) {
        OPENVINO_THROW("DeepseekV2MLP down projection parameter not registered");
    }
    return down_proj_param_->value();
}

Tensor DeepseekV2MLP::forward(const Tensor& x) const {
    auto gate = ops::linear(x, gate_proj_weight());
    auto up = ops::linear(x, up_proj_weight());
    auto gated = ops::silu(gate) * up;
    return ops::linear(gated, down_proj_weight());
}

DeepseekV2MoE::DeepseekV2MoE(BuilderContext& ctx,
                             const std::string& name,
                             const DeepseekV2TextConfig& cfg,
                             Module* parent)
    : Module(name, ctx, parent),
      hidden_size_(cfg.hidden_size),
      inter_size_(cfg.moe_intermediate_size > 0 ? cfg.moe_intermediate_size : cfg.intermediate_size),
      num_experts_(cfg.n_routed_experts),
      top_k_(cfg.num_experts_per_tok > 0 ? cfg.num_experts_per_tok : 1),
      shared_experts_(cfg.n_shared_experts),
      norm_topk_prob_(cfg.norm_topk_prob),
      routed_scaling_factor_(cfg.routed_scaling_factor) {
    if (!cfg.hidden_act.empty() && cfg.hidden_act != "silu") {
        OPENVINO_THROW("Unsupported DeepseekV2 MoE activation: ", cfg.hidden_act);
    }
    if (hidden_size_ <= 0 || inter_size_ <= 0 || num_experts_ <= 0) {
        OPENVINO_THROW("Invalid DeepseekV2 MoE configuration");
    }
    if (top_k_ <= 0 || top_k_ > num_experts_) {
        OPENVINO_THROW("Invalid DeepseekV2 MoE top-k configuration");
    }

    gate_param_ = &register_parameter("gate.weight");
    gate_experts_param_.resize(num_experts_);
    up_experts_param_.resize(num_experts_);
    down_experts_param_.resize(num_experts_);

    for (int32_t i = 0; i < num_experts_; ++i) {
        const std::string prefix = "experts." + std::to_string(i) + ".";
        gate_experts_param_[i] = &register_parameter(prefix + "gate_proj.weight");
        up_experts_param_[i] = &register_parameter(prefix + "up_proj.weight");
        down_experts_param_[i] = &register_parameter(prefix + "down_proj.weight");
    }

    if (shared_experts_ > 0) {
        shared_gate_param_ = &register_parameter("shared_experts.gate_proj.weight");
        shared_up_param_ = &register_parameter("shared_experts.up_proj.weight");
        shared_down_param_ = &register_parameter("shared_experts.down_proj.weight");
    }
}

const Tensor& DeepseekV2MoE::gate_weight() const {
    if (!gate_param_) {
        OPENVINO_THROW("DeepseekV2MoE gate parameter not registered");
    }
    return gate_param_->value();
}

std::vector<Tensor> DeepseekV2MoE::gate_expert_weights() const {
    std::vector<Tensor> out;
    out.reserve(gate_experts_param_.size());
    for (auto* param : gate_experts_param_) {
        out.push_back(param->value());
    }
    return out;
}

std::vector<Tensor> DeepseekV2MoE::up_expert_weights() const {
    std::vector<Tensor> out;
    out.reserve(up_experts_param_.size());
    for (auto* param : up_experts_param_) {
        out.push_back(param->value());
    }
    return out;
}

std::vector<Tensor> DeepseekV2MoE::down_expert_weights() const {
    std::vector<Tensor> out;
    out.reserve(down_experts_param_.size());
    for (auto* param : down_experts_param_) {
        out.push_back(param->value());
    }
    return out;
}

const Tensor& DeepseekV2MoE::shared_gate_weight() const {
    if (!shared_gate_param_) {
        OPENVINO_THROW("DeepseekV2MoE shared gate parameter not registered");
    }
    return shared_gate_param_->value();
}

const Tensor& DeepseekV2MoE::shared_up_weight() const {
    if (!shared_up_param_) {
        OPENVINO_THROW("DeepseekV2MoE shared up parameter not registered");
    }
    return shared_up_param_->value();
}

const Tensor& DeepseekV2MoE::shared_down_weight() const {
    if (!shared_down_param_) {
        OPENVINO_THROW("DeepseekV2MoE shared down parameter not registered");
    }
    return shared_down_param_->value();
}

Tensor DeepseekV2MoE::forward(const Tensor& x) const {
    auto* ctx = x.context();
    const auto input_dtype = x.dtype();

    auto flat = x.reshape({-1, hidden_size_});
    auto flat_f32 = flat.to(ov::element::f32);
    auto gate_w = gate_weight().to(ov::element::f32);
    auto logits = ops::linear(flat_f32, gate_w);
    auto scores = logits.softmax(1);

    auto k_node = ops::const_scalar(ctx, static_cast<int64_t>(top_k_));
    auto topk = std::make_shared<ov::op::v11::TopK>(
        scores.output(),
        k_node,
        -1,
        ov::op::v11::TopK::Mode::MAX,
        ov::op::v11::TopK::SortType::SORT_VALUES,
        ov::element::i64);

    Tensor topk_vals(topk->output(0), ctx);
    Tensor topk_idx(topk->output(1), ctx);

    if (norm_topk_prob_) {
        auto reduce_axis = ops::const_vec(ctx, std::vector<int64_t>{-1});
        auto sum = std::make_shared<ov::op::v1::ReduceSum>(topk_vals.output(), reduce_axis, true);
        topk_vals = topk_vals / Tensor(sum, ctx);
    }
    if (routed_scaling_factor_ != 1.0f) {
        topk_vals = topk_vals * routed_scaling_factor_;
    }

    auto zeros = shape::broadcast_to(Tensor(ops::const_scalar(ctx, 0.0f), ctx), shape::of(scores));
    auto scatter_axis = ops::const_scalar(ctx, static_cast<int64_t>(1));
    auto scatter = std::make_shared<ov::op::v12::ScatterElementsUpdate>(
        zeros.output(),
        topk_idx.output(),
        topk_vals.output(),
        scatter_axis);
    Tensor routing(scatter, ctx);  // [T, E]

    auto perm = ops::const_vec(ctx, std::vector<int64_t>{1, 0});
    auto routing_t = Tensor(std::make_shared<ov::op::v1::Transpose>(routing.output(), perm), ctx);
    auto routing_3d = routing_t.unsqueeze(-1);  // [E, T, 1]

    auto flat_3d = flat.unsqueeze(0);
    auto tiled = ops::tensor::tile(flat_3d, {num_experts_, 1, 1});  // [E, T, H]

    auto gate_weights = ops::tensor::stack(gate_expert_weights(), 0);
    auto up_weights = ops::tensor::stack(up_expert_weights(), 0);

    auto down_weights = ops::tensor::stack(down_expert_weights(), 0);

    auto gate_bmm = ops::matmul(tiled, gate_weights, false, true);
    auto up_bmm = ops::matmul(tiled, up_weights, false, true);
    auto swiglu = ops::silu(gate_bmm) * up_bmm;
    auto down_bmm = ops::matmul(swiglu, down_weights, false, true);

    auto down_f32 = down_bmm.to(ov::element::f32);
    auto weighted = down_f32 * routing_3d;
    auto reduce_axis = ops::const_vec(ctx, std::vector<int64_t>{0});
    auto reduced = std::make_shared<ov::op::v1::ReduceSum>(weighted.output(), reduce_axis, false);
    Tensor routed_out(reduced, ctx);
    routed_out = routed_out.reshape(shape::of(x), false);
    routed_out = routed_out.to(input_dtype);

    if (shared_experts_ > 0) {
        auto shared_gate = ops::linear(x, shared_gate_weight());
        auto shared_up = ops::linear(x, shared_up_weight());
        auto shared_hidden = ops::silu(shared_gate) * shared_up;
        auto shared_out = ops::linear(shared_hidden, shared_down_weight());
        shared_out = shared_out.to(routed_out.dtype());
        return routed_out + shared_out;
    }
    return routed_out;
}

DeepseekV2DecoderLayer::DeepseekV2DecoderLayer(BuilderContext& ctx,
                                               const std::string& name,
                                               const DeepseekV2TextConfig& cfg,
                                               bool is_moe,
                                               Module* parent)
    : Module(name, ctx, parent),
      self_attn_(ctx, "self_attn", cfg, this),
      input_layernorm_(ctx, "input_layernorm", cfg.rms_norm_eps, this),
      post_attention_layernorm_(ctx, "post_attention_layernorm", cfg.rms_norm_eps, this),
      is_moe_(is_moe) {
    if (is_moe_) {
        moe_ = std::make_unique<DeepseekV2MoE>(ctx, "mlp", cfg, this);
    } else {
        mlp_ = std::make_unique<DeepseekV2MLP>(ctx, "mlp", cfg, this);
    }
}

std::pair<Tensor, Tensor> DeepseekV2DecoderLayer::forward(const Tensor& hidden_states,
                                                          const Tensor& beam_idx,
                                                          const Tensor& rope_cos,
                                                          const Tensor& rope_sin,
                                                          const Tensor& causal_mask,
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

    auto attn_out = self_attn_.forward(normed, beam_idx, rope_cos, rope_sin, causal_mask);
    auto post_norm = post_attention_layernorm_.forward(attn_out, next_residual);

    Tensor mlp_out;
    if (is_moe_) {
        mlp_out = moe_->forward(post_norm.first);
    } else {
        mlp_out = mlp_->forward(post_norm.first);
    }
    return {mlp_out, post_norm.second};
}

DeepseekV2Model::DeepseekV2Model(BuilderContext& ctx, const DeepseekV2TextConfig& cfg, Module* parent)
    : Module("model", ctx, parent),
      cfg_(cfg),
      embed_tokens_(ctx, "embed_tokens", this),
      embedding_injector_(ctx, "embedding_injector", this),
      layers_(),
      norm_(ctx, "norm", cfg.rms_norm_eps, this),
      head_dim_(cfg.resolved_head_dim()),
      rope_theta_(cfg.rope_theta) {
    cfg_.validate();

    layers_.reserve(static_cast<size_t>(cfg.num_hidden_layers));
    for (int32_t i = 0; i < cfg.num_hidden_layers; ++i) {
        const bool is_moe = cfg.is_moe_layer(i);
        layers_.emplace_back(ctx, "layers[" + std::to_string(i) + "]", cfg, is_moe, this);
    }
}

Tensor DeepseekV2Model::forward(const Tensor& input_ids,
                                const Tensor& position_ids,
                                const Tensor& beam_idx,
                                const Tensor* visual_embeds,
                                const Tensor* images_seq_mask) {
    auto embeds = embed_tokens_.forward(input_ids);
    return forward_embeds(embeds, position_ids, beam_idx, visual_embeds, images_seq_mask);
}

Tensor DeepseekV2Model::forward_embeds(const Tensor& inputs_embeds,
                                       const Tensor& position_ids,
                                       const Tensor& beam_idx,
                                       const Tensor* visual_embeds,
                                       const Tensor* images_seq_mask) {
    Tensor hidden_states = inputs_embeds;
    if (visual_embeds && images_seq_mask) {
        hidden_states = embedding_injector_.forward(hidden_states, *visual_embeds, *images_seq_mask);
    }

    auto* policy = &ctx().op_policy();
    auto cos_sin = ops::llm::rope_cos_sin(position_ids, head_dim_, rope_theta_, policy);
    auto seq_len = Tensor(shape::dim(position_ids, 1), position_ids.context()).squeeze(0);
    auto causal_mask = ops::llm::causal_mask_from_seq_len(seq_len);

    std::optional<Tensor> residual;
    for (auto& layer : layers_) {
        auto layer_out = layer.forward(hidden_states, beam_idx, cos_sin.first, cos_sin.second, causal_mask, residual);
        hidden_states = layer_out.first;
        residual = layer_out.second;
    }

    if (residual) {
        return norm_.forward(hidden_states, *residual).first;
    }
    return norm_.forward(hidden_states);
}

VocabEmbedding& DeepseekV2Model::embed_tokens() {
    return embed_tokens_;
}

RMSNorm& DeepseekV2Model::norm() {
    return norm_;
}

DeepseekV2ForCausalLM::DeepseekV2ForCausalLM(BuilderContext& ctx,
                                             const DeepseekV2TextConfig& cfg,
                                             Module* parent)
    : Module("", ctx, parent),
      cfg_(cfg),
      model_(ctx, cfg, this),
      lm_head_(ctx, "lm_head", this) {
    if (cfg_.tie_word_embeddings) {
        lm_head_.tie_to(model_.embed_tokens().weight_param());
    }
}

Tensor DeepseekV2ForCausalLM::forward(const Tensor& input_ids,
                                      const Tensor& position_ids,
                                      const Tensor& beam_idx,
                                      const Tensor* visual_embeds,
                                      const Tensor* images_seq_mask) {
    auto hidden = model_.forward(input_ids, position_ids, beam_idx, visual_embeds, images_seq_mask);
    return lm_head_.forward(hidden);
}

Tensor DeepseekV2ForCausalLM::forward_embeds(const Tensor& inputs_embeds,
                                             const Tensor& position_ids,
                                             const Tensor& beam_idx,
                                             const Tensor* visual_embeds,
                                             const Tensor* images_seq_mask) {
    auto hidden = model_.forward_embeds(inputs_embeds, position_ids, beam_idx, visual_embeds, images_seq_mask);
    return lm_head_.forward(hidden);
}

DeepseekV2Model& DeepseekV2ForCausalLM::model() {
    return model_;
}

LMHead& DeepseekV2ForCausalLM::lm_head() {
    return lm_head_;
}

std::shared_ptr<ov::Model> create_deepseek_v2_text_model(
    const DeepseekOCR2LanguageConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer,
    bool use_inputs_embeds,
    bool enable_visual_inputs) {
    DeepseekV2TextConfig tcfg;
    tcfg.hidden_size = cfg.hidden_size;
    tcfg.num_attention_heads = cfg.num_attention_heads;
    tcfg.num_key_value_heads = cfg.resolved_kv_heads();
    tcfg.intermediate_size = cfg.intermediate_size;
    tcfg.moe_intermediate_size = cfg.moe_intermediate_size;
    tcfg.num_hidden_layers = cfg.num_hidden_layers;
    tcfg.n_routed_experts = cfg.n_routed_experts;
    tcfg.n_shared_experts = cfg.n_shared_experts;
    tcfg.num_experts_per_tok = cfg.num_experts_per_tok;
    tcfg.moe_layer_freq = cfg.moe_layer_freq;
    tcfg.first_k_dense_replace = cfg.first_k_dense_replace;
    tcfg.rms_norm_eps = cfg.rms_norm_eps;
    tcfg.rope_theta = cfg.rope_theta;
    tcfg.hidden_act = cfg.hidden_act;
    tcfg.attention_bias = cfg.attention_bias;
    tcfg.use_mla = cfg.use_mla;

    BuilderContext ctx;
    DeepseekV2ForCausalLM model(ctx, tcfg);

    ov::genai::modeling::weights::LoadOptions options;
    options.allow_unmatched = true;
    options.allow_missing = false;
    options.report_missing = true;
    options.report_unmatched = true;
    auto report = ov::genai::modeling::weights::load_model(model, source, finalizer, options);
    (void)report;

    const auto float_type = ov::element::f32;

    auto attention_mask = ctx.parameter(DeepseekV2TextIO::kAttentionMask,
                                        ov::element::i64,
                                        ov::PartialShape{-1, -1});
    auto position_ids = ctx.parameter(DeepseekV2TextIO::kPositionIds,
                                      ov::element::i64,
                                      ov::PartialShape{-1, -1});
    auto beam_idx = ctx.parameter(DeepseekV2TextIO::kBeamIdx,
                                  ov::element::i32,
                                  ov::PartialShape{-1});

    (void)attention_mask;

    const Tensor* visual_embeds_ptr = nullptr;
    const Tensor* images_seq_mask_ptr = nullptr;
    Tensor visual_embeds;
    Tensor images_seq_mask;
    if (enable_visual_inputs) {
        visual_embeds = ctx.parameter(DeepseekV2TextIO::kVisualEmbeds,
                                      float_type,
                                      ov::PartialShape{-1, -1, tcfg.hidden_size});
        images_seq_mask = ctx.parameter(DeepseekV2TextIO::kImagesSeqMask,
                                        ov::element::boolean,
                                        ov::PartialShape{-1, -1});
        visual_embeds_ptr = &visual_embeds;
        images_seq_mask_ptr = &images_seq_mask;
    }

    Tensor logits;
    if (use_inputs_embeds) {
        auto inputs_embeds = ctx.parameter(DeepseekV2TextIO::kInputsEmbeds,
                                           float_type,
                                           ov::PartialShape{-1, -1, tcfg.hidden_size});
        logits = model.forward_embeds(inputs_embeds,
                                      position_ids,
                                      beam_idx,
                                      visual_embeds_ptr,
                                      images_seq_mask_ptr);
    } else {
        auto input_ids = ctx.parameter(DeepseekV2TextIO::kInputIds,
                                       ov::element::i64,
                                       ov::PartialShape{-1, -1});
        logits = model.forward(input_ids,
                               position_ids,
                               beam_idx,
                               visual_embeds_ptr,
                               images_seq_mask_ptr);
    }

    auto result = std::make_shared<ov::op::v0::Result>(logits.output());
    set_name(result, DeepseekV2TextIO::kLogits);
    return ctx.build_model({result->output(0)});
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
