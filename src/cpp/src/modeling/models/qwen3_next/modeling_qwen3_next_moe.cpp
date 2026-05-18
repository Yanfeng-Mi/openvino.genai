// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_next/modeling_qwen3_next_moe.hpp"

#include <vector>

#include <openvino/core/except.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/models/qwen3_next/modeling_qwen3_next.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/ops/shape.hpp"
#include "modeling/ops/tensor_ops.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

Qwen3NextSparseMoeBlock::Qwen3NextSparseMoeBlock(BuilderContext& ctx,
                                                 const std::string& name,
                                                 const Qwen3NextConfig& cfg,
                                                 Module* parent)
    : Module(name, ctx, parent),
      hidden_size_(cfg.hidden_size),
      expert_intermediate_size_(cfg.moe_intermediate_size),
      shared_intermediate_size_(cfg.shared_expert_intermediate_size),
      num_experts_(cfg.num_experts),
      top_k_(cfg.num_experts_per_tok > 0 ? cfg.num_experts_per_tok : 1),
      norm_topk_prob_(cfg.norm_topk_prob),
      group_size_((cfg.group_size > 0) ? static_cast<size_t>(cfg.group_size) : 128) {
    if (!cfg.hidden_act.empty() && cfg.hidden_act != "silu") {
        OPENVINO_THROW("Unsupported Qwen3Next MoE activation: ", cfg.hidden_act);
    }
    if (hidden_size_ <= 0 || expert_intermediate_size_ <= 0 || shared_intermediate_size_ <= 0 || num_experts_ <= 0) {
        OPENVINO_THROW("Invalid Qwen3Next MoE configuration");
    }
    if (top_k_ <= 0 || top_k_ > num_experts_) {
        OPENVINO_THROW("Invalid Qwen3Next MoE top-k configuration");
    }

    gate_param_ = &register_parameter("gate.weight");
    shared_expert_gate_param_ = &register_parameter("shared_expert_gate.weight");
    shared_gate_proj_param_ = &register_parameter("shared_expert.gate_proj.weight");
    shared_up_proj_param_ = &register_parameter("shared_expert.up_proj.weight");
    shared_down_proj_param_ = &register_parameter("shared_expert.down_proj.weight");

    // Initialize expert parameter vectors
    gate_experts_param_.resize(static_cast<size_t>(num_experts_));
    up_experts_param_.resize(static_cast<size_t>(num_experts_));
    down_experts_param_.resize(static_cast<size_t>(num_experts_));

    // Initialize scales and zps vectors for quantized MoE weights
    gate_exps_scales_.resize(static_cast<size_t>(num_experts_));
    gate_exps_zps_.resize(static_cast<size_t>(num_experts_));
    up_exps_scales_.resize(static_cast<size_t>(num_experts_));
    up_exps_zps_.resize(static_cast<size_t>(num_experts_));
    down_exps_scales_.resize(static_cast<size_t>(num_experts_));
    down_exps_zps_.resize(static_cast<size_t>(num_experts_));

    for (int32_t i = 0; i < num_experts_; ++i) {
        const std::string prefix = "experts." + std::to_string(i) + ".";
        const size_t idx = static_cast<size_t>(i);

        gate_experts_param_[idx] = &register_parameter(prefix + "gate_proj.weight");
        up_experts_param_[idx] = &register_parameter(prefix + "up_proj.weight");
        down_experts_param_[idx] = &register_parameter(prefix + "down_proj.weight");

        gate_experts_param_[idx]->set_weight_loader([this, idx](WeightParameter& param,
                                                                 weights::WeightSource& source,
                                                                 weights::WeightFinalizer& finalizer,
                                                                 const std::string& weight_name,
                                                                 const std::optional<int>& shard_id) {
            (void)shard_id;
            if (!param.context()) {
                OPENVINO_THROW("WeightParameter has no OpContext: ", param.name());
            }
            auto weight = finalizer.finalize(weight_name, source, *param.context());
            param.bind(weight);
            if (weight.get_auxiliary("scales") != std::nullopt &&
                weight.get_auxiliary("zps") != std::nullopt) {
                gate_exps_scales_[idx] = weight.auxiliary.at("scales");
                gate_exps_zps_[idx] = weight.auxiliary.at("zps");
            }
        });

        up_experts_param_[idx]->set_weight_loader([this, idx](WeightParameter& param,
                                                               weights::WeightSource& source,
                                                               weights::WeightFinalizer& finalizer,
                                                               const std::string& weight_name,
                                                               const std::optional<int>& shard_id) {
            (void)shard_id;
            if (!param.context()) {
                OPENVINO_THROW("WeightParameter has no OpContext: ", param.name());
            }
            auto weight = finalizer.finalize(weight_name, source, *param.context());
            param.bind(weight);
            if (weight.get_auxiliary("scales") != std::nullopt &&
                weight.get_auxiliary("zps") != std::nullopt) {
                up_exps_scales_[idx] = weight.auxiliary.at("scales");
                up_exps_zps_[idx] = weight.auxiliary.at("zps");
            }
        });

        down_experts_param_[idx]->set_weight_loader([this, idx](WeightParameter& param,
                                                                 weights::WeightSource& source,
                                                                 weights::WeightFinalizer& finalizer,
                                                                 const std::string& weight_name,
                                                                 const std::optional<int>& shard_id) {
            (void)shard_id;
            if (!param.context()) {
                OPENVINO_THROW("WeightParameter has no OpContext: ", param.name());
            }
            auto weight = finalizer.finalize(weight_name, source, *param.context());
            param.bind(weight);
            if (weight.get_auxiliary("scales") != std::nullopt &&
                weight.get_auxiliary("zps") != std::nullopt) {
                down_exps_scales_[idx] = weight.auxiliary.at("scales");
                down_exps_zps_[idx] = weight.auxiliary.at("zps");
            }
        });
    }
}

const Tensor& Qwen3NextSparseMoeBlock::gate_weight() const {
    if (!gate_param_) {
        OPENVINO_THROW("Qwen3NextSparseMoeBlock gate parameter is not registered");
    }
    return gate_param_->value();
}

const Tensor& Qwen3NextSparseMoeBlock::shared_expert_gate_weight() const {
    if (!shared_expert_gate_param_) {
        OPENVINO_THROW("Qwen3NextSparseMoeBlock shared_expert_gate parameter is not registered");
    }
    return shared_expert_gate_param_->value();
}

const Tensor& Qwen3NextSparseMoeBlock::shared_gate_proj_weight() const {
    if (!shared_gate_proj_param_) {
        OPENVINO_THROW("Qwen3NextSparseMoeBlock shared gate_proj parameter is not registered");
    }
    return shared_gate_proj_param_->value();
}

const Tensor& Qwen3NextSparseMoeBlock::shared_up_proj_weight() const {
    if (!shared_up_proj_param_) {
        OPENVINO_THROW("Qwen3NextSparseMoeBlock shared up_proj parameter is not registered");
    }
    return shared_up_proj_param_->value();
}

const Tensor& Qwen3NextSparseMoeBlock::shared_down_proj_weight() const {
    if (!shared_down_proj_param_) {
        OPENVINO_THROW("Qwen3NextSparseMoeBlock shared down_proj parameter is not registered");
    }
    return shared_down_proj_param_->value();
}

Tensor Qwen3NextSparseMoeBlock::gate_expert_weights() const {
    std::vector<Tensor> ws;
    ws.reserve(gate_experts_param_.size());
    for (auto* p : gate_experts_param_) {
        ws.push_back(p->value());
    }
    auto result = ops::concat(ws, 0);
    result.output().get_node()->get_rt_info()["postponed_constant"] = true;
    return result;
}

Tensor Qwen3NextSparseMoeBlock::up_expert_weights() const {
    std::vector<Tensor> ws;
    ws.reserve(up_experts_param_.size());
    for (auto* p : up_experts_param_) {
        ws.push_back(p->value());
    }
    auto result = ops::concat(ws, 0);
    result.output().get_node()->get_rt_info()["postponed_constant"] = true;
    return result;
}

Tensor Qwen3NextSparseMoeBlock::down_expert_weights() const {
    std::vector<Tensor> ws;
    ws.reserve(down_experts_param_.size());
    for (auto* p : down_experts_param_) {
        ws.push_back(p->value());
    }
    auto result = ops::concat(ws, 0);
    result.output().get_node()->get_rt_info()["postponed_constant"] = true;
    return result;
}

Tensor Qwen3NextSparseMoeBlock::gate_exps_scales() const {
    auto result = ops::concat(gate_exps_scales_, 0);
    result.output().get_node()->get_rt_info()["postponed_constant"] = true;
    return result;
}

Tensor Qwen3NextSparseMoeBlock::gate_exps_zps() const {
    auto result = ops::concat(gate_exps_zps_, 0);
    result.output().get_node()->get_rt_info()["postponed_constant"] = true;
    return result;
}

Tensor Qwen3NextSparseMoeBlock::up_exps_scales() const {
    auto result = ops::concat(up_exps_scales_, 0);
    result.output().get_node()->get_rt_info()["postponed_constant"] = true;
    return result;
}

Tensor Qwen3NextSparseMoeBlock::up_exps_zps() const {
    auto result = ops::concat(up_exps_zps_, 0);
    result.output().get_node()->get_rt_info()["postponed_constant"] = true;
    return result;
}

Tensor Qwen3NextSparseMoeBlock::down_exps_scales() const {
    auto result = ops::concat(down_exps_scales_, 0);
    result.output().get_node()->get_rt_info()["postponed_constant"] = true;
    return result;
}

Tensor Qwen3NextSparseMoeBlock::down_exps_zps() const {
    auto result = ops::concat(down_exps_zps_, 0);
    result.output().get_node()->get_rt_info()["postponed_constant"] = true;
    return result;
}

Tensor Qwen3NextSparseMoeBlock::forward(const Tensor& hidden_states) const {
    auto* op_ctx = hidden_states.context();
    auto input_dtype = hidden_states.dtype();

    auto flat = hidden_states.reshape({-1, hidden_size_});
    auto flat_f32 = flat.to(ov::element::f32);

    // Use optimized MoE kernel for routed experts
    auto routed_out = ops::moe3gemm_fused_compressed(
        flat_f32,
        gate_weight(),
        gate_expert_weights(),
        gate_exps_scales(),
        gate_exps_zps(),
        up_expert_weights(),
        up_exps_scales(),
        up_exps_zps(),
        down_expert_weights(),
        down_exps_scales(),
        down_exps_zps(),
        hidden_size_,
        expert_intermediate_size_,
        num_experts_,
        top_k_,
        group_size_,
        ov::element::f16);

    // Shared expert computation (using standard linear ops)
    auto shared_gate = ops::linear(flat_f32, shared_gate_proj_weight().to(ov::element::f32));
    auto shared_up = ops::linear(flat_f32, shared_up_proj_weight().to(ov::element::f32));
    auto shared_hidden = ops::silu(shared_gate) * shared_up;
    auto shared_out = ops::linear(shared_hidden, shared_down_proj_weight().to(ov::element::f32));

    // Gated combination of routed and shared expert outputs
    auto shared_gate_logits = ops::linear(flat_f32, shared_expert_gate_weight().to(ov::element::f32));
    auto shared_gate_sigmoid = Tensor(std::make_shared<ov::op::v0::Sigmoid>(shared_gate_logits.output()), op_ctx);
    auto combined = routed_out + (shared_out * shared_gate_sigmoid);
    auto restored = combined.reshape(shape::of(hidden_states), false);
    return restored.to(input_dtype);
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
