// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "modeling/module.hpp"
#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

struct Qwen3NextConfig;

class Qwen3NextSparseMoeBlock : public Module {
public:
    Qwen3NextSparseMoeBlock(BuilderContext& ctx, const std::string& name, const Qwen3NextConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states) const;

private:
    const Tensor& gate_weight() const;
    const Tensor& shared_expert_gate_weight() const;
    const Tensor& shared_gate_proj_weight() const;
    const Tensor& shared_up_proj_weight() const;
    const Tensor& shared_down_proj_weight() const;

    // MoE expert weights (stacked)
    Tensor gate_expert_weights() const;
    Tensor up_expert_weights() const;
    Tensor down_expert_weights() const;

    // MoE quantization scales and zero-points (stacked)
    Tensor gate_exps_scales() const;
    Tensor gate_exps_zps() const;
    Tensor up_exps_scales() const;
    Tensor up_exps_zps() const;
    Tensor down_exps_scales() const;
    Tensor down_exps_zps() const;

    int32_t hidden_size_ = 0;
    int32_t expert_intermediate_size_ = 0;
    int32_t shared_intermediate_size_ = 0;
    int32_t num_experts_ = 0;
    int32_t top_k_ = 1;
    bool norm_topk_prob_ = true;
    size_t group_size_ = 128;

    WeightParameter* gate_param_ = nullptr;
    WeightParameter* shared_expert_gate_param_ = nullptr;
    WeightParameter* shared_gate_proj_param_ = nullptr;
    WeightParameter* shared_up_proj_param_ = nullptr;
    WeightParameter* shared_down_proj_param_ = nullptr;
    std::vector<WeightParameter*> gate_experts_param_;
    std::vector<WeightParameter*> up_experts_param_;
    std::vector<WeightParameter*> down_experts_param_;

    // Quantization scales and zero-points for each expert
    std::vector<Tensor> gate_exps_scales_;
    std::vector<Tensor> gate_exps_zps_;
    std::vector<Tensor> up_exps_scales_;
    std::vector<Tensor> up_exps_zps_;
    std::vector<Tensor> down_exps_scales_;
    std::vector<Tensor> down_exps_zps_;
};

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
