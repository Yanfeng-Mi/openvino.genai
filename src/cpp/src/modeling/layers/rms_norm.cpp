// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/layers/rms_norm.hpp"

#include <openvino/core/except.hpp>
#include <openvino/openvino.hpp>
#include <ov_ops/rms.hpp>

namespace ov {
namespace genai {
namespace modeling {

RMSNorm::RMSNorm(const Tensor& weight, float eps) : Module(), weight_(weight), eps_(eps) {}

RMSNorm::RMSNorm(BuilderContext& ctx, const std::string& name, float eps, Module* parent)
    : Module(name, ctx, parent), eps_(eps) {
    weight_param_ = &register_parameter("weight");
}

WeightParameter& RMSNorm::weight_param() {
    if (!weight_param_) {
        OPENVINO_THROW("RMSNorm has no registered parameter");
    }
    return *weight_param_;
}

const WeightParameter& RMSNorm::weight_param() const {
    if (!weight_param_) {
        OPENVINO_THROW("RMSNorm has no registered parameter");
    }
    return *weight_param_;
}

const Tensor& RMSNorm::weight() const {
    if (weight_param_) {
        return weight_param_->value();
    }
    return weight_;
}

Tensor RMSNorm::forward(const Tensor& x) const {
    auto orig_dtype = x.dtype();
    auto xf = x.to(ov::element::f32);
    auto wf = weight().to(ov::element::f32);
    auto rms_node = std::make_shared<ov::op::internal::RMS>(
        xf.output(), wf.output(), static_cast<double>(eps_), orig_dtype);
    return Tensor(rms_node->output(0), x.context());
}

std::pair<Tensor, Tensor> RMSNorm::forward(const Tensor& x, const Tensor& residual) const {
    auto orig_dtype = x.dtype();
    auto xf = x.to(ov::element::f32);
    auto rf = residual.to(ov::element::f32);
    auto sum = xf + rf;
    auto residual_out = sum.to(orig_dtype);
    auto wf = weight().to(ov::element::f32);
    auto rms_node = std::make_shared<ov::op::internal::RMS>(
        sum.output(), wf.output(), static_cast<double>(eps_), orig_dtype);
    return {Tensor(rms_node->output(0), x.context()), residual_out};
}

}  // namespace modeling
}  // namespace genai
}  // namespace ov

