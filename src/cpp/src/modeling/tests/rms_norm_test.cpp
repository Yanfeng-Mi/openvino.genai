// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>
#include <ov_ops/rms.hpp>

#include "modeling/layers/rms_norm.hpp"
#include "modeling/ops/context.hpp"
#include "modeling/ops/tensor.hpp"
#include "modeling/tests/test_utils.hpp"

namespace test_utils = ov::genai::modeling::tests;

namespace {

const float kRmsEps = 1e-6f;
const ov::Shape kRmsInputShape{1, 2, 4};
const std::vector<float> kRmsWeight = {1.0f, 0.5f, 2.0f, -1.0f};
const std::vector<float> kRmsInput = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
const std::vector<float> kRmsExpected =
    test_utils::rms_ref(
        kRmsInput,
        kRmsWeight,
        kRmsInputShape[0] * kRmsInputShape[1],
        kRmsInputShape[2],
        kRmsEps);

std::shared_ptr<ov::Model> build_model_from_output(
    const ov::Output<ov::Node>& output,
    const ov::ParameterVector& params) {
    auto result = std::make_shared<ov::op::v0::Result>(output);
    return std::make_shared<ov::Model>(ov::OutputVector{result}, params);
}

void run_rms_model_test(const std::shared_ptr<ov::Model>& model,
                        const std::vector<float>& input,
                        const std::vector<float>& expected,
                        const ov::Shape& input_shape,
                        const std::string& model_file,
                        float tol = test_utils::k_tol_default) {
    ov::serialize(model, model_file);

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    ov::Tensor input_tensor(ov::element::f32, input_shape);
    std::memcpy(input_tensor.data(), input.data(), input.size() * sizeof(float));
    request.set_input_tensor(input_tensor);
    request.infer();

    ov::Tensor output = request.get_output_tensor();
    test_utils::expect_tensor_near(output, expected, tol);
}

ov::Output<ov::Node> make_rms_norm_opset(const ov::Output<ov::Node>& input,
                                         const ov::Output<ov::Node>& weight,
                                         float eps) {
    auto eps_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, eps);
    auto power_const = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, 2.0f);
    auto square = std::make_shared<ov::op::v1::Power>(input, power_const);
    auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, -1);
    auto variance = std::make_shared<ov::op::v1::ReduceMean>(square, axis, true);
    auto add_eps = std::make_shared<ov::op::v1::Add>(variance, eps_node);
    auto sqrt_node = std::make_shared<ov::op::v0::Sqrt>(add_eps);
    auto one_const = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, 1.0f);
    auto reciprocal = std::make_shared<ov::op::v1::Divide>(one_const, sqrt_node);
    auto norm = std::make_shared<ov::op::v1::Multiply>(reciprocal, input, ov::op::AutoBroadcastType::NUMPY);

    auto weight_f32 = std::make_shared<ov::op::v0::Convert>(weight, ov::element::f32);
    return std::make_shared<ov::op::v1::Multiply>(norm, weight_f32, ov::op::AutoBroadcastType::NUMPY);
}

ov::Output<ov::Node> make_rms_norm_internal(const ov::Output<ov::Node>& input,
                                            const ov::Output<ov::Node>& weight,
                                            float eps) {
    auto weight_f32 = std::make_shared<ov::op::v0::Convert>(weight, ov::element::f32);
    return std::make_shared<ov::op::internal::RMS>(input, weight_f32, static_cast<double>(eps), ov::element::f32);
}

}  // namespace

TEST(RMSNormLayer, ModelingApi) {
    ov::genai::modeling::OpContext ctx;

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, kRmsInputShape);
    ov::genai::modeling::Tensor x(param, &ctx);

    auto weight_node = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{4}, kRmsWeight);
    ov::genai::modeling::Tensor w(weight_node, &ctx);

    ov::genai::modeling::RMSNorm rms(w, kRmsEps);
    auto output = rms.forward(x);

    auto model = build_model_from_output(output.output(), {param});
    run_rms_model_test(model, kRmsInput, kRmsExpected, kRmsInputShape, "rms_ult_model_modeling.xml");
}

TEST(RMSNormLayer, InternalOp) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, kRmsInputShape);
    auto weight_node = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{4}, kRmsWeight);

    auto output = make_rms_norm_internal(param, weight_node, kRmsEps);
    auto model = build_model_from_output(output, {param});
    run_rms_model_test(model, kRmsInput, kRmsExpected, kRmsInputShape, "rms_ult_model_internal.xml");
}

TEST(RMSNormLayer, Opset) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, kRmsInputShape);
    auto weight_node = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{4}, kRmsWeight);

    auto output = make_rms_norm_opset(param, weight_node, kRmsEps);
    auto model = build_model_from_output(output, {param});
    run_rms_model_test(model, kRmsInput, kRmsExpected, kRmsInputShape, "rms_ult_model_opset.xml");
}

