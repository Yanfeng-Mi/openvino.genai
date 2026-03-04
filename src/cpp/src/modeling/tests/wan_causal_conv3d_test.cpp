// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/nn.hpp"
#include "modeling/ops/tensor.hpp"
#include "modeling/tests/test_utils.hpp"

namespace test_utils = ov::genai::modeling::tests;

namespace {

std::shared_ptr<ov::Model> build_model_from_output(const ov::Output<ov::Node>& output,
                                                   const ov::ParameterVector& params) {
    auto result = std::make_shared<ov::op::v0::Result>(output);
    return std::make_shared<ov::Model>(ov::OutputVector{result}, params);
}

void run_model_test(const std::shared_ptr<ov::Model>& model,
                    const ov::Tensor& input_tensor,
                    const std::vector<float>& expected,
                    float tol = test_utils::k_tol_default) {
    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();
    request.set_input_tensor(input_tensor);
    request.infer();
    ov::Tensor output = request.get_output_tensor();
    test_utils::expect_tensor_near(output, expected, tol);
}

}  // namespace

TEST(WanCausalConv3dTest, CausalPaddingTimeAxis) {
    ov::genai::modeling::OpContext ctx;
    const ov::Shape in_shape{1, 1, 3, 1, 1};
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, in_shape);
    ov::genai::modeling::Tensor x(param, &ctx);

    std::vector<float> weight_data{1.0f, 1.0f, 1.0f};
    auto weight = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 3, 1, 1}, weight_data);
    ov::genai::modeling::Tensor w(weight, &ctx);

    auto out = ov::genai::modeling::ops::nn::causal_conv3d(x, w, {1, 1, 1}, {1, 0, 0});
    auto model = build_model_from_output(out.output(), {param});

    std::vector<float> input_data{1.0f, 2.0f, 3.0f};
    ov::Tensor input_tensor(ov::element::f32, in_shape);
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));

    std::vector<float> expected{1.0f, 3.0f, 6.0f};
    run_model_test(model, input_tensor, expected, test_utils::k_tol_default);
}
