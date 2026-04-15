// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/rope.hpp"
#include "modeling/ops/tensor.hpp"
#include "modeling/tests/test_utils.hpp"

namespace test_utils = ov::genai::modeling::tests;

namespace {

std::shared_ptr<ov::Model> build_model_from_output(const ov::Output<ov::Node>& output,
                                                   const ov::ParameterVector& params) {
    auto result = std::make_shared<ov::op::v0::Result>(output);
    return std::make_shared<ov::Model>(ov::OutputVector{result}, params);
}

}  // namespace

TEST(RopeOps, MropeInterleaved) {
    ov::genai::modeling::OpContext ctx;
    const ov::Shape shape{3, 1, 1, 6};
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    ov::genai::modeling::Tensor freqs(param, &ctx);

    std::vector<int32_t> section{0, 1, 1};
    auto out = ov::genai::modeling::ops::rope::mrope_interleaved(freqs, section);
    auto model = build_model_from_output(out.output(), {param});

    std::vector<float> input_data = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
        20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f};
    ov::Tensor input_tensor(ov::element::f32, shape);
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));

    std::vector<float> expected = {0.0f, 11.0f, 22.0f, 3.0f, 4.0f, 5.0f};

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();
    request.set_input_tensor(input_tensor);
    request.infer();

    ov::Tensor output = request.get_output_tensor();
    test_utils::expect_tensor_near(output, expected, test_utils::k_tol_linear);
}
