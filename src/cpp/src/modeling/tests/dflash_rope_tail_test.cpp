// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>

#include "modeling/builder_context.hpp"
#include "modeling/ops/llm.hpp"
#include "modeling/tests/test_utils.hpp"

namespace test_utils = ov::genai::modeling::tests;

TEST(DFlashOps, RopeTailSlices) {
    ov::genai::modeling::BuilderContext ctx;

    auto cos = ctx.parameter("cos", ov::element::f32, ov::PartialShape{1, 5, 2});
    auto q = ctx.parameter("q", ov::element::f32, ov::PartialShape{1, 1, 2, 4});

    auto tail = ov::genai::modeling::ops::llm::rope_tail(cos, q);
    auto result = std::make_shared<ov::op::v0::Result>(tail.output());
    auto ov_model = ctx.build_model({result->output(0)});

    ov::Core core;
    auto compiled = core.compile_model(ov_model, "CPU");
    auto request = compiled.create_infer_request();

    std::vector<float> cos_data = {0.0f, 1.0f,
                                   2.0f, 3.0f,
                                   4.0f, 5.0f,
                                   6.0f, 7.0f,
                                   8.0f, 9.0f};
    ov::Tensor cos_tensor(ov::element::f32, {1, 5, 2});
    std::memcpy(cos_tensor.data(), cos_data.data(), cos_data.size() * sizeof(float));
    request.set_tensor("cos", cos_tensor);

    std::vector<float> q_data(1 * 1 * 2 * 4, 0.0f);
    ov::Tensor q_tensor(ov::element::f32, {1, 1, 2, 4});
    std::memcpy(q_tensor.data(), q_data.data(), q_data.size() * sizeof(float));
    request.set_tensor("q", q_tensor);

    request.infer();

    std::vector<float> expected = {6.0f, 7.0f, 8.0f, 9.0f};
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, 1e-6f);
}
