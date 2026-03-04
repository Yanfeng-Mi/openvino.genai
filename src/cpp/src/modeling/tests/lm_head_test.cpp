// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>
#include "modeling/builder_context.hpp"
#include "modeling/layers/lm_head.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/tests/test_utils.hpp"

namespace test_utils = ov::genai::modeling::tests;

TEST(LMHeadLayer, Decode) {
    ov::genai::modeling::BuilderContext ctx;

    const ov::Shape x_shape{2, 4};
    const ov::Shape w_shape{5, 4};

    const std::vector<float> x_data = {
        1.f, 2.f, 3.f, 4.f,  //
        5.f, 6.f, 7.f, 8.f,  //
    };
    const std::vector<float> w_data = {
        1.f, 0.f, 0.f, 0.f,   //
        0.f, 1.f, 0.f, 0.f,   //
        0.f, 0.f, 1.f, 0.f,   //
        0.f, 0.f, 0.f, 1.f,   //
        1.f, 1.f, 1.f, 1.f,   //
    };
    const std::vector<float> expected =
        test_utils::linear_ref(x_data, w_data, x_shape[0], x_shape[1], w_shape[0]);

    auto x = ctx.parameter("x", ov::element::f32, x_shape);

    ov::Tensor w_tensor(ov::element::f32, w_shape);
    std::memcpy(w_tensor.data(), w_data.data(), w_data.size() * sizeof(float));
    auto w = ov::genai::modeling::ops::constant(w_tensor, &ctx.op_context());

    ov::genai::modeling::LMHead lm_head(w);
    auto logits = lm_head.forward(x);

    auto model = ctx.build_model({logits.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    ov::Tensor input_tensor(ov::element::f32, x_shape);
    std::memcpy(input_tensor.data(), x_data.data(), x_data.size() * sizeof(float));
    request.set_input_tensor(input_tensor);
    request.infer();

    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_default);
}

TEST(LMHeadLayer, PrefillLastToken) {
    ov::genai::modeling::BuilderContext ctx;

    // Two sequences with lengths [2, 3] -> cu_seqlens_q = [0, 2, 5]
    const std::vector<int64_t> cu_seqlens = {0, 2, 5};
    const ov::Shape cu_shape{cu_seqlens.size()};

    const ov::Shape x_shape{5, 4};  // total_tokens=5, hidden=4
    const ov::Shape w_shape{5, 4};

    const std::vector<float> x_data = {
        1.f, 2.f, 3.f, 4.f,     // token 0
        5.f, 6.f, 7.f, 8.f,     // token 1 (seq0 last)
        9.f, 10.f, 11.f, 12.f,  // token 2
        13.f, 14.f, 15.f, 16.f, // token 3
        17.f, 18.f, 19.f, 20.f, // token 4 (seq1 last)
    };
    const std::vector<float> w_data = {
        1.f, 0.f, 0.f, 0.f,   //
        0.f, 1.f, 0.f, 0.f,   //
        0.f, 0.f, 1.f, 0.f,   //
        0.f, 0.f, 0.f, 1.f,   //
        1.f, 1.f, 1.f, 1.f,   //
    };

    const std::vector<float> x_last = {
        5.f, 6.f, 7.f, 8.f,      //
        17.f, 18.f, 19.f, 20.f,  //
    };
    const std::vector<float> expected = test_utils::linear_ref(x_last, w_data, 2, x_shape[1], w_shape[0]);

    auto x = ctx.parameter("x", ov::element::f32, x_shape);
    auto cu = ctx.parameter("cu", ov::element::i64, cu_shape);

    ov::Tensor w_tensor(ov::element::f32, w_shape);
    std::memcpy(w_tensor.data(), w_data.data(), w_data.size() * sizeof(float));
    auto w = ov::genai::modeling::ops::constant(w_tensor, &ctx.op_context());

    ov::genai::modeling::LMHead lm_head(w);
    auto logits = lm_head.forward(x, cu);

    auto model = ctx.build_model({logits.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    ov::Tensor x_tensor(ov::element::f32, x_shape);
    std::memcpy(x_tensor.data(), x_data.data(), x_data.size() * sizeof(float));
    request.set_input_tensor(0, x_tensor);

    ov::Tensor cu_tensor(ov::element::i64, cu_shape);
    std::memcpy(cu_tensor.data(), cu_seqlens.data(), cu_seqlens.size() * sizeof(int64_t));
    request.set_input_tensor(1, cu_tensor);

    request.infer();

    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_default);
}


