// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>
#include "modeling/builder_context.hpp"
#include "modeling/models/qwen3/modeling_qwen3.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace test_utils = ov::genai::modeling::tests;

TEST(Qwen3MLP, MatchesReference) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t batch = 1;
    const size_t seq_len = 2;
    const size_t hidden = 3;
    const size_t intermediate = 4;

    const ov::Shape mlp_up_shape{intermediate, hidden};
    const ov::Shape mlp_down_shape{hidden, intermediate};

    const auto gate_w = test_utils::make_seq(intermediate * hidden, 0.01f, 0.02f);
    const auto up_w = test_utils::make_seq(intermediate * hidden, 0.015f, 0.02f);
    const auto down_w = test_utils::make_seq(hidden * intermediate, 0.02f, 0.02f);

    ov::genai::modeling::tests::DummyWeightSource weights;
    weights.add("mlp.gate_proj.weight", test_utils::make_tensor(gate_w, mlp_up_shape));
    weights.add("mlp.up_proj.weight", test_utils::make_tensor(up_w, mlp_up_shape));
    weights.add("mlp.down_proj.weight", test_utils::make_tensor(down_w, mlp_down_shape));

    ov::genai::modeling::tests::DummyWeightFinalizer finalizer;

    ov::genai::modeling::models::Qwen3DenseConfig cfg;
    cfg.hidden_size = static_cast<int32_t>(hidden);
    cfg.intermediate_size = static_cast<int32_t>(intermediate);
    cfg.num_hidden_layers = 1;
    cfg.hidden_act = "silu";

    ov::genai::modeling::models::Qwen3MLP mlp(ctx, "mlp", cfg);
    ov::genai::modeling::weights::load_model(mlp, weights, finalizer);

    auto input = ctx.parameter("input", ov::element::f32, ov::PartialShape{batch, seq_len, hidden});
    auto output = mlp.forward(input);
    auto ov_model = ctx.build_model({output.output()});

    ov::Core core;
    auto compiled = core.compile_model(ov_model, "GPU");
    auto request = compiled.create_infer_request();

    const std::vector<float> input_data = {
        0.1f, 0.2f, 0.3f,
        0.4f, 0.5f, 0.6f,
    };
    ov::Tensor input_tensor(ov::element::f32, {batch, seq_len, hidden});
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));
    request.set_input_tensor(input_tensor);

    request.infer();

    auto expected = test_utils::mlp_ref(input_data, gate_w, up_w, down_w, batch, seq_len, hidden, intermediate);
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_default);
}
