// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>

#include "modeling/builder_context.hpp"
#include "modeling/models/deepseek_v2/modeling_deepseek_v2.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace test_utils = ov::genai::modeling::tests;

TEST(DeepseekV2MoE, MatchesReference) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t batch = 1;
    const size_t seq_len = 2;
    const size_t hidden = 4;
    const size_t moe_inter = 3;
    const size_t num_experts = 2;
    const size_t top_k = 1;
    const size_t shared_experts = 1;
    const size_t shared_inter = moe_inter * shared_experts;

    const ov::Shape gate_shape{num_experts, hidden};
    const ov::Shape expert_up_shape{moe_inter, hidden};
    const ov::Shape expert_down_shape{hidden, moe_inter};
    const ov::Shape shared_up_shape{shared_inter, hidden};
    const ov::Shape shared_down_shape{hidden, shared_inter};

    const auto gate_w = test_utils::make_seq(num_experts * hidden, 0.02f, 0.01f);

    const auto gate_w0 = test_utils::make_seq(moe_inter * hidden, 0.01f, 0.01f);
    const auto up_w0 = test_utils::make_seq(moe_inter * hidden, 0.02f, 0.01f);
    const auto down_w0 = test_utils::make_seq(hidden * moe_inter, 0.03f, 0.01f);

    const auto gate_w1 = test_utils::make_seq(moe_inter * hidden, -0.01f, 0.02f);
    const auto up_w1 = test_utils::make_seq(moe_inter * hidden, 0.01f, 0.02f);
    const auto down_w1 = test_utils::make_seq(hidden * moe_inter, -0.02f, 0.02f);

    const auto shared_gate_w = test_utils::make_seq(shared_inter * hidden, 0.04f, 0.01f);
    const auto shared_up_w = test_utils::make_seq(shared_inter * hidden, 0.05f, 0.01f);
    const auto shared_down_w = test_utils::make_seq(hidden * shared_inter, 0.06f, 0.01f);

    ov::genai::modeling::tests::DummyWeightSource weights;
    weights.add("mlp.gate.weight", test_utils::make_tensor(gate_w, gate_shape));
    weights.add("mlp.experts.0.gate_proj.weight", test_utils::make_tensor(gate_w0, expert_up_shape));
    weights.add("mlp.experts.0.up_proj.weight", test_utils::make_tensor(up_w0, expert_up_shape));
    weights.add("mlp.experts.0.down_proj.weight", test_utils::make_tensor(down_w0, expert_down_shape));
    weights.add("mlp.experts.1.gate_proj.weight", test_utils::make_tensor(gate_w1, expert_up_shape));
    weights.add("mlp.experts.1.up_proj.weight", test_utils::make_tensor(up_w1, expert_up_shape));
    weights.add("mlp.experts.1.down_proj.weight", test_utils::make_tensor(down_w1, expert_down_shape));
    weights.add("mlp.shared_experts.gate_proj.weight", test_utils::make_tensor(shared_gate_w, shared_up_shape));
    weights.add("mlp.shared_experts.up_proj.weight", test_utils::make_tensor(shared_up_w, shared_up_shape));
    weights.add("mlp.shared_experts.down_proj.weight", test_utils::make_tensor(shared_down_w, shared_down_shape));

    ov::genai::modeling::tests::DummyWeightFinalizer finalizer;

    ov::genai::modeling::models::DeepseekV2TextConfig cfg;
    cfg.hidden_size = static_cast<int32_t>(hidden);
    cfg.moe_intermediate_size = static_cast<int32_t>(moe_inter);
    cfg.n_routed_experts = static_cast<int32_t>(num_experts);
    cfg.num_experts_per_tok = static_cast<int32_t>(top_k);
    cfg.n_shared_experts = static_cast<int32_t>(shared_experts);
    cfg.hidden_act = "silu";

    ov::genai::modeling::models::DeepseekV2MoE moe(ctx, "mlp", cfg);
    ov::genai::modeling::weights::load_model(moe, weights, finalizer);

    auto input = ctx.parameter("input", ov::element::f32, ov::PartialShape{batch, seq_len, hidden});
    auto output = moe.forward(input);
    auto ov_model = ctx.build_model({output.output()});

    ov::Core core;
    auto compiled = core.compile_model(ov_model, "GPU");
    auto request = compiled.create_infer_request();

    const std::vector<float> input_data = {
        0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.6f, 0.7f, 0.8f,
    };
    ov::Tensor input_tensor(ov::element::f32, {batch, seq_len, hidden});
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));
    request.set_input_tensor(input_tensor);

    request.infer();

    std::vector<float> gate_w_all;
    gate_w_all.reserve(num_experts * moe_inter * hidden);
    gate_w_all.insert(gate_w_all.end(), gate_w0.begin(), gate_w0.end());
    gate_w_all.insert(gate_w_all.end(), gate_w1.begin(), gate_w1.end());

    std::vector<float> up_w_all;
    up_w_all.reserve(num_experts * moe_inter * hidden);
    up_w_all.insert(up_w_all.end(), up_w0.begin(), up_w0.end());
    up_w_all.insert(up_w_all.end(), up_w1.begin(), up_w1.end());

    std::vector<float> down_w_all;
    down_w_all.reserve(num_experts * hidden * moe_inter);
    down_w_all.insert(down_w_all.end(), down_w0.begin(), down_w0.end());
    down_w_all.insert(down_w_all.end(), down_w1.begin(), down_w1.end());

    auto routed = test_utils::moe_ref(input_data,
                                      gate_w,
                                      gate_w_all,
                                      up_w_all,
                                      down_w_all,
                                      batch,
                                      seq_len,
                                      hidden,
                                      moe_inter,
                                      num_experts,
                                      top_k);
    auto shared = test_utils::mlp_ref(input_data,
                                      shared_gate_w,
                                      shared_up_w,
                                      shared_down_w,
                                      batch,
                                      seq_len,
                                      hidden,
                                      shared_inter);
    auto expected = test_utils::add_ref(routed, shared);

    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_moe);
}

