// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <string>

#include <openvino/op/read_value.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/builder_context.hpp"
#include "modeling/models/qwen3_next/modeling_qwen3_next.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace test_utils = ov::genai::modeling::tests;

TEST(Qwen3NextAttention, BuildsGraphWithPartialRopeAndGate) {
    ov::genai::modeling::BuilderContext ctx;

    ov::genai::modeling::models::Qwen3NextConfig cfg;
    cfg.hidden_size = 16;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 1;
    cfg.head_dim = 8;
    cfg.partial_rotary_factor = 0.5f;
    cfg.rms_norm_eps = 1e-6f;
    cfg.rope_theta = 10000.0f;

    ov::genai::modeling::models::Qwen3NextAttention attn(ctx, "self_attn", cfg);

    test_utils::DummyWeightSource weights;
    test_utils::DummyWeightFinalizer finalizer;

    weights.add("self_attn.q_proj.weight", test_utils::make_tensor(test_utils::make_seq(32 * 16, 0.01f, 0.001f), {32, 16}));
    weights.add("self_attn.k_proj.weight", test_utils::make_tensor(test_utils::make_seq(8 * 16, 0.02f, 0.001f), {8, 16}));
    weights.add("self_attn.v_proj.weight", test_utils::make_tensor(test_utils::make_seq(8 * 16, 0.03f, 0.001f), {8, 16}));
    weights.add("self_attn.o_proj.weight", test_utils::make_tensor(test_utils::make_seq(16 * 16, 0.04f, 0.001f), {16, 16}));
    weights.add("self_attn.q_norm.weight", test_utils::make_tensor(std::vector<float>(8, 0.0f), {8}));
    weights.add("self_attn.k_norm.weight", test_utils::make_tensor(std::vector<float>(8, 0.0f), {8}));

    ov::genai::modeling::weights::load_model(attn, weights, finalizer);

    auto hidden_states = ctx.parameter("hidden_states", ov::element::f32, ov::PartialShape{1, 2, 16});
    auto beam_idx = ctx.parameter("beam_idx", ov::element::i32, ov::PartialShape{1});
    auto rope_cos = ctx.parameter("rope_cos", ov::element::f32, ov::PartialShape{1, 2, 2});
    auto rope_sin = ctx.parameter("rope_sin", ov::element::f32, ov::PartialShape{1, 2, 2});
    auto attention_mask = ctx.parameter("attention_mask", ov::element::i64, ov::PartialShape{1, 2});

    auto out = attn.forward(hidden_states, beam_idx, rope_cos, rope_sin, &attention_mask);
    auto ov_model = ctx.build_model({out.output()});

    ASSERT_EQ(ov_model->outputs().size(), 1u);
    ASSERT_EQ(ov_model->get_output_partial_shape(0).rank().get_length(), 3);

    size_t read_value_count = 0;
    size_t assign_count = 0;
    size_t sigmoid_count = 0;
    size_t concat_count = 0;
    for (const auto& op : ov_model->get_ops()) {
        if (ov::as_type_ptr<ov::op::v6::ReadValue>(op)) {
            read_value_count++;
        }
        if (ov::as_type_ptr<ov::op::v6::Assign>(op)) {
            assign_count++;
        }
        if (std::string(op->get_type_name()) == "Sigmoid") {
            sigmoid_count++;
        }
        if (std::string(op->get_type_name()) == "Concat") {
            concat_count++;
        }
    }

    EXPECT_GE(read_value_count, 2u);
    EXPECT_GE(assign_count, 2u);
    EXPECT_GE(sigmoid_count, 1u);  // gate sigmoid
    EXPECT_GE(concat_count, 1u);   // partial RoPE path concatenates rotary/pass-through parts
}
