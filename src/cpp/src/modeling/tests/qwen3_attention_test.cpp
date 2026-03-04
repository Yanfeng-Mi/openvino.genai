// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
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

TEST(Qwen3Attention, MatchesReference) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t batch = 1;
    const size_t seq_len = 2;
    const size_t hidden = 64;
    const size_t num_heads = 4;
    const size_t num_kv_heads = 1;
    const size_t head_dim = 16;
    const float rope_theta = 10000.0f;
    const size_t kv_hidden = num_kv_heads * head_dim;

    const ov::Shape q_weight_shape{hidden, hidden};
    const ov::Shape kv_weight_shape{kv_hidden, hidden};
    const ov::Shape o_weight_shape{hidden, hidden};
    const ov::Shape q_bias_shape{hidden};
    const ov::Shape kv_bias_shape{kv_hidden};
    const ov::Shape o_bias_shape{hidden};
    const ov::Shape qk_norm_shape{head_dim};

    const auto q_w = test_utils::make_seq(hidden * hidden, 0.01f, 0.01f);
    const auto k_w = test_utils::make_seq(kv_hidden * hidden, 0.02f, 0.01f);
    const auto v_w = test_utils::make_seq(kv_hidden * hidden, 0.03f, 0.01f);
    const auto o_w = test_utils::make_seq(hidden * hidden, 0.04f, 0.01f);
    const auto q_b = test_utils::make_seq(hidden, 0.05f, 0.01f);
    const auto k_b = test_utils::make_seq(kv_hidden, -0.02f, 0.01f);
    const auto v_b = test_utils::make_seq(kv_hidden, 0.03f, 0.005f);
    const auto o_b = test_utils::make_seq(hidden, -0.01f, 0.02f);
    const auto q_norm_w = test_utils::make_seq(head_dim, 1.0f, 0.02f);
    const auto k_norm_w = test_utils::make_seq(head_dim, 0.9f, 0.03f);

    ov::genai::modeling::tests::DummyWeightSource weights;
    weights.add("self_attn.q_proj.weight", test_utils::make_tensor(q_w, q_weight_shape));
    weights.add("self_attn.q_proj.bias", test_utils::make_tensor(q_b, q_bias_shape));
    weights.add("self_attn.k_proj.weight", test_utils::make_tensor(k_w, kv_weight_shape));
    weights.add("self_attn.k_proj.bias", test_utils::make_tensor(k_b, kv_bias_shape));
    weights.add("self_attn.v_proj.weight", test_utils::make_tensor(v_w, kv_weight_shape));
    weights.add("self_attn.v_proj.bias", test_utils::make_tensor(v_b, kv_bias_shape));
    weights.add("self_attn.o_proj.weight", test_utils::make_tensor(o_w, o_weight_shape));
    weights.add("self_attn.o_proj.bias", test_utils::make_tensor(o_b, o_bias_shape));
    weights.add("self_attn.q_norm.weight", test_utils::make_tensor(q_norm_w, qk_norm_shape));
    weights.add("self_attn.k_norm.weight", test_utils::make_tensor(k_norm_w, qk_norm_shape));

    ov::genai::modeling::tests::DummyWeightFinalizer finalizer;

    ov::genai::modeling::models::Qwen3DenseConfig cfg;
    cfg.hidden_size = static_cast<int32_t>(hidden);
    cfg.num_attention_heads = static_cast<int32_t>(num_heads);
    cfg.num_key_value_heads = static_cast<int32_t>(num_kv_heads);
    cfg.head_dim = static_cast<int32_t>(head_dim);
    cfg.rope_theta = rope_theta;
    // attention_bias toggles qk_norm in this impl; keep biases to cover add-bias path.
    cfg.attention_bias = false;

    ov::genai::modeling::models::Qwen3Attention attn(ctx, "self_attn", cfg);
    ov::genai::modeling::weights::load_model(attn, weights, finalizer);

    auto hidden_states = ctx.parameter("hidden_states", ov::element::f32, ov::PartialShape{batch, seq_len, hidden});
    auto positions = ctx.parameter("positions", ov::element::i64, ov::PartialShape{batch, seq_len});
    auto beam_idx = ctx.parameter("beam_idx", ov::element::i32, ov::PartialShape{batch});

    auto output = attn.forward(positions, hidden_states, beam_idx);
    auto ov_model = ctx.build_model({output.output()});

    ov::Core core;
    auto compiled = core.compile_model(ov_model, "GPU");
    auto request = compiled.create_infer_request();

    const std::vector<float> hidden_data = test_utils::make_seq(batch * seq_len * hidden, 0.01f, 0.01f);
    const std::vector<int64_t> position_ids = {0, 1};

    ov::Tensor hidden_tensor(ov::element::f32, {batch, seq_len, hidden});
    std::memcpy(hidden_tensor.data(), hidden_data.data(), hidden_data.size() * sizeof(float));
    request.set_input_tensor(0, hidden_tensor);

    ov::Tensor pos_tensor(ov::element::i64, {batch, seq_len});
    std::memcpy(pos_tensor.data(), position_ids.data(), position_ids.size() * sizeof(int64_t));
    request.set_input_tensor(1, pos_tensor);

    ov::Tensor beam_tensor(ov::element::i32, {batch});
    std::fill_n(beam_tensor.data<int32_t>(), batch, 0);
    request.set_input_tensor(2, beam_tensor);

    request.infer();

    auto expected = test_utils::attention_ref(hidden_data,
                                              q_w,
                                              q_b,
                                              k_w,
                                              k_b,
                                              v_w,
                                              v_b,
                                              o_w,
                                              o_b,
                                              &q_norm_w,
                                              &k_norm_w,
                                              position_ids,
                                              batch,
                                              seq_len,
                                              hidden,
                                              num_heads,
                                              num_kv_heads,
                                              head_dim,
                                              rope_theta,
                                              cfg.rms_norm_eps);

    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_default);
}
