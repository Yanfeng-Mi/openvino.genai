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

TEST(Qwen3DenseDummy, BuildsAndRuns) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t batch = 1;
    const size_t seq_len = 3;
    const size_t vocab = 6;
    const size_t hidden = 4;
    const size_t num_heads = 2;
    const size_t num_kv_heads = 1;
    const size_t head_dim = 2;
    const float rope_theta = 10000.0f;
    const size_t intermediate = 6;
    const size_t num_layers = 1;
    const size_t kv_hidden = num_kv_heads * head_dim;

    const ov::Shape embed_shape{vocab, hidden};
    const ov::Shape norm_shape{hidden};
    const ov::Shape lm_head_shape{vocab, hidden};
    const ov::Shape mlp_up_shape{intermediate, hidden};
    const ov::Shape mlp_down_shape{hidden, intermediate};
    const ov::Shape q_weight_shape{hidden, hidden};
    const ov::Shape kv_weight_shape{kv_hidden, hidden};
    const ov::Shape o_weight_shape{hidden, hidden};
    const ov::Shape q_bias_shape{hidden};
    const ov::Shape kv_bias_shape{kv_hidden};
    const ov::Shape o_bias_shape{hidden};

    const std::vector<float> embed_weight = {
        0.f, 1.f, 2.f, 3.f,      //
        10.f, 11.f, 12.f, 13.f,  //
        20.f, 21.f, 22.f, 23.f,  //
        30.f, 31.f, 32.f, 33.f,  //
        40.f, 41.f, 42.f, 43.f,  //
        50.f, 51.f, 52.f, 53.f,  //
    };
    const std::vector<float> input_norm_weight0 = {1.0f, 0.8f, 1.2f, 0.9f};
    const std::vector<float> post_norm_weight0 = {1.0f, 0.9f, 1.1f, 1.05f};
    const std::vector<float> norm_weight = {1.f, 1.f, 1.f, 1.f};
    const std::vector<float> lm_head_weight = {
        1.f, 0.f, 0.f, 0.f,    //
        0.f, 1.f, 0.f, 0.f,    //
        0.f, 0.f, 1.f, 0.f,    //
        0.f, 0.f, 0.f, 1.f,    //
        1.f, 1.f, 1.f, 1.f,    //
        -1.f, -1.f, -1.f, -1.f //
    };
    const auto q_w0 = test_utils::make_seq(hidden * hidden, 0.01f, 0.01f);
    const auto k_w0 = test_utils::make_seq(kv_hidden * hidden, 0.02f, 0.01f);
    const auto v_w0 = test_utils::make_seq(kv_hidden * hidden, 0.03f, 0.01f);
    const auto o_w0 = test_utils::make_seq(hidden * hidden, 0.04f, 0.01f);
    const auto q_b0 = test_utils::make_seq(hidden, 0.05f, 0.01f);
    const auto k_b0 = test_utils::make_seq(kv_hidden, -0.02f, 0.01f);
    const auto v_b0 = test_utils::make_seq(kv_hidden, 0.03f, 0.005f);
    const auto o_b0 = test_utils::make_seq(hidden, -0.01f, 0.02f);
    const auto gate_w0 = test_utils::make_seq(intermediate * hidden, 0.05f, 0.01f);
    const auto up_w0 = test_utils::make_seq(intermediate * hidden, 0.06f, 0.01f);
    const auto down_w0 = test_utils::make_seq(hidden * intermediate, 0.07f, 0.01f);

    ov::genai::modeling::tests::DummyWeightSource weights;
    weights.add("model.embed_tokens.weight", test_utils::make_tensor(embed_weight, embed_shape));
    weights.add("model.layers[0].input_layernorm.weight", test_utils::make_tensor(input_norm_weight0, norm_shape));
    weights.add("model.layers[0].self_attn.q_proj.weight", test_utils::make_tensor(q_w0, q_weight_shape));
    weights.add("model.layers[0].self_attn.q_proj.bias", test_utils::make_tensor(q_b0, q_bias_shape));
    weights.add("model.layers[0].self_attn.k_proj.weight", test_utils::make_tensor(k_w0, kv_weight_shape));
    weights.add("model.layers[0].self_attn.k_proj.bias", test_utils::make_tensor(k_b0, kv_bias_shape));
    weights.add("model.layers[0].self_attn.v_proj.weight", test_utils::make_tensor(v_w0, kv_weight_shape));
    weights.add("model.layers[0].self_attn.v_proj.bias", test_utils::make_tensor(v_b0, kv_bias_shape));
    weights.add("model.layers[0].self_attn.o_proj.weight", test_utils::make_tensor(o_w0, o_weight_shape));
    weights.add("model.layers[0].self_attn.o_proj.bias", test_utils::make_tensor(o_b0, o_bias_shape));
    weights.add("model.layers[0].post_attention_layernorm.weight", test_utils::make_tensor(post_norm_weight0, norm_shape));
    weights.add("model.layers[0].mlp.gate_proj.weight", test_utils::make_tensor(gate_w0, mlp_up_shape));
    weights.add("model.layers[0].mlp.up_proj.weight", test_utils::make_tensor(up_w0, mlp_up_shape));
    weights.add("model.layers[0].mlp.down_proj.weight", test_utils::make_tensor(down_w0, mlp_down_shape));
    weights.add("model.norm.weight", test_utils::make_tensor(norm_weight, norm_shape));
    weights.add("lm_head.weight", test_utils::make_tensor(lm_head_weight, lm_head_shape));

    ov::genai::modeling::tests::DummyWeightFinalizer finalizer;

    ov::genai::modeling::models::Qwen3DenseConfig cfg;
    cfg.architecture = "qwen3";
    cfg.hidden_size = static_cast<int32_t>(hidden);
    cfg.num_attention_heads = static_cast<int32_t>(num_heads);
    cfg.num_key_value_heads = static_cast<int32_t>(num_kv_heads);
    cfg.head_dim = static_cast<int32_t>(head_dim);
    cfg.intermediate_size = static_cast<int32_t>(intermediate);
    cfg.num_hidden_layers = static_cast<int32_t>(num_layers);
    cfg.rms_norm_eps = 1e-6f;
    cfg.rope_theta = rope_theta;
    cfg.hidden_act = "silu";
    cfg.attention_bias = true;
    cfg.tie_word_embeddings = false;

    ov::genai::modeling::models::Qwen3ForCausalLM model(ctx, cfg);
    ov::genai::modeling::weights::load_model(model, weights, finalizer);

    auto input_ids = ctx.parameter("input_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto attention_mask = ctx.parameter("attention_mask", ov::element::i64, ov::PartialShape{-1, -1});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto beam_idx = ctx.parameter("beam_idx", ov::element::i32, ov::PartialShape{-1});

    auto logits = model.forward(input_ids, position_ids, beam_idx, attention_mask);
    auto ov_model = ctx.build_model({logits.output()});
    ov::serialize(ov_model, "qwen3_dummy_original.xml");

    ov::Core core;
    auto compiled = core.compile_model(ov_model, "GPU");
    auto request = compiled.create_infer_request();
    ov::serialize(compiled.get_runtime_model(), "qwen3_dummy_compiled.xml");

    const std::vector<int64_t> input_ids_data = {0, 2, 5};
    const std::vector<int64_t> position_ids_data = {0, 1, 2};

    ov::Tensor input_ids_tensor(ov::element::i64, {batch, seq_len});
    std::memcpy(input_ids_tensor.data(), input_ids_data.data(), input_ids_data.size() * sizeof(int64_t));
    request.set_input_tensor(0, input_ids_tensor);

    ov::Tensor attention_mask_tensor(ov::element::i64, {batch, seq_len});
    std::fill_n(attention_mask_tensor.data<int64_t>(), batch * seq_len, 1);
    request.set_input_tensor(1, attention_mask_tensor);

    ov::Tensor position_ids_tensor(ov::element::i64, {batch, seq_len});
    std::memcpy(position_ids_tensor.data(), position_ids_data.data(), position_ids_data.size() * sizeof(int64_t));
    request.set_input_tensor(2, position_ids_tensor);

    ov::Tensor beam_tensor(ov::element::i32, {batch});
    std::fill_n(beam_tensor.data<int32_t>(), batch, 0);
    request.set_input_tensor(3, beam_tensor);

    request.infer();

    // Reference: embedding -> input rmsnorm -> attention -> post rmsnorm -> mlp -> final rmsnorm -> linear
    auto hidden0 = test_utils::embedding_ref(input_ids_data, embed_weight, batch, seq_len, hidden);
    std::vector<float> residual;

    auto normed = test_utils::rms_ref(hidden0, input_norm_weight0, batch * seq_len, hidden, cfg.rms_norm_eps);
    residual = hidden0;
    auto attn_out = test_utils::attention_ref(normed,
                                              q_w0,
                                              q_b0,
                                              k_w0,
                                              k_b0,
                                              v_w0,
                                              v_b0,
                                              o_w0,
                                              o_b0,
                                              nullptr,
                                              nullptr,
                                              position_ids_data,
                                              batch,
                                              seq_len,
                                              hidden,
                                              num_heads,
                                              num_kv_heads,
                                              head_dim,
                                              rope_theta,
                                              cfg.rms_norm_eps);
    auto sum = test_utils::add_ref(attn_out, residual);
    normed = test_utils::rms_ref(sum, post_norm_weight0, batch * seq_len, hidden, cfg.rms_norm_eps);
    residual = sum;

    hidden0 = test_utils::mlp_ref(normed, gate_w0, up_w0, down_w0, batch, seq_len, hidden, intermediate);

    sum = test_utils::add_ref(hidden0, residual);
    hidden0 = test_utils::rms_ref(sum, norm_weight, batch * seq_len, hidden, cfg.rms_norm_eps);
    auto expected = test_utils::linear_ref_3d(hidden0, lm_head_weight, batch, seq_len, hidden, vocab);

    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_default);
}

TEST(Qwen3DenseDummy, TiedWeights) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t batch = 1;
    const size_t seq_len = 2;
    const size_t vocab = 4;
    const size_t hidden = 4;
    const size_t num_heads = 2;
    const size_t num_kv_heads = 2;
    const size_t head_dim = 2;
    const float rope_theta = 10000.0f;
    const size_t intermediate = 4;
    const size_t num_layers = 1;
    const size_t kv_hidden = num_kv_heads * head_dim;

    const ov::Shape embed_shape{vocab, hidden};
    const ov::Shape norm_shape{hidden};
    const ov::Shape mlp_up_shape{intermediate, hidden};
    const ov::Shape mlp_down_shape{hidden, intermediate};
    const ov::Shape q_weight_shape{hidden, hidden};
    const ov::Shape kv_weight_shape{kv_hidden, hidden};
    const ov::Shape o_weight_shape{hidden, hidden};
    const ov::Shape q_bias_shape{hidden};
    const ov::Shape kv_bias_shape{kv_hidden};
    const ov::Shape o_bias_shape{hidden};

    const std::vector<float> embed_weight = {
        0.f, 1.f, 2.f, 3.f,    //
        4.f, 5.f, 6.f, 7.f,    //
        8.f, 9.f, 10.f, 11.f,  //
        12.f, 13.f, 14.f, 15.f //
    };
    const std::vector<float> input_norm_weight0 = {0.9f, 1.1f, 1.0f, 0.95f};
    const std::vector<float> post_norm_weight0 = {0.9f, 1.05f, 1.0f, 0.95f};
    const std::vector<float> norm_weight = {1.f, 1.f, 1.f, 1.f};
    const auto q_w0 = test_utils::make_seq(hidden * hidden, 0.01f, 0.01f);
    const auto k_w0 = test_utils::make_seq(kv_hidden * hidden, 0.02f, 0.01f);
    const auto v_w0 = test_utils::make_seq(kv_hidden * hidden, 0.03f, 0.01f);
    const auto o_w0 = test_utils::make_seq(hidden * hidden, 0.04f, 0.01f);
    const auto q_b0 = test_utils::make_seq(hidden, 0.05f, 0.01f);
    const auto k_b0 = test_utils::make_seq(kv_hidden, -0.02f, 0.01f);
    const auto v_b0 = test_utils::make_seq(kv_hidden, 0.03f, 0.005f);
    const auto o_b0 = test_utils::make_seq(hidden, -0.01f, 0.02f);
    const auto gate_w0 = test_utils::make_seq(intermediate * hidden, 0.05f, 0.02f);
    const auto up_w0 = test_utils::make_seq(intermediate * hidden, 0.06f, 0.02f);
    const auto down_w0 = test_utils::make_seq(hidden * intermediate, 0.07f, 0.02f);

    ov::genai::modeling::tests::DummyWeightSource weights;
    weights.add("model.embed_tokens.weight", test_utils::make_tensor(embed_weight, embed_shape));
    weights.add("model.layers[0].input_layernorm.weight", test_utils::make_tensor(input_norm_weight0, norm_shape));
    weights.add("model.layers[0].self_attn.q_proj.weight", test_utils::make_tensor(q_w0, q_weight_shape));
    weights.add("model.layers[0].self_attn.q_proj.bias", test_utils::make_tensor(q_b0, q_bias_shape));
    weights.add("model.layers[0].self_attn.k_proj.weight", test_utils::make_tensor(k_w0, kv_weight_shape));
    weights.add("model.layers[0].self_attn.k_proj.bias", test_utils::make_tensor(k_b0, kv_bias_shape));
    weights.add("model.layers[0].self_attn.v_proj.weight", test_utils::make_tensor(v_w0, kv_weight_shape));
    weights.add("model.layers[0].self_attn.v_proj.bias", test_utils::make_tensor(v_b0, kv_bias_shape));
    weights.add("model.layers[0].self_attn.o_proj.weight", test_utils::make_tensor(o_w0, o_weight_shape));
    weights.add("model.layers[0].self_attn.o_proj.bias", test_utils::make_tensor(o_b0, o_bias_shape));
    weights.add("model.layers[0].post_attention_layernorm.weight", test_utils::make_tensor(post_norm_weight0, norm_shape));
    weights.add("model.layers[0].mlp.gate_proj.weight", test_utils::make_tensor(gate_w0, mlp_up_shape));
    weights.add("model.layers[0].mlp.up_proj.weight", test_utils::make_tensor(up_w0, mlp_up_shape));
    weights.add("model.layers[0].mlp.down_proj.weight", test_utils::make_tensor(down_w0, mlp_down_shape));
    weights.add("model.norm.weight", test_utils::make_tensor(norm_weight, norm_shape));

    ov::genai::modeling::tests::DummyWeightFinalizer finalizer;

    ov::genai::modeling::models::Qwen3DenseConfig cfg;
    cfg.architecture = "qwen3";
    cfg.hidden_size = static_cast<int32_t>(hidden);
    cfg.num_attention_heads = static_cast<int32_t>(num_heads);
    cfg.num_key_value_heads = static_cast<int32_t>(num_kv_heads);
    cfg.head_dim = static_cast<int32_t>(head_dim);
    cfg.intermediate_size = static_cast<int32_t>(intermediate);
    cfg.num_hidden_layers = static_cast<int32_t>(num_layers);
    cfg.rms_norm_eps = 1e-6f;
    cfg.rope_theta = rope_theta;
    cfg.hidden_act = "silu";
    cfg.attention_bias = true;
    cfg.tie_word_embeddings = true;

    ov::genai::modeling::models::Qwen3ForCausalLM model(ctx, cfg);
    ov::genai::modeling::weights::load_model(model, weights, finalizer);

    auto input_ids = ctx.parameter("input_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto attention_mask = ctx.parameter("attention_mask", ov::element::i64, ov::PartialShape{-1, -1});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto beam_idx = ctx.parameter("beam_idx", ov::element::i32, ov::PartialShape{-1});

    auto logits = model.forward(input_ids, position_ids, beam_idx, attention_mask);
    auto ov_model = ctx.build_model({logits.output()});

    ov::serialize(ov_model, "qwen3_dummy_original.xml");

    ov::Core core;
    auto compiled = core.compile_model(ov_model, "GPU");
    auto request = compiled.create_infer_request();

    ov::serialize(compiled.get_runtime_model(), "qwen3_dummy_compiled.xml");

    const std::vector<int64_t> input_ids_data = {0, 3};
    const std::vector<int64_t> position_ids_data = {0, 1};

    ov::Tensor input_ids_tensor(ov::element::i64, {batch, seq_len});
    std::memcpy(input_ids_tensor.data(), input_ids_data.data(), input_ids_data.size() * sizeof(int64_t));
    request.set_input_tensor(0, input_ids_tensor);

    ov::Tensor attention_mask_tensor(ov::element::i64, {batch, seq_len});
    std::fill_n(attention_mask_tensor.data<int64_t>(), batch * seq_len, 1);
    request.set_input_tensor(1, attention_mask_tensor);

    ov::Tensor position_ids_tensor(ov::element::i64, {batch, seq_len});
    std::memcpy(position_ids_tensor.data(), position_ids_data.data(), position_ids_data.size() * sizeof(int64_t));
    request.set_input_tensor(2, position_ids_tensor);

    ov::Tensor beam_tensor(ov::element::i32, {batch});
    std::fill_n(beam_tensor.data<int32_t>(), batch, 0);
    request.set_input_tensor(3, beam_tensor);

    request.infer();

    auto hidden0 = test_utils::embedding_ref(input_ids_data, embed_weight, batch, seq_len, hidden);
    std::vector<float> residual;

    auto normed = test_utils::rms_ref(hidden0, input_norm_weight0, batch * seq_len, hidden, cfg.rms_norm_eps);
    residual = hidden0;
    auto attn_out = test_utils::attention_ref(normed,
                                              q_w0,
                                              q_b0,
                                              k_w0,
                                              k_b0,
                                              v_w0,
                                              v_b0,
                                              o_w0,
                                              o_b0,
                                              nullptr,
                                              nullptr,
                                              position_ids_data,
                                              batch,
                                              seq_len,
                                              hidden,
                                              num_heads,
                                              num_kv_heads,
                                              head_dim,
                                              rope_theta,
                                              cfg.rms_norm_eps);
    auto sum = test_utils::add_ref(attn_out, residual);
    normed = test_utils::rms_ref(sum, post_norm_weight0, batch * seq_len, hidden, cfg.rms_norm_eps);
    residual = sum;

    hidden0 = test_utils::mlp_ref(normed, gate_w0, up_w0, down_w0, batch, seq_len, hidden, intermediate);

    sum = test_utils::add_ref(hidden0, residual);
    hidden0 = test_utils::rms_ref(sum, norm_weight, batch * seq_len, hidden, cfg.rms_norm_eps);
    auto expected = test_utils::linear_ref_3d(hidden0, embed_weight, batch, seq_len, hidden, vocab);

    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_default);
}
