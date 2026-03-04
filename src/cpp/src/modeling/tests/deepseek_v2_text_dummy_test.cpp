// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>

#include "modeling/builder_context.hpp"
#include "modeling/models/deepseek_v2/modeling_deepseek_v2.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace test_utils = ov::genai::modeling::tests;

TEST(DeepseekV2TextDummy, BuildsAndRunsWithMoE) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t batch = 1;
    const size_t seq_len = 2;
    const size_t vocab = 5;
    const size_t hidden = 4;
    const size_t num_heads = 2;
    const size_t num_kv_heads = 1;
    const size_t head_dim = 2;
    const float rope_theta = 10000.0f;
    const size_t moe_inter = 3;
    const size_t num_experts = 2;
    const size_t top_k = 1;
    const size_t shared_experts = 1;
    const size_t shared_inter = moe_inter * shared_experts;
    const size_t num_layers = 1;
    const size_t kv_hidden = num_kv_heads * head_dim;

    const ov::Shape embed_shape{vocab, hidden};
    const ov::Shape norm_shape{hidden};
    const ov::Shape lm_head_shape{vocab, hidden};
    const ov::Shape q_weight_shape{hidden, hidden};
    const ov::Shape kv_weight_shape{kv_hidden, hidden};
    const ov::Shape o_weight_shape{hidden, hidden};
    const ov::Shape gate_shape{num_experts, hidden};
    const ov::Shape expert_up_shape{moe_inter, hidden};
    const ov::Shape expert_down_shape{hidden, moe_inter};
    const ov::Shape shared_up_shape{shared_inter, hidden};
    const ov::Shape shared_down_shape{hidden, shared_inter};

    const std::vector<float> embed_weight = {
        0.f, 1.f, 2.f, 3.f,
        4.f, 5.f, 6.f, 7.f,
        8.f, 9.f, 10.f, 11.f,
        12.f, 13.f, 14.f, 15.f,
        16.f, 17.f, 18.f, 19.f,
    };
    const std::vector<float> input_norm_weight = {1.0f, 0.9f, 1.1f, 0.95f};
    const std::vector<float> post_norm_weight = {1.0f, 1.05f, 0.95f, 1.1f};
    const std::vector<float> norm_weight = {1.f, 1.f, 1.f, 1.f};

    const auto q_w = test_utils::make_seq(hidden * hidden, 0.01f, 0.01f);
    const auto k_w = test_utils::make_seq(kv_hidden * hidden, 0.02f, 0.01f);
    const auto v_w = test_utils::make_seq(kv_hidden * hidden, 0.03f, 0.01f);
    const auto o_w = test_utils::make_seq(hidden * hidden, 0.04f, 0.01f);

    const std::vector<float> zero_qb(hidden, 0.0f);
    const std::vector<float> zero_kb(kv_hidden, 0.0f);
    const std::vector<float> zero_vb(kv_hidden, 0.0f);
    const std::vector<float> zero_ob(hidden, 0.0f);

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

    const std::vector<float> lm_head_weight = {
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 1.f, 0.f,
        0.f, 0.f, 0.f, 1.f,
        1.f, 1.f, 1.f, 1.f,
    };

    ov::genai::modeling::tests::DummyWeightSource weights;
    weights.add("model.embed_tokens.weight", test_utils::make_tensor(embed_weight, embed_shape));
    weights.add("model.layers[0].input_layernorm.weight", test_utils::make_tensor(input_norm_weight, norm_shape));
    weights.add("model.layers[0].self_attn.q_proj.weight", test_utils::make_tensor(q_w, q_weight_shape));
    weights.add("model.layers[0].self_attn.k_proj.weight", test_utils::make_tensor(k_w, kv_weight_shape));
    weights.add("model.layers[0].self_attn.v_proj.weight", test_utils::make_tensor(v_w, kv_weight_shape));
    weights.add("model.layers[0].self_attn.o_proj.weight", test_utils::make_tensor(o_w, o_weight_shape));
    weights.add("model.layers[0].post_attention_layernorm.weight", test_utils::make_tensor(post_norm_weight, norm_shape));
    weights.add("model.layers[0].mlp.gate.weight", test_utils::make_tensor(gate_w, gate_shape));
    weights.add("model.layers[0].mlp.experts.0.gate_proj.weight", test_utils::make_tensor(gate_w0, expert_up_shape));
    weights.add("model.layers[0].mlp.experts.0.up_proj.weight", test_utils::make_tensor(up_w0, expert_up_shape));
    weights.add("model.layers[0].mlp.experts.0.down_proj.weight", test_utils::make_tensor(down_w0, expert_down_shape));
    weights.add("model.layers[0].mlp.experts.1.gate_proj.weight", test_utils::make_tensor(gate_w1, expert_up_shape));
    weights.add("model.layers[0].mlp.experts.1.up_proj.weight", test_utils::make_tensor(up_w1, expert_up_shape));
    weights.add("model.layers[0].mlp.experts.1.down_proj.weight", test_utils::make_tensor(down_w1, expert_down_shape));
    weights.add("model.layers[0].mlp.shared_experts.gate_proj.weight", test_utils::make_tensor(shared_gate_w, shared_up_shape));
    weights.add("model.layers[0].mlp.shared_experts.up_proj.weight", test_utils::make_tensor(shared_up_w, shared_up_shape));
    weights.add("model.layers[0].mlp.shared_experts.down_proj.weight", test_utils::make_tensor(shared_down_w, shared_down_shape));
    weights.add("model.norm.weight", test_utils::make_tensor(norm_weight, norm_shape));
    weights.add("lm_head.weight", test_utils::make_tensor(lm_head_weight, lm_head_shape));

    ov::genai::modeling::tests::DummyWeightFinalizer finalizer;

    ov::genai::modeling::models::DeepseekV2TextConfig cfg;
    cfg.hidden_size = static_cast<int32_t>(hidden);
    cfg.num_attention_heads = static_cast<int32_t>(num_heads);
    cfg.num_key_value_heads = static_cast<int32_t>(num_kv_heads);
    cfg.head_dim = static_cast<int32_t>(head_dim);
    cfg.moe_intermediate_size = static_cast<int32_t>(moe_inter);
    cfg.n_routed_experts = static_cast<int32_t>(num_experts);
    cfg.num_experts_per_tok = static_cast<int32_t>(top_k);
    cfg.n_shared_experts = static_cast<int32_t>(shared_experts);
    cfg.num_hidden_layers = static_cast<int32_t>(num_layers);
    cfg.first_k_dense_replace = 0;
    cfg.moe_layer_freq = 1;
    cfg.rms_norm_eps = 1e-6f;
    cfg.rope_theta = rope_theta;
    cfg.hidden_act = "silu";
    cfg.attention_bias = false;
    cfg.tie_word_embeddings = false;

    ov::genai::modeling::models::DeepseekV2ForCausalLM model(ctx, cfg);
    ov::genai::modeling::weights::load_model(model, weights, finalizer);

    auto input_ids = ctx.parameter("input_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto beam_idx = ctx.parameter("beam_idx", ov::element::i32, ov::PartialShape{-1});

    auto logits = model.forward(input_ids, position_ids, beam_idx);
    auto ov_model = ctx.build_model({logits.output()});

    ov::Core core;
    auto compiled = core.compile_model(ov_model, "GPU");
    auto request = compiled.create_infer_request();

    const std::vector<int64_t> input_ids_data = {0, 3};
    const std::vector<int64_t> position_ids_data = {0, 1};

    ov::Tensor input_ids_tensor(ov::element::i64, {batch, seq_len});
    std::memcpy(input_ids_tensor.data(), input_ids_data.data(), input_ids_data.size() * sizeof(int64_t));
    request.set_input_tensor(0, input_ids_tensor);

    ov::Tensor position_ids_tensor(ov::element::i64, {batch, seq_len});
    std::memcpy(position_ids_tensor.data(), position_ids_data.data(), position_ids_data.size() * sizeof(int64_t));
    request.set_input_tensor(1, position_ids_tensor);

    ov::Tensor beam_tensor(ov::element::i32, {batch});
    std::fill_n(beam_tensor.data<int32_t>(), batch, 0);
    request.set_input_tensor(2, beam_tensor);

    request.infer();

    auto hidden0 = test_utils::embedding_ref(input_ids_data, embed_weight, batch, seq_len, hidden);
    auto normed = test_utils::rms_ref(hidden0, input_norm_weight, batch * seq_len, hidden, cfg.rms_norm_eps);
    auto attn_out = test_utils::attention_ref(normed,
                                              q_w,
                                              zero_qb,
                                              k_w,
                                              zero_kb,
                                              v_w,
                                              zero_vb,
                                              o_w,
                                              zero_ob,
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
    auto sum1 = test_utils::add_ref(attn_out, hidden0);
    auto normed2 = test_utils::rms_ref(sum1, post_norm_weight, batch * seq_len, hidden, cfg.rms_norm_eps);

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

    auto moe_out = test_utils::moe_ref(normed2,
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
    auto shared_out = test_utils::mlp_ref(normed2,
                                          shared_gate_w,
                                          shared_up_w,
                                          shared_down_w,
                                          batch,
                                          seq_len,
                                          hidden,
                                          shared_inter);
    auto moe_total = test_utils::add_ref(moe_out, shared_out);

    auto sum2 = test_utils::add_ref(moe_total, sum1);
    auto final_hidden = test_utils::rms_ref(sum2, norm_weight, batch * seq_len, hidden, cfg.rms_norm_eps);
    auto expected = test_utils::linear_ref_3d(final_hidden, lm_head_weight, batch, seq_len, hidden, vocab);

    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_moe);
}
