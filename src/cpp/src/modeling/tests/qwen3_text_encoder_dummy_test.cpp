// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/models/qwen3/modeling_qwen3.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace test_utils = ov::genai::modeling::tests;

namespace {

void add_zero_weight(test_utils::DummyWeightSource& source,
                     const std::string& name,
                     const ov::Shape& shape) {
    ov::Tensor tensor(ov::element::f32, shape);
    std::memset(tensor.data(), 0, tensor.get_byte_size());
    source.add(name, tensor);
}

}  // namespace

TEST(Qwen3TextEncoder, BuildsAndRuns) {
    const size_t batch = 1;
    const size_t seq_len = 3;
    const size_t vocab = 8;
    const size_t hidden = 4;
    const size_t num_heads = 1;
    const size_t num_kv_heads = 1;
    const size_t head_dim = 4;
    const size_t intermediate = 4;
    const size_t num_layers = 1;
    const size_t kv_hidden = num_kv_heads * head_dim;

    ov::genai::modeling::tests::DummyWeightSource weights;
    add_zero_weight(weights, "model.embed_tokens.weight", {vocab, hidden});
    add_zero_weight(weights, "model.layers[0].input_layernorm.weight", {hidden});
    add_zero_weight(weights, "model.layers[0].self_attn.q_proj.weight", {hidden, hidden});
    add_zero_weight(weights, "model.layers[0].self_attn.q_proj.bias", {hidden});
    add_zero_weight(weights, "model.layers[0].self_attn.k_proj.weight", {kv_hidden, hidden});
    add_zero_weight(weights, "model.layers[0].self_attn.k_proj.bias", {kv_hidden});
    add_zero_weight(weights, "model.layers[0].self_attn.v_proj.weight", {kv_hidden, hidden});
    add_zero_weight(weights, "model.layers[0].self_attn.v_proj.bias", {kv_hidden});
    add_zero_weight(weights, "model.layers[0].self_attn.o_proj.weight", {hidden, hidden});
    add_zero_weight(weights, "model.layers[0].self_attn.o_proj.bias", {hidden});
    add_zero_weight(weights, "model.layers[0].post_attention_layernorm.weight", {hidden});
    add_zero_weight(weights, "model.layers[0].mlp.gate_proj.weight", {intermediate, hidden});
    add_zero_weight(weights, "model.layers[0].mlp.up_proj.weight", {intermediate, hidden});
    add_zero_weight(weights, "model.layers[0].mlp.down_proj.weight", {hidden, intermediate});
    add_zero_weight(weights, "model.norm.weight", {hidden});

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
    cfg.rope_theta = 10000.0f;
    cfg.hidden_act = "silu";
    cfg.attention_bias = true;
    cfg.tie_word_embeddings = false;

    auto model = ov::genai::modeling::models::create_qwen3_text_encoder_model(cfg, weights, finalizer);

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    std::vector<int64_t> input_ids_data = {0, 1, 2};
    std::vector<int64_t> position_ids_data = {0, 1, 2};
    std::vector<int64_t> attention_mask_data = {1, 1, 1};

    ov::Tensor input_ids_tensor(ov::element::i64, {batch, seq_len});
    std::memcpy(input_ids_tensor.data(), input_ids_data.data(), input_ids_data.size() * sizeof(int64_t));
    request.set_input_tensor(0, input_ids_tensor);

    ov::Tensor attention_mask_tensor(ov::element::i64, {batch, seq_len});
    std::memcpy(attention_mask_tensor.data(), attention_mask_data.data(), attention_mask_data.size() * sizeof(int64_t));
    request.set_input_tensor(1, attention_mask_tensor);

    ov::Tensor position_ids_tensor(ov::element::i64, {batch, seq_len});
    std::memcpy(position_ids_tensor.data(), position_ids_data.data(), position_ids_data.size() * sizeof(int64_t));
    request.set_input_tensor(2, position_ids_tensor);

    request.infer();

    std::vector<float> expected(batch * seq_len * hidden, 0.0f);
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_default);
}
