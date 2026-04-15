// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>

#include "modeling/models/deepseek_ocr2/modeling_deepseek_qwen2_vision.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace test_utils = ov::genai::modeling::tests;

TEST(DeepseekQwen2MaskTest, BasicPattern) {
    ov::genai::modeling::BuilderContext ctx;
    auto token_type_ids = ctx.parameter("token_type_ids", ov::element::i64, ov::PartialShape{1, 4});

    auto mask = ov::genai::modeling::models::build_qwen2_attention_mask(token_type_ids);
    auto model = ctx.build_model({mask.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    ov::Tensor token_tensor(ov::element::i64, {1, 4});
    auto* ptr = token_tensor.data<int64_t>();
    ptr[0] = 0;
    ptr[1] = 0;
    ptr[2] = 1;
    ptr[3] = 1;

    request.set_input_tensor(0, token_tensor);
    request.infer();

    const float neg = -65504.0f;
    std::vector<float> expected = {
        0.0f, 0.0f, neg, neg,
        0.0f, 0.0f, neg, neg,
        0.0f, 0.0f, 0.0f, neg,
        0.0f, 0.0f, 0.0f, 0.0f
    };
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_default);
}

TEST(DeepseekQwen2DecoderLayerTest, IdentityWithZeroWeights) {
    ov::genai::modeling::BuilderContext ctx;

    ov::genai::modeling::models::DeepseekQwen2VisionConfig cfg;
    cfg.hidden_size = 4;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 2;
    cfg.intermediate_size = 8;
    cfg.attention_bias = true;

    ov::genai::modeling::models::DeepseekQwen2DecoderLayer layer(ctx, "layer", cfg);

    test_utils::DummyWeightSource weights;
    weights.add("layer.input_layernorm.weight", test_utils::make_tensor({1, 1, 1, 1}, {4}));
    weights.add("layer.post_attention_layernorm.weight", test_utils::make_tensor({1, 1, 1, 1}, {4}));

    weights.add("layer.self_attn.q_proj.weight", test_utils::make_tensor(std::vector<float>(4 * 4, 0.0f), {4, 4}));
    weights.add("layer.self_attn.q_proj.bias", test_utils::make_tensor(std::vector<float>(4, 0.0f), {4}));
    weights.add("layer.self_attn.k_proj.weight", test_utils::make_tensor(std::vector<float>(4 * 4, 0.0f), {4, 4}));
    weights.add("layer.self_attn.k_proj.bias", test_utils::make_tensor(std::vector<float>(4, 0.0f), {4}));
    weights.add("layer.self_attn.v_proj.weight", test_utils::make_tensor(std::vector<float>(4 * 4, 0.0f), {4, 4}));
    weights.add("layer.self_attn.v_proj.bias", test_utils::make_tensor(std::vector<float>(4, 0.0f), {4}));
    weights.add("layer.self_attn.o_proj.weight", test_utils::make_tensor(std::vector<float>(4 * 4, 0.0f), {4, 4}));
    weights.add("layer.self_attn.o_proj.bias", test_utils::make_tensor(std::vector<float>(4, 0.0f), {4}));

    weights.add("layer.mlp.gate_proj.weight", test_utils::make_tensor(std::vector<float>(8 * 4, 0.0f), {8, 4}));
    weights.add("layer.mlp.up_proj.weight", test_utils::make_tensor(std::vector<float>(8 * 4, 0.0f), {8, 4}));
    weights.add("layer.mlp.down_proj.weight", test_utils::make_tensor(std::vector<float>(4 * 8, 0.0f), {4, 8}));

    test_utils::DummyWeightFinalizer finalizer;
    ov::genai::modeling::weights::load_model(layer, weights, finalizer);

    auto hidden_states = ctx.parameter("hidden_states", ov::element::f32, ov::PartialShape{1, 4, 4});
    auto token_type_ids = ctx.parameter("token_type_ids", ov::element::i64, ov::PartialShape{1, 4});
    auto rope_cos = ctx.parameter("rope_cos", ov::element::f32, ov::PartialShape{1, 4, 1});
    auto rope_sin = ctx.parameter("rope_sin", ov::element::f32, ov::PartialShape{1, 4, 1});

    auto mask = ov::genai::modeling::models::build_qwen2_attention_mask(token_type_ids);
    auto output = layer.forward(hidden_states, rope_cos, rope_sin, mask);
    auto model = ctx.build_model({output.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    std::vector<float> hidden_data(1 * 4 * 4);
    for (size_t i = 0; i < hidden_data.size(); ++i) {
        hidden_data[i] = static_cast<float>(i) * 0.01f;
    }
    ov::Tensor hidden_tensor(ov::element::f32, {1, 4, 4});
    std::memcpy(hidden_tensor.data(), hidden_data.data(), hidden_data.size() * sizeof(float));

    ov::Tensor token_tensor(ov::element::i64, {1, 4});
    auto* tptr = token_tensor.data<int64_t>();
    tptr[0] = 0;
    tptr[1] = 0;
    tptr[2] = 1;
    tptr[3] = 1;

    ov::Tensor cos_tensor(ov::element::f32, {1, 4, 1});
    ov::Tensor sin_tensor(ov::element::f32, {1, 4, 1});
    std::fill(cos_tensor.data<float>(), cos_tensor.data<float>() + cos_tensor.get_size(), 1.0f);
    std::fill(sin_tensor.data<float>(), sin_tensor.data<float>() + sin_tensor.get_size(), 0.0f);

    request.set_input_tensor(0, hidden_tensor);
    request.set_input_tensor(1, token_tensor);
    request.set_input_tensor(2, cos_tensor);
    request.set_input_tensor(3, sin_tensor);
    request.infer();

    test_utils::expect_tensor_near(request.get_output_tensor(), hidden_data, test_utils::k_tol_default);
}
