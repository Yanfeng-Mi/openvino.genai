// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>

#include "modeling/builder_context.hpp"
#include "modeling/models/qwen3_vl/modeling_qwen3_vl_vision.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace test_utils = ov::genai::modeling::tests;

TEST(Qwen3VLVisionPatchEmbedTest, Conv3dPatchEmbed) {
    ov::genai::modeling::BuilderContext ctx;

    ov::genai::modeling::models::Qwen3VLVisionConfig cfg;
    cfg.hidden_size = 1;
    cfg.in_channels = 1;
    cfg.patch_size = 2;
    cfg.temporal_patch_size = 2;

    ov::genai::modeling::models::Qwen3VLVisionPatchEmbed embed(ctx, "patch_embed", cfg);

    test_utils::DummyWeightSource weights;
    std::vector<float> weight_data(1 * 1 * 2 * 2 * 2, 1.0f);
    weights.add("patch_embed.proj.weight", test_utils::make_tensor(weight_data, {1, 1, 2, 2, 2}));
    weights.add("patch_embed.proj.bias", test_utils::make_tensor({0.0f}, {1}));

    test_utils::DummyWeightFinalizer finalizer;
    ov::genai::modeling::weights::load_model(embed, weights, finalizer);

    auto pixel_values = ctx.parameter("pixel_values", ov::element::f32, ov::PartialShape{1, 8});
    auto output = embed.forward(pixel_values);
    auto model = ctx.build_model({output.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    std::vector<float> input_data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    ov::Tensor input_tensor(ov::element::f32, {1, 8});
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));
    request.set_input_tensor(0, input_tensor);
    request.infer();

    std::vector<float> expected{36.0f};
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_default);
}

TEST(Qwen3VLVisionAttentionTest, ZeroWeightsProduceZeroOutput) {
    ov::genai::modeling::BuilderContext ctx;

    ov::genai::modeling::models::Qwen3VLVisionConfig cfg;
    cfg.hidden_size = 4;
    cfg.num_heads = 2;

    ov::genai::modeling::models::Qwen3VLVisionAttention attn(ctx, "attn", cfg);

    test_utils::DummyWeightSource weights;
    weights.add("attn.qkv.weight", test_utils::make_tensor(std::vector<float>(12 * 4, 0.0f), {12, 4}));
    weights.add("attn.qkv.bias", test_utils::make_tensor(std::vector<float>(12, 0.0f), {12}));
    weights.add("attn.proj.weight", test_utils::make_tensor(std::vector<float>(4 * 4, 0.0f), {4, 4}));
    weights.add("attn.proj.bias", test_utils::make_tensor(std::vector<float>(4, 0.0f), {4}));

    test_utils::DummyWeightFinalizer finalizer;
    ov::genai::modeling::weights::load_model(attn, weights, finalizer);

    auto hidden_states = ctx.parameter("hidden_states", ov::element::f32, ov::PartialShape{2, 4});
    auto rotary_cos = ctx.parameter("rotary_cos", ov::element::f32, ov::PartialShape{2, 2});
    auto rotary_sin = ctx.parameter("rotary_sin", ov::element::f32, ov::PartialShape{2, 2});

    auto output = attn.forward(hidden_states, rotary_cos, rotary_sin);
    auto model = ctx.build_model({output.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    std::vector<float> hidden_data{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    ov::Tensor hidden_tensor(ov::element::f32, {2, 4});
    std::memcpy(hidden_tensor.data(), hidden_data.data(), hidden_data.size() * sizeof(float));

    std::vector<float> cos_data{1.0f, 1.0f, 1.0f, 1.0f};
    ov::Tensor cos_tensor(ov::element::f32, {2, 2});
    std::memcpy(cos_tensor.data(), cos_data.data(), cos_data.size() * sizeof(float));

    std::vector<float> sin_data{0.0f, 0.0f, 0.0f, 0.0f};
    ov::Tensor sin_tensor(ov::element::f32, {2, 2});
    std::memcpy(sin_tensor.data(), sin_data.data(), sin_data.size() * sizeof(float));

    request.set_input_tensor(0, hidden_tensor);
    request.set_input_tensor(1, cos_tensor);
    request.set_input_tensor(2, sin_tensor);
    request.infer();

    std::vector<float> expected(8, 0.0f);
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_default);
}
