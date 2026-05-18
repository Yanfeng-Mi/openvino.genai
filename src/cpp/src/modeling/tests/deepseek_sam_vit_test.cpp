// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>

#include "modeling/builder_context.hpp"
#include "modeling/models/deepseek_ocr2/modeling_deepseek_sam_vit.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace test_utils = ov::genai::modeling::tests;

TEST(DeepseekSamPatchEmbedTest, Conv2dPatchEmbed) {
    ov::genai::modeling::BuilderContext ctx;

    ov::genai::modeling::models::DeepseekSamConfig cfg;
    cfg.in_channels = 1;
    cfg.patch_size = 2;
    cfg.embed_dim = 1;

    ov::genai::modeling::models::DeepseekSamPatchEmbed embed(ctx, "patch_embed", cfg);

    test_utils::DummyWeightSource weights;
    weights.add("patch_embed.proj.weight", test_utils::make_tensor({1, 1, 1, 1}, {1, 1, 2, 2}));
    weights.add("patch_embed.proj.bias", test_utils::make_tensor({0.0f}, {1}));

    test_utils::DummyWeightFinalizer finalizer;
    ov::genai::modeling::weights::load_model(embed, weights, finalizer);

    auto pixel_values = ctx.parameter("pixel_values", ov::element::f32, ov::PartialShape{1, 1, 2, 2});
    auto output = embed.forward(pixel_values);
    auto model = ctx.build_model({output.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    std::vector<float> input_data{1.0f, 2.0f, 3.0f, 4.0f};
    ov::Tensor input_tensor(ov::element::f32, {1, 1, 2, 2});
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));
    request.set_input_tensor(0, input_tensor);
    request.infer();

    std::vector<float> expected{10.0f};
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_default);
}

TEST(DeepseekSamBlockTest, WindowPartitionIdentity) {
    ov::genai::modeling::BuilderContext ctx;

    ov::genai::modeling::models::DeepseekSamConfig cfg;
    cfg.embed_dim = 4;
    cfg.num_heads = 2;
    cfg.window_size = 2;
    cfg.use_rel_pos = true;

    ov::genai::modeling::models::DeepseekSamBlock block(ctx, "block", cfg, cfg.window_size);

    test_utils::DummyWeightSource weights;
    weights.add("block.norm1.weight", test_utils::make_tensor({1, 1, 1, 1}, {4}));
    weights.add("block.norm1.bias", test_utils::make_tensor({0, 0, 0, 0}, {4}));
    weights.add("block.norm2.weight", test_utils::make_tensor({1, 1, 1, 1}, {4}));
    weights.add("block.norm2.bias", test_utils::make_tensor({0, 0, 0, 0}, {4}));

    weights.add("block.attn.qkv.weight", test_utils::make_tensor(std::vector<float>(12 * 4, 0.0f), {12, 4}));
    weights.add("block.attn.qkv.bias", test_utils::make_tensor(std::vector<float>(12, 0.0f), {12}));
    weights.add("block.attn.proj.weight", test_utils::make_tensor(std::vector<float>(4 * 4, 0.0f), {4, 4}));
    weights.add("block.attn.proj.bias", test_utils::make_tensor(std::vector<float>(4, 0.0f), {4}));
    weights.add("block.attn.rel_pos_h", test_utils::make_tensor(std::vector<float>(3 * 2, 0.0f), {3, 2}));
    weights.add("block.attn.rel_pos_w", test_utils::make_tensor(std::vector<float>(3 * 2, 0.0f), {3, 2}));

    weights.add("block.mlp.lin1.weight", test_utils::make_tensor(std::vector<float>(8 * 4, 0.0f), {8, 4}));
    weights.add("block.mlp.lin1.bias", test_utils::make_tensor(std::vector<float>(8, 0.0f), {8}));
    weights.add("block.mlp.lin2.weight", test_utils::make_tensor(std::vector<float>(4 * 8, 0.0f), {4, 8}));
    weights.add("block.mlp.lin2.bias", test_utils::make_tensor(std::vector<float>(4, 0.0f), {4}));

    test_utils::DummyWeightFinalizer finalizer;
    ov::genai::modeling::weights::load_model(block, weights, finalizer);

    auto hidden_states = ctx.parameter("hidden_states", ov::element::f32, ov::PartialShape{1, 3, 3, 4});
    auto output = block.forward(hidden_states);
    auto model = ctx.build_model({output.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    std::vector<float> input_data(1 * 3 * 3 * 4);
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = static_cast<float>(i) * 0.01f;
    }
    ov::Tensor input_tensor(ov::element::f32, {1, 3, 3, 4});
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));
    request.set_input_tensor(0, input_tensor);
    request.infer();

    test_utils::expect_tensor_near(request.get_output_tensor(), input_data, test_utils::k_tol_default);
}
