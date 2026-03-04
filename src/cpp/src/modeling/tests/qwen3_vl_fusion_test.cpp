// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>

#include "modeling/builder_context.hpp"
#include "modeling/models/qwen3_vl/modeling_qwen3_vl_text.hpp"
#include "modeling/models/qwen3_vl/processing_qwen3_vl.hpp"
#include "modeling/tests/test_utils.hpp"

namespace test_utils = ov::genai::modeling::tests;

TEST(Qwen3VLFusion, EmbeddingInjector) {
    ov::genai::modeling::BuilderContext ctx;
    ov::genai::modeling::models::EmbeddingInjector injector(ctx, "injector");

    auto inputs = ctx.parameter("inputs", ov::element::f32, ov::PartialShape{1, 4, 2});
    auto visual = ctx.parameter("visual", ov::element::f32, ov::PartialShape{1, 4, 2});
    auto mask = ctx.parameter("mask", ov::element::i64, ov::PartialShape{1, 4});

    auto out = injector.forward(inputs, visual, mask);
    auto model = ctx.build_model({out.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    std::vector<float> inputs_data{1, 1, 2, 2, 3, 3, 4, 4};
    std::vector<float> visual_data{10, 10, 0, 0, 30, 30, 0, 0};
    std::vector<int64_t> mask_data{1, 0, 1, 0};

    ov::Tensor inputs_tensor(ov::element::f32, {1, 4, 2});
    std::memcpy(inputs_tensor.data(), inputs_data.data(), inputs_data.size() * sizeof(float));
    ov::Tensor visual_tensor(ov::element::f32, {1, 4, 2});
    std::memcpy(visual_tensor.data(), visual_data.data(), visual_data.size() * sizeof(float));
    ov::Tensor mask_tensor(ov::element::i64, {1, 4});
    std::memcpy(mask_tensor.data(), mask_data.data(), mask_data.size() * sizeof(int64_t));

    request.set_input_tensor(0, inputs_tensor);
    request.set_input_tensor(1, visual_tensor);
    request.set_input_tensor(2, mask_tensor);
    request.infer();

    std::vector<float> expected{10, 10, 2, 2, 30, 30, 4, 4};
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_default);
}

TEST(Qwen3VLFusion, DeepstackInjector) {
    ov::genai::modeling::BuilderContext ctx;
    ov::genai::modeling::models::DeepstackInjector injector(ctx, "injector");

    auto hidden = ctx.parameter("hidden", ov::element::f32, ov::PartialShape{1, 4, 2});
    auto mask = ctx.parameter("mask", ov::element::i64, ov::PartialShape{1, 4});
    auto deepstack = ctx.parameter("deepstack", ov::element::f32, ov::PartialShape{1, 4, 2});

    auto out = injector.forward(hidden, mask, deepstack);
    auto model = ctx.build_model({out.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    std::vector<float> hidden_data{1, 1, 2, 2, 3, 3, 4, 4};
    std::vector<float> deepstack_data{10, 10, 0, 0, 30, 30, 0, 0};
    std::vector<int64_t> mask_data{1, 0, 1, 0};

    ov::Tensor hidden_tensor(ov::element::f32, {1, 4, 2});
    std::memcpy(hidden_tensor.data(), hidden_data.data(), hidden_data.size() * sizeof(float));
    ov::Tensor deepstack_tensor(ov::element::f32, {1, 4, 2});
    std::memcpy(deepstack_tensor.data(), deepstack_data.data(), deepstack_data.size() * sizeof(float));
    ov::Tensor mask_tensor(ov::element::i64, {1, 4});
    std::memcpy(mask_tensor.data(), mask_data.data(), mask_data.size() * sizeof(int64_t));

    request.set_input_tensor(0, hidden_tensor);
    request.set_input_tensor(1, mask_tensor);
    request.set_input_tensor(2, deepstack_tensor);
    request.infer();

    std::vector<float> expected{11, 11, 2, 2, 33, 33, 4, 4};
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_default);
}

TEST(Qwen3VLFusion, InputPlannerTextOnly) {
    ov::genai::modeling::models::Qwen3VLConfig cfg;
    cfg.image_token_id = 151655;
    cfg.vision_start_token_id = 151652;
    cfg.vision.spatial_merge_size = 2;

    ov::genai::modeling::models::Qwen3VLInputPlanner planner(cfg);

    ov::Tensor input_ids(ov::element::i64, {1, 4});
    int64_t* ids = input_ids.data<int64_t>();
    ids[0] = 10;
    ids[1] = 11;
    ids[2] = 12;
    ids[3] = 13;

    auto plan = planner.build_plan(input_ids);
    auto pos = plan.position_ids.data<int64_t>();

    std::vector<int64_t> expected{0, 1, 2, 3};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(pos[0 * 4 + i], expected[i]);
        EXPECT_EQ(pos[1 * 4 + i], expected[i]);
        EXPECT_EQ(pos[2 * 4 + i], expected[i]);
    }
    EXPECT_EQ(plan.rope_deltas.data<int64_t>()[0], 0);
}

TEST(Qwen3VLFusion, InputPlannerWithImage) {
    ov::genai::modeling::models::Qwen3VLConfig cfg;
    cfg.image_token_id = 151655;
    cfg.vision_start_token_id = 151652;
    cfg.vision.spatial_merge_size = 2;

    ov::genai::modeling::models::Qwen3VLInputPlanner planner(cfg);

    ov::Tensor input_ids(ov::element::i64, {1, 6});
    int64_t* ids = input_ids.data<int64_t>();
    ids[0] = 1;
    ids[1] = 2;
    ids[2] = cfg.image_token_id;
    ids[3] = cfg.image_token_id;
    ids[4] = 3;
    ids[5] = 4;

    ov::Tensor grid_thw(ov::element::i64, {1, 3});
    int64_t* grid = grid_thw.data<int64_t>();
    grid[0] = 1;
    grid[1] = 2;
    grid[2] = 4;

    auto plan = planner.build_plan(input_ids, nullptr, &grid_thw);
    auto pos = plan.position_ids.data<int64_t>();
    const char* mask = plan.visual_pos_mask.data<const char>();

    std::vector<int64_t> expected_t{0, 1, 2, 2, 4, 5};
    std::vector<int64_t> expected_h{0, 1, 2, 2, 4, 5};
    std::vector<int64_t> expected_w{0, 1, 2, 3, 4, 5};
    for (size_t i = 0; i < expected_t.size(); ++i) {
        EXPECT_EQ(pos[0 * 6 + i], expected_t[i]);
        EXPECT_EQ(pos[1 * 6 + i], expected_h[i]);
        EXPECT_EQ(pos[2 * 6 + i], expected_w[i]);
    }
    EXPECT_EQ(plan.rope_deltas.data<int64_t>()[0], 0);
    EXPECT_EQ(mask[2], 1);
    EXPECT_EQ(mask[3], 1);
}

TEST(Qwen3VLFusion, ScatterVisualEmbeds) {
    ov::Tensor visual_embeds(ov::element::f32, {2, 2});
    float* embeds = visual_embeds.data<float>();
    embeds[0] = 10.0f;
    embeds[1] = 10.0f;
    embeds[2] = 30.0f;
    embeds[3] = 30.0f;

    ov::Tensor mask(ov::element::boolean, {1, 4});
    auto* mask_data = mask.data<char>();
    mask_data[0] = 1;
    mask_data[1] = 0;
    mask_data[2] = 1;
    mask_data[3] = 0;

    auto padded = ov::genai::modeling::models::Qwen3VLInputPlanner::scatter_visual_embeds(visual_embeds, mask);
    const float* out = padded.data<const float>();
    std::vector<float> expected{10, 10, 0, 0, 30, 30, 0, 0};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(out[i], expected[i], test_utils::k_tol_exact);
    }
}
