// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>
#include <openvino/op/mvn.hpp>
#include <openvino/op/transpose.hpp>
#include <ov_ops/vl_sdpa.hpp>

#include "modeling/builder_context.hpp"
#include "modeling/models/qwen3_5/export_for_vlmpipeline.hpp"
#include "modeling/models/qwen3_5/modeling_qwen3_5_vision.hpp"
#include "modeling/models/qwen3_5/qwen3_5_weight_specs.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/synthetic_weight_source.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace test_utils = ov::genai::modeling::tests;

namespace {

ov::genai::modeling::models::Qwen3_5Config make_small_cfg() {
    using namespace ov::genai::modeling::models;
    Qwen3_5Config cfg;
    cfg.model_type = "qwen3_5";

    cfg.text.model_type = "qwen3_5_text";
    cfg.text.vocab_size = 64;
    cfg.text.hidden_size = 16;
    cfg.text.intermediate_size = 32;
    cfg.text.num_hidden_layers = 2;
    cfg.text.num_attention_heads = 4;
    cfg.text.num_key_value_heads = 2;
    cfg.text.head_dim = 4;
    cfg.text.partial_rotary_factor = 0.5f;
    cfg.text.max_position_embeddings = 256;
    cfg.text.layer_types = {"linear_attention", "full_attention"};
    cfg.text.linear_conv_kernel_dim = 2;
    cfg.text.linear_key_head_dim = 4;
    cfg.text.linear_value_head_dim = 4;
    cfg.text.linear_num_key_heads = 2;
    cfg.text.linear_num_value_heads = 2;
    cfg.text.rope.mrope_interleaved = true;
    cfg.text.rope.mrope_section = {1, 1, 0};

    cfg.vision.model_type = "qwen3_5";
    cfg.vision.depth = 2;
    cfg.vision.hidden_size = 8;
    cfg.vision.intermediate_size = 16;
    cfg.vision.num_heads = 2;
    cfg.vision.in_channels = 3;
    cfg.vision.patch_size = 2;
    cfg.vision.temporal_patch_size = 1;
    cfg.vision.spatial_merge_size = 2;
    cfg.vision.out_hidden_size = cfg.text.hidden_size;
    cfg.vision.num_position_embeddings = 16;
    cfg.vision.deepstack_visual_indexes.clear();

    cfg.image_token_id = 7;
    cfg.video_token_id = 8;
    cfg.vision_start_token_id = 9;
    cfg.vision_end_token_id = 10;

    cfg.finalize();
    cfg.validate();
    return cfg;
}

}  // namespace

TEST(Qwen3_5VisionAttentionTest, ExplicitCuSeqLensBuildsVlsdpa) {
    ov::genai::modeling::BuilderContext ctx;

    ov::genai::modeling::models::Qwen3_5VisionConfig cfg;
    cfg.hidden_size = 4;
    cfg.num_heads = 2;

    ov::genai::modeling::models::Qwen3_5VisionAttention attn(ctx, "attn", cfg);

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
    auto cu_seq_lens = ctx.parameter("cu_seq_lens", ov::element::i32, ov::PartialShape{2});

    auto output = attn.forward(hidden_states, rotary_cos, rotary_sin, &cu_seq_lens);
    auto model = ctx.build_model({output.output()});

    size_t vlsdpa_count = 0;
    size_t transpose_count = 0;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::as_type_ptr<ov::op::internal::VLSDPA>(node)) {
            ++vlsdpa_count;
        }
        if (ov::as_type_ptr<ov::op::v1::Transpose>(node)) {
            ++transpose_count;
        }
    }

    EXPECT_EQ(vlsdpa_count, 1u);
    EXPECT_EQ(transpose_count, 0u);
}

TEST(Qwen3_5VisionModelTest, FullModelDerivesCuSeqLensAndBuildsVlsdpa) {
    const auto cfg = make_small_cfg();
    auto specs = ov::genai::modeling::models::build_qwen3_5_vlm_weight_specs(cfg);
    ov::genai::modeling::weights::SyntheticWeightSource source(std::move(specs), 2028u, -0.02f, 0.02f);
    test_utils::DummyWeightFinalizer finalizer;

    auto model = ov::genai::modeling::models::create_qwen3_5_vision_model(cfg, source, finalizer);

    size_t vlsdpa_count = 0;
    size_t mvn_count = 0;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::as_type_ptr<ov::op::internal::VLSDPA>(node)) {
            ++vlsdpa_count;
        }
        if (ov::as_type_ptr<ov::op::v6::MVN>(node)) {
            ++mvn_count;
        }
    }

    EXPECT_EQ(vlsdpa_count, static_cast<size_t>(cfg.vision.depth));
    EXPECT_EQ(mvn_count, static_cast<size_t>(cfg.vision.depth * 2 + 1));
}

TEST(Qwen3_5VisionPatchMergerTest, ExplicitLayerNormBuildsMvn) {
    ov::genai::modeling::BuilderContext ctx;

    ov::genai::modeling::models::Qwen3_5VisionConfig cfg;
    cfg.hidden_size = 4;
    cfg.intermediate_size = 8;
    cfg.out_hidden_size = 4;
    cfg.spatial_merge_size = 2;

    ov::genai::modeling::models::Qwen3_5VisionPatchMerger merger(ctx, "merger", cfg, false);

    test_utils::DummyWeightSource weights;
    weights.add("merger.norm.weight", test_utils::make_tensor(std::vector<float>(4, 1.0f), {4}));
    weights.add("merger.norm.bias", test_utils::make_tensor(std::vector<float>(4, 0.0f), {4}));
    weights.add("merger.linear_fc1.weight", test_utils::make_tensor(std::vector<float>(8 * 16, 0.0f), {8, 16}));
    weights.add("merger.linear_fc1.bias", test_utils::make_tensor(std::vector<float>(8, 0.0f), {8}));
    weights.add("merger.linear_fc2.weight", test_utils::make_tensor(std::vector<float>(4 * 8, 0.0f), {4, 8}));
    weights.add("merger.linear_fc2.bias", test_utils::make_tensor(std::vector<float>(4, 0.0f), {4}));

    test_utils::DummyWeightFinalizer finalizer;
    ov::genai::modeling::weights::load_model(merger, weights, finalizer);

    auto hidden_states = ctx.parameter("hidden_states", ov::element::f32, ov::PartialShape{4, 4});

    auto output = merger.forward(hidden_states);
    auto model = ctx.build_model({output.output()});

    size_t mvn_count = 0;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::as_type_ptr<ov::op::v6::MVN>(node)) {
            ++mvn_count;
        }
    }

    EXPECT_EQ(mvn_count, 1u);
}

TEST(Qwen3_5VisionModelTest, MergerExportUsesCuSeqLensAndBuildsVlsdpa) {
    const auto cfg = make_small_cfg();
    auto specs = ov::genai::modeling::models::build_qwen3_5_vlm_weight_specs(cfg);
    ov::genai::modeling::weights::SyntheticWeightSource source(std::move(specs), 2029u, -0.02f, 0.02f);
    test_utils::DummyWeightFinalizer finalizer;

    auto model = ov::genai::modeling::models::create_qwen3_5_vision_merger_model(cfg, source, finalizer);

    bool has_cu_seq_lens = false;
    bool has_attention_mask = false;
    size_t vlsdpa_count = 0;
    for (const auto& input : model->inputs()) {
        const auto& names = input.get_names();
        has_cu_seq_lens |= names.count("cu_seq_lens") != 0;
        has_attention_mask |= names.count("attention_mask") != 0;
    }
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::as_type_ptr<ov::op::internal::VLSDPA>(node)) {
            ++vlsdpa_count;
        }
    }

    EXPECT_TRUE(has_cu_seq_lens);
    EXPECT_FALSE(has_attention_mask);
    EXPECT_EQ(vlsdpa_count, static_cast<size_t>(cfg.vision.depth));
}
