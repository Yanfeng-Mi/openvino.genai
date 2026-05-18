// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "modeling/models/qwen3_5/processing_qwen3_5.hpp"
#include "modeling/models/qwen3_5/qwen3_5_weight_specs.hpp"
#include "modeling/weights/synthetic_weight_source.hpp"

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
    cfg.text.full_attention_interval = 2;
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

TEST(Qwen3_5DummyWeightSource, BuildsExpectedSpecsAndRejectsMoe) {
    const auto cfg = make_small_cfg();
    const auto specs = ov::genai::modeling::models::build_qwen3_5_vlm_weight_specs(cfg);

    ASSERT_FALSE(specs.empty());

    auto has_name = [&](const std::string& name) {
        return std::any_of(specs.begin(), specs.end(), [&](const auto& s) { return s.name == name; });
    };

    EXPECT_TRUE(has_name("model.embed_tokens.weight"));
    EXPECT_TRUE(has_name("model.layers[0].linear_attn.in_proj_qkv.weight"));
    EXPECT_TRUE(has_name("model.layers[1].self_attn.q_proj.weight"));
    EXPECT_TRUE(has_name("visual.patch_embed.proj.weight"));
    EXPECT_TRUE(has_name("visual.merger.linear_fc2.weight"));

    for (const auto& spec : specs) {
        EXPECT_EQ(spec.name.find("experts"), std::string::npos) << spec.name;
        EXPECT_EQ(spec.name.find(".moe."), std::string::npos) << spec.name;
    }
}

TEST(Qwen3_5DummyWeightSource, IsDeterministicWithSameSeed) {
    const auto cfg = make_small_cfg();
    auto specs = ov::genai::modeling::models::build_qwen3_5_vlm_weight_specs(cfg);

    ov::genai::modeling::weights::SyntheticWeightSource source_a(specs, 2026u, -0.02f, 0.02f);
    ov::genai::modeling::weights::SyntheticWeightSource source_b(std::move(specs), 2026u, -0.02f, 0.02f);

    const auto& a = source_a.get_tensor("model.embed_tokens.weight");
    const auto& b = source_b.get_tensor("model.embed_tokens.weight");
    ASSERT_EQ(a.get_shape(), b.get_shape());
    ASSERT_EQ(a.get_element_type(), ov::element::f32);

    const auto* a_ptr = a.data<const float>();
    const auto* b_ptr = b.data<const float>();
    for (size_t i = 0; i < std::min<size_t>(32, a.get_size()); ++i) {
        EXPECT_FLOAT_EQ(a_ptr[i], b_ptr[i]);
    }
}
