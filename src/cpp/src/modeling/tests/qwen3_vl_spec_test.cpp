// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_vl/processing_qwen3_vl.hpp"

#include <gtest/gtest.h>

namespace ov {
namespace genai {
namespace modeling {
namespace tests {

using ov::genai::modeling::models::Qwen3VLConfig;
using ov::genai::modeling::models::Qwen3VLGraphSpec;
using ov::genai::modeling::models::Qwen3VLModuleNames;

TEST(Qwen3VLSpecTest, ParseAndValidateConfig) {
    const char* json_text = R"({
        "model_type": "qwen3_vl",
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "image_token_id": 151655,
        "video_token_id": 151656,
        "vision_start_token_id": 151652,
        "vision_end_token_id": 151653,
        "tie_word_embeddings": true,
        "text_config": {
            "model_type": "qwen3_vl_text",
            "vocab_size": 151936,
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "rope_theta": 5000000,
            "hidden_act": "silu",
            "attention_bias": false,
            "attention_dropout": 0.0,
            "dtype": "bfloat16",
            "tie_word_embeddings": true,
            "rope_scaling": {
                "mrope_interleaved": true,
                "mrope_section": [24, 20, 20],
                "rope_type": "default"
            }
        },
        "vision_config": {
            "model_type": "qwen3_vl",
            "depth": 24,
            "hidden_size": 1024,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 4096,
            "num_heads": 16,
            "in_channels": 3,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 2048,
            "num_position_embeddings": 2304,
            "deepstack_visual_indexes": [5, 11, 17]
        }
    })";

    auto data = nlohmann::json::parse(json_text);
    Qwen3VLConfig cfg = Qwen3VLConfig::from_json(data);

    EXPECT_EQ(cfg.model_type, "qwen3_vl");
    EXPECT_EQ(cfg.text.hidden_size, 2048);
    EXPECT_EQ(cfg.text.kv_heads(), 8);
    EXPECT_EQ(cfg.text.resolved_head_dim(), 128);
    EXPECT_TRUE(cfg.text.rope.mrope_interleaved);
    EXPECT_EQ(cfg.vision.depth, 24);
    EXPECT_EQ(cfg.vision.head_dim(), 64);
    EXPECT_TRUE(cfg.tie_word_embeddings);
}

TEST(Qwen3VLSpecTest, ModuleNames) {
    EXPECT_EQ(Qwen3VLModuleNames::vision_block(3), "blocks.3");
    EXPECT_EQ(Qwen3VLModuleNames::deepstack_merger(1), "deepstack_merger_list.1");
    EXPECT_EQ(Qwen3VLModuleNames::text_layer(7), "layers.7");
}

TEST(Qwen3VLSpecTest, GraphSpecOutputs) {
    const char* json_text = R"({
        "model_type": "qwen3_vl",
        "text_config": {"hidden_size": 1, "intermediate_size": 1, "num_hidden_layers": 1,
                        "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 1},
        "vision_config": {"depth": 2, "hidden_size": 2, "num_heads": 1,
                          "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
                          "out_hidden_size": 2, "num_position_embeddings": 4,
                          "deepstack_visual_indexes": [0, 1]}
    })";
    auto data = nlohmann::json::parse(json_text);
    Qwen3VLConfig cfg = Qwen3VLConfig::from_json(data);
    auto outputs = Qwen3VLGraphSpec::vision_outputs(cfg.vision);
    EXPECT_EQ(outputs.size(), 1u + cfg.vision.deepstack_visual_indexes.size());
}

}  // namespace tests
}  // namespace modeling
}  // namespace genai
}  // namespace ov
