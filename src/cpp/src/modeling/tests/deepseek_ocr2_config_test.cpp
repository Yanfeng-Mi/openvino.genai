// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/deepseek_ocr2/processing_deepseek_ocr2.hpp"

#include <gtest/gtest.h>

namespace ov {
namespace genai {
namespace modeling {
namespace tests {

using ov::genai::modeling::models::DeepseekOCR2Config;
using ov::genai::modeling::models::DeepseekOCR2ProcessorConfig;

TEST(DeepseekOCR2ConfigTest, ParseConfig) {
    const char* json_text = R"({
        "model_type": "deepseek_vl_v2",
        "architectures": ["DeepseekOCR2ForCausalLM"],
        "candidate_resolutions": [[1024, 1024]],
        "language_config": {
            "vocab_size": 129280,
            "hidden_size": 1280,
            "intermediate_size": 6848,
            "moe_intermediate_size": 896,
            "num_hidden_layers": 12,
            "num_attention_heads": 10,
            "num_key_value_heads": 10,
            "n_routed_experts": 64,
            "n_shared_experts": 2,
            "num_experts_per_tok": 6,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "bos_token_id": 0,
            "eos_token_id": 1
        },
        "vision_config": {
            "image_size": 1024,
            "mlp_ratio": 3.7362,
            "model_name": "deepencoderv2",
            "model_type": "vision",
            "width": {
                "sam_vit_b": {
                    "width": 768,
                    "layers": 12,
                    "heads": 12,
                    "global_attn_indexes": [2, 5, 8, 11],
                    "downsample_channels": [512, 1024]
                },
                "qwen2-0-5b": {"dim": 896}
            }
        },
        "projector_config": {
            "input_dim": 896,
            "n_embed": 1280,
            "projector_type": "linear",
            "model_type": "mlp_projector"
        }
    })";

    auto data = nlohmann::json::parse(json_text);
    DeepseekOCR2Config cfg = DeepseekOCR2Config::from_json(data);

    EXPECT_EQ(cfg.model_type, "deepseek_vl_v2");
    EXPECT_EQ(cfg.language.hidden_size, 1280);
    EXPECT_EQ(cfg.language.resolved_kv_heads(), 10);
    EXPECT_EQ(cfg.vision.sam_vit_b.width, 768);
    EXPECT_EQ(cfg.vision.qwen2_0_5b.dim, 896);
    EXPECT_EQ(cfg.projector.input_dim, 896);
    EXPECT_EQ(cfg.projector.n_embed, 1280);
    EXPECT_EQ(cfg.candidate_resolutions.size(), 1u);
}

TEST(DeepseekOCR2ConfigTest, ParseProcessorConfig) {
    const char* json_text = R"({
        "patch_size": 16,
        "downsample_ratio": 4,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "normalize": true,
        "add_special_token": false,
        "mask_prompt": false,
        "image_token": "<image>",
        "pad_token": "<|pad|>",
        "sft_format": "plain",
        "ignore_id": -100,
        "candidate_resolutions": [[1024, 1024]]
    })";

    auto data = nlohmann::json::parse(json_text);
    DeepseekOCR2ProcessorConfig cfg = DeepseekOCR2ProcessorConfig::from_json(data);

    EXPECT_EQ(cfg.patch_size, 16);
    EXPECT_EQ(cfg.downsample_ratio, 4);
    EXPECT_EQ(cfg.image_token, "<image>");
    EXPECT_EQ(cfg.candidate_resolutions.size(), 1u);
}

}  // namespace tests
}  // namespace modeling
}  // namespace genai
}  // namespace ov

