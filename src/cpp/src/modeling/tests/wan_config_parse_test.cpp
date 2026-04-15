// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/wan/processing_wan.hpp"

#include <gtest/gtest.h>

namespace ov {
namespace genai {
namespace modeling {
namespace tests {

using ov::genai::modeling::models::WanTransformer3DConfig;
using ov::genai::modeling::models::WanVAEConfig;

TEST(WanConfigParseTest, TransformerConfig) {
    const char* json_text = R"({
        "_class_name": "WanTransformer3DModel",
        "patch_size": [1, 2, 2],
        "num_attention_heads": 12,
        "attention_head_dim": 128,
        "in_channels": 16,
        "out_channels": 16,
        "text_dim": 4096,
        "freq_dim": 256,
        "ffn_dim": 8960,
        "num_layers": 30,
        "cross_attn_norm": true,
        "qk_norm": "rms_norm_across_heads",
        "eps": 1e-6,
        "image_dim": null,
        "added_kv_proj_dim": null,
        "rope_max_seq_len": 1024
    })";

    auto data = nlohmann::json::parse(json_text);
    auto cfg = WanTransformer3DConfig::from_json(data);

    EXPECT_EQ(cfg.class_name, "WanTransformer3DModel");
    EXPECT_EQ(cfg.patch_size.size(), 3u);
    EXPECT_EQ(cfg.num_layers, 30);
    EXPECT_EQ(cfg.attention_head_dim, 128);
    EXPECT_EQ(cfg.inner_dim(), 12 * 128);
    EXPECT_EQ(cfg.patch_volume(), 4);
    EXPECT_FALSE(cfg.image_dim.has_value());
}

TEST(WanConfigParseTest, VAEConfig) {
    const char* json_text = R"({
        "_class_name": "AutoencoderKLWan",
        "base_dim": 96,
        "dim_mult": [1, 2, 4, 4],
        "dropout": 0.0,
        "latents_mean": [-0.1, -0.2, -0.3, -0.4],
        "latents_std": [1.1, 1.2, 1.3, 1.4],
        "num_res_blocks": 2,
        "temperal_downsample": [false, true, true],
        "z_dim": 4
    })";

    auto data = nlohmann::json::parse(json_text);
    auto cfg = WanVAEConfig::from_json(data);

    EXPECT_EQ(cfg.class_name, "AutoencoderKLWan");
    EXPECT_EQ(cfg.base_dim, 96);
    EXPECT_EQ(cfg.decoder_base_dim, 96);
    EXPECT_EQ(cfg.z_dim, 4);
    EXPECT_EQ(cfg.dim_mult.size(), 4u);
    EXPECT_EQ(cfg.temperal_downsample.size(), 3u);
    EXPECT_EQ(cfg.latents_mean.size(), 4u);
    EXPECT_EQ(cfg.latents_std.size(), 4u);
}

}  // namespace tests
}  // namespace modeling
}  // namespace genai
}  // namespace ov
