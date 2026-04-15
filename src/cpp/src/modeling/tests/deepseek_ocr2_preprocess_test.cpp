// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/deepseek_ocr2/processing_deepseek_ocr2.hpp"

#include <cstring>
#include <gtest/gtest.h>

namespace ov {
namespace genai {
namespace modeling {
namespace tests {

using ov::genai::modeling::models::DeepseekOCR2ImagePreprocessor;
using ov::genai::modeling::models::DeepseekOCR2PreprocessConfig;

TEST(DeepseekOCR2PreprocessTest, DynamicPreprocess) {
    DeepseekOCR2PreprocessConfig cfg;
    cfg.base_size = 1024;
    cfg.image_size = 768;
    cfg.patch_size = 16;
    cfg.downsample_ratio = 4;
    cfg.crop_mode = true;

    DeepseekOCR2ImagePreprocessor preprocessor(cfg);

    ov::Tensor image(ov::element::u8, {800, 800, 3});
    std::memset(image.data(), 255, image.get_byte_size());

    auto out = preprocessor.preprocess(image);

    const auto global_shape = out.global_images.get_shape();
    ASSERT_EQ(global_shape.size(), 4u);
    EXPECT_EQ(global_shape[0], 1u);
    EXPECT_EQ(global_shape[1], 3u);
    EXPECT_EQ(global_shape[2], 1024u);
    EXPECT_EQ(global_shape[3], 1024u);

    const auto local_shape = out.local_images.get_shape();
    ASSERT_EQ(local_shape.size(), 4u);
    EXPECT_EQ(local_shape[0], 4u);
    EXPECT_EQ(local_shape[1], 3u);
    EXPECT_EQ(local_shape[2], 768u);
    EXPECT_EQ(local_shape[3], 768u);

    const auto crop_shape = out.images_spatial_crop.get_shape();
    ASSERT_EQ(crop_shape.size(), 2u);
    EXPECT_EQ(crop_shape[0], 1u);
    EXPECT_EQ(crop_shape[1], 2u);

    const auto* crop = out.images_spatial_crop.data<const int64_t>();
    EXPECT_EQ(crop[0], 2);
    EXPECT_EQ(crop[1], 2);

    ASSERT_EQ(out.image_tokens.size(), 1u);
    EXPECT_EQ(out.image_tokens[0].base_tokens, 256);
    EXPECT_EQ(out.image_tokens[0].local_tokens, 576);
    EXPECT_EQ(out.image_tokens[0].total_tokens(), 833);

    const float first_val = out.global_images.data<const float>()[0];
    EXPECT_NEAR(first_val, 1.0f, 1e-3f);
}

}  // namespace tests
}  // namespace modeling
}  // namespace genai
}  // namespace ov

