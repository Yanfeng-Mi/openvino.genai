// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/deepseek_ocr2/processing_deepseek_ocr2.hpp"

#include <gtest/gtest.h>

namespace ov {
namespace genai {
namespace modeling {
namespace tests {

using ov::genai::modeling::models::DeepseekOCR2ImageTokens;
using ov::genai::modeling::models::DeepseekOCR2VisionPackager;

TEST(DeepseekOCR2VisionPackagerTest, PackWithLocalAndGlobal) {
    ov::Tensor global(ov::element::f32, {2, 2, 3});
    auto* global_ptr = global.data<float>();
    for (size_t i = 0; i < global.get_size(); ++i) {
        global_ptr[i] = static_cast<float>(i);
    }

    ov::Tensor local(ov::element::f32, {3, 2, 3});
    auto* local_ptr = local.data<float>();
    for (size_t i = 0; i < local.get_size(); ++i) {
        local_ptr[i] = static_cast<float>(100 + i);
    }

    DeepseekOCR2ImageTokens t0;
    t0.base_tokens = 2;
    t0.local_tokens = 4;
    DeepseekOCR2ImageTokens t1;
    t1.base_tokens = 2;
    t1.local_tokens = 2;

    ov::Tensor view(ov::element::f32, {3});
    auto* view_ptr = view.data<float>();
    view_ptr[0] = 1000.0f;
    view_ptr[1] = 1001.0f;
    view_ptr[2] = 1002.0f;

    DeepseekOCR2VisionPackager packager(view);
    auto outputs = packager.pack(global, &local, {t0, t1});

    ASSERT_EQ(outputs.size(), 2u);

    const auto out0_shape = outputs[0].get_shape();
    ASSERT_EQ(out0_shape.size(), 2u);
    EXPECT_EQ(out0_shape[0], 7u);
    EXPECT_EQ(out0_shape[1], 3u);

    const auto out1_shape = outputs[1].get_shape();
    ASSERT_EQ(out1_shape.size(), 2u);
    EXPECT_EQ(out1_shape[0], 5u);
    EXPECT_EQ(out1_shape[1], 3u);

    const float* out0 = outputs[0].data<const float>();
    const float* out1 = outputs[1].data<const float>();

    // out0: local rows 0-3 (local_data 0..11), global rows 0-1 (global_data 0..5), view separator
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(out0[i], local_ptr[i]);
    }
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(out0[12 + i], global_ptr[i]);
    }
    EXPECT_FLOAT_EQ(out0[18 + 0], view_ptr[0]);
    EXPECT_FLOAT_EQ(out0[18 + 1], view_ptr[1]);
    EXPECT_FLOAT_EQ(out0[18 + 2], view_ptr[2]);

    // out1: local rows 4-5 (local_data 12..17), global rows 2-3 (global_data 6..11), view separator
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(out1[i], local_ptr[12 + i]);
    }
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(out1[6 + i], global_ptr[6 + i]);
    }
    EXPECT_FLOAT_EQ(out1[12 + 0], view_ptr[0]);
    EXPECT_FLOAT_EQ(out1[12 + 1], view_ptr[1]);
    EXPECT_FLOAT_EQ(out1[12 + 2], view_ptr[2]);
}

TEST(DeepseekOCR2VisionPackagerTest, PackWithoutLocal) {
    ov::Tensor global(ov::element::f32, {1, 2, 2});
    auto* global_ptr = global.data<float>();
    for (size_t i = 0; i < global.get_size(); ++i) {
        global_ptr[i] = static_cast<float>(i + 1);
    }

    DeepseekOCR2ImageTokens tokens;
    tokens.base_tokens = 2;
    tokens.local_tokens = 0;

    ov::Tensor view(ov::element::f32, {2});
    auto* view_ptr = view.data<float>();
    view_ptr[0] = -1.0f;
    view_ptr[1] = -2.0f;

    DeepseekOCR2VisionPackager packager(view);
    auto outputs = packager.pack(global, nullptr, {tokens});

    ASSERT_EQ(outputs.size(), 1u);
    const auto shape = outputs[0].get_shape();
    ASSERT_EQ(shape.size(), 2u);
    EXPECT_EQ(shape[0], 3u);
    EXPECT_EQ(shape[1], 2u);

    const float* out = outputs[0].data<const float>();
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(out[i], global_ptr[i]);
    }
    EXPECT_FLOAT_EQ(out[4], view_ptr[0]);
    EXPECT_FLOAT_EQ(out[5], view_ptr[1]);
}

}  // namespace tests
}  // namespace modeling
}  // namespace genai
}  // namespace ov
