// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <openvino/core/except.hpp>

#include "modeling/models/wan/processing_wan.hpp"

TEST(WanT2VCfg, MatchesExpectedFormula) {
    std::vector<float> noise_pred = {1.0f, 2.0f, 3.0f};
    std::vector<float> noise_uncond = {0.5f, 1.5f, 2.5f};
    const float guidance_scale = 2.0f;

    auto out = ov::genai::modeling::models::apply_cfg(noise_pred, noise_uncond, guidance_scale);

    std::vector<float> expected = {1.5f, 2.5f, 3.5f};
    ASSERT_EQ(out.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(out[i], expected[i]);
    }
}

TEST(WanT2VCfg, ThrowsOnSizeMismatch) {
    std::vector<float> noise_pred = {1.0f};
    std::vector<float> noise_uncond = {1.0f, 2.0f};
    EXPECT_THROW(ov::genai::modeling::models::apply_cfg(noise_pred, noise_uncond, 1.0f), ov::Exception);
}
