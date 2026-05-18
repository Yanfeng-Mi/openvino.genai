// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>

#include "modeling/models/wan/processing_wan.hpp"

TEST(WanT2VPrepareLatents, ReturnsF32AndShape) {
    const size_t batch = 1;
    const size_t channels = 2;
    const size_t frames = 3;
    const size_t height = 4;
    const size_t width = 5;
    const int32_t seed = 42;

    auto latents = ov::genai::modeling::models::prepare_latents(
        batch, channels, frames, height, width, seed);
    EXPECT_EQ(latents.get_element_type(), ov::element::f32);
    EXPECT_EQ(latents.get_shape(), (ov::Shape{batch, channels, frames, height, width}));

    auto latents_repeat = ov::genai::modeling::models::prepare_latents(
        batch, channels, frames, height, width, seed);
    ASSERT_EQ(latents.get_size(), latents_repeat.get_size());

    const float* lhs = latents.data<const float>();
    const float* rhs = latents_repeat.data<const float>();
    for (size_t i = 0; i < latents.get_size(); ++i) {
        EXPECT_FLOAT_EQ(lhs[i], rhs[i]);
    }
}
