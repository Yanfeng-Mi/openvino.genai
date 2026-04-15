// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/deepseek_ocr2/processing_deepseek_ocr2.hpp"

#include <gtest/gtest.h>

namespace ov {
namespace genai {
namespace modeling {
namespace tests {

using ov::genai::modeling::models::DeepseekOCR2ImageTokens;
using ov::genai::modeling::models::build_prompt_plan_from_tokens;

TEST(DeepseekOCR2PromptTest, BuildPromptPlan) {
    std::vector<std::vector<int64_t>> segments = {{10, 11}, {12}};
    DeepseekOCR2ImageTokens tokens;
    tokens.base_tokens = 4;
    tokens.local_tokens = 0;

    auto plan = build_prompt_plan_from_tokens(segments, {tokens}, 99, true, 0, false, 1);

    const auto ids_shape = plan.input_ids.get_shape();
    ASSERT_EQ(ids_shape.size(), 2u);
    EXPECT_EQ(ids_shape[0], 1u);
    EXPECT_EQ(ids_shape[1], 9u);

    const auto* ids = plan.input_ids.data<const int64_t>();
    EXPECT_EQ(ids[0], 0);
    EXPECT_EQ(ids[1], 10);
    EXPECT_EQ(ids[2], 11);
    EXPECT_EQ(ids[3], 99);
    EXPECT_EQ(ids[7], 99);
    EXPECT_EQ(ids[8], 12);

    const auto* mask = plan.images_seq_mask.data<const char>();
    EXPECT_EQ(mask[0], 0);
    EXPECT_EQ(mask[1], 0);
    EXPECT_EQ(mask[2], 0);
    EXPECT_EQ(mask[3], 1);
    EXPECT_EQ(mask[7], 1);
    EXPECT_EQ(mask[8], 0);

    const auto* attn = plan.attention_mask.data<const int64_t>();
    for (size_t i = 0; i < plan.attention_mask.get_size(); ++i) {
        EXPECT_EQ(attn[i], 1);
    }
}

}  // namespace tests
}  // namespace modeling
}  // namespace genai
}  // namespace ov

