// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "modeling/models/dflash_draft/dflash_draft.hpp"

TEST(DFlashUtils, BuildTargetLayerIds) {
    const auto ids = ov::genai::modeling::models::build_target_layer_ids(36, 5);
    ASSERT_EQ(ids.size(), 5u);
    EXPECT_EQ(ids[0], 1);
    EXPECT_EQ(ids[1], 9);
    EXPECT_EQ(ids[2], 17);
    EXPECT_EQ(ids[3], 25);
    EXPECT_EQ(ids[4], 33);
}
