// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>

#include "modeling/ops/context.hpp"
#include "modeling/tests/test_utils.hpp"
#include "safetensors_utils/safetensors.hh"
#include "safetensors_utils/safetensors_loader.hpp"
#include "safetensors_utils/safetensors_weight_finalizer.hpp"

namespace test_utils = ov::genai::modeling::tests;

TEST(Qwen3NextFp8Loader, MapsF8E4M3ToOpenVinoType) {
    EXPECT_EQ(ov::genai::safetensors::convert_dtype(::safetensors::kFLOAT8_E4M3), ov::element::f8e4m3);
}

TEST(Qwen3NextFp8Loader, FinalizerBuildsFp8DequantSubgraphWithScaleInv) {
    test_utils::DummyWeightSource source;

    ov::Tensor fp8_weight(ov::element::f8e4m3, {4, 4});
    source.add("linear.weight", fp8_weight);

    ov::Tensor scale_inv(ov::element::f32, {1, 1});
    scale_inv.data<float>()[0] = 2.0f;
    source.add("linear.weight_scale_inv", scale_inv);

    ov::genai::safetensors::SafetensorsWeightFinalizer finalizer;
    ov::genai::modeling::OpContext op_ctx;
    auto finalized = finalizer.finalize("linear.weight", source, op_ctx);

    auto result = std::make_shared<ov::op::v0::Result>(finalized.primary.output());
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{});

    bool has_multiply = false;
    for (const auto& op : model->get_ops()) {
        has_multiply = has_multiply || std::string(op->get_type_name()) == "Multiply";
    }

    EXPECT_TRUE(has_multiply);
    EXPECT_EQ(model->get_output_element_type(0), ov::element::f32);
}
