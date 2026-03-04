// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>

#include "modeling/models/deepseek_ocr2/modeling_deepseek_ocr2_projector.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace test_utils = ov::genai::modeling::tests;

TEST(DeepseekOCR2ProjectorTest, LinearProjection) {
    ov::genai::modeling::BuilderContext ctx;

    ov::genai::modeling::models::DeepseekOCR2ProjectorConfig cfg;
    cfg.input_dim = 4;
    cfg.n_embed = 6;
    cfg.projector_type = "linear";

    ov::genai::modeling::models::DeepseekOCR2Projector projector(ctx, "projector", cfg);

    std::vector<float> weight = test_utils::make_seq(24, 0.01f, 0.01f);
    std::vector<float> bias = test_utils::make_seq(6, -0.5f, 0.1f);

    test_utils::DummyWeightSource weights;
    weights.add("projector.layers.weight", test_utils::make_tensor(weight, {6, 4}));
    weights.add("projector.layers.bias", test_utils::make_tensor(bias, {6}));

    test_utils::DummyWeightFinalizer finalizer;
    ov::genai::modeling::weights::load_model(projector, weights, finalizer);

    auto input = ctx.parameter("input", ov::element::f32, ov::PartialShape{1, 2, 4});
    auto output = projector.forward(input);
    auto model = ctx.build_model({output.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    std::vector<float> input_data = {0.2f, 0.4f, -0.5f, 1.0f,
                                     -0.3f, 0.7f, 0.1f, -0.2f};
    ov::Tensor input_tensor(ov::element::f32, {1, 2, 4});
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));

    request.set_input_tensor(0, input_tensor);
    request.infer();

    auto expected = test_utils::linear_ref_3d_bias(input_data, weight, bias, 1, 2, 4, 6);
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_default);
}
