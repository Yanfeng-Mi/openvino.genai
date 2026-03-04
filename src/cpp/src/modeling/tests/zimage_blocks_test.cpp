// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/models/zimage/modeling_zimage_dit.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace test_utils = ov::genai::modeling::tests;

namespace {

std::shared_ptr<ov::Model> build_model_from_output(const ov::Output<ov::Node>& output,
                                                   const ov::ParameterVector& params) {
    auto result = std::make_shared<ov::op::v0::Result>(output);
    return std::make_shared<ov::Model>(ov::OutputVector{result}, params);
}

}  // namespace

TEST(ZImageBlocks, AttentionZeroWeights) {
    ov::genai::modeling::BuilderContext ctx;
    constexpr int32_t dim = 256;
    constexpr int32_t heads = 8;
    constexpr int32_t kv_heads = 8;
    constexpr int32_t head_dim = dim / heads;
    constexpr int32_t rope_dim = head_dim / 2;

    ov::genai::modeling::models::ZImageAttention attn(ctx, "attn", dim, heads, kv_heads, 1e-5f, false);

    test_utils::DummyWeightSource source;
    test_utils::DummyWeightFinalizer finalizer;
    ov::Tensor zero_w(ov::element::f32, {dim, dim});
    std::memset(zero_w.data(), 0, zero_w.get_byte_size());
    source.add("attn.to_q.weight", zero_w);
    source.add("attn.to_k.weight", zero_w);
    source.add("attn.to_v.weight", zero_w);
    source.add("attn.to_out.0.weight", zero_w);

    ov::genai::modeling::weights::load_model(attn, source, finalizer);

    auto hidden = ctx.parameter("hidden", ov::element::f32, {1, 16, dim});
    auto mask = ctx.parameter("mask", ov::element::boolean, {1, 16});
    auto rope_cos = ctx.parameter("rope_cos", ov::element::f32, {1, 16, rope_dim});
    auto rope_sin = ctx.parameter("rope_sin", ov::element::f32, {1, 16, rope_dim});

    auto out = attn.forward(hidden, mask, rope_cos, rope_sin);
    auto model = build_model_from_output(out.output(), ctx.parameters());

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    std::vector<float> hidden_data(1 * 16 * dim, 1.0f);
    std::vector<char> mask_data(16, 1);
    std::vector<float> cos_data(1 * 16 * rope_dim, 1.0f);
    std::vector<float> sin_data(1 * 16 * rope_dim, 0.0f);

    ov::Tensor hidden_tensor(ov::element::f32, {1, 16, dim});
    ov::Tensor mask_tensor(ov::element::boolean, {1, 16});
    ov::Tensor cos_tensor(ov::element::f32, {1, 16, rope_dim});
    ov::Tensor sin_tensor(ov::element::f32, {1, 16, rope_dim});

    std::memcpy(hidden_tensor.data(), hidden_data.data(), hidden_data.size() * sizeof(float));
    std::memcpy(mask_tensor.data(), mask_data.data(), mask_data.size() * sizeof(char));
    std::memcpy(cos_tensor.data(), cos_data.data(), cos_data.size() * sizeof(float));
    std::memcpy(sin_tensor.data(), sin_data.data(), sin_data.size() * sizeof(float));

    request.set_input_tensor(0, hidden_tensor);
    request.set_input_tensor(1, mask_tensor);
    request.set_input_tensor(2, cos_tensor);
    request.set_input_tensor(3, sin_tensor);
    request.infer();

    std::vector<float> expected(1 * 16 * dim, 0.0f);
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, 1e-4f);
}
