// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>

#include "modeling/builder_context.hpp"
#include "modeling/ops/llm.hpp"
#include "modeling/tests/test_utils.hpp"

namespace test_utils = ov::genai::modeling::tests;

namespace {

std::vector<float> make_rope_cos(const std::vector<int64_t>& positions,
                                 size_t batch,
                                 size_t seq_len,
                                 int32_t head_dim,
                                 float rope_theta) {
    const int32_t half_dim = head_dim / 2;
    std::vector<float> cos(static_cast<size_t>(batch * seq_len * half_dim));
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            int64_t pos = positions[b * seq_len + s];
            for (int32_t i = 0; i < half_dim; ++i) {
                float exponent = static_cast<float>(2 * i) / static_cast<float>(head_dim);
                float inv_freq = 1.0f / std::pow(rope_theta, exponent);
                float angle = static_cast<float>(pos) * inv_freq;
                cos[(b * seq_len + s) * half_dim + i] = std::cos(angle);
            }
        }
    }
    return cos;
}

std::vector<float> make_rope_sin(const std::vector<int64_t>& positions,
                                 size_t batch,
                                 size_t seq_len,
                                 int32_t head_dim,
                                 float rope_theta) {
    const int32_t half_dim = head_dim / 2;
    std::vector<float> sin(static_cast<size_t>(batch * seq_len * half_dim));
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            int64_t pos = positions[b * seq_len + s];
            for (int32_t i = 0; i < half_dim; ++i) {
                float exponent = static_cast<float>(2 * i) / static_cast<float>(head_dim);
                float inv_freq = 1.0f / std::pow(rope_theta, exponent);
                float angle = static_cast<float>(pos) * inv_freq;
                sin[(b * seq_len + s) * half_dim + i] = std::sin(angle);
            }
        }
    }
    return sin;
}

std::vector<float> apply_rope_interleave_ref(const std::vector<float>& x,
                                             const std::vector<float>& cos,
                                             const std::vector<float>& sin,
                                             size_t batch,
                                             size_t heads,
                                             size_t seq_len,
                                             int32_t head_dim) {
    const int32_t half_dim = head_dim / 2;
    std::vector<float> out(x.size());
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < heads; ++h) {
            for (size_t s = 0; s < seq_len; ++s) {
                std::vector<float> interleaved(static_cast<size_t>(head_dim));
                size_t base = ((b * heads + h) * seq_len + s) * head_dim;
                for (int32_t i = 0; i < half_dim; ++i) {
                    interleaved[static_cast<size_t>(i)] = x[base + static_cast<size_t>(2 * i)];
                    interleaved[static_cast<size_t>(i + half_dim)] = x[base + static_cast<size_t>(2 * i + 1)];
                }

                for (int32_t i = 0; i < half_dim; ++i) {
                    float c = cos[(b * seq_len + s) * half_dim + i];
                    float sn = sin[(b * seq_len + s) * half_dim + i];
                    float x1 = interleaved[static_cast<size_t>(i)];
                    float x2 = interleaved[static_cast<size_t>(i + half_dim)];
                    out[base + static_cast<size_t>(i)] = x1 * c - x2 * sn;
                    out[base + static_cast<size_t>(i + half_dim)] = x1 * sn + x2 * c;
                }
            }
        }
    }
    return out;
}

std::vector<float> apply_rope_partial_ref(const std::vector<float>& x,
                                          const std::vector<float>& cos,
                                          const std::vector<float>& sin,
                                          size_t batch,
                                          size_t heads,
                                          size_t seq_len,
                                          int32_t head_dim,
                                          int32_t rotary_ndims) {
    const int32_t half_rotary_ndims = rotary_ndims / 2;
    std::vector<float> out = x;
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < heads; ++h) {
            for (size_t s = 0; s < seq_len; ++s) {
                const size_t base = ((b * heads + h) * seq_len + s) * static_cast<size_t>(head_dim);
                for (int32_t i = 0; i < half_rotary_ndims; ++i) {
                    const float c = cos[(b * seq_len + s) * static_cast<size_t>(half_rotary_ndims) + static_cast<size_t>(i)];
                    const float sn = sin[(b * seq_len + s) * static_cast<size_t>(half_rotary_ndims) + static_cast<size_t>(i)];
                    const float x1 = x[base + static_cast<size_t>(i)];
                    const float x2 = x[base + static_cast<size_t>(half_rotary_ndims + i)];
                    out[base + static_cast<size_t>(i)] = x1 * c - x2 * sn;
                    out[base + static_cast<size_t>(half_rotary_ndims + i)] = x1 * sn + x2 * c;
                }
            }
        }
    }
    return out;
}

}  // namespace

TEST(LLMOps, ApplyRopeInterleaveMatchesReference) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t batch = 1;
    const size_t heads = 1;
    const size_t seq_len = 2;
    const int32_t head_dim = 4;
    const int32_t half_dim = head_dim / 2;
    const float rope_theta = 10000.0f;

    auto x = ctx.parameter("x", ov::element::f32, ov::PartialShape{batch, heads, seq_len, head_dim});
    auto cos = ctx.parameter("cos", ov::element::f32, ov::PartialShape{batch, seq_len, half_dim});
    auto sin = ctx.parameter("sin", ov::element::f32, ov::PartialShape{batch, seq_len, half_dim});

    auto out = ov::genai::modeling::ops::llm::apply_rope_interleave(x, cos, sin, head_dim);
    auto ov_model = ctx.build_model({out.output()});

    ov::Core core;
    auto compiled = core.compile_model(ov_model, "GPU");
    auto request = compiled.create_infer_request();

    const std::vector<float> x_data = {
        0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.6f, 0.7f, 0.8f,
    };
    const std::vector<int64_t> positions = {0, 1};
    auto cos_data = make_rope_cos(positions, batch, seq_len, head_dim, rope_theta);
    auto sin_data = make_rope_sin(positions, batch, seq_len, head_dim, rope_theta);
    auto expected = apply_rope_interleave_ref(x_data, cos_data, sin_data, batch, heads, seq_len, head_dim);

    ov::Tensor x_tensor(ov::element::f32, {batch, heads, seq_len, static_cast<size_t>(head_dim)});
    std::memcpy(x_tensor.data(), x_data.data(), x_data.size() * sizeof(float));
    request.set_input_tensor(0, x_tensor);

    ov::Tensor cos_tensor(ov::element::f32, {batch, seq_len, static_cast<size_t>(half_dim)});
    std::memcpy(cos_tensor.data(), cos_data.data(), cos_data.size() * sizeof(float));
    request.set_input_tensor(1, cos_tensor);

    ov::Tensor sin_tensor(ov::element::f32, {batch, seq_len, static_cast<size_t>(half_dim)});
    std::memcpy(sin_tensor.data(), sin_data.data(), sin_data.size() * sizeof(float));
    request.set_input_tensor(2, sin_tensor);

    request.infer();

    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_default);
}

TEST(LLMOps, ApplyRopePartialHeadMatchesReference) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t batch = 1;
    const size_t heads = 2;
    const size_t seq_len = 2;
    const int32_t head_dim = 8;
    const int32_t rotary_ndims = 4;
    const int32_t half_rotary_ndims = rotary_ndims / 2;
    const float rope_theta = 10000.0f;

    auto x = ctx.parameter("x", ov::element::f32, ov::PartialShape{batch, heads, seq_len, head_dim});
    auto cos = ctx.parameter("cos", ov::element::f32, ov::PartialShape{batch, seq_len, half_rotary_ndims});
    auto sin = ctx.parameter("sin", ov::element::f32, ov::PartialShape{batch, seq_len, half_rotary_ndims});

    auto out = ov::genai::modeling::ops::llm::apply_rope(x, cos, sin, rotary_ndims, nullptr, head_dim);
    auto ov_model = ctx.build_model({out.output()});

    ov::Core core;
    auto compiled = core.compile_model(ov_model, "GPU");
    auto request = compiled.create_infer_request();

    const std::vector<float> x_data = {
        0.1f, 0.2f, 0.3f, 0.4f, 9.1f, 9.2f, 9.3f, 9.4f,
        0.5f, 0.6f, 0.7f, 0.8f, 9.5f, 9.6f, 9.7f, 9.8f,
        1.1f, 1.2f, 1.3f, 1.4f, 8.1f, 8.2f, 8.3f, 8.4f,
        1.5f, 1.6f, 1.7f, 1.8f, 8.5f, 8.6f, 8.7f, 8.8f,
    };
    const std::vector<int64_t> positions = {0, 1};
    const auto cos_data = make_rope_cos(positions, batch, seq_len, rotary_ndims, rope_theta);
    const auto sin_data = make_rope_sin(positions, batch, seq_len, rotary_ndims, rope_theta);
    const auto expected =
        apply_rope_partial_ref(x_data, cos_data, sin_data, batch, heads, seq_len, head_dim, rotary_ndims);

    ov::Tensor x_tensor(ov::element::f32, {batch, heads, seq_len, static_cast<size_t>(head_dim)});
    std::memcpy(x_tensor.data(), x_data.data(), x_data.size() * sizeof(float));
    request.set_input_tensor(0, x_tensor);

    ov::Tensor cos_tensor(ov::element::f32, {batch, seq_len, static_cast<size_t>(half_rotary_ndims)});
    std::memcpy(cos_tensor.data(), cos_data.data(), cos_data.size() * sizeof(float));
    request.set_input_tensor(1, cos_tensor);

    ov::Tensor sin_tensor(ov::element::f32, {batch, seq_len, static_cast<size_t>(half_rotary_ndims)});
    std::memcpy(sin_tensor.data(), sin_data.data(), sin_data.size() * sizeof(float));
    request.set_input_tensor(2, sin_tensor);

    request.infer();

    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_default);
}

TEST(LLMOps, PadSliceHeadDimRoundTrip) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t batch = 1;
    const size_t heads = 1;
    const size_t seq_len = 2;
    const int32_t head_dim = 2;
    const int32_t target_head_dim = 4;

    auto x = ctx.parameter("x", ov::element::f32, ov::PartialShape{batch, heads, seq_len, head_dim});
    auto padded = ov::genai::modeling::ops::llm::pad_to_head_dim(x, head_dim, target_head_dim);
    auto roundtrip = ov::genai::modeling::ops::llm::slice_to_head_dim(padded, target_head_dim, head_dim);
    auto ov_model = ctx.build_model({padded.output(), roundtrip.output()});

    ov::Core core;
    auto compiled = core.compile_model(ov_model, "GPU");
    auto request = compiled.create_infer_request();

    const std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> expected_padded = {1.0f, 2.0f, 0.0f, 0.0f, 3.0f, 4.0f, 0.0f, 0.0f};

    ov::Tensor x_tensor(ov::element::f32, {batch, heads, seq_len, static_cast<size_t>(head_dim)});
    std::memcpy(x_tensor.data(), x_data.data(), x_data.size() * sizeof(float));
    request.set_input_tensor(0, x_tensor);

    request.infer();

    test_utils::expect_tensor_near(request.get_output_tensor(0), expected_padded, test_utils::k_tol_exact);
    test_utils::expect_tensor_near(request.get_output_tensor(1), x_data, test_utils::k_tol_exact);
}
