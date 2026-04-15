// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>
#include "modeling/builder_context.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/ops/shape.hpp"
#include "modeling/ops/tensor.hpp"
#include "modeling/tests/test_utils.hpp"

namespace test_utils = ov::genai::modeling::tests;

namespace {

std::vector<float> pow_mean_rsqrt_expected(const std::vector<float>& input, size_t rows, size_t cols) {
    std::vector<float> out(rows, 0.0f);
    for (size_t r = 0; r < rows; ++r) {
        float sumsq = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            float v = input[r * cols + c];
            sumsq += v * v;
        }
        float mean = sumsq / static_cast<float>(cols);
        out[r] = 1.0f / std::sqrt(mean + 1.0f);
    }
    return out;
}

std::vector<float> mean_expected(const std::vector<float>& input, size_t rows, size_t cols) {
    std::vector<float> out(rows, 0.0f);
    for (size_t r = 0; r < rows; ++r) {
        float acc = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            acc += input[r * cols + c];
        }
        out[r] = acc / static_cast<float>(cols);
    }
    return out;
}

void expect_tensor_eq_i64(const ov::Tensor& output, const std::vector<int64_t>& expected) {
    ASSERT_EQ(output.get_size(), expected.size());
    const int64_t* out_data = output.data<const int64_t>();
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(out_data[i], expected[i]);
    }
}

}  // namespace

TEST(TensorOps, TypeConversions) {
    ov::genai::modeling::BuilderContext ctx;

    auto x = ctx.parameter("x", ov::element::f32, ov::Shape{2, 3});
    EXPECT_EQ(x.dtype(), ov::element::f32);

    auto y = x.to(ov::element::f16);
    EXPECT_EQ(y.dtype(), ov::element::f16);

    auto z = y.to(ov::element::f32);
    EXPECT_EQ(z.dtype(), ov::element::f32);

    auto model = ctx.build_model({z.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto input_data = test_utils::make_seq(6, 0.0f, 1.0f);
    ov::Tensor input_tensor(ov::element::f32, {2, 3});
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));
    request.set_input_tensor(input_tensor);
    request.infer();

    test_utils::expect_tensor_near(request.get_output_tensor(), input_data, test_utils::k_tol_default);
}

TEST(TensorOps, PowMeanRsqrtAndMean) {
    ov::genai::modeling::BuilderContext ctx;

    auto x = ctx.parameter("x", ov::element::f32, ov::Shape{2, 3});

    auto y = x.pow(2.0f).mean(-1, true);
    auto z = (y + 1.0f).rsqrt();
    auto mean_flat = x.mean(1, false);

    auto model = ctx.build_model({z.output(), mean_flat.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto input_data = test_utils::make_seq(6, 1.0f, 1.0f);
    ov::Tensor input_tensor(ov::element::f32, {2, 3});
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));
    request.set_input_tensor(input_tensor);
    request.infer();

    auto expected_z = pow_mean_rsqrt_expected(input_data, 2, 3);
    auto expected_mean = mean_expected(input_data, 2, 3);

    test_utils::expect_tensor_near(request.get_output_tensor(0), expected_z, test_utils::k_tol_default);
    test_utils::expect_tensor_near(request.get_output_tensor(1), expected_mean, test_utils::k_tol_default);
}

TEST(TensorOps, ArithmeticOperators) {
    ov::genai::modeling::BuilderContext ctx;

    auto x = ctx.parameter("x", ov::element::f32, ov::Shape{3});
    auto y = ctx.parameter("y", ov::element::f32, ov::Shape{3});

    auto sum = x + y;
    auto sum_scalar = sum + 1.0f;
    auto sum_rev = 2.0f + x;
    auto diff = x - y;
    auto diff_scalar = x - 1.0f;
    auto diff_rev = 2.0f - x;
    auto neg = -x;
    auto prod = x * y;
    auto prod_scalar = x * 2.0f;
    auto prod_scalar_rev = 2.0f * x;
    auto div = prod / (x + 1.0f);
    auto div_scalar = x / 2.0f;
    auto div_scalar_rev = 2.0f / x;

    auto model = ctx.build_model({sum.output(),
                                  sum_scalar.output(),
                                  sum_rev.output(),
                                  diff.output(),
                                  diff_scalar.output(),
                                  diff_rev.output(),
                                  neg.output(),
                                  prod.output(),
                                  prod_scalar.output(),
                                  prod_scalar_rev.output(),
                                  div.output(),
                                  div_scalar.output(),
                                  div_scalar_rev.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    std::vector<float> x_data = {1.0f, 3.0f, 7.0f};
    std::vector<float> y_data = {4.0f, 4.0f, 8.0f};
    ov::Tensor x_tensor(ov::element::f32, {3});
    ov::Tensor y_tensor(ov::element::f32, {3});
    std::memcpy(x_tensor.data(), x_data.data(), x_data.size() * sizeof(float));
    std::memcpy(y_tensor.data(), y_data.data(), y_data.size() * sizeof(float));

    request.set_input_tensor(0, x_tensor);
    request.set_input_tensor(1, y_tensor);
    request.infer();

    std::vector<float> expected_sum = {5.0f, 7.0f, 15.0f};
    std::vector<float> expected_sum_scalar = {6.0f, 8.0f, 16.0f};
    std::vector<float> expected_sum_rev = {3.0f, 5.0f, 9.0f};
    std::vector<float> expected_diff = {-3.0f, -1.0f, -1.0f};
    std::vector<float> expected_diff_scalar = {0.0f, 2.0f, 6.0f};
    std::vector<float> expected_diff_rev = {1.0f, -1.0f, -5.0f};
    std::vector<float> expected_neg = {-1.0f, -3.0f, -7.0f};
    std::vector<float> expected_prod = {4.0f, 12.0f, 56.0f};
    std::vector<float> expected_prod_scalar = {2.0f, 6.0f, 14.0f};
    std::vector<float> expected_prod_scalar_rev = {2.0f, 6.0f, 14.0f};
    std::vector<float> expected_div = {2.0f, 3.0f, 7.0f};
    std::vector<float> expected_div_scalar = {0.5f, 1.5f, 3.5f};
    std::vector<float> expected_div_scalar_rev = {2.0f, 0.6666667f, 0.2857143f};

    test_utils::expect_tensor_near(request.get_output_tensor(0), expected_sum, test_utils::k_tol_default);
    test_utils::expect_tensor_near(request.get_output_tensor(1), expected_sum_scalar, test_utils::k_tol_default);
    test_utils::expect_tensor_near(request.get_output_tensor(2), expected_sum_rev, test_utils::k_tol_default);
    test_utils::expect_tensor_near(request.get_output_tensor(3), expected_diff, test_utils::k_tol_default);
    test_utils::expect_tensor_near(request.get_output_tensor(4), expected_diff_scalar, test_utils::k_tol_default);
    test_utils::expect_tensor_near(request.get_output_tensor(5), expected_diff_rev, test_utils::k_tol_default);
    test_utils::expect_tensor_near(request.get_output_tensor(6), expected_neg, test_utils::k_tol_default);
    test_utils::expect_tensor_near(request.get_output_tensor(7), expected_prod, test_utils::k_tol_default);
    test_utils::expect_tensor_near(request.get_output_tensor(8), expected_prod_scalar, test_utils::k_tol_default);
    test_utils::expect_tensor_near(request.get_output_tensor(9), expected_prod_scalar_rev, test_utils::k_tol_default);
    test_utils::expect_tensor_near(request.get_output_tensor(10), expected_div, test_utils::k_tol_default);
    test_utils::expect_tensor_near(request.get_output_tensor(11), expected_div_scalar, test_utils::k_tol_default);
    test_utils::expect_tensor_near(request.get_output_tensor(12), expected_div_scalar_rev, test_utils::k_tol_default);
}

TEST(TensorOps, TrigExpLogSoftmax) {
    ov::genai::modeling::BuilderContext ctx;

    auto x = ctx.parameter("x", ov::element::f32, ov::Shape{1, 3});

    auto sin_x = x.sin();
    auto cos_x = x.cos();
    auto exp_x = x.exp();
    auto log_x = exp_x.log();
    auto softmax_x = x.softmax(1);

    auto model = ctx.build_model({sin_x.output(), cos_x.output(), exp_x.output(), log_x.output(), softmax_x.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    std::vector<float> x_data = {0.0f, 1.0f, 2.0f};
    ov::Tensor x_tensor(ov::element::f32, {1, 3});
    std::memcpy(x_tensor.data(), x_data.data(), x_data.size() * sizeof(float));
    request.set_input_tensor(x_tensor);
    request.infer();

    std::vector<float> expected_sin;
    std::vector<float> expected_cos;
    std::vector<float> expected_exp;
    std::vector<float> expected_log;
    std::vector<float> expected_softmax;
    expected_sin.reserve(x_data.size());
    expected_cos.reserve(x_data.size());
    expected_exp.reserve(x_data.size());
    expected_log.reserve(x_data.size());
    expected_softmax.resize(x_data.size(), 0.0f);

    float max_val = *std::max_element(x_data.begin(), x_data.end());
    float denom = 0.0f;
    for (float v : x_data) {
        expected_sin.push_back(std::sin(v));
        expected_cos.push_back(std::cos(v));
        float ev = std::exp(v);
        expected_exp.push_back(ev);
        expected_log.push_back(std::log(ev));
        denom += std::exp(v - max_val);
    }
    for (size_t i = 0; i < x_data.size(); ++i) {
        expected_softmax[i] = std::exp(x_data[i] - max_val) / denom;
    }

    test_utils::expect_tensor_near(request.get_output_tensor(0), expected_sin,
                                  test_utils::k_tol_transcendental);
    test_utils::expect_tensor_near(request.get_output_tensor(1), expected_cos,
                                  test_utils::k_tol_transcendental);
    test_utils::expect_tensor_near(request.get_output_tensor(2), expected_exp,
                                  test_utils::k_tol_transcendental);
    test_utils::expect_tensor_near(request.get_output_tensor(3), expected_log,
                                  test_utils::k_tol_transcendental);
    test_utils::expect_tensor_near(request.get_output_tensor(4), expected_softmax,
                                  test_utils::k_tol_transcendental);
}

TEST(TensorShapeOps, ReshapePermuteMerge) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t batch = 1;
    const size_t seq_len = 2;
    const size_t num_heads = 2;
    const size_t head_dim = 3;
    const size_t hidden = num_heads * head_dim;

    auto x = ctx.parameter("x", ov::element::f32, ov::Shape{batch, seq_len, hidden});

    auto heads = x.reshape({0, 0, static_cast<int64_t>(num_heads), static_cast<int64_t>(head_dim)})
                    .permute({0, 2, 1, 3});
    auto merged = heads.transpose({0, 2, 1, 3}).reshape({0, 0, static_cast<int64_t>(hidden)});

    auto model = ctx.build_model({heads.output(), merged.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto input_data = test_utils::make_seq(batch * seq_len * hidden);
    ov::Tensor input_tensor(ov::element::f32, {batch, seq_len, hidden});
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));
    request.set_input_tensor(input_tensor);
    request.infer();

    auto heads_output = request.get_output_tensor(0);
    auto merged_output = request.get_output_tensor(1);

    EXPECT_EQ(heads_output.get_shape(), (ov::Shape{batch, num_heads, seq_len, head_dim}));
    EXPECT_EQ(merged_output.get_shape(), (ov::Shape{batch, seq_len, hidden}));

    auto expected_heads = test_utils::to_heads_ref(input_data, batch, seq_len, num_heads, head_dim);
    test_utils::expect_tensor_near(heads_output, expected_heads, test_utils::k_tol_exact);
    test_utils::expect_tensor_near(merged_output, input_data, test_utils::k_tol_exact);
}

TEST(ShapeOps, ShapeHelpersBroadcast) {
    ov::genai::modeling::BuilderContext ctx;
    auto* op_ctx = &ctx.op_context();

    auto x = ctx.parameter("x", ov::element::f32, ov::Shape{1, 3});

    auto dim0 = ov::genai::modeling::shape::dim(x, 0);
    auto dim1 = ov::genai::modeling::shape::dim(x, 1);
    auto target =
        ov::genai::modeling::shape::make({ov::genai::modeling::ops::const_vec(op_ctx, std::vector<int64_t>{2}),
                                          dim1});
    auto bcast = ov::genai::modeling::shape::broadcast_to(x, target);

    auto model = ctx.build_model({dim0, dim1, bcast.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    std::vector<float> x_data = {1.0f, 2.0f, 3.0f};
    ov::Tensor x_tensor(ov::element::f32, {1, 3});
    std::memcpy(x_tensor.data(), x_data.data(), x_data.size() * sizeof(float));
    request.set_input_tensor(x_tensor);
    request.infer();

    expect_tensor_eq_i64(request.get_output_tensor(0), {1});
    expect_tensor_eq_i64(request.get_output_tensor(1), {3});
    test_utils::expect_tensor_near(request.get_output_tensor(2), {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f},
                                  test_utils::k_tol_exact);
}

TEST(TensorShapeOps, UnsqueezeSqueezeChain) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t rows = 2;
    const size_t cols = 3;

    auto x = ctx.parameter("x", ov::element::f32, ov::Shape{rows, cols});

    auto y = x.unsqueeze({0, 2}).squeeze({0, 2});
    auto model = ctx.build_model({y.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto input_data = test_utils::make_seq(rows * cols);
    ov::Tensor input_tensor(ov::element::f32, {rows, cols});
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));
    request.set_input_tensor(input_tensor);
    request.infer();

    auto output = request.get_output_tensor();
    EXPECT_EQ(output.get_shape(), (ov::Shape{rows, cols}));
    test_utils::expect_tensor_near(output, input_data, test_utils::k_tol_exact);
}
