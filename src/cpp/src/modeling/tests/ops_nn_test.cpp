// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/nn.hpp"
#include "modeling/ops/tensor_ops.hpp"
#include "modeling/ops/tensor.hpp"
#include "modeling/tests/test_utils.hpp"

namespace test_utils = ov::genai::modeling::tests;

namespace {

std::shared_ptr<ov::Model> build_model_from_output(const ov::Output<ov::Node>& output,
                                                   const ov::ParameterVector& params) {
    auto result = std::make_shared<ov::op::v0::Result>(output);
    return std::make_shared<ov::Model>(ov::OutputVector{result}, params);
}

void run_model_test(const std::shared_ptr<ov::Model>& model,
                    const ov::Tensor& input_tensor,
                    const std::vector<float>& expected,
                    float tol = test_utils::k_tol_default) {
    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();
    request.set_input_tensor(input_tensor);
    request.infer();
    ov::Tensor output = request.get_output_tensor();
    test_utils::expect_tensor_near(output, expected, tol);
}

std::vector<float> layer_norm_ref(const std::vector<float>& input,
                                  const std::vector<float>& weight,
                                  const std::vector<float>& bias,
                                  size_t rows,
                                  size_t cols,
                                  float eps) {
    std::vector<float> out(input.size());
    for (size_t r = 0; r < rows; ++r) {
        float mean = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            mean += input[r * cols + c];
        }
        mean /= static_cast<float>(cols);
        float var = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            float diff = input[r * cols + c] - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(cols);
        float inv = 1.0f / std::sqrt(var + eps);
        for (size_t c = 0; c < cols; ++c) {
            float norm = (input[r * cols + c] - mean) * inv;
            out[r * cols + c] = norm * weight[c] + bias[c];
        }
    }
    return out;
}

std::vector<float> conv2d_ref(const std::vector<float>& input,
                              const std::vector<float>& weight,
                              size_t in_h,
                              size_t in_w,
                              size_t k_h,
                              size_t k_w) {
    const size_t out_h = in_h - k_h + 1;
    const size_t out_w = in_w - k_w + 1;
    std::vector<float> out(out_h * out_w, 0.0f);
    for (size_t y = 0; y < out_h; ++y) {
        for (size_t x = 0; x < out_w; ++x) {
            float acc = 0.0f;
            for (size_t ky = 0; ky < k_h; ++ky) {
                for (size_t kx = 0; kx < k_w; ++kx) {
                    size_t in_idx = (y + ky) * in_w + (x + kx);
                    size_t w_idx = ky * k_w + kx;
                    acc += input[in_idx] * weight[w_idx];
                }
            }
            out[y * out_w + x] = acc;
        }
    }
    return out;
}

std::vector<float> group_norm_ref(const std::vector<float>& input,
                                  const std::vector<float>& weight,
                                  const std::vector<float>& bias,
                                  size_t n,
                                  size_t c,
                                  size_t h,
                                  size_t w,
                                  size_t groups,
                                  float eps) {
    std::vector<float> out(input.size(), 0.0f);
    const size_t group_size = c / groups;
    for (size_t bn = 0; bn < n; ++bn) {
        for (size_t g = 0; g < groups; ++g) {
            float mean = 0.0f;
            float var = 0.0f;
            size_t count = group_size * h * w;
            for (size_t cg = 0; cg < group_size; ++cg) {
                size_t c_idx = g * group_size + cg;
                for (size_t y = 0; y < h; ++y) {
                    for (size_t x = 0; x < w; ++x) {
                        size_t idx = ((bn * c + c_idx) * h + y) * w + x;
                        mean += input[idx];
                    }
                }
            }
            mean /= static_cast<float>(count);
            for (size_t cg = 0; cg < group_size; ++cg) {
                size_t c_idx = g * group_size + cg;
                for (size_t y = 0; y < h; ++y) {
                    for (size_t x = 0; x < w; ++x) {
                        size_t idx = ((bn * c + c_idx) * h + y) * w + x;
                        float diff = input[idx] - mean;
                        var += diff * diff;
                    }
                }
            }
            var /= static_cast<float>(count);
            float inv = 1.0f / std::sqrt(var + eps);
            for (size_t cg = 0; cg < group_size; ++cg) {
                size_t c_idx = g * group_size + cg;
                for (size_t y = 0; y < h; ++y) {
                    for (size_t x = 0; x < w; ++x) {
                        size_t idx = ((bn * c + c_idx) * h + y) * w + x;
                        float norm = (input[idx] - mean) * inv;
                        out[idx] = norm * weight[c_idx] + bias[c_idx];
                    }
                }
            }
        }
    }
    return out;
}

float gelu_tanh_ref(float x) {
    const float k0 = 0.7978845608f;
    const float k1 = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + std::tanh(k0 * (x + k1 * x3)));
}

}  // namespace

TEST(OpsNN, LayerNorm) {
    ov::genai::modeling::OpContext ctx;
    const ov::Shape shape{1, 2, 3};
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    ov::genai::modeling::Tensor x(param, &ctx);

    std::vector<float> weight_data{1.0f, 0.5f, 2.0f};
    std::vector<float> bias_data{0.1f, -0.2f, 0.3f};
    auto weight = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{3}, weight_data);
    auto bias = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{3}, bias_data);
    ov::genai::modeling::Tensor w(weight, &ctx);
    ov::genai::modeling::Tensor b(bias, &ctx);

    auto out = ov::genai::modeling::ops::nn::layer_norm(x, w, &b, 1e-5f, -1);
    auto model = build_model_from_output(out.output(), {param});

    std::vector<float> input_data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    ov::Tensor input_tensor(ov::element::f32, shape);
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));
    auto expected = layer_norm_ref(input_data, weight_data, bias_data, 2, 3, 1e-5f);
    run_model_test(model, input_tensor, expected, test_utils::k_tol_default);
}

TEST(OpsNN, GeluTanh) {
    ov::genai::modeling::OpContext ctx;
    const ov::Shape shape{3};
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    ov::genai::modeling::Tensor x(param, &ctx);

    auto out = ov::genai::modeling::ops::nn::gelu(x, true);
    auto model = build_model_from_output(out.output(), {param});

    std::vector<float> input_data{-1.0f, 0.0f, 1.0f};
    std::vector<float> expected;
    expected.reserve(input_data.size());
    for (float v : input_data) {
        expected.push_back(gelu_tanh_ref(v));
    }
    ov::Tensor input_tensor(ov::element::f32, shape);
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));
    run_model_test(model, input_tensor, expected, test_utils::k_tol_transcendental);
}

TEST(OpsNN, Conv2d) {
    ov::genai::modeling::OpContext ctx;
    const ov::Shape in_shape{1, 1, 3, 3};
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, in_shape);
    ov::genai::modeling::Tensor x(param, &ctx);

    std::vector<float> weight_data{1.0f, 0.0f, 0.0f, -1.0f};
    auto weight = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 2, 2}, weight_data);
    ov::genai::modeling::Tensor w(weight, &ctx);

    auto out = ov::genai::modeling::ops::nn::conv2d(x, w, {1, 1}, {0, 0}, {0, 0});
    auto model = build_model_from_output(out.output(), {param});

    std::vector<float> input_data{1.0f, 2.0f, 3.0f,
                                  4.0f, 5.0f, 6.0f,
                                  7.0f, 8.0f, 9.0f};
    ov::Tensor input_tensor(ov::element::f32, in_shape);
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));
    auto expected = conv2d_ref(input_data, weight_data, 3, 3, 2, 2);
    run_model_test(model, input_tensor, expected, test_utils::k_tol_default);
}

TEST(OpsNN, GroupNorm) {
    ov::genai::modeling::OpContext ctx;
    const ov::Shape in_shape{1, 4, 1, 2};
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, in_shape);
    ov::genai::modeling::Tensor x(param, &ctx);

    std::vector<float> weight_data{1.0f, 1.5f, 0.5f, 2.0f};
    std::vector<float> bias_data{0.1f, -0.1f, 0.2f, -0.2f};
    auto weight = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{4}, weight_data);
    auto bias = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{4}, bias_data);
    ov::genai::modeling::Tensor w(weight, &ctx);
    ov::genai::modeling::Tensor b(bias, &ctx);

    auto out = ov::genai::modeling::ops::nn::group_norm(x, w, &b, 2, 1e-5f);
    auto model = build_model_from_output(out.output(), {param});

    std::vector<float> input_data{1.0f, 2.0f,
                                  3.0f, 4.0f,
                                  5.0f, 6.0f,
                                  7.0f, 8.0f};
    ov::Tensor input_tensor(ov::element::f32, in_shape);
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));
    auto expected = group_norm_ref(input_data, weight_data, bias_data, 1, 4, 1, 2, 2, 1e-5f);
    run_model_test(model, input_tensor, expected, test_utils::k_tol_default);
}

TEST(OpsNN, PadConstant) {
    ov::genai::modeling::OpContext ctx;
    const ov::Shape in_shape{1, 1, 2, 2};
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, in_shape);
    ov::genai::modeling::Tensor x(param, &ctx);

    auto out = ov::genai::modeling::ops::tensor::pad(x, {0, 0, 1, 1}, {0, 0, 1, 1}, 0.0f);
    auto model = build_model_from_output(out.output(), {param});

    std::vector<float> input_data{1.0f, 2.0f,
                                  3.0f, 4.0f};
    std::vector<float> expected{
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 2.0f, 0.0f,
        0.0f, 3.0f, 4.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
    };
    ov::Tensor input_tensor(ov::element::f32, in_shape);
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));
    run_model_test(model, input_tensor, expected, test_utils::k_tol_exact);
}

TEST(OpsNN, UpsampleNearest) {
    ov::genai::modeling::OpContext ctx;
    const ov::Shape in_shape{1, 1, 2, 2};
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, in_shape);
    ov::genai::modeling::Tensor x(param, &ctx);

    auto out = ov::genai::modeling::ops::nn::upsample_nearest(x, 2, 2);
    auto model = build_model_from_output(out.output(), {param});

    std::vector<float> input_data{1.0f, 2.0f,
                                  3.0f, 4.0f};
    std::vector<float> expected{
        1.0f, 1.0f, 2.0f, 2.0f,
        1.0f, 1.0f, 2.0f, 2.0f,
        3.0f, 3.0f, 4.0f, 4.0f,
        3.0f, 3.0f, 4.0f, 4.0f,
    };
    ov::Tensor input_tensor(ov::element::f32, in_shape);
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));
    run_model_test(model, input_tensor, expected, test_utils::k_tol_exact);
}
