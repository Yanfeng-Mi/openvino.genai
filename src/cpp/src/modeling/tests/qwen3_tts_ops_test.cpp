// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Unit tests for Qwen3-TTS ops: nn ops, math ops, tensor ops

#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>
#include "modeling/builder_context.hpp"
#include "modeling/ops/nn.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/ops/tensor_ops.hpp"
#include "modeling/tests/test_utils.hpp"

namespace test_utils = ov::genai::modeling::tests;
namespace nn = ov::genai::modeling::ops::nn;
namespace tensor = ov::genai::modeling::ops::tensor;

// ==================== Reference Implementations ====================

namespace {

// ReLU reference: max(0, x)
std::vector<float> relu_ref(const std::vector<float>& x) {
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = std::max(0.0f, x[i]);
    }
    return out;
}

// Sigmoid reference: 1 / (1 + exp(-x))
std::vector<float> sigmoid_ref(const std::vector<float>& x) {
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = 1.0f / (1.0f + std::exp(-x[i]));
    }
    return out;
}

// Tanh reference
std::vector<float> tanh_ref(const std::vector<float>& x) {
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = std::tanh(x[i]);
    }
    return out;
}

// Sin reference
std::vector<float> sin_ref(const std::vector<float>& x) {
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = std::sin(x[i]);
    }
    return out;
}

// Cos reference
std::vector<float> cos_ref(const std::vector<float>& x) {
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = std::cos(x[i]);
    }
    return out;
}

// Conv1d reference (no groups, no dilation)
// input: [batch, in_channels, length]
// weight: [out_channels, in_channels, kernel_size]
// output: [batch, out_channels, out_length]
std::vector<float> conv1d_ref(const std::vector<float>& input,
                              const std::vector<float>& weight,
                              size_t batch,
                              size_t in_channels,
                              size_t length,
                              size_t out_channels,
                              size_t kernel_size,
                              size_t stride,
                              size_t pad_begin,
                              size_t pad_end) {
    size_t padded_length = length + pad_begin + pad_end;
    size_t out_length = (padded_length - kernel_size) / stride + 1;

    std::vector<float> output(batch * out_channels * out_length, 0.0f);

    for (size_t b = 0; b < batch; ++b) {
        for (size_t oc = 0; oc < out_channels; ++oc) {
            for (size_t ol = 0; ol < out_length; ++ol) {
                float sum = 0.0f;
                for (size_t ic = 0; ic < in_channels; ++ic) {
                    for (size_t k = 0; k < kernel_size; ++k) {
                        int64_t il = static_cast<int64_t>(ol * stride + k) - static_cast<int64_t>(pad_begin);
                        if (il >= 0 && il < static_cast<int64_t>(length)) {
                            size_t input_idx = b * in_channels * length + ic * length + il;
                            size_t weight_idx = oc * in_channels * kernel_size + ic * kernel_size + k;
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
                output[b * out_channels * out_length + oc * out_length + ol] = sum;
            }
        }
    }
    return output;
}

// Add bias to conv output
std::vector<float> add_bias_ref(const std::vector<float>& input,
                                const std::vector<float>& bias,
                                size_t batch,
                                size_t channels,
                                size_t length) {
    std::vector<float> output = input;
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t l = 0; l < length; ++l) {
                output[b * channels * length + c * length + l] += bias[c];
            }
        }
    }
    return output;
}

// Batch norm reference (inference mode)
// input: [batch, channels, length]
std::vector<float> batch_norm_ref(const std::vector<float>& input,
                                  const std::vector<float>& gamma,
                                  const std::vector<float>& beta,
                                  const std::vector<float>& mean,
                                  const std::vector<float>& var,
                                  size_t batch,
                                  size_t channels,
                                  size_t length,
                                  float eps) {
    std::vector<float> output(input.size());
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            float scale = gamma[c] / std::sqrt(var[c] + eps);
            float shift = beta[c] - mean[c] * scale;
            for (size_t l = 0; l < length; ++l) {
                size_t idx = b * channels * length + c * length + l;
                output[idx] = input[idx] * scale + shift;
            }
        }
    }
    return output;
}

// ReduceSum reference along single axis
std::vector<float> reduce_sum_ref(const std::vector<float>& input,
                                  const std::vector<size_t>& shape,
                                  int64_t axis,
                                  bool keepdim) {
    size_t ndim = shape.size();
    if (axis < 0) axis += ndim;

    // Calculate output shape
    std::vector<size_t> out_shape;
    for (size_t i = 0; i < ndim; ++i) {
        if (i == static_cast<size_t>(axis)) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(shape[i]);
        }
    }

    size_t out_size = 1;
    for (auto s : out_shape) out_size *= s;
    std::vector<float> output(out_size, 0.0f);

    // Calculate strides for input
    std::vector<size_t> strides(ndim);
    strides[ndim - 1] = 1;
    for (int i = static_cast<int>(ndim) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    // Sum over axis
    size_t total = input.size();
    for (size_t i = 0; i < total; ++i) {
        // Convert flat index to multi-index
        std::vector<size_t> idx(ndim);
        size_t rem = i;
        for (size_t d = 0; d < ndim; ++d) {
            idx[d] = rem / strides[d];
            rem = rem % strides[d];
        }

        // Calculate output index (skip reduced axis)
        size_t out_idx = 0;
        size_t out_stride = 1;
        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            if (d != axis) {
                out_idx += idx[d] * out_stride;
                out_stride *= (keepdim && d == axis) ? 1 : shape[d];
            }
        }
        // Recalculate with proper output strides
        std::vector<size_t> out_strides(out_shape.size());
        if (!out_shape.empty()) {
            out_strides[out_shape.size() - 1] = 1;
            for (int d = static_cast<int>(out_shape.size()) - 2; d >= 0; --d) {
                out_strides[d] = out_strides[d + 1] * out_shape[d + 1];
            }
        }
        out_idx = 0;
        size_t out_d = 0;
        for (size_t d = 0; d < ndim; ++d) {
            if (d == static_cast<size_t>(axis)) {
                if (keepdim) out_d++;
            } else {
                out_idx += idx[d] * out_strides[out_d++];
            }
        }
        output[out_idx] += input[i];
    }
    return output;
}

// Flip reference along axis
std::vector<float> flip_ref(const std::vector<float>& input,
                            const std::vector<size_t>& shape,
                            int64_t axis) {
    size_t ndim = shape.size();
    if (axis < 0) axis += ndim;

    std::vector<float> output(input.size());

    // Calculate strides
    std::vector<size_t> strides(ndim);
    strides[ndim - 1] = 1;
    for (int i = static_cast<int>(ndim) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    for (size_t i = 0; i < input.size(); ++i) {
        // Convert flat index to multi-index
        std::vector<size_t> idx(ndim);
        size_t rem = i;
        for (size_t d = 0; d < ndim; ++d) {
            idx[d] = rem / strides[d];
            rem = rem % strides[d];
        }

        // Flip the axis
        idx[axis] = shape[axis] - 1 - idx[axis];

        // Convert back to flat index
        size_t out_idx = 0;
        for (size_t d = 0; d < ndim; ++d) {
            out_idx += idx[d] * strides[d];
        }
        output[out_idx] = input[i];
    }
    return output;
}

}  // namespace

// ==================== NN Ops Tests ====================

TEST(Qwen3TTSNNOps, ReLU) {
    ov::genai::modeling::BuilderContext ctx;

    auto x = ctx.parameter("x", ov::element::f32, ov::Shape{2, 3});
    auto out = nn::relu(x);
    auto model = ctx.build_model({out.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    std::vector<float> x_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    ov::Tensor x_tensor(ov::element::f32, {2, 3});
    std::memcpy(x_tensor.data(), x_data.data(), x_data.size() * sizeof(float));

    request.set_input_tensor(x_tensor);
    request.infer();

    auto expected = relu_ref(x_data);
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_exact);
}

TEST(Qwen3TTSNNOps, Sigmoid) {
    ov::genai::modeling::BuilderContext ctx;

    auto x = ctx.parameter("x", ov::element::f32, ov::Shape{2, 3});
    auto out = nn::sigmoid(x);
    auto model = ctx.build_model({out.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    std::vector<float> x_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    ov::Tensor x_tensor(ov::element::f32, {2, 3});
    std::memcpy(x_tensor.data(), x_data.data(), x_data.size() * sizeof(float));

    request.set_input_tensor(x_tensor);
    request.infer();

    auto expected = sigmoid_ref(x_data);
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_transcendental);
}

TEST(Qwen3TTSNNOps, Tanh) {
    ov::genai::modeling::BuilderContext ctx;

    auto x = ctx.parameter("x", ov::element::f32, ov::Shape{2, 3});
    auto out = nn::tanh_activation(x);
    auto model = ctx.build_model({out.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    std::vector<float> x_data = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
    ov::Tensor x_tensor(ov::element::f32, {2, 3});
    std::memcpy(x_tensor.data(), x_data.data(), x_data.size() * sizeof(float));

    request.set_input_tensor(x_tensor);
    request.infer();

    auto expected = tanh_ref(x_data);
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_transcendental);
}

TEST(Qwen3TTSNNOps, Conv1dBasic) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t batch = 1;
    const size_t in_channels = 2;
    const size_t length = 8;
    const size_t out_channels = 3;
    const size_t kernel_size = 3;

    auto input = ctx.parameter("input", ov::element::f32, ov::Shape{batch, in_channels, length});
    auto weight = ctx.parameter("weight", ov::element::f32, ov::Shape{out_channels, in_channels, kernel_size});

    auto out = nn::conv1d(input, weight, {1}, {1}, {1});
    auto model = ctx.build_model({out.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto input_data = test_utils::make_seq(batch * in_channels * length, 0.1f, 0.1f);
    auto weight_data = test_utils::make_seq(out_channels * in_channels * kernel_size, 0.05f, 0.05f);

    ov::Tensor input_tensor(ov::element::f32, {batch, in_channels, length});
    ov::Tensor weight_tensor(ov::element::f32, {out_channels, in_channels, kernel_size});
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));
    std::memcpy(weight_tensor.data(), weight_data.data(), weight_data.size() * sizeof(float));

    request.set_input_tensor(0, input_tensor);
    request.set_input_tensor(1, weight_tensor);
    request.infer();

    auto expected = conv1d_ref(input_data, weight_data, batch, in_channels, length,
                               out_channels, kernel_size, 1, 1, 1);
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_default);
}

TEST(Qwen3TTSNNOps, Conv1dWithBias) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t batch = 1;
    const size_t in_channels = 2;
    const size_t length = 8;
    const size_t out_channels = 3;
    const size_t kernel_size = 3;

    auto input = ctx.parameter("input", ov::element::f32, ov::Shape{batch, in_channels, length});
    auto weight = ctx.parameter("weight", ov::element::f32, ov::Shape{out_channels, in_channels, kernel_size});
    auto bias = ctx.parameter("bias", ov::element::f32, ov::Shape{out_channels});

    auto out = nn::conv1d(input, weight, bias, {1}, {1}, {1});
    auto model = ctx.build_model({out.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto input_data = test_utils::make_seq(batch * in_channels * length, 0.1f, 0.1f);
    auto weight_data = test_utils::make_seq(out_channels * in_channels * kernel_size, 0.05f, 0.05f);
    std::vector<float> bias_data = {0.1f, 0.2f, 0.3f};

    ov::Tensor input_tensor(ov::element::f32, {batch, in_channels, length});
    ov::Tensor weight_tensor(ov::element::f32, {out_channels, in_channels, kernel_size});
    ov::Tensor bias_tensor(ov::element::f32, {out_channels});
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));
    std::memcpy(weight_tensor.data(), weight_data.data(), weight_data.size() * sizeof(float));
    std::memcpy(bias_tensor.data(), bias_data.data(), bias_data.size() * sizeof(float));

    request.set_input_tensor(0, input_tensor);
    request.set_input_tensor(1, weight_tensor);
    request.set_input_tensor(2, bias_tensor);
    request.infer();

    auto conv_out = conv1d_ref(input_data, weight_data, batch, in_channels, length,
                               out_channels, kernel_size, 1, 1, 1);
    size_t out_length = (length + 2 - kernel_size) / 1 + 1;
    auto expected = add_bias_ref(conv_out, bias_data, batch, out_channels, out_length);
    // Use slightly higher tolerance for conv+bias due to GPU floating-point accumulation
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, 2e-3f);
}

TEST(Qwen3TTSNNOps, ConvTranspose1dShape) {
    // ConvTranspose1d shape verification (numerical verification is complex)
    ov::genai::modeling::BuilderContext ctx;

    const size_t batch = 1;
    const size_t in_channels = 4;
    const size_t length = 8;
    const size_t out_channels = 2;
    const size_t kernel_size = 4;
    const size_t stride = 2;

    auto input = ctx.parameter("input", ov::element::f32, ov::Shape{batch, in_channels, length});
    auto weight = ctx.parameter("weight", ov::element::f32, ov::Shape{in_channels, out_channels, kernel_size});

    auto out = nn::conv_transpose1d(input, weight, {stride}, {1}, {1});
    auto model = ctx.build_model({out.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto input_data = test_utils::random_f32(batch * in_channels * length, -0.5f, 0.5f, 42);
    auto weight_data = test_utils::random_f32(in_channels * out_channels * kernel_size, -0.5f, 0.5f, 43);

    ov::Tensor input_tensor(ov::element::f32, {batch, in_channels, length});
    ov::Tensor weight_tensor(ov::element::f32, {in_channels, out_channels, kernel_size});
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));
    std::memcpy(weight_tensor.data(), weight_data.data(), weight_data.size() * sizeof(float));

    request.set_input_tensor(0, input_tensor);
    request.set_input_tensor(1, weight_tensor);
    request.infer();

    // Verify output shape: out_length = (length - 1) * stride - 2 * pad + kernel_size + output_pad
    // = (8 - 1) * 2 - 2 * 1 + 4 + 0 = 14 - 2 + 4 = 16
    auto out_shape = request.get_output_tensor().get_shape();
    ASSERT_EQ(out_shape.size(), 3);
    ASSERT_EQ(out_shape[0], batch);
    ASSERT_EQ(out_shape[1], out_channels);
    ASSERT_EQ(out_shape[2], 16);
}

TEST(Qwen3TTSNNOps, BatchNorm) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t batch = 2;
    const size_t channels = 3;
    const size_t length = 4;
    const float eps = 1e-5f;

    auto input = ctx.parameter("input", ov::element::f32, ov::Shape{batch, channels, length});
    auto gamma = ctx.parameter("gamma", ov::element::f32, ov::Shape{channels});
    auto beta = ctx.parameter("beta", ov::element::f32, ov::Shape{channels});
    auto mean = ctx.parameter("mean", ov::element::f32, ov::Shape{channels});
    auto var = ctx.parameter("var", ov::element::f32, ov::Shape{channels});

    auto out = nn::batch_norm(input, gamma, beta, mean, var, eps);
    auto model = ctx.build_model({out.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto input_data = test_utils::make_seq(batch * channels * length, 0.1f, 0.1f);
    std::vector<float> gamma_data = {1.0f, 1.5f, 0.5f};
    std::vector<float> beta_data = {0.0f, 0.1f, -0.1f};
    std::vector<float> mean_data = {0.5f, 1.0f, 1.5f};
    std::vector<float> var_data = {0.25f, 0.5f, 1.0f};

    ov::Tensor input_tensor(ov::element::f32, {batch, channels, length});
    ov::Tensor gamma_tensor(ov::element::f32, {channels});
    ov::Tensor beta_tensor(ov::element::f32, {channels});
    ov::Tensor mean_tensor(ov::element::f32, {channels});
    ov::Tensor var_tensor(ov::element::f32, {channels});

    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));
    std::memcpy(gamma_tensor.data(), gamma_data.data(), gamma_data.size() * sizeof(float));
    std::memcpy(beta_tensor.data(), beta_data.data(), beta_data.size() * sizeof(float));
    std::memcpy(mean_tensor.data(), mean_data.data(), mean_data.size() * sizeof(float));
    std::memcpy(var_tensor.data(), var_data.data(), var_data.size() * sizeof(float));

    request.set_input_tensor(0, input_tensor);
    request.set_input_tensor(1, gamma_tensor);
    request.set_input_tensor(2, beta_tensor);
    request.set_input_tensor(3, mean_tensor);
    request.set_input_tensor(4, var_tensor);
    request.infer();

    auto expected = batch_norm_ref(input_data, gamma_data, beta_data, mean_data, var_data,
                                   batch, channels, length, eps);
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_default);
}

TEST(Qwen3TTSNNOps, AvgPool1d) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t batch = 1;
    const size_t channels = 2;
    const size_t length = 8;
    const size_t kernel_size = 2;
    const size_t stride = 2;

    auto input = ctx.parameter("input", ov::element::f32, ov::Shape{batch, channels, length});
    auto out = nn::avg_pool1d(input, kernel_size, stride, 0);
    auto model = ctx.build_model({out.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    std::vector<float> input_data = {1, 2, 3, 4, 5, 6, 7, 8,
                                     2, 4, 6, 8, 10, 12, 14, 16};
    ov::Tensor input_tensor(ov::element::f32, {batch, channels, length});
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));

    request.set_input_tensor(input_tensor);
    request.infer();

    // Expected: average of each pair
    std::vector<float> expected = {1.5f, 3.5f, 5.5f, 7.5f,
                                   3.0f, 7.0f, 11.0f, 15.0f};
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_exact);
}

// ==================== Math Ops Tests ====================

TEST(Qwen3TTSMathOps, Sin) {
    ov::genai::modeling::BuilderContext ctx;

    auto x = ctx.parameter("x", ov::element::f32, ov::Shape{2, 3});
    auto out = ov::genai::modeling::ops::sin(x);
    auto model = ctx.build_model({out.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    std::vector<float> x_data = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.14159f};
    ov::Tensor x_tensor(ov::element::f32, {2, 3});
    std::memcpy(x_tensor.data(), x_data.data(), x_data.size() * sizeof(float));

    request.set_input_tensor(x_tensor);
    request.infer();

    auto expected = sin_ref(x_data);
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_transcendental);
}

TEST(Qwen3TTSMathOps, Cos) {
    ov::genai::modeling::BuilderContext ctx;

    auto x = ctx.parameter("x", ov::element::f32, ov::Shape{2, 3});
    auto out = ov::genai::modeling::ops::cos(x);
    auto model = ctx.build_model({out.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    std::vector<float> x_data = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.14159f};
    ov::Tensor x_tensor(ov::element::f32, {2, 3});
    std::memcpy(x_tensor.data(), x_data.data(), x_data.size() * sizeof(float));

    request.set_input_tensor(x_tensor);
    request.infer();

    auto expected = cos_ref(x_data);
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_transcendental);
}

TEST(Qwen3TTSMathOps, ReduceSum) {
    ov::genai::modeling::BuilderContext ctx;

    auto x = ctx.parameter("x", ov::element::f32, ov::Shape{2, 3, 4});
    auto out = ov::genai::modeling::ops::reduce_sum(x, 1, true);
    auto model = ctx.build_model({out.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto x_data = test_utils::make_seq(24, 1.0f, 1.0f);
    ov::Tensor x_tensor(ov::element::f32, {2, 3, 4});
    std::memcpy(x_tensor.data(), x_data.data(), x_data.size() * sizeof(float));

    request.set_input_tensor(x_tensor);
    request.infer();

    auto expected = reduce_sum_ref(x_data, {2, 3, 4}, 1, true);
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_exact);
}

TEST(Qwen3TTSMathOps, ReduceSumMultipleAxes) {
    ov::genai::modeling::BuilderContext ctx;

    auto x = ctx.parameter("x", ov::element::f32, ov::Shape{2, 3, 4});
    auto out = ov::genai::modeling::ops::reduce_sum(x, {1, 2}, true);
    auto model = ctx.build_model({out.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto x_data = test_utils::make_seq(24, 1.0f, 1.0f);
    ov::Tensor x_tensor(ov::element::f32, {2, 3, 4});
    std::memcpy(x_tensor.data(), x_data.data(), x_data.size() * sizeof(float));

    request.set_input_tensor(x_tensor);
    request.infer();

    // Sum over axes 1 and 2: each of the 2 batches sums to a single value
    // Batch 0: sum(1..12) = 78, Batch 1: sum(13..24) = 222
    auto out_shape = request.get_output_tensor().get_shape();
    ASSERT_EQ(out_shape.size(), 3);
    ASSERT_EQ(out_shape[0], 2);
    ASSERT_EQ(out_shape[1], 1);
    ASSERT_EQ(out_shape[2], 1);

    std::vector<float> expected = {78.0f, 222.0f};
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_exact);
}

// ==================== Tensor Ops Tests ====================

TEST(Qwen3TTSTensorOps, SplitEqual) {
    ov::genai::modeling::BuilderContext ctx;

    auto x = ctx.parameter("x", ov::element::f32, ov::Shape{1, 6, 4});
    auto splits = tensor::split(x, 3, 1);  // Split into 3 equal parts along axis 1

    std::vector<ov::Output<ov::Node>> outputs;
    for (auto& s : splits) {
        outputs.push_back(s.output());
    }
    auto model = ctx.build_model(outputs);

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto x_data = test_utils::make_seq(24, 1.0f, 1.0f);
    ov::Tensor x_tensor(ov::element::f32, {1, 6, 4});
    std::memcpy(x_tensor.data(), x_data.data(), x_data.size() * sizeof(float));

    request.set_input_tensor(x_tensor);
    request.infer();

    // Verify shapes
    ASSERT_EQ(request.get_output_tensor(0).get_shape(), (ov::Shape{1, 2, 4}));
    ASSERT_EQ(request.get_output_tensor(1).get_shape(), (ov::Shape{1, 2, 4}));
    ASSERT_EQ(request.get_output_tensor(2).get_shape(), (ov::Shape{1, 2, 4}));

    // Verify values
    std::vector<float> expected0 = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> expected1 = {9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<float> expected2 = {17, 18, 19, 20, 21, 22, 23, 24};

    test_utils::expect_tensor_near(request.get_output_tensor(0), expected0, test_utils::k_tol_exact);
    test_utils::expect_tensor_near(request.get_output_tensor(1), expected1, test_utils::k_tol_exact);
    test_utils::expect_tensor_near(request.get_output_tensor(2), expected2, test_utils::k_tol_exact);
}

TEST(Qwen3TTSTensorOps, SplitVariable) {
    ov::genai::modeling::BuilderContext ctx;

    auto x = ctx.parameter("x", ov::element::f32, ov::Shape{1, 6, 4});
    auto splits = tensor::split(x, {1, 2, 3}, 1);  // Split into parts of size 1, 2, 3

    std::vector<ov::Output<ov::Node>> outputs;
    for (auto& s : splits) {
        outputs.push_back(s.output());
    }
    auto model = ctx.build_model(outputs);

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto x_data = test_utils::make_seq(24, 1.0f, 1.0f);
    ov::Tensor x_tensor(ov::element::f32, {1, 6, 4});
    std::memcpy(x_tensor.data(), x_data.data(), x_data.size() * sizeof(float));

    request.set_input_tensor(x_tensor);
    request.infer();

    // Verify shapes
    ASSERT_EQ(request.get_output_tensor(0).get_shape(), (ov::Shape{1, 1, 4}));
    ASSERT_EQ(request.get_output_tensor(1).get_shape(), (ov::Shape{1, 2, 4}));
    ASSERT_EQ(request.get_output_tensor(2).get_shape(), (ov::Shape{1, 3, 4}));

    // Verify values
    std::vector<float> expected0 = {1, 2, 3, 4};
    std::vector<float> expected1 = {5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> expected2 = {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

    test_utils::expect_tensor_near(request.get_output_tensor(0), expected0, test_utils::k_tol_exact);
    test_utils::expect_tensor_near(request.get_output_tensor(1), expected1, test_utils::k_tol_exact);
    test_utils::expect_tensor_near(request.get_output_tensor(2), expected2, test_utils::k_tol_exact);
}

TEST(Qwen3TTSTensorOps, Flip) {
    ov::genai::modeling::BuilderContext ctx;

    auto x = ctx.parameter("x", ov::element::f32, ov::Shape{2, 3});
    auto out = tensor::flip(x, 1);  // Flip along axis 1
    auto model = ctx.build_model({out.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    std::vector<float> x_data = {1, 2, 3, 4, 5, 6};
    ov::Tensor x_tensor(ov::element::f32, {2, 3});
    std::memcpy(x_tensor.data(), x_data.data(), x_data.size() * sizeof(float));

    request.set_input_tensor(x_tensor);
    request.infer();

    auto expected = flip_ref(x_data, {2, 3}, 1);
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_exact);
}

TEST(Qwen3TTSTensorOps, Flip3D) {
    ov::genai::modeling::BuilderContext ctx;

    auto x = ctx.parameter("x", ov::element::f32, ov::Shape{2, 3, 4});
    auto out = tensor::flip(x, 2);  // Flip along last axis
    auto model = ctx.build_model({out.output()});

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto x_data = test_utils::make_seq(24, 1.0f, 1.0f);
    ov::Tensor x_tensor(ov::element::f32, {2, 3, 4});
    std::memcpy(x_tensor.data(), x_data.data(), x_data.size() * sizeof(float));

    request.set_input_tensor(x_tensor);
    request.infer();

    auto expected = flip_ref(x_data, {2, 3, 4}, 2);
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_exact);
}
