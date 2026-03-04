// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>
#include "modeling/builder_context.hpp"
#include "modeling/models/qwen3_tts/modeling_qwen3_tts_speech_decoder.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace test_utils = ov::genai::modeling::tests;

namespace {

//===----------------------------------------------------------------------===//
// Reference Implementations
//===----------------------------------------------------------------------===//

// Reference: SnakeBeta activation: x + 1/beta * sin^2(x * alpha)
// alpha and beta are stored as log values
std::vector<float> snake_beta_ref(const std::vector<float>& x,
                                  const std::vector<float>& log_alpha,
                                  const std::vector<float>& log_beta,
                                  size_t batch, size_t channels, size_t seq_len) {
    std::vector<float> out(batch * channels * seq_len);
    
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            float alpha = std::exp(log_alpha[c]);
            float beta = std::exp(log_beta[c]);
            
            for (size_t s = 0; s < seq_len; ++s) {
                size_t idx = (b * channels + c) * seq_len + s;
                float val = x[idx];
                float sin_val = std::sin(val * alpha);
                float sin_sq = sin_val * sin_val;
                out[idx] = val + sin_sq / beta;
            }
        }
    }
    return out;
}

// Reference: SwiGLU MLP
std::vector<float> swiglu_mlp_ref(const std::vector<float>& x,
                                  const std::vector<float>& gate_w,
                                  const std::vector<float>& up_w,
                                  const std::vector<float>& down_w,
                                  size_t batch, size_t seq_len,
                                  size_t hidden, size_t intermediate) {
    std::vector<float> gate(batch * seq_len * intermediate);
    std::vector<float> up(batch * seq_len * intermediate);

    for (size_t b = 0; b < batch; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            for (size_t i = 0; i < intermediate; ++i) {
                float gate_acc = 0.0f;
                float up_acc = 0.0f;
                for (size_t h = 0; h < hidden; ++h) {
                    size_t x_idx = (b * seq_len + t) * hidden + h;
                    gate_acc += x[x_idx] * gate_w[i * hidden + h];
                    up_acc += x[x_idx] * up_w[i * hidden + h];
                }
                size_t out_idx = (b * seq_len + t) * intermediate + i;
                float silu_gate = gate_acc / (1.0f + std::exp(-gate_acc));
                gate[out_idx] = silu_gate * up_acc;
            }
        }
    }

    std::vector<float> out(batch * seq_len * hidden);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            for (size_t h = 0; h < hidden; ++h) {
                float acc = 0.0f;
                for (size_t i = 0; i < intermediate; ++i) {
                    acc += gate[(b * seq_len + t) * intermediate + i] * down_w[h * intermediate + i];
                }
                out[(b * seq_len + t) * hidden + h] = acc;
            }
        }
    }
    return out;
}

// Reference: RVQ Dequantizer - single codebook gather + project
std::vector<float> rvq_single_layer_ref(const std::vector<int64_t>& codes,
                                        const std::vector<float>& codebook,
                                        const std::vector<float>& proj_weight,
                                        size_t batch, size_t seq_len,
                                        size_t codebook_dim, size_t output_dim) {
    std::vector<float> out(batch * seq_len * output_dim, 0.0f);
    
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            int64_t code = codes[b * seq_len + s];
            
            // Gather embedding
            std::vector<float> embed(codebook_dim);
            for (size_t d = 0; d < codebook_dim; ++d) {
                embed[d] = codebook[code * codebook_dim + d];
            }
            
            // Project: output_dim x codebook_dim
            for (size_t o = 0; o < output_dim; ++o) {
                float acc = 0.0f;
                for (size_t d = 0; d < codebook_dim; ++d) {
                    acc += embed[d] * proj_weight[o * codebook_dim + d];
                }
                out[(b * seq_len + s) * output_dim + o] = acc;
            }
        }
    }
    return out;
}

// Reference: Conv1D (for simple cases)
std::vector<float> conv1d_ref(const std::vector<float>& x,
                              const std::vector<float>& weight,
                              const std::vector<float>& bias,
                              size_t batch, size_t in_ch, size_t out_ch,
                              size_t seq_len, size_t kernel, size_t pad) {
    size_t out_len = seq_len + 2 * pad - kernel + 1;
    std::vector<float> out(batch * out_ch * out_len, 0.0f);
    
    // Pad input
    std::vector<float> x_padded(batch * in_ch * (seq_len + 2 * pad), 0.0f);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < in_ch; ++c) {
            for (size_t s = 0; s < seq_len; ++s) {
                x_padded[(b * in_ch + c) * (seq_len + 2 * pad) + pad + s] = 
                    x[(b * in_ch + c) * seq_len + s];
            }
        }
    }
    
    // Conv
    for (size_t b = 0; b < batch; ++b) {
        for (size_t oc = 0; oc < out_ch; ++oc) {
            for (size_t os = 0; os < out_len; ++os) {
                float acc = bias[oc];
                for (size_t ic = 0; ic < in_ch; ++ic) {
                    for (size_t k = 0; k < kernel; ++k) {
                        acc += x_padded[(b * in_ch + ic) * (seq_len + 2 * pad) + os + k] *
                               weight[(oc * in_ch + ic) * kernel + k];
                    }
                }
                out[(b * out_ch + oc) * out_len + os] = acc;
            }
        }
    }
    return out;
}

}  // namespace

//===----------------------------------------------------------------------===//
// Test: SnakeBeta Activation
//===----------------------------------------------------------------------===//

TEST(Qwen3TTSSpeechDecoderSnakeBeta, MatchesReference) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t batch = 1;
    const size_t channels = 4;
    const size_t seq_len = 8;

    // Alpha and beta are stored as log values
    auto log_alpha = test_utils::make_seq(channels, 0.0f, 0.1f);  // log(alpha)
    auto log_beta = test_utils::make_seq(channels, 0.5f, 0.1f);   // log(beta)

    test_utils::DummyWeightSource weights;
    weights.add("snake.alpha", test_utils::make_tensor(log_alpha, {channels}));
    weights.add("snake.beta", test_utils::make_tensor(log_beta, {channels}));

    test_utils::DummyWeightFinalizer finalizer;

    ov::genai::modeling::models::SnakeBetaActivation snake(ctx, "snake");
    ov::genai::modeling::weights::load_model(snake, weights, finalizer);

    auto input = ctx.parameter("input", ov::element::f32, 
                               ov::PartialShape{batch, channels, seq_len});
    auto output = snake.forward(input);
    auto ov_model = ctx.build_model({output.output()});

    ov::Core core;
    auto compiled = core.compile_model(ov_model, "GPU");
    auto request = compiled.create_infer_request();

    // Input data in NCL format
    std::vector<float> input_data(batch * channels * seq_len);
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = static_cast<float>(i) * 0.1f - 1.5f;
    }

    ov::Tensor input_tensor(ov::element::f32, {batch, channels, seq_len});
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));
    request.set_input_tensor(input_tensor);

    request.infer();

    auto expected = snake_beta_ref(input_data, log_alpha, log_beta, batch, channels, seq_len);
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_transcendental);
}

//===----------------------------------------------------------------------===//
// Test: PreTransformerMLP (SwiGLU)
//===----------------------------------------------------------------------===//

TEST(Qwen3TTSSpeechDecoderPreTransformerMLP, MatchesReference) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t batch = 1;
    const size_t seq_len = 2;
    const size_t hidden = 8;
    const size_t intermediate = 12;

    const ov::Shape gate_shape{intermediate, hidden};
    const ov::Shape up_shape{intermediate, hidden};
    const ov::Shape down_shape{hidden, intermediate};

    const auto gate_w = test_utils::make_seq(intermediate * hidden, 0.01f, 0.02f);
    const auto up_w = test_utils::make_seq(intermediate * hidden, 0.015f, 0.02f);
    const auto down_w = test_utils::make_seq(hidden * intermediate, 0.02f, 0.02f);

    test_utils::DummyWeightSource weights;
    weights.add("mlp.gate_proj.weight", test_utils::make_tensor(gate_w, gate_shape));
    weights.add("mlp.up_proj.weight", test_utils::make_tensor(up_w, up_shape));
    weights.add("mlp.down_proj.weight", test_utils::make_tensor(down_w, down_shape));

    test_utils::DummyWeightFinalizer finalizer;

    ov::genai::modeling::models::SpeechDecoderConfig cfg;
    cfg.transformer_hidden = static_cast<int32_t>(hidden);
    cfg.transformer_intermediate = static_cast<int32_t>(intermediate);

    ov::genai::modeling::models::PreTransformerMLP mlp(ctx, "mlp", cfg);
    ov::genai::modeling::weights::load_model(mlp, weights, finalizer);

    auto input = ctx.parameter("input", ov::element::f32, 
                               ov::PartialShape{batch, seq_len, hidden});
    auto output = mlp.forward(input);
    auto ov_model = ctx.build_model({output.output()});

    ov::Core core;
    auto compiled = core.compile_model(ov_model, "GPU");
    auto request = compiled.create_infer_request();

    const std::vector<float> input_data = {
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,
        0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f,
    };
    ov::Tensor input_tensor(ov::element::f32, {batch, seq_len, hidden});
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));
    request.set_input_tensor(input_tensor);

    request.infer();

    auto expected = swiglu_mlp_ref(input_data, gate_w, up_w, down_w, 
                                   batch, seq_len, hidden, intermediate);
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_default);
}

//===----------------------------------------------------------------------===//
// Test: RVQ Dequantizer - First Codebook
// Tests that first codebook gather + projection works correctly
//===----------------------------------------------------------------------===//

TEST(Qwen3TTSSpeechDecoderRVQDequantizer, FirstCodebookMatchesReference) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t batch = 1;
    const size_t seq_len = 3;
    const size_t vocab_size = 16;
    const size_t codebook_dim = 8;
    const size_t output_dim = 12;

    // First codebook weights
    auto first_codebook = test_utils::make_seq(vocab_size * codebook_dim, 0.1f, 0.01f);
    auto first_proj = test_utils::make_seq(output_dim * codebook_dim, 0.02f, 0.005f);

    test_utils::DummyWeightSource weights;
    
    // Add first codebook
    weights.add("dequant.rvq_first.vq.layers[0]._codebook.embed",
        test_utils::make_tensor(first_codebook, {vocab_size, codebook_dim}));
    weights.add("dequant.rvq_first.output_proj.weight",
        test_utils::make_tensor(first_proj, {output_dim, codebook_dim, 1}));

    // Add rest codebooks (15 layers) - needed for model construction
    for (int i = 0; i < 15; ++i) {
        auto rest_cb = test_utils::make_seq(vocab_size * codebook_dim, 0.0f, 0.0f);
        weights.add("dequant.rvq_rest.vq.layers[" + std::to_string(i) + "]._codebook.embed",
            test_utils::make_tensor(rest_cb, {vocab_size, codebook_dim}));
    }
    auto rest_proj = test_utils::make_seq(output_dim * codebook_dim, 0.0f, 0.0f);
    weights.add("dequant.rvq_rest.output_proj.weight",
        test_utils::make_tensor(rest_proj, {output_dim, codebook_dim, 1}));

    test_utils::DummyWeightFinalizer finalizer;

    ov::genai::modeling::models::SpeechDecoderConfig cfg;
    cfg.num_quantizers = 16;
    cfg.codebook_size = static_cast<int32_t>(vocab_size);
    cfg.codebook_dim = static_cast<int32_t>(codebook_dim);
    cfg.latent_dim = static_cast<int32_t>(output_dim);  // rvq output goes to latent_dim

    ov::genai::modeling::models::RVQDequantizer dequant(ctx, "dequant", cfg);
    ov::genai::modeling::weights::load_model(dequant, weights, finalizer);

    // Input: [batch, 16, seq_len] - only layer 0 has non-zero codes
    auto codes = ctx.parameter("codes", ov::element::i64, 
                               ov::PartialShape{batch, 16, seq_len});
    auto output = dequant.forward(codes);
    auto ov_model = ctx.build_model({output.output()});

    ov::Core core;
    auto compiled = core.compile_model(ov_model, "GPU");
    auto request = compiled.create_infer_request();

    // Create codes with only layer 0 having meaningful values
    std::vector<int64_t> codes_data(batch * 16 * seq_len, 0);
    codes_data[0] = 2;  // layer 0, pos 0
    codes_data[1] = 5;  // layer 0, pos 1
    codes_data[2] = 10; // layer 0, pos 2

    ov::Tensor codes_tensor(ov::element::i64, {batch, 16, seq_len});
    std::memcpy(codes_tensor.data(), codes_data.data(), codes_data.size() * sizeof(int64_t));
    request.set_input_tensor(codes_tensor);

    request.infer();

    // Compute reference for first codebook only
    std::vector<int64_t> first_codes = {2, 5, 10};
    auto expected = rvq_single_layer_ref(first_codes, first_codebook, first_proj,
                                         batch, seq_len, codebook_dim, output_dim);
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_default);
}

//===----------------------------------------------------------------------===//
// Test: RVQ Dequantizer - Multiple Codebooks Sum
// Tests that multiple codebook outputs are correctly summed
//===----------------------------------------------------------------------===//

TEST(Qwen3TTSSpeechDecoderRVQDequantizer, MultipleCodebooksSum) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t batch = 1;
    const size_t seq_len = 2;
    const size_t vocab_size = 8;
    const size_t codebook_dim = 4;
    const size_t output_dim = 6;

    // First codebook: all 0.1
    std::vector<float> first_codebook(vocab_size * codebook_dim, 0.1f);
    // First proj: identity-like (scaled)
    std::vector<float> first_proj(output_dim * codebook_dim, 0.0f);
    for (size_t i = 0; i < std::min(output_dim, codebook_dim); ++i) {
        first_proj[i * codebook_dim + i] = 1.0f;
    }

    // Rest codebooks: all 0.2
    std::vector<float> rest_codebook(vocab_size * codebook_dim, 0.2f);
    // Rest proj: same as first
    std::vector<float> rest_proj = first_proj;

    test_utils::DummyWeightSource weights;
    
    weights.add("dequant.rvq_first.vq.layers[0]._codebook.embed",
        test_utils::make_tensor(first_codebook, {vocab_size, codebook_dim}));
    weights.add("dequant.rvq_first.output_proj.weight",
        test_utils::make_tensor(first_proj, {output_dim, codebook_dim, 1}));

    for (int i = 0; i < 15; ++i) {
        weights.add("dequant.rvq_rest.vq.layers[" + std::to_string(i) + "]._codebook.embed",
            test_utils::make_tensor(rest_codebook, {vocab_size, codebook_dim}));
    }
    weights.add("dequant.rvq_rest.output_proj.weight",
        test_utils::make_tensor(rest_proj, {output_dim, codebook_dim, 1}));

    test_utils::DummyWeightFinalizer finalizer;

    ov::genai::modeling::models::SpeechDecoderConfig cfg;
    cfg.num_quantizers = 16;
    cfg.codebook_size = static_cast<int32_t>(vocab_size);
    cfg.codebook_dim = static_cast<int32_t>(codebook_dim);
    cfg.latent_dim = static_cast<int32_t>(output_dim);  // rvq output goes to latent_dim

    ov::genai::modeling::models::RVQDequantizer dequant(ctx, "dequant", cfg);
    ov::genai::modeling::weights::load_model(dequant, weights, finalizer);

    auto codes = ctx.parameter("codes", ov::element::i64, 
                               ov::PartialShape{batch, 16, seq_len});
    auto output = dequant.forward(codes);
    auto ov_model = ctx.build_model({output.output()});

    ov::Core core;
    auto compiled = core.compile_model(ov_model, "GPU");
    auto request = compiled.create_infer_request();

    // All codes = 0 for simplicity
    std::vector<int64_t> codes_data(batch * 16 * seq_len, 0);
    ov::Tensor codes_tensor(ov::element::i64, {batch, 16, seq_len});
    std::memcpy(codes_tensor.data(), codes_data.data(), codes_data.size() * sizeof(int64_t));
    request.set_input_tensor(codes_tensor);

    request.infer();

    // Expected: first codebook contrib (0.1 * 1.0 = 0.1) + 15 * rest codebook contrib (0.2 * 1.0 = 0.2)
    // Total: 0.1 + 15 * 0.2 = 0.1 + 3.0 = 3.1 for first 4 dims, 0 for rest
    auto out_tensor = request.get_output_tensor();
    auto* out_data = out_tensor.data<float>();
    
    EXPECT_EQ(out_tensor.get_shape()[0], batch);
    EXPECT_EQ(out_tensor.get_shape()[1], seq_len);
    EXPECT_EQ(out_tensor.get_shape()[2], output_dim);

    // Check that outputs are summed correctly
    for (size_t s = 0; s < seq_len; ++s) {
        for (size_t d = 0; d < std::min(output_dim, codebook_dim); ++d) {
            float expected_val = 0.1f + 15.0f * 0.2f;  // = 3.1
            EXPECT_NEAR(out_data[s * output_dim + d], expected_val, 0.01f)
                << "Mismatch at seq=" << s << ", dim=" << d;
        }
    }
}

//===----------------------------------------------------------------------===//
// Test: ConvNeXt Block Graph Structure
//===----------------------------------------------------------------------===//

TEST(Qwen3TTSSpeechDecoderConvNeXtBlock, GraphStructure) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t batch = 1;
    const size_t channels = 8;
    const size_t seq_len = 16;

    test_utils::DummyWeightSource weights;

    // ConvNeXt block weights
    // Depthwise conv: [channels, 1, 7]
    weights.add("convnext.dwconv.conv.weight",
        test_utils::make_tensor(test_utils::make_seq(channels * 7, 0.1f, 0.01f), 
                                {channels, 1, 7}));
    weights.add("convnext.dwconv.conv.bias",
        test_utils::make_tensor(test_utils::make_seq(channels, 0.0f, 0.0f), {channels}));

    // LayerNorm
    weights.add("convnext.norm.weight",
        test_utils::make_tensor(std::vector<float>(channels, 1.0f), {channels}));
    weights.add("convnext.norm.bias",
        test_utils::make_tensor(std::vector<float>(channels, 0.0f), {channels}));

    // Pointwise convs (expand 4x then contract)
    size_t expanded = channels * 4;
    weights.add("convnext.pwconv1.weight",
        test_utils::make_tensor(test_utils::make_seq(expanded * channels, 0.01f, 0.001f),
                                {expanded, channels}));
    weights.add("convnext.pwconv1.bias",
        test_utils::make_tensor(std::vector<float>(expanded, 0.0f), {expanded}));
    weights.add("convnext.pwconv2.weight",
        test_utils::make_tensor(test_utils::make_seq(channels * expanded, 0.01f, 0.001f),
                                {channels, expanded}));
    weights.add("convnext.pwconv2.bias",
        test_utils::make_tensor(std::vector<float>(channels, 0.0f), {channels}));

    // Gamma
    weights.add("convnext.gamma",
        test_utils::make_tensor(std::vector<float>(channels, 0.1f), {channels}));

    test_utils::DummyWeightFinalizer finalizer;

    ov::genai::modeling::models::ConvNeXtBlock convnext(ctx, "convnext", 
                                                        static_cast<int32_t>(channels));
    ov::genai::modeling::weights::load_model(convnext, weights, finalizer);

    auto input = ctx.parameter("input", ov::element::f32, 
                               ov::PartialShape{batch, channels, seq_len});
    auto output = convnext.forward(input);
    auto ov_model = ctx.build_model({output.output()});

    ASSERT_NE(ov_model, nullptr);
    EXPECT_EQ(ov_model->inputs().size(), 1);
    EXPECT_EQ(ov_model->outputs().size(), 1);

    // Output should have same shape as input (residual connection)
    auto output_shape = ov_model->output(0).get_partial_shape();
    EXPECT_TRUE(output_shape.rank().is_static());
    EXPECT_EQ(output_shape.rank().get_length(), 3);
    EXPECT_EQ(output_shape[1].get_length(), static_cast<int64_t>(channels));

    // Run inference
    ov::Core core;
    auto compiled = core.compile_model(ov_model, "GPU");
    auto request = compiled.create_infer_request();

    std::vector<float> input_data(batch * channels * seq_len, 0.5f);
    ov::Tensor input_tensor(ov::element::f32, {batch, channels, seq_len});
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));
    request.set_input_tensor(input_tensor);

    EXPECT_NO_THROW(request.infer());
    
    auto out_tensor = request.get_output_tensor();
    EXPECT_EQ(out_tensor.get_shape()[0], batch);
    EXPECT_EQ(out_tensor.get_shape()[1], channels);
    EXPECT_EQ(out_tensor.get_shape()[2], seq_len);
}

//===----------------------------------------------------------------------===//
// Test: Residual Unit Graph Structure
//===----------------------------------------------------------------------===//

TEST(Qwen3TTSSpeechDecoderResidualUnit, GraphStructure) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t batch = 1;
    const size_t channels = 8;
    const size_t seq_len = 16;
    const int32_t dilation = 3;

    test_utils::DummyWeightSource weights;

    // SnakeBeta activation 1
    weights.add("resunit.act1.alpha",
        test_utils::make_tensor(std::vector<float>(channels, 0.0f), {channels}));
    weights.add("resunit.act1.beta",
        test_utils::make_tensor(std::vector<float>(channels, 0.5f), {channels}));

    // SnakeBeta activation 2
    weights.add("resunit.act2.alpha",
        test_utils::make_tensor(std::vector<float>(channels, 0.0f), {channels}));
    weights.add("resunit.act2.beta",
        test_utils::make_tensor(std::vector<float>(channels, 0.5f), {channels}));

    // Conv1: dilated causal conv (kernel=7)
    weights.add("resunit.conv1.conv.weight",
        test_utils::make_tensor(test_utils::make_seq(channels * channels * 7, 0.01f, 0.001f),
                                {channels, channels, 7}));
    weights.add("resunit.conv1.conv.bias",
        test_utils::make_tensor(std::vector<float>(channels, 0.0f), {channels}));

    // Conv2: kernel=1
    weights.add("resunit.conv2.conv.weight",
        test_utils::make_tensor(test_utils::make_seq(channels * channels, 0.01f, 0.001f),
                                {channels, channels, 1}));
    weights.add("resunit.conv2.conv.bias",
        test_utils::make_tensor(std::vector<float>(channels, 0.0f), {channels}));

    test_utils::DummyWeightFinalizer finalizer;

    ov::genai::modeling::models::ResidualUnit resunit(ctx, "resunit", 
                                                       static_cast<int32_t>(channels), dilation);
    ov::genai::modeling::weights::load_model(resunit, weights, finalizer);

    auto input = ctx.parameter("input", ov::element::f32, 
                               ov::PartialShape{batch, channels, seq_len});
    auto output = resunit.forward(input);
    auto ov_model = ctx.build_model({output.output()});

    ASSERT_NE(ov_model, nullptr);

    // Run inference
    ov::Core core;
    auto compiled = core.compile_model(ov_model, "GPU");
    auto request = compiled.create_infer_request();

    std::vector<float> input_data(batch * channels * seq_len, 0.3f);
    ov::Tensor input_tensor(ov::element::f32, {batch, channels, seq_len});
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));
    request.set_input_tensor(input_tensor);

    EXPECT_NO_THROW(request.infer());
    
    auto out_tensor = request.get_output_tensor();
    // Output should preserve sequence length due to causal padding
    EXPECT_EQ(out_tensor.get_shape()[0], batch);
    EXPECT_EQ(out_tensor.get_shape()[1], channels);
    EXPECT_EQ(out_tensor.get_shape()[2], seq_len);
}

//===----------------------------------------------------------------------===//
// Test: Decoder Block Upsample Factor
// Verifies that the decoder block correctly upsamples by the expected factor
//===----------------------------------------------------------------------===//

TEST(Qwen3TTSSpeechDecoderDecoderBlock, UpsampleFactor) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t batch = 1;
    const size_t in_channels = 16;
    const size_t out_channels = 8;
    const int32_t upsample_rate = 4;
    const size_t seq_len = 10;

    test_utils::DummyWeightSource weights;

    // SnakeBeta (block.0)
    weights.add("decoder_block.block.0.alpha",
        test_utils::make_tensor(std::vector<float>(in_channels, 0.0f), {in_channels}));
    weights.add("decoder_block.block.0.beta",
        test_utils::make_tensor(std::vector<float>(in_channels, 0.5f), {in_channels}));

    // Transposed conv (block.1): kernel = 2 * upsample_rate = 8
    size_t kernel = 2 * upsample_rate;
    weights.add("decoder_block.block.1.conv.weight",
        test_utils::make_tensor(test_utils::make_seq(in_channels * out_channels * kernel, 0.01f, 0.001f),
                                {in_channels, out_channels, kernel}));
    weights.add("decoder_block.block.1.conv.bias",
        test_utils::make_tensor(std::vector<float>(out_channels, 0.0f), {out_channels}));

    // 3 Residual units (block.2, block.3, block.4) with dilations 1, 3, 9
    std::vector<int32_t> dilations = {1, 3, 9};
    for (int i = 0; i < 3; ++i) {
        std::string prefix = "decoder_block.block." + std::to_string(i + 2);
        
        // SnakeBeta activations
        weights.add(prefix + ".act1.alpha",
            test_utils::make_tensor(std::vector<float>(out_channels, 0.0f), {out_channels}));
        weights.add(prefix + ".act1.beta",
            test_utils::make_tensor(std::vector<float>(out_channels, 0.5f), {out_channels}));
        weights.add(prefix + ".act2.alpha",
            test_utils::make_tensor(std::vector<float>(out_channels, 0.0f), {out_channels}));
        weights.add(prefix + ".act2.beta",
            test_utils::make_tensor(std::vector<float>(out_channels, 0.5f), {out_channels}));
        
        // Convs
        weights.add(prefix + ".conv1.conv.weight",
            test_utils::make_tensor(test_utils::make_seq(out_channels * out_channels * 7, 0.01f, 0.0001f),
                                    {out_channels, out_channels, 7}));
        weights.add(prefix + ".conv1.conv.bias",
            test_utils::make_tensor(std::vector<float>(out_channels, 0.0f), {out_channels}));
        weights.add(prefix + ".conv2.conv.weight",
            test_utils::make_tensor(test_utils::make_seq(out_channels * out_channels, 0.01f, 0.0001f),
                                    {out_channels, out_channels, 1}));
        weights.add(prefix + ".conv2.conv.bias",
            test_utils::make_tensor(std::vector<float>(out_channels, 0.0f), {out_channels}));
    }

    test_utils::DummyWeightFinalizer finalizer;

    ov::genai::modeling::models::DecoderBlock decoder_block(
        ctx, "decoder_block", 
        static_cast<int32_t>(in_channels), static_cast<int32_t>(out_channels), upsample_rate);
    ov::genai::modeling::weights::load_model(decoder_block, weights, finalizer);

    auto input = ctx.parameter("input", ov::element::f32, 
                               ov::PartialShape{batch, in_channels, seq_len});
    auto output = decoder_block.forward(input);
    auto ov_model = ctx.build_model({output.output()});

    ov::Core core;
    auto compiled = core.compile_model(ov_model, "GPU");
    auto request = compiled.create_infer_request();

    std::vector<float> input_data(batch * in_channels * seq_len, 0.2f);
    ov::Tensor input_tensor(ov::element::f32, {batch, in_channels, seq_len});
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));
    request.set_input_tensor(input_tensor);

    request.infer();

    auto out_tensor = request.get_output_tensor();
    EXPECT_EQ(out_tensor.get_shape()[0], batch);
    EXPECT_EQ(out_tensor.get_shape()[1], out_channels);
    // Output length should be input * upsample_rate
    EXPECT_EQ(out_tensor.get_shape()[2], seq_len * upsample_rate);
}

//===----------------------------------------------------------------------===//
// Test: Speech Decoder Model Audio Length Calculation
//===----------------------------------------------------------------------===//

TEST(Qwen3TTSSpeechDecoderModel, AudioLengthCalculation) {
    ov::genai::modeling::models::SpeechDecoderConfig cfg;
    
    // Default config: pre_upsample = [2, 2] = 4x, decoder = [8, 5, 4, 3] = 480x
    // Total: 4 * 480 = 1920x
    
    ov::genai::modeling::BuilderContext ctx;
    
    // We can't easily test the full model without all weights, but we can verify
    // the audio length calculation formula
    int64_t total_upsample = 1;
    for (auto r : cfg.pre_upsample_ratios) total_upsample *= r;
    for (auto r : cfg.decoder_upsample_rates) total_upsample *= r;
    
    EXPECT_EQ(total_upsample, 1920);
    
    // Test various input lengths
    EXPECT_EQ(10 * total_upsample, 19200);   // 10 codes -> 19200 samples
    EXPECT_EQ(100 * total_upsample, 192000); // 100 codes -> 192000 samples (8 seconds at 24kHz)
}
