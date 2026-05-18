// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>
#include "modeling/builder_context.hpp"
#include "modeling/models/qwen3_tts/modeling_qwen3_tts_talker.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace test_utils = ov::genai::modeling::tests;

namespace {

//===----------------------------------------------------------------------===//
// Reference Implementations
//===----------------------------------------------------------------------===//

// Reference: text_projection (ResizeMLP): fc1 -> silu -> fc2
std::vector<float> text_projection_ref(const std::vector<float>& x,
                                       const std::vector<float>& fc1_w,
                                       const std::vector<float>& fc1_b,
                                       const std::vector<float>& fc2_w,
                                       const std::vector<float>& fc2_b,
                                       size_t batch,
                                       size_t seq_len,
                                       size_t in_features,
                                       size_t hidden_features,
                                       size_t out_features) {
    std::vector<float> fc1_out(batch * seq_len * hidden_features);
    // fc1: [B, T, in] @ [hidden, in]^T -> [B, T, hidden]
    for (size_t b = 0; b < batch; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            for (size_t h = 0; h < hidden_features; ++h) {
                float acc = fc1_b[h];
                for (size_t i = 0; i < in_features; ++i) {
                    acc += x[(b * seq_len + t) * in_features + i] *
                           fc1_w[h * in_features + i];
                }
                fc1_out[(b * seq_len + t) * hidden_features + h] = acc;
            }
        }
    }
    // silu activation
    for (auto& v : fc1_out) {
        v = v / (1.0f + std::exp(-v));  // silu(x) = x * sigmoid(x)
    }
    // fc2: [B, T, hidden] @ [out, hidden]^T -> [B, T, out]
    std::vector<float> out(batch * seq_len * out_features);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            for (size_t o = 0; o < out_features; ++o) {
                float acc = fc2_b[o];
                for (size_t h = 0; h < hidden_features; ++h) {
                    acc += fc1_out[(b * seq_len + t) * hidden_features + h] *
                           fc2_w[o * hidden_features + h];
                }
                out[(b * seq_len + t) * out_features + o] = acc;
            }
        }
    }
    return out;
}

// Reference: SwiGLU MLP (gate * up @ down)
std::vector<float> swiglu_mlp_ref(const std::vector<float>& x,
                                  const std::vector<float>& gate_w,
                                  const std::vector<float>& up_w,
                                  const std::vector<float>& down_w,
                                  size_t batch,
                                  size_t seq_len,
                                  size_t hidden,
                                  size_t intermediate) {
    // gate = x @ gate_proj^T, up = x @ up_proj^T
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
                // silu(gate) * up
                float silu_gate = gate_acc / (1.0f + std::exp(-gate_acc));
                gate[out_idx] = silu_gate * up_acc;
            }
        }
    }

    // down = intermediate @ down_proj^T
    std::vector<float> out(batch * seq_len * hidden);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            for (size_t h = 0; h < hidden; ++h) {
                float acc = 0.0f;
                for (size_t i = 0; i < intermediate; ++i) {
                    acc += gate[(b * seq_len + t) * intermediate + i] *
                           down_w[h * intermediate + i];
                }
                out[(b * seq_len + t) * hidden + h] = acc;
            }
        }
    }
    return out;
}

// Reference: embedding lookup
std::vector<float> embedding_ref(const std::vector<int64_t>& ids,
                                 const std::vector<float>& weight,
                                 size_t batch,
                                 size_t seq_len,
                                 size_t embed_dim) {
    std::vector<float> out(batch * seq_len * embed_dim);
    for (size_t i = 0; i < batch * seq_len; ++i) {
        int64_t id = ids[i];
        for (size_t d = 0; d < embed_dim; ++d) {
            out[i * embed_dim + d] = weight[id * embed_dim + d];
        }
    }
    return out;
}

}  // namespace

//===----------------------------------------------------------------------===//
// Test: Qwen3TTSTextProjection
//===----------------------------------------------------------------------===//

TEST(Qwen3TTSTalkerTextProjection, MatchesReference) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t batch = 1;
    const size_t seq_len = 2;
    const size_t text_hidden = 8;    // text_hidden_size (small for test)
    const size_t hidden = 12;        // hidden_size (small for test)

    const ov::Shape fc1_weight_shape{hidden, text_hidden};
    const ov::Shape fc1_bias_shape{hidden};
    const ov::Shape fc2_weight_shape{hidden, hidden};
    const ov::Shape fc2_bias_shape{hidden};

    const auto fc1_w = test_utils::make_seq(hidden * text_hidden, 0.01f, 0.01f);
    const auto fc1_b = test_utils::make_seq(hidden, 0.005f, 0.002f);
    const auto fc2_w = test_utils::make_seq(hidden * hidden, 0.02f, 0.005f);
    const auto fc2_b = test_utils::make_seq(hidden, -0.01f, 0.001f);

    test_utils::DummyWeightSource weights;
    weights.add("text_projection.linear_fc1.weight", test_utils::make_tensor(fc1_w, fc1_weight_shape));
    weights.add("text_projection.linear_fc1.bias", test_utils::make_tensor(fc1_b, fc1_bias_shape));
    weights.add("text_projection.linear_fc2.weight", test_utils::make_tensor(fc2_w, fc2_weight_shape));
    weights.add("text_projection.linear_fc2.bias", test_utils::make_tensor(fc2_b, fc2_bias_shape));

    test_utils::DummyWeightFinalizer finalizer;

    ov::genai::modeling::models::Qwen3TTSTalkerConfig cfg;
    cfg.text_hidden_size = static_cast<int32_t>(text_hidden);
    cfg.hidden_size = static_cast<int32_t>(hidden);

    ov::genai::modeling::models::Qwen3TTSTextProjection projection(ctx, "text_projection", cfg);
    ov::genai::modeling::weights::load_model(projection, weights, finalizer);

    auto input = ctx.parameter("input", ov::element::f32, ov::PartialShape{batch, seq_len, text_hidden});
    auto output = projection.forward(input);
    auto ov_model = ctx.build_model({output.output()});

    ov::Core core;
    auto compiled = core.compile_model(ov_model, "GPU");
    auto request = compiled.create_infer_request();

    const std::vector<float> input_data = test_utils::make_seq(batch * seq_len * text_hidden, 0.1f, 0.05f);
    ov::Tensor input_tensor(ov::element::f32, {batch, seq_len, text_hidden});
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));
    request.set_input_tensor(input_tensor);

    request.infer();

    auto expected = text_projection_ref(input_data, fc1_w, fc1_b, fc2_w, fc2_b,
                                        batch, seq_len, text_hidden, hidden, hidden);
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_default);
}

//===----------------------------------------------------------------------===//
// Test: Qwen3TTSTalkerMLP (SwiGLU)
//===----------------------------------------------------------------------===//

TEST(Qwen3TTSTalkerMLP, MatchesReference) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t batch = 1;
    const size_t seq_len = 2;
    const size_t hidden = 6;
    const size_t intermediate = 8;

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

    ov::genai::modeling::models::Qwen3TTSTalkerConfig cfg;
    cfg.hidden_size = static_cast<int32_t>(hidden);
    cfg.intermediate_size = static_cast<int32_t>(intermediate);
    cfg.hidden_act = "silu";

    ov::genai::modeling::models::Qwen3TTSTalkerMLP mlp(ctx, "mlp", cfg);
    ov::genai::modeling::weights::load_model(mlp, weights, finalizer);

    auto input = ctx.parameter("input", ov::element::f32, ov::PartialShape{batch, seq_len, hidden});
    auto output = mlp.forward(input);
    auto ov_model = ctx.build_model({output.output()});

    ov::Core core;
    auto compiled = core.compile_model(ov_model, "GPU");
    auto request = compiled.create_infer_request();

    const std::vector<float> input_data = {
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
        0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f,
    };
    ov::Tensor input_tensor(ov::element::f32, {batch, seq_len, hidden});
    std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));
    request.set_input_tensor(input_tensor);

    request.infer();

    auto expected = swiglu_mlp_ref(input_data, gate_w, up_w, down_w, batch, seq_len, hidden, intermediate);
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_default);
}

//===----------------------------------------------------------------------===//
// Test: Embedding Model Structure
// This test verifies the embedding model graph structure without reference comparison
// (Text embedding + projection + codec embedding summation)
//===----------------------------------------------------------------------===//

TEST(Qwen3TTSTalkerEmbedding, GraphStructure) {
    // Create a minimal config for testing
    ov::genai::modeling::models::Qwen3TTSTalkerConfig cfg;
    cfg.text_vocab_size = 100;
    cfg.vocab_size = 50;
    cfg.text_hidden_size = 16;
    cfg.hidden_size = 32;
    cfg.intermediate_size = 64;
    cfg.num_hidden_layers = 2;
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 2;
    cfg.head_dim = 8;
    cfg.rope_theta = 10000.0f;
    cfg.rms_norm_eps = 1e-6f;
    cfg.mrope_section = {4, 2, 2};  // smaller for test

    const size_t batch = 1;
    const size_t seq_len = 3;

    // Create dummy weights - note: factory uses "talker." prefix for embeddings/projection
    test_utils::DummyWeightSource weights;

    // Text embedding: [text_vocab_size, text_hidden_size]
    auto text_embed_w = test_utils::make_seq(cfg.text_vocab_size * cfg.text_hidden_size, 0.01f, 0.001f);
    weights.add("talker.model.text_embedding.weight", test_utils::make_tensor(text_embed_w,
        {static_cast<size_t>(cfg.text_vocab_size), static_cast<size_t>(cfg.text_hidden_size)}));

    // Codec embedding: [vocab_size, hidden_size]
    auto codec_embed_w = test_utils::make_seq(cfg.vocab_size * cfg.hidden_size, 0.02f, 0.001f);
    weights.add("talker.model.codec_embedding.weight", test_utils::make_tensor(codec_embed_w,
        {static_cast<size_t>(cfg.vocab_size), static_cast<size_t>(cfg.hidden_size)}));

    // Text projection: fc1 [hidden, text_hidden], fc2 [hidden, hidden]
    auto fc1_w = test_utils::make_seq(cfg.hidden_size * cfg.text_hidden_size, 0.01f, 0.01f);
    auto fc1_b = test_utils::make_seq(cfg.hidden_size, 0.0f, 0.0f);
    auto fc2_w = test_utils::make_seq(cfg.hidden_size * cfg.hidden_size, 0.01f, 0.01f);
    auto fc2_b = test_utils::make_seq(cfg.hidden_size, 0.0f, 0.0f);
    weights.add("talker.text_projection.linear_fc1.weight", test_utils::make_tensor(fc1_w,
        {static_cast<size_t>(cfg.hidden_size), static_cast<size_t>(cfg.text_hidden_size)}));
    weights.add("talker.text_projection.linear_fc1.bias", test_utils::make_tensor(fc1_b,
        {static_cast<size_t>(cfg.hidden_size)}));
    weights.add("talker.text_projection.linear_fc2.weight", test_utils::make_tensor(fc2_w,
        {static_cast<size_t>(cfg.hidden_size), static_cast<size_t>(cfg.hidden_size)}));
    weights.add("talker.text_projection.linear_fc2.bias", test_utils::make_tensor(fc2_b,
        {static_cast<size_t>(cfg.hidden_size)}));

    test_utils::DummyWeightFinalizer finalizer;

    // Build the embedding model
    auto model = ov::genai::modeling::models::create_qwen3_tts_embedding_model(cfg, weights, finalizer);

    // Verify model structure
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->inputs().size(), 3);   // text_input_ids, codec_input_ids, codec_mask
    EXPECT_EQ(model->outputs().size(), 1);  // inputs_embeds

    // Verify output shape: [batch, seq, hidden_size]
    auto output_shape = model->output(0).get_partial_shape();
    EXPECT_TRUE(output_shape.rank().is_static());
    EXPECT_EQ(output_shape.rank().get_length(), 3);
    EXPECT_EQ(output_shape[2].get_length(), cfg.hidden_size);

    // Compile and run inference
    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    // Input data
    std::vector<int64_t> text_ids = {1, 5, 10};
    std::vector<int64_t> codec_ids = {0, 2, 4};
    std::vector<float> codec_mask_data = {1.0f, 1.0f, 1.0f};

    ov::Tensor text_tensor(ov::element::i64, {batch, seq_len});
    std::memcpy(text_tensor.data(), text_ids.data(), text_ids.size() * sizeof(int64_t));
    request.set_input_tensor(0, text_tensor);

    ov::Tensor codec_tensor(ov::element::i64, {batch, seq_len});
    std::memcpy(codec_tensor.data(), codec_ids.data(), codec_ids.size() * sizeof(int64_t));
    request.set_input_tensor(1, codec_tensor);

    ov::Tensor mask_tensor(ov::element::f32, {batch, seq_len});
    std::memcpy(mask_tensor.data(), codec_mask_data.data(), codec_mask_data.size() * sizeof(float));
    request.set_input_tensor(2, mask_tensor);

    // Should not throw
    EXPECT_NO_THROW(request.infer());

    // Verify output dimensions
    auto output_tensor = request.get_output_tensor();
    EXPECT_EQ(output_tensor.get_shape().size(), 3);
    EXPECT_EQ(output_tensor.get_shape()[0], batch);
    EXPECT_EQ(output_tensor.get_shape()[1], seq_len);
    EXPECT_EQ(output_tensor.get_shape()[2], static_cast<size_t>(cfg.hidden_size));
}

//===----------------------------------------------------------------------===//
// Test: Codec-Only Embedding Model
//===----------------------------------------------------------------------===//

TEST(Qwen3TTSTalkerCodecEmbedding, GraphStructure) {
    ov::genai::modeling::models::Qwen3TTSTalkerConfig cfg;
    cfg.vocab_size = 50;
    cfg.hidden_size = 32;

    test_utils::DummyWeightSource weights;

    // Note: factory uses "talker.model." prefix for codec_embedding
    auto codec_embed_w = test_utils::make_seq(cfg.vocab_size * cfg.hidden_size, 0.02f, 0.001f);
    weights.add("talker.model.codec_embedding.weight", test_utils::make_tensor(codec_embed_w,
        {static_cast<size_t>(cfg.vocab_size), static_cast<size_t>(cfg.hidden_size)}));

    test_utils::DummyWeightFinalizer finalizer;

    auto model = ov::genai::modeling::models::create_qwen3_tts_codec_embedding_model(cfg, weights, finalizer);

    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->inputs().size(), 1);   // codec_input_ids only
    EXPECT_EQ(model->outputs().size(), 1);  // codec_embeds

    // Compile and run
    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    const size_t batch = 1;
    const size_t seq_len = 1;
    std::vector<int64_t> codec_ids = {5};

    ov::Tensor codec_tensor(ov::element::i64, {batch, seq_len});
    std::memcpy(codec_tensor.data(), codec_ids.data(), codec_ids.size() * sizeof(int64_t));
    request.set_input_tensor(codec_tensor);

    EXPECT_NO_THROW(request.infer());

    auto output_tensor = request.get_output_tensor();
    EXPECT_EQ(output_tensor.get_shape()[2], static_cast<size_t>(cfg.hidden_size));
}

//===----------------------------------------------------------------------===//
// Test: Combined Embedding Reference
// Verifies that: output = text_projection(text_embed(text_ids)) + codec_embed(codec_ids)
//===----------------------------------------------------------------------===//

TEST(Qwen3TTSTalkerEmbedding, MatchesReference) {
    ov::genai::modeling::models::Qwen3TTSTalkerConfig cfg;
    cfg.text_vocab_size = 20;
    cfg.vocab_size = 10;
    cfg.text_hidden_size = 4;
    cfg.hidden_size = 6;

    const size_t batch = 1;
    const size_t seq_len = 2;

    // Create weights - note: factory uses "talker." prefix for embeddings/projection
    const auto text_embed_w = test_utils::make_seq(cfg.text_vocab_size * cfg.text_hidden_size, 0.1f, 0.01f);
    const auto codec_embed_w = test_utils::make_seq(cfg.vocab_size * cfg.hidden_size, 0.05f, 0.02f);
    const auto fc1_w = test_utils::make_seq(cfg.hidden_size * cfg.text_hidden_size, 0.02f, 0.01f);
    const auto fc1_b = test_utils::make_seq(cfg.hidden_size, 0.01f, 0.002f);
    const auto fc2_w = test_utils::make_seq(cfg.hidden_size * cfg.hidden_size, 0.015f, 0.008f);
    const auto fc2_b = test_utils::make_seq(cfg.hidden_size, -0.005f, 0.001f);

    test_utils::DummyWeightSource weights;
    weights.add("talker.model.text_embedding.weight", test_utils::make_tensor(text_embed_w,
        {static_cast<size_t>(cfg.text_vocab_size), static_cast<size_t>(cfg.text_hidden_size)}));
    weights.add("talker.model.codec_embedding.weight", test_utils::make_tensor(codec_embed_w,
        {static_cast<size_t>(cfg.vocab_size), static_cast<size_t>(cfg.hidden_size)}));
    weights.add("talker.text_projection.linear_fc1.weight", test_utils::make_tensor(fc1_w,
        {static_cast<size_t>(cfg.hidden_size), static_cast<size_t>(cfg.text_hidden_size)}));
    weights.add("talker.text_projection.linear_fc1.bias", test_utils::make_tensor(fc1_b,
        {static_cast<size_t>(cfg.hidden_size)}));
    weights.add("talker.text_projection.linear_fc2.weight", test_utils::make_tensor(fc2_w,
        {static_cast<size_t>(cfg.hidden_size), static_cast<size_t>(cfg.hidden_size)}));
    weights.add("talker.text_projection.linear_fc2.bias", test_utils::make_tensor(fc2_b,
        {static_cast<size_t>(cfg.hidden_size)}));

    test_utils::DummyWeightFinalizer finalizer;

    auto model = ov::genai::modeling::models::create_qwen3_tts_embedding_model(cfg, weights, finalizer);

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    std::vector<int64_t> text_ids = {3, 7};
    std::vector<int64_t> codec_ids = {1, 4};
    std::vector<float> codec_mask_data = {1.0f, 1.0f};

    ov::Tensor text_tensor(ov::element::i64, {batch, seq_len});
    std::memcpy(text_tensor.data(), text_ids.data(), text_ids.size() * sizeof(int64_t));
    request.set_input_tensor(0, text_tensor);

    ov::Tensor codec_tensor(ov::element::i64, {batch, seq_len});
    std::memcpy(codec_tensor.data(), codec_ids.data(), codec_ids.size() * sizeof(int64_t));
    request.set_input_tensor(1, codec_tensor);

    ov::Tensor mask_tensor(ov::element::f32, {batch, seq_len});
    std::memcpy(mask_tensor.data(), codec_mask_data.data(), codec_mask_data.size() * sizeof(float));
    request.set_input_tensor(2, mask_tensor);

    request.infer();

    // Compute reference:
    // 1. text_embed = embedding(text_ids, text_embed_w)
    auto text_embed = embedding_ref(text_ids, text_embed_w, batch, seq_len, cfg.text_hidden_size);

    // 2. text_projected = text_projection(text_embed)
    auto text_projected = text_projection_ref(text_embed, fc1_w, fc1_b, fc2_w, fc2_b,
        batch, seq_len, cfg.text_hidden_size, cfg.hidden_size, cfg.hidden_size);

    // 3. codec_embed = embedding(codec_ids, codec_embed_w)
    auto codec_embed = embedding_ref(codec_ids, codec_embed_w, batch, seq_len, cfg.hidden_size);

    // 4. output = text_projected + codec_embed
    std::vector<float> expected(text_projected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = text_projected[i] + codec_embed[i];
    }

    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_default);
}
