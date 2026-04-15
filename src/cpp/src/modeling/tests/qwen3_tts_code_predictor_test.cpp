// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>
#include "modeling/builder_context.hpp"
#include "modeling/models/qwen3_tts/modeling_qwen3_tts_code_predictor.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace test_utils = ov::genai::modeling::tests;

namespace {

//===----------------------------------------------------------------------===//
// Reference Implementations
//===----------------------------------------------------------------------===//

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

// Reference: LM Head (linear projection)
std::vector<float> lm_head_ref(const std::vector<float>& hidden_states,
                               const std::vector<float>& weight,
                               size_t batch,
                               size_t seq_len,
                               size_t hidden_size,
                               size_t vocab_size) {
    std::vector<float> out(batch * seq_len * vocab_size);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            for (size_t v = 0; v < vocab_size; ++v) {
                float acc = 0.0f;
                for (size_t h = 0; h < hidden_size; ++h) {
                    acc += hidden_states[(b * seq_len + t) * hidden_size + h] *
                           weight[v * hidden_size + h];
                }
                out[(b * seq_len + t) * vocab_size + v] = acc;
            }
        }
    }
    return out;
}

}  // namespace

//===----------------------------------------------------------------------===//
// Test: Qwen3TTSCodePredictorMLP (SwiGLU)
//===----------------------------------------------------------------------===//

TEST(Qwen3TTSCodePredictorMLP, MatchesReference) {
    ov::genai::modeling::BuilderContext ctx;

    // Code Predictor uses hidden_size=1024, intermediate_size=3072
    // Use smaller sizes for testing
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

    ov::genai::modeling::models::Qwen3TTSCodePredictorConfig cfg;
    cfg.hidden_size = static_cast<int32_t>(hidden);
    cfg.intermediate_size = static_cast<int32_t>(intermediate);
    cfg.hidden_act = "silu";

    ov::genai::modeling::models::Qwen3TTSCodePredictorMLP mlp(ctx, "mlp", cfg);
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
// Test: Single Codec Embedding Model
// Verifies the single codec embedding lookup for a specific layer
//===----------------------------------------------------------------------===//

TEST(Qwen3TTSCodePredictorSingleCodecEmbed, GraphStructure) {
    ov::genai::modeling::models::Qwen3TTSCodePredictorConfig cfg;
    cfg.vocab_size = 2048;
    cfg.hidden_size = 16;  // Small for testing
    cfg.num_code_groups = 16;

    const int codec_layer = 3;  // Test layer 4 (0-indexed as 3)

    test_utils::DummyWeightSource weights;

    // Code Predictor codec embeddings for layers 1-15 (0-indexed 0-14)
    // Module path: talker.code_predictor.model.codec_embedding[i].weight
    for (int i = 0; i < 15; ++i) {
        auto embed_w = test_utils::make_seq(cfg.vocab_size * cfg.hidden_size, 0.01f + i * 0.001f, 0.0001f);
        weights.add("talker.code_predictor.model.codec_embedding[" + std::to_string(i) + "].weight",
            test_utils::make_tensor(embed_w,
                {static_cast<size_t>(cfg.vocab_size), static_cast<size_t>(cfg.hidden_size)}));
    }

    // LM heads (needed for model construction, but not used in this test)
    for (int i = 0; i < 15; ++i) {
        auto lm_w = test_utils::make_seq(cfg.vocab_size * cfg.hidden_size, 0.02f, 0.0001f);
        weights.add("talker.code_predictor.lm_head[" + std::to_string(i) + "].weight",
            test_utils::make_tensor(lm_w,
                {static_cast<size_t>(cfg.vocab_size), static_cast<size_t>(cfg.hidden_size)}));
    }

    test_utils::DummyWeightFinalizer finalizer;

    auto model = ov::genai::modeling::models::create_qwen3_tts_code_predictor_single_codec_embed_model(
        cfg, codec_layer, weights, finalizer);

    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->inputs().size(), 1);   // codec_input
    EXPECT_EQ(model->outputs().size(), 1);  // codec_embed

    // Verify output shape: [batch, seq, hidden_size]
    auto output_shape = model->output(0).get_partial_shape();
    EXPECT_TRUE(output_shape.rank().is_static());
    EXPECT_EQ(output_shape.rank().get_length(), 3);
    EXPECT_EQ(output_shape[2].get_length(), cfg.hidden_size);

    // Compile and run
    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    const size_t batch = 1;
    const size_t seq_len = 1;
    std::vector<int64_t> codec_ids = {42};  // Token ID

    ov::Tensor codec_tensor(ov::element::i64, {batch, seq_len});
    std::memcpy(codec_tensor.data(), codec_ids.data(), codec_ids.size() * sizeof(int64_t));
    request.set_input_tensor(codec_tensor);

    EXPECT_NO_THROW(request.infer());

    auto output_tensor = request.get_output_tensor();
    EXPECT_EQ(output_tensor.get_shape()[2], static_cast<size_t>(cfg.hidden_size));
}

//===----------------------------------------------------------------------===//
// Test: Single Codec Embedding Reference
// Verifies embedding lookup matches reference for a specific layer
//===----------------------------------------------------------------------===//

TEST(Qwen3TTSCodePredictorSingleCodecEmbed, MatchesReference) {
    ov::genai::modeling::models::Qwen3TTSCodePredictorConfig cfg;
    cfg.vocab_size = 32;
    cfg.hidden_size = 8;
    cfg.num_code_groups = 16;

    const int codec_layer = 5;  // Test layer 6 (0-indexed as 5)

    test_utils::DummyWeightSource weights;

    // Store embedding weights for reference
    std::vector<std::vector<float>> all_embed_weights(15);

    for (int i = 0; i < 15; ++i) {
        all_embed_weights[i] = test_utils::make_seq(cfg.vocab_size * cfg.hidden_size, 0.1f + i * 0.05f, 0.01f);
        weights.add("talker.code_predictor.model.codec_embedding[" + std::to_string(i) + "].weight",
            test_utils::make_tensor(all_embed_weights[i],
                {static_cast<size_t>(cfg.vocab_size), static_cast<size_t>(cfg.hidden_size)}));
    }

    for (int i = 0; i < 15; ++i) {
        auto lm_w = test_utils::make_seq(cfg.vocab_size * cfg.hidden_size, 0.02f, 0.001f);
        weights.add("talker.code_predictor.lm_head[" + std::to_string(i) + "].weight",
            test_utils::make_tensor(lm_w,
                {static_cast<size_t>(cfg.vocab_size), static_cast<size_t>(cfg.hidden_size)}));
    }

    test_utils::DummyWeightFinalizer finalizer;

    auto model = ov::genai::modeling::models::create_qwen3_tts_code_predictor_single_codec_embed_model(
        cfg, codec_layer, weights, finalizer);

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    const size_t batch = 1;
    const size_t seq_len = 2;
    std::vector<int64_t> codec_ids = {7, 15};

    ov::Tensor codec_tensor(ov::element::i64, {batch, seq_len});
    std::memcpy(codec_tensor.data(), codec_ids.data(), codec_ids.size() * sizeof(int64_t));
    request.set_input_tensor(codec_tensor);

    request.infer();

    // Compute reference using the specific layer's embedding
    auto expected = embedding_ref(codec_ids, all_embed_weights[codec_layer], batch, seq_len, cfg.hidden_size);
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_default);
}

//===----------------------------------------------------------------------===//
// Test: Combined Codec Embedding Model (Sum of 15 layers)
//===----------------------------------------------------------------------===//

TEST(Qwen3TTSCodePredictorCodecEmbedSum, GraphStructure) {
    ov::genai::modeling::models::Qwen3TTSCodePredictorConfig cfg;
    cfg.vocab_size = 64;
    cfg.hidden_size = 8;
    cfg.num_code_groups = 16;

    test_utils::DummyWeightSource weights;

    for (int i = 0; i < 15; ++i) {
        auto embed_w = test_utils::make_seq(cfg.vocab_size * cfg.hidden_size, 0.01f, 0.0001f);
        weights.add("talker.code_predictor.model.codec_embedding[" + std::to_string(i) + "].weight",
            test_utils::make_tensor(embed_w,
                {static_cast<size_t>(cfg.vocab_size), static_cast<size_t>(cfg.hidden_size)}));
    }

    for (int i = 0; i < 15; ++i) {
        auto lm_w = test_utils::make_seq(cfg.vocab_size * cfg.hidden_size, 0.02f, 0.0001f);
        weights.add("talker.code_predictor.lm_head[" + std::to_string(i) + "].weight",
            test_utils::make_tensor(lm_w,
                {static_cast<size_t>(cfg.vocab_size), static_cast<size_t>(cfg.hidden_size)}));
    }

    test_utils::DummyWeightFinalizer finalizer;

    auto model = ov::genai::modeling::models::create_qwen3_tts_code_predictor_codec_embed_model(
        cfg, weights, finalizer);

    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->inputs().size(), 15);  // codec_input_0 to codec_input_14
    EXPECT_EQ(model->outputs().size(), 1);  // codec_embeds_sum
}

//===----------------------------------------------------------------------===//
// Test: Combined Codec Embedding Reference
// Verifies that output = sum(embed_i(codec_input_i) for i in 0..14)
//===----------------------------------------------------------------------===//

TEST(Qwen3TTSCodePredictorCodecEmbedSum, MatchesReference) {
    ov::genai::modeling::models::Qwen3TTSCodePredictorConfig cfg;
    cfg.vocab_size = 32;
    cfg.hidden_size = 6;
    cfg.num_code_groups = 16;

    test_utils::DummyWeightSource weights;

    std::vector<std::vector<float>> all_embed_weights(15);

    for (int i = 0; i < 15; ++i) {
        all_embed_weights[i] = test_utils::make_seq(cfg.vocab_size * cfg.hidden_size, 0.05f + i * 0.02f, 0.005f);
        weights.add("talker.code_predictor.model.codec_embedding[" + std::to_string(i) + "].weight",
            test_utils::make_tensor(all_embed_weights[i],
                {static_cast<size_t>(cfg.vocab_size), static_cast<size_t>(cfg.hidden_size)}));
    }

    for (int i = 0; i < 15; ++i) {
        auto lm_w = test_utils::make_seq(cfg.vocab_size * cfg.hidden_size, 0.01f, 0.001f);
        weights.add("talker.code_predictor.lm_head[" + std::to_string(i) + "].weight",
            test_utils::make_tensor(lm_w,
                {static_cast<size_t>(cfg.vocab_size), static_cast<size_t>(cfg.hidden_size)}));
    }

    test_utils::DummyWeightFinalizer finalizer;

    auto model = ov::genai::modeling::models::create_qwen3_tts_code_predictor_codec_embed_model(
        cfg, weights, finalizer);

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    const size_t batch = 1;
    const size_t seq_len = 1;

    // Create 15 different codec IDs for each layer
    std::vector<std::vector<int64_t>> all_codec_ids = {
        {3}, {7}, {11}, {5}, {9},
        {2}, {8}, {14}, {1}, {6},
        {10}, {4}, {12}, {0}, {15}
    };

    // Set input tensors
    for (int i = 0; i < 15; ++i) {
        ov::Tensor codec_tensor(ov::element::i64, {batch, seq_len});
        std::memcpy(codec_tensor.data(), all_codec_ids[i].data(), all_codec_ids[i].size() * sizeof(int64_t));
        request.set_input_tensor(i, codec_tensor);
    }

    request.infer();

    // Compute reference: sum of all embeddings
    std::vector<float> expected(batch * seq_len * cfg.hidden_size, 0.0f);
    for (int i = 0; i < 15; ++i) {
        auto embed = embedding_ref(all_codec_ids[i], all_embed_weights[i], batch, seq_len, cfg.hidden_size);
        for (size_t j = 0; j < expected.size(); ++j) {
            expected[j] += embed[j];
        }
    }

    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_default);
}

//===----------------------------------------------------------------------===//
// Test: AR Model Graph Structure
// Verifies the AR model for a specific generation step
//===----------------------------------------------------------------------===//

TEST(Qwen3TTSCodePredictorARModel, GraphStructure) {
    ov::genai::modeling::models::Qwen3TTSCodePredictorConfig cfg;
    cfg.vocab_size = 2048;
    cfg.hidden_size = 16;       // Small for testing
    cfg.talker_hidden_size = 16; // Same as hidden_size to skip projection
    cfg.intermediate_size = 32;
    cfg.num_hidden_layers = 2;  // Reduced layers for testing
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 2;
    cfg.head_dim = 4;
    cfg.rope_theta = 1000000.0f;
    cfg.rms_norm_eps = 1e-6f;
    cfg.num_code_groups = 16;

    const int generation_step = 3;  // Test step 3 (predicting layer 4)

    test_utils::DummyWeightSource weights;

    // Model layers
    for (int layer = 0; layer < cfg.num_hidden_layers; ++layer) {
        std::string prefix = "talker.code_predictor.model.layers[" + std::to_string(layer) + "]";

        // Attention weights
        auto q_w = test_utils::make_seq(cfg.num_attention_heads * cfg.head_dim * cfg.hidden_size, 0.01f, 0.001f);
        auto k_w = test_utils::make_seq(cfg.num_key_value_heads * cfg.head_dim * cfg.hidden_size, 0.01f, 0.001f);
        auto v_w = test_utils::make_seq(cfg.num_key_value_heads * cfg.head_dim * cfg.hidden_size, 0.01f, 0.001f);
        auto o_w = test_utils::make_seq(cfg.hidden_size * cfg.num_attention_heads * cfg.head_dim, 0.01f, 0.001f);
        auto q_norm_w = test_utils::make_seq(cfg.head_dim, 1.0f, 0.0f);
        auto k_norm_w = test_utils::make_seq(cfg.head_dim, 1.0f, 0.0f);

        weights.add(prefix + ".self_attn.q_proj.weight",
            test_utils::make_tensor(q_w, {static_cast<size_t>(cfg.num_attention_heads * cfg.head_dim),
                                          static_cast<size_t>(cfg.hidden_size)}));
        weights.add(prefix + ".self_attn.k_proj.weight",
            test_utils::make_tensor(k_w, {static_cast<size_t>(cfg.num_key_value_heads * cfg.head_dim),
                                          static_cast<size_t>(cfg.hidden_size)}));
        weights.add(prefix + ".self_attn.v_proj.weight",
            test_utils::make_tensor(v_w, {static_cast<size_t>(cfg.num_key_value_heads * cfg.head_dim),
                                          static_cast<size_t>(cfg.hidden_size)}));
        weights.add(prefix + ".self_attn.o_proj.weight",
            test_utils::make_tensor(o_w, {static_cast<size_t>(cfg.hidden_size),
                                          static_cast<size_t>(cfg.num_attention_heads * cfg.head_dim)}));
        weights.add(prefix + ".self_attn.q_norm.weight",
            test_utils::make_tensor(q_norm_w, {static_cast<size_t>(cfg.head_dim)}));
        weights.add(prefix + ".self_attn.k_norm.weight",
            test_utils::make_tensor(k_norm_w, {static_cast<size_t>(cfg.head_dim)}));

        // MLP weights
        auto gate_w = test_utils::make_seq(cfg.intermediate_size * cfg.hidden_size, 0.01f, 0.001f);
        auto up_w = test_utils::make_seq(cfg.intermediate_size * cfg.hidden_size, 0.01f, 0.001f);
        auto down_w = test_utils::make_seq(cfg.hidden_size * cfg.intermediate_size, 0.01f, 0.001f);

        weights.add(prefix + ".mlp.gate_proj.weight",
            test_utils::make_tensor(gate_w, {static_cast<size_t>(cfg.intermediate_size),
                                             static_cast<size_t>(cfg.hidden_size)}));
        weights.add(prefix + ".mlp.up_proj.weight",
            test_utils::make_tensor(up_w, {static_cast<size_t>(cfg.intermediate_size),
                                           static_cast<size_t>(cfg.hidden_size)}));
        weights.add(prefix + ".mlp.down_proj.weight",
            test_utils::make_tensor(down_w, {static_cast<size_t>(cfg.hidden_size),
                                             static_cast<size_t>(cfg.intermediate_size)}));

        // LayerNorm weights
        auto input_ln_w = test_utils::make_seq(cfg.hidden_size, 1.0f, 0.0f);
        auto post_attn_ln_w = test_utils::make_seq(cfg.hidden_size, 1.0f, 0.0f);

        weights.add(prefix + ".input_layernorm.weight",
            test_utils::make_tensor(input_ln_w, {static_cast<size_t>(cfg.hidden_size)}));
        weights.add(prefix + ".post_attention_layernorm.weight",
            test_utils::make_tensor(post_attn_ln_w, {static_cast<size_t>(cfg.hidden_size)}));
    }

    // Final norm
    auto norm_w = test_utils::make_seq(cfg.hidden_size, 1.0f, 0.0f);
    weights.add("talker.code_predictor.model.norm.weight",
        test_utils::make_tensor(norm_w, {static_cast<size_t>(cfg.hidden_size)}));

    // Codec embeddings and LM heads
    for (int i = 0; i < 15; ++i) {
        auto embed_w = test_utils::make_seq(cfg.vocab_size * cfg.hidden_size, 0.01f, 0.0001f);
        weights.add("talker.code_predictor.model.codec_embedding[" + std::to_string(i) + "].weight",
            test_utils::make_tensor(embed_w,
                {static_cast<size_t>(cfg.vocab_size), static_cast<size_t>(cfg.hidden_size)}));

        auto lm_w = test_utils::make_seq(cfg.vocab_size * cfg.hidden_size, 0.01f, 0.0001f);
        weights.add("talker.code_predictor.lm_head[" + std::to_string(i) + "].weight",
            test_utils::make_tensor(lm_w,
                {static_cast<size_t>(cfg.vocab_size), static_cast<size_t>(cfg.hidden_size)}));
    }

    test_utils::DummyWeightFinalizer finalizer;

    auto model = ov::genai::modeling::models::create_qwen3_tts_code_predictor_ar_model(
        cfg, generation_step, weights, finalizer);

    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->inputs().size(), 2);   // inputs_embeds, position_ids
    EXPECT_EQ(model->outputs().size(), 1);  // logits

    // Verify output shape: [batch, 1, vocab_size] (last position only)
    auto output_shape = model->output(0).get_partial_shape();
    EXPECT_TRUE(output_shape.rank().is_static());
    EXPECT_EQ(output_shape.rank().get_length(), 3);
    EXPECT_EQ(output_shape[2].get_length(), cfg.vocab_size);

    // Compile and run
    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    const size_t batch = 1;
    const size_t seq_len = 3;

    auto input_embeds = test_utils::make_seq(batch * seq_len * cfg.hidden_size, 0.1f, 0.01f);
    ov::Tensor embeds_tensor(ov::element::f32, {batch, seq_len, static_cast<size_t>(cfg.hidden_size)});
    std::memcpy(embeds_tensor.data(), input_embeds.data(), input_embeds.size() * sizeof(float));
    request.set_input_tensor(0, embeds_tensor);

    std::vector<int64_t> position_ids = {0, 1, 2};
    ov::Tensor pos_tensor(ov::element::i64, {batch, seq_len});
    std::memcpy(pos_tensor.data(), position_ids.data(), position_ids.size() * sizeof(int64_t));
    request.set_input_tensor(1, pos_tensor);

    EXPECT_NO_THROW(request.infer());

    auto output_tensor = request.get_output_tensor();
    EXPECT_EQ(output_tensor.get_shape()[0], batch);
    EXPECT_EQ(output_tensor.get_shape()[1], 1u);  // AR model returns last position only
    EXPECT_EQ(output_tensor.get_shape()[2], static_cast<size_t>(cfg.vocab_size));
}

//===----------------------------------------------------------------------===//
// Test: Full Model Graph Structure
// Verifies the full model with all 15 logit outputs
//===----------------------------------------------------------------------===//

TEST(Qwen3TTSCodePredictorFullModel, GraphStructure) {
    ov::genai::modeling::models::Qwen3TTSCodePredictorConfig cfg;
    cfg.vocab_size = 128;
    cfg.hidden_size = 8;
    cfg.intermediate_size = 16;
    cfg.num_hidden_layers = 1;  // Minimal for testing
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 1;
    cfg.head_dim = 4;
    cfg.rope_theta = 1000000.0f;
    cfg.rms_norm_eps = 1e-6f;
    cfg.num_code_groups = 16;

    test_utils::DummyWeightSource weights;

    // Single layer weights
    std::string prefix = "talker.code_predictor.model.layers[0]";

    auto q_w = test_utils::make_seq(cfg.num_attention_heads * cfg.head_dim * cfg.hidden_size, 0.01f, 0.001f);
    auto k_w = test_utils::make_seq(cfg.num_key_value_heads * cfg.head_dim * cfg.hidden_size, 0.01f, 0.001f);
    auto v_w = test_utils::make_seq(cfg.num_key_value_heads * cfg.head_dim * cfg.hidden_size, 0.01f, 0.001f);
    auto o_w = test_utils::make_seq(cfg.hidden_size * cfg.num_attention_heads * cfg.head_dim, 0.01f, 0.001f);
    auto q_norm_w = test_utils::make_seq(cfg.head_dim, 1.0f, 0.0f);
    auto k_norm_w = test_utils::make_seq(cfg.head_dim, 1.0f, 0.0f);

    weights.add(prefix + ".self_attn.q_proj.weight",
        test_utils::make_tensor(q_w, {static_cast<size_t>(cfg.num_attention_heads * cfg.head_dim),
                                      static_cast<size_t>(cfg.hidden_size)}));
    weights.add(prefix + ".self_attn.k_proj.weight",
        test_utils::make_tensor(k_w, {static_cast<size_t>(cfg.num_key_value_heads * cfg.head_dim),
                                      static_cast<size_t>(cfg.hidden_size)}));
    weights.add(prefix + ".self_attn.v_proj.weight",
        test_utils::make_tensor(v_w, {static_cast<size_t>(cfg.num_key_value_heads * cfg.head_dim),
                                      static_cast<size_t>(cfg.hidden_size)}));
    weights.add(prefix + ".self_attn.o_proj.weight",
        test_utils::make_tensor(o_w, {static_cast<size_t>(cfg.hidden_size),
                                      static_cast<size_t>(cfg.num_attention_heads * cfg.head_dim)}));
    weights.add(prefix + ".self_attn.q_norm.weight",
        test_utils::make_tensor(q_norm_w, {static_cast<size_t>(cfg.head_dim)}));
    weights.add(prefix + ".self_attn.k_norm.weight",
        test_utils::make_tensor(k_norm_w, {static_cast<size_t>(cfg.head_dim)}));

    auto gate_w = test_utils::make_seq(cfg.intermediate_size * cfg.hidden_size, 0.01f, 0.001f);
    auto up_w = test_utils::make_seq(cfg.intermediate_size * cfg.hidden_size, 0.01f, 0.001f);
    auto down_w = test_utils::make_seq(cfg.hidden_size * cfg.intermediate_size, 0.01f, 0.001f);

    weights.add(prefix + ".mlp.gate_proj.weight",
        test_utils::make_tensor(gate_w, {static_cast<size_t>(cfg.intermediate_size),
                                         static_cast<size_t>(cfg.hidden_size)}));
    weights.add(prefix + ".mlp.up_proj.weight",
        test_utils::make_tensor(up_w, {static_cast<size_t>(cfg.intermediate_size),
                                       static_cast<size_t>(cfg.hidden_size)}));
    weights.add(prefix + ".mlp.down_proj.weight",
        test_utils::make_tensor(down_w, {static_cast<size_t>(cfg.hidden_size),
                                         static_cast<size_t>(cfg.intermediate_size)}));

    auto input_ln_w = test_utils::make_seq(cfg.hidden_size, 1.0f, 0.0f);
    auto post_attn_ln_w = test_utils::make_seq(cfg.hidden_size, 1.0f, 0.0f);

    weights.add(prefix + ".input_layernorm.weight",
        test_utils::make_tensor(input_ln_w, {static_cast<size_t>(cfg.hidden_size)}));
    weights.add(prefix + ".post_attention_layernorm.weight",
        test_utils::make_tensor(post_attn_ln_w, {static_cast<size_t>(cfg.hidden_size)}));

    auto norm_w = test_utils::make_seq(cfg.hidden_size, 1.0f, 0.0f);
    weights.add("talker.code_predictor.model.norm.weight",
        test_utils::make_tensor(norm_w, {static_cast<size_t>(cfg.hidden_size)}));

    for (int i = 0; i < 15; ++i) {
        auto embed_w = test_utils::make_seq(cfg.vocab_size * cfg.hidden_size, 0.01f, 0.0001f);
        weights.add("talker.code_predictor.model.codec_embedding[" + std::to_string(i) + "].weight",
            test_utils::make_tensor(embed_w,
                {static_cast<size_t>(cfg.vocab_size), static_cast<size_t>(cfg.hidden_size)}));

        auto lm_w = test_utils::make_seq(cfg.vocab_size * cfg.hidden_size, 0.01f, 0.0001f);
        weights.add("talker.code_predictor.lm_head[" + std::to_string(i) + "].weight",
            test_utils::make_tensor(lm_w,
                {static_cast<size_t>(cfg.vocab_size), static_cast<size_t>(cfg.hidden_size)}));
    }

    test_utils::DummyWeightFinalizer finalizer;

    auto model = ov::genai::modeling::models::create_qwen3_tts_code_predictor_model(
        cfg, weights, finalizer);

    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->inputs().size(), 2);   // inputs_embeds, position_ids
    EXPECT_EQ(model->outputs().size(), 15); // logits_0 to logits_14

    // Verify each output has correct shape
    for (int i = 0; i < 15; ++i) {
        auto output_shape = model->output(i).get_partial_shape();
        EXPECT_TRUE(output_shape.rank().is_static());
        EXPECT_EQ(output_shape.rank().get_length(), 3);
        EXPECT_EQ(output_shape[2].get_length(), cfg.vocab_size);
    }
}

