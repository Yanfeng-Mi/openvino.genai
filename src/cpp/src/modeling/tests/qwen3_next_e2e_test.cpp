// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include <gtest/gtest.h>

#include <openvino/core/except.hpp>
#include <openvino/op/read_value.hpp>
#include <openvino/openvino.hpp>

#include "modeling/models/qwen3_next/modeling_qwen3_next.hpp"
#include "modeling/tests/test_utils.hpp"

namespace test_utils = ov::genai::modeling::tests;

namespace {

ov::genai::modeling::models::Qwen3NextConfig make_min_config() {
    ov::genai::modeling::models::Qwen3NextConfig cfg;
    cfg.architecture = "qwen3_next";
    cfg.hidden_size = 8;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 1;
    cfg.head_dim = 4;
    cfg.intermediate_size = 12;
    cfg.num_hidden_layers = 2;
    cfg.vocab_size = 16;
    cfg.max_position_embeddings = 64;
    cfg.rms_norm_eps = 1e-6f;
    cfg.rope_theta = 10000.0f;
    cfg.partial_rotary_factor = 0.5f;
    cfg.hidden_act = "silu";
    cfg.attention_bias = false;
    cfg.tie_word_embeddings = false;
    cfg.layer_types = {"linear_attention", "full_attention"};
    cfg.full_attention_interval = 4;

    cfg.linear_conv_kernel_dim = 2;
    cfg.linear_key_head_dim = 2;
    cfg.linear_value_head_dim = 2;
    cfg.linear_num_key_heads = 2;
    cfg.linear_num_value_heads = 2;

    cfg.decoder_sparse_step = 1;
    cfg.num_experts = 0;
    cfg.num_experts_per_tok = 1;
    cfg.norm_topk_prob = true;
    return cfg;
}

void add_dense_mlp_weights(test_utils::DummyWeightSource& source,
                           const std::string& prefix,
                           int32_t hidden_size,
                           int32_t intermediate_size,
                           float seed) {
    source.add(prefix + "gate_proj.weight",
               test_utils::make_tensor(
                   test_utils::make_seq(static_cast<size_t>(intermediate_size * hidden_size), seed, 0.001f),
                   {static_cast<size_t>(intermediate_size), static_cast<size_t>(hidden_size)}));
    source.add(prefix + "up_proj.weight",
               test_utils::make_tensor(
                   test_utils::make_seq(static_cast<size_t>(intermediate_size * hidden_size), seed + 0.1f, 0.001f),
                   {static_cast<size_t>(intermediate_size), static_cast<size_t>(hidden_size)}));
    source.add(prefix + "down_proj.weight",
               test_utils::make_tensor(
                   test_utils::make_seq(static_cast<size_t>(hidden_size * intermediate_size), seed + 0.2f, 0.001f),
                   {static_cast<size_t>(hidden_size), static_cast<size_t>(intermediate_size)}));
}

void add_linear_attn_weights(test_utils::DummyWeightSource& source,
                             const std::string& prefix,
                             const ov::genai::modeling::models::Qwen3NextConfig& cfg,
                             float seed) {
    const int32_t key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
    const int32_t value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    const int32_t conv_dim = key_dim * 2 + value_dim;
    const int32_t proj_qkvz = key_dim * 2 + value_dim * 2;
    const int32_t proj_ba = cfg.linear_num_value_heads * 2;

    source.add(prefix + "in_proj_qkvz.weight",
               test_utils::make_tensor(
                   test_utils::make_seq(static_cast<size_t>(proj_qkvz * cfg.hidden_size), seed, 0.001f),
                   {static_cast<size_t>(proj_qkvz), static_cast<size_t>(cfg.hidden_size)}));
    source.add(prefix + "in_proj_ba.weight",
               test_utils::make_tensor(
                   test_utils::make_seq(static_cast<size_t>(proj_ba * cfg.hidden_size), seed + 0.1f, 0.001f),
                   {static_cast<size_t>(proj_ba), static_cast<size_t>(cfg.hidden_size)}));
    source.add(prefix + "conv1d.weight",
               test_utils::make_tensor(
                   test_utils::make_seq(static_cast<size_t>(conv_dim * cfg.linear_conv_kernel_dim), seed + 0.2f, 0.001f),
                   {static_cast<size_t>(conv_dim), static_cast<size_t>(cfg.linear_conv_kernel_dim)}));
    source.add(prefix + "A_log",
               test_utils::make_tensor(
                   test_utils::make_seq(static_cast<size_t>(cfg.linear_num_value_heads), seed + 0.3f, 0.001f),
                   {static_cast<size_t>(cfg.linear_num_value_heads)}));
    source.add(prefix + "dt_bias",
               test_utils::make_tensor(
                   test_utils::make_seq(static_cast<size_t>(cfg.linear_num_value_heads), seed + 0.4f, 0.001f),
                   {static_cast<size_t>(cfg.linear_num_value_heads)}));
    source.add(prefix + "norm.weight",
               test_utils::make_tensor(
                   test_utils::make_seq(static_cast<size_t>(cfg.linear_value_head_dim), 0.0f, 0.0f),
                   {static_cast<size_t>(cfg.linear_value_head_dim)}));
    source.add(prefix + "out_proj.weight",
               test_utils::make_tensor(
                   test_utils::make_seq(static_cast<size_t>(cfg.hidden_size * value_dim), seed + 0.5f, 0.001f),
                   {static_cast<size_t>(cfg.hidden_size), static_cast<size_t>(value_dim)}));
}

void add_full_attn_weights(test_utils::DummyWeightSource& source,
                           const std::string& prefix,
                           const ov::genai::modeling::models::Qwen3NextConfig& cfg,
                           float seed) {
    const int32_t q_out = cfg.num_attention_heads * cfg.head_dim * 2;
    const int32_t kv_out = cfg.num_key_value_heads * cfg.head_dim;
    const int32_t hidden = cfg.hidden_size;
    source.add(prefix + "q_proj.weight",
               test_utils::make_tensor(
                   test_utils::make_seq(static_cast<size_t>(q_out * hidden), seed, 0.001f),
                   {static_cast<size_t>(q_out), static_cast<size_t>(hidden)}));
    source.add(prefix + "k_proj.weight",
               test_utils::make_tensor(
                   test_utils::make_seq(static_cast<size_t>(kv_out * hidden), seed + 0.1f, 0.001f),
                   {static_cast<size_t>(kv_out), static_cast<size_t>(hidden)}));
    source.add(prefix + "v_proj.weight",
               test_utils::make_tensor(
                   test_utils::make_seq(static_cast<size_t>(kv_out * hidden), seed + 0.2f, 0.001f),
                   {static_cast<size_t>(kv_out), static_cast<size_t>(hidden)}));
    source.add(prefix + "o_proj.weight",
               test_utils::make_tensor(
                   test_utils::make_seq(static_cast<size_t>(hidden * hidden), seed + 0.3f, 0.001f),
                   {static_cast<size_t>(hidden), static_cast<size_t>(hidden)}));
    source.add(prefix + "q_norm.weight",
               test_utils::make_tensor(
                   test_utils::make_seq(static_cast<size_t>(cfg.head_dim), 0.0f, 0.0f),
                   {static_cast<size_t>(cfg.head_dim)}));
    source.add(prefix + "k_norm.weight",
               test_utils::make_tensor(
                   test_utils::make_seq(static_cast<size_t>(cfg.head_dim), 0.0f, 0.0f),
                   {static_cast<size_t>(cfg.head_dim)}));
}

test_utils::DummyWeightSource make_min_source(const ov::genai::modeling::models::Qwen3NextConfig& cfg) {
    test_utils::DummyWeightSource source;

    source.add("model.embed_tokens.weight",
               test_utils::make_tensor(test_utils::make_seq(static_cast<size_t>(cfg.vocab_size * cfg.hidden_size), 0.01f, 0.001f),
                                       {static_cast<size_t>(cfg.vocab_size), static_cast<size_t>(cfg.hidden_size)}));
    source.add("model.norm.weight",
               test_utils::make_tensor(test_utils::make_seq(static_cast<size_t>(cfg.hidden_size), 0.0f, 0.0f),
                                       {static_cast<size_t>(cfg.hidden_size)}));
    source.add("lm_head.weight",
               test_utils::make_tensor(test_utils::make_seq(static_cast<size_t>(cfg.vocab_size * cfg.hidden_size), 0.02f, 0.001f),
                                       {static_cast<size_t>(cfg.vocab_size), static_cast<size_t>(cfg.hidden_size)}));

    for (int32_t i = 0; i < cfg.num_hidden_layers; ++i) {
        const std::string layer_prefix = "model.layers[" + std::to_string(i) + "].";
        source.add(layer_prefix + "input_layernorm.weight",
                   test_utils::make_tensor(test_utils::make_seq(static_cast<size_t>(cfg.hidden_size), 0.0f, 0.0f),
                                           {static_cast<size_t>(cfg.hidden_size)}));
        source.add(layer_prefix + "post_attention_layernorm.weight",
                   test_utils::make_tensor(test_utils::make_seq(static_cast<size_t>(cfg.hidden_size), 0.0f, 0.0f),
                                           {static_cast<size_t>(cfg.hidden_size)}));

        if (cfg.layer_types[static_cast<size_t>(i)] == "linear_attention") {
            add_linear_attn_weights(source, layer_prefix + "linear_attn.", cfg, 0.1f * static_cast<float>(i + 1));
        } else {
            add_full_attn_weights(source, layer_prefix + "self_attn.", cfg, 0.1f * static_cast<float>(i + 1));
        }
        add_dense_mlp_weights(source, layer_prefix + "mlp.", cfg.hidden_size, cfg.intermediate_size, 0.5f + 0.1f * static_cast<float>(i));
    }

    return source;
}

bool has_input_name(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    for (const auto& input : model->inputs()) {
        if (input.get_names().count(name) > 0) {
            return true;
        }
    }
    return false;
}

}  // namespace

TEST(Qwen3NextE2EModeling, BuildsCausalLmGraphAndAllowsExpectedUnmatchedKeys) {
    auto cfg = make_min_config();
    auto source = make_min_source(cfg);

    source.add("mtp.fake.weight", test_utils::make_tensor({1.0f}, {1}));
    source.add("model.layers.0.linear_attn.in_proj_qkvz.weight_scale_inv", test_utils::make_tensor({1.0f}, {1, 1}));

    test_utils::DummyWeightFinalizer finalizer;
    auto model = ov::genai::modeling::models::create_qwen3_next_model(cfg, source, finalizer);

    ASSERT_NE(model, nullptr);
    EXPECT_TRUE(has_input_name(model, "input_ids"));
    EXPECT_TRUE(has_input_name(model, "attention_mask"));
    EXPECT_TRUE(has_input_name(model, "position_ids"));
    EXPECT_TRUE(has_input_name(model, "beam_idx"));
    EXPECT_EQ(model->output(0).get_any_name(), "logits");

    bool has_attention_cache = false;
    bool has_linear_cache = false;
    for (const auto& op : model->get_ops()) {
        if (auto read = ov::as_type_ptr<ov::op::v6::ReadValue>(op)) {
            const auto id = read->get_variable_id();
            has_attention_cache = has_attention_cache || id.find("past_key_values.") != std::string::npos;
            has_linear_cache = has_linear_cache || id.find("linear_states.") != std::string::npos;
        }
    }

    EXPECT_TRUE(has_attention_cache);
    EXPECT_TRUE(has_linear_cache);
}

TEST(Qwen3NextE2EModeling, ThrowsOnUnexpectedUnmatchedWeight) {
    auto cfg = make_min_config();
    auto source = make_min_source(cfg);
    source.add("unexpected.weight", test_utils::make_tensor({1.0f}, {1}));

    test_utils::DummyWeightFinalizer finalizer;
    EXPECT_THROW(ov::genai::modeling::models::create_qwen3_next_model(cfg, source, finalizer), ov::Exception);
}
