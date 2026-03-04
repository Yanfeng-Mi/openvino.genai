// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>

#include "modeling/models/dflash_draft/dflash_draft.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace test_utils = ov::genai::modeling::tests;

TEST(DFlashDraft, BuildsAndRuns) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t hidden = 8;
    const size_t num_heads = 2;
    const size_t num_kv_heads = 1;
    const size_t head_dim = 4;
    const size_t inter = 16;
    const size_t num_layers = 2;

    ov::genai::modeling::models::DFlashDraftConfig cfg;
    cfg.hidden_size = static_cast<int32_t>(hidden);
    cfg.intermediate_size = static_cast<int32_t>(inter);
    cfg.num_hidden_layers = static_cast<int32_t>(num_layers);
    cfg.num_target_layers = 4;
    cfg.num_attention_heads = static_cast<int32_t>(num_heads);
    cfg.num_key_value_heads = static_cast<int32_t>(num_kv_heads);
    cfg.head_dim = static_cast<int32_t>(head_dim);
    cfg.block_size = 4;

    ov::genai::modeling::models::DFlashDraftModel model(ctx, cfg);

    test_utils::DummyWeightSource weights;
    test_utils::DummyWeightFinalizer finalizer;

    const ov::Shape q_weight_shape{hidden, hidden};
    const ov::Shape kv_weight_shape{num_kv_heads * head_dim, hidden};
    const ov::Shape o_weight_shape{hidden, hidden};
    const ov::Shape mlp_up_shape{inter, hidden};
    const ov::Shape mlp_down_shape{hidden, inter};
    const ov::Shape norm_shape{hidden};
    const ov::Shape qk_norm_shape{head_dim};
    const ov::Shape fc_shape{hidden, hidden * num_layers};

    weights.add("fc.weight", test_utils::make_tensor(test_utils::make_seq(hidden * hidden * num_layers, 0.01f, 0.01f), fc_shape));
    weights.add("hidden_norm.weight", test_utils::make_tensor(test_utils::make_seq(hidden, 0.1f, 0.01f), norm_shape));
    weights.add("norm.weight", test_utils::make_tensor(test_utils::make_seq(hidden, 0.2f, 0.01f), norm_shape));

    for (size_t i = 0; i < num_layers; ++i) {
        const std::string prefix = "layers." + std::to_string(i) + ".";
        weights.add(prefix + "input_layernorm.weight",
                    test_utils::make_tensor(test_utils::make_seq(hidden, 0.3f, 0.01f), norm_shape));
        weights.add(prefix + "post_attention_layernorm.weight",
                    test_utils::make_tensor(test_utils::make_seq(hidden, 0.4f, 0.01f), norm_shape));
        weights.add(prefix + "self_attn.q_proj.weight",
                    test_utils::make_tensor(test_utils::make_seq(hidden * hidden, 0.01f, 0.01f), q_weight_shape));
        weights.add(prefix + "self_attn.k_proj.weight",
                    test_utils::make_tensor(test_utils::make_seq(num_kv_heads * head_dim * hidden, 0.02f, 0.01f),
                                            kv_weight_shape));
        weights.add(prefix + "self_attn.v_proj.weight",
                    test_utils::make_tensor(test_utils::make_seq(num_kv_heads * head_dim * hidden, 0.03f, 0.01f),
                                            kv_weight_shape));
        weights.add(prefix + "self_attn.o_proj.weight",
                    test_utils::make_tensor(test_utils::make_seq(hidden * hidden, 0.04f, 0.01f), o_weight_shape));
        weights.add(prefix + "self_attn.q_norm.weight",
                    test_utils::make_tensor(test_utils::make_seq(head_dim, 1.0f, 0.01f), qk_norm_shape));
        weights.add(prefix + "self_attn.k_norm.weight",
                    test_utils::make_tensor(test_utils::make_seq(head_dim, 0.9f, 0.01f), qk_norm_shape));
        weights.add(prefix + "mlp.gate_proj.weight",
                    test_utils::make_tensor(test_utils::make_seq(inter * hidden, 0.01f, 0.01f), mlp_up_shape));
        weights.add(prefix + "mlp.up_proj.weight",
                    test_utils::make_tensor(test_utils::make_seq(inter * hidden, 0.02f, 0.01f), mlp_up_shape));
        weights.add(prefix + "mlp.down_proj.weight",
                    test_utils::make_tensor(test_utils::make_seq(hidden * inter, 0.03f, 0.01f), mlp_down_shape));
    }

    ov::genai::modeling::weights::load_model(model, weights, finalizer);

    const size_t ctx_len = 3;
    const size_t q_len = 4;

    auto target_hidden = ctx.parameter("target_hidden", ov::element::f32,
                                       ov::PartialShape{1, static_cast<int64_t>(ctx_len),
                                                        static_cast<int64_t>(hidden * num_layers)});
    auto noise_embedding = ctx.parameter("noise_embedding", ov::element::f32,
                                         ov::PartialShape{1, static_cast<int64_t>(q_len), static_cast<int64_t>(hidden)});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64,
                                      ov::PartialShape{1, static_cast<int64_t>(ctx_len + q_len)});

    auto output = model.forward(target_hidden, noise_embedding, position_ids);
    auto ov_model = ctx.build_model({output.output()});

    ov::Core core;
    auto compiled = core.compile_model(ov_model, "GPU");
    auto request = compiled.create_infer_request();

    const auto target_data = test_utils::make_seq(ctx_len * hidden * num_layers, 0.01f, 0.01f);
    const auto noise_data = test_utils::make_seq(q_len * hidden, 0.02f, 0.01f);

    ov::Tensor target_tensor(ov::element::f32, {1, ctx_len, hidden * num_layers});
    std::memcpy(target_tensor.data(), target_data.data(), target_data.size() * sizeof(float));
    request.set_input_tensor(0, target_tensor);

    ov::Tensor noise_tensor(ov::element::f32, {1, q_len, hidden});
    std::memcpy(noise_tensor.data(), noise_data.data(), noise_data.size() * sizeof(float));
    request.set_input_tensor(1, noise_tensor);

    std::vector<int64_t> pos_vals(ctx_len + q_len);
    for (size_t i = 0; i < pos_vals.size(); ++i) {
        pos_vals[i] = static_cast<int64_t>(i);
    }
    ov::Tensor pos_tensor(ov::element::i64, {1, ctx_len + q_len});
    std::memcpy(pos_tensor.data(), pos_vals.data(), pos_vals.size() * sizeof(int64_t));
    request.set_input_tensor(2, pos_tensor);

    request.infer();

    auto out = request.get_output_tensor();
    EXPECT_EQ(out.get_shape(), (ov::Shape{1, q_len, hidden}));
}
