// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>

#include "modeling/builder_context.hpp"
#include "modeling/models/wan/modeling_wan_dit.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace test_utils = ov::genai::modeling::tests;

namespace {

void add_const_weight(test_utils::DummyWeightSource& source,
                      const std::string& name,
                      const ov::Shape& shape,
                      float value) {
    ov::Tensor tensor(ov::element::f32, shape);
    std::fill_n(tensor.data<float>(), tensor.get_size(), value);
    source.add(name, tensor);
}

}  // namespace

TEST(WanTransformer3DDummyTest, BuildsAndRuns) {
    ov::genai::modeling::BuilderContext ctx;
    ov::genai::modeling::models::WanTransformer3DConfig cfg;
    cfg.patch_size = {1, 2, 2};
    cfg.num_attention_heads = 2;
    cfg.attention_head_dim = 6;
    cfg.in_channels = 2;
    cfg.out_channels = 2;
    cfg.text_dim = 6;
    cfg.freq_dim = 8;
    cfg.ffn_dim = 16;
    cfg.num_layers = 1;
    cfg.cross_attn_norm = true;
    cfg.qk_norm = "rms_norm_across_heads";
    cfg.eps = 1e-6f;
    cfg.rope_max_seq_len = 16;
    cfg.finalize();

    ov::genai::modeling::models::WanTransformer3DModel model(ctx, cfg);

    const int32_t inner_dim = cfg.inner_dim();
    const int32_t patch_volume = cfg.patch_volume();

    test_utils::DummyWeightSource source;
    test_utils::DummyWeightFinalizer finalizer;

    add_const_weight(source, "patch_embedding.weight",
                     {static_cast<size_t>(inner_dim),
                      static_cast<size_t>(cfg.in_channels),
                      static_cast<size_t>(cfg.patch_size[0]),
                      static_cast<size_t>(cfg.patch_size[1]),
                      static_cast<size_t>(cfg.patch_size[2])},
                     0.0f);
    add_const_weight(source, "patch_embedding.bias", {static_cast<size_t>(inner_dim)}, 0.0f);

    add_const_weight(source, "condition_embedder.time_embedder.linear_1.weight",
                     {static_cast<size_t>(inner_dim), static_cast<size_t>(cfg.freq_dim)}, 0.0f);
    add_const_weight(source, "condition_embedder.time_embedder.linear_1.bias",
                     {static_cast<size_t>(inner_dim)}, 0.0f);
    add_const_weight(source, "condition_embedder.time_embedder.linear_2.weight",
                     {static_cast<size_t>(inner_dim), static_cast<size_t>(inner_dim)}, 0.0f);
    add_const_weight(source, "condition_embedder.time_embedder.linear_2.bias",
                     {static_cast<size_t>(inner_dim)}, 0.0f);
    add_const_weight(source, "condition_embedder.time_proj.weight",
                     {static_cast<size_t>(inner_dim * 6), static_cast<size_t>(inner_dim)}, 0.0f);
    add_const_weight(source, "condition_embedder.time_proj.bias",
                     {static_cast<size_t>(inner_dim * 6)}, 0.0f);

    add_const_weight(source, "condition_embedder.text_embedder.linear_1.weight",
                     {static_cast<size_t>(inner_dim), static_cast<size_t>(cfg.text_dim)}, 0.0f);
    add_const_weight(source, "condition_embedder.text_embedder.linear_1.bias",
                     {static_cast<size_t>(inner_dim)}, 0.0f);
    add_const_weight(source, "condition_embedder.text_embedder.linear_2.weight",
                     {static_cast<size_t>(inner_dim), static_cast<size_t>(inner_dim)}, 0.0f);
    add_const_weight(source, "condition_embedder.text_embedder.linear_2.bias",
                     {static_cast<size_t>(inner_dim)}, 0.0f);

    add_const_weight(source, "scale_shift_table", {1, 2, static_cast<size_t>(inner_dim)}, 0.0f);
    add_const_weight(source, "proj_out.weight",
                     {static_cast<size_t>(cfg.out_channels * patch_volume), static_cast<size_t>(inner_dim)}, 0.0f);
    add_const_weight(source, "proj_out.bias",
                     {static_cast<size_t>(cfg.out_channels * patch_volume)}, 0.0f);

    add_const_weight(source, "blocks.0.scale_shift_table", {1, 6, static_cast<size_t>(inner_dim)}, 0.0f);
    add_const_weight(source, "blocks.0.norm2.weight", {static_cast<size_t>(inner_dim)}, 1.0f);
    add_const_weight(source, "blocks.0.norm2.bias", {static_cast<size_t>(inner_dim)}, 0.0f);

    const ov::Shape attn_weight{static_cast<size_t>(inner_dim), static_cast<size_t>(inner_dim)};
    const ov::Shape attn_bias{static_cast<size_t>(inner_dim)};
    add_const_weight(source, "blocks.0.attn1.to_q.weight", attn_weight, 0.0f);
    add_const_weight(source, "blocks.0.attn1.to_q.bias", attn_bias, 0.0f);
    add_const_weight(source, "blocks.0.attn1.to_k.weight", attn_weight, 0.0f);
    add_const_weight(source, "blocks.0.attn1.to_k.bias", attn_bias, 0.0f);
    add_const_weight(source, "blocks.0.attn1.to_v.weight", attn_weight, 0.0f);
    add_const_weight(source, "blocks.0.attn1.to_v.bias", attn_bias, 0.0f);
    add_const_weight(source, "blocks.0.attn1.to_out.0.weight", attn_weight, 0.0f);
    add_const_weight(source, "blocks.0.attn1.to_out.0.bias", attn_bias, 0.0f);
    add_const_weight(source, "blocks.0.attn1.norm_q.weight", attn_bias, 1.0f);
    add_const_weight(source, "blocks.0.attn1.norm_k.weight", attn_bias, 1.0f);

    add_const_weight(source, "blocks.0.attn2.to_q.weight", attn_weight, 0.0f);
    add_const_weight(source, "blocks.0.attn2.to_q.bias", attn_bias, 0.0f);
    add_const_weight(source, "blocks.0.attn2.to_k.weight", attn_weight, 0.0f);
    add_const_weight(source, "blocks.0.attn2.to_k.bias", attn_bias, 0.0f);
    add_const_weight(source, "blocks.0.attn2.to_v.weight", attn_weight, 0.0f);
    add_const_weight(source, "blocks.0.attn2.to_v.bias", attn_bias, 0.0f);
    add_const_weight(source, "blocks.0.attn2.to_out.0.weight", attn_weight, 0.0f);
    add_const_weight(source, "blocks.0.attn2.to_out.0.bias", attn_bias, 0.0f);
    add_const_weight(source, "blocks.0.attn2.norm_q.weight", attn_bias, 1.0f);
    add_const_weight(source, "blocks.0.attn2.norm_k.weight", attn_bias, 1.0f);

    add_const_weight(source, "blocks.0.ffn.net.0.proj.weight",
                     {static_cast<size_t>(cfg.ffn_dim), static_cast<size_t>(inner_dim)}, 0.0f);
    add_const_weight(source, "blocks.0.ffn.net.0.proj.bias",
                     {static_cast<size_t>(cfg.ffn_dim)}, 0.0f);
    add_const_weight(source, "blocks.0.ffn.net.2.weight",
                     {static_cast<size_t>(inner_dim), static_cast<size_t>(cfg.ffn_dim)}, 0.0f);
    add_const_weight(source, "blocks.0.ffn.net.2.bias",
                     {static_cast<size_t>(inner_dim)}, 0.0f);

    ov::genai::modeling::weights::load_model(model, source, finalizer);

    auto latents = ctx.parameter("hidden_states",
                                 ov::element::f32,
                                 ov::PartialShape{1, cfg.in_channels, 2, 4, 4});
    auto timesteps = ctx.parameter("timestep", ov::element::f32, ov::PartialShape{1});
    auto text = ctx.parameter("encoder_hidden_states",
                              ov::element::f32,
                              ov::PartialShape{1, 3, cfg.text_dim});

    auto output = model.forward(latents, timesteps, text, nullptr);
    auto ov_model = ctx.build_model({output.output()});

    ov::Core core;
    auto compiled = core.compile_model(ov_model, "CPU");
    auto request = compiled.create_infer_request();

    std::vector<float> latents_data(static_cast<size_t>(1 * cfg.in_channels * 2 * 4 * 4), 0.0f);
    ov::Tensor latents_tensor(ov::element::f32,
                              {1,
                               static_cast<size_t>(cfg.in_channels),
                               2,
                               4,
                               4});
    std::memcpy(latents_tensor.data(), latents_data.data(), latents_data.size() * sizeof(float));
    request.set_input_tensor(0, latents_tensor);

    ov::Tensor timestep_tensor(ov::element::f32, {1});
    timestep_tensor.data<float>()[0] = 0.0f;
    request.set_input_tensor(1, timestep_tensor);

    std::vector<float> text_data(static_cast<size_t>(1 * 3 * cfg.text_dim), 0.0f);
    ov::Tensor text_tensor(ov::element::f32, {1, 3, static_cast<size_t>(cfg.text_dim)});
    std::memcpy(text_tensor.data(), text_data.data(), text_data.size() * sizeof(float));
    request.set_input_tensor(2, text_tensor);

    request.infer();

    std::vector<float> expected(static_cast<size_t>(1 * cfg.out_channels * 2 * 4 * 4), 0.0f);
    test_utils::expect_tensor_near(request.get_output_tensor(0), expected, test_utils::k_tol_default);
}
