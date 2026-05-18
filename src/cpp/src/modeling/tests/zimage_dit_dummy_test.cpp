// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/models/zimage/modeling_zimage_dit.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace test_utils = ov::genai::modeling::tests;

namespace {

void add_zero_weight(test_utils::DummyWeightSource& source,
                     const std::string& name,
                     const ov::Shape& shape) {
    ov::Tensor tensor(ov::element::f32, shape);
    std::memset(tensor.data(), 0, tensor.get_byte_size());
    source.add(name, tensor);
}

}  // namespace

TEST(ZImageDiT, BuildsAndRuns) {
    ov::genai::modeling::BuilderContext ctx;
    ov::genai::modeling::models::ZImageDiTConfig cfg;
    cfg.n_layers = 1;
    cfg.n_refiner_layers = 1;
    cfg.dim = 4;
    cfg.n_heads = 1;
    cfg.n_kv_heads = 1;
    cfg.n_layers = 1;
    cfg.n_refiner_layers = 1;
    cfg.in_channels = 1;
    cfg.cap_feat_dim = 4;
    cfg.adaln_embed_dim = 4;
    cfg.frequency_embedding_size = 4;
    cfg.t_mid_dim = 4;
    cfg.qk_norm = false;

    ov::genai::modeling::models::ZImageDiTModel model(ctx, cfg);

    test_utils::DummyWeightSource source;
    test_utils::DummyWeightFinalizer finalizer;

    // t_embedder
    add_zero_weight(source, "t_embedder.mlp.0.weight", {static_cast<size_t>(cfg.t_mid_dim),
                                                        static_cast<size_t>(cfg.frequency_embedding_size)});
    add_zero_weight(source, "t_embedder.mlp.0.bias", {static_cast<size_t>(cfg.t_mid_dim)});
    add_zero_weight(source, "t_embedder.mlp.2.weight", {static_cast<size_t>(cfg.adaln_embed_dim),
                                                        static_cast<size_t>(cfg.t_mid_dim)});
    add_zero_weight(source, "t_embedder.mlp.2.bias", {static_cast<size_t>(cfg.adaln_embed_dim)});

    // x_embedder
    add_zero_weight(source, "all_x_embedder.2-1.weight", {static_cast<size_t>(cfg.dim),
                                                          static_cast<size_t>(cfg.patch_dim())});
    add_zero_weight(source, "all_x_embedder.2-1.bias", {static_cast<size_t>(cfg.dim)});

    // cap_embedder
    add_zero_weight(source, "cap_embedder.0.weight", {static_cast<size_t>(cfg.cap_feat_dim)});
    add_zero_weight(source, "cap_embedder.1.weight", {static_cast<size_t>(cfg.dim),
                                                      static_cast<size_t>(cfg.cap_feat_dim)});
    add_zero_weight(source, "cap_embedder.1.bias", {static_cast<size_t>(cfg.dim)});

    // pad tokens
    add_zero_weight(source, "x_pad_token", {1, static_cast<size_t>(cfg.dim)});
    add_zero_weight(source, "cap_pad_token", {1, static_cast<size_t>(cfg.dim)});

    // noise_refiner.0
    add_zero_weight(source, "noise_refiner.0.attention.to_q.weight", {static_cast<size_t>(cfg.dim),
                                                                      static_cast<size_t>(cfg.dim)});
    add_zero_weight(source, "noise_refiner.0.attention.to_k.weight", {static_cast<size_t>(cfg.dim),
                                                                      static_cast<size_t>(cfg.dim)});
    add_zero_weight(source, "noise_refiner.0.attention.to_v.weight", {static_cast<size_t>(cfg.dim),
                                                                      static_cast<size_t>(cfg.dim)});
    add_zero_weight(source, "noise_refiner.0.attention.to_out.0.weight", {static_cast<size_t>(cfg.dim),
                                                                          static_cast<size_t>(cfg.dim)});
    add_zero_weight(source, "noise_refiner.0.attention_norm1.weight", {static_cast<size_t>(cfg.dim)});
    add_zero_weight(source, "noise_refiner.0.attention_norm2.weight", {static_cast<size_t>(cfg.dim)});
    add_zero_weight(source, "noise_refiner.0.ffn_norm1.weight", {static_cast<size_t>(cfg.dim)});
    add_zero_weight(source, "noise_refiner.0.ffn_norm2.weight", {static_cast<size_t>(cfg.dim)});
    add_zero_weight(source, "noise_refiner.0.feed_forward.w1.weight", {static_cast<size_t>(cfg.ffn_hidden_dim()),
                                                                       static_cast<size_t>(cfg.dim)});
    add_zero_weight(source, "noise_refiner.0.feed_forward.w2.weight", {static_cast<size_t>(cfg.dim),
                                                                       static_cast<size_t>(cfg.ffn_hidden_dim())});
    add_zero_weight(source, "noise_refiner.0.feed_forward.w3.weight", {static_cast<size_t>(cfg.ffn_hidden_dim()),
                                                                       static_cast<size_t>(cfg.dim)});
    add_zero_weight(source, "noise_refiner.0.adaLN_modulation.0.weight", {static_cast<size_t>(cfg.dim * 4),
                                                                          static_cast<size_t>(cfg.adaln_embed_dim)});
    add_zero_weight(source, "noise_refiner.0.adaLN_modulation.0.bias", {static_cast<size_t>(cfg.dim * 4)});

    // context_refiner.0
    add_zero_weight(source, "context_refiner.0.attention.to_q.weight", {static_cast<size_t>(cfg.dim),
                                                                        static_cast<size_t>(cfg.dim)});
    add_zero_weight(source, "context_refiner.0.attention.to_k.weight", {static_cast<size_t>(cfg.dim),
                                                                        static_cast<size_t>(cfg.dim)});
    add_zero_weight(source, "context_refiner.0.attention.to_v.weight", {static_cast<size_t>(cfg.dim),
                                                                        static_cast<size_t>(cfg.dim)});
    add_zero_weight(source, "context_refiner.0.attention.to_out.0.weight", {static_cast<size_t>(cfg.dim),
                                                                            static_cast<size_t>(cfg.dim)});
    add_zero_weight(source, "context_refiner.0.attention_norm1.weight", {static_cast<size_t>(cfg.dim)});
    add_zero_weight(source, "context_refiner.0.attention_norm2.weight", {static_cast<size_t>(cfg.dim)});
    add_zero_weight(source, "context_refiner.0.ffn_norm1.weight", {static_cast<size_t>(cfg.dim)});
    add_zero_weight(source, "context_refiner.0.ffn_norm2.weight", {static_cast<size_t>(cfg.dim)});
    add_zero_weight(source, "context_refiner.0.feed_forward.w1.weight", {static_cast<size_t>(cfg.ffn_hidden_dim()),
                                                                         static_cast<size_t>(cfg.dim)});
    add_zero_weight(source, "context_refiner.0.feed_forward.w2.weight", {static_cast<size_t>(cfg.dim),
                                                                         static_cast<size_t>(cfg.ffn_hidden_dim())});
    add_zero_weight(source, "context_refiner.0.feed_forward.w3.weight", {static_cast<size_t>(cfg.ffn_hidden_dim()),
                                                                         static_cast<size_t>(cfg.dim)});

    // layers.0
    add_zero_weight(source, "layers.0.attention.to_q.weight", {static_cast<size_t>(cfg.dim),
                                                               static_cast<size_t>(cfg.dim)});
    add_zero_weight(source, "layers.0.attention.to_k.weight", {static_cast<size_t>(cfg.dim),
                                                               static_cast<size_t>(cfg.dim)});
    add_zero_weight(source, "layers.0.attention.to_v.weight", {static_cast<size_t>(cfg.dim),
                                                               static_cast<size_t>(cfg.dim)});
    add_zero_weight(source, "layers.0.attention.to_out.0.weight", {static_cast<size_t>(cfg.dim),
                                                                   static_cast<size_t>(cfg.dim)});
    add_zero_weight(source, "layers.0.attention_norm1.weight", {static_cast<size_t>(cfg.dim)});
    add_zero_weight(source, "layers.0.attention_norm2.weight", {static_cast<size_t>(cfg.dim)});
    add_zero_weight(source, "layers.0.ffn_norm1.weight", {static_cast<size_t>(cfg.dim)});
    add_zero_weight(source, "layers.0.ffn_norm2.weight", {static_cast<size_t>(cfg.dim)});
    add_zero_weight(source, "layers.0.feed_forward.w1.weight", {static_cast<size_t>(cfg.ffn_hidden_dim()),
                                                                static_cast<size_t>(cfg.dim)});
    add_zero_weight(source, "layers.0.feed_forward.w2.weight", {static_cast<size_t>(cfg.dim),
                                                                static_cast<size_t>(cfg.ffn_hidden_dim())});
    add_zero_weight(source, "layers.0.feed_forward.w3.weight", {static_cast<size_t>(cfg.ffn_hidden_dim()),
                                                                static_cast<size_t>(cfg.dim)});
    add_zero_weight(source, "layers.0.adaLN_modulation.0.weight", {static_cast<size_t>(cfg.dim * 4),
                                                                   static_cast<size_t>(cfg.adaln_embed_dim)});
    add_zero_weight(source, "layers.0.adaLN_modulation.0.bias", {static_cast<size_t>(cfg.dim * 4)});

    // final layer
    add_zero_weight(source, "all_final_layer.2-1.linear.weight", {static_cast<size_t>(cfg.patch_dim()),
                                                                  static_cast<size_t>(cfg.dim)});
    add_zero_weight(source, "all_final_layer.2-1.linear.bias", {static_cast<size_t>(cfg.patch_dim())});
    add_zero_weight(source, "all_final_layer.2-1.adaLN_modulation.1.weight", {static_cast<size_t>(cfg.dim),
                                                                              static_cast<size_t>(cfg.adaln_embed_dim)});
    add_zero_weight(source, "all_final_layer.2-1.adaLN_modulation.1.bias", {static_cast<size_t>(cfg.dim)});

    ov::genai::modeling::weights::load_model(model, source, finalizer);

    auto x_tokens = ctx.parameter("x_tokens", ov::element::f32, {1, 2, cfg.patch_dim()});
    auto x_mask = ctx.parameter("x_mask", ov::element::boolean, {1, 2});
    auto cap_feats = ctx.parameter("cap_feats", ov::element::f32, {1, 2, cfg.cap_feat_dim});
    auto cap_mask = ctx.parameter("cap_mask", ov::element::boolean, {1, 2});
    auto timesteps = ctx.parameter("timesteps", ov::element::f32, {1});
    auto x_rope_cos = ctx.parameter("x_rope_cos", ov::element::f32, {1, 2, cfg.head_dim() / 2});
    auto x_rope_sin = ctx.parameter("x_rope_sin", ov::element::f32, {1, 2, cfg.head_dim() / 2});
    auto cap_rope_cos = ctx.parameter("cap_rope_cos", ov::element::f32, {1, 2, cfg.head_dim() / 2});
    auto cap_rope_sin = ctx.parameter("cap_rope_sin", ov::element::f32, {1, 2, cfg.head_dim() / 2});

    auto out = model.forward(x_tokens,
                             x_mask,
                             cap_feats,
                             cap_mask,
                             timesteps,
                             x_rope_cos,
                             x_rope_sin,
                             cap_rope_cos,
                             cap_rope_sin);
    auto ov_model = ctx.build_model({out.output()});

    ov::Core core;
    auto compiled = core.compile_model(ov_model, "GPU");
    auto request = compiled.create_infer_request();

    std::vector<float> x_data(1 * 2 * cfg.patch_dim(), 1.0f);
    std::vector<float> cap_data(1 * 2 * cfg.cap_feat_dim, 1.0f);
    std::vector<char> mask_data(2, 1);
    std::vector<float> rope_cos_data(1 * 2 * (cfg.head_dim() / 2), 1.0f);
    std::vector<float> rope_sin_data(1 * 2 * (cfg.head_dim() / 2), 0.0f);
    std::vector<float> t_data(1, 0.5f);

    ov::Tensor x_tensor(ov::element::f32, {1, 2, static_cast<size_t>(cfg.patch_dim())});
    ov::Tensor x_mask_tensor(ov::element::boolean, {1, 2});
    ov::Tensor cap_tensor(ov::element::f32, {1, 2, static_cast<size_t>(cfg.cap_feat_dim)});
    ov::Tensor cap_mask_tensor(ov::element::boolean, {1, 2});
    ov::Tensor t_tensor(ov::element::f32, {1});
    ov::Tensor x_cos_tensor(ov::element::f32, {1, 2, static_cast<size_t>(cfg.head_dim() / 2)});
    ov::Tensor x_sin_tensor(ov::element::f32, {1, 2, static_cast<size_t>(cfg.head_dim() / 2)});
    ov::Tensor cap_cos_tensor(ov::element::f32, {1, 2, static_cast<size_t>(cfg.head_dim() / 2)});
    ov::Tensor cap_sin_tensor(ov::element::f32, {1, 2, static_cast<size_t>(cfg.head_dim() / 2)});

    std::memcpy(x_tensor.data(), x_data.data(), x_data.size() * sizeof(float));
    std::memcpy(x_mask_tensor.data(), mask_data.data(), mask_data.size() * sizeof(char));
    std::memcpy(cap_tensor.data(), cap_data.data(), cap_data.size() * sizeof(float));
    std::memcpy(cap_mask_tensor.data(), mask_data.data(), mask_data.size() * sizeof(char));
    std::memcpy(t_tensor.data(), t_data.data(), t_data.size() * sizeof(float));
    std::memcpy(x_cos_tensor.data(), rope_cos_data.data(), rope_cos_data.size() * sizeof(float));
    std::memcpy(x_sin_tensor.data(), rope_sin_data.data(), rope_sin_data.size() * sizeof(float));
    std::memcpy(cap_cos_tensor.data(), rope_cos_data.data(), rope_cos_data.size() * sizeof(float));
    std::memcpy(cap_sin_tensor.data(), rope_sin_data.data(), rope_sin_data.size() * sizeof(float));

    request.set_input_tensor(0, x_tensor);
    request.set_input_tensor(1, x_mask_tensor);
    request.set_input_tensor(2, cap_tensor);
    request.set_input_tensor(3, cap_mask_tensor);
    request.set_input_tensor(4, t_tensor);
    request.set_input_tensor(5, x_cos_tensor);
    request.set_input_tensor(6, x_sin_tensor);
    request.set_input_tensor(7, cap_cos_tensor);
    request.set_input_tensor(8, cap_sin_tensor);
    request.infer();

    std::vector<float> expected(1 * 2 * cfg.patch_dim(), 0.0f);
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_default);
}
