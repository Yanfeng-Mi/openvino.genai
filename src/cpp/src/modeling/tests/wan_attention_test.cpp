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

void add_attention_weights(test_utils::DummyWeightSource& source,
                           const std::string& prefix,
                           int32_t dim,
                           int32_t added_kv_dim,
                           bool with_added_kv) {
    const ov::Shape weight_shape{static_cast<size_t>(dim), static_cast<size_t>(dim)};
    const ov::Shape bias_shape{static_cast<size_t>(dim)};

    add_const_weight(source, prefix + ".to_q.weight", weight_shape, 0.0f);
    add_const_weight(source, prefix + ".to_q.bias", bias_shape, 0.0f);
    add_const_weight(source, prefix + ".to_k.weight", weight_shape, 0.0f);
    add_const_weight(source, prefix + ".to_k.bias", bias_shape, 0.0f);
    add_const_weight(source, prefix + ".to_v.weight", weight_shape, 0.0f);
    add_const_weight(source, prefix + ".to_v.bias", bias_shape, 0.0f);
    add_const_weight(source, prefix + ".to_out.0.weight", weight_shape, 0.0f);
    add_const_weight(source, prefix + ".to_out.0.bias", bias_shape, 0.0f);

    add_const_weight(source, prefix + ".norm_q.weight", bias_shape, 1.0f);
    add_const_weight(source, prefix + ".norm_k.weight", bias_shape, 1.0f);

    if (with_added_kv) {
        const ov::Shape add_weight_shape{static_cast<size_t>(dim), static_cast<size_t>(added_kv_dim)};
        const ov::Shape add_bias_shape{static_cast<size_t>(dim)};
        add_const_weight(source, prefix + ".add_k_proj.weight", add_weight_shape, 0.0f);
        add_const_weight(source, prefix + ".add_k_proj.bias", add_bias_shape, 0.0f);
        add_const_weight(source, prefix + ".add_v_proj.weight", add_weight_shape, 0.0f);
        add_const_weight(source, prefix + ".add_v_proj.bias", add_bias_shape, 0.0f);
        add_const_weight(source, prefix + ".norm_added_k.weight", add_bias_shape, 1.0f);
    }
}

}  // namespace

TEST(WanAttentionTest, SelfAndCrossPaths) {
    ov::genai::modeling::BuilderContext ctx;

    const int32_t dim = 12;
    const int32_t heads = 2;
    const int32_t head_dim = 6;
    const int32_t added_kv_dim = 6;

    ov::genai::modeling::models::WanAttention self_attn(ctx,
                                                        "attn1",
                                                        dim,
                                                        heads,
                                                        head_dim,
                                                        1e-6f,
                                                        "rms_norm_across_heads",
                                                        std::nullopt);
    ov::genai::modeling::models::WanAttention cross_attn(ctx,
                                                         "attn2",
                                                         dim,
                                                         heads,
                                                         head_dim,
                                                         1e-6f,
                                                         "rms_norm_across_heads",
                                                         added_kv_dim);

    test_utils::DummyWeightSource source_self;
    test_utils::DummyWeightSource source_cross;
    test_utils::DummyWeightFinalizer finalizer;
    add_attention_weights(source_self, "attn1", dim, added_kv_dim, false);
    add_attention_weights(source_cross, "attn2", dim, added_kv_dim, true);

    ov::genai::modeling::weights::load_model(self_attn, source_self, finalizer);
    ov::genai::modeling::weights::load_model(cross_attn, source_cross, finalizer);

    auto hidden_states = ctx.parameter("hidden_states", ov::element::f32, ov::PartialShape{1, 3, dim});
    auto encoder_hidden_states = ctx.parameter("encoder_hidden_states", ov::element::f32, ov::PartialShape{1, 4, dim});
    auto encoder_hidden_states_image =
        ctx.parameter("encoder_hidden_states_image", ov::element::f32, ov::PartialShape{1, 2, added_kv_dim});
    auto rotary_cos = ctx.parameter("rotary_cos", ov::element::f32, ov::PartialShape{1, 3, head_dim / 2});
    auto rotary_sin = ctx.parameter("rotary_sin", ov::element::f32, ov::PartialShape{1, 3, head_dim / 2});

    auto output_self = self_attn.forward(hidden_states, nullptr, nullptr, &rotary_cos, &rotary_sin);
    auto output_cross = cross_attn.forward(hidden_states,
                                           &encoder_hidden_states,
                                           &encoder_hidden_states_image,
                                           nullptr,
                                           nullptr);
    auto ov_model = ctx.build_model({output_self.output(), output_cross.output()});

    ov::Core core;
    auto compiled = core.compile_model(ov_model, "CPU");
    auto request = compiled.create_infer_request();

    std::vector<float> hidden_data(static_cast<size_t>(1 * 3 * dim), 0.25f);
    ov::Tensor hidden_tensor(ov::element::f32, {1, 3, static_cast<size_t>(dim)});
    std::memcpy(hidden_tensor.data(), hidden_data.data(), hidden_data.size() * sizeof(float));
    request.set_input_tensor(0, hidden_tensor);

    std::vector<float> text_data(static_cast<size_t>(1 * 4 * dim), 0.1f);
    ov::Tensor text_tensor(ov::element::f32, {1, 4, static_cast<size_t>(dim)});
    std::memcpy(text_tensor.data(), text_data.data(), text_data.size() * sizeof(float));
    request.set_input_tensor(1, text_tensor);

    std::vector<float> image_data(static_cast<size_t>(1 * 2 * added_kv_dim), 0.2f);
    ov::Tensor image_tensor(ov::element::f32, {1, 2, static_cast<size_t>(added_kv_dim)});
    std::memcpy(image_tensor.data(), image_data.data(), image_data.size() * sizeof(float));
    request.set_input_tensor(2, image_tensor);

    std::vector<float> cos_data(static_cast<size_t>(1 * 3 * (head_dim / 2)), 1.0f);
    ov::Tensor cos_tensor(ov::element::f32, {1, 3, static_cast<size_t>(head_dim / 2)});
    std::memcpy(cos_tensor.data(), cos_data.data(), cos_data.size() * sizeof(float));
    request.set_input_tensor(3, cos_tensor);

    std::vector<float> sin_data(static_cast<size_t>(1 * 3 * (head_dim / 2)), 0.0f);
    ov::Tensor sin_tensor(ov::element::f32, {1, 3, static_cast<size_t>(head_dim / 2)});
    std::memcpy(sin_tensor.data(), sin_data.data(), sin_data.size() * sizeof(float));
    request.set_input_tensor(4, sin_tensor);

    request.infer();

    std::vector<float> expected(static_cast<size_t>(1 * 3 * dim), 0.0f);
    test_utils::expect_tensor_near(request.get_output_tensor(0), expected, test_utils::k_tol_default);
    test_utils::expect_tensor_near(request.get_output_tensor(1), expected, test_utils::k_tol_default);
}
