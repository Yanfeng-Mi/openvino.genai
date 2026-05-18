// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>

#include "modeling/models/wan/modeling_wan_vae.hpp"
#include "modeling/tests/test_utils.hpp"

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

void add_conv3d_weights(test_utils::DummyWeightSource& source,
                        const std::string& prefix,
                        size_t out_ch,
                        size_t in_ch,
                        size_t kt,
                        size_t kh,
                        size_t kw) {
    add_const_weight(source, prefix + ".weight", {out_ch, in_ch, kt, kh, kw}, 0.0f);
    add_const_weight(source, prefix + ".bias", {out_ch}, 0.0f);
}

void add_conv2d_weights(test_utils::DummyWeightSource& source,
                        const std::string& prefix,
                        size_t out_ch,
                        size_t in_ch,
                        size_t kh,
                        size_t kw) {
    add_const_weight(source, prefix + ".weight", {out_ch, in_ch, kh, kw}, 0.0f);
    add_const_weight(source, prefix + ".bias", {out_ch}, 0.0f);
}

void add_rms_gamma(test_utils::DummyWeightSource& source,
                   const std::string& name,
                   size_t channels,
                   bool images) {
    if (images) {
        add_const_weight(source, name, {channels, 1, 1}, 1.0f);
    } else {
        add_const_weight(source, name, {channels, 1, 1, 1}, 1.0f);
    }
}

void add_residual_block(test_utils::DummyWeightSource& source,
                        const std::string& prefix,
                        size_t in_dim,
                        size_t out_dim) {
    add_rms_gamma(source, prefix + ".norm1.gamma", in_dim, false);
    add_conv3d_weights(source, prefix + ".conv1", out_dim, in_dim, 3, 3, 3);
    add_rms_gamma(source, prefix + ".norm2.gamma", out_dim, false);
    add_conv3d_weights(source, prefix + ".conv2", out_dim, out_dim, 3, 3, 3);
    if (in_dim != out_dim) {
        add_conv3d_weights(source, prefix + ".conv_shortcut", out_dim, in_dim, 1, 1, 1);
    }
}

void add_attention_block(test_utils::DummyWeightSource& source,
                         const std::string& prefix,
                         size_t dim) {
    add_rms_gamma(source, prefix + ".norm.gamma", dim, true);
    add_conv2d_weights(source, prefix + ".to_qkv", dim * 3, dim, 1, 1);
    add_conv2d_weights(source, prefix + ".proj", dim, dim, 1, 1);
}

}  // namespace

TEST(WanVAEEncoderTest, BuildsAndRuns) {
    ov::genai::modeling::models::WanVAEConfig cfg;
    cfg.base_dim = 8;
    cfg.decoder_base_dim = 8;
    cfg.z_dim = 4;
    cfg.dim_mult = {1, 2};
    cfg.num_res_blocks = 1;
    cfg.attn_scales = {};
    cfg.temperal_downsample = {false};
    cfg.in_channels = 3;
    cfg.out_channels = 3;
    cfg.patch_size = std::nullopt;
    cfg.finalize();

    test_utils::DummyWeightSource source;
    test_utils::DummyWeightFinalizer finalizer;

    const size_t base_dim = static_cast<size_t>(cfg.base_dim);
    const size_t in_channels = static_cast<size_t>(cfg.in_channels);
    const size_t z_dim2 = static_cast<size_t>(cfg.z_dim * 2);
    const size_t dim_last = static_cast<size_t>(cfg.base_dim * cfg.dim_mult.back());

    add_conv3d_weights(source, "encoder.conv_in", base_dim, in_channels, 3, 3, 3);
    add_residual_block(source, "encoder.down_blocks.0", base_dim, base_dim);
    add_conv2d_weights(source, "encoder.down_blocks.1.resample.1", base_dim, base_dim, 3, 3);
    add_residual_block(source, "encoder.down_blocks.2", base_dim, dim_last);

    add_residual_block(source, "encoder.mid_block.resnets.0", dim_last, dim_last);
    add_attention_block(source, "encoder.mid_block.attentions.0", dim_last);
    add_residual_block(source, "encoder.mid_block.resnets.1", dim_last, dim_last);

    add_rms_gamma(source, "encoder.norm_out.gamma", dim_last, false);
    add_conv3d_weights(source, "encoder.conv_out", z_dim2, dim_last, 3, 3, 3);
    add_conv3d_weights(source, "quant_conv", z_dim2, z_dim2, 1, 1, 1);

    auto model = ov::genai::modeling::models::create_wan_vae_encoder_model(cfg, source, finalizer);

    ov::Core core;
    auto compiled = core.compile_model(model, "CPU");
    auto request = compiled.create_infer_request();

    const ov::Shape input_shape{1, static_cast<size_t>(cfg.in_channels), 2, 4, 4};
    ov::Tensor input(ov::element::f32, input_shape);
    std::fill_n(input.data<float>(), input.get_size(), 0.0f);
    request.set_tensor("sample", input);
    request.infer();

    ov::Tensor output = request.get_tensor("latent");
    EXPECT_EQ(output.get_element_type(), ov::element::f32);
    EXPECT_EQ(output.get_shape(), (ov::Shape{1, z_dim2, 2, 2, 2}));
    std::vector<float> expected(output.get_size(), 0.0f);
    test_utils::expect_tensor_near(output, expected, test_utils::k_tol_default);
}
