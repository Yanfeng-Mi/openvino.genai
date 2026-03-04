// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/models/zimage/modeling_zimage_vae.hpp"
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

TEST(ZImageVAE, DecoderBuildsAndRuns) {
    ov::genai::modeling::BuilderContext ctx;
    ov::genai::modeling::models::ZImageVAEConfig cfg;
    cfg.in_channels = 3;
    cfg.out_channels = 3;
    cfg.latent_channels = 2;
    cfg.block_out_channels = {4, 8};
    cfg.layers_per_block = 1;
    cfg.norm_num_groups = 1;
    cfg.mid_block_add_attention = true;

    ov::genai::modeling::models::ZImageVAEDecoder model(ctx, cfg);

    test_utils::DummyWeightSource source;
    test_utils::DummyWeightFinalizer finalizer;

    // conv_in
    add_zero_weight(source, "decoder.conv_in.weight", {8, 2, 3, 3});
    add_zero_weight(source, "decoder.conv_in.bias", {8});

    // mid_block resnets
    for (int32_t i = 0; i < 2; ++i) {
        const std::string base = "decoder.mid_block.resnets." + std::to_string(i) + ".";
        add_zero_weight(source, base + "norm1.weight", {8});
        add_zero_weight(source, base + "norm1.bias", {8});
        add_zero_weight(source, base + "conv1.weight", {8, 8, 3, 3});
        add_zero_weight(source, base + "conv1.bias", {8});
        add_zero_weight(source, base + "norm2.weight", {8});
        add_zero_weight(source, base + "norm2.bias", {8});
        add_zero_weight(source, base + "conv2.weight", {8, 8, 3, 3});
        add_zero_weight(source, base + "conv2.bias", {8});
    }

    // mid_block attention
    add_zero_weight(source, "decoder.mid_block.attentions.0.group_norm.weight", {8});
    add_zero_weight(source, "decoder.mid_block.attentions.0.group_norm.bias", {8});
    add_zero_weight(source, "decoder.mid_block.attentions.0.to_q.weight", {8, 8});
    add_zero_weight(source, "decoder.mid_block.attentions.0.to_q.bias", {8});
    add_zero_weight(source, "decoder.mid_block.attentions.0.to_k.weight", {8, 8});
    add_zero_weight(source, "decoder.mid_block.attentions.0.to_k.bias", {8});
    add_zero_weight(source, "decoder.mid_block.attentions.0.to_v.weight", {8, 8});
    add_zero_weight(source, "decoder.mid_block.attentions.0.to_v.bias", {8});
    add_zero_weight(source, "decoder.mid_block.attentions.0.to_out.0.weight", {8, 8});
    add_zero_weight(source, "decoder.mid_block.attentions.0.to_out.0.bias", {8});

    // up_blocks
    std::vector<int32_t> reversed = cfg.block_out_channels;
    std::reverse(reversed.begin(), reversed.end());
    int32_t prev_out = reversed.front();
    const int32_t num_layers = cfg.layers_per_block + 1;

    for (size_t i = 0; i < reversed.size(); ++i) {
        int32_t out_ch = reversed[i];
        for (int32_t j = 0; j < num_layers; ++j) {
            int32_t in_ch = (j == 0) ? prev_out : out_ch;
            std::string base = "decoder.up_blocks." + std::to_string(i) + ".resnets." +
                               std::to_string(j) + ".";
            add_zero_weight(source, base + "norm1.weight", {static_cast<size_t>(in_ch)});
            add_zero_weight(source, base + "norm1.bias", {static_cast<size_t>(in_ch)});
            add_zero_weight(source, base + "conv1.weight",
                            {static_cast<size_t>(out_ch), static_cast<size_t>(in_ch), 3, 3});
            add_zero_weight(source, base + "conv1.bias", {static_cast<size_t>(out_ch)});
            add_zero_weight(source, base + "norm2.weight", {static_cast<size_t>(out_ch)});
            add_zero_weight(source, base + "norm2.bias", {static_cast<size_t>(out_ch)});
            add_zero_weight(source, base + "conv2.weight",
                            {static_cast<size_t>(out_ch), static_cast<size_t>(out_ch), 3, 3});
            add_zero_weight(source, base + "conv2.bias", {static_cast<size_t>(out_ch)});
            if (in_ch != out_ch) {
                add_zero_weight(source, base + "conv_shortcut.weight",
                                {static_cast<size_t>(out_ch), static_cast<size_t>(in_ch), 1, 1});
                add_zero_weight(source, base + "conv_shortcut.bias", {static_cast<size_t>(out_ch)});
            }
        }
        if (i + 1 < reversed.size()) {
            std::string base = "decoder.up_blocks." + std::to_string(i) + ".upsamplers.0.conv.";
            add_zero_weight(source, base + "weight",
                            {static_cast<size_t>(out_ch), static_cast<size_t>(out_ch), 3, 3});
            add_zero_weight(source, base + "bias", {static_cast<size_t>(out_ch)});
        }
        prev_out = out_ch;
    }

    // conv_norm_out + conv_out
    add_zero_weight(source, "decoder.conv_norm_out.weight", {4});
    add_zero_weight(source, "decoder.conv_norm_out.bias", {4});
    add_zero_weight(source, "decoder.conv_out.weight", {3, 4, 3, 3});
    add_zero_weight(source, "decoder.conv_out.bias", {3});

    ov::genai::modeling::weights::load_model(model, source, finalizer);

    const int64_t batch = 1;
    const int64_t in_h = 4;
    const int64_t in_w = 4;
    auto latents = ctx.parameter("latents", ov::element::f32, {batch, cfg.latent_channels, in_h, in_w});
    auto output = model.forward(latents);
    auto ov_model = ctx.build_model({output.output()});

    ov::Core core;
    auto compiled = core.compile_model(ov_model, "CPU");
    auto request = compiled.create_infer_request();

    std::vector<float> latents_data(static_cast<size_t>(batch * cfg.latent_channels * in_h * in_w), 0.0f);
    ov::Tensor latents_tensor(ov::element::f32,
                              {static_cast<size_t>(batch),
                               static_cast<size_t>(cfg.latent_channels),
                               static_cast<size_t>(in_h),
                               static_cast<size_t>(in_w)});
    std::memcpy(latents_tensor.data(), latents_data.data(), latents_data.size() * sizeof(float));

    request.set_input_tensor(0, latents_tensor);
    request.infer();

    const int64_t up_factor = 1LL << (static_cast<int64_t>(cfg.block_out_channels.size()) - 1);
    const int64_t out_h = in_h * up_factor;
    const int64_t out_w = in_w * up_factor;
    std::vector<float> expected(static_cast<size_t>(batch * cfg.out_channels * out_h * out_w), 0.0f);
    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_default);
}
