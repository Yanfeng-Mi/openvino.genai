// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/zimage/modeling_zimage_vae.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <openvino/core/except.hpp>
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/llm.hpp"
#include "modeling/ops/nn.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/ops/shape.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace {

ov::genai::modeling::Tensor add_bias_if_present(const ov::genai::modeling::Tensor& x,
                                                const ov::genai::modeling::Tensor* bias) {
    if (!bias) {
        return x;
    }
    return x + *bias;
}

ov::genai::modeling::Tensor linear_mm(const ov::genai::modeling::Tensor& x,
                                      const ov::genai::modeling::Tensor& weight,
                                      const ov::genai::modeling::Tensor* bias) {
    auto out = ov::genai::modeling::ops::matmul(x, weight, false, true);
    if (bias) {
        out = out + *bias;
    }
    return out;
}

auto set_name = [](auto node, const std::string& name) {
    node->output(0).set_names({name});
    node->set_friendly_name(name);
};

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

VAEGroupNorm::VAEGroupNorm(BuilderContext& ctx,
                           const std::string& name,
                           int32_t num_groups,
                           float eps,
                           Module* parent)
    : Module(name, ctx, parent),
      num_groups_(num_groups),
      eps_(eps) {
    weight_ = &register_parameter("weight");
    bias_ = &register_parameter("bias");
}

Tensor VAEGroupNorm::forward(const Tensor& input) const {
    return ops::nn::group_norm(input, weight_->value(), &bias_->value(), num_groups_, eps_);
}

VAEResnetBlock2D::VAEResnetBlock2D(BuilderContext& ctx,
                                   const std::string& name,
                                   int32_t in_channels,
                                   int32_t out_channels,
                                   int32_t norm_groups,
                                   float eps,
                                   Module* parent)
    : Module(name, ctx, parent),
      in_channels_(in_channels),
      out_channels_(out_channels),
      norm1_(ctx, "norm1", norm_groups, eps, this),
      norm2_(ctx, "norm2", norm_groups, eps, this) {
    register_module("norm1", &norm1_);
    register_module("norm2", &norm2_);

    conv1_weight_ = &register_parameter("conv1.weight");
    conv1_bias_ = &register_parameter("conv1.bias");
    conv2_weight_ = &register_parameter("conv2.weight");
    conv2_bias_ = &register_parameter("conv2.bias");

    if (in_channels_ != out_channels_) {
        conv_shortcut_weight_ = &register_parameter("conv_shortcut.weight");
        conv_shortcut_bias_ = &register_parameter("conv_shortcut.bias");
    }
}

Tensor VAEResnetBlock2D::forward(const Tensor& input) const {
    auto h = norm1_.forward(input);
    h = ops::silu(h);
    h = h.to(conv1_weight_->value().dtype());
    h = ops::nn::conv2d(h, conv1_weight_->value(), conv1_bias_->value(), {1, 1}, {1, 1}, {1, 1});

    h = norm2_.forward(h);
    h = ops::silu(h);
    h = h.to(conv2_weight_->value().dtype());
    h = ops::nn::conv2d(h, conv2_weight_->value(), conv2_bias_->value(), {1, 1}, {1, 1}, {1, 1});

    auto residual = input;
    if (conv_shortcut_weight_) {
        residual = residual.to(conv_shortcut_weight_->value().dtype());
        residual = ops::nn::conv2d(residual, conv_shortcut_weight_->value(), conv_shortcut_bias_->value(),
                                   {1, 1}, {0, 0}, {0, 0});
    }
    if (residual.dtype() != h.dtype()) {
        residual = residual.to(h.dtype());
    }
    return residual + h;
}

VAEAttention::VAEAttention(BuilderContext& ctx,
                           const std::string& name,
                           int32_t channels,
                           int32_t norm_groups,
                           float eps,
                           Module* parent)
    : Module(name, ctx, parent),
      channels_(channels),
      group_norm_(ctx, "group_norm", norm_groups, eps, this) {
    if (channels_ <= 0) {
        OPENVINO_THROW("VAEAttention requires positive channels");
    }
    register_module("group_norm", &group_norm_);

    q_weight_ = &register_parameter("to_q.weight");
    q_bias_ = &register_parameter("to_q.bias");
    k_weight_ = &register_parameter("to_k.weight");
    k_bias_ = &register_parameter("to_k.bias");
    v_weight_ = &register_parameter("to_v.weight");
    v_bias_ = &register_parameter("to_v.bias");
    o_weight_ = &register_parameter("to_out.0.weight");
    o_bias_ = &register_parameter("to_out.0.bias");
}

Tensor VAEAttention::forward(const Tensor& input) const {
    auto normed = group_norm_.forward(input);
    auto flat = normed.reshape({0, channels_, -1}).permute({0, 2, 1});
    flat = flat.to(q_weight_->value().dtype());

    auto q = linear_mm(flat, q_weight_->value(), nullptr);
    if (q_bias_) {
        q = q + q_bias_->value();
    }
    auto k = linear_mm(flat, k_weight_->value(), nullptr);
    if (k_bias_) {
        k = k + k_bias_->value();
    }
    auto v = linear_mm(flat, v_weight_->value(), nullptr);
    if (v_bias_) {
        v = v + v_bias_->value();
    }

    const int32_t num_heads = 1;
    const int32_t head_dim = channels_;
    const float scaling = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto q_heads = q.reshape({0, 0, num_heads, head_dim}).permute({0, 2, 1, 3});
    auto k_heads = k.reshape({0, 0, num_heads, head_dim}).permute({0, 2, 1, 3});
    auto v_heads = v.reshape({0, 0, num_heads, head_dim}).permute({0, 2, 1, 3});

    auto* policy = &ctx().op_policy();
    auto context = ops::llm::sdpa(q_heads, k_heads, v_heads, scaling, 3, nullptr, false, policy);

    auto merged = context.permute({0, 2, 1, 3}).reshape({0, 0, channels_});
    auto out = linear_mm(merged, o_weight_->value(), nullptr);
    if (o_bias_) {
        out = out + o_bias_->value();
    }

    auto out_chw = out.permute({0, 2, 1});
    auto target_shape = shape::make({shape::dim(input, 0),
                                     shape::dim(input, 1),
                                     shape::dim(input, 2),
                                     shape::dim(input, 3)});
    auto out_reshaped = out_chw.reshape(target_shape);

    auto residual = input;
    if (residual.dtype() != out_reshaped.dtype()) {
        residual = residual.to(out_reshaped.dtype());
    }
    return residual + out_reshaped;
}

VAEUpsample2D::VAEUpsample2D(BuilderContext& ctx, const std::string& name, int32_t channels, Module* parent)
    : Module(name, ctx, parent),
      channels_(channels) {
    conv_weight_ = &register_parameter("conv.weight");
    conv_bias_ = &register_parameter("conv.bias");
}

Tensor VAEUpsample2D::forward(const Tensor& input) const {
    auto up = ops::nn::upsample_nearest(input, 2, 2);
    up = up.to(conv_weight_->value().dtype());
    return ops::nn::conv2d(up, conv_weight_->value(), conv_bias_->value(), {1, 1}, {1, 1}, {1, 1});
}

VAEUpDecoderBlock2D::VAEUpDecoderBlock2D(BuilderContext& ctx,
                                         const std::string& name,
                                         int32_t in_channels,
                                         int32_t out_channels,
                                         int32_t num_layers,
                                         int32_t norm_groups,
                                         float eps,
                                         bool add_upsample,
                                         Module* parent)
    : Module(name, ctx, parent) {
    resnets_.reserve(static_cast<size_t>(num_layers));
    for (int32_t i = 0; i < num_layers; ++i) {
        const std::string resnet_name = "resnets." + std::to_string(i);
        const int32_t res_in = (i == 0) ? in_channels : out_channels;
        resnets_.emplace_back(ctx, resnet_name, res_in, out_channels, norm_groups, eps, this);
        register_module(resnet_name, &resnets_.back());
    }

    if (add_upsample) {
        upsampler_ = std::make_unique<VAEUpsample2D>(ctx, "upsamplers.0", out_channels, this);
        register_module("upsamplers.0", upsampler_.get());
    }
}

Tensor VAEUpDecoderBlock2D::forward(const Tensor& input) const {
    auto x = input;
    for (const auto& resnet : resnets_) {
        x = resnet.forward(x);
    }
    if (upsampler_) {
        x = upsampler_->forward(x);
    }
    return x;
}

VAEMidBlock2D::VAEMidBlock2D(BuilderContext& ctx,
                             const std::string& name,
                             int32_t channels,
                             int32_t norm_groups,
                             float eps,
                             bool add_attention,
                             Module* parent)
    : Module(name, ctx, parent),
      resnet1_(ctx, "resnets.0", channels, channels, norm_groups, eps, this),
      resnet2_(ctx, "resnets.1", channels, channels, norm_groups, eps, this) {
    register_module("resnets.0", &resnet1_);
    register_module("resnets.1", &resnet2_);

    if (add_attention) {
        attention_ = std::make_unique<VAEAttention>(ctx, "attentions.0", channels, norm_groups, eps, this);
        register_module("attentions.0", attention_.get());
    }
}

Tensor VAEMidBlock2D::forward(const Tensor& input) const {
    auto x = resnet1_.forward(input);
    if (attention_) {
        x = attention_->forward(x);
    }
    return resnet2_.forward(x);
}

ZImageVAEDecoder::ZImageVAEDecoder(BuilderContext& ctx, const ZImageVAEConfig& cfg, Module* parent)
    : Module("decoder", ctx, parent),
      cfg_(cfg),
      mid_block_(ctx, "mid_block", cfg.block_out_channels.back(), cfg.norm_num_groups, 1e-6f,
                 cfg.mid_block_add_attention, this),
      norm_out_(ctx, "conv_norm_out", cfg.norm_num_groups, 1e-6f, this) {
    conv_in_weight_ = &register_parameter("conv_in.weight");
    conv_in_bias_ = &register_parameter("conv_in.bias");
    register_module("mid_block", &mid_block_);
    register_module("conv_norm_out", &norm_out_);
    conv_out_weight_ = &register_parameter("conv_out.weight");
    conv_out_bias_ = &register_parameter("conv_out.bias");

    std::vector<int32_t> reversed_channels = cfg.block_out_channels;
    std::reverse(reversed_channels.begin(), reversed_channels.end());
    int32_t prev_output_channel = reversed_channels.front();

    up_blocks_.reserve(reversed_channels.size());
    for (size_t i = 0; i < reversed_channels.size(); ++i) {
        int32_t out_channels = reversed_channels[i];
        bool add_upsample = (i + 1) < reversed_channels.size();
        const std::string name = "up_blocks." + std::to_string(i);
        up_blocks_.emplace_back(ctx, name, prev_output_channel, out_channels,
                                cfg.layers_per_block + 1, cfg.norm_num_groups, 1e-6f, add_upsample, this);
        register_module(name, &up_blocks_.back());
        prev_output_channel = out_channels;
    }
}

Tensor ZImageVAEDecoder::forward(const Tensor& latents) const {
    auto x = latents.to(conv_in_weight_->value().dtype());
    x = ops::nn::conv2d(x, conv_in_weight_->value(), conv_in_bias_->value(), {1, 1}, {1, 1}, {1, 1});
    x = mid_block_.forward(x);
    for (const auto& block : up_blocks_) {
        x = block.forward(x);
    }
    x = norm_out_.forward(x);
    x = ops::silu(x);
    x = x.to(conv_out_weight_->value().dtype());
    return ops::nn::conv2d(x, conv_out_weight_->value(), conv_out_bias_->value(), {1, 1}, {1, 1}, {1, 1});
}

std::shared_ptr<ov::Model> create_zimage_vae_decoder_model(
    const ZImageVAEConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    ZImageVAEDecoder model(ctx, cfg);

    ov::genai::modeling::weights::LoadOptions options;
    options.allow_unmatched = true;
    options.allow_missing = false;
    options.report_missing = true;
    options.report_unmatched = true;
    auto report = ov::genai::modeling::weights::load_model(model, source, finalizer, options);
    (void)report;

    auto latents = ctx.parameter("latents",
                                 ov::element::f32,
                                 ov::PartialShape{-1, cfg.latent_channels, -1, -1});
    auto output = model.forward(latents);
    auto result = std::make_shared<ov::op::v0::Result>(output.output());
    set_name(result, "sample");
    return ctx.build_model({result->output(0)});
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
