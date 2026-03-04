// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/wan/modeling_wan_vae.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <openvino/core/except.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/llm.hpp"
#include "modeling/ops/nn.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/ops/shape.hpp"
#include "modeling/ops/tensor_ops.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace {

auto set_name = [](const std::shared_ptr<ov::Node>& node, const std::string& name) {
    node->output(0).set_names({name});
    node->set_friendly_name(name);
};

ov::Output<ov::Node> div_dim(const ov::Output<ov::Node>& dim, int64_t divisor, ov::genai::modeling::OpContext* ctx) {
    auto denom = ov::genai::modeling::ops::const_scalar(ctx, static_cast<int64_t>(divisor));
    auto div = std::make_shared<ov::op::v1::Divide>(dim, denom, ov::op::AutoBroadcastType::NUMPY);
    return std::make_shared<ov::op::v0::Convert>(div, ov::element::i64);
}

ov::Output<ov::Node> mul_dim(const ov::Output<ov::Node>& a, const ov::Output<ov::Node>& b) {
    return std::make_shared<ov::op::v1::Multiply>(a, b);
}

ov::genai::modeling::Tensor clamp(const ov::genai::modeling::Tensor& input, float min_val, float max_val) {
    auto* ctx = input.context();
    auto min_node = ov::genai::modeling::ops::const_scalar(ctx, min_val);
    auto max_node = ov::genai::modeling::ops::const_scalar(ctx, max_val);
    auto max_op = std::make_shared<ov::op::v1::Maximum>(input.output(), min_node);
    auto min_op = std::make_shared<ov::op::v1::Minimum>(max_op, max_node);
    return ov::genai::modeling::Tensor(min_op, ctx);
}

ov::genai::modeling::Tensor patchify(const ov::genai::modeling::Tensor& input, int32_t patch_size) {
    if (patch_size <= 1) {
        return input;
    }
    auto* ctx = input.context();
    auto batch = ov::genai::modeling::shape::dim(input, 0);
    auto channels = ov::genai::modeling::shape::dim(input, 1);
    auto frames = ov::genai::modeling::shape::dim(input, 2);
    auto height = ov::genai::modeling::shape::dim(input, 3);
    auto width = ov::genai::modeling::shape::dim(input, 4);
    auto p = ov::genai::modeling::ops::const_vec(ctx, std::vector<int64_t>{patch_size});
    auto h_div = div_dim(height, patch_size, ctx);
    auto w_div = div_dim(width, patch_size, ctx);

    auto shape1 = ov::genai::modeling::shape::make({batch, channels, frames, h_div, p, w_div, p});
    auto reshaped = input.reshape(shape1);
    auto permuted = reshaped.permute({0, 1, 6, 4, 2, 3, 5});

    auto c_mul = mul_dim(channels, p);
    auto c_mul2 = mul_dim(c_mul, p);
    auto shape2 = ov::genai::modeling::shape::make({batch, c_mul2, frames, h_div, w_div});
    return permuted.reshape(shape2);
}

ov::genai::modeling::Tensor unpatchify(const ov::genai::modeling::Tensor& input, int32_t patch_size) {
    if (patch_size <= 1) {
        return input;
    }
    auto* ctx = input.context();
    auto batch = ov::genai::modeling::shape::dim(input, 0);
    auto channels_p = ov::genai::modeling::shape::dim(input, 1);
    auto frames = ov::genai::modeling::shape::dim(input, 2);
    auto height = ov::genai::modeling::shape::dim(input, 3);
    auto width = ov::genai::modeling::shape::dim(input, 4);
    auto p = ov::genai::modeling::ops::const_vec(ctx, std::vector<int64_t>{patch_size});
    auto p2 = mul_dim(p, p);
    ov::Output<ov::Node> channels = std::make_shared<ov::op::v1::Divide>(channels_p, p2, ov::op::AutoBroadcastType::NUMPY);
    channels = std::make_shared<ov::op::v0::Convert>(channels, ov::element::i64);

    auto shape1 = ov::genai::modeling::shape::make({batch, channels, p, p, frames, height, width});
    auto reshaped = input.reshape(shape1);
    auto permuted = reshaped.permute({0, 1, 4, 5, 3, 6, 2});

    auto h_mul = mul_dim(height, p);
    auto w_mul = mul_dim(width, p);
    auto shape2 = ov::genai::modeling::shape::make({batch, channels, frames, h_mul, w_mul});
    return permuted.reshape(shape2);
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

WanRMSNorm::WanRMSNorm(BuilderContext& ctx,
                       const std::string& name,
                       int32_t dim,
                       bool channel_first,
                       bool images,
                       Module* parent)
    : Module(name, ctx, parent),
      dim_(dim),
      channel_first_(channel_first),
      images_(images) {
    gamma_ = &register_parameter("gamma");
    (void)dim_;
    (void)images_;
}

Tensor WanRMSNorm::forward(const Tensor& input) const {
    const int64_t axis = channel_first_ ? 1 : -1;
    auto orig_dtype = input.dtype();
    auto x = input.to(ov::element::f32);
    auto var = x.pow(2.0f).mean(axis, true);
    auto norm = x * (var + eps_).rsqrt();
    auto out = norm.to(orig_dtype) * gamma_->value();
    if (bias_) {
        out = out + bias_->value();
    }
    return out;
}

WanCausalConv3d::WanCausalConv3d(BuilderContext& ctx,
                                 const std::string& name,
                                 int32_t in_channels,
                                 int32_t out_channels,
                                 const std::vector<int64_t>& kernel,
                                 const std::vector<int64_t>& stride,
                                 const std::vector<int64_t>& padding,
                                 Module* parent)
    : Module(name, ctx, parent),
      kernel_(kernel),
      stride_(stride),
      padding_(padding),
      dilation_({1, 1, 1}) {
    (void)in_channels;
    (void)out_channels;
    if (kernel_.size() != 3 || stride_.size() != 3 || padding_.size() != 3) {
        OPENVINO_THROW("WanCausalConv3d expects 3D kernel/stride/padding");
    }
    weight_ = &register_parameter("weight");
    bias_ = &register_parameter("bias");
}

Tensor WanCausalConv3d::forward(const Tensor& input) const {
    auto x = input.to(weight_->value().dtype());
    return ops::nn::causal_conv3d(x, weight_->value(), bias_->value(), stride_, padding_, dilation_);
}

WanResidualBlock::WanResidualBlock(BuilderContext& ctx,
                                   const std::string& name,
                                   int32_t in_dim,
                                   int32_t out_dim,
                                   float dropout,
                                   Module* parent)
    : Module(name, ctx, parent),
      in_dim_(in_dim),
      out_dim_(out_dim),
      dropout_(dropout),
      norm1_(ctx, "norm1", in_dim, true, false, this),
      conv1_(ctx, "conv1", in_dim, out_dim, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, this),
      norm2_(ctx, "norm2", out_dim, true, false, this),
      conv2_(ctx, "conv2", out_dim, out_dim, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, this) {
    (void)dropout_;
    register_module("norm1", &norm1_);
    register_module("conv1", &conv1_);
    register_module("norm2", &norm2_);
    register_module("conv2", &conv2_);

    if (in_dim_ != out_dim_) {
        conv_shortcut_ = std::make_unique<WanCausalConv3d>(ctx,
                                                           "conv_shortcut",
                                                           in_dim,
                                                           out_dim,
                                                           std::vector<int64_t>{1, 1, 1},
                                                           std::vector<int64_t>{1, 1, 1},
                                                           std::vector<int64_t>{0, 0, 0},
                                                           this);
        register_module("conv_shortcut", conv_shortcut_.get());
    }
}

Tensor WanResidualBlock::forward(const Tensor& input) const {
    auto residual = input;
    if (conv_shortcut_) {
        residual = conv_shortcut_->forward(residual);
    }

    auto x = norm1_.forward(input);
    x = ops::silu(x);
    x = conv1_.forward(x);
    x = norm2_.forward(x);
    x = ops::silu(x);
    x = conv2_.forward(x);

    if (residual.dtype() != x.dtype()) {
        residual = residual.to(x.dtype());
    }
    return residual + x;
}

WanAttentionBlock::WanAttentionBlock(BuilderContext& ctx,
                                     const std::string& name,
                                     int32_t dim,
                                     Module* parent)
    : Module(name, ctx, parent),
      dim_(dim),
      scaling_(1.0f / std::sqrt(static_cast<float>(dim))),
      norm_(ctx, "norm", dim, true, true, this) {
    register_module("norm", &norm_);
    to_qkv_weight_ = &register_parameter("to_qkv.weight");
    to_qkv_bias_ = &register_parameter("to_qkv.bias");
    proj_weight_ = &register_parameter("proj.weight");
    proj_bias_ = &register_parameter("proj.bias");
}

Tensor WanAttentionBlock::forward(const Tensor& input) const {
    auto* ctx = input.context();
    auto batch = shape::dim(input, 0);
    auto channels = shape::dim(input, 1);
    auto time = shape::dim(input, 2);
    auto height = shape::dim(input, 3);
    auto width = shape::dim(input, 4);
    auto bt = mul_dim(batch, time);
    auto hw = mul_dim(height, width);

    auto x = input.permute({0, 2, 1, 3, 4});
    auto x_flat = x.reshape(shape::make({bt, channels, height, width}));
    x_flat = norm_.forward(x_flat);

    x_flat = x_flat.to(to_qkv_weight_->value().dtype());
    auto qkv = ops::nn::conv2d(x_flat,
                               to_qkv_weight_->value(),
                               to_qkv_bias_->value(),
                               {1, 1},
                               {0, 0},
                               {0, 0});

    auto three = ops::const_vec(ctx, std::vector<int64_t>{3});
    auto qkv_shape = shape::make({bt, three, channels, hw});
    auto qkv_reshaped = qkv.reshape(qkv_shape).permute({0, 1, 3, 2});

    auto q = ops::slice(qkv_reshaped, 0, 1, 1, 1);
    auto k = ops::slice(qkv_reshaped, 1, 2, 1, 1);
    auto v = ops::slice(qkv_reshaped, 2, 3, 1, 1);

    auto context = ops::llm::sdpa(q, k, v, scaling_, 3, nullptr, false, nullptr);

    auto merged = context.squeeze(1).permute({0, 2, 1});
    auto out_2d = merged.reshape(shape::make({bt, channels, height, width}));
    out_2d = out_2d.to(proj_weight_->value().dtype());
    auto proj = ops::nn::conv2d(out_2d,
                                proj_weight_->value(),
                                proj_bias_->value(),
                                {1, 1},
                                {0, 0},
                                {0, 0});

    auto out_shape = shape::make({batch, time, shape::dim(proj, 1), shape::dim(proj, 2), shape::dim(proj, 3)});
    auto out_5d = proj.reshape(out_shape).permute({0, 2, 1, 3, 4});

    auto residual = input;
    if (residual.dtype() != out_5d.dtype()) {
        residual = residual.to(out_5d.dtype());
    }
    return residual + out_5d;
}

WanResample::WanResample(BuilderContext& ctx,
                         const std::string& name,
                         int32_t dim,
                         const std::string& mode,
                         int32_t upsample_out_dim,
                         Module* parent)
    : Module(name, ctx, parent),
      dim_(dim),
      upsample_out_dim_(upsample_out_dim > 0 ? upsample_out_dim : dim / 2),
      mode_(mode) {
    if (mode_ != "none") {
        resample_weight_ = &register_parameter("resample.1.weight");
        resample_bias_ = &register_parameter("resample.1.bias");
    }

    if (mode_ == "upsample3d") {
        time_conv_ = std::make_unique<WanCausalConv3d>(ctx,
                                                       "time_conv",
                                                       dim,
                                                       dim * 2,
                                                       std::vector<int64_t>{3, 1, 1},
                                                       std::vector<int64_t>{1, 1, 1},
                                                       std::vector<int64_t>{1, 0, 0},
                                                       this);
        register_module("time_conv", time_conv_.get());
    } else if (mode_ == "downsample3d") {
        time_conv_ = std::make_unique<WanCausalConv3d>(ctx,
                                                       "time_conv",
                                                       dim,
                                                       dim,
                                                       std::vector<int64_t>{3, 1, 1},
                                                       std::vector<int64_t>{2, 1, 1},
                                                       std::vector<int64_t>{0, 0, 0},
                                                       this);
        register_module("time_conv", time_conv_.get());
    }
}

Tensor WanResample::forward(const Tensor& input) const {
    if (mode_ == "none") {
        return input;
    }

    auto* ctx = input.context();
    auto batch = shape::dim(input, 0);
    auto channels = shape::dim(input, 1);
    auto time = shape::dim(input, 2);
    auto height = shape::dim(input, 3);
    auto width = shape::dim(input, 4);
    auto two = ops::const_vec(ctx, std::vector<int64_t>{2});

    auto x = input;
    if (mode_ == "upsample3d") {
        if (!time_conv_) {
            OPENVINO_THROW("WanResample upsample3d requires time_conv");
        }
        x = time_conv_->forward(x);
        auto shape1 = shape::make({batch, two, channels, time, height, width});
        x = x.reshape(shape1).permute({0, 2, 3, 1, 4, 5});
        auto time2 = mul_dim(time, two);
        auto shape2 = shape::make({batch, channels, time2, height, width});
        x = x.reshape(shape2);
    }

    auto t_out = shape::dim(x, 2);
    auto h_out = shape::dim(x, 3);
    auto w_out = shape::dim(x, 4);
    auto bt = mul_dim(batch, t_out);
    auto flat = x.permute({0, 2, 1, 3, 4}).reshape(shape::make({bt, channels, h_out, w_out}));

    if (mode_ == "upsample2d" || mode_ == "upsample3d") {
        auto up = ops::nn::upsample_nearest(flat, 2, 2);
        up = up.to(resample_weight_->value().dtype());
        flat = ops::nn::conv2d(up,
                               resample_weight_->value(),
                               resample_bias_->value(),
                               {1, 1},
                               {1, 1},
                               {1, 1});
    } else if (mode_ == "downsample2d" || mode_ == "downsample3d") {
        auto padded = ops::tensor::pad(flat, {0, 0, 0, 0}, {0, 0, 1, 1}, 0.0f);
        padded = padded.to(resample_weight_->value().dtype());
        flat = ops::nn::conv2d(padded,
                               resample_weight_->value(),
                               resample_bias_->value(),
                               {2, 2},
                               {0, 0},
                               {0, 0});
    }

    auto out_shape = shape::make({batch,
                                  t_out,
                                  shape::dim(flat, 1),
                                  shape::dim(flat, 2),
                                  shape::dim(flat, 3)});
    auto out = flat.reshape(out_shape).permute({0, 2, 1, 3, 4});

    if (mode_ == "downsample3d") {
        if (!time_conv_) {
            OPENVINO_THROW("WanResample downsample3d requires time_conv");
        }
        out = time_conv_->forward(out);
    }

    return out;
}

WanMidBlock::WanMidBlock(BuilderContext& ctx,
                         const std::string& name,
                         int32_t dim,
                         float dropout,
                         int32_t num_layers,
                         Module* parent)
    : Module(name, ctx, parent) {
    resnets_.reserve(static_cast<size_t>(num_layers + 1));
    attentions_.reserve(static_cast<size_t>(num_layers));

    resnets_.emplace_back(ctx, "resnets.0", dim, dim, dropout, this);
    register_module("resnets.0", &resnets_.back());

    for (int32_t i = 0; i < num_layers; ++i) {
        const std::string attn_name = "attentions." + std::to_string(i);
        attentions_.emplace_back(ctx, attn_name, dim, this);
        register_module(attn_name, &attentions_.back());

        const std::string res_name = "resnets." + std::to_string(i + 1);
        resnets_.emplace_back(ctx, res_name, dim, dim, dropout, this);
        register_module(res_name, &resnets_.back());
    }
}

Tensor WanMidBlock::forward(const Tensor& input) const {
    auto x = resnets_.front().forward(input);
    for (size_t i = 0; i < attentions_.size(); ++i) {
        x = attentions_[i].forward(x);
        x = resnets_[i + 1].forward(x);
    }
    return x;
}

WanEncoder3d::WanEncoder3d(BuilderContext& ctx,
                           const std::string& name,
                           const WanVAEConfig& cfg,
                           int32_t z_dim,
                           Module* parent)
    : Module(name, ctx, parent),
      cfg_(cfg),
      z_dim_(z_dim),
      conv_in_(ctx,
               "conv_in",
               cfg.in_channels,
               cfg.base_dim,
               std::vector<int64_t>{3, 3, 3},
               std::vector<int64_t>{1, 1, 1},
               std::vector<int64_t>{1, 1, 1},
               this),
      mid_block_(ctx, "mid_block", cfg.base_dim * cfg.dim_mult.back(), cfg.dropout, 1, this),
      norm_out_(ctx, "norm_out", cfg.base_dim * cfg.dim_mult.back(), true, false, this),
      conv_out_(ctx,
                "conv_out",
                cfg.base_dim * cfg.dim_mult.back(),
                z_dim_,
                std::vector<int64_t>{3, 3, 3},
                std::vector<int64_t>{1, 1, 1},
                std::vector<int64_t>{1, 1, 1},
                this) {
    register_module("conv_in", &conv_in_);
    register_module("mid_block", &mid_block_);
    register_module("norm_out", &norm_out_);
    register_module("conv_out", &conv_out_);

    if (cfg.is_residual) {
        OPENVINO_THROW("WanEncoder3d residual down blocks are not implemented");
    }

    std::vector<int32_t> dims;
    dims.reserve(cfg.dim_mult.size() + 1);
    dims.push_back(cfg.base_dim);
    for (int32_t mult : cfg.dim_mult) {
        dims.push_back(cfg.base_dim * mult);
    }

    float scale = 1.0f;
    int32_t layer_index = 0;
    for (size_t i = 0; i + 1 < dims.size(); ++i) {
        int32_t in_dim = dims[i];
        int32_t out_dim = dims[i + 1];

        for (int32_t j = 0; j < cfg.num_res_blocks; ++j) {
            const std::string name = "down_blocks." + std::to_string(layer_index++);
            auto resnet = std::make_unique<WanResidualBlock>(ctx, name, in_dim, out_dim, cfg.dropout, this);
            register_module(name, resnet.get());
            down_layers_.push_back({Layer::Kind::Resnet, resnet.get(), nullptr, nullptr});
            resnets_.push_back(std::move(resnet));

            if (std::find(cfg.attn_scales.begin(), cfg.attn_scales.end(), scale) != cfg.attn_scales.end()) {
                const std::string attn_name = "down_blocks." + std::to_string(layer_index++);
                auto attention = std::make_unique<WanAttentionBlock>(ctx, attn_name, out_dim, this);
                register_module(attn_name, attention.get());
                down_layers_.push_back({Layer::Kind::Attention, nullptr, attention.get(), nullptr});
                attentions_.push_back(std::move(attention));
            }
            in_dim = out_dim;
        }

        if (i + 1 < dims.size() - 1) {
            const bool temporal = (!cfg.temperal_downsample.empty() && i < cfg.temperal_downsample.size())
                                      ? cfg.temperal_downsample[i]
                                      : false;
            const std::string mode = temporal ? "downsample3d" : "downsample2d";
            const std::string name = "down_blocks." + std::to_string(layer_index++);
            auto resample = std::make_unique<WanResample>(ctx, name, out_dim, mode, -1, this);
            register_module(name, resample.get());
            down_layers_.push_back({Layer::Kind::Resample, nullptr, nullptr, resample.get()});
            resamples_.push_back(std::move(resample));
            scale /= 2.0f;
        }
    }

}

Tensor WanEncoder3d::forward(const Tensor& input) const {
    auto x = conv_in_.forward(input);
    for (const auto& layer : down_layers_) {
        if (layer.kind == Layer::Kind::Resnet) {
            x = layer.resnet->forward(x);
        } else if (layer.kind == Layer::Kind::Attention) {
            x = layer.attention->forward(x);
        } else if (layer.kind == Layer::Kind::Resample) {
            x = layer.resample->forward(x);
        }
    }
    x = mid_block_.forward(x);
    x = norm_out_.forward(x);
    x = ops::silu(x);
    x = conv_out_.forward(x);
    return x;
}

WanUpBlock::WanUpBlock(BuilderContext& ctx,
                       const std::string& name,
                       int32_t in_dim,
                       int32_t out_dim,
                       int32_t num_res_blocks,
                       float dropout,
                       const std::string& upsample_mode,
                       Module* parent)
    : Module(name, ctx, parent) {
    resnets_.reserve(static_cast<size_t>(num_res_blocks + 1));
    int32_t current_dim = in_dim;
    for (int32_t i = 0; i < num_res_blocks + 1; ++i) {
        const std::string resnet_name = "resnets." + std::to_string(i);
        resnets_.emplace_back(ctx, resnet_name, current_dim, out_dim, dropout, this);
        register_module(resnet_name, &resnets_.back());
        current_dim = out_dim;
    }

    if (!upsample_mode.empty()) {
        upsampler_ = std::make_unique<WanResample>(ctx, "upsamplers.0", out_dim, upsample_mode, -1, this);
        register_module("upsamplers.0", upsampler_.get());
    }
}

Tensor WanUpBlock::forward(const Tensor& input) const {
    auto x = input;
    for (const auto& resnet : resnets_) {
        x = resnet.forward(x);
    }
    if (upsampler_) {
        x = upsampler_->forward(x);
    }
    return x;
}

WanDecoder3d::WanDecoder3d(BuilderContext& ctx,
                           const std::string& name,
                           const WanVAEConfig& cfg,
                           Module* parent)
    : Module(name, ctx, parent),
      cfg_(cfg),
      conv_in_(ctx,
               "conv_in",
               cfg.z_dim,
               cfg.decoder_base_dim * cfg.dim_mult.back(),
               std::vector<int64_t>{3, 3, 3},
               std::vector<int64_t>{1, 1, 1},
               std::vector<int64_t>{1, 1, 1},
               this),
      mid_block_(ctx, "mid_block", cfg.decoder_base_dim * cfg.dim_mult.back(), cfg.dropout, 1, this),
      norm_out_(ctx, "norm_out", cfg.decoder_base_dim * cfg.dim_mult.front(), true, false, this),
      conv_out_(ctx,
                "conv_out",
                cfg.decoder_base_dim * cfg.dim_mult.front(),
                cfg.out_channels,
                std::vector<int64_t>{3, 3, 3},
                std::vector<int64_t>{1, 1, 1},
                std::vector<int64_t>{1, 1, 1},
                this) {
    register_module("conv_in", &conv_in_);
    register_module("mid_block", &mid_block_);
    register_module("norm_out", &norm_out_);
    register_module("conv_out", &conv_out_);

    if (cfg.is_residual) {
        OPENVINO_THROW("WanDecoder3d residual up blocks are not implemented");
    }

    std::vector<int32_t> dims;
    dims.reserve(cfg.dim_mult.size() + 1);
    dims.push_back(cfg.decoder_base_dim * cfg.dim_mult.back());
    for (auto it = cfg.dim_mult.rbegin(); it != cfg.dim_mult.rend(); ++it) {
        dims.push_back(cfg.decoder_base_dim * *it);
    }

    std::vector<bool> temperal_upsample;
    temperal_upsample.reserve(cfg.temperal_downsample.size());
    for (auto it = cfg.temperal_downsample.rbegin(); it != cfg.temperal_downsample.rend(); ++it) {
        temperal_upsample.push_back(*it);
    }

    up_blocks_.reserve(dims.size() - 1);
    for (size_t i = 0; i + 1 < dims.size(); ++i) {
        int32_t in_dim = dims[i];
        const int32_t out_dim = dims[i + 1];
        if (i > 0 && !cfg.is_residual) {
            in_dim /= 2;
        }
        const bool up_flag = i + 1 < dims.size() - 1;
        std::string upsample_mode;
        if (up_flag) {
            const bool temporal = (i < temperal_upsample.size()) ? temperal_upsample[i] : false;
            upsample_mode = temporal ? "upsample3d" : "upsample2d";
        }

        const std::string name = "up_blocks." + std::to_string(i);
        up_blocks_.emplace_back(ctx, name, in_dim, out_dim, cfg.num_res_blocks, cfg.dropout, upsample_mode, this);
        register_module(name, &up_blocks_.back());
    }

}

Tensor WanDecoder3d::forward(const Tensor& input) const {
    auto x = conv_in_.forward(input);
    x = mid_block_.forward(x);
    for (const auto& block : up_blocks_) {
        x = block.forward(x);
    }
    x = norm_out_.forward(x);
    x = ops::silu(x);
    x = conv_out_.forward(x);
    return x;
}

namespace {

class WanVAEEncoderModel : public Module {
public:
    WanVAEEncoderModel(BuilderContext& ctx, const WanVAEConfig& cfg, Module* parent = nullptr)
        : Module("", ctx, parent),
          cfg_(cfg),
          encoder_(ctx, "encoder", cfg, cfg.z_dim * 2, this),
          quant_conv_(ctx,
                      "quant_conv",
                      cfg.z_dim * 2,
                      cfg.z_dim * 2,
                      std::vector<int64_t>{1, 1, 1},
                      std::vector<int64_t>{1, 1, 1},
                      std::vector<int64_t>{0, 0, 0},
                      this) {
        register_module("encoder", &encoder_);
        register_module("quant_conv", &quant_conv_);
    }

    Tensor forward(const Tensor& input) const {
        auto x = input;
        if (cfg_.patch_size.has_value()) {
            x = patchify(x, cfg_.patch_size.value());
        }
        x = encoder_.forward(x);
        x = quant_conv_.forward(x);
        return x;
    }

private:
    WanVAEConfig cfg_;
    WanEncoder3d encoder_;
    WanCausalConv3d quant_conv_;
};

class WanVAEDecoderModel : public Module {
public:
    WanVAEDecoderModel(BuilderContext& ctx, const WanVAEConfig& cfg, Module* parent = nullptr)
        : Module("", ctx, parent),
          cfg_(cfg),
          post_quant_conv_(ctx,
                           "post_quant_conv",
                           cfg.z_dim,
                           cfg.z_dim,
                           std::vector<int64_t>{1, 1, 1},
                           std::vector<int64_t>{1, 1, 1},
                           std::vector<int64_t>{0, 0, 0},
                           this),
          decoder_(ctx, "decoder", cfg, this) {
        register_module("post_quant_conv", &post_quant_conv_);
        register_module("decoder", &decoder_);
    }

    Tensor forward(const Tensor& input) const {
        auto x = post_quant_conv_.forward(input);
        x = decoder_.forward(x);
        if (cfg_.patch_size.has_value()) {
            x = unpatchify(x, cfg_.patch_size.value());
        }
        return clamp(x, -1.0f, 1.0f);
    }

private:
    WanVAEConfig cfg_;
    WanCausalConv3d post_quant_conv_;
    WanDecoder3d decoder_;
};

}  // namespace

std::shared_ptr<ov::Model> create_wan_vae_encoder_model(
    const WanVAEConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    WanVAEEncoderModel model(ctx, cfg);

    WanWeightMapping::apply_vae_packed_mapping(model);
    ov::genai::modeling::weights::LoadOptions load_opts;
    load_opts.allow_unmatched = true;
    load_opts.allow_missing = true;
    load_opts.report_missing = false;
    load_opts.report_unmatched = false;
    ov::genai::modeling::weights::load_model(model, source, finalizer, load_opts);

    auto sample = ctx.parameter("sample",
                                ov::element::f32,
                                ov::PartialShape{-1, cfg.in_channels, -1, -1, -1});
    auto output = model.forward(sample);
    auto result = std::make_shared<ov::op::v0::Result>(output.output());
    set_name(result, "latent");
    return ctx.build_model({result->output(0)});
}

std::shared_ptr<ov::Model> create_wan_vae_decoder_model(
    const WanVAEConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    WanVAEDecoderModel model(ctx, cfg);

    WanWeightMapping::apply_vae_packed_mapping(model);
    ov::genai::modeling::weights::LoadOptions load_opts;
    load_opts.allow_unmatched = true;
    load_opts.allow_missing = true;
    load_opts.report_missing = false;
    load_opts.report_unmatched = false;
    ov::genai::modeling::weights::load_model(model, source, finalizer, load_opts);

    auto latents = ctx.parameter("latents",
                                 ov::element::f32,
                                 ov::PartialShape{-1, cfg.z_dim, -1, -1, -1});
    auto output = model.forward(latents);
    auto result = std::make_shared<ov::op::v0::Result>(output.output());
    set_name(result, "sample");
    return ctx.build_model({result->output(0)});
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
