// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ov {
class Model;
}  // namespace ov

namespace ov {
namespace genai {
namespace modeling {
namespace weights {
class WeightFinalizer;
class WeightSource;
}  // namespace weights
}  // namespace modeling
}  // namespace genai
}  // namespace ov

#include "modeling/models/wan/processing_wan.hpp"
#include "modeling/module.hpp"
#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

class WanRMSNorm : public Module {
public:
    WanRMSNorm(BuilderContext& ctx,
               const std::string& name,
               int32_t dim,
               bool channel_first,
               bool images,
               Module* parent = nullptr);

    Tensor forward(const Tensor& input) const;

private:
    int32_t dim_ = 0;
    bool channel_first_ = true;
    bool images_ = true;
    float eps_ = 1e-12f;
    WeightParameter* gamma_ = nullptr;
    WeightParameter* bias_ = nullptr;
};

class WanCausalConv3d : public Module {
public:
    WanCausalConv3d(BuilderContext& ctx,
                    const std::string& name,
                    int32_t in_channels,
                    int32_t out_channels,
                    const std::vector<int64_t>& kernel,
                    const std::vector<int64_t>& stride,
                    const std::vector<int64_t>& padding,
                    Module* parent = nullptr);

    Tensor forward(const Tensor& input) const;

private:
    std::vector<int64_t> kernel_;
    std::vector<int64_t> stride_;
    std::vector<int64_t> padding_;
    std::vector<int64_t> dilation_;
    WeightParameter* weight_ = nullptr;
    WeightParameter* bias_ = nullptr;
};

class WanResidualBlock : public Module {
public:
    WanResidualBlock(BuilderContext& ctx,
                     const std::string& name,
                     int32_t in_dim,
                     int32_t out_dim,
                     float dropout,
                     Module* parent = nullptr);

    Tensor forward(const Tensor& input) const;

private:
    int32_t in_dim_ = 0;
    int32_t out_dim_ = 0;
    float dropout_ = 0.0f;

    WanRMSNorm norm1_;
    WanCausalConv3d conv1_;
    WanRMSNorm norm2_;
    WanCausalConv3d conv2_;
    std::unique_ptr<WanCausalConv3d> conv_shortcut_;
};

class WanAttentionBlock : public Module {
public:
    WanAttentionBlock(BuilderContext& ctx,
                      const std::string& name,
                      int32_t dim,
                      Module* parent = nullptr);

    Tensor forward(const Tensor& input) const;

private:
    int32_t dim_ = 0;
    float scaling_ = 1.0f;
    WanRMSNorm norm_;
    WeightParameter* to_qkv_weight_ = nullptr;
    WeightParameter* to_qkv_bias_ = nullptr;
    WeightParameter* proj_weight_ = nullptr;
    WeightParameter* proj_bias_ = nullptr;
};

class WanResample : public Module {
public:
    WanResample(BuilderContext& ctx,
                const std::string& name,
                int32_t dim,
                const std::string& mode,
                int32_t upsample_out_dim = -1,
                Module* parent = nullptr);

    Tensor forward(const Tensor& input) const;

private:
    int32_t dim_ = 0;
    int32_t upsample_out_dim_ = 0;
    std::string mode_;

    WeightParameter* resample_weight_ = nullptr;
    WeightParameter* resample_bias_ = nullptr;
    std::unique_ptr<WanCausalConv3d> time_conv_;
};

class WanMidBlock : public Module {
public:
    WanMidBlock(BuilderContext& ctx,
                const std::string& name,
                int32_t dim,
                float dropout,
                int32_t num_layers,
                Module* parent = nullptr);

    Tensor forward(const Tensor& input) const;

private:
    std::vector<WanResidualBlock> resnets_;
    std::vector<WanAttentionBlock> attentions_;
};

class WanEncoder3d : public Module {
public:
    WanEncoder3d(BuilderContext& ctx,
                 const std::string& name,
                 const WanVAEConfig& cfg,
                 int32_t z_dim,
                 Module* parent = nullptr);

    Tensor forward(const Tensor& input) const;

private:
    struct Layer {
        enum class Kind { Resnet, Attention, Resample };
        Kind kind = Kind::Resnet;
        WanResidualBlock* resnet = nullptr;
        WanAttentionBlock* attention = nullptr;
        WanResample* resample = nullptr;
    };

    WanVAEConfig cfg_;
    int32_t z_dim_ = 0;

    WanCausalConv3d conv_in_;
    std::vector<std::unique_ptr<WanResidualBlock>> resnets_;
    std::vector<std::unique_ptr<WanAttentionBlock>> attentions_;
    std::vector<std::unique_ptr<WanResample>> resamples_;
    std::vector<Layer> down_layers_;
    WanMidBlock mid_block_;
    WanRMSNorm norm_out_;
    WanCausalConv3d conv_out_;
};

class WanUpBlock : public Module {
public:
    WanUpBlock(BuilderContext& ctx,
               const std::string& name,
               int32_t in_dim,
               int32_t out_dim,
               int32_t num_res_blocks,
               float dropout,
               const std::string& upsample_mode,
               Module* parent = nullptr);

    Tensor forward(const Tensor& input) const;

private:
    std::vector<WanResidualBlock> resnets_;
    std::unique_ptr<WanResample> upsampler_;
};

class WanDecoder3d : public Module {
public:
    WanDecoder3d(BuilderContext& ctx,
                 const std::string& name,
                 const WanVAEConfig& cfg,
                 Module* parent = nullptr);

    Tensor forward(const Tensor& input) const;

private:
    WanVAEConfig cfg_;
    WanCausalConv3d conv_in_;
    WanMidBlock mid_block_;
    std::vector<WanUpBlock> up_blocks_;
    WanRMSNorm norm_out_;
    WanCausalConv3d conv_out_;
};

std::shared_ptr<ov::Model> create_wan_vae_encoder_model(
    const WanVAEConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

std::shared_ptr<ov::Model> create_wan_vae_decoder_model(
    const WanVAEConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
