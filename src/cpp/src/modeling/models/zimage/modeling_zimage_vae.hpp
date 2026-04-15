// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "modeling/module.hpp"
#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

struct ZImageVAEConfig {
    int32_t in_channels = 3;
    int32_t out_channels = 3;
    int32_t latent_channels = 16;
    std::vector<int32_t> block_out_channels = {128, 256, 512, 512};
    int32_t layers_per_block = 2;
    int32_t norm_num_groups = 32;
    bool mid_block_add_attention = true;
    float scaling_factor = 0.3611f;
    float shift_factor = 0.1159f;
};

class VAEGroupNorm : public Module {
public:
    VAEGroupNorm(BuilderContext& ctx,
                 const std::string& name,
                 int32_t num_groups,
                 float eps,
                 Module* parent = nullptr);

    Tensor forward(const Tensor& input) const;

private:
    int32_t num_groups_ = 1;
    float eps_ = 1e-6f;
    WeightParameter* weight_ = nullptr;
    WeightParameter* bias_ = nullptr;
};

class VAEResnetBlock2D : public Module {
public:
    VAEResnetBlock2D(BuilderContext& ctx,
                     const std::string& name,
                     int32_t in_channels,
                     int32_t out_channels,
                     int32_t norm_groups,
                     float eps,
                     Module* parent = nullptr);

    Tensor forward(const Tensor& input) const;

private:
    int32_t in_channels_ = 0;
    int32_t out_channels_ = 0;

    VAEGroupNorm norm1_;
    VAEGroupNorm norm2_;
    WeightParameter* conv1_weight_ = nullptr;
    WeightParameter* conv1_bias_ = nullptr;
    WeightParameter* conv2_weight_ = nullptr;
    WeightParameter* conv2_bias_ = nullptr;
    WeightParameter* conv_shortcut_weight_ = nullptr;
    WeightParameter* conv_shortcut_bias_ = nullptr;
};

class VAEAttention : public Module {
public:
    VAEAttention(BuilderContext& ctx,
                 const std::string& name,
                 int32_t channels,
                 int32_t norm_groups,
                 float eps,
                 Module* parent = nullptr);

    Tensor forward(const Tensor& input) const;

private:
    int32_t channels_ = 0;
    VAEGroupNorm group_norm_;
    WeightParameter* q_weight_ = nullptr;
    WeightParameter* q_bias_ = nullptr;
    WeightParameter* k_weight_ = nullptr;
    WeightParameter* k_bias_ = nullptr;
    WeightParameter* v_weight_ = nullptr;
    WeightParameter* v_bias_ = nullptr;
    WeightParameter* o_weight_ = nullptr;
    WeightParameter* o_bias_ = nullptr;
};

class VAEUpsample2D : public Module {
public:
    VAEUpsample2D(BuilderContext& ctx, const std::string& name, int32_t channels, Module* parent = nullptr);

    Tensor forward(const Tensor& input) const;

private:
    int32_t channels_ = 0;
    WeightParameter* conv_weight_ = nullptr;
    WeightParameter* conv_bias_ = nullptr;
};

class VAEUpDecoderBlock2D : public Module {
public:
    VAEUpDecoderBlock2D(BuilderContext& ctx,
                        const std::string& name,
                        int32_t in_channels,
                        int32_t out_channels,
                        int32_t num_layers,
                        int32_t norm_groups,
                        float eps,
                        bool add_upsample,
                        Module* parent = nullptr);

    Tensor forward(const Tensor& input) const;

private:
    std::vector<VAEResnetBlock2D> resnets_;
    std::unique_ptr<VAEUpsample2D> upsampler_;
};

class VAEMidBlock2D : public Module {
public:
    VAEMidBlock2D(BuilderContext& ctx,
                  const std::string& name,
                  int32_t channels,
                  int32_t norm_groups,
                  float eps,
                  bool add_attention,
                  Module* parent = nullptr);

    Tensor forward(const Tensor& input) const;

private:
    VAEResnetBlock2D resnet1_;
    VAEResnetBlock2D resnet2_;
    std::unique_ptr<VAEAttention> attention_;
};

class ZImageVAEDecoder : public Module {
public:
    ZImageVAEDecoder(BuilderContext& ctx, const ZImageVAEConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& latents) const;

private:
    ZImageVAEConfig cfg_;
    WeightParameter* conv_in_weight_ = nullptr;
    WeightParameter* conv_in_bias_ = nullptr;
    VAEMidBlock2D mid_block_;
    std::vector<VAEUpDecoderBlock2D> up_blocks_;
    VAEGroupNorm norm_out_;
    WeightParameter* conv_out_weight_ = nullptr;
    WeightParameter* conv_out_bias_ = nullptr;
};

std::shared_ptr<ov::Model> create_zimage_vae_decoder_model(
    const ZImageVAEConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
