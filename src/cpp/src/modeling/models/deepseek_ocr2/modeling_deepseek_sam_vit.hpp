// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
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

#include "modeling/builder_context.hpp"
#include "modeling/models/deepseek_ocr2/processing_deepseek_ocr2.hpp"
#include "modeling/module.hpp"
#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

struct DeepseekSamIO {
    static constexpr const char* kPixelValues = "pixel_values";
    static constexpr const char* kVisionFeats = "vision_feats";
};

struct DeepseekSamConfig {
    int32_t image_size = 1024;
    int32_t patch_size = 16;
    int32_t in_channels = 3;
    int32_t embed_dim = 768;
    int32_t depth = 12;
    int32_t num_heads = 12;
    float mlp_ratio = 4.0f;
    int32_t out_chans = 256;
    int32_t window_size = 14;
    std::vector<int32_t> global_attn_indexes;
    float layer_norm_eps = 1e-6f;
    bool use_rel_pos = true;
    int32_t net2_channels = 512;
    int32_t net3_channels = 896;
};

class DeepseekSamPatchEmbed : public Module {
public:
    DeepseekSamPatchEmbed(BuilderContext& ctx, const std::string& name, const DeepseekSamConfig& cfg,
                          Module* parent = nullptr);

    Tensor forward(const Tensor& pixel_values) const;

private:
    const Tensor& weight() const;
    const Tensor* bias() const;

    int32_t in_channels_ = 0;
    int32_t patch_size_ = 0;
    int32_t embed_dim_ = 0;

    WeightParameter* weight_param_ = nullptr;
    WeightParameter* bias_param_ = nullptr;
};

class DeepseekSamLayerNorm2d : public Module {
public:
    DeepseekSamLayerNorm2d(BuilderContext& ctx, const std::string& name, float eps, Module* parent = nullptr);

    Tensor forward(const Tensor& input) const;

private:
    const Tensor& weight() const;
    const Tensor& bias() const;

    float eps_ = 1e-6f;
    WeightParameter* weight_param_ = nullptr;
    WeightParameter* bias_param_ = nullptr;
};

class DeepseekSamMLP : public Module {
public:
    DeepseekSamMLP(BuilderContext& ctx, const std::string& name, Module* parent = nullptr);

    Tensor forward(const Tensor& input) const;

private:
    const Tensor& fc1_weight() const;
    const Tensor* fc1_bias() const;
    const Tensor& fc2_weight() const;
    const Tensor* fc2_bias() const;

    WeightParameter* fc1_weight_param_ = nullptr;
    WeightParameter* fc1_bias_param_ = nullptr;
    WeightParameter* fc2_weight_param_ = nullptr;
    WeightParameter* fc2_bias_param_ = nullptr;
};

class DeepseekSamAttention : public Module {
public:
    DeepseekSamAttention(BuilderContext& ctx, const std::string& name, const DeepseekSamConfig& cfg,
                         Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states) const;

private:
    const Tensor& qkv_weight() const;
    const Tensor* qkv_bias() const;
    const Tensor& proj_weight() const;
    const Tensor* proj_bias() const;
    const Tensor& rel_pos_h() const;
    const Tensor& rel_pos_w() const;

    int32_t hidden_size_ = 0;
    int32_t num_heads_ = 0;
    int32_t head_dim_ = 0;
    bool use_rel_pos_ = true;

    WeightParameter* qkv_weight_param_ = nullptr;
    WeightParameter* qkv_bias_param_ = nullptr;
    WeightParameter* proj_weight_param_ = nullptr;
    WeightParameter* proj_bias_param_ = nullptr;
    WeightParameter* rel_pos_h_param_ = nullptr;
    WeightParameter* rel_pos_w_param_ = nullptr;
};

class DeepseekSamBlock : public Module {
public:
    DeepseekSamBlock(BuilderContext& ctx, const std::string& name, const DeepseekSamConfig& cfg,
                     int32_t window_size, Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states) const;

private:
    const Tensor& norm1_weight() const;
    const Tensor& norm1_bias() const;
    const Tensor& norm2_weight() const;
    const Tensor& norm2_bias() const;

    DeepseekSamAttention attn_;
    DeepseekSamMLP mlp_;
    int32_t window_size_ = 0;
    float eps_ = 1e-6f;

    WeightParameter* norm1_weight_param_ = nullptr;
    WeightParameter* norm1_bias_param_ = nullptr;
    WeightParameter* norm2_weight_param_ = nullptr;
    WeightParameter* norm2_bias_param_ = nullptr;
};

class DeepseekSamVisionModel : public Module {
public:
    DeepseekSamVisionModel(BuilderContext& ctx, const DeepseekSamConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& pixel_values) const;

private:
    const Tensor& pos_embed() const;
    const Tensor& neck0_weight() const;
    const Tensor& neck2_weight() const;
    const Tensor& net2_weight() const;
    const Tensor& net3_weight() const;

    DeepseekSamConfig cfg_;
    DeepseekSamPatchEmbed patch_embed_;
    std::vector<DeepseekSamBlock> blocks_;
    DeepseekSamLayerNorm2d neck_norm1_;
    DeepseekSamLayerNorm2d neck_norm2_;

    WeightParameter* pos_embed_param_ = nullptr;
    WeightParameter* neck0_weight_param_ = nullptr;
    WeightParameter* neck1_weight_param_ = nullptr;
    WeightParameter* neck1_bias_param_ = nullptr;
    WeightParameter* neck2_weight_param_ = nullptr;
    WeightParameter* neck3_weight_param_ = nullptr;
    WeightParameter* neck3_bias_param_ = nullptr;
    WeightParameter* net2_weight_param_ = nullptr;
    WeightParameter* net3_weight_param_ = nullptr;
};

std::shared_ptr<ov::Model> create_deepseek_sam_model(
    const DeepseekOCR2VisionConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
