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
#include "modeling/models/glm_ocr/processing_glm_ocr.hpp"
#include "modeling/module.hpp"
#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

class GlmOcrVisionPatchEmbed : public Module {
public:
    GlmOcrVisionPatchEmbed(BuilderContext& ctx, const std::string& name, const GlmOcrVisionConfig& cfg,
                           Module* parent = nullptr);

    Tensor forward(const Tensor& pixel_values) const;

private:
    const Tensor& weight() const;
    const Tensor* bias() const;

    int32_t in_channels_ = 0;
    int32_t patch_size_ = 0;
    int32_t temporal_patch_size_ = 0;
    int32_t embed_dim_ = 0;

    WeightParameter* weight_param_ = nullptr;
    WeightParameter* bias_param_ = nullptr;
};

class GlmOcrVisionAttention : public Module {
public:
    GlmOcrVisionAttention(BuilderContext& ctx, const std::string& name, const GlmOcrVisionConfig& cfg,
                          Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states,
                   const Tensor& rotary_cos,
                   const Tensor& rotary_sin) const;

private:
    const Tensor& qkv_weight() const;
    const Tensor* qkv_bias() const;
    const Tensor& proj_weight() const;
    const Tensor* proj_bias() const;
    const Tensor& q_norm_weight() const;
    const Tensor& k_norm_weight() const;

    Tensor apply_rotary(const Tensor& x,
                        const Tensor& cos,
                        const Tensor& sin) const;

    int32_t hidden_size_ = 0;
    int32_t num_heads_ = 0;
    int32_t head_dim_ = 0;
    float scaling_ = 1.0f;

    WeightParameter* qkv_weight_param_ = nullptr;
    WeightParameter* qkv_bias_param_ = nullptr;
    WeightParameter* proj_weight_param_ = nullptr;
    WeightParameter* proj_bias_param_ = nullptr;
    WeightParameter* q_norm_weight_param_ = nullptr;
    WeightParameter* k_norm_weight_param_ = nullptr;
};

class GlmOcrVisionMLP : public Module {
public:
    GlmOcrVisionMLP(BuilderContext& ctx, const std::string& name, const GlmOcrVisionConfig& cfg,
                    Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states) const;

private:
    const Tensor& gate_proj_weight() const;
    const Tensor* gate_proj_bias() const;
    const Tensor& up_proj_weight() const;
    const Tensor* up_proj_bias() const;
    const Tensor& down_proj_weight() const;
    const Tensor* down_proj_bias() const;

    WeightParameter* gate_proj_weight_param_ = nullptr;
    WeightParameter* gate_proj_bias_param_ = nullptr;
    WeightParameter* up_proj_weight_param_ = nullptr;
    WeightParameter* up_proj_bias_param_ = nullptr;
    WeightParameter* down_proj_weight_param_ = nullptr;
    WeightParameter* down_proj_bias_param_ = nullptr;
};

class GlmOcrVisionBlock : public Module {
public:
    GlmOcrVisionBlock(BuilderContext& ctx, const std::string& name, const GlmOcrVisionConfig& cfg,
                      Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states,
                   const Tensor& rotary_cos,
                   const Tensor& rotary_sin) const;

private:
    const Tensor& norm1_weight() const;
    const Tensor& norm2_weight() const;

    GlmOcrVisionAttention attn_;
    GlmOcrVisionMLP mlp_;
    float eps_ = 1e-5f;

    WeightParameter* norm1_weight_param_ = nullptr;
    WeightParameter* norm2_weight_param_ = nullptr;
};

// proj(1536,1536,no bias)+LayerNorm(1536)+GELU, then gate/up/down(1536↔4608,no bias,SiLU)
class GlmOcrVisionPatchMerger : public Module {
public:
    GlmOcrVisionPatchMerger(BuilderContext& ctx,
                            const std::string& name,
                            const GlmOcrVisionConfig& cfg,
                            Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states) const;

private:
    const Tensor& proj_weight() const;
    const Tensor& norm_weight() const;
    const Tensor& norm_bias() const;
    const Tensor& gate_proj_weight() const;
    const Tensor& up_proj_weight() const;
    const Tensor& down_proj_weight() const;

    int32_t hidden_size_ = 0;
    int32_t out_hidden_size_ = 0;
    int32_t spatial_merge_size_ = 0;
    float eps_ = 1e-5f;

    WeightParameter* proj_weight_param_ = nullptr;
    WeightParameter* norm_weight_param_ = nullptr;
    WeightParameter* norm_bias_param_ = nullptr;
    WeightParameter* gate_proj_weight_param_ = nullptr;
    WeightParameter* up_proj_weight_param_ = nullptr;
    WeightParameter* down_proj_weight_param_ = nullptr;
};

class GlmOcrVisionModel : public Module {
public:
    GlmOcrVisionModel(BuilderContext& ctx, const GlmOcrVisionConfig& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& pixel_values,
                   const Tensor& grid_thw,
                   const Tensor& rotary_cos,
                   const Tensor& rotary_sin);

    GlmOcrVisionPatchEmbed& patch_embed();
    GlmOcrVisionPatchMerger& merger();

private:
    const Tensor& post_layernorm_weight() const;
    const Tensor& downsample_weight() const;
    const Tensor* downsample_bias() const;

    GlmOcrVisionConfig cfg_;
    GlmOcrVisionPatchEmbed patch_embed_;
    std::vector<GlmOcrVisionBlock> blocks_;
    GlmOcrVisionPatchMerger merger_;
    float eps_ = 1e-5f;

    WeightParameter* post_layernorm_weight_param_ = nullptr;
    WeightParameter* downsample_weight_param_ = nullptr;
    WeightParameter* downsample_bias_param_ = nullptr;
};

std::shared_ptr<ov::Model> create_glm_ocr_vision_model(
    const GlmOcrConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
