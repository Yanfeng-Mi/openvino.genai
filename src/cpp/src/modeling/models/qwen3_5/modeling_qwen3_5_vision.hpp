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
#include "modeling/models/qwen3_5/processing_qwen3_5.hpp"
#include "modeling/module.hpp"
#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

struct Qwen3_5VisionOutput {
    Tensor visual_embeds;
    std::vector<Tensor> deepstack_embeds;
};

class Qwen3_5VisionPatchEmbed : public Module {
public:
    Qwen3_5VisionPatchEmbed(BuilderContext& ctx, const std::string& name, const Qwen3_5VisionConfig& cfg,
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

class Qwen3_5VisionAttention : public Module {
public:
    Qwen3_5VisionAttention(BuilderContext& ctx, const std::string& name, const Qwen3_5VisionConfig& cfg,
                           Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states,
                   const Tensor& rotary_cos,
                   const Tensor& rotary_sin) const;

private:
    const Tensor& qkv_weight() const;
    const Tensor* qkv_bias() const;
    const Tensor& proj_weight() const;
    const Tensor* proj_bias() const;

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
};

class Qwen3_5VisionMLP : public Module {
public:
    Qwen3_5VisionMLP(BuilderContext& ctx, const std::string& name, const Qwen3_5VisionConfig& cfg,
                     Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states) const;

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

class Qwen3_5VisionBlock : public Module {
public:
    Qwen3_5VisionBlock(BuilderContext& ctx, const std::string& name, const Qwen3_5VisionConfig& cfg,
                       Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states,
                   const Tensor& rotary_cos,
                   const Tensor& rotary_sin) const;

private:
    const Tensor& norm1_weight() const;
    const Tensor& norm1_bias() const;
    const Tensor& norm2_weight() const;
    const Tensor& norm2_bias() const;

    Qwen3_5VisionAttention attn_;
    Qwen3_5VisionMLP mlp_;
    float eps_ = 1e-6f;

    WeightParameter* norm1_weight_param_ = nullptr;
    WeightParameter* norm1_bias_param_ = nullptr;
    WeightParameter* norm2_weight_param_ = nullptr;
    WeightParameter* norm2_bias_param_ = nullptr;
};

class Qwen3_5VisionPatchMerger : public Module {
public:
    Qwen3_5VisionPatchMerger(BuilderContext& ctx,
                             const std::string& name,
                             const Qwen3_5VisionConfig& cfg,
                             bool use_postshuffle_norm,
                             Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states) const;

private:
    const Tensor& norm_weight() const;
    const Tensor& norm_bias() const;
    const Tensor& fc1_weight() const;
    const Tensor* fc1_bias() const;
    const Tensor& fc2_weight() const;
    const Tensor* fc2_bias() const;

    int32_t hidden_size_ = 0;
    int32_t merged_hidden_size_ = 0;
    bool use_postshuffle_norm_ = false;
    float eps_ = 1e-6f;

    WeightParameter* norm_weight_param_ = nullptr;
    WeightParameter* norm_bias_param_ = nullptr;
    WeightParameter* fc1_weight_param_ = nullptr;
    WeightParameter* fc1_bias_param_ = nullptr;
    WeightParameter* fc2_weight_param_ = nullptr;
    WeightParameter* fc2_bias_param_ = nullptr;
};

class Qwen3_5VisionModel : public Module {
public:
    Qwen3_5VisionModel(BuilderContext& ctx, const Qwen3_5VisionConfig& cfg, Module* parent = nullptr);

    Qwen3_5VisionOutput forward(const Tensor& pixel_values,
                                const Tensor& grid_thw,
                                const Tensor& pos_embeds,
                                const Tensor& rotary_cos,
                                const Tensor& rotary_sin);

    Qwen3_5VisionPatchEmbed& patch_embed();
    Qwen3_5VisionPatchMerger& merger();

private:
    Qwen3_5VisionConfig cfg_;
    Qwen3_5VisionPatchEmbed patch_embed_;
    std::vector<Qwen3_5VisionBlock> blocks_;
    Qwen3_5VisionPatchMerger merger_;
    std::vector<Qwen3_5VisionPatchMerger> deepstack_mergers_;
    std::vector<int32_t> deepstack_indexes_;
};

std::shared_ptr<ov::Model> create_qwen3_5_vision_model(
    const Qwen3_5Config& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov

