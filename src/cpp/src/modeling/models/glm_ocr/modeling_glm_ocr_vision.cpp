// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/glm_ocr/modeling_glm_ocr_vision.hpp"

#include <algorithm>
#include <cmath>

#include <openvino/openvino.hpp>
#include <openvino/core/except.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/llm.hpp"
#include "modeling/ops/nn.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace {

ov::genai::modeling::Tensor add_bias_if_present(const ov::genai::modeling::Tensor& x,
                                                const ov::genai::modeling::Tensor* bias) {
    if (!bias) {
        return x;
    }
    return x + *bias;
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

// ======================== PatchEmbed ========================

GlmOcrVisionPatchEmbed::GlmOcrVisionPatchEmbed(BuilderContext& ctx,
                                                const std::string& name,
                                                const GlmOcrVisionConfig& cfg,
                                                Module* parent)
    : Module(name, ctx, parent),
      in_channels_(cfg.in_channels),
      patch_size_(cfg.patch_size),
      temporal_patch_size_(cfg.temporal_patch_size),
      embed_dim_(cfg.hidden_size) {
    weight_param_ = &register_parameter("proj.weight");
    bias_param_ = &register_parameter("proj.bias");
}

const Tensor& GlmOcrVisionPatchEmbed::weight() const {
    if (!weight_param_) {
        OPENVINO_THROW("GlmOcrVisionPatchEmbed weight parameter not registered");
    }
    return weight_param_->value();
}

const Tensor* GlmOcrVisionPatchEmbed::bias() const {
    return (bias_param_ && bias_param_->is_bound()) ? &bias_param_->value() : nullptr;
}

Tensor GlmOcrVisionPatchEmbed::forward(const Tensor& pixel_values) const {
    auto x = pixel_values.reshape({0, in_channels_, temporal_patch_size_, patch_size_, patch_size_});
    x = x.to(weight().dtype());
    const std::vector<int64_t> strides = {temporal_patch_size_, patch_size_, patch_size_};
    const std::vector<int64_t> pads = {0, 0, 0};
    Tensor conv;
    if (const auto* b = bias()) {
        conv = ops::nn::conv3d(x, weight(), *b, strides, pads, pads);
    } else {
        conv = ops::nn::conv3d(x, weight(), strides, pads, pads);
    }
    return conv.reshape({0, embed_dim_});
}

// ======================== VisionAttention ========================

GlmOcrVisionAttention::GlmOcrVisionAttention(BuilderContext& ctx,
                                              const std::string& name,
                                              const GlmOcrVisionConfig& cfg,
                                              Module* parent)
    : Module(name, ctx, parent),
      hidden_size_(cfg.hidden_size),
      num_heads_(cfg.num_heads),
      head_dim_(cfg.head_dim()),
      scaling_(1.0f / std::sqrt(static_cast<float>(head_dim_))) {
    if (hidden_size_ <= 0 || num_heads_ <= 0 || head_dim_ <= 0) {
        OPENVINO_THROW("Invalid GlmOcrVisionAttention configuration");
    }
    if (hidden_size_ != num_heads_ * head_dim_) {
        OPENVINO_THROW("GlmOcrVisionAttention hidden_size must equal num_heads * head_dim");
    }
    if (head_dim_ % 2 != 0) {
        OPENVINO_THROW("GlmOcrVisionAttention head_dim must be even for rotary embeddings");
    }
    qkv_weight_param_ = &register_parameter("qkv.weight");
    qkv_bias_param_ = &register_parameter("qkv.bias");
    proj_weight_param_ = &register_parameter("proj.weight");
    proj_bias_param_ = &register_parameter("proj.bias");
    q_norm_weight_param_ = &register_parameter("q_norm.weight");
    k_norm_weight_param_ = &register_parameter("k_norm.weight");
}

const Tensor& GlmOcrVisionAttention::qkv_weight() const {
    if (!qkv_weight_param_) {
        OPENVINO_THROW("GlmOcrVisionAttention qkv weight parameter not registered");
    }
    return qkv_weight_param_->value();
}

const Tensor* GlmOcrVisionAttention::qkv_bias() const {
    return (qkv_bias_param_ && qkv_bias_param_->is_bound()) ? &qkv_bias_param_->value() : nullptr;
}

const Tensor& GlmOcrVisionAttention::proj_weight() const {
    if (!proj_weight_param_) {
        OPENVINO_THROW("GlmOcrVisionAttention proj weight parameter not registered");
    }
    return proj_weight_param_->value();
}

const Tensor* GlmOcrVisionAttention::proj_bias() const {
    return (proj_bias_param_ && proj_bias_param_->is_bound()) ? &proj_bias_param_->value() : nullptr;
}

const Tensor& GlmOcrVisionAttention::q_norm_weight() const {
    if (!q_norm_weight_param_) {
        OPENVINO_THROW("GlmOcrVisionAttention q_norm weight parameter not registered");
    }
    return q_norm_weight_param_->value();
}

const Tensor& GlmOcrVisionAttention::k_norm_weight() const {
    if (!k_norm_weight_param_) {
        OPENVINO_THROW("GlmOcrVisionAttention k_norm weight parameter not registered");
    }
    return k_norm_weight_param_->value();
}

Tensor GlmOcrVisionAttention::apply_rotary(const Tensor& x,
                                           const Tensor& cos,
                                           const Tensor& sin) const {
    auto orig_dtype = x.dtype();
    auto x_f = x.to(ov::element::f32);
    auto cos_f = cos.to(ov::element::f32).unsqueeze(1);
    auto sin_f = sin.to(ov::element::f32).unsqueeze(1);

    const int64_t half = static_cast<int64_t>(head_dim_ / 2);
    auto x1 = ops::slice(x_f, 0, half, 1, 2);
    auto x2 = ops::slice(x_f, half, static_cast<int64_t>(head_dim_), 1, 2);
    auto rotated = ops::concat({-x2, x1}, 2);

    auto out = x_f * cos_f + rotated * sin_f;
    return out.to(orig_dtype);
}

Tensor GlmOcrVisionAttention::forward(const Tensor& hidden_states,
                                      const Tensor& rotary_cos,
                                      const Tensor& rotary_sin) const {
    auto qkv = add_bias_if_present(ops::linear(hidden_states, qkv_weight()), qkv_bias());
    auto qkv_reshaped = qkv.reshape({0, 3, num_heads_, head_dim_});
    auto q = ops::slice(qkv_reshaped, 0, 1, 1, 1).squeeze(1);
    auto k = ops::slice(qkv_reshaped, 1, 2, 1, 1).squeeze(1);
    auto v = ops::slice(qkv_reshaped, 2, 3, 1, 1).squeeze(1);

    // Apply RMSNorm to q and k (per-head, dim=head_dim=64)
    auto q_normed = ops::nn::rms_norm(q, q_norm_weight(), 1e-5f, -1);
    auto k_normed = ops::nn::rms_norm(k, k_norm_weight(), 1e-5f, -1);

    auto q_rot = apply_rotary(q_normed, rotary_cos, rotary_sin);
    auto k_rot = apply_rotary(k_normed, rotary_cos, rotary_sin);

    auto q_heads = q_rot.permute({1, 0, 2}).unsqueeze(0);
    auto k_heads = k_rot.permute({1, 0, 2}).unsqueeze(0);
    auto v_heads = v.permute({1, 0, 2}).unsqueeze(0);

    auto* policy = &ctx().op_policy();
    auto context = ops::llm::sdpa(q_heads, k_heads, v_heads, scaling_, 3, nullptr, false, policy);
    const int64_t attn_out_dim = static_cast<int64_t>(hidden_size_);
    auto merged = context.permute({0, 2, 1, 3}).reshape({0, 0, attn_out_dim});
    auto merged_2d = merged.squeeze(0);
    auto out = add_bias_if_present(ops::linear(merged_2d, proj_weight()), proj_bias());
    return out;
}

// ======================== VisionMLP (gate/up/down SiLU) ========================

GlmOcrVisionMLP::GlmOcrVisionMLP(BuilderContext& ctx,
                                  const std::string& name,
                                  const GlmOcrVisionConfig& cfg,
                                  Module* parent)
    : Module(name, ctx, parent) {
    if (!cfg.hidden_act.empty() && cfg.hidden_act != "silu") {
        OPENVINO_THROW("Unsupported GlmOcrVision MLP activation: ", cfg.hidden_act);
    }
    gate_proj_weight_param_ = &register_parameter("gate_proj.weight");
    gate_proj_bias_param_ = &register_parameter("gate_proj.bias");
    up_proj_weight_param_ = &register_parameter("up_proj.weight");
    up_proj_bias_param_ = &register_parameter("up_proj.bias");
    down_proj_weight_param_ = &register_parameter("down_proj.weight");
    down_proj_bias_param_ = &register_parameter("down_proj.bias");
}

const Tensor& GlmOcrVisionMLP::gate_proj_weight() const {
    if (!gate_proj_weight_param_) OPENVINO_THROW("GlmOcrVisionMLP gate_proj weight not registered");
    return gate_proj_weight_param_->value();
}

const Tensor* GlmOcrVisionMLP::gate_proj_bias() const {
    return (gate_proj_bias_param_ && gate_proj_bias_param_->is_bound()) ? &gate_proj_bias_param_->value() : nullptr;
}

const Tensor& GlmOcrVisionMLP::up_proj_weight() const {
    if (!up_proj_weight_param_) OPENVINO_THROW("GlmOcrVisionMLP up_proj weight not registered");
    return up_proj_weight_param_->value();
}

const Tensor* GlmOcrVisionMLP::up_proj_bias() const {
    return (up_proj_bias_param_ && up_proj_bias_param_->is_bound()) ? &up_proj_bias_param_->value() : nullptr;
}

const Tensor& GlmOcrVisionMLP::down_proj_weight() const {
    if (!down_proj_weight_param_) OPENVINO_THROW("GlmOcrVisionMLP down_proj weight not registered");
    return down_proj_weight_param_->value();
}

const Tensor* GlmOcrVisionMLP::down_proj_bias() const {
    return (down_proj_bias_param_ && down_proj_bias_param_->is_bound()) ? &down_proj_bias_param_->value() : nullptr;
}

Tensor GlmOcrVisionMLP::forward(const Tensor& hidden_states) const {
    auto gate = add_bias_if_present(ops::linear(hidden_states, gate_proj_weight()), gate_proj_bias());
    auto up = add_bias_if_present(ops::linear(hidden_states, up_proj_weight()), up_proj_bias());
    auto gated = ops::silu(gate) * up;
    return add_bias_if_present(ops::linear(gated, down_proj_weight()), down_proj_bias());
}

// ======================== VisionBlock ========================

GlmOcrVisionBlock::GlmOcrVisionBlock(BuilderContext& ctx,
                                      const std::string& name,
                                      const GlmOcrVisionConfig& cfg,
                                      Module* parent)
    : Module(name, ctx, parent),
      attn_(ctx, "attn", cfg, this),
      mlp_(ctx, "mlp", cfg, this),
      eps_(cfg.rms_norm_eps) {
    register_module("attn", &attn_);
    register_module("mlp", &mlp_);

    norm1_weight_param_ = &register_parameter("norm1.weight");
    norm2_weight_param_ = &register_parameter("norm2.weight");
}

const Tensor& GlmOcrVisionBlock::norm1_weight() const {
    if (!norm1_weight_param_) OPENVINO_THROW("GlmOcrVisionBlock norm1 weight not registered");
    return norm1_weight_param_->value();
}

const Tensor& GlmOcrVisionBlock::norm2_weight() const {
    if (!norm2_weight_param_) OPENVINO_THROW("GlmOcrVisionBlock norm2 weight not registered");
    return norm2_weight_param_->value();
}

Tensor GlmOcrVisionBlock::forward(const Tensor& hidden_states,
                                  const Tensor& rotary_cos,
                                  const Tensor& rotary_sin) const {
    // RMSNorm (no bias) instead of LayerNorm
    auto norm1 = ops::nn::rms_norm(hidden_states, norm1_weight(), eps_, -1);
    auto attn_out = attn_.forward(norm1, rotary_cos, rotary_sin);
    auto resid1 = hidden_states + attn_out;
    auto norm2 = ops::nn::rms_norm(resid1, norm2_weight(), eps_, -1);
    auto mlp_out = mlp_.forward(norm2);
    return resid1 + mlp_out;
}

// ======================== PatchMerger ========================

GlmOcrVisionPatchMerger::GlmOcrVisionPatchMerger(BuilderContext& ctx,
                                                  const std::string& name,
                                                  const GlmOcrVisionConfig& cfg,
                                                  Module* parent)
    : Module(name, ctx, parent),
      hidden_size_(cfg.hidden_size),
      out_hidden_size_(cfg.out_hidden_size),
      spatial_merge_size_(cfg.spatial_merge_size),
      eps_(cfg.rms_norm_eps) {
    // proj(out_hidden_size, out_hidden_size) - no bias
    proj_weight_param_ = &register_parameter("proj.weight");
    // post_projection_norm: LayerNorm(out_hidden_size) - has weight+bias
    norm_weight_param_ = &register_parameter("post_projection_norm.weight");
    norm_bias_param_ = &register_parameter("post_projection_norm.bias");
    // gate/up/down MLP (out_hidden_size ↔ 4608, no bias, SiLU)
    gate_proj_weight_param_ = &register_parameter("gate_proj.weight");
    up_proj_weight_param_ = &register_parameter("up_proj.weight");
    down_proj_weight_param_ = &register_parameter("down_proj.weight");
}

const Tensor& GlmOcrVisionPatchMerger::proj_weight() const {
    if (!proj_weight_param_) OPENVINO_THROW("GlmOcrVisionPatchMerger proj weight not registered");
    return proj_weight_param_->value();
}

const Tensor& GlmOcrVisionPatchMerger::norm_weight() const {
    if (!norm_weight_param_) OPENVINO_THROW("GlmOcrVisionPatchMerger norm weight not registered");
    return norm_weight_param_->value();
}

const Tensor& GlmOcrVisionPatchMerger::norm_bias() const {
    if (!norm_bias_param_) OPENVINO_THROW("GlmOcrVisionPatchMerger norm bias not registered");
    return norm_bias_param_->value();
}

const Tensor& GlmOcrVisionPatchMerger::gate_proj_weight() const {
    if (!gate_proj_weight_param_) OPENVINO_THROW("GlmOcrVisionPatchMerger gate_proj weight not registered");
    return gate_proj_weight_param_->value();
}

const Tensor& GlmOcrVisionPatchMerger::up_proj_weight() const {
    if (!up_proj_weight_param_) OPENVINO_THROW("GlmOcrVisionPatchMerger up_proj weight not registered");
    return up_proj_weight_param_->value();
}

const Tensor& GlmOcrVisionPatchMerger::down_proj_weight() const {
    if (!down_proj_weight_param_) OPENVINO_THROW("GlmOcrVisionPatchMerger down_proj weight not registered");
    return down_proj_weight_param_->value();
}

Tensor GlmOcrVisionPatchMerger::forward(const Tensor& hidden_states) const {
    // Input: [num_patches, out_hidden_size] (after downsample conv2d)
    // proj + LayerNorm + GELU
    auto projected = ops::linear(hidden_states, proj_weight());
    auto normed = ops::nn::layer_norm(projected, norm_weight(), &norm_bias(), eps_, -1);
    auto activated = ops::nn::gelu(normed, false);  // exact GELU (erf), matching PyTorch nn.GELU()
    // gate/up/down MLP with SiLU
    auto gate = ops::linear(activated, gate_proj_weight());
    auto up = ops::linear(activated, up_proj_weight());
    auto gated = ops::silu(gate) * up;
    return ops::linear(gated, down_proj_weight());
}

// ======================== VisionModel ========================

GlmOcrVisionModel::GlmOcrVisionModel(BuilderContext& ctx,
                                      const GlmOcrVisionConfig& cfg,
                                      Module* parent)
    : Module(GlmOcrModuleNames::kVision, ctx, parent),
      cfg_(cfg),
      patch_embed_(ctx, "patch_embed", cfg, this),
      blocks_(),
      merger_(ctx, "merger", cfg, this),
      eps_(cfg.rms_norm_eps) {
    register_module("patch_embed", &patch_embed_);
    register_module("merger", &merger_);

    blocks_.reserve(static_cast<size_t>(cfg.depth));
    for (int32_t i = 0; i < cfg.depth; ++i) {
        blocks_.emplace_back(ctx, GlmOcrModuleNames::vision_block(i), cfg, this);
        register_module(GlmOcrModuleNames::vision_block(i), &blocks_.back());
    }

    post_layernorm_weight_param_ = &register_parameter("post_layernorm.weight");
    downsample_weight_param_ = &register_parameter("downsample.weight");
    downsample_bias_param_ = &register_parameter("downsample.bias");
}

const Tensor& GlmOcrVisionModel::post_layernorm_weight() const {
    if (!post_layernorm_weight_param_) OPENVINO_THROW("post_layernorm weight not registered");
    return post_layernorm_weight_param_->value();
}

const Tensor& GlmOcrVisionModel::downsample_weight() const {
    if (!downsample_weight_param_) OPENVINO_THROW("downsample weight not registered");
    return downsample_weight_param_->value();
}

const Tensor* GlmOcrVisionModel::downsample_bias() const {
    return (downsample_bias_param_ && downsample_bias_param_->is_bound()) ? &downsample_bias_param_->value() : nullptr;
}

Tensor GlmOcrVisionModel::forward(const Tensor& pixel_values,
                                  const Tensor& grid_thw,
                                  const Tensor& rotary_cos,
                                  const Tensor& rotary_sin) {
    auto hidden_states = patch_embed_.forward(pixel_values);

    for (size_t layer_idx = 0; layer_idx < blocks_.size(); ++layer_idx) {
        hidden_states = blocks_[layer_idx].forward(hidden_states, rotary_cos, rotary_sin);
    }

    // post_layernorm: RMSNorm
    hidden_states = ops::nn::rms_norm(hidden_states, post_layernorm_weight(), eps_, -1);

    // Downsample via Conv2d(1024→1536, k=2, s=2)
    // Match PyTorch's view(-1, merge_size, merge_size, C) + Conv2d
    // Patches are ordered block_h→block_w→merge_h→merge_w, so every
    // merge_size*merge_size consecutive patches form one spatial merge group.
    // PyTorch: hidden_states.view(-1, merge, merge, C).permute(0,3,1,2) -> conv2d
    auto hidden_for_conv = hidden_states.to(downsample_weight().dtype());

    // Reshape: [num_patches, C] -> [-1, merge, merge, C]
    auto reshaped = hidden_for_conv.reshape({-1,
        static_cast<int64_t>(cfg_.spatial_merge_size),
        static_cast<int64_t>(cfg_.spatial_merge_size),
        static_cast<int64_t>(cfg_.hidden_size)});

    // Permute to [-1, C, merge, merge] for conv2d
    auto nchw = reshaped.permute({0, 3, 1, 2});

    // Apply Conv2d(hidden_size→out_hidden_size, kernel=merge, stride=merge)
    // kernel=2, stride=2 on a 2x2 input → 1x1 output per group
    const std::vector<int64_t> strides_2d = {
        static_cast<int64_t>(cfg_.spatial_merge_size),
        static_cast<int64_t>(cfg_.spatial_merge_size)};
    const std::vector<int64_t> pads_2d = {0, 0};
    Tensor conv_out;
    if (const auto* b = downsample_bias()) {
        conv_out = ops::nn::conv2d(nchw, downsample_weight(), *b, strides_2d, pads_2d, pads_2d);
    } else {
        conv_out = ops::nn::conv2d(nchw, downsample_weight(), strides_2d, pads_2d, pads_2d);
    }

    // conv_out: [-1, out_hidden_size, 1, 1] -> reshape to [-1, out_hidden_size]
    auto flat = conv_out.reshape({-1, cfg_.out_hidden_size});

    // Apply merger (proj + LN + GELU + gate/up/down MLP)
    return merger_.forward(flat);
}

GlmOcrVisionPatchEmbed& GlmOcrVisionModel::patch_embed() {
    return patch_embed_;
}

GlmOcrVisionPatchMerger& GlmOcrVisionModel::merger() {
    return merger_;
}

std::shared_ptr<ov::Model> create_glm_ocr_vision_model(
    const GlmOcrConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    GlmOcrVisionModel model(ctx, cfg.vision);
    model.packed_mapping().rules.push_back({"model.", "", 0});

    ov::genai::modeling::weights::LoadOptions options;
    options.allow_unmatched = true;
    options.allow_missing = false;
    options.report_missing = true;
    options.report_unmatched = true;
    auto report = ov::genai::modeling::weights::load_model(model, source, finalizer, options);
    (void)report;

    auto pixel_values = ctx.parameter(GlmOcrVisionIO::kPixelValues,
                                      ov::element::f32,
                                      ov::PartialShape{-1,
                                                       cfg.vision.in_channels,
                                                       cfg.vision.temporal_patch_size,
                                                       cfg.vision.patch_size,
                                                       cfg.vision.patch_size});
    auto grid_thw = ctx.parameter(GlmOcrVisionIO::kGridThw,
                                  ov::element::i64,
                                  ov::PartialShape{-1, 3});
    auto rotary_cos = ctx.parameter(GlmOcrVisionIO::kRotaryCos,
                                    ov::element::f32,
                                    ov::PartialShape{-1, cfg.vision.head_dim()});
    auto rotary_sin = ctx.parameter(GlmOcrVisionIO::kRotarySin,
                                    ov::element::f32,
                                    ov::PartialShape{-1, cfg.vision.head_dim()});

    auto output = model.forward(pixel_values, grid_thw, rotary_cos, rotary_sin);

    auto visual = std::make_shared<ov::op::v0::Result>(output.output());
    set_name(visual, GlmOcrVisionIO::kVisualEmbeds);

    return ctx.build_model({visual->output(0)});
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
