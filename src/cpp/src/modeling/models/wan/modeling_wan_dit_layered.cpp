// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/wan/modeling_wan_dit_layered.hpp"

#include <cmath>
#include <vector>

#include <openvino/core/except.hpp>
#include <openvino/core/model.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/llm.hpp"
#include "modeling/ops/nn.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/ops/shape.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace {

// Helper function to load model weights with allow_unmatched=true
// This is needed because each layered sub-model only contains a subset of the full model's parameters
void load_model_partial(ov::genai::modeling::Module& model,
                        ov::genai::modeling::weights::WeightSource& source,
                        ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    ov::genai::modeling::weights::LoadOptions options;
    options.allow_missing = true;
    options.allow_unmatched = true;  // Allow weights that don't match any parameter
    options.report_missing = false;
    options.report_unmatched = false;
    (void)ov::genai::modeling::weights::load_model(model, source, finalizer, options);
}

auto set_name = [](const std::shared_ptr<ov::Node>& node, const std::string& name) {
    node->output(0).set_names({name});
    node->set_friendly_name(name);
};

ov::Output<ov::Node> div_dim(const ov::Output<ov::Node>& dim, int64_t divisor, ov::genai::modeling::OpContext* ctx) {
    auto denom = ov::genai::modeling::ops::const_scalar(ctx, static_cast<int64_t>(divisor));
    auto div = std::make_shared<ov::op::v1::Divide>(dim, denom, ov::op::AutoBroadcastType::NUMPY);
    return std::make_shared<ov::op::v0::Convert>(div, ov::element::i64);
}

ov::genai::modeling::Tensor add_bias_if_present(const ov::genai::modeling::Tensor& x,
                                                const ov::genai::modeling::Tensor* bias) {
    if (!bias) {
        return x;
    }
    return x + *bias;
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

// ============================================================================
// WanDitPreprocess Implementation
// ============================================================================

WanDitPreprocess::WanDitPreprocess(BuilderContext& ctx,
                                   const WanTransformer3DConfig& cfg,
                                   Module* parent)
    : Module("", ctx, parent),
      cfg_(cfg),
      inner_dim_(cfg.inner_dim()),
      rope_(ctx,
            "rope",
            cfg.attention_head_dim,
            cfg.patch_size,
            cfg.rope_max_seq_len,
            10000.0f,
            this),
      condition_embedder_(ctx,
                          "condition_embedder",
                          inner_dim_,
                          cfg.freq_dim,
                          inner_dim_ * 6,
                          cfg.text_dim,
                          cfg.image_dim,
                          cfg.pos_embed_seq_len,
                          this) {
    register_module("rope", &rope_);
    register_module("condition_embedder", &condition_embedder_);

    patch_weight_ = &register_parameter("patch_embedding.weight");
    patch_bias_ = &register_parameter("patch_embedding.bias");
}

WanDitPreprocessOutput WanDitPreprocess::forward(const Tensor& hidden_states,
                                                  const Tensor& timestep,
                                                  const Tensor& encoder_hidden_states,
                                                  const Tensor* encoder_hidden_states_image) {
    auto* ctx = hidden_states.context();
    auto batch = shape::dim(hidden_states, 0);
    auto frames = shape::dim(hidden_states, 2);
    auto height = shape::dim(hidden_states, 3);
    auto width = shape::dim(hidden_states, 4);

    auto ppf = div_dim(frames, cfg_.patch_size[0], ctx);
    auto pph = div_dim(height, cfg_.patch_size[1], ctx);
    auto ppw = div_dim(width, cfg_.patch_size[2], ctx);

    // Compute RoPE embeddings
    auto rope = rope_.forward(hidden_states);

    // Patch embedding via 3D convolution
    auto x = hidden_states.to(patch_weight_->value().dtype());
    x = ops::nn::conv3d(x,
                        patch_weight_->value(),
                        patch_bias_->value(),
                        {cfg_.patch_size[0], cfg_.patch_size[1], cfg_.patch_size[2]},
                        {0, 0, 0},
                        {0, 0, 0});

    // Reshape to token sequence
    auto seq = std::make_shared<ov::op::v1::Multiply>(ppf, pph);
    seq = std::make_shared<ov::op::v1::Multiply>(seq, ppw);
    auto embed = shape::dim(x, 1);
    auto shape_tokens = shape::make({batch, embed, seq});
    auto tokens = x.reshape(shape_tokens).permute({0, 2, 1});

    // Compute condition embeddings
    auto cond = condition_embedder_.forward(timestep, encoder_hidden_states, encoder_hidden_states_image);
    auto timestep_proj = cond.timestep_proj.reshape({0, 6, inner_dim_});

    WanDitPreprocessOutput output;
    output.tokens = tokens;
    output.rotary_cos = rope.first;
    output.rotary_sin = rope.second;
    output.temb = cond.temb;
    output.timestep_proj = timestep_proj;
    output.text_embeds = cond.text_embeds;
    output.image_embeds = cond.image_embeds;

    return output;
}

// ============================================================================
// WanDitBlockGroup Implementation
// ============================================================================

WanDitBlockGroup::WanDitBlockGroup(BuilderContext& ctx,
                                   const WanTransformer3DConfig& cfg,
                                   int32_t start_layer,
                                   int32_t num_layers,
                                   Module* parent)
    : Module("", ctx, parent),  // Empty name to ensure correct weight paths
      cfg_(cfg),
      start_layer_(start_layer) {
    if (num_layers <= 0) {
        OPENVINO_THROW("WanDitBlockGroup requires num_layers > 0");
    }
    if (start_layer < 0 || start_layer + num_layers > cfg.num_layers) {
        OPENVINO_THROW("WanDitBlockGroup layer range out of bounds");
    }

    blocks_.reserve(static_cast<size_t>(num_layers));
    for (int32_t i = 0; i < num_layers; ++i) {
        int32_t layer_idx = start_layer + i;
        std::string block_name = "blocks." + std::to_string(layer_idx);
        blocks_.emplace_back(ctx, block_name, cfg, this);
        register_module(block_name, &blocks_.back());
    }
}

Tensor WanDitBlockGroup::forward(const Tensor& hidden_states,
                                 const Tensor& text_embeds,
                                 const Tensor& timestep_proj,
                                 const Tensor& rotary_cos,
                                 const Tensor& rotary_sin,
                                 const Tensor* image_embeds) {
    auto hs = hidden_states;
    for (auto& block : blocks_) {
        hs = block.forward(hs,
                           text_embeds,
                           timestep_proj,
                           rotary_cos,
                           rotary_sin,
                           image_embeds);
    }
    return hs;
}

// ============================================================================
// WanDitPostprocess Implementation
// ============================================================================

WanDitPostprocess::WanDitPostprocess(BuilderContext& ctx,
                                     const WanTransformer3DConfig& cfg,
                                     Module* parent)
    : Module("", ctx, parent),
      cfg_(cfg),
      inner_dim_(cfg.inner_dim()),
      norm_out_(ctx, "norm_out", cfg.eps, false, false, this) {
    register_module("norm_out", &norm_out_);

    proj_out_weight_ = &register_parameter("proj_out.weight");
    proj_out_bias_ = &register_parameter("proj_out.bias");
    scale_shift_table_ = &register_parameter("scale_shift_table");
}

Tensor WanDitPostprocess::forward(const Tensor& hidden_states,
                                  const Tensor& temb,
                                  const ov::Output<ov::Node>& batch,
                                  const ov::Output<ov::Node>& ppf,
                                  const ov::Output<ov::Node>& pph,
                                  const ov::Output<ov::Node>& ppw) {
    auto* ctx = hidden_states.context();

    // Apply scale/shift modulation
    auto table = scale_shift_table_->value().to(ov::element::f32);
    auto temb_f = temb.to(ov::element::f32).unsqueeze(1);
    auto mod = table + temb_f;
    auto shift = ops::slice(mod, 0, 1, 1, 1);
    auto scale = ops::slice(mod, 1, 2, 1, 1);

    // Layer norm with modulation
    auto normed = norm_out_.forward(hidden_states.to(ov::element::f32));
    normed = (normed * (scale + 1.0f) + shift).to(hidden_states.dtype());

    // Output projection
    auto proj_in = normed.to(proj_out_weight_->value().dtype());
    auto out = add_bias_if_present(ops::linear(proj_in, proj_out_weight_->value()),
                                   proj_out_bias_ ? &proj_out_bias_->value() : nullptr);

    // Unpatch: reshape back to video format
    auto p_t = ops::const_scalar(ctx, static_cast<int64_t>(cfg_.patch_size[0]));
    auto p_h = ops::const_scalar(ctx, static_cast<int64_t>(cfg_.patch_size[1]));
    auto p_w = ops::const_scalar(ctx, static_cast<int64_t>(cfg_.patch_size[2]));

    auto out_ch = ops::const_vec(ctx, std::vector<int64_t>{cfg_.out_channels});
    auto p_t_vec = ops::const_vec(ctx, std::vector<int64_t>{cfg_.patch_size[0]});
    auto p_h_vec = ops::const_vec(ctx, std::vector<int64_t>{cfg_.patch_size[1]});
    auto p_w_vec = ops::const_vec(ctx, std::vector<int64_t>{cfg_.patch_size[2]});

    // Reshape scalar inputs (shape {}) to 1D (shape {1}) for compatibility with shape::make
    // shape::make uses Concat which requires all inputs to have the same rank
    auto target_shape_1d = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto ppf_1d = std::make_shared<ov::op::v1::Reshape>(ppf, target_shape_1d, false);
    auto pph_1d = std::make_shared<ov::op::v1::Reshape>(pph, target_shape_1d, false);
    auto ppw_1d = std::make_shared<ov::op::v1::Reshape>(ppw, target_shape_1d, false);

    auto reshape_shape = shape::make({batch, ppf_1d, pph_1d, ppw_1d, p_t_vec, p_h_vec, p_w_vec, out_ch});
    auto unpatched = out.reshape(reshape_shape).permute({0, 7, 1, 4, 2, 5, 3, 6});

    auto out_frames = std::make_shared<ov::op::v1::Multiply>(ppf_1d, p_t);
    auto out_height = std::make_shared<ov::op::v1::Multiply>(pph_1d, p_h);
    auto out_width = std::make_shared<ov::op::v1::Multiply>(ppw_1d, p_w);
    auto final_shape = shape::make({batch, out_ch, out_frames, out_height, out_width});
    return unpatched.reshape(final_shape);
}

// ============================================================================
// Model Creation Functions
// ============================================================================

std::shared_ptr<ov::Model> create_wan_dit_preprocess_model(
    const WanTransformer3DConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    ctx.op_policy().use_internal_rope = false;
    WanDitPreprocess model(ctx, cfg);

    WanWeightMapping::apply_transformer_packed_mapping(model);
    load_model_partial(model, source, finalizer);

    // Define inputs
    auto latents = ctx.parameter("hidden_states",
                                 ov::element::f32,
                                 ov::PartialShape{-1, cfg.in_channels, -1, -1, -1});
    auto timesteps = ctx.parameter("timestep", ov::element::f32, ov::PartialShape{-1});
    auto text = ctx.parameter("encoder_hidden_states",
                              ov::element::f32,
                              ov::PartialShape{-1, -1, cfg.text_dim});

    // Forward pass
    auto output = model.forward(latents, timesteps, text, nullptr);

    // Compute shape info for output
    auto* op_ctx = &ctx.op_context();
    auto batch = shape::dim(latents, 0);
    auto frames = shape::dim(latents, 2);
    auto height = shape::dim(latents, 3);
    auto width = shape::dim(latents, 4);

    auto ppf = div_dim(frames, cfg.patch_size[0], op_ctx);
    auto pph = div_dim(height, cfg.patch_size[1], op_ctx);
    auto ppw = div_dim(width, cfg.patch_size[2], op_ctx);

    // Create results for all outputs
    auto result_tokens = std::make_shared<ov::op::v0::Result>(output.tokens.output());
    set_name(result_tokens, "tokens");

    auto result_rotary_cos = std::make_shared<ov::op::v0::Result>(output.rotary_cos.output());
    set_name(result_rotary_cos, "rotary_cos");

    auto result_rotary_sin = std::make_shared<ov::op::v0::Result>(output.rotary_sin.output());
    set_name(result_rotary_sin, "rotary_sin");

    auto result_temb = std::make_shared<ov::op::v0::Result>(output.temb.output());
    set_name(result_temb, "temb");

    auto result_timestep_proj = std::make_shared<ov::op::v0::Result>(output.timestep_proj.output());
    set_name(result_timestep_proj, "timestep_proj");

    auto result_text_embeds = std::make_shared<ov::op::v0::Result>(output.text_embeds.output());
    set_name(result_text_embeds, "text_embeds");

    // Output shape info as scalars for postprocess
    auto result_ppf = std::make_shared<ov::op::v0::Result>(ppf);
    set_name(result_ppf, "ppf");

    auto result_pph = std::make_shared<ov::op::v0::Result>(pph);
    set_name(result_pph, "pph");

    auto result_ppw = std::make_shared<ov::op::v0::Result>(ppw);
    set_name(result_ppw, "ppw");

    auto ov_model = ctx.build_model({
        result_tokens->output(0),
        result_rotary_cos->output(0),
        result_rotary_sin->output(0),
        result_temb->output(0),
        result_timestep_proj->output(0),
        result_text_embeds->output(0),
        result_ppf->output(0),
        result_pph->output(0),
        result_ppw->output(0)
    });
    ov_model->set_friendly_name("wan_dit_layered_preprocess");
    return ov_model;
}

std::shared_ptr<ov::Model> create_wan_dit_block_group_model(
    const WanTransformer3DConfig& cfg,
    int32_t start_layer,
    int32_t num_layers,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    ctx.op_policy().use_internal_rope = false;

    // Create block group with empty name to ensure weight paths are correct
    // (e.g., "blocks.0.attn1.to_q.weight" instead of "group.blocks.0.attn1.to_q.weight")
    WanDitBlockGroup model(ctx, cfg, start_layer, num_layers);

    WanWeightMapping::apply_transformer_packed_mapping(model);
    load_model_partial(model, source, finalizer);

    const int32_t inner_dim = cfg.inner_dim();

    // Define inputs matching preprocess outputs
    auto hidden_states = ctx.parameter("hidden_states",
                                       ov::element::f32,
                                       ov::PartialShape{-1, -1, inner_dim});
    auto text_embeds = ctx.parameter("text_embeds",
                                     ov::element::f32,
                                     ov::PartialShape{-1, -1, inner_dim});
    auto timestep_proj = ctx.parameter("timestep_proj",
                                       ov::element::f32,
                                       ov::PartialShape{-1, 6, inner_dim});
    auto rotary_cos = ctx.parameter("rotary_cos",
                                    ov::element::f32,
                                    ov::PartialShape{-1, -1, -1});
    auto rotary_sin = ctx.parameter("rotary_sin",
                                    ov::element::f32,
                                    ov::PartialShape{-1, -1, -1});

    // Forward pass
    auto output = model.forward(hidden_states, text_embeds, timestep_proj,
                                rotary_cos, rotary_sin, nullptr);

    auto result = std::make_shared<ov::op::v0::Result>(output.output());
    set_name(result, "hidden_states_out");

    auto ov_model = ctx.build_model({result->output(0)});
    ov_model->set_friendly_name("wan_dit_layered_block_group_" + std::to_string(start_layer) +
                                "_n" + std::to_string(num_layers));
    return ov_model;
}

std::shared_ptr<ov::Model> create_wan_dit_postprocess_model(
    const WanTransformer3DConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    ctx.op_policy().use_internal_rope = false;
    WanDitPostprocess model(ctx, cfg);

    WanWeightMapping::apply_transformer_packed_mapping(model);
    load_model_partial(model, source, finalizer);

    const int32_t inner_dim = cfg.inner_dim();

    // Define inputs
    auto hidden_states = ctx.parameter("hidden_states",
                                       ov::element::f32,
                                       ov::PartialShape{-1, -1, inner_dim});
    auto temb = ctx.parameter("temb",
                              ov::element::f32,
                              ov::PartialShape{-1, inner_dim});
    // Shape {1} to match preprocess output (div_dim uses Gather which produces {1})
    auto ppf_param = ctx.parameter("ppf", ov::element::i64, ov::PartialShape{1});
    auto pph_param = ctx.parameter("pph", ov::element::i64, ov::PartialShape{1});
    auto ppw_param = ctx.parameter("ppw", ov::element::i64, ov::PartialShape{1});

    // Get batch from hidden_states
    auto batch = shape::dim(hidden_states, 0);

    // Forward pass
    auto output = model.forward(hidden_states, temb, batch,
                                ppf_param.output(), pph_param.output(), ppw_param.output());

    auto result = std::make_shared<ov::op::v0::Result>(output.output());
    set_name(result, "sample");

    auto ov_model = ctx.build_model({result->output(0)});
    ov_model->set_friendly_name("wan_dit_layered_postprocess");
    return ov_model;
}

WanDitLayeredModels create_wan_dit_layered_models(
    const WanTransformer3DConfig& cfg,
    const WanDitLayeredConfig& layered_cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    WanDitLayeredModels models;

    // Create preprocess model
    models.preprocess = create_wan_dit_preprocess_model(cfg, source, finalizer);

    // Create block group models
    const int32_t layers_per_group = layered_cfg.layers_per_block_group;
    const int32_t total_layers = cfg.num_layers;
    const int32_t num_groups = (total_layers + layers_per_group - 1) / layers_per_group;

    models.block_groups.reserve(static_cast<size_t>(num_groups));
    for (int32_t g = 0; g < num_groups; ++g) {
        int32_t start = g * layers_per_group;
        int32_t count = std::min(layers_per_group, total_layers - start);
        models.block_groups.push_back(
            create_wan_dit_block_group_model(cfg, start, count, source, finalizer));
    }

    // Create postprocess model
    models.postprocess = create_wan_dit_postprocess_model(cfg, source, finalizer);

    return models;
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
