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

#include "modeling/layers/layer_norm.hpp"
#include "modeling/layers/rms_norm.hpp"
#include "modeling/models/wan/modeling_wan_dit.hpp"
#include "modeling/models/wan/processing_wan.hpp"
#include "modeling/module.hpp"
#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

// ============================================================================
// Layered DIT Model Structures
// ============================================================================

/**
 * @brief Output structure for the preprocess stage of WAN DIT model.
 *
 * Contains all intermediate tensors needed by transformer blocks.
 */
struct WanDitPreprocessOutput {
    Tensor tokens;           // Patched and embedded hidden states [B, seq, inner_dim]
    Tensor rotary_cos;       // RoPE cosine embeddings [B, seq, head_dim/2]
    Tensor rotary_sin;       // RoPE sine embeddings [B, seq, head_dim/2]
    Tensor temb;             // Time embedding [B, inner_dim]
    Tensor timestep_proj;    // Projected timestep [B, 6, inner_dim]
    Tensor text_embeds;      // Projected text embeddings [B, text_seq, inner_dim]
    std::optional<Tensor> image_embeds;  // Optional image embeddings
};

/**
 * @brief Preprocess module for WAN DIT model.
 *
 * Handles:
 * - RoPE position embedding computation
 * - Patch embedding (3D convolution)
 * - Condition embedding (time, text, optional image)
 */
class WanDitPreprocess : public Module {
public:
    WanDitPreprocess(BuilderContext& ctx,
                     const WanTransformer3DConfig& cfg,
                     Module* parent = nullptr);

    WanDitPreprocessOutput forward(const Tensor& hidden_states,
                                   const Tensor& timestep,
                                   const Tensor& encoder_hidden_states,
                                   const Tensor* encoder_hidden_states_image = nullptr);

    const WanTransformer3DConfig& config() const { return cfg_; }

private:
    WanTransformer3DConfig cfg_;
    int32_t inner_dim_ = 0;

    WanRotaryPosEmbed rope_;
    WanTimeTextImageEmbedding condition_embedder_;

    WeightParameter* patch_weight_ = nullptr;
    WeightParameter* patch_bias_ = nullptr;
};

/**
 * @brief A group of transformer blocks.
 *
 * Contains N consecutive WanTransformerBlock layers that can be
 * compiled and executed as a single OV model.
 *
 * Note: The module name is intentionally empty to ensure correct weight paths.
 * Internal blocks will have paths like "blocks.0", "blocks.1", etc.
 */
class WanDitBlockGroup : public Module {
public:
    WanDitBlockGroup(BuilderContext& ctx,
                     const WanTransformer3DConfig& cfg,
                     int32_t start_layer,
                     int32_t num_layers,
                     Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states,
                   const Tensor& text_embeds,
                   const Tensor& timestep_proj,
                   const Tensor& rotary_cos,
                   const Tensor& rotary_sin,
                   const Tensor* image_embeds = nullptr);

    int32_t start_layer() const { return start_layer_; }
    int32_t num_layers() const { return static_cast<int32_t>(blocks_.size()); }

private:
    WanTransformer3DConfig cfg_;
    int32_t start_layer_ = 0;
    std::vector<WanTransformerBlock> blocks_;
};

/**
 * @brief Postprocess module for WAN DIT model.
 *
 * Handles:
 * - Final layer normalization with scale/shift modulation
 * - Output projection
 * - Unpatching (reshape back to video format)
 */
class WanDitPostprocess : public Module {
public:
    WanDitPostprocess(BuilderContext& ctx,
                      const WanTransformer3DConfig& cfg,
                      Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states,
                   const Tensor& temb,
                   const ov::Output<ov::Node>& batch,
                   const ov::Output<ov::Node>& ppf,
                   const ov::Output<ov::Node>& pph,
                   const ov::Output<ov::Node>& ppw);

    const WanTransformer3DConfig& config() const { return cfg_; }

private:
    WanTransformer3DConfig cfg_;
    int32_t inner_dim_ = 0;

    FP32LayerNorm norm_out_;

    WeightParameter* proj_out_weight_ = nullptr;
    WeightParameter* proj_out_bias_ = nullptr;
    WeightParameter* scale_shift_table_ = nullptr;
};

// ============================================================================
// Layered Model Creation Functions
// ============================================================================

/**
 * @brief Configuration for layered DIT model creation.
 */
struct WanDitLayeredConfig {
    int32_t layers_per_block_group = 1;  // Number of transformer layers per block group
};

/**
 * @brief Collection of layered DIT models.
 */
struct WanDitLayeredModels {
    std::shared_ptr<ov::Model> preprocess;
    std::vector<std::shared_ptr<ov::Model>> block_groups;
    std::shared_ptr<ov::Model> postprocess;

    int32_t num_block_groups() const { return static_cast<int32_t>(block_groups.size()); }
};

/**
 * @brief Create the preprocess model for layered DIT.
 *
 * @param cfg Transformer configuration
 * @param source Weight source for loading model weights
 * @param finalizer Weight finalizer for processing weights
 * @return Shared pointer to the preprocess OV model
 *
 * Model inputs:
 * - hidden_states: [B, C, F, H, W] float32
 * - timestep: [B] float32
 * - encoder_hidden_states: [B, seq, text_dim] float32
 *
 * Model outputs:
 * - tokens: [B, seq, inner_dim] - patched hidden states
 * - rotary_cos: [B, seq, head_dim/2] - RoPE cosine
 * - rotary_sin: [B, seq, head_dim/2] - RoPE sine
 * - temb: [B, inner_dim] - time embedding
 * - timestep_proj: [B, 6, inner_dim] - timestep projection
 * - text_embeds: [B, text_seq, inner_dim] - text embeddings
 * - ppf: [] int64 - patched frames dimension
 * - pph: [] int64 - patched height dimension
 * - ppw: [] int64 - patched width dimension
 */
std::shared_ptr<ov::Model> create_wan_dit_preprocess_model(
    const WanTransformer3DConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

/**
 * @brief Create a block group model for layered DIT.
 *
 * @param cfg Transformer configuration
 * @param start_layer Starting layer index (0-based)
 * @param num_layers Number of layers in this group
 * @param source Weight source for loading model weights
 * @param finalizer Weight finalizer for processing weights
 * @return Shared pointer to the block group OV model
 *
 * Model inputs:
 * - hidden_states: [B, seq, inner_dim] float32
 * - text_embeds: [B, text_seq, inner_dim] float32
 * - timestep_proj: [B, 6, inner_dim] float32
 * - rotary_cos: [B, seq, head_dim/2] float32
 * - rotary_sin: [B, seq, head_dim/2] float32
 *
 * Model outputs:
 * - hidden_states: [B, seq, inner_dim] - updated hidden states
 */
std::shared_ptr<ov::Model> create_wan_dit_block_group_model(
    const WanTransformer3DConfig& cfg,
    int32_t start_layer,
    int32_t num_layers,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

/**
 * @brief Create the postprocess model for layered DIT.
 *
 * @param cfg Transformer configuration
 * @param source Weight source for loading model weights
 * @param finalizer Weight finalizer for processing weights
 * @return Shared pointer to the postprocess OV model
 *
 * Model inputs:
 * - hidden_states: [B, seq, inner_dim] float32
 * - temb: [B, inner_dim] float32
 * - ppf: [] int64 - patched frames dimension
 * - pph: [] int64 - patched height dimension
 * - ppw: [] int64 - patched width dimension
 *
 * Model outputs:
 * - sample: [B, out_channels, F, H, W] - final output
 */
std::shared_ptr<ov::Model> create_wan_dit_postprocess_model(
    const WanTransformer3DConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

/**
 * @brief Create all layered DIT models.
 *
 * @param cfg Transformer configuration
 * @param layered_cfg Layered model configuration (layers per group)
 * @param source Weight source for loading model weights
 * @param finalizer Weight finalizer for processing weights
 * @return WanDitLayeredModels containing all sub-models
 */
WanDitLayeredModels create_wan_dit_layered_models(
    const WanTransformer3DConfig& cfg,
    const WanDitLayeredConfig& layered_cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
