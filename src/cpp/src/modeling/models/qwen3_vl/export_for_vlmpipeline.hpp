// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>

namespace ov {
class Model;
}

namespace ov::genai::modeling::weights {
class WeightSource;
class WeightFinalizer;
}  // namespace ov::genai::modeling::weights

namespace ov::genai::modeling::models {

struct Qwen3VLConfig;

/// Options for exporting Qwen3-VL models in VLMPipeline-compatible format.
struct Qwen3VLExportOptions {
    /// Quantization mode for vision encoder weights ("NONE", "INT4", "INT8").
    std::string vision_quant_mode = "NONE";
    int vision_quant_group_size = 128;
    std::string vision_quant_backup_mode = "NONE";

    /// Quantization mode for text decoder weights ("NONE", "INT4", "INT8").
    std::string text_quant_mode = "NONE";
    int text_quant_group_size = 128;
    std::string text_quant_backup_mode = "NONE";
};

/// @brief Export a HuggingFace Qwen3-VL model as 5 VLMPipeline-compatible OV IR files.
///
/// Reads safetensors weights from @p model_dir, constructs 5 OpenVINO models
/// matching the VLMPipeline I/O contract, serializes them as .xml/.bin files
/// to @p output_dir, and generates the required config.json.
///
/// Output files:
///   - openvino_vision_embeddings_model.xml/bin  (PatchEmbed)
///   - openvino_vision_embeddings_merger_model.xml/bin (Blocks + Merger + Deepstack)
///   - openvino_vision_embeddings_pos_model.xml/bin (Position embedding lookup)
///   - openvino_text_embeddings_model.xml/bin (VocabEmbedding)
///   - openvino_language_model.xml/bin (Decoder + DeepstackInjector + LMHead)
///   - config.json (VLMPipeline-compatible configuration)
///
/// @param model_dir  Path to HuggingFace model directory containing safetensors
///                   and config.json.
/// @param output_dir Path to output directory. Created if it doesn't exist.
/// @param options    Quantization and export options.
void export_qwen3_vl_for_vlmpipeline(
    const std::filesystem::path& model_dir,
    const std::filesystem::path& output_dir,
    const Qwen3VLExportOptions& options = {});

// ---- Individual model creation functions (advanced use) ----

/// Create the PatchEmbed-only model (vision_embeddings).
/// Input: pixel_values [N, C*T*P*P] f32
/// Output: patch embeddings [N, hidden_size] f32
std::shared_ptr<ov::Model> create_qwen3_vl_vision_embeddings_model(
    const Qwen3VLConfig& cfg,
    weights::WeightSource& source,
    weights::WeightFinalizer& finalizer);

/// Create the Vision merger model (Blocks + Merger + DeepstackMergers).
/// Input: hidden_states [N, hidden_size] f32, rotary_pos_emb [N, head_dim] f32,
///        cu_seq_lens [num_segments + 1] i32
/// Output: last_hidden_state [N, out_hidden_size] f32,
///         deepstack_feature_lists [num_ds_layers, N, out_hidden_size] f32
std::shared_ptr<ov::Model> create_qwen3_vl_vision_merger_model(
    const Qwen3VLConfig& cfg,
    weights::WeightSource& source,
    weights::WeightFinalizer& finalizer);

/// Create the position embedding lookup model.
/// Input: input [4, N] i64 (bilinear interpolation corner indices)
/// Output: pos_embeds [4, N, embed_dim] f32
std::shared_ptr<ov::Model> create_qwen3_vl_vision_pos_model(
    const Qwen3VLConfig& cfg,
    weights::WeightSource& source);

/// Create the text embeddings model (VocabEmbedding only).
/// Input: input_ids [B, S] i64
/// Output: embeddings [B, S, hidden_size] f32
std::shared_ptr<ov::Model> create_qwen3_vl_text_embeddings_model(
    const Qwen3VLConfig& cfg,
    weights::WeightSource& source,
    weights::WeightFinalizer& finalizer);

/// Create the language model for VLMPipeline (Decoder + DeepstackInjector + LMHead).
/// Input: inputs_embeds [B, S, H] f32, position_ids [3, B, S] i64,
///        attention_mask [B, S] i64, beam_idx [B] i32,
///        deepstack_visual_embeds [num_ds, S, H] f32,
///        visual_pos_masks [B, S] boolean
/// Output: logits [B, S, V] f32
std::shared_ptr<ov::Model> create_qwen3_vl_language_model(
    const Qwen3VLConfig& cfg,
    weights::WeightSource& source,
    weights::WeightFinalizer& finalizer);

}  // namespace ov::genai::modeling::models
