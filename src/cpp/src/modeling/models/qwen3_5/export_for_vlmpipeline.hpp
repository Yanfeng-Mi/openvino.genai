// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include <openvino/runtime/tensor.hpp>

namespace ov {
class Model;
}

namespace ov::genai::modeling::weights {
class WeightSource;
class WeightFinalizer;
}  // namespace ov::genai::modeling::weights

namespace ov::genai::modeling::models {

struct Qwen3_5Config;

/// VLMPipeline ModelsMap type alias for convenience.
using ModelsMap = std::map<std::string, std::pair<std::string, ov::Tensor>>;

/// Options for exporting Qwen3.5 models in VLMPipeline-compatible format.
struct Qwen3_5ExportOptions {
    /// Quantization mode for vision encoder weights ("NONE", "INT4", "INT8").
    std::string vision_quant_mode = "NONE";
    int vision_quant_group_size = 128;
    std::string vision_quant_backup_mode = "NONE";

    /// Quantization mode for text decoder weights ("NONE", "INT4", "INT8").
    std::string text_quant_mode = "NONE";
    int text_quant_group_size = 128;
    std::string text_quant_backup_mode = "NONE";
};

/// @brief Export a HuggingFace Qwen3.5 model as 5 VLMPipeline-compatible OV IR files.
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
///   - openvino_language_model.xml/bin (Decoder with hybrid attention + LMHead)
///   - config.json (VLMPipeline-compatible configuration)
///
/// @param model_dir  Path to HuggingFace model directory containing safetensors
///                   and config.json.
/// @param output_dir Path to output directory. Created if it doesn't exist.
/// @param options    Quantization and export options.
void export_qwen3_5_for_vlmpipeline(
    const std::filesystem::path& model_dir,
    const std::filesystem::path& output_dir,
    const Qwen3_5ExportOptions& options = {});

// ---- Individual model creation functions (advanced use) ----

/// Create the PatchEmbed-only model (vision_embeddings).
std::shared_ptr<ov::Model> create_qwen3_5_vision_embeddings_model(
    const Qwen3_5Config& cfg,
    weights::WeightSource& source,
    weights::WeightFinalizer& finalizer);

/// Create the Vision merger model (Blocks + Merger + DeepstackMergers).
std::shared_ptr<ov::Model> create_qwen3_5_vision_merger_model(
    const Qwen3_5Config& cfg,
    weights::WeightSource& source,
    weights::WeightFinalizer& finalizer);

/// Create the position embedding lookup model.
std::shared_ptr<ov::Model> create_qwen3_5_vision_pos_model(
    const Qwen3_5Config& cfg,
    weights::WeightSource& source);

/// Create the text embeddings model (VocabEmbedding only).
std::shared_ptr<ov::Model> create_qwen3_5_text_embeddings_model(
    const Qwen3_5Config& cfg,
    weights::WeightSource& source,
    weights::WeightFinalizer& finalizer);

/// Create the language model for VLMPipeline (hybrid attention Decoder + LMHead).
/// Phase 1: EmbeddingInjector for initial visual merge, no per-layer DeepStack.
std::shared_ptr<ov::Model> create_qwen3_5_language_model(
    const Qwen3_5Config& cfg,
    weights::WeightSource& source,
    weights::WeightFinalizer& finalizer);

// ---- In-memory integration with VLMPipeline ----

/// @brief Serialize an ov::Model to an in-memory ModelsMap entry.
///
/// Converts an ov::Model to the serialized XML string + weights Tensor pair
/// that VLMPipeline's ModelsMap constructor expects.
///
/// @param model The OV model to serialize.
/// @return A pair of {xml_string, weights_tensor} suitable for ModelsMap.
std::pair<std::string, ov::Tensor> serialize_model_to_memory(
    const std::shared_ptr<ov::Model>& model);

/// @brief Build all 5 Qwen3.5 sub-models and return them as a VLMPipeline ModelsMap.
///
/// This function constructs the 5 sub-models from HuggingFace safetensors weights
/// and serializes them in-memory, ready to pass directly to VLMPipeline(ModelsMap, ...).
///
/// The config.json and tokenizer files still need to be on disk at @p model_dir
/// (or a separate config_dir) for VLMPipeline's config_dir_path parameter.
///
/// Usage:
/// @code
///   auto models_map = build_qwen3_5_models_map(model_dir, options);
///   ov::genai::VLMPipeline pipe(models_map, tokenizer, model_dir, "GPU");
/// @endcode
///
/// @param model_dir  Path to HuggingFace model directory (safetensors + config.json).
/// @param options    Quantization and export options.
/// @return ModelsMap with keys: "vision_embeddings", "vision_embeddings_merger",
///         "vision_embeddings_pos", "text_embeddings", "language".
ModelsMap build_qwen3_5_models_map(
    const std::filesystem::path& model_dir,
    const Qwen3_5ExportOptions& options = {});

}  // namespace ov::genai::modeling::models
