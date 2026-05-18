// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/// @file export_qwen3_5.cpp
/// @brief Sample: Export a HuggingFace Qwen3.5-VL model as VLMPipeline-compatible
///        OpenVINO IR files.
///
/// Usage:
///   export_qwen3_5 <MODEL_DIR> <OUTPUT_DIR> [TEXT_QUANT] [TEXT_GS] [VISION_QUANT] [VISION_GS]
///
/// Arguments:
///   MODEL_DIR     Path to HuggingFace model directory (safetensors + config.json)
///   OUTPUT_DIR    Path to output directory for VLMPipeline IR files
///   TEXT_QUANT    Text model quantization: NONE | INT4 | INT8  (default: NONE)
///   TEXT_GS       Text quantization group size  (default: 128)
///   VISION_QUANT  Vision model quantization: NONE | INT4 | INT8  (default: NONE)
///   VISION_GS     Vision quantization group size  (default: 128)
///
/// Output:
///   openvino_vision_embeddings_model.xml/bin
///   openvino_vision_embeddings_merger_model.xml/bin
///   openvino_vision_embeddings_pos_model.xml/bin
///   openvino_text_embeddings_model.xml/bin
///   openvino_language_model.xml/bin
///   config.json, preprocessor_config.json, tokenizer files
///
/// After export, load with VLMPipeline:
///   ov::genai::VLMPipeline pipe(output_dir, "GPU");
///   auto result = pipe.generate(prompt, images);

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include "modeling/models/qwen3_5/export_for_vlmpipeline.hpp"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <MODEL_DIR> <OUTPUT_DIR>"
                  << " [TEXT_QUANT] [TEXT_GS] [VISION_QUANT] [VISION_GS]\n"
                  << "\n"
                  << "  MODEL_DIR     HuggingFace model directory with safetensors\n"
                  << "  OUTPUT_DIR    Output directory for VLMPipeline-compatible IRs\n"
                  << "  TEXT_QUANT    NONE | INT4 | INT8 (default: NONE)\n"
                  << "  TEXT_GS       Text quantization group size (default: 128)\n"
                  << "  VISION_QUANT  NONE | INT4 | INT8 (default: NONE)\n"
                  << "  VISION_GS     Vision quantization group size (default: 128)\n";
        return 1;
    }

    const std::filesystem::path model_dir = argv[1];
    const std::filesystem::path output_dir = argv[2];

    ov::genai::modeling::models::Qwen3_5ExportOptions options;

    if (argc > 3) options.text_quant_mode = argv[3];
    if (argc > 4) options.text_quant_group_size = std::atoi(argv[4]);
    if (argc > 5) options.vision_quant_mode = argv[5];
    if (argc > 6) options.vision_quant_group_size = std::atoi(argv[6]);

    std::cout << "=== Qwen3.5-VL VLMPipeline Export (Phase 1: No DeepStack) ===\n"
              << "  Model:   " << model_dir << "\n"
              << "  Output:  " << output_dir << "\n"
              << "  Text Q:  " << options.text_quant_mode
              << " (gs=" << options.text_quant_group_size << ")\n"
              << "  Vision Q: " << options.vision_quant_mode
              << " (gs=" << options.vision_quant_group_size << ")\n"
              << std::endl;

    const auto start = std::chrono::steady_clock::now();

    ov::genai::modeling::models::export_qwen3_5_for_vlmpipeline(
        model_dir, output_dir, options);

    const auto end = std::chrono::steady_clock::now();
    const double elapsed_s = std::chrono::duration<double>(end - start).count();

    std::cout << "\nExport completed in " << elapsed_s << " seconds.\n"
              << "Output directory: " << output_dir << "\n"
              << "\nTo run with VLMPipeline:\n"
              << "  ov::genai::VLMPipeline pipe(\"" << output_dir.string()
              << "\", \"GPU\");\n"
              << "  auto result = pipe.generate(prompt, images);\n";

    return 0;
}
