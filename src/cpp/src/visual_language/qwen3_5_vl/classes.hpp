// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "visual_language/qwen3_vl/classes.hpp"

namespace ov::genai {

/// @brief VisionEncoder for Qwen3.5-VL. Reuses Qwen3-VL implementation
/// since both models share the same vision architecture.
class VisionEncoderQwen3_5VL : public VisionEncoderQwen3VL {
public:
    using VisionEncoderQwen3VL::VisionEncoderQwen3VL;
};

/// @brief InputsEmbedder for Qwen3.5-VL (Phase 1: No DeepStack in LM).
///
/// Inherits all Qwen3-VL vision encoding and token merging logic.
/// Overrides get_lm_extra_inputs() to return empty map since the Qwen3.5
/// language model does not use per-layer DeepStack injection.
class InputsEmbedderQwen3_5VL : public InputsEmbedderQwen3VL {
public:
    using InputsEmbedderQwen3VL::InputsEmbedderQwen3VL;

    ov::Tensor get_inputs_embeds(
        const std::string& prompt,
        const std::vector<ov::genai::EncodedImage>& images,
        const std::vector<ov::genai::EncodedVideo>& videos,
        ov::genai::VLMPerfMetrics& metrics,
        bool recalculate_merged_embeddings = true,
        const std::vector<size_t>& image_sequence = {},
        const std::vector<size_t>& videos_sequence = {},
        const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count = {}) override;

    const std::unordered_map<std::string, ov::Tensor>& get_lm_extra_inputs() const override {
        static const std::unordered_map<std::string, ov::Tensor> empty_map;
        return empty_map;
    }

protected:
    ov::Tensor get_rotary_pos_emb(const std::vector<std::array<size_t, 3>>& grids_thw) const override;
};

}  // namespace ov::genai
