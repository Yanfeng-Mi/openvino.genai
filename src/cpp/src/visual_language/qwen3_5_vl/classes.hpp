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

    const std::unordered_map<std::string, ov::Tensor>& get_lm_extra_inputs() const override {
        static const std::unordered_map<std::string, ov::Tensor> empty_map;
        return empty_map;
    }
};

}  // namespace ov::genai
