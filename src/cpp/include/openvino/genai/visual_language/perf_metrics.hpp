// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/visibility.hpp"


namespace ov::genai {

struct OPENVINO_GENAI_EXPORTS VLMRawPerfMetrics {
    /** @brief Duration of preparation of embeddings */
    std::vector<MicroSeconds> prepare_embeddings_durations;
    /** @brief Duration of visual encoding before prompt assembly */
    std::vector<MicroSeconds> vision_encode_durations;
    /** @brief Duration of prompt normalization before multimodal assembly */
    std::vector<MicroSeconds> prompt_normalize_durations;
    /** @brief Duration of the language-model prefill infer request */
    std::vector<MicroSeconds> prefill_inference_durations;
};

struct OPENVINO_GENAI_EXPORTS VLMPerfMetrics : public PerfMetrics {
    /** @brief Mean and standard deviation of preparation of embeddings in milliseconds */
    MeanStdPair prepare_embeddings_duration;
    /** @brief Mean and standard deviation of visual encoding in milliseconds */
    MeanStdPair vision_encode_duration;
    /** @brief Mean and standard deviation of prompt normalization in milliseconds */
    MeanStdPair prompt_normalize_duration;
    /** @brief Mean and standard deviation of LM prefill inference in milliseconds */
    MeanStdPair prefill_inference_duration;

    MeanStdPair get_prepare_embeddings_duration();
    MeanStdPair get_vision_encode_duration();
    MeanStdPair get_prompt_normalize_duration();
    MeanStdPair get_prefill_inference_duration();

    VLMPerfMetrics() = default;

    VLMPerfMetrics(PerfMetrics& perf_metrics)
        : PerfMetrics(perf_metrics),
          prepare_embeddings_duration(),
          vision_encode_duration(),
          prompt_normalize_duration(),
          prefill_inference_duration() {};

    void evaluate_statistics(std::optional<TimePoint> start_time = std::nullopt) override;

    VLMPerfMetrics operator+(const VLMPerfMetrics& metrics) const;

    VLMRawPerfMetrics vlm_raw_metrics;
};

}
