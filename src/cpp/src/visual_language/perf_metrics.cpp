// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/visual_language/perf_metrics.hpp"

namespace ov::genai {
MeanStdPair calc_mean_and_std(const std::vector<MicroSeconds>& durations);

MeanStdPair VLMPerfMetrics::get_prepare_embeddings_duration() {
    evaluate_statistics();
    return prepare_embeddings_duration;
}

MeanStdPair VLMPerfMetrics::get_vision_encode_duration() {
    evaluate_statistics();
    return vision_encode_duration;
}

MeanStdPair VLMPerfMetrics::get_prompt_normalize_duration() {
    evaluate_statistics();
    return prompt_normalize_duration;
}

MeanStdPair VLMPerfMetrics::get_prefill_inference_duration() {
    evaluate_statistics();
    return prefill_inference_duration;
}

void VLMPerfMetrics::evaluate_statistics(std::optional<TimePoint> start_time) {
    if (m_evaluated) {
        return;
    }

    prepare_embeddings_duration = ov::genai::calc_mean_and_std(vlm_raw_metrics.prepare_embeddings_durations);
    vision_encode_duration = ov::genai::calc_mean_and_std(vlm_raw_metrics.vision_encode_durations);
    prompt_normalize_duration = ov::genai::calc_mean_and_std(vlm_raw_metrics.prompt_normalize_durations);
    prefill_inference_duration = ov::genai::calc_mean_and_std(vlm_raw_metrics.prefill_inference_durations);
    PerfMetrics::evaluate_statistics(start_time);
};

VLMPerfMetrics VLMPerfMetrics::operator+(const VLMPerfMetrics& right) const {
    PerfMetrics base_result = PerfMetrics::operator+(right);
    VLMPerfMetrics result{base_result};

    result.vlm_raw_metrics = vlm_raw_metrics;

    auto& result_prepare_embeddings_durations = result.vlm_raw_metrics.prepare_embeddings_durations;
    auto& right_prepare_embeddings_durations = right.vlm_raw_metrics.prepare_embeddings_durations;
    result_prepare_embeddings_durations.insert(result_prepare_embeddings_durations.end(),
                                                right_prepare_embeddings_durations.begin(),
                                                right_prepare_embeddings_durations.end());

    auto& result_vision_encode_durations = result.vlm_raw_metrics.vision_encode_durations;
    auto& right_vision_encode_durations = right.vlm_raw_metrics.vision_encode_durations;
    result_vision_encode_durations.insert(result_vision_encode_durations.end(),
                                          right_vision_encode_durations.begin(),
                                          right_vision_encode_durations.end());

    auto& result_prompt_normalize_durations = result.vlm_raw_metrics.prompt_normalize_durations;
    auto& right_prompt_normalize_durations = right.vlm_raw_metrics.prompt_normalize_durations;
    result_prompt_normalize_durations.insert(result_prompt_normalize_durations.end(),
                                             right_prompt_normalize_durations.begin(),
                                             right_prompt_normalize_durations.end());

    auto& result_prefill_inference_durations = result.vlm_raw_metrics.prefill_inference_durations;
    auto& right_prefill_inference_durations = right.vlm_raw_metrics.prefill_inference_durations;
    result_prefill_inference_durations.insert(result_prefill_inference_durations.end(),
                                              right_prefill_inference_durations.begin(),
                                              right_prefill_inference_durations.end());
    return result;
}
}
