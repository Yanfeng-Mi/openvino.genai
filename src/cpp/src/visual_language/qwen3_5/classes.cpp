// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/qwen3_5/classes.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

#include <openvino/core/except.hpp>

namespace ov::genai {

std::pair<ov::Tensor, int64_t> InputsEmbedderQwen3_5::create_position_ids(
    const ov::Tensor& input_ids_tensor,
    const std::vector<std::array<size_t, 3>>& images_grid_thw,
    const std::vector<size_t>& images_sequence,
    const size_t image_id,
    const std::vector<std::array<size_t, 3>>& videos_grid_thw,
    const std::vector<size_t>& videos_sequence,
    const size_t video_id,
    const int64_t vision_start_token_id,
    const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count) {
    const auto [vision_position_ids, rope_delta] = InputsEmbedderQwen2VL::create_position_ids(input_ids_tensor,
                                                                                              images_grid_thw,
                                                                                              images_sequence,
                                                                                              image_id,
                                                                                              videos_grid_thw,
                                                                                              videos_sequence,
                                                                                              video_id,
                                                                                              vision_start_token_id,
                                                                                              history_vision_count);
    const auto& vision_shape = vision_position_ids.get_shape();
    const size_t batch_size = vision_shape.at(1);
    const size_t seq_len = vision_shape.at(2);

    ov::Tensor position_ids{vision_position_ids.get_element_type(), {4, batch_size, seq_len}};
    int64_t* dst = position_ids.data<int64_t>();
    const int64_t* src = vision_position_ids.data<const int64_t>();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            dst[b * seq_len + s] = static_cast<int64_t>(s);
        }
    }
    std::memcpy(dst + batch_size * seq_len, src, 3 * batch_size * seq_len * sizeof(int64_t));

    return {position_ids, rope_delta};
}

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedderQwen3_5::get_generation_phase_position_ids(
    const size_t inputs_embeds_size,
    const size_t history_size,
    int64_t rope_delta) {
    const auto vision_position_ids = InputsEmbedderQwen2VL::get_generation_phase_position_ids(inputs_embeds_size,
                                                                                              history_size,
                                                                                              rope_delta).first;
    ov::Tensor position_ids{vision_position_ids.get_element_type(), {4, 1, inputs_embeds_size}};
    int64_t* dst = position_ids.data<int64_t>();
    const int64_t* src = vision_position_ids.data<const int64_t>();

    std::fill_n(dst, inputs_embeds_size, static_cast<int64_t>(history_size));
    std::memcpy(dst + inputs_embeds_size, src, 3 * inputs_embeds_size * sizeof(int64_t));

    return {position_ids, rope_delta};
}

ov::Tensor InputsEmbedderQwen3_5::get_rotary_pos_emb(const std::vector<std::array<size_t, 3>>& grids_thw) const {
    const size_t spatial_merge_size = m_vision_encoder->get_processor_config().merge_size;

    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(m_ireq_queue_vision_embeddings_merger.get());
    ov::InferRequest& vision_embeddings_merger = infer_request_guard.get();
    const size_t head_dim = vision_embeddings_merger.get_tensor("rotary_pos_emb").get_shape().at(1);
    OPENVINO_ASSERT(head_dim % 4 == 0, "Qwen3.5 vision rotary head dimension must be divisible by 4.");

    const size_t rotary_dim = head_dim / 2;
    const size_t inv_len = rotary_dim / 2;
    std::vector<float> inv_freq(inv_len);
    constexpr float theta = 10000.0f;
    for (size_t i = 0; i < inv_len; ++i) {
        inv_freq[i] = 1.0f / std::pow(theta, static_cast<float>(2 * i) / static_cast<float>(rotary_dim));
    }

    size_t total_positions = 0;
    for (const auto& grid_thw : grids_thw) {
        total_positions += grid_thw.at(0) * grid_thw.at(1) * grid_thw.at(2);
    }

    ov::Tensor rotary_pos_emb(ov::element::f32, {total_positions, head_dim});
    float* output_data = rotary_pos_emb.data<float>();
    size_t offset = 0;

    for (const auto& grid_thw : grids_thw) {
        const size_t grid_t = grid_thw.at(0);
        const size_t grid_h = grid_thw.at(1);
        const size_t grid_w = grid_thw.at(2);
        OPENVINO_ASSERT(grid_h % spatial_merge_size == 0 && grid_w % spatial_merge_size == 0,
                        "Qwen3.5 grid_thw must be divisible by spatial merge size.");

        for (size_t t = 0; t < grid_t; ++t) {
            (void)t;
            for (size_t block_h = 0; block_h < grid_h / spatial_merge_size; ++block_h) {
                for (size_t block_w = 0; block_w < grid_w / spatial_merge_size; ++block_w) {
                    for (size_t merge_h = 0; merge_h < spatial_merge_size; ++merge_h) {
                        for (size_t merge_w = 0; merge_w < spatial_merge_size; ++merge_w) {
                            const size_t row = block_h * spatial_merge_size + merge_h;
                            const size_t col = block_w * spatial_merge_size + merge_w;
                            float* row_out = output_data + offset * head_dim;
                            float* col_out = row_out + inv_len;
                            for (size_t i = 0; i < inv_len; ++i) {
                                row_out[i] = static_cast<float>(row) * inv_freq[i];
                                col_out[i] = static_cast<float>(col) * inv_freq[i];
                            }
                            std::copy_n(row_out, rotary_dim, row_out + rotary_dim);
                            ++offset;
                        }
                    }
                }
            }
        }
    }

    return rotary_pos_emb;
}

}  // namespace ov::genai
