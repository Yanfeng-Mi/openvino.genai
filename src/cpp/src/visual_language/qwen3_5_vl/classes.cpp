// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/qwen3_5_vl/classes.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <optional>
#include <vector>

#include <openvino/core/except.hpp>

#include "modeling/models/qwen3_5/processing_qwen3_5.hpp"

namespace ov::genai {

namespace {

ov::Tensor build_grid_thw_tensor(const std::vector<std::array<size_t, 3>>& grids_thw,
                                 const std::vector<size_t>& sequence) {
    std::vector<std::array<int64_t, 3>> ordered_grids;
    if (!sequence.empty()) {
        ordered_grids.reserve(sequence.size());
        for (size_t grid_id : sequence) {
            OPENVINO_ASSERT(grid_id < grids_thw.size(), "Grid index is out of range for Qwen3.5-VL position planning.");
            const auto& grid = grids_thw.at(grid_id);
            ordered_grids.push_back({static_cast<int64_t>(grid[0]),
                                     static_cast<int64_t>(grid[1]),
                                     static_cast<int64_t>(grid[2])});
        }
    } else {
        ordered_grids.reserve(grids_thw.size());
        for (const auto& grid : grids_thw) {
            ordered_grids.push_back({static_cast<int64_t>(grid[0]),
                                     static_cast<int64_t>(grid[1]),
                                     static_cast<int64_t>(grid[2])});
        }
    }

    ov::Tensor grid_tensor{ov::element::i64, {ordered_grids.size(), 3}};
    if (!ordered_grids.empty()) {
        std::memcpy(grid_tensor.data<int64_t>(),
                    ordered_grids.data(),
                    ordered_grids.size() * sizeof(ordered_grids.front()));
    }
    return grid_tensor;
}

ov::Tensor get_current_input_ids(utils::KVCacheState& kv_cache_state, size_t prev_hist_length) {
    const std::vector<int64_t>& tokenized_history = kv_cache_state.get_state();
    OPENVINO_ASSERT(prev_hist_length <= tokenized_history.size(),
                    "Current prompt offset exceeds stored token history for Qwen3.5-VL.");

    const size_t current_input_size = tokenized_history.size() - prev_hist_length;
    ov::Tensor input_ids{ov::element::i64, {1, current_input_size}};
    if (current_input_size != 0) {
        std::memcpy(input_ids.data<int64_t>(),
                    tokenized_history.data() + prev_hist_length,
                    current_input_size * sizeof(int64_t));
    }
    return input_ids;
}

}  // namespace

ov::Tensor InputsEmbedderQwen3_5VL::get_rotary_pos_emb(const std::vector<std::array<size_t, 3>>& grids_thw) const {
    const size_t spatial_merge_size = m_vision_encoder->get_processor_config().merge_size;

    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(m_ireq_queue_vision_embeddings_merger.get());
    ov::InferRequest& vision_embeddings_merger = infer_request_guard.get();
    const size_t head_dim = vision_embeddings_merger.get_tensor("rotary_pos_emb").get_shape().at(1);
    OPENVINO_ASSERT(head_dim % 4 == 0, "Qwen3.5-VL vision rotary head dimension must be divisible by 4.");

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
                        "Qwen3.5-VL grid_thw must be divisible by spatial merge size.");

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

ov::Tensor InputsEmbedderQwen3_5VL::get_inputs_embeds(
    const std::string& prompt,
    const std::vector<ov::genai::EncodedImage>& images,
    const std::vector<ov::genai::EncodedVideo>& videos,
    ov::genai::VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& image_sequence,
    const std::vector<size_t>& videos_sequence,
    const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count) {
    ov::Tensor inputs_embeds = InputsEmbedderQwen3VL::get_inputs_embeds(prompt,
                                                                        images,
                                                                        videos,
                                                                        metrics,
                                                                        recalculate_merged_embeddings,
                                                                        image_sequence,
                                                                        videos_sequence,
                                                                        history_vision_count);

    if (is_cdpruner_active() || !history_vision_count.empty()) {
        return inputs_embeds;
    }

    modeling::models::Qwen3_5Config planner_cfg;
    planner_cfg.image_token_id = static_cast<int32_t>(m_vision_token_ids.at("image_pad"));
    planner_cfg.video_token_id = static_cast<int32_t>(m_vision_token_ids.at("video_pad"));
    planner_cfg.vision_start_token_id = static_cast<int32_t>(m_vision_token_ids.at("vision_start"));
    planner_cfg.vision.spatial_merge_size = static_cast<int32_t>(m_vision_encoder->get_processor_config().merge_size);

    modeling::models::Qwen3_5InputPlanner input_planner(planner_cfg);

    std::vector<std::array<size_t, 3>> images_grid_thw;
    images_grid_thw.reserve(images.size());
    for (const auto& encoded_image : images) {
        images_grid_thw.push_back({1,
                                   encoded_image.resized_source_size.height,
                                   encoded_image.resized_source_size.width});
    }

    std::vector<std::array<size_t, 3>> videos_grid_thw;
    videos_grid_thw.reserve(videos.size());
    for (const auto& encoded_video : videos) {
        videos_grid_thw.push_back({encoded_video.frame_num,
                                   encoded_video.resized_source_size.height,
                                   encoded_video.resized_source_size.width});
    }

    const ov::Tensor input_ids = get_current_input_ids(m_kv_cache_state, m_prev_hist_length);
    const std::optional<ov::Tensor> image_grid_tensor = images.empty()
        ? std::nullopt
        : std::optional<ov::Tensor>(build_grid_thw_tensor(images_grid_thw, image_sequence));
    const std::optional<ov::Tensor> video_grid_tensor = videos.empty()
        ? std::nullopt
        : std::optional<ov::Tensor>(build_grid_thw_tensor(videos_grid_thw, videos_sequence));

    const auto input_plan = input_planner.build_plan(input_ids,
                                                     nullptr,
                                                     image_grid_tensor ? &*image_grid_tensor : nullptr,
                                                     video_grid_tensor ? &*video_grid_tensor : nullptr);

    m_position_ids = input_plan.position_ids;
    const int64_t* rope_delta_data = input_plan.rope_deltas.data<const int64_t>();
    m_rope_delta = input_plan.rope_deltas.get_size() == 0 ? 0 : rope_delta_data[0];

    return inputs_embeds;
}

}  // namespace ov::genai
