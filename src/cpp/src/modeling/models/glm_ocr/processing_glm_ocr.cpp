// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/glm_ocr/processing_glm_ocr.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

#include <openvino/core/except.hpp>
#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/float16.hpp>

#include "json_utils.hpp"

namespace {

void read_config_json_file(const std::filesystem::path& path, nlohmann::json& data) {
    std::ifstream file(path);
    if (!file.is_open()) {
        OPENVINO_THROW("Failed to open config file: ", path.string());
    }
    file >> data;
}

void read_preprocess_json_file(const std::filesystem::path& path, nlohmann::json& data) {
    std::ifstream file(path);
    if (!file.is_open()) {
        OPENVINO_THROW("Failed to open preprocessor config: ", path.string());
    }
    file >> data;
}

std::filesystem::path resolve_config_path(const std::filesystem::path& path) {
    if (std::filesystem::is_directory(path)) {
        return path / "config.json";
    }
    return path;
}

void parse_rope_config(const nlohmann::json& data, ov::genai::modeling::models::GlmOcrRopeConfig& cfg) {
    using ov::genai::utils::read_json_param;
    read_json_param(data, "mrope_section", cfg.mrope_section);
    read_json_param(data, "rope_type", cfg.rope_type);
    read_json_param(data, "rope_theta", cfg.rope_theta);
}

void parse_text_config(const nlohmann::json& data, ov::genai::modeling::models::GlmOcrTextConfig& cfg) {
    using ov::genai::utils::read_json_param;
    read_json_param(data, "model_type", cfg.model_type);
    read_json_param(data, "vocab_size", cfg.vocab_size);
    read_json_param(data, "hidden_size", cfg.hidden_size);
    read_json_param(data, "intermediate_size", cfg.intermediate_size);
    read_json_param(data, "num_hidden_layers", cfg.num_hidden_layers);
    read_json_param(data, "num_attention_heads", cfg.num_attention_heads);
    read_json_param(data, "num_key_value_heads", cfg.num_key_value_heads);
    read_json_param(data, "head_dim", cfg.head_dim);
    read_json_param(data, "max_position_embeddings", cfg.max_position_embeddings);
    read_json_param(data, "rms_norm_eps", cfg.rms_norm_eps);
    read_json_param(data, "rope_theta", cfg.rope_theta);
    read_json_param(data, "hidden_act", cfg.hidden_act);
    read_json_param(data, "attention_bias", cfg.attention_bias);
    read_json_param(data, "tie_word_embeddings", cfg.tie_word_embeddings);
    read_json_param(data, "dtype", cfg.dtype);

    if (data.contains("rope_parameters")) {
        parse_rope_config(data.at("rope_parameters"), cfg.rope);
    }
    if (data.contains("rope_scaling")) {
        parse_rope_config(data.at("rope_scaling"), cfg.rope);
    }

    cfg.finalize();
}

void parse_vision_config(const nlohmann::json& data, ov::genai::modeling::models::GlmOcrVisionConfig& cfg) {
    using ov::genai::utils::read_json_param;
    read_json_param(data, "model_type", cfg.model_type);
    read_json_param(data, "depth", cfg.depth);
    read_json_param(data, "hidden_size", cfg.hidden_size);
    read_json_param(data, "hidden_act", cfg.hidden_act);
    read_json_param(data, "intermediate_size", cfg.intermediate_size);
    read_json_param(data, "num_heads", cfg.num_heads);
    read_json_param(data, "in_channels", cfg.in_channels);
    read_json_param(data, "patch_size", cfg.patch_size);
    read_json_param(data, "spatial_merge_size", cfg.spatial_merge_size);
    read_json_param(data, "temporal_patch_size", cfg.temporal_patch_size);
    read_json_param(data, "out_hidden_size", cfg.out_hidden_size);
    read_json_param(data, "rms_norm_eps", cfg.rms_norm_eps);
    read_json_param(data, "attention_bias", cfg.attention_bias);

    cfg.finalize();
}

template <typename T>
bool mask_value_at(const ov::Tensor& mask, size_t index) {
    const T* data = mask.data<const T>();
    return data[index] != static_cast<T>(0);
}

bool mask_value(const ov::Tensor& mask, size_t index) {
    switch (mask.get_element_type()) {
        case ov::element::boolean:
            return mask_value_at<char>(mask, index);
        case ov::element::i32:
            return mask_value_at<int32_t>(mask, index);
        case ov::element::i64:
            return mask_value_at<int64_t>(mask, index);
        case ov::element::u8:
            return mask_value_at<uint8_t>(mask, index);
        default:
            OPENVINO_THROW("Unsupported attention_mask dtype");
    }
}

void set_bool(ov::Tensor& mask, size_t index, bool value) {
    auto* data = mask.data<char>();
    data[index] = value ? 1 : 0;
}

std::pair<size_t, size_t> smart_resize(size_t height,
                                       size_t width,
                                       size_t factor,
                                       size_t min_pixels,
                                       size_t max_pixels) {
    if (height < factor || width < factor) {
        OPENVINO_THROW("Height and width must be >= resize factor");
    }
    if (std::max(height, width) / std::min(height, width) > 200) {
        OPENVINO_THROW("Absolute aspect ratio must be smaller than 200");
    }

    auto round_to_factor = [factor](double value) {
        return static_cast<size_t>(std::round(value / static_cast<double>(factor)) * factor);
    };

    size_t h_bar = round_to_factor(static_cast<double>(height));
    size_t w_bar = round_to_factor(static_cast<double>(width));

    const double pixels = static_cast<double>(height) * static_cast<double>(width);
    if (static_cast<double>(h_bar) * static_cast<double>(w_bar) > static_cast<double>(max_pixels)) {
        double beta = std::sqrt(pixels / static_cast<double>(max_pixels));
        h_bar = std::max(factor, static_cast<size_t>(std::floor(height / beta / factor) * factor));
        w_bar = std::max(factor, static_cast<size_t>(std::floor(width / beta / factor) * factor));
    } else if (static_cast<double>(h_bar) * static_cast<double>(w_bar) < static_cast<double>(min_pixels)) {
        double beta = std::sqrt(static_cast<double>(min_pixels) / pixels);
        h_bar = static_cast<size_t>(std::ceil(height * beta / factor) * factor);
        w_bar = static_cast<size_t>(std::ceil(width * beta / factor) * factor);
    }

    return {h_bar, w_bar};
}

// Catmull-Rom (Keys a=-0.5) cubic kernel, matching PIL/Pillow BICUBIC
float cubic_kernel(float x) {
    float abs_x = std::abs(x);
    if (abs_x <= 1.0f) {
        return (1.5f * abs_x - 2.5f) * abs_x * abs_x + 1.0f;
    } else if (abs_x < 2.0f) {
        return ((-0.5f * abs_x + 2.5f) * abs_x - 4.0f) * abs_x + 2.0f;
    }
    return 0.0f;
}

// Bicubic resize with antialiasing (matching PIL/Pillow BICUBIC behavior).
// When downscaling, the kernel is widened by the scale factor to avoid aliasing.
// Output is in CHW float format with rescale (÷255) and normalize (mean/std).
void resize_bicubic_to_chw(const uint8_t* src,
                            size_t src_h,
                            size_t src_w,
                            size_t channels,
                            bool nchw,
                            size_t dst_h,
                            size_t dst_w,
                            const std::array<float, 3>& mean,
                            const std::array<float, 3>& std,
                            std::vector<float>& dst_chw) {
    dst_chw.assign(channels * dst_h * dst_w, 0.0f);
    const float scale_y = static_cast<float>(src_h) / static_cast<float>(dst_h);
    const float scale_x = static_cast<float>(src_w) / static_cast<float>(dst_w);
    // Antialiasing: widen kernel when downscaling (scale > 1)
    const float filter_scale_y = std::max(1.0f, scale_y);
    const float filter_scale_x = std::max(1.0f, scale_x);
    // Bicubic support radius is 2; scaled by filter factor for antialiasing
    const float support_y = 2.0f * filter_scale_y;
    const float support_x = 2.0f * filter_scale_x;

    auto fetch = [&](int y, int x, size_t c) -> float {
        y = std::max(0, std::min(y, static_cast<int>(src_h) - 1));
        x = std::max(0, std::min(x, static_cast<int>(src_w) - 1));
        size_t idx = 0;
        if (nchw) {
            idx = (c * src_h + static_cast<size_t>(y)) * src_w + static_cast<size_t>(x);
        } else {
            idx = (static_cast<size_t>(y) * src_w + static_cast<size_t>(x)) * channels + c;
        }
        return static_cast<float>(src[idx]);
    };

    for (size_t y = 0; y < dst_h; ++y) {
        float center_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        int y_start = static_cast<int>(std::floor(center_y - support_y + 0.5f));
        int y_end = static_cast<int>(std::floor(center_y + support_y + 0.5f));
        for (size_t x = 0; x < dst_w; ++x) {
            float center_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            int x_start = static_cast<int>(std::floor(center_x - support_x + 0.5f));
            int x_end = static_cast<int>(std::floor(center_x + support_x + 0.5f));
            for (size_t c = 0; c < channels; ++c) {
                float sum = 0.0f;
                float weight_sum = 0.0f;
                for (int iy = y_start; iy <= y_end; ++iy) {
                    float wy = cubic_kernel((static_cast<float>(iy) - center_y) / filter_scale_y);
                    for (int ix = x_start; ix <= x_end; ++ix) {
                        float wx = cubic_kernel((static_cast<float>(ix) - center_x) / filter_scale_x);
                        float w = wy * wx;
                        sum += w * fetch(iy, ix, c);
                        weight_sum += w;
                    }
                }
                float val = weight_sum > 0.0f ? sum / weight_sum : 0.0f;
                // Clamp to [0, 255] like PIL does (bicubic can overshoot)
                val = std::max(0.0f, std::min(val, 255.0f));
                val = val / 255.0f;
                val = (val - mean[c]) / std[c];
                dst_chw[(c * dst_h + y) * dst_w + x] = val;
            }
        }
    }
}

struct PreparedImage {
    std::vector<float> data;
    size_t frames = 0;
    size_t height = 0;
    size_t width = 0;
    int64_t grid_t = 0;
    int64_t grid_h = 0;
    int64_t grid_w = 0;
};

std::pair<ov::Tensor, ov::Tensor> build_rotary_cos_sin(
    const ov::Tensor& grid_thw,
    const ov::genai::modeling::models::GlmOcrVisionConfig& cfg,
    int32_t merge_size) {
    if (grid_thw.get_element_type() != ov::element::i64) {
        OPENVINO_THROW("grid_thw must be i64");
    }
    const auto grid_shape = grid_thw.get_shape();
    if (grid_shape.size() != 2 || grid_shape[1] != 3) {
        OPENVINO_THROW("grid_thw must have shape [N, 3]");
    }
    const int64_t* grid = grid_thw.data<const int64_t>();
    const size_t num_images = grid_shape[0];

    int64_t total_tokens = 0;
    for (size_t i = 0; i < num_images; ++i) {
        const int64_t t = grid[i * 3 + 0];
        const int64_t h = grid[i * 3 + 1];
        const int64_t w = grid[i * 3 + 2];
        total_tokens += t * h * w;
    }

    const int32_t head_dim = cfg.head_dim();
    if (head_dim <= 0 || head_dim % 2 != 0) {
        OPENVINO_THROW("Invalid head_dim for GlmOcr rotary embedding");
    }
    const int32_t rotary_dim = head_dim / 2;
    if (rotary_dim % 2 != 0) {
        OPENVINO_THROW("Vision rotary_dim must be even");
    }
    const int32_t inv_len = rotary_dim / 2;
    const float theta = 10000.0f;

    std::vector<float> inv_freq(static_cast<size_t>(inv_len));
    for (int32_t i = 0; i < inv_len; ++i) {
        float exponent = static_cast<float>(2 * i) / static_cast<float>(rotary_dim);
        inv_freq[static_cast<size_t>(i)] = 1.0f / std::pow(theta, exponent);
    }

    ov::Tensor rotary_cos(ov::element::f32, {static_cast<size_t>(total_tokens), static_cast<size_t>(head_dim)});
    ov::Tensor rotary_sin(ov::element::f32, {static_cast<size_t>(total_tokens), static_cast<size_t>(head_dim)});
    float* cos_out = rotary_cos.data<float>();
    float* sin_out = rotary_sin.data<float>();

    size_t offset = 0;
    for (size_t i = 0; i < num_images; ++i) {
        const int64_t t = grid[i * 3 + 0];
        const int64_t h = grid[i * 3 + 1];
        const int64_t w = grid[i * 3 + 2];
        if (t <= 0 || h <= 0 || w <= 0) {
            OPENVINO_THROW("Invalid grid_thw for rotary embedding");
        }
        if (h % merge_size != 0 || w % merge_size != 0) {
            OPENVINO_THROW("grid_thw must be divisible by merge_size");
        }
        const int64_t merged_h = h / merge_size;
        const int64_t merged_w = w / merge_size;

        for (int64_t tt = 0; tt < t; ++tt) {
            (void)tt;
            for (int64_t bh = 0; bh < merged_h; ++bh) {
                for (int64_t bw = 0; bw < merged_w; ++bw) {
                    for (int64_t mh = 0; mh < merge_size; ++mh) {
                        for (int64_t mw = 0; mw < merge_size; ++mw) {
                            const int64_t row = bh * merge_size + mh;
                            const int64_t col = bw * merge_size + mw;
                            float* cos_ptr = cos_out + offset * static_cast<size_t>(head_dim);
                            float* sin_ptr = sin_out + offset * static_cast<size_t>(head_dim);

                            for (int32_t j = 0; j < inv_len; ++j) {
                                const float inv = inv_freq[static_cast<size_t>(j)];
                                const float row_freq = static_cast<float>(row) * inv;
                                const float col_freq = static_cast<float>(col) * inv;
                                const float cos_row = std::cos(row_freq);
                                const float sin_row = std::sin(row_freq);
                                const float cos_col = std::cos(col_freq);
                                const float sin_col = std::sin(col_freq);

                                const int32_t row_idx = j;
                                const int32_t col_idx = inv_len + j;

                                cos_ptr[row_idx] = cos_row;
                                sin_ptr[row_idx] = sin_row;
                                cos_ptr[col_idx] = cos_col;
                                sin_ptr[col_idx] = sin_col;
                            }
                            for (int32_t j = 0; j < rotary_dim; ++j) {
                                cos_ptr[rotary_dim + j] = cos_ptr[j];
                                sin_ptr[rotary_dim + j] = sin_ptr[j];
                            }

                            offset++;
                        }
                    }
                }
            }
        }
    }

    return {rotary_cos, rotary_sin};
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

int32_t GlmOcrTextConfig::kv_heads() const {
    return num_key_value_heads > 0 ? num_key_value_heads : num_attention_heads;
}

int32_t GlmOcrTextConfig::resolved_head_dim() const {
    if (head_dim > 0) {
        return head_dim;
    }
    if (hidden_size > 0 && num_attention_heads > 0) {
        return hidden_size / num_attention_heads;
    }
    return 0;
}

void GlmOcrTextConfig::finalize() {
    if (num_key_value_heads <= 0) {
        num_key_value_heads = num_attention_heads;
    }
    if (head_dim <= 0) {
        head_dim = resolved_head_dim();
    }
    if (rope.mrope_section.empty()) {
        rope.mrope_section = {16, 24, 24};
    }
    if (rope.rope_theta > 0 && rope_theta <= 0) {
        rope_theta = rope.rope_theta;
    }
    if (rope_theta > 0 && rope.rope_theta <= 0) {
        rope.rope_theta = rope_theta;
    }
}

void GlmOcrTextConfig::validate() const {
    if (hidden_size <= 0) {
        OPENVINO_THROW("GlmOcrTextConfig.hidden_size must be > 0");
    }
    if (num_hidden_layers <= 0) {
        OPENVINO_THROW("GlmOcrTextConfig.num_hidden_layers must be > 0");
    }
    if (num_attention_heads <= 0) {
        OPENVINO_THROW("GlmOcrTextConfig.num_attention_heads must be > 0");
    }
    if (kv_heads() <= 0) {
        OPENVINO_THROW("GlmOcrTextConfig.num_key_value_heads must be > 0");
    }
    if (num_attention_heads % kv_heads() != 0) {
        OPENVINO_THROW("GlmOcrTextConfig.num_attention_heads must be divisible by num_key_value_heads");
    }
    if (resolved_head_dim() <= 0) {
        OPENVINO_THROW("GlmOcrTextConfig.head_dim must be > 0");
    }
    if (rope.mrope_section.size() != 3) {
        OPENVINO_THROW("GlmOcrTextConfig.mrope_section must have 3 elements");
    }
}

int32_t GlmOcrVisionConfig::head_dim() const {
    if (num_heads <= 0) {
        return 0;
    }
    return hidden_size / num_heads;
}

void GlmOcrVisionConfig::finalize() {
    if (out_hidden_size <= 0) {
        out_hidden_size = hidden_size;
    }
}

void GlmOcrVisionConfig::validate() const {
    if (depth <= 0) {
        OPENVINO_THROW("GlmOcrVisionConfig.depth must be > 0");
    }
    if (hidden_size <= 0) {
        OPENVINO_THROW("GlmOcrVisionConfig.hidden_size must be > 0");
    }
    if (num_heads <= 0) {
        OPENVINO_THROW("GlmOcrVisionConfig.num_heads must be > 0");
    }
    if (hidden_size % num_heads != 0) {
        OPENVINO_THROW("GlmOcrVisionConfig.hidden_size must be divisible by num_heads");
    }
    if (patch_size <= 0 || spatial_merge_size <= 0 || temporal_patch_size <= 0) {
        OPENVINO_THROW("GlmOcrVisionConfig patch/merge sizes must be > 0");
    }
    if (out_hidden_size <= 0) {
        OPENVINO_THROW("GlmOcrVisionConfig.out_hidden_size must be > 0");
    }
}

void GlmOcrConfig::finalize() {
    if (model_type.empty()) {
        model_type = "glm_ocr";
    }
    text.finalize();
    vision.finalize();
    if (text.tie_word_embeddings) {
        tie_word_embeddings = true;
    }
    if (tie_word_embeddings) {
        text.tie_word_embeddings = true;
    }
}

void GlmOcrConfig::validate() const {
    if (model_type != "glm_ocr") {
        OPENVINO_THROW("Unsupported model_type: ", model_type);
    }
    text.validate();
    vision.validate();
    if (image_token_id < 0) {
        OPENVINO_THROW("Invalid token ids in GlmOcrConfig");
    }
}

GlmOcrConfig GlmOcrConfig::from_json(const nlohmann::json& data) {
    using ov::genai::utils::read_json_param;
    GlmOcrConfig cfg;
    read_json_param(data, "model_type", cfg.model_type);
    read_json_param(data, "architectures", cfg.architectures);
    read_json_param(data, "image_token_id", cfg.image_token_id);
    read_json_param(data, "image_start_token_id", cfg.image_start_token_id);
    read_json_param(data, "image_end_token_id", cfg.image_end_token_id);
    read_json_param(data, "tie_word_embeddings", cfg.tie_word_embeddings);

    // Parse eos_token_ids from text_config
    if (data.contains("text_config")) {
        const auto& tc = data.at("text_config");
        if (tc.contains("eos_token_id") && tc["eos_token_id"].is_array()) {
            cfg.eos_token_ids.clear();
            for (const auto& e : tc["eos_token_id"]) {
                cfg.eos_token_ids.push_back(e.get<int32_t>());
            }
        }
        parse_text_config(tc, cfg.text);
    } else {
        OPENVINO_THROW("GlmOcrConfig is missing text_config");
    }

    if (data.contains("vision_config")) {
        parse_vision_config(data.at("vision_config"), cfg.vision);
    } else {
        OPENVINO_THROW("GlmOcrConfig is missing vision_config");
    }

    cfg.finalize();
    cfg.validate();
    return cfg;
}

GlmOcrConfig GlmOcrConfig::from_json_file(const std::filesystem::path& config_path) {
    auto resolved = resolve_config_path(config_path);
    if (!std::filesystem::exists(resolved)) {
        OPENVINO_THROW("Config file not found: ", resolved.string());
    }
    nlohmann::json data;
    read_config_json_file(resolved, data);
    return from_json(data);
}

std::string GlmOcrModuleNames::vision_block(int32_t index) {
    return std::string("blocks.") + std::to_string(index);
}

std::string GlmOcrModuleNames::text_layer(int32_t index) {
    return std::string("layers.") + std::to_string(index);
}

GlmOcrInputPlanner::GlmOcrInputPlanner(const GlmOcrConfig& cfg)
    : image_token_id_(cfg.image_token_id),
      spatial_merge_size_(cfg.vision.spatial_merge_size) {}

ov::Tensor GlmOcrInputPlanner::build_visual_pos_mask(const ov::Tensor& input_ids,
                                                     const ov::Tensor* attention_mask) const {
    if (input_ids.get_element_type() != ov::element::i64) {
        OPENVINO_THROW("input_ids must be i64 for GlmOcrInputPlanner");
    }
    const auto shape = input_ids.get_shape();
    if (shape.size() != 2) {
        OPENVINO_THROW("input_ids must have shape [B, S]");
    }
    if (attention_mask && attention_mask->get_shape() != shape) {
        OPENVINO_THROW("attention_mask must have the same shape as input_ids");
    }
    ov::Tensor mask(ov::element::boolean, shape);
    const int64_t* ids = input_ids.data<const int64_t>();
    const size_t total = input_ids.get_size();

    for (size_t idx = 0; idx < total; ++idx) {
        bool active = ids[idx] == image_token_id_;
        if (attention_mask && !mask_value(*attention_mask, idx)) {
            active = false;
        }
        set_bool(mask, idx, active);
    }
    return mask;
}

GlmOcrInputPlan GlmOcrInputPlanner::build_plan(const ov::Tensor& input_ids,
                                                const ov::Tensor* attention_mask,
                                                const ov::Tensor* image_grid_thw) const {
    if (input_ids.get_element_type() != ov::element::i64) {
        OPENVINO_THROW("input_ids must be i64 for GlmOcrInputPlanner");
    }
    const auto shape = input_ids.get_shape();
    if (shape.size() != 2) {
        OPENVINO_THROW("input_ids must have shape [B, S]");
    }
    if (attention_mask && attention_mask->get_shape() != shape) {
        OPENVINO_THROW("attention_mask must have the same shape as input_ids");
    }
    if (image_grid_thw) {
        const auto grid_shape = image_grid_thw->get_shape();
        if (image_grid_thw->get_element_type() != ov::element::i64) {
            OPENVINO_THROW("image_grid_thw must be i64");
        }
        if (grid_shape.size() != 2 || grid_shape[1] != 3) {
            OPENVINO_THROW("image_grid_thw must have shape [N, 3]");
        }
    }
    if (spatial_merge_size_ <= 0) {
        OPENVINO_THROW("spatial_merge_size must be > 0");
    }
    const size_t batch = shape[0];
    const size_t seq_len = shape[1];

    ov::Tensor position_ids(ov::element::i64, {3, batch, seq_len});
    std::memset(position_ids.data(), 0, position_ids.get_byte_size());

    ov::Tensor rope_deltas(ov::element::i64, {batch, 1});
    std::memset(rope_deltas.data(), 0, rope_deltas.get_byte_size());

    auto visual_pos_mask = build_visual_pos_mask(input_ids, attention_mask);

    const int64_t* ids = input_ids.data<const int64_t>();
    const int64_t* grid = image_grid_thw ? image_grid_thw->data<const int64_t>() : nullptr;
    const size_t grid_rows = image_grid_thw ? image_grid_thw->get_shape().at(0) : 0;
    size_t grid_index = 0;

    auto pos_data = position_ids.data<int64_t>();
    auto delta_data = rope_deltas.data<int64_t>();

    for (size_t b = 0; b < batch; ++b) {
        std::vector<int64_t> tokens;
        std::vector<size_t> active_indices;
        tokens.reserve(seq_len);
        active_indices.reserve(seq_len);

        for (size_t s = 0; s < seq_len; ++s) {
            const size_t idx = b * seq_len + s;
            if (attention_mask && !mask_value(*attention_mask, idx)) {
                continue;
            }
            tokens.push_back(ids[idx]);
            active_indices.push_back(s);
        }

        if (tokens.empty()) {
            delta_data[b] = 0;
            continue;
        }

        std::vector<int64_t> pos_t;
        std::vector<int64_t> pos_h;
        std::vector<int64_t> pos_w;
        pos_t.reserve(tokens.size());
        pos_h.reserve(tokens.size());
        pos_w.reserve(tokens.size());

        int64_t last_max = -1;
        size_t st = 0;
        size_t local_grid_index = grid_index;

        auto append_text = [&](size_t length) {
            if (length == 0) {
                return;
            }
            const int64_t base = last_max + 1;
            for (size_t i = 0; i < length; ++i) {
                const int64_t value = base + static_cast<int64_t>(i);
                pos_t.push_back(value);
                pos_h.push_back(value);
                pos_w.push_back(value);
            }
            last_max = base + static_cast<int64_t>(length) - 1;
        };

        auto append_visual = [&](int64_t t, int64_t h, int64_t w) {
            if (t <= 0 || h <= 0 || w <= 0) {
                OPENVINO_THROW("Invalid grid_thw values in GlmOcrInputPlanner");
            }
            const int64_t llm_grid_t = t;
            const int64_t llm_grid_h = h / spatial_merge_size_;
            const int64_t llm_grid_w = w / spatial_merge_size_;
            if (llm_grid_h <= 0 || llm_grid_w <= 0) {
                OPENVINO_THROW("Invalid spatial_merge_size for grid_thw");
            }
            const int64_t base = last_max + 1;
            int64_t max_dim = 0;
            for (int64_t tt = 0; tt < llm_grid_t; ++tt) {
                for (int64_t hh = 0; hh < llm_grid_h; ++hh) {
                    for (int64_t ww = 0; ww < llm_grid_w; ++ww) {
                        pos_t.push_back(base + tt);
                        pos_h.push_back(base + hh);
                        pos_w.push_back(base + ww);
                        max_dim = std::max(max_dim, std::max(tt, std::max(hh, ww)));
                    }
                }
            }
            last_max = base + max_dim;
        };

        if (image_grid_thw) {
            while (true) {
                auto start_it = tokens.begin() + static_cast<std::vector<int64_t>::difference_type>(st);
                auto it = std::find(start_it, tokens.end(), image_token_id_);
                if (it == tokens.end()) {
                    break;
                }
                const size_t ed = static_cast<size_t>(std::distance(tokens.begin(), it));
                if (local_grid_index >= grid_rows) {
                    OPENVINO_THROW("image_grid_thw entries are fewer than image tokens");
                }
                append_text(ed - st);
                const int64_t t = grid[local_grid_index * 3 + 0];
                const int64_t h = grid[local_grid_index * 3 + 1];
                const int64_t w = grid[local_grid_index * 3 + 2];
                append_visual(t, h, w);

                const int64_t llm_grid_h = h / spatial_merge_size_;
                const int64_t llm_grid_w = w / spatial_merge_size_;
                const int64_t visual_len = t * llm_grid_h * llm_grid_w;
                if (ed + static_cast<size_t>(visual_len) > tokens.size()) {
                    OPENVINO_THROW("Image tokens length does not match grid_thw");
                }
                st = ed + static_cast<size_t>(visual_len);
                local_grid_index += 1;
            }
        }

        if (st < tokens.size()) {
            append_text(tokens.size() - st);
        }

        if (pos_t.size() != tokens.size()) {
            OPENVINO_THROW("Position ids length mismatch");
        }

        int64_t max_pos = pos_t.empty() ? 0 : pos_t.front();
        for (size_t i = 0; i < pos_t.size(); ++i) {
            max_pos = std::max(max_pos, pos_t[i]);
            max_pos = std::max(max_pos, pos_h[i]);
            max_pos = std::max(max_pos, pos_w[i]);
        }

        for (size_t i = 0; i < tokens.size(); ++i) {
            const size_t s = active_indices[i];
            const size_t base = b * seq_len + s;
            pos_data[0 * batch * seq_len + base] = pos_t[i];
            pos_data[1 * batch * seq_len + base] = pos_h[i];
            pos_data[2 * batch * seq_len + base] = pos_w[i];
        }

        // Masked/padding positions stay at 0 (matching PyTorch torch.zeros init)
        // position_ids was already memset to 0, so no action needed for masked positions

        // Use active token count (not raw seq_len which includes padding)
        // to match PyTorch: max_pos + 1 - len(current_input_ids)
        delta_data[b] = max_pos + 1 - static_cast<int64_t>(tokens.size());
        grid_index = local_grid_index;
    }

    return {position_ids, visual_pos_mask, rope_deltas};
}

ov::Tensor GlmOcrInputPlanner::scatter_visual_embeds(const ov::Tensor& visual_embeds,
                                                     const ov::Tensor& visual_pos_mask) {
    const auto mask_shape = visual_pos_mask.get_shape();
    if (mask_shape.size() != 2) {
        OPENVINO_THROW("visual_pos_mask must have shape [B, S]");
    }
    const auto embeds_shape = visual_embeds.get_shape();
    if (embeds_shape.size() != 2) {
        OPENVINO_THROW("visual_embeds must have shape [V, H]");
    }
    const size_t batch = mask_shape[0];
    const size_t seq_len = mask_shape[1];
    const size_t hidden = embeds_shape[1];

    ov::Tensor out(visual_embeds.get_element_type(), {batch, seq_len, hidden});
    std::memset(out.data(), 0, out.get_byte_size());

    const size_t elem_size = visual_embeds.get_element_type().size();
    const size_t row_bytes = hidden * elem_size;

    const char* src = static_cast<const char*>(visual_embeds.data());
    char* dst = static_cast<char*>(out.data());

    size_t visual_idx = 0;
    const size_t total = batch * seq_len;
    for (size_t idx = 0; idx < total; ++idx) {
        if (!mask_value(visual_pos_mask, idx)) {
            continue;
        }
        if (visual_idx >= embeds_shape[0]) {
            OPENVINO_THROW("visual_embeds shorter than visual_pos_mask");
        }
        std::memcpy(dst + idx * row_bytes, src + visual_idx * row_bytes, row_bytes);
        visual_idx++;
    }
    if (visual_idx != embeds_shape[0]) {
        OPENVINO_THROW("visual_embeds length does not match visual_pos_mask");
    }
    return out;
}

ov::Tensor GlmOcrInputPlanner::build_decode_position_ids(const ov::Tensor& rope_deltas,
                                                         int64_t past_length,
                                                         int64_t seq_len) {
    if (rope_deltas.get_element_type() != ov::element::i64) {
        OPENVINO_THROW("rope_deltas must be i64");
    }
    if (past_length < 0 || seq_len <= 0) {
        OPENVINO_THROW("Invalid past_length or seq_len for decode position ids");
    }
    const auto shape = rope_deltas.get_shape();
    size_t batch = 0;
    if (shape.size() == 1) {
        batch = shape[0];
    } else if (shape.size() == 2) {
        if (shape[1] != 1) {
            OPENVINO_THROW("rope_deltas must have shape [B] or [B, 1]");
        }
        batch = shape[0];
    } else {
        OPENVINO_THROW("rope_deltas must have shape [B] or [B, 1]");
    }

    ov::Tensor position_ids(ov::element::i64, {3, batch, static_cast<size_t>(seq_len)});
    auto* out = position_ids.data<int64_t>();
    const int64_t* deltas = rope_deltas.data<const int64_t>();
    const size_t plane_stride = batch * static_cast<size_t>(seq_len);

    for (size_t b = 0; b < batch; ++b) {
        const int64_t base = past_length + deltas[b];
        for (int64_t s = 0; s < seq_len; ++s) {
            const int64_t value = base + s;
            const size_t idx = b * static_cast<size_t>(seq_len) + static_cast<size_t>(s);
            out[idx] = value;
            out[plane_stride + idx] = value;
            out[2 * plane_stride + idx] = value;
        }
    }

    return position_ids;
}

GlmOcrVisionPreprocessConfig GlmOcrVisionPreprocessConfig::from_json_file(
    const std::filesystem::path& path) {
    nlohmann::json data;
    read_preprocess_json_file(path, data);
    GlmOcrVisionPreprocessConfig cfg;
    using ov::genai::utils::read_json_param;
    read_json_param(data, "size.shortest_edge", cfg.min_pixels);
    read_json_param(data, "size.longest_edge", cfg.max_pixels);
    read_json_param(data, "patch_size", cfg.patch_size);
    read_json_param(data, "temporal_patch_size", cfg.temporal_patch_size);
    read_json_param(data, "merge_size", cfg.merge_size);
    read_json_param(data, "image_mean", cfg.image_mean);
    read_json_param(data, "image_std", cfg.image_std);
    return cfg;
}

GlmOcrVisionPreprocessor::GlmOcrVisionPreprocessor(
    const GlmOcrVisionConfig& vision_cfg,
    const GlmOcrVisionPreprocessConfig& preprocess_cfg)
    : vision_cfg_(vision_cfg),
      preprocess_cfg_(preprocess_cfg) {}

GlmOcrVisionInputs GlmOcrVisionPreprocessor::preprocess(const ov::Tensor& images) const {
    const auto img_shape = images.get_shape();
    if (img_shape.size() != 3 && img_shape.size() != 4) {
        OPENVINO_THROW("images must have shape [H, W, C] or [B, H, W, C]");
    }
    if (images.get_element_type() != ov::element::u8) {
        OPENVINO_THROW("images must be u8 for GlmOcr preprocessing");
    }

    const bool has_batch = img_shape.size() == 4;
    const size_t batch = has_batch ? img_shape[0] : 1;
    const size_t in_h = has_batch ? img_shape[1] : img_shape[0];
    const size_t in_w = has_batch ? img_shape[2] : img_shape[1];
    const size_t channels = has_batch ? img_shape[3] : img_shape[2];
    if (channels != 3) {
        OPENVINO_THROW("images must have 3 channels");
    }

    const size_t factor = static_cast<size_t>(preprocess_cfg_.patch_size * preprocess_cfg_.merge_size);
    const uint8_t* src = images.data<const uint8_t>();
    const bool nchw = false;

    std::vector<PreparedImage> prepared;
    prepared.reserve(batch);

    for (size_t b = 0; b < batch; ++b) {
        const uint8_t* src_img = src + b * in_h * in_w * channels;
        size_t out_h = in_h;
        size_t out_w = in_w;
        if (preprocess_cfg_.do_resize) {
            auto resized = smart_resize(in_h,
                                        in_w,
                                        factor,
                                        static_cast<size_t>(preprocess_cfg_.min_pixels),
                                        static_cast<size_t>(preprocess_cfg_.max_pixels));
            out_h = resized.first;
            out_w = resized.second;
        }
        if (out_h % preprocess_cfg_.patch_size != 0 || out_w % preprocess_cfg_.patch_size != 0) {
            OPENVINO_THROW("Resized image must be divisible by patch_size");
        }

        std::vector<float> frame;
        resize_bicubic_to_chw(src_img,
                               in_h,
                               in_w,
                               channels,
                               nchw,
                               out_h,
                               out_w,
                               preprocess_cfg_.image_mean,
                               preprocess_cfg_.image_std,
                               frame);

        const size_t frames = 1;
        size_t padded_frames = frames;
        if (frames % static_cast<size_t>(preprocess_cfg_.temporal_patch_size) != 0) {
            padded_frames += static_cast<size_t>(preprocess_cfg_.temporal_patch_size) -
                             (frames % static_cast<size_t>(preprocess_cfg_.temporal_patch_size));
        }

        std::vector<float> stacked(padded_frames * frame.size());
        for (size_t t = 0; t < padded_frames; ++t) {
            const size_t dst_offset = t * frame.size();
            std::copy(frame.begin(), frame.end(), stacked.begin() + dst_offset);
        }

        PreparedImage item;
        item.data = std::move(stacked);
        item.frames = padded_frames;
        item.height = out_h;
        item.width = out_w;
        item.grid_t = static_cast<int64_t>(padded_frames / static_cast<size_t>(preprocess_cfg_.temporal_patch_size));
        item.grid_h = static_cast<int64_t>(out_h / static_cast<size_t>(preprocess_cfg_.patch_size));
        item.grid_w = static_cast<int64_t>(out_w / static_cast<size_t>(preprocess_cfg_.patch_size));
        prepared.push_back(std::move(item));
    }

    int64_t total_patches = 0;
    for (const auto& item : prepared) {
        total_patches += item.grid_t * item.grid_h * item.grid_w;
    }

    ov::Tensor grid_thw(ov::element::i64, {batch, 3});
    auto* grid = grid_thw.data<int64_t>();
    for (size_t b = 0; b < prepared.size(); ++b) {
        grid[b * 3 + 0] = prepared[b].grid_t;
        grid[b * 3 + 1] = prepared[b].grid_h;
        grid[b * 3 + 2] = prepared[b].grid_w;
    }

    const size_t patch_size = static_cast<size_t>(preprocess_cfg_.patch_size);
    const size_t temporal_patch = static_cast<size_t>(preprocess_cfg_.temporal_patch_size);
    const size_t merge_size = static_cast<size_t>(preprocess_cfg_.merge_size);
    const size_t patch_stride = channels * temporal_patch * patch_size * patch_size;

    ov::Tensor pixel_values(ov::element::f32,
                            {static_cast<size_t>(total_patches),
                             channels,
                             temporal_patch,
                             patch_size,
                             patch_size});
    float* out = pixel_values.data<float>();
    size_t patch_offset = 0;

    for (const auto& item : prepared) {
        const size_t height = item.height;
        const size_t width = item.width;
        const size_t frame_stride = channels * height * width;
        const float* data = item.data.data();
        const size_t grid_t = static_cast<size_t>(item.grid_t);
        const size_t grid_h = static_cast<size_t>(item.grid_h);
        const size_t grid_w = static_cast<size_t>(item.grid_w);
        if (grid_h % merge_size != 0 || grid_w % merge_size != 0) {
            OPENVINO_THROW("grid_h/grid_w must be divisible by merge_size");
        }
        const size_t merged_h = grid_h / merge_size;
        const size_t merged_w = grid_w / merge_size;

        for (size_t t = 0; t < grid_t; ++t) {
            for (size_t bh = 0; bh < merged_h; ++bh) {
                for (size_t bw = 0; bw < merged_w; ++bw) {
                    for (size_t mh = 0; mh < merge_size; ++mh) {
                        for (size_t mw = 0; mw < merge_size; ++mw) {
                            float* dst = out + patch_offset * patch_stride;
                            size_t dst_idx = 0;
                            const size_t h_idx = (bh * merge_size + mh) * patch_size;
                            const size_t w_idx = (bw * merge_size + mw) * patch_size;
                            for (size_t c = 0; c < channels; ++c) {
                                for (size_t tp = 0; tp < temporal_patch; ++tp) {
                                    const size_t t_idx = (t * temporal_patch + tp) * frame_stride;
                                    for (size_t ph = 0; ph < patch_size; ++ph) {
                                        for (size_t pw = 0; pw < patch_size; ++pw) {
                                            const size_t src_idx =
                                                t_idx + (c * height + h_idx + ph) * width + w_idx + pw;
                                            dst[dst_idx++] = data[src_idx];
                                        }
                                    }
                                }
                            }
                            patch_offset++;
                        }
                    }
                }
            }
        }
    }

    auto rotary = build_rotary_cos_sin(grid_thw, vision_cfg_, preprocess_cfg_.merge_size);

    return {pixel_values, grid_thw, rotary.first, rotary.second};
}

int64_t GlmOcrVisionPreprocessor::count_visual_tokens(const ov::Tensor& grid_thw,
                                                      int32_t spatial_merge_size) {
    if (grid_thw.get_element_type() != ov::element::i64) {
        OPENVINO_THROW("grid_thw must be i64");
    }
    const auto shape = grid_thw.get_shape();
    if (shape.size() != 2 || shape[1] != 3) {
        OPENVINO_THROW("grid_thw must have shape [N, 3]");
    }
    const int64_t* grid = grid_thw.data<const int64_t>();
    int64_t total = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        const int64_t t = grid[i * 3 + 0];
        const int64_t h = grid[i * 3 + 1];
        const int64_t w = grid[i * 3 + 2];
        if (t <= 0 || h <= 0 || w <= 0) {
            OPENVINO_THROW("Invalid grid_thw values");
        }
        if (h % spatial_merge_size != 0 || w % spatial_merge_size != 0) {
            OPENVINO_THROW("grid_thw must be divisible by spatial_merge_size");
        }
        total += t * (h / spatial_merge_size) * (w / spatial_merge_size);
    }
    return total;
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
