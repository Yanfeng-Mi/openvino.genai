// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <regex>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>

#include "loaders/model_config.hpp"
#include "modeling/models/qwen3_asr/modeling_qwen3_asr.hpp"
#include "modeling/weights/quantization_config.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "safetensors_utils/safetensors_loader.hpp"
#include "safetensors_utils/safetensors_weight_finalizer.hpp"
#include "safetensors_utils/safetensors_weight_source.hpp"
#include "whisper/feature_extractor.hpp"

namespace {

std::string trim_copy(const std::string& s);
std::string normalize_language_name(const std::string& raw);

bool has_ir_model_pair(const std::filesystem::path& xml_path, const std::filesystem::path& bin_path) {
    return std::filesystem::exists(xml_path) && std::filesystem::is_regular_file(xml_path) &&
           std::filesystem::exists(bin_path) && std::filesystem::is_regular_file(bin_path);
}

bool model_has_input_named(const std::shared_ptr<ov::Model>& model, const char* name) {
    for (const auto& input : model->inputs()) {
        for (const auto& input_name : input.get_names()) {
            if (input_name == name) {
                return true;
            }
        }
    }
    return false;
}

bool text_model_cache_supports_mode(const std::shared_ptr<ov::Model>& model, bool expect_audio_inputs) {
    const bool has_audio_embeds =
        model_has_input_named(model, ov::genai::modeling::models::Qwen3ASRTextIO::kAudioEmbeds);
    const bool has_audio_pos_mask =
        model_has_input_named(model, ov::genai::modeling::models::Qwen3ASRTextIO::kAudioPosMask);
    const bool has_audio_inputs = has_audio_embeds || has_audio_pos_mask;
    return has_audio_inputs == expect_audio_inputs;
}

double elapsed_ms(const std::chrono::steady_clock::time_point& start,
                  const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

struct StreamingConfig {
    bool enabled = false;
    double chunk_sec = 1.0;
    double window_sec = 15.0;
    int32_t unfixed_chunk_num = 3;
    int32_t unfixed_token_num = 5;
};

struct ASRDecodePassResult {
    ov::Shape embeds_shape;
    ov::Shape logits_shape;
    std::vector<int64_t> generated_ids;
    std::string transcript_text;
    std::string language_tag;
    int64_t audio_output_length = 0;
    size_t prompt_token_size = 0;
    size_t audio_seq_len = 0;
    double audio_encode_ms = 0.0;
    double ttft_ms = 0.0;
    double decode_ms = 0.0;
    size_t decode_tail_tokens = 0;
    size_t infer_steps = 0;
};

void print_usage(const char* argv0) {
    std::cout
        << "Qwen3-ASR modeling sample\n"
        << "Usage:\n"
        << "  " << argv0
        << " <TEXT_MODEL_DIR> [AUDIO_MODEL_DIR] [DEVICE] [--wav <AUDIO.wav>] [--device <DEVICE>] [--max_new_tokens <N>] [--text-only] [--prompt <TEXT>] [--streaming] [--stream_chunk_sec <S>] [--stream_window_sec <S>] [--stream_unfixed_chunk_num <N>] [--stream_unfixed_token_num <K>]\n\n"
        << "Examples:\n"
        << "  " << argv0 << " C:/models/Qwen3-ASR\n"
        << "  " << argv0 << " C:/models/Qwen3-ASR/text C:/models/Qwen3-ASR/audio GPU\n"
        << "  " << argv0 << " C:/models/Qwen3-ASR --wav C:/audio/test.wav --device GPU --max_new_tokens 128\n"
        << "  " << argv0 << " C:/models/Qwen3-ASR --wav C:/audio/test.wav --streaming --stream_chunk_sec 2.0 --stream_window_sec 24.0\n"
        << "  " << argv0 << " C:/models/Qwen3-ASR --text-only --prompt \"Summarize this sentence.\" --device GPU\n"
        << "  " << argv0 << " C:/models/Qwen3-ASR --cached-model\n\n"
        << "Options:\n"
        << "  --cached-model (or --cache-model)  Serialize built IR to model directory before inference\n"
        << "  --streaming                         Enable chunked streaming ASR mode\n"
        << "  --stream_chunk_sec <S>              Seconds of new audio consumed per chunk (default: 1.0)\n"
        << "  --stream_window_sec <S>             Seconds kept in the sliding audio window after warmup (default: 15.0)\n"
        << "  --stream_unfixed_chunk_num <N>      Number of warmup chunks before using the fixed window (default: 3)\n"
        << "  --stream_unfixed_token_num <K>      Number of trailing words kept provisional between chunks (default: 5)\n\n"
        << "In-flight quantization (optional, env-based):\n"
        << "  OV_GENAI_INFLIGHT_QUANT_MODE\n"
        << "  OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE\n"
        << "  OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE\n";
}

uint16_t read_le_u16(const uint8_t* ptr) {
    return static_cast<uint16_t>(ptr[0]) | (static_cast<uint16_t>(ptr[1]) << 8);
}

uint32_t read_le_u32(const uint8_t* ptr) {
    return static_cast<uint32_t>(ptr[0]) | (static_cast<uint32_t>(ptr[1]) << 8) | (static_cast<uint32_t>(ptr[2]) << 16) |
           (static_cast<uint32_t>(ptr[3]) << 24);
}

std::vector<float> linear_resample(const std::vector<float>& input, uint32_t in_rate, uint32_t out_rate) {
    if (input.empty()) {
        return {};
    }
    if (in_rate == out_rate) {
        return input;
    }
    const double ratio = static_cast<double>(out_rate) / static_cast<double>(in_rate);
    const size_t out_size = std::max<size_t>(1, static_cast<size_t>(input.size() * ratio));

    std::vector<float> output(out_size);
    for (size_t i = 0; i < out_size; ++i) {
        const double src_pos = static_cast<double>(i) / ratio;
        const size_t idx0 = static_cast<size_t>(src_pos);
        const size_t idx1 = std::min(idx0 + 1, input.size() - 1);
        const double frac = src_pos - static_cast<double>(idx0);
        output[i] = static_cast<float>((1.0 - frac) * static_cast<double>(input[idx0]) + frac * static_cast<double>(input[idx1]));
    }
    return output;
}

float decode_sample_to_f32(const uint8_t* ptr, uint16_t audio_format, uint16_t bits_per_sample) {
    // audio_format: 1 = PCM integer, 3 = IEEE float
    if (audio_format == 1) {
        switch (bits_per_sample) {
        case 8: {
            // unsigned PCM8: [0,255] -> [-1,1)
            const float v = static_cast<float>(ptr[0]);
            return (v - 128.0f) / 128.0f;
        }
        case 16: {
            int16_t v = 0;
            std::memcpy(&v, ptr, sizeof(v));
            return static_cast<float>(v) / 32768.0f;
        }
        case 24: {
            int32_t v = (static_cast<int32_t>(ptr[0]) | (static_cast<int32_t>(ptr[1]) << 8) |
                         (static_cast<int32_t>(ptr[2]) << 16));
            if ((v & 0x00800000) != 0) {
                v |= ~0x00FFFFFF;
            }
            return static_cast<float>(v) / 8388608.0f;
        }
        case 32: {
            int32_t v = 0;
            std::memcpy(&v, ptr, sizeof(v));
            return static_cast<float>(v) / 2147483648.0f;
        }
        default:
            throw std::runtime_error("Unsupported PCM bit depth: " + std::to_string(bits_per_sample));
        }
    }

    if (audio_format == 3) {
        if (bits_per_sample == 32) {
            float v = 0.0f;
            std::memcpy(&v, ptr, sizeof(v));
            return v;
        }
        if (bits_per_sample == 64) {
            double v = 0.0;
            std::memcpy(&v, ptr, sizeof(v));
            return static_cast<float>(v);
        }
        throw std::runtime_error("Unsupported IEEE float bit depth: " + std::to_string(bits_per_sample));
    }

    throw std::runtime_error("Unsupported wav audio_format: " + std::to_string(audio_format));
}

std::vector<float> read_wav_pcm_mono_or_stereo(const std::filesystem::path& wav_path, uint32_t target_sample_rate) {
    std::ifstream in(wav_path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open wav file: " + wav_path.string());
    }

    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (bytes.size() < 44) {
        throw std::runtime_error("Invalid wav file (too small): " + wav_path.string());
    }

    if (std::memcmp(bytes.data(), "RIFF", 4) != 0 || std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
        throw std::runtime_error("Unsupported wav container (expect RIFF/WAVE): " + wav_path.string());
    }

    uint16_t audio_format = 0;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;

    size_t data_offset = 0;
    size_t data_size = 0;

    size_t offset = 12;
    while (offset + 8 <= bytes.size()) {
        const uint8_t* chunk = bytes.data() + offset;
        const uint32_t chunk_size = read_le_u32(chunk + 4);
        const size_t chunk_data_offset = offset + 8;
        if (chunk_data_offset + chunk_size > bytes.size()) {
            throw std::runtime_error("Corrupted wav chunk size in: " + wav_path.string());
        }

        if (std::memcmp(chunk, "fmt ", 4) == 0) {
            if (chunk_size < 16) {
                throw std::runtime_error("Invalid fmt chunk in wav: " + wav_path.string());
            }
            const uint8_t* fmt = bytes.data() + chunk_data_offset;
            audio_format = read_le_u16(fmt + 0);
            channels = read_le_u16(fmt + 2);
            sample_rate = read_le_u32(fmt + 4);
            bits_per_sample = read_le_u16(fmt + 14);
        } else if (std::memcmp(chunk, "data", 4) == 0) {
            data_offset = chunk_data_offset;
            data_size = chunk_size;
            break;
        }

        offset = chunk_data_offset + chunk_size + (chunk_size % 2);
    }

    if (channels != 1 && channels != 2) {
        throw std::runtime_error("Only mono or stereo wav is supported: " + wav_path.string());
    }
    if (audio_format != 1 && audio_format != 3) {
        throw std::runtime_error("Only PCM or IEEE float wav is supported: " + wav_path.string());
    }
    if (data_offset == 0 || data_size == 0) {
        throw std::runtime_error("No audio data chunk found in wav: " + wav_path.string());
    }

    const size_t bytes_per_channel = static_cast<size_t>(bits_per_sample) / 8;
    if (bytes_per_channel == 0 || (static_cast<size_t>(bits_per_sample) % 8) != 0) {
        throw std::runtime_error("Unsupported non-byte-aligned bit depth: " + std::to_string(bits_per_sample));
    }
    const size_t bytes_per_frame = bytes_per_channel * channels;
    if (bytes_per_frame == 0 || (data_size % bytes_per_frame) != 0) {
        throw std::runtime_error("Corrupted wav data size/frame alignment: " + wav_path.string());
    }

    const size_t frames = data_size / bytes_per_frame;
    const uint8_t* pcm = bytes.data() + data_offset;

    std::vector<float> mono(frames);
    for (size_t i = 0; i < frames; ++i) {
        const uint8_t* frame_ptr = pcm + i * bytes_per_frame;
        if (channels == 1) {
            mono[i] = decode_sample_to_f32(frame_ptr, audio_format, bits_per_sample);
        } else {
            const float left = decode_sample_to_f32(frame_ptr, audio_format, bits_per_sample);
            const float right = decode_sample_to_f32(frame_ptr + bytes_per_channel, audio_format, bits_per_sample);
            mono[i] = 0.5f * (left + right);
        }
    }
    if (target_sample_rate == 0) {
        throw std::runtime_error("Invalid target sample rate: 0");
    }
    if (sample_rate != target_sample_rate) {
        return linear_resample(mono, sample_rate, target_sample_rate);
    }
    return mono;
}

ov::genai::modeling::models::Qwen3ASRTextConfig to_text_cfg(const ov::genai::loaders::ModelConfig& cfg) {
    using ov::genai::modeling::models::Qwen3ASRTextConfig;
    Qwen3ASRTextConfig out;
    out.architecture = "qwen3_asr";
    out.vocab_size = cfg.vocab_size;
    out.hidden_size = cfg.hidden_size;
    out.intermediate_size = cfg.intermediate_size;
    out.num_hidden_layers = cfg.num_hidden_layers;
    out.num_attention_heads = cfg.num_attention_heads;
    out.num_key_value_heads = cfg.num_key_value_heads > 0 ? cfg.num_key_value_heads : cfg.num_attention_heads;
    out.head_dim = cfg.head_dim > 0 ? cfg.head_dim : (cfg.hidden_size / std::max(1, cfg.num_attention_heads));
    out.max_position_embeddings = cfg.max_position_embeddings;
    out.rms_norm_eps = cfg.rms_norm_eps;
    out.rope_theta = cfg.rope_theta;
    out.hidden_act = cfg.hidden_act;
    out.attention_bias = cfg.attention_bias;
    out.tie_word_embeddings = cfg.tie_word_embeddings;

    if (out.hidden_size <= 0 || out.intermediate_size <= 0 || out.num_hidden_layers <= 0 ||
        out.num_attention_heads <= 0 || out.vocab_size <= 0) {
        throw std::runtime_error("Invalid text config parsed from config.json");
    }
    return out;
}

ov::genai::modeling::models::Qwen3ASRAudioConfig to_audio_cfg(const ov::genai::loaders::ModelConfig& cfg) {
    using ov::genai::modeling::models::Qwen3ASRAudioConfig;
    Qwen3ASRAudioConfig out;
    out.architecture = "qwen3_asr_audio_encoder";
    out.num_mel_bins = cfg.audio_num_mel_bins > 0 ? cfg.audio_num_mel_bins : out.num_mel_bins;
    out.d_model = cfg.audio_hidden_size > 0 ? cfg.audio_hidden_size : (cfg.hidden_size > 0 ? cfg.hidden_size : out.d_model);
    out.encoder_layers = cfg.audio_num_hidden_layers > 0 ? cfg.audio_num_hidden_layers
                                                          : (cfg.num_hidden_layers > 0 ? cfg.num_hidden_layers : out.encoder_layers);
    out.encoder_attention_heads = cfg.audio_num_attention_heads > 0 ? cfg.audio_num_attention_heads
                                                                     : (cfg.num_attention_heads > 0 ? cfg.num_attention_heads : out.encoder_attention_heads);
    out.encoder_ffn_dim = cfg.audio_intermediate_size > 0 ? cfg.audio_intermediate_size
                                                           : (cfg.intermediate_size > 0 ? cfg.intermediate_size : out.encoder_ffn_dim);
    out.max_source_positions = cfg.audio_max_position_embeddings > 0 ? cfg.audio_max_position_embeddings
                                                                      : (cfg.max_position_embeddings > 0 ? cfg.max_position_embeddings : out.max_source_positions);
    out.downsample_hidden_size = cfg.audio_downsample_hidden_size > 0 ? cfg.audio_downsample_hidden_size : out.downsample_hidden_size;
    out.output_dim = cfg.audio_output_dim > 0 ? cfg.audio_output_dim : out.output_dim;
    out.n_window = cfg.audio_n_window > 0 ? cfg.audio_n_window : out.n_window;
    out.n_window_infer = cfg.audio_n_window_infer > 0 ? cfg.audio_n_window_infer : out.n_window_infer;
    out.activation_function = !cfg.audio_hidden_act.empty() ? cfg.audio_hidden_act
                                                             : (cfg.hidden_act.empty() ? out.activation_function : cfg.hidden_act);

    if (out.d_model <= 0 || out.encoder_layers <= 0 || out.encoder_attention_heads <= 0 || out.encoder_ffn_dim <= 0) {
        throw std::runtime_error("Invalid audio config parsed from config.json");
    }
    return out;
}

ov::Tensor make_i64(const ov::Shape& shape, int64_t value = 0) {
    ov::Tensor t(ov::element::i64, shape);
    std::fill(t.data<int64_t>(), t.data<int64_t>() + t.get_size(), value);
    return t;
}

ov::Tensor make_i32(const ov::Shape& shape, int32_t value = 0) {
    ov::Tensor t(ov::element::i32, shape);
    std::fill(t.data<int32_t>(), t.data<int32_t>() + t.get_size(), value);
    return t;
}

ov::Tensor make_bool(const ov::Shape& shape, bool value = true) {
    ov::Tensor t(ov::element::boolean, shape);
    std::fill(t.data<char>(), t.data<char>() + t.get_size(), static_cast<char>(value ? 1 : 0));
    return t;
}

ov::Tensor make_audio_features(size_t batch, size_t mel_bins, size_t frames) {
    ov::Tensor t(ov::element::f32, ov::Shape{batch, mel_bins, frames});
    float* ptr = t.data<float>();
    for (size_t i = 0; i < t.get_size(); ++i) {
        ptr[i] = static_cast<float>((i % 97) - 48) / 64.0f;
    }
    return t;
}

ov::Tensor make_audio_features_from_pcm(const std::vector<float>& raw,
                                        ov::genai::WhisperFeatureExtractor& extractor,
                                        size_t expected_mel_bins) {
    if (raw.empty()) {
        throw std::runtime_error("Input audio slice has no samples");
    }

    ov::genai::WhisperFeatures features = extractor.extract(raw);
    if (features.n_frames == 0 || features.feature_size == 0) {
        throw std::runtime_error("Failed to extract audio features from input samples");
    }

    const size_t actual_mel_bins = features.feature_size;
    const size_t frames = features.n_frames;
    const size_t mel_bins = expected_mel_bins > 0 ? expected_mel_bins : actual_mel_bins;

    ov::Tensor tensor(ov::element::f32, ov::Shape{1, mel_bins, frames});
    float* dst = tensor.data<float>();
    const float* src = features.data.data();

    for (size_t m = 0; m < mel_bins; ++m) {
        const float* src_row = (m < actual_mel_bins) ? (src + m * frames) : nullptr;
        float* dst_row = dst + m * frames;
        if (src_row != nullptr) {
            std::copy(src_row, src_row + frames, dst_row);
        } else {
            std::fill(dst_row, dst_row + frames, 0.0f);
        }
    }

    return tensor;
}

ov::Tensor make_audio_features_from_wav(const std::filesystem::path& wav_path,
                                        const std::filesystem::path& preprocessor_json,
                                        size_t expected_mel_bins,
                                        double* audio_duration_seconds = nullptr,
                                        uint32_t* used_sample_rate = nullptr) {
    ov::genai::WhisperFeatureExtractor extractor(preprocessor_json);
    const uint32_t target_sample_rate = static_cast<uint32_t>(std::max<size_t>(1, extractor.sampling_rate));
    if (used_sample_rate != nullptr) {
        *used_sample_rate = target_sample_rate;
    }
    const std::vector<float> raw = read_wav_pcm_mono_or_stereo(wav_path, target_sample_rate);
    if (raw.empty()) {
        throw std::runtime_error("Input wav has no samples: " + wav_path.string());
    }
    if (audio_duration_seconds != nullptr) {
        *audio_duration_seconds = static_cast<double>(raw.size()) / static_cast<double>(target_sample_rate);
    }

    ov::genai::WhisperFeatures features = extractor.extract(raw);
    if (features.n_frames == 0 || features.feature_size == 0) {
        throw std::runtime_error("Failed to extract audio features from wav: " + wav_path.string());
    }

    const size_t actual_mel_bins = features.feature_size;
    const size_t frames = features.n_frames;
    const size_t mel_bins = expected_mel_bins > 0 ? expected_mel_bins : actual_mel_bins;

    ov::Tensor tensor(ov::element::f32, ov::Shape{1, mel_bins, frames});
    float* dst = tensor.data<float>();
    const float* src = features.data.data();

    for (size_t m = 0; m < mel_bins; ++m) {
        const float* src_row = (m < actual_mel_bins) ? (src + m * frames) : nullptr;
        float* dst_row = dst + m * frames;
        if (src_row != nullptr) {
            std::copy(src_row, src_row + frames, dst_row);
        } else {
            std::fill(dst_row, dst_row + frames, 0.0f);
        }
    }

    return tensor;
}

ov::Tensor slice_audio_feature_window(const ov::Tensor& features, size_t start_frame, size_t end_frame) {
    const auto shape = features.get_shape();
    if (shape.size() != 3 || shape[0] != 1 || features.get_element_type() != ov::element::f32) {
        throw std::runtime_error("Expected audio features shape [1, mel, frames] with f32 type");
    }
    if (start_frame > end_frame || end_frame > shape[2]) {
        throw std::runtime_error("Invalid audio feature frame slice");
    }

    const size_t mel_bins = shape[1];
    const size_t frame_count = end_frame - start_frame;
    ov::Tensor out(ov::element::f32, ov::Shape{1, mel_bins, frame_count});

    const float* src = features.data<const float>();
    float* dst = out.data<float>();
    for (size_t mel = 0; mel < mel_bins; ++mel) {
        const float* src_row = src + mel * shape[2] + start_frame;
        float* dst_row = dst + mel * frame_count;
        std::copy(src_row, src_row + frame_count, dst_row);
    }
    return out;
}

ov::Tensor make_position_ids_3d(size_t batch, size_t seq_len) {
    ov::Tensor t(ov::element::i64, ov::Shape{3, batch, seq_len});
    int64_t* ptr = t.data<int64_t>();
    for (size_t plane = 0; plane < 3; ++plane) {
        for (size_t b = 0; b < batch; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                ptr[(plane * batch + b) * seq_len + s] = static_cast<int64_t>(s);
            }
        }
    }
    return t;
}

ov::Tensor make_i64_from_vector(const std::vector<int64_t>& values) {
    ov::Tensor t(ov::element::i64, ov::Shape{1, values.size()});
    std::copy(values.begin(), values.end(), t.data<int64_t>());
    return t;
}

ov::Tensor make_audio_pos_mask_prefix(size_t batch, size_t seq_len, size_t audio_prefix_len) {
    ov::Tensor t(ov::element::boolean, ov::Shape{batch, seq_len});
    char* ptr = t.data<char>();
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            ptr[b * seq_len + s] = static_cast<char>(s < audio_prefix_len ? 1 : 0);
        }
    }
    return t;
}

ov::Tensor make_audio_pos_mask_from_flags(const std::vector<char>& flags) {
    ov::Tensor t(ov::element::boolean, ov::Shape{1, flags.size()});
    char* ptr = t.data<char>();
    std::copy(flags.begin(), flags.end(), ptr);
    return t;
}

ov::Tensor make_padded_audio_embeds(const ov::Tensor& audio_embeds, size_t target_seq_len) {
    const auto src_shape = audio_embeds.get_shape();
    if (src_shape.size() != 3) {
        throw std::runtime_error("audio_embeds rank must be 3 for padding");
    }
    const size_t batch = src_shape[0];
    const size_t src_seq = src_shape[1];
    const size_t hidden = src_shape[2];
    if (target_seq_len < src_seq) {
        throw std::runtime_error("target_seq_len is smaller than audio_embeds sequence length");
    }

    ov::Tensor out(audio_embeds.get_element_type(), ov::Shape{batch, target_seq_len, hidden});
    if (out.get_element_type() != ov::element::f32 || audio_embeds.get_element_type() != ov::element::f32) {
        throw std::runtime_error("Expected f32 audio_embeds tensor");
    }

    const float* src = audio_embeds.data<const float>();
    float* dst = out.data<float>();
    std::fill(dst, dst + out.get_size(), 0.0f);

    const size_t copy_per_batch = src_seq * hidden;
    const size_t dst_stride = target_seq_len * hidden;
    for (size_t b = 0; b < batch; ++b) {
        std::copy(src + b * copy_per_batch,
                  src + b * copy_per_batch + copy_per_batch,
                  dst + b * dst_stride);
    }
    return out;
}

std::vector<int64_t> tensor_row_to_i64_vector(const ov::Tensor& t) {
    const auto shape = t.get_shape();
    if (shape.size() != 2 || shape[0] != 1 || t.get_element_type() != ov::element::i64) {
        throw std::runtime_error("Expected tensor shape [1, N] with i64 type");
    }
    const int64_t* src = t.data<const int64_t>();
    return std::vector<int64_t>(src, src + shape[1]);
}

ov::Tensor make_audio_embeds_for_mask_positions(const ov::Tensor& audio_embeds, const std::vector<char>& audio_flags) {
    const auto src_shape = audio_embeds.get_shape();
    if (src_shape.size() != 3 || src_shape[0] != 1 || audio_embeds.get_element_type() != ov::element::f32) {
        throw std::runtime_error("Expected audio_embeds shape [1, T, H] with f32 type");
    }

    const size_t src_seq = src_shape[1];
    const size_t hidden = src_shape[2];
    size_t true_count = 0;
    for (char v : audio_flags) {
        true_count += (v != 0) ? 1 : 0;
    }
    if (true_count != src_seq) {
        throw std::runtime_error("Audio mask true-count must match audio_embeds sequence length");
    }

    ov::Tensor out(ov::element::f32, ov::Shape{1, audio_flags.size(), hidden});
    float* dst = out.data<float>();
    std::fill(dst, dst + out.get_size(), 0.0f);

    const float* src = audio_embeds.data<const float>();
    size_t src_idx = 0;
    for (size_t pos = 0; pos < audio_flags.size(); ++pos) {
        if (audio_flags[pos] == 0) {
            continue;
        }
        std::copy(src + src_idx * hidden,
                  src + (src_idx + 1) * hidden,
                  dst + pos * hidden);
        ++src_idx;
    }
    return out;
}

int64_t argmax_last_token_id(const ov::Tensor& logits) {
    const auto shape = logits.get_shape();
    if (shape.size() != 3 || shape[0] == 0 || shape[1] == 0 || shape[2] == 0) {
        throw std::runtime_error("Unexpected logits shape for argmax");
    }

    const size_t vocab = shape[2];
    const size_t offset = (shape[0] - 1) * shape[1] * vocab + (shape[1] - 1) * vocab;
    const float* row = logits.data<const float>() + offset;

    size_t best_idx = 0;
    float best_val = row[0];
    for (size_t i = 1; i < vocab; ++i) {
        if (row[i] > best_val) {
            best_val = row[i];
            best_idx = i;
        }
    }
    return static_cast<int64_t>(best_idx);
}

int64_t argmax_last_token_id_excluding(const ov::Tensor& logits, const std::unordered_set<int64_t>& excluded_ids) {
    const auto shape = logits.get_shape();
    if (shape.size() != 3 || shape[0] == 0 || shape[1] == 0 || shape[2] == 0) {
        throw std::runtime_error("Unexpected logits shape for argmax");
    }

    const size_t vocab = shape[2];
    const size_t offset = (shape[0] - 1) * shape[1] * vocab + (shape[1] - 1) * vocab;
    const float* row = logits.data<const float>() + offset;

    int64_t best_idx = -1;
    float best_val = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < vocab; ++i) {
        const int64_t tid = static_cast<int64_t>(i);
        if (excluded_ids.find(tid) != excluded_ids.end()) {
            continue;
        }
        if (row[i] > best_val) {
            best_val = row[i];
            best_idx = tid;
        }
    }

    if (best_idx < 0) {
        return argmax_last_token_id(logits);
    }
    return best_idx;
}

bool has_recent_ngram_loop(const std::vector<int64_t>& ids, size_t ngram, size_t repeats) {
    if (ngram == 0 || repeats < 2) {
        return false;
    }
    const size_t needed = ngram * repeats;
    if (ids.size() < needed) {
        return false;
    }

    const size_t base = ids.size() - needed;
    for (size_t r = 1; r < repeats; ++r) {
        for (size_t i = 0; i < ngram; ++i) {
            if (ids[base + i] != ids[base + r * ngram + i]) {
                return false;
            }
        }
    }
    return true;
}

bool trim_recent_duplicate_ngram(std::vector<int64_t>& ids, size_t min_ngram, size_t max_ngram) {
    if (min_ngram == 0 || max_ngram < min_ngram) {
        return false;
    }
    const size_t capped_max = std::min(max_ngram, ids.size() / 2);
    if (capped_max < min_ngram) {
        return false;
    }

    // Prefer trimming longer duplicated tails first.
    for (size_t n = capped_max; n >= min_ngram; --n) {
        const size_t off0 = ids.size() - 2 * n;
        const size_t off1 = ids.size() - n;
        bool same = true;
        for (size_t i = 0; i < n; ++i) {
            if (ids[off0 + i] != ids[off1 + i]) {
                same = false;
                break;
            }
        }
        if (same) {
            ids.resize(ids.size() - n);
            return true;
        }
        if (n == min_ngram) {
            break;
        }
    }
    return false;
}

bool is_language_tag_token(const std::string& token) {
    // Expected shape like: <|en|>, <|zh|>, <|yue|>
    if (token.size() < 6) {
        return false;
    }
    if (token.rfind("<|", 0) != 0 || token.substr(token.size() - 2) != "|>") {
        return false;
    }
    const std::string tag = token.substr(2, token.size() - 4);
    if (tag.size() < 2 || tag.size() > 5) {
        return false;
    }
    for (char c : tag) {
        if (!std::isalpha(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    return true;
}

std::string detect_language_from_tokens(ov::genai::Tokenizer& tokenizer, const std::vector<int64_t>& generated_ids) {
    for (size_t i = 0; i < generated_ids.size() && i < 8; ++i) {
        const std::string token = tokenizer.decode({generated_ids[i]}, {ov::genai::skip_special_tokens(false)});
        if (is_language_tag_token(token)) {
            return token;
        }
    }
    return "unknown";
}

std::string detect_language_from_language_prefix_tokens(ov::genai::Tokenizer& tokenizer,
                                                        const std::vector<int64_t>& generated_ids) {
    if (generated_ids.size() < 2) {
        return {};
    }

    std::string first = trim_copy(tokenizer.decode({generated_ids[0]}, {ov::genai::skip_special_tokens(false)}));
    std::string first_lower = first;
    std::transform(first_lower.begin(), first_lower.end(), first_lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (first_lower != "language") {
        return {};
    }

    const std::string second = trim_copy(tokenizer.decode({generated_ids[1]}, {ov::genai::skip_special_tokens(false)}));
    if (second.empty()) {
        return {};
    }
    return normalize_language_name(second);
}

int64_t find_token_id(const std::unordered_map<std::string, int64_t>& vocab,
                      const std::initializer_list<std::string>& candidates,
                      int64_t fallback = -1) {
    for (const auto& tok : candidates) {
        const auto it = vocab.find(tok);
        if (it != vocab.end()) {
            return it->second;
        }
    }
    return fallback;
}

void add_token_if_found(const std::unordered_map<std::string, int64_t>& vocab,
                        const std::string& token,
                        std::unordered_set<int64_t>& out) {
    const auto it = vocab.find(token);
    if (it != vocab.end()) {
        out.insert(it->second);
    }
}

std::string trim_copy(const std::string& s) {
    const auto begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
}

std::string normalize_language_name(const std::string& raw) {
    const std::string s = trim_copy(raw);
    if (s.empty()) {
        return {};
    }

    std::string out;
    out.reserve(s.size());
    bool seen_lower = false;
    for (char c : s) {
        if (!std::isalpha(static_cast<unsigned char>(c))) {
            break;
        }
        if (!out.empty() && seen_lower && std::isupper(static_cast<unsigned char>(c))) {
            break;
        }
        if (std::islower(static_cast<unsigned char>(c))) {
            seen_lower = true;
        }
        out.push_back(c);
    }
    return out;
}

bool is_suspicious_short_tail_word(const std::string& normalized_word) {
    static const std::unordered_set<std::string> suspicious_short_tail_words = {
        "a", "an", "the", "and", "but", "or", "so", "because", "if", "when", "while",
        "of", "to", "for", "in", "on", "at", "with", "from", "by", "as",
        "he", "she", "it", "they", "we", "i", "you", "his", "her", "their", "our", "your",
        "hmm", "hm", "mm", "uh", "um", "ah", "oh", "yeah", "ya"};
    return suspicious_short_tail_words.find(normalized_word) != suspicious_short_tail_words.end();
}

bool is_connector_tail_word(const std::string& normalized_word) {
    static const std::unordered_set<std::string> connector_tail_words = {
        "a", "an", "the", "and", "but", "or", "so", "because", "if", "when", "while",
        "of", "to", "for", "in", "on", "at", "with", "from", "by", "as"};
    return connector_tail_words.find(normalized_word) != connector_tail_words.end();
}

bool has_suspicious_connector_boundary(const std::vector<std::string>& base_words,
                                       const std::vector<std::string>& candidate_words,
                                       size_t overlap_words) {
    if (base_words.empty() || overlap_words >= candidate_words.size()) {
        return false;
    }

    auto normalize_boundary_word = [](const std::string& raw_word) {
        std::string out;
        out.reserve(raw_word.size());
        for (unsigned char c : raw_word) {
            if (std::isalnum(c)) {
                out.push_back(static_cast<char>(std::tolower(c)));
            }
        }
        return out;
    };

    const std::string left = normalize_boundary_word(base_words.back());
    const std::string right = normalize_boundary_word(candidate_words[overlap_words]);
    if (left.empty() || right.empty()) {
        return false;
    }
    if (!is_connector_tail_word(left) || !is_connector_tail_word(right)) {
        return false;
    }
    if (left == right) {
        return true;
    }

    static const std::unordered_set<std::string> hard_connector_boundary_words = {
        "and", "but", "or", "so", "because", "if", "when", "while", "of", "to", "for", "with", "from", "by", "as"};
    return hard_connector_boundary_words.find(left) != hard_connector_boundary_words.end() ||
           hard_connector_boundary_words.find(right) != hard_connector_boundary_words.end();
}

bool is_ascii_alpha_word(const std::string& word) {
    if (word.empty()) {
        return false;
    }
    for (char c : word) {
        if (!std::isalpha(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    return true;
}

std::string normalize_word_for_matching(const std::string& raw_word) {
    std::string out;
    out.reserve(raw_word.size());
    for (unsigned char c : raw_word) {
        if (c < 0x80 && std::isalnum(c)) {
            out.push_back(static_cast<char>(std::tolower(c)));
        } else if (c >= 0x80) {
            out.push_back(static_cast<char>(c));
        }
    }
    return out;
}

std::string normalize_text_for_comparison(const std::string& raw_text) {
    static const std::array<std::string, 3> utf8_punctuation = {
        "\xE3\x80\x82", "\xEF\xBC\x81", "\xEF\xBC\x9F"};

    std::string out;
    out.reserve(raw_text.size());
    for (size_t idx = 0; idx < raw_text.size();) {
        bool skipped_utf8_punctuation = false;
        for (const auto& token : utf8_punctuation) {
            if (raw_text.compare(idx, token.size(), token) == 0) {
                idx += token.size();
                skipped_utf8_punctuation = true;
                break;
            }
        }
        if (skipped_utf8_punctuation) {
            continue;
        }

        const unsigned char c = static_cast<unsigned char>(raw_text[idx]);
        if (c < 0x80) {
            if (std::isalnum(c)) {
                out.push_back(static_cast<char>(std::tolower(c)));
            }
            ++idx;
            continue;
        }

        out.push_back(static_cast<char>(c));
        ++idx;
    }
    return out;
}

size_t utf8_codepoint_count(const std::string& text) {
    size_t count = 0;
    for (unsigned char c : text) {
        if ((c & 0xC0) != 0x80) {
            ++count;
        }
    }
    return count;
}

uint32_t decode_utf8_codepoint(const std::string& text, size_t offset, size_t& width) {
    const unsigned char lead = static_cast<unsigned char>(text[offset]);
    if ((lead & 0x80) == 0) {
        width = 1;
        return lead;
    }
    if ((lead & 0xE0) == 0xC0 && offset + 1 < text.size()) {
        width = 2;
        return ((lead & 0x1F) << 6) |
               (static_cast<unsigned char>(text[offset + 1]) & 0x3F);
    }
    if ((lead & 0xF0) == 0xE0 && offset + 2 < text.size()) {
        width = 3;
        return ((lead & 0x0F) << 12) |
               ((static_cast<unsigned char>(text[offset + 1]) & 0x3F) << 6) |
               (static_cast<unsigned char>(text[offset + 2]) & 0x3F);
    }
    if ((lead & 0xF8) == 0xF0 && offset + 3 < text.size()) {
        width = 4;
        return ((lead & 0x07) << 18) |
               ((static_cast<unsigned char>(text[offset + 1]) & 0x3F) << 12) |
               ((static_cast<unsigned char>(text[offset + 2]) & 0x3F) << 6) |
               (static_cast<unsigned char>(text[offset + 3]) & 0x3F);
    }
    width = 1;
    return lead;
}

bool is_cjk_codepoint(uint32_t cp) {
    return (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x4E00 && cp <= 0x9FFF) ||
           (cp >= 0xF900 && cp <= 0xFAFF);
}

bool is_cjk_punctuation_codepoint(uint32_t cp) {
    return (cp >= 0x3000 && cp <= 0x303F) ||
           (cp >= 0xFF00 && cp <= 0xFFEF);
}

bool is_cyrillic_codepoint(uint32_t cp) {
    return (cp >= 0x0400 && cp <= 0x052F) ||
           (cp >= 0x2DE0 && cp <= 0x2DFF) ||
           (cp >= 0xA640 && cp <= 0xA69F);
}

bool ends_with(const std::string& text, const std::string& suffix) {
    return text.size() >= suffix.size() && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string strip_sentence_terminal_punctuation(const std::string& sentence) {
    static const std::array<std::string, 6> sentence_delimiters = {
        ".", "!", "?", "\xE3\x80\x82", "\xEF\xBC\x81", "\xEF\xBC\x9F"};
    std::string out = trim_copy(sentence);
    for (const auto& delimiter : sentence_delimiters) {
        if (ends_with(out, delimiter)) {
            out.resize(out.size() - delimiter.size());
            return trim_copy(out);
        }
    }
    return out;
}

std::string last_normalized_word(const std::string& text) {
    std::istringstream stream(text);
    std::string word;
    std::string last;
    while (stream >> word) {
        const std::string normalized = normalize_word_for_matching(word);
        if (!normalized.empty()) {
            last = normalized;
        }
    }
    return last;
}

bool starts_with_ascii_lower_word(const std::string& text) {
    for (unsigned char c : text) {
        if (std::isspace(c)) {
            continue;
        }
        return std::islower(c) != 0;
    }
    return false;
}

std::vector<std::string> split_terminated_sentences(const std::string& text) {
    static const std::array<std::string, 6> sentence_delimiters = {
        ".", "!", "?", "\xE3\x80\x82", "\xEF\xBC\x81", "\xEF\xBC\x9F"};
    std::vector<std::string> sentences;
    size_t sentence_start = 0;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t matched_len = 0;
        for (const auto& delimiter : sentence_delimiters) {
            if (text.compare(pos, delimiter.size(), delimiter) == 0) {
                matched_len = delimiter.size();
                break;
            }
        }
        if (matched_len == 0) {
            ++pos;
            continue;
        }
        const size_t sentence_end = pos + matched_len;
        std::string sentence = trim_copy(text.substr(sentence_start, sentence_end - sentence_start));
        if (!sentence.empty()) {
            sentences.push_back(sentence);
        }
        sentence_start = sentence_end;
        pos = sentence_end;
    }
    if (!trim_copy(text.substr(sentence_start)).empty()) {
        sentences.push_back(trim_copy(text.substr(sentence_start)));
    }
    return sentences;
}

std::string maybe_merge_broken_sentence_boundaries(const std::string& raw_text) {
    std::string text = trim_copy(raw_text);
    if (text.empty()) {
        return text;
    }

    std::vector<std::string> sentences = split_terminated_sentences(text);
    if (sentences.size() < 2) {
        return text;
    }

    std::vector<std::string> merged;
    merged.push_back(sentences.front());
    for (size_t idx = 1; idx < sentences.size(); ++idx) {
        std::string current = trim_copy(sentences[idx]);
        std::string previous_core = strip_sentence_terminal_punctuation(merged.back());
        std::string current_core = strip_sentence_terminal_punctuation(current);
        const std::string previous_last_word = last_normalized_word(previous_core);

        const bool should_merge = starts_with_ascii_lower_word(current_core) ||
                      is_connector_tail_word(previous_last_word);
        if (should_merge) {
            merged.back() = trim_copy(previous_core + " " + current);
        } else {
            merged.push_back(current);
        }
    }

    std::string out;
    for (size_t idx = 0; idx < merged.size(); ++idx) {
        if (idx > 0) {
            out += ' ';
        }
        out += trim_copy(merged[idx]);
    }
    return trim_copy(out);
}

std::string maybe_trim_repeated_filler_tail(const std::string& raw_text) {
    std::string text = trim_copy(raw_text);
    if (text.empty()) {
        return text;
    }

    std::vector<std::string> sentences = split_terminated_sentences(text);
    if (sentences.size() < 3) {
        return text;
    }

    const std::string tail_sentence = sentences.back();
    const std::string tail_core = strip_sentence_terminal_punctuation(tail_sentence);
    if (tail_core.empty()) {
        return text;
    }

    const size_t tail_codepoints = utf8_codepoint_count(tail_core);
    const std::string normalized_tail = normalize_word_for_matching(tail_core);
    if (tail_codepoints != 1 && !is_suspicious_short_tail_word(normalized_tail)) {
        return text;
    }

    size_t repeat_count = 1;
    for (size_t idx = sentences.size() - 1; idx > 0; --idx) {
        if (normalize_word_for_matching(strip_sentence_terminal_punctuation(sentences[idx - 1])) != normalized_tail) {
            break;
        }
        ++repeat_count;
    }
    if (repeat_count < 3) {
        return text;
    }

    sentences.resize(sentences.size() - repeat_count);
    std::string out;
    for (const auto& sentence : sentences) {
        out += sentence;
    }
    return trim_copy(out);
}

std::string maybe_trim_trailing_fragment_sentence(const std::string& raw_text) {
    std::string text = trim_copy(raw_text);
    if (text.empty()) {
        return text;
    }

    size_t sentence_end = text.find_last_of(".!?");
    if (sentence_end == std::string::npos || sentence_end + 1 != text.size()) {
        return text;
    }

    size_t sentence_start = text.find_last_of(".!?", sentence_end > 0 ? sentence_end - 1 : 0);
    sentence_start = (sentence_start == std::string::npos) ? 0 : sentence_start + 1;
    std::string tail = trim_copy(text.substr(sentence_start, sentence_end - sentence_start));
    if (tail.empty()) {
        return text;
    }

    std::istringstream stream(tail);
    std::vector<std::string> words;
    std::string word;
    while (stream >> word) {
        words.push_back(word);
    }
    if (words.size() != 1) {
        return text;
    }

    std::string lower = words.front();
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (!is_ascii_alpha_word(lower) || !is_suspicious_short_tail_word(lower)) {
        return text;
    }

    if (sentence_start == 0) {
        return text;
    }
    return trim_copy(text.substr(0, sentence_start));
}

std::string maybe_trim_repeated_tail_words(const std::string& raw_text) {
    std::string text = trim_copy(raw_text);
    if (text.empty()) {
        return text;
    }

    std::istringstream stream(text);
    std::vector<std::string> words;
    std::string word;
    while (stream >> word) {
        words.push_back(word);
    }
    if (words.size() < 4) {
        return text;
    }

    const std::string tail_word = normalize_word_for_matching(words.back());
    if (tail_word.empty() || !is_suspicious_short_tail_word(tail_word)) {
        return text;
    }

    size_t repeat_count = 1;
    for (size_t idx = words.size() - 1; idx > 0; --idx) {
        if (normalize_word_for_matching(words[idx - 1]) != tail_word) {
            break;
        }
        ++repeat_count;
    }
    if (repeat_count < 3 || words.size() == repeat_count) {
        return text;
    }

    words.resize(words.size() - repeat_count);
    std::string out;
    for (size_t idx = 0; idx < words.size(); ++idx) {
        if (idx > 0) {
            out += ' ';
        }
        out += words[idx];
    }
    return out;
}

std::string maybe_collapse_malformed_connector_pairs(const std::string& raw_text) {
    std::string text = trim_copy(raw_text);
    if (text.empty()) {
        return text;
    }

    std::istringstream stream(text);
    std::vector<std::string> words;
    std::string word;
    while (stream >> word) {
        words.push_back(word);
    }
    if (words.size() < 2) {
        return text;
    }

    static const std::unordered_set<std::string> malformed_pairs = {
        "but and", "and but", "or and", "and or", "but or", "or but", "so and", "and so"};

    std::vector<std::string> cleaned;
    cleaned.reserve(words.size());
    for (size_t idx = 0; idx < words.size(); ++idx) {
        if (idx + 1 < words.size()) {
            const std::string left = normalize_word_for_matching(words[idx]);
            const std::string right = normalize_word_for_matching(words[idx + 1]);
            if (malformed_pairs.find(left + " " + right) != malformed_pairs.end()) {
                cleaned.push_back(words[idx]);
                ++idx;
                continue;
            }
        }
        cleaned.push_back(words[idx]);
    }

    std::string out;
    for (size_t idx = 0; idx < cleaned.size(); ++idx) {
        if (idx > 0) {
            out += ' ';
        }
        out += cleaned[idx];
    }
    return out;
}

std::string sanitize_chinese_transcript_text(const std::string& raw_text) {
    std::string stripped;
    stripped.reserve(raw_text.size());
    for (size_t idx = 0; idx < raw_text.size();) {
        size_t width = 0;
        const uint32_t cp = decode_utf8_codepoint(raw_text, idx, width);
        if (cp < 0x80 && std::isalpha(static_cast<unsigned char>(cp))) {
            idx += width;
            while (idx < raw_text.size()) {
                size_t inner_width = 0;
                const uint32_t inner_cp = decode_utf8_codepoint(raw_text, idx, inner_width);
                if (inner_cp < 0x80 && std::isalpha(static_cast<unsigned char>(inner_cp))) {
                    idx += inner_width;
                    continue;
                }
                break;
            }
            while (idx < raw_text.size() && std::isspace(static_cast<unsigned char>(raw_text[idx]))) {
                ++idx;
            }
            continue;
        }
        if (is_cyrillic_codepoint(cp)) {
            idx += width;
            while (idx < raw_text.size()) {
                size_t inner_width = 0;
                const uint32_t inner_cp = decode_utf8_codepoint(raw_text, idx, inner_width);
                if (is_cyrillic_codepoint(inner_cp)) {
                    idx += inner_width;
                    continue;
                }
                break;
            }
            while (idx < raw_text.size() && std::isspace(static_cast<unsigned char>(raw_text[idx]))) {
                ++idx;
            }
            continue;
        }
        stripped.append(raw_text, idx, width);
        idx += width;
    }

    std::string compact;
    compact.reserve(stripped.size());
    for (size_t idx = 0; idx < stripped.size();) {
        size_t width = 0;
        const uint32_t cp = decode_utf8_codepoint(stripped, idx, width);
        if (cp < 0x80 && std::isspace(static_cast<unsigned char>(cp))) {
            size_t next_idx = idx + width;
            while (next_idx < stripped.size() && std::isspace(static_cast<unsigned char>(stripped[next_idx]))) {
                ++next_idx;
            }

            bool drop_space = false;
            if (!compact.empty() && next_idx < stripped.size()) {
                size_t prev_width = 0;
                size_t prev_offset = compact.size();
                do {
                    --prev_offset;
                } while (prev_offset > 0 && (static_cast<unsigned char>(compact[prev_offset]) & 0xC0) == 0x80);
                const uint32_t prev_cp = decode_utf8_codepoint(compact, prev_offset, prev_width);

                size_t next_width = 0;
                const uint32_t next_cp = decode_utf8_codepoint(stripped, next_idx, next_width);
                const bool prev_cjkish = is_cjk_codepoint(prev_cp) || is_cjk_punctuation_codepoint(prev_cp);
                const bool next_cjkish = is_cjk_codepoint(next_cp) || is_cjk_punctuation_codepoint(next_cp);
                drop_space = prev_cjkish || next_cjkish;
            }

            if (!drop_space && !compact.empty() && compact.back() != ' ') {
                compact.push_back(' ');
            }
            idx = next_idx;
            continue;
        }

        compact.append(stripped, idx, width);
        idx += width;
    }

    std::vector<std::string> sentences = split_terminated_sentences(trim_copy(compact));
    if (sentences.size() >= 2) {
        std::vector<std::string> filtered;
        for (size_t idx = 0; idx < sentences.size(); ++idx) {
            const std::string current_norm = normalize_text_for_comparison(strip_sentence_terminal_punctuation(sentences[idx]));
            if (idx + 1 < sentences.size()) {
                const std::string next_norm = normalize_text_for_comparison(strip_sentence_terminal_punctuation(sentences[idx + 1]));
                if (!current_norm.empty() &&
                    next_norm.size() > current_norm.size() &&
                    next_norm.rfind(current_norm, 0) == 0) {
                    continue;
                }
            }
            filtered.push_back(sentences[idx]);
        }

        std::string rebuilt;
        for (const auto& sentence : filtered) {
            rebuilt += trim_copy(sentence);
        }
        compact = rebuilt;
    }

    return trim_copy(compact);
}

std::string sanitize_english_transcript_text(const std::string& raw_text) {
    std::string stripped;
    stripped.reserve(raw_text.size());
    for (size_t idx = 0; idx < raw_text.size();) {
        size_t width = 0;
        const uint32_t cp = decode_utf8_codepoint(raw_text, idx, width);
        if (is_cjk_codepoint(cp) || is_cjk_punctuation_codepoint(cp) || is_cyrillic_codepoint(cp)) {
            idx += width;
            while (idx < raw_text.size()) {
                size_t inner_width = 0;
                const uint32_t inner_cp = decode_utf8_codepoint(raw_text, idx, inner_width);
                if (is_cjk_codepoint(inner_cp) || is_cjk_punctuation_codepoint(inner_cp) || is_cyrillic_codepoint(inner_cp)) {
                    idx += inner_width;
                    continue;
                }
                break;
            }
            while (idx < raw_text.size() && std::isspace(static_cast<unsigned char>(raw_text[idx]))) {
                ++idx;
            }
            continue;
        }
        stripped.append(raw_text, idx, width);
        idx += width;
    }

    std::string compact = trim_copy(stripped);
    while (!compact.empty()) {
        unsigned char c = static_cast<unsigned char>(compact.front());
        if (std::isalnum(c) || c == '"' || c == '\'') {
            break;
        }
        compact.erase(compact.begin());
        compact = trim_copy(compact);
    }

    for (size_t idx = 0; idx + 1 < compact.size(); ++idx) {
        const unsigned char next = static_cast<unsigned char>(compact[idx + 1]);
        if ((compact[idx] == '.' || compact[idx] == '!' || compact[idx] == '?') && std::isspace(next)) {
            size_t next_idx = idx + 1;
            while (next_idx < compact.size() && std::isspace(static_cast<unsigned char>(compact[next_idx]))) {
                ++next_idx;
            }
            if (next_idx < compact.size() && std::islower(static_cast<unsigned char>(compact[next_idx]))) {
                compact[next_idx] = static_cast<char>(std::toupper(static_cast<unsigned char>(compact[next_idx])));
            }
        }
    }

    return trim_copy(compact);
}

std::string postprocess_transcript_for_language(const std::string& raw_text, const std::string& language_tag) {
    if (language_tag == "Chinese") {
        return sanitize_chinese_transcript_text(raw_text);
    }
    if (language_tag == "English") {
        return sanitize_english_transcript_text(raw_text);
    }
    return trim_copy(raw_text);
}

std::string clean_transcript_text(const std::string& raw_text) {
    std::string out = raw_text;
    const std::array<std::string, 6> control_tokens = {
        "<asr_text>", "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>", "<|im_start|>", "<|im_end|>"};
    for (const auto& token : control_tokens) {
        std::string::size_type pos = 0;
        while ((pos = out.find(token, pos)) != std::string::npos) {
            out.erase(pos, token.size());
        }
    }
    out = trim_copy(out);
    while (!out.empty() && (out.front() == '.' || out.front() == ',' || out.front() == ';' || out.front() == ':' ||
                             out.front() == '!' || out.front() == '?' || std::isspace(static_cast<unsigned char>(out.front())))) {
        out.erase(out.begin());
    }
    out = trim_copy(out);
    while (true) {
        const std::string previous = out;
        out = maybe_merge_broken_sentence_boundaries(out);
        out = maybe_collapse_malformed_connector_pairs(out);
        out = maybe_trim_repeated_filler_tail(out);
        out = maybe_trim_trailing_fragment_sentence(out);
        out = maybe_trim_repeated_tail_words(out);
        out = trim_copy(out);
        if (out == previous) {
            break;
        }
    }
    return trim_copy(out);
}

std::vector<std::string> split_words(const std::string& text) {
    std::istringstream stream(text);
    std::vector<std::string> words;
    std::string word;
    while (stream >> word) {
        words.push_back(word);
    }
    return words;
}

std::string join_words(const std::vector<std::string>& words) {
    std::string out;
    for (size_t i = 0; i < words.size(); ++i) {
        if (i > 0) {
            out += ' ';
        }
        out += words[i];
    }
    return out;
}

size_t suffix_prefix_word_overlap(const std::vector<std::string>& left,
                                  const std::vector<std::string>& right,
                                  size_t max_words) {
    const size_t limit = std::min({left.size(), right.size(), max_words});
    for (size_t overlap = limit; overlap > 0; --overlap) {
        bool same = true;
        const size_t left_base = left.size() - overlap;
        for (size_t i = 0; i < overlap; ++i) {
            if (normalize_word_for_matching(left[left_base + i]) != normalize_word_for_matching(right[i])) {
                same = false;
                break;
            }
        }
        if (same) {
            return overlap;
        }
    }
    return 0;
}

double repeated_word_ratio(const std::vector<std::string>& words) {
    if (words.empty()) {
        return 0.0;
    }
    std::unordered_set<std::string> unique(words.begin(), words.end());
    return 1.0 - (static_cast<double>(unique.size()) / static_cast<double>(words.size()));
}

std::vector<std::string> concat_words(const std::vector<std::string>& prefix,
                                      const std::vector<std::string>& suffix) {
    std::vector<std::string> out = prefix;
    out.insert(out.end(), suffix.begin(), suffix.end());
    return out;
}

std::string tail_words_text(const std::vector<std::string>& words, size_t max_words) {
    if (words.size() <= max_words) {
        return join_words(words);
    }
    return join_words(std::vector<std::string>(words.end() - static_cast<std::ptrdiff_t>(max_words), words.end()));
}

bool should_accept_streaming_candidate(const std::vector<std::string>& current_words,
                                       const std::vector<std::string>& base_words,
                                       const std::vector<std::string>& candidate_words,
                                       const std::vector<std::string>& merged_words,
                                       int32_t rollback_words,
                                       size_t overlap_words) {
    if (candidate_words.empty()) {
        return false;
    }
    if (candidate_words.size() >= 16 && repeated_word_ratio(candidate_words) > 0.45) {
        return false;
    }
    const size_t allowed_shrink = static_cast<size_t>(std::max<int32_t>(8, rollback_words * 2));
    if (!current_words.empty() && merged_words.size() + allowed_shrink < current_words.size()) {
        return false;
    }
    if (!current_words.empty()) {
        const size_t min_expected_overlap = static_cast<size_t>(std::max<int32_t>(2, rollback_words / 2));
        const size_t duplicate_growth = merged_words.size() > current_words.size() ? (merged_words.size() - current_words.size()) : 0;
        const size_t suspicious_growth = static_cast<size_t>(std::max<int32_t>(16, rollback_words * 3));
        if (overlap_words < min_expected_overlap && duplicate_growth > suspicious_growth) {
            return false;
        }
    }
    if (has_suspicious_connector_boundary(base_words, candidate_words, overlap_words)) {
        return false;
    }
    return true;
}

int32_t compute_streaming_decode_cap(size_t audio_seq_len, int32_t max_new_tokens) {
    const int32_t heuristic_cap = static_cast<int32_t>(std::max<size_t>(64, audio_seq_len / 2));
    return std::min(max_new_tokens, heuristic_cap);
}

std::string build_asr_instruction_prompt(bool text_only,
                                         const std::optional<std::string>& prompt_override,
                                         const std::string& transcript_prefix = {}) {
    if (text_only) {
        return prompt_override.value_or("Please answer briefly: hello.");
    }
    if (transcript_prefix.empty()) {
        return "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>Transcribe this audio.<|im_end|>\n<|im_start|>assistant\n";
    }
    return "<|im_start|>user\nPrevious confirmed transcript:\n" + transcript_prefix +
           "\nContinue transcribing the following audio.\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n";
}

std::string extract_language_from_prefix(std::string& text) {
    // Example prefix seen in some generations: "language English ..."
    static const std::regex lang_prefix(R"(^\s*language\s+([A-Za-z]+)\s*)", std::regex_constants::icase);
    std::smatch m;
    if (std::regex_search(text, m, lang_prefix) && m.size() >= 2) {
        const std::string lang = normalize_language_name(m[1].str());
        text.erase(0, static_cast<size_t>(m.position(0) + m.length(0)));
        text = trim_copy(text);
        return lang;
    }
    return {};
}

}  // namespace

int main(int argc, char* argv[]) try {
    if (argc < 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
        print_usage(argv[0]);
        return argc < 2 ? 1 : 0;
    }

    std::vector<std::string> positional;
    std::optional<std::filesystem::path> wav_path;
    std::optional<std::string> device_override;
    std::optional<int32_t> max_new_tokens_override;
    std::optional<std::string> prompt_override;
    StreamingConfig streaming_cfg;
    bool text_only = false;
    bool cache_model = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--wav") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --wav");
            }
            wav_path = std::filesystem::path(argv[++i]);
        } else if (arg == "--device") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --device");
            }
            device_override = std::string(argv[++i]);
        } else if (arg == "--max_new_tokens") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --max_new_tokens");
            }
            max_new_tokens_override = std::stoi(argv[++i]);
        } else if (arg == "--prompt") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --prompt");
            }
            prompt_override = std::string(argv[++i]);
        } else if (arg == "--streaming") {
            streaming_cfg.enabled = true;
        } else if (arg == "--stream_chunk_sec") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --stream_chunk_sec");
            }
            streaming_cfg.chunk_sec = std::stod(argv[++i]);
        } else if (arg == "--stream_window_sec") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --stream_window_sec");
            }
            streaming_cfg.window_sec = std::stod(argv[++i]);
        } else if (arg == "--stream_unfixed_chunk_num") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --stream_unfixed_chunk_num");
            }
            streaming_cfg.unfixed_chunk_num = std::stoi(argv[++i]);
        } else if (arg == "--stream_unfixed_token_num") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --stream_unfixed_token_num");
            }
            streaming_cfg.unfixed_token_num = std::stoi(argv[++i]);
        } else if (arg == "--text-only" || arg == "--text_only") {
            text_only = true;
        } else if (arg == "--cached-model" || arg == "--cache-model") {
            cache_model = true;
        } else if (arg.rfind("--", 0) == 0) {
            throw std::runtime_error("Unknown option: " + arg);
        } else {
            positional.push_back(arg);
        }
    }

    if (positional.empty()) {
        throw std::runtime_error("Missing <TEXT_MODEL_DIR>");
    }
    if (positional.size() > 3) {
        throw std::runtime_error("Too many positional arguments");
    }

    const std::filesystem::path text_model_dir = positional[0];
    const std::filesystem::path audio_model_dir = positional.size() > 1 ? std::filesystem::path(positional[1]) : text_model_dir;
    std::string device = positional.size() > 2 ? positional[2] : "GPU";
    if (device_override.has_value()) {
        device = *device_override;
    }
    const int32_t max_new_tokens = max_new_tokens_override.has_value() ? *max_new_tokens_override : 128;
    if (max_new_tokens <= 0) {
        throw std::runtime_error("--max_new_tokens must be > 0");
    }
    if (streaming_cfg.enabled && text_only) {
        throw std::runtime_error("--streaming is only supported for audio transcription mode");
    }
    if (streaming_cfg.chunk_sec <= 0.0) {
        throw std::runtime_error("--stream_chunk_sec must be > 0");
    }
    if (streaming_cfg.window_sec < streaming_cfg.chunk_sec) {
        throw std::runtime_error("--stream_window_sec must be >= --stream_chunk_sec");
    }
    if (streaming_cfg.unfixed_chunk_num < 0) {
        throw std::runtime_error("--stream_unfixed_chunk_num must be >= 0");
    }
    if (streaming_cfg.unfixed_token_num < 0) {
        throw std::runtime_error("--stream_unfixed_token_num must be >= 0");
    }

    const auto text_loader_cfg = ov::genai::loaders::ModelConfig::from_hf_json(text_model_dir / "config.json");
    const auto audio_loader_cfg = text_only ? text_loader_cfg
                                            : ov::genai::loaders::ModelConfig::from_hf_json(audio_model_dir / "config.json");

    const auto text_cfg = to_text_cfg(text_loader_cfg);
    const auto audio_cfg = to_audio_cfg(audio_loader_cfg);

    std::shared_ptr<ov::Model> text_model;
    std::shared_ptr<ov::Model> audio_model;
    const bool expect_audio_inputs_in_text_model = !text_only;
    const std::filesystem::path text_xml = text_model_dir /
        (expect_audio_inputs_in_text_model ? "modeling_qwen3_asr_text_with_audio.xml"
                                           : "modeling_qwen3_asr_text_only.xml");
    const std::filesystem::path text_bin = text_model_dir /
        (expect_audio_inputs_in_text_model ? "modeling_qwen3_asr_text_with_audio.bin"
                                           : "modeling_qwen3_asr_text_only.bin");
    const std::filesystem::path audio_xml = audio_model_dir / "modeling_qwen3_asr_audio.xml";
    const std::filesystem::path audio_bin = audio_model_dir / "modeling_qwen3_asr_audio.bin";

    auto quant_cfg = ov::genai::modeling::weights::parse_quantization_config_from_env();
    if (quant_cfg.enabled() && quant_cfg.group_size < -1) {
        throw std::runtime_error(
            "OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE must be > -1 when OV_GENAI_INFLIGHT_QUANT_MODE is enabled");
    }
    std::cout << "[quant] enabled=" << (quant_cfg.enabled() ? "true" : "false")
              << ", group_size=" << quant_cfg.group_size << std::endl;

    ov::Core core;
    bool load_text_from_ir = false;
    if (cache_model) {
        if (has_ir_model_pair(text_xml, text_bin)) {
            auto candidate = core.read_model(text_xml.string(), text_bin.string());
            if (!text_model_cache_supports_mode(candidate, expect_audio_inputs_in_text_model)) {
                std::cout << "[cache] skipped incompatible text IR: " << text_xml.string()
                          << " (expected audio_inputs="
                          << (expect_audio_inputs_in_text_model ? "true" : "false") << ")" << std::endl;
            } else {
                text_model = std::move(candidate);
                load_text_from_ir = true;
                std::cout << "[cache] loaded text IR: " << text_xml.string() << std::endl;
            }
        }
    }
    const bool load_audio_from_ir = !text_only && cache_model && has_ir_model_pair(audio_xml, audio_bin);

    const auto build_start = std::chrono::steady_clock::now();
    if (!text_only && load_audio_from_ir) {
        audio_model = core.read_model(audio_xml.string(), audio_bin.string());
        std::cout << "[cache] loaded audio IR: " << audio_xml.string() << std::endl;
    }

    if (!load_text_from_ir) {
        auto text_data = ov::genai::safetensors::load_safetensors(text_model_dir);
        ov::genai::safetensors::SafetensorsWeightSource text_source(std::move(text_data));
        ov::genai::safetensors::SafetensorsWeightFinalizer text_finalizer(quant_cfg);
        text_model = ov::genai::modeling::models::create_qwen3_asr_text_model(
            text_cfg,
            text_source,
            text_finalizer,
            false,
            !text_only);
    }

    if (!text_only && !load_audio_from_ir) {
        auto audio_data = ov::genai::safetensors::load_safetensors(audio_model_dir);
        ov::genai::safetensors::SafetensorsWeightSource audio_source(std::move(audio_data));
        ov::genai::safetensors::SafetensorsWeightFinalizer audio_finalizer;
        audio_model = ov::genai::modeling::models::create_qwen3_asr_audio_encoder_model(
            audio_cfg,
            audio_source,
            audio_finalizer);
    }
    const auto build_end = std::chrono::steady_clock::now();

    if (cache_model) {
        if (!load_text_from_ir) {
            ov::serialize(text_model, text_xml.string(), text_bin.string());
            std::cout << "[cache] saved text IR: " << text_xml.string() << std::endl;
        }

        if (!text_only && !load_audio_from_ir) {
            ov::serialize(audio_model, audio_xml.string(), audio_bin.string());
            std::cout << "[cache] saved audio IR: " << audio_xml.string() << std::endl;
        }
    }

    const auto compile_start = std::chrono::steady_clock::now();
    auto compiled_text = core.compile_model(text_model, device);
    std::optional<ov::CompiledModel> compiled_audio;
    if (!text_only) {
        compiled_audio.emplace(core.compile_model(audio_model, device));
    }
    const auto compile_end = std::chrono::steady_clock::now();

    std::optional<ov::InferRequest> audio_request;
    if (!text_only) {
        audio_request.emplace(compiled_audio->create_infer_request());
    }
    auto text_request = compiled_text.create_infer_request();

    const size_t batch = 1;
    const size_t mel_bins = static_cast<size_t>(std::max(1, audio_cfg.num_mel_bins));
    ov::Tensor input_audio_features;
    std::optional<ov::genai::WhisperFeatureExtractor> feature_extractor;
    std::vector<float> input_audio_samples;
    double input_audio_duration_seconds = 0.0;
    uint32_t input_audio_sample_rate = 0;
    std::filesystem::path preprocessor_cfg;
    const auto feature_extract_start = std::chrono::steady_clock::now();
    if (!text_only) {
        if (wav_path.has_value()) {
            preprocessor_cfg = audio_model_dir / "preprocessor_config.json";
            if (!std::filesystem::exists(preprocessor_cfg)) {
                preprocessor_cfg = text_model_dir / "preprocessor_config.json";
            }
            feature_extractor.emplace(preprocessor_cfg);
            input_audio_sample_rate = static_cast<uint32_t>(std::max<size_t>(1, feature_extractor->sampling_rate));
            input_audio_samples = read_wav_pcm_mono_or_stereo(*wav_path, input_audio_sample_rate);
            if (input_audio_samples.empty()) {
                throw std::runtime_error("Input wav has no audio samples: " + wav_path->string());
            }
            input_audio_duration_seconds = static_cast<double>(input_audio_samples.size()) /
                                           static_cast<double>(input_audio_sample_rate);
            if (!streaming_cfg.enabled) {
                input_audio_features = make_audio_features_from_pcm(input_audio_samples, *feature_extractor, mel_bins);
            }
        } else {
            input_audio_features = make_audio_features(batch, mel_bins, 300);
            input_audio_duration_seconds = static_cast<double>(input_audio_features.get_shape()[2]) / 100.0;
        }
    }
    const auto feature_extract_end = std::chrono::steady_clock::now();

    ov::genai::Tokenizer tokenizer(text_model_dir, ov::AnyMap{{"fix_mistral_regex", true}});
    const auto vocab = tokenizer.get_vocab();
    const int64_t audio_pad_token_id = find_token_id(vocab, {"<|audio_pad|>"}, -1);
    if (!text_only && audio_pad_token_id < 0) {
        throw std::runtime_error("Failed to find <|audio_pad|> token id in tokenizer vocab");
    }

    int64_t bos_token_id = tokenizer.get_bos_token_id();
    const int64_t eos_token_id = tokenizer.get_eos_token_id();
    if (bos_token_id < 0) {
        bos_token_id = eos_token_id >= 0 ? eos_token_id : 1;
    }

    std::unordered_set<int64_t> excluded_decode_ids;
    add_token_if_found(vocab, "<|audio_pad|>", excluded_decode_ids);
    add_token_if_found(vocab, "<|audio_start|>", excluded_decode_ids);
    add_token_if_found(vocab, "<|audio_end|>", excluded_decode_ids);
    add_token_if_found(vocab, "<|im_start|>", excluded_decode_ids);
    add_token_if_found(vocab, "<|im_end|>", excluded_decode_ids);
    add_token_if_found(vocab, "<non_speech>", excluded_decode_ids);
    add_token_if_found(vocab, "<asr_text>", excluded_decode_ids);
    add_token_if_found(vocab, "<|asr_text|>", excluded_decode_ids);
    for (int i = 1; i <= 27; ++i) {
        add_token_if_found(vocab, "<blank" + std::to_string(i) + ">", excluded_decode_ids);
    }

    const int64_t dot_token_id = find_token_id(vocab, {"."}, -1);
    const int64_t excl_token_id = find_token_id(vocab, {"!"}, -1);
    const int64_t qmark_token_id = find_token_id(vocab, {"?"}, -1);

    std::unordered_set<int64_t> stop_token_ids;
    add_token_if_found(vocab, "<|endoftext|>", stop_token_ids);
    add_token_if_found(vocab, "<|im_end|>", stop_token_ids);
    add_token_if_found(vocab, "<|eot_id|>", stop_token_ids);
    auto run_asr_decode_pass = [&](const ov::Tensor& pass_audio_features,
                                   int32_t requested_max_new_tokens,
                                   const std::string& transcript_prefix,
                                   bool dynamic_decode_cap) -> ASRDecodePassResult {
        ASRDecodePassResult result;
        ov::Tensor audio_embeds;
        ov::Tensor audio_out_lengths;
        size_t audio_seq_len = 0;
        int32_t pass_max_new_tokens = requested_max_new_tokens;

        if (!text_only) {
            const auto audio_encode_start = std::chrono::steady_clock::now();
            const size_t audio_frames = pass_audio_features.get_shape()[2];

            audio_request->set_tensor(ov::genai::modeling::models::Qwen3ASRAudioIO::kInputAudioFeatures, pass_audio_features);
            auto input_lengths = make_i64({batch});
            input_lengths.data<int64_t>()[0] = static_cast<int64_t>(audio_frames);
            audio_request->set_tensor(ov::genai::modeling::models::Qwen3ASRAudioIO::kAudioFeatureLengths, input_lengths);
            audio_request->infer();
            const auto audio_encode_end = std::chrono::steady_clock::now();
            result.audio_encode_ms = elapsed_ms(audio_encode_start, audio_encode_end);

            audio_embeds = audio_request->get_tensor(ov::genai::modeling::models::Qwen3ASRAudioIO::kAudioEmbeds);
            audio_out_lengths = audio_request->get_tensor(ov::genai::modeling::models::Qwen3ASRAudioIO::kAudioOutputLengths);
            result.embeds_shape = audio_embeds.get_shape();
            if (result.embeds_shape.size() != 3) {
                throw std::runtime_error("Audio encoder output rank must be 3");
            }
            audio_seq_len = result.embeds_shape[1];
            result.audio_seq_len = audio_seq_len;
            result.audio_output_length = audio_out_lengths.data<const int64_t>()[0];

            const size_t embed_dim = result.embeds_shape[2];
            if (embed_dim != static_cast<size_t>(text_cfg.hidden_size)) {
                throw std::runtime_error("audio_embeds hidden dimension does not match text hidden_size");
            }
            if (dynamic_decode_cap) {
                pass_max_new_tokens = compute_streaming_decode_cap(audio_seq_len, requested_max_new_tokens);
            }
        }

        std::string instruction_prompt = build_asr_instruction_prompt(text_only, prompt_override, transcript_prefix);
        auto encoded_prompt = tokenizer.encode(instruction_prompt, {ov::genai::add_special_tokens(false)});
        std::vector<int64_t> prompt_ids = tensor_row_to_i64_vector(encoded_prompt.input_ids);

        std::vector<int64_t> input_ids;
        input_ids.reserve(prompt_ids.size() + audio_seq_len + static_cast<size_t>(pass_max_new_tokens));
        if (text_only) {
            input_ids = prompt_ids;
            if (input_ids.empty()) {
                input_ids.push_back(bos_token_id);
            }
        } else {
            bool replaced_audio_placeholder = false;
            for (int64_t tid : prompt_ids) {
                if (!replaced_audio_placeholder && tid == audio_pad_token_id) {
                    input_ids.insert(input_ids.end(), audio_seq_len, audio_pad_token_id);
                    replaced_audio_placeholder = true;
                } else {
                    input_ids.push_back(tid);
                }
            }
            if (!replaced_audio_placeholder) {
                input_ids.insert(input_ids.begin(), audio_seq_len, audio_pad_token_id);
                input_ids.push_back(bos_token_id);
            }
        }
        result.prompt_token_size = input_ids.size();

        std::vector<char> base_audio_pos_flags(input_ids.size(), 0);
        for (size_t i = 0; i < input_ids.size(); ++i) {
            if (input_ids[i] == audio_pad_token_id) {
                base_audio_pos_flags[i] = 1;
            }
        }

        ov::Tensor prompt_audio_embeds;
        ov::Tensor prompt_audio_pos_mask;
        ov::Tensor step_audio_embeds;
        ov::Tensor step_audio_pos_mask;
        if (!text_only) {
            prompt_audio_embeds = make_audio_embeds_for_mask_positions(audio_embeds, base_audio_pos_flags);
            prompt_audio_pos_mask = make_audio_pos_mask_from_flags(base_audio_pos_flags);

            std::vector<char> no_audio_flags(1, 0);
            ov::Tensor empty_audio_embeds(ov::element::f32, ov::Shape{1, 0, result.embeds_shape[2]});
            step_audio_embeds = make_audio_embeds_for_mask_positions(empty_audio_embeds, no_audio_flags);
            step_audio_pos_mask = make_audio_pos_mask_from_flags(no_audio_flags);
        }

        std::vector<int64_t> generated_ids;
        generated_ids.reserve(static_cast<size_t>(pass_max_new_tokens));
        int64_t prev_token_id = std::numeric_limits<int64_t>::min();
        int32_t same_token_run = 0;

        auto should_stop_after_push = [&](int64_t pushed_token_id) -> bool {
            if (pushed_token_id == prev_token_id) {
                ++same_token_run;
            } else {
                prev_token_id = pushed_token_id;
                same_token_run = 1;
            }
            if (same_token_run >= 32) {
                return true;
            }

            if (text_only) {
                if (has_recent_ngram_loop(generated_ids, 8, 3) || has_recent_ngram_loop(generated_ids, 12, 3)) {
                    return true;
                }
                if (generated_ids.size() >= 64) {
                    if (pushed_token_id == dot_token_id || pushed_token_id == excl_token_id || pushed_token_id == qmark_token_id) {
                        return true;
                    }
                }
                return false;
            }

            bool repeated_loop = has_recent_ngram_loop(generated_ids, 8, 3) ||
                                 has_recent_ngram_loop(generated_ids, 12, 3) ||
                                 has_recent_ngram_loop(generated_ids, 16, 2);
            if (!repeated_loop) {
                for (size_t n = 6; n <= 16; ++n) {
                    if (has_recent_ngram_loop(generated_ids, n, 2)) {
                        repeated_loop = true;
                        break;
                    }
                }
            }
            return repeated_loop;
        };

        text_request.reset_state();
        ov::Tensor beam_idx = make_i32({batch}, 0);
        text_request.set_tensor(ov::genai::modeling::models::Qwen3ASRTextIO::kBeamIdx, beam_idx);

        const size_t max_total_seq_len = result.prompt_token_size + static_cast<size_t>(pass_max_new_tokens);
        std::vector<int64_t> attention_mask_storage(max_total_seq_len, 1);
        auto make_attention_mask_view = [&](size_t seq_len) -> ov::Tensor {
            if (seq_len == 0 || seq_len > max_total_seq_len) {
                throw std::runtime_error("Invalid seq_len for attention mask view");
            }
            return ov::Tensor(ov::element::i64, ov::Shape{batch, seq_len}, attention_mask_storage.data());
        };

        const size_t prompt_seq_len = input_ids.size();
        text_request.set_tensor(ov::genai::modeling::models::Qwen3ASRTextIO::kInputIds, make_i64_from_vector(input_ids));
        text_request.set_tensor(ov::genai::modeling::models::Qwen3ASRTextIO::kAttentionMask, make_attention_mask_view(prompt_seq_len));
        text_request.set_tensor(ov::genai::modeling::models::Qwen3ASRTextIO::kPositionIds, make_position_ids_3d(batch, prompt_seq_len));
        if (!text_only) {
            text_request.set_tensor(ov::genai::modeling::models::Qwen3ASRTextIO::kAudioEmbeds, prompt_audio_embeds);
            text_request.set_tensor(ov::genai::modeling::models::Qwen3ASRTextIO::kAudioPosMask, prompt_audio_pos_mask);
        }

        const auto prefill_start = std::chrono::steady_clock::now();
        text_request.infer();
        const auto prefill_end = std::chrono::steady_clock::now();
        result.ttft_ms = elapsed_ms(prefill_start, prefill_end);
        result.infer_steps += 1;

        size_t total_seq_len = prompt_seq_len;
        auto process_logits_and_append = [&](const ov::Tensor& logits) -> bool {
            result.logits_shape = logits.get_shape();
            const int64_t next_token_id = argmax_last_token_id_excluding(logits, excluded_decode_ids);
            if ((eos_token_id >= 0 && next_token_id == eos_token_id) ||
                (stop_token_ids.find(next_token_id) != stop_token_ids.end())) {
                return true;
            }
            generated_ids.push_back(next_token_id);
            return should_stop_after_push(next_token_id);
        };

        bool stop_generation = false;
        {
            ov::Tensor logits = text_request.get_tensor(ov::genai::modeling::models::Qwen3ASRTextIO::kLogits);
            stop_generation = process_logits_and_append(logits);
        }

        ov::Tensor one_token_ids(ov::element::i64, ov::Shape{1, 1});
        ov::Tensor one_token_pos_ids(ov::element::i64, ov::Shape{3, 1, 1});
        int64_t* one_token_pos_ptr = one_token_pos_ids.data<int64_t>();

        while (!stop_generation && generated_ids.size() < static_cast<size_t>(pass_max_new_tokens)) {
            const int64_t token_to_feed = generated_ids.back();
            one_token_ids.data<int64_t>()[0] = token_to_feed;

            total_seq_len += 1;
            const int64_t pos = static_cast<int64_t>(total_seq_len - 1);
            one_token_pos_ptr[0] = pos;
            one_token_pos_ptr[1] = pos;
            one_token_pos_ptr[2] = pos;

            text_request.set_tensor(ov::genai::modeling::models::Qwen3ASRTextIO::kInputIds, one_token_ids);
            text_request.set_tensor(ov::genai::modeling::models::Qwen3ASRTextIO::kAttentionMask, make_attention_mask_view(total_seq_len));
            text_request.set_tensor(ov::genai::modeling::models::Qwen3ASRTextIO::kPositionIds, one_token_pos_ids);
            if (!text_only) {
                text_request.set_tensor(ov::genai::modeling::models::Qwen3ASRTextIO::kAudioEmbeds, step_audio_embeds);
                text_request.set_tensor(ov::genai::modeling::models::Qwen3ASRTextIO::kAudioPosMask, step_audio_pos_mask);
            }

            const auto step_start = std::chrono::steady_clock::now();
            text_request.infer();
            const auto step_end = std::chrono::steady_clock::now();
            result.decode_ms += elapsed_ms(step_start, step_end);
            result.decode_tail_tokens += 1;
            result.infer_steps += 1;

            ov::Tensor logits = text_request.get_tensor(ov::genai::modeling::models::Qwen3ASRTextIO::kLogits);
            if (process_logits_and_append(logits)) {
                stop_generation = true;
                break;
            }
        }

        if (!text_only) {
            while (trim_recent_duplicate_ngram(generated_ids, 6, 16)) {
            }
        }

        result.generated_ids = generated_ids;
        if (!generated_ids.empty()) {
            result.transcript_text = tokenizer.decode(generated_ids, {ov::genai::skip_special_tokens(true)});
        }
        result.transcript_text = clean_transcript_text(result.transcript_text);

        result.language_tag = detect_language_from_tokens(tokenizer, generated_ids);
        const std::string language_from_tokens = detect_language_from_language_prefix_tokens(tokenizer, generated_ids);
        if (!language_from_tokens.empty()) {
            result.language_tag = language_from_tokens;
        }

        std::string cleaned = result.transcript_text;
        const std::string language_from_text = extract_language_from_prefix(cleaned);
        if (!language_from_text.empty()) {
            if (result.language_tag.empty() || result.language_tag == "unknown") {
                result.language_tag = language_from_text;
            }
            result.transcript_text = cleaned;
        }
        if (text_only && result.language_tag == "unknown") {
            result.language_tag = "n/a";
        }
        result.transcript_text = postprocess_transcript_for_language(result.transcript_text, result.language_tag);
        return result;
    };

    double feature_extract_ms = elapsed_ms(feature_extract_start, feature_extract_end);
    double audio_encode_ms = 0.0;
    double ttft_ms = 0.0;
    double decode_ms = 0.0;
    size_t decode_tail_tokens = 0;
    size_t infer_steps = 0;
    double asr_infer_ms = 0.0;
    size_t prompt_token_size = 0;
    ov::Shape logits_shape;
    ASRDecodePassResult final_pass;
    std::string transcript_text;
    std::string language_tag = text_only ? "n/a" : "unknown";
    size_t streaming_chunk_count = 0;

    if (streaming_cfg.enabled) {
        std::vector<std::string> confirmed_words;
        std::vector<std::string> provisional_words;
        std::vector<std::string> current_words;

        if (wav_path.has_value()) {
            const size_t chunk_samples = std::max<size_t>(1, static_cast<size_t>(std::llround(streaming_cfg.chunk_sec * input_audio_sample_rate)));
            const size_t window_samples = std::max<size_t>(chunk_samples, static_cast<size_t>(std::llround(streaming_cfg.window_sec * input_audio_sample_rate)));
            streaming_chunk_count = (input_audio_samples.size() + chunk_samples - 1) / chunk_samples;

            for (size_t chunk_idx = 0; chunk_idx < streaming_chunk_count; ++chunk_idx) {
                const size_t chunk_end = std::min(input_audio_samples.size(), (chunk_idx + 1) * chunk_samples);
                const size_t chunk_start = (chunk_idx < static_cast<size_t>(streaming_cfg.unfixed_chunk_num))
                    ? 0
                    : (chunk_end > window_samples ? chunk_end - window_samples : 0);
                std::vector<float> chunk_audio(input_audio_samples.begin() + static_cast<std::ptrdiff_t>(chunk_start),
                                              input_audio_samples.begin() + static_cast<std::ptrdiff_t>(chunk_end));

                const auto chunk_feature_start = std::chrono::steady_clock::now();
                ov::Tensor chunk_features = make_audio_features_from_pcm(chunk_audio, *feature_extractor, mel_bins);
                const auto chunk_feature_end = std::chrono::steady_clock::now();
                feature_extract_ms += elapsed_ms(chunk_feature_start, chunk_feature_end);

                ASRDecodePassResult pass = run_asr_decode_pass(chunk_features,
                                                               max_new_tokens,
                                                               tail_words_text(confirmed_words, 64),
                                                               true);
                final_pass = pass;
                if ((language_tag.empty() || language_tag == "unknown") &&
                    !pass.language_tag.empty() && pass.language_tag != "unknown") {
                    language_tag = pass.language_tag;
                }
                audio_encode_ms += pass.audio_encode_ms;
                if (chunk_idx == 0) {
                    ttft_ms = pass.ttft_ms;
                }
                decode_ms += pass.decode_ms;
                decode_tail_tokens += pass.decode_tail_tokens;
                infer_steps += pass.infer_steps;
                asr_infer_ms += pass.ttft_ms + pass.decode_ms;
                prompt_token_size = pass.prompt_token_size;
                logits_shape = pass.logits_shape;

                auto candidate_words = split_words(pass.transcript_text);
                auto base_words = concat_words(confirmed_words, provisional_words);
                const std::string normalized_base_text = normalize_text_for_comparison(join_words(base_words));
                const std::string normalized_candidate_text = normalize_text_for_comparison(join_words(candidate_words));
                size_t overlap = 0;
                std::vector<std::string> merged_words;
                if (!normalized_base_text.empty() &&
                    normalized_candidate_text.size() > normalized_base_text.size() &&
                    normalized_candidate_text.rfind(normalized_base_text, 0) == 0) {
                    overlap = base_words.size();
                    merged_words = candidate_words;
                } else {
                    overlap = suffix_prefix_word_overlap(base_words,
                                                         candidate_words,
                                                         static_cast<size_t>(std::max<int32_t>(16, streaming_cfg.unfixed_token_num * 8)));
                    merged_words = base_words;
                    merged_words.insert(merged_words.end(), candidate_words.begin() + static_cast<std::ptrdiff_t>(overlap), candidate_words.end());
                }

                if (should_accept_streaming_candidate(current_words,
                                                     base_words,
                                                     candidate_words,
                                                     merged_words,
                                                     streaming_cfg.unfixed_token_num,
                                                     overlap)) {
                    const int32_t rollback_words = (chunk_idx < static_cast<size_t>(streaming_cfg.unfixed_chunk_num))
                        ? static_cast<int32_t>(merged_words.size())
                        : streaming_cfg.unfixed_token_num;
                    if (merged_words.size() > static_cast<size_t>(rollback_words)) {
                        confirmed_words.assign(merged_words.begin(), merged_words.end() - rollback_words);
                        provisional_words.assign(merged_words.end() - rollback_words, merged_words.end());
                    } else {
                        confirmed_words.clear();
                        provisional_words = merged_words;
                    }
                    current_words = merged_words;
                    transcript_text = join_words(current_words);
                    if (!pass.language_tag.empty() && pass.language_tag != "unknown") {
                        language_tag = pass.language_tag;
                    }
                }

                const double window_start_sec = static_cast<double>(chunk_start) / static_cast<double>(input_audio_sample_rate);
                const double window_end_sec = static_cast<double>(chunk_end) / static_cast<double>(input_audio_sample_rate);
                std::cout << "[stream] chunk " << (chunk_idx + 1) << "/" << streaming_chunk_count
                          << " window_s=[" << std::fixed << std::setprecision(2) << window_start_sec << ", " << window_end_sec << "]"
                          << " text=" << transcript_text << std::endl;
            }
        } else {
            const size_t total_frames = input_audio_features.get_shape()[2];
            const size_t chunk_frames = std::max<size_t>(1, static_cast<size_t>(std::llround(streaming_cfg.chunk_sec * 100.0)));
            const size_t window_frames = std::max<size_t>(chunk_frames, static_cast<size_t>(std::llround(streaming_cfg.window_sec * 100.0)));
            streaming_chunk_count = (total_frames + chunk_frames - 1) / chunk_frames;

            for (size_t chunk_idx = 0; chunk_idx < streaming_chunk_count; ++chunk_idx) {
                const size_t chunk_end = std::min(total_frames, (chunk_idx + 1) * chunk_frames);
                const size_t chunk_start = (chunk_idx < static_cast<size_t>(streaming_cfg.unfixed_chunk_num))
                    ? 0
                    : (chunk_end > window_frames ? chunk_end - window_frames : 0);

                const auto chunk_feature_start = std::chrono::steady_clock::now();
                ov::Tensor chunk_features = slice_audio_feature_window(input_audio_features, chunk_start, chunk_end);
                const auto chunk_feature_end = std::chrono::steady_clock::now();
                feature_extract_ms += elapsed_ms(chunk_feature_start, chunk_feature_end);

                ASRDecodePassResult pass = run_asr_decode_pass(chunk_features,
                                                               max_new_tokens,
                                                               tail_words_text(confirmed_words, 64),
                                                               true);
                final_pass = pass;
                if ((language_tag.empty() || language_tag == "unknown") &&
                    !pass.language_tag.empty() && pass.language_tag != "unknown") {
                    language_tag = pass.language_tag;
                }
                audio_encode_ms += pass.audio_encode_ms;
                if (chunk_idx == 0) {
                    ttft_ms = pass.ttft_ms;
                }
                decode_ms += pass.decode_ms;
                decode_tail_tokens += pass.decode_tail_tokens;
                infer_steps += pass.infer_steps;
                asr_infer_ms += pass.ttft_ms + pass.decode_ms;
                prompt_token_size = pass.prompt_token_size;
                logits_shape = pass.logits_shape;

                auto candidate_words = split_words(pass.transcript_text);
                auto base_words = concat_words(confirmed_words, provisional_words);
                const std::string normalized_base_text = normalize_text_for_comparison(join_words(base_words));
                const std::string normalized_candidate_text = normalize_text_for_comparison(join_words(candidate_words));
                size_t overlap = 0;
                std::vector<std::string> merged_words;
                if (!normalized_base_text.empty() &&
                    normalized_candidate_text.size() > normalized_base_text.size() &&
                    normalized_candidate_text.rfind(normalized_base_text, 0) == 0) {
                    overlap = base_words.size();
                    merged_words = candidate_words;
                } else {
                    overlap = suffix_prefix_word_overlap(base_words,
                                                         candidate_words,
                                                         static_cast<size_t>(std::max<int32_t>(16, streaming_cfg.unfixed_token_num * 8)));
                    merged_words = base_words;
                    merged_words.insert(merged_words.end(), candidate_words.begin() + static_cast<std::ptrdiff_t>(overlap), candidate_words.end());
                }

                if (should_accept_streaming_candidate(current_words,
                                                     base_words,
                                                     candidate_words,
                                                     merged_words,
                                                     streaming_cfg.unfixed_token_num,
                                                     overlap)) {
                    const int32_t rollback_words = (chunk_idx < static_cast<size_t>(streaming_cfg.unfixed_chunk_num))
                        ? static_cast<int32_t>(merged_words.size())
                        : streaming_cfg.unfixed_token_num;
                    if (merged_words.size() > static_cast<size_t>(rollback_words)) {
                        confirmed_words.assign(merged_words.begin(), merged_words.end() - rollback_words);
                        provisional_words.assign(merged_words.end() - rollback_words, merged_words.end());
                    } else {
                        confirmed_words.clear();
                        provisional_words = merged_words;
                    }
                    current_words = merged_words;
                    transcript_text = join_words(current_words);
                    if (!pass.language_tag.empty() && pass.language_tag != "unknown") {
                        language_tag = pass.language_tag;
                    }
                }

                const double window_start_sec = static_cast<double>(chunk_start) / 100.0;
                const double window_end_sec = static_cast<double>(chunk_end) / 100.0;
                std::cout << "[stream] chunk " << (chunk_idx + 1) << "/" << streaming_chunk_count
                          << " window_s=[" << std::fixed << std::setprecision(2) << window_start_sec << ", " << window_end_sec << "]"
                          << " text=" << transcript_text << std::endl;
            }
        }

        if (transcript_text.empty()) {
            transcript_text = final_pass.transcript_text;
        }
        transcript_text = clean_transcript_text(transcript_text);
        if ((language_tag.empty() || language_tag == "unknown") && !final_pass.language_tag.empty()) {
            language_tag = final_pass.language_tag;
        }
        transcript_text = postprocess_transcript_for_language(transcript_text, language_tag);
    } else {
        final_pass = run_asr_decode_pass(input_audio_features, max_new_tokens, {}, false);
        audio_encode_ms = final_pass.audio_encode_ms;
        ttft_ms = final_pass.ttft_ms;
        decode_ms = final_pass.decode_ms;
        decode_tail_tokens = final_pass.decode_tail_tokens;
        infer_steps = final_pass.infer_steps;
        asr_infer_ms = final_pass.ttft_ms + final_pass.decode_ms;
        prompt_token_size = final_pass.prompt_token_size;
        logits_shape = final_pass.logits_shape;
        transcript_text = final_pass.transcript_text;
        language_tag = final_pass.language_tag;
    }

    std::string preview;
    const size_t preview_count = std::min<size_t>(final_pass.generated_ids.size(), 10);
    for (size_t i = 0; i < preview_count; ++i) {
        if (i > 0) {
            preview += " | ";
        }
        preview += std::to_string(final_pass.generated_ids[i]);
        preview += ":";
        preview += tokenizer.decode({final_pass.generated_ids[i]}, {ov::genai::skip_special_tokens(false)});
    }

    std::cout << "Qwen3-ASR smoke run completed" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  device: " << device << std::endl;
    std::cout << "  text_only: " << (text_only ? "true" : "false") << std::endl;
    std::cout << "  streaming: " << (streaming_cfg.enabled ? "true" : "false") << std::endl;
    if (wav_path.has_value()) {
        std::cout << "  wav: " << wav_path->string() << std::endl;
    }
    if (!text_only) {
        std::cout << "  audio_embeds shape: [" << final_pass.embeds_shape[0] << ", " << final_pass.embeds_shape[1] << ", " << final_pass.embeds_shape[2]
                  << "]" << std::endl;
        std::cout << "  audio_output_lengths[0]: " << final_pass.audio_output_length << std::endl;
        if (input_audio_sample_rate > 0) {
            std::cout << "  audio_input_sample_rate_hz: " << input_audio_sample_rate << std::endl;
        }
    }
    if (streaming_cfg.enabled) {
        std::cout << "  streaming_chunks: " << streaming_chunk_count << std::endl;
        std::cout << "  stream_chunk_sec: " << streaming_cfg.chunk_sec << std::endl;
        std::cout << "  stream_window_sec: " << streaming_cfg.window_sec << std::endl;
    }
    std::cout << "  logits shape: [" << logits_shape[0] << ", " << logits_shape[1] << ", " << logits_shape[2] << "]"
              << std::endl;
    std::cout << "  generated tokens: " << final_pass.generated_ids.size() << std::endl;
    std::cout << "  token preview: " << preview << std::endl;
    std::cout << "  language: " << language_tag << std::endl;
    std::cout << "  text: " << transcript_text << std::endl;

    const double model_build_ms = elapsed_ms(build_start, build_end);
    const double model_compile_ms = elapsed_ms(compile_start, compile_end);
    const double tpot_ms = decode_tail_tokens > 0 ? (decode_ms / static_cast<double>(decode_tail_tokens)) : 0.0;
    const double throughput = decode_ms > 0.0 ? (static_cast<double>(decode_tail_tokens) * 1000.0 / decode_ms) : 0.0;
    const double asr_rtf = (!text_only && input_audio_duration_seconds > 0.0)
                               ? (asr_infer_ms / 1000.0) / input_audio_duration_seconds
                               : 0.0;

    size_t output_token_size = final_pass.generated_ids.size();
    if (streaming_cfg.enabled && !transcript_text.empty()) {
        auto final_tokens = tokenizer.encode(transcript_text, {ov::genai::add_special_tokens(false)});
        output_token_size = tensor_row_to_i64_vector(final_tokens.input_ids).size();
    }

    // Keep compatibility with reporting scripts that scan for these labels.
    std::cout << "Prompt token size: " << prompt_token_size << std::endl;
    std::cout << "Output token size: " << output_token_size << std::endl;
    std::cout << "Generate time: " << asr_infer_ms << " ms" << std::endl;
    std::cout << "TTFT: " << ttft_ms << " ms" << std::endl;
    if (decode_tail_tokens > 0) {
        std::cout << "TPOT: " << tpot_ms << " ms" << std::endl;
        std::cout << "Throughput: " << throughput << " tokens/s" << std::endl;
    } else {
        std::cout << "TPOT: N/A" << std::endl;
        std::cout << "Throughput: N/A" << std::endl;
    }

    std::cout << "  perf.build_model_ms: " << model_build_ms << std::endl;
    std::cout << "  perf.compile_model_ms: " << model_compile_ms << std::endl;
    std::cout << "  perf.feature_extract_ms: " << feature_extract_ms << std::endl;
    std::cout << "  perf.audio_encode_ms: " << audio_encode_ms << std::endl;
    if (!text_only) {
        std::cout << "  perf.audio_duration_s: " << input_audio_duration_seconds << std::endl;
        std::cout << "  perf.asr_infer_ms: " << asr_infer_ms << std::endl;
        std::cout << "  perf.asr_rtf: " << asr_rtf << std::endl;
    }
    std::cout << "  perf.ttft_ms: " << ttft_ms << std::endl;
    std::cout << "  perf.decode_ms: " << decode_ms << std::endl;
    std::cout << "  perf.decode_steps: " << decode_tail_tokens << std::endl;
    std::cout << "  perf.infer_steps_total: " << infer_steps << std::endl;
    if (decode_tail_tokens > 0) {
        std::cout << "  perf.tpot_ms: " << tpot_ms << std::endl;
        std::cout << "  perf.throughput_toks_per_s: " << throughput << std::endl;
    } else {
        std::cout << "  perf.tpot_ms: N/A" << std::endl;
        std::cout << "  perf.throughput_toks_per_s: N/A" << std::endl;
    }

    return 0;
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
}
