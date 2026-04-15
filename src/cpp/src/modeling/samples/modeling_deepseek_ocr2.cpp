// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>
#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/float16.hpp>

#include "load_image.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "safetensors_utils/safetensors_loader.hpp"
#include "safetensors_utils/safetensors_weight_finalizer.hpp"
#include "safetensors_utils/safetensors_weight_source.hpp"

#include "modeling/models/deepseek_ocr2/modeling_deepseek_ocr2_projector.hpp"
#include "modeling/models/deepseek_ocr2/processing_deepseek_ocr2.hpp"
#include "modeling/models/deepseek_ocr2/modeling_deepseek_qwen2_vision.hpp"
#include "modeling/models/deepseek_ocr2/modeling_deepseek_sam_vit.hpp"
#include "modeling/models/deepseek_v2/modeling_deepseek_v2.hpp"
#include "modeling/weights/quantization_config.hpp"

namespace {

int64_t argmax_last_token(const ov::Tensor& logits) {
    const auto shape = logits.get_shape();
    if (shape.size() != 3) {
        throw std::runtime_error("logits must have shape [B, S, V]");
    }
    if (shape[0] != 1) {
        throw std::runtime_error("Only batch=1 is supported in this sample");
    }
    const size_t seq_len = shape[1];
    const size_t vocab = shape[2];
    const size_t offset = (seq_len - 1) * vocab;

    if (logits.get_element_type() == ov::element::f16) {
        const auto* data = logits.data<const ov::float16>() + offset;
        ov::float16 max_val = data[0];
        size_t max_idx = 0;
        for (size_t i = 1; i < vocab; ++i) {
            if (data[i] > max_val) {
                max_val = data[i];
                max_idx = i;
            }
        }
        return static_cast<int64_t>(max_idx);
    }
    if (logits.get_element_type() == ov::element::bf16) {
        const auto* data = logits.data<const ov::bfloat16>() + offset;
        ov::bfloat16 max_val = data[0];
        size_t max_idx = 0;
        for (size_t i = 1; i < vocab; ++i) {
            if (data[i] > max_val) {
                max_val = data[i];
                max_idx = i;
            }
        }
        return static_cast<int64_t>(max_idx);
    }
    if (logits.get_element_type() != ov::element::f32) {
        throw std::runtime_error("Unsupported logits dtype");
    }
    const auto* data = logits.data<const float>() + offset;
    float max_val = data[0];
    size_t max_idx = 0;
    for (size_t i = 1; i < vocab; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    return static_cast<int64_t>(max_idx);
}

ov::Tensor make_beam_idx(size_t batch) {
    ov::Tensor beam_idx(ov::element::i32, {batch});
    auto* data = beam_idx.data<int32_t>();
    for (size_t i = 0; i < batch; ++i) {
        data[i] = static_cast<int32_t>(i);
    }
    return beam_idx;
}

ov::Tensor make_zero_tensor(const ov::element::Type& type, const ov::Shape& shape) {
    ov::Tensor tensor(type, shape);
    std::memset(tensor.data(), 0, tensor.get_byte_size());
    return tensor;
}

ov::Tensor build_position_ids(size_t batch, size_t seq_len, int64_t start_pos = 0) {
    ov::Tensor pos(ov::element::i64, {batch, seq_len});
    auto* data = pos.data<int64_t>();
    for (size_t b = 0; b < batch; ++b) {
        for (size_t i = 0; i < seq_len; ++i) {
            data[b * seq_len + i] = static_cast<int64_t>(i) + start_pos;
        }
    }
    return pos;
}

ov::Tensor to_f32_copy(const ov::Tensor& src) {
    if (src.get_element_type() == ov::element::f32) {
        return src;
    }
    ov::Tensor dst(ov::element::f32, src.get_shape());
    const size_t count = src.get_size();
    if (src.get_element_type() == ov::element::f16) {
        const auto* in = src.data<const ov::float16>();
        auto* out = dst.data<float>();
        for (size_t i = 0; i < count; ++i) {
            out[i] = static_cast<float>(in[i]);
        }
        return dst;
    }
    if (src.get_element_type() == ov::element::bf16) {
        const auto* in = src.data<const ov::bfloat16>();
        auto* out = dst.data<float>();
        for (size_t i = 0; i < count; ++i) {
            out[i] = static_cast<float>(in[i]);
        }
        return dst;
    }
    throw std::runtime_error("Unsupported query embedding dtype for conversion to f32");
}

ov::Tensor expand_query_embeds(const ov::Tensor& query, size_t batch) {
    const auto shape = query.get_shape();
    if (shape.size() == 3 && shape[0] == batch) {
        return query;
    }
    if (shape.size() == 2) {
        const size_t tokens = shape[0];
        const size_t hidden = shape[1];
        ov::Tensor out(query.get_element_type(), {batch, tokens, hidden});
        const size_t row_bytes = hidden * query.get_element_type().size();
        const size_t block_bytes = tokens * row_bytes;
        const char* src = static_cast<const char*>(query.data());
        char* dst = static_cast<char*>(out.data());
        for (size_t b = 0; b < batch; ++b) {
            std::memcpy(dst + b * block_bytes, src, block_bytes);
        }
        return out;
    }
    if (shape.size() == 3 && shape[0] == 1 && batch > 1) {
        const size_t tokens = shape[1];
        const size_t hidden = shape[2];
        ov::Tensor out(query.get_element_type(), {batch, tokens, hidden});
        const size_t row_bytes = hidden * query.get_element_type().size();
        const size_t block_bytes = tokens * row_bytes;
        const char* src = static_cast<const char*>(query.data());
        char* dst = static_cast<char*>(out.data());
        for (size_t b = 0; b < batch; ++b) {
            std::memcpy(dst + b * block_bytes, src, block_bytes);
        }
        return out;
    }
    throw std::runtime_error("Unsupported query embedding shape for expansion");
}

ov::Tensor scatter_visual_tokens(const std::vector<ov::Tensor>& packed_tokens,
                                 const ov::Tensor& images_seq_mask) {
    const auto mask_shape = images_seq_mask.get_shape();
    if (mask_shape.size() != 2) {
        throw std::runtime_error("images_seq_mask must have shape [B, S]");
    }
    const size_t batch = mask_shape[0];
    const size_t seq_len = mask_shape[1];
    if (packed_tokens.size() != batch) {
        throw std::runtime_error("packed_tokens size must match batch");
    }

    const auto dtype = packed_tokens[0].get_element_type();
    const size_t hidden = packed_tokens[0].get_shape().at(1);
    ov::Tensor out = make_zero_tensor(dtype, {batch, seq_len, hidden});

    const auto* mask = images_seq_mask.data<const uint8_t>();
    for (size_t b = 0; b < batch; ++b) {
        const auto& tokens = packed_tokens[b];
        if (tokens.get_shape().size() != 2 || tokens.get_shape()[1] != hidden) {
            throw std::runtime_error("packed_tokens must have shape [V, H]");
        }
        const size_t visual_tokens = tokens.get_shape()[0];
        size_t mask_count = 0;
        for (size_t s = 0; s < seq_len; ++s) {
            if (mask[b * seq_len + s]) {
                ++mask_count;
            }
        }
        if (mask_count != visual_tokens) {
            throw std::runtime_error("images_seq_mask count does not match visual token length");
        }

        size_t token_idx = 0;
        if (dtype == ov::element::f32) {
            const float* src = tokens.data<const float>();
            float* dst = out.data<float>();
            for (size_t s = 0; s < seq_len; ++s) {
                if (!mask[b * seq_len + s]) {
                    continue;
                }
                std::memcpy(dst + (b * seq_len + s) * hidden,
                            src + token_idx * hidden,
                            hidden * sizeof(float));
                token_idx++;
            }
        } else if (dtype == ov::element::f16) {
            const ov::float16* src = tokens.data<const ov::float16>();
            ov::float16* dst = out.data<ov::float16>();
            for (size_t s = 0; s < seq_len; ++s) {
                if (!mask[b * seq_len + s]) {
                    continue;
                }
                std::memcpy(dst + (b * seq_len + s) * hidden,
                            src + token_idx * hidden,
                            hidden * sizeof(ov::float16));
                token_idx++;
            }
        } else if (dtype == ov::element::bf16) {
            const ov::bfloat16* src = tokens.data<const ov::bfloat16>();
            ov::bfloat16* dst = out.data<ov::bfloat16>();
            for (size_t s = 0; s < seq_len; ++s) {
                if (!mask[b * seq_len + s]) {
                    continue;
                }
                std::memcpy(dst + (b * seq_len + s) * hidden,
                            src + token_idx * hidden,
                            hidden * sizeof(ov::bfloat16));
                token_idx++;
            }
        } else {
            throw std::runtime_error("Unsupported dtype for visual token scatter");
        }
    }
    return out;
}

double elapsed_ms(const std::chrono::steady_clock::time_point& start,
                  const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

int main(int argc, char* argv[]) try {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <MODEL_DIR> <IMAGE_PATH> [PROMPT] [DEVICE] [MAX_NEW_TOKENS]\n";
        return 1;
    }

    const std::filesystem::path model_dir = argv[1];
    const std::filesystem::path image_path = argv[2];
    std::string user_prompt = (argc > 3) ? argv[3] : "<image>\nPlease read the text in the image.";
    const std::string device = (argc > 4) ? argv[4] : "GPU";
    const int max_new_tokens = (argc > 5) ? std::stoi(argv[5]) : 64;

    auto cfg = ov::genai::modeling::models::DeepseekOCR2Config::from_json_file(model_dir);
    auto processor_cfg = ov::genai::modeling::models::DeepseekOCR2ProcessorConfig::from_json_file(model_dir);

    ov::genai::modeling::models::DeepseekOCR2PreprocessConfig pre_cfg;
    pre_cfg.base_size = cfg.vision.image_size;
    pre_cfg.patch_size = processor_cfg.patch_size;
    pre_cfg.downsample_ratio = processor_cfg.downsample_ratio;
    pre_cfg.image_mean = processor_cfg.image_mean;
    pre_cfg.image_std = processor_cfg.image_std;

    ov::genai::modeling::models::DeepseekOCR2ImagePreprocessor preprocessor(pre_cfg);

    auto data = ov::genai::safetensors::load_safetensors(model_dir);
    ov::genai::safetensors::SafetensorsWeightSource source(std::move(data));

    ov::genai::safetensors::SafetensorsWeightFinalizer vision_finalizer;
    auto sam_model = ov::genai::modeling::models::create_deepseek_sam_model(cfg.vision, source, vision_finalizer);
    auto qwen2_model = ov::genai::modeling::models::create_deepseek_qwen2_encoder_model(cfg.vision, source, vision_finalizer);
    auto projector_model = ov::genai::modeling::models::create_deepseek_ocr2_projector_model(cfg.projector, source, vision_finalizer);

    auto quant_cfg = ov::genai::modeling::weights::parse_quantization_config_from_env();
    ov::genai::safetensors::SafetensorsWeightFinalizer text_finalizer(quant_cfg);
    auto text_model = ov::genai::modeling::models::create_deepseek_v2_text_model(cfg.language,
                                                                                 source,
                                                                                 text_finalizer,
                                                                                 false,
                                                                                 true);

    ov::Core core;
    auto compiled_sam = core.compile_model(sam_model, device);
    auto compiled_qwen2 = core.compile_model(qwen2_model, device);
    auto compiled_projector = core.compile_model(projector_model, device);
    auto compiled_text = core.compile_model(text_model, device);

    ov::Tensor image = utils::load_image(image_path);
    const auto preprocess_start = std::chrono::steady_clock::now();
    auto vision_inputs = preprocessor.preprocess(image);
    const auto preprocess_end = std::chrono::steady_clock::now();

    auto sam_request = compiled_sam.create_infer_request();
    sam_request.set_tensor(ov::genai::modeling::models::DeepseekSamIO::kPixelValues,
                           vision_inputs.global_images);
    const auto vision_start = std::chrono::steady_clock::now();
    sam_request.infer();
    const auto vision_end = std::chrono::steady_clock::now();

    ov::Tensor global_feats =
        sam_request.get_tensor(ov::genai::modeling::models::DeepseekSamIO::kVisionFeats);

    const auto query_1024_raw = source.get_tensor(ov::genai::modeling::models::DeepseekOCR2WeightNames::kQuery1024);
    const auto query_768_raw = source.get_tensor(ov::genai::modeling::models::DeepseekOCR2WeightNames::kQuery768);
    ov::Tensor query_1024 = to_f32_copy(query_1024_raw);
    ov::Tensor query_768 = to_f32_copy(query_768_raw);

    auto qwen2_request = compiled_qwen2.create_infer_request();
    qwen2_request.set_tensor(ov::genai::modeling::models::DeepseekQwen2VisionIO::kVisionFeats, global_feats);
    qwen2_request.set_tensor(ov::genai::modeling::models::DeepseekQwen2VisionIO::kQueryEmbeds,
                             expand_query_embeds(query_1024, global_feats.get_shape().at(0)));
    qwen2_request.infer();
    ov::Tensor global_query_feats =
        qwen2_request.get_tensor(ov::genai::modeling::models::DeepseekQwen2VisionIO::kQueryFeats);

    auto projector_request = compiled_projector.create_infer_request();
    projector_request.set_tensor(ov::genai::modeling::models::DeepseekOCR2ProjectorIO::kInput,
                                 global_query_feats);
    projector_request.infer();
    ov::Tensor global_embeds =
        projector_request.get_tensor(ov::genai::modeling::models::DeepseekOCR2ProjectorIO::kOutput);

    ov::Tensor local_embeds;
    bool has_local = false;
    int64_t total_local_tokens = 0;
    for (const auto& tokens : vision_inputs.image_tokens) {
        total_local_tokens += tokens.local_tokens;
    }
    if (total_local_tokens > 0) {
        has_local = true;
        auto sam_local_request = compiled_sam.create_infer_request();
        sam_local_request.set_tensor(ov::genai::modeling::models::DeepseekSamIO::kPixelValues,
                                     vision_inputs.local_images);
        sam_local_request.infer();
        ov::Tensor local_feats =
            sam_local_request.get_tensor(ov::genai::modeling::models::DeepseekSamIO::kVisionFeats);

        auto qwen2_local_request = compiled_qwen2.create_infer_request();
        qwen2_local_request.set_tensor(ov::genai::modeling::models::DeepseekQwen2VisionIO::kVisionFeats,
                                       local_feats);
        qwen2_local_request.set_tensor(ov::genai::modeling::models::DeepseekQwen2VisionIO::kQueryEmbeds,
                                       expand_query_embeds(query_768, local_feats.get_shape().at(0)));
        qwen2_local_request.infer();
        ov::Tensor local_query_feats =
            qwen2_local_request.get_tensor(ov::genai::modeling::models::DeepseekQwen2VisionIO::kQueryFeats);

        auto projector_local_request = compiled_projector.create_infer_request();
        projector_local_request.set_tensor(ov::genai::modeling::models::DeepseekOCR2ProjectorIO::kInput,
                                           local_query_feats);
        projector_local_request.infer();
        local_embeds =
            projector_local_request.get_tensor(ov::genai::modeling::models::DeepseekOCR2ProjectorIO::kOutput);
    }

    const ov::Tensor view_sep = source.get_tensor(ov::genai::modeling::models::DeepseekOCR2WeightNames::kViewSeparator);
    ov::genai::modeling::models::DeepseekOCR2VisionPackager packager(view_sep);

    const ov::Tensor* local_ptr = has_local ? &local_embeds : nullptr;
    auto packed_tokens = packager.pack(global_embeds, local_ptr, vision_inputs.image_tokens);

    ov::genai::Tokenizer tokenizer(model_dir);
    if (user_prompt.find(processor_cfg.image_token) == std::string::npos &&
        !vision_inputs.image_tokens.empty()) {
        user_prompt = processor_cfg.image_token + std::string("\n") + user_prompt;
    }
    auto plan = ov::genai::modeling::models::build_prompt_plan(tokenizer,
                                                               user_prompt,
                                                               vision_inputs.image_tokens,
                                                               processor_cfg,
                                                               true,
                                                               cfg.bos_token_id,
                                                               false,
                                                               cfg.eos_token_id);

    const size_t batch = plan.input_ids.get_shape().at(0);
    const size_t prompt_len = plan.input_ids.get_shape().at(1);
    ov::Tensor position_ids = build_position_ids(batch, prompt_len, 0);
    ov::Tensor visual_padded = scatter_visual_tokens(packed_tokens, plan.images_seq_mask);

    auto beam_idx = make_beam_idx(batch);
    auto text_request = compiled_text.create_infer_request();
    text_request.reset_state();
    text_request.set_tensor(ov::genai::modeling::models::DeepseekV2TextIO::kInputIds, plan.input_ids);
    text_request.set_tensor(ov::genai::modeling::models::DeepseekV2TextIO::kAttentionMask, plan.attention_mask);
    text_request.set_tensor(ov::genai::modeling::models::DeepseekV2TextIO::kPositionIds, position_ids);
    text_request.set_tensor(ov::genai::modeling::models::DeepseekV2TextIO::kBeamIdx, beam_idx);
    text_request.set_tensor(ov::genai::modeling::models::DeepseekV2TextIO::kVisualEmbeds, visual_padded);
    text_request.set_tensor(ov::genai::modeling::models::DeepseekV2TextIO::kImagesSeqMask, plan.images_seq_mask);

    const auto prefill_start = std::chrono::steady_clock::now();
    text_request.infer();
    const auto prefill_end = std::chrono::steady_clock::now();

    ov::Tensor logits = text_request.get_tensor(ov::genai::modeling::models::DeepseekV2TextIO::kLogits);
    int64_t next_id = argmax_last_token(logits);

    std::vector<int64_t> generated;
    generated.reserve(static_cast<size_t>(max_new_tokens));
    generated.push_back(next_id);

    const int64_t eos_token_id = tokenizer.get_eos_token_id();
    ov::Tensor step_ids(ov::element::i64, {batch, 1});
    ov::Tensor step_mask(ov::element::i64, {batch, 1});
    auto* step_mask_data = step_mask.data<int64_t>();
    for (size_t b = 0; b < batch; ++b) {
        step_mask_data[b] = 1;
    }
    ov::Tensor decode_visual = make_zero_tensor(ov::element::f32,
                                                {batch, 1, static_cast<size_t>(cfg.language.hidden_size)});
    ov::Tensor decode_mask = make_zero_tensor(ov::element::boolean, {batch, 1});

    int64_t past_len = static_cast<int64_t>(prompt_len);
    size_t decode_steps = 0;
    const auto decode_start = std::chrono::steady_clock::now();
    for (int step = 1; step < max_new_tokens; ++step) {
        if (eos_token_id >= 0 && next_id == eos_token_id) {
            break;
        }
        auto* step_data = step_ids.data<int64_t>();
        for (size_t b = 0; b < batch; ++b) {
            step_data[b] = next_id;
        }
        auto step_positions = build_position_ids(batch, 1, past_len);

        text_request.set_tensor(ov::genai::modeling::models::DeepseekV2TextIO::kInputIds, step_ids);
        text_request.set_tensor(ov::genai::modeling::models::DeepseekV2TextIO::kAttentionMask, step_mask);
        text_request.set_tensor(ov::genai::modeling::models::DeepseekV2TextIO::kPositionIds, step_positions);
        text_request.set_tensor(ov::genai::modeling::models::DeepseekV2TextIO::kBeamIdx, beam_idx);
        text_request.set_tensor(ov::genai::modeling::models::DeepseekV2TextIO::kVisualEmbeds, decode_visual);
        text_request.set_tensor(ov::genai::modeling::models::DeepseekV2TextIO::kImagesSeqMask, decode_mask);

        text_request.infer();
        logits = text_request.get_tensor(ov::genai::modeling::models::DeepseekV2TextIO::kLogits);
        next_id = argmax_last_token(logits);
        generated.push_back(next_id);
        decode_steps += 1;
        past_len += 1;
    }
    const auto decode_end = std::chrono::steady_clock::now();

    std::string output = tokenizer.decode(generated, ov::genai::skip_special_tokens(true));
    const double preprocess_ms = elapsed_ms(preprocess_start, preprocess_end);
    const double vision_ms = elapsed_ms(vision_start, vision_end);
    const double ttft_ms = elapsed_ms(prefill_start, prefill_end);
    const double decode_ms = elapsed_ms(decode_start, decode_end);
    const double tpot_ms = decode_steps > 0 ? (decode_ms / static_cast<double>(decode_steps)) : 0.0;
    const double throughput = decode_steps > 0 && decode_ms > 0.0
                                  ? (static_cast<double>(decode_steps) * 1000.0 / decode_ms)
                                  : 0.0;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Prompt token size: " << prompt_len << std::endl;
    std::cout << "Output token size: " << generated.size() << std::endl;
    std::cout << "Preprocess time: " << preprocess_ms << " ms" << std::endl;
    std::cout << "Vision encode time: " << vision_ms << " ms" << std::endl;
    std::cout << "TTFT: " << ttft_ms << " ms" << std::endl;
    std::cout << "Decode time: " << decode_ms << " ms" << std::endl;
    if (decode_steps > 0) {
        std::cout << "TPOT: " << tpot_ms << " ms/token" << std::endl;
        std::cout << "Throughput: " << throughput << " tokens/s" << std::endl;
    } else {
        std::cout << "TPOT: N/A" << std::endl;
        std::cout << "Throughput: N/A" << std::endl;
    }
    std::cout << output << std::endl;
    return 0;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {
    }
    return 1;
}

