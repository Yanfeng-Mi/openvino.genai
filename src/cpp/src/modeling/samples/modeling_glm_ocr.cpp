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
#include <unordered_set>
#include <vector>

#include <openvino/openvino.hpp>
#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/float16.hpp>

#include "load_image.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "safetensors_utils/safetensors_loader.hpp"
#include "safetensors_utils/safetensors_weight_finalizer.hpp"
#include "safetensors_utils/safetensors_weight_source.hpp"

#include "modeling/models/glm_ocr/modeling_glm_ocr_text.hpp"
#include "modeling/models/glm_ocr/processing_glm_ocr.hpp"
#include "modeling/models/glm_ocr/modeling_glm_ocr_vision.hpp"
#include "safetensors_utils/quantization_utils.hpp"

namespace {

std::string build_prompt(const std::string& user_prompt, int64_t image_tokens) {
    // GLM-OCR prompt template: [gMASK]<sop><|user|>\n<|begin_of_image|><|image|>...<|end_of_image|>\nPROMPT/nothink<|assistant|>\n
    std::string prompt = "[gMASK]<sop><|user|>\n<|begin_of_image|>";
    prompt.reserve(prompt.size() + static_cast<size_t>(image_tokens) * 10 + user_prompt.size() + 128);
    for (int64_t i = 0; i < image_tokens; ++i) {
        prompt += "<|image|>";
    }
    prompt += "<|end_of_image|>\n";
    prompt += user_prompt;
    prompt += "/nothink<|assistant|>\n<think></think>\n";
    return prompt;
}

// Apply repetition penalty to logits (f32 buffer) in-place.
// For tokens in penalty_set: positive logits are divided by penalty, negative are multiplied.
void apply_repetition_penalty(float* logits,
                              size_t vocab,
                              const std::unordered_set<int64_t>& penalty_set,
                              float penalty) {
    if (penalty == 1.0f || penalty_set.empty()) return;
    for (int64_t token_id : penalty_set) {
        if (static_cast<size_t>(token_id) >= vocab) continue;
        float& val = logits[static_cast<size_t>(token_id)];
        if (val >= 0.0f) {
            val /= penalty;
        } else {
            val *= penalty;
        }
    }
}

int64_t argmax_with_penalty(const ov::Tensor& logits,
                            const std::unordered_set<int64_t>& penalty_set,
                            float repetition_penalty) {
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

    // Convert last-token logits to f32 for penalty application
    std::vector<float> buf(vocab);
    if (logits.get_element_type() == ov::element::f16) {
        const auto* data = logits.data<const ov::float16>() + offset;
        for (size_t i = 0; i < vocab; ++i) buf[i] = static_cast<float>(data[i]);
    } else if (logits.get_element_type() == ov::element::bf16) {
        const auto* data = logits.data<const ov::bfloat16>() + offset;
        for (size_t i = 0; i < vocab; ++i) buf[i] = static_cast<float>(data[i]);
    } else if (logits.get_element_type() == ov::element::f32) {
        const auto* data = logits.data<const float>() + offset;
        std::copy(data, data + vocab, buf.begin());
    } else {
        throw std::runtime_error("Unsupported logits dtype");
    }

    apply_repetition_penalty(buf.data(), vocab, penalty_set, repetition_penalty);

    float max_val = buf[0];
    size_t max_idx = 0;
    for (size_t i = 1; i < vocab; ++i) {
        if (buf[i] > max_val) {
            max_val = buf[i];
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

double elapsed_ms(const std::chrono::steady_clock::time_point& start,
                  const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

bool is_eos(int64_t token_id, const std::vector<int32_t>& eos_ids) {
    for (int32_t eos : eos_ids) {
        if (token_id == static_cast<int64_t>(eos)) {
            return true;
        }
    }
    return false;
}

}  // namespace

int main(int argc, char* argv[]) try {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <MODEL_DIR> <IMAGE_PATH> [PROMPT] [DEVICE] [MAX_NEW_TOKENS] [REPETITION_PENALTY] "
                  << "[VISION_QUANT] [VISION_GS] [VISION_BACKUP] "
                  << "[TEXT_QUANT] [TEXT_GS] [TEXT_BACKUP]\n";
        return 1;
    }

    const std::filesystem::path model_dir = argv[1];
    const std::filesystem::path image_path = argv[2];
    const std::string user_prompt = (argc > 3) ? argv[3] : "<|begin_of_image|><|image|><|end_of_image|>\nConvert the document to markdown./nothink";
    const std::string device = (argc > 4) ? argv[4] : "GPU";
    const int max_new_tokens = (argc > 5) ? std::stoi(argv[5]) : 300;
    const float repetition_penalty = (argc > 6) ? std::stof(argv[6]) : 1.0f;

    // Optional quantization args
    std::string vision_quant_mode = (argc > 7) ? argv[7] : "";
    int vision_group_size = (argc > 8) ? std::stoi(argv[8]) : 128;
    std::string vision_backup_mode = (argc > 9) ? argv[9] : "";

    std::string text_quant_mode = (argc > 10) ? argv[10] : "";
    int text_group_size = (argc > 11) ? std::stoi(argv[11]) : 128;
    std::string text_backup_mode = (argc > 12) ? argv[12] : "";

    auto vision_quant_config = create_quantization_config(vision_quant_mode, vision_group_size, vision_backup_mode);
    auto text_quant_config = create_quantization_config(text_quant_mode, text_group_size, text_backup_mode);

    auto cfg = ov::genai::modeling::models::GlmOcrConfig::from_json_file(model_dir);
    ov::genai::modeling::models::GlmOcrVisionPreprocessConfig pre_cfg;
    const auto pre_cfg_path = model_dir / "preprocessor_config.json";
    if (std::filesystem::exists(pre_cfg_path)) {
        pre_cfg = ov::genai::modeling::models::GlmOcrVisionPreprocessConfig::from_json_file(pre_cfg_path);
    }

    auto data = ov::genai::safetensors::load_safetensors(model_dir);
    ov::genai::safetensors::SafetensorsWeightSource source(std::move(data));
    std::shared_ptr<ov::Model> vision_model;
    {
        ov::genai::safetensors::SafetensorsWeightFinalizer vision_finalizer(vision_quant_config);
        vision_model = ov::genai::modeling::models::create_glm_ocr_vision_model(cfg, source, vision_finalizer);
    }

    std::shared_ptr<ov::Model> text_model;
    {
        ov::genai::safetensors::SafetensorsWeightFinalizer text_finalizer(text_quant_config);
        text_model = ov::genai::modeling::models::create_glm_ocr_text_model(cfg, source, text_finalizer);
    }

    ov::Core core;
    auto compiled_vision = core.compile_model(vision_model, device);
    auto compiled_text = core.compile_model(text_model, device);

    auto image = utils::load_image(image_path);
    {
        auto s = image.get_shape();
        std::cout << "Image shape: [";
        for (size_t i = 0; i < s.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << s[i];
        }
        std::cout << "]" << std::endl;
    }

    ov::genai::modeling::models::GlmOcrVisionPreprocessor preprocessor(cfg.vision, pre_cfg);
    const auto preprocess_start = std::chrono::steady_clock::now();
    auto vision_inputs = preprocessor.preprocess(image);
    const auto preprocess_end = std::chrono::steady_clock::now();

    // Debug: print pixel_values and grid_thw info
    {
        auto pv_shape = vision_inputs.pixel_values.get_shape();
        std::cout << "pixel_values shape: [";
        for (size_t i = 0; i < pv_shape.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << pv_shape[i];
        }
        std::cout << "]" << std::endl;

        auto g = vision_inputs.grid_thw.data<const int64_t>();
        auto gs = vision_inputs.grid_thw.get_shape();
        for (size_t i = 0; i < gs[0]; ++i) {
            std::cout << "grid_thw[" << i << "]: t=" << g[i*3] << " h=" << g[i*3+1] << " w=" << g[i*3+2] << std::endl;
        }

        const float* pv = vision_inputs.pixel_values.data<const float>();
        size_t pv_size = vision_inputs.pixel_values.get_size();
        float pv_min = pv[0], pv_max = pv[0], pv_sum = 0;
        for (size_t i = 0; i < pv_size; ++i) {
            pv_min = std::min(pv_min, pv[i]);
            pv_max = std::max(pv_max, pv[i]);
            pv_sum += pv[i];
        }
        std::cout << "pixel_values: min=" << pv_min << " max=" << pv_max
                  << " mean=" << (pv_sum / static_cast<float>(pv_size))
                  << " first5=[" << pv[0] << "," << pv[1] << "," << pv[2] << "," << pv[3] << "," << pv[4] << "]"
                  << std::endl;
    }

    auto vision_request = compiled_vision.create_infer_request();
    vision_request.set_tensor(ov::genai::modeling::models::GlmOcrVisionIO::kPixelValues, vision_inputs.pixel_values);
    vision_request.set_tensor(ov::genai::modeling::models::GlmOcrVisionIO::kGridThw, vision_inputs.grid_thw);
    vision_request.set_tensor(ov::genai::modeling::models::GlmOcrVisionIO::kRotaryCos, vision_inputs.rotary_cos);
    vision_request.set_tensor(ov::genai::modeling::models::GlmOcrVisionIO::kRotarySin, vision_inputs.rotary_sin);
    const auto vision_start = std::chrono::steady_clock::now();
    vision_request.infer();
    const auto vision_end = std::chrono::steady_clock::now();

    ov::Tensor visual_embeds =
        vision_request.get_tensor(ov::genai::modeling::models::GlmOcrVisionIO::kVisualEmbeds);

    ov::genai::Tokenizer tokenizer(model_dir);
    const int64_t image_tokens =
        ov::genai::modeling::models::GlmOcrVisionPreprocessor::count_visual_tokens(
            vision_inputs.grid_thw, cfg.vision.spatial_merge_size);
    const std::string prompt = build_prompt(user_prompt, image_tokens);
    auto tokenized = tokenizer.encode(prompt, ov::genai::add_special_tokens(false));

    auto input_ids = tokenized.input_ids;
    auto attention_mask = tokenized.attention_mask;
    const size_t batch = input_ids.get_shape().at(0);
    const int64_t prompt_len = static_cast<int64_t>(input_ids.get_shape().at(1));

    ov::genai::modeling::models::GlmOcrInputPlanner planner(cfg);
    auto plan = planner.build_plan(input_ids, &attention_mask, &vision_inputs.grid_thw);

    auto visual_padded =
        ov::genai::modeling::models::GlmOcrInputPlanner::scatter_visual_embeds(visual_embeds, plan.visual_pos_mask);

    auto beam_idx = make_beam_idx(batch);

    auto text_request = compiled_text.create_infer_request();
    text_request.reset_state();
    text_request.set_tensor(ov::genai::modeling::models::GlmOcrTextIO::kInputIds, input_ids);
    text_request.set_tensor(ov::genai::modeling::models::GlmOcrTextIO::kAttentionMask, attention_mask);
    text_request.set_tensor(ov::genai::modeling::models::GlmOcrTextIO::kPositionIds, plan.position_ids);
    text_request.set_tensor(ov::genai::modeling::models::GlmOcrTextIO::kBeamIdx, beam_idx);
    text_request.set_tensor(ov::genai::modeling::models::GlmOcrTextIO::kVisualEmbeds, visual_padded);
    text_request.set_tensor(ov::genai::modeling::models::GlmOcrTextIO::kVisualPosMask, plan.visual_pos_mask);
    const auto prefill_start = std::chrono::steady_clock::now();
    text_request.infer();

    ov::Tensor logits = text_request.get_tensor(ov::genai::modeling::models::GlmOcrTextIO::kLogits);

    // Build penalty set from prompt tokens + generated tokens
    std::unordered_set<int64_t> penalty_set;
    {
        const int64_t* prompt_ids = input_ids.data<const int64_t>();
        for (size_t i = 0; i < input_ids.get_size(); ++i) {
            penalty_set.insert(prompt_ids[i]);
        }
    }

    int64_t next_id = argmax_with_penalty(logits, penalty_set, repetition_penalty);
    const auto prefill_end = std::chrono::steady_clock::now();
    std::vector<int64_t> generated;
    generated.reserve(static_cast<size_t>(max_new_tokens));
    generated.push_back(next_id);
    penalty_set.insert(next_id);

    ov::Tensor step_ids(ov::element::i64, {batch, 1});
    ov::Tensor step_mask(ov::element::i64, {batch, 1});
    auto* step_mask_data = step_mask.data<int64_t>();
    for (size_t b = 0; b < batch; ++b) {
        step_mask_data[b] = 1;
    }

    ov::Tensor decode_visual =
        make_zero_tensor(ov::element::f32, {batch, 1, static_cast<size_t>(cfg.text.hidden_size)});
    ov::Tensor decode_visual_mask = make_zero_tensor(ov::element::boolean, {batch, 1});

    int64_t past_len = prompt_len;
    size_t decode_steps = 0;
    const auto decode_start = std::chrono::steady_clock::now();
    for (int step = 1; step < max_new_tokens; ++step) {
        if (is_eos(next_id, cfg.eos_token_ids)) {
            break;
        }
        auto* step_data = step_ids.data<int64_t>();
        for (size_t b = 0; b < batch; ++b) {
            step_data[b] = next_id;
        }

        auto position_ids =
            ov::genai::modeling::models::GlmOcrInputPlanner::build_decode_position_ids(
                plan.rope_deltas, past_len, 1);

        text_request.set_tensor(ov::genai::modeling::models::GlmOcrTextIO::kInputIds, step_ids);
        text_request.set_tensor(ov::genai::modeling::models::GlmOcrTextIO::kAttentionMask, step_mask);
        text_request.set_tensor(ov::genai::modeling::models::GlmOcrTextIO::kPositionIds, position_ids);
        text_request.set_tensor(ov::genai::modeling::models::GlmOcrTextIO::kBeamIdx, beam_idx);
        text_request.set_tensor(ov::genai::modeling::models::GlmOcrTextIO::kVisualEmbeds, decode_visual);
        text_request.set_tensor(ov::genai::modeling::models::GlmOcrTextIO::kVisualPosMask, decode_visual_mask);

        text_request.infer();
        logits = text_request.get_tensor(ov::genai::modeling::models::GlmOcrTextIO::kLogits);
        next_id = argmax_with_penalty(logits, penalty_set, repetition_penalty);
        generated.push_back(next_id);
        penalty_set.insert(next_id);
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
    std::cout << "Repetition penalty: " << repetition_penalty << std::endl;
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
