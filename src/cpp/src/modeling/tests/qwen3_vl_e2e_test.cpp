// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>

#include "openvino/genai/tokenizer.hpp"
#include "safetensors_utils/safetensors_loader.hpp"
#include "safetensors_utils/safetensors_weight_finalizer.hpp"
#include "safetensors_utils/safetensors_weight_source.hpp"

#include "modeling/models/qwen3_vl/modeling_qwen3_vl_text.hpp"
#include "modeling/models/qwen3_vl/processing_qwen3_vl.hpp"
#include "modeling/models/qwen3_vl/modeling_qwen3_vl_vision.hpp"

namespace {

const char* get_env(const char* name) {
    const char* value = std::getenv(name);
    return value && value[0] != '\0' ? value : nullptr;
}

std::string resolve_pos_embed_name(ov::genai::modeling::weights::WeightSource& source) {
    const std::vector<std::string> candidates = {
        "model.visual.pos_embed.weight",
        "visual.pos_embed.weight",
        "pos_embed.weight"
    };
    for (const auto& name : candidates) {
        if (source.has(name)) {
            return name;
        }
    }
    for (const auto& name : source.keys()) {
        if (name.find("pos_embed.weight") != std::string::npos) {
            return name;
        }
    }
    throw std::runtime_error("Failed to locate visual.pos_embed.weight in safetensors");
}

std::string build_prompt(const std::string& user_prompt, int64_t image_tokens) {
    std::string prompt = "<|im_start|>user\n<|vision_start|>";
    for (int64_t i = 0; i < image_tokens; ++i) {
        prompt += "<|image_pad|>";
    }
    prompt += "<|vision_end|>\n";
    prompt += user_prompt;
    prompt += "<|im_end|>\n<|im_start|>assistant\n";
    return prompt;
}

int64_t argmax_last_token(const ov::Tensor& logits) {
    const auto shape = logits.get_shape();
    if (shape.size() != 3) {
        throw std::runtime_error("logits must have shape [B, S, V]");
    }
    if (shape[0] != 1) {
        throw std::runtime_error("Only batch=1 is supported in this test");
    }
    const size_t seq_len = shape[1];
    const size_t vocab = shape[2];
    const size_t offset = (seq_len - 1) * vocab;
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

}  // namespace

TEST(Qwen3VLE2E, PrefillAndDecode) {
    const char* model_dir_env = get_env("QWEN3_VL_MODEL_DIR");
    if (!model_dir_env) {
        GTEST_SKIP() << "QWEN3_VL_MODEL_DIR is not set";
    }
    const std::string device = "GPU";
    

    const std::filesystem::path model_dir = model_dir_env;
    auto cfg = ov::genai::modeling::models::Qwen3VLConfig::from_json_file(model_dir);
    ov::genai::modeling::models::Qwen3VLVisionPreprocessConfig pre_cfg;
    const auto pre_cfg_path = model_dir / "preprocessor_config.json";
    if (std::filesystem::exists(pre_cfg_path)) {
        pre_cfg = ov::genai::modeling::models::Qwen3VLVisionPreprocessConfig::from_json_file(pre_cfg_path);
    }

    auto data = ov::genai::safetensors::load_safetensors(model_dir);
    ov::genai::safetensors::SafetensorsWeightSource source(std::move(data));

    ov::genai::safetensors::SafetensorsWeightFinalizer vision_finalizer;
    auto vision_model = ov::genai::modeling::models::create_qwen3_vl_vision_model(cfg, source, vision_finalizer);

    ov::genai::safetensors::SafetensorsWeightFinalizer text_finalizer;
    auto text_model = ov::genai::modeling::models::create_qwen3_vl_text_model(cfg, source, text_finalizer, false, true);

    ov::Core core;
    auto compiled_vision = core.compile_model(vision_model, device);
    auto compiled_text = core.compile_model(text_model, device);

    ov::Tensor image(ov::element::u8, {1, 32, 32, 3});
    std::memset(image.data(), 127, image.get_byte_size());

    const std::string pos_embed_name = resolve_pos_embed_name(source);
    const ov::Tensor pos_embed_weight = source.get_tensor(pos_embed_name);

    ov::genai::modeling::models::Qwen3VLVisionPreprocessor preprocessor(cfg.vision, pre_cfg);
    auto vision_inputs = preprocessor.preprocess(image, pos_embed_weight);

    auto vision_request = compiled_vision.create_infer_request();
    vision_request.set_tensor(ov::genai::modeling::models::Qwen3VLVisionIO::kPixelValues, vision_inputs.pixel_values);
    vision_request.set_tensor(ov::genai::modeling::models::Qwen3VLVisionIO::kGridThw, vision_inputs.grid_thw);
    vision_request.set_tensor(ov::genai::modeling::models::Qwen3VLVisionIO::kPosEmbeds, vision_inputs.pos_embeds);
    vision_request.set_tensor(ov::genai::modeling::models::Qwen3VLVisionIO::kRotaryCos, vision_inputs.rotary_cos);
    vision_request.set_tensor(ov::genai::modeling::models::Qwen3VLVisionIO::kRotarySin, vision_inputs.rotary_sin);
    vision_request.infer();

    ov::Tensor visual_embeds =
        vision_request.get_tensor(ov::genai::modeling::models::Qwen3VLVisionIO::kVisualEmbeds);
    std::vector<ov::Tensor> deepstack_embeds;
    deepstack_embeds.reserve(cfg.vision.deepstack_visual_indexes.size());
    for (size_t i = 0; i < cfg.vision.deepstack_visual_indexes.size(); ++i) {
        std::string name =
            std::string(ov::genai::modeling::models::Qwen3VLVisionIO::kDeepstackEmbedsPrefix) + "." +
            std::to_string(i);
        deepstack_embeds.push_back(vision_request.get_tensor(name));
    }

    ov::genai::Tokenizer tokenizer(model_dir);
    const int64_t image_tokens =
        ov::genai::modeling::models::Qwen3VLVisionPreprocessor::count_visual_tokens(
            vision_inputs.grid_thw, cfg.vision.spatial_merge_size);
    const std::string prompt = build_prompt("Describe the image.", image_tokens);
    auto tokenized = tokenizer.encode(prompt, ov::genai::add_special_tokens(false));

    auto input_ids = tokenized.input_ids;
    auto attention_mask = tokenized.attention_mask;
    const size_t batch = input_ids.get_shape().at(0);
    const int64_t prompt_len = static_cast<int64_t>(input_ids.get_shape().at(1));

    ov::genai::modeling::models::Qwen3VLInputPlanner planner(cfg);
    auto plan = planner.build_plan(input_ids, &attention_mask, &vision_inputs.grid_thw);

    auto visual_padded =
        ov::genai::modeling::models::Qwen3VLInputPlanner::scatter_visual_embeds(visual_embeds, plan.visual_pos_mask);
    auto deepstack_padded =
        ov::genai::modeling::models::Qwen3VLInputPlanner::scatter_deepstack_embeds(
            deepstack_embeds, plan.visual_pos_mask);

    ov::Tensor beam_idx(ov::element::i32, {batch});
    auto* beam = beam_idx.data<int32_t>();
    for (size_t b = 0; b < batch; ++b) {
        beam[b] = static_cast<int32_t>(b);
    }

    auto text_request = compiled_text.create_infer_request();
    text_request.reset_state();
    text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kInputIds, input_ids);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kAttentionMask, attention_mask);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kPositionIds, plan.position_ids);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kBeamIdx, beam_idx);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kVisualEmbeds, visual_padded);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kVisualPosMask, plan.visual_pos_mask);
    for (size_t i = 0; i < deepstack_padded.size(); ++i) {
        std::string name =
            std::string(ov::genai::modeling::models::Qwen3VLTextIO::kDeepstackEmbedsPrefix) + "." +
            std::to_string(i);
        text_request.set_tensor(name, deepstack_padded[i]);
    }
    text_request.infer();

    ov::Tensor logits = text_request.get_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kLogits);
    const int64_t next_id = argmax_last_token(logits);
    EXPECT_GE(next_id, 0);

    ov::Tensor step_ids(ov::element::i64, {batch, 1});
    auto* step_data = step_ids.data<int64_t>();
    for (size_t b = 0; b < batch; ++b) {
        step_data[b] = next_id;
    }
    ov::Tensor step_mask(ov::element::i64, {batch, 1});
    auto* step_mask_data = step_mask.data<int64_t>();
    for (size_t b = 0; b < batch; ++b) {
        step_mask_data[b] = 1;
    }
    ov::Tensor decode_visual(ov::element::f32, {batch, 1, static_cast<size_t>(cfg.text.hidden_size)});
    std::memset(decode_visual.data(), 0, decode_visual.get_byte_size());
    ov::Tensor decode_visual_mask(ov::element::boolean, {batch, 1});
    std::memset(decode_visual_mask.data(), 0, decode_visual_mask.get_byte_size());

    std::vector<ov::Tensor> decode_deepstack;
    decode_deepstack.reserve(deepstack_padded.size());
    for (size_t i = 0; i < deepstack_padded.size(); ++i) {
        ov::Tensor ds(ov::element::f32, {batch, 1, static_cast<size_t>(cfg.text.hidden_size)});
        std::memset(ds.data(), 0, ds.get_byte_size());
        decode_deepstack.push_back(ds);
    }

    auto position_ids =
        ov::genai::modeling::models::Qwen3VLInputPlanner::build_decode_position_ids(
            plan.rope_deltas, prompt_len, 1);

    text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kInputIds, step_ids);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kAttentionMask, step_mask);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kPositionIds, position_ids);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kBeamIdx, beam_idx);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kVisualEmbeds, decode_visual);
    text_request.set_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kVisualPosMask, decode_visual_mask);
    for (size_t i = 0; i < decode_deepstack.size(); ++i) {
        std::string name =
            std::string(ov::genai::modeling::models::Qwen3VLTextIO::kDeepstackEmbedsPrefix) + "." +
            std::to_string(i);
        text_request.set_tensor(name, decode_deepstack[i]);
    }
    text_request.infer();

    logits = text_request.get_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kLogits);
    EXPECT_EQ(logits.get_shape().size(), 3u);
}
