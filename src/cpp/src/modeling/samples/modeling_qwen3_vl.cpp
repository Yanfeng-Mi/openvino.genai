// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
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
#include "utils.hpp"

#include "modeling/models/qwen3_vl/modeling_qwen3_vl_text.hpp"
#include "modeling/models/qwen3_vl/processing_qwen3_vl.hpp"
#include "modeling/models/qwen3_vl/modeling_qwen3_vl_vision.hpp"
#include "safetensors_utils/quantization_utils.hpp"

namespace {

std::string build_prompt(const std::string& user_prompt, int64_t image_tokens) {
    std::string prompt = "<|im_start|>user\n<|vision_start|>";
    prompt.reserve(prompt.size() + static_cast<size_t>(image_tokens) * 12 + user_prompt.size() + 64);
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

static constexpr const char* kPosEmbedCacheResultName = "__pos_embed_cache__";

void embed_pos_embed_in_vision_model(std::shared_ptr<ov::Model>& model,
                                     const ov::Tensor& pos_embed) {
    auto constant = std::make_shared<ov::op::v0::Constant>(pos_embed);
    auto result = std::make_shared<ov::op::v0::Result>(constant);
    result->set_friendly_name(kPosEmbedCacheResultName);
    model->add_results({result});
}

ov::Tensor extract_pos_embed_from_vision_model(std::shared_ptr<ov::Model>& model) {
    for (const auto& result : model->get_results()) {
        if (result->get_friendly_name() == kPosEmbedCacheResultName) {
            auto const_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(
                result->input_value(0).get_node_shared_ptr());
            if (!const_node) {
                throw std::runtime_error("pos_embed cache result is not a Constant");
            }
            ov::Tensor tensor(const_node->get_element_type(), const_node->get_shape());
            std::memcpy(tensor.data(), const_node->get_data_ptr(), tensor.get_byte_size());
            model->remove_result(result);
            return tensor;
        }
    }
    throw std::runtime_error("Cached vision IR does not contain pos_embed data");
}

double elapsed_ms(const std::chrono::steady_clock::time_point& start,
                  const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

bool has_ir_model_pair(const std::filesystem::path& xml_path, const std::filesystem::path& bin_path) {
    return std::filesystem::exists(xml_path) && std::filesystem::is_regular_file(xml_path) &&
           std::filesystem::exists(bin_path) && std::filesystem::is_regular_file(bin_path);
}

std::string quant_mode_cache_token(ov::genai::modeling::weights::QuantizationConfig::Mode mode) {
    using Mode = ov::genai::modeling::weights::QuantizationConfig::Mode;
    switch (mode) {
        case Mode::INT4_SYM:
            return "4s";
        case Mode::INT4_ASYM:
            return "4a";
        case Mode::INT8_SYM:
            return "8s";
        case Mode::INT8_ASYM:
            return "8a";
        case Mode::NONE:
        default:
            return "n";
    }
}

std::string quant_cache_suffix(const ov::genai::modeling::weights::QuantizationConfig& cfg) {
    if (!cfg.enabled()) {
        return "";
    }
    return "_q" + quant_mode_cache_token(cfg.mode) + "_b" + quant_mode_cache_token(cfg.backup_mode) +
           "_g" + std::to_string(cfg.group_size);
}

struct RunStats {
    int64_t prompt_len = 0;
    size_t generated_tokens = 0;
    double preprocess_ms = 0.0;
    double vision_ms = 0.0;
    double prompt_build_ms = 0.0;
    double tokenization_ms = 0.0;
    double plan_ms = 0.0;
    double scatter_ms = 0.0;
    double embeddings_prepare_ms = 0.0;
    double prefill_ms = 0.0;
    double ttft_ms = 0.0;
    double decode_language_ms = 0.0;
    double decode_ms = 0.0;
    double tpot_ms = 0.0;
    double throughput = 0.0;
    std::string output;
};

double average_metric(const std::vector<RunStats>& runs, double RunStats::* member) {
    if (runs.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (const auto& run : runs) {
        sum += run.*member;
    }
    return sum / static_cast<double>(runs.size());
}

double average_count(const std::vector<RunStats>& runs, size_t RunStats::* member) {
    if (runs.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (const auto& run : runs) {
        sum += static_cast<double>(run.*member);
    }
    return sum / static_cast<double>(runs.size());
}

}  // namespace

int main(int argc, char* argv[]) try {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <MODEL_DIR> <IMAGE_PATH> [PROMPT] [DEVICE] [MAX_NEW_TOKENS] "
                  << "[VISION_QUANT] [VISION_GS] [VISION_BACKUP] "
                  << "[TEXT_QUANT] [TEXT_GS] [TEXT_BACKUP] [--cache-model] [--num-warmup N] [--benchmark-runs N] "
                  << "[--dump-runtime-model-dir PATH]\n";
        return 1;
    }

    const std::filesystem::path model_dir = argv[1];
    const std::filesystem::path image_path = argv[2];

    bool cache_model = false;
    size_t num_warmup = 1;
    size_t benchmark_runs = 1;
    std::optional<std::filesystem::path> dump_runtime_model_dir;
    std::vector<std::string> positional_args;
    positional_args.reserve(static_cast<size_t>(std::max(argc - 3, 0)));
    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--cache-model") {
            cache_model = true;
            continue;
        }
        if (arg == "--num-warmup" || arg == "-nw") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + arg);
            }
            num_warmup = static_cast<size_t>(std::stoul(argv[++i]));
            continue;
        }
        if (arg == "--benchmark-runs") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + arg);
            }
            benchmark_runs = static_cast<size_t>(std::stoul(argv[++i]));
            continue;
        }
        if (arg == "--dump-runtime-model-dir") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + arg);
            }
            dump_runtime_model_dir = std::filesystem::path(argv[++i]);
            continue;
        }
        if (arg.rfind("--", 0) == 0) {
            throw std::runtime_error("Unknown option: " + arg);
        }
        positional_args.push_back(arg);
    }
    if (positional_args.size() > 9) {
        throw std::runtime_error("Too many positional arguments. Expected at most 9 after <MODEL_DIR> <IMAGE_PATH>.");
    }

    const std::string user_prompt = positional_args.size() > 0 ? positional_args[0] : "Describe the image.";
    const std::string device = positional_args.size() > 1 ? positional_args[1] : "GPU";
    const int max_new_tokens = positional_args.size() > 2 ? std::stoi(positional_args[2]) : 64;

    // Optional quantization args: "INT4", "INT8", "NONE"
    std::string vision_quant_mode = positional_args.size() > 3 ? positional_args[3] : "";
    int vision_group_size = positional_args.size() > 4 ? std::stoi(positional_args[4]) : 128;
    std::string vision_backup_mode = positional_args.size() > 5 ? positional_args[5] : "";

    std::string text_quant_mode = positional_args.size() > 6 ? positional_args[6] : "";
    int text_group_size = positional_args.size() > 7 ? std::stoi(positional_args[7]) : 128;
    std::string text_backup_mode = positional_args.size() > 8 ? positional_args[8] : "";

    // Parse configs
    auto vision_quant_config = create_quantization_config(vision_quant_mode, vision_group_size, vision_backup_mode);
    auto text_quant_config = create_quantization_config(text_quant_mode, text_group_size, text_backup_mode);

    auto cfg = ov::genai::modeling::models::Qwen3VLConfig::from_json_file(model_dir);
    ov::genai::modeling::models::Qwen3VLVisionPreprocessConfig pre_cfg;
    const auto pre_cfg_path = model_dir / "preprocessor_config.json";
    if (std::filesystem::exists(pre_cfg_path)) {
        pre_cfg = ov::genai::modeling::models::Qwen3VLVisionPreprocessConfig::from_json_file(pre_cfg_path);
    }

    const std::string vision_ir_stem = "qwen3_vl_vision" + quant_cache_suffix(vision_quant_config);
    const std::string text_ir_stem = "qwen3_vl_text" + quant_cache_suffix(text_quant_config);
    const auto vision_xml_path = model_dir / (vision_ir_stem + ".xml");
    const auto vision_bin_path = model_dir / (vision_ir_stem + ".bin");
    const auto text_xml_path = model_dir / (text_ir_stem + ".xml");
    const auto text_bin_path = model_dir / (text_ir_stem + ".bin");

    const bool load_vision_from_ir = cache_model && has_ir_model_pair(vision_xml_path, vision_bin_path);
    const bool load_text_from_ir = cache_model && has_ir_model_pair(text_xml_path, text_bin_path);

    if (cache_model) {
        std::cout << "[cache-model] Vision IR: " << vision_xml_path
                  << (load_vision_from_ir ? " [FOUND]" : " [NOT FOUND]") << std::endl;
        std::cout << "[cache-model] Text   IR: " << text_xml_path
                  << (load_text_from_ir ? " [FOUND]" : " [NOT FOUND]") << std::endl;
    }

    if (dump_runtime_model_dir.has_value()) {
    #ifdef _WIN32
        _putenv_s("OV_GENAI_DUMP_RUNTIME_MODEL_DIR", dump_runtime_model_dir->string().c_str());
    #else
        setenv("OV_GENAI_DUMP_RUNTIME_MODEL_DIR", dump_runtime_model_dir->string().c_str(), 1);
    #endif
        std::cout << "[runtime-model-dump] Enabled at " << *dump_runtime_model_dir << std::endl;
    }
    std::cout << "Warmup iterations: " << num_warmup << std::endl;
    std::cout << "Benchmark iterations: " << benchmark_runs << std::endl;

    ov::Core core;
    core.get_versions("CPU");

    std::unique_ptr<ov::genai::safetensors::SafetensorsWeightSource> source;
    auto ensure_weight_source = [&]() -> ov::genai::safetensors::SafetensorsWeightSource& {
        if (!source) {
            auto data = ov::genai::safetensors::load_safetensors(model_dir);
            source = std::make_unique<ov::genai::safetensors::SafetensorsWeightSource>(std::move(data));
        }
        return *source;
    };

    std::shared_ptr<ov::Model> vision_model;
    ov::Tensor cached_pos_embed;
    if (load_vision_from_ir) {
        try {
            std::cout << "[cache-model] Reusing cached vision IR: " << vision_xml_path << std::endl;
            vision_model = core.read_model(vision_xml_path.string(), vision_bin_path.string());
            cached_pos_embed = extract_pos_embed_from_vision_model(vision_model);
            std::cout << "[cache-model] Extracted pos_embed from vision IR" << std::endl;
        } catch (const std::exception& error) {
            std::cout << "[cache-model] WARNING: Failed to load cached vision IR: " << error.what() << std::endl;
            std::cout << "[cache-model] Falling back to building vision model from safetensors" << std::endl;
            vision_model.reset();
        }
    }
    if (!vision_model) {
        auto& weight_source = ensure_weight_source();
        ov::genai::safetensors::SafetensorsWeightFinalizer vision_finalizer(vision_quant_config);
        vision_model = ov::genai::modeling::models::create_qwen3_vl_vision_model(cfg, weight_source, vision_finalizer);
        if (cache_model) {
            const std::string pos_embed_name = resolve_pos_embed_name(weight_source);
            cached_pos_embed = weight_source.get_tensor(pos_embed_name);
            embed_pos_embed_in_vision_model(vision_model, cached_pos_embed);
            ov::serialize(vision_model, vision_xml_path.string(), vision_bin_path.string());
            std::cout << "[cache-model] Saved vision IR (with pos_embed): " << vision_xml_path << std::endl;
            for (const auto& result : vision_model->get_results()) {
                if (result->get_friendly_name() == kPosEmbedCacheResultName) {
                    vision_model->remove_result(result);
                    break;
                }
            }
        }
    }

    std::shared_ptr<ov::Model> text_model;
    if (load_text_from_ir) {
        try {
            std::cout << "[cache-model] Reusing cached text IR: " << text_xml_path << std::endl;
            text_model = core.read_model(text_xml_path.string(), text_bin_path.string());
        } catch (const std::exception& error) {
            std::cout << "[cache-model] WARNING: Failed to load cached text IR: " << error.what() << std::endl;
            std::cout << "[cache-model] Falling back to building text model from safetensors" << std::endl;
        }
    }
    if (!text_model) {
        auto& weight_source = ensure_weight_source();
        ov::genai::safetensors::SafetensorsWeightFinalizer text_finalizer(text_quant_config);
        text_model = ov::genai::modeling::models::create_qwen3_vl_text_model(cfg, weight_source, text_finalizer);
        if (cache_model) {
            ov::serialize(text_model, text_xml_path.string(), text_bin_path.string());
            std::cout << "[cache-model] Saved text IR: " << text_xml_path << std::endl;
        }
    }
    auto compiled_vision = core.compile_model(vision_model, device);
    auto compiled_text = core.compile_model(text_model, device);
    ov::genai::utils::dump_runtime_model_if_requested(compiled_vision, "modeling_qwen3_vl_vision_compiled");
    ov::genai::utils::dump_runtime_model_if_requested(compiled_text, "modeling_qwen3_vl_text_compiled");

    auto image = utils::load_image(image_path);
    ov::Tensor pos_embed_weight;
    if (cached_pos_embed) {
        pos_embed_weight = cached_pos_embed;
    } else {
        auto& weight_source = ensure_weight_source();
        const std::string pos_embed_name = resolve_pos_embed_name(weight_source);
        pos_embed_weight = weight_source.get_tensor(pos_embed_name);
    }

    ov::genai::modeling::models::Qwen3VLVisionPreprocessor preprocessor(cfg.vision, pre_cfg);
    ov::genai::Tokenizer tokenizer(model_dir);
    ov::genai::modeling::models::Qwen3VLInputPlanner planner(cfg);

    auto run_once = [&]() -> RunStats {
        RunStats stats;
        const auto ttft_start = std::chrono::steady_clock::now();

        const auto preprocess_start = std::chrono::steady_clock::now();
        auto vision_inputs = preprocessor.preprocess(image, pos_embed_weight);
        const auto preprocess_end = std::chrono::steady_clock::now();

        auto vision_request = compiled_vision.create_infer_request();
        vision_request.set_tensor(ov::genai::modeling::models::Qwen3VLVisionIO::kPixelValues, vision_inputs.pixel_values);
        vision_request.set_tensor(ov::genai::modeling::models::Qwen3VLVisionIO::kGridThw, vision_inputs.grid_thw);
        vision_request.set_tensor(ov::genai::modeling::models::Qwen3VLVisionIO::kPosEmbeds, vision_inputs.pos_embeds);
        vision_request.set_tensor(ov::genai::modeling::models::Qwen3VLVisionIO::kRotaryCos, vision_inputs.rotary_cos);
        vision_request.set_tensor(ov::genai::modeling::models::Qwen3VLVisionIO::kRotarySin, vision_inputs.rotary_sin);
        const auto vision_start = std::chrono::steady_clock::now();
        vision_request.infer();
        const auto vision_end = std::chrono::steady_clock::now();

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

        const int64_t image_tokens =
            ov::genai::modeling::models::Qwen3VLVisionPreprocessor::count_visual_tokens(
                vision_inputs.grid_thw, cfg.vision.spatial_merge_size);
        const auto prompt_build_start = std::chrono::steady_clock::now();
        const std::string prompt = build_prompt(user_prompt, image_tokens);
        const auto prompt_build_end = std::chrono::steady_clock::now();

        const auto tokenization_start = std::chrono::steady_clock::now();
        auto tokenized = tokenizer.encode(prompt, ov::genai::add_special_tokens(false));
        const auto tokenization_end = std::chrono::steady_clock::now();

        auto input_ids = tokenized.input_ids;
        auto attention_mask = tokenized.attention_mask;
        const size_t batch = input_ids.get_shape().at(0);
        stats.prompt_len = static_cast<int64_t>(input_ids.get_shape().at(1));

        const auto plan_start = std::chrono::steady_clock::now();
        auto plan = planner.build_plan(input_ids, &attention_mask, &vision_inputs.grid_thw);
        const auto plan_end = std::chrono::steady_clock::now();

        const auto scatter_start = std::chrono::steady_clock::now();
        auto visual_padded =
            ov::genai::modeling::models::Qwen3VLInputPlanner::scatter_visual_embeds(visual_embeds, plan.visual_pos_mask);
        auto deepstack_padded =
            ov::genai::modeling::models::Qwen3VLInputPlanner::scatter_deepstack_embeds(
                deepstack_embeds, plan.visual_pos_mask);
        const auto scatter_end = std::chrono::steady_clock::now();

        auto beam_idx = make_beam_idx(batch);

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
        const auto prefill_start = std::chrono::steady_clock::now();
        text_request.infer();

        ov::Tensor logits = text_request.get_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kLogits);
        int64_t next_id = argmax_last_token(logits);
        const auto prefill_end = std::chrono::steady_clock::now();
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

        ov::Tensor decode_visual =
            make_zero_tensor(ov::element::f32, {batch, 1, static_cast<size_t>(cfg.text.hidden_size)});
        ov::Tensor decode_visual_mask = make_zero_tensor(ov::element::boolean, {batch, 1});
        std::vector<ov::Tensor> decode_deepstack;
        decode_deepstack.reserve(deepstack_padded.size());
        for (size_t i = 0; i < deepstack_padded.size(); ++i) {
            decode_deepstack.push_back(
                make_zero_tensor(ov::element::f32, {batch, 1, static_cast<size_t>(cfg.text.hidden_size)}));
        }

        int64_t past_len = stats.prompt_len;
        size_t decode_steps = 0;
        double decode_language_ms = 0.0;
        const auto decode_start = std::chrono::steady_clock::now();
        for (int step = 1; step < max_new_tokens; ++step) {
            if (eos_token_id >= 0 && next_id == eos_token_id) {
                break;
            }
            auto* step_data = step_ids.data<int64_t>();
            for (size_t b = 0; b < batch; ++b) {
                step_data[b] = next_id;
            }

            auto position_ids =
                ov::genai::modeling::models::Qwen3VLInputPlanner::build_decode_position_ids(
                    plan.rope_deltas, past_len, 1);

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

            const auto decode_language_start = std::chrono::steady_clock::now();
            text_request.infer();
            const auto decode_language_end = std::chrono::steady_clock::now();
            decode_language_ms += elapsed_ms(decode_language_start, decode_language_end);
            logits = text_request.get_tensor(ov::genai::modeling::models::Qwen3VLTextIO::kLogits);
            next_id = argmax_last_token(logits);
            generated.push_back(next_id);
            decode_steps += 1;
            past_len += 1;
        }
        const auto decode_end = std::chrono::steady_clock::now();

        stats.generated_tokens = generated.size();
        stats.output = tokenizer.decode(generated, ov::genai::skip_special_tokens(true));
        stats.preprocess_ms = elapsed_ms(preprocess_start, preprocess_end);
        stats.vision_ms = elapsed_ms(vision_start, vision_end);
        stats.prompt_build_ms = elapsed_ms(prompt_build_start, prompt_build_end);
        stats.tokenization_ms = elapsed_ms(tokenization_start, tokenization_end);
        stats.plan_ms = elapsed_ms(plan_start, plan_end);
        stats.scatter_ms = elapsed_ms(scatter_start, scatter_end);
        stats.embeddings_prepare_ms = elapsed_ms(prompt_build_start, scatter_end);
        stats.prefill_ms = elapsed_ms(prefill_start, prefill_end);
        stats.ttft_ms = elapsed_ms(ttft_start, prefill_end);
        stats.decode_language_ms = decode_language_ms;
        stats.decode_ms = elapsed_ms(decode_start, decode_end);
        stats.tpot_ms = decode_steps > 0 ? (stats.decode_ms / static_cast<double>(decode_steps)) : 0.0;
        stats.throughput = decode_steps > 0 && stats.decode_ms > 0.0
                               ? (static_cast<double>(decode_steps) * 1000.0 / stats.decode_ms)
                               : 0.0;
        return stats;
    };

    for (size_t i = 0; i < num_warmup; ++i) {
        run_once();
    }

    std::vector<RunStats> measured_runs;
    measured_runs.reserve(benchmark_runs);
    for (size_t i = 0; i < benchmark_runs; ++i) {
        measured_runs.push_back(run_once());
    }
    const RunStats& stats = measured_runs.back();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Prompt token size: " << stats.prompt_len << std::endl;
    std::cout << "Output token size: " << average_count(measured_runs, &RunStats::generated_tokens) << std::endl;
    std::cout << "Preprocess time: " << average_metric(measured_runs, &RunStats::preprocess_ms) << " ms" << std::endl;
    std::cout << "Vision encode time: " << average_metric(measured_runs, &RunStats::vision_ms) << " ms" << std::endl;
    std::cout << "Prompt build time: " << average_metric(measured_runs, &RunStats::prompt_build_ms) << " ms" << std::endl;
    std::cout << "Tokenization time: " << average_metric(measured_runs, &RunStats::tokenization_ms) << " ms" << std::endl;
    std::cout << "Planner time: " << average_metric(measured_runs, &RunStats::plan_ms) << " ms" << std::endl;
    std::cout << "Scatter time: " << average_metric(measured_runs, &RunStats::scatter_ms) << " ms" << std::endl;
    std::cout << "Embeddings preparation time: " << average_metric(measured_runs, &RunStats::embeddings_prepare_ms) << " ms" << std::endl;
    std::cout << "Text prefill infer time: " << average_metric(measured_runs, &RunStats::prefill_ms) << " ms" << std::endl;
    std::cout << "TTFT: " << average_metric(measured_runs, &RunStats::ttft_ms) << " ms" << std::endl;
    std::cout << "Decode infer time: " << average_metric(measured_runs, &RunStats::decode_language_ms) << " ms" << std::endl;
    std::cout << "Decode time: " << average_metric(measured_runs, &RunStats::decode_ms) << " ms" << std::endl;
    if (stats.generated_tokens > 1) {
        std::cout << "TPOT: " << average_metric(measured_runs, &RunStats::tpot_ms) << " ms/token" << std::endl;
        std::cout << "Throughput: " << average_metric(measured_runs, &RunStats::throughput) << " tokens/s" << std::endl;
    } else {
        std::cout << "TPOT: N/A" << std::endl;
        std::cout << "Throughput: N/A" << std::endl;
    }
    std::cout << stats.output << std::endl;
    return 0;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {
    }
    return 1;
}
