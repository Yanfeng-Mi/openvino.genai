// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "openvino/genai/chat_history.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "../../../src/cpp/src/modeling/weights/quantization_config.hpp"

int main(int argc, char* argv[]) try {
    if (3 > argc)
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> \"<PROMPT>\" [DEVICE] [NUM_WARMUP] [NUM_ITER] [MAX_NEW_TOKENS] [COMPRESSION_MODE] [GROUP_SIZE] [BACKUP_MODE]");

    std::string models_path = argv[1];
    std::string prompt = argv[2];
    std::string device = argc > 3 ? argv[3] : "GPU";  // GPU can be used as well
    std::size_t num_warmup = argc > 4 ? std::stoul(argv[4]) : 1;
    std::size_t num_iter = argc > 5 ? std::stoul(argv[5]) : 3;
    std::size_t max_new_tokens = argc > 6 ? std::stoul(argv[6]) : 100;

    // Parse compression config
    ov::genai::modeling::weights::QuantizationConfig quant_config;
    if (argc > 7) {
        std::string mode_str = argv[7];
        if (mode_str == "int4_sym") quant_config.mode = ov::genai::modeling::weights::QuantizationConfig::Mode::INT4_SYM;
        else if (mode_str == "int4_asym") quant_config.mode = ov::genai::modeling::weights::QuantizationConfig::Mode::INT4_ASYM;
        else if (mode_str == "int8_sym") quant_config.mode = ov::genai::modeling::weights::QuantizationConfig::Mode::INT8_SYM;
        else if (mode_str == "int8_asym") quant_config.mode = ov::genai::modeling::weights::QuantizationConfig::Mode::INT8_ASYM;
        else if (mode_str != "none") throw std::runtime_error("Unknown compression mode: " + mode_str);
    }

    if (argc > 8) {
        quant_config.group_size = std::stoi(argv[8]);
    }

    if (argc > 9) {
        std::string mode_str = argv[9];
        if (mode_str == "int4_sym") quant_config.backup_mode = ov::genai::modeling::weights::QuantizationConfig::Mode::INT4_SYM;
        else if (mode_str == "int4_asym") quant_config.backup_mode = ov::genai::modeling::weights::QuantizationConfig::Mode::INT4_ASYM;
        else if (mode_str == "int8_sym") quant_config.backup_mode = ov::genai::modeling::weights::QuantizationConfig::Mode::INT8_SYM;
        else if (mode_str == "int8_asym") quant_config.backup_mode = ov::genai::modeling::weights::QuantizationConfig::Mode::INT8_ASYM;
        else if (mode_str == "none") quant_config.backup_mode = ov::genai::modeling::weights::QuantizationConfig::Mode::NONE;
        else throw std::runtime_error("Unknown backup mode: " + mode_str);
    }

    num_iter = std::max<std::size_t>(num_iter, 1);
    ov::AnyMap pipe_config;
    pipe_config[ov::genai::enable_save_ov_model.name()] = false;

    if (quant_config.mode != ov::genai::modeling::weights::QuantizationConfig::Mode::NONE) {
        pipe_config["QUANTIZATION_CONFIG"] = quant_config;
    }

    ov::genai::LLMPipeline pipe(models_path, device, pipe_config);
    ov::genai::GenerationConfig config = pipe.get_generation_config();
    config.max_new_tokens = max_new_tokens;
    config.do_sample = false;  // keep greedy decoding but preserve model-specific defaults

    auto tokenizer = pipe.get_tokenizer();
    ov::genai::TokenizedInputs input_data;
    if (config.apply_chat_template && !tokenizer.get_chat_template().empty()) {
        ov::genai::ChatHistory history({{{"role", "user"}, {"content", prompt}}});
        constexpr bool add_generation_prompt = true;
        auto templated_prompt = tokenizer.apply_chat_template(history, add_generation_prompt);
        input_data = tokenizer.encode(templated_prompt, ov::genai::add_special_tokens(false));
    } else {
        input_data = tokenizer.encode(prompt, ov::genai::add_special_tokens(true));
    }
    std::size_t prompt_token_size = input_data.input_ids.get_shape()[1];
    std::cout << "Prompt token size: " << prompt_token_size << std::endl;

    for (std::size_t i = 0; i < num_warmup; ++i)
        pipe.generate(prompt, config);

    ov::genai::DecodedResults res = pipe.generate(prompt, config);
    ov::genai::PerfMetrics metrics = res.perf_metrics;
    for (std::size_t i = 0; i + 1 < num_iter; ++i) {
        res = pipe.generate(prompt, config);
        metrics = metrics + res.perf_metrics;
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Output token size: " << res.perf_metrics.get_num_generated_tokens() << std::endl;
    std::cout << "Load time: " << metrics.get_load_time() << " ms" << std::endl;
    std::cout << "Generate time: " << metrics.get_generate_duration().mean << " ± " << metrics.get_generate_duration().std << " ms" << std::endl;
    std::cout << "Tokenization time: " << metrics.get_tokenization_duration().mean << " ± " << metrics.get_tokenization_duration().std << " ms" << std::endl;
    std::cout << "Detokenization time: " << metrics.get_detokenization_duration().mean << " ± " << metrics.get_detokenization_duration().std << " ms" << std::endl;
    std::cout << "TTFT: " << metrics.get_ttft().mean  << " ± " << metrics.get_ttft().std << " ms" << std::endl;
    std::cout << "TPOT: " << metrics.get_tpot().mean  << " ± " << metrics.get_tpot().std << " ms/token " << std::endl;
    std::cout << "Throughput: " << metrics.get_throughput().mean  << " ± " << metrics.get_throughput().std << " tokens/s" << std::endl;
    std::cout << "Generated text: " << res << std::endl;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
