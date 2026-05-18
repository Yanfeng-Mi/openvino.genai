// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file modeling_wan_dit_layer.cpp
 * @brief Test program for layered WAN DIT model.
 *
 * This sample demonstrates:
 * 1. Loading the WAN 2.1 DIT transformer model
 * 2. Splitting the model into preprocess, block groups, and postprocess
 * 3. Running inference with the layered models
 * 4. Comparing results with the monolithic model for correctness verification
 *
 * Usage:
 *   modeling_wan_dit_layer <TRANSFORMER_DIR> [LAYERS_PER_GROUP] [DEVICE] [DUMP_IR]
 *   modeling_wan_dit_layer <TRANSFORMER_DIR> [LAYERS_PER_GROUP] [DEVICE] [--dump-ir]
 *
 * Arguments:
 *   TRANSFORMER_DIR   - Path to the transformer model directory
 *                       (e.g., D:/data/models/Huggingface/Wan2.1-T2V-1.3B-Diffusers/transformer)
 *   LAYERS_PER_GROUP  - Number of transformer layers per block group (default: 1)
 *   DEVICE            - OpenVINO device to use (default: CPU)
 *   DUMP_IR           - Optional: 1/true/yes/on to dump IR, 0/false/no/off to disable (default: off)
 *   --dump-ir         - Optional flag to dump IR (default: off)
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>
#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/float16.hpp>

#include "safetensors_utils/safetensors_loader.hpp"
#include "safetensors_utils/safetensors_weight_finalizer.hpp"
#include "safetensors_utils/safetensors_weight_source.hpp"

#include "modeling/models/wan/modeling_wan_dit.hpp"
#include "modeling/models/wan/modeling_wan_dit_layered.hpp"
#include "modeling/models/wan/processing_wan.hpp"

namespace {

// ============================================================================
// Utility Functions
// ============================================================================

std::vector<float> tensor_to_float_vector(const ov::Tensor& src) {
    const size_t count = src.get_size();
    std::vector<float> out(count, 0.0f);
    if (count == 0) {
        return out;
    }
    const auto type = src.get_element_type();
    if (type == ov::element::f32) {
        std::memcpy(out.data(), src.data<const float>(), count * sizeof(float));
        return out;
    }
    if (type == ov::element::f16) {
        const auto* data = src.data<const ov::float16>();
        for (size_t i = 0; i < count; ++i) {
            out[i] = static_cast<float>(data[i]);
        }
        return out;
    }
    if (type == ov::element::bf16) {
        const auto* data = src.data<const ov::bfloat16>();
        for (size_t i = 0; i < count; ++i) {
            out[i] = static_cast<float>(data[i]);
        }
        return out;
    }
    if (type == ov::element::i64) {
        const auto* data = src.data<const int64_t>();
        for (size_t i = 0; i < count; ++i) {
            out[i] = static_cast<float>(data[i]);
        }
        return out;
    }
    throw std::runtime_error("Unsupported tensor dtype for conversion");
}

ov::Tensor create_f32_tensor(const ov::Shape& shape, const std::vector<float>& data) {
    ov::Tensor tensor(ov::element::f32, shape);
    if (data.size() != tensor.get_size()) {
        throw std::runtime_error("Data size mismatch for tensor creation");
    }
    std::memcpy(tensor.data(), data.data(), data.size() * sizeof(float));
    return tensor;
}

ov::Tensor create_i64_tensor(const ov::Shape& shape, int64_t value) {
    ov::Tensor tensor(ov::element::i64, shape);
    tensor.data<int64_t>()[0] = value;
    return tensor;
}

std::vector<float> generate_random_data(size_t count, float mean, float std, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(mean, std);
    std::vector<float> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dist(rng);
    }
    return data;
}

std::vector<float> generate_fixed_data(size_t count, float scale = 1.0f) {
    std::vector<float> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = std::sin(static_cast<float>(i)) * scale;
    }
    return data;
}

bool parse_bool_value(const std::string& value, bool* out) {
    std::string lower;
    lower.resize(value.size());
    std::transform(value.begin(), value.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lower == "1" || lower == "true" || lower == "yes" || lower == "on") {
        *out = true;
        return true;
    }
    if (lower == "0" || lower == "false" || lower == "no" || lower == "off") {
        *out = false;
        return true;
    }
    return false;
}

std::filesystem::path get_executable_dir(const char* argv0) {
    std::filesystem::path exe_path(argv0 ? argv0 : "");
    if (exe_path.empty()) {
        return std::filesystem::current_path();
    }
    if (exe_path.is_relative()) {
        exe_path = std::filesystem::absolute(exe_path);
    }
    return exe_path.parent_path();
}

void dump_ir_model(const std::shared_ptr<ov::Model>& model, const std::filesystem::path& xml_path) {
    if (!model) {
        throw std::runtime_error("Cannot dump null model to IR");
    }
    std::cout << "  Dumping IR: " << xml_path << "\n";
    ov::serialize(model, xml_path.string());
}

struct ComparisonResult {
    float max_abs_diff = 0.0f;
    float mean_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    float mean_rel_diff = 0.0f;
    size_t num_elements = 0;
    size_t num_mismatches = 0;  // Count of elements with rel_diff > threshold
};

ComparisonResult compare_tensors(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 float rel_threshold = 1e-3f) {
    ComparisonResult result;
    if (a.size() != b.size()) {
        throw std::runtime_error("Tensor size mismatch in comparison");
    }
    result.num_elements = a.size();

    double sum_abs_diff = 0.0;
    double sum_rel_diff = 0.0;

    for (size_t i = 0; i < a.size(); ++i) {
        float abs_diff = std::abs(a[i] - b[i]);
        float max_val = std::max(std::abs(a[i]), std::abs(b[i]));
        float rel_diff = (max_val > 1e-8f) ? (abs_diff / max_val) : 0.0f;

        result.max_abs_diff = std::max(result.max_abs_diff, abs_diff);
        result.max_rel_diff = std::max(result.max_rel_diff, rel_diff);
        sum_abs_diff += static_cast<double>(abs_diff);
        sum_rel_diff += static_cast<double>(rel_diff);

        if (rel_diff > rel_threshold) {
            result.num_mismatches++;
        }
    }

    result.mean_abs_diff = static_cast<float>(sum_abs_diff / static_cast<double>(a.size()));
    result.mean_rel_diff = static_cast<float>(sum_rel_diff / static_cast<double>(a.size()));

    return result;
}

void print_comparison_result(const std::string& name, const ComparisonResult& result) {
    std::cout << "  " << name << ":\n";
    std::cout << "    Elements: " << result.num_elements << "\n";
    std::cout << "    Max absolute diff: " << std::scientific << std::setprecision(6) << result.max_abs_diff << "\n";
    std::cout << "    Mean absolute diff: " << result.mean_abs_diff << "\n";
    std::cout << "    Max relative diff: " << result.max_rel_diff << "\n";
    std::cout << "    Mean relative diff: " << result.mean_rel_diff << "\n";
    std::cout << "    Mismatches (>1e-3): " << result.num_mismatches << "\n";
    std::cout << std::fixed;
}

// ============================================================================
// Test Configuration
// ============================================================================

struct TestConfig {
    // Input dimensions (small for quick testing)
    int32_t batch_size = 1;
    int32_t latent_frames = 5;     // Number of frames in latent space
    int32_t latent_height = 30;    // Height in latent space (actual height / 8)
    int32_t latent_width = 52;     // Width in latent space (actual width / 8)
    int32_t text_seq_len = 64;     // Text sequence length

    // Random seed for reproducibility
    uint32_t seed = 42;

    // Timestep value for testing
    float timestep = 0.5f;
};

}  // namespace

// ============================================================================
// Main Test Function
// ============================================================================

int main(int argc, char* argv[]) try {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <TRANSFORMER_DIR> [LAYERS_PER_GROUP] [DEVICE] [DUMP_IR] [--dump-ir]\n";
        std::cerr << "\nArguments:\n";
        std::cerr << "  TRANSFORMER_DIR   - Path to the transformer model directory\n";
        std::cerr << "  LAYERS_PER_GROUP  - Number of transformer layers per block group (default: 1)\n";
        std::cerr << "  DEVICE            - OpenVINO device to use (default: CPU)\n";
        std::cerr << "  DUMP_IR           - Optional: 1/true/yes/on to dump IR, 0/false/no/off to disable (default: off)\n";
        std::cerr << "  --dump-ir         - Optional flag to dump IR (default: off)\n";
        std::cerr << "\nExample:\n";
        std::cerr << "  " << argv[0]
                  << " D:/data/models/Huggingface/Wan2.1-T2V-1.3B-Diffusers/transformer 2 CPU --dump-ir\n";
        return 1;
    }

    bool dump_ir = false;
    std::vector<std::string> positional_args;
    positional_args.reserve(static_cast<size_t>(argc));
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dump-ir" || arg == "--dump") {
            dump_ir = true;
            continue;
        }
        const std::string prefix = "--dump-ir=";
        if (arg.rfind(prefix, 0) == 0) {
            const std::string value = arg.substr(prefix.size());
            if (!parse_bool_value(value, &dump_ir)) {
                throw std::runtime_error("Invalid value for --dump-ir: " + value);
            }
            continue;
        }
        if (!arg.empty() && arg[0] == '-') {
            throw std::runtime_error("Unknown option: " + arg);
        }
        positional_args.push_back(std::move(arg));
    }

    if (positional_args.empty()) {
        throw std::runtime_error("TRANSFORMER_DIR is required");
    }

    const std::filesystem::path transformer_dir = positional_args[0];
    const int32_t layers_per_group = (positional_args.size() > 1) ? std::stoi(positional_args[1]) : 1;
    const std::string device = (positional_args.size() > 2) ? positional_args[2] : "CPU";
    if (positional_args.size() > 3) {
        bool positional_dump = false;
        if (!parse_bool_value(positional_args[3], &positional_dump)) {
            throw std::runtime_error("Unexpected extra argument: " + positional_args[3]);
        }
        dump_ir = positional_dump;
    }
    if (positional_args.size() > 4) {
        throw std::runtime_error("Too many arguments");
    }

    std::cout << "========================================\n";
    std::cout << "WAN DIT Layered Model Test\n";
    std::cout << "========================================\n";
    std::cout << "Transformer directory: " << transformer_dir << "\n";
    std::cout << "Layers per group: " << layers_per_group << "\n";
    std::cout << "Device: " << device << "\n";
    std::cout << "Dump IR: " << (dump_ir ? "enabled" : "disabled") << "\n";
    std::cout << "\n";

    // Load configuration
    std::cout << "[1] Loading transformer configuration...\n";
    auto transformer_cfg = ov::genai::modeling::models::WanTransformer3DConfig::from_json_file(
        transformer_dir / "config.json");

    std::cout << "  Model config:\n";
    std::cout << "    num_layers: " << transformer_cfg.num_layers << "\n";
    std::cout << "    num_attention_heads: " << transformer_cfg.num_attention_heads << "\n";
    std::cout << "    attention_head_dim: " << transformer_cfg.attention_head_dim << "\n";
    std::cout << "    inner_dim: " << transformer_cfg.inner_dim() << "\n";
    std::cout << "    in_channels: " << transformer_cfg.in_channels << "\n";
    std::cout << "    text_dim: " << transformer_cfg.text_dim << "\n";
    std::cout << "    ffn_dim: " << transformer_cfg.ffn_dim << "\n";
    std::cout << "\n";

    // Calculate number of block groups
    const int32_t num_groups = (transformer_cfg.num_layers + layers_per_group - 1) / layers_per_group;
    std::cout << "  Layered configuration:\n";
    std::cout << "    Total layers: " << transformer_cfg.num_layers << "\n";
    std::cout << "    Layers per group: " << layers_per_group << "\n";
    std::cout << "    Number of block groups: " << num_groups << "\n";
    std::cout << "\n";

    // Load weights (just for info display)
    std::cout << "[2] Loading model weights...\n";
    auto weights_data = ov::genai::safetensors::load_safetensors(transformer_dir);
    std::cout << "  Loaded " << weights_data.tensor_infos.size() << " weight tensors\n";
    std::cout << "\n";

    // ========================================================================
    // Create Monolithic Model
    // ========================================================================
    std::cout << "[3] Creating monolithic DIT model...\n";

    ov::genai::safetensors::SafetensorsWeightSource mono_source(
        ov::genai::safetensors::load_safetensors(transformer_dir));
    ov::genai::safetensors::SafetensorsWeightFinalizer mono_finalizer;

    auto mono_model = ov::genai::modeling::models::create_wan_dit_model(
        transformer_cfg, mono_source, mono_finalizer);

    std::cout << "  Monolithic model created successfully\n";
    std::cout << "\n";

    // ========================================================================
    // Create Layered Models
    // ========================================================================
    std::cout << "[4] Creating layered DIT models...\n";

    ov::genai::modeling::models::WanDitLayeredConfig layered_cfg;
    layered_cfg.layers_per_block_group = layers_per_group;

    ov::genai::safetensors::SafetensorsWeightSource layered_source(
        ov::genai::safetensors::load_safetensors(transformer_dir));
    ov::genai::safetensors::SafetensorsWeightFinalizer layered_finalizer;

    auto layered_models = ov::genai::modeling::models::create_wan_dit_layered_models(
        transformer_cfg, layered_cfg, layered_source, layered_finalizer);

    std::cout << "  Preprocess model created\n";
    std::cout << "  " << layered_models.num_block_groups() << " block group models created\n";
    std::cout << "  Postprocess model created\n";
    std::cout << "\n";

    if (dump_ir) {
        std::cout << "[4.1] Dumping IR models...\n";
        const std::filesystem::path exe_dir = get_executable_dir(argv[0]);
        const std::string layered_prefix = "wan_dit_layered_lpg" + std::to_string(layers_per_group);

        dump_ir_model(mono_model, exe_dir / "wan_dit_full.xml");
        dump_ir_model(layered_models.preprocess, exe_dir / (layered_prefix + "_preprocess.xml"));

        const int32_t total_layers = transformer_cfg.num_layers;
        const size_t num_groups = layered_models.block_groups.size();
        for (size_t g = 0; g < num_groups; ++g) {
            const int32_t start = static_cast<int32_t>(g) * layers_per_group;
            const int32_t count = std::min(layers_per_group, total_layers - start);
            const std::string name = layered_prefix + "_block_group_" + std::to_string(g) +
                                     "_l" + std::to_string(start) + "_n" + std::to_string(count) + ".xml";
            dump_ir_model(layered_models.block_groups[g], exe_dir / name);
        }

        dump_ir_model(layered_models.postprocess, exe_dir / (layered_prefix + "_postprocess.xml"));
        std::cout << "  IR dump completed in " << exe_dir << "\n\n";
    }

    // ========================================================================
    // Compile Models
    // ========================================================================
    std::cout << "[5] Compiling models...\n";

    ov::Core core;
    ov::AnyMap compile_props;
    if (device.find("GPU") != std::string::npos) {
        compile_props[ov::hint::inference_precision.name()] = ov::element::f32;
    }

    auto compile_model = [&](const std::shared_ptr<ov::Model>& model, const std::string& name) {
        std::cout << "  Compiling " << name << "...\n";
        return compile_props.empty()
            ? core.compile_model(model, device)
            : core.compile_model(model, device, compile_props);
    };

    std::cout << "  Compiler ready; models will be compiled on demand\n";
    std::cout << "\n";

    // ========================================================================
    // Prepare Test Inputs
    // ========================================================================
    std::cout << "[6] Preparing test inputs...\n";

    TestConfig test_cfg;

    // Calculate input shapes
    const size_t batch = static_cast<size_t>(test_cfg.batch_size);
    const size_t channels = static_cast<size_t>(transformer_cfg.in_channels);
    const size_t frames = static_cast<size_t>(test_cfg.latent_frames);
    const size_t height = static_cast<size_t>(test_cfg.latent_height);
    const size_t width = static_cast<size_t>(test_cfg.latent_width);
    const size_t text_seq = static_cast<size_t>(test_cfg.text_seq_len);
    const size_t text_dim = static_cast<size_t>(transformer_cfg.text_dim);

    std::cout << "  Input shapes:\n";
    std::cout << "    hidden_states: [" << batch << ", " << channels << ", "
              << frames << ", " << height << ", " << width << "]\n";
    std::cout << "    timestep: [" << batch << "]\n";
    std::cout << "    encoder_hidden_states: [" << batch << ", " << text_seq << ", " << text_dim << "]\n";
    std::cout << "\n";

    // Generate random input data
    auto hidden_states_data = generate_random_data(
        batch * channels * frames * height * width, 0.0f, 1.0f, test_cfg.seed);
    auto text_embed_data = generate_random_data(
        batch * text_seq * text_dim, 0.0f, 1.0f, test_cfg.seed + 1);

    // Create tensors
    ov::Tensor hidden_states_tensor = create_f32_tensor(
        {batch, channels, frames, height, width}, hidden_states_data);
    ov::Tensor timestep_tensor(ov::element::f32, {batch});
    timestep_tensor.data<float>()[0] = test_cfg.timestep;
    ov::Tensor text_embed_tensor = create_f32_tensor(
        {batch, text_seq, text_dim}, text_embed_data);

    std::cout << "  Test inputs prepared\n";
    std::cout << "\n";

    // ========================================================================
    // Run Monolithic Model
    // ========================================================================
    std::cout << "[7] Running monolithic model inference...\n";

    std::vector<float> mono_output_vec;
    ov::Shape mono_output_shape;
    std::chrono::milliseconds mono_duration(0);
    {
        auto compiled_mono = compile_model(mono_model, "monolithic model");
        auto mono_request = compiled_mono.create_infer_request();
        mono_request.set_tensor("hidden_states", hidden_states_tensor);
        mono_request.set_tensor("timestep", timestep_tensor);
        mono_request.set_tensor("encoder_hidden_states", text_embed_tensor);

        auto mono_start = std::chrono::high_resolution_clock::now();
        mono_request.infer();
        auto mono_end = std::chrono::high_resolution_clock::now();

        auto mono_output = mono_request.get_output_tensor(0);
        mono_output_vec = tensor_to_float_vector(mono_output);
        mono_output_shape = mono_output.get_shape();

        mono_duration = std::chrono::duration_cast<std::chrono::milliseconds>(mono_end - mono_start);
        std::cout << "  Monolithic model inference completed in " << mono_duration.count() << " ms\n";
        std::cout << "  Output shape: " << mono_output_shape << "\n";
        std::cout << "\n";
    }

    mono_model.reset();

    // ========================================================================
    // Run Layered Models
    // ========================================================================
    std::cout << "[8] Running layered model inference...\n";

    auto layered_start = std::chrono::high_resolution_clock::now();

    // Step 1: Preprocess
    std::cout << "  Running preprocess...\n";
    ov::Tensor tokens;
    ov::Tensor rotary_cos;
    ov::Tensor rotary_sin;
    ov::Tensor temb;
    ov::Tensor timestep_proj;
    ov::Tensor text_embeds;
    ov::Tensor ppf_tensor;
    ov::Tensor pph_tensor;
    ov::Tensor ppw_tensor;
    {
        auto compiled_preprocess = compile_model(layered_models.preprocess, "preprocess");
        auto preprocess_request = compiled_preprocess.create_infer_request();
        preprocess_request.set_tensor("hidden_states", hidden_states_tensor);
        preprocess_request.set_tensor("timestep", timestep_tensor);
        preprocess_request.set_tensor("encoder_hidden_states", text_embed_tensor);
        preprocess_request.infer();

        // Get preprocess outputs
        tokens = preprocess_request.get_tensor("tokens");
        rotary_cos = preprocess_request.get_tensor("rotary_cos");
        rotary_sin = preprocess_request.get_tensor("rotary_sin");
        temb = preprocess_request.get_tensor("temb");
        timestep_proj = preprocess_request.get_tensor("timestep_proj");
        text_embeds = preprocess_request.get_tensor("text_embeds");
        ppf_tensor = preprocess_request.get_tensor("ppf");
        pph_tensor = preprocess_request.get_tensor("pph");
        ppw_tensor = preprocess_request.get_tensor("ppw");
    }
    layered_models.preprocess.reset();

    std::cout << "    tokens shape: " << tokens.get_shape() << "\n";
    std::cout << "    rotary_cos shape: " << rotary_cos.get_shape() << "\n";
    std::cout << "    text_embeds shape: " << text_embeds.get_shape() << "\n";

    // Step 2: Block Groups
    ov::Tensor current_hidden_states = tokens;
    for (size_t i = 0; i < layered_models.block_groups.size(); ++i) {
        std::cout << "  Running block_group_" << i << "...\n";

        auto compiled_block_group =
            compile_model(layered_models.block_groups[i], "block_group_" + std::to_string(i));
        auto block_request = compiled_block_group.create_infer_request();
        block_request.set_tensor("hidden_states", current_hidden_states);
        block_request.set_tensor("text_embeds", text_embeds);
        block_request.set_tensor("timestep_proj", timestep_proj);
        block_request.set_tensor("rotary_cos", rotary_cos);
        block_request.set_tensor("rotary_sin", rotary_sin);
        block_request.infer();

        current_hidden_states = block_request.get_tensor("hidden_states_out");
        layered_models.block_groups[i].reset();
    }

    // Step 3: Postprocess
    std::cout << "  Running postprocess...\n";
    ov::Tensor layered_output;
    std::vector<float> layered_output_vec;
    {
        auto compiled_postprocess = compile_model(layered_models.postprocess, "postprocess");
        auto postprocess_request = compiled_postprocess.create_infer_request();
        postprocess_request.set_tensor("hidden_states", current_hidden_states);
        postprocess_request.set_tensor("temb", temb);
        postprocess_request.set_tensor("ppf", ppf_tensor);
        postprocess_request.set_tensor("pph", pph_tensor);
        postprocess_request.set_tensor("ppw", ppw_tensor);
        postprocess_request.infer();

        layered_output = postprocess_request.get_tensor("sample");
        layered_output_vec = tensor_to_float_vector(layered_output);
    }
    layered_models.postprocess.reset();

    auto layered_end = std::chrono::high_resolution_clock::now();
    auto layered_duration = std::chrono::duration_cast<std::chrono::milliseconds>(layered_end - layered_start);

    std::cout << "  Layered model inference completed in " << layered_duration.count() << " ms\n";
    std::cout << "  Output shape: " << layered_output.get_shape() << "\n";
    std::cout << "\n";

    // ========================================================================
    // Compare Results
    // ========================================================================
    std::cout << "[9] Comparing outputs...\n";

    if (mono_output_shape != layered_output.get_shape()) {
        std::cerr << "ERROR: Output shapes do not match!\n";
        std::cerr << "  Monolithic: " << mono_output_shape << "\n";
        std::cerr << "  Layered: " << layered_output.get_shape() << "\n";
        return 1;
    }

    auto comparison = compare_tensors(mono_output_vec, layered_output_vec);
    print_comparison_result("Output comparison", comparison);

    // Determine if test passed
    const float pass_threshold = 1e-4f;  // Allow small numerical differences
    bool passed = (comparison.max_abs_diff < pass_threshold) ||
                  (comparison.max_rel_diff < 1e-3f && comparison.num_mismatches == 0);

    std::cout << "\n";
    std::cout << "========================================\n";
    if (passed) {
        std::cout << "TEST PASSED: Layered model output matches monolithic model!\n";
    } else {
        std::cout << "TEST FAILED: Outputs differ significantly.\n";
        std::cout << "  This may indicate a bug in the layered model implementation.\n";
    }
    std::cout << "========================================\n";

    // Print timing summary
    std::cout << "\nTiming Summary:\n";
    std::cout << "  Monolithic model: " << mono_duration.count() << " ms\n";
    std::cout << "  Layered model: " << layered_duration.count() << " ms\n";
    std::cout << "  Overhead: " << (layered_duration.count() - mono_duration.count()) << " ms\n";

    return passed ? 0 : 1;

} catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
}
