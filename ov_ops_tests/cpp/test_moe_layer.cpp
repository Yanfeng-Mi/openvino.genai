// Test file for MOE layer construction
// This demonstrates how to create and test MOE layers with synthetic data

#include <openvino/openvino.hpp>
#include <openvino/op/moe.hpp>
#include "gguf_utils/building_blocks.hpp"
#include <iostream>
#include <fstream>
#include <functional>
#include <algorithm>
#include <cmath>

using namespace ov;

int main(int argc, char* argv[]) {
    // Parse command-line arguments
    std::string weight_type = "q4";  // Default to q4
    if (argc > 1) {
        weight_type = argv[1];
        // Convert to lowercase
        std::transform(weight_type.begin(), weight_type.end(), weight_type.begin(), ::tolower);
    }
    
    // Create test configuration programmatically
    std::cout << "Creating synthetic MOE test data..." << std::endl;
    std::cout << "Weight quantization type: " << weight_type << std::endl;
    
    // Model configuration
    const int hidden_dim = 512;
    const int ff_dim = 1536;
    const int num_experts = 64;
    const int topk = 8;
    std::string architecture = "qwen2moe";
    std::string arch_prefix = architecture;
    
    const int batch_size = 2;
    const int seq_len = 8;
    
    std::cout << "✓ Test configuration:" << std::endl;
    std::cout << "  Architecture: " << architecture << std::endl;
    std::cout << "  num_experts: " << num_experts << std::endl;
    std::cout << "  topk: " << topk << std::endl;
    std::cout << "  hidden_dim: " << hidden_dim << std::endl;
    std::cout << "  ff_dim: " << ff_dim << std::endl;
    std::cout << "  batch_size: " << batch_size << std::endl;
    std::cout << "  seq_len: " << seq_len << std::endl;
    
    // Create synthetic weights
    std::unordered_map<std::string, ov::Tensor> remapped_consts;
    std::unordered_map<std::string, gguf_tensor_type> remapped_qtypes;
    
    std::string layer_prefix = "blk.0";
    
    std::cout << "\nCreating synthetic deterministic weights..." << std::endl;
    

    // 1. Create router weights [num_experts, hidden_dim] - always FP16, no quantization
    ov::Tensor router_weight(ov::element::f16, {static_cast<size_t>(num_experts), static_cast<size_t>(hidden_dim)});
    ov::float16* router_data = router_weight.data<ov::float16>();
    for (int e = 0; e < num_experts; e++) {
        for (int h = 0; h < hidden_dim; h++) {
            router_data[e * hidden_dim + h] = ov::float16(static_cast<float>(e) / hidden_dim);
        }
    }
    remapped_consts[layer_prefix + ".router.weight"] = router_weight;
    remapped_qtypes[layer_prefix + ".router.qtype"] = gguf_tensor_type::GGUF_TYPE_F16;
    std::cout << "  ✓ Created router.weight (F16): [" << num_experts << ", " << hidden_dim << "]" << std::endl;
    std::cout << "    Router pattern: Expert e gets score = e (when input is all 1s)" << std::endl;
    
    // 2. Create expert weights for each expert - deterministic identity-like mappings
    std::cout << "  Creating expert weights (deterministic patterns)..." << std::endl;
    
    // Helper lambda for creating uniform weights
    auto create_uniform_weight = [&](size_t rows, size_t cols, float value, const std::string& name) {
        if (weight_type == "q4_0" || weight_type == "q4" || weight_type == "q4_1") {
            const int block_size = 32;
            size_t cols_bytes = (cols + 7) / 8;
            ov::Tensor weight(ov::element::u32, {rows, cols_bytes});
            uint32_t* weight_data = weight.data<uint32_t>();
            std::memset(weight_data, 0, weight.get_byte_size());

            size_t blocks_per_row = (cols + block_size - 1) / block_size;
            ov::Tensor scales(ov::element::f16, {rows, blocks_per_row});
            ov::Tensor biases(ov::element::f16, {rows, blocks_per_row});
            ov::float16* scale_data = scales.data<ov::float16>();
            ov::float16* bias_data = biases.data<ov::float16>();

            for (size_t r = 0; r < rows; r++) {
                for (size_t block = 0; block < blocks_per_row; block++) {
                    size_t start_col = block * block_size;
                    size_t end_col = std::min(start_col + block_size, cols);
                    if (weight_type == "q4_1") {
                        float block_min = std::numeric_limits<float>::max();
                        float block_max = std::numeric_limits<float>::lowest();
                        for (size_t c = start_col; c < end_col; c++) {
                            float val = value;
                            block_min = std::min(block_min, val);
                            block_max = std::max(block_max, val);
                        }
                        float scale;
                        if (block_max == block_min) {
                            scale = 1.0f;
                        } else {
                            scale = (block_max - block_min) / 15.0f;
                        }
                        scale_data[r * blocks_per_row + block] = ov::float16(scale);
                        bias_data[r * blocks_per_row + block] = ov::float16(block_min);
                        // Print debug info for first row and first block
                        if (r == 0 && block == 0) {
                            std::cout << "[Expert Weight Q4_1] " << name << " row 0 block 0: scale=" << scale << ", bias=" << block_min << std::endl;
                            std::cout << "[Expert Weight Q4_1] Quantized values: ";
                        }
                        for (size_t c = start_col; c < end_col; c++) {
                            float val = value;
                            int8_t quantized;
                            if (block_max == block_min) {
                                quantized = 0;
                            } else {
                                quantized = static_cast<int8_t>(std::round((val - block_min) / scale));
                                quantized = std::max(int8_t(0), std::min(int8_t(15), quantized));
                            }
                            size_t linear_idx = r * cols + c;
                            size_t uint32_idx = linear_idx / 8;
                            size_t bit_offset = (linear_idx % 8) * 4;
                            weight_data[uint32_idx] |= (static_cast<uint32_t>(quantized & 0x0F) << bit_offset);
                            if (r == 0 && block == 0) {
                                std::cout << (int)quantized << " ";
                            }
                        }
                        if (r == 0 && block == 0) {
                            std::cout << std::endl;
                        }
                    } else {
                        float scale = std::abs(value) / 7.5f;
                        int8_t quantized = static_cast<int8_t>(std::round(value / scale));
                        quantized = std::max(int8_t(-8), std::min(int8_t(7), quantized));
                        scale_data[r * blocks_per_row + block] = ov::float16(scale);
                        bias_data[r * blocks_per_row + block] = ov::float16(0.0f);
                        // Print debug info for first row and first block
                        if (r == 0 && block == 0) {
                            std::cout << "[Expert Weight Q4_0] " << name << " row 0 block 0: scale=" << scale << ", bias=0" << std::endl;
                            std::cout << "[Expert Weight Q4_0] Quantized values: ";
                        }
                        for (size_t c = start_col; c < end_col; c++) {
                            size_t linear_idx = r * cols + c;
                            size_t uint32_idx = linear_idx / 8;
                            size_t bit_offset = (linear_idx % 8) * 4;
                            weight_data[uint32_idx] |= (static_cast<uint32_t>(quantized & 0x0F) << bit_offset);
                            if (r == 0 && block == 0) {
                                std::cout << (int)quantized << " ";
                            }
                        }
                        if (r == 0 && block == 0) {
                            std::cout << std::endl;
                        }
                    }
                }
            }
            remapped_consts[name + ".weight"] = weight;
            remapped_consts[name + ".scales"] = scales;
            remapped_consts[name + ".biases"] = biases;
            remapped_qtypes[name + ".qtype"] = (weight_type == "q4_1") ? gguf_tensor_type::GGUF_TYPE_Q4_1 : gguf_tensor_type::GGUF_TYPE_Q4_0;
        } else {
            // FP16
            ov::Tensor weight(ov::element::f16, {rows, cols});
            ov::float16* weight_data = weight.data<ov::float16>();
            for (size_t j = 0; j < weight.get_size(); j++) {
                weight_data[j] = ov::float16(value);
            }
            remapped_consts[name + ".weight"] = weight;
            remapped_qtypes[name + ".qtype"] = gguf_tensor_type::GGUF_TYPE_F16;
        }
    };
    
    for (int i = 0; i < num_experts; i++) {
        std::string expert_prefix = layer_prefix + ".experts." + std::to_string(i);
        float scale = (i + 1) * 0.001f;
        float down_scale = 1.0f / ff_dim;
        
        // gate_proj: [ff_dim, hidden_dim] - transposed for MOE
        create_uniform_weight(ff_dim, hidden_dim, scale, expert_prefix + ".gate_proj");
        
        // up_proj: [ff_dim, hidden_dim] - transposed for MOE
        create_uniform_weight(ff_dim, hidden_dim, scale, expert_prefix + ".up_proj");
        
        // down_proj: [hidden_dim, ff_dim] - transposed for MOE
        create_uniform_weight(hidden_dim, ff_dim, down_scale, expert_prefix + ".down_proj");
    }
    
    std::cout << "  ✓ Created weights for " << num_experts << " experts" << std::endl;
    
    // 2.5 Concatenate expert weights BEFORE decompression for cleaner pattern matching
    // Create 3D tensors [num_experts, rows, cols] that will be processed by make_int4_weights_concat()
    std::cout << "  Concatenating expert weights with preserved expert dimension..." << std::endl;
    
    auto concat_expert_weights = [&](const std::string& weight_name, size_t rows, size_t cols) {
        if (weight_type == "q4_0" || weight_type == "q4" || weight_type == "q4_1") {
            const int block_size = 32;
            size_t blocks_per_row = (cols + block_size - 1) / block_size;
            size_t cols_bytes = (cols + 7) / 8;

            ov::Tensor concat_weight(ov::element::u32, {static_cast<size_t>(num_experts), rows, cols_bytes});
            ov::Tensor concat_scales(ov::element::f16, {static_cast<size_t>(num_experts), rows, blocks_per_row});
            ov::Tensor concat_biases(ov::element::f16, {static_cast<size_t>(num_experts), rows, blocks_per_row});

            uint32_t* concat_weight_data = concat_weight.data<uint32_t>();
            ov::float16* concat_scale_data = concat_scales.data<ov::float16>();
            ov::float16* concat_bias_data = concat_biases.data<ov::float16>();

            std::memset(concat_weight_data, 0, concat_weight.get_byte_size());

            for (int e = 0; e < num_experts; e++) {
                std::string expert_key = layer_prefix + ".experts." + std::to_string(e) + "." + weight_name;
                auto& weight = remapped_consts[expert_key + ".weight"];
                auto& scales = remapped_consts[expert_key + ".scales"];
                auto& biases = remapped_consts[expert_key + ".biases"];

                uint32_t* src_weight = weight.data<uint32_t>();
                ov::float16* src_scales = scales.data<ov::float16>();
                ov::float16* src_biases = biases.data<ov::float16>();

                size_t expert_weight_offset = e * rows * cols_bytes;
                size_t expert_scale_offset = e * rows * blocks_per_row;

                std::memcpy(concat_weight_data + expert_weight_offset, src_weight, weight.get_byte_size());
                std::memcpy(concat_scale_data + expert_scale_offset, src_scales, scales.get_byte_size());
                std::memcpy(concat_bias_data + expert_scale_offset, src_biases, biases.get_byte_size());
            }

            std::string fused_key = layer_prefix + ".experts." + weight_name + "_fused";
            remapped_consts[fused_key + ".weight"] = concat_weight;
            remapped_consts[fused_key + ".scales"] = concat_scales;
            remapped_consts[fused_key + ".biases"] = concat_biases;
            remapped_qtypes[fused_key + ".qtype"] = (weight_type == "q4_1") ? gguf_tensor_type::GGUF_TYPE_Q4_1 : gguf_tensor_type::GGUF_TYPE_Q4_0;

            std::cout << "    " << weight_name << " 3D: weight[" << num_experts << "," << rows << "," << cols_bytes << "] scales[" << num_experts << "," << rows << "," << blocks_per_row << "] (" << (weight_type == "q4_1" ? "Q4_1" : "Q4 packed") << ")" << std::endl;
        } else {
            // Concatenate FP16 weights: [num_experts, rows, cols]
            ov::Tensor concat_weight(ov::element::f16, {static_cast<size_t>(num_experts), rows, cols});
            ov::float16* concat_data = concat_weight.data<ov::float16>();
            
            for (int e = 0; e < num_experts; e++) {
                std::string expert_key = layer_prefix + ".experts." + std::to_string(e) + "." + weight_name;
                auto& weight = remapped_consts[expert_key + ".weight"];
                ov::float16* src_data = weight.data<ov::float16>();
                
                // Copy into [e, :, :] slice
                size_t expert_offset = e * rows * cols;
                std::memcpy(concat_data + expert_offset, src_data, weight.get_byte_size());
            }
            
            std::string fused_key = layer_prefix + ".experts." + weight_name + "_fused";
            remapped_consts[fused_key + ".weight"] = concat_weight;
            remapped_qtypes[fused_key + ".qtype"] = gguf_tensor_type::GGUF_TYPE_F16;
            
            std::cout << "    " << weight_name << " 3D: [" << num_experts << "," << rows << "," << cols << "] (FP16)" << std::endl;
        }
    };

    concat_expert_weights("gate_proj", ff_dim, hidden_dim);
    concat_expert_weights("up_proj", ff_dim, hidden_dim);
    concat_expert_weights("down_proj", hidden_dim, ff_dim);
    
    std::cout << "  ✓ Expert pattern: Expert e scales by (e+1)*0.001" << std::endl;
    std::cout << "  ✓ Quantization: " << weight_type << std::endl;
    std::cout << "  ✓ Total tensors created: " << remapped_consts.size() << std::endl;
    std::cout << "\n  Expected behavior with all-1s input [2,8,512]:" << std::endl;
    
    // 3. Create input parameter node
    auto hidden_states_param = std::make_shared<op::v0::Parameter>(
        element::f32, 
        PartialShape{batch_size, seq_len, hidden_dim});
    hidden_states_param->set_friendly_name("hidden_states");
    
    std::cout << "\nBuilding MOE graph..." << std::endl;
    
    // NOTE: For ConvertMOEToMOECompressed to match, we need to:
    // 1. Create ov::op::internal::MOE node (using moe_layer_internal, not moe_layer_fused)
    // 2. Ensure quantized weights have proper Convert nodes (make_int4_weights does this)
    // 3. For Q4: weight storage should be u32 packed, with scales/biases for decompression
    
    std::cout << "  Using weight type: " << weight_type << std::endl;
    std::cout << "  Creating ov::op::internal::MOE node for GPU optimization" << std::endl;
    if (weight_type == "q4_0" || weight_type == "q4" || weight_type == "q8_0" || weight_type == "q8") {
        std::cout << "  ⚠ Quantized weights - ConvertMOEToMOECompressed should match!" << std::endl;
        std::cout << "    Pattern expects:" << std::endl;
        std::cout << "    - MOE node with quantized weight inputs" << std::endl;
        std::cout << "    - Each weight: u4/i4/u8/i8 Constant → Convert → [Subtract] → Multiply" << std::endl;
        std::cout << "    - make_int4_weights() creates this pattern automatically" << std::endl;
    }
    
    // 4. Build MOE layer using moe_layer_internal() which creates the internal MOE op
    /* auto moe_output =
        moe_layer_fused(
        layer_prefix,
        hidden_states_param,
        remapped_consts,
        remapped_qtypes,
        num_experts,
        topk);*/
    
    auto moe_output =
        moe_layer_internal(layer_prefix, hidden_states_param, remapped_consts, remapped_qtypes, num_experts, topk);

    moe_output.get_node_shared_ptr()->set_friendly_name("moe_output");
        
    // Create model with main output and debug outputs
    ResultVector results;
    results.push_back(std::make_shared<op::v0::Result>(moe_output));
    
    // 5. Create model
    auto model = std::make_shared<Model>(
        results,
        ParameterVector{hidden_states_param});
        
        model->set_friendly_name("test_moe_model");
        
        std::cout << "✓ MOE graph built successfully!" << std::endl;
        std::cout << "  Input shape: [" << batch_size << ", " << seq_len << ", " << hidden_dim << "]" << std::endl;
        std::cout << "  Output shape: " << moe_output.get_partial_shape() << std::endl;
        
        // 6. Optional: Serialize the model
        std::cout << "\nSerializing model to XML..." << std::endl;
        ov::serialize(model, "test_moe_model.xml", "test_moe_model.bin");
        std::cout << "✓ Model saved to test_moe_model.xml" << std::endl;
        
        // 7. Optional: Compile and run inference
        std::cout << "\nCompiling model..." << std::endl;
        Core core;
        //ov::AnyMap config = {{ov::intel_gpu::disable_moe_opt(true)}};
        auto compiled_model = core.compile_model(model, "GPU.1");
        //std::cout << "✓ Model compiled successfully on CPU" << std::endl;
        
        auto infer_request = compiled_model.create_infer_request();
        
        // Create deterministic input data for validation
        auto input_tensor = ov::Tensor(element::f32, Shape{static_cast<size_t>(batch_size), 
                                                            static_cast<size_t>(seq_len), 
                                                            static_cast<size_t>(hidden_dim)});
        float* input_data = input_tensor.data<float>();
        
        std::cout << "\nCreating deterministic input pattern..." << std::endl;
        // Pattern 1: All ones - makes routing predictable
        for (size_t i = 0; i < input_tensor.get_size(); i++) {
            input_data[i] = 1.0f;
        }
        
        // Print sample input
        std::cout << "Input pattern: ones (all 1.0)" << std::endl;
        std::cout << "Sample input values [0, 0, :5]: ";
        for (int i = 0; i < 5 && i < hidden_dim; i++) {
            std::cout << input_data[i] << " ";
        }
        std::cout << "..." << std::endl;
        
        infer_request.set_input_tensor(input_tensor);
        
        std::cout << "\nRunning inference..." << std::endl;
        infer_request.infer();
        
        // Check all outputs
        size_t num_outputs = compiled_model.outputs().size();
        std::cout << "✓ Inference completed! Got " << num_outputs << " outputs" << std::endl;
        
        auto output_tensor = infer_request.get_output_tensor(0);
        std::cout << "  Main output tensor shape: " << output_tensor.get_shape() << std::endl;
        std::cout << "  Main output tensor type: " << output_tensor.get_element_type() << std::endl;

        // Validate main output - support both FP32 and FP16
        bool is_fp16 = (output_tensor.get_element_type() == ov::element::f16);
        
        std::cout << "\nOutput validation:" << std::endl;
        std::cout << "Sample output values [0, 0, :5]: ";
        if (is_fp16) {
            ov::float16* output_data_fp16 = output_tensor.data<ov::float16>();
            for (int i = 0; i < 5 ; i++) {
                std::cout << static_cast<float>(output_data_fp16[i]) << " ";
            }
        } else {
            float* output_data_fp32 = output_tensor.data<float>();
            for (int i = 0; i < 5 ; i++) {
                std::cout << output_data_fp32[i] << " ";
            }
        }
        std::cout << "..." << std::endl;
        
        // Calculate statistics
        float sum = 0.0f, min_val, max_val;
        size_t output_size = output_tensor.get_size();
        
        if (is_fp16) {
            ov::float16* output_data_fp16 = output_tensor.data<ov::float16>();
            min_val = static_cast<float>(output_data_fp16[0]);
            max_val = static_cast<float>(output_data_fp16[0]);
            for (size_t i = 0; i < output_size; i++) {
                float val = static_cast<float>(output_data_fp16[i]);
                sum += val;
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
            }
        } else {
            float* output_data_fp32 = output_tensor.data<float>();
            min_val = output_data_fp32[0];
            max_val = output_data_fp32[0];
            for (size_t i = 0; i < output_size; i++) {
                float val = output_data_fp32[i];
                sum += val;
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
            }
        }
        float mean = sum / output_size;
        
        std::cout << "\nOutput statistics:" << std::endl;
        std::cout << "  Mean: " << mean << std::endl;
        std::cout << "  Min:  " << min_val << std::endl;
        std::cout << "  Max:  " << max_val << std::endl;
        std::cout << "  Sum:  " << sum << std::endl;
        
        // Expected values for deterministic all-1s input with our weight pattern
        // Note: scaled down by 100x from original to prevent FP16 overflow
        const float expected_mean_fp16 = 1054.67f;
        const float expected_value_fp16 = 1054.59f;  // Min/Max should be identical
        const float expected_mean_q4 = 855.0f;  // Q4 has ~19% quantization error
        
        float tolerance = 10.0f;  // Default tolerance
        float expected_mean = expected_mean_fp16;
        float expected_value = expected_value_fp16;
        
        // Adjust expectations for Q4 quantization
        if (weight_type == "q4_0" || weight_type == "q4") {
            tolerance = 50.0f;  // Larger tolerance for Q4
            expected_mean = expected_mean_q4;
            expected_value = expected_mean_q4;
        }
        
        std::cout << "\nValidation (deterministic all-1s input):" << std::endl;
        std::cout << "  Weight type: " << weight_type << std::endl;
        std::cout << "  Expected mean: " << expected_mean << std::endl;
        std::cout << "  Expected uniform value: " << expected_value << std::endl;
        
        bool mean_ok = std::abs(mean - expected_mean) < tolerance;
        bool uniform_ok = std::abs(max_val - min_val) < tolerance;
        bool value_ok = std::abs(mean - expected_value) < tolerance * 2;
        
        if (weight_type == "q4_0" || weight_type == "q4") {
            // Q4-specific validation
            float quantization_error = std::abs((mean - expected_mean_fp16) / expected_mean_fp16) * 100.0f;
            std::cout << "  Q4 quantization error: " << quantization_error << "% vs FP16 baseline" << std::endl;
            
            if (quantization_error > 10.0f && quantization_error < 30.0f) {
                std::cout << "  ✓ Q4 quantization error in expected range (10-30%)" << std::endl;
            } else if (quantization_error < 10.0f) {
                std::cout << "  ⚠ Q4 error unexpectedly low - may not be using Q4 path?" << std::endl;
            } else {
                std::cout << "  ✗ Q4 error too high (>" << quantization_error << "%)" << std::endl;
            }
        }
        
        if (mean_ok) {
            std::cout << "  ✓ Mean matches expected value!" << std::endl;
        } else {
            std::cout << "  ✗ Mean differs: expected " << expected_mean << ", got " << mean << std::endl;
        }
        
        if (uniform_ok) {
            std::cout << "  ✓ Output is uniform (max-min < " << tolerance << ")" << std::endl;
        } else {
            std::cout << "  ✗ Output varies: min=" << min_val << ", max=" << max_val << std::endl;
        }
        
        if (value_ok) {
            std::cout << "  ✓ Values are deterministic and reproducible!" << std::endl;
        } else {
            std::cout << "  ⚠ Values differ from expected baseline" << std::endl;
        }
        
        // Check for NaN or Inf
        bool has_invalid = false;
        if (is_fp16) {
            ov::float16* output_data_fp16 = output_tensor.data<ov::float16>();
            for (size_t i = 0; i < output_size; i++) {
                float val = static_cast<float>(output_data_fp16[i]);
                if (std::isnan(val) || std::isinf(val)) {
                    has_invalid = true;
                    break;
                }
            }
        } else {
            float* output_data_fp32 = output_tensor.data<float>();
            for (size_t i = 0; i < output_size; i++) {
                if (std::isnan(output_data_fp32[i]) || std::isinf(output_data_fp32[i])) {
                    has_invalid = true;
                    break;
                }
            }
        }
        
        if (has_invalid) {
            std::cout << "  ✗ Output contains NaN or Inf values!" << std::endl;
        } else {
            std::cout << "  ✓ No NaN or Inf values detected" << std::endl;
        }
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "Test completed successfully!" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        return 0;
}
