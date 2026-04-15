// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>
#include <cstdarg>

#include <openvino/openvino.hpp>

#include "gguf_utils/gguf.hpp"

ov::Output<ov::Node> make_weights_subgraph(const std::string& key,
                                          const std::unordered_map<std::string, ov::Tensor>& consts,
                                          gguf_tensor_type qtype,
                                          bool reorder,
                                          int head_size);

ov::Output<ov::Node> make_lm_head(
    const std::string& key,
    const ov::Output<ov::Node>& input,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    const ov::Output<ov::Node>& embeddings_node,
    gguf_tensor_type qtype);

ov::Output<ov::Node> make_rms_norm(
    const std::string& key,
    const ov::Output<ov::Node>& input,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    float epsilon);

std::tuple<ov::Output<ov::Node>, ov::Output<ov::Node>> make_embedding(
    const std::string& key,
    const ov::Output<ov::Node>& input,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    gguf_tensor_type qtype);

std::tuple<ov::Output<ov::Node>, 
           ov::SinkVector,
           ov::Output<ov::Node>,
           std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>>,
           std::shared_ptr<ov::Node>> 
    layer(const std::map<std::string, GGUFMetaData>& configs,
        std::unordered_map<std::string, ov::Tensor>& consts,
        std::unordered_map<std::string, gguf_tensor_type>& qtypes,
        int layer_idx,
        const ov::Output<ov::Node>& hidden_states,
        const ov::Output<ov::Node>& attn_mask,
        const ov::Output<ov::Node>& causal_mask,
        const ov::Output<ov::Node>& position_ids,
        const ov::Output<ov::Node>& rope_const,
        const ov::Output<ov::Node>& beam_idx,
        const ov::Output<ov::Node>& batch_dim,
        const ov::Output<ov::Node>& hidden_dim,
        const std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>>& cos_sin_cached,
        const std::shared_ptr<ov::Node>& output_shape);

// MOE Layer using ov::op::internal::MOE node (for GPU optimization)
// Creates the internal MOE op which can be matched by ConvertMOEToMOECompressed
ov::Output<ov::Node> moe_layer_internal(
    const std::string& layer_prefix,
    const ov::Output<ov::Node>& hidden_states,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    const std::unordered_map<std::string, gguf_tensor_type>& qtypes,
    int num_experts,
    int topk);

ov::Output<ov::Node> moe_layer_fused(
    const std::string& layer_prefix,
    const ov::Output<ov::Node>& hidden_states,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    const std::unordered_map<std::string, gguf_tensor_type>& qtypes,
    int num_experts,
    int topk);

ov::Output<ov::Node> init_rope(
    int64_t head_dim,
    int64_t max_position_embeddings = 2048,
    float base = 10000.0f,
    float scaling_factor = 1.0f);

// ============================================================================
// In-Flight Quantization Support
// ============================================================================

/**
 * @brief Create a dequantization subgraph for in-flight INT4 quantized weights
 * 
 * This function takes pre-quantized INT4 weights (from NNCF or other quantizers)
 * and builds an OpenVINO subgraph that performs dequantization:
 *   U4 → Convert(FP16) → Subtract(zero_point) → Multiply(scale) → Reshape
 * 
 * @param compressed_weight INT4 packed weight tensor (U4 element type)
 * @param scale Scale tensor for dequantization (FP16)
 * @param zero_point Zero point tensor (U4, nullable for symmetric quantization)
 * @param original_shape Original shape of the weight before group quantization
 * @param group_size Number of elements per quantization group (default: 128)
 * @param name Friendly name prefix for the nodes
 * @return Output node producing FP32 dequantized weights
 */
ov::Output<ov::Node> make_inflight_int4_weights(
    const ov::Tensor& compressed_weight,
    const ov::Tensor& scale,
    const ov::Tensor* zero_point,
    const ov::Shape& original_shape,
    size_t group_size = 128,
    const std::string& name = "");

/**
 * @brief Create a dequantization subgraph for in-flight INT8 quantized weights
 * 
 * @param compressed_weight INT8 packed weight tensor (I8 or U8 element type)
 * @param scale Scale tensor for dequantization (FP16)
 * @param zero_point Zero point tensor (I8/U8, nullable for symmetric quantization)
 * @param original_shape Original shape of the weight before group quantization
 * @param group_size Number of elements per quantization group (default: 128)
 * @param name Friendly name prefix for the nodes
 * @return Output node producing FP32 dequantized weights
 */
ov::Output<ov::Node> make_inflight_int8_weights(
    const ov::Tensor& compressed_weight,
    const ov::Tensor& scale,
    const ov::Tensor* zero_point,
    const ov::Shape& original_shape,
    size_t group_size = 128,
    const std::string& name = "");
