// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_vl/export_for_vlmpipeline.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>
#include <openvino/op/result.hpp>
#include <openvino/runtime/properties.hpp>
#include <nlohmann/json.hpp>

#include "modeling/builder_context.hpp"
#include "modeling/models/qwen3_vl/modeling_qwen3_vl_text.hpp"
#include "modeling/models/qwen3_vl/modeling_qwen3_vl_vision.hpp"
#include "modeling/models/qwen3_vl/processing_qwen3_vl.hpp"
#include "modeling/module.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/ops/tensor.hpp"
#include "modeling/ops/tensor_ops.hpp"
#include "modeling/weights/weight_loader.hpp"
#include "safetensors_utils/safetensors_loader.hpp"
#include "safetensors_utils/safetensors_weight_finalizer.hpp"
#include "safetensors_utils/safetensors_weight_source.hpp"
#include "safetensors_utils/quantization_utils.hpp"

namespace {

using namespace ov::genai::modeling;
using namespace ov::genai::modeling::models;

auto set_name = [](auto node, const std::string& name) {
    node->output(0).set_names({name});
    node->set_friendly_name(name);
};

/// Resolve the position embedding weight name from safetensors keys.
std::string resolve_pos_embed_name(weights::WeightSource& source) {
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
    OPENVINO_THROW("Failed to locate visual.pos_embed.weight in safetensors");
}

}  // namespace

namespace ov::genai::modeling::models {

// =============================================================================
// 1. Vision Embeddings Model (PatchEmbed only)
// =============================================================================

std::shared_ptr<ov::Model> create_qwen3_vl_vision_embeddings_model(
    const Qwen3VLConfig& cfg,
    weights::WeightSource& source,
    weights::WeightFinalizer& finalizer) {

    BuilderContext ctx;
    // Create full vision model for weight loading, but only use PatchEmbed
    Qwen3VLVisionModel model(ctx, cfg.vision);
    model.packed_mapping().rules.push_back({"model.", "", 0});

    weights::LoadOptions options;
    options.allow_unmatched = true;
    options.allow_missing = true;  // Non-PatchEmbed weights may be missing
    options.report_missing = false;
    options.report_unmatched = true;
    weights::load_model(model, source, finalizer, options);

    const int32_t in_channels = cfg.vision.in_channels;
    const int32_t temporal_patch = cfg.vision.temporal_patch_size;
    const int32_t patch_size = cfg.vision.patch_size;
    const int64_t channel_dim = static_cast<int64_t>(in_channels) * temporal_patch * patch_size * patch_size;

    auto pixel_values = ctx.parameter("pixel_values",
                                       ov::element::f32,
                                       ov::PartialShape{-1, channel_dim});

    // Reshape to [N, C, T, P, P] for Conv3d PatchEmbed, then forward
    auto output = model.patch_embed().forward(pixel_values);

    auto result = std::make_shared<ov::op::v0::Result>(output.output());
    set_name(result, "output");
    return ctx.build_model({result->output(0)});
}

// =============================================================================
// 2. Vision Merger Model (Blocks + Merger + DeepstackMergers)
// =============================================================================

std::shared_ptr<ov::Model> create_qwen3_vl_vision_merger_model(
    const Qwen3VLConfig& cfg,
    weights::WeightSource& source,
    weights::WeightFinalizer& finalizer) {

    BuilderContext ctx;
    Qwen3VLVisionModel model(ctx, cfg.vision);
    model.packed_mapping().rules.push_back({"model.", "", 0});

    weights::LoadOptions options;
    options.allow_unmatched = true;
    options.allow_missing = true;
    options.report_missing = false;
    options.report_unmatched = true;
    weights::load_model(model, source, finalizer, options);

    const int32_t head_dim = cfg.vision.head_dim();

    // Inputs matching VLMPipeline's merger model I/O
    auto hidden_states = ctx.parameter("hidden_states",
                                        ov::element::f32,
                                        ov::PartialShape{-1, cfg.vision.hidden_size});

    auto rotary_pos_emb = ctx.parameter("rotary_pos_emb",
                                         ov::element::f32,
                                         ov::PartialShape{-1, head_dim});

    auto cu_seq_lens = ctx.parameter("cu_seq_lens",
                                      ov::element::i32,
                                      ov::PartialShape{-1});

    // Compute cos/sin from rotary_pos_emb inside the graph
    auto rotary_cos = rotary_pos_emb.cos();
    auto rotary_sin = rotary_pos_emb.sin();

    // Run blocks + merger + deepstack via forward_blocks (no PatchEmbed)
    auto output = model.forward_blocks(hidden_states, rotary_cos, rotary_sin, nullptr, &cu_seq_lens);

    // Build results
    ov::OutputVector results;

    // Primary output: last_hidden_state
    auto lhs_result = std::make_shared<ov::op::v0::Result>(output.visual_embeds.output());
    set_name(lhs_result, "last_hidden_state");
    results.push_back(lhs_result->output(0));

    // Deepstack output: stack individual outputs into [num_layers, N, hidden_size]
    if (!output.deepstack_embeds.empty()) {
        auto stacked = ops::tensor::stack(output.deepstack_embeds, 0);
        auto ds_result = std::make_shared<ov::op::v0::Result>(stacked.output());
        set_name(ds_result, "deepstack_feature_lists");
        results.push_back(ds_result->output(0));
    }

    return ctx.build_model(results);
}

// =============================================================================
// 3. Vision Position Embedding Model (Lookup/Gather)
// =============================================================================

std::shared_ptr<ov::Model> create_qwen3_vl_vision_pos_model(
    const Qwen3VLConfig& cfg,
    weights::WeightSource& source) {

    BuilderContext ctx;

    // Locate pos_embed weight
    std::string pos_embed_name = resolve_pos_embed_name(source);
    const ov::Tensor& pos_weight_raw = source.get_tensor(pos_embed_name);

    // pos_embed_weight: [num_position_embeddings, embed_dim]
    auto pos_weight = ops::constant(pos_weight_raw, &ctx.op_context());

    // Input: 4 sets of corner indices for bilinear interpolation
    // Shape: [4, num_positions]
    auto input_indices = ctx.parameter("input",
                                        ov::element::i64,
                                        ov::PartialShape{4, -1});

    // Gather for each of 4 corners
    std::vector<Tensor> corner_embeds;
    corner_embeds.reserve(4);
    for (int64_t corner = 0; corner < 4; ++corner) {
        auto corner_indices = ops::slice(input_indices, corner, corner + 1, 1, 0).squeeze(0);
        auto gathered = ops::gather(pos_weight, corner_indices, 0);
        corner_embeds.push_back(gathered);
    }

    // Stack: [4, num_positions, embed_dim]
    auto output = ops::tensor::stack(corner_embeds, 0);

    auto result = std::make_shared<ov::op::v0::Result>(output.output());
    set_name(result, "output");
    return ctx.build_model({result->output(0)});
}

// =============================================================================
// 4. Text Embeddings Model (VocabEmbedding only)
// =============================================================================

std::shared_ptr<ov::Model> create_qwen3_vl_text_embeddings_model(
    const Qwen3VLConfig& cfg,
    weights::WeightSource& source,
    weights::WeightFinalizer& finalizer) {

    BuilderContext ctx;
    Qwen3VLTextForCausalLM model(ctx, cfg.text);
    model.packed_mapping().rules.push_back({"model.", "", 0});

    weights::LoadOptions options;
    options.allow_unmatched = true;
    options.allow_missing = true;
    options.report_missing = false;
    options.report_unmatched = true;
    weights::load_model(model, source, finalizer, options);

    auto input_ids = ctx.parameter("input_ids",
                                    ov::element::i64,
                                    ov::PartialShape{-1, -1});

    auto embeddings = model.model().embed_tokens().forward(input_ids);

    auto result = std::make_shared<ov::op::v0::Result>(embeddings.output());
    set_name(result, "output");
    return ctx.build_model({result->output(0)});
}

// =============================================================================
// 5. Language Model (Decoder + DeepstackInjector + LMHead)
// =============================================================================

std::shared_ptr<ov::Model> create_qwen3_vl_language_model(
    const Qwen3VLConfig& cfg,
    weights::WeightSource& source,
    weights::WeightFinalizer& finalizer) {

    BuilderContext ctx;
    Qwen3VLTextForCausalLM model(ctx, cfg.text);
    model.packed_mapping().rules.push_back({"model.", "", 0});

    weights::LoadOptions options;
    options.allow_unmatched = true;
    options.allow_missing = false;
    options.report_missing = true;
    options.report_unmatched = true;
    weights::load_model(model, source, finalizer, options);

    const int32_t hidden_size = cfg.text.hidden_size;
    const size_t num_deepstack = cfg.vision.deepstack_visual_indexes.size();

    // Inputs matching VLMPipeline language model I/O
    auto inputs_embeds = ctx.parameter("inputs_embeds",
                                        ov::element::f32,
                                        ov::PartialShape{-1, -1, hidden_size});

    auto attention_mask = ctx.parameter("attention_mask",
                                         ov::element::i64,
                                         ov::PartialShape{-1, -1});

    auto position_ids = ctx.parameter("position_ids",
                                       ov::element::i64,
                                       ov::PartialShape{3, -1, -1});

    auto beam_idx = ctx.parameter("beam_idx",
                                   ov::element::i32,
                                   ov::PartialShape{-1});

    // VLMPipeline-specific visual inputs
    auto deepstack_visual_embeds = ctx.parameter("deepstack_visual_embeds",
                                                  ov::element::f32,
                                                  ov::PartialShape{static_cast<int64_t>(num_deepstack), -1, hidden_size});

    auto visual_pos_masks = ctx.parameter("visual_pos_masks",
                                           ov::element::boolean,
                                           ov::PartialShape{-1, -1});

    // Slice deepstack into per-layer tensors
    std::vector<Tensor> ds_slices;
    ds_slices.reserve(num_deepstack);
    for (size_t i = 0; i < num_deepstack; ++i) {
        auto slice = ops::slice(deepstack_visual_embeds,
                                static_cast<int64_t>(i),
                                static_cast<int64_t>(i + 1),
                                1, 0).squeeze(0);
        ds_slices.push_back(slice);
    }

    // Forward: skip EmbeddingInjector (visual_embeds=nullptr), apply DeepstackInjector
    auto logits = model.forward_embeds(inputs_embeds,
                                       position_ids,
                                       beam_idx,
                                       &attention_mask,
                                       nullptr,            // visual_embeds: skip EmbeddingInjector
                                       &visual_pos_masks,  // needed for DeepstackInjector
                                       &ds_slices);

    auto result = std::make_shared<ov::op::v0::Result>(logits.output());
    set_name(result, "logits");
    auto ov_model = ctx.build_model({result->output(0)});
    ov_model->set_rt_info(ov::element::f16, {"runtime_options", ov::hint::kv_cache_precision.name()});
    ov_model->set_rt_info(8.0f, {"runtime_options", ov::hint::activations_scale_factor.name()});
    return ov_model;
}

// =============================================================================
// Config Generation
// =============================================================================

namespace {

void generate_vlmpipeline_config(const Qwen3VLConfig& cfg,
                                 const std::filesystem::path& output_dir) {
    nlohmann::json config;
    config["model_type"] = "qwen3_vl";
    config["architectures"] = nlohmann::json::array({"Qwen3VLForConditionalGeneration"});
    config["hidden_size"] = cfg.text.hidden_size;

    // Vision config
    nlohmann::json vision_config;
    vision_config["depth"] = cfg.vision.depth;
    vision_config["hidden_size"] = cfg.vision.hidden_size;
    vision_config["num_heads"] = cfg.vision.num_heads;
    vision_config["in_channels"] = cfg.vision.in_channels;
    vision_config["patch_size"] = cfg.vision.patch_size;
    vision_config["spatial_merge_size"] = cfg.vision.spatial_merge_size;
    vision_config["temporal_patch_size"] = cfg.vision.temporal_patch_size;
    vision_config["out_hidden_size"] = cfg.vision.out_hidden_size;
    vision_config["num_position_embeddings"] = cfg.vision.num_position_embeddings;
    vision_config["intermediate_size"] = cfg.vision.intermediate_size;

    nlohmann::json ds_indexes = nlohmann::json::array();
    for (auto idx : cfg.vision.deepstack_visual_indexes) {
        ds_indexes.push_back(idx);
    }
    vision_config["deepstack_visual_indexes"] = ds_indexes;
    config["vision_config"] = vision_config;

    // Text config
    nlohmann::json text_config;
    text_config["vocab_size"] = cfg.text.vocab_size;
    text_config["hidden_size"] = cfg.text.hidden_size;
    text_config["intermediate_size"] = cfg.text.intermediate_size;
    text_config["num_hidden_layers"] = cfg.text.num_hidden_layers;
    text_config["num_attention_heads"] = cfg.text.num_attention_heads;
    text_config["num_key_value_heads"] = cfg.text.num_key_value_heads;
    text_config["max_position_embeddings"] = cfg.text.max_position_embeddings;
    text_config["rms_norm_eps"] = cfg.text.rms_norm_eps;
    text_config["rope_theta"] = cfg.text.rope_theta;
    text_config["attention_bias"] = cfg.text.attention_bias;
    text_config["tie_word_embeddings"] = cfg.text.tie_word_embeddings;
    config["text_config"] = text_config;

    // Token IDs
    config["image_token_id"] = cfg.image_token_id;
    config["video_token_id"] = cfg.video_token_id;
    config["vision_start_token_id"] = cfg.vision_start_token_id;
    config["vision_end_token_id"] = cfg.vision_end_token_id;

    std::ofstream out(output_dir / "config.json");
    OPENVINO_ASSERT(out.is_open(), "Failed to create config.json");
    out << config.dump(2);
}

void generate_preprocessor_config(const Qwen3VLConfig& cfg,
                                  const std::filesystem::path& output_dir) {
    nlohmann::json preproc;
    preproc["image_processor_type"] = "Qwen3VLImageProcessor";

    nlohmann::json size;
    size["shortest_edge"] = 56 * 56;          // min_pixels
    size["longest_edge"] = 28 * 28 * 1280;    // max_pixels
    preproc["size"] = size;

    preproc["patch_size"] = cfg.vision.patch_size;
    preproc["temporal_patch_size"] = cfg.vision.temporal_patch_size;
    preproc["merge_size"] = cfg.vision.spatial_merge_size;

    preproc["image_mean"] = {0.5, 0.5, 0.5};
    preproc["image_std"] = {0.5, 0.5, 0.5};
    preproc["do_resize"] = true;
    preproc["do_rescale"] = true;
    preproc["do_normalize"] = true;

    std::ofstream out(output_dir / "preprocessor_config.json");
    OPENVINO_ASSERT(out.is_open(), "Failed to create preprocessor_config.json");
    out << preproc.dump(2);
}

}  // namespace

// =============================================================================
// Main Export Function
// =============================================================================

void export_qwen3_vl_for_vlmpipeline(
    const std::filesystem::path& model_dir,
    const std::filesystem::path& output_dir,
    const Qwen3VLExportOptions& options) {

    // Load configuration
    auto cfg = Qwen3VLConfig::from_json_file(model_dir / "config.json");

    // Load safetensors weights
    auto data = ov::genai::safetensors::load_safetensors(model_dir);
    ov::genai::safetensors::SafetensorsWeightSource source(std::move(data));

    // Create output directory
    std::filesystem::create_directories(output_dir);

    ov::Core core;

    auto vision_quant_config = create_quantization_config(
        options.vision_quant_mode, options.vision_quant_group_size,
        options.vision_quant_backup_mode);
    auto text_quant_config = create_quantization_config(
        options.text_quant_mode, options.text_quant_group_size,
        options.text_quant_backup_mode);

    // 1. Export vision_embeddings_model (PatchEmbed)
    {
        ov::genai::safetensors::SafetensorsWeightFinalizer finalizer(vision_quant_config);
        auto model = create_qwen3_vl_vision_embeddings_model(cfg, source, finalizer);
        ov::serialize(model,
                      (output_dir / "openvino_vision_embeddings_model.xml").string(),
                      (output_dir / "openvino_vision_embeddings_model.bin").string());
    }

    // 2. Export vision_embeddings_merger_model (Blocks + Merger + Deepstack)
    {
        ov::genai::safetensors::SafetensorsWeightFinalizer finalizer(vision_quant_config);
        auto model = create_qwen3_vl_vision_merger_model(cfg, source, finalizer);
        ov::serialize(model,
                      (output_dir / "openvino_vision_embeddings_merger_model.xml").string(),
                      (output_dir / "openvino_vision_embeddings_merger_model.bin").string());
    }

    // 3. Export vision_embeddings_pos_model (Position embedding lookup)
    {
        auto model = create_qwen3_vl_vision_pos_model(cfg, source);
        ov::serialize(model,
                      (output_dir / "openvino_vision_embeddings_pos_model.xml").string(),
                      (output_dir / "openvino_vision_embeddings_pos_model.bin").string());
    }

    // 4. Export text_embeddings_model (VocabEmbedding)
    {
        ov::genai::safetensors::SafetensorsWeightFinalizer finalizer(text_quant_config);
        auto model = create_qwen3_vl_text_embeddings_model(cfg, source, finalizer);
        ov::serialize(model,
                      (output_dir / "openvino_text_embeddings_model.xml").string(),
                      (output_dir / "openvino_text_embeddings_model.bin").string());
    }

    // 5. Export language_model (Decoder + DeepstackInjector + LMHead)
    {
        ov::genai::safetensors::SafetensorsWeightFinalizer finalizer(text_quant_config);
        auto model = create_qwen3_vl_language_model(cfg, source, finalizer);
        ov::serialize(model,
                      (output_dir / "openvino_language_model.xml").string(),
                      (output_dir / "openvino_language_model.bin").string());
    }

    // 6. Generate VLMPipeline-compatible config files
    generate_vlmpipeline_config(cfg, output_dir);
    generate_preprocessor_config(cfg, output_dir);

    // 7. Copy tokenizer files if present
    for (const auto& tokenizer_file : {"tokenizer.json", "tokenizer_config.json",
                                        "special_tokens_map.json", "vocab.json",
                                        "merges.txt"}) {
        auto src_path = model_dir / tokenizer_file;
        if (std::filesystem::exists(src_path)) {
            std::filesystem::copy_file(src_path, output_dir / tokenizer_file,
                                       std::filesystem::copy_options::overwrite_existing);
        }
    }
}

}  // namespace ov::genai::modeling::models
