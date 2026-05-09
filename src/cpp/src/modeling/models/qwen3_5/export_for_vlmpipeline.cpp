// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/qwen3_5/export_for_vlmpipeline.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>
#include <openvino/op/result.hpp>
#include <openvino/pass/serialize.hpp>
#include <nlohmann/json.hpp>

#include "modeling/builder_context.hpp"
#include "modeling/models/qwen3_5/modeling_qwen3_5_text.hpp"
#include "modeling/models/qwen3_5/modeling_qwen3_5_vision.hpp"
#include "modeling/models/qwen3_5/processing_qwen3_5.hpp"
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

void enable_zero_copy_safetensors_if_unset() {
    if (std::getenv("OV_GENAI_USE_ZERO_COPY") != nullptr) {
        return;
    }
#ifdef _WIN32
    _putenv_s("OV_GENAI_USE_ZERO_COPY", "1");
#else
    setenv("OV_GENAI_USE_ZERO_COPY", "1", 0);
#endif
}

}  // namespace

namespace ov::genai::modeling::models {

// =============================================================================
// 1. Vision Embeddings Model (PatchEmbed only)
// =============================================================================

std::shared_ptr<ov::Model> create_qwen3_5_vision_embeddings_model(
    const Qwen3_5Config& cfg,
    weights::WeightSource& source,
    weights::WeightFinalizer& finalizer) {

    BuilderContext ctx;
    Qwen3_5VisionModel model(ctx, cfg.vision);
    model.packed_mapping().rules.push_back({"model.", "", 0});

    weights::LoadOptions options;
    options.allow_unmatched = true;
    options.allow_missing = true;
    options.report_missing = false;
    options.report_unmatched = true;
    weights::load_model(model, source, finalizer, options);

    const int32_t in_channels = cfg.vision.in_channels;
    const int32_t temporal_patch = cfg.vision.temporal_patch_size;
    const int32_t patch_size = cfg.vision.patch_size;
    const int64_t channel_dim = static_cast<int64_t>(in_channels) * temporal_patch * patch_size * patch_size;

    // VLMPipeline calls set_tensor("hidden_states", ...) on this model
    auto hidden_states = ctx.parameter("hidden_states",
                                       ov::element::f32,
                                       ov::PartialShape{-1, channel_dim});

    auto output = model.patch_embed().forward(hidden_states);

    auto result = std::make_shared<ov::op::v0::Result>(output.output());
    set_name(result, "output");
    return ctx.build_model({result->output(0)});
}

// =============================================================================
// 2. Vision Merger Model (Blocks + Merger + DeepstackMergers)
// =============================================================================

std::shared_ptr<ov::Model> create_qwen3_5_vision_merger_model(
    const Qwen3_5Config& cfg,
    weights::WeightSource& source,
    weights::WeightFinalizer& finalizer) {

    (void)cfg;

    BuilderContext ctx;
    Qwen3_5VisionModel model(ctx, cfg.vision);
    model.packed_mapping().rules.push_back({"model.", "", 0});

    weights::LoadOptions options;
    options.allow_unmatched = true;
    options.allow_missing = true;
    options.report_missing = false;
    options.report_unmatched = true;
    weights::load_model(model, source, finalizer, options);

    const int32_t head_dim = cfg.vision.head_dim();

    auto hidden_states = ctx.parameter("hidden_states",
                                        ov::element::f32,
                                        ov::PartialShape{-1, cfg.vision.hidden_size});

    auto rotary_pos_emb = ctx.parameter("rotary_pos_emb",
                                         ov::element::f32,
                                         ov::PartialShape{-1, head_dim});

    auto attention_mask = ctx.parameter("attention_mask",
                                         ov::element::f32,
                                         ov::PartialShape{1, -1, -1});

    // Compute cos/sin from rotary_pos_emb
    auto rotary_cos = rotary_pos_emb.cos();
    auto rotary_sin = rotary_pos_emb.sin();

    // Run blocks + merger + deepstack via forward_blocks (no PatchEmbed)
    auto output = model.forward_blocks(hidden_states, rotary_cos, rotary_sin);

    ov::OutputVector results;

    auto lhs_result = std::make_shared<ov::op::v0::Result>(output.visual_embeds.output());
    set_name(lhs_result, "last_hidden_state");
    results.push_back(lhs_result->output(0));

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

std::shared_ptr<ov::Model> create_qwen3_5_vision_pos_model(
    const Qwen3_5Config& cfg,
    weights::WeightSource& source) {

    BuilderContext ctx;

    std::string pos_embed_name = resolve_pos_embed_name(source);
    const ov::Tensor& pos_weight_raw = source.get_tensor(pos_embed_name);

    auto pos_weight = ops::constant(pos_weight_raw, &ctx.op_context());

    auto input_indices = ctx.parameter("input",
                                        ov::element::i64,
                                        ov::PartialShape{4, -1});

    std::vector<Tensor> corner_embeds;
    corner_embeds.reserve(4);
    for (int64_t corner = 0; corner < 4; ++corner) {
        auto corner_indices = ops::slice(input_indices, corner, corner + 1, 1, 0).squeeze(0);
        auto gathered = ops::gather(pos_weight, corner_indices, 0);
        corner_embeds.push_back(gathered);
    }

    auto output = ops::tensor::stack(corner_embeds, 0);

    auto result = std::make_shared<ov::op::v0::Result>(output.output());
    set_name(result, "output");
    return ctx.build_model({result->output(0)});
}

// =============================================================================
// 4. Text Embeddings Model (VocabEmbedding only)
// =============================================================================

std::shared_ptr<ov::Model> create_qwen3_5_text_embeddings_model(
    const Qwen3_5Config& cfg,
    weights::WeightSource& source,
    weights::WeightFinalizer& finalizer) {

    BuilderContext ctx;

    VocabEmbedding embed_tokens(ctx, "model.embed_tokens");
    embed_tokens.packed_mapping().rules.push_back({"model.language_model.", "model.", 0});
    embed_tokens.packed_mapping().rules.push_back({"language_model.", "model.", 0});

    weights::LoadOptions options;
    options.allow_unmatched = true;
    options.allow_missing = false;
    options.report_missing = false;
    options.report_unmatched = false;
    weights::load_model(embed_tokens, source, finalizer, options);

    auto input_ids = ctx.parameter("input_ids",
                                    ov::element::i64,
                                    ov::PartialShape{-1, -1});

    auto embeddings = embed_tokens.forward(input_ids);

    auto result = std::make_shared<ov::op::v0::Result>(embeddings.output());
    set_name(result, "output");
    return ctx.build_model({result->output(0)});
}

// =============================================================================
// 5. Language Model (Hybrid Attention Decoder + LMHead)
// =============================================================================

std::shared_ptr<ov::Model> create_qwen3_5_language_model(
    const Qwen3_5Config& cfg,
    weights::WeightSource& source,
    weights::WeightFinalizer& finalizer) {

    // Export language model with inputs_embeds mode, no visual inputs.
    // VLMPipeline's InputsEmbedder handles the visual-text token merge externally,
    // so the language model just receives pre-merged inputs_embeds.
    //
    // Phase 1: No per-layer DeepStack injection, no EmbeddingInjector.
    // Model inputs: inputs_embeds [B, S, H], attention_mask [B, S],
    //               position_ids [4, B, S] (text + MRoPE), beam_idx [B]
    // Model outputs: logits [B, S, V]
    return create_qwen3_5_text_model(cfg, source, finalizer,
                                     /*use_inputs_embeds=*/true,
                                     /*enable_visual_inputs=*/false);
}

// =============================================================================
// Config Generation
// =============================================================================

namespace {

void generate_vlmpipeline_config(const Qwen3_5Config& cfg,
                                 const std::filesystem::path& output_dir) {
    nlohmann::json config;
    config["model_type"] = "qwen3_5";
    config["architectures"] = nlohmann::json::array({"Qwen3_5ForConditionalGeneration"});
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
    text_config["partial_rotary_factor"] = cfg.text.partial_rotary_factor;
    text_config["full_attention_interval"] = cfg.text.full_attention_interval;

    // MRoPE config
    nlohmann::json rope_config;
    rope_config["mrope_interleaved"] = cfg.text.rope.mrope_interleaved;
    rope_config["mrope_section"] = cfg.text.rope.mrope_section;
    text_config["rope_scaling"] = rope_config;

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

void generate_preprocessor_config(const Qwen3_5Config& cfg,
                                  const std::filesystem::path& output_dir) {
    nlohmann::json preproc;
    preproc["image_processor_type"] = "Qwen3_5ImageProcessor";

    nlohmann::json size;
    size["shortest_edge"] = 56 * 56;
    size["longest_edge"] = 28 * 28 * 1280;
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

void export_qwen3_5_for_vlmpipeline(
    const std::filesystem::path& model_dir,
    const std::filesystem::path& output_dir,
    const Qwen3_5ExportOptions& options) {

    enable_zero_copy_safetensors_if_unset();

    auto cfg = Qwen3_5Config::from_json_file(model_dir / "config.json");

    auto data = ov::genai::safetensors::load_safetensors(model_dir);
    ov::genai::safetensors::SafetensorsWeightSource source(std::move(data));

    std::filesystem::create_directories(output_dir);

    auto vision_quant_config = create_quantization_config(
        options.vision_quant_mode, options.vision_quant_group_size,
        options.vision_quant_backup_mode);
    auto text_quant_config = create_quantization_config(
        options.text_quant_mode, options.text_quant_group_size,
        options.text_quant_backup_mode);

    // 1. Export vision_embeddings_model (PatchEmbed)
    {
        ov::genai::safetensors::SafetensorsWeightFinalizer finalizer(vision_quant_config);
        auto model = create_qwen3_5_vision_embeddings_model(cfg, source, finalizer);
        ov::serialize(model,
                      (output_dir / "openvino_vision_embeddings_model.xml").string(),
                      (output_dir / "openvino_vision_embeddings_model.bin").string());
    }

    // 2. Export vision_embeddings_merger_model (Blocks + Merger + Deepstack)
    {
        ov::genai::safetensors::SafetensorsWeightFinalizer finalizer(vision_quant_config);
        auto model = create_qwen3_5_vision_merger_model(cfg, source, finalizer);
        ov::serialize(model,
                      (output_dir / "openvino_vision_embeddings_merger_model.xml").string(),
                      (output_dir / "openvino_vision_embeddings_merger_model.bin").string());
    }

    // 3. Export vision_embeddings_pos_model (Position embedding lookup)
    {
        auto model = create_qwen3_5_vision_pos_model(cfg, source);
        ov::serialize(model,
                      (output_dir / "openvino_vision_embeddings_pos_model.xml").string(),
                      (output_dir / "openvino_vision_embeddings_pos_model.bin").string());
    }

    // 4. Export text_embeddings_model (VocabEmbedding)
    {
        ov::genai::safetensors::SafetensorsWeightFinalizer finalizer(text_quant_config);
        auto model = create_qwen3_5_text_embeddings_model(cfg, source, finalizer);
        ov::serialize(model,
                      (output_dir / "openvino_text_embeddings_model.xml").string(),
                      (output_dir / "openvino_text_embeddings_model.bin").string());
    }

    // 5. Export language_model (Hybrid Attention Decoder + LMHead)
    {
        ov::genai::safetensors::SafetensorsWeightFinalizer finalizer(text_quant_config);
        auto model = create_qwen3_5_language_model(cfg, source, finalizer);
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
                                        "merges.txt", "openvino_tokenizer.xml",
                                        "openvino_tokenizer.bin", "openvino_detokenizer.xml",
                                        "openvino_detokenizer.bin"}) {
        auto src_path = model_dir / tokenizer_file;
        if (std::filesystem::exists(src_path)) {
            std::filesystem::copy_file(src_path, output_dir / tokenizer_file,
                                       std::filesystem::copy_options::overwrite_existing);
        }
    }
}

// =============================================================================
// In-memory serialization for ModelsMap
// =============================================================================

std::pair<std::string, ov::Tensor> serialize_qwen3_5_model_to_memory(
    const std::shared_ptr<ov::Model>& model) {
    std::ostringstream xml_stream;
    std::ostringstream bin_stream;

    ov::pass::Serialize serializer(xml_stream, bin_stream);
    serializer.run_on_model(std::const_pointer_cast<ov::Model>(model));

    std::string xml_str = xml_stream.str();
    std::string bin_str = bin_stream.str();

    ov::Tensor weights(ov::element::u8, {bin_str.size()});
    std::memcpy(weights.data(), bin_str.data(), bin_str.size());

    return {std::move(xml_str), std::move(weights)};
}

ModelsMap build_qwen3_5_models_map(
    const std::filesystem::path& model_dir,
    const Qwen3_5ExportOptions& options) {

    enable_zero_copy_safetensors_if_unset();

    auto cfg = Qwen3_5Config::from_json_file(model_dir / "config.json");

    auto data = ov::genai::safetensors::load_safetensors(model_dir);
    ov::genai::safetensors::SafetensorsWeightSource source(std::move(data));

    auto vision_quant_config = create_quantization_config(
        options.vision_quant_mode, options.vision_quant_group_size,
        options.vision_quant_backup_mode);
    auto text_quant_config = create_quantization_config(
        options.text_quant_mode, options.text_quant_group_size,
        options.text_quant_backup_mode);

    ModelsMap models_map;

    // 1. Vision embeddings (PatchEmbed)
    {
        ov::genai::safetensors::SafetensorsWeightFinalizer finalizer(vision_quant_config);
        auto model = create_qwen3_5_vision_embeddings_model(cfg, source, finalizer);
        models_map["vision_embeddings"] = serialize_qwen3_5_model_to_memory(model);
    }

    // 2. Vision merger (Blocks + Merger + Deepstack)
    {
        ov::genai::safetensors::SafetensorsWeightFinalizer finalizer(vision_quant_config);
        auto model = create_qwen3_5_vision_merger_model(cfg, source, finalizer);
        models_map["vision_embeddings_merger"] = serialize_qwen3_5_model_to_memory(model);
    }

    // 3. Vision position embeddings (Gather lookup)
    {
        auto model = create_qwen3_5_vision_pos_model(cfg, source);
        models_map["vision_embeddings_pos"] = serialize_qwen3_5_model_to_memory(model);
    }

    // 4. Text embeddings (VocabEmbedding)
    {
        ov::genai::safetensors::SafetensorsWeightFinalizer finalizer(text_quant_config);
        auto model = create_qwen3_5_text_embeddings_model(cfg, source, finalizer);
        models_map["text_embeddings"] = serialize_qwen3_5_model_to_memory(model);
    }

    // 5. Language model (Decoder + LMHead)
    {
        ov::genai::safetensors::SafetensorsWeightFinalizer finalizer(text_quant_config);
        auto model = create_qwen3_5_language_model(cfg, source, finalizer);
        models_map["language"] = serialize_qwen3_5_model_to_memory(model);
    }

    return models_map;
}

}  // namespace ov::genai::modeling::models
