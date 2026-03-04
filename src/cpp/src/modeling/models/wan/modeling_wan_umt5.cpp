// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/wan/modeling_wan_umt5.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>

#include <openvino/core/except.hpp>
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>

#include "json_utils.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/ops/shape.hpp"
#include "modeling/ops/nn.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace {

void read_config_json_file(const std::filesystem::path& path, nlohmann::json& data) {
    std::ifstream file(path);
    if (!file.is_open()) {
        OPENVINO_THROW("Failed to open config file: ", path.string());
    }
    file >> data;
}

std::filesystem::path resolve_config_path(const std::filesystem::path& path) {
    if (std::filesystem::is_directory(path)) {
        return path / "config.json";
    }
    return path;
}

ov::genai::modeling::Tensor abs_tensor(const ov::genai::modeling::Tensor& x) {
    auto node = std::make_shared<ov::opset13::Abs>(x.output());
    return ov::genai::modeling::Tensor(node, x.context());
}

ov::genai::modeling::Tensor floor_tensor(const ov::genai::modeling::Tensor& x) {
    auto node = std::make_shared<ov::opset13::Floor>(x.output());
    return ov::genai::modeling::Tensor(node, x.context());
}

ov::genai::modeling::Tensor minimum_tensor(const ov::genai::modeling::Tensor& a,
                                            const ov::genai::modeling::Tensor& b) {
    auto node = std::make_shared<ov::opset13::Minimum>(a.output(), b.output());
    return ov::genai::modeling::Tensor(node, a.context());
}

auto set_name = [](auto node, const std::string& name) {
    node->output(0).set_names({name});
    node->set_friendly_name(name);
};

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

int32_t UMT5Config::inner_dim() const {
    return num_heads > 0 ? num_heads * d_kv : 0;
}

void UMT5Config::finalize() {
    if (dense_act_fn.empty()) {
        dense_act_fn = "gelu_new";
    }
}

void UMT5Config::validate() const {
    if (vocab_size <= 0) {
        OPENVINO_THROW("UMT5Config.vocab_size must be > 0");
    }
    if (d_model <= 0 || d_ff <= 0 || d_kv <= 0) {
        OPENVINO_THROW("UMT5Config.d_model/d_ff/d_kv must be > 0");
    }
    if (num_heads <= 0 || num_layers <= 0) {
        OPENVINO_THROW("UMT5Config.num_heads/num_layers must be > 0");
    }
    if (relative_attention_num_buckets <= 0 || relative_attention_max_distance <= 0) {
        OPENVINO_THROW("UMT5Config.relative_attention_num_buckets/max_distance must be > 0");
    }
    if (relative_attention_num_buckets < 4 || (relative_attention_num_buckets % 2) != 0) {
        OPENVINO_THROW("UMT5Config.relative_attention_num_buckets must be an even value >= 4");
    }
    if (inner_dim() <= 0) {
        OPENVINO_THROW("UMT5Config.inner_dim must be > 0");
    }
}

UMT5Config UMT5Config::from_json(const nlohmann::json& data) {
    using ov::genai::utils::read_json_param;
    UMT5Config cfg;

    read_json_param(data, "_class_name", cfg.class_name);
    read_json_param(data, "vocab_size", cfg.vocab_size);
    read_json_param(data, "d_model", cfg.d_model);
    read_json_param(data, "d_ff", cfg.d_ff);
    read_json_param(data, "d_kv", cfg.d_kv);
    read_json_param(data, "num_heads", cfg.num_heads);
    read_json_param(data, "num_layers", cfg.num_layers);
    read_json_param(data, "relative_attention_num_buckets", cfg.relative_attention_num_buckets);
    read_json_param(data, "relative_attention_max_distance", cfg.relative_attention_max_distance);
    read_json_param(data, "dropout_rate", cfg.dropout_rate);
    read_json_param(data, "layer_norm_epsilon", cfg.layer_norm_epsilon);
    read_json_param(data, "dense_act_fn", cfg.dense_act_fn);
    read_json_param(data, "is_gated_act", cfg.is_gated_act);

    std::string feed_forward_proj;
    read_json_param(data, "feed_forward_proj", feed_forward_proj);
    if (!feed_forward_proj.empty() && feed_forward_proj.find("gated") != std::string::npos) {
        cfg.is_gated_act = true;
    }

    cfg.finalize();
    cfg.validate();
    return cfg;
}

UMT5Config UMT5Config::from_json_file(const std::filesystem::path& config_path) {
    nlohmann::json data;
    read_config_json_file(resolve_config_path(config_path), data);
    return from_json(data);
}

UMT5Attention::UMT5Attention(BuilderContext& ctx,
                             const std::string& name,
                             const UMT5Config& cfg,
                             Module* parent)
    : Module(name, ctx, parent),
      num_heads_(cfg.num_heads),
      head_dim_(cfg.d_kv),
      inner_dim_(cfg.inner_dim()),
      num_buckets_(cfg.relative_attention_num_buckets),
      max_distance_(cfg.relative_attention_max_distance) {
    q_weight_ = &register_parameter("q.weight");
    k_weight_ = &register_parameter("k.weight");
    v_weight_ = &register_parameter("v.weight");
    o_weight_ = &register_parameter("o.weight");
    relative_attention_bias_ = &register_parameter("relative_attention_bias.weight");
}

Tensor UMT5Attention::compute_bias(const Tensor& seq_len) const {
    auto* ctx = seq_len.context();
    auto positions = ops::range(seq_len, 0, 1, ov::element::i64);
    auto q_pos = positions.unsqueeze(1);
    auto k_pos = positions.unsqueeze(0);
    auto rel_pos = k_pos - q_pos;  // [seq, seq]

    const int32_t half_buckets = num_buckets_ / 2;
    const int32_t max_exact = half_buckets / 2;

    auto one_i64 = Tensor(ops::const_scalar(ctx, int64_t(1)), ctx);
    auto zero_i64 = Tensor(ops::const_scalar(ctx, int64_t(0)), ctx);
    auto half_bucket_i64 = Tensor(ops::const_scalar(ctx, int64_t(half_buckets)), ctx);

    auto is_pos = ops::greater_equal(rel_pos, one_i64);
    auto pos_offset = ops::where(is_pos, half_bucket_i64, zero_i64);
    auto rel_pos_abs = abs_tensor(rel_pos);

    auto max_exact_i64 = Tensor(ops::const_scalar(ctx, int64_t(max_exact)), ctx);
    auto max_exact_minus_one = Tensor(ops::const_scalar(ctx, int64_t(max_exact - 1)), ctx);
    auto is_small = ops::less_equal(rel_pos_abs, max_exact_minus_one);

    auto rel_pos_f = rel_pos_abs.to(ov::element::f32);
    auto max_exact_f = Tensor(ops::const_scalar(ctx, static_cast<float>(max_exact)), ctx);
    auto safe_rel_pos = ops::where(is_small, max_exact_f, rel_pos_f);

    const float log_base = std::log(static_cast<float>(max_distance_) / static_cast<float>(max_exact));
    auto log_base_f = Tensor(ops::const_scalar(ctx, log_base), ctx);
    auto log_ratio = safe_rel_pos.log() / log_base_f;
    auto bucket_scale = Tensor(ops::const_scalar(ctx, static_cast<float>(half_buckets - max_exact)), ctx);
    auto scaled = log_ratio * bucket_scale;
    auto rel_pos_large = floor_tensor(scaled).to(ov::element::i64) + max_exact_i64;

    auto max_bucket_i64 = Tensor(ops::const_scalar(ctx, int64_t(half_buckets - 1)), ctx);
    auto rel_pos_large_clamped = minimum_tensor(rel_pos_large, max_bucket_i64);

    auto rel_bucket = pos_offset + ops::where(is_small, rel_pos_abs, rel_pos_large_clamped);
    auto rel_bucket_i32 = rel_bucket.to(ov::element::i32);

    auto bias = ops::gather(relative_attention_bias_->value(), rel_bucket_i32, 0);
    return bias.permute({2, 0, 1}).unsqueeze(0);
}

Tensor UMT5Attention::forward(const Tensor& hidden_states, const Tensor* attention_mask) const {
    auto q = ops::linear(hidden_states, q_weight_->value());
    auto k = ops::linear(hidden_states, k_weight_->value());
    auto v = ops::linear(hidden_states, v_weight_->value());

    auto q_heads = q.reshape({0, 0, num_heads_, head_dim_}).permute({0, 2, 1, 3});
    auto k_heads = k.reshape({0, 0, num_heads_, head_dim_}).permute({0, 2, 1, 3});
    auto v_heads = v.reshape({0, 0, num_heads_, head_dim_}).permute({0, 2, 1, 3});

    auto scores = ops::matmul(q_heads, k_heads, false, true);

    auto seq_len = Tensor(shape::dim(hidden_states, 1), hidden_states.context()).squeeze(0);
    auto position_bias = compute_bias(seq_len).to(scores.dtype());
    auto logits = scores + position_bias;

    if (attention_mask) {
        logits = logits + attention_mask->to(scores.dtype());
    }

    auto attn = logits.softmax(3);
    auto context = ops::matmul(attn, v_heads);
    auto merged = context.permute({0, 2, 1, 3}).reshape({0, 0, inner_dim_});
    return ops::linear(merged, o_weight_->value());
}

UMT5DenseReluDense::UMT5DenseReluDense(BuilderContext& ctx,
                                       const std::string& name,
                                       const UMT5Config& cfg,
                                       Module* parent)
    : Module(name, ctx, parent),
      gated_(cfg.is_gated_act),
      gelu_approximate_(cfg.dense_act_fn == "gelu_new") {
    if (gated_) {
        wi_0_weight_ = &register_parameter("wi_0.weight");
        wi_1_weight_ = &register_parameter("wi_1.weight");
    } else {
        wi_weight_ = &register_parameter("wi.weight");
    }
    wo_weight_ = &register_parameter("wo.weight");
}

Tensor UMT5DenseReluDense::forward(const Tensor& hidden_states) const {
    if (gated_) {
        auto gelu = ops::nn::gelu(ops::linear(hidden_states, wi_0_weight_->value()), gelu_approximate_);
        auto linear = ops::linear(hidden_states, wi_1_weight_->value());
        auto gated = gelu * linear;
        return ops::linear(gated, wo_weight_->value());
    }
    auto proj = ops::linear(hidden_states, wi_weight_->value());
    auto act = ops::nn::gelu(proj, gelu_approximate_);
    return ops::linear(act, wo_weight_->value());
}

UMT5LayerSelfAttention::UMT5LayerSelfAttention(BuilderContext& ctx,
                                               const std::string& name,
                                               const UMT5Config& cfg,
                                               Module* parent)
    : Module(name, ctx, parent),
      self_attention_(ctx, "SelfAttention", cfg, this),
      layer_norm_(ctx, "layer_norm", cfg.layer_norm_epsilon, this) {
    register_module("SelfAttention", &self_attention_);
    register_module("layer_norm", &layer_norm_);
}

Tensor UMT5LayerSelfAttention::forward(const Tensor& hidden_states, const Tensor* attention_mask) const {
    auto normed = layer_norm_.forward(hidden_states);
    auto attn_out = self_attention_.forward(normed, attention_mask);
    return hidden_states + attn_out;
}

UMT5LayerFF::UMT5LayerFF(BuilderContext& ctx,
                         const std::string& name,
                         const UMT5Config& cfg,
                         Module* parent)
    : Module(name, ctx, parent),
      dense_(ctx, "DenseReluDense", cfg, this),
      layer_norm_(ctx, "layer_norm", cfg.layer_norm_epsilon, this) {
    register_module("DenseReluDense", &dense_);
    register_module("layer_norm", &layer_norm_);
}

Tensor UMT5LayerFF::forward(const Tensor& hidden_states) const {
    auto normed = layer_norm_.forward(hidden_states);
    auto out = dense_.forward(normed);
    return hidden_states + out;
}

UMT5Block::UMT5Block(BuilderContext& ctx,
                     const std::string& name,
                     const UMT5Config& cfg,
                     Module* parent)
    : Module(name, ctx, parent),
      self_attention_(ctx, "layer.0", cfg, this),
      ffn_(ctx, "layer.1", cfg, this) {
    register_module("layer.0", &self_attention_);
    register_module("layer.1", &ffn_);
}

Tensor UMT5Block::forward(const Tensor& hidden_states, const Tensor* attention_mask) const {
    auto attn_out = self_attention_.forward(hidden_states, attention_mask);
    return ffn_.forward(attn_out);
}

UMT5EncoderStack::UMT5EncoderStack(BuilderContext& ctx,
                                   const std::string& name,
                                   const UMT5Config& cfg,
                                   Module* parent)
    : Module(name, ctx, parent),
      blocks_(),
      final_layer_norm_(ctx, "final_layer_norm", cfg.layer_norm_epsilon, this) {
    blocks_.reserve(static_cast<size_t>(cfg.num_layers));
    for (int32_t i = 0; i < cfg.num_layers; ++i) {
        std::string block_name = "block." + std::to_string(i);
        blocks_.emplace_back(ctx, block_name, cfg, this);
        register_module(block_name, &blocks_.back());
    }
    register_module("final_layer_norm", &final_layer_norm_);
}

Tensor UMT5EncoderStack::forward(const Tensor& hidden_states, const Tensor& attention_mask) const {
    Tensor output = hidden_states;
    for (const auto& block : blocks_) {
        output = block.forward(output, &attention_mask);
    }
    return final_layer_norm_.forward(output);
}

UMT5EncoderModel::UMT5EncoderModel(BuilderContext& ctx,
                                   const UMT5Config& cfg,
                                   Module* parent)
    : Module("", ctx, parent),
      cfg_(cfg),
      shared_(ctx, "shared", this),
      encoder_(ctx, "encoder", cfg, this) {
    register_module("shared", &shared_);
    register_module("encoder", &encoder_);
}

Tensor UMT5EncoderModel::prepare_attention_mask(const Tensor& attention_mask) const {
    auto* ctx = attention_mask.context();
    auto mask_i64 = attention_mask.to(ov::element::i64);
    auto one = Tensor(ops::const_scalar(ctx, int64_t(1)), ctx);
    auto keep = ops::greater_equal(mask_i64, one);

    auto zero = Tensor(ops::const_scalar(ctx, 0.0f), ctx);
    auto neg = Tensor(ops::const_scalar(ctx, -65504.0f), ctx);
    auto mask = ops::where(keep, zero, neg);
    return mask.unsqueeze({1, 2});
}

Tensor UMT5EncoderModel::forward(const Tensor& input_ids, const Tensor& attention_mask) {
    auto hidden_states = shared_.forward(input_ids);
    auto attn_mask = prepare_attention_mask(attention_mask);
    return encoder_.forward(hidden_states, attn_mask);
}

std::shared_ptr<ov::Model> create_umt5_text_encoder_model(
    const UMT5Config& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    BuilderContext ctx;
    UMT5EncoderModel model(ctx, cfg);

    ov::genai::modeling::weights::load_model(model, source, finalizer);

    auto input_ids = ctx.parameter("input_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto attention_mask = ctx.parameter("attention_mask", ov::element::i64, ov::PartialShape{-1, -1});

    auto output = model.forward(input_ids, attention_mask);
    auto result = std::make_shared<ov::op::v0::Result>(output.output());
    set_name(result, "last_hidden_state");
    return ctx.build_model({result->output(0)});
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
