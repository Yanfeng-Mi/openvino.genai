// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace ov {
class Model;
}  // namespace ov

namespace ov {
namespace genai {
namespace modeling {
namespace weights {
class WeightFinalizer;
class WeightSource;
}  // namespace weights
}  // namespace modeling
}  // namespace genai
}  // namespace ov

#include "modeling/layers/rms_norm.hpp"
#include "modeling/layers/vocab_embedding.hpp"
#include "modeling/module.hpp"
#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

struct UMT5Config {
    std::string class_name = "UMT5EncoderModel";
    int32_t vocab_size = 0;
    int32_t d_model = 0;
    int32_t d_ff = 0;
    int32_t d_kv = 0;
    int32_t num_heads = 0;
    int32_t num_layers = 0;
    int32_t relative_attention_num_buckets = 32;
    int32_t relative_attention_max_distance = 128;
    float dropout_rate = 0.0f;
    float layer_norm_epsilon = 1e-6f;
    bool is_gated_act = false;
    std::string dense_act_fn = "gelu_new";

    int32_t inner_dim() const;
    void finalize();
    void validate() const;

    static UMT5Config from_json(const nlohmann::json& data);
    static UMT5Config from_json_file(const std::filesystem::path& config_path);
};

class UMT5Attention : public Module {
public:
    UMT5Attention(BuilderContext& ctx,
                  const std::string& name,
                  const UMT5Config& cfg,
                  Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states, const Tensor* attention_mask) const;

private:
    Tensor compute_bias(const Tensor& seq_len) const;

    int32_t num_heads_ = 0;
    int32_t head_dim_ = 0;
    int32_t inner_dim_ = 0;
    int32_t num_buckets_ = 0;
    int32_t max_distance_ = 0;

    WeightParameter* q_weight_ = nullptr;
    WeightParameter* k_weight_ = nullptr;
    WeightParameter* v_weight_ = nullptr;
    WeightParameter* o_weight_ = nullptr;
    WeightParameter* relative_attention_bias_ = nullptr;
};

class UMT5DenseReluDense : public Module {
public:
    UMT5DenseReluDense(BuilderContext& ctx,
                       const std::string& name,
                       const UMT5Config& cfg,
                       Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states) const;

private:
    bool gated_ = false;
    bool gelu_approximate_ = true;
    WeightParameter* wi_weight_ = nullptr;
    WeightParameter* wi_0_weight_ = nullptr;
    WeightParameter* wi_1_weight_ = nullptr;
    WeightParameter* wo_weight_ = nullptr;
};

class UMT5LayerSelfAttention : public Module {
public:
    UMT5LayerSelfAttention(BuilderContext& ctx,
                           const std::string& name,
                           const UMT5Config& cfg,
                           Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states, const Tensor* attention_mask) const;

private:
    UMT5Attention self_attention_;
    RMSNorm layer_norm_;
};

class UMT5LayerFF : public Module {
public:
    UMT5LayerFF(BuilderContext& ctx,
                const std::string& name,
                const UMT5Config& cfg,
                Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states) const;

private:
    UMT5DenseReluDense dense_;
    RMSNorm layer_norm_;
};

class UMT5Block : public Module {
public:
    UMT5Block(BuilderContext& ctx,
              const std::string& name,
              const UMT5Config& cfg,
              Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states, const Tensor* attention_mask) const;

private:
    UMT5LayerSelfAttention self_attention_;
    UMT5LayerFF ffn_;
};

class UMT5EncoderStack : public Module {
public:
    UMT5EncoderStack(BuilderContext& ctx,
                     const std::string& name,
                     const UMT5Config& cfg,
                     Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states, const Tensor& attention_mask) const;

private:
    std::vector<UMT5Block> blocks_;
    RMSNorm final_layer_norm_;
};

class UMT5EncoderModel : public Module {
public:
    UMT5EncoderModel(BuilderContext& ctx,
                     const UMT5Config& cfg,
                     Module* parent = nullptr);

    Tensor forward(const Tensor& input_ids, const Tensor& attention_mask);

private:
    Tensor prepare_attention_mask(const Tensor& attention_mask) const;

    UMT5Config cfg_;
    VocabEmbedding shared_;
    UMT5EncoderStack encoder_;
};

std::shared_ptr<ov::Model> create_umt5_text_encoder_model(
    const UMT5Config& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
