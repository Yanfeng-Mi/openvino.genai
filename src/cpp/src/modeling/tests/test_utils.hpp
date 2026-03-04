// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <openvino/core/except.hpp>
#include <openvino/openvino.hpp>

#include "modeling/ops/ops.hpp"
#include "modeling/weights/weight_finalizer.hpp"
#include "modeling/weights/weight_source.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace tests {

ov::Tensor make_tensor(const std::vector<float>& data, const ov::Shape& shape);
std::vector<float> make_seq(size_t n, float start = 0.0f, float step = 1.0f);

std::vector<float> matmul_ref(const std::vector<float>& a,
                              const std::vector<float>& b,
                              size_t m,
                              size_t k,
                              size_t n);
std::vector<float> matmul_ref_transpose_a(const std::vector<float>& a,
                                          const std::vector<float>& b,
                                          size_t m,
                                          size_t k,
                                          size_t n);
std::vector<float> linear_ref(const std::vector<float>& x,
                              const std::vector<float>& w,
                              size_t rows,
                              size_t in_features,
                              size_t out_features);
std::vector<float> linear_ref_3d(const std::vector<float>& x,
                                 const std::vector<float>& w,
                                 size_t batch,
                                 size_t seq_len,
                                 size_t in_features,
                                 size_t out_features);
std::vector<float> linear_ref_3d_bias(const std::vector<float>& x,
                                      const std::vector<float>& w,
                                      const std::vector<float>& bias,
                                      size_t batch,
                                      size_t seq_len,
                                      size_t in_features,
                                      size_t out_features);
std::vector<float> mean_ref(const std::vector<float>& x, size_t rows, size_t cols);
std::vector<float> rms_ref(const std::vector<float>& x,
                           const std::vector<float>& weight,
                           size_t rows,
                           size_t cols,
                           float eps);
std::vector<float> embedding_ref(const std::vector<int64_t>& ids,
                                 const std::vector<float>& weight,
                                 size_t rows,
                                 size_t cols,
                                 size_t embed_dim);
std::vector<float> add_ref(const std::vector<float>& a, const std::vector<float>& b);
std::vector<float> mul_ref(const std::vector<float>& a, const std::vector<float>& b);
std::vector<float> silu_ref(const std::vector<float>& x);
std::vector<float> mlp_ref(const std::vector<float>& x,
                           const std::vector<float>& gate_w,
                           const std::vector<float>& up_w,
                           const std::vector<float>& down_w,
                           size_t batch,
                           size_t seq_len,
                           size_t hidden,
                           size_t intermediate);
std::vector<float> random_f32(size_t count, float low, float high, uint32_t seed);
struct Q41Quantized {
    ov::Tensor weights_u4;
    ov::Tensor scales_f16;
    ov::Tensor zps_u4;
    std::vector<uint8_t> weights_packed;
    std::vector<uint8_t> zps_packed;
    std::vector<ov::float16> scales;
    size_t group_num = 0;
    size_t group_size = 0;
    size_t k = 0;
};

// Helper to construct dequantization reference subgraph
ov::genai::modeling::Tensor make_dequant_subgraph(const Q41Quantized& q_weights,
                                                  ov::genai::modeling::OpContext* op_ctx);

Q41Quantized quantize_q41(const std::vector<float>& weights_f32,
                          size_t num_experts,
                          size_t n,
                          size_t k,
                          size_t group_size);
Q41Quantized quantize_q41(const std::vector<float>& weights_f32,
                          size_t n,
                          size_t k,
                          size_t group_size);

std::vector<float> dequantize_q41(const Q41Quantized& q,
                                  size_t num_experts,
                                  size_t n,
                                  size_t k);
std::vector<float> moe_ref(const std::vector<float>& hidden_states,
                           const std::vector<float>& gate_inp,
                           const std::vector<float>& gate_w,
                           const std::vector<float>& up_w,
                           const std::vector<float>& down_w,
                           size_t batch,
                           size_t seq_len,
                           size_t hidden_size,
                           size_t inter_size,
                           size_t num_experts,
                           size_t top_k);
std::vector<float> to_heads_ref(const std::vector<float>& x,
                                size_t batch,
                                size_t seq_len,
                                size_t num_heads,
                                size_t head_dim);
std::vector<float> rmsnorm_heads_ref(const std::vector<float>& x,
                                     const std::vector<float>& weight,
                                     size_t batch,
                                     size_t num_heads,
                                     size_t seq_len,
                                     size_t head_dim,
                                     float eps);
std::vector<float> merge_heads_ref(const std::vector<float>& x,
                                   size_t batch,
                                   size_t seq_len,
                                   size_t num_heads,
                                   size_t head_dim);
std::vector<float> apply_rope_ref(const std::vector<float>& x,
                                  const std::vector<int64_t>& positions,
                                  size_t batch,
                                  size_t seq_len,
                                  size_t num_heads,
                                  size_t head_dim,
                                  float rope_theta);
std::vector<float> repeat_kv_ref(const std::vector<float>& x,
                                 size_t batch,
                                 size_t num_heads,
                                 size_t num_kv_heads,
                                 size_t seq_len,
                                 size_t head_dim);
std::vector<float> attention_ref(const std::vector<float>& hidden,
                                 const std::vector<float>& q_w,
                                 const std::vector<float>& q_b,
                                 const std::vector<float>& k_w,
                                 const std::vector<float>& k_b,
                                 const std::vector<float>& v_w,
                                 const std::vector<float>& v_b,
                                 const std::vector<float>& o_w,
                                 const std::vector<float>& o_b,
                                 const std::vector<float>* q_norm_w,
                                 const std::vector<float>* k_norm_w,
                                 const std::vector<int64_t>& positions,
                                 size_t batch,
                                 size_t seq_len,
                                 size_t hidden_size,
                                 size_t num_heads,
                                 size_t num_kv_heads,
                                 size_t head_dim,
                                 float rope_theta,
                                 float rms_norm_eps,
                                 bool use_rope = true);
std::pair<std::vector<float>, std::vector<float>> linear_attention_ref(const std::vector<float>& q,
                                                                        const std::vector<float>& k,
                                                                        const std::vector<float>& v,
                                                                        const std::vector<float>& beta,
                                                                        const std::vector<float>& g,
                                                                        const std::vector<float>& initial_state,
                                                                        size_t batch,
                                                                        size_t seq_len,
                                                                        size_t num_heads,
                                                                        size_t head_dim);

/// Recommended tol by operator:
///   tol    | Usage Scenarios
///   -------|---------------------------
///   1e-6   | reshape, permute, tile, pad
///   1e-5   | rope, 
///   1e-3   | matmul, linear, MLP, attention
///   2.5e-3 | sin, cos, exp, log, softmax
///   5e-1   | MoE, Q41 fusion gemm
///   1.5e-2 | linear_attention (fp16 recurrent accumulation)
constexpr float k_tol_exact = 1e-6f;
constexpr float k_tol_linear = 1e-5f;
constexpr float k_tol_default = 1e-3f;
constexpr float k_tol_transcendental = 2.5e-3f;
constexpr float k_tol_linear_attn = 1.5e-2f;
constexpr float k_tol_moe = 5e-1f;

void expect_tensor_near(const ov::Tensor& output, const std::vector<float>& expected, float tol);

class DummyWeightSource : public weights::WeightSource {
public:
    void add(const std::string& name, const ov::Tensor& tensor) {
        if (!weights_.count(name)) {
            keys_.push_back(name);
        }
        weights_[name] = tensor;
    }

    std::vector<std::string> keys() const override {
        return keys_;
    }

    bool has(const std::string& name) const override {
        return weights_.count(name) != 0;
    }

    const ov::Tensor& get_tensor(const std::string& name) const override {
        auto it = weights_.find(name);
        if (it == weights_.end()) {
            OPENVINO_THROW("Unknown weight: ", name);
        }
        return it->second;
    }

private:
    std::unordered_map<std::string, ov::Tensor> weights_;
    std::vector<std::string> keys_;
};

class DummyWeightFinalizer : public weights::WeightFinalizer {
public:
    weights::FinalizedWeight finalize(const std::string& name, weights::WeightSource& source, OpContext& ctx) override {
        const auto& tensor = source.get_tensor(name);
        return weights::FinalizedWeight(ops::constant(tensor, &ctx), {});
    }
};

}  // namespace tests
}  // namespace modeling
}  // namespace genai
}  // namespace ov
