// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>

#include "modeling/builder_context.hpp"
#include "modeling/models/youtu/modeling_youtu.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace test_utils = ov::genai::modeling::tests;

namespace {

std::vector<float> make_rope_cos(const std::vector<int64_t>& positions,
                                 size_t batch,
                                 size_t seq_len,
                                 int32_t head_dim,
                                 float rope_theta) {
    const int32_t half_dim = head_dim / 2;
    std::vector<float> cos(static_cast<size_t>(batch * seq_len * half_dim));
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            int64_t pos = positions[b * seq_len + s];
            for (int32_t i = 0; i < half_dim; ++i) {
                float exponent = static_cast<float>(2 * i) / static_cast<float>(head_dim);
                float inv_freq = 1.0f / std::pow(rope_theta, exponent);
                float angle = static_cast<float>(pos) * inv_freq;
                cos[(b * seq_len + s) * half_dim + i] = std::cos(angle);
            }
        }
    }
    return cos;
}

std::vector<float> make_rope_sin(const std::vector<int64_t>& positions,
                                 size_t batch,
                                 size_t seq_len,
                                 int32_t head_dim,
                                 float rope_theta) {
    const int32_t half_dim = head_dim / 2;
    std::vector<float> sin(static_cast<size_t>(batch * seq_len * half_dim));
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            int64_t pos = positions[b * seq_len + s];
            for (int32_t i = 0; i < half_dim; ++i) {
                float exponent = static_cast<float>(2 * i) / static_cast<float>(head_dim);
                float inv_freq = 1.0f / std::pow(rope_theta, exponent);
                float angle = static_cast<float>(pos) * inv_freq;
                sin[(b * seq_len + s) * half_dim + i] = std::sin(angle);
            }
        }
    }
    return sin;
}

std::vector<float> slice_last_dim(const std::vector<float>& x,
                                  size_t batch,
                                  size_t heads,
                                  size_t seq_len,
                                  size_t head_dim,
                                  size_t start,
                                  size_t length) {
    std::vector<float> out(batch * heads * seq_len * length, 0.0f);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < heads; ++h) {
            for (size_t s = 0; s < seq_len; ++s) {
                const size_t in_base = ((b * heads + h) * seq_len + s) * head_dim + start;
                const size_t out_base = ((b * heads + h) * seq_len + s) * length;
                for (size_t d = 0; d < length; ++d) {
                    out[out_base + d] = x[in_base + d];
                }
            }
        }
    }
    return out;
}

std::vector<float> concat_last_dim(const std::vector<float>& a,
                                   const std::vector<float>& b,
                                   size_t batch,
                                   size_t heads,
                                   size_t seq_len,
                                   size_t dim_a,
                                   size_t dim_b) {
    std::vector<float> out(batch * heads * seq_len * (dim_a + dim_b), 0.0f);
    for (size_t b_idx = 0; b_idx < batch; ++b_idx) {
        for (size_t h = 0; h < heads; ++h) {
            for (size_t s = 0; s < seq_len; ++s) {
                const size_t base_out = ((b_idx * heads + h) * seq_len + s) * (dim_a + dim_b);
                const size_t base_a = ((b_idx * heads + h) * seq_len + s) * dim_a;
                const size_t base_b = ((b_idx * heads + h) * seq_len + s) * dim_b;
                for (size_t d = 0; d < dim_a; ++d) {
                    out[base_out + d] = a[base_a + d];
                }
                for (size_t d = 0; d < dim_b; ++d) {
                    out[base_out + dim_a + d] = b[base_b + d];
                }
            }
        }
    }
    return out;
}

std::vector<float> apply_rope_interleave_ref(const std::vector<float>& x,
                                             const std::vector<float>& cos,
                                             const std::vector<float>& sin,
                                             size_t batch,
                                             size_t heads,
                                             size_t seq_len,
                                             int32_t head_dim) {
    const int32_t half_dim = head_dim / 2;
    std::vector<float> out(x.size(), 0.0f);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < heads; ++h) {
            for (size_t s = 0; s < seq_len; ++s) {
                std::vector<float> interleaved(static_cast<size_t>(head_dim));
                const size_t base = ((b * heads + h) * seq_len + s) * head_dim;
                for (int32_t i = 0; i < half_dim; ++i) {
                    interleaved[static_cast<size_t>(i)] = x[base + static_cast<size_t>(2 * i)];
                    interleaved[static_cast<size_t>(i + half_dim)] = x[base + static_cast<size_t>(2 * i + 1)];
                }

                for (int32_t i = 0; i < half_dim; ++i) {
                    float c = cos[(b * seq_len + s) * half_dim + i];
                    float sn = sin[(b * seq_len + s) * half_dim + i];
                    float x1 = interleaved[static_cast<size_t>(i)];
                    float x2 = interleaved[static_cast<size_t>(i + half_dim)];
                    out[base + static_cast<size_t>(i)] = x1 * c - x2 * sn;
                    out[base + static_cast<size_t>(i + half_dim)] = x1 * sn + x2 * c;
                }
            }
        }
    }
    return out;
}

std::vector<float> broadcast_heads(const std::vector<float>& x,
                                   size_t batch,
                                   size_t seq_len,
                                   size_t head_dim,
                                   size_t heads) {
    std::vector<float> out(batch * heads * seq_len * head_dim, 0.0f);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            const size_t in_base = (b * seq_len + s) * head_dim;
            for (size_t h = 0; h < heads; ++h) {
                const size_t out_base = ((b * heads + h) * seq_len + s) * head_dim;
                for (size_t d = 0; d < head_dim; ++d) {
                    out[out_base + d] = x[in_base + d];
                }
            }
        }
    }
    return out;
}

std::vector<float> attention_ref(const std::vector<float>& q,
                                 const std::vector<float>& k,
                                 const std::vector<float>& v,
                                 size_t batch,
                                 size_t heads,
                                 size_t seq_len,
                                 size_t qk_head_dim,
                                 size_t v_head_dim,
                                 float scale) {
    std::vector<float> context(batch * heads * seq_len * v_head_dim, 0.0f);
    std::vector<float> scores(seq_len, 0.0f);

    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < heads; ++h) {
            for (size_t i = 0; i < seq_len; ++i) {
                float max_score = -1e30f;
                for (size_t j = 0; j < seq_len; ++j) {
                    float acc = 0.0f;
                    if (j > i) {
                        scores[j] = -65504.0f;
                        continue;
                    }
                    const size_t q_base = ((b * heads + h) * seq_len + i) * qk_head_dim;
                    const size_t k_base = ((b * heads + h) * seq_len + j) * qk_head_dim;
                    for (size_t d = 0; d < qk_head_dim; ++d) {
                        acc += q[q_base + d] * k[k_base + d];
                    }
                    acc *= scale;
                    scores[j] = acc;
                    if (acc > max_score) {
                        max_score = acc;
                    }
                }
                float sum = 0.0f;
                for (size_t j = 0; j < seq_len; ++j) {
                    scores[j] = std::exp(scores[j] - max_score);
                    sum += scores[j];
                }
                const size_t ctx_base = ((b * heads + h) * seq_len + i) * v_head_dim;
                for (size_t d = 0; d < v_head_dim; ++d) {
                    float acc = 0.0f;
                    for (size_t j = 0; j < seq_len; ++j) {
                        const size_t v_base = ((b * heads + h) * seq_len + j) * v_head_dim;
                        acc += (scores[j] / sum) * v[v_base + d];
                    }
                    context[ctx_base + d] = acc;
                }
            }
        }
    }

    return context;
}

std::vector<float> youtu_mla_ref(const std::vector<float>& hidden,
                                 const std::vector<float>& q_a_w,
                                 const std::vector<float>& q_b_w,
                                 const std::vector<float>& kv_a_w,
                                 const std::vector<float>& kv_b_w,
                                 const std::vector<float>& o_w,
                                 const std::vector<float>& q_a_ln_w,
                                 const std::vector<float>& kv_a_ln_w,
                                 const std::vector<float>& rope_cos,
                                 const std::vector<float>& rope_sin,
                                 size_t batch,
                                 size_t seq_len,
                                 size_t hidden_size,
                                 size_t num_heads,
                                 size_t q_lora_rank,
                                 size_t kv_lora_rank,
                                 size_t qk_nope_head_dim,
                                 size_t qk_rope_head_dim,
                                 size_t qk_head_dim,
                                 size_t v_head_dim,
                                 float rms_norm_eps) {
    auto q_a = test_utils::linear_ref_3d(hidden, q_a_w, batch, seq_len, hidden_size, q_lora_rank);
    auto q_a_norm = test_utils::rms_ref(q_a, q_a_ln_w, batch * seq_len, q_lora_rank, rms_norm_eps);
    auto q_b = test_utils::linear_ref_3d(q_a_norm, q_b_w, batch, seq_len, q_lora_rank, num_heads * qk_head_dim);

    auto q_heads = test_utils::to_heads_ref(q_b, batch, seq_len, num_heads, qk_head_dim);
    auto q_pass = slice_last_dim(q_heads, batch, num_heads, seq_len, qk_head_dim, 0, qk_nope_head_dim);
    auto q_rot = slice_last_dim(q_heads, batch, num_heads, seq_len, qk_head_dim, qk_nope_head_dim, qk_rope_head_dim);
    q_rot = apply_rope_interleave_ref(q_rot, rope_cos, rope_sin, batch, num_heads, seq_len,
                                      static_cast<int32_t>(qk_rope_head_dim));

    auto kv_a = test_utils::linear_ref_3d(hidden, kv_a_w, batch, seq_len, hidden_size,
                                          kv_lora_rank + qk_rope_head_dim);
    std::vector<float> k_pass_latent(batch * seq_len * kv_lora_rank, 0.0f);
    std::vector<float> k_rot(batch * seq_len * qk_rope_head_dim, 0.0f);
    for (size_t i = 0; i < batch * seq_len; ++i) {
        std::memcpy(&k_pass_latent[i * kv_lora_rank],
                    &kv_a[i * (kv_lora_rank + qk_rope_head_dim)],
                    kv_lora_rank * sizeof(float));
        std::memcpy(&k_rot[i * qk_rope_head_dim],
                    &kv_a[i * (kv_lora_rank + qk_rope_head_dim) + kv_lora_rank],
                    qk_rope_head_dim * sizeof(float));
    }

    auto k_pass_norm = test_utils::rms_ref(k_pass_latent, kv_a_ln_w, batch * seq_len, kv_lora_rank, rms_norm_eps);
    auto kv_b = test_utils::linear_ref_3d(k_pass_norm, kv_b_w, batch, seq_len, kv_lora_rank,
                                          num_heads * (qk_nope_head_dim + v_head_dim));
    auto kv_heads = test_utils::to_heads_ref(kv_b, batch, seq_len, num_heads, qk_nope_head_dim + v_head_dim);
    auto k_pass = slice_last_dim(kv_heads, batch, num_heads, seq_len, qk_nope_head_dim + v_head_dim, 0,
                                 qk_nope_head_dim);
    auto v_heads = slice_last_dim(kv_heads, batch, num_heads, seq_len, qk_nope_head_dim + v_head_dim,
                                  qk_nope_head_dim, v_head_dim);

    auto k_rot_heads = broadcast_heads(k_rot, batch, seq_len, qk_rope_head_dim, num_heads);
    k_rot_heads = apply_rope_interleave_ref(k_rot_heads, rope_cos, rope_sin, batch, num_heads, seq_len,
                                            static_cast<int32_t>(qk_rope_head_dim));

    auto query_states = concat_last_dim(q_pass, q_rot, batch, num_heads, seq_len, qk_nope_head_dim, qk_rope_head_dim);
    auto key_states = concat_last_dim(k_pass, k_rot_heads, batch, num_heads, seq_len,
                                      qk_nope_head_dim, qk_rope_head_dim);

    const float scale = 1.0f / std::sqrt(static_cast<float>(qk_head_dim));
    auto context = attention_ref(query_states, key_states, v_heads, batch, num_heads, seq_len,
                                 qk_head_dim, v_head_dim, scale);
    auto merged = test_utils::merge_heads_ref(context, batch, seq_len, num_heads, v_head_dim);
    return test_utils::linear_ref_3d(merged, o_w, batch, seq_len, num_heads * v_head_dim, hidden_size);
}

}  // namespace

TEST(YoutuMLAttention, MatchesReference) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t batch = 1;
    const size_t seq_len = 2;
    const size_t hidden = 4;
    const size_t num_heads = 2;
    const size_t q_lora_rank = 2;
    const size_t kv_lora_rank = 2;
    const size_t qk_nope_head_dim = 2;
    const size_t qk_rope_head_dim = 4;
    const size_t qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;
    const size_t v_head_dim = 2;
    const size_t rope_half_dim = qk_rope_head_dim / 2;
    const float rope_theta = 10000.0f;
    const float rms_norm_eps = 1e-6f;

    const ov::Shape q_a_w_shape{q_lora_rank, hidden};
    const ov::Shape q_b_w_shape{num_heads * qk_head_dim, q_lora_rank};
    const ov::Shape kv_a_w_shape{kv_lora_rank + qk_rope_head_dim, hidden};
    const ov::Shape kv_b_w_shape{num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank};
    const ov::Shape o_w_shape{hidden, num_heads * v_head_dim};
    const ov::Shape q_a_ln_shape{q_lora_rank};
    const ov::Shape kv_a_ln_shape{kv_lora_rank};

    const auto q_a_w = test_utils::make_seq(q_a_w_shape[0] * q_a_w_shape[1], 0.01f, 0.01f);
    const auto q_b_w = test_utils::make_seq(q_b_w_shape[0] * q_b_w_shape[1], 0.02f, 0.01f);
    const auto kv_a_w = test_utils::make_seq(kv_a_w_shape[0] * kv_a_w_shape[1], 0.03f, 0.01f);
    const auto kv_b_w = test_utils::make_seq(kv_b_w_shape[0] * kv_b_w_shape[1], 0.04f, 0.01f);
    const auto o_w = test_utils::make_seq(o_w_shape[0] * o_w_shape[1], 0.05f, 0.01f);
    const auto q_a_ln_w = test_utils::make_seq(q_a_ln_shape[0], 1.0f, 0.02f);
    const auto kv_a_ln_w = test_utils::make_seq(kv_a_ln_shape[0], 0.9f, 0.03f);

    ov::genai::modeling::tests::DummyWeightSource weights;
    weights.add("self_attn.q_a_proj.weight", test_utils::make_tensor(q_a_w, q_a_w_shape));
    weights.add("self_attn.q_b_proj.weight", test_utils::make_tensor(q_b_w, q_b_w_shape));
    weights.add("self_attn.kv_a_proj_with_mqa.weight", test_utils::make_tensor(kv_a_w, kv_a_w_shape));
    weights.add("self_attn.kv_b_proj.weight", test_utils::make_tensor(kv_b_w, kv_b_w_shape));
    weights.add("self_attn.o_proj.weight", test_utils::make_tensor(o_w, o_w_shape));
    weights.add("self_attn.q_a_layernorm.weight", test_utils::make_tensor(q_a_ln_w, q_a_ln_shape));
    weights.add("self_attn.kv_a_layernorm.weight", test_utils::make_tensor(kv_a_ln_w, kv_a_ln_shape));

    ov::genai::modeling::tests::DummyWeightFinalizer finalizer;

    ov::genai::modeling::models::YoutuConfig cfg;
    cfg.hidden_size = static_cast<int32_t>(hidden);
    cfg.num_attention_heads = static_cast<int32_t>(num_heads);
    cfg.num_key_value_heads = static_cast<int32_t>(num_heads);
    cfg.q_lora_rank = static_cast<int32_t>(q_lora_rank);
    cfg.kv_lora_rank = static_cast<int32_t>(kv_lora_rank);
    cfg.qk_rope_head_dim = static_cast<int32_t>(qk_rope_head_dim);
    cfg.qk_nope_head_dim = static_cast<int32_t>(qk_nope_head_dim);
    cfg.qk_head_dim = static_cast<int32_t>(qk_head_dim);
    cfg.v_head_dim = static_cast<int32_t>(v_head_dim);
    cfg.rope_theta = rope_theta;
    cfg.rope_interleave = true;
    cfg.rms_norm_eps = rms_norm_eps;

    ov::genai::modeling::models::YoutuMLAttention attn(ctx, "self_attn", cfg);
    ov::genai::modeling::weights::load_model(attn, weights, finalizer);

    auto hidden_states = ctx.parameter("hidden_states", ov::element::f32, ov::PartialShape{batch, seq_len, hidden});
    auto beam_idx = ctx.parameter("beam_idx", ov::element::i32, ov::PartialShape{batch});
    auto rope_cos = ctx.parameter("rope_cos", ov::element::f32, ov::PartialShape{batch, seq_len, rope_half_dim});
    auto rope_sin = ctx.parameter("rope_sin", ov::element::f32, ov::PartialShape{batch, seq_len, rope_half_dim});

    auto output = attn.forward(hidden_states, beam_idx, rope_cos, rope_sin);
    auto ov_model = ctx.build_model({output.output()});

    ov::Core core;
    auto compiled = core.compile_model(ov_model, "GPU");
    auto request = compiled.create_infer_request();

    const std::vector<float> hidden_data = {
        0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.6f, 0.7f, 0.8f,
    };
    const std::vector<int64_t> positions = {0, 1};
    auto cos_data = make_rope_cos(positions, batch, seq_len, static_cast<int32_t>(qk_rope_head_dim), rope_theta);
    auto sin_data = make_rope_sin(positions, batch, seq_len, static_cast<int32_t>(qk_rope_head_dim), rope_theta);

    auto expected = youtu_mla_ref(hidden_data,
                                  q_a_w,
                                  q_b_w,
                                  kv_a_w,
                                  kv_b_w,
                                  o_w,
                                  q_a_ln_w,
                                  kv_a_ln_w,
                                  cos_data,
                                  sin_data,
                                  batch,
                                  seq_len,
                                  hidden,
                                  num_heads,
                                  q_lora_rank,
                                  kv_lora_rank,
                                  qk_nope_head_dim,
                                  qk_rope_head_dim,
                                  qk_head_dim,
                                  v_head_dim,
                                  rms_norm_eps);

    ov::Tensor hidden_tensor(ov::element::f32, {batch, seq_len, hidden});
    std::memcpy(hidden_tensor.data(), hidden_data.data(), hidden_data.size() * sizeof(float));
    request.set_input_tensor(0, hidden_tensor);

    ov::Tensor beam_tensor(ov::element::i32, {batch});
    std::fill_n(beam_tensor.data<int32_t>(), batch, 0);
    request.set_input_tensor(1, beam_tensor);

    ov::Tensor cos_tensor(ov::element::f32, {batch, seq_len, rope_half_dim});
    std::memcpy(cos_tensor.data(), cos_data.data(), cos_data.size() * sizeof(float));
    request.set_input_tensor(2, cos_tensor);

    ov::Tensor sin_tensor(ov::element::f32, {batch, seq_len, rope_half_dim});
    std::memcpy(sin_tensor.data(), sin_data.data(), sin_data.size() * sizeof(float));
    request.set_input_tensor(3, sin_tensor);

    request.infer();

    test_utils::expect_tensor_near(request.get_output_tensor(), expected, test_utils::k_tol_default);
}
