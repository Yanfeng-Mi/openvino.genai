// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/openvino.hpp>

#include "modeling/models/dflash_draft/dflash_draft.hpp"
#include "modeling/ops/llm.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace test_utils = ov::genai::modeling::tests;

namespace {

std::vector<float> concat_seq(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> out = a;
    out.insert(out.end(), b.begin(), b.end());
    return out;
}

std::vector<float> non_causal_attention_ref(const std::vector<float>& qh,
                                            const std::vector<float>& kh,
                                            const std::vector<float>& vh,
                                            size_t num_heads,
                                            size_t q_len,
                                            size_t kv_len,
                                            size_t head_dim) {
    std::vector<float> context(num_heads * q_len * head_dim, 0.0f);
    std::vector<float> scores(kv_len, 0.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t i = 0; i < q_len; ++i) {
            float max_score = -1e30f;
            for (size_t j = 0; j < kv_len; ++j) {
                float acc = 0.0f;
                const size_t q_base = (h * q_len + i) * head_dim;
                const size_t k_base = (h * kv_len + j) * head_dim;
                for (size_t d = 0; d < head_dim; ++d) {
                    acc += qh[q_base + d] * kh[k_base + d];
                }
                acc *= scale;
                scores[j] = acc;
                max_score = std::max(max_score, acc);
            }

            float sum = 0.0f;
            for (size_t j = 0; j < kv_len; ++j) {
                scores[j] = std::exp(scores[j] - max_score);
                sum += scores[j];
            }

            const size_t ctx_base = (h * q_len + i) * head_dim;
            for (size_t d = 0; d < head_dim; ++d) {
                float acc = 0.0f;
                for (size_t j = 0; j < kv_len; ++j) {
                    const size_t v_base = (h * kv_len + j) * head_dim;
                    acc += (scores[j] / sum) * vh[v_base + d];
                }
                context[ctx_base + d] = acc;
            }
        }
    }
    return context;
}

std::vector<float> dflash_attention_ref(const std::vector<float>& target_hidden,
                                        const std::vector<float>& noise_hidden,
                                        const std::vector<float>& q_w,
                                        const std::vector<float>& k_w,
                                        const std::vector<float>& v_w,
                                        const std::vector<float>& o_w,
                                        const std::vector<float>& q_norm_w,
                                        const std::vector<float>& k_norm_w,
                                        const std::vector<int64_t>& positions,
                                        size_t ctx_len,
                                        size_t q_len,
                                        size_t hidden,
                                        size_t num_heads,
                                        size_t num_kv_heads,
                                        size_t head_dim,
                                        float rope_theta,
                                        float rms_norm_eps) {
    const size_t kv_hidden = num_kv_heads * head_dim;
    const size_t kv_len = ctx_len + q_len;

    auto q = test_utils::linear_ref_3d(noise_hidden, q_w, 1, q_len, hidden, hidden);
    auto k_ctx = test_utils::linear_ref_3d(target_hidden, k_w, 1, ctx_len, hidden, kv_hidden);
    auto k_noise = test_utils::linear_ref_3d(noise_hidden, k_w, 1, q_len, hidden, kv_hidden);
    auto v_ctx = test_utils::linear_ref_3d(target_hidden, v_w, 1, ctx_len, hidden, kv_hidden);
    auto v_noise = test_utils::linear_ref_3d(noise_hidden, v_w, 1, q_len, hidden, kv_hidden);

    auto k = concat_seq(k_ctx, k_noise);
    auto v = concat_seq(v_ctx, v_noise);

    auto qh = test_utils::to_heads_ref(q, 1, q_len, num_heads, head_dim);
    auto kh = test_utils::to_heads_ref(k, 1, kv_len, num_kv_heads, head_dim);
    auto vh = test_utils::to_heads_ref(v, 1, kv_len, num_kv_heads, head_dim);

    auto qh_norm = test_utils::rmsnorm_heads_ref(qh, q_norm_w, 1, num_heads, q_len, head_dim, rms_norm_eps);
    auto kh_norm = test_utils::rmsnorm_heads_ref(kh, k_norm_w, 1, num_kv_heads, kv_len, head_dim, rms_norm_eps);

    std::vector<int64_t> pos_tail(positions.end() - static_cast<std::ptrdiff_t>(q_len), positions.end());
    auto qh_rope = test_utils::apply_rope_ref(qh_norm, pos_tail, 1, q_len, num_heads, head_dim, rope_theta);
    auto kh_rope = test_utils::apply_rope_ref(kh_norm, positions, 1, kv_len, num_kv_heads, head_dim, rope_theta);

    auto kh_exp = test_utils::repeat_kv_ref(kh_rope, 1, num_heads, num_kv_heads, kv_len, head_dim);
    auto vh_exp = test_utils::repeat_kv_ref(vh, 1, num_heads, num_kv_heads, kv_len, head_dim);

    auto context = non_causal_attention_ref(qh_rope, kh_exp, vh_exp, num_heads, q_len, kv_len, head_dim);
    auto merged = test_utils::merge_heads_ref(context, 1, q_len, num_heads, head_dim);
    return test_utils::linear_ref_3d(merged, o_w, 1, q_len, hidden, hidden);
}

}  // namespace

TEST(DFlashAttention, MatchesReference) {
    ov::genai::modeling::BuilderContext ctx;

    const size_t ctx_len = 3;
    const size_t q_len = 2;
    const size_t hidden = 4;
    const size_t num_heads = 2;
    const size_t num_kv_heads = 1;
    const size_t head_dim = 2;
    const size_t kv_hidden = num_kv_heads * head_dim;

    ov::genai::modeling::models::DFlashDraftConfig cfg;
    cfg.hidden_size = static_cast<int32_t>(hidden);
    cfg.num_attention_heads = static_cast<int32_t>(num_heads);
    cfg.num_key_value_heads = static_cast<int32_t>(num_kv_heads);
    cfg.head_dim = static_cast<int32_t>(head_dim);
    cfg.rms_norm_eps = 1e-6f;
    cfg.rope_theta = 10000.0f;
    cfg.attention_bias = false;

    ov::genai::modeling::models::DFlashAttention attn(ctx, "self_attn", cfg);

    test_utils::DummyWeightSource weights;
    test_utils::DummyWeightFinalizer finalizer;

    const auto q_w = test_utils::make_seq(hidden * hidden, 0.01f, 0.01f);
    const auto k_w = test_utils::make_seq(kv_hidden * hidden, 0.02f, 0.01f);
    const auto v_w = test_utils::make_seq(kv_hidden * hidden, 0.03f, 0.01f);
    const auto o_w = test_utils::make_seq(hidden * hidden, 0.04f, 0.01f);
    const auto q_norm_w = test_utils::make_seq(head_dim, 1.0f, 0.01f);
    const auto k_norm_w = test_utils::make_seq(head_dim, 0.9f, 0.01f);

    weights.add("self_attn.q_proj.weight", test_utils::make_tensor(q_w, {hidden, hidden}));
    weights.add("self_attn.k_proj.weight", test_utils::make_tensor(k_w, {kv_hidden, hidden}));
    weights.add("self_attn.v_proj.weight", test_utils::make_tensor(v_w, {kv_hidden, hidden}));
    weights.add("self_attn.o_proj.weight", test_utils::make_tensor(o_w, {hidden, hidden}));
    weights.add("self_attn.q_norm.weight", test_utils::make_tensor(q_norm_w, {head_dim}));
    weights.add("self_attn.k_norm.weight", test_utils::make_tensor(k_norm_w, {head_dim}));

    ov::genai::modeling::weights::load_model(attn, weights, finalizer);

    auto target_hidden = ctx.parameter("target_hidden", ov::element::f32, ov::PartialShape{1, static_cast<int64_t>(ctx_len), static_cast<int64_t>(hidden)});
    auto noise_hidden = ctx.parameter("noise_hidden", ov::element::f32, ov::PartialShape{1, static_cast<int64_t>(q_len), static_cast<int64_t>(hidden)});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{1, static_cast<int64_t>(ctx_len + q_len)});

    auto* policy = &ctx.op_policy();
    auto cos_sin = ov::genai::modeling::ops::llm::rope_cos_sin(position_ids, static_cast<int32_t>(head_dim), cfg.rope_theta, policy);
    auto out = attn.forward(target_hidden, noise_hidden, cos_sin.first, cos_sin.second);

    auto result = std::make_shared<ov::op::v0::Result>(out.output());
    auto ov_model = ctx.build_model({result->output(0)});

    ov::Core core;
    auto compiled = core.compile_model(ov_model, "GPU");
    auto request = compiled.create_infer_request();

    const auto target_data = test_utils::make_seq(ctx_len * hidden, 0.01f, 0.01f);
    const auto noise_data = test_utils::make_seq(q_len * hidden, 0.02f, 0.01f);
    ov::Tensor target_tensor(ov::element::f32, {1, ctx_len, hidden});
    std::memcpy(target_tensor.data(), target_data.data(), target_data.size() * sizeof(float));
    request.set_tensor("target_hidden", target_tensor);

    ov::Tensor noise_tensor(ov::element::f32, {1, q_len, hidden});
    std::memcpy(noise_tensor.data(), noise_data.data(), noise_data.size() * sizeof(float));
    request.set_tensor("noise_hidden", noise_tensor);

    std::vector<int64_t> pos_vals(ctx_len + q_len);
    for (size_t i = 0; i < pos_vals.size(); ++i) {
        pos_vals[i] = static_cast<int64_t>(i);
    }
    ov::Tensor pos_tensor(ov::element::i64, {1, ctx_len + q_len});
    std::memcpy(pos_tensor.data(), pos_vals.data(), pos_vals.size() * sizeof(int64_t));
    request.set_tensor("position_ids", pos_tensor);

    request.infer();

    std::vector<float> expected = dflash_attention_ref(
        target_data,
        noise_data,
        q_w,
        k_w,
        v_w,
        o_w,
        q_norm_w,
        k_norm_w,
        pos_vals,
        ctx_len,
        q_len,
        hidden,
        num_heads,
        num_kv_heads,
        head_dim,
        cfg.rope_theta,
        cfg.rms_norm_eps);

    auto output = request.get_output_tensor();
    EXPECT_EQ(output.get_shape(), (ov::Shape{1, q_len, hidden}));
    test_utils::expect_tensor_near(output, expected, 1e-3f);
}
