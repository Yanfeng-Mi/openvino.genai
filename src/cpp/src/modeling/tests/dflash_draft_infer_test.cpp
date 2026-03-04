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

std::vector<float> rms_norm_ref(const std::vector<float>& x,
                                const std::vector<float>& weight,
                                size_t rows,
                                size_t cols,
                                float eps) {
    std::vector<float> out(rows * cols, 0.0f);
    for (size_t r = 0; r < rows; ++r) {
        float sum_sq = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            float v = x[r * cols + c];
            sum_sq += v * v;
        }
        float mean = sum_sq / static_cast<float>(cols);
        float inv = 1.0f / std::sqrt(mean + eps);
        for (size_t c = 0; c < cols; ++c) {
            out[r * cols + c] = x[r * cols + c] * inv * weight[c];
        }
    }
    return out;
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

    std::vector<float> k = k_ctx;
    k.insert(k.end(), k_noise.begin(), k_noise.end());
    std::vector<float> v = v_ctx;
    v.insert(v.end(), v_noise.begin(), v_noise.end());

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

    // Non-causal attention reference.
    std::vector<float> context(num_heads * q_len * head_dim, 0.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t i = 0; i < q_len; ++i) {
            std::vector<float> scores(kv_len, 0.0f);
            float max_score = -1e30f;
            for (size_t j = 0; j < kv_len; ++j) {
                float acc = 0.0f;
                const size_t q_base = (h * q_len + i) * head_dim;
                const size_t k_base = (h * kv_len + j) * head_dim;
                for (size_t d = 0; d < head_dim; ++d) {
                    acc += qh_rope[q_base + d] * kh_exp[k_base + d];
                }
                acc *= scale;
                scores[j] = acc;
                max_score = std::max(max_score, acc);
            }
            float sum = 0.0f;
            for (float& s : scores) {
                s = std::exp(s - max_score);
                sum += s;
            }
            const size_t ctx_base = (h * q_len + i) * head_dim;
            for (size_t d = 0; d < head_dim; ++d) {
                float acc = 0.0f;
                for (size_t j = 0; j < kv_len; ++j) {
                    const size_t v_base = (h * kv_len + j) * head_dim;
                    acc += (scores[j] / sum) * vh_exp[v_base + d];
                }
                context[ctx_base + d] = acc;
            }
        }
    }
    auto merged = test_utils::merge_heads_ref(context, 1, q_len, num_heads, head_dim);
    return test_utils::linear_ref_3d(merged, o_w, 1, q_len, hidden, hidden);
}

}  // namespace

TEST(DFlashDraft, ForwardMatchesReference) {
    using namespace ov::genai::modeling;
    using namespace ov::genai::modeling::models;

    const size_t ctx_len = 3;
    const size_t q_len = 2;
    const size_t hidden = 4;
    const size_t num_layers = 2;
    const size_t num_heads = 2;
    const size_t num_kv_heads = 1;
    const size_t head_dim = 2;
    const size_t intermediate = 6;
    const float rope_theta = 10000.0f;
    const float rms_eps = 1e-6f;

    DFlashDraftConfig cfg;
    cfg.hidden_size = static_cast<int32_t>(hidden);
    cfg.intermediate_size = static_cast<int32_t>(intermediate);
    cfg.num_hidden_layers = static_cast<int32_t>(num_layers);
    cfg.num_target_layers = static_cast<int32_t>(num_layers);
    cfg.num_attention_heads = static_cast<int32_t>(num_heads);
    cfg.num_key_value_heads = static_cast<int32_t>(num_kv_heads);
    cfg.head_dim = static_cast<int32_t>(head_dim);
    cfg.block_size = 4;
    cfg.rms_norm_eps = rms_eps;
    cfg.rope_theta = rope_theta;
    cfg.attention_bias = false;

    // Prepare deterministic weights.
    test_utils::DummyWeightSource weights;
    test_utils::DummyWeightFinalizer finalizer;

    const size_t ctx_dim = hidden * num_layers;
    const auto fc_w = test_utils::make_seq(hidden * ctx_dim, 0.01f, 0.01f);
    weights.add("fc.weight", test_utils::make_tensor(fc_w, {hidden, ctx_dim}));
    weights.add("hidden_norm.weight", test_utils::make_tensor(test_utils::make_seq(hidden, 1.0f, 0.01f), {hidden}));
    weights.add("norm.weight", test_utils::make_tensor(test_utils::make_seq(hidden, 1.1f, 0.02f), {hidden}));

    for (size_t i = 0; i < num_layers; ++i) {
        const std::string prefix = "layers." + std::to_string(i) + ".";
        weights.add(prefix + "self_attn.q_proj.weight",
                    test_utils::make_tensor(test_utils::make_seq(hidden * hidden, 0.02f, 0.01f),
                                            {hidden, hidden}));
        weights.add(prefix + "self_attn.k_proj.weight",
                    test_utils::make_tensor(test_utils::make_seq((hidden / num_heads) * hidden, 0.03f, 0.01f),
                                            {hidden / num_heads, hidden}));
        weights.add(prefix + "self_attn.v_proj.weight",
                    test_utils::make_tensor(test_utils::make_seq((hidden / num_heads) * hidden, 0.04f, 0.01f),
                                            {hidden / num_heads, hidden}));
        weights.add(prefix + "self_attn.o_proj.weight",
                    test_utils::make_tensor(test_utils::make_seq(hidden * hidden, 0.05f, 0.01f),
                                            {hidden, hidden}));
        weights.add(prefix + "self_attn.q_norm.weight",
                    test_utils::make_tensor(test_utils::make_seq(head_dim, 1.0f, 0.01f), {head_dim}));
        weights.add(prefix + "self_attn.k_norm.weight",
                    test_utils::make_tensor(test_utils::make_seq(head_dim, 0.9f, 0.01f), {head_dim}));

        weights.add(prefix + "input_layernorm.weight",
                    test_utils::make_tensor(test_utils::make_seq(hidden, 1.2f, 0.01f), {hidden}));
        weights.add(prefix + "post_attention_layernorm.weight",
                    test_utils::make_tensor(test_utils::make_seq(hidden, 1.3f, 0.01f), {hidden}));

        weights.add(prefix + "mlp.gate_proj.weight",
                    test_utils::make_tensor(test_utils::make_seq(intermediate * hidden, 0.06f, 0.01f),
                                            {intermediate, hidden}));
        weights.add(prefix + "mlp.up_proj.weight",
                    test_utils::make_tensor(test_utils::make_seq(intermediate * hidden, 0.07f, 0.01f),
                                            {intermediate, hidden}));
        weights.add(prefix + "mlp.down_proj.weight",
                    test_utils::make_tensor(test_utils::make_seq(hidden * intermediate, 0.08f, 0.01f),
                                            {hidden, intermediate}));
    }

    // Build model.
    auto model = create_dflash_draft_model(cfg, weights, finalizer, ov::element::f32);

    // Inputs.
    auto target_vals = test_utils::make_seq(ctx_len * ctx_dim, 0.01f, 0.01f);
    auto noise_vals = test_utils::make_seq(q_len * hidden, 0.02f, 0.01f);
    std::vector<int64_t> pos_vals(ctx_len + q_len);
    for (size_t i = 0; i < pos_vals.size(); ++i) {
        pos_vals[i] = static_cast<int64_t>(i);
    }

    ov::Tensor target_tensor(ov::element::f32, {1, ctx_len, ctx_dim});
    std::memcpy(target_tensor.data(), target_vals.data(), target_vals.size() * sizeof(float));
    ov::Tensor noise_tensor(ov::element::f32, {1, q_len, hidden});
    std::memcpy(noise_tensor.data(), noise_vals.data(), noise_vals.size() * sizeof(float));
    ov::Tensor pos_tensor(ov::element::i64, {1, pos_vals.size()});
    std::memcpy(pos_tensor.data(), pos_vals.data(), pos_vals.size() * sizeof(int64_t));

    ov::Core core;
    std::string device = "GPU";
    {
        auto available = core.get_available_devices();
        if (std::find(available.begin(), available.end(), "GPU") == available.end()) {
            GTEST_SKIP() << "GPU device is required for this test (CPU plugin lacks PlaceholderExtension support)";
        }
    }
    ov::AnyMap compile_cfg = {{ov::hint::inference_precision.name(), ov::element::f32}};
    auto compiled = core.compile_model(model, device, compile_cfg);
    auto request = compiled.create_infer_request();
    request.set_tensor("target_hidden", target_tensor);
    request.set_tensor("noise_embedding", noise_tensor);
    request.set_tensor("position_ids", pos_tensor);
    request.infer();
    auto output = request.get_tensor("draft_hidden");

    // Reference computation.
    auto conditioned = test_utils::linear_ref_3d(target_vals, fc_w, 1, ctx_len, ctx_dim, hidden);
    auto context_hidden = rms_norm_ref(conditioned,
                                       test_utils::make_seq(hidden, 1.0f, 0.01f),
                                       1 * ctx_len,
                                       hidden,
                                       rms_eps);

    std::vector<float> hidden_states = noise_vals;
    for (size_t layer = 0; layer < num_layers; ++layer) {
        const std::string prefix = "layers." + std::to_string(layer) + ".";
        auto in_ln = test_utils::make_seq(hidden, 1.2f, 0.01f);
        auto post_ln = test_utils::make_seq(hidden, 1.3f, 0.01f);
        auto gate_w = test_utils::make_seq(intermediate * hidden, 0.06f, 0.01f);
        auto up_w = test_utils::make_seq(intermediate * hidden, 0.07f, 0.01f);
        auto down_w = test_utils::make_seq(hidden * intermediate, 0.08f, 0.01f);

        auto normed = rms_norm_ref(hidden_states, in_ln, 1 * q_len, hidden, rms_eps);
        auto attn_out = dflash_attention_ref(context_hidden,
                                             normed,
                                             test_utils::make_seq(hidden * hidden, 0.02f, 0.01f),
                                             test_utils::make_seq((hidden / num_heads) * hidden, 0.03f, 0.01f),
                                             test_utils::make_seq((hidden / num_heads) * hidden, 0.04f, 0.01f),
                                             test_utils::make_seq(hidden * hidden, 0.05f, 0.01f),
                                             test_utils::make_seq(head_dim, 1.0f, 0.01f),
                                             test_utils::make_seq(head_dim, 0.9f, 0.01f),
                                             pos_vals,
                                             ctx_len,
                                             q_len,
                                             hidden,
                                             num_heads,
                                             num_kv_heads,
                                             head_dim,
                                             rope_theta,
                                             rms_eps);
        auto attn_res = test_utils::add_ref(hidden_states, attn_out);
        auto post_norm = rms_norm_ref(attn_res, post_ln, 1 * q_len, hidden, rms_eps);
        auto mlp_out = test_utils::mlp_ref(post_norm,
                                           gate_w,
                                           up_w,
                                           down_w,
                                           1,
                                           q_len,
                                           hidden,
                                           intermediate);
        hidden_states = test_utils::add_ref(attn_res, mlp_out);
    }

    auto expected = rms_norm_ref(hidden_states,
                                 test_utils::make_seq(hidden, 1.1f, 0.02f),
                                 1 * q_len,
                                 hidden,
                                 rms_eps);

    test_utils::expect_tensor_near(output, expected, 1e-3f);

    // Extra guard: ensure no NaNs.
    const float* out_data = output.data<const float>();
    const size_t total = output.get_size();
    for (size_t i = 0; i < total; ++i) {
        ASSERT_TRUE(std::isfinite(out_data[i])) << "Found non-finite at " << i;
    }
}
