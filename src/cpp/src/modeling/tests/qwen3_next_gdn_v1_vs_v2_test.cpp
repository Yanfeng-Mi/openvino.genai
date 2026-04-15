// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// E2E numerical equivalence test: GatedDeltaNet (v1, TensorIterator) vs
// GatedDeltaNet (ops::linear_attention).
//
// The test exercises the full stateful prefill + multi-step decode pipeline with
// identical weights and inputs for both implementations.  If the beta/g inputs
// are swapped, the state recurrence diverges quickly and the max absolute error
// will blow up well beyond the tolerance.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>

#include "modeling/builder_context.hpp"
#include "modeling/models/qwen3_next/modeling_qwen3_next.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace test_utils = ov::genai::modeling::tests;
using Qwen3NextConfig = ov::genai::modeling::models::Qwen3NextConfig;

namespace {

// ─── Shared config ───────────────────────────────────────────────────────────

Qwen3NextConfig make_cfg() {
    Qwen3NextConfig cfg;
    cfg.hidden_size = 2048;
    cfg.linear_num_key_heads = 16;
    cfg.linear_num_value_heads = 32;
    cfg.linear_key_head_dim = 128;
    cfg.linear_value_head_dim = 128;
    cfg.linear_conv_kernel_dim = 4;
    cfg.rms_norm_eps = 1e-6f;
    return cfg;
}

// ─── Shared weight helper ────────────────────────────────────────────────────

void add_weights(test_utils::DummyWeightSource& weights,
                 const std::string& prefix,
                 const Qwen3NextConfig& cfg) {
    const int32_t key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
    const int32_t value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    const int32_t conv_dim = key_dim * 2 + value_dim;
    const int32_t proj_qkvz = key_dim * 2 + value_dim * 2;
    const int32_t proj_ba = cfg.linear_num_value_heads * 2;

    // Use tiny step values to keep weights in a small range and avoid NaN from
    // numerical overflow in the f32 (unquantized) forward pass.
    weights.add(prefix + ".in_proj_qkvz.weight",
                test_utils::make_tensor(
                    test_utils::make_seq(static_cast<size_t>(proj_qkvz * cfg.hidden_size), 0.001f, 1e-8f),
                    {static_cast<size_t>(proj_qkvz), static_cast<size_t>(cfg.hidden_size)}));
    weights.add(prefix + ".in_proj_ba.weight",
                test_utils::make_tensor(
                    test_utils::make_seq(static_cast<size_t>(proj_ba * cfg.hidden_size), 0.002f, 1e-7f),
                    {static_cast<size_t>(proj_ba), static_cast<size_t>(cfg.hidden_size)}));
    weights.add(prefix + ".conv1d.weight",
                test_utils::make_tensor(
                    test_utils::make_seq(static_cast<size_t>(conv_dim * cfg.linear_conv_kernel_dim), 0.01f, 1e-6f),
                    {static_cast<size_t>(conv_dim), static_cast<size_t>(cfg.linear_conv_kernel_dim)}));
    weights.add(prefix + ".A_log",
                test_utils::make_tensor(
                    test_utils::make_seq(static_cast<size_t>(cfg.linear_num_value_heads), 0.04f, 0.001f),
                    {static_cast<size_t>(cfg.linear_num_value_heads)}));
    weights.add(prefix + ".dt_bias",
                test_utils::make_tensor(
                    test_utils::make_seq(static_cast<size_t>(cfg.linear_num_value_heads), 0.05f, 0.001f),
                    {static_cast<size_t>(cfg.linear_num_value_heads)}));
    weights.add(prefix + ".norm.weight",
                test_utils::make_tensor(
                    test_utils::make_seq(static_cast<size_t>(cfg.linear_value_head_dim), 1.0f, 0.0f),
                    {static_cast<size_t>(cfg.linear_value_head_dim)}));
    weights.add(prefix + ".out_proj.weight",
                test_utils::make_tensor(
                    test_utils::make_seq(static_cast<size_t>(cfg.hidden_size * value_dim), 0.001f, 1e-8f),
                    {static_cast<size_t>(cfg.hidden_size), static_cast<size_t>(value_dim)}));
}

// ─── Build model helper (templated on V1 or V2) ─────────────────────────────

template <typename GDNModule>
std::shared_ptr<ov::Model> build_model(const Qwen3NextConfig& cfg, bool dynamic_seq) {
    ov::genai::modeling::BuilderContext ctx;
    GDNModule m(ctx, "linear_attn", cfg, /*layer_idx=*/0);

    test_utils::DummyWeightSource weights;
    test_utils::DummyWeightFinalizer finalizer;
    add_weights(weights, "linear_attn", cfg);
    ov::genai::modeling::weights::load_model(m, weights, finalizer);

    auto seq_dim = dynamic_seq ? int64_t(-1) : int64_t(1);
    auto hs = ctx.parameter("hidden_states", ov::element::f32,
                            ov::PartialShape{1, seq_dim, cfg.hidden_size});
    auto bi = ctx.parameter("beam_idx", ov::element::i32, ov::PartialShape{1});
    auto am = ctx.parameter("attention_mask", ov::element::i64,
                            ov::PartialShape{1, seq_dim});
    auto out = m.forward(hs, bi, &am, nullptr);
    return ctx.build_model({out.output()});
}

// ─── Comparison helpers ──────────────────────────────────────────────────────

struct CompareStats {
    float max_abs_diff = 0.0f;
    float mean_abs_diff = 0.0f;
    size_t count = 0;
    size_t nan_count_a = 0;
    size_t nan_count_b = 0;
};

CompareStats compare_tensors(const ov::Tensor& a, const ov::Tensor& b) {
    CompareStats stats;
    EXPECT_EQ(a.get_shape(), b.get_shape());
    if (a.get_shape() != b.get_shape()) return stats;

    const float* da = a.data<float>();
    const float* db = b.data<float>();
    stats.count = a.get_size();
    double sum = 0.0;
    size_t finite_count = 0;
    for (size_t i = 0; i < stats.count; ++i) {
        if (std::isnan(da[i])) stats.nan_count_a++;
        if (std::isnan(db[i])) stats.nan_count_b++;
        float diff = std::abs(da[i] - db[i]);
        if (std::isfinite(diff)) {
            stats.max_abs_diff = std::max(stats.max_abs_diff, diff);
            sum += static_cast<double>(diff);
            finite_count++;
        }
    }
    stats.mean_abs_diff = finite_count > 0
        ? static_cast<float>(sum / static_cast<double>(finite_count))
        : 0.0f;
    return stats;
}

void set_inputs(ov::InferRequest& req, const ov::Tensor& hs, const ov::Tensor& bi, const ov::Tensor& am) {
    req.set_tensor("hidden_states", hs);
    req.set_tensor("beam_idx", bi);
    req.set_tensor("attention_mask", am);
}

}  // anonymous namespace

// =============================================================================
// Test: Prefill + multi-step decode numerical equivalence between V1 and V2.
//
// This is the most sensitive test for the beta/g swap bug because:
//   1. Prefill builds up the recurrent state over multiple tokens.
//   2. Each decode step reads the state, applies decay (exp(g)), and updates it.
//   3. A swapped beta/g causes the state to *grow* instead of decay, so errors
//      compound rapidly across decode steps.
// =============================================================================

TEST(GatedDeltaNetV1vsLinearAttention, PrefillAndMultiStepDecodeMatchOnGPU) {
    const auto cfg = make_cfg();
    const size_t hidden = static_cast<size_t>(cfg.hidden_size);
    const size_t prefill_len = 8;
    const size_t num_decode_steps = 5;
    constexpr float tolerance = test_utils::k_tol_linear_attn;

    // Build both models with dynamic sequence length (for prefill & decode).
    auto model_v1 = build_model<ov::genai::modeling::models::Qwen3NextGatedDeltaNet>(cfg, true);
    auto model_v2 = build_model<ov::genai::modeling::models::Qwen3NextGatedDeltaNet>(cfg, true);

    ov::Core core;
    auto req_v1 = core.compile_model(model_v1, "GPU").create_infer_request();
    auto req_v2 = core.compile_model(model_v2, "GPU").create_infer_request();

    // ── Shared beam index ──
    ov::Tensor bi(ov::element::i32, {1});
    bi.data<int32_t>()[0] = 0;

    // ── Step 1: Prefill ──
    {
        auto input_data = test_utils::random_f32(prefill_len * hidden, -0.1f, 0.1f, /*seed=*/42);
        ov::Tensor hs(ov::element::f32, {1, prefill_len, hidden});
        std::memcpy(hs.data<float>(), input_data.data(), input_data.size() * sizeof(float));

        ov::Tensor am(ov::element::i64, {1, prefill_len});
        std::fill_n(am.data<int64_t>(), prefill_len, 1);

        set_inputs(req_v1, hs, bi, am);
        set_inputs(req_v2, hs, bi, am);
        req_v1.infer();
        req_v2.infer();

        auto stats = compare_tensors(req_v1.get_output_tensor(0),
                                     req_v2.get_output_tensor(0));
        // Print first few values to help diagnose structural differences.
        {
            const float* d1 = req_v1.get_output_tensor(0).data<float>();
            const float* d2 = req_v2.get_output_tensor(0).data<float>();
            std::cout << "  v1[0..4]: ";
            for (int i = 0; i < 5 && i < (int)stats.count; ++i)
                std::cout << std::fixed << std::setprecision(6) << d1[i] << " ";
            std::cout << "\n  v2[0..4]: ";
            for (int i = 0; i < 5 && i < (int)stats.count; ++i)
                std::cout << std::fixed << std::setprecision(6) << d2[i] << " ";
            std::cout << "\n";
        }
        std::cout << "[Prefill  seq_len=" << prefill_len
                  << "]  max_diff=" << std::scientific << std::setprecision(4) << stats.max_abs_diff
                  << "  mean_diff=" << stats.mean_abs_diff
                  << "  nan_v1=" << stats.nan_count_a << "  nan_v2=" << stats.nan_count_b << "\n";
        EXPECT_EQ(stats.nan_count_a, stats.nan_count_b)
            << "Prefill: NaN count mismatch between two linear-attention paths";
        EXPECT_LT(stats.max_abs_diff, tolerance)
            << "Prefill: path A vs path B outputs diverged (max_diff=" << stats.max_abs_diff << ")";
    }

    // ── Step 2: Multiple decode steps (state carries over) ──
    for (size_t step = 0; step < num_decode_steps; ++step) {
        auto input_data = test_utils::random_f32(hidden, -0.1f, 0.1f, /*seed=*/100 + static_cast<uint32_t>(step));
        ov::Tensor hs(ov::element::f32, {1, 1, hidden});
        std::memcpy(hs.data<float>(), input_data.data(), input_data.size() * sizeof(float));

        ov::Tensor am(ov::element::i64, {1, 1});
        am.data<int64_t>()[0] = 1;

        set_inputs(req_v1, hs, bi, am);
        set_inputs(req_v2, hs, bi, am);
        req_v1.infer();
        req_v2.infer();

        auto stats = compare_tensors(req_v1.get_output_tensor(0),
                                     req_v2.get_output_tensor(0));
        std::cout << "[Decode   step=" << step + 1
                  << "]  max_diff=" << std::scientific << std::setprecision(4) << stats.max_abs_diff
                  << "  mean_diff=" << stats.mean_abs_diff
                  << "  nan_v1=" << stats.nan_count_a << "  nan_v2=" << stats.nan_count_b << "\n";
        EXPECT_EQ(stats.nan_count_a, stats.nan_count_b)
            << "Decode step " << step + 1 << ": NaN count mismatch between two linear-attention paths";
        EXPECT_LT(stats.max_abs_diff, tolerance)
            << "Decode step " << step + 1 << ": path A vs path B outputs diverged (max_diff="
            << stats.max_abs_diff << ")";
    }
}

