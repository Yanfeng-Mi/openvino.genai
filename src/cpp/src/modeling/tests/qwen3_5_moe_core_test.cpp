// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/core/except.hpp>
#include <openvino/openvino.hpp>

#include "modeling/builder_context.hpp"
#include "modeling/models/qwen3_5/modeling_qwen3_5_moe.hpp"
#include "modeling/models/qwen3_5/modeling_qwen3_5_text.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/weight_loader.hpp"
#include "modeling/weights/weight_source.hpp"

namespace {

using namespace ov::genai::modeling;
using namespace ov::genai::modeling::models;
using namespace ov::genai::modeling::tests;
using namespace ov::genai::modeling::weights;

struct MoeShape {
    int32_t hidden_size = 256;
    int32_t moe_intermediate_size = 128;
    int32_t shared_expert_intermediate_size = 128;
    int32_t num_experts = 16;
    int32_t num_experts_per_tok = 4;
    int64_t batch = 1;
    int64_t seq_len = 8;
};

enum class RefMode {
    FP32,
    INT4
};

enum class Int4RouteMode {
    ForceOpsetFallback,
    EnableFused
};

struct CpuRefWeights {
    std::vector<float> gate_inp;
    std::vector<float> gate_proj;
    std::vector<float> up_proj;
    std::vector<float> down_proj;
    std::vector<float> shared_expert_gate;
    std::vector<float> shared_gate_proj;
    std::vector<float> shared_up_proj;
    std::vector<float> shared_down_proj;
};

int32_t parse_env_int(const char* name, int32_t default_value, int32_t min_value) {
    const char* raw = std::getenv(name);
    if (!raw || raw[0] == '\0') {
        return default_value;
    }
    try {
        const long long parsed = std::stoll(raw);
        if (parsed < static_cast<long long>(min_value) || parsed > std::numeric_limits<int32_t>::max()) {
            return default_value;
        }
        return static_cast<int32_t>(parsed);
    } catch (...) {
        return default_value;
    }
}

MoeShape make_ref_shape() {
    MoeShape s;
    s.hidden_size = parse_env_int("OV_GENAI_QWEN35_MOE_REF_HIDDEN", 256, 1);
    s.moe_intermediate_size = parse_env_int("OV_GENAI_QWEN35_MOE_REF_INTER", 128, 1);
    s.shared_expert_intermediate_size = parse_env_int("OV_GENAI_QWEN35_MOE_REF_SHARED_INTER", 128, 1);
    // GPU fused MoE kernel is more stable with production-like expert counts.
    s.num_experts = parse_env_int("OV_GENAI_QWEN35_MOE_REF_EXPERTS", 128, 1);
    s.num_experts_per_tok = std::min<int32_t>(parse_env_int("OV_GENAI_QWEN35_MOE_REF_TOPK", 8, 1), s.num_experts);
    s.batch = parse_env_int("OV_GENAI_QWEN35_MOE_REF_BATCH", 1, 1);
    s.seq_len = parse_env_int("OV_GENAI_QWEN35_MOE_REF_SEQ", 8, 1);
    return s;
}

// Real Qwen3.5-35B-A3B text MoE parameters from config:
// hidden=2048, moe_inter=512, shared_inter=512, experts=256, topk=8.
MoeShape make_qwen35_full_shape() {
    MoeShape s;
    s.hidden_size = 2048;
    s.moe_intermediate_size = 512;
    s.shared_expert_intermediate_size = 512;
    s.num_experts = 256;
    s.num_experts_per_tok = 8;
    s.batch = parse_env_int("OV_GENAI_QWEN35_MOE_FULL_BATCH", 1, 1);
    // Keep fused MoE perf comparison on decode-like shape by default.
    s.seq_len = parse_env_int("OV_GENAI_QWEN35_MOE_FULL_SEQ", 1, 1);
    return s;
}

MoeShape make_fused_ref_shape() {
    MoeShape s = make_ref_shape();
    // The current fused GPU path is stable on single-token decode shape.
    s.batch = parse_env_int("OV_GENAI_QWEN35_MOE_FUSED_BATCH", 1, 1);
    s.seq_len = parse_env_int("OV_GENAI_QWEN35_MOE_FUSED_SEQ", 1, 1);
    return s;
}

ov::Tensor make_input_tensor(const MoeShape& s, uint32_t seed = 0x5A5A0137u) {
    ov::Tensor t(ov::element::f32,
                 ov::Shape{static_cast<size_t>(s.batch), static_cast<size_t>(s.seq_len), static_cast<size_t>(s.hidden_size)});
    auto* out = t.data<float>();
    uint32_t st = seed;
    for (size_t i = 0; i < t.get_size(); ++i) {
        st = st * 1664525u + 1013904223u;
        const float u = static_cast<float>((st >> 8) & 0x00FFFFFFu) / static_cast<float>(0x00FFFFFFu);
        out[i] = (u * 2.0f - 1.0f) * 0.5f;
    }
    return t;
}

std::vector<float> tensor_to_f32_vector(const ov::Tensor& t) {
    std::vector<float> out(t.get_size(), 0.0f);
    if (t.get_element_type() == ov::element::f32) {
        const auto* src = t.data<const float>();
        std::copy(src, src + out.size(), out.begin());
    } else if (t.get_element_type() == ov::element::f16) {
        const auto* src = t.data<const ov::float16>();
        for (size_t i = 0; i < out.size(); ++i) {
            out[i] = static_cast<float>(src[i]);
        }
    } else {
        OPENVINO_THROW("Unsupported tensor type in tensor_to_f32_vector: ", t.get_element_type());
    }
    return out;
}

ov::Tensor make_f32_tensor_from_f32(const std::vector<float>& v, const ov::Shape& shape) {
    ov::Tensor t(ov::element::f32, shape);
    auto* dst = t.data<float>();
    std::copy(v.begin(), v.end(), dst);
    return t;
}

ov::Tensor make_seeded_f16_tensor(const ov::Shape& shape, uint32_t seed, float scale = 0.02f) {
    ov::Tensor t(ov::element::f16, shape);
    auto* out = t.data<ov::float16>();
    uint32_t st = seed;
    const size_t n = t.get_size();
    for (size_t i = 0; i < n; ++i) {
        st = st * 1103515245u + 12345u;
        const float u = static_cast<float>((st >> 9) & 0x007FFFFFu) / static_cast<float>(0x007FFFFFu);
        out[i] = ov::float16((u * 2.0f - 1.0f) * scale);
    }
    return t;
}

std::vector<float> make_seeded_f32(size_t n, uint32_t seed, float scale = 0.02f) {
    std::vector<float> out(n);
    uint32_t st = seed;
    for (size_t i = 0; i < n; ++i) {
        st = st * 1103515245u + 12345u;
        const float u = static_cast<float>((st >> 9) & 0x007FFFFFu) / static_cast<float>(0x007FFFFFu);
        out[i] = (u * 2.0f - 1.0f) * scale;
    }
    return out;
}

struct WeightSpec {
    std::string name;
    ov::Shape shape;
    uint32_t seed = 0;
};

std::vector<WeightSpec> make_weight_specs(const MoeShape& s) {
    const size_t e = static_cast<size_t>(s.num_experts);
    const size_t h = static_cast<size_t>(s.hidden_size);
    const size_t i = static_cast<size_t>(s.moe_intermediate_size);
    const size_t si = static_cast<size_t>(s.shared_expert_intermediate_size);
    return {
        {"mlp.gate.weight", {e, h}, 101u},
        {"mlp.experts.gate_up_proj", {e, 2 * i, h}, 103u},
        {"mlp.experts.down_proj", {e, h, i}, 107u},
        {"mlp.shared_expert_gate.weight", {1, h}, 109u},
        {"mlp.shared_expert.gate_proj.weight", {si, h}, 113u},
        {"mlp.shared_expert.up_proj.weight", {si, h}, 127u},
        {"mlp.shared_expert.down_proj.weight", {h, si}, 131u},
    };
}

class DeterministicMoeWeightSource : public WeightSource {
public:
    explicit DeterministicMoeWeightSource(const MoeShape& s) {
        for (const auto& spec : make_weight_specs(s)) {
            specs_.emplace(spec.name, spec);
            keys_.push_back(spec.name);
        }
    }

    std::vector<std::string> keys() const override {
        return keys_;
    }

    bool has(const std::string& name) const override {
        return specs_.find(name) != specs_.end();
    }

    const ov::Tensor& get_tensor(const std::string& name) const override {
        auto it = specs_.find(name);
        OPENVINO_ASSERT(it != specs_.end(), "Unknown deterministic MoE weight: ", name);
        if (cached_name_ != name) {
            cached_name_ = name;
            cached_tensor_ = make_seeded_f16_tensor(it->second.shape, it->second.seed);
        }
        return cached_tensor_;
    }

private:
    std::unordered_map<std::string, WeightSpec> specs_;
    std::vector<std::string> keys_;
    mutable std::string cached_name_;
    mutable ov::Tensor cached_tensor_;
};

std::vector<float> get_raw_weight_vector(const MoeShape& s, const std::string& name) {
    for (const auto& spec : make_weight_specs(s)) {
        if (spec.name == name) {
            return make_seeded_f32(ov::shape_size(spec.shape), spec.seed);
        }
    }
    OPENVINO_THROW("Unknown weight name for get_raw_weight_vector: ", name);
}

void split_gate_up(const MoeShape& s,
                   const std::vector<float>& fused_gate_up,
                   std::vector<float>& gate_proj,
                   std::vector<float>& up_proj) {
    const size_t e = static_cast<size_t>(s.num_experts);
    const size_t i = static_cast<size_t>(s.moe_intermediate_size);
    const size_t h = static_cast<size_t>(s.hidden_size);

    gate_proj.resize(e * i * h);
    up_proj.resize(e * i * h);

    for (size_t ex = 0; ex < e; ++ex) {
        for (size_t row = 0; row < i; ++row) {
            const size_t src_gate = (ex * (2 * i) + row) * h;
            const size_t src_up = (ex * (2 * i) + (i + row)) * h;
            const size_t dst = (ex * i + row) * h;
            std::copy_n(fused_gate_up.data() + src_gate, h, gate_proj.data() + dst);
            std::copy_n(fused_gate_up.data() + src_up, h, up_proj.data() + dst);
        }
    }
}

CpuRefWeights build_cpu_ref_weights(const MoeShape& s, RefMode mode) {
    CpuRefWeights w;
    w.gate_inp = get_raw_weight_vector(s, "mlp.gate.weight");
    w.shared_expert_gate = get_raw_weight_vector(s, "mlp.shared_expert_gate.weight");
    w.shared_gate_proj = get_raw_weight_vector(s, "mlp.shared_expert.gate_proj.weight");
    w.shared_up_proj = get_raw_weight_vector(s, "mlp.shared_expert.up_proj.weight");
    w.shared_down_proj = get_raw_weight_vector(s, "mlp.shared_expert.down_proj.weight");

    const auto fused_gate_up = get_raw_weight_vector(s, "mlp.experts.gate_up_proj");
    const auto raw_down = get_raw_weight_vector(s, "mlp.experts.down_proj");
    std::vector<float> raw_gate;
    std::vector<float> raw_up;
    split_gate_up(s, fused_gate_up, raw_gate, raw_up);

    if (mode == RefMode::FP32) {
        w.gate_proj = std::move(raw_gate);
        w.up_proj = std::move(raw_up);
        w.down_proj = raw_down;
        return w;
    }

    const size_t e = static_cast<size_t>(s.num_experts);
    const size_t i = static_cast<size_t>(s.moe_intermediate_size);
    const size_t h = static_cast<size_t>(s.hidden_size);
    constexpr size_t kGroup = 128;

    OPENVINO_ASSERT((h % kGroup) == 0, "hidden_size must be divisible by 128 for INT4 test");
    OPENVINO_ASSERT((i % kGroup) == 0, "moe_intermediate_size must be divisible by 128 for INT4 test");

    const auto q_gate = quantize_q41(raw_gate, e, i, h, kGroup);
    const auto q_up = quantize_q41(raw_up, e, i, h, kGroup);
    const auto q_down = quantize_q41(raw_down, e, h, i, kGroup);

    w.gate_proj = dequantize_q41(q_gate, e, i, h);
    w.up_proj = dequantize_q41(q_up, e, i, h);
    w.down_proj = dequantize_q41(q_down, e, h, i);
    return w;
}

std::vector<float> build_qwen35_moe_cpu_ref(const MoeShape& s,
                                            const ov::Tensor& input_tensor,
                                            const CpuRefWeights& w) {
    OPENVINO_ASSERT(input_tensor.get_element_type() == ov::element::f32, "CPU ref expects f32 input");
    const auto input = tensor_to_f32_vector(input_tensor);

    const auto routed = moe_ref(input,
                                w.gate_inp,
                                w.gate_proj,
                                w.up_proj,
                                w.down_proj,
                                static_cast<size_t>(s.batch),
                                static_cast<size_t>(s.seq_len),
                                static_cast<size_t>(s.hidden_size),
                                static_cast<size_t>(s.moe_intermediate_size),
                                static_cast<size_t>(s.num_experts),
                                static_cast<size_t>(s.num_experts_per_tok));

    const auto shared = mlp_ref(input,
                                w.shared_gate_proj,
                                w.shared_up_proj,
                                w.shared_down_proj,
                                static_cast<size_t>(s.batch),
                                static_cast<size_t>(s.seq_len),
                                static_cast<size_t>(s.hidden_size),
                                static_cast<size_t>(s.shared_expert_intermediate_size));

    const auto shared_gate_logits = linear_ref_3d(input,
                                                  w.shared_expert_gate,
                                                  static_cast<size_t>(s.batch),
                                                  static_cast<size_t>(s.seq_len),
                                                  static_cast<size_t>(s.hidden_size),
                                                  1);

    std::vector<float> out = routed;
    const size_t tokens = static_cast<size_t>(s.batch) * static_cast<size_t>(s.seq_len);
    const size_t hidden = static_cast<size_t>(s.hidden_size);
    for (size_t t = 0; t < tokens; ++t) {
        const float g = 1.0f / (1.0f + std::exp(-shared_gate_logits[t]));
        for (size_t h = 0; h < hidden; ++h) {
            out[t * hidden + h] += shared[t * hidden + h] * g;
        }
    }
    return out;
}

class TestInt4MoeFinalizer : public WeightFinalizer {
public:
    explicit TestInt4MoeFinalizer(Int4RouteMode route_mode, size_t group_size = 128)
        : route_mode_(route_mode),
          group_size_(group_size) {}

    FinalizedWeight finalize(const std::string& name, WeightSource& source, OpContext& ctx) override {
        const ov::Tensor& tensor = source.get_tensor(name);

        const bool is_gate = name.find("mlp.experts.gate_proj.weight") != std::string::npos;
        const bool is_up = name.find("mlp.experts.up_proj.weight") != std::string::npos;
        const bool is_down = name.find("mlp.experts.down_proj") != std::string::npos;
        if (!is_gate && !is_up && !is_down) {
            // Keep non-expert weights in fp32 to avoid MatMul dtype mismatch (f32 x f16).
            auto w = ops::constant(tensor, &ctx);
            if (tensor.get_element_type() != ov::element::f32) {
                w = w.to(ov::element::f32);
            }
            return FinalizedWeight(w, {});
        }

        const auto shape = tensor.get_shape();
        OPENVINO_ASSERT(shape.size() == 3, "Expected 3D expert tensor shape for ", name);
        const size_t e = shape[0];
        const size_t o = shape[1];
        const size_t i = shape[2];
        OPENVINO_ASSERT((i % group_size_) == 0,
                        "Input dim must be divisible by group_size for INT4 test quantization: ", name);

        const auto w_f32 = tensor_to_f32_vector(tensor);
        const auto q = quantize_q41(w_f32, e, o, i, group_size_);

        if (is_down && route_mode_ == Int4RouteMode::ForceOpsetFallback) {
            // Keep INT4 quantized values as source of truth, then bind dequantized dense tensor
            // without auxiliary scales/zps so fused path is disabled.
            const auto deq = dequantize_q41(q, e, o, i);
            const auto dense = make_f32_tensor_from_f32(deq, {e, o, i});
            return FinalizedWeight(ops::constant(dense, &ctx), {});
        }

        ov::Tensor packed(ov::element::u4, {e, o, q.group_num, q.group_size});
        std::memcpy(packed.data(), q.weights_packed.data(), packed.get_byte_size());

        std::unordered_map<std::string, Tensor> aux;
        aux.emplace("scales", ops::constant(q.scales_f16, &ctx));
        aux.emplace("zps", ops::constant(q.zps_u4, &ctx));
        return FinalizedWeight(ops::constant(packed, &ctx), aux);
    }

private:
    Int4RouteMode route_mode_;
    size_t group_size_;
};

Qwen3_5TextModelConfig make_moe_text_cfg(const MoeShape& s) {
    Qwen3_5TextModelConfig cfg;
    cfg.hidden_size = s.hidden_size;
    cfg.moe_intermediate_size = s.moe_intermediate_size;
    cfg.shared_expert_intermediate_size = s.shared_expert_intermediate_size;
    cfg.num_experts = s.num_experts;
    cfg.num_experts_per_tok = s.num_experts_per_tok;
    cfg.norm_topk_prob = true;
    cfg.hidden_act = "silu";
    return cfg;
}

std::shared_ptr<ov::Model> build_qwen3_5_moe_only_model(const MoeShape& s,
                                                         WeightSource& source,
                                                         WeightFinalizer& finalizer) {
    BuilderContext ctx;
    Qwen3_5SparseMoeBlock moe(ctx, "mlp", make_moe_text_cfg(s));
    (void)load_model(moe, source, finalizer);

    auto hidden_states = ctx.parameter("hidden_states",
                                       ov::element::f32,
                                       ov::Shape{static_cast<size_t>(s.batch),
                                                 static_cast<size_t>(s.seq_len),
                                                 static_cast<size_t>(s.hidden_size)});
    auto out = moe.forward(hidden_states);
    return ctx.build_model({out.output()});
}

bool model_contains_internal_fused_moe(const std::shared_ptr<ov::Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        const std::string type_name = node->get_type_name();
        if (type_name.find("MOE3GemmFusedCompressed") != std::string::npos) {
            return true;
        }
    }
    return false;
}

bool has_gpu_device() {
    ov::Core core;
    for (const auto& device : core.get_available_devices()) {
        if (device.rfind("GPU", 0) == 0) {
            return true;
        }
    }
    return false;
}

ov::Tensor run_model_on_gpu(const std::shared_ptr<ov::Model>& model, const ov::Tensor& input) {
    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();
    request.set_input_tensor(0, input);
    request.infer();
    return request.get_output_tensor(0);
}

struct TensorDiffStats {
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
};

TensorDiffStats compare_tensors(const ov::Tensor& a, const ov::Tensor& b) {
    OPENVINO_ASSERT(a.get_shape() == b.get_shape(), "Tensor shape mismatch");
    OPENVINO_ASSERT(a.get_element_type() == ov::element::f32 && b.get_element_type() == ov::element::f32,
                    "compare_tensors expects f32");

    const auto* pa = a.data<const float>();
    const auto* pb = b.data<const float>();
    const size_t n = a.get_size();

    TensorDiffStats stats;
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const float d = std::abs(pa[i] - pb[i]);
        stats.max_abs = std::max(stats.max_abs, d);
        sum += static_cast<double>(d);
    }
    stats.mean_abs = static_cast<float>(sum / static_cast<double>(n));
    return stats;
}

class Qwen3_5MoeCoreULT : public ::testing::Test {
protected:
    static void skip_if_no_gpu() {
        if (!has_gpu_device()) {
            GTEST_SKIP() << "GPU device is not available for Qwen3.5 MoE ULT.";
        }
    }
};

TEST_F(Qwen3_5MoeCoreULT, OpsetMoeFp32MatchesCpuRefOnGpu) {
    skip_if_no_gpu();
    const auto shape = make_ref_shape();
    const auto input = make_input_tensor(shape, 0x101u);
    const auto ref_weights = build_cpu_ref_weights(shape, RefMode::FP32);
    const auto expected = build_qwen35_moe_cpu_ref(shape, input, ref_weights);

    DeterministicMoeWeightSource source(shape);
    DummyWeightFinalizer finalizer;
    auto model = build_qwen3_5_moe_only_model(shape, source, finalizer);
    ASSERT_FALSE(model_contains_internal_fused_moe(model));

    auto out = run_model_on_gpu(model, input);
    expect_tensor_near(out, expected, 2e-2f);
}

TEST_F(Qwen3_5MoeCoreULT, OpsetMoeInt4MatchesCpuRefOnGpu) {
    skip_if_no_gpu();
    const auto shape = make_ref_shape();
    const auto input = make_input_tensor(shape, 0x202u);
    const auto ref_weights = build_cpu_ref_weights(shape, RefMode::INT4);
    const auto expected = build_qwen35_moe_cpu_ref(shape, input, ref_weights);

    DeterministicMoeWeightSource source(shape);
    TestInt4MoeFinalizer finalizer(Int4RouteMode::ForceOpsetFallback, 128);
    auto model = build_qwen3_5_moe_only_model(shape, source, finalizer);
    ASSERT_FALSE(model_contains_internal_fused_moe(model));

    auto out = run_model_on_gpu(model, input);
    expect_tensor_near(out, expected, k_tol_moe);
}

TEST_F(Qwen3_5MoeCoreULT, FusedMoeInt4MatchesCpuRefOnGpu) {
    skip_if_no_gpu();
    const auto shape = make_fused_ref_shape();
    if (shape.num_experts < 32) {
        GTEST_SKIP() << "Fused MoE GPU test requires num_experts >= 32 for stable kernel path.";
    }
    if (shape.batch * shape.seq_len != 1) {
        GTEST_SKIP() << "Fused MoE ULT is restricted to single-token shape (batch*seq==1) to avoid unstable multi-token path.";
    }
    const auto input = make_input_tensor(shape, 0x303u);
    const auto ref_weights = build_cpu_ref_weights(shape, RefMode::INT4);
    const auto expected = build_qwen35_moe_cpu_ref(shape, input, ref_weights);

    DeterministicMoeWeightSource source(shape);
    TestInt4MoeFinalizer finalizer(Int4RouteMode::EnableFused, 128);
    auto model = build_qwen3_5_moe_only_model(shape, source, finalizer);
    ASSERT_TRUE(model_contains_internal_fused_moe(model));

    auto out = run_model_on_gpu(model, input);
    expect_tensor_near(out, expected, k_tol_moe);
}

TEST_F(Qwen3_5MoeCoreULT, FusedAndOpsetOutputsStayConsistentOnGpu) {
    skip_if_no_gpu();

    // This profile test intentionally uses real Qwen3.5-35B-A3B MoE dimensions.
    const auto shape = make_qwen35_full_shape();
    if (shape.num_experts < 32) {
        GTEST_SKIP() << "Fused MoE GPU test requires num_experts >= 32 for stable kernel path.";
    }
    if (shape.batch * shape.seq_len != 1) {
        GTEST_SKIP() << "Set OV_GENAI_QWEN35_MOE_FULL_BATCH=1 and OV_GENAI_QWEN35_MOE_FULL_SEQ=1 for stable fused GPU comparison.";
    }
    const auto input = make_input_tensor(shape, 0x404u);

    DeterministicMoeWeightSource opset_source(shape);
    DeterministicMoeWeightSource fused_source(shape);
    TestInt4MoeFinalizer opset_finalizer(Int4RouteMode::ForceOpsetFallback, 128);
    TestInt4MoeFinalizer fused_finalizer(Int4RouteMode::EnableFused, 128);

    auto opset_model = build_qwen3_5_moe_only_model(shape, opset_source, opset_finalizer);
    auto fused_model = build_qwen3_5_moe_only_model(shape, fused_source, fused_finalizer);

    ASSERT_FALSE(model_contains_internal_fused_moe(opset_model));
    ASSERT_TRUE(model_contains_internal_fused_moe(fused_model));

    auto out_opset = run_model_on_gpu(opset_model, input);
    auto out_fused = run_model_on_gpu(fused_model, input);

    ASSERT_EQ(out_opset.get_shape(), out_fused.get_shape());
    ASSERT_EQ(out_opset.get_element_type(), ov::element::f32);
    ASSERT_EQ(out_fused.get_element_type(), ov::element::f32);

    const auto stats = compare_tensors(out_opset, out_fused);
    EXPECT_LE(stats.max_abs, 1.2f);
    EXPECT_LE(stats.mean_abs, 0.2f);
}

}  // namespace
