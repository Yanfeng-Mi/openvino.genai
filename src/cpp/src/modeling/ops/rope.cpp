// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/ops/rope.hpp"

#include <openvino/core/except.hpp>
#include <openvino/opsets/opset13.hpp>

#include "modeling/ops/ops.hpp"
#include "modeling/ops/shape.hpp"

namespace {

ov::genai::modeling::Tensor tensor_mod(const ov::genai::modeling::Tensor& a,
                                       const ov::genai::modeling::Tensor& b) {
    auto* ctx = a.context();
    auto node = std::make_shared<ov::op::v1::Mod>(a.output(), b.output(), ov::op::AutoBroadcastType::NUMPY);
    return ov::genai::modeling::Tensor(node, ctx);
}

ov::genai::modeling::Tensor tensor_equal(const ov::genai::modeling::Tensor& a,
                                         const ov::genai::modeling::Tensor& b) {
    auto* ctx = a.context();
    auto node = std::make_shared<ov::op::v1::Equal>(a.output(), b.output(), ov::op::AutoBroadcastType::NUMPY);
    return ov::genai::modeling::Tensor(node, ctx);
}

ov::genai::modeling::Tensor tensor_less(const ov::genai::modeling::Tensor& a,
                                        const ov::genai::modeling::Tensor& b) {
    auto* ctx = a.context();
    auto node = std::make_shared<ov::op::v1::Less>(a.output(), b.output(), ov::op::AutoBroadcastType::NUMPY);
    return ov::genai::modeling::Tensor(node, ctx);
}

ov::genai::modeling::Tensor tensor_and(const ov::genai::modeling::Tensor& a,
                                       const ov::genai::modeling::Tensor& b) {
    auto* ctx = a.context();
    auto node = std::make_shared<ov::op::v1::LogicalAnd>(a.output(), b.output(), ov::op::AutoBroadcastType::NUMPY);
    return ov::genai::modeling::Tensor(node, ctx);
}

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace ops {
namespace rope {

Tensor mrope_interleaved(const Tensor& freqs, const std::vector<int32_t>& mrope_section) {
    if (mrope_section.size() != 3) {
        OPENVINO_THROW("mrope_interleaved expects mrope_section size 3");
    }
    auto* ctx = freqs.context();
    auto t = ops::slice(freqs, 0, 1, 1, 0).squeeze(0);
    auto h = ops::slice(freqs, 1, 2, 1, 0).squeeze(0);
    auto w = ops::slice(freqs, 2, 3, 1, 0).squeeze(0);

    int64_t h_len = static_cast<int64_t>(mrope_section[1]) * 3;
    int64_t w_len = static_cast<int64_t>(mrope_section[2]) * 3;
    if (h_len <= 0 && w_len <= 0) {
        return t;
    }

    auto d_dim = Tensor(shape::dim(t, 2), ctx).squeeze(0);
    auto idx = ops::range(d_dim, 0, 1, ov::element::i64);
    auto mod = tensor_mod(idx, Tensor(ops::const_scalar(ctx, static_cast<int64_t>(3)), ctx));

    auto one = Tensor(ops::const_scalar(ctx, static_cast<int64_t>(1)), ctx);
    auto two = Tensor(ops::const_scalar(ctx, static_cast<int64_t>(2)), ctx);
    auto eq1 = tensor_equal(mod, one);
    auto eq2 = tensor_equal(mod, two);

    auto out = t;
    if (h_len > 0) {
        auto h_limit = Tensor(ops::const_scalar(ctx, h_len), ctx);
        auto in_h = tensor_less(idx, h_limit);
        auto mask_h = tensor_and(in_h, eq1);
        out = ops::where(mask_h, h, out);
    }
    if (w_len > 0) {
        auto w_limit = Tensor(ops::const_scalar(ctx, w_len), ctx);
        auto in_w = tensor_less(idx, w_limit);
        auto mask_w = tensor_and(in_w, eq2);
        out = ops::where(mask_w, w, out);
    }
    return out;
}

Tensor mrope_chunked(const Tensor& freqs, const std::vector<int32_t>& mrope_section) {
    if (mrope_section.size() != 3) {
        OPENVINO_THROW("mrope_chunked expects mrope_section size 3");
    }
    auto* ctx = freqs.context();
    // freqs shape: [3, batch, seq, half_dim]
    // Extract temporal (0), height (1), width (2) components
    auto t = ops::slice(freqs, 0, 1, 1, 0).squeeze(0);  // [batch, seq, half_dim]
    auto h = ops::slice(freqs, 1, 2, 1, 0).squeeze(0);
    auto w = ops::slice(freqs, 2, 3, 1, 0).squeeze(0);

    const int64_t s0 = static_cast<int64_t>(mrope_section[0]);
    const int64_t s1 = static_cast<int64_t>(mrope_section[1]);
    const int64_t s2 = static_cast<int64_t>(mrope_section[2]);

    // Slice contiguous blocks from each component along the last dim
    auto t_chunk = ops::slice(t, 0, s0, 1, 2);                // [batch, seq, s0]
    auto h_chunk = ops::slice(h, s0, s0 + s1, 1, 2);          // [batch, seq, s1]
    auto w_chunk = ops::slice(w, s0 + s1, s0 + s1 + s2, 1, 2); // [batch, seq, s2]

    // Concatenate: [batch, seq, s0+s1+s2] = [batch, seq, half_dim]
    return ops::concat({t_chunk, h_chunk, w_chunk}, 2);
}

}  // namespace rope
}  // namespace ops
}  // namespace modeling
}  // namespace genai
}  // namespace ov
