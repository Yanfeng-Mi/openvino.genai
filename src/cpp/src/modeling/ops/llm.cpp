// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/ops/llm.hpp"

#include <cmath>
#include <vector>

#include <openvino/opsets/opset13.hpp>
#include <openvino/core/except.hpp>
#include <ov_ops/rotary_positional_embeddings.hpp>

#include "modeling/ops/ops.hpp"
#include "modeling/ops/shape.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace ops {
namespace llm {

std::pair<Tensor, Tensor> rope_cos_sin(const Tensor& positions,
                                       int32_t head_dim,
                                       float rope_theta,
                                       const OpPolicy* policy) {
    (void)policy;
    auto* ctx = positions.context();
    const int32_t half_dim = head_dim / 2;
    std::vector<float> inv_freq(static_cast<size_t>(half_dim));
    for (int32_t i = 0; i < half_dim; ++i) {
        float exponent = static_cast<float>(2 * i) / static_cast<float>(head_dim);
        inv_freq[static_cast<size_t>(i)] = 1.0f / std::pow(rope_theta, exponent);
    }

    auto inv_freq_const = const_vec(ctx, inv_freq);
    auto inv_freq_shape =
        const_vec(ctx, std::vector<int64_t>{1, 1, static_cast<int64_t>(half_dim)});
    Tensor inv_freq_tensor(inv_freq_const, ctx);
    auto inv_freq_reshaped = inv_freq_tensor.reshape(inv_freq_shape, false);

    auto pos_f = positions.to(ov::element::f32);
    auto freqs = pos_f.unsqueeze(2) * inv_freq_reshaped;
    return {freqs.cos(), freqs.sin()};
}

Tensor apply_rope(const Tensor& x,
                  const Tensor& cos,
                  const Tensor& sin,
                  int32_t head_dim,
                  const OpPolicy* policy) {
    if (head_dim % 2 != 0) {
        OPENVINO_THROW("apply_rope expects even head_dim");
    }
    (void)policy;
    // Align rotary factors with activation dtype to avoid mixed-type multiplies.
    auto dtype = x.dtype();
    auto cos_unsq = cos.to(dtype).unsqueeze(1);
    auto sin_unsq = sin.to(dtype).unsqueeze(1);
    int64_t half_dim = head_dim / 2;

    const bool use_internal = policy ? policy->use_internal_rope : true;
    if (!use_internal) {
        const int32_t half_dim = head_dim / 2;
        auto cos_cast = cos.to(x.dtype()).unsqueeze(1);  // [B, 1, S, half]
        auto sin_cast = sin.to(x.dtype()).unsqueeze(1);  // [B, 1, S, half]

        auto x1 = slice(x, 0, half_dim, 1, 3);
        auto x2 = slice(x, half_dim, head_dim, 1, 3);

        auto out1 = x1 * cos_cast - x2 * sin_cast;
        auto out2 = x1 * sin_cast + x2 * cos_cast;
        return concat({out1, out2}, 3);
    }

    // Use internal RoPE op directly for optimal GPU performance.
    // This avoids relying on RoPEFusion transformation to match patterns.
    //
    // Input shapes:
    //   x: [batch, heads, seq, head_dim]
    //   cos/sin: [batch, seq, half_dim] where half_dim = head_dim / 2
    //
    // RoPE op expects cos/sin with shape [batch, seq, rotary_ndims] and will
    // handle the rotation internally.

    op::internal::RoPE::Config config;
    config.rotary_ndims = static_cast<size_t>(head_dim);
    config.is_interleaved = false;
    config.input_trans0213 = false;
    config.output_trans0213 = false;

    // Expand cos/sin from half_dim to head_dim by concatenating with themselves
    auto cos_full = concat({cos, cos}, 2);  // [batch, seq, head_dim]
    auto sin_full = concat({sin, sin}, 2);  // [batch, seq, head_dim]

    // Unsqueeze to 4D: [batch, 1, seq, head_dim] to use GPU's 4D RoPE path
    // which is more thoroughly tested than the 3D path that has indexing bugs
    auto cos_4d = cos_full.unsqueeze(1);  // [batch, 1, seq, head_dim]
    auto sin_4d = sin_full.unsqueeze(1);  // [batch, 1, seq, head_dim]

    // Since we expanded cos/sin to full head_dim, set cos_sin_ndims accordingly
    config.cos_sin_ndims = static_cast<size_t>(head_dim);

    auto rope_node = std::make_shared<op::internal::RoPE>(
        ov::OutputVector{x.output(), cos_4d.output(), sin_4d.output()},
        config);

    return Tensor(rope_node, x.context());
}

Tensor apply_rope_interleave(const Tensor& x,
                             const Tensor& cos,
                             const Tensor& sin,
                             int32_t head_dim,
                             const OpPolicy* policy) {
    (void)policy;
    const int32_t half_dim = head_dim / 2;
    auto interleaved = x.reshape({0, 0, 0, half_dim, 2})
                           .permute({0, 1, 2, 4, 3})
                           .reshape({0, 0, 0, head_dim});
    return apply_rope(interleaved, cos, sin, head_dim, policy);
}

Tensor rope_tail(const Tensor& cos_or_sin, const Tensor& q) {
    auto* ctx = cos_or_sin.context();
    auto total_len = shape::dim(cos_or_sin, 1);
    auto q_len = shape::dim(q, 2);

    auto total_len_scalar = Tensor(total_len, ctx).squeeze(0);
    auto q_len_scalar = Tensor(q_len, ctx).squeeze(0);
    auto start = total_len_scalar - q_len_scalar;

    auto indices = range(start, total_len_scalar, 1, ov::element::i64);
    return gather(cos_or_sin, indices, 1);
}

Tensor pad_to_head_dim(const Tensor& x, int32_t head_dim, int32_t target_head_dim) {
    if (target_head_dim <= head_dim) {
        return x;
    }
    auto* ctx = x.context();
    const int32_t pad = target_head_dim - head_dim;

    auto batch = shape::dim(x, 0);
    auto heads = shape::dim(x, 1);
    auto seq = shape::dim(x, 2);
    auto pad_dim = const_vec(ctx, std::vector<int64_t>{static_cast<int64_t>(pad)});
    auto pad_shape = shape::make({batch, heads, seq, pad_dim});

    auto zero = Tensor(const_scalar(ctx, 0.0f), ctx).to(x.dtype());
    auto pad_tensor = shape::broadcast_to(zero, pad_shape);
    return concat({x, pad_tensor}, 3);
}

Tensor slice_to_head_dim(const Tensor& x, int32_t head_dim, int32_t target_head_dim) {
    if (target_head_dim >= head_dim) {
        return x;
    }
    return slice(x, 0, target_head_dim, 1, 3);
}

Tensor repeat_kv(const Tensor& x, int32_t num_heads, int32_t num_kv_heads, int32_t head_dim) {
    if (num_heads == num_kv_heads) {
        return x;
    }
    auto* ctx = x.context();
    const int32_t repeats = num_heads / num_kv_heads;
    auto unsq = x.unsqueeze(2);

    auto batch = shape::dim(x, 0);
    auto seq = shape::dim(x, 2);

    auto kv_heads = const_vec(ctx, std::vector<int64_t>{static_cast<int64_t>(num_kv_heads)});
    auto rep = const_vec(ctx, std::vector<int64_t>{static_cast<int64_t>(repeats)});
    auto hdim = const_vec(ctx, std::vector<int64_t>{static_cast<int64_t>(head_dim)});

    auto target = shape::make({batch, kv_heads, rep, seq, hdim});
    auto broadcast = shape::broadcast_to(unsq, target);

    auto heads = const_vec(ctx, std::vector<int64_t>{static_cast<int64_t>(num_heads)});
    auto reshape_shape = shape::make({batch, heads, seq, hdim});
    return broadcast.reshape(reshape_shape, false);
}

Tensor causal_mask_from_seq_len(const Tensor& seq_len) {
    auto* ctx = seq_len.context();
    auto idx = range(seq_len, 0, 1, ov::element::i64);
    auto row = idx.unsqueeze(1);
    auto col = idx.unsqueeze(0);
    auto ge = greater_equal(row, col);

    auto zero = Tensor(const_scalar(ctx, 0.0f), ctx);
    auto neg = Tensor(const_scalar(ctx, -65504.0f), ctx);
    auto mask2d = where(ge, zero, neg);
    return mask2d.unsqueeze({0, 1});
}

Tensor causal_mask(const Tensor& scores) {
    auto scores_shape = shape::of(scores);
    auto seq = Tensor(shape::dim(scores, 2), scores.context()).squeeze(0);
    auto mask4d = causal_mask_from_seq_len(seq);
    return shape::broadcast_to(mask4d, scores_shape);
}

namespace {

Tensor build_kv_causal_mask_impl(ov::genai::modeling::OpContext* ctx,
                                 const ov::Output<ov::Node>& batch,
                                 const Tensor& q_len_1d,
                                 const Tensor& kv_len_1d,
                                 const Tensor* attention_mask) {
    // Squeeze [1] length tensors to scalars for Range ops.
    auto q_len_scalar = q_len_1d.squeeze(0);
    auto kv_len_scalar = kv_len_1d.squeeze(0);

    // cache_len = kv_len - q_len.
    auto cache_len_scalar = Tensor(
        std::make_shared<ov::opset13::Subtract>(kv_len_scalar.output(), q_len_scalar.output())->output(0), ctx);

    auto cache_len_i32 = Tensor(
        std::make_shared<ov::op::v0::Convert>(cache_len_scalar.output(), ov::element::i32)->output(0), ctx);
    auto q_len_i32 = Tensor(
        std::make_shared<ov::op::v0::Convert>(q_len_scalar.output(), ov::element::i32)->output(0), ctx);
    auto kv_len_i32 = Tensor(
        std::make_shared<ov::op::v0::Convert>(kv_len_scalar.output(), ov::element::i32)->output(0), ctx);

    // col indices: [1, kv_len], row indices: [q_len, 1] in absolute token positions.
    auto col_indices = range(kv_len_i32, 0, 1, ov::element::i32).unsqueeze(0);
    auto row_indices = range(cache_len_i32, cache_len_i32 + q_len_i32, 1, ov::element::i32).unsqueeze(1);
    auto causal_cond = less_equal(col_indices, row_indices);  // [q_len, kv_len] bool

    auto zero_val = Tensor(const_scalar(ctx, 0.0f), ctx);
    auto neg_inf = Tensor(const_scalar(ctx, -65504.0f), ctx);

    if (!attention_mask) {
        auto mask_2d = where(causal_cond, zero_val, neg_inf);
        auto mask_4d = mask_2d.unsqueeze({0, 1});
        auto one_val = const_vec(ctx, std::vector<int64_t>{1});
        auto target_shape = shape::make({batch, one_val, q_len_1d.output(), kv_len_1d.output()});
        return shape::broadcast_to(mask_4d, target_shape);
    }

    // Combine causal and padding constraints in boolean space:
    // attend = causal_cond && (attention_mask != 0), then map to {0, -inf}.
    auto attn_i32 = Tensor(
        std::make_shared<ov::op::v0::Convert>(attention_mask->output(), ov::element::i32)->output(0), ctx);
    auto zero_i32 = Tensor(const_scalar(ctx, int32_t{0}), ctx);
    auto valid_tokens = Tensor(
        std::make_shared<ov::op::v1::NotEqual>(attn_i32.output(), zero_i32.output())->output(0), ctx);  // [B, kv_len]

    auto causal_4d = causal_cond.unsqueeze({0, 1});      // [1, 1, q_len, kv_len]
    auto valid_tokens_4d = valid_tokens.unsqueeze({1, 2});  // [B, 1, 1, kv_len]
    auto can_attend = Tensor(
        std::make_shared<ov::op::v1::LogicalAnd>(causal_4d.output(), valid_tokens_4d.output())->output(0), ctx);
    auto combined_mask = where(can_attend, zero_val, neg_inf);  // [B, 1, q_len, kv_len]

    // Keep NPU/NPUW-identifiable SDPA mask pattern.
    auto zero_1d = const_vec(ctx, std::vector<int64_t>{0});
    auto max_1d = const_vec(ctx, std::vector<int64_t>{std::numeric_limits<int64_t>::max()});
    auto one_1d = const_vec(ctx, std::vector<int64_t>{1});
    auto axis_1d = const_vec(ctx, std::vector<int64_t>{3});
    auto slice_node = std::make_shared<ov::op::v8::Slice>(combined_mask.output(), zero_1d, max_1d, one_1d, axis_1d);
    return Tensor(slice_node, ctx);
}

}  // namespace

Tensor build_kv_causal_mask(const Tensor& q, const Tensor& k) {
    // Build causal mask for KV cache scenario:
    // Q: [batch, heads, q_len, head_dim]
    // K: [batch, heads, kv_len, head_dim]
    // Output: [batch, 1, q_len, kv_len] where mask[i,j]=0 if col_j <= row_i_absolute, else -inf
    //
    // For prefill: q_len=N, kv_len=N, behaves like standard causal mask
    // For decode: q_len=1, kv_len=cache_len+1, allows attending to all positions
    auto* ctx = q.context();
    auto batch = shape::dim(q, 0);
    auto q_len = Tensor(shape::dim(q, 2), ctx);
    auto kv_len = Tensor(shape::dim(k, 2), ctx);
    return build_kv_causal_mask_impl(ctx, batch, q_len, kv_len, nullptr);
}

Tensor build_kv_causal_mask_with_attention(const Tensor& q, const Tensor& k, const Tensor& attention_mask) {
    // Build causal mask for KV cache scenario with attention_mask integration.
    // attention_mask: [batch, kv_len] where 1=attend, 0=mask (padding)
    // Q: [batch, heads, q_len, head_dim]
    // K: [batch, heads, kv_len, head_dim]
    // Output: [batch, 1, q_len, kv_len]
    auto* ctx = q.context();
    auto batch = shape::dim(q, 0);
    auto q_len = Tensor(shape::dim(q, 2), ctx);
    auto kv_len = Tensor(shape::dim(k, 2), ctx);
    return build_kv_causal_mask_impl(ctx, batch, q_len, kv_len, &attention_mask);
}

Tensor build_kv_causal_mask_with_attention_from_q_len(const Tensor& q_len, const Tensor& attention_mask) {
    auto* ctx = q_len.context() ? q_len.context() : attention_mask.context();
    if (q_len.context() && attention_mask.context() && q_len.context() != attention_mask.context()) {
        OPENVINO_THROW("Tensor contexts do not match");
    }
    auto batch = shape::dim(attention_mask, 0);
    auto kv_len = Tensor(shape::dim(attention_mask, 1), ctx);

    Tensor q_len_1d = q_len;
    const auto q_rank = q_len.output().get_partial_shape().rank();
    if (q_rank.is_static() && q_rank.get_length() == 0) {
        q_len_1d = q_len.unsqueeze(0);
    }

    return build_kv_causal_mask_impl(ctx, batch, q_len_1d, kv_len, &attention_mask);
}

Tensor sdpa(const Tensor& q,
            const Tensor& k,
            const Tensor& v,
            float scale,
            int64_t softmax_axis,
            const Tensor* mask,
            bool causal,
            const OpPolicy* policy) {
    (void)policy;
    (void)softmax_axis;  // Native SDPA handles this internally

    auto* ctx = q.context();
    
    // Create scale constant - this is critical for NPU compatibility
    // The scale is typically 1/sqrt(head_dim)
    auto scale_const = std::make_shared<ov::op::v0::Constant>(
        ov::element::f32, ov::Shape{}, std::vector<float>{scale});

    // Use native ScaledDotProductAttention for optimal GPU performance
    if (mask) {
        auto sdpa_node = std::make_shared<ov::op::v13::ScaledDotProductAttention>(
            q.output(), k.output(), v.output(), mask->output(), scale_const, causal);
        return Tensor(sdpa_node, ctx);
    } else {
        auto sdpa_node = std::make_shared<ov::op::v13::ScaledDotProductAttention>(
            q.output(), k.output(), v.output(), scale_const, causal);
        return Tensor(sdpa_node, ctx);
    }
}

}  // namespace llm
}  // namespace ops
}  // namespace modeling
}  // namespace genai
}  // namespace ov
