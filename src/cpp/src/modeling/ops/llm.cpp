// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/ops/llm.hpp"

#include <cmath>
#include <vector>

#include <openvino/opsets/opset13.hpp>
#include <openvino/core/except.hpp>
#include <ov_ops/rotary_positional_embeddings.hpp>
#include <ov_ops/vl_sdpa.hpp>

#include "modeling/ops/ops.hpp"
#include "modeling/ops/shape.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace ops {
namespace llm {

namespace {

Tensor build_attention_visibility_from_attention_mask(const Tensor& attention_mask) {
    auto* ctx = attention_mask.context();
    auto mask_i32 = ops::convert(attention_mask, ov::element::i32);
    auto abs_mask = Tensor(std::make_shared<ov::op::v0::Abs>(mask_i32.output())->output(0), ctx);
    auto visible = Tensor(std::make_shared<ov::op::v0::Convert>(abs_mask.output(), ov::element::boolean)->output(0), ctx);
    return visible.unsqueeze({1, 2});
}

Tensor build_attention_visibility_from_padding_mask(const Tensor& padding_mask) {
    auto* ctx = padding_mask.context();
    auto zero = Tensor(ops::const_scalar(ctx, 0.0f), ctx).to(padding_mask.dtype());
    auto node = std::make_shared<ov::op::v1::Equal>(padding_mask.output(), zero.output(), ov::op::AutoBroadcastType::NUMPY);
    return Tensor(node, ctx);
}

Tensor logical_and(const Tensor& a, const Tensor& b) {
    auto* ctx = a.context() ? a.context() : b.context();
    auto node = std::make_shared<ov::op::v1::LogicalAnd>(a.output(), b.output(), ov::op::AutoBroadcastType::NUMPY);
    return Tensor(node, ctx);
}

Tensor normalize_length_1d(const Tensor& length) {
    const auto rank = length.output().get_partial_shape().rank();
    if (rank.is_static() && rank.get_length() == 0) {
        return length.unsqueeze(0);
    }
    return length;
}

Tensor build_kv_causal_mask_with_attention_from_lengths(const Tensor& q_len,
                                                        const Tensor& kv_len,
                                                        const Tensor& attention_mask,
                                                        const Tensor* precomputed_padding_mask) {
    auto* ctx = q_len.context() ? q_len.context() : (kv_len.context() ? kv_len.context() : attention_mask.context());
    if (!ctx) {
        OPENVINO_THROW("Tensor context is null");
    }
    if ((q_len.context() && q_len.context() != ctx) ||
        (kv_len.context() && kv_len.context() != ctx) ||
        (attention_mask.context() && attention_mask.context() != ctx)) {
        OPENVINO_THROW("Tensor contexts do not match");
    }

    auto q_len_scalar = normalize_length_1d(q_len).squeeze(0);
    auto kv_len_scalar = normalize_length_1d(kv_len).squeeze(0);

    auto cache_len_scalar = Tensor(
        std::make_shared<ov::opset13::Subtract>(kv_len_scalar.output(), q_len_scalar.output())->output(0), ctx);

    auto cache_len_i32 = Tensor(
        std::make_shared<ov::op::v0::Convert>(cache_len_scalar.output(), ov::element::i32)->output(0), ctx);
    auto q_len_i32 = Tensor(
        std::make_shared<ov::op::v0::Convert>(q_len_scalar.output(), ov::element::i32)->output(0), ctx);
    auto kv_len_i32 = Tensor(
        std::make_shared<ov::op::v0::Convert>(kv_len_scalar.output(), ov::element::i32)->output(0), ctx);

    auto col_range = range(kv_len_i32, 0, 1, ov::element::i32);
    auto col_indices = col_range.unsqueeze(0);

    auto q_len_plus_cache = cache_len_i32 + q_len_i32;
    auto row_range = range(cache_len_i32, q_len_plus_cache, 1, ov::element::i32);
    auto row_indices = row_range.unsqueeze(1);

    auto causal_cond = less_equal(col_indices, row_indices);
    auto causal_visible = causal_cond.unsqueeze({0, 1});

    Tensor padding_visible = precomputed_padding_mask
                                 ? build_attention_visibility_from_padding_mask(*precomputed_padding_mask)
                                 : build_attention_visibility_from_attention_mask(attention_mask);
    auto combined_visible = logical_and(causal_visible, padding_visible);

    auto zero_val = Tensor(const_scalar(ctx, 0.0f), ctx);
    auto neg_inf = Tensor(const_scalar(ctx, -65504.0f), ctx);
    auto mask = where(combined_visible, zero_val, neg_inf);

    auto zero_1d = const_vec(ctx, std::vector<int64_t>{0});
    auto max_1d = const_vec(ctx, std::vector<int64_t>{std::numeric_limits<int64_t>::max()});
    auto one_1d = const_vec(ctx, std::vector<int64_t>{1});
    auto axis_1d = const_vec(ctx, std::vector<int64_t>{3});

    auto slice_node = std::make_shared<ov::op::v8::Slice>(mask.output(), zero_1d, max_1d, one_1d, axis_1d);
    return Tensor(slice_node, ctx);
}

}  // namespace

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
                  int32_t rotary_ndims,
                  const OpPolicy* policy,
                  int32_t head_size) {
    if (rotary_ndims % 2 != 0) {
        OPENVINO_THROW("apply_rope expects even rotary_ndims");
    }
    if (head_size < 0) {
        head_size = rotary_ndims;
    }
    if (head_size < rotary_ndims) {
        OPENVINO_THROW("apply_rope expects head_size >= rotary_ndims");
    }
    const auto x_ps = x.output().get_partial_shape();
    const auto x_rank = x_ps.rank();
    if (!x_rank.is_static() || (x_rank.get_length() != 3 && x_rank.get_length() != 4)) {
        OPENVINO_THROW("apply_rope expects rank-3 or rank-4 x input");
    }
    const auto x_rank_len = x_rank.get_length();
    if (x_rank.is_static() && x_rank.get_length() > 0) {
        const auto& last_dim = x_ps[x_rank.get_length() - 1];
        if (last_dim.is_static() && last_dim.get_length() != head_size) {
            OPENVINO_THROW("apply_rope head_size mismatch: expected last dim ", head_size, ", got ", last_dim.get_length());
        }
    }
    const int32_t half_rotary_ndims = rotary_ndims / 2;

    const bool use_internal = policy ? policy->use_internal_rope : true;
    if (!use_internal) {
        auto cos_cast = cos.to(x.dtype());
        auto sin_cast = sin.to(x.dtype());
        const auto cos_rank = cos.output().get_partial_shape().rank();
        const auto sin_rank = sin.output().get_partial_shape().rank();
        if (x_rank_len == 4) {
            if (cos_rank.is_static() && cos_rank.get_length() == 3) {
                cos_cast = cos_cast.unsqueeze(1);  // [B, 1, S, half]
            }
            if (sin_rank.is_static() && sin_rank.get_length() == 3) {
                sin_cast = sin_cast.unsqueeze(1);  // [B, 1, S, half]
            }
        } else {
            if (cos_rank.is_static() && cos_rank.get_length() == 2) {
                cos_cast = cos_cast.unsqueeze(1);  // [S, 1, half]
            }
            if (sin_rank.is_static() && sin_rank.get_length() == 2) {
                sin_cast = sin_cast.unsqueeze(1);  // [S, 1, half]
            }
        }

        auto x1 = slice(x, 0, half_rotary_ndims, 1, x_rank_len - 1);
        auto x2 = slice(x, half_rotary_ndims, rotary_ndims, 1, x_rank_len - 1);

        auto out1 = x1 * cos_cast - x2 * sin_cast;
        auto out2 = x1 * sin_cast + x2 * cos_cast;
        auto rotated = concat({out1, out2}, x_rank_len - 1);
        if (head_size == rotary_ndims) {
            return rotated;
        }

        auto tail = slice(x, rotary_ndims, head_size, 1, x_rank_len - 1);
        return concat({rotated, tail}, x_rank_len - 1);
    }

    // Use internal RoPE op directly for optimal GPU performance.
    // This avoids relying on RoPEFusion transformation to match patterns.
    //
    // Input shapes:
    //   x: [batch, heads, seq, head_dim]
    //   cos/sin: [batch, seq, half_rotary_ndims]
    //
    // Keep x at full head_size and pass half-width cos/sin tables directly.
    // The RoPE kernel can then rotate only rotary_ndims and preserve the tail
    // without materializing slice/concat scaffolding in the graph.

    op::internal::RoPE::Config config;
    config.rotary_ndims = static_cast<size_t>(rotary_ndims);
    config.cos_sin_ndims = static_cast<size_t>(half_rotary_ndims);
    config.is_interleaved = false;
    config.input_trans0213 = false;
    config.output_trans0213 = false;
    config.support_3d_rope = (x_rank_len == 3);
    config.head_size = static_cast<size_t>(head_size);

    if (x_ps[1].is_static()) {
        config.head_cnt = static_cast<size_t>(x_ps[1].get_length());
    }

    // Keep cos/sin tables in their original precision.
    // Internal RoPE kernels already support mixed x/cos/sin precision, and
    // downcasting trig tables to x.dtype regresses rotation accuracy on GPU.
    // CPU RoPE also reads cos/sin buffers as float tables.
    Tensor rope_cos = cos;
    Tensor rope_sin = sin;
    const auto cos_rank = cos.output().get_partial_shape().rank();
    const auto sin_rank = sin.output().get_partial_shape().rank();
    if (x_rank_len == 4) {
        if (cos_rank.is_static() && cos_rank.get_length() == 3) {
            rope_cos = cos.unsqueeze(1);  // [batch, 1, seq, half_rotary_ndims]
        }
        if (sin_rank.is_static() && sin_rank.get_length() == 3) {
            rope_sin = sin.unsqueeze(1);  // [batch, 1, seq, half_rotary_ndims]
        }
    } else {
        if (cos_rank.is_static() && cos_rank.get_length() == 2) {
            rope_cos = cos.unsqueeze(1);  // [seq, 1, half_rotary_ndims]
        }
        if (sin_rank.is_static() && sin_rank.get_length() == 2) {
            rope_sin = sin.unsqueeze(1);  // [seq, 1, half_rotary_ndims]
        }
    }

    auto rope_node = std::make_shared<op::internal::RoPE>(
        ov::OutputVector{x.output(), rope_cos.output(), rope_sin.output()},
        config);

    return Tensor(rope_node, x.context());
}

Tensor apply_rope_interleave(const Tensor& x,
                             const Tensor& cos,
                             const Tensor& sin,
                             int32_t head_dim,
                             const OpPolicy* policy) {
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
    // Use BIDIRECTIONAL broadcast to match GPU's Unsqueeze+Broadcast+Reshape+SDPA fusion pattern.
    auto broadcast = Tensor(
        std::make_shared<ov::op::v3::Broadcast>(unsq.output(), target, ov::op::BroadcastType::BIDIRECTIONAL), ctx);

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

Tensor build_kv_causal_mask(const Tensor& q, const Tensor& k) {
    // Build causal mask for KV cache scenario:
    // Q: [batch, heads, q_len, head_dim]
    // K: [batch, heads, kv_len, head_dim]
    // Output: [batch, 1, q_len, kv_len] where mask[i,j]=0 if col_j <= row_i_absolute, else -inf
    //
    // For prefill: q_len=N, kv_len=N, behaves like standard causal mask
    // For decode: q_len=1, kv_len=cache_len+1, allows attending to all positions
    auto* ctx = q.context();

    // Get dimensions as shape [1] tensors
    auto batch = shape::dim(q, 0);  // [1]
    auto q_len = shape::dim(q, 2);  // [1]
    auto kv_len = shape::dim(k, 2); // [1]

    // Squeeze to scalars for Range op
    auto q_len_scalar = Tensor(q_len, ctx).squeeze(0);
    auto kv_len_scalar = Tensor(kv_len, ctx).squeeze(0);

    // Calculate cache_seq_len = kv_len - q_len (as scalar)
    auto cache_len_scalar = Tensor(
        std::make_shared<ov::opset13::Subtract>(kv_len_scalar.output(), q_len_scalar.output())->output(0), ctx);

    // Convert to i32 for Range
    auto cache_len_i32 = Tensor(
        std::make_shared<ov::op::v0::Convert>(cache_len_scalar.output(), ov::element::i32)->output(0), ctx);
    auto q_len_i32 = Tensor(
        std::make_shared<ov::op::v0::Convert>(q_len_scalar.output(), ov::element::i32)->output(0), ctx);
    auto kv_len_i32 = Tensor(
        std::make_shared<ov::op::v0::Convert>(kv_len_scalar.output(), ov::element::i32)->output(0), ctx);

    // Create col indices: [0, 1, 2, ..., kv_len-1] -> [1, kv_len]
    auto col_range = range(kv_len_i32, 0, 1, ov::element::i32);
    auto col_indices = col_range.unsqueeze(0);  // [1, kv_len]

    // Create row indices: [cache_len, cache_len+1, ..., cache_len+q_len-1] -> [q_len, 1]
    // These represent the absolute positions of query tokens
    auto q_len_plus_cache = cache_len_i32 + q_len_i32;
    auto row_range = range(cache_len_i32, q_len_plus_cache, 1, ov::element::i32);
    auto row_indices = row_range.unsqueeze(1);  // [q_len, 1]

    // Causal condition: col <= row (attend to current and past positions)
    auto causal_cond = less_equal(col_indices, row_indices);  // [q_len, kv_len]

    // Build mask: 0 where can attend, -inf where masked
    auto zero_val = Tensor(const_scalar(ctx, 0.0f), ctx);
    auto neg_inf = Tensor(const_scalar(ctx, -65504.0f), ctx);
    auto mask_2d = where(causal_cond, zero_val, neg_inf);  // [q_len, kv_len]

    // Expand to [batch, 1, q_len, kv_len]
    auto mask_4d = mask_2d.unsqueeze({0, 1});

    // Broadcast to batch size
    auto one_val = const_vec(ctx, std::vector<int64_t>{1});
    auto target_shape = shape::make({batch, one_val, q_len, kv_len});
    return shape::broadcast_to(mask_4d, target_shape);
}

Tensor build_kv_padding_mask_from_attention(const Tensor& attention_mask) {
    auto* ctx = attention_mask.context();

    auto zero_val = Tensor(const_scalar(ctx, 0.0f), ctx);
    auto neg_inf = Tensor(const_scalar(ctx, -65504.0f), ctx);
    auto attn_mask_f32 =
        Tensor(std::make_shared<ov::op::v0::Convert>(attention_mask.output(), ov::element::f32)->output(0), ctx);
    auto attn_mask_cond =
        Tensor(std::make_shared<ov::op::v1::Equal>(attn_mask_f32.output(), zero_val.output())->output(0), ctx);
    auto padding_mask = where(attn_mask_cond, neg_inf, zero_val);
    return padding_mask.unsqueeze({1, 2});
}

Tensor build_kv_causal_mask_with_attention(const Tensor& q,
                                           const Tensor& k,
                                           const Tensor& attention_mask,
                                           const Tensor* precomputed_padding_mask) {
    auto* ctx = q.context();

    return build_kv_causal_mask_with_attention_from_lengths(
        Tensor(shape::dim(q, 2), ctx),
        Tensor(shape::dim(k, 2), ctx),
        attention_mask,
        precomputed_padding_mask);
}

Tensor build_kv_causal_mask_with_attention_from_q_len(const Tensor& q_len,
                                                      const Tensor& kv_len,
                                                      const Tensor& attention_mask) {
    return build_kv_causal_mask_with_attention_from_lengths(q_len, kv_len, attention_mask, nullptr);
}

Tensor vlsdpa(const Tensor& q,
              const Tensor& k,
              const Tensor& v,
              const Tensor& cu_seq_lens,
              const std::vector<int64_t>& order_q,
              const std::vector<int64_t>& order_k,
              const std::vector<int64_t>& order_v,
              const std::vector<int64_t>& order_out) {
    auto* ctx = q.context();
    auto vlsdpa_node = std::make_shared<ov::op::internal::VLSDPA>(
        ov::OutputVector{q.output(), k.output(), v.output(), cu_seq_lens.output()},
        order_q,
        order_k,
        order_v,
        order_out);
    return Tensor(vlsdpa_node, ctx);
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
