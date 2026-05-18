// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace ops {
namespace rope {

// freqs: [3, batch, seq, head_dim/2]
// returns: [batch, seq, head_dim/2]
Tensor mrope_interleaved(const Tensor& freqs, const std::vector<int32_t>& mrope_section);

// freqs: [3, batch, seq, head_dim/2], section=[s0, s1, s2] (sum = head_dim/2)
// returns: [batch, seq, head_dim/2] — temporal[0:s0] + height[s0:s0+s1] + width[s0+s1:s0+s1+s2]
Tensor mrope_chunked(const Tensor& freqs, const std::vector<int32_t>& mrope_section);

}  // namespace rope
}  // namespace ops
}  // namespace modeling
}  // namespace genai
}  // namespace ov
