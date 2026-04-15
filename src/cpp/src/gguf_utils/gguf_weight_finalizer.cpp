// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_utils/gguf_weight_finalizer.hpp"

#include <openvino/core/except.hpp>

#include "gguf_utils/building_blocks.hpp"

namespace {

bool ends_with(const std::string& value, const std::string& suffix) {
    if (value.size() < suffix.size()) {
        return false;
    }
    return value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

namespace ov {
namespace genai {
namespace gguf {

GGUFWeightFinalizer::GGUFWeightFinalizer(const std::unordered_map<std::string, ov::Tensor>& consts,
                                         const std::unordered_map<std::string, gguf_tensor_type>& qtypes) :
    consts_(consts),
    qtypes_(qtypes) {}

gguf_tensor_type GGUFWeightFinalizer::resolve_qtype(const std::string& base_key) const {
    const bool has_scales = consts_.count(base_key + ".scales") > 0;
    if (!has_scales) {
        return gguf_tensor_type::GGUF_TYPE_F16;
    }
    const auto it = qtypes_.find(base_key + ".qtype");
    if (it == qtypes_.end()) {
        return gguf_tensor_type::GGUF_TYPE_F16;
    }
    return it->second;
}

std::string GGUFWeightFinalizer::base_key_from_name(const std::string& name) const {
    const std::string suffix = ".weight";
    if (ends_with(name, suffix)) {
        return name.substr(0, name.size() - suffix.size());
    }
    return name;
}

ov::genai::modeling::weights::FinalizedWeight GGUFWeightFinalizer::finalize(const std::string& name,
                                                                            ov::genai::modeling::weights::WeightSource& source,
                                                                            ov::genai::modeling::OpContext& ctx) {
    if (!source.has(name)) {
        OPENVINO_THROW("Missing GGUF tensor: ", name);
    }
    const std::string base_key = base_key_from_name(name);
    const auto it_cached = cache_.find(base_key);
    std::unordered_map<std::string, ov::genai::modeling::Tensor> empty_auxiliary;
    if (it_cached != cache_.end()) {
        return ov::genai::modeling::weights::FinalizedWeight(
            ov::genai::modeling::Tensor(it_cached->second, &ctx),
            empty_auxiliary);
    }

    const auto qtype = resolve_qtype(base_key);
    auto node = make_weights_subgraph(base_key, consts_, qtype, /*reorder=*/false, /*head_size=*/-1);
    cache_.emplace(base_key, node);
    return ov::genai::modeling::weights::FinalizedWeight(
        ov::genai::modeling::Tensor(node, &ctx),
        empty_auxiliary);
}

}  // namespace gguf
}  // namespace genai
}  // namespace ov
