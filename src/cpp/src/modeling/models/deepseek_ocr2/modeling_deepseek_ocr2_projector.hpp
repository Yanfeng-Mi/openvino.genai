// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>

namespace ov {
class Model;
}  // namespace ov

namespace ov {
namespace genai {
namespace modeling {
namespace weights {
class WeightFinalizer;
class WeightSource;
}  // namespace weights
}  // namespace modeling
}  // namespace genai
}  // namespace ov

#include "modeling/builder_context.hpp"
#include "modeling/models/deepseek_ocr2/processing_deepseek_ocr2.hpp"
#include "modeling/module.hpp"
#include "modeling/ops/tensor.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace models {

struct DeepseekOCR2ProjectorIO {
    static constexpr const char* kInput = "query_feats";
    static constexpr const char* kOutput = "visual_embeds";
};

class DeepseekOCR2Projector : public Module {
public:
    DeepseekOCR2Projector(BuilderContext& ctx,
                          const std::string& name,
                          const DeepseekOCR2ProjectorConfig& cfg,
                          Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states) const;

private:
    const Tensor& weight() const;
    const Tensor* bias() const;

    DeepseekOCR2ProjectorConfig cfg_;
    WeightParameter* weight_param_ = nullptr;
    WeightParameter* bias_param_ = nullptr;
};

std::shared_ptr<ov::Model> create_deepseek_ocr2_projector_model(
    const DeepseekOCR2ProjectorConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer);

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
