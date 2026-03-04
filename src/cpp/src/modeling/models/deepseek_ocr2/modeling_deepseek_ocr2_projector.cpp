// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/deepseek_ocr2/modeling_deepseek_ocr2_projector.hpp"

#include <openvino/openvino.hpp>
#include <openvino/core/except.hpp>

#include "modeling/ops/ops.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace {

ov::genai::modeling::Tensor add_bias_if_present(const ov::genai::modeling::Tensor& x,
                                                const ov::genai::modeling::Tensor* bias) {
    if (!bias) {
        return x;
    }
    return x + *bias;
}

auto set_name = [](const std::shared_ptr<ov::Node>& node, const std::string& name) {
    node->output(0).set_names({name});
    node->set_friendly_name(name);
};

}  // namespace

namespace ov {
namespace genai {
namespace modeling {
namespace models {

DeepseekOCR2Projector::DeepseekOCR2Projector(BuilderContext& ctx,
                                             const std::string& name,
                                             const DeepseekOCR2ProjectorConfig& cfg,
                                             Module* parent)
    : Module(name, ctx, parent),
      cfg_(cfg) {
    if (cfg_.projector_type.empty()) {
        cfg_.projector_type = "linear";
    }
    weight_param_ = &register_parameter("layers.weight");
    bias_param_ = &register_parameter("layers.bias");
    if (cfg_.projector_type == "linear") {
        bias_param_->set_optional(true);
    }
}

const Tensor& DeepseekOCR2Projector::weight() const {
    if (!weight_param_) {
        OPENVINO_THROW("DeepseekOCR2Projector weight parameter not registered");
    }
    return weight_param_->value();
}

const Tensor* DeepseekOCR2Projector::bias() const {
    return (bias_param_ && bias_param_->is_bound()) ? &bias_param_->value() : nullptr;
}

Tensor DeepseekOCR2Projector::forward(const Tensor& hidden_states) const {
    if (cfg_.projector_type == "identity") {
        return hidden_states;
    }
    if (cfg_.projector_type != "linear") {
        OPENVINO_THROW("Unsupported projector_type: ", cfg_.projector_type);
    }
    return add_bias_if_present(ops::linear(hidden_states, weight()), bias());
}

std::shared_ptr<ov::Model> create_deepseek_ocr2_projector_model(
    const DeepseekOCR2ProjectorConfig& cfg,
    ov::genai::modeling::weights::WeightSource& source,
    ov::genai::modeling::weights::WeightFinalizer& finalizer) {
    if (cfg.input_dim <= 0 || cfg.n_embed <= 0) {
        OPENVINO_THROW("Invalid DeepseekOCR2 projector config");
    }
    BuilderContext ctx;
    DeepseekOCR2Projector model(ctx, "projector", cfg);
    model.packed_mapping().rules.push_back({DeepseekOCR2WeightNames::kProjectorPrefix, "projector.", 0});

    ov::genai::modeling::weights::LoadOptions options;
    options.allow_unmatched = true;
    options.allow_missing = false;
    options.report_missing = true;
    options.report_unmatched = true;
    auto report = ov::genai::modeling::weights::load_model(model, source, finalizer, options);
    (void)report;

    auto input = ctx.parameter(DeepseekOCR2ProjectorIO::kInput,
                               ov::element::f32,
                               ov::PartialShape{-1, -1, cfg.input_dim});
    auto output = model.forward(input);

    auto result = std::make_shared<ov::op::v0::Result>(output.output());
    set_name(result, DeepseekOCR2ProjectorIO::kOutput);
    return ctx.build_model({result->output(0)});
}

}  // namespace models
}  // namespace modeling
}  // namespace genai
}  // namespace ov
