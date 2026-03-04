// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <openvino/core/except.hpp>

#include "modeling/builder_context.hpp"
#include "modeling/module.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace tests {

class DummyModule : public Module {
public:
    explicit DummyModule(BuilderContext& ctx) : Module("dummy", ctx) {
        weight = &register_parameter("weight");
    }
    WeightParameter* weight = nullptr;
};

TEST(WeightLoaderOptions, ReportsMissingInLenientMode) {
    BuilderContext ctx;
    DummyModule module(ctx);
    DummyWeightSource source;
    DummyWeightFinalizer finalizer;

    weights::LoadOptions options = weights::LoadOptions::lenient();
    auto report = weights::load_model(module, source, finalizer, options);
    EXPECT_EQ(report.missing.size(), 1u);
    EXPECT_EQ(report.unmatched.size(), 0u);
}

TEST(WeightLoaderOptions, ThrowsOnMissingInStrictMode) {
    BuilderContext ctx;
    DummyModule module(ctx);
    DummyWeightSource source;
    DummyWeightFinalizer finalizer;

    weights::LoadOptions options = weights::LoadOptions::strict();
    EXPECT_THROW(weights::load_model(module, source, finalizer, options), ov::Exception);
}

TEST(WeightLoaderOptions, AllowsUnmatchedWeights) {
    BuilderContext ctx;
    DummyModule module(ctx);
    DummyWeightSource source;
    DummyWeightFinalizer finalizer;

    ov::Tensor dummy_weight(ov::element::f32, ov::Shape{1});
    source.add("unknown.weight", dummy_weight);

    weights::LoadOptions options = weights::LoadOptions::lenient();
    auto report = weights::load_model(module, source, finalizer, options);
    EXPECT_EQ(report.unmatched.size(), 1u);
}

}  // namespace tests
}  // namespace modeling
}  // namespace genai
}  // namespace ov
