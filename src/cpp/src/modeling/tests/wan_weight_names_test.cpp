// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/models/wan/processing_wan.hpp"

#include <gtest/gtest.h>

#include "modeling/builder_context.hpp"
#include "modeling/tests/test_utils.hpp"
#include "modeling/weights/weight_loader.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace tests {

namespace {

class WanDummyTransformer : public ov::genai::modeling::Module {
public:
    explicit WanDummyTransformer(ov::genai::modeling::BuilderContext& ctx) : Module("", ctx) {
        register_parameter("patch_embedding.weight");
        register_parameter("blocks.0.attn1.to_q.weight");
    }
};

class WanDummyVAE : public ov::genai::modeling::Module {
public:
    explicit WanDummyVAE(ov::genai::modeling::BuilderContext& ctx) : Module("", ctx) {
        register_parameter("encoder.conv_in.weight");
        register_parameter("decoder.conv_out.bias");
    }
};

}  // namespace

TEST(WanWeightNamesTest, TransformerPackedMapping) {
    ov::genai::modeling::BuilderContext ctx;
    WanDummyTransformer model(ctx);
    ov::genai::modeling::models::WanWeightMapping::apply_transformer_packed_mapping(model);

    DummyWeightSource source;
    source.add("transformer.patch_embedding.weight", make_tensor({1.0f}, {1}));
    source.add("model.blocks.0.attn1.to_q.weight", make_tensor({2.0f}, {1}));
    DummyWeightFinalizer finalizer;

    auto report = ov::genai::modeling::weights::load_model(
        model, source, finalizer, ov::genai::modeling::weights::LoadOptions::strict());

    EXPECT_EQ(report.matched.size(), 2u);
    EXPECT_TRUE(model.get_parameter("patch_embedding.weight").is_bound());
    EXPECT_TRUE(model.get_parameter("blocks.0.attn1.to_q.weight").is_bound());
}

TEST(WanWeightNamesTest, VAEPackedMapping) {
    ov::genai::modeling::BuilderContext ctx;
    WanDummyVAE model(ctx);
    ov::genai::modeling::models::WanWeightMapping::apply_vae_packed_mapping(model);

    DummyWeightSource source;
    source.add("vae.encoder.conv_in.weight", make_tensor({1.0f}, {1}));
    source.add("model.decoder.conv_out.bias", make_tensor({2.0f}, {1}));
    DummyWeightFinalizer finalizer;

    auto report = ov::genai::modeling::weights::load_model(
        model, source, finalizer, ov::genai::modeling::weights::LoadOptions::strict());

    EXPECT_EQ(report.matched.size(), 2u);
    EXPECT_TRUE(model.get_parameter("encoder.conv_in.weight").is_bound());
    EXPECT_TRUE(model.get_parameter("decoder.conv_out.bias").is_bound());
}

}  // namespace tests
}  // namespace modeling
}  // namespace genai
}  // namespace ov
