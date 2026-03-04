// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <openvino/op/util/variable.hpp>
#include <openvino/opsets/opset13.hpp>
#include <openvino/openvino.hpp>

#include "utils.hpp"

namespace {

std::shared_ptr<ov::Model> build_stateful_axes_model() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1});

    auto linear_init = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 4, 0}, std::vector<float>{});
    ov::op::util::VariableInfo linear_info{ov::PartialShape{-1, 4, 0}, ov::element::f32, "linear_states.0.conv"};
    auto linear_var = std::make_shared<ov::op::util::Variable>(linear_info);
    auto linear_read = std::make_shared<ov::op::v6::ReadValue>(linear_init, linear_var);
    auto linear_assign = std::make_shared<ov::opset13::Assign>(linear_read->output(0), linear_var);

    auto kv_init = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 0, 1, 1}, std::vector<float>{});
    ov::op::util::VariableInfo kv_info{ov::PartialShape{-1, 0, 1, 1}, ov::element::f32, "past_key_values.0.key_cache"};
    auto kv_var = std::make_shared<ov::op::util::Variable>(kv_info);
    auto kv_read = std::make_shared<ov::op::v6::ReadValue>(kv_init, kv_var);
    auto kv_assign = std::make_shared<ov::opset13::Assign>(kv_read->output(0), kv_var);

    auto axis_3d = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 1, 2});
    auto axis_4d = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 2, 3});
    auto linear_sum = std::make_shared<ov::opset13::ReduceSum>(linear_read->output(0), axis_3d->output(0), false);
    auto kv_sum = std::make_shared<ov::opset13::ReduceSum>(kv_read->output(0), axis_4d->output(0), false);
    auto state_sum = std::make_shared<ov::opset13::Add>(linear_sum->output(0), kv_sum->output(0));
    auto state_sum_2d = std::make_shared<ov::opset13::Reshape>(
        state_sum->output(0),
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 1}),
        false);
    auto out = std::make_shared<ov::opset13::Add>(input->output(0), state_sum_2d->output(0));
    auto result = std::make_shared<ov::op::v0::Result>(out->output(0));

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
    model->add_sinks({linear_assign, kv_assign});
    return model;
}

std::shared_ptr<ov::Model> build_stateful_trim_model() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1});

    auto linear_init = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 4, 0}, std::vector<float>{});
    ov::op::util::VariableInfo linear_info{ov::PartialShape{-1, 4, -1}, ov::element::f32, "linear_states.0.conv"};
    auto linear_var = std::make_shared<ov::op::util::Variable>(linear_info);
    auto linear_read = std::make_shared<ov::op::v6::ReadValue>(linear_init, linear_var);
    auto linear_assign = std::make_shared<ov::opset13::Assign>(linear_read->output(0), linear_var);

    auto kv_init = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 0, 1, 1}, std::vector<float>{});
    ov::op::util::VariableInfo kv_info{ov::PartialShape{-1, -1, 1, 1}, ov::element::f32, "past_key_values.0.key_cache"};
    auto kv_var = std::make_shared<ov::op::util::Variable>(kv_info);
    auto kv_read = std::make_shared<ov::op::v6::ReadValue>(kv_init, kv_var);
    auto kv_assign = std::make_shared<ov::opset13::Assign>(kv_read->output(0), kv_var);

    auto axis_3d = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 1, 2});
    auto axis_4d = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 2, 3});
    auto linear_sum = std::make_shared<ov::opset13::ReduceSum>(linear_read->output(0), axis_3d->output(0), false);
    auto kv_sum = std::make_shared<ov::opset13::ReduceSum>(kv_read->output(0), axis_4d->output(0), false);
    auto state_sum = std::make_shared<ov::opset13::Add>(linear_sum->output(0), kv_sum->output(0));
    auto state_sum_2d = std::make_shared<ov::opset13::Reshape>(
        state_sum->output(0),
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 1}),
        false);
    auto out = std::make_shared<ov::opset13::Add>(input->output(0), state_sum_2d->output(0));
    auto result = std::make_shared<ov::op::v0::Result>(out->output(0));

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
    model->add_sinks({linear_assign, kv_assign});
    return model;
}

}  // namespace

TEST(Qwen3NextCacheRuntime, KVAxesDetectionSkipsLinearStates) {
    auto model = build_stateful_axes_model();
    auto kv_pos = ov::genai::utils::get_kv_axes_pos(model);

    EXPECT_EQ(kv_pos.batch, 0u);
    EXPECT_EQ(kv_pos.seq_len, 1u);
}

TEST(Qwen3NextCacheRuntime, TrimOnlyAffectsAttentionStates) {
    auto model = build_stateful_trim_model();
    ov::Core core;
    auto compiled = core.compile_model(model, "GPU");
    auto request = compiled.create_infer_request();

    auto states = request.query_state();
    ASSERT_EQ(states.size(), 2u);
    for (auto& state : states) {
        if (state.get_name().find("past_key_values.") != std::string::npos) {
            ov::Tensor kv_state(ov::element::f32, {1, 6, 1, 1});
            state.set_state(kv_state);
        } else if (state.get_name().find("linear_states.") != std::string::npos) {
            ov::Tensor linear_state(ov::element::f32, {1, 4, 5});
            state.set_state(linear_state);
        }
    }

    ov::genai::utils::KVCacheState kv_cache_state;
    kv_cache_state.seq_length_axis = 1;
    kv_cache_state.num_tokens_to_trim = 2;
    ov::genai::utils::trim_kv_cache(request, kv_cache_state, std::nullopt);

    ov::Shape kv_shape;
    ov::Shape linear_shape;
    states = request.query_state();
    for (auto& state : states) {
        if (state.get_name().find("past_key_values.") != std::string::npos) {
            kv_shape = state.get_state().get_shape();
        } else if (state.get_name().find("linear_states.") != std::string::npos) {
            linear_shape = state.get_state().get_shape();
        }
    }

    ASSERT_EQ(kv_shape.size(), 4u);
    ASSERT_EQ(linear_shape.size(), 3u);
    EXPECT_EQ(kv_shape[1], 4u);
    EXPECT_EQ(linear_shape[2], 5u);
}
