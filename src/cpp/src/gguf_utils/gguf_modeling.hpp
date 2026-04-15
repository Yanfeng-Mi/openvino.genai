// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstring>

#include "openvino/openvino.hpp"

/**
 * @deprecated Use loaders::LoaderRegistry and loaders::ModelBuilder instead.
 * 
 * Migration example:
 * @code
 * auto& registry = loaders::LoaderRegistry::instance();
 * auto loader = registry.get_loader_for_path(model_path);
 * auto config = loader->load_config(model_path);
 * auto source = loader->create_weight_source(model_path);
 * auto finalizer = loader->create_weight_finalizer(config);
 * auto model = loaders::ModelBuilder::instance().build(config, *source, *finalizer);
 * @endcode
 */
[[deprecated("Use loaders::LoaderRegistry and loaders::ModelBuilder instead")]]
std::shared_ptr<ov::Model> create_from_gguf(const std::string& model_path, const bool enable_save_ov_model);
