// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "loaders/gguf/gguf_loader.hpp"

#include <filesystem>
#include <stdexcept>

#include "gguf_utils/gguf.hpp"
#include "gguf_utils/gguf_tokenizer.hpp"
#include "gguf_utils/gguf_weight_source.hpp"
#include "gguf_utils/gguf_weight_finalizer.hpp"
#include "loaders/loader_registry.hpp"

namespace ov {
namespace genai {

// Use the modeling weights types
using modeling::weights::WeightSource;
using modeling::weights::WeightFinalizer;

namespace loaders {

bool GGUFLoader::supports(const std::string& path) const {
    std::filesystem::path p(path);
    return p.extension() == ".gguf";
}

void GGUFLoader::ensure_loaded(const std::string& path) const {
    if (cached_path_ == path && !cached_metadata_.empty()) {
        return;  // Already loaded
    }

    // Load GGUF file using existing utility
    auto [metadata, tensors, qtypes] = load_gguf(path);
    
    cached_path_ = path;
    cached_metadata_ = std::move(metadata);
    cached_tensors_ = std::move(tensors);
    cached_qtypes_ = std::move(qtypes);
}

ModelConfig GGUFLoader::load_config(const std::string& path) const {
    ensure_loaded(path);
    
    // cached_metadata_ is already std::map, pass it directly
    auto config = ModelConfig::from_gguf(cached_metadata_);
    
    // Detect tie_word_embeddings: if output.weight doesn't exist in tensors,
    // it means lm_head shares weights with token embeddings
    config.tie_word_embeddings = (cached_tensors_.find("output.weight") == cached_tensors_.end());
    
    return config;
}

std::shared_ptr<WeightSource> GGUFLoader::create_weight_source(
    const std::string& path) const {
    ensure_loaded(path);
    
    // Create a shared copy of tensors for the weight source
    auto tensors_copy = std::make_shared<std::unordered_map<std::string, ov::Tensor>>(cached_tensors_);
    
    // GGUFWeightSource takes a reference, so we need to ensure the tensors outlive the source
    // We create a custom shared_ptr that holds both
    class GGUFWeightSourceWithData : public ov::genai::gguf::GGUFWeightSource {
    public:
        GGUFWeightSourceWithData(std::shared_ptr<std::unordered_map<std::string, ov::Tensor>> tensors)
            : ov::genai::gguf::GGUFWeightSource(*tensors), tensors_(std::move(tensors)) {}
    private:
        std::shared_ptr<std::unordered_map<std::string, ov::Tensor>> tensors_;
    };
    
    return std::make_shared<GGUFWeightSourceWithData>(std::move(tensors_copy));
}

std::shared_ptr<WeightFinalizer> GGUFLoader::create_weight_finalizer(
    const ModelConfig& config) const {
    if (cached_path_.empty()) {
        throw std::runtime_error("GGUFLoader: load_config() must be called before create_weight_finalizer()");
    }
    
    // Create shared copies for the finalizer
    auto tensors_copy = std::make_shared<std::unordered_map<std::string, ov::Tensor>>(cached_tensors_);
    auto qtypes_copy = std::make_shared<std::unordered_map<std::string, gguf_tensor_type>>(cached_qtypes_);
    
    // GGUFWeightFinalizer takes references, so we need to ensure the data outlives the finalizer
    class GGUFWeightFinalizerWithData : public ov::genai::gguf::GGUFWeightFinalizer {
    public:
        GGUFWeightFinalizerWithData(
            std::shared_ptr<std::unordered_map<std::string, ov::Tensor>> tensors,
            std::shared_ptr<std::unordered_map<std::string, gguf_tensor_type>> qtypes)
            : ov::genai::gguf::GGUFWeightFinalizer(*tensors, *qtypes),
              tensors_(std::move(tensors)),
              qtypes_(std::move(qtypes)) {}
    private:
        std::shared_ptr<std::unordered_map<std::string, ov::Tensor>> tensors_;
        std::shared_ptr<std::unordered_map<std::string, gguf_tensor_type>> qtypes_;
    };
    
    return std::make_shared<GGUFWeightFinalizerWithData>(
        std::move(tensors_copy), std::move(qtypes_copy));
}

TokenizerPair GGUFLoader::load_tokenizer(const std::string& path) const {
    // NOTE: This function is currently not used in the actual workflow.
    // Tokenizer loading is handled by the Tokenizer class (tokenizer_impl.cpp)
    // which independently loads/converts tokenizers based on model directory.
    // This interface exists for potential future unified loader integration.
    std::cout << "[GGUFLoader::load_tokenizer] Called but NOT used - "
              << "Tokenizer class handles loading independently. Path: " << path << std::endl;
    return TokenizerPair{nullptr, nullptr};
}

// Register the GGUF loader with the registry
namespace {
static bool registered = []() {
    LoaderRegistry::instance().register_loader(
        ModelFormat::GGUF,
        []() { return std::make_unique<GGUFLoader>(); }
    );
    return true;
}();
}  // namespace

}  // namespace loaders
}  // namespace genai
}  // namespace ov
