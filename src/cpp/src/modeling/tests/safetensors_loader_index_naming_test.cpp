// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "safetensors_utils/safetensors_loader.hpp"

namespace {

namespace fs = std::filesystem;

class TempDirGuard {
public:
    explicit TempDirGuard(const std::string& suffix) {
        dir_path = fs::temp_directory_path() / fs::path("ov_genai_safetensors_loader_" + suffix);
        fs::remove_all(dir_path);
        fs::create_directories(dir_path);
    }

    ~TempDirGuard() {
        std::error_code ec;
        fs::remove_all(dir_path, ec);
    }

    fs::path path() const {
        return dir_path;
    }

private:
    fs::path dir_path;
};

void write_text_file(const fs::path& path, const std::string& content) {
    std::ofstream out(path, std::ios::binary);
    ASSERT_TRUE(out.is_open()) << "Failed to open file: " << path.string();
    out << content;
    out.close();
}

void touch_empty_file(const fs::path& path) {
    std::ofstream out(path, std::ios::binary);
    ASSERT_TRUE(out.is_open()) << "Failed to create file: " << path.string();
}

void expect_load_uses_shard_name_from_index(const fs::path& model_dir, const std::string& expected_shard_name) {
    try {
        (void)ov::genai::safetensors::load_safetensors(model_dir);
        FAIL() << "Expected load_safetensors to fail for empty shard file";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find(expected_shard_name), std::string::npos)
            << "Actual error: " << msg;
    }
}

}  // namespace

TEST(SafetensorsLoaderIndexNaming, SupportsModelShardNamingWithPrefixModelDash) {
    TempDirGuard temp_dir("model_dash");
    const std::string shard_name = "model-00001-of-00015.safetensors";

    touch_empty_file(temp_dir.path() / shard_name);
    write_text_file(
        temp_dir.path() / "model.safetensors.index.json",
        "{\n"
        "  \"metadata\": {\"total_size\": 1},\n"
        "  \"weight_map\": {\n"
        "    \"model.embed_tokens.weight\": \"" + shard_name + "\"\n"
        "  }\n"
        "}\n");

    expect_load_uses_shard_name_from_index(temp_dir.path(), shard_name);
}

TEST(SafetensorsLoaderIndexNaming, SupportsModelShardNamingWithEmbeddedSafetensorsSuffix) {
    TempDirGuard temp_dir("embedded_safetensors_suffix");
    const std::string shard_name = "model.safetensors-00001-of-00014.safetensors";

    touch_empty_file(temp_dir.path() / shard_name);
    write_text_file(
        temp_dir.path() / "model.safetensors.index.json",
        "{\n"
        "  \"metadata\": {\"total_size\": 1},\n"
        "  \"weight_map\": {\n"
        "    \"model.embed_tokens.weight\": \"" + shard_name + "\"\n"
        "  }\n"
        "}\n");

    expect_load_uses_shard_name_from_index(temp_dir.path(), shard_name);
}

