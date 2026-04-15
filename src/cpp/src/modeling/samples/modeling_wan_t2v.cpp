// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>
#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/float16.hpp>

#include "imwrite.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "safetensors_utils/safetensors_loader.hpp"
#include "safetensors_utils/safetensors_weight_finalizer.hpp"
#include "safetensors_utils/safetensors_weight_source.hpp"

#include "image_generation/schedulers/flow_match_euler_discrete.hpp"
#include "modeling/models/wan/modeling_wan_dit.hpp"
#include "modeling/models/wan/modeling_wan_vae.hpp"
#include "modeling/models/wan/processing_wan.hpp"
#include "modeling/models/wan/modeling_wan_umt5.hpp"

namespace {

std::vector<float> tensor_to_float_vector(const ov::Tensor& src) {
    const size_t count = src.get_size();
    std::vector<float> out(count, 0.0f);
    if (count == 0) {
        return out;
    }
    const auto type = src.get_element_type();
    if (type == ov::element::f32) {
        std::memcpy(out.data(), src.data<const float>(), count * sizeof(float));
        return out;
    }
    if (type == ov::element::f16) {
        const auto* data = src.data<const ov::float16>();
        for (size_t i = 0; i < count; ++i) {
            out[i] = static_cast<float>(data[i]);
        }
        return out;
    }
    if (type == ov::element::bf16) {
        const auto* data = src.data<const ov::bfloat16>();
        for (size_t i = 0; i < count; ++i) {
            out[i] = static_cast<float>(data[i]);
        }
        return out;
    }
    throw std::runtime_error("Unsupported tensor dtype for conversion");
}

std::vector<int64_t> tensor_to_int64_vector(const ov::Tensor& src) {
    const size_t count = src.get_size();
    std::vector<int64_t> out(count, 0);
    if (count == 0) {
        return out;
    }
    const auto type = src.get_element_type();
    if (type == ov::element::i64) {
        std::memcpy(out.data(), src.data<const int64_t>(), count * sizeof(int64_t));
        return out;
    }
    if (type == ov::element::i32) {
        const auto* data = src.data<const int32_t>();
        for (size_t i = 0; i < count; ++i) {
            out[i] = static_cast<int64_t>(data[i]);
        }
        return out;
    }
    if (type == ov::element::boolean) {
        const auto* data = src.data<const char>();
        for (size_t i = 0; i < count; ++i) {
            out[i] = data[i] ? 1 : 0;
        }
        return out;
    }
    throw std::runtime_error("Unsupported tensor dtype for int64 conversion");
}

ov::Tensor apply_attention_mask(const ov::Tensor& hidden_states, const ov::Tensor& attention_mask) {
    auto mask_vec = tensor_to_int64_vector(attention_mask);
    auto out_vec = tensor_to_float_vector(hidden_states);
    const auto shape = hidden_states.get_shape();
    if (shape.size() != 3) {
        throw std::runtime_error("Text encoder output must have shape [B, S, H]");
    }
    const size_t batch = shape[0];
    const size_t seq_len = shape[1];
    const size_t hidden = shape[2];
    if (mask_vec.size() != batch * seq_len) {
        throw std::runtime_error("attention_mask size mismatch with hidden_states");
    }
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            if (mask_vec[b * seq_len + s] == 0) {
                const size_t base = (b * seq_len + s) * hidden;
                std::fill_n(out_vec.data() + base, hidden, 0.0f);
            }
        }
    }
    ov::Tensor out(ov::element::f32, shape);
    std::memcpy(out.data(), out_vec.data(), out_vec.size() * sizeof(float));
    return out;
}

ov::Tensor repeat_prompt_embeds(const ov::Tensor& embeds, int32_t repeat) {
    if (repeat <= 1) {
        return embeds;
    }
    const auto shape = embeds.get_shape();
    if (shape.size() != 3) {
        throw std::runtime_error("prompt_embeds must have shape [B, S, H]");
    }
    const size_t batch = shape[0];
    const size_t seq_len = shape[1];
    const size_t hidden = shape[2];
    const size_t out_batch = static_cast<size_t>(repeat) * batch;

    auto data = tensor_to_float_vector(embeds);
    std::vector<float> out(out_batch * seq_len * hidden, 0.0f);
    for (size_t b = 0; b < batch; ++b) {
        const size_t src_base = b * seq_len * hidden;
        for (int32_t r = 0; r < repeat; ++r) {
            const size_t dst_base = (static_cast<size_t>(r) * batch + b) * seq_len * hidden;
            std::memcpy(out.data() + dst_base, data.data() + src_base, seq_len * hidden * sizeof(float));
        }
    }
    ov::Tensor out_tensor(ov::element::f32, {out_batch, seq_len, hidden});
    std::memcpy(out_tensor.data(), out.data(), out.size() * sizeof(float));
    return out_tensor;
}

ov::Tensor decode_to_u8_frames(const ov::Tensor& video) {
    const auto shape = video.get_shape();
    if (shape.size() != 4 && shape.size() != 5) {
        throw std::runtime_error("VAE output must have shape [B, C, H, W] or [B, C, F, H, W]");
    }
    const bool has_frames = shape.size() == 5;
    const size_t batch = shape[0];
    const size_t channels = shape[1];
    const size_t frames = has_frames ? shape[2] : 1;
    const size_t height = has_frames ? shape[3] : shape[2];
    const size_t width = has_frames ? shape[4] : shape[3];

    if (batch != 1 || channels != 3) {
        throw std::runtime_error("Sample supports batch=1 and channels=3 only");
    }

    auto data = tensor_to_float_vector(video);
    ov::Tensor out_u8(ov::element::u8, {frames, height, width, 3});
    auto* dst = out_u8.data<uint8_t>();

    const size_t stride_w = 1;
    const size_t stride_h = width;
    const size_t stride_f = height * width;
    const size_t stride_c = frames * stride_f;

    for (size_t f = 0; f < frames; ++f) {
        for (size_t h = 0; h < height; ++h) {
            for (size_t w = 0; w < width; ++w) {
                for (size_t c = 0; c < 3; ++c) {
                    const size_t src_idx = c * stride_c + f * stride_f + h * stride_h + w * stride_w;
                    float val = data[src_idx];
                    val = val / 2.0f + 0.5f;
                    val = std::min(std::max(val, 0.0f), 1.0f);
                    dst[((f * height + h) * width + w) * 3 + c] =
                        static_cast<uint8_t>(std::round(val * 255.0f));
                }
            }
        }
    }
    return out_u8;
}

}  // namespace

int main(int argc, char* argv[]) try {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <MODEL_DIR> <PROMPT> [OUTPUT_DIR] [DEVICE] [HEIGHT] [WIDTH] [FRAMES] [STEPS]"
                  << " [SEED] [GUIDANCE] [NEG_PROMPT] [MAX_SEQ_LEN]\n";
        return 1;
    }

    const std::filesystem::path model_dir = argv[1];
    const std::string prompt = argv[2];
    const std::filesystem::path output_dir = (argc > 3) ? argv[3] : "wan_t2v_out";
    const std::string device = (argc > 4) ? argv[4] : "CPU";
    int32_t height = (argc > 5) ? std::stoi(argv[5]) : 480;
    int32_t width = (argc > 6) ? std::stoi(argv[6]) : 832;
    int32_t num_frames = (argc > 7) ? std::stoi(argv[7]) : 81;
    const int32_t steps = (argc > 8) ? std::stoi(argv[8]) : 50;
    const int32_t seed = (argc > 9) ? std::stoi(argv[9]) : 0;
    const float guidance_scale = (argc > 10) ? std::stof(argv[10]) : 5.0f;
    const std::string negative_prompt = (argc > 11) ? argv[11] : "";
    const int32_t max_seq_len = (argc > 12) ? std::stoi(argv[12]) : 512;

    if (height % 16 != 0 || width % 16 != 0) {
        throw std::runtime_error("height and width must be divisible by 16");
    }

    const std::filesystem::path text_dir = model_dir / "text_encoder";
    const std::filesystem::path transformer_dir = model_dir / "transformer";
    const std::filesystem::path vae_dir = model_dir / "vae";
    const std::filesystem::path scheduler_dir = model_dir / "scheduler";
    const std::filesystem::path tokenizer_dir = model_dir / "tokenizer";

    auto text_cfg = ov::genai::modeling::models::UMT5Config::from_json_file(text_dir / "config.json");
    auto transformer_cfg = ov::genai::modeling::models::WanTransformer3DConfig::from_json_file(transformer_dir / "config.json");
    auto vae_cfg = ov::genai::modeling::models::WanVAEConfig::from_json_file(vae_dir / "config.json");
    ov::genai::FlowMatchEulerDiscreteScheduler::Config sched_cfg(scheduler_dir / "scheduler_config.json");

    if (transformer_cfg.text_dim != text_cfg.d_model) {
        std::cout << "[WanT2V] Warning: text_dim != text encoder hidden size\n";
    }
    if (transformer_cfg.in_channels != vae_cfg.z_dim) {
        std::cout << "[WanT2V] Warning: transformer in_channels != VAE z_dim\n";
    }

    const int32_t scale_t = vae_cfg.scale_factor_temporal;
    const int32_t scale_s = vae_cfg.scale_factor_spatial;
    if (num_frames % scale_t != 1) {
        std::cout << "[WanT2V] Warning: num_frames-1 is not divisible by " << scale_t << ", rounding.\n";
        num_frames = num_frames / scale_t * scale_t + 1;
    }
    const int32_t latent_frames = (num_frames - 1) / scale_t + 1;
    const int32_t latent_height = height / scale_s;
    const int32_t latent_width = width / scale_s;

    ov::genai::Tokenizer tokenizer(tokenizer_dir);

    auto text_data = ov::genai::safetensors::load_safetensors(text_dir);
    ov::genai::safetensors::SafetensorsWeightSource text_source(std::move(text_data));
    ov::genai::safetensors::SafetensorsWeightFinalizer text_finalizer;
    auto text_model = ov::genai::modeling::models::create_umt5_text_encoder_model(text_cfg, text_source, text_finalizer);

    auto transformer_data = ov::genai::safetensors::load_safetensors(transformer_dir);
    ov::genai::safetensors::SafetensorsWeightSource transformer_source(std::move(transformer_data));
    ov::genai::safetensors::SafetensorsWeightFinalizer transformer_finalizer;
    auto transformer_model = ov::genai::modeling::models::create_wan_dit_model(
        transformer_cfg, transformer_source, transformer_finalizer);

    auto vae_data = ov::genai::safetensors::load_safetensors(vae_dir);
    ov::genai::safetensors::SafetensorsWeightSource vae_source(std::move(vae_data));
    ov::genai::safetensors::SafetensorsWeightFinalizer vae_finalizer;
    auto vae_model = ov::genai::modeling::models::create_wan_vae_decoder_model(vae_cfg, vae_source, vae_finalizer);

    ov::Core core;
    ov::AnyMap compile_props;
    if (device.find("GPU") != std::string::npos) {
        compile_props[ov::hint::inference_precision.name()] = ov::element::f32;
    }

    auto compiled_text = compile_props.empty()
                             ? core.compile_model(text_model, device)
                             : core.compile_model(text_model, device, compile_props);
    auto compiled_transformer = compile_props.empty()
                                    ? core.compile_model(transformer_model, device)
                                    : core.compile_model(transformer_model, device, compile_props);
    auto compiled_vae = compile_props.empty()
                            ? core.compile_model(vae_model, device)
                            : core.compile_model(vae_model, device, compile_props);

    const bool do_cfg = guidance_scale > 1.0f;
    auto text_inputs = ov::genai::modeling::models::tokenize_prompts(
        tokenizer, {prompt}, max_seq_len, true);
    ov::genai::modeling::models::WanTextInputs neg_inputs;
    if (do_cfg) {
        neg_inputs = ov::genai::modeling::models::tokenize_prompts(
            tokenizer, {negative_prompt}, max_seq_len, true);
    }

    auto text_request = compiled_text.create_infer_request();
    text_request.set_tensor("input_ids", text_inputs.input_ids);
    text_request.set_tensor("attention_mask", text_inputs.attention_mask);
    text_request.infer();
    ov::Tensor prompt_embeds = apply_attention_mask(text_request.get_output_tensor(0),
                                                    text_inputs.attention_mask);

    ov::Tensor negative_embeds;
    if (do_cfg) {
        text_request.set_tensor("input_ids", neg_inputs.input_ids);
        text_request.set_tensor("attention_mask", neg_inputs.attention_mask);
        text_request.infer();
        negative_embeds = apply_attention_mask(text_request.get_output_tensor(0),
                                               neg_inputs.attention_mask);
    }

    prompt_embeds = repeat_prompt_embeds(prompt_embeds, 1);
    if (do_cfg) {
        negative_embeds = repeat_prompt_embeds(negative_embeds, 1);
    }

    const size_t batch = 1;
    ov::Tensor latents_tensor = ov::genai::modeling::models::prepare_latents(
        batch,
        static_cast<size_t>(transformer_cfg.in_channels),
        static_cast<size_t>(latent_frames),
        static_cast<size_t>(latent_height),
        static_cast<size_t>(latent_width),
        seed);

    ov::genai::FlowMatchEulerDiscreteScheduler scheduler(sched_cfg);
    scheduler.set_timesteps(static_cast<size_t>(steps), 1.0f);
    auto timesteps = scheduler.get_float_timesteps();

    auto transformer_request = compiled_transformer.create_infer_request();
    ov::Tensor timestep_tensor(ov::element::f32, {batch});

    for (size_t i = 0; i < timesteps.size(); ++i) {
        const float t = timesteps[i];
        timestep_tensor.data<float>()[0] = t;

        transformer_request.set_tensor("hidden_states", latents_tensor);
        transformer_request.set_tensor("timestep", timestep_tensor);
        transformer_request.set_tensor("encoder_hidden_states", prompt_embeds);
        transformer_request.infer();
        ov::Tensor noise_pred_tensor = transformer_request.get_output_tensor(0);
        auto noise_pred = tensor_to_float_vector(noise_pred_tensor);

        if (do_cfg) {
            transformer_request.set_tensor("encoder_hidden_states", negative_embeds);
            transformer_request.infer();
            ov::Tensor noise_uncond_tensor = transformer_request.get_output_tensor(0);
            auto noise_uncond = tensor_to_float_vector(noise_uncond_tensor);
            noise_pred = ov::genai::modeling::models::apply_cfg(noise_pred, noise_uncond, guidance_scale);
        }

        ov::Tensor noise_pred_f32(ov::element::f32, latents_tensor.get_shape());
        std::memcpy(noise_pred_f32.data(), noise_pred.data(), noise_pred.size() * sizeof(float));
        auto step_out = scheduler.step(noise_pred_f32, latents_tensor, i, nullptr);
        latents_tensor = step_out.at("latent");
    }

    std::vector<float> latents_out = tensor_to_float_vector(latents_tensor);
    if (!vae_cfg.latents_mean.empty() && !vae_cfg.latents_std.empty()) {
        if (vae_cfg.latents_mean.size() != static_cast<size_t>(vae_cfg.z_dim) ||
            vae_cfg.latents_std.size() != static_cast<size_t>(vae_cfg.z_dim)) {
            throw std::runtime_error("latents_mean/std size must match z_dim");
        }
        const size_t z_dim = static_cast<size_t>(vae_cfg.z_dim);
        const size_t spatial = static_cast<size_t>(latent_frames) *
                               static_cast<size_t>(latent_height) *
                               static_cast<size_t>(latent_width);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t c = 0; c < z_dim; ++c) {
                const float mean = vae_cfg.latents_mean[c];
                const float std = vae_cfg.latents_std[c];
                const size_t base = (b * z_dim + c) * spatial;
                for (size_t i = 0; i < spatial; ++i) {
                    latents_out[base + i] = latents_out[base + i] * std + mean;
                }
            }
        }
    }

    ov::Tensor vae_input(ov::element::f32,
                         {batch,
                          static_cast<size_t>(vae_cfg.z_dim),
                          static_cast<size_t>(latent_frames),
                          static_cast<size_t>(latent_height),
                          static_cast<size_t>(latent_width)});
    std::memcpy(vae_input.data(), latents_out.data(), latents_out.size() * sizeof(float));

    auto vae_request = compiled_vae.create_infer_request();
    vae_request.set_tensor("latents", vae_input);
    vae_request.infer();
    ov::Tensor video = vae_request.get_output_tensor(0);

    ov::Tensor frames_u8 = decode_to_u8_frames(video);
    std::filesystem::create_directories(output_dir);
    std::ostringstream pattern;
    pattern << (output_dir / "frame_%03d.bmp").string();
    imwrite(pattern.str(), frames_u8, false);

    std::cout << "Saved frames to " << output_dir.string() << std::endl;
    return 0;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {
    }
    return 1;
}
