# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import openvino_genai as ov_genai
import pytest
import soundfile as sf


@dataclass(frozen=True)
class Qwen3ASRTestAssets:
    model_dir: Path
    audio_path: Path
    audio_model_dir: Path | None
    device: str
    context: str
    language: str | None
    max_new_tokens: int


def _normalize_audio(audio_path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1).astype(np.float32, copy=False)
    if sample_rate != 16000:
        audio = librosa.resample(audio, orig_sr=int(sample_rate), target_sr=16000).astype(np.float32)

    if audio.size:
        peak = float(np.max(np.abs(audio)))
        if peak > 1.0:
            audio = audio / peak
        audio = np.clip(audio, -1.0, 1.0)
    return audio


def _read_required_path(var_name: str) -> Path:
    raw = os.environ.get(var_name, "").strip()
    if not raw:
        pytest.skip(f"{var_name} is not set")

    path = Path(raw)
    if not path.exists():
        pytest.skip(f"{var_name} does not exist: {path}")
    return path


@pytest.fixture(scope="module")
def qwen3_asr_assets() -> Qwen3ASRTestAssets:
    model_dir = _read_required_path("QWEN3_ASR_TEST_MODEL_DIR")
    audio_path = _read_required_path("QWEN3_ASR_TEST_AUDIO")

    audio_model_raw = os.environ.get("QWEN3_ASR_TEST_AUDIO_MODEL_DIR", "").strip()
    audio_model_dir = None
    if audio_model_raw:
        audio_model_dir = Path(audio_model_raw)
        if not audio_model_dir.exists():
            pytest.skip(f"QWEN3_ASR_TEST_AUDIO_MODEL_DIR does not exist: {audio_model_dir}")

    return Qwen3ASRTestAssets(
        model_dir=model_dir,
        audio_path=audio_path,
        audio_model_dir=audio_model_dir,
        device=os.environ.get("QWEN3_ASR_TEST_DEVICE", "GPU").strip() or "GPU",
        context=os.environ.get("QWEN3_ASR_TEST_CONTEXT", ""),
        language=os.environ.get("QWEN3_ASR_TEST_LANGUAGE", "").strip() or None,
        max_new_tokens=int(os.environ.get("QWEN3_ASR_TEST_MAX_NEW_TOKENS", "256")),
    )


@pytest.fixture(scope="module")
def qwen3_asr_audio(qwen3_asr_assets: Qwen3ASRTestAssets):
    return _normalize_audio(qwen3_asr_assets.audio_path)


@pytest.fixture(scope="module")
def qwen3_asr_engine(qwen3_asr_assets: Qwen3ASRTestAssets):
    return ov_genai.Qwen3ASRInferenceEngine(
        qwen3_asr_assets.model_dir,
        device=qwen3_asr_assets.device,
        max_new_tokens=qwen3_asr_assets.max_new_tokens,
        audio_model_path=qwen3_asr_assets.audio_model_dir,
    )


def _assert_result_shape(result: ov_genai.Qwen3ASRDecodedResult) -> None:
    assert isinstance(result.raw_text, str)
    assert isinstance(result.text, str)
    assert isinstance(result.language, str)
    assert result.prompt_token_size > 0
    assert result.generated_tokens >= 0
    assert result.audio_output_length >= 0
    assert result.feature_extract_ms >= 0.0
    assert result.audio_encode_ms >= 0.0
    assert result.ttft_ms >= 0.0
    assert result.decode_ms >= 0.0
    assert result.decode_steps >= 0
    assert result.infer_steps >= 0
    assert result.stream_steps >= 0
    assert result.infer_ms >= 0.0
    assert result.audio_duration_s > 0.0
    assert result.asr_rtf >= 0.0


@pytest.mark.real_models
def test_qwen3_asr_engine_generate_smoke(
    qwen3_asr_engine,
    qwen3_asr_assets: Qwen3ASRTestAssets,
    qwen3_asr_audio,
):
    result = qwen3_asr_engine.generate(
        qwen3_asr_audio,
        context=qwen3_asr_assets.context,
        language=qwen3_asr_assets.language,
    )

    _assert_result_shape(result)


@pytest.mark.real_models
def test_qwen3_asr_engine_generate_streaming_smoke(
    qwen3_asr_engine,
    qwen3_asr_assets: Qwen3ASRTestAssets,
    qwen3_asr_audio,
):
    result = qwen3_asr_engine.generate_streaming(
        qwen3_asr_audio,
        context=qwen3_asr_assets.context,
        language=qwen3_asr_assets.language,
        stream_chunk_sec=0.5,
        stream_window_sec=8.0,
        stream_unfixed_chunk_num=2,
        stream_unfixed_token_num=5,
    )

    _assert_result_shape(result)
    assert result.stream_steps >= 1


@pytest.mark.real_models
def test_qwen3_asr_streaming_matches_offline_language(
    qwen3_asr_engine,
    qwen3_asr_assets: Qwen3ASRTestAssets,
    qwen3_asr_audio,
):
    offline = qwen3_asr_engine.generate(
        qwen3_asr_audio,
        context=qwen3_asr_assets.context,
        language=qwen3_asr_assets.language,
    )

    streaming = qwen3_asr_engine.generate_streaming(
        qwen3_asr_audio,
        context=qwen3_asr_assets.context,
        language=qwen3_asr_assets.language,
        stream_chunk_sec=0.5,
        stream_window_sec=8.0,
        stream_unfixed_chunk_num=2,
        stream_unfixed_token_num=5,
    )

    assert streaming.language == offline.language
    assert isinstance(streaming.text, str)
    assert isinstance(offline.text, str)