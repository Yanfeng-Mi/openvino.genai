#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import openvino as ov


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare WAN DIT full model inference vs layered IR models.\n"
            "Requires IRs dumped by modeling_wan_layered_dit (wan_dit_full.xml and layered IRs)."
        )
    )
    parser.add_argument("--ir-dir", type=Path, default=Path("."),
                        help="Directory containing dumped IR XML/BIN files (default: current dir)")
    parser.add_argument("--layers-per-group", type=int, default=1,
                        help="Layers per group used for layered IR names (default: 1)")
    parser.add_argument("--full-xml", type=Path, default=None,
                        help="Path to monolithic IR XML (default: <ir-dir>/wan_dit_full.xml)")
    parser.add_argument("--device", type=str, default="CPU", help="OpenVINO device (default: CPU)")
    parser.add_argument("--batch", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--frames", type=int, default=5, help="Latent frames (default: 5)")
    parser.add_argument("--height", type=int, default=30, help="Latent height (default: 30)")
    parser.add_argument("--width", type=int, default=52, help="Latent width (default: 52)")
    parser.add_argument("--text-seq", type=int, default=64, help="Text sequence length (default: 64)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--in-channels", type=int, default=None,
                        help="Override in_channels if model input is dynamic")
    parser.add_argument("--text-dim", type=int, default=None,
                        help="Override text_dim if model input is dynamic")
    parser.add_argument("--image-seq", type=int, default=None,
                        help="Image sequence length if encoder_hidden_states_image is present")
    parser.add_argument("--image-dim", type=int, default=None,
                        help="Override image_dim if encoder_hidden_states_image is dynamic")
    parser.add_argument("--abs-threshold", type=float, default=1e-4,
                        help="Max abs diff threshold (default: 1e-4)")
    parser.add_argument("--rel-threshold", type=float, default=1e-3,
                        help="Max rel diff threshold (default: 1e-3)")
    return parser.parse_args()


def _get_static_dim(shape: ov.PartialShape, idx: int) -> int | None:
    if shape.rank.is_dynamic:
        return None
    dim = shape[idx]
    if dim.is_dynamic:
        return None
    return int(dim.get_length())


def _load_layered_paths(ir_dir: Path, layers_per_group: int) -> tuple[Path, Path, list[Path]]:
    prefix = f"wan_dit_layered_lpg{layers_per_group}"
    preprocess = ir_dir / f"{prefix}_preprocess.xml"
    postprocess = ir_dir / f"{prefix}_postprocess.xml"
    block_group_files = list(ir_dir.glob(f"{prefix}_block_group_*_l*_n*.xml"))
    if not block_group_files:
        raise FileNotFoundError(f"No block group IRs found with prefix {prefix} in {ir_dir}")

    def _group_index(path: Path) -> int:
        match = re.search(r"_block_group_(\d+)_l", path.name)
        if not match:
            return 1_000_000
        return int(match.group(1))

    block_group_files.sort(key=_group_index)
    return preprocess, postprocess, block_group_files


def _tensor_from_rng(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.normal(loc=0.0, scale=1.0, size=shape).astype(np.float32)


def _compare_outputs(a: np.ndarray, b: np.ndarray, rel_threshold: float) -> dict:
    if a.shape != b.shape:
        raise RuntimeError(f"Output shapes differ: {a.shape} vs {b.shape}")
    abs_diff = np.abs(a - b)
    max_abs = float(abs_diff.max(initial=0.0))
    mean_abs = float(abs_diff.mean() if abs_diff.size else 0.0)
    max_val = np.maximum(np.abs(a), np.abs(b))
    rel_diff = np.where(max_val > 1e-8, abs_diff / max_val, 0.0)
    max_rel = float(rel_diff.max(initial=0.0))
    mean_rel = float(rel_diff.mean() if rel_diff.size else 0.0)
    mismatches = int((rel_diff > rel_threshold).sum())
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_rel": max_rel,
        "mean_rel": mean_rel,
        "mismatches": mismatches,
        "num_elements": int(abs_diff.size),
    }


def _to_ov_tensor(value: object) -> ov.Tensor:
    if isinstance(value, ov.Tensor):
        return value
    return ov.Tensor(value)


def _to_numpy(value: object) -> np.ndarray:
    if isinstance(value, ov.Tensor):
        return np.asarray(value.data)
    return np.asarray(value)


def main() -> int:
    args = parse_args()
    ir_dir = args.ir_dir

    full_xml = args.full_xml if args.full_xml is not None else (ir_dir / "wan_dit_full.xml")
    if not full_xml.exists():
        raise FileNotFoundError(f"Monolithic IR not found: {full_xml}")

    preprocess_xml, postprocess_xml, block_group_xmls = _load_layered_paths(
        ir_dir, args.layers_per_group
    )
    for path in [preprocess_xml, postprocess_xml, *block_group_xmls]:
        if not path.exists():
            raise FileNotFoundError(f"Missing IR file: {path}")

    core = ov.Core()
    full_model = core.read_model(str(full_xml))

    hidden_input = full_model.input("hidden_states")
    text_input = full_model.input("encoder_hidden_states")

    in_channels = _get_static_dim(hidden_input.get_partial_shape(), 1) or args.in_channels
    text_dim = _get_static_dim(text_input.get_partial_shape(), 2) or args.text_dim
    if in_channels is None:
        raise RuntimeError("in_channels is dynamic; please pass --in-channels")
    if text_dim is None:
        raise RuntimeError("text_dim is dynamic; please pass --text-dim")

    batch = args.batch
    frames = args.frames
    height = args.height
    width = args.width
    text_seq = args.text_seq

    rng = np.random.default_rng(args.seed)
    hidden_states = _tensor_from_rng(rng, (batch, in_channels, frames, height, width))
    timestep = np.full((batch,), 0.5, dtype=np.float32)
    text_embeds = _tensor_from_rng(rng, (batch, text_seq, text_dim))

    inputs = {
        "hidden_states": hidden_states,
        "timestep": timestep,
        "encoder_hidden_states": text_embeds,
    }

    if "encoder_hidden_states_image" in {inp.get_any_name() for inp in full_model.inputs}:
        image_input = full_model.input("encoder_hidden_states_image")
        image_dim = _get_static_dim(image_input.get_partial_shape(), 2) or args.image_dim
        if image_dim is None:
            raise RuntimeError("image_dim is dynamic; please pass --image-dim")
        image_seq = args.image_seq or text_seq
        inputs["encoder_hidden_states_image"] = _tensor_from_rng(
            rng, (batch, image_seq, image_dim)
        )

    compiled_full = core.compile_model(full_model, args.device)
    full_request = compiled_full.create_infer_request()
    for name, value in inputs.items():
        full_request.set_tensor(name, _to_ov_tensor(value))
    full_request.infer()
    full_output = _to_numpy(full_request.get_output_tensor(0))

    preprocess = core.compile_model(str(preprocess_xml), args.device)
    preprocess_request = preprocess.create_infer_request()
    for name, value in inputs.items():
        if name in {inp.get_any_name() for inp in preprocess.inputs}:
            preprocess_request.set_tensor(name, _to_ov_tensor(value))
    preprocess_request.infer()

    tokens = preprocess_request.get_tensor("tokens")
    rotary_cos = preprocess_request.get_tensor("rotary_cos")
    rotary_sin = preprocess_request.get_tensor("rotary_sin")
    temb = preprocess_request.get_tensor("temb")
    timestep_proj = preprocess_request.get_tensor("timestep_proj")
    text_proj = preprocess_request.get_tensor("text_embeds")
    ppf = preprocess_request.get_tensor("ppf")
    pph = preprocess_request.get_tensor("pph")
    ppw = preprocess_request.get_tensor("ppw")

    current = tokens
    for block_xml in block_group_xmls:
        compiled_block = core.compile_model(str(block_xml), args.device)
        block_request = compiled_block.create_infer_request()
        block_request.set_tensor("hidden_states", current)
        block_request.set_tensor("text_embeds", text_proj)
        block_request.set_tensor("timestep_proj", timestep_proj)
        block_request.set_tensor("rotary_cos", rotary_cos)
        block_request.set_tensor("rotary_sin", rotary_sin)
        block_request.infer()
        current = block_request.get_tensor("hidden_states_out")

    postprocess = core.compile_model(str(postprocess_xml), args.device)
    post_request = postprocess.create_infer_request()
    post_request.set_tensor("hidden_states", current)
    post_request.set_tensor("temb", temb)
    post_request.set_tensor("ppf", ppf)
    post_request.set_tensor("pph", pph)
    post_request.set_tensor("ppw", ppw)
    post_request.infer()
    layered_output = _to_numpy(post_request.get_tensor("sample"))

    stats = _compare_outputs(full_output, layered_output, args.rel_threshold)
    print("Output comparison:")
    print(f"  Elements: {stats['num_elements']}")
    print(f"  Max absolute diff: {stats['max_abs']:.6e}")
    print(f"  Mean absolute diff: {stats['mean_abs']:.6e}")
    print(f"  Max relative diff: {stats['max_rel']:.6e}")
    print(f"  Mean relative diff: {stats['mean_rel']:.6e}")
    print(f"  Mismatches (>{args.rel_threshold}): {stats['mismatches']}")

    passed = (stats["max_abs"] < args.abs_threshold) or (
        stats["max_rel"] < args.rel_threshold and stats["mismatches"] == 0
    )
    print("RESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
