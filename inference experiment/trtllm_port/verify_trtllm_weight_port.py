#!/usr/bin/env python3
"""Verify TensorRT-LLM checkpoint tensors against the source Fairseq checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from convert_indicxlit_to_trtllm import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_OUTPUT_DIR,
    decoder_weights,
    encoder_weights,
)
from inspect_indicxlit_checkpoint import layer_count  # noqa: E402


DEFAULT_REPORT = SCRIPT_DIR / "artifacts" / "trtllm_weight_port_verification.json"


def compare(expected: dict[str, torch.Tensor], actual_path: Path) -> dict[str, object]:
    actual = load_file(str(actual_path), device="cpu")
    expected_names = set(expected)
    actual_names = set(actual)
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    shape_mismatches = {}
    value_mismatches = {}
    max_abs_diff = 0.0

    for name in sorted(expected_names & actual_names):
        left = expected[name].detach().cpu().float()
        right = actual[name].detach().cpu().float()
        if tuple(left.shape) != tuple(right.shape):
            shape_mismatches[name] = {
                "expected": list(left.shape),
                "actual": list(right.shape),
            }
            continue
        diff = (left - right).abs()
        tensor_max = float(diff.max().item()) if diff.numel() else 0.0
        max_abs_diff = max(max_abs_diff, tensor_max)
        if tensor_max > 0.0:
            value_mismatches[name] = tensor_max

    return {
        "actual_path": str(actual_path),
        "expected_count": len(expected),
        "actual_count": len(actual),
        "missing": missing,
        "extra": extra,
        "shape_mismatches": shape_mismatches,
        "value_mismatch_count": len(value_mismatches),
        "value_mismatches": value_mismatches,
        "max_abs_diff": max_abs_diff,
        "exact": not missing and not extra and not shape_mismatches and not value_mismatches,
    }


def fairseq_semantics(checkpoint: dict) -> dict[str, object]:
    args = checkpoint.get("args")
    cfg_model = checkpoint["cfg"]["model"]
    hidden_size = int(getattr(cfg_model, "encoder_embed_dim"))
    return {
        "encoder_learned_pos": getattr(args, "encoder_learned_pos", None),
        "decoder_learned_pos": getattr(args, "decoder_learned_pos", None),
        "no_token_positional_embeddings": getattr(args, "no_token_positional_embeddings", None),
        "no_scale_embedding": getattr(args, "no_scale_embedding", None),
        "fairseq_embedding_scale": 1.0 if getattr(args, "no_scale_embedding", False) else math.sqrt(hidden_size),
        "layernorm_embedding": getattr(cfg_model, "layernorm_embedding", None),
        "encoder_normalize_before": getattr(cfg_model, "encoder_normalize_before", None),
        "decoder_normalize_before": getattr(cfg_model, "decoder_normalize_before", None),
        "activation_fn": getattr(cfg_model, "activation_fn", None),
        "notes": [
            "Fairseq learned positional embedding weights are absent when encoder_learned_pos/decoder_learned_pos are false.",
            "If no_token_positional_embeddings is false, Fairseq still applies sinusoidal positional embeddings as model math, not checkpoint weights.",
            "If no_scale_embedding is false, Fairseq scales token embeddings by sqrt(embed_dim).",
        ],
    }


def trt_config_summary(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    pc = config.get("pretrained_config", config)
    return {
        "config_path": str(path),
        "architecture": pc.get("architecture"),
        "has_position_embedding": pc.get("has_position_embedding"),
        "has_embedding_scale": pc.get("has_embedding_scale"),
        "has_embedding_layernorm": pc.get("has_embedding_layernorm"),
        "has_model_final_layernorm": pc.get("has_model_final_layernorm"),
        "layernorm_position": pc.get("layernorm_position"),
        "hidden_act": pc.get("hidden_act"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify TensorRT-LLM checkpoint tensors against Fairseq.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--trtllm-checkpoint-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    encoder_layers = layer_count(state, "encoder")
    decoder_layers = layer_count(state, "decoder")

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fairseq_checkpoint": str(args.checkpoint),
        "trtllm_checkpoint_dir": str(args.trtllm_checkpoint_dir),
        "encoder": compare(
            encoder_weights(state, encoder_layers),
            args.trtllm_checkpoint_dir / "encoder" / "rank0.safetensors",
        ),
        "decoder": compare(
            decoder_weights(state, decoder_layers),
            args.trtllm_checkpoint_dir / "decoder" / "rank0.safetensors",
        ),
        "fairseq_semantics": fairseq_semantics(checkpoint),
        "trtllm_semantics": {
            "encoder": trt_config_summary(args.trtllm_checkpoint_dir / "encoder" / "config.json"),
            "decoder": trt_config_summary(args.trtllm_checkpoint_dir / "decoder" / "config.json"),
        },
    }
    report["exact_tensor_port"] = report["encoder"]["exact"] and report["decoder"]["exact"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {args.output}")
    print(
        f"exact_tensor_port={report['exact_tensor_port']} "
        f"encoder_max_abs_diff={report['encoder']['max_abs_diff']} "
        f"decoder_max_abs_diff={report['decoder']['max_abs_diff']}"
    )
    print(
        "fairseq_embedding_scale="
        f"{report['fairseq_semantics']['fairseq_embedding_scale']} "
        "trt_encoder_has_embedding_scale="
        f"{report['trtllm_semantics']['encoder']['has_embedding_scale']}"
    )
    print(
        "fairseq_uses_token_positions="
        f"{not report['fairseq_semantics']['no_token_positional_embeddings']} "
        "trt_encoder_has_position_embedding="
        f"{report['trtllm_semantics']['encoder']['has_position_embedding']}"
    )
    return 0 if report["exact_tensor_port"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
