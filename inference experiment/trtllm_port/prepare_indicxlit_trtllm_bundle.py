#!/usr/bin/env python3
"""Prepare a portable IndicXlit checkpoint bundle for TensorRT-LLM conversion.

This script does not build TensorRT-LLM engines. It extracts the Fairseq
checkpoint, dictionaries, and descriptive weight mapping into a version-neutral
bundle that a TensorRT-LLM-specific converter can consume on the target machine.
The final TRT-LLM checkpoint field names still need to be matched to the
installed TensorRT-LLM version.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
APP_DIR = REPO_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from inspect_indicxlit_checkpoint import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CORPUS_BIN,
    DEFAULT_LANG_LIST,
    build_mapping,
    layer_count,
    read_dictionary,
    read_language_tokens,
    shape_of,
)


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "artifacts" / "trtllm_bundle_en_hi"


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(
            f"Missing {label}: {path}\n"
            "Install/download the IndicXlit model assets first, or pass the "
            "correct --checkpoint/--corpus-bin paths."
        )


def fairseq_symbols(path: Path, lang_list: Path | None = None) -> list[str]:
    symbols = ["<s>", "<pad>", "</s>", "<unk>"]
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        pieces = line.rsplit(" ", 1)
        if len(pieces) == 2:
            symbols.append(pieces[0])
    symbols.extend(read_language_tokens(lang_list) if lang_list else [])
    return symbols


def write_vocab(path: Path, symbols: list[str]) -> None:
    payload = {
        "symbols": symbols,
        "indices": {symbol: index for index, symbol in enumerate(symbols)},
        "special_ids": {"bos": 0, "pad": 1, "eos": 2, "unk": 3},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def tensor_to_numpy(tensor):
    return tensor.detach().cpu().contiguous().numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an IndicXlit TRT-LLM conversion bundle.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--corpus-bin", type=Path, default=DEFAULT_CORPUS_BIN)
    parser.add_argument("--lang-list", type=Path, default=DEFAULT_LANG_LIST)
    parser.add_argument("--source-lang", default="en")
    parser.add_argument("--target-lang", default="hi")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    require_file(args.checkpoint, "Fairseq checkpoint")
    require_file(args.corpus_bin / f"dict.{args.source_lang}.txt", "source dictionary")
    require_file(args.corpus_bin / f"dict.{args.target_lang}.txt", "target dictionary")
    require_file(args.lang_list, "language list")

    import torch

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    cfg_model = checkpoint["cfg"]["model"]
    encoder_layers = layer_count(state, "encoder")
    decoder_layers = layer_count(state, "decoder")
    mapping, missing, ignored = build_mapping(state, encoder_layers, decoder_layers)
    if missing:
        raise SystemExit("Missing required tensors:\n" + "\n".join(f"  {name}" for name in missing))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_symbols = fairseq_symbols(args.corpus_bin / f"dict.{args.source_lang}.txt", args.lang_list)
    target_symbols = fairseq_symbols(args.corpus_bin / f"dict.{args.target_lang}.txt", args.lang_list)
    write_vocab(args.output_dir / f"vocab.{args.source_lang}.json", source_symbols)
    write_vocab(args.output_dir / f"vocab.{args.target_lang}.json", target_symbols)

    weights = {}
    weight_manifest = []
    for row in mapping:
        source = row["source"]
        target = row["target"]
        tensor = state[source]
        weights[target] = tensor_to_numpy(tensor)
        weight_manifest.append(
            {
                "source": source,
                "target": target,
                "shape": shape_of(tensor),
                "dtype": str(tensor.dtype),
            }
        )

    np.savez_compressed(args.output_dir / "weights_descriptive.npz", **weights)

    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "descriptive_bundle_ready",
        "source_lang": args.source_lang,
        "target_lang": args.target_lang,
        "checkpoint": str(args.checkpoint),
        "corpus_bin": str(args.corpus_bin),
        "architecture": {
            "model_type": "fairseq_transformer_encoder_decoder",
            "encoder_layers": encoder_layers,
            "decoder_layers": decoder_layers,
            "hidden_size": getattr(cfg_model, "encoder_embed_dim", None),
            "ffn_hidden_size": getattr(cfg_model, "encoder_ffn_embed_dim", None),
            "num_attention_heads": getattr(cfg_model, "encoder_attention_heads", None),
            "activation": getattr(cfg_model, "activation_fn", None),
            "encoder_normalize_before": getattr(cfg_model, "encoder_normalize_before", None),
            "decoder_normalize_before": getattr(cfg_model, "decoder_normalize_before", None),
            "layernorm_embedding": getattr(cfg_model, "layernorm_embedding", None),
            "share_decoder_input_output_embed": getattr(cfg_model, "share_decoder_input_output_embed", None),
            "source_vocab_size": len(source_symbols),
            "target_vocab_size": len(target_symbols),
            "special_ids": {"bos": 0, "pad": 1, "eos": 2, "unk": 3},
        },
        "recommended_engine_build": {
            "precision": "float32",
            "tp_size": 1,
            "pp_size": 1,
            "max_beam_width": 1,
            "max_batch_size": 512,
            "max_input_len": 128,
            "max_seq_len": 64,
        },
        "files": {
            "weights": "weights_descriptive.npz",
            "source_vocab": f"vocab.{args.source_lang}.json",
            "target_vocab": f"vocab.{args.target_lang}.json",
            "mapping": "mapping_descriptive.json",
            "config": "config.json",
        },
        "notes": [
            "Weight names are descriptive, not guaranteed TensorRT-LLM schema names.",
            "Map these names to the installed TensorRT-LLM encoder-decoder checkpoint API on the cloud machine.",
            "Keep fp32 until greedy parity is understood.",
        ],
    }

    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "mapping_descriptive.json").write_text(
        json.dumps({"mapping": weight_manifest, "ignored": ignored}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Wrote bundle to {args.output_dir}")
    print(f"weights={len(weights)} source_vocab={len(source_symbols)} target_vocab={len(target_symbols)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
