#!/usr/bin/env python3
"""Inspect IndicXlit Fairseq checkpoint and emit a TRT-LLM mapping report."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = REPO_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


DEFAULT_MODEL_ROOT = (
    REPO_ROOT
    / "app"
    / "ai4bharat"
    / "transliteration"
    / "transformer"
    / "models"
    / "en2indic"
    / "v1.0"
)
DEFAULT_CHECKPOINT = DEFAULT_MODEL_ROOT / "transformer" / "indicxlit.pt"
DEFAULT_CORPUS_BIN = DEFAULT_MODEL_ROOT / "corpus-bin"
DEFAULT_LANG_LIST = DEFAULT_MODEL_ROOT.parent / "lang_list.txt"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "artifacts" / "indicxlit_checkpoint_mapping.json"


def shape_of(tensor) -> list[int]:
    return list(tensor.shape) if hasattr(tensor, "shape") else []


def tensor_info(state: dict[str, object], name: str) -> dict[str, object]:
    value = state.get(name)
    return {
        "name": name,
        "exists": value is not None,
        "shape": shape_of(value) if value is not None else None,
        "dtype": str(value.dtype) if hasattr(value, "dtype") else None,
    }


def expect(state: dict[str, object], name: str, missing: list[str], mapping: list[dict[str, object]], target: str):
    info = tensor_info(state, name)
    if not info["exists"]:
        missing.append(name)
    mapping.append({"source": name, "target": target, **info})


def read_language_tokens(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [
        f"__{line.strip()}__"
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_dictionary(path: Path, lang_list: Path | None = None) -> dict[str, object]:
    symbols = []
    counts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        pieces = line.rsplit(" ", 1)
        if len(pieces) != 2:
            continue
        symbols.append(pieces[0])
        try:
            counts.append(int(pieces[1]))
        except ValueError:
            counts.append(None)

    special_symbols = ["<s>", "<pad>", "</s>", "<unk>"]
    language_tokens = read_language_tokens(lang_list) if lang_list else []
    all_symbols = special_symbols + symbols + language_tokens
    return {
        "path": str(path),
        "lang_list": str(lang_list) if lang_list else None,
        "size_without_specials": len(symbols),
        "language_token_count": len(language_tokens),
        "size_with_fairseq_specials": len(all_symbols),
        "first_symbols_without_specials": symbols[:30],
        "language_tokens": language_tokens,
        "special_ids": {symbol: index for index, symbol in enumerate(special_symbols)},
        "language_token_ids": {
            symbol: index
            for index, symbol in enumerate(all_symbols)
            if symbol.startswith("__") and symbol.endswith("__")
        },
        "symbol_counts_sample": dict(zip(symbols[:30], counts[:30])),
    }


def layer_count(state: dict[str, object], prefix: str) -> int:
    indices = set()
    marker = f"{prefix}.layers."
    for key in state:
        if key.startswith(marker):
            rest = key[len(marker) :]
            piece = rest.split(".", 1)[0]
            if piece.isdigit():
                indices.add(int(piece))
    return max(indices) + 1 if indices else 0


def build_mapping(state: dict[str, object], encoder_layers: int, decoder_layers: int) -> tuple[list[dict[str, object]], list[str], list[dict[str, object]]]:
    mapping: list[dict[str, object]] = []
    missing: list[str] = []
    ignored: list[dict[str, object]] = []

    expect(state, "encoder.embed_tokens.weight", missing, mapping, "encoder.embedding.vocab_embedding.weight")
    expect(state, "decoder.embed_tokens.weight", missing, mapping, "decoder.embedding.vocab_embedding.weight")
    expect(state, "decoder.output_projection.weight", missing, mapping, "decoder.lm_head.weight")
    expect(state, "encoder.layernorm_embedding.weight", missing, mapping, "encoder.embedding.layernorm.weight")
    expect(state, "encoder.layernorm_embedding.bias", missing, mapping, "encoder.embedding.layernorm.bias")
    expect(state, "decoder.layernorm_embedding.weight", missing, mapping, "decoder.embedding.layernorm.weight")
    expect(state, "decoder.layernorm_embedding.bias", missing, mapping, "decoder.embedding.layernorm.bias")
    expect(state, "decoder.layer_norm.weight", missing, mapping, "decoder.final_layernorm.weight")
    expect(state, "decoder.layer_norm.bias", missing, mapping, "decoder.final_layernorm.bias")

    for layer in range(encoder_layers):
        prefix = f"encoder.layers.{layer}"
        target = f"encoder.layers.{layer}"
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            expect(state, f"{prefix}.self_attn.{proj}.weight", missing, mapping, f"{target}.self_attention.{proj}.weight")
            expect(state, f"{prefix}.self_attn.{proj}.bias", missing, mapping, f"{target}.self_attention.{proj}.bias")
        for norm in ["self_attn_layer_norm", "final_layer_norm"]:
            expect(state, f"{prefix}.{norm}.weight", missing, mapping, f"{target}.{norm}.weight")
            expect(state, f"{prefix}.{norm}.bias", missing, mapping, f"{target}.{norm}.bias")
        for ff in ["fc1", "fc2"]:
            expect(state, f"{prefix}.{ff}.weight", missing, mapping, f"{target}.mlp.{ff}.weight")
            expect(state, f"{prefix}.{ff}.bias", missing, mapping, f"{target}.mlp.{ff}.bias")

    for layer in range(decoder_layers):
        prefix = f"decoder.layers.{layer}"
        target = f"decoder.layers.{layer}"
        for attention_name in ["self_attn", "encoder_attn"]:
            target_attention = "self_attention" if attention_name == "self_attn" else "cross_attention"
            for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]:
                expect(
                    state,
                    f"{prefix}.{attention_name}.{proj}.weight",
                    missing,
                    mapping,
                    f"{target}.{target_attention}.{proj}.weight",
                )
                expect(
                    state,
                    f"{prefix}.{attention_name}.{proj}.bias",
                    missing,
                    mapping,
                    f"{target}.{target_attention}.{proj}.bias",
                )
        for norm in ["self_attn_layer_norm", "encoder_attn_layer_norm", "final_layer_norm"]:
            expect(state, f"{prefix}.{norm}.weight", missing, mapping, f"{target}.{norm}.weight")
            expect(state, f"{prefix}.{norm}.bias", missing, mapping, f"{target}.{norm}.bias")
        for ff in ["fc1", "fc2"]:
            expect(state, f"{prefix}.{ff}.weight", missing, mapping, f"{target}.mlp.{ff}.weight")
            expect(state, f"{prefix}.{ff}.bias", missing, mapping, f"{target}.mlp.{ff}.bias")

    mapped_sources = {row["source"] for row in mapping}
    for key, value in state.items():
        if key in mapped_sources:
            continue
        reason = "not needed for inference conversion"
        if key.endswith("._float_tensor"):
            reason = "Fairseq positional embedding sentinel, not a trainable weight"
        elif key.endswith(".version"):
            reason = "Fairseq module version marker"
        ignored.append({"source": key, "shape": shape_of(value), "reason": reason})

    return mapping, missing, ignored


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect IndicXlit checkpoint for TensorRT-LLM conversion.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--corpus-bin", type=Path, default=DEFAULT_CORPUS_BIN)
    parser.add_argument("--lang-list", type=Path, default=DEFAULT_LANG_LIST)
    parser.add_argument("--source-lang", default="en")
    parser.add_argument("--target-lang", default="hi")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    import torch

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    cfg_model = checkpoint["cfg"]["model"]
    encoder_layers = layer_count(state, "encoder")
    decoder_layers = layer_count(state, "decoder")
    mapping, missing, ignored = build_mapping(state, encoder_layers, decoder_layers)

    tensor_shapes = Counter(tuple(shape_of(value)) for value in state.values() if shape_of(value))
    source_dict = read_dictionary(args.corpus_bin / f"dict.{args.source_lang}.txt", args.lang_list)
    target_dict = read_dictionary(args.corpus_bin / f"dict.{args.target_lang}.txt", args.lang_list)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "status": "pass" if not missing else "fail",
        "config": {
            "task": getattr(cfg_model, "task", None),
            "arch": getattr(cfg_model, "arch", None),
            "encoder_layers": encoder_layers,
            "decoder_layers": decoder_layers,
            "encoder_embed_dim": getattr(cfg_model, "encoder_embed_dim", None),
            "decoder_embed_dim": getattr(cfg_model, "decoder_embed_dim", None),
            "encoder_ffn_embed_dim": getattr(cfg_model, "encoder_ffn_embed_dim", None),
            "decoder_ffn_embed_dim": getattr(cfg_model, "decoder_ffn_embed_dim", None),
            "encoder_attention_heads": getattr(cfg_model, "encoder_attention_heads", None),
            "decoder_attention_heads": getattr(cfg_model, "decoder_attention_heads", None),
            "activation_fn": getattr(cfg_model, "activation_fn", None),
            "encoder_normalize_before": getattr(cfg_model, "encoder_normalize_before", None),
            "decoder_normalize_before": getattr(cfg_model, "decoder_normalize_before", None),
            "layernorm_embedding": getattr(cfg_model, "layernorm_embedding", None),
            "share_decoder_input_output_embed": getattr(cfg_model, "share_decoder_input_output_embed", None),
            "max_source_positions": getattr(cfg_model, "max_source_positions", None),
            "max_target_positions": getattr(cfg_model, "max_target_positions", None),
        },
        "dictionaries": {
            "source": source_dict,
            "target": target_dict,
        },
        "tensor_count": len(state),
        "tensor_shape_histogram": {str(list(shape)): count for shape, count in tensor_shapes.items()},
        "mapping_count": len(mapping),
        "missing_required_sources": missing,
        "mapping": mapping,
        "ignored": ignored,
        "notes": [
            "Fairseq uses BOS id 0 (<s>), PAD id 1, EOS id 2, UNK id 3 by default.",
            "IndicXlit preprocessing prefixes the source character sequence with the target language token, e.g. __hi__.",
            "TRT-LLM converter must preserve Fairseq pre-layernorm and GELU semantics.",
            "TRT-LLM target names in this report are descriptive placeholders; final names must match the installed TensorRT-LLM checkpoint schema.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {args.output}")
    print(
        f"status={report['status']} tensors={report['tensor_count']} "
        f"mapped={report['mapping_count']} missing={len(missing)} ignored={len(ignored)}"
    )
    if missing:
        print("Missing required tensors:")
        for name in missing:
            print(f"  {name}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
