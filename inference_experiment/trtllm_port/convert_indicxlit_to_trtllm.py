#!/usr/bin/env python3
"""Convert IndicXlit Fairseq weights to TensorRT-LLM encoder/decoder checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from inspect_indicxlit_checkpoint import DEFAULT_CHECKPOINT, layer_count  # noqa: E402


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "artifacts" / "trtllm_checkpoint_en_hi"


def require(name: str, state: dict[str, torch.Tensor]) -> torch.Tensor:
    if name not in state:
        raise KeyError(f"Missing Fairseq tensor: {name}")
    return state[name].detach().cpu().contiguous().float()


def cat_qkv(state: dict[str, torch.Tensor], prefix: str, suffix: str) -> torch.Tensor:
    return torch.cat(
        [
            require(f"{prefix}.q_proj.{suffix}", state),
            require(f"{prefix}.k_proj.{suffix}", state),
            require(f"{prefix}.v_proj.{suffix}", state),
        ],
        dim=0,
    )


def fairseq_sinusoidal_positions(num_embeddings: int, embedding_dim: int, padding_idx: int = 1) -> torch.Tensor:
    """Create Fairseq-compatible sinusoidal positional rows for TRT position ids.

    Fairseq's non-padding positions start at `padding_idx + 1`, while TRT-LLM's
    generated position ids start at 0. Row 0 in this table therefore maps to
    Fairseq row 2 for the first non-padding token.
    """
    fairseq_rows = num_embeddings + padding_idx + 1
    half_dim = embedding_dim // 2
    scale = math.log(10000.0) / (half_dim - 1)
    freqs = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -scale)
    positions = torch.arange(fairseq_rows, dtype=torch.float32).unsqueeze(1)
    table = positions * freqs.unsqueeze(0)
    table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
    if embedding_dim % 2 == 1:
        table = torch.cat([table, torch.zeros(fairseq_rows, 1)], dim=1)
    table[padding_idx, :] = 0
    return table[padding_idx + 1 : padding_idx + 1 + num_embeddings].contiguous()


def common_config(*, architecture: str, vocab_size: int, layers: int, hidden: int, ffn: int, heads: int, dtype: str):
    from tensorrt_llm.functional import LayerNormPositionType, LayerNormType, MLPType
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import PretrainedConfig

    return PretrainedConfig(
        architecture=architecture,
        dtype=dtype,
        logits_dtype=dtype,
        vocab_size=vocab_size,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_kv_heads=heads,
        head_size=hidden // heads,
        intermediate_size=ffn,
        hidden_act="gelu",
        max_position_embeddings=1024,
        has_position_embedding=True,
        has_embedding_layernorm=True,
        has_embedding_scale=True,
        has_attention_qkvo_bias=True,
        has_mlp_bias=True,
        has_model_final_layernorm=True,
        norm_epsilon=1e-5,
        layernorm_position=LayerNormPositionType.pre_layernorm,
        layernorm_type=LayerNormType.LayerNorm,
        q_scaling=1.0,
        residual_scaling=1.0,
        relative_attention=False,
        max_distance=0,
        num_buckets=0,
        model_type="bart",
        mlp_type=MLPType.MLP,
        use_parallel_embedding=False,
        embedding_sharding_dim=0,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
    )


def encoder_weights(state: dict[str, torch.Tensor], layers: int) -> dict[str, torch.Tensor]:
    hidden = int(require("encoder.embed_tokens.weight", state).shape[1])
    weights = {
        "transformer.vocab_embedding.weight": require("encoder.embed_tokens.weight", state),
        "transformer.position_embedding.weight": fairseq_sinusoidal_positions(1024, hidden),
        "transformer.ln_embed.weight": require("encoder.layernorm_embedding.weight", state),
        "transformer.ln_embed.bias": require("encoder.layernorm_embedding.bias", state),
        "transformer.ln_f.weight": require("encoder.layer_norm.weight", state),
        "transformer.ln_f.bias": require("encoder.layer_norm.bias", state),
    }
    for layer in range(layers):
        src = f"encoder.layers.{layer}"
        dst = f"transformer.layers.{layer}"
        weights[f"{dst}.attention.qkv.weight"] = cat_qkv(state, f"{src}.self_attn", "weight")
        weights[f"{dst}.attention.qkv.bias"] = cat_qkv(state, f"{src}.self_attn", "bias")
        weights[f"{dst}.attention.dense.weight"] = require(f"{src}.self_attn.out_proj.weight", state)
        weights[f"{dst}.attention.dense.bias"] = require(f"{src}.self_attn.out_proj.bias", state)
        weights[f"{dst}.attention_layernorm.weight"] = require(f"{src}.self_attn_layer_norm.weight", state)
        weights[f"{dst}.attention_layernorm.bias"] = require(f"{src}.self_attn_layer_norm.bias", state)
        weights[f"{dst}.mlp.fc.weight"] = require(f"{src}.fc1.weight", state)
        weights[f"{dst}.mlp.fc.bias"] = require(f"{src}.fc1.bias", state)
        weights[f"{dst}.mlp.proj.weight"] = require(f"{src}.fc2.weight", state)
        weights[f"{dst}.mlp.proj.bias"] = require(f"{src}.fc2.bias", state)
        weights[f"{dst}.mlp_layernorm.weight"] = require(f"{src}.final_layer_norm.weight", state)
        weights[f"{dst}.mlp_layernorm.bias"] = require(f"{src}.final_layer_norm.bias", state)
    return weights


def decoder_weights(state: dict[str, torch.Tensor], layers: int) -> dict[str, torch.Tensor]:
    hidden = int(require("decoder.embed_tokens.weight", state).shape[1])
    weights = {
        "transformer.vocab_embedding.weight": require("decoder.embed_tokens.weight", state),
        "transformer.position_embedding.weight": fairseq_sinusoidal_positions(1024, hidden),
        "transformer.ln_embed.weight": require("decoder.layernorm_embedding.weight", state),
        "transformer.ln_embed.bias": require("decoder.layernorm_embedding.bias", state),
        "transformer.ln_f.weight": require("decoder.layer_norm.weight", state),
        "transformer.ln_f.bias": require("decoder.layer_norm.bias", state),
        "lm_head.weight": require("decoder.output_projection.weight", state),
    }
    for layer in range(layers):
        src = f"decoder.layers.{layer}"
        dst = f"transformer.layers.{layer}"
        for fairseq_attn, trt_attn in [("self_attn", "self_attention"), ("encoder_attn", "cross_attention")]:
            weights[f"{dst}.{trt_attn}.qkv.weight"] = cat_qkv(state, f"{src}.{fairseq_attn}", "weight")
            weights[f"{dst}.{trt_attn}.qkv.bias"] = cat_qkv(state, f"{src}.{fairseq_attn}", "bias")
            weights[f"{dst}.{trt_attn}.dense.weight"] = require(f"{src}.{fairseq_attn}.out_proj.weight", state)
            weights[f"{dst}.{trt_attn}.dense.bias"] = require(f"{src}.{fairseq_attn}.out_proj.bias", state)
        weights[f"{dst}.self_attention_layernorm.weight"] = require(f"{src}.self_attn_layer_norm.weight", state)
        weights[f"{dst}.self_attention_layernorm.bias"] = require(f"{src}.self_attn_layer_norm.bias", state)
        weights[f"{dst}.cross_attention_layernorm.weight"] = require(f"{src}.encoder_attn_layer_norm.weight", state)
        weights[f"{dst}.cross_attention_layernorm.bias"] = require(f"{src}.encoder_attn_layer_norm.bias", state)
        weights[f"{dst}.mlp.fc.weight"] = require(f"{src}.fc1.weight", state)
        weights[f"{dst}.mlp.fc.bias"] = require(f"{src}.fc1.bias", state)
        weights[f"{dst}.mlp.proj.weight"] = require(f"{src}.fc2.weight", state)
        weights[f"{dst}.mlp.proj.bias"] = require(f"{src}.fc2.bias", state)
        weights[f"{dst}.mlp_layernorm.weight"] = require(f"{src}.final_layer_norm.weight", state)
        weights[f"{dst}.mlp_layernorm.bias"] = require(f"{src}.final_layer_norm.bias", state)
    return weights


def cast_floating_weights(weights: dict[str, torch.Tensor], dtype: str) -> dict[str, torch.Tensor]:
    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]
    return {
        name: tensor.to(dtype=torch_dtype) if tensor.is_floating_point() else tensor
        for name, tensor in weights.items()
    }


def validate_and_save(component: str, model_cls, config, weights: dict[str, torch.Tensor], output_dir: Path, dtype: str) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model_cls(config)
    expected = {name: list(param.raw_value.shape) for name, param in model.named_parameters()}
    provided = {name: list(tensor.shape) for name, tensor in weights.items()}
    missing = sorted(set(expected) - set(provided))
    extra = sorted(set(provided) - set(expected))
    shape_mismatches = {
        name: {"expected": expected[name], "actual": provided[name]}
        for name in sorted(set(expected) & set(provided))
        if expected[name] != provided[name]
    }
    if missing or extra or shape_mismatches:
        raise SystemExit(
            json.dumps(
                {
                    "component": component,
                    "missing": missing,
                    "extra": extra,
                    "shape_mismatches": shape_mismatches,
                },
                indent=2,
            )
        )
    model.load(cast_floating_weights(weights, dtype))
    model.save_checkpoint(str(output_dir), save_config=True)
    return {
        "component": component,
        "output_dir": str(output_dir),
        "tensor_count": len(weights),
        "config": output_dir.joinpath("config.json").name,
        "weights": "rank0.safetensors",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert IndicXlit Fairseq checkpoint to TensorRT-LLM checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"Missing checkpoint: {args.checkpoint}")

    from tensorrt_llm.models.enc_dec.model import DecoderModel, EncoderModel

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    cfg_model = checkpoint["cfg"]["model"]
    encoder_layers = layer_count(state, "encoder")
    decoder_layers = layer_count(state, "decoder")
    hidden = int(getattr(cfg_model, "encoder_embed_dim"))
    ffn = int(getattr(cfg_model, "encoder_ffn_embed_dim"))
    heads = int(getattr(cfg_model, "encoder_attention_heads"))
    source_vocab = int(require("encoder.embed_tokens.weight", state).shape[0])
    target_vocab = int(require("decoder.embed_tokens.weight", state).shape[0])

    encoder_config = common_config(
        architecture="EncoderModel",
        vocab_size=source_vocab,
        layers=encoder_layers,
        hidden=hidden,
        ffn=ffn,
        heads=heads,
        dtype=args.dtype,
    )
    decoder_config = common_config(
        architecture="DecoderModel",
        vocab_size=target_vocab,
        layers=decoder_layers,
        hidden=hidden,
        ffn=ffn,
        heads=heads,
        dtype=args.dtype,
    )
    decoder_config.encoder_hidden_size = hidden
    decoder_config.encoder_num_heads = heads
    decoder_config.encoder_num_kv_heads = heads
    decoder_config.encoder_head_size = hidden // heads
    decoder_config.skip_cross_kv = False

    args.output_dir.mkdir(parents=True, exist_ok=True)
    components = [
        validate_and_save(
            "encoder",
            EncoderModel,
            encoder_config,
            encoder_weights(state, encoder_layers),
            args.output_dir / "encoder",
            args.dtype,
        ),
        validate_and_save(
            "decoder",
            DecoderModel,
            decoder_config,
            decoder_weights(state, decoder_layers),
            args.output_dir / "decoder",
            args.dtype,
        ),
    ]

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "output_dir": str(args.output_dir),
        "precision": args.dtype,
        "source_vocab": source_vocab,
        "target_vocab": target_vocab,
        "encoder_layers": encoder_layers,
        "decoder_layers": decoder_layers,
        "embedding_scale": math.sqrt(hidden),
        "position_embedding": "fairseq sinusoidal positions shifted so TRT position id 0 maps to Fairseq position padding_idx+1",
        "components": components,
    }
    args.output_dir.joinpath("manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote TensorRT-LLM checkpoint to {args.output_dir}")
    print(f"encoder_tensors={components[0]['tensor_count']} decoder_tensors={components[1]['tensor_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
