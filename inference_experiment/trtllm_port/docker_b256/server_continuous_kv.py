#!/usr/bin/env python3
"""Minimal HTTP server for IndicXlit via the TensorRT-LLM EncDecModelRunner.

Serves a continuous-KV (+ fused cross-attention) decoder engine directly through
the runtime runner. This is the path that actually exercises the cross-attention
patch (Triton's inflight backend is paged-only).

Endpoint is Triton-ensemble-compatible:
    POST /v2/models/indicxlit/infer
    inputs: text_input, target_lang, max_tokens, beam_width, topk
    outputs: text_output, candidates_json
"""

import json
import os
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, Request

from tensorrt_llm.runtime import EncDecModelRunner

SPECIALS = ["<s>", "<pad>", "</s>", "<unk>"]
BOS_ID, PAD_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

MODEL_ROOT = Path(os.environ.get("INDICXLIT_MODEL_ROOT", "/models/assets/en2indic"))
ENGINE_DIR = Path(os.environ.get("ENGINE_DIR", "/models/engines/b256_cont"))
SOURCE_DICT = MODEL_ROOT / "corpus-bin" / "dict.en.txt"
LANG_LIST = Path(os.environ.get("INDICXLIT_LANG_LIST", str(MODEL_ROOT.parent / "lang_list.txt")))
DEFAULT_LANG = os.environ.get("INDICXLIT_LANG", "hi")
HOST = os.environ.get("INDICXLIT_HOST", "0.0.0.0")
PORT = int(os.environ.get("INDICXLIT_PORT", "8000"))


def _read_base_symbols(path: Path) -> list[str]:
    symbols = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        pieces = line.rsplit(" ", 1)
        if len(pieces) == 2:
            symbols.append(pieces[0])
    return symbols


def _read_language_tokens(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [f"__{ln.strip()}__" for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


SYMBOLS = SPECIALS + _read_base_symbols(SOURCE_DICT) + _read_language_tokens(LANG_LIST)
TOKEN_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
SUPPORTED_LANGS = {ln.strip().removeprefix("__").removesuffix("__") for ln in
                   [f"__{x}__" for x in _read_language_tokens(LANG_LIST)]}

# Target vocab is per output language (dict.<lang>.txt); load lazily and cache.
_TGT_SYMBOLS: dict[str, list[str]] = {}


def _target_symbols(target_lang: str) -> list[str]:
    if target_lang not in _TGT_SYMBOLS:
        tgt_dict = MODEL_ROOT / "corpus-bin" / f"dict.{target_lang}.txt"
        syms = SPECIALS + _read_base_symbols(tgt_dict) + _read_language_tokens(LANG_LIST)
        _TGT_SYMBOLS[target_lang] = syms
    return _TGT_SYMBOLS[target_lang]


def _decode_ids(ids: list[int], symbols: list[str]) -> str:
    pieces = []
    for tid in ids:
        tid = int(tid)
        if tid == EOS_ID:
            break
        if tid in (BOS_ID, PAD_ID):
            continue
        pieces.append(symbols[tid] if 0 <= tid < len(symbols) else "<unk>")
    return "".join(" ".join(pieces).split(" "))

print(f"[indicxlit] vocab size: {len(SYMBOLS)}; supported langs: {sorted(SUPPORTED_LANGS)}", flush=True)
print(f"[indicxlit] loading engine: {ENGINE_DIR}", flush=True)
RUNNER = EncDecModelRunner.from_engine("rank0.engine", str(ENGINE_DIR))
print("[indicxlit] runner ready", flush=True)

app = FastAPI(title="indicxlit-continuous-kv")


def _preprocess(word: str, target_lang: str) -> str:
    return f"__{target_lang}__ " + " ".join(list(word.lower()))


def _encode(text: str) -> list[int]:
    ids = [TOKEN_TO_ID.get(p, UNK_ID) for p in text.split()]
    ids.append(EOS_ID)
    return ids


def _run(words: list[str], target_lang: str, max_tokens: int, beam_width: int) -> list[list[str]]:
    prepped = [_preprocess(w, target_lang) for w in words]
    rows = [_encode(t) for t in prepped]
    max_len = max(len(r) for r in rows)
    encoder_input = torch.tensor([r + [PAD_ID] * (max_len - len(r)) for r in rows], dtype=torch.int32)
    decoder_input = torch.full((len(words), 1), EOS_ID, dtype=torch.int32)
    attention_mask = (encoder_input != PAD_ID).to(dtype=torch.int32)
    output = RUNNER.generate(
        encoder_input, decoder_input,
        max_new_tokens=max_tokens, num_beams=beam_width, pad_token_id=PAD_ID,
        eos_token_id=EOS_ID, bos_token_id=BOS_ID, attention_mask=attention_mask,
        return_dict=False,
    )
    if isinstance(output, dict):
        output = output.get("output_ids", output.get("sequences", output.get("outputs")))
    out = output.detach().cpu()
    results = []
    tgt_syms = _target_symbols(target_lang)
    for b in range(len(words)):
        beams = []
        if out.dim() == 3:
            seqs = [out[b, k, 1:].tolist() for k in range(out.shape[1])]
        else:
            seqs = [out[b, 1:].tolist()]
        for seq in seqs[:beam_width]:
            beams.append(_decode_ids(seq, tgt_syms))
        results.append(beams)
    return results


def _input(req: dict, name: str, default=None):
    for inp in req.get("inputs", []):
        if inp.get("name") == name:
            data = inp.get("data")
            return data[0] if isinstance(data, list) and data else data
    return default


@app.get("/v2/health/ready")
def health():
    return {"health": "ready"}


@app.post("/v2/models/indicxlit/infer")
async def infer(request: Request):
    req = await request.json()
    text = _input(req, "text_input") or ""
    target_lang = _input(req, "target_lang") or DEFAULT_LANG
    if target_lang not in SUPPORTED_LANGS:
        target_lang = DEFAULT_LANG
    max_tokens = int(_input(req, "max_tokens") or 32)
    beam_width = int(_input(req, "beam_width") or 5)
    topk = int(_input(req, "topk") or beam_width)

    words = [w for w in str(text).splitlines() if w] or [str(text)]
    results = _run(words, target_lang, max_tokens, beam_width)
    best = results[0][0] if results and results[0] else ""
    cands = results[0][:topk] if results else []

    return {
        "model_name": "indicxlit",
        "outputs": [
            {"name": "text_output", "datatype": "BYTES", "shape": [1, 1], "data": [best]},
            {"name": "candidates_json", "datatype": "BYTES", "shape": [1, 1], "data": [json.dumps(cands, ensure_ascii=False)]},
        ],
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
