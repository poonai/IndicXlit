#!/usr/bin/env python3
"""JSON-line worker around TensorRT-LLM EncDecModelRunner.

Rust owns HTTP batching and sends one JSON request per line on stdin. This
process owns Python/TensorRT-LLM state so tensor prep stays identical to the
known-good EncDecModelRunner path.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import torch
from tensorrt_llm.runtime import EncDecModelRunner

SPECIALS = ["<s>", "<pad>", "</s>", "<unk>"]
BOS_ID, PAD_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

MODEL_ROOT = Path(os.environ.get("INDICXLIT_MODEL_ROOT", "/models/assets/en2indic"))
ENGINE_DIR = Path(os.environ.get("ENGINE_DIR", "/models/engines/b256_cont"))
SOURCE_DICT = MODEL_ROOT / "corpus-bin" / "dict.en.txt"
LANG_LIST = Path(os.environ.get("INDICXLIT_LANG_LIST", str(MODEL_ROOT.parent / "lang_list.txt")))
DEFAULT_LANG = os.environ.get("INDICXLIT_LANG", "hi")


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def read_base_symbols(path: Path) -> list[str]:
    symbols: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        pieces = line.rsplit(" ", 1)
        if len(pieces) == 2:
            symbols.append(pieces[0])
    return symbols


def read_langs(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def language_tokens(langs: list[str]) -> list[str]:
    return [f"__{lang}__" for lang in langs]


LANGS = read_langs(LANG_LIST)
SUPPORTED_LANGS = set(LANGS)
if not SUPPORTED_LANGS:
    SUPPORTED_LANGS.add(DEFAULT_LANG)

SYMBOLS = SPECIALS + read_base_symbols(SOURCE_DICT) + language_tokens(LANGS)
TOKEN_TO_ID = {symbol: idx for idx, symbol in enumerate(SYMBOLS)}
TARGET_SYMBOLS: dict[str, list[str]] = {}


def target_symbols(target_lang: str) -> list[str]:
    if target_lang not in TARGET_SYMBOLS:
        tgt_dict = MODEL_ROOT / "corpus-bin" / f"dict.{target_lang}.txt"
        TARGET_SYMBOLS[target_lang] = SPECIALS + read_base_symbols(tgt_dict) + language_tokens(LANGS)
    return TARGET_SYMBOLS[target_lang]


def decode_ids(ids: list[int], symbols: list[str]) -> str:
    pieces: list[str] = []
    for token_id in ids:
        token_id = int(token_id)
        if token_id == EOS_ID:
            break
        if token_id in (BOS_ID, PAD_ID):
            continue
        pieces.append(symbols[token_id] if 0 <= token_id < len(symbols) else "<unk>")
    return "".join(" ".join(pieces).split(" "))


def preprocess(word: str, target_lang: str) -> str:
    return f"__{target_lang}__ " + " ".join(list(word.lower()))


def encode(text: str) -> list[int]:
    ids = [TOKEN_TO_ID.get(piece, UNK_ID) for piece in text.split()]
    ids.append(EOS_ID)
    return ids


log(f"[indicxlit-worker] vocab size: {len(SYMBOLS)}; supported langs: {sorted(SUPPORTED_LANGS)}")
log(f"[indicxlit-worker] loading engine: {ENGINE_DIR}")
RUNNER = EncDecModelRunner.from_engine("rank0.engine", str(ENGINE_DIR))
log("[indicxlit-worker] runner ready")


def run(words: list[str], target_lang: str, max_tokens: int, beam_width: int, topk: int) -> list[str]:
    if target_lang not in SUPPORTED_LANGS:
        target_lang = DEFAULT_LANG
    if not words:
        words = [""]

    prepped = [preprocess(word, target_lang) for word in words]
    rows = [encode(text) for text in prepped]
    max_len = max(len(row) for row in rows)
    encoder_input = torch.tensor([row + [PAD_ID] * (max_len - len(row)) for row in rows], dtype=torch.int32)
    decoder_input = torch.full((len(words), 1), EOS_ID, dtype=torch.int32)
    attention_mask = (encoder_input != PAD_ID).to(dtype=torch.int32)

    output = RUNNER.generate(
        encoder_input,
        decoder_input,
        max_new_tokens=max_tokens,
        num_beams=beam_width,
        pad_token_id=PAD_ID,
        eos_token_id=EOS_ID,
        bos_token_id=BOS_ID,
        attention_mask=attention_mask,
        return_dict=False,
    )
    if isinstance(output, dict):
        output = output.get("output_ids", output.get("sequences", output.get("outputs")))
    out = output.detach().cpu()

    symbols = target_symbols(target_lang)
    results: list[str] = []
    for batch_idx in range(len(words)):
        candidates: list[str] = []
        if out.dim() == 3:
            seqs = [out[batch_idx, beam_idx, 1:].tolist() for beam_idx in range(out.shape[1])]
        else:
            seqs = [out[batch_idx, 1:].tolist()]
        for seq in seqs[: max(1, min(topk, len(seqs)))]:
            candidates.append(decode_ids(seq, symbols))
        results.append(json.dumps(candidates, ensure_ascii=False))
    return results


def respond(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def main() -> int:
    for line in sys.stdin:
        try:
            request = json.loads(line)
            op = request.get("op")
            if op == "ping":
                respond({"ok": True, "outputs": []})
            elif op == "infer":
                outputs = run(
                    [str(word) for word in request.get("words", [])],
                    str(request.get("target_lang") or DEFAULT_LANG),
                    int(request.get("max_tokens") or 32),
                    int(request.get("beam_width") or 1),
                    int(request.get("topk") or request.get("beam_width") or 1),
                )
                respond({"ok": True, "outputs": outputs})
            else:
                respond({"ok": False, "error": f"unknown op: {op}"})
        except Exception as exc:  # Keep the worker alive after request errors.
            traceback.print_exc(file=sys.stderr)
            respond({"ok": False, "error": str(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
