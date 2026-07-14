#!/usr/bin/env python3
"""Run IndicXlit TensorRT-LLM encoder-decoder generation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
APP_DIR = REPO_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from inspect_indicxlit_checkpoint import DEFAULT_CORPUS_BIN, DEFAULT_LANG_LIST, read_language_tokens  # noqa: E402


DEFAULT_ENGINE_DIR = SCRIPT_DIR / "artifacts" / "trtllm_engines_en_hi_beam4"
DEFAULT_WORDS_FILE = SCRIPT_DIR / "words_en_hi.txt"
DEFAULT_OUTPUT = SCRIPT_DIR / "artifacts" / "trtllm_beam_output.json"


SPECIALS = ["<s>", "<pad>", "</s>", "<unk>"]
BOS_ID = 0
PAD_ID = 1
EOS_ID = 2
UNK_ID = 3


def read_base_symbols(path: Path) -> list[str]:
    symbols = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        pieces = line.rsplit(" ", 1)
        if len(pieces) == 2:
            symbols.append(pieces[0])
    return symbols


def load_vocab(dict_path: Path, lang_list: Path) -> tuple[list[str], dict[str, int]]:
    symbols = SPECIALS + read_base_symbols(dict_path) + read_language_tokens(lang_list)
    return symbols, {symbol: index for index, symbol in enumerate(symbols)}


def read_words(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def preprocess_words(words: list[str], target_lang: str) -> list[str]:
    return [f"__{target_lang}__ " + " ".join(list(word.lower())) for word in words]


def encode_preprocessed(text: str, token_to_id: dict[str, int]) -> list[int]:
    ids = [token_to_id.get(piece, UNK_ID) for piece in text.split()]
    ids.append(EOS_ID)
    return ids


def pad_rows(rows: list[list[int]], pad_id: int) -> torch.Tensor:
    max_len = max(len(row) for row in rows)
    padded = [row + [pad_id] * (max_len - len(row)) for row in rows]
    return torch.tensor(padded, dtype=torch.int32)


def attention_mask_from_padded(padded: torch.Tensor, pad_id: int) -> torch.Tensor:
    return (padded != pad_id).to(dtype=torch.int32)


def decode_ids(ids: list[int], id_to_token: list[str]) -> tuple[list[int], str]:
    cleaned = []
    for token_id in ids:
        token_id = int(token_id)
        if token_id == EOS_ID:
            break
        if token_id in (BOS_ID, PAD_ID):
            continue
        cleaned.append(token_id)
    pieces = [id_to_token[token_id] if 0 <= token_id < len(id_to_token) else "<unk>" for token_id in cleaned]
    return cleaned, " ".join(pieces)


def postprocess_raw(raw: str) -> str:
    return "".join(raw.split(" "))


def normalize_output(output) -> torch.Tensor:
    if isinstance(output, dict):
        for key in ["output_ids", "sequences", "outputs"]:
            if key in output:
                output = output[key]
                break
    if not torch.is_tensor(output):
        raise TypeError(f"Unsupported runner output type: {type(output)!r}")
    return output.detach().cpu()


def extract_sequences(output: torch.Tensor, batch_size: int, prompt_len: int) -> list[list[list[int]]]:
    # Common TRT-LLM enc-dec shape is [batch, beam, seq]. Some builds return
    # [batch, seq] for greedy.
    if output.dim() == 3:
        return [
            [output[index, beam_index, prompt_len:].tolist() for beam_index in range(output.shape[1])]
            for index in range(batch_size)
        ]
    if output.dim() == 2:
        return [[output[index, prompt_len:].tolist()] for index in range(batch_size)]
    raise ValueError(f"Unsupported output shape: {tuple(output.shape)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run IndicXlit TRT-LLM generation.")
    parser.add_argument("--engine-dir", type=Path, default=DEFAULT_ENGINE_DIR)
    parser.add_argument("--corpus-bin", type=Path, default=DEFAULT_CORPUS_BIN)
    parser.add_argument("--lang-list", type=Path, default=DEFAULT_LANG_LIST)
    parser.add_argument("--words-file", type=Path, default=DEFAULT_WORDS_FILE)
    parser.add_argument("--lang", default="hi")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    from tensorrt_llm.runtime import EncDecModelRunner

    src_symbols, src_to_id = load_vocab(args.corpus_bin / "dict.en.txt", args.lang_list)
    tgt_symbols, _ = load_vocab(args.corpus_bin / f"dict.{args.lang}.txt", args.lang_list)

    base_words = read_words(args.words_file)
    if not base_words:
        raise SystemExit(f"No words found in {args.words_file}")
    words = [base_words[index % len(base_words)] for index in range(args.batch_size)]
    preprocessed = preprocess_words(words, args.lang)
    source_ids = [encode_preprocessed(row, src_to_id) for row in preprocessed]

    encoder_input_ids = pad_rows(source_ids, PAD_ID)
    decoder_input_ids = torch.full((len(words), 1), EOS_ID, dtype=torch.int32)

    runner = EncDecModelRunner.from_engine("rank0.engine", str(args.engine_dir))

    started = time.perf_counter()
    output = runner.generate(
        encoder_input_ids,
        decoder_input_ids,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.beam_width,
        pad_token_id=PAD_ID,
        eos_token_id=EOS_ID,
        bos_token_id=BOS_ID,
        attention_mask=attention_mask_from_padded(encoder_input_ids, PAD_ID),
        return_dict=False,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    output_tensor = normalize_output(output)
    sequences = extract_sequences(output_tensor, len(words), decoder_input_ids.shape[1])
    rows = []
    for index, beam_sequences in enumerate(sequences):
        candidates = []
        for beam_index, sequence in enumerate(beam_sequences[: args.topk]):
            generated_ids, raw = decode_ids(sequence, tgt_symbols)
            candidates.append(
                {
                    "rank": beam_index + 1,
                    "generated_token_ids": generated_ids,
                    "raw_decoded": raw,
                    "final_postprocessed": postprocess_raw(raw),
                }
            )
        best = candidates[0] if candidates else {
            "generated_token_ids": [],
            "raw_decoded": "",
            "final_postprocessed": "",
        }
        rows.append(
            {
                "index": index,
                "word": words[index],
                "preprocessed": preprocessed[index],
                "source_token_ids": source_ids[index],
                "generated_token_ids": best["generated_token_ids"],
                "raw_decoded_best": best["raw_decoded"],
                "final_postprocessed_best": best["final_postprocessed"],
                "candidates": candidates,
            }
        )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": "trtllm",
        "engine_dir": str(args.engine_dir),
        "lang": args.lang,
        "beam": args.beam_width,
        "topk": args.topk,
        "postprocess": True,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "elapsed_seconds": elapsed,
        "items_per_second": len(words) / elapsed if elapsed else None,
        "output_shape": list(output_tensor.shape),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(
        f"batch={len(words)} beam={args.beam_width} output_shape={list(output_tensor.shape)} "
        f"items_per_second={payload['items_per_second']:.2f}"
    )
    print(f"first={rows[0]['word']} -> {rows[0]['final_postprocessed_best']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
