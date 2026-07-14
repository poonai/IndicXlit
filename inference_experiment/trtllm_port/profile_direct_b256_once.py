#!/usr/bin/env python3
"""Small direct TensorRT-LLM run for Nsight Systems profiling."""

from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
APP_DIR = REPO_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_trtllm_greedy import (  # noqa: E402
    BOS_ID,
    DEFAULT_CORPUS_BIN,
    DEFAULT_LANG_LIST,
    DEFAULT_WORDS_FILE,
    EOS_ID,
    PAD_ID,
    attention_mask_from_padded,
    encode_preprocessed,
    load_vocab,
    pad_rows,
    preprocess_words,
    read_words,
)


def default_engine_dir() -> Path:
    candidates = [
        SCRIPT_DIR / "artifacts" / "trtllm_engines_en_hi_beam5_host_fp16_b256_continuous_kv",
        SCRIPT_DIR / "artifacts" / "trtllm_engines_en_hi_beam5_host_fp16_b256",
        SCRIPT_DIR / "artifacts" / "trtllm_engines_en_hi_beam5_triton_fp16_b256",
    ]
    for candidate in candidates:
        if (candidate / "encoder" / "rank0.engine").is_file() and (
            candidate / "decoder" / "rank0.engine"
        ).is_file():
            return candidate
    return candidates[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Small direct TensorRT-LLM run for Nsight Systems profiling.")
    parser.add_argument(
        "--engine-dir",
        type=Path,
        default=default_engine_dir(),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    from tensorrt_llm.runtime import EncDecModelRunner

    _, src_to_id = load_vocab(DEFAULT_CORPUS_BIN / "dict.en.txt", DEFAULT_LANG_LIST)
    base_words = read_words(DEFAULT_WORDS_FILE)
    words = [base_words[index % len(base_words)] for index in range(256)]
    preprocessed = preprocess_words(words, "hi")
    source_ids = [encode_preprocessed(row, src_to_id) for row in preprocessed]
    encoder_input_ids = pad_rows(source_ids, PAD_ID)
    decoder_input_ids = torch.full((len(words), 1), EOS_ID, dtype=torch.int32)
    attention_mask = attention_mask_from_padded(encoder_input_ids, PAD_ID)

    runner = EncDecModelRunner.from_engine("rank0.engine", str(args.engine_dir))

    torch.cuda.nvtx.range_push("warmup_generate_b256_beam5")
    try:
        for _ in range(args.warmup):
            runner.generate(
                encoder_input_ids,
                decoder_input_ids,
                max_new_tokens=32,
                num_beams=5,
                pad_token_id=PAD_ID,
                eos_token_id=EOS_ID,
                bos_token_id=BOS_ID,
                attention_mask=attention_mask,
                return_dict=False,
            )
    finally:
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

    started = time.perf_counter()
    torch.cuda.nvtx.range_push("measured_generate_b256_beam5")
    try:
        for _ in range(args.iterations):
            torch.cuda.nvtx.range_push("generate_iteration")
            try:
                runner.generate(
                    encoder_input_ids,
                    decoder_input_ids,
                    max_new_tokens=32,
                    num_beams=5,
                    pad_token_id=PAD_ID,
                    eos_token_id=EOS_ID,
                    bos_token_id=BOS_ID,
                    attention_mask=attention_mask,
                    return_dict=False,
                )
            finally:
                torch.cuda.nvtx.range_pop()
    finally:
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    print(
        f"batch=256 iterations={args.iterations} elapsed={elapsed:.6f}s "
        f"items_per_second={256 * args.iterations / elapsed:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
