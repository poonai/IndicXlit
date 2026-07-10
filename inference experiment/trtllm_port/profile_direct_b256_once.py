#!/usr/bin/env python3
"""Small direct TensorRT-LLM run for Nsight Systems profiling."""

from __future__ import annotations

import sys
import time
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


def main() -> int:
    from tensorrt_llm.runtime import EncDecModelRunner

    engine_dir = SCRIPT_DIR / "artifacts" / "trtllm_engines_en_hi_beam5_triton_b256"
    _, src_to_id = load_vocab(DEFAULT_CORPUS_BIN / "dict.en.txt", DEFAULT_LANG_LIST)
    base_words = read_words(DEFAULT_WORDS_FILE)
    words = [base_words[index % len(base_words)] for index in range(256)]
    preprocessed = preprocess_words(words, "hi")
    source_ids = [encode_preprocessed(row, src_to_id) for row in preprocessed]
    encoder_input_ids = pad_rows(source_ids, PAD_ID)
    decoder_input_ids = torch.full((len(words), 1), EOS_ID, dtype=torch.int32)
    attention_mask = attention_mask_from_padded(encoder_input_ids, PAD_ID)

    runner = EncDecModelRunner.from_engine("rank0.engine", str(engine_dir))

    for _ in range(2):
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
    torch.cuda.synchronize()

    started = time.perf_counter()
    iterations = 5
    for _ in range(iterations):
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
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    print(
        f"batch=256 iterations={iterations} elapsed={elapsed:.6f}s "
        f"items_per_second={256 * iterations / elapsed:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
