#!/usr/bin/env python3
"""Sustained direct TensorRT-LLM encoder-decoder benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
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


class GpuSampler:
    def __init__(self, interval: float):
        self.interval = interval
        self.samples: list[dict[str, int]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self):
        while not self._stop.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,utilization.memory,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=2,
                ).strip()
                if output:
                    util_gpu, util_mem, mem_used = [int(part.strip()) for part in output.splitlines()[0].split(",")]
                    self.samples.append(
                        {
                            "gpu_util_percent": util_gpu,
                            "mem_util_percent": util_mem,
                            "memory_used_mib": mem_used,
                        }
                    )
            except Exception:
                pass
            self._stop.wait(self.interval)

    def summary(self) -> dict[str, float | int | None]:
        if not self.samples:
            return {
                "sample_count": 0,
                "avg_gpu_util_percent": None,
                "max_gpu_util_percent": None,
                "avg_mem_util_percent": None,
                "max_mem_util_percent": None,
                "max_memory_used_mib": None,
            }
        return {
            "sample_count": len(self.samples),
            "avg_gpu_util_percent": sum(sample["gpu_util_percent"] for sample in self.samples) / len(self.samples),
            "max_gpu_util_percent": max(sample["gpu_util_percent"] for sample in self.samples),
            "avg_mem_util_percent": sum(sample["mem_util_percent"] for sample in self.samples) / len(self.samples),
            "max_mem_util_percent": max(sample["mem_util_percent"] for sample in self.samples),
            "max_memory_used_mib": max(sample["memory_used_mib"] for sample in self.samples),
        }


def make_batch(words_file: Path, corpus_bin: Path, lang_list: Path, lang: str, batch_size: int):
    _, src_to_id = load_vocab(corpus_bin / "dict.en.txt", lang_list)
    base_words = read_words(words_file)
    if not base_words:
        raise SystemExit(f"No words found in {words_file}")
    words = [base_words[index % len(base_words)] for index in range(batch_size)]
    preprocessed = preprocess_words(words, lang)
    source_ids = [encode_preprocessed(row, src_to_id) for row in preprocessed]
    encoder_input_ids = pad_rows(source_ids, PAD_ID)
    decoder_input_ids = torch.full((len(words), 1), EOS_ID, dtype=torch.int32)
    attention_mask = attention_mask_from_padded(encoder_input_ids, PAD_ID)
    return encoder_input_ids, decoder_input_ids, attention_mask


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark direct TRT-LLM generation.")
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus-bin", type=Path, default=DEFAULT_CORPUS_BIN)
    parser.add_argument("--lang-list", type=Path, default=DEFAULT_LANG_LIST)
    parser.add_argument("--words-file", type=Path, default=DEFAULT_WORDS_FILE)
    parser.add_argument("--lang", default="hi")
    parser.add_argument("--batches", default="64,128,192,256")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--sample-interval", type=float, default=0.1)
    args = parser.parse_args()

    from tensorrt_llm.runtime import EncDecModelRunner

    runner = EncDecModelRunner.from_engine("rank0.engine", str(args.engine_dir))
    results = []
    for batch_size in [int(piece) for piece in args.batches.split(",") if piece.strip()]:
        encoder_input_ids, decoder_input_ids, attention_mask = make_batch(
            args.words_file,
            args.corpus_bin,
            args.lang_list,
            args.lang,
            batch_size,
        )

        for _ in range(args.warmup):
            runner.generate(
                encoder_input_ids,
                decoder_input_ids,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.beam_width,
                pad_token_id=PAD_ID,
                eos_token_id=EOS_ID,
                bos_token_id=BOS_ID,
                attention_mask=attention_mask,
                return_dict=False,
            )
        torch.cuda.synchronize()

        iteration_times = []
        with GpuSampler(args.sample_interval) as sampler:
            started = time.perf_counter()
            for _ in range(args.iterations):
                iter_started = time.perf_counter()
                runner.generate(
                    encoder_input_ids,
                    decoder_input_ids,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.beam_width,
                    pad_token_id=PAD_ID,
                    eos_token_id=EOS_ID,
                    bos_token_id=BOS_ID,
                    attention_mask=attention_mask,
                    return_dict=False,
                )
                torch.cuda.synchronize()
                iteration_times.append(time.perf_counter() - iter_started)
            elapsed = time.perf_counter() - started

        item_count = batch_size * args.iterations
        result = {
            "batch_size": batch_size,
            "iterations": args.iterations,
            "elapsed_seconds": elapsed,
            "items_per_second": item_count / elapsed if elapsed else None,
            "avg_iteration_ms": 1000 * sum(iteration_times) / len(iteration_times),
            "min_iteration_ms": 1000 * min(iteration_times),
            "max_iteration_ms": 1000 * max(iteration_times),
            "gpu": sampler.summary(),
        }
        results.append(result)
        print(
            f"batch={batch_size} items/s={result['items_per_second']:.2f} "
            f"avg_iter_ms={result['avg_iteration_ms']:.2f} "
            f"avg_gpu={result['gpu']['avg_gpu_util_percent']}"
        )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "engine_dir": str(args.engine_dir),
        "beam_width": args.beam_width,
        "max_new_tokens": args.max_new_tokens,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
