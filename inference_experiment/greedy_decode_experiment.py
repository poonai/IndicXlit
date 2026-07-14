#!/usr/bin/env python3
"""
Experimental greedy decoder for IndicXlit's Fairseq transformer model.

This script intentionally lives outside the app and outside site-packages. It is
a throwaway workspace for testing whether a specialized beam=1 decoder can beat
Fairseq's generic SequenceGenerator path.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


DEFAULT_WORDS = [
    "bharat",
    "namaste",
    "amma",
    "delhi",
    "karnataka",
    "transliteration",
    "computer",
    "school",
    "mobile",
    "india",
    "ramayan",
    "pradesh",
    "language",
    "server",
    "request",
    "throughput",
]


def parse_batch_sizes(value: str) -> list[int]:
    sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("batch sizes must be positive integers")
    return sizes


def build_batch(words: list[str], batch_size: int) -> list[str]:
    return [words[i % len(words)] for i in range(batch_size)]


def cpu_seconds(process) -> float:
    times = process.cpu_times()
    return float(times.user + times.system)


def maybe_sync(torch_module):
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def make_incremental_states(torch_module, model) -> List[Dict[str, Dict[str, Optional[object]]]]:
    return [
        torch_module.jit.annotate(Dict[str, Dict[str, Optional[object]]], {})
        for _ in range(model.models_size)
    ]


def greedy_generate(transliterator, inputs: list[str], max_len_b: int):
    import torch
    from ai4bharat.transliteration.transformer.custom_interactive import make_batches

    generator = transliterator.generator
    model = generator.model
    cfg = transliterator.cfg
    task = transliterator.task

    outputs = [None] * len(inputs)
    for batch in make_batches(inputs, cfg, task, transliterator.max_positions, transliterator.encode_fn):
        src_tokens = batch.src_tokens
        src_lengths = batch.src_lengths
        if transliterator.use_cuda:
            src_tokens = src_tokens.cuda()
            src_lengths = src_lengths.cuda()

        net_input = {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
        }
        bsz, src_len = src_tokens.size()[:2]
        max_len = min(int(generator.max_len_a * src_len + max_len_b), generator.max_len - 1)

        encoder_outs = model.forward_encoder(net_input)
        incremental_states = make_incremental_states(torch, model)

        tokens = (
            torch.full(
                (bsz, max_len + 2),
                fill_value=generator.pad,
                dtype=torch.long,
                device=src_tokens.device,
            )
        )
        tokens[:, 0] = generator.eos
        scores = torch.zeros((bsz, max_len + 1), dtype=torch.float, device=src_tokens.device)
        finished = torch.zeros((bsz,), dtype=torch.bool, device=src_tokens.device)

        last_step = 0
        for step in range(max_len + 1):
            lprobs, _ = model.forward_decoder(
                tokens[:, : step + 1],
                encoder_outs,
                incremental_states,
                generator.temperature,
            )
            lprobs[lprobs != lprobs] = torch.tensor(-math.inf, device=lprobs.device)
            lprobs[:, generator.pad] = -math.inf
            lprobs[:, generator.unk] -= generator.unk_penalty

            if step < generator.min_len:
                lprobs[:, generator.eos] = -math.inf
            if step >= max_len:
                lprobs[:, : generator.eos] = -math.inf
                lprobs[:, generator.eos + 1 :] = -math.inf

            next_scores, next_tokens = torch.max(lprobs, dim=1)
            next_tokens = torch.where(
                finished,
                torch.full_like(next_tokens, generator.eos),
                next_tokens,
            )
            tokens[:, step + 1] = next_tokens
            scores[:, step] = next_scores
            finished = finished | next_tokens.eq(generator.eos)
            last_step = step
            if bool(finished.all()):
                break

        for row in range(bsz):
            row_tokens = tokens[row, 1 : last_step + 2]
            eos_positions = (row_tokens == generator.eos).nonzero(as_tuple=False)
            if eos_positions.numel() > 0:
                row_tokens = row_tokens[: eos_positions[0].item() + 1]
            word = transliterator.tgt_dict.string(
                row_tokens,
                cfg.common_eval.post_process,
                extra_symbols_to_ignore=generator.symbols_to_strip_from_output,
            )
            outputs[int(batch.ids[row].item())] = "".join(word.split(" "))

    return outputs


def run_mode(args, mode: str, engine, batch_words: list[str], repeats: int, process, torch_module):
    inputs = engine.pre_process(batch_words, "en", args.lang)

    for _ in range(args.warmup):
        if mode == "fairseq":
            engine.transliterator.translate(inputs, nbest=1)
        else:
            greedy_generate(engine.transliterator, inputs, args.max_len_b)
        maybe_sync(torch_module)

    cpu_start = cpu_seconds(process)
    wall_start = time.perf_counter()
    for _ in range(repeats):
        if mode == "fairseq":
            engine.transliterator.translate(inputs, nbest=1)
        else:
            greedy_generate(engine.transliterator, inputs, args.max_len_b)
        maybe_sync(torch_module)

    wall = time.perf_counter() - wall_start
    cpu_used = cpu_seconds(process) - cpu_start
    total_items = len(batch_words) * repeats
    return {
        "mode": mode,
        "batch_size": len(batch_words),
        "repeats": repeats,
        "items_per_second": total_items / wall,
        "wall_ms_per_item": (wall / total_items) * 1000,
        "cpu_ms_per_item": (cpu_used / total_items) * 1000,
        "effective_cpu_cores": cpu_used / wall if wall else 0.0,
    }


def print_row(row):
    print(
        f"mode={row['mode']:<7} "
        f"batch={row['batch_size']:>5} "
        f"items/s={row['items_per_second']:>8.2f} "
        f"wall_ms/item={row['wall_ms_per_item']:>7.2f} "
        f"cpu_ms/item={row['cpu_ms_per_item']:>7.2f} "
        f"cores={row['effective_cpu_cores']:>5.2f}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Fairseq beam=1 generation with a custom greedy decoder.")
    parser.add_argument("--lang", default="hi")
    parser.add_argument("--batch-sizes", type=parse_batch_sizes, default=parse_batch_sizes("128,512,1024"))
    parser.add_argument("--target-items", type=int, default=4096)
    parser.add_argument("--min-repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-len-b", type=int, default=32)
    parser.add_argument("--words", default=",".join(DEFAULT_WORDS))
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--mode", choices=["both", "fairseq", "greedy"], default="both")
    args = parser.parse_args()

    import psutil
    import torch
    from ai4bharat.transliteration.xlit_src import XlitEngine

    process = psutil.Process(os.getpid())
    words = [word.strip() for word in args.words.split(",") if word.strip()]

    print("Loading transformer with beam_width=1...", flush=True)
    engine = XlitEngine(args.lang, beam_width=1, rescore=False, model_type="transformer", src_script_type="roman")
    print(
        f"cuda={int(torch.cuda.is_available())} "
        f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
        flush=True,
    )

    sample_words = build_batch(words, args.sample_count)
    sample_inputs = engine.pre_process(sample_words, "en", args.lang)
    fairseq_raw = engine.transliterator.translate(sample_inputs, nbest=1)
    fairseq_words = engine.post_process(fairseq_raw, args.lang)
    greedy_words = greedy_generate(engine.transliterator, sample_inputs, args.max_len_b)
    print("sample comparison:", flush=True)
    for src, normal, greedy in zip(sample_words, fairseq_words, greedy_words):
        marker = "OK" if normal == greedy else "DIFF"
        print(f"  {marker} {src}: fairseq={normal} greedy={greedy}", flush=True)

    for batch_size in args.batch_sizes:
        repeats = max(args.min_repeats, math.ceil(args.target_items / batch_size))
        batch_words = build_batch(words, batch_size)
        if args.mode in {"both", "fairseq"}:
            print_row(run_mode(args, "fairseq", engine, batch_words, repeats, process, torch))
        if args.mode in {"both", "greedy"}:
            print_row(run_mode(args, "greedy", engine, batch_words, repeats, process, torch))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
