#!/usr/bin/env python3
"""
Benchmark Fairseq batch decoding throughput for IndicXlit.

Run from the repository root, for example:

    .venv/bin/python "inference experiment/benchmark_batch_decode.py"

The benchmark calls the internal Fairseq Transliterator with batches of words.
It does not start the Flask server and does not measure HTTP overhead.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


DEFAULT_ROMAN_WORDS = [
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

DEFAULT_INDIC_WORDS = {
    "hi": [
        "\u092d\u093e\u0930\u0924",
        "\u0928\u092e\u0938\u094d\u0924\u0947",
        "\u0926\u093f\u0932\u094d\u0932\u0940",
        "\u0915\u0930\u094d\u0928\u093e\u091f\u0915",
        "\u0915\u0902\u092a\u094d\u092f\u0942\u091f\u0930",
        "\u0938\u094d\u0915\u0942\u0932",
        "\u092e\u094b\u092c\u093e\u0907\u0932",
        "\u092d\u093e\u0937\u093e",
    ],
}


def parse_batch_sizes(value: str) -> list[int]:
    sizes = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        size = int(part)
        if size <= 0:
            raise argparse.ArgumentTypeError("batch sizes must be positive")
        sizes.append(size)
    if not sizes:
        raise argparse.ArgumentTypeError("at least one batch size is required")
    return sizes


def load_words(args: argparse.Namespace) -> list[str]:
    if args.words_file:
        words = [
            line.strip()
            for line in Path(args.words_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif args.words:
        words = [word.strip() for word in args.words.split(",") if word.strip()]
    elif args.direction == "en2indic":
        words = DEFAULT_ROMAN_WORDS
    else:
        words = DEFAULT_INDIC_WORDS.get(args.lang, [])

    if not words:
        raise SystemExit(
            "No benchmark words available. Pass --words or --words-file for this direction/lang."
        )
    return words


def make_engine(args: argparse.Namespace):
    from ai4bharat.transliteration.xlit_src import XlitEngine

    if args.direction == "en2indic":
        return XlitEngine(
            lang2use=args.lang,
            beam_width=args.beam_width,
            rescore=args.rescore,
            model_type="transformer",
            src_script_type="roman",
        )

    return XlitEngine(
        beam_width=args.beam_width,
        rescore=args.rescore,
        model_type="transformer",
        src_script_type="indic",
    )


def process_snapshot(process):
    try:
        memory = process.memory_info()
        return {
            "rss_mb": memory.rss / 1024 / 1024,
            "vms_mb": memory.vms / 1024 / 1024,
            "threads": process.num_threads(),
        }
    except Exception:
        return {"rss_mb": 0.0, "vms_mb": 0.0, "threads": 0}


def cpu_seconds(process) -> float:
    times = process.cpu_times()
    return float(times.user + times.system)


def torch_gpu_snapshot(torch_module) -> dict[str, float | int | str]:
    if not torch_module.cuda.is_available():
        return {
            "cuda_available": 0,
            "cuda_device_count": 0,
            "cuda_current_device": "",
            "torch_gpu_allocated_mb": 0.0,
            "torch_gpu_reserved_mb": 0.0,
            "torch_gpu_max_allocated_mb": 0.0,
            "torch_gpu_max_reserved_mb": 0.0,
        }

    current_device = torch_module.cuda.current_device()
    return {
        "cuda_available": 1,
        "cuda_device_count": torch_module.cuda.device_count(),
        "cuda_current_device": current_device,
        "torch_gpu_allocated_mb": torch_module.cuda.memory_allocated(current_device) / 1024 / 1024,
        "torch_gpu_reserved_mb": torch_module.cuda.memory_reserved(current_device) / 1024 / 1024,
        "torch_gpu_max_allocated_mb": torch_module.cuda.max_memory_allocated(current_device) / 1024 / 1024,
        "torch_gpu_max_reserved_mb": torch_module.cuda.max_memory_reserved(current_device) / 1024 / 1024,
    }


def nvidia_smi_snapshot() -> dict[str, float | int | str]:
    query = (
        "index,name,utilization.gpu,utilization.memory,"
        "memory.used,memory.total,power.draw,temperature.gpu"
    )
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return {
            "nvidia_smi_available": 0,
            "gpu_count": 0,
            "gpu_util_percent_max": 0.0,
            "gpu_mem_util_percent_max": 0.0,
            "gpu_mem_used_mb_total": 0.0,
            "gpu_mem_total_mb_total": 0.0,
            "gpu_power_watts_total": 0.0,
            "gpu_temperature_c_max": 0.0,
            "gpu_names": "",
        }

    rows = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 8:
            continue
        index, name, gpu_util, mem_util, mem_used, mem_total, power, temp = parts
        try:
            rows.append(
                {
                    "index": index,
                    "name": name,
                    "gpu_util": float(gpu_util),
                    "mem_util": float(mem_util),
                    "mem_used": float(mem_used),
                    "mem_total": float(mem_total),
                    "power": 0.0 if power in {"[Not Supported]", "N/A"} else float(power),
                    "temp": float(temp),
                }
            )
        except ValueError:
            continue

    if not rows:
        return {
            "nvidia_smi_available": 1,
            "gpu_count": 0,
            "gpu_util_percent_max": 0.0,
            "gpu_mem_util_percent_max": 0.0,
            "gpu_mem_used_mb_total": 0.0,
            "gpu_mem_total_mb_total": 0.0,
            "gpu_power_watts_total": 0.0,
            "gpu_temperature_c_max": 0.0,
            "gpu_names": "",
        }

    return {
        "nvidia_smi_available": 1,
        "gpu_count": len(rows),
        "gpu_util_percent_max": max(row["gpu_util"] for row in rows),
        "gpu_mem_util_percent_max": max(row["mem_util"] for row in rows),
        "gpu_mem_used_mb_total": sum(row["mem_used"] for row in rows),
        "gpu_mem_total_mb_total": sum(row["mem_total"] for row in rows),
        "gpu_power_watts_total": sum(row["power"] for row in rows),
        "gpu_temperature_c_max": max(row["temp"] for row in rows),
        "gpu_names": "|".join(row["name"] for row in rows),
    }


def build_batch(words: list[str], batch_size: int) -> list[str]:
    return [words[i % len(words)] for i in range(batch_size)]


def benchmark_batch(args, engine, process, torch_module, words, batch_size: int) -> dict[str, float | int | str]:
    src_lang = "en" if args.direction == "en2indic" else args.lang
    tgt_lang = args.lang if args.direction == "en2indic" else "en"
    batch_words = build_batch(words, batch_size)
    repeats = max(args.min_repeats, math.ceil(args.target_items / batch_size))

    for _ in range(args.warmup):
        inputs = engine.pre_process(batch_words, src_lang, tgt_lang)
        raw = engine.transliterator.translate(inputs, nbest=args.topk)
        if args.postprocess:
            engine.post_process(raw, tgt_lang)

    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
        torch_module.cuda.reset_peak_memory_stats()

    snapshot_before = process_snapshot(process)
    gpu_before = torch_gpu_snapshot(torch_module)
    smi_before = nvidia_smi_snapshot()
    cpu_start = cpu_seconds(process)
    wall_start = time.perf_counter()

    for _ in range(repeats):
        inputs = engine.pre_process(batch_words, src_lang, tgt_lang)
        raw = engine.transliterator.translate(inputs, nbest=args.topk)
        if args.postprocess:
            engine.post_process(raw, tgt_lang)

    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()

    wall_seconds = time.perf_counter() - wall_start
    cpu_used = cpu_seconds(process) - cpu_start
    snapshot_after = process_snapshot(process)
    gpu_after = torch_gpu_snapshot(torch_module)
    smi_after = nvidia_smi_snapshot()
    total_items = batch_size * repeats

    return {
        "direction": args.direction,
        "lang": args.lang,
        "batch_size": batch_size,
        "repeats": repeats,
        "total_items": total_items,
        "topk": args.topk,
        "beam_width": args.beam_width,
        "postprocess": int(args.postprocess),
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_used,
        "items_per_second": total_items / wall_seconds if wall_seconds else 0.0,
        "wall_ms_per_item": (wall_seconds / total_items) * 1000 if total_items else 0.0,
        "cpu_ms_per_item": (cpu_used / total_items) * 1000 if total_items else 0.0,
        "effective_cpu_cores": cpu_used / wall_seconds if wall_seconds else 0.0,
        "rss_mb_before": snapshot_before["rss_mb"],
        "rss_mb_after": snapshot_after["rss_mb"],
        "vms_mb_after": snapshot_after["vms_mb"],
        "threads_after": snapshot_after["threads"],
        "cuda_available": gpu_after["cuda_available"],
        "cuda_device_count": gpu_after["cuda_device_count"],
        "cuda_current_device": gpu_after["cuda_current_device"],
        "torch_gpu_allocated_mb_before": gpu_before["torch_gpu_allocated_mb"],
        "torch_gpu_allocated_mb_after": gpu_after["torch_gpu_allocated_mb"],
        "torch_gpu_reserved_mb_after": gpu_after["torch_gpu_reserved_mb"],
        "torch_gpu_max_allocated_mb": gpu_after["torch_gpu_max_allocated_mb"],
        "torch_gpu_max_reserved_mb": gpu_after["torch_gpu_max_reserved_mb"],
        "nvidia_smi_available": smi_after["nvidia_smi_available"],
        "gpu_count": smi_after["gpu_count"],
        "gpu_util_percent_before": smi_before["gpu_util_percent_max"],
        "gpu_util_percent_after": smi_after["gpu_util_percent_max"],
        "gpu_mem_util_percent_after": smi_after["gpu_mem_util_percent_max"],
        "gpu_mem_used_mb_before": smi_before["gpu_mem_used_mb_total"],
        "gpu_mem_used_mb_after": smi_after["gpu_mem_used_mb_total"],
        "gpu_mem_total_mb": smi_after["gpu_mem_total_mb_total"],
        "gpu_power_watts_after": smi_after["gpu_power_watts_total"],
        "gpu_temperature_c_after": smi_after["gpu_temperature_c_max"],
        "gpu_names": smi_after["gpu_names"],
    }


def format_row(row: dict[str, float | int | str]) -> str:
    return (
        f"batch={row['batch_size']:>4} "
        f"items/s={row['items_per_second']:>8.2f} "
        f"wall_ms/item={row['wall_ms_per_item']:>7.2f} "
        f"cpu_ms/item={row['cpu_ms_per_item']:>7.2f} "
        f"cores={row['effective_cpu_cores']:>5.2f} "
        f"rss_mb={row['rss_mb_after']:>7.1f} "
        f"torch_gpu_mb={row['torch_gpu_allocated_mb_after']:>7.1f} "
        f"smi_gpu_mb={row['gpu_mem_used_mb_after']:>7.1f} "
        f"gpu_util%={row['gpu_util_percent_after']:>5.1f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark IndicXlit Fairseq batch decoding.")
    parser.add_argument("--direction", choices=["en2indic", "indic2en"], default="en2indic")
    parser.add_argument("--lang", default="hi", help="Language code, for example hi, ta, bn.")
    parser.add_argument("--batch-sizes", type=parse_batch_sizes, default=parse_batch_sizes("1,4,8,16,32,64,128,256,512"))
    parser.add_argument("--target-items", type=int, default=512, help="Minimum items to decode per batch-size row.")
    parser.add_argument("--min-repeats", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--rescore", action="store_true", help="Enable dictionary rescoring. This can use a lot of RAM.")
    parser.add_argument("--no-postprocess", dest="postprocess", action="store_false", help="Skip Fairseq text post-processing parse.")
    parser.set_defaults(postprocess=True)
    parser.add_argument("--words", help="Comma-separated benchmark words.")
    parser.add_argument("--words-file", help="UTF-8 file containing one benchmark word per line.")
    parser.add_argument("--torch-threads", type=int, help="Set torch.set_num_threads before loading the model.")
    parser.add_argument("--csv", help="Optional path to write CSV results.")
    args = parser.parse_args()

    if args.target_items <= 0:
        raise SystemExit("--target-items must be positive")
    if args.min_repeats <= 0:
        raise SystemExit("--min-repeats must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup cannot be negative")

    import psutil
    import torch

    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)

    process = psutil.Process(os.getpid())
    words = load_words(args)

    print("Loading model...", flush=True)
    load_start = time.perf_counter()
    engine = make_engine(args)
    load_seconds = time.perf_counter() - load_start
    loaded = process_snapshot(process)
    loaded_gpu = torch_gpu_snapshot(torch)
    loaded_smi = nvidia_smi_snapshot()
    print(
        f"Loaded direction={args.direction} lang={args.lang} "
        f"in {load_seconds:.2f}s rss_mb={loaded['rss_mb']:.1f} "
        f"threads={loaded['threads']} "
        f"cuda={loaded_gpu['cuda_available']} "
        f"torch_gpu_mb={loaded_gpu['torch_gpu_allocated_mb']:.1f} "
        f"smi_gpu_mb={loaded_smi['gpu_mem_used_mb_total']:.1f}",
        flush=True,
    )

    rows = []
    for batch_size in args.batch_sizes:
        row = benchmark_batch(args, engine, process, torch, words, batch_size)
        rows.append(row)
        print(format_row(row), flush=True)

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote CSV: {csv_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
