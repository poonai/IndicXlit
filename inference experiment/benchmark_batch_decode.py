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
import concurrent.futures
import csv
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
TRTLLM_PORT_DIR = REPO_ROOT / "inference experiment" / "trtllm_port"
if str(TRTLLM_PORT_DIR) not in sys.path:
    sys.path.insert(0, str(TRTLLM_PORT_DIR))


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


class TRTLLMBenchmarkEngine:
    def __init__(self, args: argparse.Namespace):
        from tensorrt_llm.runtime import EncDecModelRunner

        from inspect_indicxlit_checkpoint import DEFAULT_CORPUS_BIN, DEFAULT_LANG_LIST
        from run_trtllm_greedy import (
            EOS_ID,
            PAD_ID,
            BOS_ID,
            decode_ids,
            encode_preprocessed,
            extract_sequences,
            load_vocab,
            normalize_output,
            pad_rows,
            postprocess_raw,
            preprocess_words,
        )

        self.lang = args.lang
        self.max_new_tokens = args.max_new_tokens
        self.pad_id = PAD_ID
        self.eos_id = EOS_ID
        self.bos_id = BOS_ID
        self.decode_ids = decode_ids
        self.encode_preprocessed = encode_preprocessed
        self.extract_sequences = extract_sequences
        self.normalize_output = normalize_output
        self.pad_rows = pad_rows
        self.postprocess_raw = postprocess_raw
        self.preprocess_words = preprocess_words

        corpus_bin = args.corpus_bin or DEFAULT_CORPUS_BIN
        lang_list = args.lang_list or DEFAULT_LANG_LIST
        _, self.src_to_id = load_vocab(corpus_bin / "dict.en.txt", lang_list)
        self.tgt_symbols, _ = load_vocab(corpus_bin / f"dict.{args.lang}.txt", lang_list)
        self.runner = EncDecModelRunner.from_engine("rank0.engine", str(args.trtllm_engine_dir))

    def pre_process(self, words: list[str], src_lang: str, tgt_lang: str) -> dict[str, object]:
        import torch

        del src_lang, tgt_lang
        preprocessed = self.preprocess_words(words, self.lang)
        source_ids = [self.encode_preprocessed(row, self.src_to_id) for row in preprocessed]
        return {
            "words": words,
            "preprocessed": preprocessed,
            "source_ids": source_ids,
            "encoder_input_ids": self.pad_rows(source_ids, self.pad_id),
            "decoder_input_ids": torch.full((len(words), 1), self.eos_id, dtype=torch.int32),
        }

    def translate(self, inputs: dict[str, object]):
        return self.runner.generate(
            inputs["encoder_input_ids"],
            inputs["decoder_input_ids"],
            max_new_tokens=self.max_new_tokens,
            num_beams=1,
            pad_token_id=self.pad_id,
            eos_token_id=self.eos_id,
            bos_token_id=self.bos_id,
            return_dict=False,
        )

    def post_process(self, output, inputs: dict[str, object]) -> list[str]:
        output_tensor = self.normalize_output(output)
        beam_sequences = self.extract_sequences(
            output_tensor,
            len(inputs["words"]),
            inputs["decoder_input_ids"].shape[1],
        )
        decoded = []
        for sequences in beam_sequences:
            sequence = sequences[0] if sequences else []
            _, raw = self.decode_ids(sequence, self.tgt_symbols)
            decoded.append(self.postprocess_raw(raw))
        return decoded


def make_trtllm_engine(args: argparse.Namespace) -> TRTLLMBenchmarkEngine:
    if args.direction != "en2indic":
        raise SystemExit("TensorRT-LLM benchmark currently supports --direction en2indic only")
    if args.model_instances != 1:
        raise SystemExit("TensorRT-LLM benchmark supports one model instance only")
    if args.beam_width != 1 or args.topk != 1:
        raise SystemExit("TensorRT-LLM benchmark currently supports greedy only: pass --beam-width 1 --topk 1")
    return TRTLLMBenchmarkEngine(args)


def apply_torch_compile(args: argparse.Namespace, engine, torch_module) -> list[str]:
    if args.torch_compile == "none":
        return []
    if not hasattr(torch_module, "compile"):
        raise SystemExit("This PyTorch build does not expose torch.compile")

    import torch._dynamo

    torch._dynamo.config.suppress_errors = args.torch_compile_suppress_errors
    compile_kwargs = {
        "mode": args.torch_compile_mode,
        "fullgraph": args.torch_compile_fullgraph,
        "dynamic": args.torch_compile_dynamic,
    }
    compiled = []

    def compile_attr(owner, attr_name, label):
        original = getattr(owner, attr_name, None)
        if original is None:
            return
        setattr(owner, attr_name, torch_module.compile(original, **compile_kwargs))
        compiled.append(label)

    models = getattr(engine.transliterator, "models", [])
    for idx, model in enumerate(models):
        if args.torch_compile in {"encoder", "encoder-decoder", "all"} and hasattr(model, "encoder"):
            compile_attr(model.encoder, "forward_torchscript", f"model{idx}.encoder.forward_torchscript")
        if args.torch_compile in {"decoder", "encoder-decoder", "all"} and hasattr(model, "decoder"):
            compile_attr(model.decoder, "forward", f"model{idx}.decoder.forward")
        if args.torch_compile in {"model", "all"}:
            compile_attr(model, "get_normalized_probs", f"model{idx}.get_normalized_probs")

    if args.torch_compile in {"generator", "all"}:
        compile_attr(engine.transliterator.generator, "forward", "generator.forward")
        compile_attr(engine.transliterator.generator, "_generate", "generator._generate")

    return compiled


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
    preprocess_seconds = 0.0
    inference_seconds = 0.0
    postprocess_seconds = 0.0

    for _ in range(repeats):
        step_start = time.perf_counter()
        inputs = engine.pre_process(batch_words, src_lang, tgt_lang)
        preprocess_seconds += time.perf_counter() - step_start

        step_start = time.perf_counter()
        raw = engine.transliterator.translate(inputs, nbest=args.topk)
        if torch_module.cuda.is_available():
            torch_module.cuda.synchronize()
        inference_seconds += time.perf_counter() - step_start

        if args.postprocess:
            step_start = time.perf_counter()
            engine.post_process(raw, tgt_lang)
            postprocess_seconds += time.perf_counter() - step_start

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
        "model_instances": 1,
        "batch_size": batch_size,
        "repeats": repeats,
        "total_items": total_items,
        "topk": args.topk,
        "beam_width": args.beam_width,
        "postprocess": int(args.postprocess),
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_used,
        "preprocess_seconds": preprocess_seconds,
        "inference_seconds": inference_seconds,
        "postprocess_seconds": postprocess_seconds,
        "items_per_second": total_items / wall_seconds if wall_seconds else 0.0,
        "items_per_second_per_instance": total_items / wall_seconds if wall_seconds else 0.0,
        "wall_ms_per_item": (wall_seconds / total_items) * 1000 if total_items else 0.0,
        "preprocess_ms_per_item": (preprocess_seconds / total_items) * 1000 if total_items else 0.0,
        "inference_ms_per_item": (inference_seconds / total_items) * 1000 if total_items else 0.0,
        "postprocess_ms_per_item": (postprocess_seconds / total_items) * 1000 if total_items else 0.0,
        "preprocess_percent": (preprocess_seconds / wall_seconds) * 100 if wall_seconds else 0.0,
        "inference_percent": (inference_seconds / wall_seconds) * 100 if wall_seconds else 0.0,
        "postprocess_percent": (postprocess_seconds / wall_seconds) * 100 if wall_seconds else 0.0,
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


def benchmark_trtllm_batch(args, engine, process, torch_module, words, batch_size: int) -> dict[str, float | int | str]:
    src_lang = "en"
    tgt_lang = args.lang
    batch_words = build_batch(words, batch_size)
    repeats = max(args.min_repeats, math.ceil(args.target_items / batch_size))

    for _ in range(args.warmup):
        inputs = engine.pre_process(batch_words, src_lang, tgt_lang)
        output = engine.translate(inputs)
        if torch_module.cuda.is_available():
            torch_module.cuda.synchronize()
        if args.postprocess:
            engine.post_process(output, inputs)

    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
        torch_module.cuda.reset_peak_memory_stats()

    snapshot_before = process_snapshot(process)
    gpu_before = torch_gpu_snapshot(torch_module)
    smi_before = nvidia_smi_snapshot()
    cpu_start = cpu_seconds(process)
    wall_start = time.perf_counter()
    preprocess_seconds = 0.0
    inference_seconds = 0.0
    postprocess_seconds = 0.0

    for _ in range(repeats):
        step_start = time.perf_counter()
        inputs = engine.pre_process(batch_words, src_lang, tgt_lang)
        preprocess_seconds += time.perf_counter() - step_start

        step_start = time.perf_counter()
        output = engine.translate(inputs)
        if torch_module.cuda.is_available():
            torch_module.cuda.synchronize()
        inference_seconds += time.perf_counter() - step_start

        if args.postprocess:
            step_start = time.perf_counter()
            engine.post_process(output, inputs)
            postprocess_seconds += time.perf_counter() - step_start

    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()

    wall_seconds = time.perf_counter() - wall_start
    cpu_used = cpu_seconds(process) - cpu_start
    snapshot_after = process_snapshot(process)
    gpu_after = torch_gpu_snapshot(torch_module)
    smi_after = nvidia_smi_snapshot()
    total_items = batch_size * repeats

    return {
        "backend": "trtllm",
        "direction": args.direction,
        "lang": args.lang,
        "model_instances": 1,
        "batch_size": batch_size,
        "repeats": repeats,
        "total_items": total_items,
        "topk": args.topk,
        "beam_width": args.beam_width,
        "postprocess": int(args.postprocess),
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_used,
        "preprocess_seconds": preprocess_seconds,
        "inference_seconds": inference_seconds,
        "postprocess_seconds": postprocess_seconds,
        "items_per_second": total_items / wall_seconds if wall_seconds else 0.0,
        "items_per_second_per_instance": total_items / wall_seconds if wall_seconds else 0.0,
        "wall_ms_per_item": (wall_seconds / total_items) * 1000 if total_items else 0.0,
        "preprocess_ms_per_item": (preprocess_seconds / total_items) * 1000 if total_items else 0.0,
        "inference_ms_per_item": (inference_seconds / total_items) * 1000 if total_items else 0.0,
        "postprocess_ms_per_item": (postprocess_seconds / total_items) * 1000 if total_items else 0.0,
        "preprocess_percent": (preprocess_seconds / wall_seconds) * 100 if wall_seconds else 0.0,
        "inference_percent": (inference_seconds / wall_seconds) * 100 if wall_seconds else 0.0,
        "postprocess_percent": (postprocess_seconds / wall_seconds) * 100 if wall_seconds else 0.0,
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


def run_thread_worker(
    args,
    engine,
    torch_module,
    words: list[str],
    batch_size: int,
    repeats: int,
    start_barrier: threading.Barrier,
) -> dict[str, float | int]:
    src_lang = "en" if args.direction == "en2indic" else args.lang
    tgt_lang = args.lang if args.direction == "en2indic" else "en"
    batch_words = build_batch(words, batch_size)

    for _ in range(args.warmup):
        inputs = engine.pre_process(batch_words, src_lang, tgt_lang)
        raw = engine.transliterator.translate(inputs, nbest=args.topk)
        if args.postprocess:
            engine.post_process(raw, tgt_lang)

    start_barrier.wait()

    wall_start = time.perf_counter()
    preprocess_seconds = 0.0
    inference_seconds = 0.0
    postprocess_seconds = 0.0

    for _ in range(repeats):
        step_start = time.perf_counter()
        inputs = engine.pre_process(batch_words, src_lang, tgt_lang)
        preprocess_seconds += time.perf_counter() - step_start

        step_start = time.perf_counter()
        raw = engine.transliterator.translate(inputs, nbest=args.topk)
        inference_seconds += time.perf_counter() - step_start

        if args.postprocess:
            step_start = time.perf_counter()
            engine.post_process(raw, tgt_lang)
            postprocess_seconds += time.perf_counter() - step_start

    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()

    return {
        "wall_seconds": time.perf_counter() - wall_start,
        "preprocess_seconds": preprocess_seconds,
        "inference_seconds": inference_seconds,
        "postprocess_seconds": postprocess_seconds,
        "total_items": batch_size * repeats,
    }


def benchmark_multi_instance_batch(
    args,
    engines,
    process,
    torch_module,
    words,
    batch_size: int,
) -> dict[str, float | int | str]:
    model_instances = len(engines)
    repeats = max(args.min_repeats, math.ceil(args.target_items / batch_size))

    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
        torch_module.cuda.reset_peak_memory_stats()

    snapshot_before = process_snapshot(process)
    gpu_before = torch_gpu_snapshot(torch_module)
    smi_before = nvidia_smi_snapshot()
    start_barrier = threading.Barrier(model_instances + 1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=model_instances) as executor:
        futures = [
            executor.submit(
                run_thread_worker,
                args,
                engine,
                torch_module,
                words,
                batch_size,
                repeats,
                start_barrier,
            )
            for engine in engines
        ]

        start_barrier.wait()
        cpu_start = cpu_seconds(process)
        wall_start = time.perf_counter()
        worker_rows = [future.result() for future in futures]

    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()

    wall_seconds = time.perf_counter() - wall_start
    cpu_used = cpu_seconds(process) - cpu_start
    snapshot_after = process_snapshot(process)
    gpu_after = torch_gpu_snapshot(torch_module)
    smi_after = nvidia_smi_snapshot()
    total_items = sum(row["total_items"] for row in worker_rows)
    preprocess_seconds = sum(row["preprocess_seconds"] for row in worker_rows)
    inference_seconds = sum(row["inference_seconds"] for row in worker_rows)
    postprocess_seconds = sum(row["postprocess_seconds"] for row in worker_rows)
    worker_wall_seconds = [row["wall_seconds"] for row in worker_rows]

    return {
        "direction": args.direction,
        "lang": args.lang,
        "model_instances": model_instances,
        "batch_size": batch_size,
        "repeats": repeats,
        "total_items": total_items,
        "topk": args.topk,
        "beam_width": args.beam_width,
        "postprocess": int(args.postprocess),
        "wall_seconds": wall_seconds,
        "worker_wall_seconds_min": min(worker_wall_seconds) if worker_wall_seconds else 0.0,
        "worker_wall_seconds_max": max(worker_wall_seconds) if worker_wall_seconds else 0.0,
        "cpu_seconds": cpu_used,
        "preprocess_seconds": preprocess_seconds,
        "inference_seconds": inference_seconds,
        "postprocess_seconds": postprocess_seconds,
        "items_per_second": total_items / wall_seconds if wall_seconds else 0.0,
        "items_per_second_per_instance": (
            total_items / wall_seconds / model_instances if wall_seconds and model_instances else 0.0
        ),
        "wall_ms_per_item": (wall_seconds / total_items) * 1000 if total_items else 0.0,
        "preprocess_ms_per_item": (preprocess_seconds / total_items) * 1000 if total_items else 0.0,
        "inference_ms_per_item": (inference_seconds / total_items) * 1000 if total_items else 0.0,
        "postprocess_ms_per_item": (postprocess_seconds / total_items) * 1000 if total_items else 0.0,
        "preprocess_percent": (preprocess_seconds / wall_seconds) * 100 if wall_seconds else 0.0,
        "inference_percent": (inference_seconds / wall_seconds) * 100 if wall_seconds else 0.0,
        "postprocess_percent": (postprocess_seconds / wall_seconds) * 100 if wall_seconds else 0.0,
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
    instances = int(row.get("model_instances", 1))
    instance_text = f"instances={instances:>2} " if instances > 1 else ""
    per_instance_text = (
        f"per_inst_items/s={row['items_per_second_per_instance']:>8.2f} "
        if instances > 1
        else ""
    )
    return (
        f"{instance_text}"
        f"batch={row['batch_size']:>4} "
        f"items/s={row['items_per_second']:>8.2f} "
        f"{per_instance_text}"
        f"wall_ms/item={row['wall_ms_per_item']:>7.2f} "
        f"wall_s={row['wall_seconds']:>7.2f} "
        f"pre_ms/item={row['preprocess_ms_per_item']:>6.2f} "
        f"inf_ms/item={row['inference_ms_per_item']:>6.2f} "
        f"post_ms/item={row['postprocess_ms_per_item']:>6.2f} "
        f"cpu_ms/item={row['cpu_ms_per_item']:>7.2f} "
        f"cores={row['effective_cpu_cores']:>5.2f} "
        f"rss_mb={row['rss_mb_after']:>7.1f} "
        f"torch_gpu_mb={row['torch_gpu_allocated_mb_after']:>7.1f} "
        f"smi_gpu_mb={row['gpu_mem_used_mb_after']:>7.1f} "
        f"gpu_util%={row['gpu_util_percent_after']:>5.1f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark IndicXlit batch decoding.")
    parser.add_argument("--backend", choices=["fairseq", "trtllm"], default="fairseq")
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
    parser.add_argument(
        "--model-instances",
        type=int,
        default=1,
        help="Load this many independent model instances and run one benchmark thread per instance.",
    )
    parser.add_argument(
        "--torch-compile",
        choices=["none", "encoder", "decoder", "encoder-decoder", "model", "generator", "all"],
        default="none",
        help="Experimentally wrap Fairseq callables with torch.compile.",
    )
    parser.add_argument("--torch-compile-mode", default="reduce-overhead")
    parser.add_argument("--torch-compile-fullgraph", action="store_true")
    parser.add_argument("--torch-compile-dynamic", action="store_true")
    parser.add_argument(
        "--no-torch-compile-suppress-errors",
        dest="torch_compile_suppress_errors",
        action="store_false",
        help="Let torch.compile failures raise instead of falling back.",
    )
    parser.set_defaults(torch_compile_suppress_errors=True)
    parser.add_argument("--csv", help="Optional path to write CSV results.")
    parser.add_argument(
        "--trtllm-engine-dir",
        type=Path,
        default=TRTLLM_PORT_DIR / "artifacts" / "trtllm_engines_en_hi",
        help="TensorRT-LLM engine directory for --backend trtllm.",
    )
    parser.add_argument("--corpus-bin", type=Path, help="Corpus-bin directory for --backend trtllm.")
    parser.add_argument("--lang-list", type=Path, help="Language token list for --backend trtllm.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    if args.target_items <= 0:
        raise SystemExit("--target-items must be positive")
    if args.min_repeats <= 0:
        raise SystemExit("--min-repeats must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup cannot be negative")
    if args.model_instances <= 0:
        raise SystemExit("--model-instances must be positive")

    import psutil
    import torch

    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)

    process = psutil.Process(os.getpid())
    words = load_words(args)

    print(f"Loading {args.model_instances} model instance(s) for backend={args.backend}...", flush=True)
    load_start = time.perf_counter()
    engines = []
    compiled_by_engine = []
    if args.backend == "trtllm":
        engine = make_trtllm_engine(args)
        engines.append(engine)
        compiled_by_engine.append([])
    else:
        for index in range(args.model_instances):
            engine = make_engine(args)
            compiled = apply_torch_compile(args, engine, torch)
            engines.append(engine)
            compiled_by_engine.append(compiled)
            if args.model_instances > 1:
                loaded = process_snapshot(process)
                loaded_gpu = torch_gpu_snapshot(torch)
                loaded_smi = nvidia_smi_snapshot()
                print(
                    f"Loaded instance={index + 1}/{args.model_instances} "
                    f"rss_mb={loaded['rss_mb']:.1f} "
                    f"torch_gpu_mb={loaded_gpu['torch_gpu_allocated_mb']:.1f} "
                    f"smi_gpu_mb={loaded_smi['gpu_mem_used_mb_total']:.1f} "
                    f"compiled={','.join(compiled) if compiled else 'none'}",
                    flush=True,
                )
    load_seconds = time.perf_counter() - load_start
    loaded = process_snapshot(process)
    loaded_gpu = torch_gpu_snapshot(torch)
    loaded_smi = nvidia_smi_snapshot()
    compiled_labels = sorted({label for compiled in compiled_by_engine for label in compiled})
    print(
        f"Loaded backend={args.backend} direction={args.direction} lang={args.lang} "
        f"in {load_seconds:.2f}s rss_mb={loaded['rss_mb']:.1f} "
        f"threads={loaded['threads']} "
        f"cuda={loaded_gpu['cuda_available']} "
        f"torch_gpu_mb={loaded_gpu['torch_gpu_allocated_mb']:.1f} "
        f"smi_gpu_mb={loaded_smi['gpu_mem_used_mb_total']:.1f} "
        f"compiled={','.join(compiled_labels) if compiled_labels else 'none'}",
        flush=True,
    )

    rows = []
    for batch_size in args.batch_sizes:
        if args.backend == "trtllm":
            row = benchmark_trtllm_batch(args, engines[0], process, torch, words, batch_size)
        elif args.model_instances == 1:
            row = benchmark_batch(args, engines[0], process, torch, words, batch_size)
        else:
            row = benchmark_multi_instance_batch(args, engines, process, torch, words, batch_size)
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
