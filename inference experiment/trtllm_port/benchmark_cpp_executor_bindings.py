#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import importlib
import importlib.util
import json
import os
import queue
import statistics
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_ENGINE_DIR = SCRIPT_DIR / "artifacts" / "trtllm_engines_en_hi_beam5_triton_fp16_b256"
DEFAULT_MODEL_ROOT = (
    REPO_ROOT
    / "app"
    / "ai4bharat"
    / "transliteration"
    / "transformer"
    / "models"
    / "en2indic"
)
DEFAULT_DAKSHINA = (
    SCRIPT_DIR
    / "artifacts"
    / "dakshina_data"
    / "dakshina_dataset_v1.0"
    / "hi"
    / "lexicons"
    / "hi.translit.sampled.test.tsv"
)
PREPROCESS_PATH = (
    SCRIPT_DIR
    / "triton_indicxlit"
    / "model_repository"
    / "indicxlit_preprocess"
    / "1"
    / "model.py"
)
TRTLLM_SITE = REPO_ROOT / ".venv-trtllm" / "lib" / "python3.10" / "site-packages"

EOS_ID = 2
PAD_ID = 1


def ensure_ld_path() -> None:
    if os.environ.get("INDICXLIT_CPP_EXECUTOR_LD_READY") == "1":
        return
    nvidia_root = TRTLLM_SITE / "nvidia"
    lib_dirs = [
        TRTLLM_SITE / "torch" / "lib",
        TRTLLM_SITE / "tensorrt_llm" / "libs",
        TRTLLM_SITE / "tensorrt_libs",
    ]
    if nvidia_root.exists():
        lib_dirs.extend(sorted(path for path in nvidia_root.glob("*/lib") if path.is_dir()))
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(str(path) for path in lib_dirs) + (":" + existing if existing else "")
    os.environ["INDICXLIT_CPP_EXECUTOR_LD_READY"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


def load_executor_bindings():
    # Avoid executing tensorrt_llm/__init__.py, which imports a large Torch stack.
    pkg = types.ModuleType("tensorrt_llm")
    pkg.__path__ = [str(TRTLLM_SITE / "tensorrt_llm")]
    sys.modules["tensorrt_llm"] = pkg

    so_path = TRTLLM_SITE / "tensorrt_llm" / "bindings.cpython-310-x86_64-linux-gnu.so"
    spec = importlib.util.spec_from_file_location("tensorrt_llm.bindings", so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load bindings extension from {so_path}")
    bindings = importlib.util.module_from_spec(spec)
    bindings.__path__ = [str(TRTLLM_SITE / "tensorrt_llm" / "bindings")]
    sys.modules["tensorrt_llm.bindings"] = bindings
    spec.loader.exec_module(bindings)

    plugin_lib = TRTLLM_SITE / "tensorrt_llm" / "libs" / "libnvinfer_plugin_tensorrt_llm.so"
    plugin_handle = ctypes.CDLL(str(plugin_lib), mode=ctypes.RTLD_GLOBAL)
    init_plugins = plugin_handle.initTrtLlmPlugins
    init_plugins.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    init_plugins.restype = ctypes.c_bool
    if not init_plugins(None, b"tensorrt_llm"):
        raise RuntimeError("initTrtLlmPlugins returned false")

    return importlib.import_module("tensorrt_llm.bindings.executor")


def load_preprocess_module():
    spec = importlib.util.spec_from_file_location("indicxlit_preprocess_model", PREPROCESS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load preprocess module from {PREPROCESS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_words(path: Path | None, limit: int) -> list[str]:
    if path and path.exists():
        words: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            words.append((parts[1] if len(parts) > 1 else parts[0]).strip())
            if len(words) >= limit:
                break
        if words:
            return words
    seed = ["bharat", "namaste", "karnataka", "vidyalaya", "maharashtra", "pradhan", "sangeet", "computer"]
    return [seed[i % len(seed)] for i in range(limit)]


def sample_gpu(stop: threading.Event, out: list[dict[str, float]], interval_s: float = 0.2) -> None:
    while not stop.is_set():
        try:
            raw = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,utilization.memory,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=2,
            )
            util_gpu, util_mem, mem_used = [float(x.strip()) for x in raw.splitlines()[0].split(",")]
            out.append({"gpu_util": util_gpu, "mem_util": util_mem, "mem_used_mib": mem_used})
        except Exception:
            pass
        stop.wait(interval_s)


def build_requests(ex: Any, encoded: dict[str, Any], count: int, beam_width: int, topk: int, output_len: int):
    import torch

    sampling = ex.SamplingConfig(beam_width)
    sampling.num_return_sequences = topk
    output_config = ex.OutputConfig(False, False, False, True, False, False, None)
    requests = []
    rows = encoded["INPUT_ID"]
    lengths = encoded["REQUEST_INPUT_LEN"]
    for i in range(count):
        row = i % len(rows)
        src_len = int(lengths[row, 0])
        encoder_ids = [int(x) for x in rows[row, :src_len].tolist()]
        cross_attention_mask = torch.ones((output_len, src_len), dtype=torch.bool)
        requests.append(
            ex.Request(
                [EOS_ID],
                output_len,
                streaming=False,
                sampling_config=sampling,
                output_config=output_config,
                end_id=EOS_ID,
                pad_id=PAD_ID,
                encoder_input_token_ids=encoder_ids,
                cross_attention_mask=cross_attention_mask,
            )
        )
    return requests


def drain_responses(executor: Any, want: int, request_start: dict[int, float], latencies: list[float]) -> int:
    done = 0
    while done < want:
        responses = executor.await_responses()
        for response in responses:
            if response.has_error():
                raise RuntimeError(f"Executor error for request {response.request_id}: {response.error_msg}")
            result = response.result
            if result.is_final:
                done += 1
                start = request_start.pop(int(response.request_id), None)
                if start is not None:
                    latencies.append(time.perf_counter() - start)
        if not responses:
            time.sleep(0.001)
    return done


def run_window(executor: Any, requests: list[Any], concurrency: int, total_requests: int):
    next_request = 0
    completed = 0
    request_start: dict[int, float] = {}
    latencies: list[float] = []

    gpu_samples: list[dict[str, float]] = []
    stop_gpu = threading.Event()
    gpu_thread = threading.Thread(target=sample_gpu, args=(stop_gpu, gpu_samples), daemon=True)
    gpu_thread.start()

    wall_start = time.perf_counter()
    initial = min(concurrency, total_requests)
    for _ in range(initial):
        req_id = int(executor.enqueue_request(requests[next_request % len(requests)]))
        request_start[req_id] = time.perf_counter()
        next_request += 1

    while completed < total_requests:
        responses = executor.await_responses()
        for response in responses:
            if response.has_error():
                raise RuntimeError(f"Executor error for request {response.request_id}: {response.error_msg}")
            if response.result.is_final:
                completed += 1
                start = request_start.pop(int(response.request_id), None)
                if start is not None:
                    latencies.append(time.perf_counter() - start)
                if next_request < total_requests:
                    req_id = int(executor.enqueue_request(requests[next_request % len(requests)]))
                    request_start[req_id] = time.perf_counter()
                    next_request += 1
        if not responses:
            time.sleep(0.001)

    wall_s = time.perf_counter() - wall_start
    stop_gpu.set()
    gpu_thread.join(timeout=2)

    result: dict[str, Any] = {
        "concurrency": concurrency,
        "requested": total_requests,
        "ok": completed,
        "wall_s": wall_s,
        "throughput_req_s": completed / wall_s if wall_s > 0 else 0.0,
    }
    if latencies:
        sorted_lat = sorted(latencies)
        result.update(
            {
                "latency_avg_ms": statistics.mean(latencies) * 1000,
                "latency_p50_ms": sorted_lat[int(0.50 * (len(sorted_lat) - 1))] * 1000,
                "latency_p95_ms": sorted_lat[int(0.95 * (len(sorted_lat) - 1))] * 1000,
                "latency_p99_ms": sorted_lat[int(0.99 * (len(sorted_lat) - 1))] * 1000,
            }
        )
    if gpu_samples:
        result.update(
            {
                "gpu_util_avg": statistics.mean(x["gpu_util"] for x in gpu_samples),
                "gpu_util_max": max(x["gpu_util"] for x in gpu_samples),
                "gpu_mem_util_avg": statistics.mean(x["mem_util"] for x in gpu_samples),
                "gpu_mem_util_max": max(x["mem_util"] for x in gpu_samples),
                "gpu_sample_count": len(gpu_samples),
                "gpu_mem_used_max_mib": max(x["mem_used_mib"] for x in gpu_samples),
            }
        )
    return result


def main() -> None:
    ensure_ld_path()
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-dir", type=Path, default=DEFAULT_ENGINE_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DAKSHINA)
    parser.add_argument("--lang", default="hi")
    parser.add_argument("--num-words", type=int, default=4096)
    parser.add_argument("--request-pool-size", type=int, default=4096)
    parser.add_argument("--total-requests", type=int, default=2048)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[64, 128, 192, 256, 320, 384])
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--max-batch-size", type=int, default=256)
    parser.add_argument("--max-num-tokens", type=int, default=8192)
    parser.add_argument("--kv-cache-free-gpu-mem-fraction", type=float, default=0.35)
    parser.add_argument("--cross-kv-cache-fraction", type=float, default=0.35)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ex = load_executor_bindings()
    preprocess = load_preprocess_module()
    words = load_words(args.dataset, args.num_words)
    encoded = preprocess.encode_batch(words, args.lang, args.output_len, DEFAULT_MODEL_ROOT)

    kv_cache_config = ex.KvCacheConfig(
        free_gpu_memory_fraction=args.kv_cache_free_gpu_mem_fraction,
        cross_kv_cache_fraction=args.cross_kv_cache_fraction,
    )
    executor_config = ex.ExecutorConfig(
        max_beam_width=args.beam_width,
        kv_cache_config=kv_cache_config,
        max_batch_size=args.max_batch_size,
        max_num_tokens=args.max_num_tokens,
        batching_type=ex.BatchingType.INFLIGHT,
        enable_chunked_context=False,
    )

    encoder_dir = args.engine_dir / "encoder"
    decoder_dir = args.engine_dir / "decoder"
    requests = build_requests(
        ex,
        encoded,
        count=min(args.request_pool_size, len(words)),
        beam_width=args.beam_width,
        topk=args.topk,
        output_len=args.output_len,
    )

    results = []
    with ex.Executor(str(encoder_dir), str(decoder_dir), ex.ModelType.ENCODER_DECODER, executor_config) as executor:
        # Warm-up a small batch.
        warm = min(16, len(requests))
        start_map: dict[int, float] = {}
        dummy_latencies: list[float] = []
        for i in range(warm):
            rid = int(executor.enqueue_request(requests[i]))
            start_map[rid] = time.perf_counter()
        drain_responses(executor, warm, start_map, dummy_latencies)

        for concurrency in args.concurrency:
            total_requests = max(args.total_requests, concurrency * 4)
            result = run_window(executor, requests, concurrency, total_requests)
            print(json.dumps(result, indent=2), flush=True)
            results.append(result)

    payload = {
        "engine_dir": str(args.engine_dir),
        "dataset": str(args.dataset),
        "num_words": len(words),
        "beam_width": args.beam_width,
        "topk": args.topk,
        "output_len": args.output_len,
        "kv_cache_free_gpu_mem_fraction": args.kv_cache_free_gpu_mem_fraction,
        "cross_kv_cache_fraction": args.cross_kv_cache_fraction,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
