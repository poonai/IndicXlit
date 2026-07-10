#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import queue
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import tritonclient.http as httpclient


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PREPROCESS_PATH = (
    SCRIPT_DIR
    / "triton_indicxlit"
    / "model_repository"
    / "indicxlit_preprocess"
    / "1"
    / "model.py"
)
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
            # Dakshina sampled lexicon is: native<TAB>roman<TAB>count.
            word = parts[1] if len(parts) > 1 else parts[0]
            words.append(word.strip())
            if len(words) >= limit:
                break
        if words:
            return words
    seed = [
        "bharat",
        "namaste",
        "karnataka",
        "vidyalaya",
        "maharashtra",
        "pradhan",
        "sangeet",
        "computer",
        "mobile",
        "krishna",
    ]
    return [seed[i % len(seed)] for i in range(limit)]


def make_inputs(encoded: dict[str, np.ndarray], row: int, beam_width: int, topk: int):
    input_len = int(encoded["REQUEST_INPUT_LEN"][row, 0])
    output_len = int(encoded["REQUEST_OUTPUT_LEN"][row, 0])
    fields = {
        "input_ids": encoded["INPUT_ID"][row : row + 1],
        "decoder_input_ids": encoded["DECODER_INPUT_ID"][row : row + 1],
        "input_lengths": encoded["REQUEST_INPUT_LEN"][row : row + 1],
        "decoder_input_lengths": encoded["REQUEST_DECODER_INPUT_LEN"][row : row + 1],
        "request_output_len": encoded["REQUEST_OUTPUT_LEN"][row : row + 1],
        "end_id": encoded["OUT_END_ID"][row : row + 1],
        "pad_id": encoded["OUT_PAD_ID"][row : row + 1],
        "beam_width": np.asarray([[beam_width]], dtype=np.int32),
        "num_return_sequences": np.asarray([[topk]], dtype=np.int32),
        "return_log_probs": np.asarray([[True]], dtype=np.bool_),
        "cross_attention_mask": np.ones((1, output_len, input_len), dtype=np.bool_),
    }
    inputs = []
    for name, value in fields.items():
        triton_input = httpclient.InferInput(name, value.shape, np_to_triton_dtype(value.dtype))
        triton_input.set_data_from_numpy(value)
        inputs.append(triton_input)
    return inputs


def np_to_triton_dtype(dtype: np.dtype[Any]) -> str:
    dtype = np.dtype(dtype)
    if dtype == np.int32:
        return "INT32"
    if dtype == np.bool_:
        return "BOOL"
    raise TypeError(f"Unsupported dtype: {dtype}")


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


def request_once(url: str, model: str, inputs, outputs):
    client = httpclient.InferenceServerClient(url=url, concurrency=1)
    return client.infer(model_name=model, inputs=inputs, outputs=outputs)


def run_concurrency(url: str, model: str, request_inputs, concurrency: int, total_requests: int):
    outputs = [
        httpclient.InferRequestedOutput("output_ids"),
        httpclient.InferRequestedOutput("sequence_length"),
        httpclient.InferRequestedOutput("cum_log_probs"),
    ]
    work_q: queue.Queue[int] = queue.Queue()
    for i in range(total_requests):
        work_q.put(i % len(request_inputs))

    latencies: list[float] = []
    errors: list[str] = []

    def worker():
        client = httpclient.InferenceServerClient(url=url, concurrency=1)
        local_latencies = []
        while True:
            try:
                idx = work_q.get_nowait()
            except queue.Empty:
                return local_latencies
            start = time.perf_counter()
            try:
                client.infer(model_name=model, inputs=request_inputs[idx], outputs=outputs)
                local_latencies.append(time.perf_counter() - start)
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))
            finally:
                work_q.task_done()

    gpu_samples: list[dict[str, float]] = []
    stop_gpu = threading.Event()
    gpu_thread = threading.Thread(target=sample_gpu, args=(stop_gpu, gpu_samples), daemon=True)
    gpu_thread.start()
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker) for _ in range(concurrency)]
        for future in as_completed(futures):
            latencies.extend(future.result())
    wall_s = time.perf_counter() - wall_start
    stop_gpu.set()
    gpu_thread.join(timeout=2)

    ok = len(latencies)
    result = {
        "concurrency": concurrency,
        "requested": total_requests,
        "ok": ok,
        "errors": errors[:5],
        "wall_s": wall_s,
        "throughput_req_s": ok / wall_s if wall_s > 0 else 0.0,
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
                "gpu_mem_util_avg": statistics.mean(x["mem_util"] for x in gpu_samples),
                "gpu_mem_used_max_mib": max(x["mem_used_mib"] for x in gpu_samples),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="localhost:8010")
    parser.add_argument("--model", default="indicxlit_tensorrt_llm")
    parser.add_argument("--lang", default="hi")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DAKSHINA)
    parser.add_argument("--num-words", type=int, default=4096)
    parser.add_argument("--total-requests", type=int, default=2048)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[64, 128, 192, 256, 320, 384])
    parser.add_argument("--request-output-len", type=int, default=32)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    preprocess = load_preprocess_module()
    words = load_words(args.dataset, args.num_words)
    encoded = preprocess.encode_batch(words, args.lang, args.request_output_len, DEFAULT_MODEL_ROOT)
    request_inputs = [
        make_inputs(encoded, i, args.beam_width, args.topk)
        for i in range(len(words))
    ]

    client = httpclient.InferenceServerClient(url=args.url)
    if not client.is_model_ready(args.model):
        raise RuntimeError(f"Model {args.model!r} is not ready at {args.url}")

    results = []
    for concurrency in args.concurrency:
        total_requests = max(args.total_requests, concurrency * 4)
        result = run_concurrency(args.url, args.model, request_inputs, concurrency, total_requests)
        print(json.dumps(result, indent=2), flush=True)
        results.append(result)

    payload = {
        "url": args.url,
        "model": args.model,
        "lang": args.lang,
        "dataset": str(args.dataset),
        "num_words": len(words),
        "beam_width": args.beam_width,
        "topk": args.topk,
        "request_output_len": args.request_output_len,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
