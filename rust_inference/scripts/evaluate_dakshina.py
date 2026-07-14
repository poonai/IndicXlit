#!/usr/bin/env python3
"""Evaluate the Rust IndicXlit HTTP runtime on Dakshina lexicon data."""

from __future__ import annotations

import argparse
import json
import re
import tarfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_ROOT = PROJECT_DIR / "artifacts" / "dakshina_data"
DEFAULT_OUTPUT = PROJECT_DIR / "artifacts" / "dakshina_hi_rust_http_eval.json"
DEFAULT_URL = "http://127.0.0.1:8000/v2/models/indicxlit/infer"
DAKSHINA_ARCHIVE_URL = "https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar"
SCRIPT_FILTERS = {
    "hi": r"[^\u0900-\u097F]",
}


def dakshina_tsv(data_root: Path, lang: str, split: str) -> Path:
    return data_root / "dakshina_dataset_v1.0" / lang / "lexicons" / f"{lang}.translit.sampled.{split}.tsv"


def download_file(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, output.open("wb") as out:
        total = int(response.headers.get("content-length") or 0)
        copied = 0
        last_report = time.perf_counter()
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            copied += len(chunk)
            now = time.perf_counter()
            if now - last_report >= 5:
                if total:
                    print(f"downloaded {copied / 1e9:.2f}/{total / 1e9:.2f} GB")
                else:
                    print(f"downloaded {copied / 1e9:.2f} GB")
                last_report = now


def ensure_dakshina(data_root: Path, lang: str, split: str, download: bool) -> Path:
    tsv = dakshina_tsv(data_root, lang, split)
    if tsv.is_file():
        return tsv
    if not download:
        raise SystemExit(
            f"Missing Dakshina TSV: {tsv}\n"
            "Pass --download-dakshina to download/extract it, or pass --dakshina-tsv."
        )

    archive = data_root / "dakshina_dataset_v1.0.tar"
    if not archive.is_file():
        print(f"Downloading Dakshina archive to {archive}")
        download_file(DAKSHINA_ARCHIVE_URL, archive)

    print(f"Extracting {archive} to {data_root}")
    data_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r") as tar:
        safe_extract(tar, data_root)

    if not tsv.is_file():
        raise SystemExit(f"Dakshina TSV still missing after extraction: {tsv}")
    return tsv


def safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in tar.getmembers():
        target = (destination / member.name).resolve()
        if destination != target and destination not in target.parents:
            raise RuntimeError(f"Refusing to extract path outside destination: {member.name}")
    tar.extractall(destination)


def load_dakshina_pairs(path: Path, lang: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not path.is_file():
        raise SystemExit(f"Missing Dakshina TSV: {path}")

    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    stats = {"raw_rows": len(rows)}

    pairs = []
    for row in rows:
        pieces = row.split("\t")
        if len(pieces) >= 2:
            pairs.append((pieces[0], pieces[1]))
    stats["first_two_columns"] = len(pairs)

    roman_re = re.compile(r"[^a-zA-Z]")
    pairs = [(native, roman) for native, roman in pairs if not roman_re.search(roman)]
    stats["roman_ascii_rows"] = len(pairs)

    script_pattern = SCRIPT_FILTERS.get(lang)
    if script_pattern:
        script_re = re.compile(script_pattern)
        pairs = [(native, roman) for native, roman in pairs if not script_re.search(native)]
    stats["native_script_rows"] = len(pairs)

    pairs = [(native, roman.lower()) for native, roman in pairs]
    pairs = sorted(set(pairs), key=lambda pair: (pair[1], pair[0]))
    stats["deduped_lower_rows"] = len(pairs)

    return [
        {"index": index, "target": native, "word": roman}
        for index, (native, roman) in enumerate(pairs)
    ], stats


def iter_batches(rows: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {body}") from error


def extract_outputs(response: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    outputs = {item.get("name"): item for item in response.get("outputs", [])}
    text_output = outputs.get("text_output", {}).get("data", [])
    candidates_raw = outputs.get("candidates_json", {}).get("data", [])
    candidates = []
    for value in candidates_raw:
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            parsed = []
        candidates.append([str(item) for item in parsed])
    return [str(item) for item in text_output], candidates


def request_payload(
    words: list[str],
    lang: str,
    beam_width: int,
    topk: int,
    max_tokens: int,
    rescore: bool | None,
) -> dict[str, Any]:
    inputs = [
        {"name": "text_input", "data": ["\n".join(words)]},
        {"name": "target_lang", "data": [lang]},
        {"name": "beam_width", "data": [beam_width]},
        {"name": "topk", "data": [topk]},
        {"name": "max_tokens", "data": [max_tokens]},
    ]
    if rescore is not None:
        inputs.append({"name": "rescore", "data": [rescore]})
    return {
        "inputs": inputs
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Rust IndicXlit HTTP runtime on Dakshina.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--lang", default="hi")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dakshina-tsv", type=Path)
    parser.add_argument(
        "--download-dakshina",
        dest="download_dakshina",
        action="store_true",
        default=True,
        help="Download/extract Dakshina if the selected TSV is missing. This is the default.",
    )
    parser.add_argument(
        "--no-download-dakshina",
        dest="download_dakshina",
        action="store_false",
        help="Fail instead of downloading Dakshina when the selected TSV is missing.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--rescore", dest="rescore", action="store_true", default=None)
    parser.add_argument("--no-rescore", dest="rescore", action="store_false")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int, help="Limit rows for a quick smoke evaluation.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--save-rows", action="store_true", help="Store per-row predictions in the output JSON.")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.beam_width <= 0:
        raise SystemExit("--beam-width must be positive")
    if args.topk <= 0 or args.topk > args.beam_width:
        raise SystemExit("--topk must be in [1, beam-width]")
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive")

    tsv = args.dakshina_tsv or ensure_dakshina(args.data_root, args.lang, args.split, args.download_dakshina)
    rows, filter_stats = load_dakshina_pairs(tsv, args.lang)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No rows to evaluate after filtering")

    started = time.perf_counter()
    request_latencies = []
    result_rows = []
    correct_top1 = 0
    correct_topk = 0

    for batch_index, batch in enumerate(iter_batches(rows, args.batch_size), start=1):
        words = [row["word"] for row in batch]
        request_started = time.perf_counter()
        response = post_json(
            args.url,
            request_payload(words, args.lang, args.beam_width, args.topk, args.max_tokens, args.rescore),
            args.timeout,
        )
        request_latencies.append(time.perf_counter() - request_started)
        predictions, candidates_by_row = extract_outputs(response)
        if len(predictions) != len(batch):
            raise RuntimeError(f"Expected {len(batch)} predictions, got {len(predictions)}")

        for row, prediction, candidates in zip(batch, predictions, candidates_by_row):
            target = row["target"]
            top1_match = prediction == target
            topk_candidates = candidates[: args.topk]
            topk_match = target in topk_candidates
            correct_top1 += int(top1_match)
            correct_topk += int(topk_match)
            if args.save_rows:
                result_rows.append(
                    {
                        **row,
                        "prediction": prediction,
                        "top1_match": top1_match,
                        f"top{args.topk}_match": topk_match,
                        "candidates": topk_candidates,
                    }
                )

        if batch_index == 1 or batch_index % 10 == 0:
            done = min(batch_index * args.batch_size, len(rows))
            print(f"evaluated {done}/{len(rows)}")

    elapsed = time.perf_counter() - started
    total = len(rows)
    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": "rust_executor_http",
        "url": args.url,
        "dataset": "dakshina",
        "dakshina_tsv": str(tsv),
        "lang": args.lang,
        "split": args.split,
        "beam": args.beam_width,
        "topk": args.topk,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "rescore": args.rescore,
        "filter_stats": filter_stats,
        "evaluated_count": total,
        "top1_correct": correct_top1,
        f"top{args.topk}_correct": correct_topk,
        "top1_accuracy": correct_top1 / total if total else 0.0,
        f"top{args.topk}_accuracy": correct_topk / total if total else 0.0,
        "elapsed_seconds": elapsed,
        "items_per_second": total / elapsed if elapsed else 0.0,
        "request_count": len(request_latencies),
        "avg_request_latency_seconds": sum(request_latencies) / len(request_latencies) if request_latencies else 0.0,
        "max_request_latency_seconds": max(request_latencies) if request_latencies else 0.0,
    }
    if args.save_rows:
        report["rows"] = result_rows

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {args.output}")
    print(
        f"count={total} top1={report['top1_accuracy']:.4f} "
        f"top{args.topk}={report[f'top{args.topk}_accuracy']:.4f} "
        f"items/s={report['items_per_second']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
