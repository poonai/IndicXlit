#!/usr/bin/env python3
"""Capture deterministic Fairseq outputs and benchmark metrics for TRT-LLM parity."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = REPO_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


DEFAULT_WORDS_FILE = Path(__file__).resolve().parent / "words_en_hi.txt"
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts" / "fairseq_baseline"


def read_words(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def parse_translation(raw: str) -> dict[int, dict[str, object]]:
    rows: dict[int, dict[str, object]] = {}
    for line in raw.splitlines():
        if not line or "-" not in line or "\t" not in line:
            continue
        tag, *parts = line.split("\t")
        prefix, index_text = tag.split("-", 1)
        if not index_text.isdigit():
            continue
        index = int(index_text)
        row = rows.setdefault(
            index,
            {
                "source_tokens_text": None,
                "worker_seconds": None,
                "hypotheses": [],
                "detokenized": [],
                "positional_scores": [],
            },
        )
        if prefix == "S" and parts:
            row["source_tokens_text"] = parts[0]
        elif prefix == "W" and parts:
            match = re.search(r"[-+]?[0-9]*\.?[0-9]+", parts[0])
            row["worker_seconds"] = float(match.group(0)) if match else None
        elif prefix == "H" and len(parts) >= 2:
            row["hypotheses"].append({"score": float(parts[0]), "text": parts[1]})
        elif prefix == "D" and len(parts) >= 2:
            row["detokenized"].append({"score": float(parts[0]), "text": parts[1]})
        elif prefix == "P" and parts:
            row["positional_scores"].append(parts[0])
    return rows


def token_ids(dictionary, text: str) -> list[int]:
    return dictionary.encode_line(text, append_eos=True, add_if_not_exist=False).tolist()


def capture_outputs(lang: str, words: list[str], beam: int, topk: int, postprocess: bool) -> dict[str, object]:
    from ai4bharat.transliteration.xlit_src import XlitEngine

    engine = XlitEngine(
        lang2use=lang,
        beam_width=beam,
        rescore=False,
        model_type="transformer",
        src_script_type="roman",
    )
    preprocessed = engine.pre_process(words, "en", lang)
    raw = engine.transliterator.translate(preprocessed, nbest=topk)
    parsed = parse_translation(raw)
    final = engine.post_process(raw, lang) if postprocess else None
    src_dict = engine.transliterator.src_dict

    rows = []
    for index, word in enumerate(words):
        item = parsed.get(index, {})
        final_index = index * topk
        rows.append(
            {
                "index": index,
                "word": word,
                "preprocessed": preprocessed[index],
                "source_token_ids": token_ids(src_dict, preprocessed[index]),
                "raw_source_tokens_text": item.get("source_tokens_text"),
                "hypotheses": item.get("hypotheses", []),
                "detokenized": item.get("detokenized", []),
                "positional_scores": item.get("positional_scores", []),
                "final_postprocessed_best": final[final_index] if final and final_index < len(final) else None,
                "final_postprocessed_all": (
                    final[final_index : final_index + topk] if final and final_index < len(final) else []
                ),
            }
        )

    cfg = engine.transliterator.cfg
    model = engine.transliterator.models[0]
    return {
        "lang": lang,
        "beam": beam,
        "topk": topk,
        "postprocess": postprocess,
        "model": {
            "encoder_layers": len(model.encoder.layers),
            "decoder_layers": len(model.decoder.layers),
            "encoder_embed_dim": model.encoder.embed_tokens.embedding_dim,
            "decoder_embed_dim": model.decoder.embed_tokens.embedding_dim,
            "src_vocab_size": len(engine.transliterator.src_dict),
            "tgt_vocab_size": len(engine.transliterator.tgt_dict),
            "pad": engine.transliterator.tgt_dict.pad(),
            "eos": engine.transliterator.tgt_dict.eos(),
            "unk": engine.transliterator.tgt_dict.unk(),
        },
        "generation": {
            "max_len_a": cfg.generation.max_len_a,
            "max_len_b": cfg.generation.max_len_b,
            "min_len": cfg.generation.min_len,
            "lenpen": cfg.generation.lenpen,
        },
        "rows": rows,
        "raw_translation": raw,
    }


def run_benchmark(
    python: str,
    output_csv: Path,
    lang: str,
    beam: int,
    topk: int,
    postprocess: bool,
    batch_sizes: str,
    target_items: int,
    words_file: Path,
) -> dict[str, object]:
    cmd = [
        python,
        str(REPO_ROOT / "inference experiment" / "benchmark_batch_decode.py"),
        "--direction",
        "en2indic",
        "--lang",
        lang,
        "--beam-width",
        str(beam),
        "--topk",
        str(topk),
        "--batch-sizes",
        batch_sizes,
        "--target-items",
        str(target_items),
        "--min-repeats",
        "2",
        "--warmup",
        "2",
        "--words-file",
        str(words_file),
        "--csv",
        str(output_csv),
    ]
    if not postprocess:
        cmd.append("--no-postprocess")

    started = time.perf_counter()
    completed = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return {
        "cmd": cmd,
        "returncode": completed.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture Fairseq baseline artifacts for TRT-LLM parity.")
    parser.add_argument("--lang", default="hi")
    parser.add_argument("--words-file", type=Path, default=DEFAULT_WORDS_FILE)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--batch-sizes", default="1,32,128")
    parser.add_argument("--target-items", type=int, default=256)
    parser.add_argument("--skip-benchmarks", action="store_true")
    args = parser.parse_args()

    words = read_words(args.words_file)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    cases = [
        {"name": "greedy_no_postprocess", "beam": 1, "topk": 1, "postprocess": False},
        {"name": "beam8_top5_no_postprocess", "beam": 8, "topk": 5, "postprocess": False},
        {"name": "greedy_postprocess", "beam": 1, "topk": 1, "postprocess": True},
        {"name": "beam8_top5_postprocess", "beam": 8, "topk": 5, "postprocess": True},
    ]

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "words_file": str(args.words_file),
        "word_count": len(words),
        "cases": [],
    }

    for case in cases:
        print(f"Capturing {case['name']}...", flush=True)
        output = capture_outputs(args.lang, words, case["beam"], case["topk"], case["postprocess"])
        output_path = args.artifact_dir / f"{case['name']}.json"
        output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

        bench_csv = args.artifact_dir / f"{case['name']}_benchmark.csv"
        bench_result = None
        if not args.skip_benchmarks:
            bench_result = run_benchmark(
                python,
                bench_csv,
                args.lang,
                case["beam"],
                case["topk"],
                case["postprocess"],
                args.batch_sizes,
                args.target_items,
                args.words_file,
            )
            bench_log = args.artifact_dir / f"{case['name']}_benchmark_log.json"
            bench_log.write_text(json.dumps(bench_result, indent=2, ensure_ascii=False), encoding="utf-8")
            if bench_result["returncode"] != 0:
                raise SystemExit(f"Benchmark failed for {case['name']}; see {bench_log}")

        manifest["cases"].append(
            {
                **case,
                "output_json": str(output_path),
                "benchmark_csv": str(bench_csv) if bench_csv.exists() else None,
                "benchmark_rows": csv_rows(bench_csv),
            }
        )

    manifest_path = args.artifact_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
