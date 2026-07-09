#!/usr/bin/env python3
"""Evaluate the IndicXlit TensorRT-LLM port on Dakshina lexicon TSV files."""

from __future__ import annotations

import argparse
import json
import re
import sys
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

from inspect_indicxlit_checkpoint import DEFAULT_CORPUS_BIN, DEFAULT_LANG_LIST  # noqa: E402
from run_trtllm_greedy import (  # noqa: E402
    BOS_ID,
    EOS_ID,
    PAD_ID,
    decode_ids,
    encode_preprocessed,
    extract_sequences,
    attention_mask_from_padded,
    load_vocab,
    normalize_output,
    pad_rows,
    postprocess_raw,
    preprocess_words,
)


DEFAULT_ENGINE_DIR = SCRIPT_DIR / "artifacts" / "trtllm_engines_en_hi_beam4"
DEFAULT_DAKSHINA_TEST = (
    SCRIPT_DIR
    / "artifacts"
    / "dakshina_data"
    / "dakshina_dataset_v1.0"
    / "hi"
    / "lexicons"
    / "hi.translit.sampled.test.tsv"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "artifacts" / "dakshina_hi_trtllm_beam4_eval.json"
SCRIPT_FILTERS = {
    "hi": r"[^\u0900-\u097F]",
}


def load_dakshina_pairs(path: Path, lang: str) -> tuple[list[dict], dict]:
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


def iter_batches(rows: list[dict], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate TRT-LLM IndicXlit on Dakshina lexicon data.")
    parser.add_argument("--engine-dir", type=Path, default=DEFAULT_ENGINE_DIR)
    parser.add_argument("--dakshina-tsv", type=Path, default=DEFAULT_DAKSHINA_TEST)
    parser.add_argument("--corpus-bin", type=Path, default=DEFAULT_CORPUS_BIN)
    parser.add_argument("--lang-list", type=Path, default=DEFAULT_LANG_LIST)
    parser.add_argument("--lang", default="hi")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, help="Optional limit for quick smoke runs.")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.beam_width <= 0:
        raise SystemExit("--beam-width must be positive")
    if args.topk <= 0 or args.topk > args.beam_width:
        raise SystemExit("--topk must be in [1, beam_width]")

    from tensorrt_llm.runtime import EncDecModelRunner

    rows, filter_stats = load_dakshina_pairs(args.dakshina_tsv, args.lang)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No rows to evaluate after filtering")

    _, src_to_id = load_vocab(args.corpus_bin / "dict.en.txt", args.lang_list)
    tgt_symbols, _ = load_vocab(args.corpus_bin / f"dict.{args.lang}.txt", args.lang_list)
    runner = EncDecModelRunner.from_engine("rank0.engine", str(args.engine_dir))

    started = time.perf_counter()
    result_rows = []
    correct_top1 = 0
    correct_topk = 0
    for batch in iter_batches(rows, args.batch_size):
        words = [row["word"] for row in batch]
        preprocessed = preprocess_words(words, args.lang)
        source_ids = [encode_preprocessed(row, src_to_id) for row in preprocessed]
        encoder_input_ids = pad_rows(source_ids, PAD_ID)
        decoder_input_ids = torch.full((len(words), 1), EOS_ID, dtype=torch.int32)
        output = runner.generate(
            encoder_input_ids,
            decoder_input_ids,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.beam_width,
            pad_token_id=PAD_ID,
            eos_token_id=EOS_ID,
            bos_token_id=BOS_ID,
            attention_mask=attention_mask_from_padded(encoder_input_ids, PAD_ID),
            return_dict=False,
        )
        torch.cuda.synchronize()
        output_tensor = normalize_output(output)
        beam_sequences = extract_sequences(output_tensor, len(words), decoder_input_ids.shape[1])

        for local_index, sequences in enumerate(beam_sequences):
            candidates = []
            for beam_index, sequence in enumerate(sequences[: args.topk]):
                generated_ids, raw = decode_ids(sequence, tgt_symbols)
                candidates.append(
                    {
                        "rank": beam_index + 1,
                        "generated_token_ids": generated_ids,
                        "raw_decoded": raw,
                        "final_postprocessed": postprocess_raw(raw),
                    }
                )
            target = batch[local_index]["target"]
            best = candidates[0]["final_postprocessed"] if candidates else ""
            topk_predictions = [candidate["final_postprocessed"] for candidate in candidates]
            top1_match = best == target
            topk_match = target in topk_predictions
            correct_top1 += int(top1_match)
            correct_topk += int(topk_match)
            result_rows.append(
                {
                    **batch[local_index],
                    "preprocessed": preprocessed[local_index],
                    "source_token_ids": source_ids[local_index],
                    "prediction": best,
                    "top1_match": top1_match,
                    "topk_match": topk_match,
                    "candidates": candidates,
                }
            )

    elapsed = time.perf_counter() - started
    total = len(result_rows)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": "trtllm",
        "dataset": "dakshina",
        "dakshina_tsv": str(args.dakshina_tsv),
        "engine_dir": str(args.engine_dir),
        "lang": args.lang,
        "beam": args.beam_width,
        "topk": args.topk,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "filter_stats": filter_stats,
        "evaluated_count": total,
        "top1_correct": correct_top1,
        "topk_correct": correct_topk,
        "top1_accuracy": correct_top1 / total if total else 0.0,
        "topk_accuracy": correct_topk / total if total else 0.0,
        "elapsed_seconds": elapsed,
        "items_per_second": total / elapsed if elapsed else 0.0,
        "rows": result_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(
        f"count={total} top1={report['top1_accuracy']:.4f} "
        f"top{args.topk}={report['topk_accuracy']:.4f} items/s={report['items_per_second']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
