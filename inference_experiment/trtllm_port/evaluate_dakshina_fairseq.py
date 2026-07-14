#!/usr/bin/env python3
"""Evaluate the Fairseq IndicXlit checkpoint on Dakshina lexicon TSV files."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import ujson
import torch


_ORIGINAL_TORCH_LOAD = torch.load


def _torch_load_legacy_checkpoint(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _ORIGINAL_TORCH_LOAD(*args, **kwargs)


torch.load = _torch_load_legacy_checkpoint


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
APP_DIR = REPO_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ai4bharat.transliteration.transformer.custom_interactive import Transliterator  # noqa: E402


DEFAULT_MODEL_ROOT = APP_DIR / "ai4bharat/transliteration/transformer/models/en2indic/v1.0"
DEFAULT_DAKSHINA_TEST = (
    SCRIPT_DIR
    / "artifacts"
    / "dakshina_data"
    / "dakshina_dataset_v1.0"
    / "hi"
    / "lexicons"
    / "hi.translit.sampled.test.tsv"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "artifacts" / "dakshina_hi_fairseq_beam4_eval.json"
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
        yield start, rows[start : start + batch_size]


def preprocess_words(words: list[str], lang: str) -> list[str]:
    return [f"__{lang}__ {' '.join(list(word.lower()))}" for word in words]


def joined_prediction(prediction: str) -> str:
    return "".join(prediction.split(" "))


def parse_fairseq_output(translation_str: str) -> dict[int, list[dict]]:
    candidates: dict[int, list[dict]] = defaultdict(list)
    for line in translation_str.splitlines():
        if not line.startswith("H-"):
            continue
        pieces = line.split("\t")
        if len(pieces) < 3:
            continue
        local_id = int(pieces[0].split("-", 1)[1])
        score_bits = float(pieces[1])
        raw = pieces[2]
        candidates[local_id].append(
            {
                "raw_decoded": raw,
                "model_score_bits": score_bits,
                "model_score_prob": math.pow(2.0, score_bits),
                "final_postprocessed": joined_prediction(raw),
            }
        )

    for local_id in candidates:
        candidates[local_id].sort(key=lambda item: item["model_score_prob"], reverse=True)
        for rank, candidate in enumerate(candidates[local_id], start=1):
            candidate["rank"] = rank
    return candidates


def rescore_candidates(candidates: list[dict], word_prob_dict: dict, alpha: float) -> list[dict]:
    total_model_score = sum(candidate["model_score_prob"] for candidate in candidates)
    dict_candidates = [
        candidate for candidate in candidates if candidate["final_postprocessed"] in word_prob_dict
    ]
    total_dict_score = sum(word_prob_dict[candidate["final_postprocessed"]] for candidate in dict_candidates)

    rescored = []
    for candidate in candidates:
        word = candidate["final_postprocessed"]
        if not total_model_score or word not in word_prob_dict or not total_dict_score:
            score = 0.0
        else:
            model_norm = candidate["model_score_prob"] / total_model_score
            dict_norm = word_prob_dict[word] / total_dict_score
            score = alpha * model_norm + (1.0 - alpha) * dict_norm
        rescored.append({**candidate, "rescore_score": score})

    rescored.sort(key=lambda item: item["rescore_score"], reverse=True)
    for rank, candidate in enumerate(rescored, start=1):
        candidate["rescored_rank"] = rank
    return rescored


def build_lang_pairs(model_root: Path) -> str:
    lang_list = (model_root / "../lang_list.txt").resolve().read_text(encoding="utf-8").splitlines()
    langs = [lang for lang in lang_list if lang and lang != "en"]
    return ",".join(f"en-{lang}" for lang in langs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Fairseq IndicXlit on Dakshina lexicon data.")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--dakshina-tsv", type=Path, default=DEFAULT_DAKSHINA_TEST)
    parser.add_argument("--lang", default="hi")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--rescore-alpha", type=float, default=0.9)
    parser.add_argument("--no-rescore", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.topk <= 0 or args.topk > args.beam_width:
        raise SystemExit("--topk must be in [1, beam_width]")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    rows, filter_stats = load_dakshina_pairs(args.dakshina_tsv, args.lang)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No rows to evaluate after filtering")

    word_prob_dict = None
    if not args.no_rescore:
        dict_path = args.model_root / "word_prob_dicts" / f"{args.lang}_word_prob_dict.json"
        if not dict_path.is_file():
            raise SystemExit(f"Missing rescoring dictionary: {dict_path}")
        word_prob_dict = ujson.load(open(dict_path, encoding="utf-8"))

    transliterator = Transliterator(
        str(args.model_root / "corpus-bin"),
        str(args.model_root / "transformer" / "indicxlit.pt"),
        lang_pairs_csv=build_lang_pairs(args.model_root),
        lang_list_file=str((args.model_root / "../lang_list.txt").resolve()),
        beam=args.beam_width,
        batch_size=args.batch_size,
    )

    started = time.perf_counter()
    result_rows = []
    raw_top1_correct = 0
    raw_topk_correct = 0
    rescored_top1_correct = 0
    rescored_topk_correct = 0
    for _, batch in iter_batches(rows, args.batch_size):
        preprocessed = preprocess_words([row["word"] for row in batch], args.lang)
        translation_str = transliterator.translate(preprocessed, nbest=args.topk)
        by_id = parse_fairseq_output(translation_str)

        for local_index, row in enumerate(batch):
            candidates = by_id.get(local_index, [])[: args.topk]
            raw_predictions = [candidate["final_postprocessed"] for candidate in candidates]
            raw_prediction = raw_predictions[0] if raw_predictions else ""
            raw_top1_match = raw_prediction == row["target"]
            raw_topk_match = row["target"] in raw_predictions
            raw_top1_correct += int(raw_top1_match)
            raw_topk_correct += int(raw_topk_match)

            rescored = None
            rescored_prediction = None
            rescored_top1_match = None
            rescored_topk_match = None
            if word_prob_dict is not None:
                rescored = rescore_candidates(candidates, word_prob_dict, args.rescore_alpha)
                rescored_predictions = [candidate["final_postprocessed"] for candidate in rescored]
                rescored_prediction = rescored_predictions[0] if rescored_predictions else ""
                rescored_top1_match = rescored_prediction == row["target"]
                rescored_topk_match = row["target"] in rescored_predictions
                rescored_top1_correct += int(rescored_top1_match)
                rescored_topk_correct += int(rescored_topk_match)

            result_rows.append(
                {
                    **row,
                    "preprocessed": preprocessed[local_index],
                    "raw_prediction": raw_prediction,
                    "raw_top1_match": raw_top1_match,
                    "raw_topk_match": raw_topk_match,
                    "rescored_prediction": rescored_prediction,
                    "rescored_top1_match": rescored_top1_match,
                    "rescored_topk_match": rescored_topk_match,
                    "candidates": candidates,
                    "rescored_candidates": rescored,
                }
            )

    elapsed = time.perf_counter() - started
    total = len(result_rows)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": "fairseq",
        "dataset": "dakshina",
        "dakshina_tsv": str(args.dakshina_tsv),
        "model_root": str(args.model_root),
        "lang": args.lang,
        "beam": args.beam_width,
        "topk": args.topk,
        "batch_size": args.batch_size,
        "rescore": word_prob_dict is not None,
        "rescore_alpha": args.rescore_alpha if word_prob_dict is not None else None,
        "filter_stats": filter_stats,
        "evaluated_count": total,
        "raw_top1_correct": raw_top1_correct,
        "raw_topk_correct": raw_topk_correct,
        "raw_top1_accuracy": raw_top1_correct / total if total else 0.0,
        "raw_topk_accuracy": raw_topk_correct / total if total else 0.0,
        "rescored_top1_correct": rescored_top1_correct if word_prob_dict is not None else None,
        "rescored_topk_correct": rescored_topk_correct if word_prob_dict is not None else None,
        "rescored_top1_accuracy": rescored_top1_correct / total if word_prob_dict is not None and total else None,
        "rescored_topk_accuracy": rescored_topk_correct / total if word_prob_dict is not None and total else None,
        "elapsed_seconds": elapsed,
        "items_per_second": total / elapsed if elapsed else 0.0,
        "rows": result_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(
        f"count={total} raw_top1={report['raw_top1_accuracy']:.4f} "
        f"raw_top{args.topk}={report['raw_topk_accuracy']:.4f} "
        f"items/s={report['items_per_second']:.2f}"
    )
    if word_prob_dict is not None:
        print(
            f"rescored_top1={report['rescored_top1_accuracy']:.4f} "
            f"rescored_top{args.topk}={report['rescored_topk_accuracy']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
