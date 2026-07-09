#!/usr/bin/env python3
"""Compare TensorRT-LLM experiment outputs against locked Fairseq artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FAIRSEQ = SCRIPT_DIR / "artifacts" / "fairseq_baseline" / "greedy_postprocess.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "artifacts" / "trtllm_parity_report.json"


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def rows_by_index(payload: dict) -> dict[int, dict]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise SystemExit("Expected payload to contain a top-level rows list.")
    result = {}
    for row in rows:
        if "index" not in row:
            raise SystemExit("Every row must contain an index.")
        result[int(row["index"])] = row
    return result


def best_text(row: dict, field: str) -> str | None:
    if field in row:
        return row[field]
    if field == "raw_decoded_best":
        hypotheses = row.get("hypotheses") or []
        if hypotheses:
            return hypotheses[0].get("text")
    return None


def token_diff(expected: list[int] | None, actual: list[int] | None) -> dict | None:
    if expected is None and actual is None:
        return None
    expected = expected or []
    actual = actual or []
    first_mismatch = None
    for idx, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            first_mismatch = idx
            break
    if first_mismatch is None and len(expected) != len(actual):
        first_mismatch = min(len(expected), len(actual))
    return {
        "expected": expected,
        "actual": actual,
        "first_mismatch": first_mismatch,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare TRT-LLM rows against Fairseq baseline rows.")
    parser.add_argument("--fairseq", type=Path, default=DEFAULT_FAIRSEQ)
    parser.add_argument("--trtllm", type=Path, required=True)
    parser.add_argument(
        "--field",
        choices=["final_postprocessed_best", "raw_decoded_best"],
        default="final_postprocessed_best",
        help="Text field to compare. raw_decoded_best falls back to hypotheses[0].text.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    fairseq = rows_by_index(load_json(args.fairseq))
    trtllm = rows_by_index(load_json(args.trtllm))
    all_indices = sorted(fairseq)
    missing = [index for index in all_indices if index not in trtllm]

    mismatches = []
    matches = 0
    compared = 0
    for index in all_indices:
        if index not in trtllm:
            continue
        expected_row = fairseq[index]
        actual_row = trtllm[index]
        expected = best_text(expected_row, args.field)
        actual = best_text(actual_row, args.field)
        compared += 1
        if expected == actual:
            matches += 1
            continue
        mismatches.append(
            {
                "index": index,
                "word": expected_row.get("word", actual_row.get("word")),
                "expected": expected,
                "actual": actual,
                "expected_preprocessed": expected_row.get("preprocessed"),
                "actual_preprocessed": actual_row.get("preprocessed"),
                "token_diff": token_diff(
                    expected_row.get("generated_token_ids"),
                    actual_row.get("generated_token_ids"),
                ),
            }
        )

    exact_match_rate = (matches / compared) if compared else 0.0
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fairseq": str(args.fairseq),
        "trtllm": str(args.trtllm),
        "field": args.field,
        "expected_count": len(all_indices),
        "compared_count": compared,
        "missing_count": len(missing),
        "match_count": matches,
        "mismatch_count": len(mismatches),
        "exact_match_rate": exact_match_rate,
        "passes_greedy_gate": exact_match_rate >= 0.95 and not missing,
        "missing_indices": missing,
        "mismatches": mismatches,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(
        f"matches={matches}/{compared} missing={len(missing)} "
        f"exact_match_rate={exact_match_rate:.3f} passes_greedy_gate={report['passes_greedy_gate']}"
    )
    return 0 if report["passes_greedy_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
