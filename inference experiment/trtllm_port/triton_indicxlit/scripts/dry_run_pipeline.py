#!/usr/bin/env python3
"""Dry-run the Triton Python preprocess/postprocess around EncDecModelRunner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
TRITON_ROOT = SCRIPT_DIR.parent
PORT_ROOT = TRITON_ROOT.parent
REPO_ROOT = PORT_ROOT.parents[1]

if str(PORT_ROOT) not in sys.path:
    sys.path.insert(0, str(PORT_ROOT))

from run_trtllm_greedy import normalize_output  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    from tensorrt_llm.runtime import EncDecModelRunner

    preprocess = load_module(
        "indicxlit_preprocess_model",
        TRITON_ROOT / "model_repository" / "indicxlit_preprocess" / "1" / "model.py",
    )
    postprocess = load_module(
        "indicxlit_postprocess_model",
        TRITON_ROOT / "model_repository" / "indicxlit_postprocess" / "1" / "model.py",
    )

    model_root = REPO_ROOT / "app" / "ai4bharat" / "transliteration" / "transformer" / "models" / "en2indic"
    engine_dir = PORT_ROOT / "artifacts" / "trtllm_engines_en_hi_beam5"
    words = ["bharat", "aayenge", "aavritti", "angrzi"]
    lang = "hi"

    encoded = preprocess.encode_batch(words, lang, request_output_len=32, model_root=model_root)
    runner = EncDecModelRunner.from_engine("rank0.engine", str(engine_dir))
    output = runner.generate(
        torch.tensor(encoded["INPUT_ID"], dtype=torch.int32),
        torch.tensor(encoded["DECODER_INPUT_ID"], dtype=torch.int32),
        max_new_tokens=32,
        num_beams=5,
        pad_token_id=preprocess.PAD_ID,
        eos_token_id=preprocess.EOS_ID,
        bos_token_id=preprocess.BOS_ID,
        attention_mask=torch.tensor(encoded["INPUT_ID"] != preprocess.PAD_ID, dtype=torch.int32),
        return_dict=True,
    )
    torch.cuda.synchronize()

    output_ids = normalize_output(output).numpy()
    cum_log_probs = output.get("cum_log_probs")
    if torch.is_tensor(cum_log_probs):
        cum_log_probs = cum_log_probs.detach().cpu().numpy()

    processor = postprocess.IndicXlitPostprocessor(model_root)
    decoded = processor.decode_batch(
        output_ids,
        langs=[lang] * len(words),
        cum_log_probs=cum_log_probs,
        topk=5,
        rescore=False,
    )
    rescored = processor.decode_batch(
        output_ids,
        langs=[lang] * len(words),
        cum_log_probs=cum_log_probs,
        topk=5,
        rescore=True,
    )

    for word, raw_candidates, rescored_candidates in zip(words, decoded, rescored):
        print(f"{word}: raw={raw_candidates[:5]} rescored={rescored_candidates[:5]}")

    expected = {
        "bharat": "भारत",
        "aayenge": "आयेंगे",
        "aavritti": "आवृत्ति",
        "angrzi": "अंग्रज़ी",
    }
    failed = [
        (word, candidates)
        for word, candidates in zip(words, decoded)
        if expected[word] not in candidates
    ]
    if failed:
        raise SystemExit(f"Expected candidate missing from top-5: {failed}")
    print("Dry-run pipeline passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
