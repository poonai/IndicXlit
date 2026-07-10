#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request


def infer_payload(input_ids: list[int], output_len: int = 32) -> dict:
    input_len = len(input_ids)
    mask = [True] * (output_len * input_len)
    return {
        "inputs": [
            {"name": "input_ids", "shape": [1, input_len], "datatype": "INT32", "data": input_ids},
            {"name": "decoder_input_ids", "shape": [1, 1], "datatype": "INT32", "data": [2]},
            {"name": "input_lengths", "shape": [1, 1], "datatype": "INT32", "data": [input_len]},
            {"name": "decoder_input_lengths", "shape": [1, 1], "datatype": "INT32", "data": [1]},
            {"name": "request_output_len", "shape": [1, 1], "datatype": "INT32", "data": [output_len]},
            {"name": "end_id", "shape": [1, 1], "datatype": "INT32", "data": [2]},
            {"name": "pad_id", "shape": [1, 1], "datatype": "INT32", "data": [1]},
            {"name": "beam_width", "shape": [1, 1], "datatype": "INT32", "data": [5]},
            {"name": "num_return_sequences", "shape": [1, 1], "datatype": "INT32", "data": [5]},
            {"name": "return_log_probs", "shape": [1, 1], "datatype": "BOOL", "data": [True]},
            {
                "name": "cross_attention_mask",
                "shape": [1, output_len, input_len],
                "datatype": "BOOL",
                "data": mask,
            },
        ],
        "outputs": [
            {"name": "output_ids"},
            {"name": "sequence_length"},
            {"name": "cum_log_probs"},
        ],
    }


def post_json(url: str, payload: dict, timeout: float = 30.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--wait", type=float, default=120.0)
    args = parser.parse_args()

    ready_url = args.url.rstrip("/") + "/v2/health/ready"
    infer_url = args.url.rstrip("/") + "/v2/models/indicxlit_tensorrt_llm/infer"

    deadline = time.time() + args.wait
    while True:
        try:
            urllib.request.urlopen(ready_url, timeout=2).read()
            break
        except Exception:
            if time.time() > deadline:
                raise
            time.sleep(2)

    # Pre-tokenized Hindi target "bharat" from the IndicXlit preprocess model.
    response = post_json(infer_url, infer_payload([3, 22, 7, 4, 9, 4, 8, 2]))
    print(json.dumps(response, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()

