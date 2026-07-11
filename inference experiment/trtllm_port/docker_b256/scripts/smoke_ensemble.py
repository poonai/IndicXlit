#!/usr/bin/env python3
import argparse
import json
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://localhost:8000")
args = parser.parse_args()
payload = {
    "inputs": [
        {"name": "text_input", "shape": [1, 1], "datatype": "BYTES", "data": ["bharat"]},
        {"name": "target_lang", "shape": [1, 1], "datatype": "BYTES", "data": ["hi"]},
        {"name": "max_tokens", "shape": [1, 1], "datatype": "INT32", "data": [32]},
        {"name": "beam_width", "shape": [1, 1], "datatype": "INT32", "data": [5]},
        {"name": "topk", "shape": [1, 1], "datatype": "INT32", "data": [5]},
        {"name": "rescore", "shape": [1, 1], "datatype": "BOOL", "data": [False]},
    ],
    "outputs": [{"name": "text_output"}, {"name": "candidates_json"}],
}
request = urllib.request.Request(
    f"{args.url}/v2/models/indicxlit_ensemble/infer",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=120) as response:
    result = json.load(response)
outputs = {item["name"]: item["data"] for item in result["outputs"]}
assert outputs["text_output"][0] == "भारत", outputs
assert len(json.loads(outputs["candidates_json"][0])) == 5, outputs
print(json.dumps(outputs, ensure_ascii=False))
