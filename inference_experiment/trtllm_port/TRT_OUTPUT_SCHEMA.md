# TRT-LLM Output Schema

Use this JSON shape for TensorRT-LLM experiment outputs so
`compare_trtllm_parity.py` can compare them against the locked Fairseq baseline.

```json
{
  "backend": "trtllm",
  "lang": "hi",
  "beam": 1,
  "topk": 1,
  "postprocess": true,
  "rows": [
    {
      "index": 0,
      "word": "bharat",
      "preprocessed": "__hi__ b h a r a t",
      "source_token_ids": [38, 22, 7, 4, 9, 4, 8, 2],
      "generated_token_ids": [/* target ids, preferably without BOS/PAD */],
      "raw_decoded_best": "भ ा र त",
      "final_postprocessed_best": "भारत"
    }
  ]
}
```

Notes:

- Preserve input order in `rows`.
- `index` must match the Fairseq baseline row index.
- `final_postprocessed_best` should be the same postprocessing surface used by
  `BaseEngineTransformer.post_process`.
- During greedy parity, compare `final_postprocessed_best` first. If that fails,
  add `generated_token_ids` and compare token-level diffs.
- For beam search, keep the best candidate in the `*_best` fields and add an
  optional `candidates` list for top-k analysis.
