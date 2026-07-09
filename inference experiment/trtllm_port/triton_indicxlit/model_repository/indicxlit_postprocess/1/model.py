import json
import math
from pathlib import Path

import numpy as np

try:
    import triton_python_backend_utils as pb_utils
except ImportError:  # Allows local dry-run tests without tritonserver.
    pb_utils = None

try:
    import ujson
except ImportError:
    ujson = json


SPECIALS = ["<s>", "<pad>", "</s>", "<unk>"]
BOS_ID = 0
PAD_ID = 1
EOS_ID = 2


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[6]


def read_language_tokens(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_base_symbols(path: Path) -> list[str]:
    symbols = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        pieces = line.rsplit(" ", 1)
        if len(pieces) == 2:
            symbols.append(pieces[0])
    return symbols


def load_symbols(dict_path: Path, lang_list: Path) -> list[str]:
    return SPECIALS + read_base_symbols(dict_path) + read_language_tokens(lang_list)


def normalize_text_value(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def decode_ids(ids, id_to_token: list[str]) -> str:
    pieces = []
    for token_id in ids:
        token_id = int(token_id)
        if token_id == EOS_ID:
            break
        if token_id in (BOS_ID, PAD_ID):
            continue
        pieces.append(id_to_token[token_id] if 0 <= token_id < len(id_to_token) else "<unk>")
    return " ".join(pieces)


def postprocess_raw(raw: str) -> str:
    return "".join(raw.split(" "))


def normalize_output_ids(output_ids: np.ndarray, prompt_len: int = 1) -> np.ndarray:
    output_ids = np.asarray(output_ids)
    if output_ids.ndim == 2:
        output_ids = output_ids[:, None, :]
    if prompt_len and output_ids.shape[-1] > prompt_len:
        output_ids = output_ids[..., prompt_len:]
    return output_ids


def rescore_candidates(candidates: list[dict], word_prob_dict: dict, alpha: float) -> list[dict]:
    total_model_score = sum(candidate["model_score_prob"] for candidate in candidates)
    dict_candidates = [
        candidate for candidate in candidates if candidate["text"] in word_prob_dict
    ]
    total_dict_score = sum(word_prob_dict[candidate["text"]] for candidate in dict_candidates)

    rescored = []
    for candidate in candidates:
        word = candidate["text"]
        if not total_model_score or word not in word_prob_dict or not total_dict_score:
            score = 0.0
        else:
            score = (
                alpha * (candidate["model_score_prob"] / total_model_score)
                + (1.0 - alpha) * (word_prob_dict[word] / total_dict_score)
            )
        rescored.append({**candidate, "rescore_score": score})
    rescored.sort(key=lambda item: item["rescore_score"], reverse=True)
    return rescored


class IndicXlitPostprocessor:
    def __init__(self, model_root: Path, rescore_alpha: float = 0.9):
        self.model_root = model_root
        self.rescore_alpha = rescore_alpha
        self._symbols: dict[str, list[str]] = {}
        self._word_probs: dict[str, dict] = {}

    def symbols(self, lang: str) -> list[str]:
        if lang not in self._symbols:
            self._symbols[lang] = load_symbols(
                self.model_root / "v1.0" / "corpus-bin" / f"dict.{lang}.txt",
                self.model_root / "lang_list.txt",
            )
        return self._symbols[lang]

    def word_probs(self, lang: str) -> dict:
        if lang not in self._word_probs:
            path = self.model_root / "v1.0" / "word_prob_dicts" / f"{lang}_word_prob_dict.json"
            self._word_probs[lang] = ujson.load(open(path, encoding="utf-8")) if path.is_file() else {}
        return self._word_probs[lang]

    def decode_batch(
        self,
        output_ids: np.ndarray,
        langs: list[str],
        cum_log_probs: np.ndarray | None = None,
        sequence_lengths: np.ndarray | None = None,
        topk: int | None = None,
        rescore: bool = False,
        prompt_len: int = 1,
    ) -> list[list[str]]:
        output_ids = normalize_output_ids(output_ids, prompt_len=prompt_len)
        batch, beams, _ = output_ids.shape
        if topk is None:
            topk = beams
        sequence_lengths = np.asarray(sequence_lengths) if sequence_lengths is not None else None
        results = []
        for row_index in range(batch):
            lang = langs[row_index]
            symbols = self.symbols(lang)
            candidates = []
            for beam_index in range(min(topk, beams)):
                ids = output_ids[row_index, beam_index]
                if sequence_lengths is not None:
                    if sequence_lengths.ndim == 2:
                        seq_len = int(sequence_lengths[row_index, beam_index])
                    else:
                        seq_len = int(sequence_lengths.reshape(-1)[row_index * beams + beam_index])
                    ids = ids[:seq_len]
                raw = decode_ids(ids, symbols)
                text = postprocess_raw(raw)
                model_score = 1.0
                if cum_log_probs is not None:
                    score_array = np.asarray(cum_log_probs)
                    if score_array.ndim == 2:
                        model_score = math.exp(float(score_array[row_index, beam_index]))
                    elif score_array.size > row_index * beams + beam_index:
                        model_score = math.exp(float(score_array.reshape(-1)[row_index * beams + beam_index]))
                candidates.append({"text": text, "raw": raw, "model_score_prob": model_score})
            if rescore:
                candidates = rescore_candidates(candidates, self.word_probs(lang), self.rescore_alpha)
            results.append([candidate["text"] for candidate in candidates])
        return results


class TritonPythonModel:
    def initialize(self, args):
        config = json.loads(args["model_config"])
        params = config.get("parameters", {})
        default_model_root = _repo_root() / "app" / "ai4bharat" / "transliteration" / "transformer" / "models" / "en2indic"
        model_root = Path(params.get("model_root", {}).get("string_value", str(default_model_root)))
        alpha = float(params.get("rescore_alpha", {}).get("string_value", "0.9"))
        self.prompt_len = int(params.get("strip_decoder_prompt_len", {}).get("string_value", "0"))
        self.processor = IndicXlitPostprocessor(model_root, alpha)

    def execute(self, requests):
        responses = []
        for request in requests:
            tokens = pb_utils.get_input_tensor_by_name(request, "TOKENS_BATCH").as_numpy()
            lang_tensor = pb_utils.get_input_tensor_by_name(request, "TARGET_LANG")
            topk_tensor = pb_utils.get_input_tensor_by_name(request, "TOPK")
            rescore_tensor = pb_utils.get_input_tensor_by_name(request, "RESCORE")
            cum_log_probs_tensor = pb_utils.get_input_tensor_by_name(request, "CUM_LOG_PROBS")
            sequence_lengths_tensor = pb_utils.get_input_tensor_by_name(request, "SEQUENCE_LENGTH")

            langs = ["hi"] * tokens.shape[0]
            if lang_tensor is not None:
                lang_values = lang_tensor.as_numpy().reshape(-1)
                langs = [normalize_text_value(value) for value in lang_values]
            topk = None
            if topk_tensor is not None:
                topk = int(topk_tensor.as_numpy().reshape(-1)[0])
            rescore = False
            if rescore_tensor is not None:
                rescore = bool(rescore_tensor.as_numpy().reshape(-1)[0])
            cum_log_probs = cum_log_probs_tensor.as_numpy() if cum_log_probs_tensor is not None else None
            sequence_lengths = (
                sequence_lengths_tensor.as_numpy() if sequence_lengths_tensor is not None else None
            )

            decoded = self.processor.decode_batch(
                tokens,
                langs,
                cum_log_probs,
                sequence_lengths,
                topk,
                rescore,
                prompt_len=self.prompt_len,
            )
            best = np.asarray([[row[0] if row else ""] for row in decoded], dtype=object)
            candidates = np.asarray([[json.dumps(row, ensure_ascii=False)] for row in decoded], dtype=object)
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("TEXT_OUTPUT", best),
                        pb_utils.Tensor("CANDIDATES_JSON", candidates),
                    ]
                )
            )
        return responses
