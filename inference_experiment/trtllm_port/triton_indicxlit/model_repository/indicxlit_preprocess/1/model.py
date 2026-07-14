import json
import sys
from pathlib import Path

import numpy as np

try:
    import triton_python_backend_utils as pb_utils
except ImportError:  # Allows local dry-run tests without tritonserver.
    pb_utils = None


SPECIALS = ["<s>", "<pad>", "</s>", "<unk>"]
BOS_ID = 0
PAD_ID = 1
EOS_ID = 2
UNK_ID = 3


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


def load_vocab(dict_path: Path, lang_list: Path) -> dict[str, int]:
    symbols = SPECIALS + read_base_symbols(dict_path) + read_language_tokens(lang_list)
    return {symbol: index for index, symbol in enumerate(symbols)}


def normalize_text_value(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def normalize_lang(value, default_lang: str) -> str:
    lang = normalize_text_value(value).strip().lower()
    return lang or default_lang


def preprocess_words(words: list[str], target_lang: str) -> list[str]:
    return [f"__{target_lang}__ " + " ".join(list(word.lower())) for word in words]


def encode_preprocessed(text: str, token_to_id: dict[str, int]) -> list[int]:
    ids = [token_to_id.get(piece, UNK_ID) for piece in text.split()]
    ids.append(EOS_ID)
    return ids


def pad_rows(rows: list[list[int]], pad_id: int = PAD_ID) -> np.ndarray:
    max_len = max(len(row) for row in rows)
    return np.asarray([row + [pad_id] * (max_len - len(row)) for row in rows], dtype=np.int32)


def encode_batch(
    words: list[str],
    target_lang: str,
    request_output_len: int,
    model_root: Path,
    token_to_id: dict[str, int] | None = None,
) -> dict[str, np.ndarray]:
    if token_to_id is None:
        lang_list = model_root / "lang_list.txt"
        corpus_bin = model_root / "v1.0" / "corpus-bin"
        token_to_id = load_vocab(corpus_bin / "dict.en.txt", lang_list)
    preprocessed = preprocess_words(words, target_lang)
    source_ids = [encode_preprocessed(row, token_to_id) for row in preprocessed]
    input_ids = pad_rows(source_ids, PAD_ID)
    batch_size = len(words)
    cross_attention_mask = np.zeros((batch_size, request_output_len, input_ids.shape[1]), dtype=np.bool_)
    for row_index, row in enumerate(source_ids):
        cross_attention_mask[row_index, :, : len(row)] = True
    return {
        "INPUT_ID": input_ids,
        "CROSS_ATTENTION_MASK": cross_attention_mask,
        "REQUEST_INPUT_LEN": np.asarray([[len(row)] for row in source_ids], dtype=np.int32),
        "DECODER_INPUT_ID": np.full((batch_size, 1), EOS_ID, dtype=np.int32),
        "REQUEST_DECODER_INPUT_LEN": np.ones((batch_size, 1), dtype=np.int32),
        "REQUEST_OUTPUT_LEN": np.full((batch_size, 1), request_output_len, dtype=np.int32),
        "OUT_END_ID": np.full((batch_size, 1), EOS_ID, dtype=np.int32),
        "OUT_PAD_ID": np.full((batch_size, 1), PAD_ID, dtype=np.int32),
        "SOURCE_WORD": np.asarray([[word] for word in words], dtype=object),
        "TARGET_LANG": np.asarray([[target_lang] for _ in words], dtype=object),
    }


class TritonPythonModel:
    def initialize(self, args):
        config = json.loads(args["model_config"])
        params = config.get("parameters", {})
        default_model_root = Path("/models/assets/en2indic")
        self.model_root = Path(
            params.get("model_root", {}).get("string_value", str(default_model_root))
        )
        self.default_lang = params.get("default_lang", {}).get("string_value", "hi").strip().lower() or "hi"
        lang_list = self.model_root / "lang_list.txt"
        corpus_bin = self.model_root / "v1.0" / "corpus-bin"
        self.src_to_id = load_vocab(corpus_bin / "dict.en.txt", lang_list)
        self.supported_langs = set(read_language_tokens(lang_list))

    def execute(self, requests):
        responses = []
        for request in requests:
            text_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
            lang_tensor = pb_utils.get_input_tensor_by_name(request, "TARGET_LANG")
            max_tokens_tensor = pb_utils.get_input_tensor_by_name(request, "REQUEST_OUTPUT_LEN")

            words = [normalize_text_value(row[0]) for row in text_tensor.as_numpy()]
            target_lang = self.default_lang
            if lang_tensor is not None:
                lang_values = lang_tensor.as_numpy()
                if lang_values.size:
                    target_lang = normalize_lang(lang_values.reshape(-1)[0], self.default_lang)
            if target_lang not in self.supported_langs:
                message = (
                    f"Unsupported target_lang={target_lang!r}; "
                    f"supported={sorted(self.supported_langs)}"
                )
                responses.append(pb_utils.InferenceResponse(error=pb_utils.TritonError(message)))
                continue
            request_output_len = 32
            if max_tokens_tensor is not None:
                request_output_len = int(max_tokens_tensor.as_numpy().reshape(-1)[0])

            outputs = encode_batch(words, target_lang, request_output_len, self.model_root, self.src_to_id)
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor(name, value)
                        for name, value in outputs.items()
                    ]
                )
            )
        return responses
