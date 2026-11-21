"""Token counting. Prefers the bge HF tokenizer (spec step 3), then tiktoken,
then a whitespace estimate. Returns (count, method) so the UI can flag estimates."""
from functools import lru_cache

from core.config import EMBEDDING_MODEL_DEFAULT


@lru_cache(maxsize=1)
def _load():
    try:  # lightweight: tokenizers + a small tokenizer.json, no torch
        from tokenizers import Tokenizer

        tok = Tokenizer.from_pretrained(EMBEDDING_MODEL_DEFAULT)
        return ("bge", lambda t: len(tok.encode(t, add_special_tokens=False).ids))
    except Exception:
        pass
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_DEFAULT)
        return ("bge", lambda t: len(tok.encode(t, add_special_tokens=False)))
    except Exception:
        pass
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return ("tiktoken", lambda t: len(enc.encode(t)))
    except Exception:
        return ("estimate", lambda t: int(len(t.split()) * 1.3) + 1)


def count_tokens(text: str) -> int:
    return _load()[1](text)


def tokenizer_method() -> str:
    return _load()[0]
