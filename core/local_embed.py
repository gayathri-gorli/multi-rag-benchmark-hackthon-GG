"""Local embeddings from an ALREADY-CACHED BAAI/bge-small-en-v1.5 using only numpy + tokenizers
(no torch/transformers, no downloads). A plain BERT forward pass over the safetensors weights,
CLS pooling, L2-normalised, as in the model's sentence-transformers config.

NOTE: bge-small is 384-d. The spec names bge-large-en-v1.5 (1024-d); this is a stand-in."""
import json
import math
import os
import struct
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings

MODEL_REPO = "models--BAAI--bge-small-en-v1.5"
REQUIRED = ("model.safetensors", "tokenizer.json", "config.json")


def find_model_dir() -> Path | None:
    override = os.environ.get("LOCAL_EMBEDDING_MODEL_DIR")
    roots = [Path(override)] if override else []
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub" / MODEL_REPO / "snapshots"
    if hub.exists():
        roots += sorted(hub.iterdir())
    for d in roots:
        if all((d / f).exists() and (d / f).stat().st_size > 0 for f in REQUIRED):  # stat follows symlinks
            return d
    return None


def _read_safetensors(path: Path) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, meta in header.items():
            if name == "__metadata__" or meta["dtype"] != "F32":
                continue
            s, e = meta["data_offsets"]
            f.seek(base + s)
            out[name] = np.frombuffer(f.read(e - s), dtype="<f4").reshape(meta["shape"])
    return out


def _erf(x: np.ndarray) -> np.ndarray:
    # Abramowitz & Stegun 7.1.26 (max error 1.5e-7) - numpy has no erf and scipy is not installed.
    s = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return s * y


def _gelu(x):
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


class LocalBgeSmallEmbeddings(Embeddings):
    dim = 384
    model_name = "BAAI/bge-small-en-v1.5"

    def __init__(self, model_dir: Path | None = None):
        d = model_dir or find_model_dir()
        if d is None:
            raise FileNotFoundError("bge-small-en-v1.5 is not in the local Hugging Face cache.")
        from tokenizers import Tokenizer

        cfg = json.loads((d / "config.json").read_text())
        self.h, self.L, self.heads = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_attention_heads"]
        self.eps, self.max_len = cfg["layer_norm_eps"], cfg["max_position_embeddings"]
        self.W = _read_safetensors(d / "model.safetensors")
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(max_length=self.max_len)

    def _ln(self, x, prefix):
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + self.eps) * self.W[prefix + ".weight"] + self.W[prefix + ".bias"]

    def _lin(self, x, prefix):
        return x @ self.W[prefix + ".weight"].T + self.W[prefix + ".bias"]

    def _embed_one(self, text: str) -> np.ndarray:
        W = self.W
        ids = np.array(self.tok.encode(text).ids)
        n = len(ids)
        x = W["embeddings.word_embeddings.weight"][ids] + W["embeddings.position_embeddings.weight"][:n] \
            + W["embeddings.token_type_embeddings.weight"][0]
        x = self._ln(x, "embeddings.LayerNorm")
        dh = self.h // self.heads
        for i in range(self.L):
            p = f"encoder.layer.{i}."
            q, k, v = (self._lin(x, p + f"attention.self.{t}").reshape(n, self.heads, dh).transpose(1, 0, 2)
                       for t in ("query", "key", "value"))
            att = q @ k.transpose(0, 2, 1) / math.sqrt(dh)
            att = np.exp(att - att.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            ctx = (att @ v).transpose(1, 0, 2).reshape(n, self.h)
            x = self._ln(x + self._lin(ctx, p + "attention.output.dense"), p + "attention.output.LayerNorm")
            x = self._ln(x + self._lin(_gelu(self._lin(x, p + "intermediate.dense")), p + "output.dense"),
                         p + "output.LayerNorm")
        cls = x[0]  # CLS pooling
        return cls / (np.linalg.norm(cls) or 1.0)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t).astype(float).tolist() for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
