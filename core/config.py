"""Hyperparameters (spec section 4). Defaults are the problem-statement values."""
from pydantic import BaseModel, Field


class Hyperparams(BaseModel):
    chunk_size: int = Field(600, ge=100, le=2000, description="tokens per chunk")
    chunk_overlap: int = Field(60, ge=0, le=500, description="token overlap")
    hyde_n: int = Field(3, ge=1, le=6, description="Multi-HyDE passage count N")
    hop_k: int = Field(2, ge=1, le=4, description="Graph traversal hop radius k")
    rrf_alpha: int = Field(60, ge=1, le=200, description="RRF constant alpha")
    # Not in the spec's four; operational setting for how many chunks reach synthesis.
    top_k: int = Field(5, ge=1, le=20)


EMBEDDING_MODEL_DEFAULT = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024
EMBEDDING_MAX_TOKENS = 512  # bge-large-en-v1.5 hard limit; chunks are 600 per spec
