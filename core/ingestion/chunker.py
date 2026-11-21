"""Step 2 (text part): RecursiveCharacterTextSplitter, 600 tokens / 60 overlap, measured with
the bge tokenizer. Separators prefer clause/section boundaries before sentence/word ones."""
import hashlib

from core.config import Hyperparams
from core.tokenizer import count_tokens

CLAUSE_SEPARATORS = ["\n\n\n", "\n\n", r"\n(?=\s*(?:Section|Rule|Clause|Article)\s+\d)", "\n", r"\. ", " ", ""]


def chunk_records(records: list[dict], hp: Hyperparams | None = None) -> list[dict]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    hp = hp or Hyperparams()
    if hp.chunk_overlap >= hp.chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=hp.chunk_size, chunk_overlap=hp.chunk_overlap,
        length_function=count_tokens, separators=CLAUSE_SEPARATORS, is_separator_regex=True,
    )
    chunks = []
    for rec in records:
        for idx, text in enumerate(splitter.split_text(rec["text"])):
            cid = hashlib.sha1(f"{rec['metadata']['source']}|{rec['metadata']['page']}|{idx}|{text}".encode()).hexdigest()[:16]
            chunks.append({"text": text, "metadata": {**rec["metadata"], "chunk_id": cid, "chunk_index": idx}})
    return chunks
