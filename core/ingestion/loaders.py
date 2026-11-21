"""Step 1: load PDF / DOCX / JSON / CSV into page-level records with provenance.
Uses UnstructuredPDFLoader when available (spec), falling back to pypdf."""
import csv
import json
from pathlib import Path


def _pdf(path: Path) -> list[dict]:
    try:
        from langchain_community.document_loaders import UnstructuredPDFLoader

        docs = UnstructuredPDFLoader(str(path), mode="elements").load()
        pages: dict[int, list[str]] = {}
        for d in docs:
            pages.setdefault(d.metadata.get("page_number", 1), []).append(d.page_content)
        return [{"text": "\n".join(t), "page": p} for p, t in sorted(pages.items()) if "".join(t).strip()]
    except Exception:
        from pypdf import PdfReader

        return [{"text": (pg.extract_text() or ""), "page": i + 1}
                for i, pg in enumerate(PdfReader(str(path)).pages)]


def _docx(path: Path) -> list[dict]:
    import docx

    d = docx.Document(str(path))
    text = "\n".join(p.text for p in d.paragraphs if p.text.strip())
    for t in d.tables:  # keep tabular content (e.g. tax tables)
        for row in t.rows:
            text += "\n" + " | ".join(c.text.strip() for c in row.cells)
    return [{"text": text, "page": None}]


def _json(path: Path) -> list[dict]:
    return [{"text": json.dumps(json.loads(path.read_text()), indent=1, ensure_ascii=False), "page": None}]


def _csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [{"text": "; ".join(f"{k}: {v}" for k, v in r.items()), "page": None} for r in rows]


def _txt(path: Path) -> list[dict]:
    return [{"text": path.read_text(encoding="utf-8", errors="ignore"), "page": None}]


LOADERS = {".pdf": _pdf, ".docx": _docx, ".json": _json, ".csv": _csv, ".txt": _txt, ".md": _txt}


def load_file(path: str | Path, practice_area: str) -> list[dict]:
    """Returns [{'text', 'metadata': {source, page, practice_area}}]."""
    path = Path(path)
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return [
        {"text": r["text"], "metadata": {"source": path.name, "page": r["page"], "practice_area": practice_area}}
        for r in loader(path) if r["text"].strip()
    ]
