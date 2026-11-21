"""Bundled SYNTHETIC sample documents (data/<area>/), for testing without real corpora."""
from pathlib import Path

from core.guardrails import PracticeArea

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
AREA_DIRS = {PracticeArea.TAX: "tax", PracticeArea.GRC: "grc", PracticeArea.DEALS: "deals"}


def sample_files(area: PracticeArea) -> list[Path]:
    return sorted((DATA_DIR / AREA_DIRS[area]).glob("*.md"))
