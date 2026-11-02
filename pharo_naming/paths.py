"""Default locations used by the pipeline, relative to the package root."""

from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = ROOT_DIR / "results"
CORPUS_DIR = RESULTS_DIR / "corpus"
CACHE_DIR = ROOT_DIR / ".cache"
