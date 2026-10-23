"""
Project-IQ: Centralized configuration.

* All settings come from environment variables (optionally loaded from `.env`).
* Canonical prefix is ``PROJECTIQ_``; the legacy ``PROJECT_IQ_`` prefix is still accepted.
* Secrets (``GEMINI_API_KEY``, ``PROJECTIQ_ADMIN_PASSKEY``) are read live through
  properties so a key set after import (Streamlit secrets, tests, a `.env` reload)
  is always honoured instead of being frozen at import time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
_PREFIXES = ("PROJECTIQ_", "PROJECT_IQ_")


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read ``PROJECTIQ_<name>`` (or legacy ``PROJECT_IQ_<name>``); empty values count as unset."""
    for prefix in _PREFIXES:
        value = os.getenv(prefix + name)
        if value not in (None, ""):
            return value
    return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _env_path(name: str, default: Path) -> Path:
    value = _env(name)
    return Path(value) if value else default


@dataclass
class Config:
    # --- paths -------------------------------------------------------------
    root_dir: Path = ROOT_DIR
    data_dir: Path = field(default_factory=lambda: _env_path("DATA_DIR", ROOT_DIR / "data"))
    raw_dir: Path = field(default_factory=lambda: _env_path("RAW_DIR", ROOT_DIR / "data" / "raw"))
    extracted_dir: Path = field(default_factory=lambda: _env_path("EXTRACTED_DIR", ROOT_DIR / "data" / "extracted"))
    cleaned_dir: Path = field(default_factory=lambda: _env_path("CLEANED_DIR", ROOT_DIR / "data" / "cleaned"))
    chunks_file: Path = field(default_factory=lambda: _env_path("CHUNKS_FILE", ROOT_DIR / "data" / "chunks" / "chunks.jsonl"))
    chroma_dir: Path = field(default_factory=lambda: _env_path("CHROMA_DIR", ROOT_DIR / "data" / "chroma_db"))
    reports_dir: Path = field(default_factory=lambda: _env_path("REPORTS_DIR", ROOT_DIR / "data" / "reports"))
    logs_dir: Path = field(default_factory=lambda: _env_path("LOGS_DIR", ROOT_DIR / "logs"))

    # --- models ------------------------------------------------------------
    embedding_model_name: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))  # type: ignore[arg-type]
    llm_model_name: str = field(default_factory=lambda: _env("LLM_MODEL", "gemini-3.5-flash"))  # type: ignore[arg-type]
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))  # type: ignore[arg-type]

    # --- chunking ----------------------------------------------------------
    target_chunk_tokens: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 450))
    chunk_overlap_tokens: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 80))
    hard_max_tokens: int = field(default_factory=lambda: _env_int("HARD_MAX_TOKENS", 520))
    min_chunk_words: int = 20
    shingle_size: int = 6
    min_shared_shingles: int = 2
    jaccard_threshold: float = 0.60

    # --- retrieval / originality ------------------------------------------
    top_k_retrieval: int = 5
    # Cosine-similarity cut-offs. NOT yet calibrated on labelled duplicate pairs:
    # treat them as tunable defaults and override via env until you have labels.
    duplicate_threshold: float = field(default_factory=lambda: _env_float("DUPLICATE_THRESHOLD", 0.70))
    moderate_threshold: float = field(default_factory=lambda: _env_float("MODERATE_THRESHOLD", 0.45))
    min_relevance: float = field(default_factory=lambda: _env_float("MIN_RELEVANCE", 0.20))

    # --- secrets (live, never frozen at import) ------------------------------
    _gemini_override: Optional[str] = field(default=None, repr=False)

    @property
    def gemini_api_key(self) -> Optional[str]:
        if self._gemini_override:
            return self._gemini_override
        return os.getenv("GEMINI_API_KEY") or None

    @gemini_api_key.setter
    def gemini_api_key(self, value: Optional[str]) -> None:
        self._gemini_override = value or None

    @property
    def admin_passkey(self) -> Optional[str]:
        """Admin portal passkey. There is deliberately NO default: unset means the portal stays locked."""
        return _env("ADMIN_PASSKEY")

    def ensure_directories(self) -> None:
        """Create every directory the pipeline writes to."""
        for d in (self.data_dir, self.raw_dir, self.extracted_dir, self.cleaned_dir,
                  self.chunks_file.parent, self.chroma_dir, self.reports_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)


config = Config()
ProjectIQConfig = Config
config.ensure_directories()
