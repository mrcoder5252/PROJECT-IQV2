"""
Project-IQ: Centralized Production Configuration
Provides environment-driven settings, path resolution, and default hyperparameters.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

@dataclass
class Config:
    root_dir: Path = ROOT_DIR
    data_dir: Path = ROOT_DIR / "data"
    raw_dir: Path = field(default_factory=lambda: Path(os.getenv("PROJECTIQ_RAW_DIR", ROOT_DIR / "data" / "raw")))
    extracted_dir: Path = field(default_factory=lambda: Path(os.getenv("PROJECTIQ_EXTRACTED_DIR", ROOT_DIR / "data" / "extracted")))
    cleaned_dir: Path = field(default_factory=lambda: Path(os.getenv("PROJECTIQ_CLEANED_DIR", ROOT_DIR / "data" / "cleaned")))
    chunks_file: Path = field(default_factory=lambda: Path(os.getenv("PROJECTIQ_CHUNKS_FILE", ROOT_DIR / "data" / "chunks" / "chunks.jsonl")))
    chroma_dir: Path = field(default_factory=lambda: Path(os.getenv("PROJECTIQ_CHROMA_DIR", ROOT_DIR / "data" / "chroma_db")))
    reports_dir: Path = field(default_factory=lambda: Path(os.getenv("PROJECTIQ_REPORTS_DIR", ROOT_DIR / "data" / "reports")))
    logs_dir: Path = field(default_factory=lambda: Path(os.getenv("PROJECTIQ_LOGS_DIR", ROOT_DIR / "logs")))

    embedding_model_name: str = os.getenv("PROJECTIQ_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    llm_model_name: str = os.getenv("PROJECTIQ_LLM_MODEL", "gemini-1.5-flash")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    log_level: str = os.getenv("PROJECTIQ_LOG_LEVEL", "INFO")

    target_chunk_tokens: int = int(os.getenv("PROJECTIQ_CHUNK_SIZE", "450"))
    chunk_overlap_tokens: int = int(os.getenv("PROJECTIQ_CHUNK_OVERLAP", "80"))
    hard_max_tokens: int = int(os.getenv("PROJECTIQ_HARD_MAX_TOKENS", "520"))
    min_chunk_words: int = 20
    shingle_size: int = 6
    min_shared_shingles: int = 2
    jaccard_threshold: float = 0.60
    top_k_retrieval: int = 5

    def ensure_directories(self):
        """Ensure all required directories exist."""
        for d in [self.data_dir, self.raw_dir, self.extracted_dir, self.cleaned_dir,
                  self.chunks_file.parent, self.chroma_dir, self.reports_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

config = Config()
ProjectIQConfig = Config
config.ensure_directories()
