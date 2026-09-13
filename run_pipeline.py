"""
Project-IQ: Unified End-to-End Pipeline Orchestrator
Executes extraction, privacy cleaning, chunking, vector indexing,
gap analysis, and team matching.
"""

from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.logger import get_logger
from src.extract import extract_all
from src.clean import clean_all
from src.chunk_for_rag import chunk_all
from src.vector_store import ProjectIQVectorStore
from src.gap_analyzer import GapAnalyzer
from src.team_matcher import TeamMatcher

logger = get_logger("project_iq.pipeline")


def run_full_pipeline(
    raw_dir: Path,
    extracted_dir: Path,
    cleaned_dir: Path,
    chunks_file: Path,
    chroma_dir: Path,
    reports_dir: Path,
    reset: bool = False,
    domain_filter: str | None = None
) -> Dict[str, Any]:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    logger.info("=" * 60)
    logger.info("STARTING PROJECT-IQ PIPELINE")
    logger.info("=" * 60)
    start_total = time.time()

    if reset:
        logger.info("Reset requested: clearing previous processed outputs...")
        for p in [extracted_dir, cleaned_dir, chunks_file.parent, chroma_dir, reports_dir]:
            if p.exists():
                if p.is_file():
                    p.unlink()
                else:
                    shutil.rmtree(p, ignore_errors=True)

    extracted_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    chunks_file.parent.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extraction
    logger.info("[STAGE 1/5] Extracting raw multi-format academic documents...")
    t0 = time.time()
    manifest = extract_all(raw_dir, extracted_dir)
    logger.info(f"Extraction completed in {time.time() - t0:.2f}s ({manifest['extracted_files']} files extracted)")

    # 2. Cleaning & Privacy
    logger.info("[STAGE 2/5] Cleaning administrative front-matter & sanitizing PII...")
    t0 = time.time()
    clean_summary = clean_all(extracted_dir, cleaned_dir, reports_dir)
    logger.info(f"Sanitization completed in {time.time() - t0:.2f}s ({clean_summary['cleaned_documents']} documents cleaned)")

    # 3. Chunking & Deduplication
    logger.info("[STAGE 3/5] Token-bounded chunking and near-duplicate clustering...")
    t0 = time.time()
    chunks = chunk_all(cleaned_dir, chunks_file)
    logger.info(f"Chunking completed in {time.time() - t0:.2f}s ({len(chunks)} chunks produced)")

    # 4. Vector Storage & Indexing
    logger.info("[STAGE 4/5] Indexing canonical chunks into ChromaDB...")
    t0 = time.time()
    store = ProjectIQVectorStore(db_path=chroma_dir)
    indexed_count = store.index_chunks(chunks_file)
    logger.info(f"Indexing completed in {time.time() - t0:.2f}s ({indexed_count} chunks indexed)")

    # 5. Gap Analysis & Project Ideation
    logger.info("[STAGE 5/5] Extracting research gaps and synthesizing novel project proposals...")
    t0 = time.time()
    analyzer = GapAnalyzer(vector_store=store)
    gaps = analyzer.extract_project_gaps(top_k_per_section=10)
    proposals = analyzer.generate_novel_projects(domain_filter=domain_filter, num_proposals=2)

    proposals_file = reports_dir / "novel_project_proposals.json"
    proposals_file.write_text(json.dumps(proposals, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Ideation completed in {time.time() - t0:.2f}s ({len(proposals)} proposals generated)")

    total_time = time.time() - start_total
    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETE in {total_time:.2f}s")
    logger.info(f"Chunks: {chunks_file}")
    logger.info(f"ChromaDB: {chroma_dir}")
    logger.info(f"Proposals: {proposals_file}")
    logger.info("=" * 60)

    summary_result = {
        "extracted_files": manifest["extracted_files"],
        "cleaned_docs": clean_summary["cleaned_documents"],
        "total_chunks": len(chunks),
        "indexed_chunks": indexed_count,
        "gaps_found": len(gaps),
        "proposals_generated": len(proposals),
        "total_time_seconds": round(total_time, 2)
    }

    # Save summary report
    summary_file = reports_dir / "pipeline_summary.json"
    summary_file.write_text(json.dumps(summary_result, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary_result


def main():
    parser = argparse.ArgumentParser(description="Run the end-to-end Project-IQ processing pipeline.")
    parser.add_argument("--raw-dir", default=str(config.raw_dir), help="Path to raw document directory")
    parser.add_argument("--extracted-dir", default=str(config.extracted_dir), help="Path to extracted directory")
    parser.add_argument("--cleaned-dir", default=str(config.cleaned_dir), help="Path to cleaned directory")
    parser.add_argument("--chunks-file", default=str(config.chunks_file), help="Path to chunks.jsonl")
    parser.add_argument("--chroma-dir", default=str(config.chroma_dir), help="Path to ChromaDB directory")
    parser.add_argument("--reports-dir", default=str(config.reports_dir), help="Path to reports directory")
    parser.add_argument("--reset", action="store_true", help="Clear all previous output data before running")
    parser.add_argument("--domain", default=None, help="Optional domain filter for proposal generation")
    args = parser.parse_args()

    run_full_pipeline(
        raw_dir=Path(args.raw_dir),
        extracted_dir=Path(args.extracted_dir),
        cleaned_dir=Path(args.cleaned_dir),
        chunks_file=Path(args.chunks_file),
        chroma_dir=Path(args.chroma_dir),
        reports_dir=Path(args.reports_dir),
        reset=args.reset,
        domain_filter=args.domain
    )


if __name__ == "__main__":
    main()
