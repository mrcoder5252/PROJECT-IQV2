"""
Project-IQ: Similarity & Duplicate Project Detector
Feature 3 from Springer paper: Compares proposed project titles and abstracts
against the historical college project corpus to detect duplicate ideas, warn students
about saturation, and safeguard academic novelty.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import config
from src.logger import get_logger
from src.vector_store import ProjectIQVectorStore

logger = get_logger("project_iq.duplicate_detector")


class DuplicateDetector:
    def __init__(self, vector_store: Optional[ProjectIQVectorStore] = None):
        self.vector_store = vector_store or ProjectIQVectorStore(db_path=config.chroma_dir)

    def check_originality(
        self,
        title: str,
        abstract: str = "",
        threshold_duplicate: float = 0.82,
        threshold_moderate: float = 0.65,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate proposed project idea for duplication and originality.
        Returns similarity percentage, risk verdict, and nearest historical projects.
        """
        t_clean = (str(title) if title is not None else "").strip()
        a_clean = (str(abstract) if abstract is not None else "").strip()
        parts = [p for p in [t_clean, a_clean] if p]
        query_text = ". ".join(parts)

        if not query_text:
            return {
                "title": title,
                "max_similarity_pct": 0.0,
                "originality_score": 100.0,
                "verdict": "Empty Input",
                "verdict_level": "info",
                "warning_message": "Please provide a project title or abstract to assess originality.",
                "matching_projects": []
            }

        try:
            results = self.vector_store.query_similar(query_text=query_text, n_results=top_k)
        except Exception as e:
            logger.warning(f"Error querying vector store for originality check: {e}")
            results = []

        if not results:
            return {
                "title": title,
                "max_similarity_pct": 0.0,
                "originality_score": 100.0,
                "verdict": "Completely Novel",
                "verdict_level": "success",
                "warning_message": "No closely matching projects found in institutional repository.",
                "matching_projects": []
            }

        max_sim = max((r.get("similarity", 0.0) for r in results), default=0.0)
        max_sim_pct = round(max_sim * 100, 1)

        if max_sim >= threshold_duplicate:
            verdict = "High Redundancy Detected"
            verdict_level = "danger"
            warning = (
                f"Warning: This concept has an {max_sim_pct}% semantic overlap with past projects. "
                "High risk of rejection for lack of novelty. Review existing projects to find unique extensions."
            )
        elif max_sim >= threshold_moderate:
            verdict = "Moderate Overlap"
            verdict_level = "warning"
            warning = (
                f"Notice: This concept shares {max_sim_pct}% similarity with previous work. "
                "Ensure your proposed methodology or dataset clearly differentiates your scope."
            )
        else:
            verdict = "High Originality"
            verdict_level = "success"
            warning = (
                f"Excellent: Project demonstrates strong originality ({max_sim_pct}% maximum historical match). "
                "No saturated overlap found in the departmental archive."
            )

        # Deduplicate results by document_id/project_name
        seen_docs = set()
        unique_matches = []
        for r in results:
            doc_id = r["metadata"].get("document_id") or r["metadata"].get("source_file")
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                unique_matches.append({
                    "similarity_pct": round(r.get("similarity", 0.0) * 100, 1),
                    "source_file": r["metadata"].get("source_file", "Unknown"),
                    "project_name": r["metadata"].get("project_name", "Unknown"),
                    "department": r["metadata"].get("department", "General"),
                    "group_id": r["metadata"].get("group_id", "General"),
                    "category": r["metadata"].get("category", "Report"),
                    "section": r["metadata"].get("section", "General"),
                    "excerpt": r.get("text", "")[:320] + ("..." if len(r.get("text", "")) > 320 else "")
                })

        return {
            "title": title,
            "max_similarity_pct": max_sim_pct,
            "originality_score": max(0.0, round(100.0 - max_sim_pct, 1)),
            "verdict": verdict,
            "verdict_level": verdict_level,
            "warning_message": warning,
            "matching_projects": unique_matches
        }
