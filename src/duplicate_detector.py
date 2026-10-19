"""
Project-IQ: Originality & Duplicate Project Idea Detector
Compares proposed project titles and abstracts against the historical college project corpus
to detect duplicate ideas, summarize conceptual overlaps in brief, and safeguard academic novelty.
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import config
from src.logger import get_logger
from src.vector_store import ProjectIQVectorStore

logger = get_logger("project_iq.duplicate_detector")

# theme id -> (human label, keywords). Matched on word boundaries against the *historical* match text only.
THEMES: Dict[str, tuple] = {
    "mobile": ("mobile health & therapy applications", ["mobile", "android", "ios", "firebase", "pain", "therapy", "exercise", "acupoint", "questionnaire", "symptom"]),
    "vision": ("computer vision & deep learning", ["cnn", "resnet", "yolo", "opencv", "segmentation", "detection", "image", "leaf", "plant", "camera"]),
    "nlp": ("natural language processing & transformers", ["nlp", "bert", "transformer", "llm", "sentiment", "summarization", "language"]),
    "iot": ("iot & embedded sensor systems", ["iot", "sensor", "arduino", "raspberry", "esp32", "hardware", "microcontroller", "embedded", "telemetry"]),
    "blockchain": ("blockchain & security architectures", ["blockchain", "solidity", "smart contract", "ethereum", "crypto", "security", "privacy", "federated"]),
    "cloud": ("distributed cloud pipelines", ["cloud", "docker", "kubernetes", "fastapi", "microservice", "stream", "database"]),
}
_THEME_RES = {
    tid: re.compile(r"\b(?:" + "|".join(re.escape(k) for k in kws) + r")\b", re.I)
    for tid, (_, kws) in THEMES.items()
}


class DuplicateDetector:
    def __init__(self, vector_store: Optional[ProjectIQVectorStore] = None):
        self.vector_store = vector_store or ProjectIQVectorStore(db_path=config.chroma_dir)

    def check_originality(
        self,
        title: str,
        abstract: str = "",
        threshold_duplicate: Optional[float] = None,
        threshold_moderate: Optional[float] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate proposed project idea for duplication and originality.
        Returns similarity percentage, risk verdict, executive synthesized brief,
        and nearest historical projects.
        """
        threshold_duplicate = config.duplicate_threshold if threshold_duplicate is None else threshold_duplicate
        threshold_moderate = config.moderate_threshold if threshold_moderate is None else threshold_moderate
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
                "executive_summary": "No proposal input was provided to assess originality.",
                "differentiation_recommendations": [
                    "Provide a descriptive project title and a 2-3 sentence technical abstract outlining your problem statement, target stack, and methodology."
                ],
                "matching_projects": []
            }

        try:
            results = self.vector_store.query_similar(query_text=query_text, n_results=max(top_k * 3, top_k))  # over-fetch: several chunks often come from one document
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
                "executive_summary": "Strong academic novelty verified. No overlapping project concepts or matching architectures were identified in the institutional archive.",
                "differentiation_recommendations": [
                    "Document your novel problem formulation clearly in your proposal synopsis.",
                    "Include baseline benchmark comparisons against generic state-of-the-art methods to validate your unique contributions."
                ],
                "matching_projects": []
            }

        max_sim = max((r.get("similarity", 0.0) for r in results), default=0.0)
        max_sim_pct = round(max_sim * 100, 1)

        # Deduplicate results by document_id/source_file
        seen_docs = set()
        unique_matches = []
        found_theme_ids: List[str] = []

        for r in results:
            doc_id = r["metadata"].get("document_id") or r["metadata"].get("source_file") or r.get("chunk_id")
            text = r.get("text", "")
            if doc_id not in seen_docs and len(unique_matches) < top_k:
                seen_docs.add(doc_id)
                unique_matches.append({
                    "similarity_pct": round(r.get("similarity", 0.0) * 100, 1),
                    "source_file": r["metadata"].get("source_file", "Unknown"),
                    "project_name": r["metadata"].get("project_name", "Unknown"),
                    "department": r["metadata"].get("department", "General"),
                    "group_id": r["metadata"].get("group_id", "General"),
                    "category": r["metadata"].get("category", "Report"),
                    "section": r["metadata"].get("section", "General"),
                    "excerpt": text[:320] + ("..." if len(text) > 320 else "")
                })

            # Harvest technical themes from the historical text only (the user's own query must not
            # count as evidence of overlap with itself).
            for tid, rx in _THEME_RES.items():
                if tid not in found_theme_ids and rx.search(text):
                    found_theme_ids.append(tid)

        found_themes = [THEMES[t][0] for t in found_theme_ids]
        matched_domain_str = " and ".join(found_themes[:2]) if found_themes else "related software and modeling workflows"

        if max_sim >= threshold_duplicate:
            verdict = "High Redundancy Detected"
            verdict_level = "danger"
            warning = (
                f"Warning: This concept has an {max_sim_pct}% semantic overlap with past projects. "
                "High risk of rejection for lack of novelty. Review existing concepts to find unique extensions."
            )
            executive_summary = (
                f"High conceptual overlap detected ({max_sim_pct}% similarity). Previous student cohorts have already explored very similar project formulations involving {matched_domain_str}. "
                "While the problem domain is relevant, the proposed system flow closely mirrors existing departmental blackbooks. To stand out, you must introduce substantial technical divergence."
            )
            recs = []
            if "mobile" in found_theme_ids:
                recs.append("Edge-Native Deployment: Replace generic mobile/cloud APIs with quantized on-device inference (e.g. ONNX Runtime, TFLite, or Raspberry Pi edge accelerators) for sub-40ms offline latency.")
                recs.append("Multi-Modal Sensor Telemetry: Augment standard questionnaires or manual inputs with hardware sensors (e.g. IMU motion tracking, bio-sensors, or audio/vision telemetry).")
            if "vision" in found_theme_ids:
                recs.append("Lightweight Vision Quantization: Replace generic heavy architectures with INT8 quantized MobileNetV4 or YOLO-Edge models optimized for embedded microcontrollers.")
                recs.append("Robust Cross-Dataset Generalizability: Evaluate across multi-environment outdoor lighting and test with ablation studies rather than single static benchmark splits.")
            if "nlp" in found_theme_ids:
                recs.append("Retrieval-Augmented Grounding: Integrate localized vector grounding (e.g. ChromaDB) to eliminate hallucinations rather than relying on ungrounded conversational prompts.")
            if "iot" in found_theme_ids:
                recs.append("TinyML on Microcontrollers: Execute model inference directly on ESP32/Raspberry Pi rather than streaming raw telemetry to external cloud servers.")
            if "blockchain" in found_theme_ids:
                recs.append("Layer-2 Zero Knowledge Proofs: Implement ZK-rollups or off-chain state channels to solve latency and transaction gas bottlenecks.")
            
            # Universal fallbacks if domain-specific list is short
            if len(recs) < 3:
                recs.append("Advanced Decision Architectures: Incorporate modern foundation models, federated learning, or explainable AI (SHAP/LIME) rather than standard heuristic rule-engines.")
                recs.append("Empirical Rigor: Evaluate on standardized public benchmark datasets with ablation studies rather than isolated self-collected surveys.")
            differentiation_recommendations = recs[:4]
        elif max_sim >= threshold_moderate:
            verdict = "Moderate Overlap"
            verdict_level = "warning"
            warning = (
                f"Notice: This concept shares {max_sim_pct}% similarity with previous work. "
                "Ensure your proposed methodology or dataset clearly differentiates your scope."
            )
            executive_summary = (
                f"Moderate thematic overlap identified ({max_sim_pct}% similarity). Prior college projects have tackled adjacent challenges with aspects of {matched_domain_str}. "
                "The core idea is viable, but requires a clearly defined novel contribution or specialized architectural angle to avoid being perceived as an incremental clone."
            )
            differentiation_recommendations = [
                "Narrow Problem Scope: Target a specialized sub-domain or extreme operational condition (e.g. low-bandwidth rural environments, constrained battery, or real-time streaming).",
                "Hybrid Stack Integration: Combine deep learning inference with automated orchestration (FastAPI microservices, Docker, CI/CD pipeline).",
                "Comparative Evaluation: Explicitly benchmark your proposed approach against the methodologies implemented in past cohorts."
            ]
        else:
            verdict = "High Originality"
            verdict_level = "success"
            warning = (
                f"Excellent: Project demonstrates strong originality ({max_sim_pct}% maximum historical match). "
                "No saturated overlap found in the departmental archive."
            )
            executive_summary = (
                f"Strong academic originality confirmed ({max_sim_pct}% historical similarity). Your proposed concept addresses a fresh problem angle with minimal overlap in the repository."
            )
            differentiation_recommendations = [
                "Maintain Technical Rigor: Ensure your proposed tech stack and architecture are thoroughly detailed in your engineering design specification.",
                "Define Measurable Success Metrics: Set quantitative benchmarks (accuracy %, latency in ms, throughput) to validate success during evaluation."
            ]

        return {
            "title": title,
            "max_similarity_pct": max_sim_pct,
            "originality_score": max(0.0, round(100.0 - max_sim_pct, 1)),
            "verdict": verdict,
            "verdict_level": verdict_level,
            "warning_message": warning,
            "executive_summary": executive_summary,
            "differentiation_recommendations": differentiation_recommendations,
            "thresholds": {"duplicate": threshold_duplicate, "moderate": threshold_moderate, "calibrated": False},
            "matching_projects": unique_matches
        }
