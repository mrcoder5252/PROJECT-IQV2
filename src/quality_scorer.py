"""
Project-IQ: Rule-Based Proposal Quality Scorer.

This is a transparent, deterministic heuristic (regex + keyword counts + fixed weights).
It is NOT a trained model. ``feature_contributions`` shows how far each dimension sits
from a neutral baseline, weighted - a simple additive breakdown, not SHAP values.
(``shap_attributions`` is kept as a deprecated alias for older callers.)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.logger import get_logger

logger = get_logger("project_iq.quality_scorer")

BASELINE = 50.0

_METRIC_WORDS = re.compile(
    r"\b(accuracy|precision|recall|f1(?:[\s-]*score)?|latency|throughput|fps|bleu|auc|rmse|mae|roc|mse)\b", re.I
)
_METRIC_MAP = re.compile(r"\bmAP\b")  # case-sensitive: avoids the English word "map"
_METRIC_PERCENT = re.compile(r"\d+(?:\.\d+)?\s?%")
_METRIC_MS = re.compile(r"\b\d+(?:\.\d+)?\s?(?:ms|milliseconds?)\b", re.I)

_STEPS = re.compile(
    r"(?im)(?:^\s*\d+[.)]\s)|\bstep\s*\d|\bphase\s*\d|\b(?:pipeline|architecture|workflow|algorithm)\b"
)
_CITATIONS = re.compile(r"\[\d+(?:\s*[,\-–]\s*\d+)*\]|\([A-Z][A-Za-z]+(?: et al\.)?,\s*\d{4}\)")
_PROBLEM_MARKERS = re.compile(
    r"(?i)\b(limitation|bottleneck|inefficien\w*|lack of|challenge|vulnerabilit\w*|gap|overhead)\b"
)
_FULL_LIFECYCLE = {"docker", "fastapi", "flask", "react", "streamlit", "aws", "gcp", "kubernetes", "onnx"}

# Domain-label keyword -> evidence keywords expected in the proposal text itself.
# Evidence is checked against the proposal text only (never the domain label), so it is not circular.
_DOMAIN_EVIDENCE = {
    ("vision", "healthcare"): ["vision", "image", "medical", "disease", "pathology", "camera", "cnn"],
    ("agriculture", "iot"): ["iot", "sensor", "hardware", "agriculture", "crop", "plant", "esp32", "arduino"],
    ("security", "blockchain", "cyber"): ["blockchain", "security", "contract", "crypto", "privacy", "encryption"],
    ("cloud", "distributed"): ["cloud", "docker", "api", "microservice", "kubernetes", "distributed"],
    ("nlp", "intelligent", "language"): ["nlp", "text", "bert", "transformer", "llm", "language"],
}


def _word_in(text: str, word: str) -> bool:
    return re.search(r"\b" + re.escape(word) + r"\b", text) is not None


class QualityScorer:
    def __init__(self) -> None:
        self.feature_weights = {
            "methodology_depth": 0.25,
            "metric_rigor": 0.20,
            "literature_grounding": 0.20,
            "tech_stack_coherence": 0.15,
            "problem_clarity": 0.20,
        }

    def extract_features(
        self,
        title: str,
        problem_statement: str,
        proposed_methodology: str,
        tech_stack: List[str] | str,
        citations: Optional[List[Any]] = None,
        domain: Optional[str] = None,
        document_text: Optional[str] = None,
    ) -> Dict[str, float]:
        citations = citations or []
        title_s, prob_s = str(title or ""), str(problem_statement or "")
        meth_s, doc_s = str(proposed_methodology or ""), str(document_text or "")

        if isinstance(tech_stack, str):
            stack_items = [t.strip() for t in re.split(r"[,;|\n]+", tech_stack) if t.strip()]
        elif isinstance(tech_stack, list):
            stack_items = [str(t).strip() for t in tech_stack if str(t).strip()]
        else:
            stack_items = []

        full_text = f"{title_s} {prob_s} {meth_s} {doc_s}"  # proposal evidence only; domain label excluded
        low_text = full_text.lower()

        # 1. Methodology depth: amount of method text + presence of structured steps
        words_meth = len(meth_s.split()) + len(doc_s.split()) // 3
        has_steps = bool(_STEPS.search(full_text))
        meth_score = min(100.0, (words_meth / 25.0) * 55.0 + (45.0 if has_steps else 15.0))

        # 2. Metric rigor: DISTINCT metrics named (repeating "accuracy" ten times does not help)
        metrics = {m.lower() for m in _METRIC_WORDS.findall(full_text)}
        if _METRIC_MAP.search(full_text):
            metrics.add("map")
        if _METRIC_MS.search(full_text):
            metrics.add("latency_ms")
        n_percent = min(2, len(_METRIC_PERCENT.findall(full_text)))  # concrete numbers count, capped
        metric_score = min(100.0, (len(metrics) + n_percent) * 25.0)

        # 3. Literature grounding: distinct in-text citations plus explicitly supplied ones
        in_text = set(_CITATIONS.findall(full_text))
        num_cites = len(citations) + len(in_text)
        grounding_score = min(100.0, num_cites * 25.0)

        # 4. Tech stack: breadth, deployment tooling, and (non-circular) domain evidence
        has_lifecycle = any(s.lower() in _FULL_LIFECYCLE for s in stack_items)
        tech_score = min(100.0, len(stack_items) * 20.0 + (25.0 if has_lifecycle else 0.0))
        if domain:
            dom = str(domain).lower()
            for labels, evidence in _DOMAIN_EVIDENCE.items():
                if any(l in dom for l in labels) and any(_word_in(low_text, e) for e in evidence):
                    tech_score = min(100.0, tech_score + 10.0)
                    break

        # 5. Problem clarity
        has_markers = bool(_PROBLEM_MARKERS.search(prob_s))
        clarity_score = min(100.0, (len(prob_s.split()) / 15.0) * 55.0 + (45.0 if has_markers else 15.0))

        return {
            "methodology_depth": round(meth_score, 1),
            "metric_rigor": round(metric_score, 1),
            "literature_grounding": round(grounding_score, 1),
            "tech_stack_coherence": round(tech_score, 1),
            "problem_clarity": round(clarity_score, 1),
        }

    def score_proposal(
        self,
        title: str,
        problem_statement: str,
        proposed_methodology: str,
        tech_stack: List[str] | str,
        citations: Optional[List[Any]] = None,
        domain: Optional[str] = None,
        document_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        feats = self.extract_features(title, problem_statement, proposed_methodology, tech_stack, citations, domain, document_text)

        contributions = {k: round((feats[k] - BASELINE) * w, 1) for k, w in self.feature_weights.items()}
        overall = round(sum(feats[k] * w for k, w in self.feature_weights.items()), 1)
        overall = min(100.0, max(0.0, overall))
        stars = round(overall / 100.0 * 5.0, 1)

        dimensions = {
            "Innovation & Novelty": min(100.0, round(feats["literature_grounding"] * 0.4 + feats["metric_rigor"] * 0.3 + 30.0, 1)),
            "Technical Feasibility": min(100.0, round(feats["tech_stack_coherence"] * 0.7 + (25.0 if overall >= 50 else 10.0), 1)),
            "Architecture Depth": round(feats["methodology_depth"], 1),
            "Scope Clarity": round(feats["problem_clarity"], 1),
            "Metric Rigor": round(feats["metric_rigor"], 1),
        }

        if overall >= 85:
            grade = "Grade A+ (Distinction)"
        elif overall >= 75:
            grade = "Grade A (Production Grade)"
        elif overall >= 65:
            grade = "Grade B+ (Strong Feasibility)"
        elif overall >= 50:
            grade = "Grade B (Moderate Feasibility)"
        else:
            grade = "Grade C (Preliminary Draft)"

        if overall >= 75:
            tier = "Production Ready / High Feasibility"
        elif overall >= 50:
            tier = "Feasible / Refinements Recommended"
        else:
            tier = "Significant Gaps"

        advice: List[str] = []
        if feats["metric_rigor"] < 50.0:
            advice.append("Define concrete quantitative targets (e.g. latency < 50 ms, F1 > 0.90, throughput benchmarks) and name at least three distinct evaluation metrics.")
        if feats["literature_grounding"] < 40.0:
            advice.append("Cite at least 2-3 prior papers or reference architectures whose limitations you directly address.")
        if feats["methodology_depth"] < 60.0:
            advice.append("Break the architecture into clear numbered phases (data ingestion -> model training/inference -> verification/deployment).")
        if feats["tech_stack_coherence"] < 50.0:
            advice.append("Specify end-to-end technologies for both the intelligence layer (e.g. PyTorch/Transformers) and deployment (e.g. FastAPI/Docker/Streamlit).")
        if feats["problem_clarity"] < 50.0:
            advice.append("State the specific bottlenecks of existing solutions (computational overhead, lack of interpretability, edge latency, ...).")
        if not advice:
            advice.append("Strong proposal: methodology depth, metrics, citations and tooling all meet the heuristic thresholds.")

        return {
            "overall_score": overall,
            "star_rating": stars,
            "grade": grade,
            "readiness_tier": tier,
            "domain": domain or "General Engineering",
            "dimensions": dimensions,
            "features": feats,
            "feature_contributions": contributions,
            "shap_attributions": contributions,  # deprecated alias (kept for older callers)
            "actionable_advice": advice,
            "method": "rule-based heuristic (not a trained model)",
        }
