"""
Project-IQ: Explainable Machine Learning Quality Scorer
Feature 6 from Springer paper: Regression-based quality evaluation of project proposals
with SHAP-style explainability translating feature weights into specific improvement advice.
"""

from __future__ import annotations
import math
import re
from typing import Any, Dict, List, Optional

from src.logger import get_logger

logger = get_logger("project_iq.quality_scorer")


class QualityScorer:
    def __init__(self):
        # Established feature weights inspired by rubric evaluation criteria
        self.feature_weights = {
            "methodology_depth": 0.25,
            "metric_rigor": 0.20,
            "literature_grounding": 0.20,
            "tech_stack_coherence": 0.15,
            "problem_clarity": 0.20
        }

    def extract_features(
        self,
        title: str,
        problem_statement: str,
        proposed_methodology: str,
        tech_stack: List[str] | str,
        citations: Optional[List[Any]] = None
    ) -> Dict[str, float]:
        """Extract quantitative features from proposal text."""
        citations = citations or []
        title_s = str(title or "")
        prob_s = str(problem_statement or "")
        meth_s = str(proposed_methodology or "")

        if isinstance(tech_stack, str):
            stack_items = [t.strip() for t in re.split(r"[,;|\n]+", tech_stack) if t.strip()]
        elif isinstance(tech_stack, list):
            stack_items = [str(t).strip() for t in tech_stack if str(t).strip()]
        else:
            stack_items = []

        full_text = f"{title_s} {prob_s} {meth_s}"

        # 1. Methodology Depth (length, step markers, technical terms)
        words_meth = len(meth_s.split())
        has_steps = bool(re.search(r"\b(1\.|2\.|step\s*\d|phase\s*\d|pipeline|architecture)\b", meth_s, re.I))
        meth_score = min(100.0, (words_meth / 25.0) * 55.0 + (45.0 if has_steps else 15.0))

        # 2. Metric Rigor (quantitative metrics: %, accuracy, ms, F1, latency, etc.)
        metric_matches = re.findall(
            r"(?i)\b(accuracy|precision|recall|f1(?:\s*score)?|latency|throughput|fps|loss|bleu|map|auc|rmse|sub-\d+ms|\d+(?:\.\d+)?%)\b",
            full_text
        )
        metric_score = min(100.0, len(metric_matches) * 35.0)

        # 3. Literature Grounding (citations to past work or papers)
        cite_matches = re.findall(r"\[\d+\]|\(\w+,\s*\d{4}\)", full_text)
        num_cites = len(citations) + len(cite_matches)
        grounding_score = min(100.0, num_cites * 50.0)

        # 4. Tech Stack Coherence (richness and modern engineering tools)
        valid_stack_count = len(stack_items)
        has_full_lifecycle = any(s.lower() in {"docker", "fastapi", "flask", "react", "streamlit", "aws", "gcp"} for s in stack_items)
        tech_score = min(100.0, (valid_stack_count * 20.0) + (25.0 if has_full_lifecycle else 0.0))

        # 5. Problem Clarity (word count & focused problem phrasing)
        words_prob = len(prob_s.split())
        has_problem_markers = bool(re.search(r"(?i)\b(limitation|bottleneck|inefficiency|lack of|challenge|vulnerability|gap)\b", prob_s))
        clarity_score = min(100.0, (words_prob / 15.0) * 55.0 + (45.0 if has_problem_markers else 15.0))

        return {
            "methodology_depth": round(meth_score, 1),
            "metric_rigor": round(metric_score, 1),
            "literature_grounding": round(grounding_score, 1),
            "tech_stack_coherence": round(tech_score, 1),
            "problem_clarity": round(clarity_score, 1)
        }

    def score_proposal(
        self,
        title: str,
        problem_statement: str,
        proposed_methodology: str,
        tech_stack: List[str] | str,
        citations: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculates rubric-aligned quality score (0-100), 5-star rating,
        SHAP-style feature attribution, and targeted improvement advice.
        """
        feats = self.extract_features(
            title=title,
            problem_statement=problem_statement,
            proposed_methodology=proposed_methodology,
            tech_stack=tech_stack,
            citations=citations
        )

        # Base expectation score (prior mean)
        baseline = 50.0

        # Compute feature contributions (SHAP-style deviations from baseline)
        attributions = {}
        total_score = 0.0
        for feat_name, weight in self.feature_weights.items():
            val = feats[feat_name]
            contribution = (val - baseline) * weight
            attributions[feat_name] = round(contribution, 1)
            total_score += val * weight

        overall_score = min(100.0, max(0.0, round(total_score, 1)))
        star_rating = round((overall_score / 100.0) * 5.0, 1)

        # Generate targeted improvement recommendations for lagging areas
        recommendations = []
        if feats["metric_rigor"] < 50.0:
            recommendations.append(
                "Define concrete quantitative performance targets (e.g. inference latency < 50ms, F1-score > 90%, or throughput benchmarks) to strengthen evaluation rigor."
            )
        if feats["literature_grounding"] < 40.0:
            recommendations.append(
                "Ground proposal in academic literature by citing at least 2-3 prior papers or past departmental project reports whose gaps are directly addressed."
            )
        if feats["methodology_depth"] < 60.0:
            recommendations.append(
                "Detail your system architecture into clear sequential phases (Data Ingestion -> Model Training/Inference -> Verification/UI Delivery)."
            )
        if feats["tech_stack_coherence"] < 50.0:
            recommendations.append(
                "Specify end-to-end technologies covering both the core intelligence layer (PyTorch/Transformers) and the deployment pipeline (FastAPI/Docker/Streamlit)."
            )
        if feats["problem_clarity"] < 50.0:
            recommendations.append(
                "State the specific limitations of existing solutions (e.g., computational overhead, lack of interpretability, or poor edge feasibility)."
            )

        if not recommendations:
            recommendations.append("Outstanding proposal specification! Meets all key criteria for faculty committee approval.")

        return {
            "overall_score": overall_score,
            "star_rating": star_rating,
            "readiness_tier": "Approved (Ready for Submission)" if overall_score >= 75 else ("Revision Recommended" if overall_score >= 50 else "Significant Gaps"),
            "features": feats,
            "shap_attributions": attributions,
            "actionable_advice": recommendations
        }
