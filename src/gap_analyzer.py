"""
Project-IQ: Research Gap Analysis & Novel Project Ideation Engine
Extracts technical limitations, constraints, and future scope items across past projects,
synthesizes recurring engineering gaps, and generates novel, well-scoped final year project proposals.
Supports both Google Gemini API and structured local synthesis.
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.llm import generate_text, make_client
from src.logger import get_logger
from src.vector_store import ProjectIQVectorStore

logger = get_logger("project_iq.gap_analyzer")

GAP_INDICATORS = re.compile(
    r"(?i)\b(limitation|future work|future scope|drawback|disadvantage|challenge|"
    r"restrict|bottleneck|inability|further research|unresolved|accuracy drops|"
    r"computational cost|small dataset|manual intervention|edge device|real-time)\b"
)


def compute_feasibility_score(indicators: List[str], similarity: float = 0.0, has_domain_match: bool = False) -> int:
    """
    Transparent 3-10 heuristic (replaces a hard-coded 8): base 5, up to +3 for the number of
    distinct gap indicators found in the evidence, +2 if the evidence is semantically close
    to the requested domain, +1 if a domain filter was supplied and matched.
    """
    score = 5 + min(3, len(set(indicators)))
    if similarity > 0.3:
        score += 2
    if has_domain_match:
        score += 1
    return max(3, min(10, score))


class GapAnalyzer:
    def __init__(self, vector_store: Optional[ProjectIQVectorStore] = None, api_key: Optional[str] = None):
        self.store = vector_store or ProjectIQVectorStore()
        self.api_key = api_key or config.gemini_api_key

    def extract_project_gaps(self, top_k_per_section: int = 15) -> List[Dict[str, Any]]:
        target_sections = ["FUTURE SCOPE", "CONCLUSION", "RESULTS", "METHODOLOGY"]
        collected_chunks = []
        seen_ids = set()

        for sec in target_sections:
            chunks = self.store.query_by_section(section=sec, n_results=top_k_per_section)
            for c in chunks:
                if c["chunk_id"] not in seen_ids:
                    seen_ids.add(c["chunk_id"])
                    collected_chunks.append(c)

        gap_candidates = []
        for c in collected_chunks:
            text = c["text"]
            matches = GAP_INDICATORS.findall(text)
            if matches or c["metadata"].get("section") == "FUTURE SCOPE":
                gap_candidates.append({
                    "chunk_id": c["chunk_id"],
                    "document_id": c["metadata"].get("document_id"),
                    "source_file": c["metadata"].get("source_file"),
                    "category": c["metadata"].get("category"),
                    "section": c["metadata"].get("section"),
                    "matched_indicators": list(set(m.lower() for m in matches)),
                    "text": text
                })

        logger.info(f"Extracted {len(gap_candidates)} candidate gaps across past projects.")
        return gap_candidates

    def generate_novel_projects(
        self,
        domain_filter: Optional[str] = None,
        num_proposals: int = 3
    ) -> List[Dict[str, Any]]:
        gaps = self.extract_project_gaps(top_k_per_section=12)

        if not gaps:
            if not domain_filter:
                logger.info("No research gaps found in repository. Returning empty list.")
                return []
            gaps = [
                {
                    "chunk_id": f"seed_{i+1}",
                    "source_file": f"Curricular Taxonomy ({domain_filter})",
                    "section": "DOMAIN CHALLENGES",
                    "category": "domain_spec",
                    "matched_indicators": ["scalability", "real-time", "latency", "security"],
                    "text": f"Foundational engineering challenge in {domain_filter} requiring modular architecture, low latency, and secure decentralized implementation."
                }
                for i in range(num_proposals)
            ]

        relevant_chunks: List[Dict[str, Any]] = []
        if domain_filter:
            relevant_chunks = self.store.query_similar(query_text=domain_filter, n_results=8)
            context_text = "\n\n---\n\n".join(
                f"[Source: {c['metadata'].get('source_file')} | Section: {c['metadata'].get('section')}]:\n{c['text']}"
                for c in relevant_chunks
            )
        else:
            context_text = "\n\n---\n\n".join(
                f"[Source: {g.get('source_file')} | Section: {g.get('section')}]:\n{g.get('text')}"
                for g in gaps[:10]
            )

        if self.api_key:
            try:
                return self._generate_with_gemini(context_text, domain_filter, num_proposals)
            except Exception as e:
                logger.warning(f"Gemini API generation encountered error: {e}. Falling back to structured gap synthesizer.")

        evidence = gaps
        if relevant_chunks:  # domain-specific evidence first, so local proposals match the requested domain
            evidence = [
                {
                    "chunk_id": c["chunk_id"],
                    "source_file": c["metadata"].get("source_file"),
                    "category": c["metadata"].get("category"),
                    "section": c["metadata"].get("section"),
                    "matched_indicators": sorted({m.lower() for m in GAP_INDICATORS.findall(c["text"])}),
                    "similarity": c.get("similarity", 0.0),
                    "text": c["text"],
                }
                for c in relevant_chunks
            ] + gaps
        return self._generate_local_synthesis(evidence, domain_filter, num_proposals)

    def _generate_with_gemini(
        self,
        context_text: str,
        domain_filter: Optional[str],
        num_proposals: int
    ) -> List[Dict[str, Any]]:
        from google import genai
        from google.genai import types

        client = make_client(self.api_key, timeout_s=60)
        domain_instruction = f" focusing especially on the domain: '{domain_filter}'" if domain_filter else ""

        prompt = f"""You are an elite Senior Engineering Professor and Research Director evaluating final-year undergraduate engineering projects.
Below are excerpts of past student project reports detailing their methodologies, limitations, and future scope:

{context_text}

Analyze the limitations and future work in these past projects. Then design {num_proposals} novel, high-scoring final year project proposals{domain_instruction} that solve these exact gaps.

You MUST respond strictly with a valid JSON array of objects having the following schema:
[
  {{
    "title": "Clear, professional project title",
    "domain": "e.g. Computer Vision & Healthcare / Cloud IoT / NLP",
    "problem_statement": "Specific problem being solved and why past approaches failed",
    "past_project_citations": [
      {{
        "source_file": "Name of past project file cited",
        "identified_gap": "Exact limitation or gap addressed"
      }}
    ],
    "core_innovation": "What makes this project uniquely advanced compared to past work",
    "proposed_methodology": "Step-by-step system pipeline and algorithms",
    "tech_stack": ["Python", "PyTorch", "FastAPI", "..."],
    "feasibility_score": 8,
    "expected_deliverables": ["Web Dashboard", "Trained Model Weights", "REST API", "Benchmark Report"]
  }}
]
"""
        raw_text = generate_text(
            client, prompt,
            types.GenerateContentConfig(temperature=0.3, response_mime_type="application/json"),
        ) or ""
        cleaned_json = raw_text.strip()
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text, re.IGNORECASE)
        if match:
            cleaned_json = match.group(1).strip()
        elif cleaned_json.startswith("```"):
            cleaned_json = re.sub(r"^```(?:json)?\s*", "", cleaned_json, flags=re.IGNORECASE)
            cleaned_json = re.sub(r"\s*```$", "", cleaned_json).strip()
        proposals = json.loads(cleaned_json)
        if isinstance(proposals, dict):  # model wrapped the list in an object
            proposals = next((v for v in proposals.values() if isinstance(v, list)), [proposals])
        if not isinstance(proposals, list):
            raise ValueError("Gemini returned an unexpected JSON shape")
        for p in proposals:
            try:
                p["feasibility_score"] = max(1, min(10, int(p.get("feasibility_score", 7))))
            except (TypeError, ValueError):
                p["feasibility_score"] = 7
        return proposals[:num_proposals]

    def _generate_local_synthesis(
        self,
        gaps: List[Dict[str, Any]],
        domain_filter: Optional[str],
        num_proposals: int
    ) -> List[Dict[str, Any]]:
        if not gaps:
            logger.info("No research gaps found or provided for proposal synthesis. Returning empty list.")
            return []

        default_domain = domain_filter or "Computer Science & Engineering"
        proposals = []

        tech_catalog = [
            "PyTorch", "TensorFlow", "FastAPI", "Docker", "ONNX Runtime",
            "ChromaDB", "Streamlit", "React", "Kafka", "PostgreSQL",
            "OpenCV", "Scikit-Learn", "BERT", "YOLO", "ResNet"
        ]

        for i, g in enumerate(gaps[:max(num_proposals, 1)]):
            gap_text = g.get("text", "")
            gap_sec = g.get("section", "FUTURE SCOPE")
            src_file = g.get("source_file", "past_project_report.pdf")
            indicators = g.get("matched_indicators", [])

            # Extract technologies explicitly mentioned in gap or relevant defaults
            found_techs = [t for t in tech_catalog if re.search(r"\b" + re.escape(t) + r"\b", gap_text, re.I)]
            if not found_techs:
                found_techs = ["Python", "FastAPI", "PyTorch", "Docker"]
            else:
                for base_dep in ["Python", "FastAPI", "Docker"]:
                    if base_dep not in found_techs:
                        found_techs.append(base_dep)

            # Analyze limitation theme from text and matched indicators
            theme = "Adaptive Optimization & Robustness"
            if any(ind in ["latency", "real-time", "computational cost", "edge device"] for ind in indicators) or re.search(r"latency|real-time|edge|cost|hardware", gap_text, re.I):
                theme = "Edge-Accelerated Low-Latency Optimization"
            elif any(ind in ["small dataset", "data", "manual intervention"] for ind in indicators) or re.search(r"dataset|small dataset|few samples|labeling", gap_text, re.I):
                theme = "Self-Supervised & Few-Shot Generalization"
            elif any(ind in ["accuracy drops", "robustness", "unresolved"] for ind in indicators) or re.search(r"accuracy drops|noise|drift|robust", gap_text, re.I):
                theme = "Robust Multimodal Representation"
            elif any(ind in ["privacy", "security", "vulnerability"] for ind in indicators) or re.search(r"privacy|security|packet loss|breach", gap_text, re.I):
                theme = "Privacy-Preserving Decentralized Pipeline"

            # Domain selection
            if domain_filter:
                domain = domain_filter
            else:
                cat = g.get("category", "")
                if cat and cat not in ["blackbook", "research_paper", "general"]:
                    domain = f"{cat.replace('_', ' ').title()} ({theme})"
                else:
                    domain = f"{default_domain} ({theme})"

            # Extract concise limitation snippet for citations and problem statement
            clean_snippet = re.sub(r"\s+", " ", gap_text).strip()
            if len(clean_snippet) > 140:
                clean_snippet = clean_snippet[:137] + "..."

            citation = {
                "source_file": src_file,
                "identified_gap": f"Reported in {gap_sec}: {clean_snippet}"
            }

            title = f"{theme}: {domain.split('(')[0].strip()}"
            problem_statement = (
                f"Analysis of previous student project '{src_file}' ({gap_sec}) revealed an unaddressed limitation: "
                f"\"{clean_snippet}\". Prior methodologies lacked adaptive mechanisms to overcome these operational constraints."
            )
            core_innovation = (
                f"Directly resolves the limitation highlighted in '{src_file}' by formulating a modular architecture "
                f"combining {theme.lower()} with modern engineering tooling ({', '.join(found_techs[:3])})."
            )
            methodology = (
                f"1. Ingestion and targeted normalization tailored to resolve the bottleneck in '{src_file}'.\n"
                f"2. Implementation of modern architectural components ({', '.join(found_techs[:3])}).\n"
                f"3. Rigorous validation against historical benchmarks from '{src_file}'.\n"
                f"4. Production packaging using containerized services and reproducible evaluation metrics."
            )

            proposals.append({
                "title": title,
                "domain": domain,
                "problem_statement": problem_statement,
                "past_project_citations": [citation],
                "core_innovation": core_innovation,
                "proposed_methodology": methodology,
                "tech_stack": found_techs,
                "feasibility_score": compute_feasibility_score(
                    indicators, g.get("similarity", 0.0), bool(domain_filter)
                ),
                "expected_deliverables": [
                    "Engineered Pipeline & Pre-Trained Weights",
                    "Benchmark Evaluation Report comparing with Previous Baselines",
                    "REST API Service & Container Deployment",
                    "Technical Documentation & Verification Suite"
                ]
            })

            if len(proposals) >= num_proposals:
                break

        # If more proposals requested than unique gaps, synthesize secondary specialized variants
        while len(proposals) < num_proposals and gaps:
            base_idx = len(proposals) % len(gaps)
            base_g = gaps[base_idx]
            var_num = len(proposals) + 1
            src_file = base_g.get("source_file", "past_project_report.pdf")
            snippet = re.sub(r"\s+", " ", base_g.get("text", "")).strip()[:120]
            citation = {
                "source_file": src_file,
                "identified_gap": f"Secondary scope extension from {base_g.get('section', 'FUTURE SCOPE')}: {snippet}..."
            }
            domain = domain_filter or "Advanced Software & Systems"
            proposals.append({
                "title": f"Decentralized Robust Architecture for {domain} (Variant {var_num})",
                "domain": domain,
                "problem_statement": f"Builds upon findings in '{src_file}' to resolve secondary scalability gaps: {snippet}...",
                "past_project_citations": [citation],
                "core_innovation": f"Extends analysis from '{src_file}' through fault-tolerant distributed pipelines and real-time streaming.",
                "proposed_methodology": "1. Distributed telemetry processing. 2. Low-overhead inference. 3. Verification and resilience testing.",
                "tech_stack": ["Python", "FastAPI", "Docker", "PyTorch"],
                "feasibility_score": compute_feasibility_score(
                    base_g.get("matched_indicators", []), base_g.get("similarity", 0.0), bool(domain_filter)
                ),
                "expected_deliverables": [
                    "Distributed Prototype",
                    "Performance Evaluation",
                    "Containerized REST API"
                ]
            })

        return proposals


def main():
    parser = argparse.ArgumentParser(description="Analyze project gaps and generate novel project proposals.")
    parser.add_argument("--domain", default=None, help="Target domain for ideation")
    parser.add_argument("--num", type=int, default=2, help="Number of project proposals to generate")
    parser.add_argument("--db", default=None, help="Path to ChromaDB directory")
    parser.add_argument("--output", default=str(config.reports_dir / "proposals.json"), help="Path to save proposals")
    args = parser.parse_args()

    store = ProjectIQVectorStore(db_path=args.db) if args.db else None
    analyzer = GapAnalyzer(vector_store=store)
    proposals = analyzer.generate_novel_projects(domain_filter=args.domain, num_proposals=args.num)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(proposals, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"Generated {len(proposals)} novel project proposals saved to {out_path}.")
    for p in proposals:
        logger.info(f"* Title: {p['title']} | Domain: {p['domain']}")


if __name__ == "__main__":
    main()
