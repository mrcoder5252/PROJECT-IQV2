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
from src.logger import get_logger
from src.vector_store import ProjectIQVectorStore

logger = get_logger("project_iq.gap_analyzer")


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
        gap_indicators = re.compile(
            r"(?i)\b(limitation|future work|future scope|drawback|disadvantage|challenge|"
            r"restrict|bottleneck|inability|further research|unresolved|accuracy drops|"
            r"computational cost|small dataset|manual intervention|edge device|real-time)\b"
        )

        for c in collected_chunks:
            text = c["text"]
            matches = gap_indicators.findall(text)
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

        return self._generate_local_synthesis(gaps, domain_filter, num_proposals)

    def _generate_with_gemini(
        self,
        context_text: str,
        domain_filter: Optional[str],
        num_proposals: int
    ) -> List[Dict[str, Any]]:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
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
        response = client.models.generate_content(
            model=config.llm_model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                response_mime_type="application/json"
            )
        )
        proposals = json.loads(response.text)
        return proposals

    def _generate_local_synthesis(
        self,
        gaps: List[Dict[str, Any]],
        domain_filter: Optional[str],
        num_proposals: int
    ) -> List[Dict[str, Any]]:
        domain = domain_filter or "Artificial Intelligence & Edge Systems"
        citations = []
        for g in gaps[:4]:
            citations.append({
                "source_file": g.get("source_file", "past_project_report.pdf"),
                "identified_gap": f"Identified limitation in {g.get('section', 'FUTURE SCOPE')}: {g.get('text', '')[:120]}..."
            })

        if not citations:
            citations.append({
                "source_file": "past_student_corpus",
                "identified_gap": "Past implementations relied on centralized batch processing without edge inference or real-time adaptation."
            })

        # Pre-configured rich structured templates covering all institutional engineering domains
        proposals_pool = [
            {
                "title": f"Next-Gen Adaptive {domain.title()}: Real-Time Edge-Optimized Multi-Modal Framework",
                "domain": domain,
                "problem_statement": "Past final-year student implementations suffered from high latency, reliance on static datasets, and lack of real-time edge deployment capabilities.",
                "past_project_citations": citations[:2],
                "core_innovation": "Integrates lightweight quantized neural architectures (ONNX/TensorRT) with asynchronous event-driven streaming to enable sub-50ms inference on commodity edge hardware.",
                "proposed_methodology": "1. Multi-modal data ingestion and normalization. 2. Edge model quantization and knowledge distillation. 3. Zero-shot anomaly detection with fallback alerts. 4. Interactive telemetry monitoring dashboard.",
                "tech_stack": ["Python", "FastAPI", "PyTorch / ONNX Runtime", "ChromaDB", "Streamlit", "Docker"],
                "feasibility_score": 9,
                "expected_deliverables": [
                    "Lightweight Edge-Ready Model Pipeline",
                    "Real-Time REST & WebSocket API",
                    "Comprehensive Benchmark Comparison against Past Projects",
                    "Interactive Web Verification Dashboard"
                ]
            },
            {
                "title": f"Privacy-Preserving Federated Intelligence for {domain.title()}",
                "domain": f"Privacy & {domain}",
                "problem_statement": "Previous student projects required centralized collection of sensitive user records, introducing severe data privacy and compliance risks.",
                "past_project_citations": citations[1:3] if len(citations) > 2 else citations[:1],
                "core_innovation": "Applies federated learning with differential privacy guarantees, training decentralized models locally on distributed client nodes without ever transmitting raw data to a central server.",
                "proposed_methodology": "1. Local client node parameter training. 2. Secure federated aggregation server (FedAvg with DP noise). 3. Integrity verification and drift monitoring. 4. End-to-end evaluation against centralized benchmarks.",
                "tech_stack": ["Python", "Flower (FL framework)", "PyTorch", "FastAPI", "React / Streamlit"],
                "feasibility_score": 8,
                "expected_deliverables": [
                    "Decentralized Client-Server Simulation",
                    "Differential Privacy Verification Suite",
                    "Performance & Accuracy Trade-Off Report",
                    "Full Engineering Documentation & API"
                ]
            },
            {
                "title": f"Autonomous Self-Supervised Telemetry & Anomaly Detection in {domain.title()}",
                "domain": domain,
                "problem_statement": "Prior departmental projects heavily depended on manual labeling and static thresholding, causing high false alarm rates under changing environmental conditions.",
                "past_project_citations": citations[:1],
                "core_innovation": "Combines temporal transformer embeddings with self-supervised contrastive learning to detect anomalies without manual threshold calibration.",
                "proposed_methodology": "1. Telemetry ingestion via MQTT/Kafka streams. 2. Contrastive representation pre-training. 3. Online drift detection and adaptive alert filtering. 4. Live reporting interface.",
                "tech_stack": ["Python", "PyTorch", "Kafka", "PostgreSQL", "Docker", "Grafana / Streamlit"],
                "feasibility_score": 8,
                "expected_deliverables": [
                    "Self-Supervised Contrastive Encoder",
                    "Live Telemetry Streaming Pipeline",
                    "Ablation Study on Detection Latency and False Positives",
                    "Interactive Operational Dashboard"
                ]
            }
        ]
        return proposals_pool[:num_proposals]


def main():
    parser = argparse.ArgumentParser(description="Analyze project gaps and generate novel project proposals.")
    parser.add_argument("--domain", default=None, help="Target domain for ideation")
    parser.add_argument("--num", type=int, default=2, help="Number of project proposals to generate")
    parser.add_argument("--db", default=None, help="Path to ChromaDB directory")
    parser.add_argument("--output", default="data/reports/proposals.json", help="Path to save proposals")
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
