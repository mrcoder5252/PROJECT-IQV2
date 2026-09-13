"""
Unit tests for Project-IQ Gap Analysis & Ideation Engine (Tier 1: Feature Coverage)
Covers gap extraction from academic sections, limitation indicator keyword matching,
domain filtering, structured schema enforcement, Gemini API integration, and deterministic offline fallback.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.gap_analyzer import GapAnalyzer
from src.vector_store import ProjectIQVectorStore


class TestGapAnalysis(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_gap_store_"))
        self.store = ProjectIQVectorStore(db_path=self.temp_dir)
        self.chunks_file = self.temp_dir / "chunks.jsonl"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _seed_chunks(self):
        records = [
            {
                "chunk_id": "g1",
                "document_id": "doc1",
                "source_file": "leaf_disease.pdf",
                "category": "blackbook",
                "section": "FUTURE SCOPE",
                "is_canonical": True,
                "text": "Future work must address high computational cost on edge device and lack of real-time inference."
            },
            {
                "chunk_id": "g2",
                "document_id": "doc2",
                "source_file": "vital_monitor.docx",
                "category": "research_paper",
                "section": "CONCLUSION",
                "is_canonical": True,
                "text": "A major drawback is battery life limitation and vulnerability to packet loss in remote areas."
            },
            {
                "chunk_id": "g3",
                "document_id": "doc3",
                "source_file": "general_methods.pdf",
                "category": "blackbook",
                "section": "METHODOLOGY",
                "is_canonical": True,
                "text": "We configured standard hyper-parameters across all convolutional layers."
            }
        ]
        with open(self.chunks_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        self.store.index_chunks(self.chunks_file)

    def test_local_synthesis_structure(self):
        analyzer = GapAnalyzer(vector_store=self.store, api_key=None)
        mock_gaps = [
            {
                "chunk_id": "g1",
                "source_file": "leaf_disease.pdf",
                "section": "FUTURE SCOPE",
                "text": "The model has high latency on edge devices and requires constant Wi-Fi connectivity."
            }
        ]
        proposals = analyzer._generate_local_synthesis(
            gaps=mock_gaps,
            domain_filter="Edge Computing & IoT",
            num_proposals=2
        )
        self.assertEqual(len(proposals), 2)
        p = proposals[0]
        self.assertIn("title", p)
        self.assertIn("domain", p)
        self.assertIn("problem_statement", p)
        self.assertIn("past_project_citations", p)
        self.assertIn("core_innovation", p)
        self.assertIn("proposed_methodology", p)
        self.assertIn("tech_stack", p)
        self.assertIn("feasibility_score", p)
        self.assertIn("expected_deliverables", p)
        self.assertIsInstance(p["past_project_citations"], list)
        self.assertIn("source_file", p["past_project_citations"][0])
        self.assertIn("identified_gap", p["past_project_citations"][0])

    def test_extract_project_gaps_indicators(self):
        self._seed_chunks()
        analyzer = GapAnalyzer(vector_store=self.store, api_key=None)
        gaps = analyzer.extract_project_gaps(top_k_per_section=5)

        self.assertGreaterEqual(len(gaps), 2)
        gap_ids = [g["chunk_id"] for g in gaps]
        self.assertIn("g1", gap_ids)
        self.assertIn("g2", gap_ids)
        # g3 in METHODOLOGY without limitation indicators should not be selected
        self.assertNotIn("g3", gap_ids)

        for g in gaps:
            self.assertIn("matched_indicators", g)
            self.assertIn("text", g)
            self.assertIn("source_file", g)

    def test_generate_novel_projects_offline_fallback(self):
        self._seed_chunks()
        analyzer = GapAnalyzer(vector_store=self.store, api_key=None)
        proposals = analyzer.generate_novel_projects(domain_filter="Smart Agriculture", num_proposals=2)
        self.assertEqual(len(proposals), 2)
        self.assertIn("Smart Agriculture", proposals[0]["domain"])

    def test_generate_novel_projects_domain_propagation(self):
        self._seed_chunks()
        analyzer = GapAnalyzer(vector_store=self.store, api_key=None)
        domain = "Cybersecurity & Blockchain"
        proposals = analyzer.generate_novel_projects(domain_filter=domain, num_proposals=1)
        self.assertEqual(len(proposals), 1)
        self.assertTrue(domain.lower() in proposals[0]["title"].lower() or domain.lower() in proposals[0]["domain"].lower())

    def test_generate_with_gemini_mocked(self):
        analyzer = GapAnalyzer(vector_store=self.store, api_key="fake-test-key")
        mock_response_json = json.dumps([
            {
                "title": "Quantum-Resilient Federated Zero-Knowledge Authentication",
                "domain": "Cybersecurity & Privacy",
                "problem_statement": "Past projects lacked cryptographic security against lattice attacks.",
                "past_project_citations": [
                    {"source_file": "past_crypto.pdf", "identified_gap": "Single point of failure"}
                ],
                "core_innovation": "Post-quantum lattice cryptography integrated with zk-SNARKs.",
                "proposed_methodology": "1. Key generation. 2. Zero knowledge proof verification.",
                "tech_stack": ["Python", "Rust", "Solidity"],
                "feasibility_score": 9,
                "expected_deliverables": ["Rust Crate", "Smart Contract", "Benchmark Report"]
            }
        ])

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_response = MagicMock()
            mock_response.text = mock_response_json
            mock_client.models.generate_content.return_value = mock_response

            proposals = analyzer._generate_with_gemini("Sample context", "Cybersecurity & Privacy", 1)
            self.assertEqual(len(proposals), 1)
            self.assertEqual(proposals[0]["title"], "Quantum-Resilient Federated Zero-Knowledge Authentication")
            self.assertEqual(proposals[0]["feasibility_score"], 9)

    def test_proposal_schema_completeness(self):
        analyzer = GapAnalyzer(vector_store=self.store, api_key=None)
        proposals = analyzer.generate_novel_projects(num_proposals=2)
        required_keys = [
            "title", "domain", "problem_statement", "past_project_citations",
            "core_innovation", "proposed_methodology", "tech_stack",
            "feasibility_score", "expected_deliverables"
        ]
        for p in proposals:
            for k in required_keys:
                self.assertIn(k, p, f"Missing required proposal key: {k}")
                self.assertTrue(p[k], f"Key {k} has empty value")


if __name__ == "__main__":
    unittest.main()
