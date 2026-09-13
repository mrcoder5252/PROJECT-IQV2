"""
Unit tests for Project-IQ Token-Bounded Chunking & Deduplication (Tier 1: Feature Coverage)
Covers section-aware splitting, hard token boundary enforcement (<= 520 tokens),
shingling, Jaccard duplicate clustering, ML feature extraction, and chunk manifest generation.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.chunk_for_rag import (
    HARD_MAX_TOKENS,
    MIN_CHUNK_WORDS,
    UnionFind,
    chunk_all,
    chunk_text,
    chunk_text_by_section,
    count_tokens,
    extract_ml_features,
    find_duplicate_clusters,
    shingles,
    split_by_sections,
)


class TestChunking(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="projectiq_chunk_test_"))

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_token_counting(self):
        text = "Deep learning models require structured token boundaries for effective embedding."
        tokens = count_tokens(text)
        self.assertGreater(tokens, 5)
        self.assertLess(tokens, 25)

    def test_section_splitting(self):
        doc = """ABSTRACT
This is the abstract text describing the scope of this project.

METHODOLOGY
Here is the methodology detailing neural architectures and data pipelines.

FUTURE SCOPE
In the future, the model should be ported to mobile microcontrollers.
"""
        sections = split_by_sections(doc)
        sec_names = [s[0] for s in sections]
        self.assertIn("ABSTRACT", sec_names)
        self.assertIn("METHODOLOGY", sec_names)
        self.assertIn("FUTURE SCOPE", sec_names)

    def test_section_splitting_fallback_body(self):
        text = "This document does not contain explicit uppercase section headers anywhere."
        sections = split_by_sections(text)
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0][0], "BODY")

    def test_hard_token_bound(self):
        long_para = "This is an extensive research sentence demonstrating token bounding and chunking safety. " * 60
        chunks = chunk_text(long_para, chunk_size=450, overlap=80)
        self.assertGreater(len(chunks), 0)
        for c in chunks:
            tc = count_tokens(c)
            self.assertLessEqual(tc, HARD_MAX_TOKENS, f"Chunk exceeded hard max: {tc} > {HARD_MAX_TOKENS}")

    def test_shingling(self):
        s = "deep convolutional neural network framework for leaf disease"
        sh = shingles(s, n=4)
        self.assertTrue(any("deep convolutional neural network" in x for x in sh))

    def test_ml_features_extraction(self):
        text = "In 2024, our model achieved 98.4% accuracy [1]. Does it generalize to other crops?"
        feats = extract_ml_features(text)
        self.assertEqual(feats["sentence_count"], 2)
        self.assertTrue(feats["has_numbers"])
        self.assertTrue(feats["has_citation"])

        no_feats = extract_ml_features("Simple text without figures or reference citations")
        self.assertFalse(no_feats["has_numbers"])
        self.assertFalse(no_feats["has_citation"])

    def test_union_find_clustering(self):
        ids = ["c1", "c2", "c3", "c4"]
        uf = UnionFind(ids)
        uf.union("c1", "c2")
        uf.union("c2", "c3")
        self.assertEqual(uf.find("c1"), uf.find("c3"))
        self.assertNotEqual(uf.find("c1"), uf.find("c4"))

    def test_find_duplicate_clusters_high_jaccard(self):
        # Two chunks with identical text should cluster together
        passage = (
            "Decentralized ledger technology enables verifiable audit trails for academic "
            "credential verification across distributed institutions and university departments."
        )
        chunk_records = [
            {"chunk_id": "c1", "text": passage},
            {"chunk_id": "c2", "text": passage},
            {"chunk_id": "c3", "text": "Completely unrelated text concerning agricultural soil moisture telemetry."}
        ]
        clusters = find_duplicate_clusters(chunk_records)
        self.assertEqual(clusters["c1"], clusters["c2"])
        self.assertNotEqual(clusters["c1"], clusters["c3"])

    def test_chunk_all_end_to_end(self):
        cleaned_dir = self.test_dir / "cleaned"
        cleaned_dir.mkdir(parents=True, exist_ok=True)
        chunks_file = self.test_dir / "chunks" / "chunks.jsonl"

        doc = {
            "document": {
                "document_id": "doc_abc123",
                "source_file": "report.pdf",
                "department": "CS_1",
                "group_id": "Group 1",
                "category": "blackbook",
                "document_type": "pdf"
            },
            "text": (
                "ABSTRACT\n"
                "This project investigates deep learning architectures for anomaly detection in cloud microservices. "
                "We benchmark latency, throughput, and error rates across varying simulated workloads.\n\n"
                "METHODOLOGY\n"
                "The system implements an autoencoder trained on Prometheus metrics streams collected every 5 seconds. "
                "Anomalous score thresholds are dynamically adapted using moving average filters.\n\n"
                "FUTURE SCOPE\n"
                "Current latency is 200ms per inference batch. Future iterations must evaluate lightweight ONNX runtimes."
            )
        }
        (cleaned_dir / "doc_abc123_cleaned.json").write_text(json.dumps(doc), encoding="utf-8")

        chunks = chunk_all(cleaned_dir, chunks_file)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(chunks_file.exists())

        for c in chunks:
            self.assertIn("chunk_id", c)
            self.assertIn("document_id", c)
            self.assertIn("section", c)
            self.assertIn("token_count", c)
            self.assertLessEqual(c["token_count"], HARD_MAX_TOKENS)
            self.assertIn("is_canonical", c)


if __name__ == "__main__":
    unittest.main()
