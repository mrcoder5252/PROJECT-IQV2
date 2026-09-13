"""
Unit tests for Project-IQ Vector Storage & Hybrid Retrieval (Tier 1: Feature Coverage)
Covers ChromaDB indexing, SentenceTransformer embedding generation, section-filtered search,
category filtering, thresholding, canonical filtering, and persistence.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.vector_store import ProjectIQVectorStore


class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_chroma_"))
        self.store = ProjectIQVectorStore(db_path=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write_sample_chunks(self, chunks_file: Path):
        sample_records = [
            {
                "chunk_id": "c1",
                "document_id": "doc1",
                "source_file": "doc1.pdf",
                "department": "CS_1",
                "group_id": "Group 1",
                "project_name": "Leaf Disease Detection",
                "category": "blackbook",
                "section": "METHODOLOGY",
                "is_canonical": True,
                "has_citation": True,
                "token_count": 50,
                "word_count": 35,
                "duplicate_group_id": "grp1",
                "text": "We utilized a ResNet-50 deep neural network for leaf disease classification in agriculture."
            },
            {
                "chunk_id": "c2",
                "document_id": "doc1",
                "source_file": "doc1.pdf",
                "department": "CS_1",
                "group_id": "Group 1",
                "project_name": "Leaf Disease Detection",
                "category": "blackbook",
                "section": "FUTURE SCOPE",
                "is_canonical": True,
                "has_citation": False,
                "token_count": 45,
                "word_count": 30,
                "duplicate_group_id": "grp2",
                "text": "Future work will deploy the quantized model on low-power agricultural IoT microcontrollers."
            },
            {
                "chunk_id": "c3_duplicate",
                "document_id": "doc2",
                "source_file": "doc2.docx",
                "department": "CS_1",
                "group_id": "Group 1",
                "project_name": "Leaf Disease Detection Synopsis",
                "category": "abstract_synopsis",
                "section": "METHODOLOGY",
                "is_canonical": False,
                "has_citation": False,
                "token_count": 48,
                "word_count": 32,
                "duplicate_group_id": "grp1",
                "text": "We utilized a ResNet-50 deep neural network for leaf disease classification."
            }
        ]
        with open(chunks_file, "w", encoding="utf-8") as f:
            for r in sample_records:
                f.write(json.dumps(r) + "\n")

    def test_indexing_and_canonical_filtering(self):
        chunks_file = self.temp_dir / "test_chunks.jsonl"
        self._write_sample_chunks(chunks_file)

        # Out of 3 chunks, only 2 are canonical
        indexed = self.store.index_chunks(chunks_file)
        self.assertEqual(indexed, 2)
        self.assertEqual(self.store.count(), 2)

    def test_query_similar_semantic_matching(self):
        chunks_file = self.temp_dir / "test_chunks.jsonl"
        self._write_sample_chunks(chunks_file)
        self.store.index_chunks(chunks_file)

        results = self.store.query_similar("deep neural network vision architecture", n_results=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["chunk_id"], "c1")
        self.assertIn("similarity", results[0])
        self.assertIn("distance", results[0])
        self.assertIn("metadata", results[0])

    def test_query_by_section_with_and_without_query(self):
        chunks_file = self.temp_dir / "test_chunks.jsonl"
        self._write_sample_chunks(chunks_file)
        self.store.index_chunks(chunks_file)

        # Section query without query text (direct get)
        future_chunks = self.store.query_by_section(section="FUTURE SCOPE")
        self.assertEqual(len(future_chunks), 1)
        self.assertEqual(future_chunks[0]["chunk_id"], "c2")

        # Section query with query text (semantic search constrained to section)
        method_chunks = self.store.query_by_section(section="METHODOLOGY", query_text="ResNet")
        self.assertEqual(len(method_chunks), 1)
        self.assertEqual(method_chunks[0]["chunk_id"], "c1")

    def test_query_with_category_filter(self):
        chunks_file = self.temp_dir / "test_chunks.jsonl"
        self._write_sample_chunks(chunks_file)
        self.store.index_chunks(chunks_file)

        results = self.store.query_similar("IoT", category="blackbook", n_results=2)
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertEqual(r["metadata"]["category"], "blackbook")

    def test_empty_chunks_file_handling(self):
        empty_file = self.temp_dir / "empty.jsonl"
        empty_file.write_text("", encoding="utf-8")
        indexed = self.store.index_chunks(empty_file)
        self.assertEqual(indexed, 0)

    def test_missing_chunks_file_raises_error(self):
        non_existent = self.temp_dir / "does_not_exist.jsonl"
        with self.assertRaises(FileNotFoundError):
            self.store.index_chunks(non_existent)

    def test_vector_store_persistence(self):
        chunks_file = self.temp_dir / "test_chunks.jsonl"
        self._write_sample_chunks(chunks_file)
        self.store.index_chunks(chunks_file)
        initial_count = self.store.count()

        # Instantiate a new store instance with the exact same path
        new_store = ProjectIQVectorStore(db_path=self.temp_dir)
        self.assertEqual(new_store.count(), initial_count)


if __name__ == "__main__":
    unittest.main()
