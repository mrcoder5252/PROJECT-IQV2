"""
Adversarial & Boundary Verification Test Suite for Project-IQ (Tier 2)
Covers corrupt files, 0-byte edge cases, extreme token bursts, zero overlap,
adversarial PII obfuscation, citation meta-character resilience, and API key absence fallback.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.chunk_for_rag import HARD_MAX_TOKENS, chunk_all, chunk_text, count_tokens
from src.clean import clean_text, redact_text
from src.extract import extract_all
from src.gap_analyzer import GapAnalyzer
from src.vector_store import ProjectIQVectorStore


class TestAdversarial(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="projectiq_adversarial_test_"))

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_corrupt_pdf_file_handling(self):
        raw_dir = self.test_dir / "raw"
        out_dir = self.test_dir / "extracted"
        raw_dir.mkdir(parents=True, exist_ok=True)

        corrupt_pdf = raw_dir / "corrupted_document.pdf"
        corrupt_pdf.write_bytes(b"NOT_A_VALID_PDF_HEADER_JUST_GARBAGE_BYTES_1234567890")

        manifest = extract_all(raw_dir, out_dir)
        self.assertEqual(manifest["total_files"], 1)
        self.assertEqual(manifest["extracted_files"], 0)
        self.assertEqual(manifest["failed_files"], 1)

    def test_corrupt_docx_file_handling(self):
        raw_dir = self.test_dir / "raw"
        out_dir = self.test_dir / "extracted"
        raw_dir.mkdir(parents=True, exist_ok=True)

        corrupt_docx = raw_dir / "corrupted_archive.docx"
        corrupt_docx.write_bytes(b"PK\x03\x04_CORRUPT_ZIP_PAYLOAD_WITHOUT_WORD_DIR")

        manifest = extract_all(raw_dir, out_dir)
        self.assertEqual(manifest["total_files"], 1)
        self.assertEqual(manifest["extracted_files"], 0)
        self.assertEqual(manifest["failed_files"], 1)

    def test_empty_zero_byte_files(self):
        raw_dir = self.test_dir / "raw"
        out_dir = self.test_dir / "extracted"
        raw_dir.mkdir(parents=True, exist_ok=True)

        (raw_dir / "empty.pdf").write_bytes(b"")
        (raw_dir / "empty.docx").write_bytes(b"")
        (raw_dir / "empty.pptx").write_bytes(b"")

        manifest = extract_all(raw_dir, out_dir)
        self.assertEqual(manifest["total_files"], 3)
        self.assertEqual(manifest["failed_files"], 3)

    def test_extreme_token_burst_hard_ceiling(self):
        # A continuous massive text without punctuation or section headings
        massive_text = "unsegmentedtokenstring " * 2500
        chunks = chunk_text(massive_text, chunk_size=450, overlap=80)
        self.assertGreater(len(chunks), 0)

        for idx, c in enumerate(chunks):
            tc = count_tokens(c)
            self.assertLessEqual(
                tc,
                HARD_MAX_TOKENS,
                f"Chunk {idx} violated hard token boundary: {tc} > {HARD_MAX_TOKENS}"
            )

    def test_zero_overlap_chunking(self):
        sample_text = (
            "Paragraph one describing computer vision architectures in medical scans. " * 30
            + "\n\n"
            + "Paragraph two discussing hyperparameter optimization with AdamW. " * 30
        )
        chunks = chunk_text(sample_text, chunk_size=300, overlap=0)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(count_tokens(c), HARD_MAX_TOKENS)

    def test_clean_text_adversarial_control_characters(self):
        nasty_input = "Corrupted\x00data\u00adstring\twith\x08control   characters.\n\n\n\n\n\n\nNext line."
        cleaned = clean_text(nasty_input)
        self.assertNotIn("\x00", cleaned)
        self.assertNotIn("\u00ad", cleaned)
        self.assertNotIn("\n\n\n", cleaned)
        self.assertIn("Corrupted data string with", cleaned)

    def test_adversarial_pii_obfuscation(self):
        obfuscated_text = (
            "Reach student at student.researcher [at] sub.college [dot] edu or lead(at)lab(dot)org. "
            "Phone is +91-98765-43210. Student ID: PRN #72018492K. "
            "Backend auth: access_token = ghp_99281829471928471928"
        )
        redacted, counts = redact_text(obfuscated_text, local_names=[])
        self.assertIn("[EMAIL]", redacted)
        self.assertNotIn("student.researcher", redacted)
        self.assertIn("[PHONE]", redacted)
        self.assertNotIn("98765-43210", redacted)
        self.assertIn("[STUDENT_ID]", redacted)
        self.assertNotIn("72018492K", redacted)
        self.assertIn("[REDACTED_CREDENTIAL]", redacted)
        self.assertNotIn("ghp_99281829471928471928", redacted)

    def test_citation_meta_character_preservation_complex(self):
        citation_text = (
            "Benchmark comparisons based on IEEE standards [1], [2], and [3]-[5]. "
            "According to recent findings (Vaswani et al., 2017) and (LeCun & Bengio, 2020), "
            "sparse attention outperforms recurrent baselines. Contact: author@ai.edu"
        )
        redacted, counts = redact_text(citation_text, local_names=[])
        self.assertIn("[1]", redacted)
        self.assertIn("[2]", redacted)
        self.assertIn("(Vaswani et al., 2017)", redacted)
        self.assertIn("(LeCun & Bengio, 2020)", redacted)
        self.assertIn("[EMAIL]", redacted)
        self.assertEqual(counts["student_id"], 0)

    def test_empty_cleaned_dir_chunk_all(self):
        cleaned_dir = self.test_dir / "cleaned_empty"
        cleaned_dir.mkdir(parents=True, exist_ok=True)
        chunks_file = self.test_dir / "chunks_empty.jsonl"

        chunks = chunk_all(cleaned_dir, chunks_file)
        self.assertEqual(len(chunks), 0)
        self.assertTrue(chunks_file.exists())
        self.assertEqual(chunks_file.stat().st_size, 0)

    def test_missing_gemini_api_key_deterministic_degradation(self):
        store = ProjectIQVectorStore(db_path=self.test_dir / "chroma_dummy")
        with patch.dict(os.environ, {}, clear=True):
            analyzer = GapAnalyzer(vector_store=store, api_key=None)
            self.assertIsNone(analyzer.api_key)
            proposals = analyzer.generate_novel_projects(domain_filter="Edge Systems", num_proposals=2)
            self.assertEqual(len(proposals), 2)
            self.assertIn("title", proposals[0])
            self.assertIn("feasibility_score", proposals[0])
            self.assertGreaterEqual(proposals[0]["feasibility_score"], 1)


if __name__ == "__main__":
    unittest.main()
