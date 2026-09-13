"""
Unit tests for Project-IQ Privacy Sanitization & Front-Matter Cleaning (Tier 1: Feature Coverage)
Covers administrative front-matter stripping, granular PII redaction,
credential masking, citation preservation, and deduplication.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.clean import (
    clean_all,
    clean_record_data,
    discover_local_names,
    get_nlp,
    heading_like,
    is_admin_front_matter,
    redact_text,
)


class TestCleaning(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="projectiq_clean_test_"))
        self.nlp = get_nlp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_pii_redaction_email_phone_sid(self):
        text = "Contact student John at john.doe@college.edu or +91 9876543210. PRN: 72018492K."
        redacted, counts = redact_text(text, local_names=["John Doe", "John"])
        self.assertIn("[EMAIL]", redacted)
        self.assertNotIn("john.doe@college.edu", redacted)
        self.assertIn("[PHONE]", redacted)
        self.assertNotIn("9876543210", redacted)
        self.assertIn("[STUDENT_ID]", redacted)
        self.assertNotIn("72018492K", redacted)
        self.assertIn("[NAME]", redacted)
        self.assertEqual(counts["email"], 1)
        self.assertEqual(counts["phone"], 1)
        self.assertEqual(counts["student_id"], 1)

    def test_pii_redaction_credentials(self):
        text = "Database connection: api_key = sk_live_992182049281 and password: supersecretpwd!"
        redacted, counts = redact_text(text, local_names=[])
        self.assertIn("[REDACTED_CREDENTIAL]", redacted)
        self.assertNotIn("sk_live_992182049281", redacted)
        self.assertNotIn("supersecretpwd!", redacted)
        self.assertGreaterEqual(counts["credential"], 1)

    def test_citation_preservation(self):
        text = (
            "According to LeCun et al. [1], deep convolution enables robust vision features. "
            "Recent transformer architectures (Vaswani et al., 2017) outperform recurrent units [12]."
        )
        redacted, counts = redact_text(text, local_names=[])
        self.assertIn("[1]", redacted)
        self.assertIn("[12]", redacted)
        self.assertIn("(Vaswani et al., 2017)", redacted)
        self.assertEqual(counts["email"], 0)
        self.assertEqual(counts["phone"], 0)
        self.assertEqual(counts["student_id"], 0)

    def test_admin_front_matter_detection(self):
        cert_text = "This is to certify that the project report entitled Smart Agriculture is a bonafide work..."
        ack_text = "Acknowledgements: We express our sincere gratitude to our guide and faculty members..."
        decl_text = "Declaration of candidate: I hereby declare that this work is original."
        plag_text = "Turnitin Similarity Report: Overall index 8%."
        body_text = "The deep convolutional neural network was trained on 10,000 augmented leaf images."

        self.assertTrue(is_admin_front_matter(cert_text))
        self.assertTrue(is_admin_front_matter(ack_text))
        self.assertTrue(is_admin_front_matter(decl_text))
        self.assertTrue(is_admin_front_matter(plag_text))
        self.assertFalse(is_admin_front_matter(body_text))

    def test_heading_like(self):
        self.assertTrue(heading_like("CHAPTER 3 METHODOLOGY"))
        self.assertTrue(heading_like("3.1 SYSTEM ARCHITECTURE"))
        self.assertTrue(heading_like("FUTURE SCOPE"))
        self.assertTrue(heading_like("RESULTS AND DISCUSSION"))
        self.assertFalse(heading_like("This is a long sentence explaining the algorithmic breakdown of our system in full detail."))
        self.assertFalse(heading_like(""))

    def test_discover_local_names(self):
        blocks = [
            {
                "text": "SUBMITTED BY:\nVAIBHAV KADAM (Roll No: 42)\nAMIT DESHMUKH\nGUIDED BY: PROF. S. K. SHARMA",
                "block_type": "paragraph"
            },
            {
                "text": "The proposed ResNet-50 architecture was evaluated on 5,000 test images.",
                "block_type": "paragraph"
            }
        ]
        names = discover_local_names(blocks, self.nlp)
        self.assertIsInstance(names, list)
        self.assertGreater(len(names), 0)

    def test_clean_record_data_pipeline(self):
        raw_record = {
            "document": {
                "document_id": "doc_1234567890ab",
                "source_file": "report.pdf",
                "department": "CS_1",
                "group_id": "Group 1",
                "category": "blackbook",
                "file_sha256": "abc123sha"
            },
            "blocks": [
                {
                    "block_id": "b0",
                    "text": "This is to certify that student Rahul Sharma carried out this bonafide project.",
                    "page": 1,
                    "block_type": "paragraph"
                },
                {
                    "block_id": "b1",
                    "text": "Contact author at rahul.sharma@college.edu or +91 9123456780. Roll No: CS-101.",
                    "page": 1,
                    "block_type": "paragraph"
                },
                {
                    "block_id": "b2",
                    "text": "METHODOLOGY\nWe implemented a custom YOLOv8 model for real-time edge detection [1].",
                    "page": 2,
                    "block_type": "paragraph"
                }
            ]
        }

        cleaned = clean_record_data(raw_record, self.nlp)
        self.assertIn("document", cleaned)
        self.assertIn("privacy", cleaned)
        self.assertIn("blocks", cleaned)
        # Block b0 is front matter, should be stripped
        block_texts = [b["text"] for b in cleaned["blocks"]]
        self.assertFalse(any("bonafide project" in t for t in block_texts))
        # Block b1 has PII redacted
        self.assertTrue(any("[EMAIL]" in t for t in block_texts))
        self.assertTrue(any("[PHONE]" in t for t in block_texts))
        self.assertTrue(any("[STUDENT_ID]" in t for t in block_texts))
        # Block b2 preserves methodology and citation
        self.assertTrue(any("YOLOv8" in t and "[1]" in t for t in block_texts))

    def test_clean_all_deduplication_and_summary(self):
        inp_dir = self.test_dir / "extracted"
        out_dir = self.test_dir / "cleaned"
        report_dir = self.test_dir / "reports"
        inp_dir.mkdir(parents=True, exist_ok=True)

        doc1 = {
            "document": {"document_id": "doc_1", "file_sha256": "same_hash_123", "source_file": "file1.pdf"},
            "blocks": [{"block_id": "b1", "text": "Unique research content on neural networks."}]
        }
        # Duplicate doc with same file_sha256
        doc2 = {
            "document": {"document_id": "doc_2", "file_sha256": "same_hash_123", "source_file": "file2.pdf"},
            "blocks": [{"block_id": "b1", "text": "Unique research content on neural networks."}]
        }

        (inp_dir / "doc_1.json").write_text(json.dumps(doc1), encoding="utf-8")
        (inp_dir / "doc_2.json").write_text(json.dumps(doc2), encoding="utf-8")

        summary = clean_all(inp_dir, out_dir, report_dir)
        # Only 1 unique document should be cleaned due to hash deduplication
        self.assertEqual(summary["cleaned_documents"], 1)
        self.assertTrue((report_dir / "cleaning_report.json").exists())


if __name__ == "__main__":
    unittest.main()
