"""
Unit tests for Project-IQ Duplicate Detector and Explainable Quality Scorer.
"""

import unittest
from src.duplicate_detector import DuplicateDetector
from src.quality_scorer import QualityScorer

class TestQualityAndDuplicates(unittest.TestCase):
    def setUp(self):
        self.detector = DuplicateDetector()
        self.scorer = QualityScorer()

    def test_quality_scorer_complete_proposal(self):
        result = self.scorer.score_proposal(
            title="Edge-Optimized CNN for Real-Time Plant Pathology",
            problem_statement="Existing models suffer from severe latency bottlenecks on embedded edge devices.",
            proposed_methodology="1. Collect crop images. 2. Quantize ResNet model via ONNX. 3. Deploy on Raspberry Pi with sub-40ms latency and 96.2% accuracy [1].",
            tech_stack=["PyTorch", "ONNX", "FastAPI", "Docker", "Raspberry Pi"],
            citations=["Ref [1] 2024"]
        )
        self.assertGreaterEqual(result["overall_score"], 70.0)
        self.assertGreaterEqual(result["star_rating"], 3.5)
        self.assertIn("methodology_depth", result["features"])
        self.assertIn("shap_attributions", result)
        self.assertIsInstance(result["actionable_advice"], list)

    def test_quality_scorer_weak_proposal_triggers_advice(self):
        result = self.scorer.score_proposal(
            title="App for stuff",
            problem_statement="Make an app",
            proposed_methodology="Build it",
            tech_stack=["Python"]
        )
        self.assertLess(result["overall_score"], 50.0)
        self.assertEqual(result["readiness_tier"], "Significant Gaps")
        self.assertGreater(len(result["actionable_advice"]), 1)

    def test_duplicate_detector_empty_input(self):
        res = self.detector.check_originality("")
        self.assertEqual(res["max_similarity_pct"], 0.0)
        self.assertEqual(res["verdict_level"], "info")

    def test_duplicate_detector_existing_corpus_topic(self):
        res = self.detector.check_originality(
            title="Acupoint therapy pain management mobile app with AI questionnaire",
            abstract="Android application for musculoskeletal pain relief with Firebase and personalized exercise recommendation."
        )
        self.assertIn("max_similarity_pct", res)
        self.assertIn("matching_projects", res)
        self.assertGreater(len(res["matching_projects"]), 0)

if __name__ == "__main__":
    unittest.main()
