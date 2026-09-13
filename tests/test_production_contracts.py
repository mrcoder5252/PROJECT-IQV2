"""
Production Contract & Graceful Fallback Tests for Project-IQ
Verifies defensive programming against null, empty, or unexpected inputs across all core modules.
"""

import tempfile
import unittest
from pathlib import Path
from src.duplicate_detector import DuplicateDetector
from src.gap_analyzer import GapAnalyzer
from src.quality_scorer import QualityScorer
from src.team_matcher import TeamMatcher
from src.vector_store import ProjectIQVectorStore


class TestProductionContracts(unittest.TestCase):
    def test_duplicate_detector_none_inputs(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            store = ProjectIQVectorStore(db_path=Path(tmp))
            detector = DuplicateDetector(vector_store=store)
            res = detector.check_originality(title=None, abstract=None)
            self.assertEqual(res["max_similarity_pct"], 0.0)
            self.assertEqual(res["originality_score"], 100.0)
            self.assertEqual(res["verdict_level"], "info")

    def test_quality_scorer_none_and_edge_inputs(self):
        scorer = QualityScorer()
        res = scorer.score_proposal(
            title=None,
            problem_statement=None,
            proposed_methodology=None,
            tech_stack=None,
            citations=None
        )
        self.assertIn("overall_score", res)
        self.assertIn("star_rating", res)
        self.assertIn("readiness_tier", res)
        self.assertIn("shap_attributions", res)
        self.assertIn("actionable_advice", res)
        self.assertLess(res["overall_score"], 50.0)

    def test_team_matcher_empty_skills_resilience(self):
        matcher = TeamMatcher()
        res = matcher.evaluate_team(skills=[], interests=[], top_k=2)
        self.assertEqual(len(res["top_domains"]), 2)
        self.assertEqual(res["evaluated_skills"], [])
        self.assertEqual(res["top_domains"][0]["team_readiness"], "Low")

    def test_gap_analyzer_fallback_domain_diversity(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            store = ProjectIQVectorStore(db_path=Path(tmp))
            analyzer = GapAnalyzer(vector_store=store, api_key=None)
            proposals = analyzer.generate_novel_projects(
                domain_filter="Cybersecurity & Blockchain",
                num_proposals=2
            )
            self.assertEqual(len(proposals), 2)
            self.assertIn("Cybersecurity", proposals[0]["domain"])
            self.assertGreaterEqual(proposals[0]["feasibility_score"], 1)

    def test_vector_store_empty_db_no_exceptions(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            store = ProjectIQVectorStore(db_path=Path(tmp))
            self.assertEqual(store.count(), 0)
            results = store.query_similar("sample query", n_results=5)
            self.assertEqual(results, [])
            sec_results = store.query_by_section("FUTURE SCOPE", n_results=5)
            self.assertEqual(sec_results, [])


if __name__ == "__main__":
    unittest.main()
