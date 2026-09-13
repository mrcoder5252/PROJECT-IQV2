"""
Unit tests for Project-IQ Team Matcher (Tier 1: Feature Coverage)
Covers multi-domain taxonomy matching, cosine embedding similarity + keyword overlap composite scoring,
readiness tiering (High/Moderate/Low), skill gap recommendations, and top-k filtering.
"""

import unittest
from src.team_matcher import DOMAIN_TAXONOMY, TeamMatcher


class TestTeamMatcher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matcher = TeamMatcher()

    def test_evaluate_cv_team(self):
        skills = ["Python", "OpenCV", "PyTorch", "CNN", "YOLO"]
        interests = ["Medical Diagnosis", "Healthcare"]
        eval_result = self.matcher.evaluate_team(skills=skills, interests=interests, top_k=2)

        self.assertIn("top_domains", eval_result)
        self.assertGreater(len(eval_result["top_domains"]), 0)

        top = eval_result["top_domains"][0]
        self.assertIn("Computer Vision & Healthcare Diagnostics", top["domain"])
        self.assertGreaterEqual(top["match_percentage"], 50.0)
        self.assertIn("team_readiness", top)
        self.assertIn("suggested_to_learn", top)

    def test_evaluate_blockchain_team(self):
        skills = ["Solidity", "Ethereum", "Smart Contracts", "Web3", "JavaScript"]
        interests = ["Decentralized Finance", "Security"]
        eval_result = self.matcher.evaluate_team(skills=skills, interests=interests, top_k=2)

        top = eval_result["top_domains"][0]
        self.assertIn("Cybersecurity, Privacy & Blockchain Systems", top["domain"])
        self.assertGreaterEqual(top["match_percentage"], 40.0)

    def test_evaluate_iot_agriculture_team(self):
        skills = ["IoT", "Sensors", "ESP32", "Arduino", "MQTT", "Embedded"]
        interests = ["Smart Agriculture", "Soil Monitoring"]
        eval_result = self.matcher.evaluate_team(skills=skills, interests=interests, top_k=1)

        top = eval_result["top_domains"][0]
        self.assertIn("IoT, Edge AI & Smart Agriculture", top["domain"])
        self.assertIn("High", top["team_readiness"])

    def test_evaluate_nlp_team(self):
        skills = ["NLP", "Transformers", "BERT", "HuggingFace", "Python"]
        interests = ["Automated Grading", "Education"]
        eval_result = self.matcher.evaluate_team(skills=skills, interests=interests, top_k=1)

        top = eval_result["top_domains"][0]
        self.assertIn("NLP, LLMs & Educational Technology", top["domain"])
        self.assertGreaterEqual(top["match_percentage"], 50.0)

    def test_readiness_tier_low_for_unrelated_skills(self):
        skills = ["Cooking", "Gardening", "Painting"]
        eval_result = self.matcher.evaluate_team(skills=skills, interests=[], top_k=1)

        top = eval_result["top_domains"][0]
        self.assertEqual(top["team_readiness"], "Low")
        self.assertLess(top["match_percentage"], 45.0)

    def test_top_k_parameter_bounds(self):
        skills = ["Python", "Docker"]
        for k in [1, 3, 5]:
            eval_result = self.matcher.evaluate_team(skills=skills, top_k=k)
            self.assertEqual(len(eval_result["top_domains"]), k)
            self.assertEqual(len(eval_result["all_ranked_domains"]), len(DOMAIN_TAXONOMY))

    def test_suggested_to_learn_recommendations(self):
        # Team with partial skills in Computer Vision
        skills = ["OpenCV", "Python"]
        eval_result = self.matcher.evaluate_team(skills=skills, top_k=1)
        top = eval_result["top_domains"][0]
        self.assertIsInstance(top["suggested_to_learn"], list)
        self.assertGreater(len(top["suggested_to_learn"]), 0)


if __name__ == "__main__":
    unittest.main()
