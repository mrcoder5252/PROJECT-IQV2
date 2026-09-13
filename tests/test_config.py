"""
Unit tests for Project-IQ Production Configuration (src/config.py)
"""

import os
import unittest
from pathlib import Path
from src.config import ProjectIQConfig, config


class TestConfig(unittest.TestCase):
    def test_default_paths(self):
        self.assertTrue(config.root_dir.exists())
        self.assertIsInstance(config.target_chunk_tokens, int)
        self.assertEqual(config.hard_max_tokens, 520)
        self.assertGreater(config.target_chunk_tokens, 0)
        self.assertLess(config.target_chunk_tokens, config.hard_max_tokens)

    def test_environment_override(self):
        custom_cfg = ProjectIQConfig(
            embedding_model_name="test-embedding-model",
            target_chunk_tokens=400,
            hard_max_tokens=500
        )
        self.assertEqual(custom_cfg.embedding_model_name, "test-embedding-model")
        self.assertEqual(custom_cfg.target_chunk_tokens, 400)
        self.assertEqual(custom_cfg.hard_max_tokens, 500)

    def test_ensure_directories(self):
        config.ensure_directories()
        self.assertTrue(config.data_dir.exists())
        self.assertTrue(config.logs_dir.exists())


if __name__ == "__main__":
    unittest.main()
