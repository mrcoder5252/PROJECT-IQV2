"""
Unit tests for Project-IQ Production Logger (src/logger.py)
"""

import logging
import tempfile
import unittest
from pathlib import Path
from src.logger import get_logger, setup_logger


class TestLogger(unittest.TestCase):
    def test_setup_logger_creates_file(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = setup_logger("test_module", log_file=log_file, level="DEBUG")

            self.assertEqual(logger.level, logging.DEBUG)
            logger.info("Test log message for unit testing.")

            # Flush & close handlers for Windows file release
            for h in list(logger.handlers):
                h.flush()
                h.close()
                logger.removeHandler(h)

            self.assertTrue(log_file.exists())
            content = log_file.read_text(encoding="utf-8")
            self.assertIn("Test log message for unit testing.", content)
            self.assertIn("INFO", content)

    def test_get_logger_singleton(self):
        l1 = get_logger("singleton_test")
        l2 = get_logger("singleton_test")
        self.assertIs(l1, l2)


if __name__ == "__main__":
    unittest.main()
