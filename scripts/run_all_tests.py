"""
Project-IQ: Complete Automated Test Runner
Runs all unit and integration test suites in tests/ and outputs a structured summary.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_all_tests():
    loader = unittest.TestLoader()
    suite = loader.discover(str(PROJECT_ROOT / "tests"), pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 60)

    if not result.wasSuccessful():
        sys.exit(1)
    return result


if __name__ == "__main__":
    run_all_tests()
