"""
Test runner for Sprint 1 (Imbalance + Outlier Detection).

Run with: python run_sprint1_tests.py
"""

import sys
import subprocess

def run_tests():
    """Run Sprint 1 tests and report results."""
    print("=" * 60)
    print("SPRINT 1 TEST SUITE: Imbalance + Outlier Detection")
    print("=" * 60)
    print()

    # Test 1: Imbalance detection
    print("Running test_imbalance.py...")
    result1 = subprocess.run(
        [sys.executable, "-m", "pytest", "spectral_predict_v3/tests/test_imbalance.py", "-v"],
        capture_output=True,
        text=True
    )
    print(result1.stdout)
    if result1.stderr:
        print(result1.stderr)

    # Test 2: Outlier detection
    print("\n" + "=" * 60)
    print("Running test_outlier_detection.py...")
    result2 = subprocess.run(
        [sys.executable, "-m", "pytest", "spectral_predict_v3/tests/test_outlier_detection.py", "-v"],
        capture_output=True,
        text=True
    )
    print(result2.stdout)
    if result2.stderr:
        print(result2.stderr)

    # Coverage report (if pytest-cov is installed)
    print("\n" + "=" * 60)
    print("Generating coverage report...")
    result3 = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "spectral_predict_v3/tests/test_imbalance.py",
            "spectral_predict_v3/tests/test_outlier_detection.py",
            "--cov=spectral_predict_v3/core/imbalance",
            "--cov=spectral_predict_v3/core/outlier_detection",
            "--cov-report=term-missing"
        ],
        capture_output=True,
        text=True
    )
    print(result3.stdout)
    if result3.stderr:
        print(result3.stderr)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    if result1.returncode == 0 and result2.returncode == 0:
        print("✓ All tests PASSED")
        return 0
    else:
        print("✗ Some tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
