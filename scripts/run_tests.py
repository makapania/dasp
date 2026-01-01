#!/usr/bin/env python
"""
Test automation script for AI agent execution.

Usage:
    python scripts/run_tests.py --suite smoke
    python scripts/run_tests.py --suite all --coverage
    python scripts/run_tests.py --suite numerical --verbose
"""

import argparse
import subprocess
import sys
from pathlib import Path

SUITES = {
    "smoke": ["-m", "smoke"],
    "unit": ["-m", "unit"],
    "integration": ["-m", "integration"],
    "numerical": ["-m", "numerical"],
    "fast": ["-m", "smoke or unit"],
    "all": []
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run test suites for Spectral Predict",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available test suites:
  smoke       : Quick sanity checks (< 60s)
  unit        : Fast unit tests (< 2 min)
  integration : Multi-module tests (< 5 min)
  numerical   : Mathematical correctness (< 3 min)
  fast        : smoke + unit tests
  all         : All tests (< 10 min)

Examples:
  python scripts/run_tests.py --suite smoke
  python scripts/run_tests.py --suite all --coverage
  python scripts/run_tests.py --suite numerical --verbose
        """
    )
    parser.add_argument(
        "--suite",
        choices=list(SUITES.keys()),
        default="smoke",
        help="Test suite to run (default: smoke)"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--failfast",
        "-x",
        action="store_true",
        help="Stop on first failure"
    )
    return parser.parse_args()


def build_pytest_command(suite_name, coverage=False, verbose=False, failfast=False):
    """Build the pytest command with appropriate arguments."""
    cmd = ["pytest"]

    # Add suite-specific arguments
    cmd.extend(SUITES[suite_name])

    # Add coverage if requested
    if coverage:
        cmd.extend([
            "--cov=src/spectral_predict",
            "--cov-report=term-missing",
            "--cov-report=html",
            "--cov-report=json"
        ])

    # Add verbose flag if requested (unless already in addopts)
    if verbose:
        if "-v" not in cmd and "--verbose" not in cmd:
            cmd.append("-vv")

    # Add failfast if requested
    if failfast:
        cmd.append("-x")

    return cmd


def print_header(suite_name, coverage=False):
    """Print a formatted header for the test run."""
    print("=" * 70)
    print(f"  Spectral Predict Test Suite: {suite_name.upper()}")
    if coverage:
        print("  Coverage Analysis: ENABLED")
    print("=" * 70)
    print()


def print_footer(returncode):
    """Print a formatted footer with test results."""
    print()
    print("=" * 70)
    if returncode == 0:
        print("  RESULT: ALL TESTS PASSED")
        status = "SUCCESS"
    else:
        print("  RESULT: TESTS FAILED")
        status = "FAILURE"
    print("=" * 70)
    return status


def main():
    """Main execution function."""
    args = parse_args()

    # Verify we're in the project root
    project_root = Path(__file__).parent.parent
    if not (project_root / "pyproject.toml").exists():
        print("ERROR: Must run from project root containing pyproject.toml")
        return 1

    # Build the pytest command
    cmd = build_pytest_command(
        args.suite,
        coverage=args.coverage,
        verbose=args.verbose,
        failfast=args.failfast
    )

    # Print header
    print_header(args.suite, args.coverage)

    # Print the command being run
    print(f"Running: {' '.join(cmd)}")
    print()

    # Execute pytest with PYTHONPATH set to project root
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)

    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            env=env,
            check=False
        )
        returncode = result.returncode
    except KeyboardInterrupt:
        print("\n\nTest run interrupted by user")
        returncode = 130
    except Exception as e:
        print(f"\n\nERROR: Failed to execute tests: {e}")
        returncode = 1

    # Print footer with results
    status = print_footer(returncode)

    # Print additional info if coverage was generated
    if args.coverage and returncode == 0:
        print()
        print("Coverage reports generated:")
        print(f"  - Terminal: (shown above)")
        print(f"  - HTML: {project_root / 'htmlcov' / 'index.html'}")
        print(f"  - JSON: {project_root / 'coverage.json'}")

    return returncode


if __name__ == "__main__":
    sys.exit(main())
