#!/usr/bin/env python
"""
Coverage analysis script for beta readiness.

Usage:
    python scripts/check_coverage.py
    python scripts/check_coverage.py --threshold 75
    python scripts/check_coverage.py --verbose
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze test coverage for Spectral Predict",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script runs pytest with coverage and analyzes the results.
It generates three report formats:
  - Terminal report with missing lines
  - HTML report in htmlcov/
  - JSON report for programmatic analysis

The script checks coverage against a threshold (default: 70%)
and prints a module-by-module breakdown.

Examples:
  python scripts/check_coverage.py
  python scripts/check_coverage.py --threshold 75
  python scripts/check_coverage.py --verbose
        """
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=70.0,
        help="Minimum coverage threshold percentage (default: 70)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output"
    )
    parser.add_argument(
        "--suite",
        choices=["smoke", "unit", "integration", "fast", "all"],
        default="all",
        help="Test suite to run (default: all)"
    )
    return parser.parse_args()


def run_tests_with_coverage(suite="all", verbose=False, project_root=None):
    """Run pytest with coverage reporting."""
    import os

    cmd = [
        "pytest",
        "--cov=src/spectral_predict",
        "--cov-report=term-missing",
        "--cov-report=html",
        "--cov-report=json",
        "--cov-branch"
    ]

    # Add suite marker if not all
    if suite != "all":
        cmd.extend(["-m", suite])

    if verbose:
        cmd.append("-vv")

    print("Running tests with coverage analysis...")
    print(f"Command: {' '.join(cmd)}")
    print()

    # Set PYTHONPATH to project root
    env = os.environ.copy()
    if project_root:
        env["PYTHONPATH"] = str(project_root)

    result = subprocess.run(cmd, check=False, env=env, cwd=project_root)
    return result.returncode


def load_coverage_json(project_root: Path) -> Dict:
    """Load coverage data from JSON file."""
    coverage_file = project_root / "coverage.json"
    if not coverage_file.exists():
        raise FileNotFoundError(
            f"Coverage JSON not found at {coverage_file}. "
            "Make sure pytest ran successfully."
        )

    with open(coverage_file, "r") as f:
        return json.load(f)


def analyze_coverage(coverage_data: Dict) -> Tuple[float, List[Tuple[str, float]]]:
    """
    Analyze coverage data and return overall percentage and per-module breakdown.

    Returns:
        (overall_percent, [(module_name, percent), ...])
    """
    files = coverage_data.get("files", {})
    module_coverage = []

    for filepath, data in files.items():
        # Convert path to module name
        path_obj = Path(filepath)
        if "spectral_predict" in path_obj.parts:
            # Extract module path relative to spectral_predict
            parts = path_obj.parts
            sp_index = parts.index("spectral_predict")
            module_parts = parts[sp_index:]
            module_name = ".".join(module_parts).replace(".py", "")

            # Get coverage percentage
            summary = data.get("summary", {})
            num_statements = summary.get("num_statements", 0)
            covered = summary.get("covered_lines", 0)

            if num_statements > 0:
                percent = (covered / num_statements) * 100
                module_coverage.append((module_name, percent, num_statements, covered))

    # Calculate overall coverage
    total_percent = coverage_data.get("totals", {}).get("percent_covered", 0.0)

    return total_percent, sorted(module_coverage)


def print_coverage_report(
    total_percent: float,
    module_coverage: List[Tuple[str, float, int, int]],
    threshold: float,
    verbose: bool = False
):
    """Print formatted coverage report."""
    print()
    print("=" * 80)
    print("  COVERAGE ANALYSIS REPORT")
    print("=" * 80)
    print()

    # Overall coverage
    print(f"Overall Coverage: {total_percent:.2f}%")
    print(f"Threshold:        {threshold:.2f}%")
    print()

    # Status
    if total_percent >= threshold:
        status = "PASSED"
        symbol = "✓"
    else:
        status = "FAILED"
        symbol = "✗"

    print(f"Status: {symbol} {status}")
    print()

    # Module breakdown
    print("-" * 80)
    print("Module-by-Module Coverage:")
    print("-" * 80)
    print(f"{'Module':<50} {'Coverage':<12} {'Lines':<15}")
    print("-" * 80)

    for module_name, percent, total_lines, covered_lines in module_coverage:
        # Color-code based on coverage level
        if percent >= 80:
            status_symbol = "✓"
        elif percent >= threshold:
            status_symbol = "~"
        else:
            status_symbol = "✗"

        # Truncate long module names if needed
        display_name = module_name
        if len(display_name) > 48:
            display_name = "..." + display_name[-45:]

        lines_info = f"{covered_lines}/{total_lines}"
        print(f"{status_symbol} {display_name:<48} {percent:>6.2f}%     {lines_info:<15}")

    print("-" * 80)
    print()

    # Summary statistics
    if verbose:
        print("Summary Statistics:")
        print(f"  Total modules: {len(module_coverage)}")

        high_coverage = sum(1 for _, p, _, _ in module_coverage if p >= 80)
        medium_coverage = sum(1 for _, p, _, _ in module_coverage if threshold <= p < 80)
        low_coverage = sum(1 for _, p, _, _ in module_coverage if p < threshold)

        print(f"  High coverage (≥80%):     {high_coverage}")
        print(f"  Medium coverage (≥{threshold}%):  {medium_coverage}")
        print(f"  Low coverage (<{threshold}%):     {low_coverage}")
        print()


def main():
    """Main execution function."""
    args = parse_args()

    # Verify we're in the project root
    project_root = Path(__file__).parent.parent
    if not (project_root / "pyproject.toml").exists():
        print("ERROR: Must run from project root containing pyproject.toml")
        return 1

    print("=" * 80)
    print("  Spectral Predict Coverage Analysis")
    print(f"  Threshold: {args.threshold}%")
    print(f"  Test Suite: {args.suite}")
    print("=" * 80)
    print()

    # Run tests with coverage
    test_result = run_tests_with_coverage(args.suite, args.verbose, project_root)

    if test_result != 0:
        print()
        print("WARNING: Some tests failed, but continuing with coverage analysis...")
        print()

    # Load and analyze coverage data
    try:
        coverage_data = load_coverage_json(project_root)
        total_percent, module_coverage = analyze_coverage(coverage_data)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    except Exception as e:
        print(f"ERROR: Failed to analyze coverage: {e}")
        return 1

    # Print report
    print_coverage_report(total_percent, module_coverage, args.threshold, args.verbose)

    # Print report locations
    print("Coverage Reports Generated:")
    print(f"  - HTML:  {project_root / 'htmlcov' / 'index.html'}")
    print(f"  - JSON:  {project_root / 'coverage.json'}")
    print()

    # Return appropriate exit code
    if total_percent >= args.threshold:
        print(f"SUCCESS: Coverage ({total_percent:.2f}%) meets threshold ({args.threshold}%)")
        return 0
    else:
        print(f"FAILURE: Coverage ({total_percent:.2f}%) below threshold ({args.threshold}%)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
