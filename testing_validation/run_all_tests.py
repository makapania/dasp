"""
Automated Test Runner - Complete DASP Validation
=================================================

This script runs the complete validation pipeline:
1. DASP regression tests
2. Comparison with R results
3. Generate reports

Usage: python run_all_tests.py

Options:
  --skip-dasp : Skip DASP tests (use existing results)
  --skip-r : Skip R tests (use existing results)
  --datasets : Specify datasets to test (comma-separated: bone,d13c)
"""

import subprocess
import sys
from pathlib import Path
import time
import argparse

BASE_DIR = Path(__file__).parent

def run_command(description, command, cwd=None, timeout=None):
    """Run a command and handle output."""
    print("\n" + "=" * 80)
    print(f"RUNNING: {description}")
    print("=" * 80)
    print(f"Command: {' '.join(command)}")
    print()

    start_time = time.time()

    try:
        result = subprocess.run(
            command,
            cwd=cwd or BASE_DIR,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        # Print output
        if result.stdout:
            print(result.stdout)

        if result.returncode != 0:
            print(f"\nERROR: Command failed with return code {result.returncode}")
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
            return False

        elapsed = time.time() - start_time
        print(f"\n✓ Completed in {elapsed:.1f} seconds")
        return True

    except subprocess.TimeoutExpired:
        print(f"\nERROR: Command timed out after {timeout} seconds")
        return False
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        return False

def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Run DASP validation tests")
    parser.add_argument('--skip-dasp', action='store_true', help="Skip DASP tests")
    parser.add_argument('--skip-r', action='store_true', help="Skip R tests")
    parser.add_argument('--datasets', type=str, help="Datasets to test (bone,d13c)")
    parser.add_argument('--quick', action='store_true', help="Run quick test (fewer models)")

    args = parser.parse_args()

    print("=" * 80)
    print("DASP VALIDATION TEST SUITE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Skip DASP tests: {args.skip_dasp}")
    print(f"  Skip R tests: {args.skip_r}")
    print(f"  Datasets: {args.datasets or 'all'}")
    print(f"  Quick mode: {args.quick}")

    start_time = time.time()
    results = {}

    # Step 1: Run R tests
    if not args.skip_r:
        print("\n" + "#" * 80)
        print("# STEP 1: R REGRESSION TESTS")
        print("#" * 80)

        success = run_command(
            description="R Regression Tests (Comprehensive)",
            command=["Rscript", "regression_models_comprehensive.R"],
            cwd=BASE_DIR / "r_scripts",
            timeout=1800  # 30 minutes
        )
        results['r_tests'] = success

        if not success:
            print("\nWARNING: R tests failed. Continuing anyway...")
    else:
        print("\n[SKIPPED] R tests")
        results['r_tests'] = None

    # Step 2: Run DASP tests
    if not args.skip_dasp:
        print("\n" + "#" * 80)
        print("# STEP 2: DASP REGRESSION TESTS")
        print("#" * 80)

        python_exe = sys.executable
        success = run_command(
            description="DASP Regression Tests",
            command=[python_exe, "dasp_regression.py"],
            timeout=1800  # 30 minutes
        )
        results['dasp_tests'] = success

        if not success:
            print("\nERROR: DASP tests failed!")
            return 1
    else:
        print("\n[SKIPPED] DASP tests")
        results['dasp_tests'] = None

    # Step 3: Compare results
    print("\n" + "#" * 80)
    print("# STEP 3: COMPARE DASP VS. R RESULTS")
    print("#" * 80)

    python_exe = sys.executable
    success = run_command(
        description="Compare Regression Results",
        command=[python_exe, "compare_regression_results.py"],
        timeout=300  # 5 minutes
    )
    results['comparison'] = success

    if not success:
        print("\nWARNING: Comparison failed. Check if both DASP and R results exist.")

    # Final summary
    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print("\nTest Results:")
    for step, success in results.items():
        if success is None:
            status = "[SKIPPED]"
        elif success:
            status = "[SUCCESS]"
        else:
            status = "[FAILED]"
        print(f"  {step:20s}: {status}")

    print(f"\nTotal time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

    # Check if all tests passed
    failures = [k for k, v in results.items() if v is False]
    if failures:
        print(f"\n❌ FAILED: {len(failures)} test(s) failed: {', '.join(failures)}")
        return 1
    else:
        print("\n✅ SUCCESS: All tests completed successfully!")

        print("\nGenerated files:")
        print("  - results/r_regression/bone_collagen/*")
        print("  - results/r_regression/d13c/*")
        print("  - results/dasp_regression/bone_collagen/*")
        print("  - results/dasp_regression/d13c/*")
        print("  - results/comparisons/bone_collagen_comparison.csv")
        print("  - results/comparisons/d13c_comparison.csv")
        print("  - results/comparisons/comparison_report.md")

        print("\nNext steps:")
        print("  1. Review comparison_report.md")
        print("  2. Check for any models with large discrepancies")
        print("  3. Run classification tests if needed")

        return 0

if __name__ == "__main__":
    sys.exit(main())
