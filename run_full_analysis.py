"""Run full analysis with GUI - for testing."""

import subprocess
import sys

# Run spectral-predict with GUI
print("=" * 70)
print("RUNNING FULL ANALYSIS WITH GUI")
print("=" * 70)
print()
print("This will:")
print("1. Open GUI with data preview and plots")
print("2. You can click 'Convert to Absorbance' if desired")
print("3. Click 'Continue to Model Search' when ready")
print("4. Model search will run (this takes a few minutes)")
print("5. Results will be saved to outputs/ and reports/")
print()
print("=" * 70)
print()

# Run the command
cmd = [
    sys.executable, "-m", "spectral_predict.cli",
    "--asd-dir", "example/quick_start/",
    "--reference", "example/quick_start/reference.csv",
    "--id-column", "File Number",
    "--target", "%Collagen",
    "--gui"
]

result = subprocess.run(cmd, cwd="C:/Users/sponheim/git/dasp")

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print()
print("Check the results:")
print("  - outputs/results.csv")
print("  - reports/%Collagen.md")
print()

sys.exit(result.returncode)
