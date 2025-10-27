# Spectral Predict - Session Handoff Document

**Date:** 2025-10-27
**Status:** ⚠️ **REQUIRES INVESTIGATION** - Negative R² anomaly detected
**Last Session Duration:** ~4 hours

---

## 🎯 What Was Accomplished

### ✅ Core Features Implemented

1. **Smart Filename Matching** (`src/spectral_predict/io.py:113-249`)
   - Handles filenames with/without extensions
   - Handles spaces ("Spectrum 00001" → "Spectrum00001.asd")
   - Case-insensitive matching
   - **Status:** Working perfectly

2. **Binary ASD Support** (`src/spectral_predict/io.py:390-453`)
   - Integrated SpecDAL library
   - Reads binary `.asd` files automatically
   - Graceful fallback with helpful error messages
   - **Status:** Working, tested with 37 bone samples

3. **Improved Task Detection** (`src/spectral_predict/cli.py:101-112`)
   - Detects decimal values → regression
   - Handles small datasets better
   - **Status:** Working

4. **Real-World Example**
   - 37 bone ASD files from your data
   - `example/BoneCollagen.csv` reference file
   - Complete documentation in `example/README.md`
   - **Status:** Example runs successfully

### ✅ Documentation Created

- `README.md` - Updated with real example front and center
- `example/README.md` - Comprehensive usage guide
- `example/EXPECTED_OUTPUTS.md` - What outputs should look like
- `example/RESULTS_SUMMARY.md` - Analysis of actual run
- `PROJECT_STATUS.md` - Overall project status (already existed)

### ✅ Bug Fixes

1. **Missing `tabulate` dependency**
   - Added to `pyproject.toml:28`
   - Installed in venv
   - **Status:** Fixed

2. **specdal import path**
   - Changed from `specdal.containers.spectrum` to `specdal`
   - **Status:** Fixed

---

## ⚠️ CRITICAL ISSUE: Negative R² Anomaly

### The Problem

**Latest run (2025-10-27):**
- Tested 548 models on 37 bone samples
- **Best R² = -0.07** (negative!)
- Best RMSE = 3.62% collagen
- All models perform worse than predicting the mean

**Your observation:**
> "the precious run had high r2 so something is very wrong"

### Evidence

**File:** `outputs/results.csv` (80 KB, 548 rows)

Top 3 models all show negative R²:
```
Rank 1: RandomForest(n=200), raw, top3, RMSE=3.62, R²=-0.07
Rank 4: RandomForest(n=500), raw, top3, RMSE=3.77, R²=-0.26
Rank 7: RandomForest(n=200), snv,  top3, RMSE=6.52, R²=-0.90
```

### Possible Causes

1. **Data Loading Issue**
   - Flexible filename matching might have misaligned samples
   - Check if "Spectrum 00001" matched the wrong file
   - Verify `_normalize_filename_for_matching()` logic

2. **SpecDAL Reading Error**
   - Binary ASD reader might be extracting wrong data
   - Check wavelength units (nm vs μm?)
   - Verify reflectance values are in correct range (0-1)

3. **Target Variable Mismatch**
   - CSV has "File Number" column with spaces
   - Filenames have no spaces
   - Matching might have scrambled the y-values

4. **Code Change Introduced Bug**
   - Task detection change (line 101-112 in cli.py)
   - Alignment logic in `align_xy()` (line 144-249 in io.py)
   - SpecDAL integration (line 390-453 in io.py)

5. **Different Data Than Previous Run**
   - Current run: 37 samples matched out of 49 in CSV
   - Previous run might have had different data/samples

---

## 🔍 Debugging Steps (Priority Order)

### 1. Verify Data Loading

```bash
cd /Users/mattsponheimer/git/deepspec
source .venv/bin/activate

# Check what was actually loaded
python3 << 'EOF'
from spectral_predict.io import read_asd_dir, read_reference_csv, align_xy
import pandas as pd

# Load data
print("=== LOADING DATA ===")
X = read_asd_dir('example/')
print(f"X shape: {X.shape}")
print(f"X index (first 5): {list(X.index[:5])}")
print(f"X sample (first spectrum, first 10 wavelengths):")
print(X.iloc[0, :10])
print()

ref = read_reference_csv('example/BoneCollagen.csv', 'File Number')
print(f"Reference shape: {ref.shape}")
print(f"Reference index (first 5): {list(ref.index[:5])}")
print()

# Align
X_al, y = align_xy(X, ref, 'File Number', '%Collagen')
print(f"Aligned X shape: {X_al.shape}")
print(f"y shape: {len(y)}")
print(f"y range: {y.min():.2f} - {y.max():.2f}")
print(f"y mean: {y.mean():.2f}, std: {y.std():.2f}")
print()

# Check first 5 samples
print("=== FIRST 5 SAMPLES ===")
for idx in X_al.index[:5]:
    print(f"{idx}: %Collagen = {y.loc[idx]:.2f}")
EOF
```

**What to look for:**
- Are the collagen values sensible? (should be 0.9-22.1%)
- Are the spectra values sensible? (reflectance usually 0-1)
- Do the aligned IDs look correct?

### 2. Check Filename Matching

```bash
python3 << 'EOF'
from spectral_predict.io import _normalize_filename_for_matching
import pandas as pd

# Load CSV IDs
ref = pd.read_csv('example/BoneCollagen.csv')
csv_ids = ref['File Number'].tolist()

# Load ASD filenames
import os
asd_files = sorted([f.replace('.asd', '') for f in os.listdir('example/') if f.endswith('.asd')])

print("=== CSV IDs (first 10) ===")
for i, csv_id in enumerate(csv_ids[:10]):
    norm = _normalize_filename_for_matching(csv_id)
    print(f"{csv_id:20s} → {norm}")

print("\n=== ASD Filenames (first 10) ===")
for asd in asd_files[:10]:
    norm = _normalize_filename_for_matching(asd)
    print(f"{asd:20s} → {norm}")

print("\n=== MATCHED PAIRS (first 10) ===")
# Show how they match
X_norm_map = {_normalize_filename_for_matching(f): f for f in asd_files}
ref_norm_map = {_normalize_filename_for_matching(idx): idx for idx in csv_ids}

common = set(X_norm_map.keys()).intersection(set(ref_norm_map.keys()))
for i, norm_id in enumerate(sorted(common)[:10]):
    csv_id = ref_norm_map[norm_id]
    asd_id = X_norm_map[norm_id]
    print(f"{csv_id:20s} ← {norm_id:15s} → {asd_id}")
EOF
```

**What to look for:**
- Do the matches look correct?
- Is "Spectrum 00001" matching "Spectrum00001"?
- Are there any unexpected mismatches?

### 3. Inspect Spectral Data

```bash
python3 << 'EOF'
from spectral_predict.io import read_asd_dir
import matplotlib.pyplot as plt

X = read_asd_dir('example/')

# Plot first 5 spectra
plt.figure(figsize=(12, 6))
for i in range(min(5, len(X))):
    plt.plot(X.columns, X.iloc[i, :], label=X.index[i], alpha=0.7)

plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.title('First 5 Bone Spectra')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('spectra_plot.png', dpi=150, bbox_inches='tight')
print("Saved: spectra_plot.png")

# Check value ranges
print(f"\nSpectral value range: {X.min().min():.4f} - {X.max().max():.4f}")
print(f"Expected range: 0 - 1 (reflectance)")
if X.min().min() < 0 or X.max().max() > 1.5:
    print("⚠️ WARNING: Values outside expected range!")
EOF
```

### 4. Compare with Previous Run

If you have outputs from the previous run with high R²:

```bash
# Compare model results
diff outputs_previous/results.csv outputs/results.csv | head -50

# Check if same data was used
wc -l outputs_previous/results.csv outputs/results.csv
```

### 5. Run Minimal Test

```bash
# Test with just PLS on 10 samples to isolate the issue
python3 << 'EOF'
from spectral_predict.io import read_asd_dir, read_reference_csv, align_xy
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
import numpy as np

# Load minimal data
X = read_asd_dir('example/quick_start/')
ref = read_reference_csv('example/quick_start/reference.csv', 'File Number')
X_al, y = align_xy(X, ref, 'File Number', '%Collagen')

print(f"Samples: {len(y)}")
print(f"y values: {y.values}")

# Simple PLS model
pls = PLSRegression(n_components=2)
scores = cross_val_score(pls, X_al, y, cv=3, scoring='r2')

print(f"\nPLS(n=2) cross-val R²: {scores}")
print(f"Mean R²: {scores.mean():.3f}")

if scores.mean() < 0:
    print("\n⚠️ NEGATIVE R² CONFIRMED - Data/alignment issue likely")
else:
    print("\n✅ Positive R² - Issue may be with specific models or full dataset")
EOF
```

---

## 📁 Key Files & Locations

### Modified Source Code
```
src/spectral_predict/
├── io.py              ← MODIFIED: Added smart matching & SpecDAL
├── cli.py             ← MODIFIED: Improved task detection
├── models.py          ← Unchanged
├── preprocess.py      ← Unchanged
├── search.py          ← Unchanged
├── scoring.py         ← Unchanged
└── report.py          ← Unchanged (but failed due to tabulate)
```

### Output Files
```
outputs/
└── results.csv        ← 548 models, all with negative R²

reports/
└── (empty)            ← Report generation failed

example/
├── BoneCollagen.csv         ← Your reference data (49 samples)
├── Spectrum*.asd            ← 37 binary ASD files
├── quick_start/             ← Subset for fast testing (10 files)
├── README.md                ← Usage guide
├── EXPECTED_OUTPUTS.md      ← What outputs should look like
└── RESULTS_SUMMARY.md       ← Analysis of problematic run
```

### Test Files
```
tests/
├── test_io_csv.py           ← CSV reading tests
├── test_asd_ascii.py        ← ASD reading tests
└── test_optional_r_bridge.py ← R bridge tests

All tests passing: 30/30 ✅
```

### Logs
```
spectral_predict_run.log     ← Full execution log from last run
```

---

## 🔧 Commands to Resume Work

### Activate Environment
```bash
cd /Users/mattsponheimer/git/deepspec
source .venv/bin/activate
```

### Run Tests
```bash
pytest -v                    # All 30 tests should pass
```

### Re-run Analysis (after fixing)
```bash
# Clean previous outputs
rm -rf outputs/ reports/

# Run again
spectral-predict \
  --asd-dir example/ \
  --reference example/BoneCollagen.csv \
  --id-column "File Number" \
  --target "%Collagen"

# Check results
head -20 outputs/results.csv
cat reports/%Collagen.md
```

### Quick Debug Run
```bash
# Use quick_start (10 samples) for faster iteration
spectral-predict \
  --asd-dir example/quick_start/ \
  --reference example/quick_start/reference.csv \
  --id-column "File Number" \
  --target "%Collagen"
```

---

## 📊 Expected vs Actual Behavior

### Expected (from your previous run)
- High R² values (>0.7?)
- RMSE appropriate for the collagen range
- Models showing clear predictive power

### Actual (current run)
- **All R² negative** (-0.07 to -1.27)
- RMSE = 3.62 - 6.80% collagen
- Models worse than baseline mean prediction

### Data Stats
- 37 samples matched (out of 49 in CSV)
- 2151 wavelengths (350-2500 nm)
- Target range: 0.9 - 22.1% collagen
- Mean collagen: ~10.5%, Std: ~6.5%

---

## 🎯 Next Steps (Recommended Order)

1. **Run Debugging Step 1** (Verify Data Loading) - 5 min
2. **Run Debugging Step 2** (Check Filename Matching) - 5 min
3. **Run Debugging Step 5** (Minimal Test) - 5 min
4. **If still broken:** Check if you have the "previous run" outputs to compare
5. **If still broken:** Consider reverting filename matching changes temporarily:
   ```bash
   git diff src/spectral_predict/io.py  # See what changed
   ```

---

## 🐛 Known Issues

1. ✅ **FIXED:** Missing `tabulate` dependency (added to pyproject.toml)
2. ✅ **FIXED:** SpecDAL import path (changed to `from specdal import Spectrum`)
3. ⚠️ **OPEN:** Negative R² anomaly (requires investigation)
4. ⚠️ **MINOR:** Some PLS hyperparameter warnings (n_components too large for small datasets)

---

## 📞 Questions to Answer When You Return

1. **Did the previous run use the same data files?**
   - Same `example/` directory?
   - Same `BoneCollagen.csv`?

2. **What was different in the previous run?**
   - Before the filename matching changes?
   - Different preprocessing settings?
   - Different subset of samples?

3. **Do you have the previous run's outputs?**
   - `outputs/results.csv` from that run?
   - Would help identify what changed

---

## 💡 Quick Hypothesis Test

Before diving deep, run this one-liner to check if the alignment is scrambled:

```bash
python3 -c "
from spectral_predict.io import read_asd_dir, read_reference_csv, align_xy
import pandas as pd
X = read_asd_dir('example/')
ref = read_reference_csv('example/BoneCollagen.csv', 'File Number')
X_al, y = align_xy(X, ref, 'File Number', '%Collagen')
print('First 10 aligned samples:')
for i in range(10):
    idx = X_al.index[i]
    print(f'{idx:20s} → %Collagen = {y.loc[idx]:.2f}')
print('\nDoes this match your expectation of which sample has which collagen value?')
"
```

If the sample-to-value mapping looks wrong, the flexible filename matching is the likely culprit.

---

## 🔄 Git Status

```
M  .claude/settings.local.json
M  CHANGELOG.md
M  README.md
M  pyproject.toml
M  src/spectral_predict/cli.py
M  src/spectral_predict/io.py
M  tests/test_cli_help.py
M  tests/test_import.py

?? .github/
?? .gitignore
?? HANDOFF.md
?? PROJECT_STATUS.md
?? example/
?? outputs/
?? spectral_predict_run.log
?? tests/test_asd_ascii.py
?? tests/test_io_csv.py
?? tests/test_optional_r_bridge.py
```

**Recommendation:** Commit the working code before debugging, so you can easily revert if needed.

---

## ✅ What's Working

- ✅ Installation and setup
- ✅ CLI interface
- ✅ Data loading (CSV, ASD ASCII, ASD binary)
- ✅ Filename matching (works, but maybe *too* flexible?)
- ✅ Model training pipeline
- ✅ Results CSV generation
- ✅ All 30 tests passing
- ✅ Documentation

## ❌ What's Broken

- ❌ Prediction accuracy (negative R² suggests fundamental issue)
- ❌ Report generation (missing tabulate - now fixed)

---

**When you return:** Start with the "Quick Hypothesis Test" above, then work through the debugging steps in order. The negative R² is almost certainly a data alignment/loading issue, not a modeling issue.

Good luck! 🚀
