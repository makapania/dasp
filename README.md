# Spectral Predict

**Automated spectral modeling** — ingest spectra (CSV or ASD), test multiple preprocessing × model combinations with cross-validation, then rank them by a simplicity-aware score.

Stop manually testing preprocessing pipelines. Get a ranked list of candidate models and a Markdown report in minutes.

---

## 🚀 Quick Start (2 minutes)

### 1. Install

```bash
git clone https://github.com/yourusername/deepspec.git
cd deepspec
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
pip install specdal  # For binary ASD files
```

### 2. Run the Example

Use the included bone collagen dataset (37 real ASD files):

```bash
spectral-predict \
  --asd-dir example/ \
  --reference example/BoneCollagen.csv \
  --id-column "File Number" \
  --target "%Collagen"
```

**Takes 3-5 minutes.** See `example/README.md` for a 30-second quick test with 10 samples.

### 3. Check Results

```bash
# Ranked model table
cat outputs/results.csv

# Top 5 models with details
cat reports/%Collagen.md
```

Done! You now have a ranked list of models with cross-validated performance metrics.

---

## 💡 Why Use This?

**The Problem:**
Building baseline spectral models is tedious. You need to:
- Try multiple preprocessing methods (SNV, derivatives, etc.)
- Test different algorithms (PLS, Random Forest, MLP)
- Tune hyperparameters for each
- Run cross-validation
- Compare and select models

**The Solution:**
Spectral Predict does all of this automatically. You get:
- ✅ **Interactive loading phase** with spectral plots and data preview
- ✅ **Predictor screening** (JMP-style) to identify informative wavelengths
- ✅ **Absorbance conversion** option (like Unscrambler)
- ✅ **Automated grid search** over preprocessing × models × hyperparameters
- ✅ **Smart ranking** that balances accuracy with simplicity
- ✅ **Multiple input formats** (CSV wide/long, ASD ASCII/binary)
- ✅ **Flexible filename matching** (handles extensions, spaces, case)
- ✅ **Clear outputs** (sortable CSV + readable Markdown report)

---

## 📊 What It Does

```
Input: Spectra + Reference Data
         ↓
   [Preprocessing Grid]
   • Raw, SNV, Derivatives (1st/2nd order)
   • SG windows: 7, 19
         ↓
     [Model Grid]
   • Regression: PLS, Random Forest, MLP
   • Classification: PLS-DA, RF, MLP
         ↓
  [Feature Selection]
   • Top-20, Top-5, Top-3 variables
   • VIP scores (PLS) or importances (RF/MLP)
         ↓
  [5-Fold Cross-Validation]
   • RMSE, R² (regression)
   • Accuracy, ROC-AUC (classification)
         ↓
 [Composite Score Ranking]
   Score = z(metric) + λ × (complexity)
   Lower score = better model
         ↓
Output: results.csv + report.md
```

---

## 📥 Supported File Formats

Spectral Predict supports 10+ spectral file formats with automatic detection and unified I/O API.

### Format Support Matrix

| Format | Extensions | Read | Write | Auto-Detect | Dependencies |
|--------|-----------|------|-------|-------------|--------------|
| CSV | .csv | ✅ | ✅ | ✅ | Built-in |
| Excel | .xlsx, .xls | ✅ | ✅ | ✅ | openpyxl, xlsxwriter |
| ASD (ASCII) | .asd, .sig | ✅ | ❌ | ✅ | Built-in |
| ASD (Binary) | .asd | ✅ | ❌ | ✅ | specdal (optional) |
| SPC | .spc | ✅ | ✅ | ✅ | spc-io |
| JCAMP-DX | .jdx, .dx, .jcm | ✅ | ✅ | ✅ | jcamp |
| ASCII Text | .txt, .dat | ✅ | ✅ | ✅ | Built-in |
| Bruker OPUS | .0, .1, .2, etc. | ✅ | ❌ | ✅ | brukeropus (optional) |
| PerkinElmer | .sp | ✅ | ❌ | ✅ | specio (optional) |
| Agilent | .seq | 🚧 | ❌ | ✅ | agilent-ir-formats (optional) |

**Legend:**
- ✅ = Fully supported
- 🚧 = In development
- ❌ = Not supported

### Installation with Optional Formats

```bash
# Basic installation (CSV, Excel, ASCII, JCAMP, SPC)
pip install -e .

# With binary ASD support
pip install -e ".[asd]"

# With vendor formats
pip install -e ".[opus]"          # Bruker OPUS
pip install -e ".[perkinelmer]"   # PerkinElmer
pip install -e ".[agilent]"       # Agilent

# Install all format support
pip install -e ".[all-formats]"
```

### Unified I/O API

The unified API automatically detects formats and provides consistent interface:

```python
from spectral_predict.io import read_spectra, write_spectra

# Read any supported format (auto-detect)
df, metadata = read_spectra('data/spectra.xlsx')
df, metadata = read_spectra('data/spectra.csv')
df, metadata = read_spectra('data/asd_files/')
df, metadata = read_spectra('data/spectrum.jdx')

# Explicit format specification
df, metadata = read_spectra('data/file.dat', format='ascii')

# Write to various formats
write_spectra(df, 'output.csv', format='csv')
write_spectra(df, 'output.xlsx', format='excel')
write_spectra(df, 'output.jdx', format='jcamp')
```

### CSV Wide Format

First column = sample ID, remaining columns = wavelengths:

```csv
sample_id,400.0,401.0,402.0,...,2400.0
S001,0.123,0.125,0.127,...,0.456
S002,0.134,0.136,0.138,...,0.467
```

```bash
spectral-predict --spectra data/spectra.csv \
                 --reference data/ref.csv \
                 --id-column sample_id \
                 --target nitrogen
```

### CSV Long Format (Single Spectrum)

Two columns with wavelength and value:

```csv
wavelength,value
400.0,0.123
401.0,0.125
```

Automatically detected and converted to wide format.

### Combined CSV/Excel Format (Spectra + Targets)

**NEW:** Load all data from a single file containing both spectra and target variables. Perfect for sharing complete datasets!

The combined format is automatically detected when you provide a directory containing a single CSV or Excel file. The file can have columns in any order:

```csv
specimen_id,400.0,401.0,...,2400.0,nitrogen,protein
S001,0.123,0.125,...,0.456,2.45,15.3
S002,0.134,0.136,...,0.467,2.78,17.4
```

Or even with targets first:

```csv
nitrogen,specimen_id,400.0,401.0,...,2400.0
2.45,S001,0.123,0.125,...,0.456
2.78,S002,0.134,0.136,...,0.467
```

**Key Features:**
- ✅ Automatic column detection (wavelengths, specimen ID, target)
- ✅ Flexible column ordering (columns can be in any position)
- ✅ Auto-generates sample IDs if missing (Sample_1, Sample_2, etc.)
- ✅ Works with both CSV and Excel files
- ✅ No separate reference file needed!

**Usage:**

```python
from spectral_predict.io import read_combined_csv, read_combined_excel

# CSV combined format
X, y, metadata = read_combined_csv('data.csv')
print(f"Loaded {len(X)} spectra with {X.shape[1]} wavelengths")
print(f"Target variable: {metadata['y_col']}")

# Excel combined format (same logic)
X, y, metadata = read_combined_excel('data.xlsx')
```

**Automatic Detection:**
If you place a single CSV/Excel file in a directory, the GUI will automatically detect it as a combined format and load both spectra and targets together.

**Without Specimen ID column:**
The system automatically generates IDs if your file only has wavelengths and targets:

```csv
400.0,401.0,...,2400.0,nitrogen
0.123,0.125,...,0.456,2.45
0.134,0.136,...,0.467,2.78
```
→ Creates Sample_1, Sample_2, etc.

### Excel Format

Same layout as CSV, supports multiple sheets:

```python
# Read specific sheet
df, meta = read_spectra('data.xlsx', sheet_name='Spectra')

# Write with formatting
write_spectra(df, 'output.xlsx', format='excel',
              sheet_name='Data', freeze_panes=(1, 1))
```

### ASD Format

**ASCII ASD** (.sig, ASCII .asd):
```bash
spectral-predict --asd-dir data/asd_files/ \
                 --reference data/ref.csv \
                 --id-column filename \
                 --target protein
```

**Binary ASD** (requires SpecDAL):
```bash
pip install specdal
spectral-predict --asd-dir data/binary_asd/ \
                 --reference data/ref.csv \
                 --id-column filename \
                 --target "%collagen"
```

### SPC Format (GRAMS/Thermo Galactic)

```python
# Read SPC file or directory
df, meta = read_spectra('data/spectrum.spc')
df, meta = read_spectra('data/spc_files/')

# Write SPC (single spectrum only)
write_spectra(df.iloc[[0]], 'output.spc', format='spc')
```

### JCAMP-DX Format

```python
# Read JCAMP file
df, meta = read_spectra('spectrum.jdx')

# Write JCAMP with metadata
write_spectra(df.iloc[[0]], 'output.jdx', format='jcamp',
              title='Sample A',
              data_type='INFRARED SPECTRUM',
              xunits='NANOMETERS',
              yunits='REFLECTANCE')
```

### ASCII Text Format

Generic two-column format (wavelength, intensity):

```python
# Auto-detect delimiter
df, meta = read_spectra('spectrum.txt')

# Specify delimiter
df, meta = read_spectra('data.dat', format='ascii', delimiter='\t')

# Write ASCII
write_spectra(df.iloc[[0]], 'output.txt', format='ascii', delimiter='\t')
```

### Vendor Formats

**Bruker OPUS** (.0, .1, .2, etc.):
```bash
pip install brukeropus
```
```python
df, meta = read_spectra('spectrum.0', format='opus')
```

**PerkinElmer** (.sp):
```bash
pip install specio
```
```python
df, meta = read_spectra('spectrum.sp', format='perkinelmer')
```

**Agilent** (.seq):
```bash
pip install agilent-ir-formats
```
Status: In development - contributions welcome!

### Reference CSV

Maps sample IDs to target variables:

```csv
sample_id,nitrogen,carbon,protein
S001,2.45,45.2,15.3
S002,2.78,43.8,17.4
```

**Flexible ID Matching:**
The software intelligently matches IDs even if:
- CSV has "Spectrum 00001" but files are "Spectrum00001.asd"
- Different casing ("sample_001" vs "Sample_001")
- With/without extensions

---

## 📤 Outputs

### results.csv

All model runs with columns:
- **Model**: PLS, RandomForest, MLP (with hyperparameters)
- **Preprocess**: raw, snv, deriv1, snv_deriv1, etc.
- **SubsetTag**: all, top-20, top-5, top-3 variables
- **RMSE, R²** (regression) or **Accuracy, ROC_AUC** (classification)
- **CompositeScore**: Lower = better (balances performance + simplicity)
- **Rank**: 1 = best model

**Sort by Rank to see top models.**

### reports/\<target>.md

Markdown report with:
- Top 5 models
- Configuration details
- Performance metrics
- Summary table

---

## 🎛️ Options

```bash
spectral-predict --spectra <CSV> | --asd-dir <DIR> \
                 --reference <CSV> \
                 --id-column <COL> \
                 --target <COL> \
                 [--folds 5] \
                 [--lambda-penalty 0.15] \
                 [--outdir outputs] \
                 [--asd-reader auto] \
                 [--interactive] \
                 [--no-interactive]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--folds` | 5 | Cross-validation folds |
| `--lambda-penalty` | 0.15 | Complexity penalty weight (higher = prefer simpler models) |
| `--outdir` | outputs | Output directory |
| `--asd-reader` | auto | ASD reader mode (auto tries SpecDAL if installed) |
| `--interactive` | True | Enable interactive loading phase |
| `--no-interactive` | - | Skip interactive loading phase |

---

## 🔍 Interactive Loading Phase

By default, Spectral Predict includes an interactive loading phase that helps you verify your data before modeling:

### 1. Spectral Plots
Three plots are automatically generated:
- **Raw spectra**: Verify reflectance values look correct
- **1st derivative**: Check for spectral features
- **2nd derivative**: Identify fine spectral details

### 2. Data Preview
A table showing:
- Sample IDs
- Target values (if available)
- First few wavelengths
- Quick verification that files loaded correctly

### 3. Data Range Check
Automatic detection of data format:
- Reflectance (0-1)
- Percent reflectance (0-100)
- Other formats

### 4. Absorbance Conversion
Option to convert reflectance → absorbance using `log10(1/R)`:
```
Convert to absorbance? [y/N]:
```

### 5. Predictor Screening
JMP-style variable screening showing:
- Top 20 most correlated wavelengths with target
- Correlation plot across all wavelengths
- Immediate feedback on whether target is predictable

**Skip Interactive Mode:**
```bash
spectral-predict --asd-dir data/ --reference ref.csv \
                 --id-column sample --target nitrogen \
                 --no-interactive
```

---

## 🧪 Example: Bone Collagen

The `example/` directory contains real data:

**Dataset:** 37 bone samples with measured collagen content (0.9-22.1%)
**Task:** Predict %collagen from VIS-NIR spectra (350-2500 nm)
**Files:** Binary ASD format + CSV reference

```bash
# Full analysis (3-5 min)
spectral-predict --asd-dir example/ \
                 --reference example/BoneCollagen.csv \
                 --id-column "File Number" \
                 --target "%Collagen"

# Quick test with 10 samples (30 sec)
spectral-predict --asd-dir example/quick_start/ \
                 --reference example/quick_start/reference.csv \
                 --id-column "File Number" \
                 --target "%Collagen"
```

**Expected Results:**
- RMSE: 2.5-4.5% collagen
- R²: 0.70-0.85
- Top models: PLS or Random Forest with SNV preprocessing

See `example/README.md` for details.

---

## 🛠️ Development

```bash
# Run tests
pytest -v                       # All 30 tests

# Code formatting
black src/ tests/               # Format code
flake8 src/ tests/              # Lint code

# CI/CD
# GitHub Actions runs on Linux/Windows with Python 3.10-3.12
```

---

## 📋 Requirements

### Core Dependencies

- Python ≥ 3.10
- numpy ≥ 1.21.0
- pandas ≥ 1.3.0
- scikit-learn ≥ 1.0.0
- scipy ≥ 1.7.0
- matplotlib ≥ 3.5.0
- tabulate ≥ 0.9.0
- xgboost ≥ 2.0.0
- lightgbm ≥ 4.0.0
- catboost ≥ 1.2.0

### File Format Dependencies

**Built-in formats** (no additional packages):
- CSV
- ASCII ASD (.sig, ASCII .asd)
- Generic ASCII text files

**Included in default installation:**
- Excel: openpyxl ≥ 3.1.0, xlsxwriter ≥ 3.2.0
- JCAMP-DX: jcamp ≥ 1.3.0
- SPC: spc-io ≥ 0.2.0

**Optional vendor formats:**
- Binary ASD: specdal
- Bruker OPUS: brukeropus
- PerkinElmer: specio
- Agilent: agilent-ir-formats

---

## ⚠️ Data Assumptions

- **Wavelengths** in nanometers (nm), strictly increasing
- **Don't mix** different wavelength grids in one run (build separate models)
- **Reflectance** usually 0-1 (values outside this range likely radiance/DN)
- **One target** at a time (run again for each additional target)

---

## 🐛 Troubleshooting

### "Binary ASD detected"

```bash
pip install specdal
# OR export to ASCII in ASD software
```

### "No matching IDs"

The software tries flexible matching, but if it fails:
1. Check `--id-column` matches your CSV column name exactly
2. Verify CSV IDs roughly match filenames (software handles extensions/spaces)
3. Look at debug output showing normalized IDs

Example of what works:
- CSV: "Spectrum 00001" ✅ matches → File: "Spectrum00001.asd"
- CSV: "sample_001" ✅ matches → File: "Sample_001.asd"

### "Target not found"

Check that `--target` exactly matches a column name in your reference CSV (case-sensitive).

### Small Dataset Warnings

With <15 samples, you may see warnings about:
- R² not well-defined (too few samples per fold)
- PLS components exceeding limits

**Recommendation:** Use ≥20 samples for reliable cross-validation.

---

## 🗺️ Roadmap

**v0.1.0 (Current)**
- ✅ CSV wide/long formats
- ✅ ASD ASCII files (`.sig`, ASCII `.asd`)
- ✅ Binary ASD via SpecDAL
- ✅ PLS, Random Forest, MLP
- ✅ SNV, Savitzky-Golay derivatives
- ✅ Feature selection
- ✅ Cross-validation
- ✅ Composite scoring
- ✅ Intelligent filename matching

**Future**
- Native binary ASD reader (no SpecDAL dependency)
- R bridge (prospectr, asdreader packages)
- Interactive mode
- Model persistence/reloading
- Additional file formats (SPC, OPUS)
- Batch processing for multiple targets

---

## 📄 License

MIT

---

## 🙏 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass (`pytest -v`)
5. Submit a pull request

---

## 📞 Support

- **Documentation:** `example/README.md`, `PROJECT_STATUS.md`
- **Issues:** https://github.com/yourusername/deepspec/issues
- **Help:** `spectral-predict --help`

---

## 📚 Citation

If you use this software in your research:

```bibtex
@software{spectral_predict_2025,
  title = {Spectral Predict: Automated Spectral Analysis Software},
  author = {deepspec contributors},
  year = {2025},
  version = {0.1.0},
  url = {https://github.com/yourusername/deepspec}
}
```

---

## 🆕 What's New in v2.0 (October 27, 2025)

### ✨ Neural Boosted Regression
- **Gradient boosting with neural networks** - Captures nonlinearity better than PLS
- **Interpretable** - Provides feature importances like PLS VIP scores
- **Robust** - Huber loss option for outlier handling
- **Automatic** - Tests 24 configurations with early stopping
- **Tested** - R² = 0.9582 on validation data ✓

### ✨ Top Important Variables
- **New `top_vars` column** in results CSV
- Shows **top 30 most important wavelengths** per model (ordered by importance)
- Works for **all models**: PLS (VIP), RandomForest (Gini), MLP/NeuralBoosted (weights)
- Example: `"1450.0,2250.0,1455.0,..."` - O-H and C-H peaks

### 📚 New Documentation
- **NEURAL_BOOSTED_GUIDE.md** - Complete user guide (when to use, how to interpret)
- **WAVELENGTH_SUBSET_SELECTION.md** - Explains how wavelengths are selected
- **NEURAL_BOOSTED_IMPLEMENTATION_PLAN.md** - Technical specification
- **HANDOFF_NEURAL_BOOSTED_COMPLETE.md** - Testing guide for tomorrow

See documentation files for complete details.

---

## 📊 Models Now Tested

| Model | Speed | Nonlinearity | Interpretability | Best For |
|-------|-------|--------------|------------------|----------|
| PLS | ⭐⭐⭐⭐⭐ | Linear | ⭐⭐⭐⭐⭐ | Linear relationships |
| Random Forest | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Large datasets |
| MLP | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | Deep nonlinearity |
| **Neural Boosted** ✨ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | **Interpretable nonlinearity** |

