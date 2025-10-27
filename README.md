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

## 📥 Input Formats

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

### ASD ASCII Format

Text-based `.sig` or ASCII `.asd` files:

```bash
spectral-predict --asd-dir data/asd_files/ \
                 --reference data/ref.csv \
                 --id-column filename \
                 --target protein
```

### ASD Binary Format

Binary `.asd` files (requires SpecDAL):

```bash
pip install specdal
spectral-predict --asd-dir data/binary_asd/ \
                 --reference data/ref.csv \
                 --id-column filename \
                 --target "%collagen"
```

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
                 [--asd-reader auto]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--folds` | 5 | Cross-validation folds |
| `--lambda-penalty` | 0.15 | Complexity penalty weight (higher = prefer simpler models) |
| `--outdir` | outputs | Output directory |
| `--asd-reader` | auto | ASD reader mode (auto tries SpecDAL if installed) |

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

- Python ≥ 3.10
- numpy ≥ 1.21.0
- pandas ≥ 1.3.0
- scikit-learn ≥ 1.0.0
- scipy ≥ 1.7.0

**Optional:**
- specdal (for binary ASD files)

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
