# Bone Collagen Spectral Analysis Example

This directory contains real ASD spectral data from bone samples with measured collagen content.

## 📁 What's Here

- **37 ASD files** (`Spectrum00001.asd` through `Spectrum00050.asd`)
  - Binary ASD format (requires SpecDAL or can export to ASCII)
  - 2151 wavelengths (350-2500 nm, typical VIS-NIR range)
  - Bone tissue reflectance measurements

- **BoneCollagen.csv** - Reference data linking spectra to measurements
  - **File Number**: Spectrum identifier (e.g., "Spectrum 00001")
  - **Sample no.**: Specimen ID (e.g., "A-53", "G-21")
  - **%Collagen**: Measured collagen content (0.9-22.1%)

## 🎯 The Challenge

Use spectral reflectance to predict bone collagen content. This is a **regression** task - predicting continuous numeric values from spectral data.

## 🚀 Quick Start

### 1. Install Dependencies

The binary ASD files require SpecDAL:

```bash
cd /path/to/deepspec
source .venv/bin/activate
pip install specdal
```

### 2. Run the Analysis

```bash
spectral-predict \
  --asd-dir example/ \
  --reference example/BoneCollagen.csv \
  --id-column "File Number" \
  --target "%Collagen"
```

**Note:** This analyzes all 37 samples and takes **3-5 minutes** on most machines. See below for a faster test.

### 3. Check Results

After completion, you'll find:

- `outputs/results.csv` - All model runs ranked by performance
- `reports/%Collagen.md` - Top 5 models with detailed metrics

## ⚡ Quick Test (Faster)

For a faster test with the first 10 samples:

```bash
# Already set up in quick_start/ subdirectory
spectral-predict \
  --asd-dir example/quick_start/ \
  --reference example/quick_start/reference.csv \
  --id-column "File Number" \
  --target "%Collagen"
```

Runs in ~30 seconds.

## 📊 Understanding the Data Structure

### CSV Format

```csv
File Number,Sample no.,%Collagen
Spectrum 00001,A-53,6.4
Spectrum 00002,A-83,7.9
```

**Important Notes:**
- `File Number` has **spaces** ("Spectrum 00001")
- Actual files have **no spaces** ("Spectrum00001.asd")
- **The software handles this automatically!** ✓

###  Column Roles

When running the analysis:

| Argument | Value | Meaning |
|----------|-------|---------|
| `--id-column` | `"File Number"` | Which column matches filenames |
| `--target` | `"%Collagen"` | Which column to predict |

The `Sample no.` column is **optional** - it's a specimen identifier but not used in this analysis.

## 🧠 What the Software Does

1. **Reads ASD Files** (using SpecDAL for binary format)
2. **Matches Filenames Intelligently**
   - Handles "Spectrum 00001" → "Spectrum00001.asd"
   - Case-insensitive, ignores extensions and spaces
3. **Detects Task Type**
   - Sees %Collagen has decimals → regression task
4. **Tests Models**
   - PLS, Random Forest, MLP
   - Multiple preprocessing (SNV, derivatives)
   - Feature selection (top-20, top-5, top-3 variables)
   - 5-fold cross-validation
5. **Ranks by Composite Score**
   - Balances accuracy with model simplicity
   - Lower score = better

## 📈 Expected Results

With the full 37-sample dataset, you should see:

- **RMSE**: 2.5-4.5% collagen (depending on model)
- **R²**: 0.70-0.85 (70-85% variance explained)
- **Top models**: Usually PLS or Random Forest with SNV preprocessing

These are **respectable results** for a small dataset with high biological variability.

## ⚠️ Troubleshooting

### "Binary ASD file detected"

If you don't have SpecDAL installed:

```bash
pip install specdal
```

Or export your ASD files to ASCII format (`.sig` or ASCII `.asd`) in the ASD software.

### "No matching IDs"

The software tries to match filenames flexibly, but if it fails:

1. Check that `--id-column` is correct (`"File Number"`)
2. Verify filenames match CSV entries (even approximately)
3. Look at the debug output showing normalized IDs

### Small Sample Warnings

If using the `quick_start/` subset (10 samples), you may see warnings about:
- R² not well-defined (too few samples per fold)
- PLS components exceeding limits

These are **expected** with very small datasets. Use the full 37-sample dataset for production analysis.

## 📚 Learn More

- See `../README.md` for full software documentation
- See `../PROJECT_STATUS.md` for implementation details
- Check `outputs/results.csv` to see all model configurations tested

## 🎓 Citation

If you use this example dataset in your work:

```
Bone spectral data courtesy of [your attribution here]
Analysis performed with Spectral Predict v0.1.0
https://github.com/yourusername/deepspec
```

---

**Questions?** Open an issue on GitHub or check the main README.
