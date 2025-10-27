# Expected Output Files

When you run `spectral-predict`, you should see **two output files** created:

## 1. outputs/results.csv

**What it is:** Complete table of ALL models tested, ranked by performance

**Columns:**
- `Rank` - 1 = best model
- `Model` - Algorithm name with hyperparameters
  - Examples: `PLS(n=6)`, `RandomForest(n=500,depth=15)`, `MLP(hidden=(128,64))`
- `Preprocess` - What was done to the data
  - Examples: `raw`, `snv`, `deriv1_win7`, `snv_deriv1_win7`
- `SubsetTag` - How many variables used
  - `all` = full spectrum (2151 wavelengths)
  - `top-20` = 20 most important wavelengths
  - `top-5` = 5 most important wavelengths
  - `top-3` = 3 most important wavelengths
- **RMSE** - Root mean squared error (lower = better)
- **R2** - R-squared, variance explained (higher = better, 0-1 scale)
- **CompositeScore** - Overall ranking score (lower = better)
  - Balances accuracy with simplicity
  - Formula: `z(RMSE) + 0.15 × (n_components/25 + n_vars/full_vars)`
- `n_vars` - Number of variables used
- `full_vars` - Total variables available (2151)

### Example rows (what you'd see):

```csv
Rank,Model,Preprocess,SubsetTag,RMSE,R2,CompositeScore,n_vars,full_vars
1,PLS(n=6),snv_deriv1_win7,top-20,2.87,0.82,-0.45,20,2151
2,RandomForest(n=500,depth=15),snv,top-20,3.12,0.79,-0.31,20,2151
3,PLS(n=8),deriv1_win7,all,3.25,0.77,-0.18,2151,2151
4,RandomForest(n=500,depth=None),snv_deriv1_win19,top-5,3.41,0.74,0.03,5,2151
5,MLP(hidden=(128,64)),snv,top-20,3.58,0.71,0.19,20,2151
...
```

**How to use it:**
```bash
# View top 10 models
head -11 outputs/results.csv | column -t -s,

# Sort by R² (highest first)
cat outputs/results.csv | sort -t, -k6 -rn | head -10

# Filter for models using <10 variables
awk -F, '$8 < 10' outputs/results.csv
```

---

## 2. reports/%Collagen.md

**What it is:** Human-readable report with TOP 5 models explained in detail

### Example content:

````markdown
# Spectral Analysis Report: %Collagen

**Generated:** 2025-01-27 14:30:15

## Dataset Summary
- Samples: 37
- Wavelengths: 2151 (350-2500 nm)
- Task: Regression
- Target range: 0.9 - 22.1% collagen
- Cross-validation: 5-fold

---

## Top 5 Models

### Rank 1: PLS(n=6) + SNV + Deriv1(win=7) [top-20 vars]

**Performance:**
- RMSE: 2.87% collagen
- R²: 0.82 (82% variance explained)
- Composite Score: -0.45

**Configuration:**
- Preprocessing: SNV normalization → 1st derivative (window=7, poly=2)
- Algorithm: Partial Least Squares Regression
- Components: 6
- Variables: 20 most important wavelengths
- Selected wavelengths: 1450, 1480, 1510, 1680, 1720, 1760, 1940, 1980, 2020, 2060, 2100, 2140, 2180, 2220, 2260, 2300, 2340, 2380, 2420, 2460 nm

**Interpretation:**
This model uses C-H stretches (1400-1500 nm) and N-H/O-H overtones (1900-2500 nm), which are characteristic of collagen protein structure.

---

### Rank 2: RandomForest(n=500, depth=15) + SNV [top-20 vars]

**Performance:**
- RMSE: 3.12% collagen
- R²: 0.79
- Composite Score: -0.31

**Configuration:**
- Preprocessing: SNV normalization only
- Algorithm: Random Forest
- Trees: 500
- Max depth: 15
- Variables: 20 most important wavelengths

---

*[Ranks 3-5 follow similar format]*

---

## Recommendations

1. **Best overall**: Rank 1 (PLS with derivatives and top-20 variables)
2. **Simplest good model**: Check models with `top-5` or `top-3` variables
3. **For deployment**: Consider Rank 2 (RF) if you need non-linear relationships

## Next Steps

1. **Validate externally**: Test top models on held-out samples
2. **Inspect predictions**: Plot predicted vs. actual for top model
3. **Feature interpretation**: Examine which wavelengths matter most
4. **Refine**: Try narrower wavelength ranges based on important regions

---

**Full results:** `outputs/results.csv`
````

---

## Summary

**What you get:**
1. ✅ **results.csv** - Sortable table of ALL models (typically 300-800 rows)
2. ✅ **\<target\>.md** - Top 5 report with interpretation

**Why it matters:**
- No manual testing dozens of preprocessing combinations
- See exactly which models work best
- Understand the trade-off between accuracy and simplicity
- Get wavelength importance for physical interpretation

**Typical results for bone collagen:**
- Best RMSE: 2.5-4.5% collagen
- Best R²: 0.70-0.85
- Top model: Usually PLS with SNV + derivatives
- Important wavelengths: Usually 1400-1500 nm (C-H), 1900-2300 nm (N-H/O-H)

---

## Current Run Status

The analysis is currently running. On the 37-sample bone dataset, expect:
- **Time**: 5-10 minutes (varies by CPU)
- **Models tested**: ~500-800 configurations
- **Output size**: results.csv ~150 KB, report ~10 KB

**Check when complete:**
```bash
ls -lh outputs/ reports/
head -20 outputs/results.csv
cat reports/%Collagen.md
```
