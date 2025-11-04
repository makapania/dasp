# SpectralPredict GUI - Troubleshooting Guide

## ✅ GUI is Now Fixed!

The `UndefVarError: load_spectral_dataset` error has been fixed.

## 🚀 To Start the GUI

```bash
cd /Users/mattsponheimer/git/dasp/julia_port/SpectralPredict
./start_gui.sh
```

Then open your browser to: **http://localhost:8080**

---

## Common Errors & Solutions

### 1. "Cannot connect" or Page Won't Load

**Symptoms:**
- Browser shows "This site can't be reached"
- Page loading forever

**Solution:**
```bash
# Stop any running instance (Ctrl+C in terminal)
# Then restart:
cd /Users/mattsponheimer/git/dasp/julia_port/SpectralPredict
julia --project=. gui.jl

# Wait for this message:
# "Server running at: http://localhost:8080"

# Then open browser to: http://localhost:8080
```

---

### 2. "File not found" or "Directory not found"

**Symptoms:**
- Error after clicking "Run Analysis"
- Terminal shows file path errors

**Solution:**

✅ **Use FULL paths:**
```
/Users/mattsponheimer/git/dasp/example
```

❌ **Don't use:**
```
~/git/dasp/example          # Doesn't work
./example                    # Doesn't work
example                      # Doesn't work
```

**How to get the full path:**
```bash
# In Terminal, navigate to your folder:
cd /path/to/your/folder

# Then run:
pwd

# Copy the output and paste it in the GUI
```

---

### 3. Column Name Not Found

**Symptoms:**
- "Column 'sample_id' not found"
- "Column '%Collagen' not found"

**Solution:**

Column names are **case-sensitive** and must match **exactly**!

✅ **Check your CSV file:**
```bash
# Look at the header of your reference file:
head -1 /path/to/your/reference.csv
```

Common mistakes:
- `sample_id` vs `Sample ID` vs `Sample_ID`
- `%Collagen` vs `%collagen` vs `Collagen`
- Extra spaces: `sample_id ` (with space at end)

---

### 4. No Data Loaded / Empty Results

**Symptoms:**
- "0 samples loaded"
- Analysis runs but returns no results

**Solution:**

**Check sample IDs match:**

Your reference CSV sample IDs must match your spectra filenames:

✅ **Correct:**
```
Reference CSV: sample001, sample002, sample003
Spectra files: sample001.csv, sample002.csv, sample003.csv
```

❌ **Won't match:**
```
Reference CSV: Sample001
Spectra files: sample001.csv    (case doesn't match)
```

**Check file formats:**

Spectra files must be:
- CSV format (`.csv` extension)
- OR ASD format (`.asd` extension)
- One file per sample

---

### 5. Analysis Takes Forever / No Progress

**Symptoms:**
- Status shows "Running analysis..." for 30+ minutes
- No progress updates

**Solutions:**

**1. Reduce complexity:**
- Use **3 CV folds** instead of 10
- Select **Ridge only** (uncheck other models)
- Use **SNV only** (uncheck derivatives)
- **Uncheck** variable and region subsets

**2. Check dataset size:**
```bash
# Count samples:
ls /path/to/spectra | wc -l

# Count wavelengths:
head -1 /path/to/spectra/first_file.csv | tr ',' '\n' | wc -l
```

Large datasets (1000+ samples or 2000+ wavelengths) take longer.

**3. Check terminal for errors:**
Look in the terminal window where the GUI is running for error messages.

---

### 6. "Address already in use"

**Symptoms:**
- Can't start GUI
- Error: "address already in use :8080"

**Solution:**

Another program is using port 8080.

```bash
# Find what's using the port:
lsof -i :8080

# Kill the process (replace PID with actual number):
kill -9 PID

# Or just use a different terminal window and restart
```

---

### 7. Out of Memory

**Symptoms:**
- Julia crashes
- "Out of memory" error
- Computer becomes very slow

**Solution:**

Your dataset is too large for available RAM.

**Reduce memory usage:**
1. Use **3 CV folds** instead of 10
2. Disable **variable subsets**
3. Disable **region subsets**
4. Select **fewer models** (just Ridge)

**Or split your analysis:**
- Analyze subsets of samples separately
- Combine results afterwards

---

### 8. Results Look Wrong / Poor Performance

**Symptoms:**
- All R² values near 0 or negative
- RMSE very high

**Possible Causes:**

**1. Target variable mismatch:**
- Check you're predicting the right column
- Verify data is numeric (not text)

**2. Preprocessing needed:**
- Try different preprocessing (SNV, derivatives)
- Raw data might not work well

**3. Data quality issues:**
- Check for missing values
- Check for outliers
- Verify spectral data is clean

**Quick test:**
```julia
# In Julia REPL:
using CSV, DataFrames
ref = CSV.read("your_reference.csv", DataFrame)
println(describe(ref))  # Check data statistics
```

---

### 9. GUI Shows Results but CSV Not Saved

**Symptoms:**
- Results display in browser
- Can't find output CSV file

**Solution:**

CSV is saved in the **current directory** where you started the GUI.

```bash
# Check where you are:
pwd

# List recent CSV files:
ls -lt *.csv | head -5

# The file is named:
# spectralpredict_results_2025-10-30_14-30-45.csv
```

---

### 10. Can't Stop the GUI

**Symptoms:**
- Pressing Ctrl+C doesn't work
- GUI keeps running

**Solution:**

```bash
# Press Ctrl+C multiple times (2-3 times)

# If still running, close the terminal window

# Or find and kill the process:
ps aux | grep julia
kill -9 PID  # Replace PID with the process ID
```

---

## 🔧 Advanced Troubleshooting

### Check Julia Installation

```bash
julia --version
# Should show: julia version 1.12.1
```

### Check Packages

```bash
cd /Users/mattsponheimer/git/dasp/julia_port/SpectralPredict
julia --project=. -e 'using Pkg; Pkg.status()'
# Should list all installed packages
```

### Reinstall Packages

```bash
cd /Users/mattsponheimer/git/dasp/julia_port/SpectralPredict
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Test Core Functionality

```bash
cd /Users/mattsponheimer/git/dasp/julia_port/SpectralPredict
julia --project=.
```

```julia
# In Julia REPL:
include("src/SpectralPredict.jl")
using .SpectralPredict

# Test with random data:
X = randn(50, 100)
y = randn(50)
wavelengths = collect(400.0:4.0:796.0)

results = SpectralPredict.run_search(
    X, y, wavelengths,
    models=["Ridge"],
    preprocessing=["snv"],
    n_folds=3
)

println("Success! Tested ", size(results, 1), " configurations")
```

If this works, the core system is fine - issue is likely with data paths or format.

---

## 📞 Still Having Issues?

### Before Asking for Help

1. **Check the terminal** - Error messages appear there
2. **Copy error message** - Full text helps diagnose
3. **Note what you tried** - Helps avoid repeating steps
4. **Check file paths** - 90% of issues are path-related

### What to Report

Include:
1. Full error message from terminal
2. Paths you're using
3. First few lines of your reference CSV
4. Number of samples and wavelengths
5. What you've already tried

### Quick Diagnostics

```bash
# System info:
julia --version
pwd
ls -la

# Check your files:
ls /path/to/your/spectra | head -5
head -3 /path/to/your/reference.csv

# Check GUI loads:
cd /Users/mattsponheimer/git/dasp/julia_port/SpectralPredict
julia --project=. -e 'include("gui.jl")'
# Should end with no errors
```

---

## ✅ Prevention Checklist

Before running analysis:

- [ ] Using full paths starting with `/Users/...`
- [ ] Column names match exactly (case-sensitive)
- [ ] Spectra files are CSV or ASD format
- [ ] Reference CSV has matching sample IDs
- [ ] At least one model selected
- [ ] At least one preprocessing method selected
- [ ] Started with simple settings first (Ridge + SNV + 5 folds)

---

## 🎯 Recommended First Run

To minimize issues on first try:

1. **Data:**
   - Use test data first: `/Users/mattsponheimer/git/dasp/example`
   - Reference: `/Users/mattsponheimer/git/dasp/example/BoneCollagen.csv`
   - ID: `File Number`
   - Target: `%Collagen`

2. **Settings:**
   - Models: **Ridge only**
   - Preprocessing: **SNV only**
   - CV Folds: **5**
   - Variable subsets: **Unchecked**
   - Region subsets: **Unchecked**

3. **Run and wait 2-3 minutes**

4. **If successful, try with your data**

---

This should solve 99% of issues! If you're still stuck, collect the diagnostics above and ask for help.
