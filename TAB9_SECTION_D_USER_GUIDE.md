# Tab 9 Section D: Export Equalized Spectra - User Guide

## Quick Start

### Purpose
Combine spectral data from multiple instruments into a single harmonized dataset on a common wavelength grid.

### When to Use
- You have spectra collected on different instruments
- Each instrument may have different wavelength ranges or spacing
- You want to merge them into a single dataset for analysis
- You need a standardized format for downstream processing

---

## Step-by-Step Usage

### Step 1: Organize Your Data

Create a directory structure where each subdirectory represents one instrument:

```
my_project/
  ├── ASD_FieldSpec3/
  │   ├── soil_sample_001.asd
  │   ├── soil_sample_002.asd
  │   └── soil_sample_003.asd
  ├── ASD_FieldSpec4/
  │   ├── soil_sample_101.asd
  │   └── soil_sample_102.asd
  └── ASD_HandHeld2/
      ├── soil_sample_201.asd
      └── soil_sample_202.asd
```

**Important:**
- Subdirectory names will be used as instrument identifiers
- Use descriptive names (e.g., "ASD_Lab1", "PSR_Field", "ASD_2024")
- Each subdirectory should contain spectra files (.asd, .csv, or .spc)
- You need at least 2 instruments for equalization

### Step 2: Load Multi-Instrument Dataset

1. Click **"Load Multi-Instrument Dataset..."** button
2. Navigate to and select your base directory (e.g., `my_project/` in example above)
3. Click "Select Folder"
4. Wait for loading to complete

**What Happens:**
- GUI scans for subdirectories
- Loads spectra from each subdirectory
- Displays summary showing:
  - Number of instruments found
  - Sample count per instrument
  - Wavelength range per instrument

**Example Summary:**
```
Loaded 3 instruments:

ASD_FieldSpec3: 3 samples, 2151 wavelengths (350.0-2500.0 nm)
ASD_FieldSpec4: 2 samples, 2151 wavelengths (350.0-2500.0 nm)
ASD_HandHeld2: 2 samples, 512 wavelengths (325.0-1075.0 nm)
```

**Troubleshooting:**
- **"No subdirectories found"** → Make sure you selected the base directory, not an instrument directory
- **"Need at least 2 instruments"** → Add more instrument subdirectories
- **"Failed to load spectra from X"** → Check that subdirectory contains valid spectral files

### Step 3: Equalize and Export

1. Click **"Equalize & Export..."** button (green accent button)
2. Wait for processing (status shows "Processing equalization...")
3. When prompted, choose output location and filename
4. Click "Save"
5. Wait for export to complete

**What Happens:**
- Finds common wavelength range (intersection of all instruments)
- Calculates appropriate wavelength spacing (coarsest among instruments)
- Resamples all spectra to common grid
- Creates unique sample IDs with instrument prefixes
- Exports to CSV format

**Example Output Summary:**
```
Equalization complete and exported!

Common wavelength grid: 375 points
Range: 350.0 - 1075.0 nm
Total samples: 7
Data shape: (7, 375)

Exported to:
C:/my_project/equalized_dataset.csv
```

**CSV File Format:**
```csv
sample_id,350.00,351.00,352.00,...,1075.00
ASD_FieldSpec3_sample001,0.123,0.145,0.167,...,0.876
ASD_FieldSpec3_sample002,0.234,0.256,0.278,...,0.765
ASD_FieldSpec3_sample003,0.145,0.167,0.189,...,0.654
ASD_FieldSpec4_sample001,0.256,0.278,0.290,...,0.543
ASD_FieldSpec4_sample002,0.367,0.389,0.401,...,0.432
ASD_HandHeld2_sample001,0.478,0.490,0.512,...,0.321
ASD_HandHeld2_sample002,0.589,0.601,0.623,...,0.210
```

---

## Understanding the Results

### Common Wavelength Grid

The equalization process creates a **common wavelength grid** that:
- Covers the **overlapping range** of all instruments
- Uses the **coarsest spacing** to avoid over-sampling
- Ensures all instruments can contribute valid data

**Example:**
```
Instrument 1: 350-2500 nm, 1 nm spacing
Instrument 2: 350-2500 nm, 1 nm spacing
Instrument 3: 325-1075 nm, 2 nm spacing  ← Coarsest spacing, narrowest range

Common grid: 350-1075 nm, 2 nm spacing
             └─ Intersection  └─ Coarsest spacing
```

### Sample IDs

Each sample gets a unique ID combining:
- **Instrument ID** (subdirectory name)
- **Sample number** (3-digit zero-padded)

Format: `{instrument_id}_sample{###}`

Examples:
- `ASD_Lab1_sample001`
- `PSR_Field_sample042`
- `Spectrometer_2024_sample123`

---

## Use Cases

### Use Case 1: Multi-Laboratory Study
**Scenario:** Soil samples analyzed at 3 different labs with different instruments

**Steps:**
1. Organize data by lab: `Lab1/`, `Lab2/`, `Lab3/`
2. Load and equalize
3. Use exported CSV for unified analysis

**Benefit:** Compare results across labs on common wavelength grid

### Use Case 2: Instrument Upgrade
**Scenario:** Historical data from old instrument + new data from upgraded instrument

**Steps:**
1. Organize data by instrument: `OldInstrument/`, `NewInstrument/`
2. Load and equalize
3. Export combined dataset

**Benefit:** Maintain continuity across instrument change

### Use Case 3: Field vs. Lab Instruments
**Scenario:** Field measurements (limited wavelength range) + lab measurements (full range)

**Steps:**
1. Organize data by location: `Field/`, `Laboratory/`
2. Load and equalize (will use field instrument's narrower range)
3. Export for analysis

**Benefit:** Combine complementary datasets

---

## Advanced Tips

### Tip 1: Instrument Naming
Use descriptive, consistent names for subdirectories:
- **Good:** `ASD_FieldSpec3_Lab1`, `ASD_FieldSpec4_Lab2`
- **Avoid:** `data1`, `backup`, `temp`

### Tip 2: Pre-Check Wavelength Ranges
Before equalization, review the summary to understand:
- Which instrument has the narrowest range (limits common grid)
- Which instrument has coarsest spacing (determines grid resolution)

### Tip 3: Data Organization
Keep a separate copy of raw data before equalization:
```
project/
  ├── raw_data/
  │   ├── instrument1/
  │   └── instrument2/
  └── processed_data/
      └── equalized_dataset.csv
```

### Tip 4: Verify Export
After export, open CSV in Excel or Python to verify:
- Sample count matches expectation
- Wavelength columns are correct
- No missing or invalid values
- Sample IDs are properly formatted

### Tip 5: Downstream Analysis
The exported CSV can be:
- Imported into other tabs for modeling
- Analyzed in R, Python, MATLAB
- Opened in Excel for quick inspection
- Shared with collaborators

---

## Troubleshooting

### Problem: "No subdirectories found"
**Solution:** You selected a leaf directory. Navigate up one level to the base directory containing instrument folders.

### Problem: "Need at least 2 instruments"
**Solution:** Equalization requires multiple instruments. Add more subdirectories or use a different analysis approach.

### Problem: "Failed to load spectra from X"
**Causes:**
- No spectral files in subdirectory
- Unsupported file format
- Corrupted files

**Solution:** Check subdirectory contains .asd, .csv, or .spc files. Remove or fix problematic files.

### Problem: Common grid has very few wavelengths
**Cause:** Instruments have minimal overlap in wavelength ranges

**Example:**
```
Instrument1: 350-1000 nm
Instrument2: 900-2500 nm
Overlap: only 900-1000 nm (very small!)
```

**Solutions:**
- Use instruments with better overlap
- Remove instruments with non-overlapping ranges
- Consider analyzing instruments separately

### Problem: Export cancelled but want to try again
**Solution:** Just click "Equalize & Export..." again. The equalization is already done, it will only re-prompt for export location.

### Problem: CSV file is very large
**Cause:** Many samples and/or fine wavelength spacing

**Solutions:**
- Filter samples before equalization
- Use a coarser wavelength grid (requires code modification)
- Export to binary format (.npz) instead of CSV (requires code modification)

---

## FAQ

**Q: Can I use instruments with different file formats?**
A: Yes! The loader auto-detects .asd, .csv, and .spc formats. All will be loaded and equalized together.

**Q: What happens to wavelengths outside the common range?**
A: They are discarded. Only the overlapping wavelength range is retained.

**Q: Can I equalize just 2 instruments?**
A: Yes! The minimum is 2 instruments. There's no maximum limit.

**Q: Are the spectra calibrated/standardized between instruments?**
A: The basic equalization only resamples to a common grid. For true calibration transfer (removing inter-instrument bias), use Tab 9 Sections A-C to build and apply transfer models first.

**Q: Can I change the sample ID naming scheme?**
A: Currently, the format is fixed as `{instrument}_sample{###}`. Modification requires code changes.

**Q: What if instruments have different wavelength units?**
A: All instruments must use the same units (typically nanometers). Mixed units are not supported.

**Q: Can I exclude certain instruments after loading?**
A: Currently, no. You would need to reload with a different directory structure. This could be a future enhancement.

**Q: What's the difference between this and Tab 9 Sections A-C?**
A:
- **Sections A-C:** Build calibration transfer models (DS/PDS) to standardize spectra from a slave instrument to match a master instrument
- **Section D:** Simple multi-instrument equalization to common wavelength grid (no bias correction)

Use Section D for quick merging, use Sections A-C for rigorous standardization.

---

## Technical Details

### Wavelength Grid Selection Algorithm
1. Find minimum wavelength across all instruments → `min_wl`
2. Find maximum wavelength across all instruments → `max_wl`
3. Find intersection: `common_min = max(all_minimums)`, `common_max = min(all_maximums)`
4. Calculate median spacing for each instrument → `Δλ₁, Δλ₂, ..., Δλₙ`
5. Use coarsest (largest) spacing → `Δλ_common = max(Δλ₁, Δλ₂, ..., Δλₙ)`
6. Generate grid: `λ_common = [common_min, common_min + Δλ, common_min + 2Δλ, ..., common_max]`

### Resampling Method
- Uses linear interpolation (from `resample_to_grid()`)
- Preserves spectral features while changing wavelength sampling
- No smoothing or filtering applied (unless instrument profiles are used)

### File Format
- CSV (Comma-Separated Values)
- First row: header with wavelengths
- First column: sample IDs
- Remaining cells: reflectance/absorbance values
- Compatible with Excel, R, Python pandas, MATLAB

---

## Examples

### Example 1: Minimal Workflow
```
1. Organize data:
   mydata/
     ├── inst1/
     │   └── sample1.asd
     └── inst2/
         └── sample1.asd

2. Load Multi-Instrument Dataset → Select "mydata/"
3. Equalize & Export → Save as "merged.csv"
4. Done!
```

### Example 2: Large Study
```
1. Organize data:
   soil_study/
     ├── ASD_Lab1/ (50 samples)
     ├── ASD_Lab2/ (75 samples)
     ├── ASD_Lab3/ (60 samples)
     └── PSR_Field/ (100 samples)

2. Load Multi-Instrument Dataset → Select "soil_study/"
   Summary shows: 4 instruments, 285 total samples

3. Equalize & Export → Save as "soil_study_equalized.csv"
   Result: 285 rows × 401 wavelength columns (example)

4. Import CSV into Tab 1 for modeling
```

### Example 3: Checking Results in Python
```python
import pandas as pd

# Load exported dataset
df = pd.read_csv('equalized_dataset.csv', index_col=0)

# Check dimensions
print(f"Samples: {len(df)}")
print(f"Wavelengths: {len(df.columns)}")

# View first few samples
print(df.head())

# Check wavelength range
wavelengths = df.columns.astype(float)
print(f"Range: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")

# Check for missing values
print(f"Missing values: {df.isnull().sum().sum()}")

# Plot first spectrum
import matplotlib.pyplot as plt
plt.plot(wavelengths, df.iloc[0])
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.title(df.index[0])
plt.show()
```

---

## Summary

Tab 9 Section D provides a straightforward way to combine multi-instrument spectral datasets into a unified format. The equalization process handles different wavelength ranges and spacings automatically, producing a clean CSV file ready for analysis.

**Key Benefits:**
- Simple directory-based organization
- Automatic wavelength grid selection
- Handles multiple file formats
- Produces analysis-ready CSV output
- Preserves instrument provenance in sample IDs

**Limitations:**
- Only resamples to common grid (no bias correction)
- Requires overlapping wavelength ranges
- Limited to intersection of all ranges
- No advanced preprocessing options

For more sophisticated calibration transfer, use Tab 9 Sections A-C in combination with Section D.

---

**Need Help?** Contact the development team or refer to the full implementation documentation in `AGENT1_TAB9_SECTION_D_IMPLEMENTATION.md`.
