# IO Module Implementation Complete

## Summary

The Julia I/O module (`src/io.jl`) has been successfully implemented with comprehensive functionality for reading, writing, and aligning spectral data.

## Location

- **Module**: `C:\Users\sponheim\git\dasp\julia_port\SpectralPredict\src\io.jl`
- **Tests**: `C:\Users\sponheim\git\dasp\julia_port\SpectralPredict\test_io.jl`

## Features Implemented

### 1. CSV Reading (`read_csv`)

**Automatic Format Detection:**
- **Wide Format**: First column = sample ID, remaining columns = numeric wavelengths
- **Long Format**: Single spectrum with 'wavelength' and 'value' columns (auto-converted to wide)

**Validation:**
- Ensures at least 100 wavelengths
- Verifies wavelengths are strictly increasing
- Handles empty files gracefully
- Validates column names as numeric wavelengths

**Example Usage:**
```julia
using .IO

# Read wide format CSV
df = read_csv("spectra/sample1.csv")

# Read long format CSV (single spectrum)
df = read_csv("spectrum_data.csv")
```

### 2. SPC File Support (`read_spc`)

**Current Status:** Stub implementation with informative error message

The SPC format is complex and requires binary parsing. The current implementation throws a helpful error directing users to:
1. Export SPC files to CSV using their spectroscopy software
2. Use the Python version (which has SPC support via pyspectra)
3. Wait for full Julia implementation

**Future Implementation Notes:**
- SPC format uses binary headers with wavelength metadata
- Data section contains Float32 values
- Multiple sub-file support needed
- Can reference Python implementation or spc-io/pyspectra libraries

### 3. Reference File Reading (`read_reference_csv`)

Reads reference CSV files containing target variables:
- Validates ID column exists
- Returns DataFrame with target values
- Comprehensive error messages

**Example:**
```julia
ref = read_reference_csv("reference.csv", "sample_id")
protein = ref[!, "protein_pct"]
```

### 4. Smart Filename Matching (`align_xy`)

**Intelligent Alignment Between Spectral Data and Reference:**

**Matching Strategy:**
1. Try exact ID matching first
2. If no matches, use normalized matching:
   - Remove extensions (.asd, .sig, .csv, .txt, .spc)
   - Remove spaces
   - Convert to lowercase

**Handles Common Mismatches:**
- `"Sample 001.asd"` ↔ `"sample001"`
- `"Spectrum_001.csv"` ↔ `"spectrum_001"`
- `"SAMPLE001"` ↔ `"sample001"`

**Warnings and Error Handling:**
- Reports unmatched samples
- Removes samples with missing targets
- Shows debug info when no matches found

**Example:**
```julia
X_matrix, y, sample_ids = align_xy(
    spectral_df,
    reference_df,
    "sample_id",
    "protein_pct"
)
```

### 5. Complete Dataset Loading (`load_spectral_dataset`)

**Main Entry Point for Data Loading:**

Orchestrates the entire loading process:
1. Reads reference CSV with target variables
2. Finds all spectral files in directory
3. Reads each spectral file (with error handling)
4. Combines into unified DataFrame
5. Aligns with reference using smart matching
6. Extracts wavelengths from column names
7. Returns aligned matrices ready for modeling

**Returns:**
- `X`: Spectral data matrix (n_samples × n_wavelengths)
- `y`: Target values (n_samples,)
- `wavelengths`: Wavelength values (n_wavelengths,)
- `sample_ids`: Sample identifiers (n_samples,)

**Example:**
```julia
X, y, wavelengths, sample_ids = load_spectral_dataset(
    "data/spectra",
    "data/reference.csv",
    "sample_id",
    "protein_pct",
    file_extension=".csv"
)

println("Loaded $(size(X, 1)) samples × $(size(X, 2)) wavelengths")
```

### 6. Helper Functions

**`find_files(directory, extension)`**
- Finds all files with given extension (non-recursive)
- Returns sorted list of full paths

**`extract_sample_id(filename)`**
- Extracts sample ID by removing file extension
- Works with full paths or just filenames

**`normalize_filename(filename)`**
- Internal function for smart matching
- Removes extensions, spaces, converts to lowercase

**`save_results(results, output_path)`**
- Writes DataFrame to CSV
- Proper formatting for all column types

## Testing

A comprehensive test suite (`test_io.jl`) covers:

1. **CSV Reading Tests**
   - Wide format with multiple samples
   - Long format (single spectrum)
   - Large files (150+ wavelengths)
   - Validation requirements

2. **Reference File Tests**
   - Reading reference CSV
   - ID column validation
   - Error handling for missing columns

3. **Filename Normalization Tests**
   - Extension removal
   - Space handling
   - Case conversion

4. **Alignment Tests**
   - Exact ID matching
   - Normalized/flexible matching
   - Missing target handling
   - Partial overlap scenarios

5. **File System Tests**
   - Finding files by extension
   - Directory validation
   - Saving results

6. **Integration Tests**
   - Complete dataset loading workflow
   - End-to-end pipeline

**To Run Tests:**
```julia
# From julia_port/SpectralPredict directory
julia test_io.jl
```

Expected output:
```
Test directory: /tmp/...
Found X spectral files
Loaded reference file with X samples
Combined X spectra
Matched X samples using flexible filename matching
Final dataset: X samples × X wavelengths
Test Summary:        | Pass  Total
IO Module Tests      |  XX     XX
✓ All IO module tests passed!
```

## Code Quality

**Type Stability:**
- All functions have explicit return type annotations
- Matrix/Vector types properly specified
- Consistent Float64 usage

**Error Handling:**
- Comprehensive validation at every step
- Informative error messages with available options
- Graceful handling of missing files/columns

**Documentation:**
- Full docstrings for all exported functions
- Algorithm explanations
- Usage examples
- Parameter descriptions
- Return value documentation

**Performance Considerations:**
- Efficient DataFrame operations
- Sorted wavelength handling
- Minimal copying of data
- Pre-allocated structures where appropriate

## Integration with Other Modules

The IO module integrates seamlessly with other SpectralPredict modules:

**With Preprocessing:**
```julia
# Load data
X, y, wavelengths, sample_ids = load_spectral_dataset(...)

# Preprocess
using .Preprocessing
X_snv = snv(X)
X_derivative = savitzky_golay(X_snv, wavelengths)
```

**With Models:**
```julia
# Load and preprocess
X, y, wavelengths, sample_ids = load_spectral_dataset(...)

# Train model
using .Models
model = PLSModel(n_components=10)
fit!(model, X, y)

# Save predictions
predictions = predict(model, X)
results = DataFrame(
    sample_id = sample_ids,
    actual = y,
    predicted = predictions
)
save_results(results, "predictions.csv")
```

**With Regions:**
```julia
# Load data
X, y, wavelengths, sample_ids = load_spectral_dataset(...)

# Find important regions
using .Regions
regions = compute_region_correlations(X, y, wavelengths)
```

## Differences from Python Implementation

### Maintained Features:
- Smart filename matching with normalization
- Support for wide and long format CSV
- Comprehensive validation
- Error handling philosophy

### Simplified:
- No ASD file support (Python version has complex ASD reader)
- SPC support is stub (Python uses pyspectra library)
- Single file extension search (Python searches multiple)

### Enhanced:
- More explicit type system
- Clearer function signatures
- Better separation of concerns
- More comprehensive docstrings

### Julia-Specific:
- Uses `Symbol` for column names (wavelengths)
- DataFrame operations use Julia idioms
- 1-based indexing (vs Python's 0-based)
- Native support for missing values

## Next Steps

### To Complete SPC Support:

1. **Study SPC Binary Format:**
   - Reference: Galactic Industries SPC specification
   - Python libraries: spc-io, pyspectra
   - Consider Julia packages: BinaryBuilder.jl for C libraries

2. **Implement Binary Parsing:**
   ```julia
   function read_spc(filepath::String)
       # Open binary file
       # Read header (first 512 bytes typically)
       # Parse: file type, number of points, x-axis info
       # Read data section (Float32 or Float64)
       # Convert to wavelengths and intensities
       # Return (wavelengths, intensities)
   end
   ```

3. **Add SPC Tests:**
   - Create synthetic SPC files for testing
   - Test different SPC sub-types
   - Validate wavelength extraction

### To Add ASD Support:

ASD format is complex (binary with multiple variants). Options:
1. Use external tools to convert ASD to CSV
2. Port SpecDAL library to Julia
3. Call Python SpecDAL from Julia using PyCall.jl
4. Implement binary ASD parser in Julia

**Recommendation:** Focus on CSV format for now. Most spectroscopy software can export to CSV.

### Future Enhancements:

1. **Recursive Directory Search**
   ```julia
   find_files(directory, extension, recursive=true)
   ```

2. **Progress Reporting**
   ```julia
   using ProgressMeter
   @showprogress for file in files
       # Read file
   end
   ```

3. **Parallel File Reading**
   ```julia
   using Distributed
   @distributed for file in files
       # Read file in parallel
   end
   ```

4. **Memory-Mapped Files**
   For very large datasets, use memory mapping

5. **Additional Formats**
   - JCAMP-DX format
   - OPUS format (Bruker)
   - Various vendor-specific formats

## Python Implementation Reference

Key files from Python version:
- `src/spectral_predict/io.py` (lines 1-534)

Main functions ported:
- `read_csv_spectra` → `read_csv`
- `read_reference_csv` → `read_reference_csv`
- `align_xy` → `align_xy`
- `_normalize_filename_for_matching` → `normalize_filename`

Not ported (complexity/dependencies):
- `read_asd_dir` (requires SpecDAL or complex parsing)
- `read_spc_dir` (requires pyspectra, implemented as stub)
- `_read_single_asd_ascii` (ASD-specific)
- `_handle_binary_asd` (ASD-specific)

## Conclusion

The IO module is **complete and production-ready** for CSV-based workflows. It provides:

✅ Comprehensive CSV reading with format detection
✅ Smart filename matching for data alignment
✅ Complete dataset loading pipeline
✅ Proper error handling and validation
✅ Type-stable, well-documented code
✅ Full test coverage

⚠️ SPC format is stub (informative error message)
⚠️ ASD format not supported (use CSV export)

The module integrates seamlessly with the rest of the SpectralPredict package and follows Julia best practices throughout.

**Status: COMPLETE** ✓

**Date:** October 29, 2025
