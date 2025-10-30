# IO Module Quick Reference

## Import

```julia
include("src/io.jl")
using .IO
```

## Main Functions

### Load Complete Dataset (Recommended)

```julia
X, y, wavelengths, sample_ids = load_spectral_dataset(
    "data/spectra",           # Directory with spectral files
    "data/reference.csv",     # Reference CSV with targets
    "sample_id",              # ID column in reference
    "protein_pct",            # Target variable
    file_extension=".csv"     # File type (default: ".csv")
)
```

**Returns:**
- `X`: Matrix (n_samples × n_wavelengths)
- `y`: Vector (n_samples) - target values
- `wavelengths`: Vector (n_wavelengths) - wavelength values in nm
- `sample_ids`: Vector (n_samples) - sample identifiers

---

### Read Single CSV File

```julia
df = read_csv("path/to/spectrum.csv")
```

**Supports two formats:**

**Wide Format:**
```csv
sample_id,400.0,402.0,404.0,...
sample1,0.123,0.145,0.167,...
```

**Long Format:**
```csv
wavelength,value
400.0,0.123
402.0,0.145
```

---

### Read Reference File

```julia
ref = read_reference_csv("reference.csv", "sample_id")
```

**Format:**
```csv
sample_id,protein,moisture,fat
sample1,12.3,45.6,7.8
sample2,23.4,56.7,8.9
```

---

### Align Spectral Data with Reference

```julia
X_matrix, y, sample_ids = align_xy(
    spectral_df,      # DataFrame from read_csv
    reference_df,     # DataFrame from read_reference_csv
    "sample_id",      # ID column name in reference
    "protein_pct"     # Target column name
)
```

**Smart matching handles:**
- Extensions: `"sample.asd"` ↔ `"sample"`
- Spaces: `"Sample 001"` ↔ `"Sample001"`
- Case: `"SAMPLE"` ↔ `"sample"`

---

### Save Results

```julia
results = DataFrame(
    sample_id = sample_ids,
    actual = y,
    predicted = predictions
)

save_results(results, "predictions.csv")
```

---

## Helper Functions

### Find Files

```julia
csv_files = find_files("data/spectra", ".csv")
spc_files = find_files("data/spectra", ".spc")
```

### Extract Sample ID

```julia
id = extract_sample_id("sample001.csv")  # "sample001"
id = extract_sample_id("/path/to/data.asd")  # "data"
```

---

## Complete Workflow Example

```julia
include("src/io.jl")
using .IO

# 1. Load data
X, y, wavelengths, ids = load_spectral_dataset(
    "data/spectra",
    "data/reference.csv",
    "sample_id",
    "protein_pct"
)

println("Loaded $(size(X, 1)) samples × $(size(X, 2)) wavelengths")
println("Range: $(wavelengths[1])-$(wavelengths[end]) nm")

# 2. Preprocess (using Preprocessing module)
include("src/preprocessing.jl")
using .Preprocessing

X_snv = snv(X)
X_deriv = savitzky_golay(X_snv, wavelengths)

# 3. Train model (using Models module)
include("src/models.jl")
using .Models

model = PLSModel(n_components=10)
fit!(model, X_deriv, y)

# 4. Make predictions
predictions = predict(model, X_deriv)

# 5. Save results
results = DataFrame(
    sample_id = ids,
    actual = y,
    predicted = predictions,
    error = abs.(y .- predictions)
)

save_results(results, "predictions.csv")
println("Results saved!")
```

---

## Error Handling

All functions provide clear error messages:

```julia
try
    df = read_csv("nonexistent.csv")
catch e
    println("Error: ", e)
    # ArgumentError: File not found: nonexistent.csv
end

try
    ref = read_reference_csv("ref.csv", "wrong_column")
catch e
    println("Error: ", e)
    # ArgumentError: Column 'wrong_column' not found. Available: sample_id, protein, ...
end
```

---

## CSV Format Specifications

### Wide Format Requirements:
- First column: Sample ID (any string)
- Remaining columns: Numeric wavelengths (e.g., 400.0, 402.0)
- At least 100 wavelengths
- Wavelengths must be strictly increasing

### Long Format Requirements:
- Must have columns: `wavelength` (or `wavelength_nm`) and `value` (or `intensity`, `reflectance`, `pct_reflect`)
- One row per wavelength
- At least 100 wavelengths
- Filename becomes sample ID

---

## SPC Format (Future)

Currently stub implementation. Throws informative error:

```julia
try
    wl, intensities = read_spc("spectrum.spc")
catch e
    println(e)
    # Suggests CSV export or Python version
end
```

**Options:**
1. Export SPC to CSV using your spectroscopy software
2. Use Python version (has pyspectra support)
3. Wait for Julia implementation

---

## Type Information

```julia
# Function signatures
read_csv(filepath::String)::DataFrame
read_reference_csv(filepath::String, id_column::String)::DataFrame
align_xy(X::DataFrame, ref::DataFrame, id_column::String,
         target_column::String)::Tuple{Matrix{Float64}, Vector{Float64}, Vector{String}}
load_spectral_dataset(...)::Tuple{Matrix{Float64}, Vector{Float64},
                                   Vector{Float64}, Vector{String}}
save_results(results::DataFrame, output_path::String)
find_files(directory::String, extension::String)::Vector{String}
extract_sample_id(filename::String)::String
```

---

## Testing

Run comprehensive tests:

```julia
julia test_io.jl
```

Tests cover:
- CSV reading (wide and long format)
- Reference file reading
- Alignment (exact and normalized matching)
- File finding
- Error handling
- Complete integration workflow

---

## See Also

- **Full Documentation**: `IO_MODULE_COMPLETE.md`
- **Example Usage**: `examples/io_example.jl`
- **Source Code**: `src/io.jl`
- **Tests**: `test_io.jl`
