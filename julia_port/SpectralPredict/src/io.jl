"""
    io.jl

Input/Output functions for reading and writing spectral data.

This module handles reading spectral data from various file formats (CSV, SPC) and
reference files, aligning spectral data with target variables, and saving results.

Key Features:
- CSV reading with automatic format detection (wide vs. long format)
- SPC (GRAMS/Thermo Galactic) file support (stub implementation)
- Smart filename matching for aligning spectral data with reference files
- Comprehensive error handling and validation
- Support for multiple spectral file formats

Typical Usage:
1. Read reference CSV with target variables
2. Read spectral data from directory
3. Align spectral data with reference targets
4. Train models and save results
"""

module IO

using CSV
using DataFrames
using Statistics

export read_csv, read_spc, load_spectral_dataset, save_results,
       extract_sample_id, find_files, read_reference_csv, align_xy


"""
    read_csv(filepath::String)::DataFrame

Read spectral data from CSV file with automatic format detection.

Supports two CSV formats:
- **Wide format**: First column = sample ID, remaining columns = numeric wavelengths (nm)
- **Long format**: Single spectrum with 'wavelength' and 'value' columns (auto-pivoted)

# Arguments
- `filepath::String`: Path to CSV file

# Returns
- `DataFrame`: Wide format matrix with:
  - Rows indexed by sample ID
  - Columns = wavelengths (Float64) in ascending order
  - Values = spectral intensities

# Format Detection

**Wide Format Example:**
```
sample_id,400.0,402.0,404.0,...
sample1,0.123,0.145,0.167,...
sample2,0.234,0.256,0.278,...
```

**Long Format Example:**
```
wavelength,value
400.0,0.123
402.0,0.145
404.0,0.167
```

For long format, the filename (without extension) is used as the sample ID.

# Validation
- Ensures at least 100 wavelengths are present
- Verifies wavelengths are strictly increasing
- Checks for empty files
- Validates all column names can be parsed as wavelengths (wide format)

# Errors
- `ErrorException`: If file is empty, has too few wavelengths, or wavelengths not increasing
- `ArgumentError`: If column names cannot be parsed as numeric wavelengths

# Examples
```julia
# Read wide format CSV
df = read_csv("spectra.csv")
@assert size(df, 2) >= 100  # At least 100 wavelengths

# Read long format CSV (single spectrum)
df = read_csv("sample001.csv")
@assert nrow(df) == 1  # Single sample
```
"""
function read_csv(filepath::String)::DataFrame
    # Check file exists
    if !isfile(filepath)
        throw(ArgumentError("File not found: $filepath"))
    end

    # Read CSV
    df = CSV.read(filepath, DataFrame)

    # Check for empty file
    if nrow(df) == 0
        throw(ErrorException("Empty CSV file: $filepath"))
    end

    # Detect format by looking for wavelength/value columns
    col_names_lower = lowercase.(string.(names(df)))
    has_wavelength = any(x -> x in ["wavelength", "wavelength_nm"], col_names_lower)
    has_value = any(x -> x in ["value", "intensity", "reflectance", "pct_reflect"], col_names_lower)

    if has_wavelength && has_value
        # Long format - single spectrum
        return _read_csv_long_format(filepath, df)
    else
        # Wide format
        return _read_csv_wide_format(filepath, df)
    end
end


"""
    _read_csv_long_format(filepath::String, df::DataFrame)::DataFrame

Read CSV in long format (wavelength, value columns) and convert to wide format.

Internal function used by `read_csv`.
"""
function _read_csv_long_format(filepath::String, df::DataFrame)::DataFrame
    # Find wavelength and value columns (case-insensitive)
    col_names_lower = lowercase.(string.(names(df)))

    wl_col_idx = findfirst(x -> x in ["wavelength", "wavelength_nm"], col_names_lower)
    val_col_idx = findfirst(x -> x in ["value", "intensity", "reflectance", "pct_reflect"], col_names_lower)

    if isnothing(wl_col_idx) || isnothing(val_col_idx)
        throw(ErrorException("Could not find wavelength and value columns"))
    end

    wl_col = names(df)[wl_col_idx]
    val_col = names(df)[val_col_idx]

    # Extract sample ID from filename (without extension)
    sample_id = splitext(basename(filepath))[1]

    # Create DataFrame with wavelength and value
    df_subset = select(df, wl_col => :wavelength, val_col => :value)
    dropmissing!(df_subset)

    # Convert to wide format: one row with wavelengths as columns
    # Create dictionary of wavelength => value
    wl_dict = Dict{Float64, Float64}()
    for row in eachrow(df_subset)
        wl = Float64(row.wavelength)
        val = Float64(row.value)
        wl_dict[wl] = val
    end

    # Create DataFrame with sorted wavelengths
    wavelengths = sort(collect(keys(wl_dict)))
    values = [wl_dict[wl] for wl in wavelengths]

    # Create wide DataFrame
    result = DataFrame()
    result[!, :sample_id] = [sample_id]
    for (wl, val) in zip(wavelengths, values)
        result[!, Symbol(string(wl))] = [val]
    end

    # Remove sample_id column and use as index (simulate pandas indexing)
    # In Julia DataFrames, we don't have true row indices, so we keep sample_id as metadata
    # but return a DataFrame where the first column is sample_id

    return _validate_spectral_dataframe(result)
end


"""
    _read_csv_wide_format(filepath::String, df::DataFrame)::DataFrame

Read CSV in wide format (first column = ID, rest = wavelength columns).

Internal function used by `read_csv`.
"""
function _read_csv_wide_format(filepath::String, df::DataFrame)::DataFrame
    # First column is ID, rest should be numeric wavelengths
    id_col = names(df)[1]

    # Try to parse all other column names as wavelengths
    wl_cols = names(df)[2:end]
    wl_map = Dict{String, Float64}()

    for col in wl_cols
        try
            wl_map[col] = parse(Float64, col)
        catch e
            throw(ArgumentError("Could not parse column name '$col' as wavelength: $e"))
        end
    end

    # Rename columns to wavelengths and sort
    sorted_cols = sort(collect(keys(wl_map)), by = x -> wl_map[x])

    # Create new DataFrame with sorted wavelength columns
    result = DataFrame()
    result[!, :sample_id] = df[!, id_col]

    for col in sorted_cols
        wl = wl_map[col]
        result[!, Symbol(string(wl))] = df[!, col]
    end

    return _validate_spectral_dataframe(result)
end


"""
    _validate_spectral_dataframe(df::DataFrame)::DataFrame

Validate spectral DataFrame has enough wavelengths and they are strictly increasing.

Internal validation function.
"""
function _validate_spectral_dataframe(df::DataFrame)::DataFrame
    # Get wavelength columns (all except sample_id)
    wl_cols = [col for col in names(df) if col != :sample_id]

    if length(wl_cols) < 100
        throw(ErrorException("Expected at least 100 wavelengths, got $(length(wl_cols))"))
    end

    # Extract wavelengths and check they're strictly increasing
    wavelengths = [parse(Float64, string(col)) for col in wl_cols]

    for i in 2:length(wavelengths)
        if wavelengths[i] <= wavelengths[i-1]
            throw(ErrorException("Wavelengths must be strictly increasing"))
        end
    end

    return df
end


"""
    read_spc(filepath::String)::Tuple{Vector{Float64}, Vector{Float64}}

Read SPC (GRAMS/Thermo Galactic) binary spectral file.

**NOTE:** This is currently a stub implementation. SPC format support requires
additional binary parsing logic or external dependencies.

# Arguments
- `filepath::String`: Path to .spc file

# Returns
- `Tuple{Vector{Float64}, Vector{Float64}}`: (wavelengths, intensities)

# SPC File Format

The SPC format is a binary format used by GRAMS and Thermo Galactic software:
- Header section with metadata (file signature, data points, wavelength range)
- Data section with Float32 spectral values
- Multiple sub-file support (for multi-spectrum files)

# Current Status

This implementation throws an informative error directing users to:
1. Use CSV format instead (export from SPC software)
2. Wait for full SPC implementation (future enhancement)

# Future Implementation

To implement full SPC support:
1. Parse binary header (see Python spc-io or pyspectra libraries)
2. Extract wavelength information (start, end, number of points)
3. Read Float32 data section
4. Handle different SPC sub-types (evenly vs. unevenly spaced wavelengths)

# Errors
- `ErrorException`: Always (not yet implemented)

# Examples
```julia
# This will currently throw an error
try
    wl, intensities = read_spc("spectrum.spc")
catch e
    println("Use CSV format: ", e)
end
```
"""
function read_spc(filepath::String)::Tuple{Vector{Float64}, Vector{Float64}}
    throw(ErrorException(
        "SPC format not yet implemented in Julia port.\n" *
        "Options:\n" *
        "  1. Export your SPC files to CSV format using your spectroscopy software\n" *
        "  2. Use Python version which supports SPC via pyspectra library\n" *
        "  3. Wait for Julia SPC implementation (coming soon)\n" *
        "\n" *
        "For CSV export, use format:\n" *
        "  wavelength,value\n" *
        "  400.0,0.123\n" *
        "  402.0,0.145\n" *
        "  ..."
    ))
end


"""
    read_reference_csv(filepath::String, id_column::String)::DataFrame

Read reference CSV file containing target variables.

# Arguments
- `filepath::String`: Path to reference CSV file
- `id_column::String`: Column name to use as sample identifier (e.g., "sample_id", "filename")

# Returns
- `DataFrame`: Reference data with sample IDs and target variables

# Format

Expected CSV format:
```
sample_id,target1,target2,...
sample1,12.3,45.6,...
sample2,23.4,56.7,...
```

# Errors
- `ArgumentError`: If file not found or id_column not in CSV
- `ErrorException`: If file is empty

# Examples
```julia
# Read reference file with sample IDs and protein content
ref = read_reference_csv("reference.csv", "sample_id")

# Access target values
protein = ref[!, "protein_pct"]
```
"""
function read_reference_csv(filepath::String, id_column::String)::DataFrame
    if !isfile(filepath)
        throw(ArgumentError("File not found: $filepath"))
    end

    df = CSV.read(filepath, DataFrame)

    if nrow(df) == 0
        throw(ErrorException("Empty reference CSV file: $filepath"))
    end

    if !(id_column in names(df))
        available = join(names(df), ", ")
        throw(ArgumentError("Column '$id_column' not found in $filepath. Available: $available"))
    end

    return df
end


"""
    normalize_filename(filename::String)::String

Normalize filename for flexible matching between spectral data and reference files.

Removes common file extensions, spaces, and converts to lowercase to enable
matching between different naming conventions.

# Arguments
- `filename::String`: Filename or sample ID to normalize

# Returns
- `String`: Normalized filename (lowercase, no extension, no spaces)

# Removed Extensions
- .asd, .sig, .csv, .txt, .spc

# Examples
```julia
normalize_filename("Sample 001.asd")  # "sample001"
normalize_filename("sample_001.csv")  # "sample_001"
normalize_filename("SAMPLE001")       # "sample001"
```
"""
function normalize_filename(filename::String)::String
    # Remove common extensions
    for ext in [".asd", ".sig", ".csv", ".txt", ".spc"]
        if endswith(lowercase(filename), ext)
            filename = filename[1:end-length(ext)]
            break
        end
    end

    # Remove spaces and convert to lowercase
    filename = replace(filename, " " => "")
    filename = lowercase(filename)

    return filename
end


"""
    align_xy(
        X::DataFrame,
        ref::DataFrame,
        id_column::String,
        target_column::String
    )::Tuple{Matrix{Float64}, Vector{Float64}, Vector{String}}

Align spectral data with reference target variable using smart filename matching.

This function matches samples between spectral data and reference files, handling:
- Files with/without extensions ("sample.asd" vs "sample")
- Filenames with/without spaces ("Spectrum 001" vs "Spectrum001")
- Case differences ("SAMPLE" vs "sample")

# Arguments
- `X::DataFrame`: Spectral data (wide format, sample_id column + wavelength columns)
- `ref::DataFrame`: Reference data with target values
- `id_column::String`: Name of ID column in reference DataFrame
- `target_column::String`: Name of target variable column

# Returns
- `Tuple{Matrix{Float64}, Vector{Float64}, Vector{String}}`:
  - `X_matrix`: Aligned spectral data matrix (n_samples × n_wavelengths)
  - `y`: Target values (n_samples,)
  - `sample_ids`: Sample identifiers (n_samples,)

# Matching Strategy

1. Try exact ID matching first
2. If no exact matches, use normalized filename matching:
   - Remove extensions (.asd, .csv, etc.)
   - Remove spaces
   - Convert to lowercase
3. Report warnings for unmatched samples
4. Remove samples with missing target values

# Errors
- `ArgumentError`: If target column not found in reference
- `ErrorException`: If no matching samples found or all samples have missing targets

# Examples
```julia
# Read data
X = read_csv("spectra_dir/sample1.csv")
ref = read_reference_csv("reference.csv", "sample_id")

# Align - handles "sample1.csv" matching "sample1" in reference
X_matrix, y, ids = align_xy(X, ref, "sample_id", "protein_pct")

println("Matched ", length(y), " samples")
```
"""
function align_xy(
    X::DataFrame,
    ref::DataFrame,
    id_column::String,
    target_column::String
)::Tuple{Matrix{Float64}, Vector{Float64}, Vector{String}}

    # Validate target column exists
    if !(target_column in names(ref))
        available = join(names(ref), ", ")
        throw(ArgumentError("Target '$target_column' not found in reference. Available: $available"))
    end

    # Get sample IDs from spectral data and reference
    X_ids = X[!, :sample_id]
    ref_ids = ref[!, id_column]

    # Try exact matching first
    common_ids = intersect(X_ids, ref_ids)

    id_mapping = Dict{String, String}()  # ref_id => X_id

    if length(common_ids) == 0
        println("No exact ID matches found. Trying flexible filename matching...")

        # Create normalized name mappings
        X_norm_map = Dict(normalize_filename(string(id)) => string(id) for id in X_ids)
        ref_norm_map = Dict(normalize_filename(string(id)) => string(id) for id in ref_ids)

        # Find common normalized IDs
        common_norm_ids = intersect(keys(X_norm_map), keys(ref_norm_map))

        if length(common_norm_ids) == 0
            # Show debug info
            println("\nSpectral data IDs (first 5): ", X_ids[1:min(5, length(X_ids))])
            println("Reference IDs (first 5): ", ref_ids[1:min(5, length(ref_ids))])
            println("\nNormalized spectral IDs (first 5): ", collect(keys(X_norm_map))[1:min(5, length(X_norm_map))])
            println("Normalized reference IDs (first 5): ", collect(keys(ref_norm_map))[1:min(5, length(ref_norm_map))])

            throw(ErrorException(
                "No matching IDs between spectral data and reference.\n" *
                "Check that '$id_column' values match between files.\n" *
                "Tried matching with and without file extensions/spaces."
            ))
        end

        # Build mapping using normalized matching
        for norm_id in common_norm_ids
            ref_id = ref_norm_map[norm_id]
            X_id = X_norm_map[norm_id]
            id_mapping[ref_id] = X_id
        end

        println("Matched $(length(id_mapping)) samples using flexible filename matching")
    else
        # Use exact matches
        for id in common_ids
            id_mapping[string(id)] = string(id)
        end

        if length(common_ids) < nrow(X)
            println("Warning: $(nrow(X) - length(common_ids)) samples from spectral data have no reference")
        end

        if length(common_ids) < nrow(ref)
            println("Warning: $(nrow(ref) - length(common_ids)) samples from reference have no spectral data")
        end
    end

    # Build aligned datasets
    aligned_ref_ids = String[]
    aligned_X_rows = Int[]

    for (i, ref_id) in enumerate(ref[!, id_column])
        ref_id_str = string(ref_id)
        if haskey(id_mapping, ref_id_str)
            X_id = id_mapping[ref_id_str]
            X_row = findfirst(==(X_id), string.(X[!, :sample_id]))
            if !isnothing(X_row)
                push!(aligned_ref_ids, ref_id_str)
                push!(aligned_X_rows, X_row)
            end
        end
    end

    # Extract aligned data
    X_aligned = X[aligned_X_rows, :]
    y_values = ref[ref[!, id_column] .∈ Ref(aligned_ref_ids), target_column]

    # Drop NaN targets
    valid_mask = .!ismissing.(y_values)

    if !all(valid_mask)
        n_dropped = sum(.!valid_mask)
        println("Warning: Dropping $n_dropped samples with missing target values")
        X_aligned = X_aligned[valid_mask, :]
        y_values = y_values[valid_mask]
        aligned_ref_ids = aligned_ref_ids[valid_mask]
    end

    if length(y_values) == 0
        throw(ErrorException("No valid samples after alignment and NaN removal"))
    end

    # Convert to matrix (exclude sample_id column)
    wl_cols = [col for col in names(X_aligned) if col != :sample_id]
    X_matrix = Matrix{Float64}(X_aligned[!, wl_cols])
    y = Vector{Float64}(y_values)

    return X_matrix, y, aligned_ref_ids
end


"""
    load_spectral_dataset(
        spectra_dir::String,
        reference_file::String,
        id_column::String,
        target_column::String;
        file_extension::String=".csv"
    )::Tuple{Matrix{Float64}, Vector{Float64}, Vector{Float64}, Vector{String}}

Load complete spectral dataset from directory and align with reference file.

This is the main entry point for loading spectral data. It:
1. Reads reference CSV with target variables
2. Finds all spectral files in directory
3. Reads each spectral file
4. Aligns spectral data with reference targets
5. Returns aligned matrices ready for modeling

# Arguments
- `spectra_dir::String`: Directory containing spectral files
- `reference_file::String`: Path to reference CSV with target values
- `id_column::String`: Column name for sample IDs in reference file
- `target_column::String`: Column name for target variable
- `file_extension::String`: File extension to search for (default: ".csv")

# Returns
- `Tuple{Matrix{Float64}, Vector{Float64}, Vector{Float64}, Vector{String}}`:
  - `X`: Spectral data matrix (n_samples × n_wavelengths)
  - `y`: Target values (n_samples,)
  - `wavelengths`: Wavelength values (n_wavelengths,)
  - `sample_ids`: Sample identifiers (n_samples,)

# Process

1. Read reference CSV
2. Find all files matching extension in directory
3. Read each spectral file (CSV format)
4. Combine into single DataFrame
5. Align with reference using smart matching
6. Extract wavelengths from column names
7. Return aligned matrices

# Errors
- `ArgumentError`: If directory not found or no files found
- `ErrorException`: If no valid matches between spectra and reference

# Examples
```julia
# Load dataset
X, y, wavelengths, sample_ids = load_spectral_dataset(
    "data/spectra",
    "data/reference.csv",
    "sample_id",
    "protein_pct"
)

println("Loaded ", size(X, 1), " samples")
println("Features: ", size(X, 2), " wavelengths")
println("Wavelength range: ", wavelengths[1], "-", wavelengths[end], " nm")
```
"""
function load_spectral_dataset(
    spectra_dir::String,
    reference_file::String,
    id_column::String,
    target_column::String;
    file_extension::String=".csv"
)::Tuple{Matrix{Float64}, Vector{Float64}, Vector{Float64}, Vector{String}}

    # Validate directory exists
    if !isdir(spectra_dir)
        throw(ArgumentError("Directory not found: $spectra_dir"))
    end

    # Find spectral files
    spectral_files = find_files(spectra_dir, file_extension)

    if length(spectral_files) == 0
        throw(ArgumentError("No $file_extension files found in $spectra_dir"))
    end

    println("Found $(length(spectral_files)) spectral files")

    # Read reference file
    ref = read_reference_csv(reference_file, id_column)
    println("Loaded reference file with $(nrow(ref)) samples")

    # Read all spectral files and combine
    all_spectra = DataFrame[]

    for filepath in spectral_files
        try
            df = read_csv(filepath)
            push!(all_spectra, df)
        catch e
            println("Warning: Could not read $filepath: $e")
        end
    end

    if length(all_spectra) == 0
        throw(ErrorException("No valid spectral files could be read"))
    end

    # Combine all spectra into single DataFrame
    X_combined = vcat(all_spectra...)
    println("Combined $(nrow(X_combined)) spectra")

    # Align with reference
    X_matrix, y, sample_ids = align_xy(X_combined, ref, id_column, target_column)

    # Extract wavelengths from column names
    wl_cols = [col for col in names(X_combined) if col != :sample_id]
    wavelengths = [parse(Float64, string(col)) for col in wl_cols]

    println("Final dataset: $(length(y)) samples × $(length(wavelengths)) wavelengths")

    return X_matrix, y, wavelengths, sample_ids
end


"""
    find_files(directory::String, extension::String)::Vector{String}

Find all files with given extension in directory (non-recursive).

# Arguments
- `directory::String`: Directory to search
- `extension::String`: File extension (e.g., ".csv", ".spc")

# Returns
- `Vector{String}`: List of full file paths

# Examples
```julia
# Find all CSV files
csv_files = find_files("data/spectra", ".csv")

# Find all SPC files
spc_files = find_files("data/spectra", ".spc")
```
"""
function find_files(directory::String, extension::String)::Vector{String}
    if !isdir(directory)
        throw(ArgumentError("Not a directory: $directory"))
    end

    files = String[]
    for entry in readdir(directory, join=true)
        if isfile(entry) && endswith(lowercase(entry), lowercase(extension))
            push!(files, entry)
        end
    end

    return sort(files)
end


"""
    extract_sample_id(filename::String)::String

Extract sample ID from filename by removing extension.

# Arguments
- `filename::String`: Filename (can be full path or just filename)

# Returns
- `String`: Sample ID (filename without extension)

# Examples
```julia
extract_sample_id("sample001.csv")           # "sample001"
extract_sample_id("/path/to/sample001.asd")  # "sample001"
extract_sample_id("SAMPLE_001.SPC")          # "SAMPLE_001"
```
"""
function extract_sample_id(filename::String)::String
    # Get just the filename (not full path)
    base = basename(filename)

    # Remove extension
    return splitext(base)[1]
end


"""
    save_results(results::DataFrame, output_path::String)

Save results DataFrame to CSV file.

# Arguments
- `results::DataFrame`: Results to save (any DataFrame)
- `output_path::String`: Output CSV file path

# Format

Writes standard CSV with:
- Header row with column names
- Proper handling of numeric types
- Quoted strings if needed

# Examples
```julia
# Save prediction results
results = DataFrame(
    sample_id = ["s1", "s2", "s3"],
    predicted = [12.3, 45.6, 78.9],
    actual = [12.1, 45.8, 78.7]
)

save_results(results, "predictions.csv")
```
"""
function save_results(results::DataFrame, output_path::String)
    CSV.write(output_path, results)
    println("Results saved to $output_path")
end


end  # module IO
