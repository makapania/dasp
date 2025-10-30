"""
Example usage of the IO module for loading spectral data

This script demonstrates how to use the IO module to:
1. Load spectral data from CSV files
2. Read reference data with targets
3. Align spectral data with reference
4. Save results
"""

# Add src to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

include("../src/io.jl")
using .IO
using DataFrames

println("=" ^ 60)
println("IO Module Example Usage")
println("=" ^ 60)

# Example 1: Reading a single CSV file
println("\n1. Reading Single CSV File")
println("-" ^ 60)

# Wide format CSV
println("Wide format: First column = ID, rest = wavelength columns")
println("Format: sample_id,400.0,402.0,404.0,...")
println("        sample1,0.123,0.145,0.167,...")

# Long format CSV
println("\nLong format: wavelength and value columns")
println("Format: wavelength,value")
println("        400.0,0.123")
println("        402.0,0.145")

println("\nUsage:")
println("  df = read_csv(\"path/to/spectrum.csv\")")
println("  # Returns DataFrame with sample_id + wavelength columns")


# Example 2: Reading reference file
println("\n\n2. Reading Reference File")
println("-" ^ 60)
println("Reference CSV contains sample IDs and target variables")
println("Format: sample_id,protein,moisture,fat,...")
println("        sample1,12.3,45.6,7.8,...")
println("        sample2,23.4,56.7,8.9,...")

println("\nUsage:")
println("  ref = read_reference_csv(\"reference.csv\", \"sample_id\")")
println("  protein = ref[!, \"protein\"]")


# Example 3: Aligning spectral data with reference
println("\n\n3. Aligning Spectral Data with Reference")
println("-" ^ 60)
println("Smart matching handles:")
println("  - Different file extensions: 'sample.asd' ↔ 'sample'")
println("  - Spaces: 'Sample 001' ↔ 'Sample001'")
println("  - Case: 'SAMPLE' ↔ 'sample'")

println("\nUsage:")
println("  X_matrix, y, sample_ids = align_xy(")
println("      spectral_df,")
println("      reference_df,")
println("      \"sample_id\",    # ID column name in reference")
println("      \"protein_pct\"   # Target variable")
println("  )")
println("  # Returns:")
println("  #   X_matrix: n_samples × n_wavelengths")
println("  #   y: target values (n_samples,)")
println("  #   sample_ids: matched sample IDs")


# Example 4: Complete dataset loading
println("\n\n4. Complete Dataset Loading (Recommended)")
println("-" ^ 60)
println("Load all spectral files and align with reference in one step")

println("\nUsage:")
println("  X, y, wavelengths, sample_ids = load_spectral_dataset(")
println("      \"data/spectra\",           # Directory with spectral files")
println("      \"data/reference.csv\",     # Reference file")
println("      \"sample_id\",              # ID column in reference")
println("      \"protein_pct\",            # Target variable")
println("      file_extension=\".csv\"     # File type to search")
println("  )")
println("  ")
println("  println(\"Loaded \$(size(X, 1)) samples\")")
println("  println(\"Features: \$(size(X, 2)) wavelengths\")")
println("  println(\"Range: \$(wavelengths[1])-\$(wavelengths[end]) nm\")")


# Example 5: Saving results
println("\n\n5. Saving Results")
println("-" ^ 60)
println("Save predictions or any results to CSV")

println("\nUsage:")
println("  results = DataFrame(")
println("      sample_id = sample_ids,")
println("      actual = y,")
println("      predicted = predictions")
println("  )")
println("  save_results(results, \"predictions.csv\")")


# Example 6: Complete workflow
println("\n\n6. Complete Workflow Example")
println("-" ^ 60)
println("Typical spectral analysis pipeline:")

println("\nCode:")
println("  # Load data")
println("  X, y, wavelengths, ids = load_spectral_dataset(")
println("      \"data/spectra\", \"data/ref.csv\", \"sample_id\", \"protein\"")
println("  )")
println("  ")
println("  # Preprocess (from Preprocessing module)")
println("  using .Preprocessing")
println("  X_snv = snv(X)")
println("  X_deriv = savitzky_golay(X_snv, wavelengths)")
println("  ")
println("  # Train model (from Models module)")
println("  using .Models")
println("  model = PLSModel(n_components=10)")
println("  fit!(model, X_deriv, y)")
println("  ")
println("  # Predict")
println("  predictions = predict(model, X_deriv)")
println("  ")
println("  # Save results")
println("  results = DataFrame(")
println("      sample_id = ids,")
println("      actual = y,")
println("      predicted = predictions,")
println("      error = abs.(y .- predictions)")
println("  )")
println("  save_results(results, \"predictions.csv\")")


# Example 7: Error handling
println("\n\n7. Error Handling")
println("-" ^ 60)
println("The module provides comprehensive error messages:")

println("\nCommon errors:")
println("  • File not found → ArgumentError with path")
println("  • Missing ID column → Lists available columns")
println("  • No matching samples → Shows first few IDs from each file")
println("  • Too few wavelengths → Reports actual count")
println("  • Missing target values → Reports count dropped")

println("\nExample:")
println("  try")
println("      df = read_csv(\"nonexistent.csv\")")
println("  catch e")
println("      println(\"Error: \", e)")
println("  end")


# Example 8: File finding utilities
println("\n\n8. File Finding Utilities")
println("-" ^ 60)

println("\nFind all files with extension:")
println("  csv_files = find_files(\"data/spectra\", \".csv\")")
println("  spc_files = find_files(\"data/spectra\", \".spc\")")

println("\nExtract sample ID from filename:")
println("  id = extract_sample_id(\"sample001.csv\")  # \"sample001\"")
println("  id = extract_sample_id(\"/path/to/data.asd\")  # \"data\"")


# Example 9: SPC format (future)
println("\n\n9. SPC Format Support")
println("-" ^ 60)
println("SPC support is currently a stub implementation.")
println("It throws an informative error with alternatives:")

println("\nOptions:")
println("  1. Export SPC to CSV from your spectroscopy software")
println("  2. Use Python version (has SPC support via pyspectra)")
println("  3. Wait for Julia SPC implementation")

println("\nExample error:")
println("  try")
println("      wl, intensities = read_spc(\"spectrum.spc\")")
println("  catch e")
println("      # Error explains CSV export format and alternatives")
println("  end")


println("\n\n" * "=" ^ 60)
println("For more details, see:")
println("  • IO_MODULE_COMPLETE.md")
println("  • test_io.jl")
println("  • src/io.jl docstrings")
println("=" ^ 60)
