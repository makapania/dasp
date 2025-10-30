"""
Test suite for io.jl module

Tests reading CSV files, aligning data, and basic I/O operations.
"""

using Test
using DataFrames
using CSV

# Add src to load path
push!(LOAD_PATH, joinpath(@__DIR__, "src"))

# Import the IO module
include("src/io.jl")
using .IO

# Create temporary test directory
test_dir = mktempdir()
println("Test directory: $test_dir")

@testset "IO Module Tests" begin

    @testset "CSV Reading - Wide Format" begin
        # Create test CSV in wide format
        csv_path = joinpath(test_dir, "test_wide.csv")
        open(csv_path, "w") do f
            write(f, "sample_id,400.0,402.0,404.0,406.0,408.0\n")
            write(f, "sample1,0.1,0.2,0.3,0.4,0.5\n")
            write(f, "sample2,0.2,0.3,0.4,0.5,0.6\n")
        end

        # Read CSV - note: will fail validation (< 100 wavelengths)
        # This tests the parsing logic, not validation
        try
            df = IO.read_csv(csv_path)
            @test false  # Should fail validation
        catch e
            @test occursin("at least 100 wavelengths", string(e))
        end

        # Create larger CSV for full test
        csv_path_large = joinpath(test_dir, "test_wide_large.csv")
        open(csv_path_large, "w") do f
            # Create header with 150 wavelengths
            header = "sample_id"
            wavelengths = collect(400.0:2.0:698.0)  # 150 wavelengths
            for wl in wavelengths
                header *= ",$wl"
            end
            write(f, header * "\n")

            # Write data rows
            for i in 1:3
                row = "sample$i"
                for j in 1:150
                    row *= ",$(0.1 * i + 0.01 * j)"
                end
                write(f, row * "\n")
            end
        end

        df = IO.read_csv(csv_path_large)
        @test nrow(df) == 3
        @test ncol(df) == 151  # 150 wavelengths + sample_id
        @test "sample_id" in names(df)
        @test df[1, :sample_id] == "sample1"
    end

    @testset "CSV Reading - Long Format" begin
        # Create test CSV in long format
        csv_path = joinpath(test_dir, "test_long.csv")
        open(csv_path, "w") do f
            write(f, "wavelength,value\n")
            for wl in 400.0:2.0:698.0  # 150 wavelengths
                write(f, "$wl,$(0.1 + 0.001 * wl)\n")
            end
        end

        df = IO.read_csv(csv_path)
        @test nrow(df) == 1  # Single sample
        @test "sample_id" in names(df)
        @test df[1, :sample_id] == "test_long"  # From filename
    end

    @testset "Reference CSV Reading" begin
        # Create reference CSV
        ref_path = joinpath(test_dir, "reference.csv")
        open(ref_path, "w") do f
            write(f, "sample_id,protein,moisture\n")
            write(f, "sample1,12.3,45.6\n")
            write(f, "sample2,23.4,56.7\n")
            write(f, "sample3,34.5,67.8\n")
        end

        ref = IO.read_reference_csv(ref_path, "sample_id")
        @test nrow(ref) == 3
        @test "protein" in names(ref)
        @test "moisture" in names(ref)

        # Test error for missing column
        @test_throws ArgumentError IO.read_reference_csv(ref_path, "nonexistent")
    end

    @testset "Filename Normalization" begin
        @test IO.normalize_filename("Sample 001.asd") == "sample001"
        @test IO.normalize_filename("sample_001.csv") == "sample_001"
        @test IO.normalize_filename("SAMPLE001") == "sample001"
        @test IO.normalize_filename("My File.spc") == "myfile"
    end

    @testset "Sample ID Extraction" begin
        @test IO.extract_sample_id("sample001.csv") == "sample001"
        @test IO.extract_sample_id("/path/to/sample001.asd") == "sample001"
        @test IO.extract_sample_id("SAMPLE_001.SPC") == "SAMPLE_001"
    end

    @testset "Align XY - Exact Match" begin
        # Create spectral data
        X = DataFrame(
            sample_id = ["sample1", "sample2", "sample3"],
            Symbol("400.0") => [0.1, 0.2, 0.3],
            Symbol("402.0") => [0.2, 0.3, 0.4],
            Symbol("404.0") => [0.3, 0.4, 0.5]
        )

        # Create reference data
        ref = DataFrame(
            sample_id = ["sample1", "sample2", "sample3", "sample4"],
            protein = [12.3, 23.4, 34.5, 45.6],
            moisture = [45.6, 56.7, 67.8, 78.9]
        )

        X_matrix, y, sample_ids = IO.align_xy(X, ref, "sample_id", "protein")

        @test size(X_matrix, 1) == 3  # 3 matched samples
        @test size(X_matrix, 2) == 3  # 3 wavelengths
        @test length(y) == 3
        @test length(sample_ids) == 3
        @test y[1] ≈ 12.3
        @test y[2] ≈ 23.4
        @test y[3] ≈ 34.5
    end

    @testset "Align XY - Normalized Match" begin
        # Create spectral data with file extensions
        X = DataFrame(
            sample_id = ["Sample 001.asd", "Sample 002.asd"],
            Symbol("400.0") => [0.1, 0.2],
            Symbol("402.0") => [0.2, 0.3],
            Symbol("404.0") => [0.3, 0.4]
        )

        # Create reference data without extensions
        ref = DataFrame(
            filename = ["sample001", "sample002", "sample003"],
            protein = [12.3, 23.4, 34.5]
        )

        X_matrix, y, sample_ids = IO.align_xy(X, ref, "filename", "protein")

        @test size(X_matrix, 1) == 2  # 2 matched samples
        @test length(y) == 2
        @test y[1] ≈ 12.3
        @test y[2] ≈ 23.4
    end

    @testset "Find Files" begin
        # Create test files
        touch(joinpath(test_dir, "file1.csv"))
        touch(joinpath(test_dir, "file2.csv"))
        touch(joinpath(test_dir, "file3.txt"))

        csv_files = IO.find_files(test_dir, ".csv")
        @test length(csv_files) == 2
        @test all(endswith(f, ".csv") for f in csv_files)

        txt_files = IO.find_files(test_dir, ".txt")
        @test length(txt_files) == 1

        # Test error for non-existent directory
        @test_throws ArgumentError IO.find_files("/nonexistent", ".csv")
    end

    @testset "Save Results" begin
        # Create test results
        results = DataFrame(
            sample_id = ["s1", "s2", "s3"],
            predicted = [12.3, 45.6, 78.9],
            actual = [12.1, 45.8, 78.7]
        )

        output_path = joinpath(test_dir, "results.csv")
        IO.save_results(results, output_path)

        @test isfile(output_path)

        # Read back and verify
        loaded = CSV.read(output_path, DataFrame)
        @test nrow(loaded) == 3
        @test ncol(loaded) == 3
        @test "sample_id" in names(loaded)
        @test "predicted" in names(loaded)
    end

    @testset "SPC Format (Stub)" begin
        # Test that SPC throws informative error
        @test_throws ErrorException IO.read_spc("dummy.spc")

        try
            IO.read_spc("dummy.spc")
        catch e
            @test occursin("not yet implemented", string(e))
            @test occursin("CSV format", string(e))
        end
    end

    @testset "Error Handling" begin
        # Non-existent file
        @test_throws ArgumentError IO.read_csv("/nonexistent.csv")

        # Empty CSV
        empty_csv = joinpath(test_dir, "empty.csv")
        open(empty_csv, "w") do f
            write(f, "header\n")  # Just header, no data
        end
        @test_throws ErrorException IO.read_csv(empty_csv)

        # Invalid wavelength columns
        invalid_csv = joinpath(test_dir, "invalid.csv")
        open(invalid_csv, "w") do f
            write(f, "id,abc,def,ghi\n")
            write(f, "s1,0.1,0.2,0.3\n")
        end
        @test_throws ArgumentError IO.read_csv(invalid_csv)
    end

    @testset "Load Spectral Dataset Integration" begin
        # Create spectral files
        spectra_dir = joinpath(test_dir, "spectra")
        mkdir(spectra_dir)

        for i in 1:3
            csv_path = joinpath(spectra_dir, "sample$i.csv")
            open(csv_path, "w") do f
                write(f, "wavelength,value\n")
                for wl in 400.0:2.0:698.0  # 150 wavelengths
                    write(f, "$wl,$(0.1 * i + 0.001 * wl)\n")
                end
            end
        end

        # Create reference
        ref_path = joinpath(test_dir, "reference_full.csv")
        open(ref_path, "w") do f
            write(f, "sample_id,protein\n")
            write(f, "sample1,12.3\n")
            write(f, "sample2,23.4\n")
            write(f, "sample3,34.5\n")
        end

        # Load dataset
        X, y, wavelengths, sample_ids = IO.load_spectral_dataset(
            spectra_dir,
            ref_path,
            "sample_id",
            "protein"
        )

        @test size(X, 1) == 3  # 3 samples
        @test size(X, 2) == 150  # 150 wavelengths
        @test length(y) == 3
        @test length(wavelengths) == 150
        @test length(sample_ids) == 3
        @test wavelengths[1] == 400.0
        @test wavelengths[end] == 698.0
        @test y[1] ≈ 12.3
    end
end

println("\n✓ All IO module tests passed!")
