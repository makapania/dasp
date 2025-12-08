# Julia Migration: Validation Framework & Agent Architecture

**Date**: 2025-01-19
**Purpose**: Automated validation to ensure model reproducibility during Julia migration
**Core Principle**: "Create model in Results → Must reproduce exactly in Model Dev"

---

## The Golden Rule

**EVERY implementation must satisfy**:
```
R²(Results_Tab) - R²(Model_Dev) < 0.001
```

**No exceptions. No compromises. No "close enough".**

This applies to:
- All preprocessing combinations
- All wavelength restrictions
- All model types
- All subset analyses
- All data transformations

---

## The Three Agents (Automated Validation)

### Agent 1: Architect Agent (Design Validator)
**Role**: Enforce architectural rules BEFORE code is written
**Implementation**: Design rules as Julia type constraints and assertions

### Agent 2: Debugger Agent (Runtime Validator)
**Role**: Catch bugs during execution with extensive instrumentation
**Implementation**: Logging, checksums, intermediate value validation

### Agent 3: Testing Agent (Regression Validator)
**Role**: Continuously test every change against ground truth
**Implementation**: Automated test suite that runs on every commit

---

## Agent 1: Architect Agent (Design Validator)

### Rule 1: Wavelength Restriction MUST Happen Before Preprocessing
**Enforcement**: Type system + immutable config

```julia
# CORRECT - enforced by type system
struct PreprocessingPipeline
    restriction::WavelengthRestriction  # Applied FIRST (immutable)
    transforms::Vector{Transform}       # Applied SECOND (immutable)
end

# WRONG - compile error, can't construct this
# No way to create a pipeline with transforms before restriction!

function preprocess(X::Matrix, wl::Vector{Float64}, pipeline::PreprocessingPipeline)
    # Step 1: Restriction (type system guarantees this is first)
    X_restricted, wl_restricted = apply_restriction(X, wl, pipeline.restriction)

    # Step 2: Transforms (SNV, derivatives - on restricted data)
    X_transformed = apply_transforms(X_restricted, pipeline.transforms)

    return X_transformed, wl_restricted
end
```

**Validation check**: Architect agent scans code for ANY preprocessing that doesn't use `PreprocessingPipeline` type → ERROR

### Rule 2: Wavelength Order MUST Be Preserved
**Enforcement**: Never allow sorting, use OrderedSet/Vector only

```julia
# CORRECT - preserves order
function select_wavelengths(available::Vector{Float64}, indices::Vector{Int})
    return available[indices]  # Maintains order from available
end

# WRONG - would be caught by architect agent
function select_wavelengths_WRONG(available::Vector{Float64}, indices::Vector{Int})
    return sort(available[indices])  # ❌ FORBIDDEN - architect agent flags this
end

# Architect agent rule: Any call to sort() on wavelengths → COMPILE ERROR
```

**Validation check**: Architect agent scans for `sort()` calls on wavelength vectors → ERROR

### Rule 3: Model Metadata MUST Store Exact Training Configuration
**Enforcement**: Mandatory fields in ModelMetadata struct

```julia
struct ModelMetadata
    # MANDATORY - architect agent ensures these are ALWAYS set
    preprocessing_config::PreprocessingPipeline    # Complete pipeline
    selected_wavelengths::Vector{Float64}          # In exact order
    training_restriction::WavelengthRestriction    # What range was used
    cv_config::CrossValidationConfig               # Exact CV setup
    training_fingerprint::String                   # Hash of training data

    # Optional metadata
    timestamp::DateTime
    user_notes::String
end

# Architect agent rule: Can't save model without COMPLETE metadata
# Attempting to save with missing fields → COMPILE ERROR
```

**Validation check**: Architect agent ensures `ModelMetadata` has no `Union{Nothing, T}` optional fields for critical data

### Rule 4: SNV Normalization MUST See Correct Wavelength Range
**Enforcement**: SNV function signature requires wavelength context

```julia
# CORRECT - SNV knows what wavelengths it's normalizing
function snv_normalize(X::Matrix, wl::Vector{Float64})
    # wl parameter makes it explicit what range we're normalizing over
    n_samples, n_wavelengths = size(X)
    @assert length(wl) == n_wavelengths "Wavelength count mismatch!"

    # Normalize each spectrum (row)
    return (X .- mean(X, dims=2)) ./ std(X, dims=2)
end

# WRONG - architect agent would flag missing wavelength parameter
function snv_normalize_WRONG(X::Matrix)
    # ❌ No wavelength context - can't verify correct range!
    return (X .- mean(X, dims=2)) ./ std(X, dims=2)
end
```

**Validation check**: Architect agent ensures SNV is NEVER called without wavelength vector parameter

### Architect Agent Implementation

```julia
# architect_validator.jl

struct ArchitectValidator
    rules::Vector{ValidationRule}
end

struct ValidationRule
    name::String
    check::Function  # Returns (passed::Bool, message::String)
end

function validate_preprocessing_order(code_ast)
    # Parse AST, ensure restriction before transforms
    # Returns (true, "") if valid, (false, "Error message") if invalid
end

function validate_no_wavelength_sorting(code_ast)
    # Scan for sort() calls on wavelength variables
    # Returns (false, "Line 42: Illegal sort() on wavelengths") if found
end

function validate_metadata_completeness(model::TrainedModel)
    required_fields = [:preprocessing_config, :selected_wavelengths,
                       :training_restriction, :cv_config, :training_fingerprint]

    for field in required_fields
        if !isdefined(model.metadata, field)
            return (false, "Missing required metadata: $field")
        end
    end

    return (true, "")
end

# Run architect validation before EVERY model save
function save_model(model::TrainedModel, path::String)
    passed, msg = validate_metadata_completeness(model)
    if !passed
        error("ARCHITECT VALIDATION FAILED: $msg")
    end

    # Only save if validation passed
    serialize(path, model)
end
```

---

## Agent 2: Debugger Agent (Runtime Validator)

### Instrumentation Strategy: Log Everything Critical

```julia
struct DebugLogger
    enabled::Bool
    log_file::IOStream
end

const GLOBAL_LOGGER = Ref{Union{Nothing, DebugLogger}}(nothing)

function enable_debug_logging(path::String)
    GLOBAL_LOGGER[] = DebugLogger(true, open(path, "w"))
end

function debug_log(stage::String, data::Dict)
    if !isnothing(GLOBAL_LOGGER[]) && GLOBAL_LOGGER[].enabled
        timestamp = now()
        entry = Dict("timestamp" => timestamp, "stage" => stage, "data" => data)
        println(GLOBAL_LOGGER[].log_file, JSON.json(entry))
        flush(GLOBAL_LOGGER[].log_file)
    end
end

function debug_checkpoint(name::String, X::Matrix, wl::Vector{Float64})
    # Log shape, checksums, wavelength range at each pipeline stage
    debug_log("checkpoint", Dict(
        "name" => name,
        "shape" => size(X),
        "checksum" => hash(X),  # Detects ANY data change
        "wl_min" => minimum(wl),
        "wl_max" => maximum(wl),
        "wl_count" => length(wl),
        "wl_first_5" => wl[1:min(5, length(wl))],
        "wl_last_5" => wl[max(1, end-4):end]
    ))
end
```

### Instrumented Preprocessing Pipeline

```julia
function preprocess_with_debugging(X::Matrix, wl::Vector{Float64}, config::PreprocessingConfig)
    debug_checkpoint("input", X, wl)

    # Step 1: Wavelength restriction
    if !isnothing(config.restriction)
        wl_min, wl_max = config.restriction
        mask = (wl .>= wl_min) .& (wl .<= wl_max)
        X = X[:, mask]
        wl = wl[mask]

        debug_checkpoint("after_restriction", X, wl)
        debug_log("restriction_applied", Dict(
            "range" => (wl_min, wl_max),
            "n_kept" => sum(mask),
            "n_removed" => length(mask) - sum(mask)
        ))
    end

    # Step 2: Derivative
    if config.derivative_order > 0
        X = savitzky_golay_derivative(X, config.savgol_window,
                                      config.savgol_polyorder,
                                      config.derivative_order)
        debug_checkpoint("after_derivative", X, wl)
    end

    # Step 3: SNV
    if config.apply_snv
        X_before_snv = copy(X)  # Save for validation
        X = snv_normalize(X, wl)
        debug_checkpoint("after_snv", X, wl)

        # Validate SNV applied correctly
        mean_per_spectrum = mean(X, dims=2)
        std_per_spectrum = std(X, dims=2)

        if !all(abs.(mean_per_spectrum) .< 1e-10)
            error("DEBUGGER AGENT: SNV failed - mean not zero! Max mean: $(maximum(abs.(mean_per_spectrum)))")
        end

        if !all(abs.(std_per_spectrum .- 1.0) .< 1e-10)
            error("DEBUGGER AGENT: SNV failed - std not 1.0! Max deviation: $(maximum(abs.(std_per_spectrum .- 1.0)))")
        end
    end

    debug_checkpoint("output", X, wl)
    return X, wl
end
```

### Cross-Check Between Results and Model Dev

```julia
struct TrainingFingerprint
    data_hash::UInt64            # Hash of input X
    wavelength_hash::UInt64      # Hash of wavelength vector
    preprocessing_hash::UInt64   # Hash of config
    output_hash::UInt64          # Hash of preprocessed X

    # Detailed checksums for debugging
    wl_first::Float64
    wl_last::Float64
    wl_count::Int
    X_shape::Tuple{Int, Int}
    X_mean::Float64
    X_std::Float64
end

function compute_fingerprint(X::Matrix, wl::Vector{Float64},
                             config::PreprocessingConfig,
                             X_output::Matrix)
    return TrainingFingerprint(
        hash(X),
        hash(wl),
        hash(config),
        hash(X_output),
        wl[1],
        wl[end],
        length(wl),
        size(X),
        mean(X),
        std(X)
    )
end

# When training in Results tab
function train_model(X, y, wl, config)
    X_preprocessed, wl_preprocessed = preprocess_with_debugging(X, wl, config)

    fingerprint = compute_fingerprint(X, wl, config, X_preprocessed)

    # Train model...
    model = fit_pls(X_preprocessed, y)

    # Store fingerprint in metadata
    metadata = ModelMetadata(
        preprocessing_config = config,
        selected_wavelengths = wl_preprocessed,
        training_fingerprint = fingerprint,
        # ... other fields
    )

    return TrainedModel(model, metadata)
end

# When testing in Model Dev tab
function test_model(trained_model::TrainedModel, X_test, wl_test)
    # Recreate EXACT preprocessing
    X_preprocessed, wl_preprocessed = preprocess_with_debugging(
        X_test,
        wl_test,
        trained_model.metadata.preprocessing_config
    )

    # Compute fingerprint
    test_fingerprint = compute_fingerprint(
        X_test,
        wl_test,
        trained_model.metadata.preprocessing_config,
        X_preprocessed
    )

    # DEBUGGER AGENT: Compare fingerprints
    if test_fingerprint.preprocessing_hash != trained_model.metadata.training_fingerprint.preprocessing_hash
        error("""
        DEBUGGER AGENT FAILURE: Preprocessing mismatch!
        Training config: $(trained_model.metadata.training_fingerprint.preprocessing_hash)
        Testing config:  $(test_fingerprint.preprocessing_hash)
        → Preprocessing pipeline CHANGED between training and testing!
        """)
    end

    if test_fingerprint.wl_count != trained_model.metadata.training_fingerprint.wl_count
        error("""
        DEBUGGER AGENT FAILURE: Wavelength count mismatch!
        Training: $(trained_model.metadata.training_fingerprint.wl_count) wavelengths
        Testing:  $(test_fingerprint.wl_count) wavelengths
        → Wavelength restriction was applied differently!
        """)
    end

    # Make prediction
    y_pred = predict(trained_model.model, X_preprocessed)

    return y_pred
end
```

---

## Agent 3: Testing Agent (Regression Validator)

### Continuous Validation Test Suite

```julia
# test/test_r2_reproducibility.jl

using Test
using DASP
using Pickle  # Load Python validation data

@testset "R² Reproducibility - The Golden Rule" begin

    # Load ground truth from Python
    ground_truth = Pickle.load("validation/ground_truth.pkl")

    @testset "Derivative-only (must be PERFECT)" begin
        config = PreprocessingConfig(
            restriction = nothing,  # Whole spectrum
            apply_snv = false,
            derivative_order = 2,
            savgol_window = 11,
            savgol_polyorder = 2
        )

        # Preprocess
        X_julia, wl_julia = preprocess_with_debugging(
            ground_truth["X"],
            ground_truth["wavelengths"],
            config
        )

        # Train PLS
        model = fit_pls(X_julia, ground_truth["y"], n_components=10)

        # Predict
        y_pred = predict(model, X_julia)

        # Calculate R²
        r2_julia = calculate_r2(ground_truth["y"], y_pred)
        r2_python = ground_truth["expected_results"]["deriv_only_r2"]

        # MUST be exact match (< 0.001)
        @test abs(r2_julia - r2_python) < 0.001

        if abs(r2_julia - r2_python) >= 0.001
            @error """
            TESTING AGENT FAILURE: Derivative-only R² mismatch!
            Python:  $(r2_python)
            Julia:   $(r2_julia)
            Diff:    $(r2_julia - r2_python)

            This should be PERFECT. Something is fundamentally broken.
            Check debug log: $(GLOBAL_LOGGER[].log_file)
            """
        end
    end

    @testset "Derivative+SNV whole spectrum" begin
        config = PreprocessingConfig(
            restriction = nothing,
            apply_snv = true,
            derivative_order = 2,
            savgol_window = 11,
            savgol_polyorder = 2
        )

        X_julia, wl_julia = preprocess_with_debugging(
            ground_truth["X"],
            ground_truth["wavelengths"],
            config
        )

        model = fit_pls(X_julia, ground_truth["y"], n_components=10)
        y_pred = predict(model, X_julia)
        r2_julia = calculate_r2(ground_truth["y"], y_pred)
        r2_python = ground_truth["expected_results"]["deriv_snv_whole_r2"]

        @test abs(r2_julia - r2_python) < 0.001
    end

    @testset "Derivative+SNV with NIR restriction (THE CRITICAL TEST)" begin
        # This is the test that FAILS in current Python implementation!
        config = PreprocessingConfig(
            restriction = (1000.0, 2500.0),  # NIR range
            apply_snv = true,
            derivative_order = 2,
            savgol_window = 11,
            savgol_polyorder = 2
        )

        X_julia, wl_julia = preprocess_with_debugging(
            ground_truth["X"],
            ground_truth["wavelengths"],
            config
        )

        # DEBUGGER AGENT: Verify restriction was applied BEFORE SNV
        @test length(wl_julia) < length(ground_truth["wavelengths"])
        @test minimum(wl_julia) >= 1000.0
        @test maximum(wl_julia) <= 2500.0

        model = fit_pls(X_julia, ground_truth["y"], n_components=10)
        y_pred = predict(model, X_julia)
        r2_julia = calculate_r2(ground_truth["y"], y_pred)
        r2_python_results = ground_truth["expected_results"]["deriv_snv_nir_results_r2"]

        # Must match Results tab R² (the one trained on restricted data)
        @test abs(r2_julia - r2_python_results) < 0.001

        # Now test Model Dev reproduction
        # Save model, then reload and test
        saved_model = TrainedModel(model, ModelMetadata(
            preprocessing_config = config,
            selected_wavelengths = wl_julia,
            training_restriction = WavelengthRestriction(1000.0, 2500.0),
            # ... other fields
        ))

        # Simulate Model Dev: use same data but go through test_model()
        y_pred_modeldev = test_model(saved_model, ground_truth["X"], ground_truth["wavelengths"])
        r2_modeldev = calculate_r2(ground_truth["y"], y_pred_modeldev)

        # THE GOLDEN RULE: Results R² must equal Model Dev R²
        @test abs(r2_julia - r2_modeldev) < 0.001

        if abs(r2_julia - r2_modeldev) >= 0.001
            @error """
            TESTING AGENT CRITICAL FAILURE!

            The defining requirement is broken:
            Results tab R²:   $(r2_julia)
            Model Dev R²:     $(r2_modeldev)
            Difference:       $(r2_modeldev - r2_julia)

            This is the EXACT bug we're trying to fix!
            Check debug log for preprocessing differences.
            """

            # Halt all further development until this passes
            error("Cannot proceed with migration until R² reproducibility is fixed!")
        end
    end

    @testset "Wavelength ordering preserved" begin
        # Test that wavelength selection preserves order
        config = PreprocessingConfig(
            restriction = (1000.0, 2500.0),
            apply_snv = true,
            derivative_order = 2,
            savgol_window = 11,
            savgol_polyorder = 2
        )

        X_julia, wl_julia = preprocess_with_debugging(
            ground_truth["X"],
            ground_truth["wavelengths"],
            config
        )

        # Train and get feature importance
        model = fit_pls(X_julia, ground_truth["y"], n_components=10)
        importance = get_feature_importance(model, X_julia)

        # Select top 175 wavelengths
        top_indices = select_top_n(importance, 175)
        selected_wl = wl_julia[top_indices]

        # Check order preservation
        for i in 2:length(selected_wl)
            @test selected_wl[i] > selected_wl[i-1]  # Must be in ascending order
        end

        # Match Python selection exactly
        python_selected = ground_truth["expected_results"]["selected_wavelengths"]
        @test selected_wl == python_selected
    end
end

# Run this test suite after EVERY code change!
```

### Automated Testing Workflow

```julia
# scripts/run_validation.jl

using Pkg
Pkg.activate(".")
Pkg.test()  # Runs entire test suite

# If any test fails, HALT development
# Do not proceed until all tests pass
```

### Git Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Running DASP validation tests..."

cd dasp-julia
julia scripts/run_validation.jl

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ TESTING AGENT FAILURE: Validation tests failed!"
    echo "Commit rejected. Fix tests before committing."
    exit 1
fi

echo "✅ All validation tests passed"
exit 0
```

---

## Continuous Integration Setup

### GitHub Actions Workflow

```yaml
# .github/workflows/validate.yml

name: DASP Validation

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Setup Julia
        uses: julia-actions/setup-julia@v1
        with:
          version: '1.9'

      - name: Install dependencies
        run: |
          julia --project=dasp-julia -e 'using Pkg; Pkg.instantiate()'

      - name: Run validation suite
        run: |
          julia --project=dasp-julia scripts/run_validation.jl

      - name: Check R² reproducibility
        run: |
          julia --project=dasp-julia -e '
          using Test, DASP
          include("test/test_r2_reproducibility.jl")
          '

      - name: Upload debug logs if failed
        if: failure()
        uses: actions/upload-artifact@v2
        with:
          name: debug-logs
          path: dasp-julia/debug_*.log
```

---

## The Development Workflow with Agents

### Every Code Change Goes Through This

```
1. Write code
   ↓
2. ARCHITECT AGENT validates design
   - Preprocessing order correct?
   - Wavelength ordering preserved?
   - Metadata complete?
   ↓ (if passes)
3. Run code with DEBUGGER AGENT enabled
   - Logging active
   - Checksums computed
   - Intermediate values validated
   ↓
4. TESTING AGENT runs full test suite
   - R² reproducibility < 0.001?
   - Wavelength ordering correct?
   - Subset analysis matches?
   ↓ (if passes)
5. Commit allowed
   ↓
6. CI/CD runs full validation again
   ↓ (if passes)
7. Code merged to main
```

**If ANY step fails**: STOP, fix, repeat from step 1.

---

## Validation Checklist for Each Feature

### Before marking ANY feature "complete"

- [ ] **Architect validation**: Design follows all rules
- [ ] **Debugger logs**: Reviewed for anomalies
- [ ] **Unit tests**: All pass
- [ ] **R² reproducibility test**: < 0.001 difference
- [ ] **Wavelength ordering test**: Exact match
- [ ] **Subset analysis test**: Results match Python
- [ ] **Integration test**: Full workflow works
- [ ] **Performance test**: Meets speedup target
- [ ] **Code review**: Human verification
- [ ] **Documentation**: Updated

**Only mark complete when ALL boxes checked.**

---

## Emergency Stop Conditions

### Development MUST halt immediately if:

1. **R² difference > 0.001** on any deterministic model (PLS, Ridge, Lasso)
2. **Wavelength ordering changes** between train and test
3. **Subset analysis diverges** from Python by >0.1%
4. **Preprocessing fingerprints differ** between Results and Model Dev
5. **Any architect validation rule violated**

**Resume only after root cause found and fixed.**

---

## Success Criteria Summary

Migration is SUCCESSFUL when:

1. ✅ All R² reproducibility tests pass (< 0.001 difference)
2. ✅ Subset analysis produces identical results to Python
3. ✅ 5x speedup achieved (preferably 10-15x)
4. ✅ All architect validation rules enforced
5. ✅ Debugger agent logs show clean execution
6. ✅ Testing agent passes on every commit
7. ✅ Zero regressions from current Python functionality

**Not a single criterion is optional.**

---

## Conclusion

The three agents work together:

1. **Architect Agent**: Prevents bugs by design (type system, immutability)
2. **Debugger Agent**: Catches bugs at runtime (logging, checksums, validation)
3. **Testing Agent**: Catches regressions (continuous validation, golden rule)

**Result**: Impossible to break R² reproducibility without immediate detection.

This is the only way to migrate safely. Anything less will result in subtle bugs that take months to find.

---

**Document Version**: 1.0
**Date**: 2025-01-19
**Status**: Validation framework specification complete
