"""
    test_cv.jl

Test suite for cross-validation framework.
"""

using Test
using Random
using Statistics

# Add src directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

include("../src/cv.jl")
include("../src/models.jl")
include("../src/preprocessing.jl")


@testset "Cross-Validation Framework Tests" begin

    @testset "CV Fold Creation" begin
        # Test basic fold creation
        folds = create_cv_folds(100, 5)
        @test length(folds) == 5

        # Check each fold
        for (train_idx, test_idx) in folds
            # Test sets should be disjoint
            @test isempty(intersect(train_idx, test_idx))

            # Should cover all samples
            @test sort([train_idx; test_idx]) == 1:100

            # Approximate sizes
            @test length(test_idx) ∈ [20, 19, 21]  # Allow ±1 for rounding
            @test length(train_idx) ∈ [80, 79, 81]
        end

        # Test uneven division
        folds = create_cv_folds(95, 10)
        @test length(folds) == 10
        test_sizes = [length(fold[2]) for fold in folds]
        @test sum(test_sizes) == 95
        @test all(s -> s ∈ [9, 10], test_sizes)

        # Test edge cases
        @test_throws ArgumentError create_cv_folds(0, 5)
        @test_throws ArgumentError create_cv_folds(10, 1)
        @test_throws ArgumentError create_cv_folds(10, 11)
    end

    @testset "Regression Metrics" begin
        # Perfect predictions
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.0, 2.0, 3.0, 4.0, 5.0]
        metrics = compute_regression_metrics(y_true, y_pred)

        @test metrics["RMSE"] ≈ 0.0 atol=1e-10
        @test metrics["R2"] ≈ 1.0 atol=1e-10
        @test metrics["MAE"] ≈ 0.0 atol=1e-10

        # Constant predictions (predict mean)
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = fill(3.0, 5)
        metrics = compute_regression_metrics(y_true, y_pred)

        @test metrics["RMSE"] ≈ sqrt(2.0) atol=1e-6
        @test metrics["R2"] ≈ 0.0 atol=1e-6
        @test metrics["MAE"] ≈ 1.2 atol=1e-6

        # Poor predictions (reversed)
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [5.0, 4.0, 3.0, 2.0, 1.0]
        metrics = compute_regression_metrics(y_true, y_pred)

        @test metrics["RMSE"] > 0.0
        @test metrics["R2"] < 0.0  # Worse than mean
        @test metrics["MAE"] > 0.0

        # Test edge cases
        @test_throws ArgumentError compute_regression_metrics(Float64[], Float64[])
        @test_throws ArgumentError compute_regression_metrics([1.0], [1.0, 2.0])
    end

    @testset "Classification Metrics" begin
        # Perfect classification
        y_true = [0.0, 0.0, 1.0, 1.0, 1.0]
        y_pred = [0.0, 0.1, 0.9, 0.95, 1.0]
        metrics = compute_classification_metrics(y_true, y_pred)

        @test metrics["Accuracy"] == 1.0
        @test metrics["ROC_AUC"] ≈ 1.0 atol=1e-6
        @test metrics["Precision"] == 1.0
        @test metrics["Recall"] == 1.0

        # Random predictions
        y_true = [0.0, 0.0, 1.0, 1.0, 1.0]
        y_pred = fill(0.5, 5)
        metrics = compute_classification_metrics(y_true, y_pred)

        @test metrics["Accuracy"] == 0.6  # All predicted as 1 (≥0.5)
        @test 0.4 < metrics["ROC_AUC"] < 0.6  # Should be near random

        # Edge cases
        @test_throws ArgumentError compute_classification_metrics(Float64[], Float64[])
    end

    @testset "Single Fold Execution" begin
        # Create synthetic data
        Random.seed!(42)
        X = rand(100, 20)
        y = rand(100)

        # Create fold
        folds = create_cv_folds(100, 5)
        train_idx, test_idx = folds[1]

        # Build model
        model = PLSModel(5)
        preprocess_config = Dict("name" => "snv")

        # Run fold
        metrics = run_single_fold(
            X, y, train_idx, test_idx,
            model, "PLS", preprocess_config, "regression"
        )

        # Check metrics exist
        @test haskey(metrics, "RMSE")
        @test haskey(metrics, "R2")
        @test haskey(metrics, "MAE")

        # Check reasonable values
        @test metrics["RMSE"] >= 0.0
        @test -1.0 <= metrics["R2"] <= 1.0
        @test metrics["MAE"] >= 0.0
    end

    @testset "Skip Preprocessing Mode" begin
        # Create data and preprocess it
        Random.seed!(42)
        X = rand(50, 10)
        y = rand(50)

        # Preprocess entire dataset
        preprocess_config = Dict("name" => "snv")
        X_preprocessed = apply_preprocessing(X, preprocess_config)

        # Create fold
        folds = create_cv_folds(50, 5)
        train_idx, test_idx = folds[1]

        # Build model
        model1 = PLSModel(3)
        model2 = PLSModel(3)

        # Run with skip_preprocessing=true (should use X_preprocessed as-is)
        metrics1 = run_single_fold(
            X_preprocessed, y, train_idx, test_idx,
            model1, "PLS", preprocess_config, "regression",
            skip_preprocessing=true
        )

        # Run with skip_preprocessing=false (should apply preprocessing again)
        metrics2 = run_single_fold(
            X_preprocessed, y, train_idx, test_idx,
            model2, "PLS", preprocess_config, "regression",
            skip_preprocessing=false
        )

        # Results should differ because double-preprocessing changes the data
        # (SNV on already-SNV'd data is different from original SNV)
        @test metrics1["RMSE"] != metrics2["RMSE"]
    end

    @testset "Full Cross-Validation" begin
        # Create synthetic regression data
        Random.seed!(42)
        X = rand(80, 15)
        y = rand(80)

        # Build model
        model = PLSModel(5)
        preprocess_config = Dict("name" => "raw")

        # Run 5-fold CV
        results = run_cross_validation(
            X, y, model, "PLS", preprocess_config, "regression",
            n_folds=5
        )

        # Check structure
        @test haskey(results, "RMSE_mean")
        @test haskey(results, "RMSE_std")
        @test haskey(results, "R2_mean")
        @test haskey(results, "R2_std")
        @test haskey(results, "MAE_mean")
        @test haskey(results, "MAE_std")
        @test haskey(results, "cv_scores")
        @test haskey(results, "n_folds")
        @test haskey(results, "task_type")

        # Check values
        @test results["n_folds"] == 5
        @test results["task_type"] == "regression"
        @test length(results["cv_scores"]) == 5

        # Check metrics are reasonable
        @test results["RMSE_mean"] >= 0.0
        @test results["RMSE_std"] >= 0.0
        @test -1.0 <= results["R2_mean"] <= 1.0
        @test results["MAE_mean"] >= 0.0
    end

    @testset "Model Config Extraction" begin
        # Test each model type
        pls = PLSModel(10)
        config = extract_model_config(pls, "PLS")
        @test config["n_components"] == 10

        ridge = RidgeModel(1.0)
        config = extract_model_config(ridge, "Ridge")
        @test config["alpha"] == 1.0

        lasso = LassoModel(0.5)
        config = extract_model_config(lasso, "Lasso")
        @test config["alpha"] == 0.5

        enet = ElasticNetModel(0.1, 0.5)
        config = extract_model_config(enet, "ElasticNet")
        @test config["alpha"] == 0.1
        @test config["l1_ratio"] == 0.5

        rf = RandomForestModel(100, "sqrt")
        config = extract_model_config(rf, "RandomForest")
        @test config["n_trees"] == 100
        @test config["max_features"] == "sqrt"

        mlp = MLPModel((50, 50), 0.01)
        config = extract_model_config(mlp, "MLP")
        @test config["hidden_layers"] == (50, 50)
        @test config["learning_rate"] == 0.01
    end

    @testset "Results Aggregation" begin
        # Create sample fold metrics
        fold_metrics = [
            Dict("RMSE" => 0.5, "R2" => 0.85, "MAE" => 0.4),
            Dict("RMSE" => 0.6, "R2" => 0.80, "MAE" => 0.5),
            Dict("RMSE" => 0.4, "R2" => 0.90, "MAE" => 0.3)
        ]

        results = aggregate_cv_results(fold_metrics, "regression", 3)

        # Check means
        @test results["RMSE_mean"] ≈ 0.5
        @test results["R2_mean"] ≈ 0.85
        @test results["MAE_mean"] ≈ 0.4

        # Check standard deviations
        @test results["RMSE_std"] ≈ std([0.5, 0.6, 0.4])
        @test results["R2_std"] ≈ std([0.85, 0.80, 0.90])
        @test results["MAE_std"] ≈ std([0.4, 0.5, 0.3])

        # Check metadata
        @test results["n_folds"] == 3
        @test results["task_type"] == "regression"
        @test results["cv_scores"] == fold_metrics
    end

    @testset "Different Models" begin
        # Test CV works with different model types
        Random.seed!(42)
        X = rand(60, 10)
        y = rand(60)
        preprocess_config = Dict("name" => "raw")

        # PLS
        model = PLSModel(3)
        results = run_cross_validation(X, y, model, "PLS", preprocess_config, "regression", n_folds=3)
        @test haskey(results, "RMSE_mean")

        # Ridge
        model = RidgeModel(1.0)
        results = run_cross_validation(X, y, model, "Ridge", preprocess_config, "regression", n_folds=3)
        @test haskey(results, "RMSE_mean")

        # Lasso
        model = LassoModel(0.1)
        results = run_cross_validation(X, y, model, "Lasso", preprocess_config, "regression", n_folds=3)
        @test haskey(results, "RMSE_mean")
    end

    @testset "Different Preprocessing" begin
        # Test CV with various preprocessing methods
        Random.seed!(42)
        X = rand(50, 20)
        y = rand(50)
        model = PLSModel(5)

        # Raw
        results = run_cross_validation(
            X, y, model, "PLS",
            Dict("name" => "raw"), "regression", n_folds=3
        )
        @test haskey(results, "RMSE_mean")

        # SNV
        results = run_cross_validation(
            X, y, model, "PLS",
            Dict("name" => "snv"), "regression", n_folds=3
        )
        @test haskey(results, "RMSE_mean")

        # Derivative
        results = run_cross_validation(
            X, y, model, "PLS",
            Dict("name" => "deriv", "deriv" => 1, "window" => 11, "polyorder" => 2),
            "regression", n_folds=3
        )
        @test haskey(results, "RMSE_mean")
    end

end

println("\n✓ All CV tests passed!")
