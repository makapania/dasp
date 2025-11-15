# ==============================================================================
# DASP Validation Testing - R Regression Models (Comprehensive)
# ==============================================================================
#
# This script tests ALL regression datasets:
#   1. Bone Collagen (n=49: 36 train, 13 test)
#   2. Enamel d13C (n=140: 105 train, 35 test)
#
# Models tested:
#   - PLS Regression
#   - Ridge Regression
#   - Lasso Regression
#   - ElasticNet Regression
#   - Random Forest
#   - XGBoost
#
# Usage: Rscript regression_models_comprehensive.R
#
# ==============================================================================

# Suppress package startup messages
suppressPackageStartupMessages({
  library(pls)
  library(glmnet)
  library(randomForest)
  library(xgboost)
  library(prospectr)
  library(jsonlite)
  library(dplyr)
})

cat("================================================================================\n")
cat("DASP Validation Testing - R Regression Models (Comprehensive)\n")
cat("================================================================================\n\n")

# Set random seed for reproducibility
set.seed(42)

# ==============================================================================
# Helper Functions
# ==============================================================================

calculate_metrics <- function(y_true, y_pred, model_name) {
  rmse <- sqrt(mean((y_true - y_pred)^2))
  mae <- mean(abs(y_true - y_pred))
  ss_res <- sum((y_true - y_pred)^2)
  ss_tot <- sum((y_true - mean(y_true))^2)
  r2 <- 1 - (ss_res / ss_tot)

  list(
    model = model_name,
    rmse = rmse,
    r2 = r2,
    mae = mae,
    n_samples = length(y_true)
  )
}

# ==============================================================================
# Function to Test All Models on a Dataset
# ==============================================================================

test_regression_models <- function(dataset_name, data_dir, target_col, results_dir) {

  cat("\n", "=", rep("=", 78), "=\n", sep="")
  cat(sprintf("TESTING DATASET: %s\n", toupper(dataset_name)))
  cat("=", rep("=", 78), "=\n", sep="")

  # Create results directory
  dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

  # Load data
  cat("\nLoading data from:", data_dir, "\n")
  X_train <- as.matrix(read.csv(file.path(data_dir, "X_train.csv")))
  X_test <- as.matrix(read.csv(file.path(data_dir, "X_test.csv")))

  y_train_data <- read.csv(file.path(data_dir, "y_train.csv"))
  y_test_data <- read.csv(file.path(data_dir, "y_test.csv"))

  y_train <- y_train_data[[target_col]]
  y_test <- y_test_data[[target_col]]

  cat(sprintf("  Train: %d samples x %d wavelengths\n", nrow(X_train), ncol(X_train)))
  cat(sprintf("  Test: %d samples\n", nrow(X_test)))
  cat(sprintf("  Target range: %.2f - %.2f\n", min(y_train), max(y_train)))

  all_results <- list()

  # ============================================================================
  # Model 1: PLS Regression
  # ============================================================================

  cat("\n", "-", rep("-", 78), "-\n", sep="")
  cat("PLS Regression\n")
  cat("-", rep("-", 78), "-\n", sep="")

  pls_n_components <- c(2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50)
  max_comp <- min(nrow(X_train) - 1, ncol(X_train))
  pls_n_components <- pls_n_components[pls_n_components <= max_comp]

  cat(sprintf("\nTesting n_components: %s\n", paste(pls_n_components, collapse=", ")))

  pls_results <- list()
  for (ncomp in pls_n_components) {
    cat(sprintf("  n_components=%d... ", ncomp))

    pls_model <- plsr(y_train ~ X_train, ncomp = ncomp,
                     validation = "none", method = "oscorespls")
    y_pred <- predict(pls_model, newdata = X_test, ncomp = ncomp)[, 1, 1]

    metrics <- calculate_metrics(y_test, y_pred, sprintf("PLS_%d", ncomp))
    pls_results[[as.character(ncomp)]] <- metrics

    cat(sprintf("R²=%.3f, RMSE=%.3f\n", metrics$r2, metrics$rmse))
  }

  all_results <- c(all_results, pls_results)
  write_json(pls_results, file.path(results_dir, "pls_results.json"),
             pretty = TRUE, auto_unbox = TRUE)

  # ============================================================================
  # Model 2: Ridge Regression
  # ============================================================================

  cat("\n", "-", rep("-", 78), "-\n", sep="")
  cat("Ridge Regression\n")
  cat("-", rep("-", 78), "-\n", sep="")

  ridge_lambdas <- c(0.001, 0.01, 0.1, 1.0, 10.0)
  cat(sprintf("\nTesting lambda values: %s\n", paste(ridge_lambdas, collapse=", ")))

  ridge_results <- list()
  for (lambda_val in ridge_lambdas) {
    cat(sprintf("  lambda=%.3f... ", lambda_val))

    ridge_model <- glmnet(X_train, y_train, alpha = 0, lambda = lambda_val,
                         standardize = FALSE)
    y_pred <- predict(ridge_model, newx = X_test, s = lambda_val)[, 1]

    metrics <- calculate_metrics(y_test, y_pred, sprintf("Ridge_%.3f", lambda_val))
    ridge_results[[as.character(lambda_val)]] <- metrics

    cat(sprintf("R²=%.3f, RMSE=%.3f\n", metrics$r2, metrics$rmse))
  }

  all_results <- c(all_results, ridge_results)
  write_json(ridge_results, file.path(results_dir, "ridge_results.json"),
             pretty = TRUE, auto_unbox = TRUE)

  # ============================================================================
  # Model 3: Lasso Regression
  # ============================================================================

  cat("\n", "-", rep("-", 78), "-\n", sep="")
  cat("Lasso Regression\n")
  cat("-", rep("-", 78), "-\n", sep="")

  lasso_lambdas <- c(0.001, 0.01, 0.1, 1.0)
  cat(sprintf("\nTesting lambda values: %s\n", paste(lasso_lambdas, collapse=", ")))

  lasso_results <- list()
  for (lambda_val in lasso_lambdas) {
    cat(sprintf("  lambda=%.3f... ", lambda_val))

    lasso_model <- glmnet(X_train, y_train, alpha = 1, lambda = lambda_val,
                         standardize = FALSE)
    y_pred <- predict(lasso_model, newx = X_test, s = lambda_val)[, 1]

    metrics <- calculate_metrics(y_test, y_pred, sprintf("Lasso_%.3f", lambda_val))
    lasso_results[[as.character(lambda_val)]] <- metrics

    cat(sprintf("R²=%.3f, RMSE=%.3f\n", metrics$r2, metrics$rmse))
  }

  all_results <- c(all_results, lasso_results)
  write_json(lasso_results, file.path(results_dir, "lasso_results.json"),
             pretty = TRUE, auto_unbox = TRUE)

  # ============================================================================
  # Model 4: Random Forest
  # ============================================================================

  cat("\n", "-", rep("-", 78), "-\n", sep="")
  cat("Random Forest Regression\n")
  cat("-", rep("-", 78), "-\n", sep="")

  rf_n_estimators <- c(100, 200)
  rf_max_depth <- c(15, 30)

  rf_results <- list()
  for (ntree in rf_n_estimators) {
    for (max_depth in c(rf_max_depth, -1)) {

      depth_label <- if (max_depth == -1) "None" else as.character(max_depth)
      config_name <- sprintf("RF_ntree%d_depth%s", ntree, depth_label)

      cat(sprintf("  ntree=%d, max_depth=%s... ", ntree, depth_label))

      if (max_depth == -1) {
        rf_model <- randomForest(X_train, y_train, ntree = ntree,
                                 mtry = sqrt(ncol(X_train)), nodesize = 1,
                                 importance = TRUE)
      } else {
        max_nodes <- 2^max_depth
        rf_model <- randomForest(X_train, y_train, ntree = ntree,
                                 mtry = sqrt(ncol(X_train)), nodesize = 1,
                                 maxnodes = max_nodes, importance = TRUE)
      }

      y_pred <- predict(rf_model, newdata = X_test)

      metrics <- calculate_metrics(y_test, y_pred, config_name)
      metrics$ntree <- ntree
      metrics$max_depth <- if (max_depth == -1) NULL else max_depth
      rf_results[[config_name]] <- metrics

      cat(sprintf("R²=%.3f, RMSE=%.3f\n", metrics$r2, metrics$rmse))
    }
  }

  all_results <- c(all_results, rf_results)
  write_json(rf_results, file.path(results_dir, "rf_results.json"),
             pretty = TRUE, auto_unbox = TRUE)

  # ============================================================================
  # Model 5: XGBoost
  # ============================================================================

  cat("\n", "-", rep("-", 78), "-\n", sep="")
  cat("XGBoost Regression\n")
  cat("-", rep("-", 78), "-\n", sep="")

  xgb_n_estimators <- c(100, 200)
  xgb_learning_rate <- c(0.05, 0.1)
  xgb_max_depth <- c(3, 6)

  dtrain <- xgb.DMatrix(data = X_train, label = y_train)
  dtest <- xgb.DMatrix(data = X_test, label = y_test)

  xgb_results <- list()
  for (nrounds in xgb_n_estimators) {
    for (eta in xgb_learning_rate) {
      for (max_depth in xgb_max_depth) {

        config_name <- sprintf("XGB_n%d_lr%.2f_depth%d", nrounds, eta, max_depth)

        cat(sprintf("  nrounds=%d, eta=%.2f, max_depth=%d... ", nrounds, eta, max_depth))

        xgb_model <- xgb.train(
          data = dtrain,
          params = list(
            objective = "reg:squarederror",
            eta = eta,
            max_depth = max_depth,
            subsample = 0.8,
            colsample_bytree = 0.8
          ),
          nrounds = nrounds,
          verbose = 0
        )

        y_pred <- predict(xgb_model, dtest)

        metrics <- calculate_metrics(y_test, y_pred, config_name)
        metrics$nrounds <- nrounds
        metrics$eta <- eta
        metrics$max_depth <- max_depth
        xgb_results[[config_name]] <- metrics

        cat(sprintf("R²=%.3f, RMSE=%.3f\n", metrics$r2, metrics$rmse))
      }
    }
  }

  all_results <- c(all_results, xgb_results)
  write_json(xgb_results, file.path(results_dir, "xgboost_results.json"),
             pretty = TRUE, auto_unbox = TRUE)

  # ============================================================================
  # Summary for this dataset
  # ============================================================================

  cat("\n", "=", rep("=", 78), "=\n", sep="")
  cat(sprintf("Summary: %s\n", dataset_name))
  cat("=", rep("=", 78), "=\n", sep="")

  all_r2 <- sapply(all_results, function(x) x$r2)
  best_idx <- which.max(all_r2)
  best_model <- all_results[[best_idx]]

  cat(sprintf("\nTotal models tested: %d\n", length(all_results)))
  cat(sprintf("Best model: %s\n", best_model$model))
  cat(sprintf("  R² = %.4f\n", best_model$r2))
  cat(sprintf("  RMSE = %.4f\n", best_model$rmse))
  cat(sprintf("  MAE = %.4f\n", best_model$mae))

  # Save combined results
  write_json(all_results, file.path(results_dir, "all_models_summary.json"),
             pretty = TRUE, auto_unbox = TRUE)

  cat(sprintf("\nResults saved to: %s\n", results_dir))

  return(all_results)
}

# ==============================================================================
# Main Execution
# ==============================================================================

# Test Dataset 1: Bone Collagen
bone_results <- test_regression_models(
  dataset_name = "Bone Collagen",
  data_dir = file.path("..", "r_data", "regression"),
  target_col = "X.Collagen",  # Column might have X. prefix
  results_dir = file.path("..", "results", "r_regression", "bone_collagen")
)

# Test Dataset 2: Enamel d13C
d13c_results <- test_regression_models(
  dataset_name = "Enamel d13C",
  data_dir = file.path("..", "r_data", "d13c"),
  target_col = "d13C",
  results_dir = file.path("..", "results", "r_regression", "d13c")
)

# ==============================================================================
# Overall Summary
# ==============================================================================

cat("\n", "=", rep("=", 78), "=\n", sep="")
cat("OVERALL SUMMARY - ALL DATASETS\n")
cat("=", rep("=", 78), "=\n", sep="")

cat("\nBone Collagen (n=49):\n")
bone_r2 <- sapply(bone_results, function(x) x$r2)
best_bone <- bone_results[[which.max(bone_r2)]]
cat(sprintf("  Best: %s (R²=%.4f, RMSE=%.4f)\n",
            best_bone$model, best_bone$r2, best_bone$rmse))

cat("\nEnamel d13C (n=140):\n")
d13c_r2 <- sapply(d13c_results, function(x) x$r2)
best_d13c <- d13c_results[[which.max(d13c_r2)]]
cat(sprintf("  Best: %s (R²=%.4f, RMSE=%.4f)\n",
            best_d13c$model, best_d13c$r2, best_d13c$rmse))

cat("\n", "=", rep("=", 78), "=\n", sep="")
cat("COMPLETE! All R regression tests finished.\n")
cat("=", rep("=", 78), "=\n", sep="")
cat("\nNext steps:\n")
cat("  1. Run DASP regression tests: python dasp_regression.py\n")
cat("  2. Compare results: python compare_regression_results.py\n")
cat("================================================================================\n")
