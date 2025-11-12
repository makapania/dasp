#!/usr/bin/env Rscript
# =============================================================================
# Random Forest Regression Validation: R vs Python
# =============================================================================
#
# This script trains Random Forest models using R's 'randomForest' package
# and compares results with Python's sklearn.ensemble.RandomForestRegressor
#
# Usage:
#   Rscript r_validation_scripts/random_forest_comparison.R
#
# Requirements:
#   install.packages("randomForest")
#   install.packages("jsonlite")
# =============================================================================

library(randomForest)
library(jsonlite)

# Set random seed for reproducibility (MUST match Python)
set.seed(42)

# Output directory
output_dir <- "r_validation_scripts/results/r"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cat(paste(rep("=", 80), collapse=""), "\n")
cat("RANDOM FOREST REGRESSION: R randomForest package\n")
cat(paste(rep("=", 80), collapse=""), "\n\n")

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
cat("Loading data...\n")

train_data <- read.csv("r_validation_scripts/results/rf_regression_train.csv")
test_data <- read.csv("r_validation_scripts/results/rf_regression_test.csv")

# Separate features and target
X_train <- as.matrix(train_data[, -ncol(train_data)])
y_train <- train_data$target

X_test <- as.matrix(test_data[, -ncol(test_data)])
y_test <- test_data$target

cat(sprintf("  Train samples: %d\n", nrow(X_train)))
cat(sprintf("  Test samples: %d\n", nrow(X_test)))
cat(sprintf("  Features: %d\n\n", ncol(X_train)))

# -----------------------------------------------------------------------------
# Train Random Forest model
# -----------------------------------------------------------------------------
# Fixed hyperparameters (MUST match Python)
n_trees <- 500
mtry <- floor(sqrt(ncol(X_train)))  # sqrt(n_features) - matches Python's 'sqrt'
nodesize <- 1  # Similar to Python's min_samples_split=2

cat("Training Random Forest model...\n")
cat(sprintf("  Trees (ntree): %d\n", n_trees))
cat(sprintf("  Features per split (mtry): %d (sqrt of %d)\n", mtry, ncol(X_train)))
cat(sprintf("  Min node size (nodesize): %d\n\n", nodesize))

# Train Random Forest model
# NOTE: randomForest in R uses different parameter names than sklearn:
#   - ntree = n_estimators
#   - mtry = max_features (when max_features='sqrt', mtry=sqrt(n_features))
#   - nodesize = min samples required to be at a leaf node (similar to min_samples_split)
rf_model <- randomForest(
  x = X_train,
  y = y_train,
  ntree = n_trees,
  mtry = mtry,
  nodesize = nodesize,
  importance = TRUE,  # Calculate feature importances
  keep.forest = TRUE
)

cat("Model trained successfully!\n\n")

# -----------------------------------------------------------------------------
# Make predictions
# -----------------------------------------------------------------------------
cat("Making predictions...\n")

y_train_pred <- predict(rf_model, X_train)
y_test_pred <- predict(rf_model, X_test)

# Calculate metrics
rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

r2 <- function(actual, predicted) {
  ss_res <- sum((actual - predicted)^2)
  ss_tot <- sum((actual - mean(actual))^2)
  1 - (ss_res / ss_tot)
}

train_rmse <- rmse(y_train, y_train_pred)
test_rmse <- rmse(y_test, y_test_pred)
train_r2 <- r2(y_train, y_train_pred)
test_r2 <- r2(y_test, y_test_pred)

cat("\nR Random Forest Results:\n")
cat(sprintf("  Train RMSE: %.6f\n", train_rmse))
cat(sprintf("  Test RMSE:  %.6f\n", test_rmse))
cat(sprintf("  Train R²:   %.6f\n", train_r2))
cat(sprintf("  Test R²:    %.6f\n\n", test_r2))

# -----------------------------------------------------------------------------
# Save results
# -----------------------------------------------------------------------------
cat("Saving results...\n")

# Save predictions and metrics
results <- list(
  predictions = list(
    train = as.vector(y_train_pred),
    test = as.vector(y_test_pred)
  ),
  model_info = list(
    model_type = "RandomForest",
    n_estimators = n_trees,
    mtry = mtry,
    nodesize = nodesize,
    train_rmse = train_rmse,
    test_rmse = test_rmse,
    train_r2 = train_r2,
    test_r2 = test_r2,
    package = "randomForest",
    r_version = R.version.string
  ),
  random_seed = 42
)

# Save as JSON
write_json(results, file.path(output_dir, "rf_regression.json"), auto_unbox = TRUE, pretty = TRUE)

# Save feature importances
# R's randomForest provides two importance measures:
#   - %IncMSE: decrease in accuracy when variable is permuted
#   - IncNodePurity: total decrease in node impurity from splits on the variable
# We'll use IncNodePurity to match sklearn's default feature_importances_
importance_values <- importance(rf_model)[, "IncNodePurity"]

# Normalize to match sklearn (which normalizes to sum=1)
importance_values_normalized <- importance_values / sum(importance_values)

importance_df <- data.frame(
  feature = paste0("feature_", 0:(length(importance_values) - 1)),
  importance = importance_values_normalized,
  importance_raw = importance_values
)
write.csv(importance_df, file.path(output_dir, "rf_regression_importance.csv"), row.names = FALSE)

cat(sprintf("  Saved: %s\n", file.path(output_dir, "rf_regression.json")))
cat(sprintf("  Saved: %s\n\n", file.path(output_dir, "rf_regression_importance.csv")))

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
cat(paste(rep("=", 80), collapse=""), "\n")
cat("RANDOM FOREST REGRESSION COMPLETE\n")
cat(paste(rep("=", 80), collapse=""), "\n\n")

cat("Next step: Compare results\n")
cat("  python r_validation_scripts/compare_results.py --model rf_regression\n\n")

cat("Expected result:\n")
cat("  Python and R predictions should be similar but NOT identical\n")
cat("  (Random Forest has stochastic elements even with same seed)\n")
cat("  RMSE should be within 5-10%\n")
cat("  Feature importances should be correlated (correlation > 0.9)\n\n")

cat("NOTE: Random Forest models in R and Python may differ slightly due to:\n")
cat("  1. Different random number generators\n")
cat("  2. Different tie-breaking rules\n")
cat("  3. Different splitting criteria implementations\n")
cat("  This is expected and doesn't indicate a problem.\n\n")
