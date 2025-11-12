#!/usr/bin/env Rscript
# =============================================================================
# PLS Regression Validation: R vs Python
# =============================================================================
#
# This script trains PLS models using R's 'pls' package and compares results
# with Python's sklearn.cross_decomposition.PLSRegression
#
# Usage:
#   Rscript r_validation_scripts/pls_comparison.R
#
# Requirements:
#   install.packages("pls")
#   install.packages("jsonlite")
# =============================================================================

library(pls)
library(jsonlite)

# Set random seed for reproducibility (MUST match Python)
set.seed(42)

# Output directory
output_dir <- "r_validation_scripts/results/r"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cat("="=paste(rep("=", 80), collapse=""), "\n")
cat("PLS REGRESSION: R pls package\n")
cat(paste(rep("=", 80), collapse=""), "\n\n")

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
cat("Loading data...\n")

train_data <- read.csv("r_validation_scripts/results/pls_regression_train.csv")
test_data <- read.csv("r_validation_scripts/results/pls_regression_test.csv")

# Separate features and target
X_train <- as.matrix(train_data[, -ncol(train_data)])
y_train <- train_data$target

X_test <- as.matrix(test_data[, -ncol(test_data)])
y_test <- test_data$target

cat(sprintf("  Train samples: %d\n", nrow(X_train)))
cat(sprintf("  Test samples: %d\n", nrow(X_test)))
cat(sprintf("  Features: %d\n\n", ncol(X_train)))

# -----------------------------------------------------------------------------
# Train PLS model
# -----------------------------------------------------------------------------
# Fixed hyperparameters (MUST match Python)
n_components <- 10

cat("Training PLS model...\n")
cat(sprintf("  Components: %d\n", n_components))
cat(sprintf("  Scale: FALSE (must match Python)\n\n"))

# Create data frame for pls
train_df <- data.frame(X_train)
train_df$y <- y_train

# Train PLS model using plsr() from pls package
# NOTE: scale=FALSE to match Python's PLSRegression(scale=False)
# validation="none" to avoid automatic CV
pls_model <- plsr(
  y ~ .,
  data = train_df,
  ncomp = n_components,
  scale = FALSE,
  validation = "none"
)

# -----------------------------------------------------------------------------
# Make predictions
# -----------------------------------------------------------------------------
cat("Making predictions...\n")

# Predict on training data
test_df <- data.frame(X_test)
y_train_pred <- predict(pls_model, newdata = train_df, ncomp = n_components)[,,1]
y_test_pred <- predict(pls_model, newdata = test_df, ncomp = n_components)[,,1]

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

cat("\nR PLS Results:\n")
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
    model_type = "PLS",
    n_components = n_components,
    scale = FALSE,
    train_rmse = train_rmse,
    test_rmse = test_rmse,
    train_r2 = train_r2,
    test_r2 = test_r2,
    x_loadings_shape = dim(pls_model$loadings),
    package = "pls",
    r_version = R.version.string
  ),
  random_seed = 42
)

# Save as JSON (for easy comparison with Python)
write_json(results, file.path(output_dir, "pls_regression.json"), auto_unbox = TRUE, pretty = TRUE)

# Save loadings for comparison
loadings_df <- data.frame(pls_model$loadings[, 1:n_components])
colnames(loadings_df) <- paste0("comp_", 1:n_components)
write.csv(loadings_df, file.path(output_dir, "pls_regression_loadings.csv"), row.names = FALSE)

cat(sprintf("  Saved: %s\n", file.path(output_dir, "pls_regression.json")))
cat(sprintf("  Saved: %s\n\n", file.path(output_dir, "pls_regression_loadings.csv")))

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
cat(paste(rep("=", 80), collapse=""), "\n")
cat("PLS REGRESSION COMPLETE\n")
cat(paste(rep("=", 80), collapse=""), "\n\n")

cat("Next step: Compare results\n")
cat("  python r_validation_scripts/compare_results.py --model pls_regression\n\n")

cat("Expected result:\n")
cat("  Python and R predictions should match within tolerance (< 1e-6)\n")
cat("  RMSE and R² should be identical\n")
cat("  Loadings should match (possibly with sign differences)\n\n")
