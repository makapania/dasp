#!/usr/bin/env Rscript
# =============================================================================
# glmnet Regression Validation: R vs Python
# =============================================================================
#
# This script trains Ridge, Lasso, and ElasticNet models using R's 'glmnet'
# package and compares results with Python's sklearn.linear_model
#
# Usage:
#   Rscript r_validation_scripts/glmnet_comparison.R
#
# Requirements:
#   install.packages("glmnet")
#   install.packages("jsonlite")
#
# Parameter naming differences:
#   Python sklearn           | R glmnet
#   -----------------------|------------------------
#   Ridge(alpha=X)         | glmnet(alpha=0, lambda=X)
#   Lasso(alpha=X)         | glmnet(alpha=1, lambda=X)
#   ElasticNet(alpha=X,    | glmnet(alpha=L1_ratio,
#              l1_ratio=Y) |         lambda=X)
#
# NOTE: "alpha" means different things in sklearn vs glmnet!
#   - sklearn alpha = regularization strength (lambda in glmnet)
#   - glmnet alpha = mixing parameter (l1_ratio in sklearn)
# =============================================================================

library(glmnet)
library(jsonlite)

# Set random seed for reproducibility (MUST match Python)
set.seed(42)

# Output directory
output_dir <- "r_validation_scripts/results/r"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cat(paste(rep("=", 80), collapse=""), "\n")
cat("GLMNET REGRESSION: R glmnet package\n")
cat(paste(rep("=", 80), collapse=""), "\n\n")

# Helper functions
rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

r2 <- function(actual, predicted) {
  ss_res <- sum((actual - predicted)^2)
  ss_tot <- sum((actual - mean(actual))^2)
  1 - (ss_res / ss_tot)
}

# =============================================================================
# 1. RIDGE REGRESSION (alpha=0 in glmnet)
# =============================================================================

cat("\n", paste(rep("-", 80), collapse=""), "\n")
cat("1. RIDGE REGRESSION (glmnet alpha=0)\n")
cat(paste(rep("-", 80), collapse=""), "\n\n")

# Load data
train_data <- read.csv("r_validation_scripts/results/ridge_regression_train.csv")
test_data <- read.csv("r_validation_scripts/results/ridge_regression_test.csv")

X_train <- as.matrix(train_data[, -ncol(train_data)])
y_train <- train_data$target
X_test <- as.matrix(test_data[, -ncol(test_data)])
y_test <- test_data$target

cat(sprintf("  Train samples: %d\n", nrow(X_train)))
cat(sprintf("  Test samples: %d\n", nrow(X_test)))
cat(sprintf("  Features: %d\n\n", ncol(X_train)))

# Fixed hyperparameters (MUST match Python)
lambda_value <- 1.0  # sklearn: alpha=1.0
alpha_glmnet <- 0    # Ridge = alpha=0 in glmnet

cat("Training Ridge model...\n")
cat(sprintf("  Lambda (sklearn alpha): %.2f\n", lambda_value))
cat(sprintf("  Alpha (mixing): %.1f (Ridge)\n\n", alpha_glmnet))

# Train glmnet model
# standardize=FALSE to match sklearn's default behavior
ridge_model <- glmnet(
  x = X_train,
  y = y_train,
  alpha = alpha_glmnet,
  lambda = lambda_value,
  standardize = FALSE,  # Match sklearn default
  intercept = TRUE
)

# Predictions
y_train_pred <- predict(ridge_model, newx = X_train, s = lambda_value)[, 1]
y_test_pred <- predict(ridge_model, newx = X_test, s = lambda_value)[, 1]

# Metrics
train_rmse <- rmse(y_train, y_train_pred)
test_rmse <- rmse(y_test, y_test_pred)
train_r2 <- r2(y_train, y_train_pred)
test_r2 <- r2(y_test, y_test_pred)

cat("\nR Ridge Results:\n")
cat(sprintf("  Train RMSE: %.6f\n", train_rmse))
cat(sprintf("  Test RMSE:  %.6f\n", test_rmse))
cat(sprintf("  Train R²:   %.6f\n", train_r2))
cat(sprintf("  Test R²:    %.6f\n\n", test_r2))

# Get coefficients
coefs <- coef(ridge_model, s = lambda_value)
coefs_vector <- as.vector(coefs)[-1]  # Exclude intercept

# Save results
results_ridge <- list(
  predictions = list(
    train = as.vector(y_train_pred),
    test = as.vector(y_test_pred)
  ),
  model_info = list(
    model_type = "Ridge",
    alpha = lambda_value,  # sklearn naming
    lambda_glmnet = lambda_value,
    alpha_glmnet = alpha_glmnet,
    train_rmse = train_rmse,
    test_rmse = test_rmse,
    train_r2 = train_r2,
    test_r2 = test_r2,
    n_nonzero_coefs = sum(abs(coefs_vector) > 1e-10),
    package = "glmnet",
    r_version = R.version.string
  ),
  random_seed = 42
)

write_json(results_ridge, file.path(output_dir, "ridge_regression.json"), auto_unbox = TRUE, pretty = TRUE)

# Save coefficients
coef_df <- data.frame(
  feature = paste0("feature_", 0:(length(coefs_vector) - 1)),
  coefficient = coefs_vector
)
write.csv(coef_df, file.path(output_dir, "ridge_regression_coefs.csv"), row.names = FALSE)

cat(sprintf("  Saved: %s\n", file.path(output_dir, "ridge_regression.json")))
cat(sprintf("  Saved: %s\n", file.path(output_dir, "ridge_regression_coefs.csv")))

# =============================================================================
# 2. LASSO REGRESSION (alpha=1 in glmnet)
# =============================================================================

cat("\n", paste(rep("-", 80), collapse=""), "\n")
cat("2. LASSO REGRESSION (glmnet alpha=1)\n")
cat(paste(rep("-", 80), collapse=""), "\n\n")

# Load data
train_data <- read.csv("r_validation_scripts/results/lasso_regression_train.csv")
test_data <- read.csv("r_validation_scripts/results/lasso_regression_test.csv")

X_train <- as.matrix(train_data[, -ncol(train_data)])
y_train <- train_data$target
X_test <- as.matrix(test_data[, -ncol(test_data)])
y_test <- test_data$target

# Fixed hyperparameters (MUST match Python)
lambda_value <- 0.1  # sklearn: alpha=0.1
alpha_glmnet <- 1    # Lasso = alpha=1 in glmnet

cat("Training Lasso model...\n")
cat(sprintf("  Lambda (sklearn alpha): %.2f\n", lambda_value))
cat(sprintf("  Alpha (mixing): %.1f (Lasso)\n\n", alpha_glmnet))

# Train glmnet model
lasso_model <- glmnet(
  x = X_train,
  y = y_train,
  alpha = alpha_glmnet,
  lambda = lambda_value,
  standardize = FALSE,
  intercept = TRUE
)

# Predictions
y_train_pred <- predict(lasso_model, newx = X_train, s = lambda_value)[, 1]
y_test_pred <- predict(lasso_model, newx = X_test, s = lambda_value)[, 1]

# Metrics
train_rmse <- rmse(y_train, y_train_pred)
test_rmse <- rmse(y_test, y_test_pred)
train_r2 <- r2(y_train, y_train_pred)
test_r2 <- r2(y_test, y_test_pred)

cat("\nR Lasso Results:\n")
cat(sprintf("  Train RMSE: %.6f\n", train_rmse))
cat(sprintf("  Test RMSE:  %.6f\n", test_rmse))
cat(sprintf("  Train R²:   %.6f\n", train_r2))
cat(sprintf("  Test R²:    %.6f\n", test_r2))

# Get coefficients
coefs <- coef(lasso_model, s = lambda_value)
coefs_vector <- as.vector(coefs)[-1]  # Exclude intercept
n_nonzero <- sum(abs(coefs_vector) > 1e-10)

cat(sprintf("  Non-zero coefficients: %d/%d\n\n", n_nonzero, length(coefs_vector)))

# Save results
results_lasso <- list(
  predictions = list(
    train = as.vector(y_train_pred),
    test = as.vector(y_test_pred)
  ),
  model_info = list(
    model_type = "Lasso",
    alpha = lambda_value,  # sklearn naming
    lambda_glmnet = lambda_value,
    alpha_glmnet = alpha_glmnet,
    train_rmse = train_rmse,
    test_rmse = test_rmse,
    train_r2 = train_r2,
    test_r2 = test_r2,
    n_nonzero_coefs = n_nonzero,
    package = "glmnet",
    r_version = R.version.string
  ),
  random_seed = 42
)

write_json(results_lasso, file.path(output_dir, "lasso_regression.json"), auto_unbox = TRUE, pretty = TRUE)

# Save coefficients
coef_df <- data.frame(
  feature = paste0("feature_", 0:(length(coefs_vector) - 1)),
  coefficient = coefs_vector
)
write.csv(coef_df, file.path(output_dir, "lasso_regression_coefs.csv"), row.names = FALSE)

cat(sprintf("  Saved: %s\n", file.path(output_dir, "lasso_regression.json")))
cat(sprintf("  Saved: %s\n", file.path(output_dir, "lasso_regression_coefs.csv")))

# =============================================================================
# 3. ELASTICNET REGRESSION (0 < alpha < 1 in glmnet)
# =============================================================================

cat("\n", paste(rep("-", 80), collapse=""), "\n")
cat("3. ELASTICNET REGRESSION (glmnet 0 < alpha < 1)\n")
cat(paste(rep("-", 80), collapse=""), "\n\n")

# Load data
train_data <- read.csv("r_validation_scripts/results/elasticnet_regression_train.csv")
test_data <- read.csv("r_validation_scripts/results/elasticnet_regression_test.csv")

X_train <- as.matrix(train_data[, -ncol(train_data)])
y_train <- train_data$target
X_test <- as.matrix(test_data[, -ncol(test_data)])
y_test <- test_data$target

# Fixed hyperparameters (MUST match Python)
lambda_value <- 0.1  # sklearn: alpha=0.1
alpha_glmnet <- 0.5  # sklearn: l1_ratio=0.5

cat("Training ElasticNet model...\n")
cat(sprintf("  Lambda (sklearn alpha): %.2f\n", lambda_value))
cat(sprintf("  Alpha (sklearn l1_ratio): %.1f\n\n", alpha_glmnet))

# Train glmnet model
elasticnet_model <- glmnet(
  x = X_train,
  y = y_train,
  alpha = alpha_glmnet,
  lambda = lambda_value,
  standardize = FALSE,
  intercept = TRUE
)

# Predictions
y_train_pred <- predict(elasticnet_model, newx = X_train, s = lambda_value)[, 1]
y_test_pred <- predict(elasticnet_model, newx = X_test, s = lambda_value)[, 1]

# Metrics
train_rmse <- rmse(y_train, y_train_pred)
test_rmse <- rmse(y_test, y_test_pred)
train_r2 <- r2(y_train, y_train_pred)
test_r2 <- r2(y_test, y_test_pred)

cat("\nR ElasticNet Results:\n")
cat(sprintf("  Train RMSE: %.6f\n", train_rmse))
cat(sprintf("  Test RMSE:  %.6f\n", test_rmse))
cat(sprintf("  Train R²:   %.6f\n", train_r2))
cat(sprintf("  Test R²:    %.6f\n", test_r2))

# Get coefficients
coefs <- coef(elasticnet_model, s = lambda_value)
coefs_vector <- as.vector(coefs)[-1]  # Exclude intercept
n_nonzero <- sum(abs(coefs_vector) > 1e-10)

cat(sprintf("  Non-zero coefficients: %d/%d\n\n", n_nonzero, length(coefs_vector)))

# Save results
results_elasticnet <- list(
  predictions = list(
    train = as.vector(y_train_pred),
    test = as.vector(y_test_pred)
  ),
  model_info = list(
    model_type = "ElasticNet",
    alpha = lambda_value,  # sklearn naming
    l1_ratio = alpha_glmnet,  # sklearn naming
    lambda_glmnet = lambda_value,
    alpha_glmnet = alpha_glmnet,
    train_rmse = train_rmse,
    test_rmse = test_rmse,
    train_r2 = train_r2,
    test_r2 = test_r2,
    n_nonzero_coefs = n_nonzero,
    package = "glmnet",
    r_version = R.version.string
  ),
  random_seed = 42
)

write_json(results_elasticnet, file.path(output_dir, "elasticnet_regression.json"), auto_unbox = TRUE, pretty = TRUE)

# Save coefficients
coef_df <- data.frame(
  feature = paste0("feature_", 0:(length(coefs_vector) - 1)),
  coefficient = coefs_vector
)
write.csv(coef_df, file.path(output_dir, "elasticnet_regression_coefs.csv"), row.names = FALSE)

cat(sprintf("  Saved: %s\n", file.path(output_dir, "elasticnet_regression.json")))
cat(sprintf("  Saved: %s\n", file.path(output_dir, "elasticnet_regression_coefs.csv")))

# =============================================================================
# Summary
# =============================================================================

cat("\n", paste(rep("=", 80), collapse=""), "\n")
cat("GLMNET REGRESSION COMPLETE\n")
cat(paste(rep("=", 80), collapse=""), "\n\n")

cat("Models trained:\n")
cat("  1. Ridge (alpha=0)\n")
cat("  2. Lasso (alpha=1)\n")
cat("  3. ElasticNet (alpha=0.5)\n\n")

cat("Next step: Compare results\n")
cat("  python r_validation_scripts/compare_results.py --model ridge_regression\n")
cat("  python r_validation_scripts/compare_results.py --model lasso_regression\n")
cat("  python r_validation_scripts/compare_results.py --model elasticnet_regression\n\n")

cat("Expected result:\n")
cat("  Python and R predictions should match very closely (< 1e-6)\n")
cat("  Coefficients should be nearly identical\n")
cat("  Some differences may occur due to:\n")
cat("    - Different convergence criteria\n")
cat("    - Different coordinate descent implementations\n")
cat("    - Numerical precision differences\n\n")

cat("PARAMETER NAMING REMINDER:\n")
cat("  sklearn alpha = glmnet lambda (regularization strength)\n")
cat("  sklearn l1_ratio = glmnet alpha (L1/L2 mixing parameter)\n\n")
