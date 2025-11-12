#!/usr/bin/env Rscript
# =============================================================================
# XGBoost Regression Validation: R vs Python
# =============================================================================
#
# This script trains XGBoost models using R's 'xgboost' package
# and compares results with Python's xgboost.XGBRegressor
#
# Usage:
#   Rscript r_validation_scripts/xgboost_comparison.R
#
# Requirements:
#   install.packages("xgboost")
#   install.packages("jsonlite")
# =============================================================================

library(xgboost)
library(jsonlite)

# Set random seed for reproducibility (MUST match Python)
set.seed(42)

# Output directory
output_dir <- "r_validation_scripts/results/r"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cat(paste(rep("=", 80), collapse=""), "\n")
cat("XGBOOST REGRESSION: R xgboost package\n")
cat(paste(rep("=", 80), collapse=""), "\n\n")

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
cat("Loading data...\n")

train_data <- read.csv("r_validation_scripts/results/xgb_regression_train.csv")
test_data <- read.csv("r_validation_scripts/results/xgb_regression_test.csv")

# Separate features and target
X_train <- as.matrix(train_data[, -ncol(train_data)])
y_train <- train_data$target

X_test <- as.matrix(test_data[, -ncol(test_data)])
y_test <- test_data$target

cat(sprintf("  Train samples: %d\n", nrow(X_train)))
cat(sprintf("  Test samples: %d\n", nrow(X_test)))
cat(sprintf("  Features: %d\n\n", ncol(X_train)))

# Create DMatrix objects (XGBoost's optimized data structure)
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest <- xgb.DMatrix(data = X_test, label = y_test)

# -----------------------------------------------------------------------------
# Train XGBoost model
# -----------------------------------------------------------------------------
# Fixed hyperparameters (MUST match Python)
# NOTE: Parameter naming differs between Python and R!
#   Python (sklearn API) -> R (native API)
#   n_estimators -> nrounds
#   learning_rate -> eta
#   max_depth -> max_depth (same)
#   subsample -> subsample (same)
#   colsample_bytree -> colsample_bytree (same)
#   reg_alpha -> alpha
#   reg_lambda -> lambda

params <- list(
  objective = "reg:squarederror",  # Regression task
  eta = 0.1,                       # learning_rate
  max_depth = 6,                   # max_depth
  subsample = 0.8,                 # subsample
  colsample_bytree = 0.8,          # colsample_bytree
  alpha = 0.1,                     # reg_alpha (L1)
  lambda = 1.0,                    # reg_lambda (L2)
  seed = 42                        # random_state
)

nrounds <- 100  # n_estimators

cat("Training XGBoost model...\n")
cat(sprintf("  Rounds (n_estimators): %d\n", nrounds))
cat(sprintf("  Learning rate (eta): %.2f\n", params$eta))
cat(sprintf("  Max depth: %d\n", params$max_depth))
cat(sprintf("  Subsample: %.2f\n", params$subsample))
cat(sprintf("  Colsample bytree: %.2f\n", params$colsample_bytree))
cat(sprintf("  Alpha (L1): %.2f\n", params$alpha))
cat(sprintf("  Lambda (L2): %.2f\n\n", params$lambda))

# Train model
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = nrounds,
  verbose = 0  # Suppress training output
)

cat("Model trained successfully!\n\n")

# -----------------------------------------------------------------------------
# Make predictions
# -----------------------------------------------------------------------------
cat("Making predictions...\n")

y_train_pred <- predict(xgb_model, dtrain)
y_test_pred <- predict(xgb_model, dtest)

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

cat("\nR XGBoost Results:\n")
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
    model_type = "XGBoost",
    n_estimators = nrounds,
    learning_rate = params$eta,
    max_depth = params$max_depth,
    subsample = params$subsample,
    colsample_bytree = params$colsample_bytree,
    reg_alpha = params$alpha,
    reg_lambda = params$lambda,
    train_rmse = train_rmse,
    test_rmse = test_rmse,
    train_r2 = train_r2,
    test_r2 = test_r2,
    package = "xgboost",
    r_version = R.version.string
  ),
  random_seed = 42
)

# Save as JSON
write_json(results, file.path(output_dir, "xgb_regression.json"), auto_unbox = TRUE, pretty = TRUE)

# Save feature importances
# XGBoost provides several importance types: "gain", "weight", "cover"
# We'll use "gain" which is similar to sklearn's default
importance_matrix <- xgb.importance(model = xgb_model)

# Create importance dataframe matching Python format
importance_df <- data.frame(
  feature = paste0("feature_", 0:(ncol(X_train) - 1)),
  importance = 0  # Initialize with zeros
)

# Fill in the importance values from XGBoost
# Note: XGBoost only reports importance for features actually used
for (i in 1:nrow(importance_matrix)) {
  feature_name <- importance_matrix$Feature[i]
  feature_idx <- as.integer(sub("V", "", feature_name))
  importance_df$importance[feature_idx] <- importance_matrix$Gain[i]
}

# Normalize to sum to 1 (to match sklearn)
importance_df$importance <- importance_df$importance / sum(importance_df$importance)

write.csv(importance_df, file.path(output_dir, "xgb_regression_importance.csv"), row.names = FALSE)

cat(sprintf("  Saved: %s\n", file.path(output_dir, "xgb_regression.json")))
cat(sprintf("  Saved: %s\n\n", file.path(output_dir, "xgb_regression_importance.csv")))

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
cat(paste(rep("=", 80), collapse=""), "\n")
cat("XGBOOST REGRESSION COMPLETE\n")
cat(paste(rep("=", 80), collapse=""), "\n\n")

cat("Next step: Compare results\n")
cat("  python r_validation_scripts/compare_results.py --model xgb_regression\n\n")

cat("Expected result:\n")
cat("  Python and R predictions should match very closely (< 1e-4)\n")
cat("  XGBoost should be highly consistent across R and Python\n")
cat("  RMSE and R² should be nearly identical\n")
cat("  Feature importances should be very similar (correlation > 0.99)\n\n")

cat("NOTE: XGBoost is designed to be consistent across platforms,\n")
cat("so results should match much more closely than Random Forest.\n\n")
