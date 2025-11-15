# ==============================================================================
# DASP Validation Testing - R Regression Models
# ==============================================================================
#
# This script fits regression models in R with parameters equivalent to DASP,
# then exports predictions and metrics for comparison.
#
# Models tested:
#   - PLS Regression
#   - Ridge Regression
#   - Lasso Regression
#   - ElasticNet Regression
#   - Random Forest
#   - XGBoost
#   - LightGBM (if available)
#
# Usage: Rscript regression_models.R
#
# ==============================================================================

# Suppress package startup messages
suppressPackageStartupMessages({
  library(pls)
  library(glmnet)
  library(randomForest)
  library(xgboost)
  library(prospectr)  # For preprocessing
  library(jsonlite)
  library(dplyr)
})

cat("================================================================================\n")
cat("DASP Validation Testing - R Regression Models\n")
cat("================================================================================\n\n")

# Set random seed for reproducibility
set.seed(42)

# ==============================================================================
# Configuration
# ==============================================================================

# Paths
DATA_DIR <- file.path("..", "r_data", "regression")
RESULTS_DIR <- file.path("..", "results", "r_regression")

# Create results directory
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

# ==============================================================================
# Preprocessing Functions (matching DASP)
# ==============================================================================

apply_snv <- function(X) {
  # Standard Normal Variate (matches DASP implementation)
  # For each row (sample): (x - mean) / std
  t(apply(X, 1, function(row) {
    (row - mean(row)) / sd(row)
  }))
}

apply_sg_derivative <- function(X, deriv = 1, window = 7, polyorder = 2) {
  # Savitzky-Golay derivative (using prospectr)
  # deriv: 0 (smoothing), 1 (first derivative), 2 (second derivative)
  # window: window size (must be odd)
  # polyorder: polynomial order

  if (window %% 2 == 0) {
    stop("Window size must be odd")
  }

  # prospectr uses different parameter names
  savitzkyGolay(X, m = deriv, p = polyorder, w = window)
}

# ==============================================================================
# Data Loading
# ==============================================================================

cat("Loading data...\n")

# Load spectral data
X_train <- as.matrix(read.csv(file.path(DATA_DIR, "X_train.csv")))
X_test <- as.matrix(read.csv(file.path(DATA_DIR, "X_test.csv")))

# Load reference data
y_data <- read.csv(file.path(DATA_DIR, "y_train.csv"))
y_train <- y_data$X.Collagen  # Column name might have X. prefix
y_test_data <- read.csv(file.path(DATA_DIR, "y_test.csv"))
y_test <- y_test_data$X.Collagen

# Load wavelengths
wavelengths <- read.csv(file.path(DATA_DIR, "wavelengths.csv"))$wavelength

cat(sprintf("  Train: %d samples x %d wavelengths\n", nrow(X_train), ncol(X_train)))
cat(sprintf("  Test: %d samples x %d wavelengths\n", nrow(X_test), ncol(X_test)))
cat(sprintf("  Target range: %.1f - %.1f%% collagen\n", min(y_train), max(y_train)))

# ==============================================================================
# Evaluation Metrics
# ==============================================================================

calculate_metrics <- function(y_true, y_pred, model_name) {
  # Calculate RMSE, R², MAE
  rmse <- sqrt(mean((y_true - y_pred)^2))
  mae <- mean(abs(y_true - y_pred))

  # R-squared
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
# Model 1: PLS Regression
# ==============================================================================

cat("\n" , "=", rep("=", 78), "=\n", sep="")
cat("PLS Regression\n")
cat("=", rep("=", 78), "=\n", sep="")

# Test multiple n_components values (matching DASP grid)
pls_n_components <- c(2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50)

# Limit to reasonable values for this dataset
max_comp <- min(nrow(X_train) - 1, ncol(X_train))
pls_n_components <- pls_n_components[pls_n_components <= max_comp]

cat(sprintf("\nTesting n_components: %s\n", paste(pls_n_components, collapse=", ")))

pls_results <- list()

for (ncomp in pls_n_components) {
  cat(sprintf("\n  n_components=%d... ", ncomp))

  # Fit PLS model
  pls_model <- plsr(y_train ~ X_train,
                    ncomp = ncomp,
                    validation = "none",  # We have separate test set
                    method = "oscorespls")

  # Predict on test set
  y_pred <- predict(pls_model, newdata = X_test, ncomp = ncomp)[, 1, 1]

  # Calculate metrics
  metrics <- calculate_metrics(y_test, y_pred, sprintf("PLS_%d", ncomp))
  pls_results[[as.character(ncomp)]] <- metrics

  cat(sprintf("R²=%.3f, RMSE=%.3f\n", metrics$r2, metrics$rmse))
}

# Export PLS results
write_json(pls_results, file.path(RESULTS_DIR, "pls_results.json"),
           pretty = TRUE, auto_unbox = TRUE)

# Find best PLS model
pls_r2_values <- sapply(pls_results, function(x) x$r2)
best_pls_ncomp <- pls_n_components[which.max(pls_r2_values)]
cat(sprintf("\nBest PLS model: n_components=%d (R²=%.3f)\n",
            best_pls_ncomp, max(pls_r2_values)))

# ==============================================================================
# Model 2: Ridge Regression (glmnet with alpha=0)
# ==============================================================================

cat("\n", "=", rep("=", 78), "=\n", sep="")
cat("Ridge Regression\n")
cat("=", rep("=", 78), "=\n", sep="")

# Test multiple alpha (lambda) values (matching DASP grid)
# NOTE: DASP calls regularization strength "alpha", but glmnet calls it "lambda"
#       glmnet's "alpha" controls L1/L2 mix (0 = Ridge, 1 = Lasso)
ridge_lambdas <- c(0.001, 0.01, 0.1, 1.0, 10.0)

cat(sprintf("\nTesting lambda values: %s\n", paste(ridge_lambdas, collapse=", ")))

ridge_results <- list()

for (lambda_val in ridge_lambdas) {
  cat(sprintf("\n  lambda=%.3f... ", lambda_val))

  # Fit Ridge model (alpha=0 for Ridge)
  ridge_model <- glmnet(X_train, y_train,
                       alpha = 0,  # Ridge
                       lambda = lambda_val,
                       standardize = FALSE)  # Data already preprocessed

  # Predict on test set
  y_pred <- predict(ridge_model, newx = X_test, s = lambda_val)[, 1]

  # Calculate metrics
  metrics <- calculate_metrics(y_test, y_pred, sprintf("Ridge_%.3f", lambda_val))
  ridge_results[[as.character(lambda_val)]] <- metrics

  cat(sprintf("R²=%.3f, RMSE=%.3f\n", metrics$r2, metrics$rmse))
}

# Export Ridge results
write_json(ridge_results, file.path(RESULTS_DIR, "ridge_results.json"),
           pretty = TRUE, auto_unbox = TRUE)

# ==============================================================================
# Model 3: Lasso Regression (glmnet with alpha=1)
# ==============================================================================

cat("\n", "=", rep("=", 78), "=\n", sep="")
cat("Lasso Regression\n")
cat("=", rep("=", 78), "=\n", sep="")

# Test multiple lambda values (matching DASP grid)
lasso_lambdas <- c(0.001, 0.01, 0.1, 1.0)

cat(sprintf("\nTesting lambda values: %s\n", paste(lasso_lambdas, collapse=", ")))

lasso_results <- list()

for (lambda_val in lasso_lambdas) {
  cat(sprintf("\n  lambda=%.3f... ", lambda_val))

  # Fit Lasso model (alpha=1 for Lasso)
  lasso_model <- glmnet(X_train, y_train,
                       alpha = 1,  # Lasso
                       lambda = lambda_val,
                       standardize = FALSE)

  # Predict on test set
  y_pred <- predict(lasso_model, newx = X_test, s = lambda_val)[, 1]

  # Calculate metrics
  metrics <- calculate_metrics(y_test, y_pred, sprintf("Lasso_%.3f", lambda_val))
  lasso_results[[as.character(lambda_val)]] <- metrics

  cat(sprintf("R²=%.3f, RMSE=%.3f\n", metrics$r2, metrics$rmse))
}

# Export Lasso results
write_json(lasso_results, file.path(RESULTS_DIR, "lasso_results.json"),
           pretty = TRUE, auto_unbox = TRUE)

# ==============================================================================
# Model 4: ElasticNet Regression
# ==============================================================================

cat("\n", "=", rep("=", 78), "=\n", sep="")
cat("ElasticNet Regression\n")
cat("=", rep("=", 78), "=\n", sep="")

# Test combinations of lambda and l1_ratio (matching DASP grid)
# glmnet's alpha = DASP's l1_ratio (L1/L2 mix)
# glmnet's lambda = DASP's alpha (regularization strength)
elasticnet_lambdas <- c(0.001, 0.01, 0.1, 1.0)
elasticnet_l1_ratios <- c(0.1, 0.3, 0.5, 0.7, 0.9)

cat(sprintf("\nTesting %d combinations\n",
            length(elasticnet_lambdas) * length(elasticnet_l1_ratios)))

elasticnet_results <- list()
counter <- 0

for (lambda_val in elasticnet_lambdas) {
  for (l1_ratio in elasticnet_l1_ratios) {
    counter <- counter + 1
    config_name <- sprintf("ElasticNet_lambda%.3f_l1ratio%.1f", lambda_val, l1_ratio)

    if (counter %% 5 == 0) {
      cat(sprintf("\n  [%d/%d] lambda=%.3f, l1_ratio=%.1f... ",
                  counter, length(elasticnet_lambdas) * length(elasticnet_l1_ratios),
                  lambda_val, l1_ratio))
    }

    # Fit ElasticNet model
    en_model <- glmnet(X_train, y_train,
                      alpha = l1_ratio,  # L1/L2 mix
                      lambda = lambda_val,  # Regularization strength
                      standardize = FALSE)

    # Predict on test set
    y_pred <- predict(en_model, newx = X_test, s = lambda_val)[, 1]

    # Calculate metrics
    metrics <- calculate_metrics(y_test, y_pred, config_name)
    metrics$lambda <- lambda_val
    metrics$l1_ratio <- l1_ratio
    elasticnet_results[[config_name]] <- metrics

    if (counter %% 5 == 0) {
      cat(sprintf("R²=%.3f\n", metrics$r2))
    }
  }
}

# Export ElasticNet results
write_json(elasticnet_results, file.path(RESULTS_DIR, "elasticnet_results.json"),
           pretty = TRUE, auto_unbox = TRUE)

# ==============================================================================
# Model 5: Random Forest
# ==============================================================================

cat("\n", "=", rep("=", 78), "=\n", sep="")
cat("Random Forest Regression\n")
cat("=", rep("=", 78), "=\n", sep="")

# Test combinations matching DASP grid
rf_n_estimators <- c(100, 200, 500)
rf_max_depth <- c(15, 30)  # NULL (unlimited) tested separately

cat(sprintf("\nTesting Random Forest configurations...\n"))

rf_results <- list()

for (ntree in rf_n_estimators) {
  for (max_depth in c(rf_max_depth, -1)) {  # -1 represents NULL (unlimited)

    depth_label <- if (max_depth == -1) "None" else as.character(max_depth)
    config_name <- sprintf("RF_ntree%d_depth%s", ntree, depth_label)

    cat(sprintf("\n  ntree=%d, max_depth=%s... ", ntree, depth_label))

    # Random Forest parameters
    # Note: R randomForest doesn't have max_depth parameter directly
    # We'll use maxnodes as approximation: maxnodes ≈ 2^max_depth
    if (max_depth == -1) {
      rf_model <- randomForest(X_train, y_train,
                               ntree = ntree,
                               mtry = sqrt(ncol(X_train)),  # Default for regression
                               nodesize = 1,  # Matches DASP min_samples_leaf=1
                               importance = TRUE)
    } else {
      max_nodes <- 2^max_depth
      rf_model <- randomForest(X_train, y_train,
                               ntree = ntree,
                               mtry = sqrt(ncol(X_train)),
                               nodesize = 1,
                               maxnodes = max_nodes,
                               importance = TRUE)
    }

    # Predict on test set
    y_pred <- predict(rf_model, newdata = X_test)

    # Calculate metrics
    metrics <- calculate_metrics(y_test, y_pred, config_name)
    metrics$ntree <- ntree
    metrics$max_depth <- if (max_depth == -1) NULL else max_depth
    rf_results[[config_name]] <- metrics

    cat(sprintf("R²=%.3f, RMSE=%.3f\n", metrics$r2, metrics$rmse))
  }
}

# Export RF results
write_json(rf_results, file.path(RESULTS_DIR, "rf_results.json"),
           pretty = TRUE, auto_unbox = TRUE)

# ==============================================================================
# Model 6: XGBoost
# ==============================================================================

cat("\n", "=", rep("=", 78), "=\n", sep="")
cat("XGBoost Regression\n")
cat("=", rep("=", 78), "=\n", sep="")

# Test combinations matching DASP grid
xgb_n_estimators <- c(100, 200)
xgb_learning_rate <- c(0.05, 0.1)
xgb_max_depth <- c(3, 6)

cat(sprintf("\nTesting %d XGBoost configurations...\n",
            length(xgb_n_estimators) * length(xgb_learning_rate) * length(xgb_max_depth)))

xgb_results <- list()

# Prepare data for xgboost
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest <- xgb.DMatrix(data = X_test, label = y_test)

for (nrounds in xgb_n_estimators) {
  for (eta in xgb_learning_rate) {
    for (max_depth in xgb_max_depth) {

      config_name <- sprintf("XGB_n%d_lr%.2f_depth%d", nrounds, eta, max_depth)

      cat(sprintf("\n  nrounds=%d, eta=%.2f, max_depth=%d... ", nrounds, eta, max_depth))

      # Train XGBoost model
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

      # Predict on test set
      y_pred <- predict(xgb_model, dtest)

      # Calculate metrics
      metrics <- calculate_metrics(y_test, y_pred, config_name)
      metrics$nrounds <- nrounds
      metrics$eta <- eta
      metrics$max_depth <- max_depth
      xgb_results[[config_name]] <- metrics

      cat(sprintf("R²=%.3f, RMSE=%.3f\n", metrics$r2, metrics$rmse))
    }
  }
}

# Export XGBoost results
write_json(xgb_results, file.path(RESULTS_DIR, "xgboost_results.json"),
           pretty = TRUE, auto_unbox = TRUE)

# ==============================================================================
# Summary
# ==============================================================================

cat("\n", "=", rep("=", 78), "=\n", sep="")
cat("Summary\n")
cat("=", rep("=", 78), "=\n", sep="")

# Combine all results
all_results <- c(pls_results, ridge_results, lasso_results,
                elasticnet_results, rf_results, xgb_results)

# Find best model overall
all_r2 <- sapply(all_results, function(x) x$r2)
best_idx <- which.max(all_r2)
best_model <- all_results[[best_idx]]

cat(sprintf("\nTotal models tested: %d\n", length(all_results)))
cat(sprintf("\nBest model: %s\n", best_model$model))
cat(sprintf("  R² = %.4f\n", best_model$r2))
cat(sprintf("  RMSE = %.4f\n", best_model$rmse))
cat(sprintf("  MAE = %.4f\n", best_model$mae))

# Export combined results
write_json(all_results, file.path(RESULTS_DIR, "all_models_summary.json"),
           pretty = TRUE, auto_unbox = TRUE)

cat(sprintf("\nResults saved to: %s\n", RESULTS_DIR))

cat("\n", "=", rep("=", 78), "=\n", sep="")
cat("Complete!\n")
cat("=", rep("=", 78), "=\n", sep="")
