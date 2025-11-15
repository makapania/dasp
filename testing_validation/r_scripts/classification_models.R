# ==============================================================================
# DASP Validation Testing - R Classification Models
# ==============================================================================
#
# This script fits classification models in R with parameters equivalent to DASP,
# then exports predictions and metrics for comparison.
#
# Models tested:
#   - PLS-DA (PLS Discriminant Analysis)
#   - Random Forest Classification
#   - XGBoost Classification
#
# Tasks tested:
#   - Binary classification (High vs. Low collagen)
#   - 4-class classification (categories A, F, G, H)
#
# Usage: Rscript classification_models.R
#
# ==============================================================================

# Suppress package startup messages
suppressPackageStartupMessages({
  library(pls)
  library(randomForest)
  library(xgboost)
  library(caret)  # For PLS-DA
  library(prospectr)  # For preprocessing
  library(jsonlite)
  library(dplyr)
  library(pROC)  # For ROC-AUC
})

cat("================================================================================\n")
cat("DASP Validation Testing - R Classification Models\n")
cat("================================================================================\n\n")

# Set random seed for reproducibility
set.seed(42)

# ==============================================================================
# Configuration
# ==============================================================================

# Paths
R_DATA_DIR <- file.path("..", "r_data")
RESULTS_DIR <- file.path("..", "results", "r_classification")

# Create results directory
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)

# ==============================================================================
# Evaluation Metrics for Classification
# ==============================================================================

calculate_classification_metrics <- function(y_true, y_pred, y_pred_proba = NULL, model_name, task_type = "binary") {
  # Calculate accuracy, precision, recall, F1

  # Confusion matrix
  cm <- table(Predicted = y_pred, Actual = y_true)

  # Accuracy
  accuracy <- sum(diag(cm)) / sum(cm)

  # For binary classification, calculate additional metrics
  if (task_type == "binary" && !is.null(y_pred_proba)) {
    # ROC-AUC (assuming y_true is 0/1)
    roc_obj <- tryCatch({
      roc(y_true, y_pred_proba, quiet = TRUE)
    }, error = function(e) NULL)

    auc <- if (!is.null(roc_obj)) as.numeric(auc(roc_obj)) else NA

    # Precision, Recall, F1 for binary
    tp <- cm[2, 2]  # True Positive
    fp <- cm[2, 1]  # False Positive
    fn <- cm[1, 2]  # False Negative
    tn <- cm[1, 1]  # True Negative

    precision <- tp / (tp + fp)
    recall <- tp / (tp + fn)
    f1 <- 2 * (precision * recall) / (precision + recall)

    list(
      model = model_name,
      task = task_type,
      accuracy = accuracy,
      auc = auc,
      precision = precision,
      recall = recall,
      f1 = f1,
      n_samples = length(y_true)
    )
  } else {
    # Multi-class: just accuracy and macro-averaged F1
    # Calculate per-class precision, recall, F1
    classes <- unique(c(as.character(y_true), as.character(y_pred)))
    n_classes <- length(classes)

    f1_scores <- numeric(n_classes)
    for (i in seq_along(classes)) {
      cls <- classes[i]
      tp <- sum(y_true == cls & y_pred == cls)
      fp <- sum(y_true != cls & y_pred == cls)
      fn <- sum(y_true == cls & y_pred != cls)

      if (tp + fp > 0) {
        precision <- tp / (tp + fp)
      } else {
        precision <- 0
      }

      if (tp + fn > 0) {
        recall <- tp / (tp + fn)
      } else {
        recall <- 0
      }

      if (precision + recall > 0) {
        f1_scores[i] <- 2 * (precision * recall) / (precision + recall)
      } else {
        f1_scores[i] <- 0
      }
    }

    macro_f1 <- mean(f1_scores)

    list(
      model = model_name,
      task = task_type,
      accuracy = accuracy,
      macro_f1 = macro_f1,
      n_samples = length(y_true),
      n_classes = n_classes
    )
  }
}

# ==============================================================================
# Function to process each task
# ==============================================================================

process_classification_task <- function(task_name) {
  cat("\n", "=", rep("=", 78), "=\n", sep="")
  cat(sprintf("Processing %s Classification Task\n", toupper(task_name)))
  cat("=", rep("=", 78), "=\n", sep="")

  # Paths
  task_dir <- file.path(R_DATA_DIR, task_name)
  task_results_dir <- file.path(RESULTS_DIR, task_name)
  dir.create(task_results_dir, showWarnings = FALSE, recursive = TRUE)

  # Load data
  cat("\nLoading data...\n")
  X_train <- as.matrix(read.csv(file.path(task_dir, "X_train.csv")))
  X_test <- as.matrix(read.csv(file.path(task_dir, "X_test.csv")))

  y_train_data <- read.csv(file.path(task_dir, "y_train.csv"))
  y_test_data <- read.csv(file.path(task_dir, "y_test.csv"))

  # Extract class labels
  if (task_name == "binary") {
    # Binary classification
    y_train <- as.factor(y_train_data$Binary_Class)
    y_test <- as.factor(y_test_data$Binary_Class)
    task_type <- "binary"
  } else {
    # Multi-class classification
    y_train <- as.factor(y_train_data$Category)
    y_test <- as.factor(y_test_data$Category)
    task_type <- "multiclass"
  }

  cat(sprintf("  Train: %d samples x %d features\n", nrow(X_train), ncol(X_train)))
  cat(sprintf("  Test: %d samples\n", nrow(X_test)))
  cat(sprintf("  Classes: %s\n", paste(levels(y_train), collapse=", ")))

  results <- list()

  # ==========================================================================
  # Model 1: PLS-DA
  # ==========================================================================

  cat("\n", "-", rep("-", 78), "-\n", sep="")
  cat("PLS-DA (PLS Discriminant Analysis)\n")
  cat("-", rep("-", 78), "-\n", sep="")

  # Test multiple n_components
  plsda_n_components <- c(2, 4, 6, 8, 10, 12, 16, 20)
  max_comp <- min(nrow(X_train) - 1, ncol(X_train), 30)
  plsda_n_components <- plsda_n_components[plsda_n_components <= max_comp]

  cat(sprintf("\nTesting n_components: %s\n", paste(plsda_n_components, collapse=", ")))

  for (ncomp in plsda_n_components) {
    cat(sprintf("\n  n_components=%d... ", ncomp))

    tryCatch({
      # Fit PLS-DA using plsr with dummy variables
      # Convert factors to numeric for PLS
      y_train_numeric <- as.numeric(y_train) - 1

      pls_model <- plsr(y_train_numeric ~ X_train,
                       ncomp = ncomp,
                       validation = "none",
                       method = "oscorespls")

      # Predict on test set
      y_pred_scores <- predict(pls_model, newdata = X_test, ncomp = ncomp)[, 1, 1]

      # For binary: threshold at 0.5
      # For multiclass: use max score (simplified)
      if (task_type == "binary") {
        y_pred <- factor(ifelse(y_pred_scores > 0.5, 1, 0), levels = levels(y_test))
        y_pred_proba <- y_pred_scores
      } else {
        # For multiclass, discretize predictions
        y_pred <- factor(round(y_pred_scores), levels = levels(y_test))
        y_pred_proba <- NULL
      }

      # Calculate metrics
      metrics <- calculate_classification_metrics(
        y_test, y_pred, y_pred_proba,
        sprintf("PLSDA_%d", ncomp),
        task_type
      )

      results[[sprintf("PLSDA_%d", ncomp)]] <- metrics

      if (task_type == "binary") {
        cat(sprintf("Accuracy=%.3f, AUC=%.3f\n", metrics$accuracy, metrics$auc))
      } else {
        cat(sprintf("Accuracy=%.3f, F1=%.3f\n", metrics$accuracy, metrics$macro_f1))
      }
    }, error = function(e) {
      cat(sprintf("ERROR: %s\n", e$message))
    })
  }

  # ==========================================================================
  # Model 2: Random Forest Classification
  # ==========================================================================

  cat("\n", "-", rep("-", 78), "-\n", sep="")
  cat("Random Forest Classification\n")
  cat("-", rep("-", 78), "-\n", sep="")

  # Test configurations
  rf_n_estimators <- c(100, 200)
  rf_max_depth <- c(15, 30)

  for (ntree in rf_n_estimators) {
    for (max_depth in c(rf_max_depth, -1)) {

      depth_label <- if (max_depth == -1) "None" else as.character(max_depth)
      config_name <- sprintf("RF_ntree%d_depth%s", ntree, depth_label)

      cat(sprintf("\n  ntree=%d, max_depth=%s... ", ntree, depth_label))

      tryCatch({
        # Fit Random Forest
        if (max_depth == -1) {
          rf_model <- randomForest(X_train, y_train,
                                   ntree = ntree,
                                   mtry = sqrt(ncol(X_train)),
                                   nodesize = 1,
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
        y_pred <- predict(rf_model, newdata = X_test, type = "class")

        # Get probabilities for AUC (binary only)
        if (task_type == "binary") {
          y_pred_proba <- predict(rf_model, newdata = X_test, type = "prob")[, 2]
        } else {
          y_pred_proba <- NULL
        }

        # Calculate metrics
        metrics <- calculate_classification_metrics(
          y_test, y_pred, y_pred_proba,
          config_name,
          task_type
        )

        results[[config_name]] <- metrics

        if (task_type == "binary") {
          cat(sprintf("Accuracy=%.3f, AUC=%.3f\n", metrics$accuracy, metrics$auc))
        } else {
          cat(sprintf("Accuracy=%.3f, F1=%.3f\n", metrics$accuracy, metrics$macro_f1))
        }
      }, error = function(e) {
        cat(sprintf("ERROR: %s\n", e$message))
      })
    }
  }

  # ==========================================================================
  # Model 3: XGBoost Classification
  # ==========================================================================

  cat("\n", "-", rep("-", 78), "-\n", sep="")
  cat("XGBoost Classification\n")
  cat("-", rep("-", 78), "-\n", sep="")

  # Test configurations
  xgb_n_estimators <- c(100, 200)
  xgb_learning_rate <- c(0.05, 0.1)
  xgb_max_depth <- c(3, 6)

  # Prepare data for xgboost
  # Convert factor labels to 0-indexed integers
  y_train_numeric <- as.numeric(y_train) - 1
  y_test_numeric <- as.numeric(y_test) - 1

  dtrain <- xgb.DMatrix(data = X_train, label = y_train_numeric)
  dtest <- xgb.DMatrix(data = X_test, label = y_test_numeric)

  # Determine objective
  num_classes <- length(levels(y_train))
  if (num_classes == 2) {
    objective <- "binary:logistic"
  } else {
    objective <- "multi:softprob"
  }

  for (nrounds in xgb_n_estimators) {
    for (eta in xgb_learning_rate) {
      for (max_depth in xgb_max_depth) {

        config_name <- sprintf("XGB_n%d_lr%.2f_depth%d", nrounds, eta, max_depth)

        cat(sprintf("\n  nrounds=%d, eta=%.2f, max_depth=%d... ", nrounds, eta, max_depth))

        tryCatch({
          # Train XGBoost model
          if (num_classes == 2) {
            xgb_model <- xgb.train(
              data = dtrain,
              params = list(
                objective = "binary:logistic",
                eta = eta,
                max_depth = max_depth,
                subsample = 0.8,
                colsample_bytree = 0.8
              ),
              nrounds = nrounds,
              verbose = 0
            )

            # Predict probabilities
            y_pred_proba <- predict(xgb_model, dtest)
            y_pred_numeric <- ifelse(y_pred_proba > 0.5, 1, 0)
          } else {
            xgb_model <- xgb.train(
              data = dtrain,
              params = list(
                objective = "multi:softprob",
                num_class = num_classes,
                eta = eta,
                max_depth = max_depth,
                subsample = 0.8,
                colsample_bytree = 0.8
              ),
              nrounds = nrounds,
              verbose = 0
            )

            # Predict probabilities (returns matrix for multiclass)
            y_pred_proba_matrix <- matrix(predict(xgb_model, dtest), ncol = num_classes, byrow = TRUE)
            y_pred_numeric <- max.col(y_pred_proba_matrix) - 1
            y_pred_proba <- NULL
          }

          # Convert back to factor
          y_pred <- factor(y_pred_numeric, levels = 0:(num_classes - 1), labels = levels(y_test))

          # Calculate metrics
          metrics <- calculate_classification_metrics(
            y_test, y_pred, y_pred_proba,
            config_name,
            task_type
          )

          results[[config_name]] <- metrics

          if (task_type == "binary") {
            cat(sprintf("Accuracy=%.3f, AUC=%.3f\n", metrics$accuracy, metrics$auc))
          } else {
            cat(sprintf("Accuracy=%.3f, F1=%.3f\n", metrics$accuracy, metrics$macro_f1))
          }
        }, error = function(e) {
          cat(sprintf("ERROR: %s\n", e$message))
        })
      }
    }
  }

  # ==========================================================================
  # Export results for this task
  # ==========================================================================

  write_json(results, file.path(task_results_dir, "all_models_results.json"),
             pretty = TRUE, auto_unbox = TRUE)

  cat(sprintf("\nResults saved to: %s\n", task_results_dir))

  return(results)
}

# ==============================================================================
# Main Execution
# ==============================================================================

# Process binary classification
binary_results <- process_classification_task("binary")

# Process 4-class classification
fourclass_results <- process_classification_task("4class")

# ==============================================================================
# Overall Summary
# ==============================================================================

cat("\n", "=", rep("=", 78), "=\n", sep="")
cat("Overall Summary\n")
cat("=", rep("=", 78), "=\n", sep="")

cat("\nBinary Classification:\n")
binary_acc <- sapply(binary_results, function(x) x$accuracy)
best_binary <- binary_results[[which.max(binary_acc)]]
cat(sprintf("  Best model: %s (Accuracy=%.3f, AUC=%.3f)\n",
            best_binary$model, best_binary$accuracy, best_binary$auc))

cat("\n4-Class Classification:\n")
fourclass_acc <- sapply(fourclass_results, function(x) x$accuracy)
best_fourclass <- fourclass_results[[which.max(fourclass_acc)]]
cat(sprintf("  Best model: %s (Accuracy=%.3f, F1=%.3f)\n",
            best_fourclass$model, best_fourclass$accuracy, best_fourclass$macro_f1))

cat("\n", "=", rep("=", 78), "=\n", sep="")
cat("Complete!\n")
cat("=", rep("=", 78), "=\n", sep="")
