"""
R code generator for reproducible analysis scripts.

Generates standalone R scripts from model configurations for use
in the R statistical environment.
"""

import base64
import gzip
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np


# Model mapping from Python to R
MODEL_MAPPING = {
    'PLS': {'package': 'pls', 'function': 'plsr', 'type': 'formula'},
    'Ridge': {'package': 'glmnet', 'function': 'glmnet', 'type': 'glmnet', 'alpha': 0},
    'Lasso': {'package': 'glmnet', 'function': 'glmnet', 'type': 'glmnet', 'alpha': 1},
    'ElasticNet': {'package': 'glmnet', 'function': 'glmnet', 'type': 'glmnet', 'alpha': 0.5},
    'RandomForest': {'package': 'randomForest', 'function': 'randomForest', 'type': 'formula'},
    'XGBoost': {'package': 'xgboost', 'function': 'xgboost', 'type': 'matrix'},
    'LightGBM': {'package': 'lightgbm', 'function': 'lgb.train', 'type': 'lgb'},
    'CatBoost': {'package': 'catboost', 'function': 'catboost.train', 'type': 'catboost'},
    'SVM': {'package': 'e1071', 'function': 'svm', 'type': 'formula'},
    'SVR': {'package': 'e1071', 'function': 'svm', 'type': 'formula'},
    'MLP': {'package': 'nnet', 'function': 'nnet', 'type': 'formula'},
    'MLPRegressor': {'package': 'nnet', 'function': 'nnet', 'type': 'formula'},
    'MLPClassifier': {'package': 'nnet', 'function': 'nnet', 'type': 'formula'},
}


class RCodeGenerator:
    """
    Generate R scripts from model configurations.

    Parameters
    ----------
    model_config : dict
        Model configuration containing:
        - model_name: str (e.g., 'PLS', 'Ridge', 'RandomForest')
        - preprocessing: str (e.g., 'snv', 'sg1', 'raw')
        - target_name: str
        - task_type: str ('regression' or 'classification')
        - params: dict of hyperparameters
        - metrics: dict of performance metrics
        - variable_indices: list or None
        - wavelengths: list or array
        - cv_folds: int
    include_data : bool
        Whether to embed actual data in the script
    data_X : np.ndarray, optional
        Spectral data array
    data_y : np.ndarray, optional
        Target values
    wavelengths : np.ndarray, optional
        Wavelength array
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        include_data: bool = False,
        data_X: Optional[np.ndarray] = None,
        data_y: Optional[np.ndarray] = None,
        wavelengths: Optional[np.ndarray] = None
    ):
        self.config = model_config
        self.include_data = include_data
        self.data_X = data_X
        self.data_y = data_y
        self.wavelengths = wavelengths

        # Extract commonly used values
        self.model_name = model_config.get('model_name', 'Unknown')
        self.preprocessing = model_config.get('preprocessing', 'raw')
        self.task_type = model_config.get('task_type', 'regression')
        self.target_name = model_config.get('target_name', 'target')
        self.params = model_config.get('params', {})
        self.cv_folds = model_config.get('cv_folds', 5)
        self.imbalance_method = model_config.get('imbalance_method', None)
        self.variable_indices = model_config.get('variable_indices', None)

        # Validate data embedding
        if self.include_data:
            self._validate_data()

    def _validate_data(self):
        """Validate data embedding requirements."""
        if self.data_X is None or self.data_y is None:
            raise ValueError("data_X and data_y must be provided when include_data=True")

        # Helper to get size from array or list
        def get_size(obj):
            if hasattr(obj, 'size'):
                return obj.size
            elif hasattr(obj, '__len__'):
                return len(obj)
            return 0

        # Check data size (100 MB limit)
        total_elements = get_size(self.data_X) + get_size(self.data_y)
        if self.wavelengths is not None:
            total_elements += get_size(self.wavelengths)
        estimated_bytes = total_elements * 8 * 0.3 * 1.33
        size_mb = estimated_bytes / (1024 * 1024)

        if size_mb > 100:
            raise ValueError(
                f"Data size ({size_mb:.1f} MB) exceeds 100 MB limit. "
                f"Consider exporting without embedded data."
            )

    def generate_script(self) -> str:
        """
        Generate a complete R script.

        Returns
        -------
        str
            Complete R script as a string
        """
        sections = []

        # 1. Header
        sections.append(self._render_header())

        # 2. Package installation instructions
        sections.append(self._render_install_packages())

        # 3. Load required packages
        sections.append(self._render_load_packages())

        # 4. Preprocessing functions
        if self.preprocessing != 'raw':
            sections.append(self._render_preprocessing_functions())

        # 5. Data loading (embedded or file-based)
        if self.include_data:
            sections.append(self._render_embedded_data())
        else:
            sections.append(self._render_data_loading())

        # 6. Apply preprocessing
        sections.append(self._render_preprocessing_application())

        # 7. Variable selection (always included - sets X_final)
        sections.append(self._render_variable_selection_application())

        # 8. Imbalance handling (if classification)
        if self.imbalance_method and self.task_type == 'classification':
            sections.append(self._render_imbalance_handling())

        # 8. Model definition and training
        sections.append(self._render_model())

        # 9. Cross-validation
        sections.append(self._render_cross_validation())

        # 10. Final model and prediction
        sections.append(self._render_final_model())

        return '\n'.join(sections)

    def save_script(self, filepath: str):
        """Save R script to file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.generate_script())

    # =========================================================================
    # Rendering methods
    # =========================================================================

    def _render_header(self) -> str:
        """Render script header with metadata."""
        model_params = ', '.join(f"{k}: {v}" for k, v in self.params.items())
        if not model_params:
            model_params = 'default parameters'

        return f'''# ============================================================================
# Spectral Analysis in R
# ============================================================================
# Generated by Spectral Predict on {datetime.now().strftime('%Y-%m-%d')}
#
# Model: {self.model_name}
# Preprocessing: {self.preprocessing}
# Task: {self.task_type}
# Parameters: {model_params}
# CV Folds: {self.cv_folds}
# ============================================================================
'''

    def _render_install_packages(self) -> str:
        """Render package installation instructions."""
        packages = self._get_required_packages()
        pkg_str = ', '.join(f'"{pkg}"' for pkg in packages)

        return f'''
# ============================================================================
# INSTALL REQUIRED PACKAGES (run once)
# ============================================================================
# Uncomment and run the following lines to install required packages:
# install.packages(c({pkg_str}))
'''

    def _render_load_packages(self) -> str:
        """Render library loading code."""
        packages = self._get_required_packages()
        load_statements = '\n'.join(f'library({pkg})' for pkg in packages)

        return f'''
# ============================================================================
# LOAD REQUIRED PACKAGES
# ============================================================================
{load_statements}
'''

    def _get_required_packages(self) -> List[str]:
        """Get list of required R packages."""
        packages = []

        # Model-specific package
        if self.model_name in MODEL_MAPPING:
            packages.append(MODEL_MAPPING[self.model_name]['package'])

        # Preprocessing packages
        if 'sg' in self.preprocessing.lower() or 'deriv' in self.preprocessing.lower():
            packages.append('prospectr')

        # Always include base stats packages
        if 'caret' not in packages:
            packages.append('caret')  # For cross-validation

        # Imbalance handling packages
        if self.imbalance_method and self.imbalance_method.lower() == 'smote':
            packages.append('smotefamily')

        return packages

    def _render_preprocessing_functions(self) -> str:
        """Render preprocessing function definitions."""
        functions = []

        if 'snv' in self.preprocessing.lower():
            functions.append('''
# Standard Normal Variate (SNV) preprocessing
apply_snv <- function(X) {
  # Apply SNV to each row (spectrum)
  t(apply(X, 1, function(row) {
    (row - mean(row)) / sd(row)
  }))
}
''')

        if 'sg' in self.preprocessing.lower() or 'deriv' in self.preprocessing.lower():
            # Determine derivative order
            deriv_order = 0
            if 'sg1' in self.preprocessing.lower() or 'deriv1' in self.preprocessing.lower():
                deriv_order = 1
            elif 'sg2' in self.preprocessing.lower() or 'deriv2' in self.preprocessing.lower():
                deriv_order = 2

            if deriv_order > 0:
                functions.append(f'''
# Savitzky-Golay derivative preprocessing
apply_savgol_derivative <- function(X, derivative = {deriv_order}, window_length = 7) {{
  # Apply Savitzky-Golay filter to each row
  # Using prospectr::savitzkyGolay
  t(apply(X, 1, function(row) {{
    prospectr::savitzkyGolay(row, m = derivative, p = 2, w = window_length)
  }}))
}}
''')

        if not functions:
            return "\n# No preprocessing - using raw spectra\n"

        return f'''
# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================
{''.join(functions)}
'''

    def _render_data_loading(self) -> str:
        """Render data loading code (file-based)."""
        return f'''
# ============================================================================
# DATA LOADING
# ============================================================================
# Load your spectral data
# Expected format: CSV file with wavelengths as columns and samples as rows
# Last column should be the target variable

# Update this path to your data file
data_file <- "your_data.csv"

# Read data
data <- read.csv(data_file, header = TRUE)

# Separate features (X) and target (y)
X <- as.matrix(data[, -ncol(data)])  # All columns except last
y <- data[, ncol(data)]  # Last column

# Optional: wavelength information
# wavelengths <- as.numeric(colnames(data)[-ncol(data)])

cat("Data loaded: X shape =", dim(X), ", y length =", length(y), "\\n")
'''

    def _render_embedded_data(self) -> str:
        """Render embedded data section."""
        if not self.include_data:
            return ""

        sections = [
            "\n# ============================================================================",
            "# EMBEDDED DATA",
            "# ============================================================================\n",
        ]

        # Encode data
        X_encoded = self._encode_array(self.data_X)
        y_encoded = self._encode_array(self.data_y)

        # Add decode function
        sections.append('''
# Function to decode embedded data
decode_embedded_data <- function(encoded_str) {
  library(jsonlite)
  decoded <- base64enc::base64decode(encoded_str)
  decompressed <- memDecompress(decoded, type = "gzip")
  data_list <- fromJSON(rawToChar(decompressed))
  return(as.matrix(data_list))
}

# Ensure base64enc is available
if (!require("base64enc", quietly = TRUE)) {
  install.packages("base64enc")
  library(base64enc)
}
''')

        # Add encoded data
        sections.append(f'\n# Encoded spectral data\nX_ENCODED <- "{X_encoded}"')
        sections.append(f'\n# Encoded target values\nY_ENCODED <- "{y_encoded}"')

        if self.wavelengths is not None:
            wl_encoded = self._encode_array(self.wavelengths)
            sections.append(f'\n# Encoded wavelengths\nWAVELENGTHS_ENCODED <- "{wl_encoded}"')

        # Decode data
        sections.append('\n# Decode embedded data')
        sections.append('X <- decode_embedded_data(X_ENCODED)')
        sections.append('y <- as.vector(decode_embedded_data(Y_ENCODED))')

        if self.wavelengths is not None:
            sections.append('wavelengths <- as.vector(decode_embedded_data(WAVELENGTHS_ENCODED))')

        sections.append('\ncat("Loaded embedded data: X shape =", dim(X), ", y length =", length(y), "\\n")')

        return '\n'.join(sections)

    @staticmethod
    def _encode_array(arr) -> str:
        """Encode array (numpy or list) to base64+gzip string."""
        # Handle both numpy arrays and lists
        if hasattr(arr, 'tolist'):
            data = arr.tolist()
        else:
            data = list(arr)
        data_bytes = json.dumps(data).encode('utf-8')
        compressed = gzip.compress(data_bytes, compresslevel=9)
        encoded = base64.b64encode(compressed).decode('ascii')
        return encoded

    def _render_preprocessing_application(self) -> str:
        """Render preprocessing application code."""
        preproc = self.preprocessing.lower()

        if preproc == 'raw':
            return '\n# No preprocessing - using raw spectra\nX_processed <- X\n'

        sections = [
            "\n# ============================================================================",
            "# APPLY PREPROCESSING",
            "# ============================================================================\n",
        ]

        if preproc == 'snv':
            sections.append("X_processed <- apply_snv(X)")
        elif 'sg1' in preproc or 'deriv1' in preproc:
            sections.append("X_processed <- apply_savgol_derivative(X, derivative = 1, window_length = 7)")
        elif 'sg2' in preproc or 'deriv2' in preproc:
            sections.append("X_processed <- apply_savgol_derivative(X, derivative = 2, window_length = 7)")
        elif 'deriv_snv' in preproc:
            sections.append("X_processed <- apply_savgol_derivative(X, derivative = 1, window_length = 7)")
            sections.append("X_processed <- apply_snv(X_processed)")

        sections.append('\ncat("Preprocessed data shape:", dim(X_processed), "\\n")')

        return '\n'.join(sections)

    def _render_variable_selection_application(self) -> str:
        """Render variable selection application code."""
        if self.variable_indices is None:
            return '''
# ============================================================================
# VARIABLE SELECTION
# ============================================================================

# No variable selection - using all variables
X_final <- X_processed
cat("Using all", ncol(X_final), "variables\\n")
'''

        # Convert indices to R format (1-indexed)
        indices = self.variable_indices
        if hasattr(indices, 'tolist'):
            indices = indices.tolist()

        # R uses 1-based indexing, so add 1 to each index
        r_indices = [i + 1 for i in indices]

        # Format indices for R
        if len(r_indices) <= 20:
            indices_str = f"c({', '.join(map(str, r_indices))})"
        else:
            # For large lists, format more compactly
            indices_str = f"c({', '.join(map(str, r_indices))})"

        var_sel_method = self.config.get('variable_selection_method', 'selection')

        return f'''
# ============================================================================
# APPLY VARIABLE SELECTION
# ============================================================================

# Selected variable indices (from {var_sel_method})
# Note: R uses 1-based indexing
selected_indices <- {indices_str}

# Apply variable selection
X_final <- X_processed[, selected_indices]

cat("After variable selection:", ncol(X_final), "variables selected\\n")
'''

    def _render_imbalance_handling(self) -> str:
        """Render imbalance handling code for classification."""
        if not self.imbalance_method:
            return ''

        method = self.imbalance_method.lower()

        if method == 'smote':
            return '''
# ============================================================================
# IMBALANCE HANDLING: SMOTE
# ============================================================================

# Apply SMOTE to handle class imbalance
# Convert to data frame for SMOTE
df_for_smote <- data.frame(X_final, class = as.factor(y))

# Apply SMOTE
library(smotefamily)
smote_result <- SMOTE(df_for_smote[, -ncol(df_for_smote)], df_for_smote$class, K = 5, dup_size = 0)

# Extract balanced data
X_balanced <- as.matrix(smote_result$data[, -ncol(smote_result$data)])
y_balanced <- as.numeric(as.character(smote_result$data$class))

cat("SMOTE applied:", nrow(X_final), "samples ->", nrow(X_balanced), "samples\\n")

# Update variables to use balanced data
X_final <- X_balanced
y <- y_balanced
'''
        elif method == 'class_weight':
            return '''
# ============================================================================
# IMBALANCE HANDLING: Class Weighting
# ============================================================================

# Note: Some R models support class weights through the 'weights' parameter
# This would need to be calculated and passed to the model
class_counts <- table(y)
class_weights <- max(class_counts) / class_counts
sample_weights <- class_weights[as.character(y)]
cat("Class weights calculated for imbalanced data\\n")
'''
        else:
            return f'\n# Imbalance method: {self.imbalance_method} (not implemented in R export)\n'

    def _render_model(self) -> str:
        """Render model definition and training code."""
        if self.model_name not in MODEL_MAPPING:
            return f"\n# ERROR: Model '{self.model_name}' not supported in R code generation\n"

        model_info = MODEL_MAPPING[self.model_name]
        sections = [
            "\n# ============================================================================",
            f"# MODEL: {self.model_name}",
            "# ============================================================================\n",
        ]

        # Generate model-specific code
        if self.model_name == 'PLS':
            n_components = self.params.get('n_components', 10)
            sections.append(f'''
# PLS Regression
# Create data frame for formula interface
df_train <- data.frame(y = y, X_final)

# Fit PLS model
model <- plsr(y ~ ., data = df_train, ncomp = {n_components}, validation = "none")

cat("PLS model trained with", {n_components}, "components\\n")
''')

        elif self.model_name in ['Ridge', 'Lasso', 'ElasticNet']:
            alpha = model_info['alpha']
            lambda_val = self.params.get('alpha', 1.0)  # Python 'alpha' maps to R 'lambda'

            sections.append(f'''
# {self.model_name} Regression using glmnet
# Note: glmnet uses alpha for mixing (0=ridge, 1=lasso), lambda for regularization

model <- glmnet(X_final, y, alpha = {alpha}, lambda = {lambda_val}, family = "gaussian")

cat("{self.model_name} model trained with alpha =", {alpha}, ", lambda =", {lambda_val}, "\\n")
''')

        elif self.model_name == 'RandomForest':
            n_trees = self.params.get('n_estimators', 100)
            sections.append(f'''
# Random Forest Regression
df_train <- data.frame(y = y, X_final)

model <- randomForest(y ~ ., data = df_train, ntree = {n_trees})

cat("Random Forest trained with", {n_trees}, "trees\\n")
''')

        elif self.model_name == 'SVM' or self.model_name == 'SVR':
            kernel = self.params.get('kernel', 'rbf')
            sections.append(f'''
# Support Vector Machine
df_train <- data.frame(y = y, X_final)

model <- svm(y ~ ., data = df_train, kernel = "{kernel}")

cat("SVM model trained with kernel =", "{kernel}", "\\n")
''')

        elif self.model_name == 'XGBoost':
            n_rounds = self.params.get('n_estimators', 100)
            max_depth = self.params.get('max_depth', 6)
            learning_rate = self.params.get('learning_rate', 0.1)

            if self.task_type == 'classification':
                objective = "binary:logistic"  # or "multi:softmax" for multiclass
            else:
                objective = "reg:squarederror"

            sections.append(f'''
# XGBoost {'Classification' if self.task_type == 'classification' else 'Regression'}
# Prepare data in matrix format
dtrain <- xgb.DMatrix(data = X_final, label = y)

# Set parameters
params <- list(
  objective = "{objective}",
  eta = {learning_rate},
  max_depth = {max_depth}
)

# Train model
model <- xgboost(data = dtrain, params = params, nrounds = {n_rounds}, verbose = 0)

cat("XGBoost model trained with", {n_rounds}, "rounds\\n")
''')

        elif self.model_name == 'LightGBM':
            n_estimators = self.params.get('n_estimators', 100)
            max_depth = self.params.get('max_depth', -1)
            learning_rate = self.params.get('learning_rate', 0.1)
            num_leaves = self.params.get('num_leaves', 31)

            if self.task_type == 'classification':
                objective = "binary"
            else:
                objective = "regression"

            sections.append(f'''
# LightGBM {'Classification' if self.task_type == 'classification' else 'Regression'}
# Prepare data in LightGBM format
dtrain <- lgb.Dataset(data = X_final, label = y)

# Set parameters
params <- list(
  objective = "{objective}",
  learning_rate = {learning_rate},
  max_depth = {max_depth},
  num_leaves = {num_leaves},
  verbose = -1
)

# Train model
model <- lgb.train(
  params = params,
  data = dtrain,
  nrounds = {n_estimators}
)

cat("LightGBM model trained with", {n_estimators}, "rounds\\n")
''')

        elif self.model_name == 'CatBoost':
            n_estimators = self.params.get('n_estimators', 100)
            max_depth = self.params.get('max_depth', 6)
            learning_rate = self.params.get('learning_rate', 0.1)

            if self.task_type == 'classification':
                loss_function = "Logloss"
            else:
                loss_function = "RMSE"

            sections.append(f'''
# CatBoost {'Classification' if self.task_type == 'classification' else 'Regression'}
# Prepare data in CatBoost format
train_pool <- catboost.load_pool(data = X_final, label = y)

# Set parameters
params <- list(
  iterations = {n_estimators},
  depth = {max_depth},
  learning_rate = {learning_rate},
  loss_function = "{loss_function}",
  verbose = 0
)

# Train model
model <- catboost.train(train_pool, NULL, params)

cat("CatBoost model trained with", {n_estimators}, "iterations\\n")
''')

        elif self.model_name in ['MLP', 'MLPRegressor', 'MLPClassifier']:
            hidden_layer_sizes = self.params.get('hidden_layer_sizes', (100,))
            max_iter = self.params.get('max_iter', 200)

            # Convert tuple to R-friendly format
            if isinstance(hidden_layer_sizes, tuple):
                size = hidden_layer_sizes[0] if len(hidden_layer_sizes) > 0 else 100
            else:
                size = hidden_layer_sizes

            # For classification, specify linout=FALSE for sigmoid output
            linout = "TRUE" if self.task_type == 'regression' else "FALSE"

            sections.append(f'''
# Multi-Layer Perceptron (Neural Network) {'Classification' if self.task_type == 'classification' else 'Regression'}
# Using nnet package
df_train <- data.frame(y = y, X_final)

# Train neural network
# Note: nnet supports single hidden layer
model <- nnet(
  y ~ .,
  data = df_train,
  size = {size},
  linout = {linout},
  maxit = {max_iter},
  trace = FALSE
)

cat("MLP model trained with", {size}, "hidden units\\n")
''')

        else:
            sections.append(f"# Model '{self.model_name}' code generation not yet implemented\n")

        return '\n'.join(sections)

    def _render_cross_validation(self) -> str:
        """Render cross-validation code."""
        if self.task_type == 'classification':
            return f'''
# ============================================================================
# CROSS-VALIDATION
# ============================================================================

# Set up {self.cv_folds}-fold cross-validation
set.seed(42)
folds <- createFolds(y, k = {self.cv_folds}, list = TRUE, returnTrain = FALSE)

# Initialize vectors for predictions and actuals
cv_predictions <- numeric(length(y))
cv_actuals <- numeric(length(y))

# Perform cross-validation
for (fold_idx in seq_along(folds)) {{
  test_indices <- folds[[fold_idx]]
  train_indices <- setdiff(seq_len(nrow(X_final)), test_indices)

  X_train_cv <- X_final[train_indices, ]
  y_train_cv <- y[train_indices]
  X_test_cv <- X_final[test_indices, ]
  y_test_cv <- y[test_indices]

  # Train fold model (model-specific code needed here)
  # This is a simplified example - actual code depends on model type

  # For demonstration, using simple logistic model
  df_train_cv <- data.frame(y = as.factor(y_train_cv), X_train_cv)
  fold_model <- glm(y ~ ., data = df_train_cv, family = binomial)

  # Predict on test fold
  df_test_cv <- data.frame(X_test_cv)
  colnames(df_test_cv) <- colnames(df_train_cv)[-1]
  fold_prob <- predict(fold_model, newdata = df_test_cv, type = "response")
  fold_predictions <- ifelse(fold_prob > 0.5, 1, 0)

  # Store predictions and actuals
  cv_predictions[test_indices] <- fold_predictions
  cv_actuals[test_indices] <- y_test_cv
}}

# Calculate classification metrics
cv_accuracy <- mean(cv_predictions == cv_actuals)
confusion_matrix <- table(Actual = cv_actuals, Predicted = cv_predictions)

# Calculate F1 score (for binary classification)
tp <- sum(cv_predictions == 1 & cv_actuals == 1)
fp <- sum(cv_predictions == 1 & cv_actuals == 0)
fn <- sum(cv_predictions == 0 & cv_actuals == 1)
precision <- ifelse(tp + fp > 0, tp / (tp + fp), 0)
recall <- ifelse(tp + fn > 0, tp / (tp + fn), 0)
cv_f1 <- ifelse(precision + recall > 0, 2 * (precision * recall) / (precision + recall), 0)

cat("\\n=== Cross-Validation Results ===\\n")
cat("Accuracy:", round(cv_accuracy, 4), "\\n")
cat("F1 Score:", round(cv_f1, 4), "\\n")
cat("\\nConfusion Matrix:\\n")
print(confusion_matrix)
'''
        else:
            return f'''
# ============================================================================
# CROSS-VALIDATION
# ============================================================================

# Set up {self.cv_folds}-fold cross-validation
set.seed(42)
folds <- createFolds(y, k = {self.cv_folds}, list = TRUE, returnTrain = FALSE)

# Initialize vectors for predictions and actuals
cv_predictions <- numeric(length(y))
cv_actuals <- numeric(length(y))

# Perform cross-validation
for (fold_idx in seq_along(folds)) {{
  test_indices <- folds[[fold_idx]]
  train_indices <- setdiff(seq_len(nrow(X_final)), test_indices)

  X_train_cv <- X_final[train_indices, ]
  y_train_cv <- y[train_indices]
  X_test_cv <- X_final[test_indices, ]
  y_test_cv <- y[test_indices]

  # Train fold model (model-specific code needed here)
  # This is a simplified example - actual code depends on model type

  # For demonstration, using simple linear model
  df_train_cv <- data.frame(y = y_train_cv, X_train_cv)
  fold_model <- lm(y ~ ., data = df_train_cv)

  # Predict on test fold
  df_test_cv <- data.frame(X_test_cv)
  colnames(df_test_cv) <- colnames(df_train_cv)[-1]
  fold_predictions <- predict(fold_model, newdata = df_test_cv)

  # Store predictions and actuals
  cv_predictions[test_indices] <- fold_predictions
  cv_actuals[test_indices] <- y_test_cv
}}

# Calculate metrics
cv_rmse <- sqrt(mean((cv_predictions - cv_actuals)^2))
cv_r2 <- 1 - sum((cv_actuals - cv_predictions)^2) / sum((cv_actuals - mean(cv_actuals))^2)
cv_mae <- mean(abs(cv_predictions - cv_actuals))

cat("\\n=== Cross-Validation Results ===\\n")
cat("RMSE:", round(cv_rmse, 4), "\\n")
cat("R²:", round(cv_r2, 4), "\\n")
cat("MAE:", round(cv_mae, 4), "\\n")
'''

    def _render_final_model(self) -> str:
        """Render final model training and prediction template."""
        return '''
# ============================================================================
# FINAL MODEL AND PREDICTIONS
# ============================================================================

# The model has already been trained on the full dataset above
# To make predictions on new data:

# Example prediction function
predict_new_samples <- function(new_X, model) {
  # Apply the same preprocessing as training data
  # (implement preprocessing steps here)
  new_X_processed <- new_X  # Replace with actual preprocessing

  # Make predictions (adjust based on model type)
  predictions <- predict(model, newdata = new_X_processed)

  return(predictions)
}

cat("\\nModel training complete. Use predict_new_samples() for new data.\\n")
'''


def generate_r_script_from_config(
    model_config: Dict[str, Any],
    include_data: bool = False,
    data_X: Optional[np.ndarray] = None,
    data_y: Optional[np.ndarray] = None,
    wavelengths: Optional[np.ndarray] = None
) -> str:
    """
    Convenience function to generate R script from model configuration.

    Parameters
    ----------
    model_config : dict
        Model configuration dictionary
    include_data : bool
        Embed actual data in script
    data_X : np.ndarray, optional
        Spectral data
    data_y : np.ndarray, optional
        Target values
    wavelengths : np.ndarray, optional
        Wavelength array

    Returns
    -------
    str
        Complete R script
    """
    generator = RCodeGenerator(
        model_config=model_config,
        include_data=include_data,
        data_X=data_X,
        data_y=data_y,
        wavelengths=wavelengths
    )
    return generator.generate_script()
