"""
Cross-validation and metrics templates for generated scripts.
"""

CROSS_VALIDATION_TEMPLATE = '''
# =============================================================================
# CROSS-VALIDATION
# =============================================================================

from sklearn.model_selection import cross_val_predict, KFold{stratified_import}

# Set up cross-validation
cv = {cv_class}(n_splits={cv_folds}, shuffle=True, random_state=42)

# Get cross-validated predictions
y_pred_cv = cross_val_predict(model, X_final, y, cv=cv)
'''

CROSS_VALIDATION_REGRESSION_TEMPLATE = '''
# =============================================================================
# CROSS-VALIDATION
# =============================================================================

from sklearn.model_selection import cross_val_predict, KFold

# Set up cross-validation
cv = KFold(n_splits={cv_folds}, shuffle=True, random_state=42)

# Get cross-validated predictions
y_pred_cv = cross_val_predict(model, X_final, y, cv=cv)
'''

CROSS_VALIDATION_CLASSIFICATION_TEMPLATE = '''
# =============================================================================
# CROSS-VALIDATION
# =============================================================================

from sklearn.model_selection import cross_val_predict, StratifiedKFold

# Set up stratified cross-validation (maintains class proportions)
cv = StratifiedKFold(n_splits={cv_folds}, shuffle=True, random_state=42)

# Get cross-validated predictions
y_pred_cv = cross_val_predict(model, X_final, y, cv=cv)

# For probability predictions (if needed):
# y_prob_cv = cross_val_predict(model, X_final, y, cv=cv, method='predict_proba')
'''

METRICS_TEMPLATE = '''
# =============================================================================
# EVALUATION METRICS
# =============================================================================

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Calculate regression metrics
rmse = np.sqrt(mean_squared_error(y, y_pred_cv))
r2 = r2_score(y, y_pred_cv)
mae = mean_absolute_error(y, y_pred_cv)
rpd = np.std(y) / rmse  # Ratio of Performance to Deviation

print(f"\\nCross-validation Results ({cv_folds}-fold):")
print(f"  RMSE: {{rmse:.4f}}")
print(f"  R²:   {{r2:.4f}}")
print(f"  MAE:  {{mae:.4f}}")
print(f"  RPD:  {{rpd:.2f}}")
'''

METRICS_CLASSIFICATION_TEMPLATE = '''
# =============================================================================
# EVALUATION METRICS
# =============================================================================

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Calculate classification metrics
accuracy = accuracy_score(y, y_pred_cv)
f1 = f1_score(y, y_pred_cv, average='weighted')

print(f"\\nCross-validation Results ({cv_folds}-fold):")
print(f"  Accuracy: {{accuracy:.4f}}")
print(f"  F1 Score (weighted): {{f1:.4f}}")

print("\\nConfusion Matrix:")
print(confusion_matrix(y, y_pred_cv))

print("\\nClassification Report:")
print(classification_report(y, y_pred_cv))
'''

FINAL_MODEL_TEMPLATE = '''
# =============================================================================
# TRAIN FINAL MODEL
# =============================================================================

# Train the model on all data
model.fit(X_final, y)
print(f"\\nFinal model trained on {{X_final.shape[0]}} samples with {{X_final.shape[1]}} features")
'''

PREDICTION_TEMPLATE = '''
# =============================================================================
# PREDICTION ON NEW DATA (Template)
# =============================================================================

# Uncomment and modify to apply the model to new data:
#
# # Load new spectra
# new_data = pd.read_csv("new_spectra.csv")
# X_new = new_data[wavelength_cols].values
#
# # Apply same preprocessing
# X_new_processed = apply_preprocessing(X_new)  # Use your preprocessing function
#
# # Apply variable selection (if used)
# X_new_final = X_new_processed[:, selected_indices]  # If variable selection was used
#
# # Make predictions
# predictions = model.predict(X_new_final)
#
# # Save predictions
# results = pd.DataFrame({{'Sample': new_data.index, 'Prediction': predictions.ravel()}})
# results.to_csv("predictions.csv", index=False)
'''


def get_cross_validation_template(task_type: str, cv_folds: int) -> str:
    """
    Get the appropriate cross-validation template.

    Parameters
    ----------
    task_type : str
        'regression' or 'classification'
    cv_folds : int
        Number of cross-validation folds

    Returns
    -------
    str
        Cross-validation code template
    """
    if task_type == 'classification':
        return CROSS_VALIDATION_CLASSIFICATION_TEMPLATE.format(cv_folds=cv_folds)
    else:
        return CROSS_VALIDATION_REGRESSION_TEMPLATE.format(cv_folds=cv_folds)


def get_metrics_template(task_type: str, cv_folds: int) -> str:
    """
    Get the appropriate metrics template.

    Parameters
    ----------
    task_type : str
        'regression' or 'classification'
    cv_folds : int
        Number of CV folds (for display)

    Returns
    -------
    str
        Metrics calculation code template
    """
    if task_type == 'classification':
        return METRICS_CLASSIFICATION_TEMPLATE.format(cv_folds=cv_folds)
    else:
        return METRICS_TEMPLATE.format(cv_folds=cv_folds)
