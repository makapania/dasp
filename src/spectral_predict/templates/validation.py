"""
Cross-validation and metrics templates for generated scripts.
"""


def _cv_splitter_code(task_type: str, cv_strategy: str, cv_folds: int, cv_n_repeats: int) -> tuple[str, str]:
    """Return (import_line, constructor_expr) for the requested CV strategy.

    Exports use the same splitter that Model Development used so generated
    scripts reproduce headline numbers. LOO ignores cv_folds by definition.
    """
    stratified = task_type == 'classification'
    if cv_strategy == 'loo':
        return 'from sklearn.model_selection import LeaveOneOut', 'LeaveOneOut()'
    if cv_strategy == 'repeated_kfold':
        if stratified:
            return (
                'from sklearn.model_selection import RepeatedStratifiedKFold',
                f'RepeatedStratifiedKFold(n_splits={cv_folds}, n_repeats={cv_n_repeats}, random_state=42)',
            )
        return (
            'from sklearn.model_selection import RepeatedKFold',
            f'RepeatedKFold(n_splits={cv_folds}, n_repeats={cv_n_repeats}, random_state=42)',
        )
    # kfold (default)
    if stratified:
        return (
            'from sklearn.model_selection import StratifiedKFold',
            f'StratifiedKFold(n_splits={cv_folds}, shuffle=True, random_state=42)',
        )
    return (
        'from sklearn.model_selection import KFold',
        f'KFold(n_splits={cv_folds}, shuffle=True, random_state=42)',
    )


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
# CROSS-VALIDATION (matches Model Development exactly)
# =============================================================================

{cv_import}
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import clone

# Set up cross-validation
cv = {cv_constructor}

# Store per-fold metrics and per-sample prediction lists.
# Collecting per-sample (not flat concatenation) matters under Repeated K-Fold,
# where each sample appears in multiple test folds — the backend averages those
# repeat predictions before scoring, so we mirror that here to reproduce
# headline numbers exactly. For plain K-Fold / LOO this is a no-op (each sample
# appears exactly once).
fold_rmse = []
fold_r2 = []
fold_mae = []
preds_per_sample = {{}}
truth_per_sample = {{}}

for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_final)):
    X_train, X_test = X_final[train_idx], X_final[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    fold_model = clone(model)
    fold_model.fit(X_train, y_train)
    y_pred_fold = fold_model.predict(X_test).ravel()

    for local_i, sample_idx in enumerate(test_idx):
        preds_per_sample.setdefault(int(sample_idx), []).append(float(y_pred_fold[local_i]))
        truth_per_sample[int(sample_idx)] = y_test[local_i]

    # Per-fold metrics kept for the per-fold printout only — not used in headline numbers.
    fold_rmse.append(np.sqrt(mean_squared_error(y_test, y_pred_fold)))
    fold_r2.append(r2_score(y_test, y_pred_fold))
    fold_mae.append(mean_absolute_error(y_test, y_pred_fold))

# Reduce per-sample predictions (mean across repeats) and compute pooled metrics.
# Pooled RMSE/MAE are required under LOO (per-fold RMSE on 1-sample folds
# degenerates to |y-ŷ| and averaging gives MAE, not RMSE). For Repeated K-Fold
# we reduce-then-score, matching backend cross_val_predict_pooled semantics.
sorted_idx = sorted(preds_per_sample.keys())
all_y_true_arr = np.array([truth_per_sample[i] for i in sorted_idx])
all_y_pred_arr = np.array([np.mean(preds_per_sample[i]) for i in sorted_idx])
rmse = float(np.sqrt(mean_squared_error(all_y_true_arr, all_y_pred_arr)))
r2 = float(r2_score(all_y_true_arr, all_y_pred_arr))
mae = float(mean_absolute_error(all_y_true_arr, all_y_pred_arr))
rpd = np.std(y) / rmse

# Also keep y_pred_cv for compatibility with visualization
y_pred_cv = all_y_pred_arr
'''

CROSS_VALIDATION_CLASSIFICATION_TEMPLATE = '''
# =============================================================================
# CROSS-VALIDATION (matches Model Development)
# =============================================================================

{cv_import}
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import clone
from collections import Counter

# Set up cross-validation
cv = {cv_constructor}

# Use binary for 2 classes, macro otherwise (matches results tab)
unique_classes = np.unique(y)
average_method = 'binary' if len(unique_classes) == 2 else 'macro'

# Per-sample prediction lists (see regression block for Repeated K-Fold note).
fold_acc = []
fold_f1 = []
preds_per_sample = {{}}
truth_per_sample = {{}}

for train_idx, test_idx in cv.split(X_final, y):
    X_train, X_test = X_final[train_idx], X_final[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    fold_model = clone(model)
    fold_model.fit(X_train, y_train)
    y_pred_fold = fold_model.predict(X_test)

    for local_i, sample_idx in enumerate(test_idx):
        preds_per_sample.setdefault(int(sample_idx), []).append(y_pred_fold[local_i])
        truth_per_sample[int(sample_idx)] = y_test[local_i]

    # Per-fold metrics kept for the per-fold printout only — not used in headline numbers.
    fold_acc.append(accuracy_score(y_test, y_pred_fold))
    fold_f1.append(f1_score(y_test, y_pred_fold, average=average_method, zero_division=0))

# Reduce per-sample predictions via majority vote (classifier predict under
# Repeated K-Fold must vote, not average — averaging class labels yields
# nonsensical fractional predictions). For plain K-Fold / LOO this is a no-op.
sorted_idx = sorted(preds_per_sample.keys())
all_y_true_arr = np.array([truth_per_sample[i] for i in sorted_idx])
all_y_pred_arr = np.array(
    [Counter(preds_per_sample[i]).most_common(1)[0][0] for i in sorted_idx]
)
accuracy = float(accuracy_score(all_y_true_arr, all_y_pred_arr))
f1 = float(f1_score(all_y_true_arr, all_y_pred_arr, average=average_method, zero_division=0))

# Keep y_pred_cv for compatibility with visualization
y_pred_cv = all_y_pred_arr
'''

METRICS_TEMPLATE = '''
# =============================================================================
# EVALUATION METRICS (matches Model Development)
# =============================================================================

print(f"\\nCross-validation Results ({cv_folds}-fold):")
print(f"  RMSE: {{rmse:.4f}} (pooled across folds — matches Model Development)")
print(f"  R²:   {{r2:.4f}} (pooled across folds)")
print(f"  MAE:  {{mae:.4f}} (pooled across folds)")
print(f"  RPD:  {{rpd:.2f}}")

# Per-fold details (for reference only — not used in headline numbers)
print(f"\\nPer-fold RMSE: {{[f'{{x:.4f}}' for x in fold_rmse]}}")
print(f"Per-fold R²:   {{[f'{{x:.4f}}' for x in fold_r2]}} (for reference only)")
'''

METRICS_CLASSIFICATION_TEMPLATE = '''
# =============================================================================
# EVALUATION METRICS
# =============================================================================

from sklearn.metrics import confusion_matrix, classification_report

print(f"\\nCross-validation Results ({cv_folds}-fold):")
print(f"  Accuracy: {{accuracy:.4f}} (pooled across folds — matches Model Development)")
print(f"  F1 Score (weighted): {{f1:.4f}} (pooled across folds)")

print("\\nConfusion Matrix:")
print(confusion_matrix(all_y_true_arr, all_y_pred_arr))

print("\\nClassification Report:")
print(classification_report(all_y_true_arr, all_y_pred_arr))
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

CROSS_VALIDATION_ONE_CLASS_TEMPLATE = '''
# =============================================================================
# CROSS-VALIDATION (One-Class — matches contamination.py:run_one_class_cv)
# =============================================================================

{cv_import}
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import clone
from sklearn.metrics import (
    balanced_accuracy_score, recall_score, precision_score,
    f1_score, accuracy_score, roc_auc_score,
)
import numpy as np

def _one_class_metrics(y_true, y_pred, scores=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_true_bin = (y_true == -1).astype(int)
    y_pred_bin = (y_pred == -1).astype(int)
    metrics = {{}}
    n_true_outliers = int(y_true_bin.sum())
    n_true_inliers = int((1 - y_true_bin).sum())
    if n_true_outliers > 0:
        metrics['sensitivity'] = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    else:
        metrics['sensitivity'] = float('nan')
    if n_true_inliers > 0:
        metrics['specificity'] = recall_score(1 - y_true_bin, 1 - y_pred_bin, zero_division=0)
    else:
        metrics['specificity'] = float('nan')
    metrics['precision'] = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    metrics['f1'] = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    metrics['accuracy'] = accuracy_score(y_true_bin, y_pred_bin)
    if n_true_outliers > 0 and n_true_inliers > 0:
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true_bin, y_pred_bin)
    else:
        metrics['balanced_accuracy'] = float('nan')
    if scores is not None and n_true_outliers > 0 and n_true_inliers > 0:
        try:
            metrics['auc'] = roc_auc_score(y_true_bin, -np.asarray(scores))
        except (ValueError, TypeError):
            metrics['auc'] = float('nan')
    else:
        metrics['auc'] = float('nan')
    return metrics

cv = {cv_constructor}

inlier_indices = np.where(y_oc == 1)[0]
outlier_indices = np.where(y_oc == -1)[0]
n_outliers = len(outlier_indices)

fold_metrics = []
all_test_labels = []
all_y_pred = []
all_scores = []
all_test_idx = []

for fold_i, (train_inlier_idx, test_inlier_idx) in enumerate(cv.split(inlier_indices)):
    train_idx = inlier_indices[train_inlier_idx]
    test_inlier = inlier_indices[test_inlier_idx]
    test_idx = np.concatenate([test_inlier, outlier_indices])
    test_labels = np.concatenate([
        np.ones(len(test_inlier), dtype=int),
        -np.ones(n_outliers, dtype=int),
    ])

    fold_model = clone(model)

    if '{model_name}' in ('OneClassSVM', 'EllipticEnvelope', 'LOF'):
        scaler = StandardScaler()
        X_train = scaler.fit_transform({x_var}[train_idx])
        X_test = scaler.transform({x_var}[test_idx])
        if '{model_name}' == 'EllipticEnvelope' and X_train.shape[1] > X_train.shape[0]:
            n_pca = max(2, X_train.shape[0] // 2)
            pca_reducer = PCA(n_components=n_pca)
            X_train = pca_reducer.fit_transform(X_train)
            X_test = pca_reducer.transform(X_test)
    else:
        X_train = {x_var}[train_idx]
        X_test = {x_var}[test_idx]

    fold_model.fit(X_train)

    scores = None
    if hasattr(fold_model, 'decision_function'):
        scores = fold_model.decision_function(X_test)
        y_pred = np.where(scores >= 0, 1, -1)
    elif hasattr(fold_model, 'score_samples'):
        scores = fold_model.score_samples(X_test)
        y_pred = fold_model.predict(X_test)
    else:
        y_pred = fold_model.predict(X_test)

    fold_result = _one_class_metrics(test_labels, y_pred, scores)
    fold_metrics.append(fold_result)

    all_test_labels.append(test_labels)
    all_y_pred.append(y_pred)
    all_test_idx.append(test_idx)
    if scores is not None:
        all_scores.append(scores)

_is_repeated = hasattr(cv, 'n_repeats')

if _is_repeated and all_test_idx:
    pooled_idx_flat = np.concatenate(all_test_idx)
    pooled_labels_flat = np.concatenate(all_test_labels)
    pooled_preds_flat = np.concatenate(all_y_pred)
    unique_samples = np.unique(pooled_idx_flat)
    pooled_labels = np.empty(len(unique_samples), dtype=pooled_labels_flat.dtype)
    pooled_preds = np.empty(len(unique_samples), dtype=pooled_preds_flat.dtype)
    for i, s in enumerate(unique_samples):
        sample_mask = pooled_idx_flat == s
        pooled_labels[i] = pooled_labels_flat[sample_mask][0]
        vals, counts = np.unique(pooled_preds_flat[sample_mask], return_counts=True)
        pooled_preds[i] = vals[np.argmax(counts)]
    pooled_scores = None
else:
    pooled_labels = np.concatenate(all_test_labels)
    pooled_preds = np.concatenate(all_y_pred)
    pooled_scores = np.concatenate(all_scores) if all_scores else None

mean_metrics = _one_class_metrics(pooled_labels, pooled_preds, pooled_scores)

if _is_repeated and fold_metrics:
    fold_aucs = [m.get('auc', float('nan')) for m in fold_metrics]
    fold_aucs = [a for a in fold_aucs if a is not None and not np.isnan(a)]
    mean_metrics['auc'] = float(np.mean(fold_aucs)) if fold_aucs else float('nan')

if np.isnan(mean_metrics['balanced_accuracy']):
    mean_metrics['balanced_accuracy'] = mean_metrics['specificity']

balanced_accuracy = mean_metrics['balanced_accuracy']
sensitivity = mean_metrics['sensitivity']
specificity = mean_metrics['specificity']
auc = mean_metrics['auc']
ber = 1.0 - balanced_accuracy

all_y_true_arr = pooled_labels
y_pred_cv = pooled_preds
'''

METRICS_ONE_CLASS_TEMPLATE = '''
# =============================================================================
# EVALUATION METRICS (One-Class)
# =============================================================================

print(f"\\nCross-validation Results ({cv_folds}-fold, one-class):")
print(f"  Balanced Accuracy:  {{balanced_accuracy:.4f}}")
print(f"  Sensitivity (outlier recall):  {{sensitivity:.4f}}")
print(f"  Specificity (inlier recall):  {{specificity:.4f}}")
print(f"  AUC:  {{auc:.4f}}")
print(f"  BER:  {{ber:.4f}}")
'''

FINAL_MODEL_ONE_CLASS_TEMPLATE = '''
# =============================================================================
# TRAIN FINAL MODEL (One-Class — fit on all inliers, evaluate on all data)
# =============================================================================

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

inlier_indices = np.where(y_oc == 1)[0]
outlier_indices = np.where(y_oc == -1)[0]

if '{model_name}' in ('OneClassSVM', 'EllipticEnvelope', 'LOF'):
    cal_scaler = StandardScaler()
    X_inlier_scaled = cal_scaler.fit_transform({x_var}[inlier_indices])
    X_all_scaled = cal_scaler.transform({x_var})
    if '{model_name}' == 'EllipticEnvelope' and X_inlier_scaled.shape[1] > X_inlier_scaled.shape[0]:
        n_pca = max(2, X_inlier_scaled.shape[0] // 2)
        cal_pca_reducer = PCA(n_components=n_pca)
        X_inlier_scaled = cal_pca_reducer.fit_transform(X_inlier_scaled)
        X_all_scaled = cal_pca_reducer.transform(X_all_scaled)
    model.fit(X_inlier_scaled)
    if hasattr(model, 'decision_function'):
        scores_cal = model.decision_function(X_all_scaled)
        y_pred_cal = np.where(scores_cal >= 0, 1, -1)
    else:
        y_pred_cal = model.predict(X_all_scaled)
        scores_cal = None
else:
    model.fit({x_var}[inlier_indices])
    if hasattr(model, 'decision_function'):
        scores_cal = model.decision_function({x_var})
        y_pred_cal = np.where(scores_cal >= 0, 1, -1)
    else:
        y_pred_cal = model.predict({x_var})
        scores_cal = None

cal_metrics = _one_class_metrics(y_oc, y_pred_cal, scores_cal)
cal_balanced_accuracy = cal_metrics['balanced_accuracy']
cal_sensitivity = cal_metrics['sensitivity']
cal_specificity = cal_metrics['specificity']
cal_auc = cal_metrics['auc']
cal_ber = 1.0 - cal_balanced_accuracy

print(f"\\nFinal model trained on {{len(inlier_indices)}} inliers with {{{x_var}.shape[1]}} features")
print(f"  Calibration Balanced Accuracy:  {{cal_balanced_accuracy:.4f}}")
print(f"  Calibration Sensitivity:  {{cal_sensitivity:.4f}}")
print(f"  Calibration Specificity:  {{cal_specificity:.4f}}")
print(f"  Calibration AUC:  {{cal_auc:.4f}}")
print(f"  Calibration BER:  {{cal_ber:.4f}}")
'''

PREDICTION_ONE_CLASS_TEMPLATE = '''
# =============================================================================
# PREDICTION ON NEW DATA (One-Class Template)
# =============================================================================

# Uncomment and modify to apply the one-class model to new data:
#
# new_data = pd.read_csv("new_spectra.csv")
# X_new = new_data[wavelength_cols].values
#
# # Apply same preprocessing
# X_new_processed = apply_preprocessing(X_new)
#
# # Apply variable selection (if used)
# # X_new_final = X_new_processed[:, selected_indices]
#
# # Apply scaling (if model needs it)
# # X_new_final = cal_scaler.transform(X_new_final)
# # if cal_pca_reducer is not None:
# #     X_new_final = cal_pca_reducer.transform(X_new_final)
#
# # Predict: +1 = inlier, -1 = outlier
# predictions = model.predict(X_new_final)
# labels = np.where(predictions == 1, "Inlier", "Outlier")
#
# # Decision scores (if available)
# if hasattr(model, 'decision_function'):
#     scores = model.decision_function(X_new_final)
# else:
#     scores = None
#
# # Save predictions
# results = pd.DataFrame({{'Sample': new_data.index, 'Label': labels, 'Prediction': predictions}})
# if scores is not None:
#     results['Score'] = scores
# results.to_csv("predictions.csv", index=False)
# print(results)
'''


def get_cross_validation_template(
    task_type: str,
    cv_folds: int,
    cv_strategy: str = 'kfold',
    cv_n_repeats: int = 5,
    model_name: str = '',
    x_var: str = 'X_final',
) -> str:
    """Return the CV code block for the requested strategy/task."""
    cv_import, cv_constructor = _cv_splitter_code(task_type, cv_strategy, cv_folds, cv_n_repeats)
    if task_type == 'one_class':
        return CROSS_VALIDATION_ONE_CLASS_TEMPLATE.format(
            cv_folds=cv_folds, cv_import=cv_import, cv_constructor=cv_constructor,
            model_name=model_name, x_var=x_var,
        )
    tmpl = (
        CROSS_VALIDATION_CLASSIFICATION_TEMPLATE
        if task_type == 'classification'
        else CROSS_VALIDATION_REGRESSION_TEMPLATE
    )
    return tmpl.format(cv_folds=cv_folds, cv_import=cv_import, cv_constructor=cv_constructor)


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
    elif task_type == 'one_class':
        return METRICS_ONE_CLASS_TEMPLATE.format(cv_folds=cv_folds)
    else:
        return METRICS_TEMPLATE.format(cv_folds=cv_folds)


def get_final_model_template(task_type: str, x_var: str = 'X_final', model_name: str = '') -> str:
    if task_type == 'one_class':
        return FINAL_MODEL_ONE_CLASS_TEMPLATE.format(x_var=x_var, model_name=model_name)
    return f'''
# =============================================================================
# TRAIN FINAL MODEL
# =============================================================================

# Train the model on all data
model.fit({x_var}, y)
print(f"\\nFinal model trained on {{{x_var}.shape[0]}} samples with {{{x_var}.shape[1]}} features")
'''


def get_prediction_template(task_type: str) -> str:
    if task_type == 'one_class':
        return PREDICTION_ONE_CLASS_TEMPLATE
    return PREDICTION_TEMPLATE
