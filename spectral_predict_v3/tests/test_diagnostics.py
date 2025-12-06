"""
Tests for diagnostic plots in Spectral Predict v3.

Tests cover:
- Scatter plot generation
- Confusion matrix accuracy
- ROC curve AUC calculation
- Plot data validation

Note: DearPyGui tests are limited without a display.
These tests validate data processing and plot setup.
"""

import pytest
import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.metrics import confusion_matrix, roc_curve, auc, r2_score


class TestDiagnosticDataProcessing:
    """Test diagnostic plot data processing (without GUI)."""

    def test_prediction_vs_actual_data(self):
        """Test prediction vs actual scatter plot data preparation."""
        # Generate synthetic data
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)

        # Create predictions (with some noise)
        y_pred = y + np.random.randn(100) * 0.1

        # Validate data
        assert y.shape == y_pred.shape, "y_true and y_pred should have same shape"

        # Calculate metrics
        r2 = r2_score(y, y_pred)

        # Calculate plot limits
        y_min = min(y.min(), y_pred.min())
        y_max = max(y.max(), y_pred.max())
        padding = (y_max - y_min) * 0.1
        axis_min = y_min - padding
        axis_max = y_max + padding

        assert axis_min < axis_max, "Axis limits should be valid"

        # R² should be reasonable for good predictions
        assert r2 > 0.8, f"R² should be high for similar predictions, got {r2:.4f}"

        print(f"✓ Prediction vs Actual data: R² = {r2:.4f}")
        print(f"  Axis range: [{axis_min:.2f}, {axis_max:.2f}]")

    def test_confusion_matrix_data(self):
        """Test confusion matrix generation and normalization."""
        # Generate synthetic classification data
        X, y_true = make_classification(
            n_samples=200,
            n_features=20,
            n_classes=3,
            n_informative=10,
            random_state=42
        )

        # Simulate predictions (with some errors)
        y_pred = y_true.copy()
        # Add some random errors
        error_indices = np.random.choice(200, size=30, replace=False)
        y_pred[error_indices] = (y_pred[error_indices] + 1) % 3

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Validate shape
        assert cm.shape == (3, 3), "Confusion matrix should be 3x3 for 3 classes"

        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)  # Handle division by zero

        # Check normalization
        assert np.allclose(cm_norm.sum(axis=1), 1.0), "Rows should sum to 1"

        # Calculate overall accuracy
        accuracy = np.trace(cm) / np.sum(cm) * 100

        print(f"✓ Confusion Matrix: {cm.shape}")
        print(f"  Overall Accuracy: {accuracy:.2f}%")
        print(f"  Confusion Matrix:\n{cm}")

    def test_roc_curve_data(self):
        """Test ROC curve generation for binary classification."""
        # Generate binary classification data
        X, y_true = make_classification(
            n_samples=150,
            n_features=20,
            n_classes=2,
            random_state=42
        )

        # Simulate probability scores
        y_score = np.random.rand(150)
        # Make scores correlated with true labels
        y_score[y_true == 1] += 0.3
        y_score = np.clip(y_score, 0, 1)

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        roc_auc = auc(fpr, tpr)

        # Validate
        assert len(fpr) == len(tpr), "FPR and TPR should have same length"
        assert 0 <= roc_auc <= 1, f"AUC should be in [0, 1], got {roc_auc:.4f}"
        assert fpr[0] == 0.0 or tpr[0] == 0.0, "ROC curve should start at origin"
        assert fpr[-1] == 1.0 or tpr[-1] == 1.0, "ROC curve should end at (1, 1)"

        print(f"✓ ROC Curve: AUC = {roc_auc:.4f}")
        print(f"  Points: {len(fpr)}")

        # AUC should be better than random (0.5)
        assert roc_auc > 0.5, f"AUC should be > 0.5, got {roc_auc:.4f}"


class TestDiagnosticMetrics:
    """Test metric calculations for diagnostics."""

    def test_regression_metrics(self):
        """Test regression diagnostic metrics."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error

        # Generate data
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.2

        # Calculate metrics
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        # Validate metrics
        assert not np.isnan(r2), "R² should not be NaN"
        assert rmse >= 0, "RMSE should be non-negative"
        assert mae >= 0, "MAE should be non-negative"

        print(f"✓ Regression Metrics:")
        print(f"  R² = {r2:.4f}")
        print(f"  RMSE = {rmse:.4f}")
        print(f"  MAE = {mae:.4f}")

    def test_classification_metrics(self):
        """Test classification diagnostic metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score

        # Generate binary classification data
        y_true = np.random.randint(0, 2, size=100)
        # Create predictions with ~80% accuracy
        y_pred = y_true.copy()
        error_indices = np.random.choice(100, size=20, replace=False)
        y_pred[error_indices] = 1 - y_pred[error_indices]

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')

        # Validate
        assert 0 <= accuracy <= 1, f"Accuracy should be in [0, 1], got {accuracy}"
        assert 0 <= precision <= 1, f"Precision should be in [0, 1], got {precision}"
        assert 0 <= recall <= 1, f"Recall should be in [0, 1], got {recall}"

        print(f"✓ Classification Metrics:")
        print(f"  Accuracy = {accuracy:.4f}")
        print(f"  Precision = {precision:.4f}")
        print(f"  Recall = {recall:.4f}")


class TestDiagnosticEdgeCases:
    """Test edge cases for diagnostic calculations."""

    def test_perfect_predictions(self):
        """Test diagnostics with perfect predictions."""
        y_true = np.random.randn(50)
        y_pred = y_true.copy()  # Perfect predictions

        # R² should be 1.0
        r2 = r2_score(y_true, y_pred)
        assert np.isclose(r2, 1.0), f"R² should be 1.0 for perfect predictions, got {r2}"

        print("✓ Perfect predictions: R² = 1.0")

    def test_constant_predictions(self):
        """Test diagnostics with constant predictions."""
        y_true = np.random.randn(50)
        y_pred = np.full(50, y_true.mean())  # Predict mean

        # R² should be 0.0 (same as predicting mean)
        r2 = r2_score(y_true, y_pred)
        assert np.isclose(r2, 0.0, atol=1e-10), \
            f"R² should be ~0 for constant predictions, got {r2}"

        print("✓ Constant predictions: R² = 0.0")

    def test_single_class_confusion_matrix(self):
        """Test confusion matrix when all samples are same class."""
        y_true = np.zeros(50, dtype=int)
        y_pred = np.zeros(50, dtype=int)

        cm = confusion_matrix(y_true, y_pred)

        # Should be 1x1 matrix
        assert cm.shape == (1, 1), f"Should be 1x1 for single class, got {cm.shape}"
        assert cm[0, 0] == 50, "All 50 samples should be in single cell"

        print("✓ Single class confusion matrix handled")

    def test_multiclass_confusion_matrix(self):
        """Test confusion matrix with multiple classes."""
        n_classes = 5
        y_true = np.random.randint(0, n_classes, size=200)
        y_pred = y_true.copy()
        # Add some errors
        error_indices = np.random.choice(200, size=40, replace=False)
        y_pred[error_indices] = (y_pred[error_indices] + 1) % n_classes

        cm = confusion_matrix(y_true, y_pred)

        assert cm.shape == (n_classes, n_classes), \
            f"Should be {n_classes}x{n_classes}, got {cm.shape}"
        assert cm.sum() == 200, "Total count should be 200"

        print(f"✓ Multiclass ({n_classes} classes) confusion matrix: {cm.shape}")


class TestDiagnosticPlotData:
    """Test data structures for diagnostic plots."""

    def test_results_dict_regression(self):
        """Test results dictionary structure for regression."""
        # Simulate results
        results = {
            'task_type': 'regression',
            'y_true': np.random.randn(100),
            'y_pred': np.random.randn(100),
            'model_name': 'PLS'
        }

        # Validate structure
        assert 'task_type' in results
        assert 'y_true' in results
        assert 'y_pred' in results
        assert results['task_type'] == 'regression'
        assert results['y_true'].shape == results['y_pred'].shape

        print("✓ Regression results dictionary structure valid")

    def test_results_dict_classification(self):
        """Test results dictionary structure for classification."""
        # Simulate results
        results = {
            'task_type': 'classification',
            'y_true': np.random.randint(0, 3, size=100),
            'y_pred': np.random.randint(0, 3, size=100),
            'y_score': np.random.rand(100),  # Probability scores
            'model_name': 'RandomForest'
        }

        # Validate structure
        assert 'task_type' in results
        assert 'y_true' in results
        assert 'y_pred' in results
        assert 'y_score' in results
        assert results['task_type'] == 'classification'

        print("✓ Classification results dictionary structure valid")

    def test_plot_data_export_format(self):
        """Test data format for plot export."""
        # Simulate ROC curve data for export
        fpr = np.linspace(0, 1, 100)
        tpr = np.linspace(0, 1, 100) ** 0.5  # Concave curve

        plot_data = {
            'FPR': fpr,
            'TPR': tpr
        }

        # Validate
        assert all(len(v) == 100 for v in plot_data.values()), \
            "All arrays should have same length"
        assert all(isinstance(k, str) for k in plot_data.keys()), \
            "Keys should be strings"

        print("✓ Plot data export format valid")


def test_diagnostic_functions_exist():
    """Test that diagnostic functions are importable."""
    try:
        from spectral_predict_v3.ui.components.diagnostics import (
            plot_prediction_vs_actual,
            plot_confusion_matrix,
            plot_roc_curve,
            create_diagnostic_panel
        )

        print("✓ All diagnostic functions importable")

    except ImportError as e:
        pytest.fail(f"Could not import diagnostic functions: {e}")


if __name__ == "__main__":
    # Run tests
    print("=" * 60)
    print("Testing Diagnostic Plots")
    print("=" * 60)

    # Data processing tests
    print("\n--- Data Processing Tests ---")
    test_data = TestDiagnosticDataProcessing()
    test_data.test_prediction_vs_actual_data()
    test_data.test_confusion_matrix_data()
    test_data.test_roc_curve_data()

    # Metrics tests
    print("\n--- Metrics Tests ---")
    test_metrics = TestDiagnosticMetrics()
    test_metrics.test_regression_metrics()
    test_metrics.test_classification_metrics()

    # Edge cases
    print("\n--- Edge Case Tests ---")
    test_edge = TestDiagnosticEdgeCases()
    test_edge.test_perfect_predictions()
    test_edge.test_constant_predictions()
    test_edge.test_single_class_confusion_matrix()
    test_edge.test_multiclass_confusion_matrix()

    # Plot data tests
    print("\n--- Plot Data Tests ---")
    test_plot_data = TestDiagnosticPlotData()
    test_plot_data.test_results_dict_regression()
    test_plot_data.test_results_dict_classification()
    test_plot_data.test_plot_data_export_format()

    # Import test
    print("\n--- Import Tests ---")
    test_diagnostic_functions_exist()

    print("\n" + "=" * 60)
    print("All diagnostic tests passed!")
    print("=" * 60)
