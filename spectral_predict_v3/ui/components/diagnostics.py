"""
Diagnostic plots for model evaluation in Spectral Predict v3.

This module provides visualization components for:
- Prediction vs Actual scatter plots (regression)
- Confusion matrices (classification)
- ROC curves with AUC (classification)
"""

import numpy as np
import dearpygui.dearpygui as dpg
from typing import Optional, Dict, Any, Tuple, List
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_prediction_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tag: str,
    parent: int,
    title: str = "Prediction vs Actual",
    width: int = 400,
    height: int = 400
) -> int:
    """
    Create a scatter plot of predicted vs actual values for regression.

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted target values
    tag : str
        Unique tag for the plot
    parent : int
        Parent DPG item ID
    title : str
        Plot title
    width : int
        Plot width in pixels
    height : int
        Plot height in pixels

    Returns
    -------
    plot_id : int
        DPG item ID of the created plot
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Calculate limits for axes (with some padding)
    y_min = min(y_true.min(), y_pred.min())
    y_max = max(y_true.max(), y_pred.max())
    padding = (y_max - y_min) * 0.1
    axis_min = y_min - padding
    axis_max = y_max + padding

    # Create plot
    with dpg.plot(
        label=title,
        tag=tag,
        parent=parent,
        width=width,
        height=height,
        anti_aliased=True
    ):
        # X axis (actual)
        dpg.add_plot_axis(dpg.mvXAxis, label="Actual", tag=f"{tag}_x")
        dpg.set_axis_limits(f"{tag}_x", axis_min, axis_max)

        # Y axis (predicted)
        dpg.add_plot_axis(dpg.mvYAxis, label="Predicted", tag=f"{tag}_y")
        dpg.set_axis_limits(f"{tag}_y", axis_min, axis_max)

        # Add 1:1 reference line
        dpg.add_line_series(
            [axis_min, axis_max],
            [axis_min, axis_max],
            label="Perfect Prediction",
            parent=f"{tag}_y",
            tag=f"{tag}_reference"
        )
        dpg.bind_item_theme(f"{tag}_reference", "__line_theme__")

        # Add scatter plot
        dpg.add_scatter_series(
            y_true.tolist(),
            y_pred.tolist(),
            label="Predictions",
            parent=f"{tag}_y",
            tag=f"{tag}_scatter"
        )
        dpg.bind_item_theme(f"{tag}_scatter", "__scatter_theme__")

    # Create themes if they don't exist
    if not dpg.does_item_exist("__line_theme__"):
        with dpg.theme(tag="__line_theme__"):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (150, 150, 150, 255))
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2)

    if not dpg.does_item_exist("__scatter_theme__"):
        with dpg.theme(tag="__scatter_theme__"):
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (52, 152, 219, 180))
                dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 4)

    return tag


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tag: str,
    parent: int,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    width: int = 400,
    height: int = 400
) -> int:
    """
    Create a heatmap visualization of a confusion matrix for classification.

    Parameters
    ----------
    y_true : np.ndarray
        True class labels
    y_pred : np.ndarray
        Predicted class labels
    tag : str
        Unique tag for the plot
    parent : int
        Parent DPG item ID
    class_names : list of str, optional
        Names of classes (if None, uses class indices)
    title : str
        Plot title
    width : int
        Plot width in pixels
    height : int
        Plot height in pixels

    Returns
    -------
    plot_id : int
        DPG item ID of the created plot
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]

    # Get class names
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    elif len(class_names) != n_classes:
        class_names = [f"Class {i}" for i in range(n_classes)]

    # Normalize confusion matrix for visualization
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # Handle division by zero

    # Create group for confusion matrix display
    with dpg.group(tag=tag, parent=parent, horizontal=False):
        dpg.add_text(title, color=(255, 255, 255))
        dpg.add_spacer(height=10)

        # Create table for confusion matrix
        with dpg.table(
            tag=f"{tag}_table",
            header_row=True,
            borders_innerH=True,
            borders_innerV=True,
            borders_outerH=True,
            borders_outerV=True,
            width=width
        ):
            # Header row
            dpg.add_table_column(label="True \\ Pred")
            for name in class_names:
                dpg.add_table_column(label=name)

            # Data rows
            for i in range(n_classes):
                with dpg.table_row():
                    dpg.add_text(class_names[i])
                    for j in range(n_classes):
                        # Color code: green on diagonal, red off diagonal
                        count = cm[i, j]
                        pct = cm_norm[i, j] * 100

                        if i == j:
                            color = (100, 200, 100)  # Green for correct
                        else:
                            color = (200, 100, 100)  # Red for incorrect

                        text = f"{count} ({pct:.1f}%)"
                        dpg.add_text(text, color=color)

        dpg.add_spacer(height=10)

        # Calculate and display accuracy
        accuracy = np.trace(cm) / np.sum(cm) * 100
        dpg.add_text(f"Overall Accuracy: {accuracy:.2f}%", color=(100, 200, 255))

    return tag


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    tag: str,
    parent: int,
    title: str = "ROC Curve",
    width: int = 400,
    height: int = 400,
    pos_label: int = 1
) -> int:
    """
    Create an ROC curve plot for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1)
    y_score : np.ndarray
        Target scores (probability estimates or decision function)
    tag : str
        Unique tag for the plot
    parent : int
        Parent DPG item ID
    title : str
        Plot title
    width : int
        Plot width in pixels
    height : int
        Plot height in pixels
    pos_label : int
        Label of the positive class (default: 1)

    Returns
    -------
    plot_id : int
        DPG item ID of the created plot
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    # Create plot
    with dpg.plot(
        label=f"{title} (AUC = {roc_auc:.3f})",
        tag=tag,
        parent=parent,
        width=width,
        height=height,
        anti_aliased=True
    ):
        # X axis (FPR)
        dpg.add_plot_axis(dpg.mvXAxis, label="False Positive Rate", tag=f"{tag}_x")
        dpg.set_axis_limits(f"{tag}_x", 0, 1)

        # Y axis (TPR)
        dpg.add_plot_axis(dpg.mvYAxis, label="True Positive Rate", tag=f"{tag}_y")
        dpg.set_axis_limits(f"{tag}_y", 0, 1)

        # Add diagonal reference line (random classifier)
        dpg.add_line_series(
            [0, 1],
            [0, 1],
            label="Random (AUC=0.5)",
            parent=f"{tag}_y",
            tag=f"{tag}_random"
        )
        dpg.bind_item_theme(f"{tag}_random", "__roc_random_theme__")

        # Add ROC curve
        dpg.add_line_series(
            fpr.tolist(),
            tpr.tolist(),
            label=f"ROC (AUC={roc_auc:.3f})",
            parent=f"{tag}_y",
            tag=f"{tag}_roc"
        )
        dpg.bind_item_theme(f"{tag}_roc", "__roc_curve_theme__")

    # Create themes if they don't exist
    if not dpg.does_item_exist("__roc_random_theme__"):
        with dpg.theme(tag="__roc_random_theme__"):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (150, 150, 150, 255))
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2)

    if not dpg.does_item_exist("__roc_curve_theme__"):
        with dpg.theme(tag="__roc_curve_theme__"):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (52, 152, 219, 255))
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 3)

    return tag


def create_diagnostic_panel(
    results: Dict[str, Any],
    tag: str,
    parent: int,
    width: int = 1200,
    height: int = 400
) -> int:
    """
    Create a comprehensive diagnostic panel based on task type.

    For regression: Shows prediction vs actual plot
    For classification: Shows confusion matrix and ROC curve (if binary)

    Parameters
    ----------
    results : dict
        Results dictionary containing:
        - 'task_type': 'regression' or 'classification'
        - 'y_true': true values
        - 'y_pred': predictions
        - 'y_score': prediction scores (for classification)
        - 'model_name': name of the model (optional)
    tag : str
        Unique tag for the panel
    parent : int
        Parent DPG item ID
    width : int
        Total width
    height : int
        Plot height

    Returns
    -------
    panel_id : int
        DPG item ID of the created panel
    """
    task_type = results.get('task_type', 'regression')
    y_true = results['y_true']
    y_pred = results['y_pred']
    model_name = results.get('model_name', 'Model')

    with dpg.group(tag=tag, parent=parent, horizontal=True):
        if task_type == 'regression':
            # Regression: prediction vs actual plot
            plot_prediction_vs_actual(
                y_true=y_true,
                y_pred=y_pred,
                tag=f"{tag}_pred_vs_actual",
                parent=tag,
                title=f"{model_name}: Prediction vs Actual",
                width=width // 2,
                height=height
            )

            # Calculate and display metrics
            with dpg.group(parent=tag):
                dpg.add_spacer(height=20)

                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

                r2 = r2_score(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)

                dpg.add_text("Performance Metrics:", color=(255, 255, 255))
                dpg.add_spacer(height=10)
                dpg.add_text(f"R² Score: {r2:.4f}", color=(100, 200, 255))
                dpg.add_text(f"RMSE: {rmse:.4f}", color=(100, 200, 255))
                dpg.add_text(f"MAE: {mae:.4f}", color=(100, 200, 255))

        else:
            # Classification: confusion matrix
            n_classes = len(np.unique(y_true))

            plot_confusion_matrix(
                y_true=y_true,
                y_pred=y_pred,
                tag=f"{tag}_confusion",
                parent=tag,
                title=f"{model_name}: Confusion Matrix",
                width=width // 3,
                height=height
            )

            # Add ROC curve if binary classification and scores are provided
            if n_classes == 2 and 'y_score' in results:
                y_score = results['y_score']
                plot_roc_curve(
                    y_true=y_true,
                    y_score=y_score,
                    tag=f"{tag}_roc",
                    parent=tag,
                    title=f"{model_name}: ROC Curve",
                    width=width // 3,
                    height=height
                )

    return tag


def export_plot_to_png(plot_tag: str, filepath: str, dpi: int = 150):
    """
    Export a DPG plot to PNG file.

    Note: This is a placeholder. DearPyGui doesn't have built-in plot export.
    For production, you would need to:
    1. Extract data from the plot
    2. Use matplotlib to recreate and save
    3. Or use screen capture utilities

    Parameters
    ----------
    plot_tag : str
        Tag of the plot to export
    filepath : str
        Output file path
    dpi : int
        Resolution in dots per inch
    """
    # TODO: Implement proper plot export
    # For now, this is a placeholder that would need matplotlib integration
    print(f"Plot export requested: {plot_tag} -> {filepath} @ {dpi} DPI")
    print("Note: Plot export requires matplotlib integration (not yet implemented)")

    # Example implementation with matplotlib:
    # 1. Get plot data from DPG
    # 2. Create matplotlib figure
    # 3. Save with plt.savefig(filepath, dpi=dpi)

    return False
