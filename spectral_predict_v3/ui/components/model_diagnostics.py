"""
Model Diagnostics Panel for Spectral Predict v3.

Provides post-training visualizations:
- Predicted vs Actual scatter plot
- Residual diagnostics (vs fitted, vs index, Q-Q)
- Confusion matrix (classification)
- ROC curves (classification)
"""

import dearpygui.dearpygui as dpg
import numpy as np
from typing import Optional, List, Tuple, Dict
from ..theme import COLORS


class ModelDiagnosticsPanel:
    """
    Panel for displaying model diagnostic plots.

    Supports both regression and classification tasks with
    appropriate visualizations for each.
    """

    def __init__(self, parent_tag: str, tag: str = "model_diagnostics"):
        """
        Initialize the diagnostics panel.

        Parameters
        ----------
        parent_tag : str
            DearPyGui tag of parent container
        tag : str
            Base tag for this component's elements
        """
        self.parent_tag = parent_tag
        self.tag = tag

        # Data storage (single model - for backward compatibility)
        self._y_true: Optional[np.ndarray] = None
        self._y_pred: Optional[np.ndarray] = None
        self._sample_ids: Optional[List[str]] = None
        self._task_type: str = 'regression'
        self._model_name: str = ''

        # Multi-model storage
        self._all_models: List[Dict] = []  # List of {name, y_true, y_pred, task_type, sample_ids}
        self._current_model_index: int = 0

        # Current view
        self._current_view = 'pred_vs_actual'

        # Build UI
        self._build_ui()

    def _build_ui(self):
        """Build the diagnostics panel UI."""
        with dpg.group(parent=self.parent_tag, tag=f"{self.tag}_container", show=False):
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=5)

            # Header with model selector and view selector
            with dpg.group(horizontal=True):
                dpg.add_text("Model Diagnostics", color=COLORS["text_secondary"])
                dpg.add_spacer(width=15)

                # Model selector (hidden when single model)
                with dpg.group(horizontal=True, tag=f"{self.tag}_model_selector", show=False):
                    dpg.add_button(
                        label="<",
                        callback=self._on_prev_model,
                        tag=f"{self.tag}_prev_btn",
                        width=25
                    )
                    dpg.add_text(
                        "1 / 1",
                        tag=f"{self.tag}_model_index_label",
                        color=COLORS["accent_primary"]
                    )
                    dpg.add_button(
                        label=">",
                        callback=self._on_next_model,
                        tag=f"{self.tag}_next_btn",
                        width=25
                    )
                    dpg.add_spacer(width=15)

                dpg.add_text("View:", color=COLORS["text_muted"])
                dpg.add_combo(
                    items=["Predicted vs Actual", "Residuals", "Q-Q Plot"],
                    default_value="Predicted vs Actual",
                    callback=self._on_view_change,
                    tag=f"{self.tag}_view_combo",
                    width=150
                )
                dpg.add_spacer(width=15)
                dpg.add_text("", tag=f"{self.tag}_model_label", color=COLORS["text_muted"])

            dpg.add_spacer(height=5)

            # Metrics row
            dpg.add_text("", tag=f"{self.tag}_metrics", color=COLORS["accent_primary"])

            dpg.add_spacer(height=5)

            # Plot area - fills remaining space
            with dpg.child_window(height=-1, border=True, tag=f"{self.tag}_plot_container"):
                # DPG plot
                with dpg.plot(
                    label="",
                    height=-1,
                    width=-1,
                    tag=f"{self.tag}_plot",
                    anti_aliased=True
                ):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="", tag=f"{self.tag}_x_axis")
                    dpg.add_plot_axis(dpg.mvYAxis, label="", tag=f"{self.tag}_y_axis")

    def set_data(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: str = 'regression',
        model_name: str = '',
        sample_ids: Optional[List[str]] = None
    ):
        """
        Set the prediction data for diagnostics.

        Parameters
        ----------
        y_true : np.ndarray
            True target values
        y_pred : np.ndarray
            Predicted values
        task_type : str
            'regression' or 'classification'
        model_name : str
            Name of the model for display
        sample_ids : list of str, optional
            Sample identifiers
        """
        self._y_true = np.asarray(y_true)
        self._y_pred = np.asarray(y_pred)
        self._task_type = task_type
        self._model_name = model_name
        self._sample_ids = sample_ids

        # Update view options based on task type
        if task_type == 'classification':
            dpg.configure_item(
                f"{self.tag}_view_combo",
                items=["Confusion Matrix", "Class Distribution"]
            )
            dpg.set_value(f"{self.tag}_view_combo", "Confusion Matrix")
            self._current_view = 'confusion_matrix'
        else:
            dpg.configure_item(
                f"{self.tag}_view_combo",
                items=["Predicted vs Actual", "Residuals", "Q-Q Plot"]
            )
            dpg.set_value(f"{self.tag}_view_combo", "Predicted vs Actual")
            self._current_view = 'pred_vs_actual'

        # Update model label
        dpg.set_value(f"{self.tag}_model_label", f"Model: {model_name}")

        # Update metrics
        self._update_metrics()

        # Show container
        dpg.configure_item(f"{self.tag}_container", show=True)

        # Update plot
        self._update_plot()

    def _update_metrics(self):
        """Update the metrics display."""
        if self._y_true is None or self._y_pred is None:
            return

        if self._task_type == 'regression':
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            rmse = np.sqrt(mean_squared_error(self._y_true, self._y_pred))
            r2 = r2_score(self._y_true, self._y_pred)
            mae = mean_absolute_error(self._y_true, self._y_pred)
            n = len(self._y_true)

            metrics_text = f"R² = {r2:.4f}  |  RMSE = {rmse:.4f}  |  MAE = {mae:.4f}  |  n = {n}"
        else:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            acc = accuracy_score(self._y_true, self._y_pred)
            n = len(self._y_true)

            try:
                # Try multiclass averages
                prec = precision_score(self._y_true, self._y_pred, average='weighted', zero_division=0)
                rec = recall_score(self._y_true, self._y_pred, average='weighted', zero_division=0)
                f1 = f1_score(self._y_true, self._y_pred, average='weighted', zero_division=0)
                metrics_text = f"Accuracy = {acc:.3f}  |  Precision = {prec:.3f}  |  Recall = {rec:.3f}  |  F1 = {f1:.3f}  |  n = {n}"
            except:
                metrics_text = f"Accuracy = {acc:.3f}  |  n = {n}"

        dpg.set_value(f"{self.tag}_metrics", metrics_text)

    def _on_view_change(self, sender, app_data):
        """Handle view selection change."""
        view_map = {
            "Predicted vs Actual": "pred_vs_actual",
            "Residuals": "residuals",
            "Q-Q Plot": "qq_plot",
            "Confusion Matrix": "confusion_matrix",
            "Class Distribution": "class_dist"
        }
        self._current_view = view_map.get(app_data, "pred_vs_actual")
        self._update_plot()

    def _on_prev_model(self, sender=None, app_data=None):
        """Navigate to previous model."""
        if len(self._all_models) <= 1:
            return
        self._current_model_index = (self._current_model_index - 1) % len(self._all_models)
        self._display_current_model()

    def _on_next_model(self, sender=None, app_data=None):
        """Navigate to next model."""
        if len(self._all_models) <= 1:
            return
        self._current_model_index = (self._current_model_index + 1) % len(self._all_models)
        self._display_current_model()

    def set_multiple_models(self, models: List[Dict]):
        """
        Set data for multiple models, enabling toggle between them.

        Parameters
        ----------
        models : list of dict
            Each dict contains: {name, y_true, y_pred, task_type, sample_ids}
        """
        self._all_models = models
        self._current_model_index = 0

        # Update model selector visibility
        if len(models) > 1:
            dpg.configure_item(f"{self.tag}_model_selector", show=True)
        else:
            dpg.configure_item(f"{self.tag}_model_selector", show=False)

        # Display first model
        if models:
            self._display_current_model()

    def _display_current_model(self):
        """Display the currently selected model from the multi-model list."""
        if not self._all_models:
            return

        model = self._all_models[self._current_model_index]

        # Update model index label
        dpg.set_value(
            f"{self.tag}_model_index_label",
            f"{self._current_model_index + 1} / {len(self._all_models)}"
        )

        # Use existing set_data to display this model
        # But preserve the current view selection
        current_view_value = dpg.get_value(f"{self.tag}_view_combo")

        self.set_data(
            y_true=model['y_true'],
            y_pred=model['y_pred'],
            task_type=model.get('task_type', 'regression'),
            model_name=model.get('name', ''),
            sample_ids=model.get('sample_ids')
        )

        # Restore view selection if it's valid for this task type
        try:
            dpg.set_value(f"{self.tag}_view_combo", current_view_value)
            self._on_view_change(None, current_view_value)
        except:
            pass  # View might not be valid for this task type

    def _update_plot(self):
        """Update the plot based on current view."""
        if self._y_true is None or self._y_pred is None:
            return

        # Clear existing plot series
        self._clear_plot()

        if self._current_view == 'pred_vs_actual':
            self._plot_pred_vs_actual()
        elif self._current_view == 'residuals':
            self._plot_residuals()
        elif self._current_view == 'qq_plot':
            self._plot_qq()
        elif self._current_view == 'confusion_matrix':
            self._plot_confusion_matrix()
        elif self._current_view == 'class_dist':
            self._plot_class_distribution()

    def _clear_plot(self):
        """Clear all series from the plot."""
        # Delete existing series
        for series_tag in [
            f"{self.tag}_scatter",
            f"{self.tag}_line",
            f"{self.tag}_ref_line",
            f"{self.tag}_zero_line",
            f"{self.tag}_qq_line",
            f"{self.tag}_qq_ref",
        ]:
            if dpg.does_item_exist(series_tag):
                dpg.delete_item(series_tag)

        # Reset axis labels
        dpg.set_item_label(f"{self.tag}_x_axis", "")
        dpg.set_item_label(f"{self.tag}_y_axis", "")

    def _plot_pred_vs_actual(self):
        """Plot predicted vs actual values with 1:1 line."""
        y_true = self._y_true
        y_pred = self._y_pred

        # Set axis labels
        dpg.set_item_label(f"{self.tag}_x_axis", "Actual")
        dpg.set_item_label(f"{self.tag}_y_axis", "Predicted")

        # Compute axis limits
        all_vals = np.concatenate([y_true, y_pred])
        min_val = np.min(all_vals)
        max_val = np.max(all_vals)
        margin = (max_val - min_val) * 0.05

        # Add scatter plot
        dpg.add_scatter_series(
            x=list(y_true),
            y=list(y_pred),
            label="Samples",
            parent=f"{self.tag}_y_axis",
            tag=f"{self.tag}_scatter"
        )

        # Add 1:1 reference line
        dpg.add_line_series(
            x=[min_val - margin, max_val + margin],
            y=[min_val - margin, max_val + margin],
            label="1:1 Line",
            parent=f"{self.tag}_y_axis",
            tag=f"{self.tag}_ref_line"
        )

        # Configure line color (red dashed)
        dpg.bind_item_theme(f"{self.tag}_ref_line", self._get_line_theme())

        # Fit axes
        dpg.fit_axis_data(f"{self.tag}_x_axis")
        dpg.fit_axis_data(f"{self.tag}_y_axis")

    def _plot_residuals(self):
        """Plot residuals vs fitted values."""
        y_pred = self._y_pred
        residuals = self._y_true - self._y_pred

        # Set axis labels
        dpg.set_item_label(f"{self.tag}_x_axis", "Fitted Values")
        dpg.set_item_label(f"{self.tag}_y_axis", "Residuals")

        # Add scatter plot
        dpg.add_scatter_series(
            x=list(y_pred),
            y=list(residuals),
            label="Residuals",
            parent=f"{self.tag}_y_axis",
            tag=f"{self.tag}_scatter"
        )

        # Add zero reference line
        min_fitted = np.min(y_pred)
        max_fitted = np.max(y_pred)
        dpg.add_line_series(
            x=[min_fitted, max_fitted],
            y=[0, 0],
            label="Zero",
            parent=f"{self.tag}_y_axis",
            tag=f"{self.tag}_zero_line"
        )

        dpg.bind_item_theme(f"{self.tag}_zero_line", self._get_line_theme())

        dpg.fit_axis_data(f"{self.tag}_x_axis")
        dpg.fit_axis_data(f"{self.tag}_y_axis")

    def _plot_qq(self):
        """Plot Q-Q plot for residual normality."""
        from scipy import stats

        residuals = self._y_true - self._y_pred

        # Set axis labels
        dpg.set_item_label(f"{self.tag}_x_axis", "Theoretical Quantiles")
        dpg.set_item_label(f"{self.tag}_y_axis", "Sample Quantiles")

        # Compute Q-Q values
        sorted_residuals = np.sort(residuals)
        n = len(residuals)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, n))

        # Standardize residuals
        standardized = (sorted_residuals - np.mean(residuals)) / np.std(residuals)

        # Add scatter
        dpg.add_scatter_series(
            x=list(theoretical_quantiles),
            y=list(standardized),
            label="Residuals",
            parent=f"{self.tag}_y_axis",
            tag=f"{self.tag}_scatter"
        )

        # Add reference diagonal
        min_q = np.min(theoretical_quantiles)
        max_q = np.max(theoretical_quantiles)
        dpg.add_line_series(
            x=[min_q, max_q],
            y=[min_q, max_q],
            label="Normal",
            parent=f"{self.tag}_y_axis",
            tag=f"{self.tag}_qq_ref"
        )

        dpg.bind_item_theme(f"{self.tag}_qq_ref", self._get_line_theme())

        dpg.fit_axis_data(f"{self.tag}_x_axis")
        dpg.fit_axis_data(f"{self.tag}_y_axis")

    def _plot_confusion_matrix(self):
        """Plot confusion matrix as text display (DPG doesn't have heatmap)."""
        from sklearn.metrics import confusion_matrix

        # Set axis labels
        dpg.set_item_label(f"{self.tag}_x_axis", "Predicted Class")
        dpg.set_item_label(f"{self.tag}_y_axis", "True Class")

        # Compute confusion matrix
        labels = np.unique(np.concatenate([self._y_true, self._y_pred]))
        cm = confusion_matrix(self._y_true, self._y_pred, labels=labels)

        n_classes = len(labels)

        # Plot each cell as a point with size proportional to count
        x_coords = []
        y_coords = []

        for i in range(n_classes):
            for j in range(n_classes):
                x_coords.append(j)
                y_coords.append(n_classes - 1 - i)  # Flip y for matrix orientation

        dpg.add_scatter_series(
            x=x_coords,
            y=y_coords,
            label="Confusion Matrix",
            parent=f"{self.tag}_y_axis",
            tag=f"{self.tag}_scatter"
        )

        # Set axis limits
        dpg.set_axis_limits(f"{self.tag}_x_axis", -0.5, n_classes - 0.5)
        dpg.set_axis_limits(f"{self.tag}_y_axis", -0.5, n_classes - 0.5)

    def _plot_class_distribution(self):
        """Plot class distribution bar chart."""
        # Set axis labels
        dpg.set_item_label(f"{self.tag}_x_axis", "Class")
        dpg.set_item_label(f"{self.tag}_y_axis", "Count")

        labels, counts = np.unique(self._y_true, return_counts=True)

        # Create bar-like scatter (DPG limitation)
        dpg.add_stem_series(
            x=list(range(len(labels))),
            y=list(counts),
            label="True Distribution",
            parent=f"{self.tag}_y_axis",
            tag=f"{self.tag}_scatter"
        )

        dpg.fit_axis_data(f"{self.tag}_x_axis")
        dpg.fit_axis_data(f"{self.tag}_y_axis")

    def _get_line_theme(self):
        """Get or create theme for reference lines."""
        theme_tag = f"{self.tag}_line_theme"
        if not dpg.does_item_exist(theme_tag):
            with dpg.theme(tag=theme_tag):
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 100, 100, 200))
        return theme_tag

    def hide(self):
        """Hide the diagnostics panel."""
        dpg.configure_item(f"{self.tag}_container", show=False)

    def show(self):
        """Show the diagnostics panel."""
        dpg.configure_item(f"{self.tag}_container", show=True)

    def is_visible(self) -> bool:
        """Check if panel is visible."""
        return dpg.is_item_visible(f"{self.tag}_container")

    def clear(self):
        """Clear all data and hide."""
        self._y_true = None
        self._y_pred = None
        self._sample_ids = None
        self._clear_plot()
        self.hide()
