"""
Data Quality Panel for Spectral Predict v3.

Comprehensive outlier detection and imbalance handling with:
- Interactive T² and Q-residual plots with threshold lines
- Outlier table with sample details and selection checkboxes
- Controls for PCA components and confidence level
- Flag/Remove/Clear actions
- Imbalance detection with method recommendations
"""

import dearpygui.dearpygui as dpg
import numpy as np
from typing import Optional, List, Callable, Set
from dataclasses import dataclass

from ..theme import COLORS
from ..tooltips import add_tooltip, TOOLTIP_CONTENT
from ...core.outlier_detection import (
    run_pca_outlier_detection,
    compute_q_residuals,
    compute_mahalanobis_distance,
    check_y_data_consistency,
    OutlierReport,
    generate_outlier_report
)
from ...core.imbalance import (
    detect_class_imbalance,
    detect_regression_imbalance,
    detect_multiclass_imbalance,
    ClassImbalanceResult,
    RegressionImbalanceResult,
    get_available_methods,
    get_method_info,
    HAS_IMBLEARN
)


@dataclass
class OutlierDetectionResults:
    """Results from outlier detection analysis."""
    t2_values: np.ndarray
    t2_threshold: float
    t2_outliers: np.ndarray  # boolean mask
    q_values: np.ndarray
    q_threshold: float
    q_outliers: np.ndarray  # boolean mask
    mahalanobis: np.ndarray
    maha_threshold: float
    maha_outliers: np.ndarray
    y_zscores: Optional[np.ndarray] = None  # Z-scores for Y values
    y_outliers: Optional[np.ndarray] = None  # boolean mask for Y outliers
    combined_outliers: np.ndarray = None  # indices
    high_confidence: np.ndarray = None  # indices (3+ methods)
    moderate_confidence: np.ndarray = None  # indices (2 methods)
    low_confidence: np.ndarray = None  # indices (1 method)


class DataQualityPanel:
    """
    Comprehensive data quality analysis panel.

    Features:
    - T² and Q-residual scatter plots with threshold lines
    - Outlier table with checkboxes for selection
    - Controls for PCA components and confidence level
    - Flag/Remove/Clear actions
    - Imbalance detection with recommendations

    Example
    -------
    >>> panel = DataQualityPanel(parent="container", on_flag=callback)
    >>> panel.set_data(dataset)
    >>> panel.run_analysis()
    """

    def __init__(
        self,
        parent: str,
        tag: str = "data_quality_panel",
        on_flag: Optional[Callable] = None,
        on_remove: Optional[Callable] = None
    ):
        """
        Initialize the data quality panel.

        Parameters
        ----------
        parent : str
            Parent container tag
        tag : str
            Unique tag for this panel
        on_flag : callable, optional
            Callback when samples are flagged (receives list of indices)
        on_remove : callable, optional
            Callback when samples should be removed (receives list of indices)
        """
        self.parent = parent
        self.tag = tag
        self.on_flag = on_flag
        self.on_remove = on_remove

        self._dataset = None
        self._results: Optional[OutlierDetectionResults] = None
        self._selected_outliers: Set[int] = set()
        self._flagged_samples: Set[int] = set()

        # Detection parameters
        self._n_components = 5
        self._confidence_level = 0.95  # 95%
        self._current_plot = "t2"  # Currently visible plot

        self._create_ui()

    def _create_ui(self):
        """Create the panel UI structure."""
        with dpg.child_window(parent=self.parent, tag=self.tag, border=False):
            # === OUTLIER DETECTION SECTION ===
            dpg.add_text("OUTLIER DETECTION", color=COLORS["text_secondary"])
            dpg.add_separator()
            dpg.add_spacer(height=5)

            # Controls row
            with dpg.group(horizontal=True):
                dpg.add_text("PCA Components:", color=COLORS["text_muted"])
                dpg.add_input_int(
                    default_value=5,
                    min_value=2,
                    max_value=20,
                    min_clamped=True,
                    max_clamped=True,
                    tag=f"{self.tag}_n_components",
                    width=60,
                    callback=self._on_params_change
                )
                dpg.add_spacer(width=10)
                dpg.add_text("Confidence:", color=COLORS["text_muted"])
                dpg.add_combo(
                    items=["95%", "99%", "99.9%"],
                    default_value="95%",
                    tag=f"{self.tag}_confidence",
                    width=70,
                    callback=self._on_params_change
                )

            dpg.add_spacer(height=5)

            # Re-run button and status (analysis runs automatically on data load)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Re-run Analysis",
                    callback=self._run_outlier_detection,
                    tag=f"{self.tag}_run_btn",
                    width=120
                )
                dpg.add_text(
                    "Load data to begin",
                    tag=f"{self.tag}_status",
                    color=COLORS["text_muted"]
                )

            dpg.add_spacer(height=10)

            # Summary stats
            dpg.add_text(
                "",
                tag=f"{self.tag}_summary",
                color=COLORS["text_primary"],
                wrap=0
            )

            dpg.add_spacer(height=10)

            # === PLOT TOGGLES ===
            dpg.add_text("Plots:", color=COLORS["text_muted"])
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="T²",
                    tag=f"{self.tag}_btn_t2",
                    callback=self._on_plot_btn_t2,
                    width=50
                )
                dpg.add_button(
                    label="Q",
                    tag=f"{self.tag}_btn_q",
                    callback=self._on_plot_btn_q,
                    width=50
                )
                dpg.add_button(
                    label="Mahal",
                    tag=f"{self.tag}_btn_maha",
                    callback=self._on_plot_btn_maha,
                    width=50
                )
                dpg.add_button(
                    label="Y-Check",
                    tag=f"{self.tag}_btn_y",
                    callback=self._on_plot_btn_y,
                    width=60
                )

            # Set T² as active initially
            self._show_plot("t2")

            dpg.add_spacer(height=5)

            # === PLOTS (only one visible at a time) ===
            # T² Plot
            with dpg.child_window(width=-1, height=200, border=True, tag=f"{self.tag}_t2_container", show=True):
                dpg.add_text("Hotelling T² (PCA distance)", color=COLORS["text_muted"])
                with dpg.plot(
                    tag=f"{self.tag}_t2_plot",
                    height=170,
                    width=-1,
                    anti_aliased=True,
                    no_title=True
                ):
                    dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthEast)
                    dpg.add_plot_axis(dpg.mvXAxis, label="Sample", tag=f"{self.tag}_t2_x")
                    dpg.add_plot_axis(dpg.mvYAxis, label="T²", tag=f"{self.tag}_t2_y")

            # Q-Residuals Plot
            with dpg.child_window(width=-1, height=200, border=True, tag=f"{self.tag}_q_container", show=False):
                dpg.add_text("Q-Residuals (reconstruction error)", color=COLORS["text_muted"])
                with dpg.plot(
                    tag=f"{self.tag}_q_plot",
                    height=170,
                    width=-1,
                    anti_aliased=True,
                    no_title=True
                ):
                    dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthEast)
                    dpg.add_plot_axis(dpg.mvXAxis, label="Sample", tag=f"{self.tag}_q_x")
                    dpg.add_plot_axis(dpg.mvYAxis, label="Q", tag=f"{self.tag}_q_y")

            # Mahalanobis Distance Plot
            with dpg.child_window(width=-1, height=200, border=True, tag=f"{self.tag}_maha_container", show=False):
                dpg.add_text("Mahalanobis Distance", color=COLORS["text_muted"])
                with dpg.plot(
                    tag=f"{self.tag}_maha_plot",
                    height=170,
                    width=-1,
                    anti_aliased=True,
                    no_title=True
                ):
                    dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthEast)
                    dpg.add_plot_axis(dpg.mvXAxis, label="Sample", tag=f"{self.tag}_maha_x")
                    dpg.add_plot_axis(dpg.mvYAxis, label="Distance", tag=f"{self.tag}_maha_y")

            # Y-Consistency Plot
            with dpg.child_window(width=-1, height=200, border=True, tag=f"{self.tag}_y_container", show=False):
                dpg.add_text("Y-Value Consistency (Z-scores)", color=COLORS["text_muted"])
                with dpg.plot(
                    tag=f"{self.tag}_y_plot",
                    height=170,
                    width=-1,
                    anti_aliased=True,
                    no_title=True
                ):
                    dpg.add_plot_legend(location=dpg.mvPlot_Location_NorthEast)
                    dpg.add_plot_axis(dpg.mvXAxis, label="Sample", tag=f"{self.tag}_y_x")
                    dpg.add_plot_axis(dpg.mvYAxis, label="Z-score", tag=f"{self.tag}_y_y")

            dpg.add_spacer(height=10)

            # === OUTLIER TABLE ===
            dpg.add_text("Detected Outliers", color=COLORS["text_secondary"])
            dpg.add_spacer(height=5)

            # Action buttons
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Select All",
                    callback=self._select_all_outliers,
                    tag=f"{self.tag}_select_all_btn",
                    enabled=False
                )
                dpg.add_button(
                    label="Clear Selection",
                    callback=self._clear_selection,
                    tag=f"{self.tag}_clear_sel_btn",
                    enabled=False
                )
                dpg.add_spacer(width=20)
                dpg.add_button(
                    label="Flag Selected",
                    callback=self._flag_selected,
                    tag=f"{self.tag}_flag_btn",
                    enabled=False
                )
                dpg.add_button(
                    label="Unflag Selected",
                    callback=self._unflag_selected,
                    tag=f"{self.tag}_unflag_btn",
                    enabled=False
                )
                dpg.add_button(
                    label="Remove Flagged",
                    callback=self._remove_flagged,
                    tag=f"{self.tag}_remove_btn",
                    enabled=False
                )

            dpg.add_spacer(height=5)

            # Outlier table
            with dpg.child_window(height=150, border=True, tag=f"{self.tag}_table_container"):
                with dpg.table(
                    tag=f"{self.tag}_outlier_table",
                    header_row=True,
                    borders_innerH=True,
                    borders_outerH=True,
                    borders_innerV=True,
                    borders_outerV=True,
                    scrollY=True,
                    resizable=True,
                    policy=dpg.mvTable_SizingStretchProp
                ):
                    dpg.add_table_column(label="", width_fixed=True, init_width_or_weight=30)  # Checkbox
                    dpg.add_table_column(label="Sample ID", init_width_or_weight=80)
                    dpg.add_table_column(label="T²", init_width_or_weight=60)
                    dpg.add_table_column(label="Q", init_width_or_weight=60)
                    dpg.add_table_column(label="Mahal.", init_width_or_weight=60)
                    dpg.add_table_column(label="Confidence", init_width_or_weight=80)

            dpg.add_spacer(height=15)

            # === IMBALANCE SECTION ===
            dpg.add_text("TARGET IMBALANCE", color=COLORS["text_secondary"])
            dpg.add_separator()
            dpg.add_spacer(height=5)

            dpg.add_text(
                "Load data with target to analyze",
                tag=f"{self.tag}_imbalance_status",
                color=COLORS["text_muted"],
                wrap=0
            )

            dpg.add_spacer(height=5)

            # Imbalance details (hidden until analysis runs)
            with dpg.group(tag=f"{self.tag}_imbalance_details", show=False):
                dpg.add_text("", tag=f"{self.tag}_imbalance_info", wrap=0)
                dpg.add_spacer(height=5)
                dpg.add_text("Recommendation:", color=COLORS["text_secondary"])
                dpg.add_text(
                    "",
                    tag=f"{self.tag}_imbalance_recommendation",
                    color=COLORS["accent_primary"],
                    wrap=0
                )

                dpg.add_spacer(height=10)
                dpg.add_text("Handling Method:", color=COLORS["text_muted"])
                # Build dynamic method list from get_available_methods
                classification_methods = get_available_methods('classification')
                method_items = ["None (use as-is)"] + [desc for _, desc in classification_methods]
                dpg.add_combo(
                    items=method_items,
                    default_value="None (use as-is)",
                    tag=f"{self.tag}_imbalance_method",
                    width=-1,
                    callback=self._on_imbalance_method_change
                )
                dpg.add_spacer(height=5)
                dpg.add_text(
                    "",
                    tag=f"{self.tag}_method_description",
                    color=COLORS["text_muted"],
                    wrap=400
                )

        # Set up tooltips
        self._setup_tooltips()

    def _setup_tooltips(self):
        """Set up tooltips for data quality panel elements."""
        # PCA components
        add_tooltip(f"{self.tag}_n_components", TOOLTIP_CONTENT['outlier_detection']['n_components'])

        # Confidence level
        add_tooltip(f"{self.tag}_confidence", TOOLTIP_CONTENT['outlier_detection']['confidence_level'])

        # Re-run button
        add_tooltip(f"{self.tag}_run_btn",
            "Run outlier detection analysis with current parameters. "
            "Analysis runs automatically when data is loaded.")

        # Plot buttons
        add_tooltip(f"{self.tag}_btn_t2", TOOLTIP_CONTENT['outlier_detection']['t2'])
        add_tooltip(f"{self.tag}_btn_q", TOOLTIP_CONTENT['outlier_detection']['q_residuals'])
        add_tooltip(f"{self.tag}_btn_maha", TOOLTIP_CONTENT['outlier_detection']['mahalanobis'])
        add_tooltip(f"{self.tag}_btn_y", TOOLTIP_CONTENT['outlier_detection']['y_consistency'])

        # Action buttons
        add_tooltip(f"{self.tag}_select_all_btn", "Select all detected outliers in the table.")
        add_tooltip(f"{self.tag}_clear_sel_btn", "Clear current selection without changing flags.")
        add_tooltip(f"{self.tag}_flag_btn",
            "Flag selected samples as potential outliers. "
            "Flagged samples are highlighted but not removed.")
        add_tooltip(f"{self.tag}_unflag_btn", "Remove flag from selected samples.")
        add_tooltip(f"{self.tag}_remove_btn",
            "Remove all flagged samples from the dataset. "
            "This action cannot be undone.")

        # Imbalance handling
        add_tooltip(f"{self.tag}_imbalance_method",
            "Method to handle class imbalance or target distribution issues. "
            "SMOTE and ADASYN create synthetic samples. "
            "Class weights adjust model training.")

    def set_data(self, dataset):
        """
        Set the dataset to analyze.

        Parameters
        ----------
        dataset : SpectralDataset
            The dataset to analyze
        """
        self._dataset = dataset
        self._results = None
        self._selected_outliers.clear()
        self._flagged_samples.clear()

        # Reset UI
        self._clear_plots()
        self._clear_table()

        if dataset is None:
            dpg.set_value(f"{self.tag}_status", "No data loaded")
            dpg.set_value(f"{self.tag}_summary", "")
            dpg.set_value(f"{self.tag}_imbalance_status", "No data loaded")
            dpg.configure_item(f"{self.tag}_imbalance_details", show=False)
        else:
            dpg.set_value(f"{self.tag}_status", f"Analyzing ({dataset.n_samples} samples)...")
            dpg.set_value(f"{self.tag}_summary", "")

            # Run outlier detection automatically
            self._run_outlier_detection()

            # Run imbalance check automatically
            self._check_imbalance()

    def _on_params_change(self, sender=None, app_data=None):
        """Handle parameter change - update stored values."""
        self._n_components = dpg.get_value(f"{self.tag}_n_components")

        conf_str = dpg.get_value(f"{self.tag}_confidence")
        conf_map = {"95%": 0.95, "99%": 0.99, "99.9%": 0.999}
        self._confidence_level = conf_map.get(conf_str, 0.95)

    def _on_imbalance_method_change(self, sender=None, app_data=None):
        """Handle imbalance method selection - show detailed description."""
        selected = dpg.get_value(f"{self.tag}_imbalance_method")

        if selected == "None (use as-is)" or not selected:
            dpg.set_value(f"{self.tag}_method_description", "")
            return

        # Determine task type from current dataset
        task_type = 'classification'
        if self._dataset and self._dataset.metadata:
            task_type = self._dataset.metadata.get('target_type', 'regression')

        # Find method key from selected description
        methods = get_available_methods(task_type)
        method_key = None
        for key, desc in methods:
            if desc == selected:
                method_key = key
                break

        if method_key:
            info = get_method_info(method_key, task_type)

            # Build detailed description text
            desc_lines = [
                info['description'],
                "",
                f"WHEN TO USE: {info['when_to_use']}",
                "",
                "PROS:",
            ]
            for pro in info.get('pros', []):
                desc_lines.append(f"  + {pro}")

            desc_lines.append("")
            desc_lines.append("CONS:")
            for con in info.get('cons', []):
                desc_lines.append(f"  - {con}")

            if info.get('key_params'):
                desc_lines.append("")
                desc_lines.append("KEY PARAMETERS:")
                for param, param_desc in info['key_params'].items():
                    desc_lines.append(f"  {param}: {param_desc}")

            dpg.set_value(f"{self.tag}_method_description", "\n".join(desc_lines))
        else:
            dpg.set_value(f"{self.tag}_method_description", "")

    def _on_plot_btn_t2(self, sender=None, app_data=None):
        """Switch to T² plot."""
        self._show_plot("t2")

    def _on_plot_btn_q(self, sender=None, app_data=None):
        """Switch to Q-residuals plot."""
        self._show_plot("q")

    def _on_plot_btn_maha(self, sender=None, app_data=None):
        """Switch to Mahalanobis plot."""
        self._show_plot("maha")

    def _on_plot_btn_y(self, sender=None, app_data=None):
        """Switch to Y-consistency plot."""
        self._show_plot("y")

    def _show_plot(self, plot_type: str):
        """Show the specified plot and hide others."""
        self._current_plot = plot_type

        # Hide all plot containers
        for pt in ["t2", "q", "maha", "y"]:
            container_tag = f"{self.tag}_{pt}_container"
            if dpg.does_item_exist(container_tag):
                dpg.configure_item(container_tag, show=(pt == plot_type))

        # Update button styles to show active state
        for pt in ["t2", "q", "maha", "y"]:
            btn_tag = f"{self.tag}_btn_{pt}"
            if dpg.does_item_exist(btn_tag):
                if pt == plot_type:
                    # Active button style
                    with dpg.theme() as active_theme:
                        with dpg.theme_component(dpg.mvButton):
                            dpg.add_theme_color(dpg.mvThemeCol_Button, COLORS["accent_primary"], category=dpg.mvThemeCat_Core)
                    dpg.bind_item_theme(btn_tag, active_theme)
                else:
                    # Inactive button - unbind theme
                    dpg.bind_item_theme(btn_tag, 0)

    def _run_outlier_detection(self, sender=None, app_data=None):
        """Run comprehensive outlier detection."""
        if self._dataset is None:
            dpg.set_value(f"{self.tag}_status", "No data loaded")
            return

        dpg.set_value(f"{self.tag}_status", "Running analysis...")

        try:
            ds = self._dataset
            X = ds.X
            y = ds.y if ds.has_target else np.zeros(ds.n_samples)

            n_components = min(self._n_components, ds.n_samples - 1, ds.n_wavelengths)

            # Run PCA outlier detection
            pca_result = run_pca_outlier_detection(X, y, n_components=n_components)

            # Compute Q-residuals
            q_result = compute_q_residuals(X, pca_result.pca_model, n_components)

            # Compute Mahalanobis distance
            maha_result = compute_mahalanobis_distance(pca_result.scores)

            # Y-consistency check
            y_zscores = None
            y_outliers = None
            if ds.has_target:
                y_result = check_y_data_consistency(y)
                y_zscores = y_result.z_scores
                y_outliers = y_result.all_outliers
            else:
                y_result = None

            # Combine flags
            flags_sum = (
                pca_result.outlier_flags.astype(int) +
                q_result.outlier_flags.astype(int) +
                maha_result.outlier_flags.astype(int)
            )
            if y_result is not None:
                flags_sum += y_result.all_outliers.astype(int)

            # Confidence levels
            high_conf = np.where(flags_sum >= 3)[0]
            mod_conf = np.where(flags_sum == 2)[0]
            low_conf = np.where(flags_sum == 1)[0]
            combined = np.where(flags_sum >= 2)[0]

            self._results = OutlierDetectionResults(
                t2_values=pca_result.hotelling_t2,
                t2_threshold=pca_result.t2_threshold,
                t2_outliers=pca_result.outlier_flags,
                q_values=q_result.q_residuals,
                q_threshold=q_result.q_threshold,
                q_outliers=q_result.outlier_flags,
                mahalanobis=maha_result.distances,
                maha_threshold=maha_result.threshold,
                maha_outliers=maha_result.outlier_flags,
                y_zscores=y_zscores,
                y_outliers=y_outliers,
                combined_outliers=combined,
                high_confidence=high_conf,
                moderate_confidence=mod_conf,
                low_confidence=low_conf
            )

            # Update UI
            self._update_plots()
            self._update_table()
            self._update_summary()

            dpg.set_value(f"{self.tag}_status", "Analysis complete")

            # Enable buttons
            has_outliers = len(combined) > 0
            dpg.configure_item(f"{self.tag}_select_all_btn", enabled=has_outliers)
            dpg.configure_item(f"{self.tag}_clear_sel_btn", enabled=has_outliers)

        except Exception as e:
            dpg.set_value(f"{self.tag}_status", f"Error: {str(e)[:50]}")
            import traceback
            traceback.print_exc()

    def _update_plots(self):
        """Update T² and Q-residual plots."""
        if self._results is None:
            return

        results = self._results
        n_samples = len(results.t2_values)
        x_indices = list(range(n_samples))

        # Clear existing series
        self._clear_plots()

        # === T² Plot ===
        t2_y_axis = f"{self.tag}_t2_y"

        # Normal points
        normal_mask = ~results.t2_outliers
        if np.any(normal_mask):
            normal_x = [i for i, m in enumerate(normal_mask) if m]
            normal_y = [results.t2_values[i] for i in normal_x]
            dpg.add_scatter_series(
                x=normal_x, y=normal_y,
                label="Normal",
                parent=t2_y_axis,
                tag=f"{self.tag}_t2_normal"
            )
            self._apply_scatter_theme(f"{self.tag}_t2_normal", (79, 195, 247, 255))  # Cyan

        # Outlier points
        outlier_mask = results.t2_outliers
        if np.any(outlier_mask):
            outlier_x = [i for i, m in enumerate(outlier_mask) if m]
            outlier_y = [results.t2_values[i] for i in outlier_x]
            dpg.add_scatter_series(
                x=outlier_x, y=outlier_y,
                label="Outlier",
                parent=t2_y_axis,
                tag=f"{self.tag}_t2_outlier"
            )
            self._apply_scatter_theme(f"{self.tag}_t2_outlier", (244, 67, 54, 255))  # Red

        # Threshold line
        dpg.add_line_series(
            x=[0, n_samples - 1],
            y=[results.t2_threshold, results.t2_threshold],
            label=f"Threshold ({self._confidence_level*100:.0f}%)",
            parent=t2_y_axis,
            tag=f"{self.tag}_t2_threshold"
        )
        self._apply_line_theme(f"{self.tag}_t2_threshold", (255, 152, 0, 255))  # Orange

        dpg.fit_axis_data(f"{self.tag}_t2_x")
        dpg.fit_axis_data(f"{self.tag}_t2_y")

        # === Q-Residuals Plot ===
        q_y_axis = f"{self.tag}_q_y"

        # Normal points
        normal_mask = ~results.q_outliers
        if np.any(normal_mask):
            normal_x = [i for i, m in enumerate(normal_mask) if m]
            normal_y = [results.q_values[i] for i in normal_x]
            dpg.add_scatter_series(
                x=normal_x, y=normal_y,
                label="Normal",
                parent=q_y_axis,
                tag=f"{self.tag}_q_normal"
            )
            self._apply_scatter_theme(f"{self.tag}_q_normal", (129, 199, 132, 255))  # Green

        # Outlier points
        outlier_mask = results.q_outliers
        if np.any(outlier_mask):
            outlier_x = [i for i, m in enumerate(outlier_mask) if m]
            outlier_y = [results.q_values[i] for i in outlier_x]
            dpg.add_scatter_series(
                x=outlier_x, y=outlier_y,
                label="Outlier",
                parent=q_y_axis,
                tag=f"{self.tag}_q_outlier"
            )
            self._apply_scatter_theme(f"{self.tag}_q_outlier", (244, 67, 54, 255))  # Red

        # Threshold line
        dpg.add_line_series(
            x=[0, n_samples - 1],
            y=[results.q_threshold, results.q_threshold],
            label="95th percentile",
            parent=q_y_axis,
            tag=f"{self.tag}_q_threshold"
        )
        self._apply_line_theme(f"{self.tag}_q_threshold", (255, 152, 0, 255))  # Orange

        dpg.fit_axis_data(f"{self.tag}_q_x")
        dpg.fit_axis_data(f"{self.tag}_q_y")

        # === Mahalanobis Plot ===
        maha_y_axis = f"{self.tag}_maha_y"

        # Normal points
        normal_mask = ~results.maha_outliers
        if np.any(normal_mask):
            normal_x = [i for i, m in enumerate(normal_mask) if m]
            normal_y = [results.mahalanobis[i] for i in normal_x]
            dpg.add_scatter_series(
                x=normal_x, y=normal_y,
                label="Normal",
                parent=maha_y_axis,
                tag=f"{self.tag}_maha_normal"
            )
            self._apply_scatter_theme(f"{self.tag}_maha_normal", (255, 193, 7, 255))  # Amber

        # Outlier points
        if np.any(results.maha_outliers):
            outlier_x = [i for i, m in enumerate(results.maha_outliers) if m]
            outlier_y = [results.mahalanobis[i] for i in outlier_x]
            dpg.add_scatter_series(
                x=outlier_x, y=outlier_y,
                label="Outlier",
                parent=maha_y_axis,
                tag=f"{self.tag}_maha_outlier"
            )
            self._apply_scatter_theme(f"{self.tag}_maha_outlier", (244, 67, 54, 255))  # Red

        # Threshold line
        dpg.add_line_series(
            x=[0, n_samples - 1],
            y=[results.maha_threshold, results.maha_threshold],
            label="3×MAD threshold",
            parent=maha_y_axis,
            tag=f"{self.tag}_maha_threshold"
        )
        self._apply_line_theme(f"{self.tag}_maha_threshold", (255, 152, 0, 255))  # Orange

        dpg.fit_axis_data(f"{self.tag}_maha_x")
        dpg.fit_axis_data(f"{self.tag}_maha_y")

        # === Y-Consistency Plot ===
        if results.y_zscores is not None:
            y_y_axis = f"{self.tag}_y_y"

            # Normal points
            y_normal_mask = ~results.y_outliers if results.y_outliers is not None else np.ones(n_samples, dtype=bool)
            if np.any(y_normal_mask):
                normal_x = [i for i, m in enumerate(y_normal_mask) if m]
                normal_y = [results.y_zscores[i] for i in normal_x]
                dpg.add_scatter_series(
                    x=normal_x, y=normal_y,
                    label="Normal",
                    parent=y_y_axis,
                    tag=f"{self.tag}_y_normal"
                )
                self._apply_scatter_theme(f"{self.tag}_y_normal", (156, 39, 176, 255))  # Purple

            # Outlier points
            if results.y_outliers is not None and np.any(results.y_outliers):
                outlier_x = [i for i, m in enumerate(results.y_outliers) if m]
                outlier_y = [results.y_zscores[i] for i in outlier_x]
                dpg.add_scatter_series(
                    x=outlier_x, y=outlier_y,
                    label="Outlier (|z|>3)",
                    parent=y_y_axis,
                    tag=f"{self.tag}_y_outlier"
                )
                self._apply_scatter_theme(f"{self.tag}_y_outlier", (244, 67, 54, 255))  # Red

            # Threshold lines at ±3
            dpg.add_line_series(
                x=[0, n_samples - 1],
                y=[3.0, 3.0],
                label="+3σ",
                parent=y_y_axis,
                tag=f"{self.tag}_y_thresh_upper"
            )
            self._apply_line_theme(f"{self.tag}_y_thresh_upper", (255, 152, 0, 255))  # Orange

            dpg.add_line_series(
                x=[0, n_samples - 1],
                y=[-3.0, -3.0],
                label="-3σ",
                parent=y_y_axis,
                tag=f"{self.tag}_y_thresh_lower"
            )
            self._apply_line_theme(f"{self.tag}_y_thresh_lower", (255, 152, 0, 255))  # Orange

            dpg.fit_axis_data(f"{self.tag}_y_x")
            dpg.fit_axis_data(f"{self.tag}_y_y")

    def _apply_scatter_theme(self, tag: str, color: tuple):
        """Apply a color theme to a scatter series."""
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, color, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, color, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 4, category=dpg.mvThemeCat_Plots)
        dpg.bind_item_theme(tag, theme)

    def _apply_line_theme(self, tag: str, color: tuple):
        """Apply a color theme to a line series."""
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2, category=dpg.mvThemeCat_Plots)
        dpg.bind_item_theme(tag, theme)

    def _clear_plots(self):
        """Clear all plot series."""
        for suffix in ['_t2_normal', '_t2_outlier', '_t2_threshold',
                       '_q_normal', '_q_outlier', '_q_threshold',
                       '_maha_normal', '_maha_outlier', '_maha_threshold',
                       '_y_normal', '_y_outlier', '_y_thresh_upper', '_y_thresh_lower']:
            tag = f"{self.tag}{suffix}"
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)

    def _update_table(self):
        """Update the outlier table."""
        self._clear_table()

        if self._results is None or self._dataset is None:
            return

        results = self._results
        ds = self._dataset

        # Get all outliers (combined: flagged by 2+ methods)
        outlier_indices = list(results.combined_outliers)

        # Sort by confidence (high first)
        def get_confidence(idx):
            if idx in results.high_confidence:
                return 0
            elif idx in results.moderate_confidence:
                return 1
            else:
                return 2

        outlier_indices.sort(key=get_confidence)

        table_tag = f"{self.tag}_outlier_table"

        for idx in outlier_indices:
            with dpg.table_row(parent=table_tag):
                # Checkbox
                dpg.add_checkbox(
                    default_value=idx in self._selected_outliers,
                    callback=self._on_outlier_checkbox,
                    user_data=idx,
                    tag=f"{self.tag}_check_{idx}"
                )

                # Sample ID
                sample_id = ds.sample_ids[idx] if idx < len(ds.sample_ids) else str(idx)
                dpg.add_text(str(sample_id))

                # T² value (red if outlier)
                t2_val = results.t2_values[idx]
                t2_color = COLORS["accent_error"] if results.t2_outliers[idx] else COLORS["text_primary"]
                dpg.add_text(f"{t2_val:.2f}", color=t2_color)

                # Q value (red if outlier)
                q_val = results.q_values[idx]
                q_color = COLORS["accent_error"] if results.q_outliers[idx] else COLORS["text_primary"]
                dpg.add_text(f"{q_val:.2f}", color=q_color)

                # Mahalanobis distance
                maha_val = results.mahalanobis[idx]
                maha_color = COLORS["accent_error"] if results.maha_outliers[idx] else COLORS["text_primary"]
                dpg.add_text(f"{maha_val:.2f}", color=maha_color)

                # Confidence level
                if idx in results.high_confidence:
                    conf_text = "HIGH"
                    conf_color = COLORS["accent_error"]
                elif idx in results.moderate_confidence:
                    conf_text = "Moderate"
                    conf_color = COLORS["accent_warning"]
                else:
                    conf_text = "Low"
                    conf_color = COLORS["text_muted"]
                dpg.add_text(conf_text, color=conf_color)

    def _clear_table(self):
        """Clear all rows from the outlier table."""
        table_tag = f"{self.tag}_outlier_table"
        children = dpg.get_item_children(table_tag, 1)
        if children:
            for child in children:
                dpg.delete_item(child)

    def _update_summary(self):
        """Update the summary statistics."""
        if self._results is None or self._dataset is None:
            return

        results = self._results
        n_total = self._dataset.n_samples
        n_t2 = int(np.sum(results.t2_outliers))
        n_q = int(np.sum(results.q_outliers))
        n_combined = len(results.combined_outliers)
        n_high = len(results.high_confidence)

        pct_t2 = 100 * n_t2 / n_total if n_total > 0 else 0
        pct_q = 100 * n_q / n_total if n_total > 0 else 0
        pct_combined = 100 * n_combined / n_total if n_total > 0 else 0

        summary = (
            f"Total: {n_total} | "
            f"T² outliers: {n_t2} ({pct_t2:.1f}%) | "
            f"Q outliers: {n_q} ({pct_q:.1f}%) | "
            f"Combined: {n_combined} ({pct_combined:.1f}%)"
        )

        if n_high > 0:
            summary += f" | High confidence: {n_high}"

        dpg.set_value(f"{self.tag}_summary", summary)

    def _on_outlier_checkbox(self, sender, app_data, user_data):
        """Handle outlier checkbox toggle."""
        idx = user_data
        if app_data:
            self._selected_outliers.add(idx)
        else:
            self._selected_outliers.discard(idx)

        # Update button states
        has_selection = len(self._selected_outliers) > 0
        dpg.configure_item(f"{self.tag}_flag_btn", enabled=has_selection)

    def _select_all_outliers(self, sender=None, app_data=None):
        """Select all outliers in the table."""
        if self._results is None:
            return

        for idx in self._results.combined_outliers:
            self._selected_outliers.add(idx)
            check_tag = f"{self.tag}_check_{idx}"
            if dpg.does_item_exist(check_tag):
                dpg.set_value(check_tag, True)

        dpg.configure_item(f"{self.tag}_flag_btn", enabled=True)

    def _clear_selection(self, sender=None, app_data=None):
        """Clear all selections."""
        for idx in list(self._selected_outliers):
            check_tag = f"{self.tag}_check_{idx}"
            if dpg.does_item_exist(check_tag):
                dpg.set_value(check_tag, False)

        self._selected_outliers.clear()
        dpg.configure_item(f"{self.tag}_flag_btn", enabled=False)

    def _flag_selected(self, sender=None, app_data=None):
        """Flag selected samples as outliers."""
        if not self._selected_outliers:
            return

        self._flagged_samples.update(self._selected_outliers)

        # Enable remove and unflag buttons
        dpg.configure_item(f"{self.tag}_remove_btn", enabled=True)
        dpg.configure_item(f"{self.tag}_unflag_btn", enabled=True)

        # Call callback
        if self.on_flag:
            self.on_flag(list(self._flagged_samples))

        # Update status
        dpg.set_value(
            f"{self.tag}_status",
            f"{len(self._flagged_samples)} samples flagged for removal"
        )

    def _unflag_selected(self, sender=None, app_data=None):
        """Unflag selected samples (remove from flagged set)."""
        if not self._selected_outliers:
            return

        # Remove selected samples from flagged set
        self._flagged_samples.difference_update(self._selected_outliers)

        # Update button states
        has_flagged = len(self._flagged_samples) > 0
        dpg.configure_item(f"{self.tag}_remove_btn", enabled=has_flagged)
        dpg.configure_item(f"{self.tag}_unflag_btn", enabled=has_flagged)

        # Update status
        if has_flagged:
            dpg.set_value(
                f"{self.tag}_status",
                f"{len(self._flagged_samples)} samples flagged for removal"
            )
        else:
            dpg.set_value(f"{self.tag}_status", "No samples flagged")

    def _remove_flagged(self, sender=None, app_data=None):
        """Remove flagged samples from the dataset."""
        if not self._flagged_samples:
            return

        # Call callback to actually remove
        if self.on_remove:
            self.on_remove(list(self._flagged_samples))

        # Clear flagged set
        removed_count = len(self._flagged_samples)
        self._flagged_samples.clear()
        self._selected_outliers.clear()

        dpg.configure_item(f"{self.tag}_remove_btn", enabled=False)
        dpg.configure_item(f"{self.tag}_unflag_btn", enabled=False)
        dpg.set_value(f"{self.tag}_status", f"Removed {removed_count} samples")

    def _check_imbalance(self):
        """Check for target imbalance."""
        if self._dataset is None or not self._dataset.has_target:
            dpg.set_value(f"{self.tag}_imbalance_status", "No target variable")
            dpg.configure_item(f"{self.tag}_imbalance_details", show=False)
            return

        ds = self._dataset
        task_type = ds.metadata.get('target_type', 'regression')

        try:
            if task_type == 'classification':
                # Check number of unique classes
                unique_classes = np.unique(ds.y)
                if len(unique_classes) > 2:
                    result = detect_multiclass_imbalance(ds.y)
                    if result.is_imbalanced:
                        self._show_multiclass_imbalance(result)
                    else:
                        dpg.set_value(f"{self.tag}_imbalance_status", "Classes are balanced")
                        dpg.configure_item(f"{self.tag}_imbalance_details", show=False)
                else:
                    result = detect_class_imbalance(ds.y)
                    if result.is_imbalanced:
                        self._show_class_imbalance(result)
                    else:
                        dpg.set_value(f"{self.tag}_imbalance_status", "Classes are balanced")
                        dpg.configure_item(f"{self.tag}_imbalance_details", show=False)
            else:
                result = detect_regression_imbalance(ds.y)
                if result.is_imbalanced:
                    self._show_regression_imbalance(result)
                else:
                    dpg.set_value(f"{self.tag}_imbalance_status", "Target distribution is balanced")
                    dpg.configure_item(f"{self.tag}_imbalance_details", show=False)

        except Exception as e:
            dpg.set_value(f"{self.tag}_imbalance_status", f"Error: {str(e)[:50]}")
            dpg.configure_item(f"{self.tag}_imbalance_details", show=False)

    def _show_class_imbalance(self, result: ClassImbalanceResult):
        """Display class imbalance results."""
        severity_colors = {
            'moderate': COLORS["accent_warning"],
            'severe': COLORS["accent_error"],
            'extreme': COLORS["accent_error"]
        }
        color = severity_colors.get(result.severity, COLORS["text_primary"])

        dpg.set_value(
            f"{self.tag}_imbalance_status",
            f"IMBALANCED - {result.severity.upper()} ({result.imbalance_ratio:.1f}:1)"
        )

        # Build info text
        info_lines = [
            f"Majority class: {result.majority_class} ({result.class_counts.get(result.majority_class, 0)} samples)",
            f"Minority class: {result.minority_class} ({result.class_counts.get(result.minority_class, 0)} samples)"
        ]
        dpg.set_value(f"{self.tag}_imbalance_info", "\n".join(info_lines))
        dpg.set_value(f"{self.tag}_imbalance_recommendation", result.recommendation)

        # Reset dropdown to classification methods
        classification_methods = get_available_methods('classification')
        classification_items = ["None (use as-is)"] + [desc for _, desc in classification_methods]
        dpg.configure_item(f"{self.tag}_imbalance_method", items=classification_items)

        # Set recommended method based on severity
        if result.severity == 'extreme':
            dpg.set_value(f"{self.tag}_imbalance_method", "SMOTETomek - Combined over/undersampling")
        elif result.severity == 'severe':
            dpg.set_value(f"{self.tag}_imbalance_method", "SMOTE - Synthetic oversampling (standard)")
        else:
            dpg.set_value(f"{self.tag}_imbalance_method", "Class weights - No resampling, weight loss function")

        dpg.configure_item(f"{self.tag}_imbalance_details", show=True)

    def _show_multiclass_imbalance(self, result):
        """Display multi-class imbalance results."""
        dpg.set_value(
            f"{self.tag}_imbalance_status",
            f"IMBALANCED - {result.severity.upper()} (max ratio: {result.max_ratio:.1f}:1)"
        )

        # Build info text
        info_lines = [
            f"Classes: {len(result.class_counts)}",
            f"Largest: {result.max_class} ({result.class_counts[result.max_class]} samples)",
            f"Smallest: {result.min_class} ({result.class_counts[result.min_class]} samples)"
        ]
        dpg.set_value(f"{self.tag}_imbalance_info", "\n".join(info_lines))
        dpg.set_value(f"{self.tag}_imbalance_recommendation",
                     "Use class_weight='balanced' or SMOTE for multi-class")

        # Reset dropdown to classification methods
        classification_methods = get_available_methods('classification')
        classification_items = ["None (use as-is)"] + [desc for _, desc in classification_methods]
        dpg.configure_item(f"{self.tag}_imbalance_method", items=classification_items)

        dpg.configure_item(f"{self.tag}_imbalance_details", show=True)

    def _show_regression_imbalance(self, result: RegressionImbalanceResult):
        """Display regression imbalance results."""
        dpg.set_value(
            f"{self.tag}_imbalance_status",
            f"IMBALANCED - {result.severity.upper()} (coverage: {result.coverage:.1%})"
        )

        # Build info text
        info_lines = [
            f"Target range: {result.target_range[0]:.2f} - {result.target_range[1]:.2f}",
            f"Sparse regions: {len(result.sparse_bins)} of {len(result.bin_counts)} bins",
            f"Total samples: {result.n_samples}"
        ]
        dpg.set_value(f"{self.tag}_imbalance_info", "\n".join(info_lines))
        dpg.set_value(f"{self.tag}_imbalance_recommendation", result.recommendation)

        # Update dropdown for regression with dynamic methods
        regression_methods = get_available_methods('regression')
        regression_items = ["None (use as-is)"] + [desc for _, desc in regression_methods]
        dpg.configure_item(f"{self.tag}_imbalance_method", items=regression_items)

        dpg.configure_item(f"{self.tag}_imbalance_details", show=True)

    def get_flagged_samples(self) -> List[int]:
        """Get list of flagged sample indices."""
        return list(self._flagged_samples)

    def get_selected_imbalance_method(self) -> str:
        """Get the selected imbalance handling method."""
        return dpg.get_value(f"{self.tag}_imbalance_method")

    def clear(self):
        """Clear all data and reset the panel."""
        self._dataset = None
        self._results = None
        self._selected_outliers.clear()
        self._flagged_samples.clear()

        self._clear_plots()
        self._clear_table()

        dpg.set_value(f"{self.tag}_status", "No data loaded")
        dpg.set_value(f"{self.tag}_summary", "")
        dpg.set_value(f"{self.tag}_imbalance_status", "No data loaded")
        dpg.configure_item(f"{self.tag}_imbalance_details", show=False)
