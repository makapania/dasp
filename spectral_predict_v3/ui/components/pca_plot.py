"""
PCA scores plot for Spectral Predict v3.

Interactive scatter plot showing PCA decomposition of spectral data.
"""

import dearpygui.dearpygui as dpg
import numpy as np
from typing import Optional, List, Callable
from sklearn.decomposition import PCA
from ..theme import COLORS


class PCAPlot:
    """
    Interactive PCA scores visualization.

    Features:
    - PC1 vs PC2 scatter plot
    - Color by target value or cluster
    - Variance explained display
    - Click to select samples
    - Zoom and pan (built into DPG plots)

    Example
    -------
    >>> plot = PCAPlot(parent="explore_panel")
    >>> plot.set_data(dataset)
    """

    def __init__(self, parent: str, tag: str = "pca_plot", on_select: Optional[Callable] = None):
        """
        Initialize the PCA plot.

        Parameters
        ----------
        parent : str
            Parent container tag
        tag : str
            Unique tag for this plot
        on_select : callable, optional
            Callback when points are selected (receives list of indices)
        """
        self.parent = parent
        self.tag = tag
        self.on_select = on_select
        self._dataset = None
        self._pca = None
        self._scores = None
        self._pc_x = 0  # PC1
        self._pc_y = 1  # PC2
        self._color_by = "target"  # "target", "none", or metadata column name
        self._n_bins = 4  # Number of color bins (shared with spectra plot)
        self._selected_indices = set()  # Currently selected sample indices

        self._create_ui()

    def _create_ui(self):
        """Create the plot UI structure."""
        with dpg.child_window(parent=self.parent, tag=self.tag, border=False):
            # Toolbar
            with dpg.group(horizontal=True):
                dpg.add_text("X Axis:", color=COLORS["text_muted"])
                dpg.add_combo(
                    items=["PC1", "PC2", "PC3", "PC4", "PC5"],
                    default_value="PC1",
                    callback=self._on_change_pc_x,
                    tag=f"{self.tag}_pc_x",
                    width=80
                )
                dpg.add_spacer(width=10)
                dpg.add_text("Y Axis:", color=COLORS["text_muted"])
                dpg.add_combo(
                    items=["PC1", "PC2", "PC3", "PC4", "PC5"],
                    default_value="PC2",
                    callback=self._on_change_pc_y,
                    tag=f"{self.tag}_pc_y",
                    width=80
                )
                dpg.add_spacer(width=20)
                dpg.add_text("Color by:", color=COLORS["text_muted"])
                dpg.add_combo(
                    items=["Target", "None"],
                    default_value="Target",
                    callback=self._on_change_color,
                    tag=f"{self.tag}_color_by",
                    width=120
                )

            dpg.add_spacer(height=5)

            # Variance info
            dpg.add_text(
                "Load data and click 'Run PCA'",
                tag=f"{self.tag}_variance",
                color=COLORS["text_muted"]
            )

            dpg.add_spacer(height=5)

            # Plot area
            with dpg.plot(
                tag=f"{self.tag}_plot",
                label="PCA Scores",
                height=-1,
                width=-1,
                anti_aliased=True,
                equal_aspects=True
            ):
                dpg.add_plot_legend()

                # X axis
                dpg.add_plot_axis(
                    dpg.mvXAxis,
                    label="PC1",
                    tag=f"{self.tag}_x_axis"
                )

                # Y axis
                dpg.add_plot_axis(
                    dpg.mvYAxis,
                    label="PC2",
                    tag=f"{self.tag}_y_axis"
                )

            # Add click handler for point selection
            with dpg.handler_registry(tag=f"{self.tag}_handler"):
                dpg.add_mouse_click_handler(callback=self._on_plot_click)

    def set_data(self, dataset):
        """
        Set the dataset to display.

        Parameters
        ----------
        dataset : SpectralDataset
            The dataset containing spectra for PCA
        """
        self._dataset = dataset
        self._pca = None
        self._scores = None

        # Update color options based on available columns
        color_options = ["Target", "None"]
        if dataset and dataset.metadata_columns:
            color_options.extend(list(dataset.metadata_columns.keys()))
        dpg.configure_item(f"{self.tag}_color_by", items=color_options)

        # Clear the plot and auto-run PCA
        self._clear_plot()
        if dataset is not None:
            self._run_pca()

    def _run_pca(self, sender=None, app_data=None):
        """Run PCA on the current dataset."""
        if self._dataset is None:
            return

        try:
            # Compute PCA
            n_components = min(5, self._dataset.n_samples, self._dataset.n_wavelengths)
            self._pca = PCA(n_components=n_components)
            self._scores = self._pca.fit_transform(self._dataset.X)

            # Update variance display
            var_explained = self._pca.explained_variance_ratio_ * 100
            var_text = "Variance: " + ", ".join([
                f"PC{i+1}={var_explained[i]:.1f}%"
                for i in range(min(3, len(var_explained)))
            ])
            if len(var_explained) > 3:
                var_text += f" (Total: {sum(var_explained[:3]):.1f}%)"
            dpg.set_value(f"{self.tag}_variance", var_text)

            # Update plot
            self._update_plot()

        except Exception as e:
            dpg.set_value(f"{self.tag}_variance", f"Error: {str(e)}")

    def _update_plot(self):
        """Update the scatter plot with current settings."""
        self._clear_plot()

        if self._scores is None or self._dataset is None:
            return

        ds = self._dataset
        scores = self._scores

        # Get PC indices
        pc_x = self._pc_x
        pc_y = self._pc_y

        if pc_x >= scores.shape[1] or pc_y >= scores.shape[1]:
            return

        x_data = scores[:, pc_x]
        y_data = scores[:, pc_y]

        # Update axis labels with variance
        var = self._pca.explained_variance_ratio_ * 100
        dpg.configure_item(f"{self.tag}_x_axis", label=f"PC{pc_x+1} ({var[pc_x]:.1f}%)")
        dpg.configure_item(f"{self.tag}_y_axis", label=f"PC{pc_y+1} ({var[pc_y]:.1f}%)")

        # Determine coloring
        color_by = dpg.get_value(f"{self.tag}_color_by")

        if color_by == "Target" and ds.has_target:
            if ds.metadata.get('target_type') == 'classification':
                self._plot_by_class(x_data, y_data, ds.y)
            else:
                self._plot_by_gradient(x_data, y_data, ds.y)
        elif color_by != "None" and color_by != "Target" and color_by in ds.metadata_columns:
            values = ds.metadata_columns[color_by]
            # Check if numeric
            try:
                numeric_values = np.array([float(v) for v in values])
                self._plot_by_gradient(x_data, y_data, numeric_values)
            except:
                self._plot_by_class(x_data, y_data, values)
        else:
            self._plot_single_color(x_data, y_data)

        # Fit axes
        dpg.fit_axis_data(f"{self.tag}_x_axis")
        dpg.fit_axis_data(f"{self.tag}_y_axis")

    def _plot_single_color(self, x_data: np.ndarray, y_data: np.ndarray):
        """Plot all points in a single color."""
        y_axis = f"{self.tag}_y_axis"

        dpg.add_scatter_series(
            x=list(x_data),
            y=list(y_data),
            label="Samples",
            parent=y_axis,
            tag=f"{self.tag}_scatter"
        )

        # Apply solid color theme
        color = (100, 149, 237, 255)  # Cornflower blue, solid
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, color, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, color, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 5, category=dpg.mvThemeCat_Plots)
        dpg.bind_item_theme(f"{self.tag}_scatter", theme)

    def _plot_by_gradient(self, x_data: np.ndarray, y_data: np.ndarray, values: np.ndarray):
        """Plot points colored by gradient based on numeric values."""
        y_axis = f"{self.tag}_y_axis"

        n_bins = self._n_bins

        # Special case: n_bins=1 means single color, no gradient
        if n_bins == 1:
            color = (100, 149, 237, 255)  # Cornflower blue, solid
            dpg.add_scatter_series(
                x=list(x_data),
                y=list(y_data),
                label="Samples",
                parent=y_axis,
                tag=f"{self.tag}_scatter_0"
            )
            with dpg.theme() as theme:
                with dpg.theme_component(dpg.mvScatterSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, color, category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, color, category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 5, category=dpg.mvThemeCat_Plots)
            dpg.bind_item_theme(f"{self.tag}_scatter_0", theme)
            return

        # Normalize values
        vmin, vmax = np.nanmin(values), np.nanmax(values)
        if vmax == vmin:
            normalized = np.zeros_like(values)
        else:
            normalized = (values - vmin) / (vmax - vmin)

        # Generate colors dynamically for any bin count (2-8)
        bin_colors = self._generate_bin_colors(n_bins)

        # Create bins dynamically based on n_bins
        for i in range(n_bins):
            bin_start = i / n_bins
            bin_end = (i + 1) / n_bins

            if i == n_bins - 1:
                mask = (normalized >= bin_start) & (normalized <= bin_end)
            else:
                mask = (normalized >= bin_start) & (normalized < bin_end)

            if np.any(mask):
                # Create label
                val_start = vmin + bin_start * (vmax - vmin)
                val_end = vmin + bin_end * (vmax - vmin)
                if n_bins == 2:
                    label = "Low" if i == 0 else "High"
                else:
                    label = f"{val_start:.1f}-{val_end:.1f}"

                color = bin_colors[i]

                dpg.add_scatter_series(
                    x=list(x_data[mask]),
                    y=list(y_data[mask]),
                    label=label,
                    parent=y_axis,
                    tag=f"{self.tag}_scatter_{i}"
                )
                # Make solid color with matching outline
                solid_color = (color[0], color[1], color[2], 255)
                with dpg.theme() as theme:
                    with dpg.theme_component(dpg.mvScatterSeries):
                        dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, solid_color, category=dpg.mvThemeCat_Plots)
                        dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, solid_color, category=dpg.mvThemeCat_Plots)
                        dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 5, category=dpg.mvThemeCat_Plots)
                dpg.bind_item_theme(f"{self.tag}_scatter_{i}", theme)

    def _generate_bin_colors(self, n_bins: int) -> list:
        """Generate distinct colors for n bins (1-8) along a rainbow spectrum."""
        # Full 8-color palette: Blue -> Cyan -> Green -> Yellow-Green -> Yellow -> Orange -> Red-Orange -> Red
        full_palette = [
            (30, 100, 255, 200),    # Blue
            (50, 180, 220, 200),    # Cyan
            (50, 200, 100, 200),    # Green
            (150, 210, 50, 200),    # Yellow-green
            (255, 220, 50, 200),    # Yellow
            (255, 160, 50, 200),    # Orange
            (255, 100, 50, 200),    # Red-orange
            (255, 50, 50, 200),     # Red
        ]

        if n_bins == 1:
            return [(100, 149, 237, 200)]  # Cornflower blue - single color
        elif n_bins == 8:
            return full_palette
        elif n_bins == 2:
            return [full_palette[0], full_palette[7]]  # Blue, Red
        else:
            # Sample evenly from the palette
            indices = [int(i * 7 / (n_bins - 1)) for i in range(n_bins)]
            return [full_palette[i] for i in indices]

    def _plot_by_class(self, x_data: np.ndarray, y_data: np.ndarray, labels):
        """Plot points colored by class label."""
        y_axis = f"{self.tag}_y_axis"

        unique_labels = list(set(labels))

        # Color palette - solid colors
        palette = [
            (31, 119, 180, 255),   # Blue
            (255, 127, 14, 255),   # Orange
            (44, 160, 44, 255),    # Green
            (214, 39, 40, 255),    # Red
            (148, 103, 189, 255),  # Purple
            (140, 86, 75, 255),    # Brown
            (227, 119, 194, 255),  # Pink
            (127, 127, 127, 255),  # Gray
            (188, 189, 34, 255),   # Olive
            (23, 190, 207, 255),   # Cyan
        ]

        for i, label in enumerate(unique_labels):
            mask = np.array([l == label for l in labels])
            color = palette[i % len(palette)]

            dpg.add_scatter_series(
                x=list(x_data[mask]),
                y=list(y_data[mask]),
                label=str(label),
                parent=y_axis,
                tag=f"{self.tag}_scatter_{i}"
            )
            with dpg.theme() as theme:
                with dpg.theme_component(dpg.mvScatterSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, color, category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, color, category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 5, category=dpg.mvThemeCat_Plots)
            dpg.bind_item_theme(f"{self.tag}_scatter_{i}", theme)

    def _clear_plot(self):
        """Clear all series from the plot."""
        y_axis = f"{self.tag}_y_axis"
        children = dpg.get_item_children(y_axis, 1)
        if children:
            for child in children:
                dpg.delete_item(child)

    def _on_change_pc_x(self, sender, app_data):
        """Handle PC X axis change."""
        pc_map = {"PC1": 0, "PC2": 1, "PC3": 2, "PC4": 3, "PC5": 4}
        self._pc_x = pc_map.get(app_data, 0)
        self._update_plot()

    def _on_change_pc_y(self, sender, app_data):
        """Handle PC Y axis change."""
        pc_map = {"PC1": 0, "PC2": 1, "PC3": 2, "PC4": 3, "PC5": 4}
        self._pc_y = pc_map.get(app_data, 1)
        self._update_plot()

    def _on_change_color(self, sender, app_data):
        """Handle color by change."""
        self._update_plot()

    def set_bins(self, n_bins: int):
        """Set the number of color bins for gradient display."""
        self._n_bins = n_bins
        self._update_plot()

    def clear(self):
        """Clear the plot."""
        self._dataset = None
        self._pca = None
        self._scores = None
        self._selected_indices = set()
        self._clear_plot()
        dpg.set_value(f"{self.tag}_variance", "No data loaded")

    def _on_plot_click(self, sender, app_data):
        """Handle mouse click on plot for point selection."""
        import sys
        print(f"PCA click handler called: sender={sender}, app_data={app_data}", file=sys.stderr)

        if self._scores is None:
            print("  No scores, returning", file=sys.stderr)
            return

        # Only respond to left click (button 0)
        if app_data != 0:
            print(f"  Not left click ({app_data}), returning", file=sys.stderr)
            return

        # Get mouse position in plot coordinates
        # This returns None if mouse is not over any plot
        mouse_pos = dpg.get_plot_mouse_pos()
        print(f"  Mouse pos: {mouse_pos}", file=sys.stderr)
        if mouse_pos is None:
            print("  Mouse not over plot, returning", file=sys.stderr)
            return

        # Also check if this is specifically our plot by checking if hovered
        # Try checking both the plot and the child window containing it
        plot_hovered = dpg.is_item_hovered(f"{self.tag}_plot")
        container_hovered = dpg.is_item_hovered(self.tag)
        print(f"  Plot hovered: {plot_hovered}, Container hovered: {container_hovered}", file=sys.stderr)

        # If neither is hovered, the click was on a different plot
        if not plot_hovered and not container_hovered:
            print("  Wrong plot, returning", file=sys.stderr)
            return

        click_x, click_y = mouse_pos

        # Get current PC scores
        pc_x = self._pc_x
        pc_y = self._pc_y
        x_data = self._scores[:, pc_x]
        y_data = self._scores[:, pc_y]

        # Find nearest point
        min_dist = float('inf')
        nearest_idx = None

        # Calculate axis ranges for normalization
        x_range = max(x_data) - min(x_data) if len(x_data) > 0 else 1
        y_range = max(y_data) - min(y_data) if len(y_data) > 0 else 1

        for i in range(len(x_data)):
            # Normalize distances by axis range for fair comparison
            dx = (x_data[i] - click_x) / x_range if x_range > 0 else 0
            dy = (y_data[i] - click_y) / y_range if y_range > 0 else 0
            dist = dx * dx + dy * dy

            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        # Only select if click is reasonably close (within 5% of plot range)
        print(f"  Nearest idx: {nearest_idx}, min_dist: {min_dist}", file=sys.stderr)
        if nearest_idx is not None and min_dist < 0.01:  # ~10% threshold
            # Check if Ctrl is held for multi-select
            ctrl_held = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)

            if ctrl_held:
                # Toggle selection
                if nearest_idx in self._selected_indices:
                    self._selected_indices.discard(nearest_idx)
                else:
                    self._selected_indices.add(nearest_idx)
            else:
                # Single select
                self._selected_indices = {nearest_idx}

            print(f"  Selected: {self._selected_indices}", file=sys.stderr)
            self._update_selection_highlight()
            if self.on_select:
                self.on_select(list(self._selected_indices))
        else:
            print(f"  Click too far from any point", file=sys.stderr)

    def set_selection(self, indices: set):
        """
        Set the current selection (for sync from other views).

        Parameters
        ----------
        indices : set
            Set of selected sample indices
        """
        self._selected_indices = set(indices)
        self._update_selection_highlight()

    def get_selection(self) -> set:
        """Get the currently selected indices."""
        return self._selected_indices.copy()

    def _update_selection_highlight(self):
        """Update visual highlighting for selected points."""
        if self._scores is None or not self._selected_indices:
            # Clear any existing selection highlight
            if dpg.does_item_exist(f"{self.tag}_selection"):
                dpg.delete_item(f"{self.tag}_selection")
            return

        # Get coordinates for selected points
        pc_x = self._pc_x
        pc_y = self._pc_y
        x_data = self._scores[:, pc_x]
        y_data = self._scores[:, pc_y]

        selected_x = [x_data[i] for i in self._selected_indices if i < len(x_data)]
        selected_y = [y_data[i] for i in self._selected_indices if i < len(y_data)]

        if not selected_x:
            return

        # Remove existing selection highlight
        if dpg.does_item_exist(f"{self.tag}_selection"):
            dpg.delete_item(f"{self.tag}_selection")

        # Add highlighted scatter series for selected points
        y_axis = f"{self.tag}_y_axis"
        dpg.add_scatter_series(
            x=selected_x,
            y=selected_y,
            label="Selected",
            parent=y_axis,
            tag=f"{self.tag}_selection"
        )

        # Highlight color - bright yellow with red outline
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, (255, 255, 0, 255), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, (255, 0, 0, 255), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 8, category=dpg.mvThemeCat_Plots)
        dpg.bind_item_theme(f"{self.tag}_selection", theme)
