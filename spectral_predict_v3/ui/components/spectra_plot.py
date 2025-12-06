"""
Interactive spectra plot for Spectral Predict v3.

GPU-accelerated plotting using Dear PyGui's native plot widgets.
"""

import dearpygui.dearpygui as dpg
import numpy as np
from typing import Optional, List, Callable
from ..theme import COLORS


class SpectraPlot:
    """
    Interactive spectra visualization with derivative overlays.

    Features:
    - Raw spectra display
    - 1st and 2nd derivative overlays
    - Zoom and pan (built into DPG plots)
    - Subsample for performance with large datasets
    - Color by target value

    Example
    -------
    >>> plot = SpectraPlot(parent="explore_panel")
    >>> plot.set_data(dataset)
    """

    def __init__(self, parent: str, tag: str = "spectra_plot", on_select: Optional[Callable] = None):
        """
        Initialize the spectra plot.

        Parameters
        ----------
        parent : str
            Parent container tag
        tag : str
            Unique tag for this plot
        on_select : callable, optional
            Callback when spectra are selected (receives list of indices)
        """
        self.parent = parent
        self.tag = tag
        self.on_select = on_select
        self._dataset = None
        self._max_spectra = 100  # Subsample for performance
        self._show_raw = True
        self._show_sg1 = False
        self._show_sg2 = False
        self._n_bins = 4  # Number of color bins for gradient (synced with PCA)
        self._selected_indices = set()  # Currently selected sample indices
        self._displayed_indices = []  # Indices of samples currently displayed (for mapping clicks)

        self._create_ui()

    def _create_ui(self):
        """Create the plot UI structure."""
        with dpg.child_window(parent=self.parent, tag=self.tag, border=False):
            # Toolbar
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="Raw",
                    default_value=True,
                    callback=self._on_toggle_raw,
                    tag=f"{self.tag}_raw_cb"
                )
                dpg.add_checkbox(
                    label="1st Derivative",
                    default_value=False,
                    callback=self._on_toggle_sg1,
                    tag=f"{self.tag}_sg1_cb"
                )
                dpg.add_checkbox(
                    label="2nd Derivative",
                    default_value=False,
                    callback=self._on_toggle_sg2,
                    tag=f"{self.tag}_sg2_cb"
                )
                dpg.add_spacer(width=20)
                dpg.add_text("Showing:", color=COLORS["text_muted"])
                dpg.add_text("0 spectra", tag=f"{self.tag}_count")

            dpg.add_spacer(height=5)

            # Plot area
            with dpg.plot(
                tag=f"{self.tag}_plot",
                label="Spectra",
                height=-1,
                width=-1,
                anti_aliased=True
            ):
                dpg.add_plot_legend()

                # X axis - wavelength
                dpg.add_plot_axis(
                    dpg.mvXAxis,
                    label="Wavelength (nm)",
                    tag=f"{self.tag}_x_axis"
                )

                # Y axis - intensity/absorbance
                dpg.add_plot_axis(
                    dpg.mvYAxis,
                    label="Value",
                    tag=f"{self.tag}_y_axis"
                )

            # Add click handler for spectrum selection
            with dpg.handler_registry(tag=f"{self.tag}_handler"):
                dpg.add_mouse_click_handler(callback=self._on_plot_click)

    def set_data(self, dataset):
        """
        Set the dataset to display.

        Parameters
        ----------
        dataset : SpectralDataset
            The dataset containing spectra to plot
        """
        self._dataset = dataset

        self._update_plot()

    def set_bins(self, n_bins: int):
        """Set the number of color bins for gradient display."""
        self._n_bins = n_bins
        self._update_plot()

    def _update_plot(self):
        """Update the plot with current data and settings."""
        # Clear existing series
        y_axis = f"{self.tag}_y_axis"

        # Delete all children of y_axis (the line series)
        children = dpg.get_item_children(y_axis, 1)
        if children:
            for child in children:
                dpg.delete_item(child)

        if self._dataset is None:
            dpg.set_value(f"{self.tag}_count", "0 spectra")
            return

        ds = self._dataset
        n_samples = ds.n_samples
        wavelengths = ds.wavelengths

        # Subsample if too many spectra
        if n_samples > self._max_spectra:
            indices = np.linspace(0, n_samples - 1, self._max_spectra, dtype=int)
            display_text = f"{self._max_spectra} of {n_samples} spectra"
        else:
            indices = np.arange(n_samples)
            display_text = f"{n_samples} spectra"

        # Store displayed indices for click-to-select
        self._displayed_indices = list(indices)

        dpg.set_value(f"{self.tag}_count", display_text)

        # Generate colors based on target if available
        self._legend_info = None  # Reset legend info
        if ds.has_target and ds.metadata.get('target_type') == 'regression':
            colors, self._legend_info = self._get_gradient_colors(ds.y[indices])
        elif ds.has_target and ds.metadata.get('target_type') == 'classification':
            colors, self._legend_info = self._get_class_colors(ds.y[indices])
        else:
            # Default blue color for all
            colors = [(100, 149, 237, 150)] * len(indices)  # Cornflower blue with alpha

        # Check if we need to normalize (multiple types shown)
        n_types_shown = sum([self._show_raw, self._show_sg1, self._show_sg2])
        normalize = n_types_shown > 1

        # Update Y axis label based on normalization
        if normalize:
            dpg.configure_item(f"{self.tag}_y_axis", label="Normalized (0-1)")
        else:
            dpg.configure_item(f"{self.tag}_y_axis", label="Value")

        # Plot raw spectra (no label - legend shows target colors only)
        if self._show_raw:
            raw_data = ds.X[indices]
            if normalize:
                raw_data = self._normalize_data(raw_data)
            for i, idx in enumerate(indices):
                dpg.add_line_series(
                    x=list(wavelengths),
                    y=list(raw_data[i]),
                    parent=y_axis,
                    tag=f"{self.tag}_raw_{i}"
                )
                dpg.bind_item_theme(f"{self.tag}_raw_{i}", self._create_line_theme(colors[i]))

        # Plot 1st derivative (no label)
        if self._show_sg1:
            sg1_data = self._compute_derivative(ds.X[indices], wavelengths, deriv=1)
            if normalize:
                sg1_data = self._normalize_data(sg1_data)
            for i, idx in enumerate(indices):
                dpg.add_line_series(
                    x=list(wavelengths),
                    y=list(sg1_data[i]),
                    parent=y_axis,
                    tag=f"{self.tag}_sg1_{i}"
                )
                # Use target colors when shown alone, orange when overlaid
                if n_types_shown == 1:
                    dpg.bind_item_theme(f"{self.tag}_sg1_{i}", self._create_line_theme(colors[i]))
                else:
                    dpg.bind_item_theme(f"{self.tag}_sg1_{i}", self._create_line_theme((255, 165, 0, 150)))

        # Plot 2nd derivative (no label)
        if self._show_sg2:
            sg2_data = self._compute_derivative(ds.X[indices], wavelengths, deriv=2)
            if normalize:
                sg2_data = self._normalize_data(sg2_data)
            for i, idx in enumerate(indices):
                dpg.add_line_series(
                    x=list(wavelengths),
                    y=list(sg2_data[i]),
                    parent=y_axis,
                    tag=f"{self.tag}_sg2_{i}"
                )
                # Use target colors when shown alone, green when overlaid
                if n_types_shown == 1:
                    dpg.bind_item_theme(f"{self.tag}_sg2_{i}", self._create_line_theme(colors[i]))
                else:
                    dpg.bind_item_theme(f"{self.tag}_sg2_{i}", self._create_line_theme((50, 205, 50, 150)))

        # Add legend entries for color scheme (only when using target colors)
        if self._legend_info and n_types_shown == 1:
            for i, (label, color) in enumerate(self._legend_info['entries']):
                # Add a dummy series for the legend
                dpg.add_line_series(
                    x=[wavelengths[0]],
                    y=[np.nan],  # Invisible point
                    label=label,
                    parent=y_axis,
                    tag=f"{self.tag}_legend_{i}"
                )
                dpg.bind_item_theme(f"{self.tag}_legend_{i}", self._create_line_theme(color))

        # Fit axes to data
        dpg.fit_axis_data(f"{self.tag}_x_axis")
        dpg.fit_axis_data(f"{self.tag}_y_axis")

    def _normalize_data(self, X: np.ndarray) -> np.ndarray:
        """Normalize data to 0-1 range across all spectra (for multi-type display)."""
        vmin = np.nanmin(X)
        vmax = np.nanmax(X)
        if vmax == vmin:
            return np.zeros_like(X)
        return (X - vmin) / (vmax - vmin)

    def _generate_bin_colors(self, n_bins: int) -> list:
        """Generate distinct colors for n bins (1-8) along a rainbow spectrum."""
        # Full 8-color palette: Blue -> Cyan -> Green -> Yellow-Green -> Yellow -> Orange -> Red-Orange -> Red
        full_palette = [
            (30, 100, 255, 180),    # Blue
            (50, 180, 220, 180),    # Cyan
            (50, 200, 100, 180),    # Green
            (150, 210, 50, 180),    # Yellow-green
            (255, 220, 50, 180),    # Yellow
            (255, 160, 50, 180),    # Orange
            (255, 100, 50, 180),    # Red-orange
            (255, 50, 50, 180),     # Red
        ]

        if n_bins == 1:
            return [(100, 149, 237, 180)]  # Cornflower blue - single color
        elif n_bins == 8:
            return full_palette
        elif n_bins == 2:
            return [full_palette[0], full_palette[7]]  # Blue, Red
        else:
            # Sample evenly from the palette
            indices = [int(i * 7 / (n_bins - 1)) for i in range(n_bins)]
            return [full_palette[i] for i in indices]

    def _compute_derivative(self, X: np.ndarray, wavelengths: np.ndarray, deriv: int = 1) -> np.ndarray:
        """Compute Savitzky-Golay derivative."""
        from scipy.signal import savgol_filter

        window = min(15, len(wavelengths) // 4)
        if window % 2 == 0:
            window += 1
        window = max(5, window)

        result = np.zeros_like(X)
        for i in range(len(X)):
            try:
                result[i] = savgol_filter(X[i], window, polyorder=2, deriv=deriv)
            except:
                result[i] = np.gradient(X[i]) if deriv == 1 else np.gradient(np.gradient(X[i]))

        return result

    def _get_gradient_colors(self, values: np.ndarray) -> tuple:
        """Generate gradient colors based on numeric values."""
        if len(values) == 0:
            return [], None

        n_bins = self._n_bins

        # Special case: n_bins=1 means single color, no gradient
        if n_bins == 1:
            single_color = (100, 149, 237, 180)  # Cornflower blue
            colors = [single_color] * len(values)
            return colors, None  # No legend for single color

        # Normalize to 0-1
        vmin, vmax = np.nanmin(values), np.nanmax(values)
        if vmax == vmin:
            normalized = np.zeros_like(values)
        else:
            normalized = (values - vmin) / (vmax - vmin)

        # Generate colors dynamically for any bin count (2-8)
        # Use a color progression: Blue -> Cyan -> Green -> Yellow -> Orange -> Red
        bin_colors = self._generate_bin_colors(n_bins)

        # Assign colors based on which bin each value falls into
        colors = []
        for v in normalized:
            bin_idx = min(int(v * n_bins), n_bins - 1)  # Clamp to valid range
            colors.append(bin_colors[bin_idx])

        # Create legend entries
        legend_entries = []
        for i in range(n_bins):
            # Calculate bin boundaries
            bin_start = i / n_bins
            bin_end = (i + 1) / n_bins

            # Value range for this bin
            val_start = vmin + bin_start * (vmax - vmin)
            val_end = vmin + bin_end * (vmax - vmin)

            color = bin_colors[i]

            # Label
            if n_bins == 2:
                label = "Low" if i == 0 else "High"
                label += f" ({val_start:.2f}-{val_end:.2f})"
            else:
                label = f"{val_start:.2f}-{val_end:.2f}"

            legend_entries.append((label, color))

        legend_info = {
            'type': 'gradient',
            'entries': legend_entries
        }

        return colors, legend_info

    def _get_class_colors(self, labels: np.ndarray) -> tuple:
        """Generate distinct colors for classification labels."""
        unique_labels = sorted(set(labels), key=str)
        n_classes = len(unique_labels)

        # Predefined colors for up to 10 classes
        palette = [
            (31, 119, 180, 150),   # Blue
            (255, 127, 14, 150),   # Orange
            (44, 160, 44, 150),    # Green
            (214, 39, 40, 150),    # Red
            (148, 103, 189, 150),  # Purple
            (140, 86, 75, 150),    # Brown
            (227, 119, 194, 150),  # Pink
            (127, 127, 127, 150),  # Gray
            (188, 189, 34, 150),   # Olive
            (23, 190, 207, 150),   # Cyan
        ]

        label_to_color = {label: palette[i % len(palette)] for i, label in enumerate(unique_labels)}
        colors = [label_to_color[label] for label in labels]

        # Legend info for classes: list of (label, color) tuples
        legend_info = {
            'type': 'classification',
            'entries': [(str(label), palette[i % len(palette)]) for i, label in enumerate(unique_labels)]
        }

        return colors, legend_info

    def _create_line_theme(self, color: tuple) -> int:
        """Create a theme for a line series with the given color."""
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)
        return theme

    def _on_toggle_raw(self, sender, app_data):
        """Handle raw spectra toggle."""
        self._show_raw = app_data
        self._update_plot()

    def _on_toggle_sg1(self, sender, app_data):
        """Handle 1st derivative toggle."""
        self._show_sg1 = app_data
        self._update_plot()

    def _on_toggle_sg2(self, sender, app_data):
        """Handle 2nd derivative toggle."""
        self._show_sg2 = app_data
        self._update_plot()

    def _on_change_bins(self, sender, app_data):
        """Handle color bins change."""
        self._n_bins = int(app_data)
        self._update_plot()

    def clear(self):
        """Clear the plot."""
        self._dataset = None
        self._selected_indices = set()
        self._displayed_indices = []
        self._update_plot()

    def _on_plot_click(self, sender, app_data):
        """Handle mouse click on plot for spectrum selection."""
        if self._dataset is None or not self._displayed_indices:
            return

        # Only respond to left click (button 0)
        if app_data != 0:
            return

        # Check if mouse is over the plot
        if not dpg.is_item_hovered(f"{self.tag}_plot"):
            return

        # Get mouse position in plot coordinates
        mouse_pos = dpg.get_plot_mouse_pos()
        if mouse_pos is None:
            return

        click_x, click_y = mouse_pos
        wavelengths = self._dataset.wavelengths

        # Find which spectrum is closest to the click
        min_dist = float('inf')
        nearest_idx = None

        # Get the spectral data for displayed samples
        ds = self._dataset

        for local_i, global_idx in enumerate(self._displayed_indices):
            spectrum = ds.X[global_idx]

            # Find the closest wavelength to click_x
            wl_idx = np.argmin(np.abs(wavelengths - click_x))

            # Get the y value at that wavelength
            y_val = spectrum[wl_idx]

            # Calculate distance to click point (only y matters for line selection)
            dist = abs(y_val - click_y)

            if dist < min_dist:
                min_dist = dist
                nearest_idx = global_idx

        # Check if click is close enough (within reasonable range)
        y_range = np.ptp(ds.X[self._displayed_indices])
        threshold = 0.05 * y_range if y_range > 0 else 0.1

        if nearest_idx is not None and min_dist < threshold:
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

            self._update_selection_highlight()
            if self.on_select:
                self.on_select(list(self._selected_indices))

    def set_selection(self, indices: set):
        """Set the current selection (for sync from other views)."""
        self._selected_indices = set(indices)
        self._update_selection_highlight()

    def get_selection(self) -> set:
        """Get the currently selected indices."""
        return self._selected_indices.copy()

    def _update_selection_highlight(self):
        """Update visual highlighting for selected spectra."""
        # Remove any existing selection highlights
        for tag in list(dpg.get_item_children(f"{self.tag}_y_axis", 1) or []):
            tag_str = dpg.get_item_alias(tag) if dpg.get_item_alias(tag) else ""
            if "_selected_" in str(tag_str):
                dpg.delete_item(tag)

        if self._dataset is None or not self._selected_indices:
            return

        wavelengths = self._dataset.wavelengths
        y_axis = f"{self.tag}_y_axis"

        # Add highlighted line for each selected spectrum
        for idx in self._selected_indices:
            if idx >= self._dataset.n_samples:
                continue

            spectrum = self._dataset.X[idx]
            tag = f"{self.tag}_selected_{idx}"

            # Remove if exists
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)

            dpg.add_line_series(
                x=list(wavelengths),
                y=list(spectrum),
                label=f"Selected ({self._dataset.sample_ids[idx]})" if idx < len(self._dataset.sample_ids) else f"Selected ({idx})",
                parent=y_axis,
                tag=tag
            )

            # Bright highlight color - thick yellow/gold line
            with dpg.theme() as theme:
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 215, 0, 255), category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 3.0, category=dpg.mvThemeCat_Plots)
            dpg.bind_item_theme(tag, theme)
