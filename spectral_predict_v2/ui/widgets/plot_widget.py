"""
Interactive Plot Widgets - Spectral Predict v2

High-performance interactive plots using pyqtgraph.

Features:
- Hardware-accelerated rendering
- Built-in pan/zoom
- Crosshair with coordinate display
- Hover tooltips
- Theme-aware styling
"""

from typing import Optional, List, Tuple
import numpy as np
import pyqtgraph as pg
from pyqtgraph import PlotWidget as PGPlotWidget, ScatterPlotItem, InfiniteLine

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QGridLayout,
    QToolButton,
    QMenu,
    QFileDialog,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPen, QBrush, QFont

from ..theme.tokens import COLORS, SPACING, TYPOGRAPHY
from ..theme.icons import Icons


# Configure pyqtgraph defaults
pg.setConfigOptions(
    antialias=True,
    background=QColor(COLORS["bg_surface"]),
    foreground=QColor(COLORS["text_primary"]),
)


class InteractivePlotWidget(QWidget):
    """
    Base interactive plot widget with common functionality.

    Features:
    - Dark theme styling
    - Crosshair with coordinate display
    - Export to PNG/SVG
    - Title and axis labels
    """

    point_clicked = Signal(int)  # Emits index of clicked point
    point_hovered = Signal(int, float, float)  # index, x, y

    def __init__(
        self,
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        show_crosshair: bool = True,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self._title = title
        self._x_label = x_label
        self._y_label = y_label

        self._setup_ui(show_crosshair)
        self._apply_theme()

    def _setup_ui(self, show_crosshair: bool):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header with title and export button
        if self._title:
            header = QHBoxLayout()
            header.setContentsMargins(SPACING["sm"], SPACING["xs"], SPACING["sm"], 0)

            title_label = QLabel(self._title)
            title_label.setStyleSheet(f"""
                color: {COLORS["text_primary"]};
                font-size: {TYPOGRAPHY["size_md"]}pt;
                font-weight: {TYPOGRAPHY["weight_semibold"]};
            """)
            header.addWidget(title_label)
            header.addStretch()

            # Export button
            export_btn = QToolButton()
            export_btn.setIcon(Icons.download(14))
            export_btn.setToolTip("Export plot")
            export_btn.setStyleSheet(f"""
                QToolButton {{
                    background: transparent;
                    border: none;
                    padding: 2px;
                }}
                QToolButton:hover {{
                    background: {COLORS["bg_elevated"]};
                    border-radius: 4px;
                }}
            """)
            export_btn.clicked.connect(self._show_export_menu)
            header.addWidget(export_btn)

            layout.addLayout(header)

        # Plot widget
        self._plot = PGPlotWidget()
        self._plot.setBackground(QColor(COLORS["bg_surface"]))
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setMouseEnabled(x=True, y=True)

        # Axis labels
        self._plot.setLabel("bottom", self._x_label)
        self._plot.setLabel("left", self._y_label)

        layout.addWidget(self._plot, 1)

        # Coordinate display
        self._coord_label = QLabel("")
        self._coord_label.setStyleSheet(f"""
            color: {COLORS["text_secondary"]};
            font-size: {TYPOGRAPHY["size_xs"]}pt;
            padding: 2px {SPACING["sm"]}px;
        """)
        layout.addWidget(self._coord_label)

        # Crosshair
        self._vline = None
        self._hline = None
        if show_crosshair:
            self._setup_crosshair()

    def _setup_crosshair(self):
        """Add crosshair that follows mouse."""
        pen = QPen(QColor(COLORS["text_tertiary"]))
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setWidth(1)

        self._vline = InfiniteLine(angle=90, movable=False, pen=pen)
        self._hline = InfiniteLine(angle=0, movable=False, pen=pen)
        self._plot.addItem(self._vline, ignoreBounds=True)
        self._plot.addItem(self._hline, ignoreBounds=True)

        # Connect mouse move
        self._plot.scene().sigMouseMoved.connect(self._on_mouse_moved)

    def _on_mouse_moved(self, pos):
        """Update crosshair and coordinate display on mouse move."""
        if self._plot.sceneBoundingRect().contains(pos):
            mouse_point = self._plot.plotItem.vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()

            if self._vline is not None:
                self._vline.setPos(x)
                self._hline.setPos(y)

            self._coord_label.setText(f"x: {x:.4f}  y: {y:.4f}")

    def _apply_theme(self):
        """Apply dark theme to plot."""
        # Axis styling
        axis_pen = QPen(QColor(COLORS["border_default"]))
        axis_pen.setWidth(1)

        for axis in ['left', 'bottom']:
            ax = self._plot.getAxis(axis)
            ax.setPen(axis_pen)
            ax.setTextPen(QColor(COLORS["text_secondary"]))
            ax.setStyle(tickFont=QFont(TYPOGRAPHY["font_family"], TYPOGRAPHY["size_xs"]))

        self.setStyleSheet(f"""
            InteractivePlotWidget {{
                background-color: {COLORS["bg_surface"]};
                border: 1px solid {COLORS["border_default"]};
                border-radius: {SPACING["sm"]}px;
            }}
        """)

    def _show_export_menu(self):
        """Show export options menu."""
        menu = QMenu(self)
        menu.addAction("Export as PNG...", lambda: self._export("png"))
        menu.addAction("Export as SVG...", lambda: self._export("svg"))
        menu.exec(self.mapToGlobal(self.sender().pos()))

    def _export(self, format: str):
        """Export plot to file."""
        filter_map = {
            "png": "PNG Image (*.png)",
            "svg": "SVG Image (*.svg)",
        }
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Plot", "", filter_map.get(format, "")
        )
        if file_path:
            exporter = pg.exporters.ImageExporter(self._plot.plotItem)
            exporter.export(file_path)

    def get_plot_item(self) -> pg.PlotItem:
        """Get the underlying PlotItem for direct manipulation."""
        return self._plot.plotItem

    def get_view_box(self) -> pg.ViewBox:
        """Get the ViewBox for coordinate transforms."""
        return self._plot.plotItem.vb

    def clear(self):
        """Clear all plot items."""
        self._plot.clear()
        if self._vline is not None:
            self._plot.addItem(self._vline, ignoreBounds=True)
            self._plot.addItem(self._hline, ignoreBounds=True)

    def auto_range(self):
        """Reset view to show all data."""
        self._plot.autoRange()


class SpectraOverlayPlot(InteractivePlotWidget):
    """
    Plot for overlaying multiple spectra.

    Features:
    - Plot multiple spectra with different colors
    - Click to select spectrum
    - Hover to highlight
    - Legend with sample IDs
    """

    spectrum_selected = Signal(int)  # Index of selected spectrum
    spectrum_hovered = Signal(int)  # Index of hovered spectrum

    def __init__(
        self,
        title: str = "Spectra",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(
            title=title,
            x_label="Wavelength",
            y_label="Absorbance",
            parent=parent,
        )
        self._spectra: List = []
        self._wavelengths: Optional[np.ndarray] = None
        self._sample_ids: List[str] = []
        self._selected_index: int = -1

    def set_data(
        self,
        spectra: np.ndarray,
        wavelengths: np.ndarray,
        sample_ids: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
    ):
        """
        Plot multiple spectra.

        Args:
            spectra: 2D array (n_samples, n_wavelengths)
            wavelengths: 1D array of wavelength values
            sample_ids: Optional list of sample identifiers
            colors: Optional list of colors for each spectrum
        """
        self.clear()
        self._spectra = []
        self._wavelengths = wavelengths

        n_samples = spectra.shape[0]

        if sample_ids is None:
            self._sample_ids = [f"Sample {i+1}" for i in range(n_samples)]
        else:
            self._sample_ids = list(sample_ids)

        # Default colors from theme
        chart_colors = [
            COLORS["chart_1"], COLORS["chart_2"], COLORS["chart_3"],
            COLORS["chart_4"], COLORS["chart_5"], COLORS["chart_6"],
            COLORS["chart_7"], COLORS["chart_8"],
        ]

        for i in range(n_samples):
            if colors:
                color = colors[i] if i < len(colors) else chart_colors[i % len(chart_colors)]
            else:
                color = chart_colors[i % len(chart_colors)]

            pen = QPen(QColor(color))
            pen.setWidth(1)

            item = self._plot.plot(
                wavelengths,
                spectra[i],
                pen=pen,
                name=self._sample_ids[i],
            )
            self._spectra.append(item)

        self.auto_range()

    def highlight_spectrum(self, index: int):
        """Highlight a specific spectrum."""
        for i, item in enumerate(self._spectra):
            pen = item.opts['pen']
            if isinstance(pen, QPen):
                if i == index:
                    pen.setWidth(3)
                else:
                    pen.setWidth(1)
                item.setPen(pen)

    def set_selected(self, index: int):
        """Set the selected spectrum."""
        self._selected_index = index
        self.highlight_spectrum(index)
        self.spectrum_selected.emit(index)


class PredVsRefPlot(InteractivePlotWidget):
    """
    Predicted vs Reference scatter plot.

    Features:
    - 1:1 reference line
    - Hover to show sample ID
    - Click to select point
    - Statistics display (R², RMSE)
    """

    point_selected = Signal(int)

    def __init__(
        self,
        title: str = "Predicted vs Reference",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(
            title=title,
            x_label="Reference",
            y_label="Predicted",
            parent=parent,
        )
        self._scatter: Optional[ScatterPlotItem] = None
        self._line = None
        self._sample_ids: List[str] = []
        self._stats_label: Optional[QLabel] = None

        # Add stats label
        self._setup_stats_label()

    def _setup_stats_label(self):
        """Add statistics label to plot."""
        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet(f"""
            color: {COLORS["text_secondary"]};
            font-size: {TYPOGRAPHY["size_sm"]}pt;
            padding: 4px {SPACING["sm"]}px;
            background-color: {COLORS["bg_elevated"]};
            border-radius: 4px;
        """)
        # Position in top-left of plot
        self.layout().insertWidget(1, self._stats_label)

    def set_data(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_ids: Optional[List[str]] = None,
        show_stats: bool = True,
    ):
        """
        Plot predicted vs reference values.

        Args:
            y_true: Reference values
            y_pred: Predicted values
            sample_ids: Optional sample identifiers for tooltips
            show_stats: Whether to show R², RMSE statistics
        """
        self.clear()

        self._sample_ids = sample_ids or [f"Sample {i+1}" for i in range(len(y_true))]

        # Calculate statistics
        if show_stats:
            ss_res = np.sum((y_true - y_pred)**2)
            ss_tot = np.sum((y_true - np.mean(y_true))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            rmse = np.sqrt(np.mean((y_true - y_pred)**2))
            bias = np.mean(y_pred - y_true)
            self._stats_label.setText(f"R² = {r2:.4f}  |  RMSE = {rmse:.4f}  |  Bias = {bias:.4f}")
        else:
            self._stats_label.setText("")

        # 1:1 reference line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        margin = (max_val - min_val) * 0.1

        line_pen = QPen(QColor(COLORS["text_tertiary"]))
        line_pen.setStyle(Qt.PenStyle.DashLine)
        line_pen.setWidth(1)

        self._line = self._plot.plot(
            [min_val - margin, max_val + margin],
            [min_val - margin, max_val + margin],
            pen=line_pen,
        )

        # Scatter plot
        brush = QBrush(QColor(COLORS["chart_1"]))
        pen = QPen(QColor(COLORS["chart_1"]))
        pen.setWidth(1)

        self._scatter = ScatterPlotItem(
            x=y_true,
            y=y_pred,
            size=8,
            pen=pen,
            brush=brush,
            hoverable=True,
            hoverPen=QPen(QColor(COLORS["accent_secondary"]), 2),
            hoverBrush=QBrush(QColor(COLORS["accent_secondary"])),
        )
        self._scatter.sigClicked.connect(self._on_point_clicked)
        self._scatter.sigHovered.connect(self._on_point_hovered)

        self._plot.addItem(self._scatter)
        self.auto_range()

    def _on_point_clicked(self, plot, points):
        if points:
            idx = points[0].index()
            self.point_selected.emit(idx)

    def _on_point_hovered(self, plot, points):
        if points:
            idx = points[0].index()
            if idx < len(self._sample_ids):
                self._coord_label.setText(f"Sample: {self._sample_ids[idx]}")

    def highlight_points(self, indices: List[int], color: str = None):
        """Highlight specific points."""
        if self._scatter is None:
            return

        n_points = len(self._scatter.data)
        if n_points == 0:
            return

        # Reset all to default
        brushes = [QBrush(QColor(COLORS["chart_1"]))] * n_points

        # Set highlight color
        highlight_color = color or COLORS["accent_danger"]
        for idx in indices:
            if 0 <= idx < len(brushes):
                brushes[idx] = QBrush(QColor(highlight_color))

        self._scatter.setBrush(brushes)


class ResidualsPlot(InteractivePlotWidget):
    """
    Residuals analysis plot.

    Features:
    - Residuals vs predicted or index
    - Zero reference line
    - ±2 sigma bands
    - Outlier highlighting
    """

    def __init__(
        self,
        title: str = "Residuals",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(
            title=title,
            x_label="Predicted",
            y_label="Residual",
            parent=parent,
        )
        self._scatter: Optional[ScatterPlotItem] = None
        self._zero_line: Optional[InfiniteLine] = None
        self._sigma_lines: List[InfiniteLine] = []
        self._residuals: Optional[np.ndarray] = None

    def set_data(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        x_axis: str = "predicted",  # "predicted" or "index"
        show_bands: bool = True,
    ):
        """
        Plot residuals.

        Args:
            y_true: Reference values
            y_pred: Predicted values
            x_axis: What to use for x-axis
            show_bands: Whether to show ±2 sigma bands
        """
        self.clear()
        self._sigma_lines = []

        self._residuals = y_pred - y_true

        if x_axis == "predicted":
            x_values = y_pred
            self._plot.setLabel("bottom", "Predicted")
        else:
            x_values = np.arange(len(self._residuals))
            self._plot.setLabel("bottom", "Sample Index")

        # Zero line
        zero_pen = QPen(QColor(COLORS["text_tertiary"]))
        zero_pen.setWidth(1)
        self._zero_line = InfiniteLine(pos=0, angle=0, pen=zero_pen)
        self._plot.addItem(self._zero_line)

        # ±2 sigma bands
        if show_bands:
            sigma = np.std(self._residuals)
            band_pen = QPen(QColor(COLORS["accent_warning"]))
            band_pen.setStyle(Qt.PenStyle.DashLine)
            band_pen.setWidth(1)

            for mult in [-2, 2]:
                line = InfiniteLine(pos=mult * sigma, angle=0, pen=band_pen)
                self._plot.addItem(line)
                self._sigma_lines.append(line)

        # Scatter plot
        brush = QBrush(QColor(COLORS["chart_2"]))
        pen = QPen(QColor(COLORS["chart_2"]))

        self._scatter = ScatterPlotItem(
            x=x_values,
            y=self._residuals,
            size=8,
            pen=pen,
            brush=brush,
            hoverable=True,
        )
        self._plot.addItem(self._scatter)

        self.auto_range()

    def get_outlier_indices(self, n_sigma: float = 2.0) -> List[int]:
        """Get indices of points outside n_sigma bands."""
        if self._residuals is None:
            return []

        sigma = np.std(self._residuals)
        outliers = np.where(np.abs(self._residuals) > n_sigma * sigma)[0]
        return outliers.tolist()


class LoadingsPlot(InteractivePlotWidget):
    """
    PLS loadings/coefficients plot.

    Features:
    - Multiple loadings (components) with different colors
    - Zero reference line
    - Wavelength-indexed x-axis
    - Peak picking
    """

    def __init__(
        self,
        title: str = "Loadings",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(
            title=title,
            x_label="Wavelength",
            y_label="Loading",
            parent=parent,
        )
        self._loadings: List = []
        self._zero_line: Optional[InfiniteLine] = None

    def set_data(
        self,
        loadings: np.ndarray,
        wavelengths: np.ndarray,
        component_names: Optional[List[str]] = None,
    ):
        """
        Plot loadings/coefficients.

        Args:
            loadings: 2D array (n_components, n_wavelengths) or 1D array
            wavelengths: Wavelength values for x-axis
            component_names: Names for each component
        """
        self.clear()
        self._loadings = []

        # Handle 1D or 2D input
        if loadings.ndim == 1:
            loadings = loadings.reshape(1, -1)

        n_components = loadings.shape[0]

        if component_names is None:
            component_names = [f"Component {i+1}" for i in range(n_components)]

        # Zero line
        zero_pen = QPen(QColor(COLORS["text_tertiary"]))
        zero_pen.setWidth(1)
        self._zero_line = InfiniteLine(pos=0, angle=0, pen=zero_pen)
        self._plot.addItem(self._zero_line)

        # Chart colors
        chart_colors = [
            COLORS["chart_1"], COLORS["chart_2"], COLORS["chart_3"],
            COLORS["chart_4"], COLORS["chart_5"],
        ]

        for i in range(n_components):
            color = chart_colors[i % len(chart_colors)]
            pen = QPen(QColor(color))
            pen.setWidth(2)

            item = self._plot.plot(
                wavelengths,
                loadings[i],
                pen=pen,
                name=component_names[i],
            )
            self._loadings.append(item)

        # Add legend
        if n_components > 1:
            self._plot.addLegend(offset=(10, 10))

        self.auto_range()

    def set_coefficients(
        self,
        coefficients: np.ndarray,
        wavelengths: np.ndarray,
        name: str = "Coefficients",
    ):
        """
        Plot regression coefficients (convenience method).

        Args:
            coefficients: 1D array of coefficients
            wavelengths: Wavelength values
            name: Label for the line
        """
        self.set_data(coefficients, wavelengths, [name])


class DiagnosticsPanel(QWidget):
    """
    Panel with 4 diagnostic plots arranged in a 2x2 grid.

    Layout:
        [Pred vs Ref]    [Residuals]
        [Loadings]       [Spectra]

    All plots are synchronized for selection highlighting.
    """

    sample_selected = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QGridLayout(self)
        layout.setContentsMargins(SPACING["sm"], SPACING["sm"], SPACING["sm"], SPACING["sm"])
        layout.setSpacing(SPACING["sm"])

        # Create plots
        self._pred_vs_ref = PredVsRefPlot("Predicted vs Reference")
        self._residuals = ResidualsPlot("Residuals")
        self._loadings = LoadingsPlot("Loadings / Coefficients")
        self._spectra = SpectraOverlayPlot("Spectra")

        # Add to grid
        layout.addWidget(self._pred_vs_ref, 0, 0)
        layout.addWidget(self._residuals, 0, 1)
        layout.addWidget(self._loadings, 1, 0)
        layout.addWidget(self._spectra, 1, 1)

        # Equal sizing
        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)

    def _connect_signals(self):
        """Connect signals for synchronized selection."""
        self._pred_vs_ref.point_selected.connect(self._on_point_selected)
        self._spectra.spectrum_selected.connect(self._on_point_selected)

    def _on_point_selected(self, index: int):
        """Handle point selection - highlight in all plots."""
        self._pred_vs_ref.highlight_points([index])
        self._spectra.highlight_spectrum(index)
        self.sample_selected.emit(index)

    def set_cv_results(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_ids: Optional[List[str]] = None,
    ):
        """
        Set cross-validation results for pred vs ref and residuals.

        Args:
            y_true: Reference values
            y_pred: CV-predicted values
            sample_ids: Sample identifiers
        """
        self._pred_vs_ref.set_data(y_true, y_pred, sample_ids)
        self._residuals.set_data(y_true, y_pred)

    def set_loadings(
        self,
        loadings: np.ndarray,
        wavelengths: np.ndarray,
        component_names: Optional[List[str]] = None,
    ):
        """Set loadings/coefficients plot data."""
        self._loadings.set_data(loadings, wavelengths, component_names)

    def set_spectra(
        self,
        spectra: np.ndarray,
        wavelengths: np.ndarray,
        sample_ids: Optional[List[str]] = None,
    ):
        """Set spectra overlay plot data."""
        self._spectra.set_data(spectra, wavelengths, sample_ids)

    def get_pred_vs_ref_plot(self) -> PredVsRefPlot:
        return self._pred_vs_ref

    def get_residuals_plot(self) -> ResidualsPlot:
        return self._residuals

    def get_loadings_plot(self) -> LoadingsPlot:
        return self._loadings

    def get_spectra_plot(self) -> SpectraOverlayPlot:
        return self._spectra

    def clear_all(self):
        """Clear all plots."""
        self._pred_vs_ref.clear()
        self._residuals.clear()
        self._loadings.clear()
        self._spectra.clear()


# =============================================================================
# BACKWARD COMPATIBILITY - Keep old PlotWidget for existing code
# =============================================================================

class PlotWidget(InteractivePlotWidget):
    """
    Backward-compatible plot widget.

    Maps old matplotlib-style API to new pyqtgraph implementation.
    """

    def __init__(self, title: str = "", parent=None):
        super().__init__(title=title, parent=parent)

    def plot_pred_vs_ref(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Predicted vs Reference",
        xlabel: str = "Reference",
        ylabel: str = "Predicted"
    ):
        """Plot predicted vs reference values with 1:1 line."""
        self.clear()
        self._plot.setLabel("bottom", xlabel)
        self._plot.setLabel("left", ylabel)

        # 1:1 line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        margin = (max_val - min_val) * 0.05
        line_range = [min_val - margin, max_val + margin]

        line_pen = QPen(QColor("#ff5555"))
        line_pen.setStyle(Qt.PenStyle.DashLine)
        self._plot.plot(line_range, line_range, pen=line_pen)

        # Scatter
        self._plot.plot(
            y_true, y_pred,
            pen=None,
            symbol='o',
            symbolSize=8,
            symbolBrush=QColor(COLORS["chart_1"]),
        )
        self.auto_range()

    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Residuals",
        xlabel: str = "Predicted",
        ylabel: str = "Residual"
    ):
        """Plot residuals vs predicted values."""
        self.clear()
        self._plot.setLabel("bottom", xlabel)
        self._plot.setLabel("left", ylabel)

        residuals = y_true - y_pred

        # Zero line
        zero_pen = QPen(QColor("#ff5555"))
        zero_pen.setStyle(Qt.PenStyle.DashLine)
        self._plot.addItem(InfiniteLine(pos=0, angle=0, pen=zero_pen))

        # ±2 sigma
        std_res = np.std(residuals)
        band_pen = QPen(QColor(COLORS["accent_warning"]))
        band_pen.setStyle(Qt.PenStyle.DotLine)
        self._plot.addItem(InfiniteLine(pos=2*std_res, angle=0, pen=band_pen))
        self._plot.addItem(InfiniteLine(pos=-2*std_res, angle=0, pen=band_pen))

        # Scatter
        self._plot.plot(
            y_pred, residuals,
            pen=None,
            symbol='o',
            symbolSize=8,
            symbolBrush=QColor(COLORS["chart_2"]),
        )
        self.auto_range()

    def plot_loadings(
        self,
        wavelengths: np.ndarray,
        loadings: np.ndarray,
        n_components: int = 3,
        title: str = "Loadings",
        xlabel: str = "Wavelength (nm)",
        ylabel: str = "Loading"
    ):
        """Plot PLS/PCA loadings."""
        self.clear()
        self._plot.setLabel("bottom", xlabel)
        self._plot.setLabel("left", ylabel)

        colors = [COLORS["chart_1"], COLORS["chart_2"], COLORS["chart_3"],
                  COLORS["chart_4"], COLORS["chart_5"]]

        # Zero line
        zero_pen = QPen(QColor(COLORS["text_tertiary"]))
        self._plot.addItem(InfiniteLine(pos=0, angle=0, pen=zero_pen))

        if loadings.ndim == 1:
            self._plot.plot(wavelengths, loadings, pen=QPen(QColor(colors[0]), 2))
        else:
            n_to_plot = min(n_components, loadings.shape[0])
            for i in range(n_to_plot):
                self._plot.plot(
                    wavelengths, loadings[i],
                    pen=QPen(QColor(colors[i % len(colors)]), 2),
                    name=f'LV{i+1}'
                )
        self.auto_range()

    def plot_coefficients(
        self,
        wavelengths: np.ndarray,
        coefficients: np.ndarray,
        title: str = "Regression Coefficients",
        xlabel: str = "Wavelength (nm)",
        ylabel: str = "Coefficient"
    ):
        """Plot regression coefficients."""
        self.plot_loadings(wavelengths, coefficients, n_components=1,
                          title=title, xlabel=xlabel, ylabel=ylabel)

    def plot_spectra(
        self,
        wavelengths: np.ndarray,
        spectra: np.ndarray,
        labels: Optional[list] = None,
        title: str = "Spectra",
        xlabel: str = "Wavelength (nm)",
        ylabel: str = "Intensity"
    ):
        """Plot one or more spectra."""
        self.clear()
        self._plot.setLabel("bottom", xlabel)
        self._plot.setLabel("left", ylabel)

        colors = [COLORS["chart_1"], COLORS["chart_2"], COLORS["chart_3"],
                  COLORS["chart_4"], COLORS["chart_5"]]

        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)

        for i, spectrum in enumerate(spectra):
            self._plot.plot(
                wavelengths, spectrum,
                pen=QPen(QColor(colors[i % len(colors)]), 1),
                name=labels[i] if labels and i < len(labels) else None
            )
        self.auto_range()

    def plot_histogram(
        self,
        values: np.ndarray,
        bins: int = 30,
        title: str = "Distribution",
        xlabel: str = "Value",
        ylabel: str = "Count"
    ):
        """Plot a histogram."""
        self.clear()
        self._plot.setLabel("bottom", xlabel)
        self._plot.setLabel("left", ylabel)

        y, x = np.histogram(values, bins=bins)
        self._plot.plot(
            x, y, stepMode="center",
            fillLevel=0,
            fillOutline=True,
            brush=QColor(COLORS["chart_1"]),
        )
        self.auto_range()

    def show_toolbar(self, show: bool = True):
        """Compatibility method - no-op for pyqtgraph."""
        pass
