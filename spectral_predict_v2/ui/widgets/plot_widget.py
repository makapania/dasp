"""
Plot Widget - Matplotlib-based plotting for model diagnostics.

Provides reusable plot types:
- Predicted vs Reference
- Residuals
- Loadings/Coefficients
- Spectra overlay
"""

from typing import Optional
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PySide6.QtCore import Qt

# Use matplotlib with Qt backend
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class PlotWidget(QWidget):
    """
    A reusable matplotlib plot widget with toolbar.
    """

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.title = title

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Title label
        if self.title:
            title_label = QLabel(self.title)
            title_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #aaa;")
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(title_label)

        # Create matplotlib figure
        self.figure = Figure(figsize=(5, 4), dpi=100, facecolor='#1e1e1e')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: #1e1e1e;")
        layout.addWidget(self.canvas)

        # Toolbar (optional, can be hidden)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setVisible(False)  # Hidden by default
        layout.addWidget(self.toolbar)

        # Create axes
        self.ax = self.figure.add_subplot(111)
        self._style_axes()

    def _style_axes(self):
        """Apply dark theme styling to axes."""
        self.ax.set_facecolor('#2d2d2d')
        self.ax.tick_params(colors='#aaa')
        self.ax.xaxis.label.set_color('#aaa')
        self.ax.yaxis.label.set_color('#aaa')
        self.ax.title.set_color('#e0e0e0')
        for spine in self.ax.spines.values():
            spine.set_color('#555')

    def clear(self):
        """Clear the plot."""
        self.ax.clear()
        self._style_axes()
        self.canvas.draw()

    def show_toolbar(self, show: bool = True):
        """Show or hide the navigation toolbar."""
        self.toolbar.setVisible(show)

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

        # Scatter plot
        self.ax.scatter(y_true, y_pred, alpha=0.6, c='#4fc3f7', edgecolors='none', s=40)

        # 1:1 line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        margin = (max_val - min_val) * 0.05
        line_range = [min_val - margin, max_val + margin]
        self.ax.plot(line_range, line_range, 'r--', alpha=0.7, label='1:1 line')

        # Labels
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title, color='#e0e0e0')

        # Equal aspect
        self.ax.set_xlim(line_range)
        self.ax.set_ylim(line_range)

        self.figure.tight_layout()
        self.canvas.draw()

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

        residuals = y_true - y_pred

        # Scatter plot
        self.ax.scatter(y_pred, residuals, alpha=0.6, c='#81c784', edgecolors='none', s=40)

        # Zero line
        self.ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)

        # +/- 2 std lines
        std_res = np.std(residuals)
        self.ax.axhline(y=2*std_res, color='orange', linestyle=':', alpha=0.5, label='+2σ')
        self.ax.axhline(y=-2*std_res, color='orange', linestyle=':', alpha=0.5, label='-2σ')

        # Labels
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title, color='#e0e0e0')

        self.figure.tight_layout()
        self.canvas.draw()

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

        colors = ['#4fc3f7', '#81c784', '#ffb74d', '#f48fb1', '#ce93d8']

        # Handle 1D or 2D loadings
        if loadings.ndim == 1:
            self.ax.plot(wavelengths, loadings, c=colors[0], alpha=0.8, label='LV1')
        else:
            n_to_plot = min(n_components, loadings.shape[0])
            for i in range(n_to_plot):
                self.ax.plot(
                    wavelengths, loadings[i],
                    c=colors[i % len(colors)],
                    alpha=0.8,
                    label=f'LV{i+1}'
                )

        # Zero line
        self.ax.axhline(y=0, color='#555', linestyle='-', alpha=0.5)

        # Labels
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title, color='#e0e0e0')
        self.ax.legend(loc='best', facecolor='#2d2d2d', edgecolor='#555', labelcolor='#aaa')

        self.figure.tight_layout()
        self.canvas.draw()

    def plot_coefficients(
        self,
        wavelengths: np.ndarray,
        coefficients: np.ndarray,
        title: str = "Regression Coefficients",
        xlabel: str = "Wavelength (nm)",
        ylabel: str = "Coefficient"
    ):
        """Plot regression coefficients."""
        self.clear()

        # Fill between positive and negative
        self.ax.fill_between(
            wavelengths, 0, coefficients,
            where=(coefficients >= 0),
            color='#4fc3f7', alpha=0.5
        )
        self.ax.fill_between(
            wavelengths, 0, coefficients,
            where=(coefficients < 0),
            color='#f48fb1', alpha=0.5
        )
        self.ax.plot(wavelengths, coefficients, c='#e0e0e0', alpha=0.8, linewidth=0.8)

        # Zero line
        self.ax.axhline(y=0, color='#555', linestyle='-', alpha=0.5)

        # Labels
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title, color='#e0e0e0')

        self.figure.tight_layout()
        self.canvas.draw()

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

        colors = ['#4fc3f7', '#81c784', '#ffb74d', '#f48fb1', '#ce93d8']

        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)

        for i, spectrum in enumerate(spectra):
            label = labels[i] if labels and i < len(labels) else None
            self.ax.plot(
                wavelengths, spectrum,
                c=colors[i % len(colors)],
                alpha=0.7,
                label=label
            )

        # Labels
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title, color='#e0e0e0')

        if labels:
            self.ax.legend(loc='best', facecolor='#2d2d2d', edgecolor='#555', labelcolor='#aaa')

        self.figure.tight_layout()
        self.canvas.draw()

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

        self.ax.hist(values, bins=bins, color='#4fc3f7', alpha=0.7, edgecolor='#2d2d2d')

        # Labels
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title, color='#e0e0e0')

        self.figure.tight_layout()
        self.canvas.draw()


class DiagnosticsPanel(QWidget):
    """
    A panel with three diagnostic plots: Pred vs Ref, Residuals, and Loadings/Coefficients.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Three plots side by side
        self.pred_ref_plot = PlotWidget("Predicted vs Reference")
        self.residuals_plot = PlotWidget("Residuals")
        self.loadings_plot = PlotWidget("Loadings / Coefficients")

        layout.addWidget(self.pred_ref_plot)
        layout.addWidget(self.residuals_plot)
        layout.addWidget(self.loadings_plot)

    def update_plots(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        wavelengths: Optional[np.ndarray] = None,
        loadings: Optional[np.ndarray] = None,
        coefficients: Optional[np.ndarray] = None
    ):
        """Update all diagnostic plots."""
        # Pred vs Ref
        self.pred_ref_plot.plot_pred_vs_ref(y_true, y_pred)

        # Residuals
        self.residuals_plot.plot_residuals(y_true, y_pred)

        # Loadings or Coefficients
        if loadings is not None and wavelengths is not None:
            self.loadings_plot.plot_loadings(wavelengths, loadings)
        elif coefficients is not None and wavelengths is not None:
            self.loadings_plot.plot_coefficients(wavelengths, coefficients)
        else:
            self.loadings_plot.clear()

    def clear_all(self):
        """Clear all plots."""
        self.pred_ref_plot.clear()
        self.residuals_plot.clear()
        self.loadings_plot.clear()
