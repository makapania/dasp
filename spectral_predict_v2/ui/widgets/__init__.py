"""Reusable UI widgets."""
from .file_drop import FileDropWidget, CompactFileDropWidget
from .plot_widget import (
    PlotWidget, DiagnosticsPanel,
    InteractivePlotWidget, SpectraOverlayPlot,
    PredVsRefPlot, ResidualsPlot, LoadingsPlot
)
from .column_config_dialog import ColumnConfigDialog
from .data_grid import SpectralDataModel, SpectralDataGrid, SpectralDataFilterProxy
from .mode_selector import ModeSelector, CompactModeSelector, AppMode
from .data_context_bar import DataContextBar, CompactDataContextBar, StatBadge

__all__ = [
    # File handling
    "FileDropWidget",
    "CompactFileDropWidget",
    "ColumnConfigDialog",
    # Plots
    "PlotWidget",
    "DiagnosticsPanel",
    "InteractivePlotWidget",
    "SpectraOverlayPlot",
    "PredVsRefPlot",
    "ResidualsPlot",
    "LoadingsPlot",
    # Data grid
    "SpectralDataModel",
    "SpectralDataGrid",
    "SpectralDataFilterProxy",
    # Navigation
    "ModeSelector",
    "CompactModeSelector",
    "AppMode",
    # Context
    "DataContextBar",
    "CompactDataContextBar",
    "StatBadge",
]
