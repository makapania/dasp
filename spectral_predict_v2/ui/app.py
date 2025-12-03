"""
Main Application Window - Spectral Predict v2

Combines mode selector, data context bar, and mode views into a unified interface.
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QStackedWidget,
    QMenuBar, QMenu, QStatusBar, QMessageBox, QFileDialog,
    QDialog, QDialogButtonBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence

from orchestration.state_store import StateStore, AppMode
from orchestration.job_runner import JobRunner
from orchestration.config_manager import ConfigManager
from .mode_selector import ModeSelector
from .data_context_bar import DataContextBar
from .modes.explore import ExploreMode
from .modes.build import BuildMode
from .modes.predict import PredictMode
from .tools import CalibrationTransferTool, DataQualityTool, InterferenceRemovalTool, PresetManagerTool


class ToolDialog(QDialog):
    """Dialog wrapper for tool panels."""

    def __init__(self, tool_widget: QWidget, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(800, 600)
        self.resize(900, 700)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(tool_widget)

        # Close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.accept)
        layout.addWidget(button_box)


class SpectralPredictApp(QMainWindow):
    """
    Main application window for Spectral Predict v2.
    """

    def __init__(self):
        super().__init__()

        # Core services
        self.state = StateStore()
        self.job_runner = JobRunner()
        self.config_manager = ConfigManager()

        # Theme state
        self._dark_mode = True

        self._setup_window()
        self._setup_menu()
        self._setup_ui()
        self._connect_signals()
        self._apply_theme()

    def _setup_window(self):
        """Configure the main window."""
        self.setWindowTitle("Spectral Predict v2")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

    def _setup_menu(self):
        """Create the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open Data...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_data)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        save_model_action = QAction("&Save Model...", self)
        save_model_action.setShortcut(QKeySequence.StandardKey.Save)
        save_model_action.triggered.connect(self._save_model)
        file_menu.addAction(save_model_action)

        load_model_action = QAction("&Load Model...", self)
        load_model_action.triggered.connect(self._load_model)
        file_menu.addAction(load_model_action)

        file_menu.addSeparator()

        export_action = QAction("&Export Results...", self)
        export_action.triggered.connect(self._export_results)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        toggle_theme_action = QAction("Toggle &Dark/Light Mode", self)
        toggle_theme_action.setShortcut("Ctrl+D")
        toggle_theme_action.triggered.connect(self._toggle_theme)
        view_menu.addAction(toggle_theme_action)

        # Tools menu
        tools_menu = menubar.addMenu("&Tools")

        quality_action = QAction("&Data Quality Check", self)
        quality_action.triggered.connect(self._show_quality_check)
        tools_menu.addAction(quality_action)

        tools_menu.addSeparator()

        transfer_action = QAction("&Calibration Transfer", self)
        transfer_action.triggered.connect(self._show_calibration_transfer)
        tools_menu.addAction(transfer_action)

        interference_action = QAction("&Interference Removal", self)
        interference_action.triggered.connect(self._show_interference_removal)
        tools_menu.addAction(interference_action)

        tools_menu.addSeparator()

        preset_action = QAction("&Manage Presets...", self)
        preset_action.triggered.connect(self._show_preset_manager)
        tools_menu.addAction(preset_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_ui(self):
        """Create the main UI layout."""
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Data context bar (always visible at top)
        self.data_bar = DataContextBar(self.state)
        layout.addWidget(self.data_bar)

        # Mode selector
        mode_container = QWidget()
        mode_layout = QVBoxLayout(mode_container)
        mode_layout.setContentsMargins(20, 12, 20, 12)

        self.mode_selector = ModeSelector()
        mode_layout.addWidget(self.mode_selector, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(mode_container)

        # Stacked widget for mode views
        self.mode_stack = QStackedWidget()

        self.explore_mode = ExploreMode(self.state, self.job_runner, self.config_manager)
        self.build_mode = BuildMode(self.state, self.job_runner)
        self.predict_mode = PredictMode(self.state, self.job_runner)

        self.mode_stack.addWidget(self.explore_mode)
        self.mode_stack.addWidget(self.build_mode)
        self.mode_stack.addWidget(self.predict_mode)

        layout.addWidget(self.mode_stack, 1)  # Stretch factor 1

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _connect_signals(self):
        """Connect signals between components."""
        # Mode switching
        self.mode_selector.mode_changed.connect(self._on_mode_changed)
        self.state.mode_changed.connect(self.mode_selector.set_mode)

        # Data bar actions
        self.data_bar.load_data_clicked.connect(self._open_data)
        self.data_bar.quality_check_clicked.connect(self._show_quality_check)

        # Job runner
        self.job_runner.job_started.connect(
            lambda _: self.status_bar.showMessage("Analysis running...")
        )
        self.job_runner.job_completed.connect(
            lambda _, r: self.status_bar.showMessage(
                "Analysis complete" if r.success else f"Error: {r.error}"
            )
        )

        # Analysis signals
        self.state.analysis_started.connect(
            lambda: self.status_bar.showMessage("Analysis started...")
        )
        self.state.analysis_completed.connect(self._on_analysis_completed)

    def _on_mode_changed(self, mode: AppMode):
        """Handle mode change."""
        self.state.set_mode(mode)

        if mode == AppMode.EXPLORE:
            self.mode_stack.setCurrentWidget(self.explore_mode)
        elif mode == AppMode.BUILD:
            self.mode_stack.setCurrentWidget(self.build_mode)
        elif mode == AppMode.PREDICT:
            self.mode_stack.setCurrentWidget(self.predict_mode)

    def _on_analysis_completed(self):
        """Handle analysis completion."""
        n_models = self.state.analysis.n_models_evaluated
        best = self.state.analysis.best_model_name
        self.status_bar.showMessage(f"Analysis complete: {n_models} models evaluated. Best: {best}")

    # --- File Actions ---

    def _open_data(self):
        """Open a data file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Spectral Data",
            "",
            "All Supported (*.csv *.xlsx *.xls *.asd *.spc *.jdx *.dx);;"\
            "CSV Files (*.csv);;"\
            "Excel Files (*.xlsx *.xls);;"\
            "ASD Files (*.asd);;"\
            "SPC Files (*.spc);;"\
            "JCAMP-DX Files (*.jdx *.dx);;"\
            "All Files (*)"
        )

        if file_path:
            # TODO: Use engine API to load data
            self.status_bar.showMessage(f"Loading: {file_path}")
            # For now, just show a message
            QMessageBox.information(
                self,
                "Load Data",
                f"Data loading will be implemented in Phase 2.\n\nFile: {file_path}"
            )

    def _save_model(self):
        """Save the current model."""
        if not self.state.model.selected_model:
            QMessageBox.warning(self, "Save Model", "No model to save. Train a model first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Model",
            "",
            "DASP Model Files (*.dasp);;All Files (*)"
        )

        if file_path:
            # TODO: Implement model saving
            QMessageBox.information(self, "Save Model", f"Model saving will be implemented.\n\nPath: {file_path}")

    def _load_model(self):
        """Load a saved model."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Model",
            "",
            "DASP Model Files (*.dasp);;All Files (*)"
        )

        if file_path:
            # TODO: Implement model loading
            QMessageBox.information(self, "Load Model", f"Model loading will be implemented.\n\nPath: {file_path}")
            # Switch to Predict mode
            self.state.set_mode(AppMode.PREDICT)

    def _export_results(self):
        """Export analysis results."""
        if self.state.analysis.results_df is None:
            QMessageBox.warning(self, "Export", "No results to export. Run an analysis first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            "",
            "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
        )

        if file_path:
            try:
                if file_path.endswith(".xlsx"):
                    self.state.analysis.results_df.to_excel(file_path, index=False)
                else:
                    self.state.analysis.results_df.to_csv(file_path, index=False)
                self.status_bar.showMessage(f"Results exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    # --- View Actions ---

    def _toggle_theme(self):
        """Toggle between dark and light mode."""
        self._dark_mode = not self._dark_mode
        self._apply_theme()

    def _apply_theme(self):
        """Apply the current theme."""
        if self._dark_mode:
            self.setStyleSheet(self._get_dark_stylesheet())
        else:
            self.setStyleSheet(self._get_light_stylesheet())

    def _get_dark_stylesheet(self) -> str:
        """Return dark mode stylesheet."""
        return """
            QMainWindow, QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QMenuBar {
                background-color: #2d2d2d;
                color: #e0e0e0;
            }
            QMenuBar::item:selected {
                background-color: #3d3d3d;
            }
            QMenu {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: 1px solid #444;
            }
            QMenu::item:selected {
                background-color: #0078d4;
            }
            QStatusBar {
                background-color: #2d2d2d;
                color: #aaa;
            }
            QScrollBar:vertical {
                background-color: #2d2d2d;
                width: 12px;
            }
            QScrollBar::handle:vertical {
                background-color: #555;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #666;
            }
            QPushButton {
                background-color: #0078d4;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1084d8;
            }
            QPushButton:pressed {
                background-color: #006cbd;
            }
            QPushButton:disabled {
                background-color: #444;
                color: #888;
            }
            QComboBox {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px 12px;
                color: #e0e0e0;
            }
            QComboBox:hover {
                border-color: #0078d4;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px;
                color: #e0e0e0;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #0078d4;
            }
            QTableWidget, QTableView {
                background-color: #2d2d2d;
                border: 1px solid #444;
                gridline-color: #444;
            }
            QTableWidget::item, QTableView::item {
                padding: 4px;
            }
            QTableWidget::item:selected, QTableView::item:selected {
                background-color: #0078d4;
            }
            QHeaderView::section {
                background-color: #3d3d3d;
                border: none;
                border-bottom: 1px solid #555;
                padding: 6px;
                font-weight: bold;
            }
            QProgressBar {
                background-color: #3d3d3d;
                border: none;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 4px;
            }
            QGroupBox {
                border: 1px solid #444;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
            }
        """

    def _get_light_stylesheet(self) -> str:
        """Return light mode stylesheet."""
        return """
            QMainWindow, QWidget {
                background-color: #f5f5f5;
                color: #333;
            }
            QMenuBar {
                background-color: #fff;
                color: #333;
            }
            QMenuBar::item:selected {
                background-color: #e0e0e0;
            }
            QMenu {
                background-color: #fff;
                color: #333;
                border: 1px solid #ccc;
            }
            QMenu::item:selected {
                background-color: #0078d4;
                color: white;
            }
            QStatusBar {
                background-color: #e0e0e0;
                color: #666;
            }
            QPushButton {
                background-color: #0078d4;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1084d8;
            }
            QComboBox {
                background-color: #fff;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox {
                background-color: #fff;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 6px;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #0078d4;
            }
            QTableWidget, QTableView {
                background-color: #fff;
                border: 1px solid #ddd;
            }
            QTableWidget::item:selected, QTableView::item:selected {
                background-color: #0078d4;
                color: white;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                border: none;
                border-bottom: 1px solid #ddd;
                padding: 6px;
                font-weight: bold;
            }
            QGroupBox {
                border: 1px solid #ddd;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
            }
        """

    # --- Tools Actions ---

    def _show_quality_check(self):
        """Show the data quality check tool."""
        if not self.state.has_data:
            QMessageBox.warning(self, "Quality Check", "Load data first to run quality checks.")
            return

        tool = DataQualityTool(self.state)
        dialog = ToolDialog(tool, "Data Quality Assessment", self)
        dialog.exec()

    def _show_calibration_transfer(self):
        """Show the calibration transfer tool."""
        tool = CalibrationTransferTool(self.state)
        dialog = ToolDialog(tool, "Calibration Transfer", self)
        dialog.exec()

    def _show_interference_removal(self):
        """Show the interference removal tool."""
        tool = InterferenceRemovalTool(self.state)
        dialog = ToolDialog(tool, "Interference Removal", self)
        dialog.exec()

    def _show_preset_manager(self):
        """Show the preset manager."""
        tool = PresetManagerTool(self.config_manager)
        tool.preset_selected.connect(self._on_preset_applied)
        dialog = ToolDialog(tool, "Preset Manager", self)
        dialog.exec()

    def _on_preset_applied(self, preset_key: str):
        """Handle preset application from preset manager."""
        self.state.set_preset(preset_key)
        # Switch to Explore mode and update preset
        self.state.set_mode(AppMode.EXPLORE)
        self.explore_mode.set_preset(preset_key)
        self.status_bar.showMessage(f"Applied preset: {preset_key}")

    # --- Help Actions ---

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Spectral Predict",
            "<h2>Spectral Predict v2.0</h2>"
            "<p>Modern Automated Spectral Analysis Platform</p>"
            "<p>A replacement for SIMCA and Unscrambler with superior automation.</p>"
            "<hr>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>Automated model discovery and ranking</li>"
            "<li>Bayesian hyperparameter optimization</li>"
            "<li>Preset-driven workflows</li>"
            "<li>Calibration transfer</li>"
            "<li>Interference removal</li>"
            "</ul>"
        )

    def closeEvent(self, event):
        """Handle window close."""
        if self.job_runner.has_active_jobs():
            reply = QMessageBox.question(
                self,
                "Analysis Running",
                "An analysis is still running. Are you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return

            self.job_runner.cancel_all()

        event.accept()
