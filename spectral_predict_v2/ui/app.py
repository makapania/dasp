"""
Main Application Window - Spectral Predict v2

Modern card-based interface with integrated theme system.
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

from .theme.tokens import COLORS, SPACING, RADIUS, TYPOGRAPHY, set_theme_mode
from .theme.styles import get_app_stylesheet, ThemeMode
from .widgets.mode_selector import ModeSelector, AppMode as WidgetAppMode
from .widgets.data_context_bar import DataContextBar
from .modes.explore import ExploreMode
from .modes.build import BuildMode
from .modes.predict import PredictMode


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

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.accept)
        layout.addWidget(button_box)


class SpectralPredictApp(QMainWindow):
    """
    Main application window for Spectral Predict v2.

    Modern dark theme with card-based layout and integrated
    data grid, visualizations, and analysis tools.
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
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)

        # Try to maximize on start for best experience
        # self.showMaximized()  # Uncomment to start maximized

    def _setup_menu(self):
        """Create the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open File...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_data)
        file_menu.addAction(open_action)

        open_folder_action = QAction("Open &Folder...", self)
        open_folder_action.setShortcut("Ctrl+Shift+O")
        open_folder_action.triggered.connect(self._open_folder)
        file_menu.addAction(open_folder_action)

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
        self.data_bar = DataContextBar()
        self.data_bar.file_clicked.connect(self._open_data)
        layout.addWidget(self.data_bar)

        # Mode selector container
        mode_container = QWidget()
        mode_container.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['bg_base']};
                border-bottom: 1px solid {COLORS['border_subtle']};
            }}
        """)
        mode_layout = QVBoxLayout(mode_container)
        mode_layout.setContentsMargins(SPACING["xl"], SPACING["md"], SPACING["xl"], SPACING["md"])

        self.mode_selector = ModeSelector()
        mode_layout.addWidget(self.mode_selector, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(mode_container)

        # Stacked widget for mode views
        self.mode_stack = QStackedWidget()
        self.mode_stack.setStyleSheet(f"background-color: {COLORS['bg_base']};")

        self.explore_mode = ExploreMode(self.state, self.job_runner, self.config_manager)
        self.build_mode = BuildMode(self.state, self.job_runner)
        self.predict_mode = PredictMode(self.state, self.job_runner)

        self.mode_stack.addWidget(self.explore_mode)
        self.mode_stack.addWidget(self.build_mode)
        self.mode_stack.addWidget(self.predict_mode)

        layout.addWidget(self.mode_stack, 1)

        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(f"""
            QStatusBar {{
                background-color: {COLORS['bg_surface']};
                color: {COLORS['text_secondary']};
                border-top: 1px solid {COLORS['border_subtle']};
                padding: {SPACING['xs']}px {SPACING['sm']}px;
            }}
        """)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _connect_signals(self):
        """Connect signals between components."""
        # Mode switching - map widget AppMode to state AppMode
        def on_widget_mode_changed(widget_mode):
            mode_map = {
                WidgetAppMode.EXPLORE: AppMode.EXPLORE,
                WidgetAppMode.BUILD: AppMode.BUILD,
                WidgetAppMode.PREDICT: AppMode.PREDICT,
            }
            self._on_mode_changed(mode_map.get(widget_mode, AppMode.EXPLORE))

        self.mode_selector.mode_changed.connect(on_widget_mode_changed)

        # State mode changes update selector
        def on_state_mode_changed(state_mode):
            mode_map = {
                AppMode.EXPLORE: WidgetAppMode.EXPLORE,
                AppMode.BUILD: WidgetAppMode.BUILD,
                AppMode.PREDICT: WidgetAppMode.PREDICT,
            }
            self.mode_selector.set_mode(mode_map.get(state_mode, WidgetAppMode.EXPLORE))

        self.state.mode_changed.connect(on_state_mode_changed)

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

        # Data changes update context bar
        self.state.data_changed.connect(self._on_data_changed)

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

    def _on_data_changed(self):
        """Handle data changes and update context bar."""
        if self.state.has_data:
            data = self.state.data
            self.data_bar.set_data(
                file_name=data.file_name or "Unknown",
                n_samples=data.n_samples,
                n_wavelengths=data.n_wavelengths,
                wavelength_range=(data.wavelength_min, data.wavelength_max),
                target_column=data.target_column,
                target_range=(data.y.min(), data.y.max()) if data.y is not None else None,
                missing_count=0,
                outlier_count=len(self.state.flagged_rows) if hasattr(self.state, 'flagged_rows') else 0
            )
        else:
            self.data_bar.clear()

    # --- File Actions ---

    def _open_data(self):
        """Open a data file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Spectral Data",
            "",
            "All Supported (*.csv *.xlsx *.xls *.asd *.spc *.jdx *.dx);;"
            "CSV Files (*.csv);;"
            "Excel Files (*.xlsx *.xls);;"
            "ASD Files (*.asd);;"
            "SPC Files (*.spc);;"
            "JCAMP-DX Files (*.jdx *.dx);;"
            "All Files (*)"
        )

        if file_path:
            self.status_bar.showMessage(f"Loading: {file_path}")
            # Switch to Explore mode and trigger file selection
            self.state.set_mode(AppMode.EXPLORE)
            self.explore_mode.file_drop.set_file(file_path)

    def _open_folder(self):
        """Open a folder containing spectral files (ASD, SPC, etc.)."""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Open Spectral Data Folder",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        if folder_path:
            self.status_bar.showMessage(f"Loading folder: {folder_path}")
            # Switch to Explore mode and trigger file selection
            self.state.set_mode(AppMode.EXPLORE)
            self.explore_mode.file_drop.set_file(folder_path)

    def _save_model(self):
        """Save the current model."""
        if not self.state.model.selected_model:
            QMessageBox.warning(self, "Save Model", "No model to save. Train a model first.")
            return

        # Delegate to Build mode
        self.build_mode._save_model()

    def _load_model(self):
        """Load a saved model."""
        # Switch to Predict mode which handles model loading
        self.state.set_mode(AppMode.PREDICT)
        self.predict_mode._load_model()

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
        mode = ThemeMode.DARK if self._dark_mode else ThemeMode.LIGHT
        set_theme_mode(mode)  # Update global COLORS dict
        self.setStyleSheet(get_app_stylesheet(mode))

    # --- Tools Actions ---

    def _show_quality_check(self):
        """Show the data quality check tool."""
        if not self.state.has_data:
            QMessageBox.warning(self, "Quality Check", "Load data first to run quality checks.")
            return

        try:
            from .tools import DataQualityTool
            tool = DataQualityTool(self.state)
            dialog = ToolDialog(tool, "Data Quality Assessment", self)
            dialog.exec()
        except ImportError:
            QMessageBox.information(self, "Coming Soon", "Data Quality tool will be available soon.")

    def _show_calibration_transfer(self):
        """Show the calibration transfer tool."""
        try:
            from .tools import CalibrationTransferTool
            tool = CalibrationTransferTool(self.state)
            dialog = ToolDialog(tool, "Calibration Transfer", self)
            dialog.exec()
        except ImportError:
            QMessageBox.information(self, "Coming Soon", "Calibration Transfer tool will be available soon.")

    def _show_interference_removal(self):
        """Show the interference removal tool."""
        try:
            from .tools import InterferenceRemovalTool
            tool = InterferenceRemovalTool(self.state)
            dialog = ToolDialog(tool, "Interference Removal", self)
            dialog.exec()
        except ImportError:
            QMessageBox.information(self, "Coming Soon", "Interference Removal tool will be available soon.")

    def _show_preset_manager(self):
        """Show the preset manager."""
        try:
            from .tools import PresetManagerTool
            tool = PresetManagerTool(self.config_manager)
            tool.preset_selected.connect(self._on_preset_applied)
            dialog = ToolDialog(tool, "Preset Manager", self)
            dialog.exec()
        except ImportError:
            QMessageBox.information(self, "Coming Soon", "Preset Manager will be available soon.")

    def _on_preset_applied(self, preset_key: str):
        """Handle preset application from preset manager."""
        self.state.set_preset(preset_key)
        self.state.set_mode(AppMode.EXPLORE)
        self.status_bar.showMessage(f"Applied preset: {preset_key}")

    # --- Help Actions ---

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Spectral Predict",
            f"""<h2 style="color: {COLORS['accent_primary']};">Spectral Predict v2.0</h2>
            <p>Modern Automated Spectral Analysis Platform</p>
            <p style="color: {COLORS['text_secondary']};">
            A replacement for SIMCA and Unscrambler with superior automation.
            </p>
            <hr>
            <p><b>Features:</b></p>
            <ul>
            <li>Automated model discovery and ranking</li>
            <li>Excel-like data editing with fill-down</li>
            <li>Interactive pyqtgraph visualizations</li>
            <li>Bayesian hyperparameter optimization</li>
            <li>Preset-driven workflows</li>
            <li>Calibration transfer</li>
            <li>Interference removal</li>
            </ul>
            """
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


def main():
    """Launch the application."""
    from PySide6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = SpectralPredictApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
