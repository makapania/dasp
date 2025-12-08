"""
Interference Removal Tool - Remove spectral interferences.

Supports methods:
- MSC (Multiplicative Scatter Correction)
- OSC (Orthogonal Signal Correction)
- EPO (External Parameter Orthogonalization)
- DOSC (Direct OSC)
"""

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QFrame,
    QFileDialog, QMessageBox, QCheckBox, QTextEdit
)
from PySide6.QtCore import Qt, QThread, Signal

from orchestration.state_store import StateStore
from ui.widgets.file_drop import FileDropWidget
from ui.widgets.plot_widget import PlotWidget


class InterferenceWorker(QThread):
    """Background worker for interference removal."""

    finished = Signal(object)  # (X_corrected, info_dict) or Exception
    progress = Signal(str)

    def __init__(self, X, y, method, params, interference_spectra=None):
        super().__init__()
        self.X = X
        self.y = y
        self.method = method
        self.params = params
        self.interference_spectra = interference_spectra

    def run(self):
        try:
            from src.spectral_predict import interference as intf

            info = {"method": self.method}

            if self.method == 'msc':
                self.progress.emit("Applying Multiplicative Scatter Correction...")
                transformer = intf.MSC()
                X_corrected = transformer.fit_transform(self.X)
                info["reference_spectrum"] = transformer.mean_spectrum_

            elif self.method == 'osc':
                self.progress.emit("Fitting Orthogonal Signal Correction...")
                n_components = self.params.get('n_components', 2)
                transformer = intf.OSC(n_components=n_components)
                X_corrected = transformer.fit_transform(self.X, self.y)
                info["n_components"] = n_components
                info["variance_removed"] = transformer.explained_variance_ratio_ if hasattr(transformer, 'explained_variance_ratio_') else None

            elif self.method == 'epo':
                self.progress.emit("Fitting External Parameter Orthogonalization...")
                if self.interference_spectra is None:
                    raise ValueError("EPO requires interference spectra")
                n_components = self.params.get('n_components', 5)
                transformer = intf.EPO(n_components=n_components)
                transformer.fit(self.interference_spectra)
                X_corrected = transformer.transform(self.X)
                info["n_components"] = n_components

            elif self.method == 'dosc':
                self.progress.emit("Fitting Direct Orthogonal Signal Correction...")
                n_components = self.params.get('n_components', 2)
                transformer = intf.DOSC(n_components=n_components)
                X_corrected = transformer.fit_transform(self.X, self.y)
                info["n_components"] = n_components

            else:
                raise ValueError(f"Unknown method: {self.method}")

            self.finished.emit((X_corrected, info))

        except Exception as e:
            self.finished.emit(e)


class InterferenceRemovalTool(QWidget):
    """
    Tool panel for removing spectral interferences.

    Methods:
    - MSC: Corrects multiplicative scatter effects
    - OSC: Removes Y-orthogonal variation
    - EPO: Removes known interferent signatures
    - DOSC: Direct version of OSC (faster)
    """

    def __init__(self, state: StateStore, parent=None):
        super().__init__(parent)
        self.state = state
        self._interference_data = None  # For EPO
        self._worker = None
        self._corrected_X = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Header
        header = QLabel("Interference Removal")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #e0e0e0;")
        layout.addWidget(header)

        desc = QLabel(
            "Remove spectral interferences such as scatter effects, moisture, "
            "or other unwanted spectral variation."
        )
        desc.setStyleSheet("color: #888; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Method selection
        method_group = QGroupBox("Method")
        method_layout = QVBoxLayout(method_group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Method:"))

        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "MSC (Multiplicative Scatter Correction)",
            "OSC (Orthogonal Signal Correction)",
            "DOSC (Direct OSC)",
            "EPO (External Parameter Orthogonalization)",
        ])
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        row1.addWidget(self.method_combo)

        row1.addStretch()
        method_layout.addLayout(row1)

        # Method descriptions
        self.method_desc = QLabel("")
        self.method_desc.setStyleSheet("color: #888; font-size: 11px; padding: 8px;")
        self.method_desc.setWordWrap(True)
        method_layout.addWidget(self.method_desc)

        layout.addWidget(method_group)

        # Parameters section
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_group)

        # Components row (for OSC, DOSC, EPO)
        self.components_row = QHBoxLayout()
        self.components_label = QLabel("Components to remove:")
        self.components_row.addWidget(self.components_label)

        self.n_components_spin = QSpinBox()
        self.n_components_spin.setRange(1, 20)
        self.n_components_spin.setValue(2)
        self.components_row.addWidget(self.n_components_spin)

        self.components_row.addStretch()
        params_layout.addLayout(self.components_row)

        # EPO interference spectra
        self.epo_frame = QFrame()
        epo_layout = QVBoxLayout(self.epo_frame)
        epo_layout.setContentsMargins(0, 0, 0, 0)

        epo_label = QLabel("Interference Spectra (for EPO):")
        epo_label.setStyleSheet("font-weight: bold;")
        epo_layout.addWidget(epo_label)

        epo_desc = QLabel(
            "Load spectra that represent the interference to remove "
            "(e.g., pure water spectra for moisture correction)."
        )
        epo_desc.setStyleSheet("color: #888; font-size: 11px;")
        epo_desc.setWordWrap(True)
        epo_layout.addWidget(epo_desc)

        self.interference_drop = FileDropWidget()
        self.interference_drop.file_selected.connect(self._on_interference_selected)
        epo_layout.addWidget(self.interference_drop)

        self.interference_info = QLabel("")
        self.interference_info.setStyleSheet("color: #888; font-size: 11px;")
        epo_layout.addWidget(self.interference_info)

        params_layout.addWidget(self.epo_frame)

        layout.addWidget(params_group)

        # Preview plot
        self.preview_plot = PlotWidget("Before/After Comparison")
        self.preview_plot.setMinimumHeight(200)
        layout.addWidget(self.preview_plot)

        # Action buttons
        btn_row = QHBoxLayout()

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888;")
        btn_row.addWidget(self.status_label)

        btn_row.addStretch()

        self.apply_btn = QPushButton("Apply Correction")
        self.apply_btn.clicked.connect(self._apply_correction)
        btn_row.addWidget(self.apply_btn)

        self.save_btn = QPushButton("Apply & Update Data")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_to_state)
        btn_row.addWidget(self.save_btn)

        layout.addLayout(btn_row)

        layout.addStretch()

        # Initialize UI state
        self._on_method_changed(0)

    def _connect_signals(self):
        """Connect state signals."""
        self.state.data_changed.connect(self._on_data_changed)

    def _on_data_changed(self):
        """Handle data changes."""
        self._corrected_X = None
        self.preview_plot.clear()
        self.save_btn.setEnabled(False)

    def _on_method_changed(self, index):
        """Update UI based on selected method."""
        methods = ['msc', 'osc', 'dosc', 'epo']
        method = methods[index]

        # Show/hide components spinner
        needs_components = method in ['osc', 'dosc', 'epo']
        self.components_label.setVisible(needs_components)
        self.n_components_spin.setVisible(needs_components)

        # Show/hide EPO interference spectra
        self.epo_frame.setVisible(method == 'epo')

        # Update description
        descriptions = {
            'msc': (
                "MSC corrects for multiplicative scatter effects by normalizing each spectrum "
                "to a reference (mean) spectrum. Good for correcting particle size effects in NIR."
            ),
            'osc': (
                "OSC removes spectral variation that is orthogonal to the Y variable. "
                "This removes interference that doesn't correlate with your target property. "
                "Requires Y values."
            ),
            'dosc': (
                "Direct OSC is a faster, PLS-based version of OSC that achieves similar results "
                "with better computational efficiency. Requires Y values."
            ),
            'epo': (
                "EPO removes known interferent signatures from spectra. You provide spectra "
                "of the pure interferent (e.g., water for moisture), and EPO projects it out. "
                "Useful when you know what interference to remove."
            ),
        }
        self.method_desc.setText(descriptions.get(method, ""))

    def _on_interference_selected(self, file_path: str):
        """Handle interference spectra file selection."""
        try:
            from engine.api import EngineAPI
            engine = EngineAPI()
            loaded = engine.load_data(file_path, for_prediction=True)
            self._interference_data = loaded.X

            self.interference_info.setText(
                f"Loaded: {loaded.metadata['n_samples']} spectra | "
                f"{loaded.metadata['n_wavelengths']} wavelengths"
            )

        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            self._interference_data = None

    def _apply_correction(self):
        """Apply the selected correction method."""
        if not self.state.has_data:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return

        if self._worker is not None and self._worker.isRunning():
            return

        methods = ['msc', 'osc', 'dosc', 'epo']
        method = methods[self.method_combo.currentIndex()]

        # Check requirements
        if method in ['osc', 'dosc'] and self.state.data.y is None:
            QMessageBox.warning(
                self, "Missing Y Values",
                f"{method.upper()} requires Y values. Please load data with a target column."
            )
            return

        if method == 'epo' and self._interference_data is None:
            QMessageBox.warning(
                self, "Missing Interference Spectra",
                "EPO requires interference spectra. Please load interferent spectra first."
            )
            return

        params = {
            'n_components': self.n_components_spin.value(),
        }

        self.apply_btn.setEnabled(False)
        self.status_label.setText("Applying correction...")

        self._worker = InterferenceWorker(
            X=self.state.get_active_X(),
            y=self.state.data.y,
            method=method,
            params=params,
            interference_spectra=self._interference_data
        )
        self._worker.progress.connect(lambda msg: self.status_label.setText(msg))
        self._worker.finished.connect(self._on_correction_complete)
        self._worker.start()

    def _on_correction_complete(self, result):
        """Handle correction completion."""
        self.apply_btn.setEnabled(True)

        if isinstance(result, Exception):
            self.status_label.setText(f"Error: {str(result)}")
            QMessageBox.critical(self, "Correction Failed", str(result))
            return

        X_corrected, info = result
        self._corrected_X = X_corrected

        self.status_label.setText("Correction applied!")
        self.status_label.setStyleSheet("color: #4caf50;")

        # Update preview plot
        self._update_preview_plot(X_corrected)

        self.save_btn.setEnabled(True)

    def _update_preview_plot(self, X_corrected):
        """Update the before/after preview plot."""
        self.preview_plot.clear()
        ax = self.preview_plot.ax

        wavelengths = self.state.data.wavelengths
        X_original = self.state.get_active_X()

        # Plot mean spectrum before and after
        mean_before = np.mean(X_original, axis=0)
        mean_after = np.mean(X_corrected, axis=0)

        ax.plot(wavelengths, mean_before, c='#f44336', alpha=0.7, label='Before')
        ax.plot(wavelengths, mean_after, c='#4caf50', alpha=0.7, label='After')

        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Intensity')
        ax.legend(loc='best', facecolor='#2d2d2d', edgecolor='#555', labelcolor='#aaa')

        self.preview_plot.figure.tight_layout()
        self.preview_plot.canvas.draw()

    def _save_to_state(self):
        """Save corrected data to state."""
        if self._corrected_X is None:
            return

        methods = ['msc', 'osc', 'dosc', 'epo']
        method = methods[self.method_combo.currentIndex()]

        self.state.apply_preprocessing(
            method.upper(),
            self._corrected_X
        )

        QMessageBox.information(
            self, "Applied",
            f"{method.upper()} correction applied to data.\n"
            f"The corrected data will be used for further analysis."
        )

        self.save_btn.setEnabled(False)
        self._corrected_X = None
