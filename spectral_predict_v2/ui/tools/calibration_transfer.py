"""
Calibration Transfer Tool - Transfer models between instruments.

Supports methods:
- DS (Direct Standardization)
- PDS (Piecewise Direct Standardization)
- TSR (Transfer Sample Regression / Shenk-Westerhaus)
"""

from pathlib import Path
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QFrame,
    QFileDialog, QMessageBox, QProgressBar, QTableWidget,
    QTableWidgetItem, QHeaderView
)
from PySide6.QtCore import Qt, QThread, Signal

from orchestration.state_store import StateStore
from ui.widgets.file_drop import FileDropWidget


class TransferWorker(QThread):
    """Background worker for calibration transfer."""

    finished = Signal(object)  # TransferModel or Exception
    progress = Signal(str)

    def __init__(self, engine, X_master, X_slave, method, params):
        super().__init__()
        self.engine = engine
        self.X_master = X_master
        self.X_slave = X_slave
        self.method = method
        self.params = params

    def run(self):
        try:
            self.progress.emit(f"Computing {self.method.upper()} transfer matrix...")

            # Import calibration transfer module
            from src.spectral_predict import calibration_transfer as ct

            if self.method == 'ds':
                A = ct.estimate_ds(
                    self.X_master,
                    self.X_slave,
                    lam=self.params.get('lambda', 0.0)
                )
                transfer_model = ct.TransferModel(
                    master_id="master",
                    slave_id="slave",
                    method="ds",
                    wavelengths_common=None,
                    params={"A": A},
                    meta={"lambda": self.params.get('lambda', 0.0)}
                )

            elif self.method == 'pds':
                B = ct.estimate_pds(
                    self.X_master,
                    self.X_slave,
                    window=self.params.get('window', 5),
                    lam=self.params.get('lambda', 0.0)
                )
                transfer_model = ct.TransferModel(
                    master_id="master",
                    slave_id="slave",
                    method="pds",
                    wavelengths_common=None,
                    params={"B": B, "window": self.params.get('window', 5)},
                    meta={"lambda": self.params.get('lambda', 0.0)}
                )

            else:
                raise ValueError(f"Unknown transfer method: {self.method}")

            self.finished.emit(transfer_model)

        except Exception as e:
            self.finished.emit(e)


class CalibrationTransferTool(QWidget):
    """
    Tool panel for calibration transfer between instruments.

    Workflow:
    1. Load master instrument spectra (transfer standards)
    2. Load slave instrument spectra (same samples)
    3. Select transfer method and parameters
    4. Compute transfer model
    5. Apply to new slave data or save for later use
    """

    def __init__(self, state: StateStore, parent=None):
        super().__init__(parent)
        self.state = state
        self._engine = None
        self._master_data = None  # (X, wavelengths, sample_ids)
        self._slave_data = None
        self._transfer_model = None
        self._transfer_worker = None

        self._setup_ui()

    def _get_engine(self):
        if self._engine is None:
            from engine.api import EngineAPI
            self._engine = EngineAPI()
        return self._engine

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Header
        header = QLabel("Calibration Transfer")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #e0e0e0;")
        layout.addWidget(header)

        desc = QLabel(
            "Transfer calibration models between instruments using standard samples "
            "measured on both instruments."
        )
        desc.setStyleSheet("color: #888; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Master data section
        master_group = QGroupBox("Master Instrument (Reference)")
        master_layout = QVBoxLayout(master_group)

        self.master_drop = FileDropWidget()
        self.master_drop.file_selected.connect(self._on_master_selected)
        master_layout.addWidget(self.master_drop)

        self.master_info = QLabel("")
        self.master_info.setStyleSheet("color: #888; font-size: 11px;")
        master_layout.addWidget(self.master_info)

        layout.addWidget(master_group)

        # Slave data section
        slave_group = QGroupBox("Slave Instrument (To Transfer)")
        slave_layout = QVBoxLayout(slave_group)

        self.slave_drop = FileDropWidget()
        self.slave_drop.file_selected.connect(self._on_slave_selected)
        slave_layout.addWidget(self.slave_drop)

        self.slave_info = QLabel("")
        self.slave_info.setStyleSheet("color: #888; font-size: 11px;")
        slave_layout.addWidget(self.slave_info)

        layout.addWidget(slave_group)

        # Method selection
        method_group = QGroupBox("Transfer Method")
        method_layout = QVBoxLayout(method_group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Method:"))

        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "DS (Direct Standardization)",
            "PDS (Piecewise Direct Standardization)",
        ])
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        row1.addWidget(self.method_combo)

        row1.addStretch()
        method_layout.addLayout(row1)

        # Parameters row
        row2 = QHBoxLayout()

        row2.addWidget(QLabel("Lambda (regularization):"))
        self.lambda_spin = QDoubleSpinBox()
        self.lambda_spin.setRange(0.0, 1000.0)
        self.lambda_spin.setValue(0.0)
        self.lambda_spin.setDecimals(4)
        row2.addWidget(self.lambda_spin)

        row2.addSpacing(20)

        self.window_label = QLabel("Window (PDS):")
        row2.addWidget(self.window_label)
        self.window_spin = QSpinBox()
        self.window_spin.setRange(3, 51)
        self.window_spin.setValue(5)
        self.window_spin.setSingleStep(2)
        row2.addWidget(self.window_spin)

        row2.addStretch()
        method_layout.addLayout(row2)

        layout.addWidget(method_group)

        # Compute button
        btn_row = QHBoxLayout()

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888;")
        btn_row.addWidget(self.status_label)

        btn_row.addStretch()

        self.compute_btn = QPushButton("Compute Transfer")
        self.compute_btn.setEnabled(False)
        self.compute_btn.clicked.connect(self._compute_transfer)
        btn_row.addWidget(self.compute_btn)

        layout.addLayout(btn_row)

        # Results section
        results_group = QGroupBox("Transfer Model")
        results_layout = QVBoxLayout(results_group)

        self.results_label = QLabel("No transfer model computed")
        self.results_label.setStyleSheet("color: #888;")
        results_layout.addWidget(self.results_label)

        # Action buttons
        action_row = QHBoxLayout()

        self.save_btn = QPushButton("Save Transfer Model")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_transfer)
        action_row.addWidget(self.save_btn)

        self.apply_btn = QPushButton("Apply to Current Data")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply_transfer)
        action_row.addWidget(self.apply_btn)

        action_row.addStretch()
        results_layout.addLayout(action_row)

        layout.addWidget(results_group)

        layout.addStretch()

        # Initial state
        self._on_method_changed(0)

    def _on_method_changed(self, index):
        """Update UI based on selected method."""
        is_pds = index == 1
        self.window_label.setVisible(is_pds)
        self.window_spin.setVisible(is_pds)

    def _on_master_selected(self, file_path: str):
        """Handle master file selection."""
        try:
            engine = self._get_engine()
            loaded = engine.load_data(file_path, for_prediction=True)
            self._master_data = (loaded.X, loaded.wavelengths, loaded.sample_ids)

            self.master_info.setText(
                f"{loaded.metadata['n_samples']} samples | "
                f"{loaded.metadata['n_wavelengths']} wavelengths"
            )
            self._check_ready()

        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            self._master_data = None

    def _on_slave_selected(self, file_path: str):
        """Handle slave file selection."""
        try:
            engine = self._get_engine()
            loaded = engine.load_data(file_path, for_prediction=True)
            self._slave_data = (loaded.X, loaded.wavelengths, loaded.sample_ids)

            self.slave_info.setText(
                f"{loaded.metadata['n_samples']} samples | "
                f"{loaded.metadata['n_wavelengths']} wavelengths"
            )
            self._check_ready()

        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            self._slave_data = None

    def _check_ready(self):
        """Check if ready to compute transfer."""
        if self._master_data is None or self._slave_data is None:
            self.compute_btn.setEnabled(False)
            return

        # Check sample counts match
        if len(self._master_data[2]) != len(self._slave_data[2]):
            self.status_label.setText("Sample counts must match!")
            self.status_label.setStyleSheet("color: #f44336;")
            self.compute_btn.setEnabled(False)
            return

        # Check wavelength counts match
        if self._master_data[0].shape[1] != self._slave_data[0].shape[1]:
            self.status_label.setText("Wavelength counts must match!")
            self.status_label.setStyleSheet("color: #f44336;")
            self.compute_btn.setEnabled(False)
            return

        self.status_label.setText("")
        self.compute_btn.setEnabled(True)

    def _compute_transfer(self):
        """Compute calibration transfer model."""
        if self._master_data is None or self._slave_data is None:
            return

        method_map = {0: 'ds', 1: 'pds'}
        method = method_map[self.method_combo.currentIndex()]

        params = {
            'lambda': self.lambda_spin.value(),
            'window': self.window_spin.value(),
        }

        self.compute_btn.setEnabled(False)
        self.status_label.setText("Computing...")

        self._transfer_worker = TransferWorker(
            engine=self._get_engine(),
            X_master=self._master_data[0],
            X_slave=self._slave_data[0],
            method=method,
            params=params
        )
        self._transfer_worker.progress.connect(lambda msg: self.status_label.setText(msg))
        self._transfer_worker.finished.connect(self._on_transfer_complete)
        self._transfer_worker.start()

    def _on_transfer_complete(self, result):
        """Handle transfer computation completion."""
        self.compute_btn.setEnabled(True)

        if isinstance(result, Exception):
            self.status_label.setText(f"Error: {str(result)}")
            self.status_label.setStyleSheet("color: #f44336;")
            QMessageBox.critical(self, "Transfer Failed", str(result))
            return

        self._transfer_model = result
        self.status_label.setText("Transfer model computed!")
        self.status_label.setStyleSheet("color: #4caf50;")

        # Update results display
        method_name = result.method.upper()
        self.results_label.setText(
            f"Method: {method_name} | "
            f"Matrix shape: {result.params.get('A', result.params.get('B', [[]])).shape}"
        )
        self.results_label.setStyleSheet("color: #4fc3f7;")

        self.save_btn.setEnabled(True)
        self.apply_btn.setEnabled(self.state.has_data)

    def _save_transfer(self):
        """Save transfer model to file."""
        if self._transfer_model is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Transfer Model",
            "transfer_model",
            "NumPy Files (*.npz);;All Files (*)"
        )

        if file_path:
            try:
                from src.spectral_predict import calibration_transfer as ct
                ct.save_transfer_model(self._transfer_model, file_path)
                QMessageBox.information(
                    self, "Saved",
                    f"Transfer model saved to:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Save Failed", str(e))

    def _apply_transfer(self):
        """Apply transfer model to current data in state."""
        if self._transfer_model is None or not self.state.has_data:
            return

        try:
            from src.spectral_predict import calibration_transfer as ct

            X_transferred = ct.apply_transfer_dispatch(
                self.state.data.X,
                self._transfer_model
            )

            # Update state with transferred data
            self.state.apply_preprocessing(
                f"Transfer-{self._transfer_model.method.upper()}",
                X_transferred
            )

            QMessageBox.information(
                self, "Applied",
                f"Transfer applied to {self.state.data.n_samples} samples.\n"
                f"Data is now standardized to master instrument."
            )

        except Exception as e:
            QMessageBox.critical(self, "Apply Failed", str(e))
