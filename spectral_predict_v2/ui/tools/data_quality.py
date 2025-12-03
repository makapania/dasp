"""
Data Quality Tool - Outlier detection and data validation.

Provides:
- PCA-based outlier detection (Hotelling T², Q-residuals)
- Mahalanobis distance analysis
- Y-value consistency checks
- Visual diagnostics
"""

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox,
    QMessageBox, QSplitter
)
from PySide6.QtCore import Qt, QThread, Signal

from orchestration.state_store import StateStore
from ui.widgets.plot_widget import PlotWidget


class OutlierDetectionWorker(QThread):
    """Background worker for outlier detection."""

    finished = Signal(object)  # dict with results or Exception
    progress = Signal(str)

    def __init__(self, X, y, n_components, alpha):
        super().__init__()
        self.X = X
        self.y = y
        self.n_components = n_components
        self.alpha = alpha

    def run(self):
        try:
            self.progress.emit("Running PCA outlier detection...")

            from src.spectral_predict import outlier_detection as od

            # Run PCA-based outlier detection
            results = od.run_pca_outlier_detection(
                self.X,
                self.y,
                n_components=self.n_components
            )

            self.progress.emit("Computing Q-residuals...")

            # Compute Q-residuals (SPE)
            q_residuals = od.compute_q_residuals(
                self.X,
                results['pca_model']
            )

            self.progress.emit("Computing Mahalanobis distances...")

            # Compute Mahalanobis distance
            mahal_dist = od.compute_mahalanobis_distance(results['scores'])

            # Combine all outlier flags
            t2_outliers = results['outlier_flags']
            q_outliers = q_residuals > np.percentile(q_residuals, 100 * (1 - self.alpha))
            combined_outliers = t2_outliers | q_outliers

            results.update({
                'q_residuals': q_residuals,
                'mahalanobis': mahal_dist,
                'q_outliers': q_outliers,
                'combined_outliers': combined_outliers,
                'n_combined_outliers': np.sum(combined_outliers),
            })

            self.finished.emit(results)

        except Exception as e:
            self.finished.emit(e)


class DataQualityTool(QWidget):
    """
    Tool panel for data quality assessment and outlier detection.

    Features:
    - PCA-based outlier detection
    - Hotelling T² and Q-residual plots
    - Sample flagging and removal
    - Quality score summary
    """

    def __init__(self, state: StateStore, parent=None):
        super().__init__(parent)
        self.state = state
        self._detection_worker = None
        self._results = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Header
        header = QLabel("Data Quality Assessment")
        header.setStyleSheet("font-size: 16px; font-weight: bold; color: #e0e0e0;")
        layout.addWidget(header)

        desc = QLabel(
            "Detect outliers using PCA-based methods (Hotelling T², Q-residuals) "
            "and flag suspicious samples for review or removal."
        )
        desc.setStyleSheet("color: #888; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Parameters section
        params_group = QGroupBox("Detection Parameters")
        params_layout = QHBoxLayout(params_group)

        params_layout.addWidget(QLabel("PCA Components:"))
        self.n_components_spin = QSpinBox()
        self.n_components_spin.setRange(2, 20)
        self.n_components_spin.setValue(5)
        params_layout.addWidget(self.n_components_spin)

        params_layout.addSpacing(20)

        params_layout.addWidget(QLabel("Confidence Level:"))
        self.alpha_combo = QComboBox()
        self.alpha_combo.addItems(["95%", "99%", "99.9%"])
        params_layout.addWidget(self.alpha_combo)

        params_layout.addStretch()

        self.run_btn = QPushButton("Run Detection")
        self.run_btn.clicked.connect(self._run_detection)
        params_layout.addWidget(self.run_btn)

        layout.addWidget(params_group)

        # Status
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888;")
        layout.addWidget(self.status_label)

        # Results splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side - plots
        plots_widget = QWidget()
        plots_layout = QVBoxLayout(plots_widget)
        plots_layout.setContentsMargins(0, 0, 0, 0)

        self.t2_plot = PlotWidget("Hotelling T²")
        self.t2_plot.setMinimumHeight(180)
        plots_layout.addWidget(self.t2_plot)

        self.q_plot = PlotWidget("Q-Residuals (SPE)")
        self.q_plot.setMinimumHeight(180)
        plots_layout.addWidget(self.q_plot)

        splitter.addWidget(plots_widget)

        # Right side - outlier table
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        table_layout.setContentsMargins(0, 0, 0, 0)

        table_header = QLabel("Detected Outliers")
        table_header.setStyleSheet("font-weight: bold; color: #e0e0e0;")
        table_layout.addWidget(table_header)

        self.outlier_table = QTableWidget()
        self.outlier_table.setColumnCount(5)
        self.outlier_table.setHorizontalHeaderLabels([
            "Sample", "T²", "Q", "Mahal", "Flag"
        ])
        self.outlier_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.outlier_table.setAlternatingRowColors(True)
        table_layout.addWidget(self.outlier_table)

        splitter.addWidget(table_widget)
        splitter.setSizes([400, 300])

        layout.addWidget(splitter, 1)

        # Summary and actions
        summary_group = QGroupBox("Summary")
        summary_layout = QVBoxLayout(summary_group)

        self.summary_label = QLabel("No detection run yet")
        self.summary_label.setStyleSheet("color: #888;")
        summary_layout.addWidget(self.summary_label)

        action_row = QHBoxLayout()

        self.flag_btn = QPushButton("Flag Selected as Outliers")
        self.flag_btn.setEnabled(False)
        self.flag_btn.clicked.connect(self._flag_outliers)
        action_row.addWidget(self.flag_btn)

        self.remove_btn = QPushButton("Remove Flagged from Data")
        self.remove_btn.setEnabled(False)
        self.remove_btn.clicked.connect(self._remove_outliers)
        action_row.addWidget(self.remove_btn)

        action_row.addStretch()

        self.clear_btn = QPushButton("Clear Flags")
        self.clear_btn.setEnabled(False)
        self.clear_btn.clicked.connect(self._clear_flags)
        action_row.addWidget(self.clear_btn)

        summary_layout.addLayout(action_row)

        layout.addWidget(summary_group)

    def _connect_signals(self):
        """Connect state signals."""
        self.state.data_changed.connect(self._on_data_changed)

    def _on_data_changed(self):
        """Handle data changes."""
        self._results = None
        self.outlier_table.setRowCount(0)
        self.t2_plot.clear()
        self.q_plot.clear()
        self.summary_label.setText("Data changed - run detection again")

        has_data = self.state.has_data
        self.run_btn.setEnabled(has_data)

    def _run_detection(self):
        """Run outlier detection."""
        if not self.state.has_data:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return

        if self._detection_worker is not None and self._detection_worker.isRunning():
            return

        # Get parameters
        n_components = self.n_components_spin.value()
        alpha_map = {"95%": 0.05, "99%": 0.01, "99.9%": 0.001}
        alpha = alpha_map[self.alpha_combo.currentText()]

        self.run_btn.setEnabled(False)
        self.status_label.setText("Running detection...")

        self._detection_worker = OutlierDetectionWorker(
            X=self.state.get_active_X(),
            y=self.state.data.y,
            n_components=n_components,
            alpha=alpha
        )
        self._detection_worker.progress.connect(lambda msg: self.status_label.setText(msg))
        self._detection_worker.finished.connect(self._on_detection_complete)
        self._detection_worker.start()

    def _on_detection_complete(self, result):
        """Handle detection completion."""
        self.run_btn.setEnabled(True)

        if isinstance(result, Exception):
            self.status_label.setText(f"Error: {str(result)}")
            QMessageBox.critical(self, "Detection Failed", str(result))
            return

        self._results = result
        self.status_label.setText("Detection complete!")

        # Update plots
        self._update_t2_plot(result)
        self._update_q_plot(result)

        # Update outlier table
        self._update_outlier_table(result)

        # Update summary
        n_t2 = result['n_outliers']
        n_q = np.sum(result['q_outliers'])
        n_combined = result['n_combined_outliers']
        n_total = len(result['hotelling_t2'])

        self.summary_label.setText(
            f"Total samples: {n_total} | "
            f"T² outliers: {n_t2} ({100*n_t2/n_total:.1f}%) | "
            f"Q outliers: {n_q} ({100*n_q/n_total:.1f}%) | "
            f"Combined: {n_combined} ({100*n_combined/n_total:.1f}%)"
        )

        # Enable action buttons
        self.flag_btn.setEnabled(n_combined > 0)
        self.remove_btn.setEnabled(len(self.state.data.outlier_indices) > 0)
        self.clear_btn.setEnabled(len(self.state.data.outlier_indices) > 0)

    def _update_t2_plot(self, results):
        """Update T² plot."""
        self.t2_plot.clear()
        ax = self.t2_plot.ax

        t2 = results['hotelling_t2']
        threshold = results['t2_threshold']
        outliers = results['outlier_flags']

        # Plot all samples
        x = np.arange(len(t2))
        ax.scatter(x[~outliers], t2[~outliers], c='#4fc3f7', alpha=0.6, s=20, label='Normal')
        ax.scatter(x[outliers], t2[outliers], c='#f44336', alpha=0.8, s=30, label='Outlier')

        # Threshold line
        ax.axhline(y=threshold, color='#ffa500', linestyle='--', alpha=0.7, label='95% threshold')

        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Hotelling T²')
        ax.legend(loc='upper right', facecolor='#2d2d2d', edgecolor='#555', labelcolor='#aaa')

        self.t2_plot.figure.tight_layout()
        self.t2_plot.canvas.draw()

    def _update_q_plot(self, results):
        """Update Q-residuals plot."""
        self.q_plot.clear()
        ax = self.q_plot.ax

        q = results['q_residuals']
        outliers = results['q_outliers']
        threshold = np.percentile(q, 95)

        # Plot all samples
        x = np.arange(len(q))
        ax.scatter(x[~outliers], q[~outliers], c='#81c784', alpha=0.6, s=20, label='Normal')
        ax.scatter(x[outliers], q[outliers], c='#f44336', alpha=0.8, s=30, label='Outlier')

        # Threshold line
        ax.axhline(y=threshold, color='#ffa500', linestyle='--', alpha=0.7, label='95% threshold')

        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Q-Residuals (SPE)')
        ax.legend(loc='upper right', facecolor='#2d2d2d', edgecolor='#555', labelcolor='#aaa')

        self.q_plot.figure.tight_layout()
        self.q_plot.canvas.draw()

    def _update_outlier_table(self, results):
        """Update outlier table with detected outliers."""
        combined = results['combined_outliers']
        outlier_indices = np.where(combined)[0]

        self.outlier_table.setRowCount(len(outlier_indices))

        for i, idx in enumerate(outlier_indices):
            # Sample ID
            sample_id = self.state.data.sample_ids[idx] if self.state.data.sample_ids else str(idx)
            self.outlier_table.setItem(i, 0, QTableWidgetItem(str(sample_id)))

            # T² value
            t2_val = results['hotelling_t2'][idx]
            t2_item = QTableWidgetItem(f"{t2_val:.2f}")
            if results['outlier_flags'][idx]:
                t2_item.setForeground(Qt.GlobalColor.red)
            self.outlier_table.setItem(i, 1, t2_item)

            # Q value
            q_val = results['q_residuals'][idx]
            q_item = QTableWidgetItem(f"{q_val:.2f}")
            if results['q_outliers'][idx]:
                q_item.setForeground(Qt.GlobalColor.red)
            self.outlier_table.setItem(i, 2, q_item)

            # Mahalanobis
            mahal_val = results['mahalanobis'][idx]
            self.outlier_table.setItem(i, 3, QTableWidgetItem(f"{mahal_val:.2f}"))

            # Flag checkbox
            flag_item = QTableWidgetItem()
            flag_item.setCheckState(Qt.CheckState.Checked)
            flag_item.setData(Qt.ItemDataRole.UserRole, idx)  # Store original index
            self.outlier_table.setItem(i, 4, flag_item)

    def _flag_outliers(self):
        """Flag selected outliers in state."""
        flagged_indices = []

        for row in range(self.outlier_table.rowCount()):
            flag_item = self.outlier_table.item(row, 4)
            if flag_item and flag_item.checkState() == Qt.CheckState.Checked:
                idx = flag_item.data(Qt.ItemDataRole.UserRole)
                flagged_indices.append(idx)

        self.state.set_outliers(flagged_indices)

        QMessageBox.information(
            self, "Outliers Flagged",
            f"Flagged {len(flagged_indices)} samples as outliers."
        )

        self.remove_btn.setEnabled(len(flagged_indices) > 0)
        self.clear_btn.setEnabled(len(flagged_indices) > 0)

    def _remove_outliers(self):
        """Remove flagged outliers from data."""
        outlier_indices = self.state.data.outlier_indices

        if not outlier_indices:
            QMessageBox.warning(self, "No Outliers", "No outliers flagged for removal.")
            return

        reply = QMessageBox.question(
            self, "Confirm Removal",
            f"Remove {len(outlier_indices)} flagged samples from the dataset?\n\n"
            "This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Create mask for samples to keep
        keep_mask = np.ones(self.state.data.n_samples, dtype=bool)
        keep_mask[outlier_indices] = False

        # Filter data
        X_clean = self.state.data.X[keep_mask]
        y_clean = self.state.data.y[keep_mask]
        sample_ids_clean = [sid for i, sid in enumerate(self.state.data.sample_ids) if keep_mask[i]]

        # Update state with cleaned data
        self.state.load_data(
            X=X_clean,
            y=y_clean,
            wavelengths=self.state.data.wavelengths,
            file_path=self.state.data.file_path,
            target_column=self.state.data.target_column,
            sample_ids=sample_ids_clean
        )

        QMessageBox.information(
            self, "Outliers Removed",
            f"Removed {len(outlier_indices)} samples.\n"
            f"Dataset now has {len(y_clean)} samples."
        )

    def _clear_flags(self):
        """Clear all outlier flags."""
        self.state.set_outliers([])
        self.remove_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)

        # Uncheck all in table
        for row in range(self.outlier_table.rowCount()):
            flag_item = self.outlier_table.item(row, 4)
            if flag_item:
                flag_item.setCheckState(Qt.CheckState.Unchecked)
