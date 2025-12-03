"""
Predict Mode - Apply saved models to new data.

For loading trained models and making predictions on new samples.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QFrame, QScrollArea, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QMessageBox, QProgressBar
)
from PySide6.QtCore import Qt, QThread, Signal

from orchestration.state_store import StateStore
from orchestration.job_runner import JobRunner
from ui.widgets.file_drop import FileDropWidget


class PredictionWorker(QThread):
    """Background worker for predictions."""

    finished = Signal(object)  # (predictions, uncertainty, ad_flags, sample_ids) or Exception
    progress = Signal(str)

    def __init__(self, engine, model, X, sample_ids):
        super().__init__()
        self.engine = engine
        self.model = model
        self.X = X
        self.sample_ids = sample_ids

    def run(self):
        try:
            self.progress.emit("Applying preprocessing...")

            # Make predictions
            predictions, uncertainty, ad_flags = self.engine.predict(
                self.X,
                self.model,
                return_uncertainty=True
            )

            self.progress.emit("Checking applicability domain...")

            # Enhanced AD check using training stats
            training_stats = self.model.config.get('training_stats', {})
            if training_stats:
                y_min = training_stats.get('y_min', -np.inf)
                y_max = training_stats.get('y_max', np.inf)
                y_range = y_max - y_min if y_max > y_min else 1

                # Flag predictions outside training range (with 10% margin)
                margin = y_range * 0.1
                ad_flags = (predictions >= y_min - margin) & (predictions <= y_max + margin)

            self.finished.emit((predictions, uncertainty, ad_flags, self.sample_ids))

        except Exception as e:
            self.finished.emit(e)


class PredictMode(QWidget):
    """
    Predict mode view for applying models to new data.

    Layout:
    - MODEL: Load saved model, show model info
    - NEW DATA: Load new spectral data for prediction
    - PREDICTIONS: Table with predictions, uncertainty, applicability domain flags
    """

    def __init__(self, state: StateStore, job_runner: JobRunner, parent=None):
        super().__init__(parent)
        self.state = state
        self.job_runner = job_runner
        self._engine = None
        self._loaded_model = None
        self._prediction_data = None  # (X, sample_ids, wavelengths)
        self._prediction_worker = None
        self._predictions_df = None

        self._setup_ui()
        self._connect_signals()

    def _get_engine(self):
        """Lazy load the engine API."""
        if self._engine is None:
            from engine.api import EngineAPI
            self._engine = EngineAPI()
        return self._engine

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 10, 20, 20)
        layout.setSpacing(16)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(16)

        # MODEL Section
        model_group = self._create_model_section()
        content_layout.addWidget(model_group)

        # NEW DATA Section
        data_group = self._create_data_section()
        content_layout.addWidget(data_group)

        # PREDICTIONS Section
        pred_group = self._create_predictions_section()
        content_layout.addWidget(pred_group, 1)

        scroll.setWidget(content)
        layout.addWidget(scroll)

    def _create_model_section(self) -> QGroupBox:
        """Create the MODEL section."""
        group = QGroupBox("MODEL")
        layout = QVBoxLayout(group)

        # Top row
        row1 = QHBoxLayout()

        self.load_model_btn = QPushButton("Load Model (.dasp)")
        self.load_model_btn.clicked.connect(self._load_model)
        row1.addWidget(self.load_model_btn)

        self.model_path_label = QLabel("No model loaded")
        self.model_path_label.setStyleSheet("color: #888;")
        row1.addWidget(self.model_path_label)

        row1.addStretch()
        layout.addLayout(row1)

        # Model info row
        self.model_info_label = QLabel("")
        self.model_info_label.setStyleSheet("color: #4fc3f7; font-size: 12px;")
        layout.addWidget(self.model_info_label)

        return group

    def _create_data_section(self) -> QGroupBox:
        """Create the NEW DATA section."""
        group = QGroupBox("NEW DATA")
        layout = QVBoxLayout(group)

        # File drop widget
        self.file_drop = FileDropWidget()
        self.file_drop.file_selected.connect(self._on_file_selected)
        layout.addWidget(self.file_drop)

        # Data info
        self.data_info_label = QLabel("")
        self.data_info_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.data_info_label)

        # Warnings
        self.warning_label = QLabel("")
        self.warning_label.setStyleSheet("color: #ffa500;")
        self.warning_label.setVisible(False)
        layout.addWidget(self.warning_label)

        # Status and predict button
        btn_row = QHBoxLayout()

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888;")
        btn_row.addWidget(self.status_label)

        btn_row.addStretch()

        self.predict_btn = QPushButton("▶ PREDICT")
        self.predict_btn.setMinimumHeight(40)
        self.predict_btn.setMinimumWidth(150)
        self.predict_btn.setEnabled(False)
        self.predict_btn.clicked.connect(self._run_prediction)
        btn_row.addWidget(self.predict_btn)

        layout.addLayout(btn_row)

        return group

    def _create_predictions_section(self) -> QGroupBox:
        """Create the PREDICTIONS section."""
        group = QGroupBox("PREDICTIONS")
        layout = QVBoxLayout(group)

        # Summary row
        self.summary_label = QLabel("")
        self.summary_label.setStyleSheet("font-size: 12px; color: #888;")
        layout.addWidget(self.summary_label)

        # Results table
        self.pred_table = QTableWidget()
        self.pred_table.setColumnCount(4)
        self.pred_table.setHorizontalHeaderLabels([
            "Sample ID", "Prediction", "Uncertainty (±)", "AD Flag"
        ])
        self.pred_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.pred_table.setAlternatingRowColors(True)
        self.pred_table.setStyleSheet("""
            QTableWidget {
                background-color: #2d2d2d;
                alternate-background-color: #353535;
                gridline-color: #444;
            }
            QHeaderView::section {
                background-color: #3d3d3d;
                color: #e0e0e0;
                padding: 4px;
                border: 1px solid #444;
            }
        """)
        layout.addWidget(self.pred_table)

        # Action buttons
        btn_row = QHBoxLayout()

        self.export_csv_btn = QPushButton("Export CSV")
        self.export_csv_btn.setEnabled(False)
        self.export_csv_btn.clicked.connect(self._export_csv)
        btn_row.addWidget(self.export_csv_btn)

        self.export_report_btn = QPushButton("Export Report")
        self.export_report_btn.setEnabled(False)
        self.export_report_btn.clicked.connect(self._export_report)
        btn_row.addWidget(self.export_report_btn)

        self.batch_btn = QPushButton("Batch Process Folder")
        self.batch_btn.setEnabled(False)
        self.batch_btn.clicked.connect(self._batch_process)
        btn_row.addWidget(self.batch_btn)

        btn_row.addStretch()

        layout.addLayout(btn_row)

        return group

    def _connect_signals(self):
        """Connect signals."""
        self.state.model_changed.connect(self._on_model_changed)

    def _on_model_changed(self):
        """Handle model change from Build mode."""
        if self.state.model.selected_model:
            # Use model from state (passed from Build mode)
            from engine.api import TrainedModel
            self._loaded_model = TrainedModel(
                model=self.state.model.selected_model,
                name=self.state.model.model_name or "Unknown",
                preprocessing=self.state.model.preprocessing or "raw",
                variable_mask=self.state.model.variable_mask,
                config=self.state.model.model_config,
                metrics=self.state.model.metrics
            )
            self._update_model_display()

    def _update_model_display(self):
        """Update the model info display."""
        if self._loaded_model:
            metrics = self._loaded_model.metrics
            rmsecv = metrics.get('rmsecv', metrics.get('rmse', '?'))
            if isinstance(rmsecv, float):
                rmsecv = f"{rmsecv:.4f}"

            self.model_path_label.setText(f"Loaded: {self._loaded_model.name}")
            self.model_path_label.setStyleSheet("color: #4caf50;")

            info_parts = [
                f"Preprocessing: {self._loaded_model.preprocessing}",
                f"RMSECV: {rmsecv}",
            ]

            training_stats = self._loaded_model.config.get('training_stats', {})
            if training_stats:
                n_samples = training_stats.get('n_samples', '?')
                y_range = f"{training_stats.get('y_min', '?'):.2f} - {training_stats.get('y_max', '?'):.2f}"
                info_parts.append(f"Trained on: {n_samples} samples")
                info_parts.append(f"Target range: {y_range}")

            created = self._loaded_model.config.get('created', '')
            if created:
                info_parts.append(f"Created: {created[:10]}")

            self.model_info_label.setText(" | ".join(info_parts))
            self.batch_btn.setEnabled(True)
            self._check_ready()

    def _load_model(self):
        """Load a saved model from .dasp file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Model",
            "",
            "DASP Model Files (*.dasp);;All Files (*)"
        )

        if file_path:
            try:
                engine = self._get_engine()
                self._loaded_model = engine.load_model(file_path)
                self._update_model_display()

            except Exception as e:
                QMessageBox.critical(
                    self, "Load Failed",
                    f"Failed to load model:\n{str(e)}"
                )

    def _on_file_selected(self, file_path: str):
        """Handle file selection for prediction."""
        try:
            engine = self._get_engine()
            loaded = engine.load_data(file_path, for_prediction=True)

            self._prediction_data = (loaded.X, loaded.sample_ids, loaded.wavelengths)

            self.data_info_label.setText(
                f"{loaded.metadata['n_samples']} samples | "
                f"{loaded.metadata['n_wavelengths']} wavelengths | "
                f"Range: {loaded.metadata['wavelength_range'][0]:.0f}-{loaded.metadata['wavelength_range'][1]:.0f} nm"
            )

            # Check wavelength compatibility
            self._check_wavelength_compatibility()
            self._check_ready()

        except Exception as e:
            self.warning_label.setText(f"Error loading file: {str(e)}")
            self.warning_label.setVisible(True)
            self._prediction_data = None

    def _check_wavelength_compatibility(self):
        """Check if prediction data wavelengths match model."""
        if self._loaded_model is None or self._prediction_data is None:
            self.warning_label.setVisible(False)
            return

        model_wavelengths = self._loaded_model.config.get('wavelengths')
        if model_wavelengths is None:
            # Can't check, proceed anyway
            self.warning_label.setVisible(False)
            return

        pred_wavelengths = self._prediction_data[2]

        # Check if wavelength ranges match reasonably
        if len(model_wavelengths) != len(pred_wavelengths):
            self.warning_label.setText(
                f"⚠️ Wavelength count mismatch: Model has {len(model_wavelengths)}, "
                f"data has {len(pred_wavelengths)}. Predictions may be unreliable."
            )
            self.warning_label.setVisible(True)
        elif not np.allclose(model_wavelengths, pred_wavelengths, rtol=0.01):
            self.warning_label.setText(
                "⚠️ Wavelength values differ from training data. Predictions may be unreliable."
            )
            self.warning_label.setVisible(True)
        else:
            self.warning_label.setVisible(False)

    def _check_ready(self):
        """Check if ready to predict."""
        has_model = self._loaded_model is not None
        has_data = self._prediction_data is not None
        self.predict_btn.setEnabled(has_model and has_data)

    def _run_prediction(self):
        """Run prediction on loaded data."""
        if self._loaded_model is None:
            QMessageBox.warning(self, "No Model", "Please load a model first.")
            return

        if self._prediction_data is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return

        if self._prediction_worker is not None and self._prediction_worker.isRunning():
            QMessageBox.warning(self, "In Progress", "Prediction is already running.")
            return

        X, sample_ids, wavelengths = self._prediction_data

        self.predict_btn.setEnabled(False)
        self.status_label.setText("Running predictions...")

        self._prediction_worker = PredictionWorker(
            engine=self._get_engine(),
            model=self._loaded_model,
            X=X,
            sample_ids=sample_ids
        )
        self._prediction_worker.progress.connect(self._on_prediction_progress)
        self._prediction_worker.finished.connect(self._on_prediction_complete)
        self._prediction_worker.start()

    def _on_prediction_progress(self, message: str):
        """Handle prediction progress."""
        self.status_label.setText(message)

    def _on_prediction_complete(self, result):
        """Handle prediction completion."""
        self.predict_btn.setEnabled(True)

        if isinstance(result, Exception):
            self.status_label.setText(f"Error: {str(result)}")
            QMessageBox.critical(self, "Prediction Failed", str(result))
            return

        predictions, uncertainty, ad_flags, sample_ids = result
        self.status_label.setText("Predictions complete!")

        # Store as DataFrame for export
        self._predictions_df = pd.DataFrame({
            'sample_id': sample_ids,
            'prediction': predictions,
            'uncertainty': uncertainty if uncertainty is not None else np.nan,
            'in_ad': ad_flags if ad_flags is not None else True,
        })

        # Update table
        self._update_predictions_table(predictions, uncertainty, ad_flags, sample_ids)

        # Update summary
        n_outside_ad = np.sum(~ad_flags) if ad_flags is not None else 0
        self.summary_label.setText(
            f"{len(predictions)} predictions | "
            f"Mean: {np.mean(predictions):.4f} | "
            f"Std: {np.std(predictions):.4f} | "
            f"Range: {np.min(predictions):.4f} - {np.max(predictions):.4f}"
            + (f" | ⚠️ {n_outside_ad} outside AD" if n_outside_ad > 0 else "")
        )

        self.export_csv_btn.setEnabled(True)
        self.export_report_btn.setEnabled(True)

    def _update_predictions_table(self, predictions, uncertainty, ad_flags, sample_ids):
        """Update the predictions table."""
        self.pred_table.setRowCount(len(predictions))

        for i, (sid, pred, unc, ad) in enumerate(zip(
            sample_ids, predictions,
            uncertainty if uncertainty is not None else [None] * len(predictions),
            ad_flags if ad_flags is not None else [True] * len(predictions)
        )):
            # Sample ID
            self.pred_table.setItem(i, 0, QTableWidgetItem(str(sid)))

            # Prediction
            pred_item = QTableWidgetItem(f"{pred:.4f}")
            pred_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.pred_table.setItem(i, 1, pred_item)

            # Uncertainty
            unc_text = f"{unc:.4f}" if unc is not None else "N/A"
            unc_item = QTableWidgetItem(unc_text)
            unc_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.pred_table.setItem(i, 2, unc_item)

            # AD flag
            ad_text = "✓" if ad else "⚠️ Outside"
            ad_item = QTableWidgetItem(ad_text)
            if not ad:
                ad_item.setForeground(Qt.GlobalColor.yellow)
            else:
                ad_item.setForeground(Qt.GlobalColor.green)
            ad_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.pred_table.setItem(i, 3, ad_item)

    def _export_csv(self):
        """Export predictions to CSV."""
        if self._predictions_df is None:
            QMessageBox.warning(self, "No Predictions", "Run predictions first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Predictions", "predictions.csv", "CSV Files (*.csv)"
        )

        if file_path:
            try:
                self._predictions_df.to_csv(file_path, index=False)
                QMessageBox.information(
                    self, "Export Complete",
                    f"Predictions exported to:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", str(e))

    def _export_report(self):
        """Export prediction report."""
        if self._predictions_df is None:
            QMessageBox.warning(self, "No Predictions", "Run predictions first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Report", "prediction_report.md", "Markdown Files (*.md)"
        )

        if file_path:
            try:
                df = self._predictions_df
                n_outside = df[~df['in_ad']].shape[0] if 'in_ad' in df.columns else 0

                report = f"""# Prediction Report

## Model
- **Name:** {self._loaded_model.name}
- **Preprocessing:** {self._loaded_model.preprocessing}
- **RMSECV:** {self._loaded_model.metrics.get('rmsecv', 'N/A')}

## Predictions Summary
- **Total samples:** {len(df)}
- **Mean prediction:** {df['prediction'].mean():.4f}
- **Std deviation:** {df['prediction'].std():.4f}
- **Min:** {df['prediction'].min():.4f}
- **Max:** {df['prediction'].max():.4f}
- **Samples outside AD:** {n_outside}

## Individual Predictions

| Sample ID | Prediction | Uncertainty | In AD |
|-----------|------------|-------------|-------|
"""
                for _, row in df.iterrows():
                    ad_flag = "Yes" if row.get('in_ad', True) else "**No**"
                    unc = f"{row['uncertainty']:.4f}" if pd.notna(row['uncertainty']) else "N/A"
                    report += f"| {row['sample_id']} | {row['prediction']:.4f} | {unc} | {ad_flag} |\n"

                report += "\n---\n*Generated by Spectral Predict 2.0*\n"

                Path(file_path).write_text(report)
                QMessageBox.information(
                    self, "Report Exported",
                    f"Report exported to:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", str(e))

    def _batch_process(self):
        """Process a folder of files."""
        if self._loaded_model is None:
            QMessageBox.warning(self, "No Model", "Please load a model first.")
            return

        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Process")

        if folder:
            try:
                folder_path = Path(folder)
                supported_ext = ['.csv', '.xlsx', '.asd', '.spc', '.jdx', '.dx']
                files = [f for f in folder_path.iterdir()
                        if f.is_file() and f.suffix.lower() in supported_ext]

                if not files:
                    QMessageBox.warning(
                        self, "No Files",
                        f"No supported files found in folder.\nSupported: {', '.join(supported_ext)}"
                    )
                    return

                # Confirm
                reply = QMessageBox.question(
                    self, "Batch Process",
                    f"Found {len(files)} files to process.\nContinue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )

                if reply != QMessageBox.StandardButton.Yes:
                    return

                # Process all files
                all_results = []
                engine = self._get_engine()

                for file in files:
                    try:
                        loaded = engine.load_data(str(file), for_prediction=True)
                        predictions, uncertainty, ad_flags = engine.predict(
                            loaded.X, self._loaded_model, return_uncertainty=True
                        )

                        for sid, pred, unc, ad in zip(
                            loaded.sample_ids, predictions,
                            uncertainty if uncertainty is not None else [None] * len(predictions),
                            ad_flags if ad_flags is not None else [True] * len(predictions)
                        ):
                            all_results.append({
                                'file': file.name,
                                'sample_id': sid,
                                'prediction': pred,
                                'uncertainty': unc,
                                'in_ad': ad,
                            })
                    except Exception as e:
                        all_results.append({
                            'file': file.name,
                            'sample_id': 'ERROR',
                            'prediction': None,
                            'uncertainty': None,
                            'in_ad': None,
                            'error': str(e),
                        })

                # Save results
                output_path = folder_path / "batch_predictions.csv"
                pd.DataFrame(all_results).to_csv(output_path, index=False)

                QMessageBox.information(
                    self, "Batch Complete",
                    f"Processed {len(files)} files.\n"
                    f"Results saved to:\n{output_path}"
                )

            except Exception as e:
                QMessageBox.critical(self, "Batch Failed", str(e))
