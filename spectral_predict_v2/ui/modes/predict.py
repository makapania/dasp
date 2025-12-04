"""
Predict Mode - Apply saved models to new data.

Modern card-based layout for loading models and making predictions.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QScrollArea, QSplitter, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal

from orchestration.state_store import StateStore
from orchestration.job_runner import JobRunner

from ..theme.tokens import COLORS, SPACING, RADIUS, TYPOGRAPHY
from ..theme.icons import Icons
from ..components.cards import Card, StatusCard
from ..components.buttons import PrimaryButton, SecondaryButton, OutlineButton
from ..widgets.file_drop import FileDropWidget, CompactFileDropWidget
from ..widgets.data_grid import SpectralDataGrid


class PredictionWorker(QThread):
    """Background worker for predictions."""

    finished = Signal(object)
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

            predictions, uncertainty, ad_flags = self.engine.predict(
                self.X,
                self.model,
                return_uncertainty=True
            )

            self.progress.emit("Checking applicability domain...")

            training_stats = self.model.config.get('training_stats', {})
            if training_stats:
                y_min = training_stats.get('y_min', -np.inf)
                y_max = training_stats.get('y_max', np.inf)
                y_range = y_max - y_min if y_max > y_min else 1
                margin = y_range * 0.1
                ad_flags = (predictions >= y_min - margin) & (predictions <= y_max + margin)

            self.finished.emit((predictions, uncertainty, ad_flags, self.sample_ids))

        except Exception as e:
            self.finished.emit(e)


class PredictMode(QWidget):
    """
    Predict mode view for applying models to new data.

    Modern card-based layout:
    - Left panel: Model loading, data loading, predict controls
    - Right panel: Predictions table with summary stats
    """

    def __init__(self, state: StateStore, job_runner: JobRunner, parent=None):
        super().__init__(parent)
        self.state = state
        self.job_runner = job_runner
        self._engine = None
        self._loaded_model = None
        self._prediction_data = None
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
        layout.setContentsMargins(SPACING["lg"], SPACING["md"], SPACING["lg"], SPACING["lg"])
        layout.setSpacing(SPACING["md"])

        # Main splitter: config left, results right
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setHandleWidth(SPACING["sm"])

        # Left panel: configuration
        left_panel = self._create_config_panel()
        left_panel.setMaximumWidth(450)
        main_splitter.addWidget(left_panel)

        # Right panel: results
        right_panel = self._create_results_panel()
        main_splitter.addWidget(right_panel)

        main_splitter.setSizes([400, 800])
        layout.addWidget(main_splitter)

    def _create_config_panel(self) -> QWidget:
        """Create the left configuration panel."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, SPACING["sm"], 0)
        layout.setSpacing(SPACING["md"])

        # MODEL Card
        model_card = Card("Model", parent=self)
        model_layout = model_card.content_layout

        # Load model button
        self.load_model_btn = OutlineButton("Load Model (.dasp)", Icons.folder_open(14, COLORS["text_primary"]))
        self.load_model_btn.clicked.connect(self._load_model)
        model_layout.addWidget(self.load_model_btn)

        # Model info display
        self.model_status = StatusCard("No model loaded", "neutral", parent=self)
        model_layout.addWidget(self.model_status)

        # Model details
        self.model_details = QLabel("")
        self.model_details.setWordWrap(True)
        self.model_details.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-size: {TYPOGRAPHY['size_sm']}pt;
            padding: {SPACING['xs']}px;
            background-color: {COLORS['bg_elevated']};
            border-radius: {RADIUS['sm']}px;
        """)
        self.model_details.setVisible(False)
        model_layout.addWidget(self.model_details)

        layout.addWidget(model_card)

        # DATA Card
        data_card = Card("Prediction Data", parent=self)
        data_layout = data_card.content_layout

        # File drop
        self.file_drop = FileDropWidget()
        self.file_drop.file_selected.connect(self._on_file_selected)
        data_layout.addWidget(self.file_drop)

        # Data info
        self.data_info = QLabel("")
        self.data_info.setStyleSheet(f"color: {COLORS['text_tertiary']}; font-size: {TYPOGRAPHY['size_sm']}pt;")
        data_layout.addWidget(self.data_info)

        # Warning label
        self.warning_label = QLabel("")
        self.warning_label.setStyleSheet(f"""
            color: {COLORS['accent_warning']};
            font-size: {TYPOGRAPHY['size_sm']}pt;
            padding: {SPACING['xs']}px;
            background-color: {COLORS['accent_warning']}20;
            border-radius: {RADIUS['sm']}px;
        """)
        self.warning_label.setWordWrap(True)
        self.warning_label.setVisible(False)
        data_layout.addWidget(self.warning_label)

        layout.addWidget(data_card)

        # PREDICT Card
        predict_card = Card(parent=self)
        predict_layout = predict_card.content_layout

        # Status
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {COLORS['text_tertiary']}; font-size: {TYPOGRAPHY['size_sm']}pt;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        predict_layout.addWidget(self.status_label)

        # Predict button
        self.predict_btn = PrimaryButton("Run Predictions", Icons.play(16, "#ffffff"))
        self.predict_btn.setMinimumHeight(48)
        self.predict_btn.setEnabled(False)
        self.predict_btn.clicked.connect(self._run_prediction)
        predict_layout.addWidget(self.predict_btn)

        layout.addWidget(predict_card)

        # BATCH Card
        batch_card = Card("Batch Processing", parent=self)
        batch_layout = batch_card.content_layout

        batch_info = QLabel("Process multiple files at once")
        batch_info.setStyleSheet(f"color: {COLORS['text_tertiary']}; font-size: {TYPOGRAPHY['size_sm']}pt;")
        batch_layout.addWidget(batch_info)

        self.batch_btn = OutlineButton("Select Folder...", Icons.folder_open(14, COLORS["text_primary"]))
        self.batch_btn.setEnabled(False)
        self.batch_btn.clicked.connect(self._batch_process)
        batch_layout.addWidget(self.batch_btn)

        layout.addWidget(batch_card)

        layout.addStretch()
        scroll.setWidget(panel)
        return scroll

    def _create_results_panel(self) -> QWidget:
        """Create the right results panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING["md"])

        # Summary Card
        summary_card = Card("Summary", parent=self)
        summary_layout = summary_card.content_layout

        # Stats grid
        stats_row = QHBoxLayout()
        stats_row.setSpacing(SPACING["md"])

        self.stat_count = self._create_stat_widget("Samples", "--")
        stats_row.addWidget(self.stat_count, 1)

        self.stat_mean = self._create_stat_widget("Mean", "--")
        stats_row.addWidget(self.stat_mean, 1)

        self.stat_std = self._create_stat_widget("Std Dev", "--")
        stats_row.addWidget(self.stat_std, 1)

        self.stat_range = self._create_stat_widget("Range", "--")
        stats_row.addWidget(self.stat_range, 1)

        self.stat_ad = self._create_stat_widget("Outside AD", "--")
        stats_row.addWidget(self.stat_ad, 1)

        summary_layout.addLayout(stats_row)
        layout.addWidget(summary_card)

        # Results Card
        results_card = Card("Predictions", parent=self)
        results_layout = results_card.content_layout

        # Results grid
        self.results_grid = SpectralDataGrid()
        self.results_grid.setMinimumHeight(300)
        results_layout.addWidget(self.results_grid)

        layout.addWidget(results_card, 1)

        # Export Card
        export_card = Card(parent=self)
        export_layout = export_card.content_layout

        btn_row = QHBoxLayout()
        btn_row.setSpacing(SPACING["sm"])

        self.export_csv_btn = PrimaryButton("Export CSV", Icons.download(14, "#ffffff"))
        self.export_csv_btn.setEnabled(False)
        self.export_csv_btn.clicked.connect(self._export_csv)
        btn_row.addWidget(self.export_csv_btn)

        self.export_report_btn = OutlineButton("Export Report", Icons.download(14, COLORS["text_primary"]))
        self.export_report_btn.setEnabled(False)
        self.export_report_btn.clicked.connect(self._export_report)
        btn_row.addWidget(self.export_report_btn)

        btn_row.addStretch()

        export_layout.addLayout(btn_row)
        layout.addWidget(export_card)

        return panel

    def _create_stat_widget(self, label: str, value: str) -> QWidget:
        """Create a statistics display widget."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(SPACING["sm"], SPACING["xs"], SPACING["sm"], SPACING["xs"])
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        label_widget = QLabel(label)
        label_widget.setStyleSheet(f"color: {COLORS['text_tertiary']}; font-size: {TYPOGRAPHY['size_xs']}pt;")
        label_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label_widget)

        value_widget = QLabel(value)
        value_widget.setObjectName("value")
        value_widget.setStyleSheet(f"""
            color: {COLORS['text_primary']};
            font-size: {TYPOGRAPHY['size_lg']}pt;
            font-weight: {TYPOGRAPHY['weight_semibold']};
        """)
        value_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(value_widget)

        container.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['bg_elevated']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: {RADIUS['sm']}px;
            }}
        """)

        return container

    def _connect_signals(self):
        """Connect signals."""
        self.state.model_changed.connect(self._on_model_changed)

    def _on_model_changed(self):
        """Handle model change from Build mode."""
        if self.state.model.selected_model:
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

            self.model_status.set_status(f"Loaded: {self._loaded_model.name}", "success")

            details = [
                f"Preprocessing: {self._loaded_model.preprocessing}",
                f"RMSECV: {rmsecv}",
            ]

            training_stats = self._loaded_model.config.get('training_stats', {})
            if training_stats:
                n_samples = training_stats.get('n_samples', '?')
                y_min = training_stats.get('y_min', 0)
                y_max = training_stats.get('y_max', 0)
                details.append(f"Trained on {n_samples} samples")
                details.append(f"Target range: {y_min:.2f} - {y_max:.2f}")

            self.model_details.setText("\n".join(details))
            self.model_details.setVisible(True)

            self.batch_btn.setEnabled(True)
            self._check_ready()

    def _load_model(self):
        """Load a saved model from .dasp file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Model", "",
            "DASP Model Files (*.dasp);;All Files (*)"
        )

        if file_path:
            try:
                engine = self._get_engine()
                self._loaded_model = engine.load_model(file_path)
                self._update_model_display()
            except Exception as e:
                QMessageBox.critical(self, "Load Failed", f"Failed to load model:\n{str(e)}")

    def _on_file_selected(self, file_path: str):
        """Handle file selection for prediction."""
        try:
            engine = self._get_engine()
            loaded = engine.load_data(file_path, for_prediction=True)

            self._prediction_data = (loaded.X, loaded.sample_ids, loaded.wavelengths)

            self.data_info.setText(
                f"{loaded.metadata['n_samples']} samples | "
                f"{loaded.metadata['n_wavelengths']} wavelengths | "
                f"Range: {loaded.metadata['wavelength_range'][0]:.0f}-{loaded.metadata['wavelength_range'][1]:.0f} nm"
            )

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
            self.warning_label.setVisible(False)
            return

        pred_wavelengths = self._prediction_data[2]

        if len(model_wavelengths) != len(pred_wavelengths):
            self.warning_label.setText(
                f"Wavelength count mismatch: Model has {len(model_wavelengths)}, "
                f"data has {len(pred_wavelengths)}. Predictions may be unreliable."
            )
            self.warning_label.setVisible(True)
        elif not np.allclose(model_wavelengths, pred_wavelengths, rtol=0.01):
            self.warning_label.setText(
                "Wavelength values differ from training data. Predictions may be unreliable."
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
        self.status_label.setStyleSheet(f"color: {COLORS['accent_secondary']}; font-size: {TYPOGRAPHY['size_sm']}pt;")

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
            self.status_label.setStyleSheet(f"color: {COLORS['accent_danger']}; font-size: {TYPOGRAPHY['size_sm']}pt;")
            QMessageBox.critical(self, "Prediction Failed", str(result))
            return

        predictions, uncertainty, ad_flags, sample_ids = result
        self.status_label.setText("Predictions complete!")
        self.status_label.setStyleSheet(f"color: {COLORS['accent_primary']}; font-size: {TYPOGRAPHY['size_sm']}pt;")

        # Store as DataFrame for export
        self._predictions_df = pd.DataFrame({
            'sample_id': sample_ids,
            'prediction': predictions,
            'uncertainty': uncertainty if uncertainty is not None else np.nan,
            'in_ad': ad_flags if ad_flags is not None else True,
        })

        # Update summary stats
        self._update_summary(predictions, ad_flags)

        # Update results grid
        self._update_results_display()

        self.export_csv_btn.setEnabled(True)
        self.export_report_btn.setEnabled(True)

    def _update_summary(self, predictions, ad_flags):
        """Update summary statistics."""
        n_outside_ad = np.sum(~ad_flags) if ad_flags is not None else 0

        self.stat_count.findChild(QLabel, "value").setText(str(len(predictions)))
        self.stat_mean.findChild(QLabel, "value").setText(f"{np.mean(predictions):.4f}")
        self.stat_std.findChild(QLabel, "value").setText(f"{np.std(predictions):.4f}")
        self.stat_range.findChild(QLabel, "value").setText(
            f"{np.min(predictions):.2f} - {np.max(predictions):.2f}"
        )

        ad_widget = self.stat_ad.findChild(QLabel, "value")
        ad_widget.setText(str(n_outside_ad))
        if n_outside_ad > 0:
            ad_widget.setStyleSheet(f"""
                color: {COLORS['accent_warning']};
                font-size: {TYPOGRAPHY['size_lg']}pt;
                font-weight: {TYPOGRAPHY['weight_semibold']};
            """)
        else:
            ad_widget.setStyleSheet(f"""
                color: {COLORS['accent_primary']};
                font-size: {TYPOGRAPHY['size_lg']}pt;
                font-weight: {TYPOGRAPHY['weight_semibold']};
            """)

    def _update_results_display(self):
        """Update the results grid with predictions."""
        if self._predictions_df is None:
            return

        df = self._predictions_df
        n_rows = len(df)
        n_cols = 4

        # Convert to numpy array for grid display
        data = np.zeros((n_rows, n_cols))
        data[:, 0] = np.arange(n_rows)  # Sample index
        data[:, 1] = df['prediction'].values
        data[:, 2] = df['uncertainty'].values if 'uncertainty' in df else np.nan
        data[:, 3] = df['in_ad'].astype(float).values if 'in_ad' in df else 1.0

        wavelengths = np.arange(n_cols)
        sample_ids = df['sample_id'].tolist()

        self.results_grid.set_data(data, wavelengths, sample_ids)
        self.results_grid.model().set_column_names(["Index", "Prediction", "Uncertainty", "In AD"])

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
                QMessageBox.information(self, "Export Complete", f"Predictions exported to:\n{file_path}")
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
                QMessageBox.information(self, "Report Exported", f"Report exported to:\n{file_path}")
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

                reply = QMessageBox.question(
                    self, "Batch Process",
                    f"Found {len(files)} files to process.\nContinue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )

                if reply != QMessageBox.StandardButton.Yes:
                    return

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

                output_path = folder_path / "batch_predictions.csv"
                pd.DataFrame(all_results).to_csv(output_path, index=False)

                QMessageBox.information(
                    self, "Batch Complete",
                    f"Processed {len(files)} files.\nResults saved to:\n{output_path}"
                )

            except Exception as e:
                QMessageBox.critical(self, "Batch Failed", str(e))
