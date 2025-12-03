"""
Build Mode - Train and refine models.

For configuring and training production models from Explore results.
"""

from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QSpinBox, QFrame, QScrollArea,
    QSplitter, QTableWidget, QMessageBox, QFileDialog,
    QCheckBox, QDoubleSpinBox, QFormLayout
)
from PySide6.QtCore import Qt, QThread, Signal
import numpy as np

from orchestration.state_store import StateStore, AppMode
from orchestration.job_runner import JobRunner
from ui.widgets.plot_widget import DiagnosticsPanel


class TrainingWorker(QThread):
    """Background worker for model training."""

    finished = Signal(object)  # TrainedModel or Exception
    progress = Signal(str)  # Status message

    def __init__(self, engine, X, y, config, wavelengths=None):
        super().__init__()
        self.engine = engine
        self.X = X
        self.y = y
        self.config = config
        self.wavelengths = wavelengths

    def run(self):
        try:
            from sklearn.model_selection import cross_val_predict, cross_val_score

            self.progress.emit("Applying preprocessing...")

            # Apply preprocessing
            X_proc = self.engine.apply_preprocessing(
                self.X,
                self.config['preprocessing']
            )

            self.progress.emit("Training model...")

            # Train model
            trained = self.engine.train_model(
                self.X,
                self.y,
                model_type=self.config['model_type'],
                preprocessing=self.config['preprocessing'],
                n_components=self.config.get('n_components', 5),
            )

            self.progress.emit("Computing cross-validation metrics...")

            # Get CV predictions for diagnostics
            from src.spectral_predict.models import get_model
            model = get_model(
                self.config['model_type'],
                n_components=self.config.get('n_components', 5)
            )

            y_pred_cv = cross_val_predict(model, X_proc, self.y, cv=5)

            # Calculate CV metrics
            from sklearn.metrics import mean_squared_error, r2_score
            rmse_cv = np.sqrt(mean_squared_error(self.y, y_pred_cv))
            r2_cv = r2_score(self.y, y_pred_cv)
            bias = np.mean(y_pred_cv - self.y)
            rpd = np.std(self.y) / rmse_cv if rmse_cv > 0 else 0

            # Update metrics
            trained.metrics['rmsecv'] = rmse_cv
            trained.metrics['r2_cv'] = r2_cv
            trained.metrics['bias'] = bias
            trained.metrics['rpd'] = rpd

            # Store CV predictions for plotting
            trained.config['y_pred_cv'] = y_pred_cv
            trained.config['y_true'] = self.y

            # Try to get loadings or coefficients for plotting
            if hasattr(trained.model, 'coef_'):
                trained.config['coefficients'] = trained.model.coef_.flatten()
            elif hasattr(trained.model, 'x_loadings_'):
                trained.config['loadings'] = trained.model.x_loadings_.T

            trained.config['wavelengths'] = self.wavelengths

            self.finished.emit(trained)

        except Exception as e:
            self.finished.emit(e)


class BuildMode(QWidget):
    """
    Build mode view for training and refining models.

    Layout:
    - MODEL SELECTION: Choose from Explore results or start fresh
    - CONFIGURATION: Preprocessing, model type, hyperparameters
    - DIAGNOSTICS: Plots and metrics for the trained model
    """

    def __init__(self, state: StateStore, job_runner: JobRunner, parent=None):
        super().__init__(parent)
        self.state = state
        self.job_runner = job_runner
        self._engine = None
        self._training_worker = None
        self._current_trained_model = None

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

        # MODEL SELECTION Section
        selection_group = self._create_selection_section()
        content_layout.addWidget(selection_group)

        # CONFIGURATION Section
        config_group = self._create_config_section()
        content_layout.addWidget(config_group)

        # DIAGNOSTICS Section
        diagnostics_group = self._create_diagnostics_section()
        content_layout.addWidget(diagnostics_group, 1)

        scroll.setWidget(content)
        layout.addWidget(scroll)

    def _create_selection_section(self) -> QGroupBox:
        """Create the MODEL SELECTION section."""
        group = QGroupBox("MODEL SELECTION")
        layout = QVBoxLayout(group)

        # Top row - selection controls
        row1 = QHBoxLayout()

        row1.addWidget(QLabel("From Explore Results:"))

        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(400)
        self.model_combo.addItem("(No results available - run Explore first)")
        self.model_combo.setEnabled(False)
        self.model_combo.currentIndexChanged.connect(self._on_model_selected)
        row1.addWidget(self.model_combo)

        row1.addWidget(QLabel("or"))

        self.fresh_btn = QPushButton("Start Fresh...")
        self.fresh_btn.clicked.connect(self._start_fresh)
        row1.addWidget(self.fresh_btn)

        row1.addStretch()
        layout.addLayout(row1)

        # Pinned models indicator
        self.pinned_label = QLabel("")
        self.pinned_label.setStyleSheet("color: #4fc3f7; font-size: 11px;")
        layout.addWidget(self.pinned_label)

        return group

    def _create_config_section(self) -> QGroupBox:
        """Create the CONFIGURATION section."""
        group = QGroupBox("CONFIGURATION")
        layout = QVBoxLayout(group)

        # Config row 1
        row1 = QHBoxLayout()

        row1.addWidget(QLabel("Preprocessing:"))
        self.preprocess_combo = QComboBox()
        self.preprocess_combo.addItems([
            "Raw", "SNV", "SG1 (1st Deriv)", "SG2 (2nd Deriv)", "MSC"
        ])
        row1.addWidget(self.preprocess_combo)

        row1.addSpacing(20)

        row1.addWidget(QLabel("Model:"))
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems([
            "PLS", "Ridge", "Lasso", "ElasticNet",
            "RandomForest", "XGBoost", "SVR", "MLP"
        ])
        self.model_type_combo.currentTextChanged.connect(self._on_model_type_changed)
        row1.addWidget(self.model_type_combo)

        row1.addStretch()
        layout.addLayout(row1)

        # Config row 2
        row2 = QHBoxLayout()

        self.components_label = QLabel("Components:")
        row2.addWidget(self.components_label)
        self.n_components = QSpinBox()
        self.n_components.setRange(1, 50)
        self.n_components.setValue(5)
        row2.addWidget(self.n_components)

        row2.addSpacing(20)

        row2.addWidget(QLabel("Variables:"))
        self.variables_combo = QComboBox()
        self.variables_combo.addItems(["All", "Top 10", "Top 20", "Top 50"])
        row2.addWidget(self.variables_combo)

        row2.addSpacing(20)

        row2.addWidget(QLabel("CV Folds:"))
        self.cv_folds = QSpinBox()
        self.cv_folds.setRange(2, 10)
        self.cv_folds.setValue(5)
        row2.addWidget(self.cv_folds)

        row2.addStretch()

        self.advanced_btn = QPushButton("Advanced ▸")
        self.advanced_btn.setCheckable(True)
        self.advanced_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 1px solid #555;
                color: #aaa;
            }
            QPushButton:checked {
                color: #4fc3f7;
                border-color: #4fc3f7;
            }
        """)
        self.advanced_btn.clicked.connect(self._toggle_advanced)
        row2.addWidget(self.advanced_btn)

        layout.addLayout(row2)

        # Advanced options (hidden by default)
        self.advanced_frame = QFrame()
        self.advanced_frame.setVisible(False)
        advanced_layout = QFormLayout(self.advanced_frame)
        advanced_layout.setContentsMargins(10, 10, 10, 10)

        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0001, 100.0)
        self.alpha_spin.setValue(1.0)
        self.alpha_spin.setDecimals(4)
        advanced_layout.addRow("Alpha/Regularization:", self.alpha_spin)

        self.l1_ratio_spin = QDoubleSpinBox()
        self.l1_ratio_spin.setRange(0.0, 1.0)
        self.l1_ratio_spin.setValue(0.5)
        self.l1_ratio_spin.setDecimals(2)
        advanced_layout.addRow("L1 Ratio (ElasticNet):", self.l1_ratio_spin)

        layout.addWidget(self.advanced_frame)

        # Train button
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888;")
        btn_row.addWidget(self.status_label)

        btn_row.addSpacing(20)

        self.train_btn = QPushButton("▶ TRAIN MODEL")
        self.train_btn.setMinimumHeight(40)
        self.train_btn.setMinimumWidth(200)
        self.train_btn.clicked.connect(self._train_model)
        btn_row.addWidget(self.train_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        return group

    def _create_diagnostics_section(self) -> QGroupBox:
        """Create the DIAGNOSTICS section with actual plots."""
        group = QGroupBox("DIAGNOSTICS")
        layout = QVBoxLayout(group)

        # Use the DiagnosticsPanel for plots
        self.diagnostics_panel = DiagnosticsPanel()
        self.diagnostics_panel.setMinimumHeight(250)
        layout.addWidget(self.diagnostics_panel)

        # Metrics row
        metrics_row = QHBoxLayout()
        self.metrics_label = QLabel("RMSECV: -- | R²: -- | RPD: -- | Bias: --")
        self.metrics_label.setStyleSheet("font-size: 14px; color: #888;")
        metrics_row.addWidget(self.metrics_label)
        metrics_row.addStretch()
        layout.addLayout(metrics_row)

        # Action buttons
        btn_row = QHBoxLayout()

        self.save_btn = QPushButton("Save Model (.dasp)")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_model)
        btn_row.addWidget(self.save_btn)

        self.export_btn = QPushButton("Export Report")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_report)
        btn_row.addWidget(self.export_btn)

        btn_row.addStretch()

        self.predict_btn = QPushButton("→ Predict Mode")
        self.predict_btn.clicked.connect(self._go_to_predict)
        btn_row.addWidget(self.predict_btn)

        layout.addLayout(btn_row)

        return group

    def _connect_signals(self):
        """Connect signals."""
        self.state.analysis_completed.connect(self._on_results_available)
        self.state.model_changed.connect(self._on_model_changed)
        self.state.data_changed.connect(self._on_data_changed)

    def _on_data_changed(self):
        """Handle data changes."""
        # Clear previous model when data changes
        self._current_trained_model = None
        self.diagnostics_panel.clear_all()
        self.metrics_label.setText("RMSECV: -- | R²: -- | RPD: -- | Bias: --")
        self.save_btn.setEnabled(False)
        self.export_btn.setEnabled(False)

    def _on_results_available(self):
        """Handle when analysis results are available from Explore."""
        if self.state.analysis.results_df is not None:
            self.model_combo.clear()
            df = self.state.analysis.results_df

            # Show pinned models first
            pinned = self.state.pinned_indices
            if pinned:
                self.pinned_label.setText(f"📌 {len(pinned)} pinned model(s) shown first")
                for idx in pinned:
                    if idx < len(df):
                        row = df.iloc[idx]
                        label = f"📌 {row.get('model', '?')} + {row.get('preprocessing', '?')} (R²: {row.get('r2_cv', row.get('r2', 0)):.3f})"
                        self.model_combo.addItem(label, idx)

                self.model_combo.insertSeparator(len(pinned))
            else:
                self.pinned_label.setText("")

            # Add top 20 results
            for i, (idx, row) in enumerate(df.head(20).iterrows()):
                if i not in pinned:
                    score = row.get('composite_score', row.get('r2_cv', row.get('r2', 0)))
                    label = f"{row.get('model', '?')} + {row.get('preprocessing', '?')} (Score: {score:.1f})"
                    self.model_combo.addItem(label, idx if isinstance(idx, int) else i)

            self.model_combo.setEnabled(True)

    def _on_model_selected(self, index):
        """Handle model selection from dropdown."""
        if index < 0 or self.state.analysis.results_df is None:
            return

        # Get the actual row index from item data
        row_idx = self.model_combo.itemData(index)
        if row_idx is None:
            return

        df = self.state.analysis.results_df
        if row_idx >= len(df):
            return

        row = df.iloc[row_idx]

        # Update configuration UI based on selected model
        preprocess = str(row.get('preprocessing', 'raw')).lower()
        preprocess_map = {
            'raw': 0, 'none': 0, 'snv': 1, 'sg1': 2, 'sg2': 3, 'msc': 4
        }
        preprocess_idx = preprocess_map.get(preprocess, 0)
        if '1' in preprocess:
            preprocess_idx = 2
        elif '2' in preprocess:
            preprocess_idx = 3
        self.preprocess_combo.setCurrentIndex(preprocess_idx)

        # Model type
        model_name = str(row.get('model', 'PLS')).upper()
        for i in range(self.model_type_combo.count()):
            if self.model_type_combo.itemText(i).upper() == model_name:
                self.model_type_combo.setCurrentIndex(i)
                break

        # Components
        if 'n_components' in row:
            self.n_components.setValue(int(row['n_components']))

    def _on_model_type_changed(self, model_type: str):
        """Update UI based on selected model type."""
        # Show/hide components spinner based on model type
        component_models = ['PLS', 'Ridge', 'Lasso', 'ElasticNet', 'SVR']
        show_components = model_type in component_models
        self.n_components.setVisible(show_components)
        self.components_label.setVisible(show_components)

        # Update label text
        if model_type == 'PLS':
            self.components_label.setText("Components:")
        else:
            self.components_label.setText("Max Iter:")

    def _on_model_changed(self):
        """Handle when a model is set in state."""
        if self.state.model.selected_model:
            metrics = self.state.model.metrics
            self.metrics_label.setText(
                f"RMSECV: {metrics.get('rmsecv', '--'):.4f} | "
                f"R²: {metrics.get('r2_cv', metrics.get('r2', '--')):.4f} | "
                f"RPD: {metrics.get('rpd', '--'):.2f} | "
                f"Bias: {metrics.get('bias', '--'):.4f}"
            )
            self.save_btn.setEnabled(True)
            self.export_btn.setEnabled(True)

    def _toggle_advanced(self):
        """Toggle advanced options visibility."""
        self.advanced_frame.setVisible(self.advanced_btn.isChecked())

    def _start_fresh(self):
        """Start with a fresh model configuration."""
        if not self.state.has_data:
            QMessageBox.warning(self, "No Data", "Please load data first in Explore mode.")
            return

        # Reset to defaults
        self.preprocess_combo.setCurrentIndex(1)  # SNV
        self.model_type_combo.setCurrentIndex(0)  # PLS
        self.n_components.setValue(5)
        self.variables_combo.setCurrentIndex(0)  # All

        self.diagnostics_panel.clear_all()
        self._current_trained_model = None
        self.metrics_label.setText("RMSECV: -- | R²: -- | RPD: -- | Bias: --")
        self.save_btn.setEnabled(False)
        self.export_btn.setEnabled(False)

    def _train_model(self):
        """Train the configured model."""
        if not self.state.has_data:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return

        if self._training_worker is not None and self._training_worker.isRunning():
            QMessageBox.warning(self, "Training in Progress", "Please wait for current training to complete.")
            return

        # Get configuration
        preprocess_text = self.preprocess_combo.currentText()
        preprocess_map = {
            'Raw': 'raw', 'SNV': 'snv', 'SG1 (1st Deriv)': 'sg1',
            'SG2 (2nd Deriv)': 'sg2', 'MSC': 'msc'
        }
        preprocessing = preprocess_map.get(preprocess_text, 'raw')

        model_type = self.model_type_combo.currentText().lower()
        n_components = self.n_components.value()

        config = {
            'preprocessing': preprocessing,
            'model_type': model_type,
            'n_components': n_components,
        }

        # Update UI
        self.train_btn.setEnabled(False)
        self.status_label.setText("Training...")

        # Start training in background
        self._training_worker = TrainingWorker(
            engine=self._get_engine(),
            X=self.state.data.X,
            y=self.state.data.y,
            config=config,
            wavelengths=self.state.data.wavelengths
        )
        self._training_worker.progress.connect(self._on_training_progress)
        self._training_worker.finished.connect(self._on_training_complete)
        self._training_worker.start()

    def _on_training_progress(self, message: str):
        """Handle training progress updates."""
        self.status_label.setText(message)

    def _on_training_complete(self, result):
        """Handle training completion."""
        self.train_btn.setEnabled(True)

        if isinstance(result, Exception):
            self.status_label.setText(f"Error: {str(result)}")
            QMessageBox.critical(self, "Training Failed", str(result))
            return

        self._current_trained_model = result
        self.status_label.setText("Training complete!")

        # Update state
        self.state.set_model(
            model=result.model,
            name=result.name,
            config=result.config,
            preprocessing=result.preprocessing,
            variable_mask=result.variable_mask,
            metrics=result.metrics
        )

        # Update diagnostics plots
        y_true = result.config.get('y_true')
        y_pred = result.config.get('y_pred_cv')
        wavelengths = result.config.get('wavelengths')
        loadings = result.config.get('loadings')
        coefficients = result.config.get('coefficients')

        if y_true is not None and y_pred is not None:
            self.diagnostics_panel.update_plots(
                y_true=y_true,
                y_pred=y_pred,
                wavelengths=wavelengths,
                loadings=loadings,
                coefficients=coefficients
            )

        # Update metrics display
        metrics = result.metrics
        self.metrics_label.setText(
            f"RMSECV: {metrics.get('rmsecv', 0):.4f} | "
            f"R²: {metrics.get('r2_cv', 0):.4f} | "
            f"RPD: {metrics.get('rpd', 0):.2f} | "
            f"Bias: {metrics.get('bias', 0):.4f}"
        )
        self.metrics_label.setStyleSheet("font-size: 14px; color: #81c784;")

        self.save_btn.setEnabled(True)
        self.export_btn.setEnabled(True)

    def _save_model(self):
        """Save the trained model."""
        if self._current_trained_model is None:
            QMessageBox.warning(self, "No Model", "Please train a model first.")
            return

        # Default filename from model name and target
        default_name = f"{self.state.data.target_column}_{self._current_trained_model.name}.dasp"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Model",
            default_name,
            "DASP Model Files (*.dasp);;All Files (*)"
        )

        if file_path:
            try:
                engine = self._get_engine()
                engine.save_model(self._current_trained_model, file_path)

                self.state.mark_model_saved(file_path)
                QMessageBox.information(
                    self, "Model Saved",
                    f"Model saved successfully to:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Save Failed", str(e))

    def _export_report(self):
        """Export a model report."""
        if self._current_trained_model is None:
            QMessageBox.warning(self, "No Model", "Please train a model first.")
            return

        default_name = f"{self.state.data.target_column}_{self._current_trained_model.name}_report.md"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Report",
            default_name,
            "Markdown Files (*.md);;All Files (*)"
        )

        if file_path:
            try:
                metrics = self._current_trained_model.metrics
                report = f"""# Model Report: {self._current_trained_model.name}

## Configuration
- **Model Type:** {self._current_trained_model.config.get('model_type', 'Unknown')}
- **Preprocessing:** {self._current_trained_model.preprocessing}
- **Components/Parameters:** {self._current_trained_model.config.get('n_components', 'N/A')}

## Data
- **File:** {self.state.data.file_name}
- **Samples:** {self.state.data.n_samples}
- **Wavelengths:** {self.state.data.n_wavelengths} ({self.state.data.wavelength_min:.0f}-{self.state.data.wavelength_max:.0f} nm)
- **Target:** {self.state.data.target_column}

## Cross-Validation Metrics (5-fold)
- **RMSECV:** {metrics.get('rmsecv', 0):.4f}
- **R² (CV):** {metrics.get('r2_cv', 0):.4f}
- **RPD:** {metrics.get('rpd', 0):.2f}
- **Bias:** {metrics.get('bias', 0):.4f}

## Calibration Metrics
- **RMSE (Cal):** {metrics.get('rmse', 0):.4f}
- **R² (Cal):** {metrics.get('r2', 0):.4f}

---
*Generated by Spectral Predict 2.0*
"""
                Path(file_path).write_text(report)
                QMessageBox.information(
                    self, "Report Exported",
                    f"Report exported to:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", str(e))

    def _go_to_predict(self):
        """Switch to Predict mode."""
        self.state.set_mode(AppMode.PREDICT)
