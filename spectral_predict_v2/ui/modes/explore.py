"""
Explore Mode - Automated model discovery.

Primary entry point for finding the best model automatically.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QSlider, QProgressBar, QTableWidget,
    QTableWidgetItem, QHeaderView, QCheckBox, QFrame, QScrollArea,
    QSplitter, QFileDialog, QMessageBox, QSpinBox, QDialog
)
from PySide6.QtCore import Qt, Signal, Slot
import numpy as np

from orchestration.state_store import StateStore, TaskType
from orchestration.job_runner import JobRunner, JobResult
from orchestration.config_manager import ConfigManager, AnalysisPreset
from engine.api import EngineAPI, AnalysisConfig
from ui.widgets.file_drop import FileDropWidget
from ui.widgets.column_config_dialog import ColumnConfigDialog


class ExploreMode(QWidget):
    """
    Explore mode view for automated model discovery.

    Layout:
    - DATA section: File drop, preset selector, target/task selection
    - ANALYSIS section: Thoroughness slider, run button, advanced options
    - RESULTS section: Sortable/filterable table with pinning for comparison
    """

    def __init__(self, state: StateStore, job_runner: JobRunner, config: ConfigManager, parent=None):
        super().__init__(parent)
        self.state = state
        self.job_runner = job_runner
        self.config = config
        self.engine = EngineAPI()

        self._current_job_id = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 10, 20, 20)
        layout.setSpacing(16)

        # Create scrollable content area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(16)

        # DATA Section
        data_group = self._create_data_section()
        content_layout.addWidget(data_group)

        # ANALYSIS Section
        analysis_group = self._create_analysis_section()
        content_layout.addWidget(analysis_group)

        # RESULTS Section
        results_group = self._create_results_section()
        content_layout.addWidget(results_group, 1)  # Stretch

        scroll.setWidget(content)
        layout.addWidget(scroll)

    def _create_data_section(self) -> QGroupBox:
        """Create the DATA section."""
        group = QGroupBox("DATA")
        layout = QHBoxLayout(group)
        layout.setSpacing(16)

        # File drop widget
        self.file_drop = FileDropWidget()
        self.file_drop.setMinimumWidth(300)
        self.file_drop.setMaximumWidth(400)
        self.file_drop.file_selected.connect(self._on_file_selected)
        layout.addWidget(self.file_drop)

        # Preset selector
        preset_layout = QVBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumWidth(180)
        self._populate_presets()
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(self.preset_combo)
        preset_layout.addStretch()
        layout.addLayout(preset_layout)

        # Target variable
        target_layout = QVBoxLayout()
        target_layout.addWidget(QLabel("Target:"))
        self.target_combo = QComboBox()
        self.target_combo.setMinimumWidth(150)
        self.target_combo.setEnabled(False)
        self.target_combo.currentIndexChanged.connect(self._on_target_changed)
        target_layout.addWidget(self.target_combo)
        target_layout.addStretch()
        layout.addLayout(target_layout)

        # Task type
        task_layout = QVBoxLayout()
        task_layout.addWidget(QLabel("Task:"))
        self.task_combo = QComboBox()
        self.task_combo.addItems(["Regression", "Classification"])
        self.task_combo.currentIndexChanged.connect(self._on_task_changed)
        task_layout.addWidget(self.task_combo)
        task_layout.addStretch()
        layout.addLayout(task_layout)

        layout.addStretch()

        return group

    def _create_analysis_section(self) -> QGroupBox:
        """Create the ANALYSIS section."""
        group = QGroupBox("ANALYSIS")
        layout = QVBoxLayout(group)

        # Thoroughness slider row
        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Thoroughness:"))

        self.thoroughness_slider = QSlider(Qt.Orientation.Horizontal)
        self.thoroughness_slider.setMinimum(0)
        self.thoroughness_slider.setMaximum(3)
        self.thoroughness_slider.setValue(1)  # Standard
        self.thoroughness_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.thoroughness_slider.setTickInterval(1)
        self.thoroughness_slider.setMinimumWidth(300)
        slider_row.addWidget(self.thoroughness_slider)

        self.time_estimate_label = QLabel("~15 min")
        self.time_estimate_label.setStyleSheet("color: #888;")
        self.time_estimate_label.setMinimumWidth(80)
        slider_row.addWidget(self.time_estimate_label)

        slider_row.addStretch()

        # Advanced options button
        self.advanced_btn = QPushButton("Advanced ▸")
        self.advanced_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 1px solid #555;
                color: #aaa;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
            }
        """)
        self.advanced_btn.clicked.connect(self._toggle_advanced)
        slider_row.addWidget(self.advanced_btn)

        layout.addLayout(slider_row)

        # Slider labels
        labels_row = QHBoxLayout()
        labels_row.addSpacing(100)  # Offset for "Thoroughness:" label
        labels_row.addWidget(QLabel("Quick"))
        labels_row.addStretch()
        labels_row.addWidget(QLabel("Standard"))
        labels_row.addStretch()
        labels_row.addWidget(QLabel("Comprehensive"))
        labels_row.addStretch()
        labels_row.addWidget(QLabel("Experimental"))
        labels_row.addSpacing(160)
        layout.addLayout(labels_row)

        # Advanced options (hidden by default)
        self.advanced_frame = QFrame()
        self.advanced_frame.setVisible(False)
        self.advanced_frame.setStyleSheet("QFrame { background-color: #252525; border-radius: 4px; padding: 8px; }")
        adv_layout = QVBoxLayout(self.advanced_frame)

        # Preprocessing options
        preprocess_row = QHBoxLayout()
        preprocess_row.addWidget(QLabel("Preprocessing:"))
        self.preprocess_checks = {}
        for name in ["Raw", "SNV", "SG1", "SG2", "MSC"]:
            cb = QCheckBox(name)
            cb.setChecked(name in ["Raw", "SNV", "SG1"])
            self.preprocess_checks[name.lower()] = cb
            preprocess_row.addWidget(cb)
        preprocess_row.addStretch()
        adv_layout.addLayout(preprocess_row)

        # Model options - container for dynamic model checkboxes
        self.model_container = QWidget()
        self.model_layout = QHBoxLayout(self.model_container)
        self.model_layout.setContentsMargins(0, 0, 0, 0)
        self.model_layout.addWidget(QLabel("Models:"))
        self.model_checks = {}

        # Define models for each task type
        self.regression_models = ["PLS", "Ridge", "ElasticNet", "RandomForest", "XGBoost", "LightGBM", "SVR"]
        self.classification_models = ["PLS-DA", "RandomForest", "XGBoost", "LightGBM", "SVM", "LDA"]

        # Create checkboxes for regression models (default)
        for name in self.regression_models:
            cb = QCheckBox(name)
            cb.setChecked(name in ["PLS", "Ridge", "ElasticNet"])
            key = name.lower().replace("-", "")
            self.model_checks[key] = cb
            self.model_layout.addWidget(cb)

        self.model_layout.addStretch()
        adv_layout.addWidget(self.model_container)

        # Bayesian option
        bayesian_row = QHBoxLayout()
        self.bayesian_check = QCheckBox("Use Bayesian Optimization (faster)")
        self.bayesian_check.setChecked(True)
        bayesian_row.addWidget(self.bayesian_check)
        bayesian_row.addWidget(QLabel("Trials:"))
        self.n_trials_spin = QSpinBox()
        self.n_trials_spin.setRange(10, 200)
        self.n_trials_spin.setValue(50)
        bayesian_row.addWidget(self.n_trials_spin)
        bayesian_row.addStretch()
        adv_layout.addLayout(bayesian_row)

        layout.addWidget(self.advanced_frame)

        # Run button
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self.run_btn = QPushButton("▶ RUN AUTO ANALYSIS")
        self.run_btn.setMinimumHeight(50)
        self.run_btn.setMinimumWidth(250)
        self.run_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                font-weight: bold;
            }
        """)
        self.run_btn.setEnabled(False)  # Disabled until data loaded
        self.run_btn.clicked.connect(self._run_analysis)
        btn_row.addWidget(self.run_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self._cancel_analysis)
        btn_row.addWidget(self.cancel_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Progress bar (hidden until analysis starts)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(24)
        layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.progress_label)

        return group

    def _create_results_section(self) -> QGroupBox:
        """Create the RESULTS section."""
        group = QGroupBox("RESULTS")
        layout = QVBoxLayout(group)

        # Filter/sort row
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))

        self.model_filter = QComboBox()
        self.model_filter.addItems(["All Models", "PLS", "Ridge", "ElasticNet", "RandomForest", "XGBoost", "LightGBM"])
        self.model_filter.currentIndexChanged.connect(self._apply_filters)
        filter_row.addWidget(self.model_filter)

        self.preprocess_filter = QComboBox()
        self.preprocess_filter.addItems(["All Preprocessing", "Raw", "SNV", "SG1", "SG2", "MSC"])
        self.preprocess_filter.currentIndexChanged.connect(self._apply_filters)
        filter_row.addWidget(self.preprocess_filter)

        filter_row.addWidget(QLabel("Score >"))
        self.score_filter = QComboBox()
        self.score_filter.addItems(["Any", "50", "70", "80", "90"])
        self.score_filter.currentIndexChanged.connect(self._apply_filters)
        filter_row.addWidget(self.score_filter)

        filter_row.addStretch()

        self.results_count_label = QLabel("No results yet")
        self.results_count_label.setStyleSheet("color: #888;")
        filter_row.addWidget(self.results_count_label)

        layout.addLayout(filter_row)

        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(7)
        self.results_table.setHorizontalHeaderLabels([
            "Pin", "Model", "Preprocessing", "RMSECV", "R²", "Variables", "Score"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.results_table.setColumnWidth(0, 40)
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.cellDoubleClicked.connect(self._on_result_double_click)
        layout.addWidget(self.results_table)

        # Action buttons
        btn_row = QHBoxLayout()

        self.compare_btn = QPushButton("Compare Pinned (0)")
        self.compare_btn.setEnabled(False)
        self.compare_btn.clicked.connect(self._compare_pinned)
        btn_row.addWidget(self.compare_btn)

        self.export_btn = QPushButton("Export CSV")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_results)
        btn_row.addWidget(self.export_btn)

        btn_row.addStretch()

        self.build_btn = QPushButton("→ Build Selected")
        self.build_btn.setEnabled(False)
        self.build_btn.clicked.connect(self._go_to_build)
        btn_row.addWidget(self.build_btn)

        layout.addLayout(btn_row)

        return group

    def _populate_presets(self):
        """Populate the preset dropdown."""
        self.preset_combo.clear()
        self.preset_combo.addItem("Custom", None)
        for key, name in self.config.list_presets():
            self.preset_combo.addItem(name, key)
        # Default to Standard preset
        idx = self.preset_combo.findText("NIR Protein (Grain)")
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)

    def _connect_signals(self):
        """Connect signals."""
        self.state.data_changed.connect(self._on_data_changed)
        self.state.analysis_completed.connect(self._on_analysis_completed)
        self.thoroughness_slider.valueChanged.connect(self._update_time_estimate)

        # Job runner signals
        self.job_runner.job_progress.connect(self._on_job_progress)
        self.job_runner.job_completed.connect(self._on_job_completed)

    def _on_file_selected(self, file_path: str):
        """Handle file selection - shows column config dialog."""
        try:
            # Show column configuration dialog
            dialog = ColumnConfigDialog(file_path, self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                self.file_drop.clear()
                return

            config = dialog.get_config()
            if not config:
                self.file_drop.clear()
                return

            # Load data with configuration
            data = self.engine.load_data_with_config(
                file_path,
                id_column=config.get('id_column'),
                target_column=config.get('target_column'),
                metadata_columns=config.get('metadata_columns', [])
            )

            # Update state
            self.state.load_data(
                X=data.X,
                y=data.y if data.y is not None else np.zeros(data.X.shape[0]),
                wavelengths=data.wavelengths,
                file_path=file_path,
                target_column=data.target_column or "Unknown",
                sample_ids=data.sample_ids
            )

            # Populate target combo
            self.target_combo.clear()
            if data.available_targets:
                self.target_combo.addItems(data.available_targets)
                self.target_combo.setEnabled(True)
                if data.target_column:
                    idx = self.target_combo.findText(data.target_column)
                    if idx >= 0:
                        self.target_combo.setCurrentIndex(idx)
            else:
                self.target_combo.addItem("(No targets detected)")
                self.target_combo.setEnabled(False)

            # Enable run button if we have target
            self.run_btn.setEnabled(data.y is not None)

            # Auto-set task type based on target
            if data.metadata.get("is_classification", False):
                self.task_combo.setCurrentIndex(1)  # Classification
                self.state.set_task_type(TaskType.CLASSIFICATION)
            else:
                self.task_combo.setCurrentIndex(0)  # Regression
                self.state.set_task_type(TaskType.REGRESSION)

            # Show success message
            n_samples = data.X.shape[0]
            n_wl = data.X.shape[1]
            wl_range = f"{data.wavelengths.min():.0f}-{data.wavelengths.max():.0f}"
            QMessageBox.information(
                self,
                "Data Loaded",
                f"Successfully loaded:\n"
                f"• {n_samples} samples\n"
                f"• {n_wl} wavelengths ({wl_range} nm)\n"
                f"• Target: {data.target_column or 'None detected'}"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Data",
                f"Could not load file:\n{str(e)}"
            )
            self.file_drop.clear()

    def _on_data_changed(self):
        """Handle data change from state."""
        has_data = self.state.has_data
        has_target = self.state.data.y is not None and len(self.state.data.y) > 0
        self.run_btn.setEnabled(has_data and has_target)

    def _on_preset_changed(self, index: int):
        """Handle preset selection."""
        preset_key = self.preset_combo.currentData()
        if preset_key:
            preset = self.config.get_preset(preset_key)
            if preset:
                self._apply_preset(preset)
                self.state.set_preset(preset_key)

    def _apply_preset(self, preset: AnalysisPreset):
        """Apply preset configuration to UI."""
        # Set task type
        if preset.task_type == "classification":
            self.task_combo.setCurrentIndex(1)
        else:
            self.task_combo.setCurrentIndex(0)

        # Set preprocessing checkboxes
        for key, cb in self.preprocess_checks.items():
            cb.setChecked(key in preset.preprocessing.methods)

        # Set model checkboxes
        for key, cb in self.model_checks.items():
            cb.setChecked(key in preset.models.model_types)

        # Set Bayesian
        self.bayesian_check.setChecked(preset.models.use_bayesian)
        self.n_trials_spin.setValue(preset.models.n_trials)

    def _on_target_changed(self, index: int):
        """Handle target selection change."""
        # TODO: Reload y values for new target
        pass

    def _on_task_changed(self, index: int):
        """Handle task type change."""
        task = TaskType.CLASSIFICATION if index == 1 else TaskType.REGRESSION
        self.state.set_task_type(task)
        self._update_model_checkboxes(task)

    def _update_model_checkboxes(self, task: TaskType):
        """Update model checkboxes based on task type."""
        # Clear existing checkboxes (except the label)
        while self.model_layout.count() > 1:
            item = self.model_layout.takeAt(1)
            if item.widget():
                item.widget().deleteLater()

        self.model_checks = {}

        if task == TaskType.CLASSIFICATION:
            models = self.classification_models
            defaults = ["PLS-DA", "RandomForest", "SVM"]
        else:
            models = self.regression_models
            defaults = ["PLS", "Ridge", "ElasticNet"]

        for name in models:
            cb = QCheckBox(name)
            cb.setChecked(name in defaults)
            key = name.lower().replace("-", "")
            self.model_checks[key] = cb
            self.model_layout.addWidget(cb)

        self.model_layout.addStretch()

    def _toggle_advanced(self):
        """Toggle advanced options visibility."""
        visible = not self.advanced_frame.isVisible()
        self.advanced_frame.setVisible(visible)
        self.advanced_btn.setText("Advanced ▾" if visible else "Advanced ▸")

    def _update_time_estimate(self, value: int):
        """Update the time estimate based on thoroughness."""
        estimates = ["~2 min", "~15 min", "~45 min", "~2+ hours"]
        self.time_estimate_label.setText(estimates[value])

    def _get_analysis_config(self) -> AnalysisConfig:
        """Build analysis config from UI settings."""
        # Get task type
        task_type = "classification" if self.task_combo.currentIndex() == 1 else "regression"

        # Get selected preprocessing
        preprocessing = [k for k, cb in self.preprocess_checks.items() if cb.isChecked()]
        if not preprocessing:
            preprocessing = ["raw"]

        # Get selected models - use appropriate defaults based on task type
        models = [k for k, cb in self.model_checks.items() if cb.isChecked()]
        if not models:
            if task_type == "classification":
                models = ["plsda", "randomforest", "svm"]
            else:
                models = ["pls", "ridge"]

        # Adjust based on thoroughness
        thoroughness = self.thoroughness_slider.value()
        n_trials = self.n_trials_spin.value()

        if thoroughness == 0:  # Quick
            n_trials = min(n_trials, 20)
            models = models[:2]  # Limit models
        elif thoroughness == 2:  # Comprehensive
            n_trials = max(n_trials, 100)
        elif thoroughness == 3:  # Experimental
            n_trials = max(n_trials, 150)

        return AnalysisConfig(
            preprocessing_methods=preprocessing,
            model_types=models,
            task_type=task_type,
            use_bayesian=self.bayesian_check.isChecked(),
            n_bayesian_trials=n_trials,
            n_folds=5,
            random_state=42,
        )

    def _run_analysis(self):
        """Start the automated analysis."""
        if not self.state.has_data:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return

        # Get configuration
        config = self._get_analysis_config()

        # Show progress UI
        self.run_btn.setVisible(False)
        self.cancel_btn.setVisible(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setVisible(True)
        self.progress_label.setText("Starting analysis...")

        # Mark analysis as started
        self.state.start_analysis()

        # Run analysis in background
        X = self.state.get_active_X()
        y = self.state.data.y

        self._current_job_id = self.job_runner.run(
            self.engine.run_analysis,
            X, y, config,
            job_id="explore_analysis"
        )

    @Slot(str, float, str)
    def _on_job_progress(self, job_id: str, progress: float, stage: str):
        """Handle job progress update."""
        if job_id == self._current_job_id:
            self.progress_bar.setValue(int(progress * 100))
            self.progress_label.setText(stage)
            self.state.update_progress(progress, stage)

    @Slot(str, JobResult)
    def _on_job_completed(self, job_id: str, result: JobResult):
        """Handle job completion."""
        if job_id != self._current_job_id:
            return

        self._current_job_id = None

        # Reset UI
        self.run_btn.setVisible(True)
        self.cancel_btn.setVisible(False)
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)

        if result.success:
            # Store results
            self.state.complete_analysis(result.result)
            self._populate_results(result.result)
            self.export_btn.setEnabled(True)
            self.build_btn.setEnabled(True)

            n_models = len(result.result)
            QMessageBox.information(
                self,
                "Analysis Complete",
                f"Evaluated {n_models} model configurations.\n"
                f"Best: {self.state.analysis.best_model_name}"
            )
        else:
            QMessageBox.critical(
                self,
                "Analysis Failed",
                f"Error: {result.error}\n\n{result.traceback}"
            )

    def _cancel_analysis(self):
        """Cancel the running analysis."""
        if self._current_job_id:
            self.job_runner.cancel(self._current_job_id)
        self.run_btn.setVisible(True)
        self.cancel_btn.setVisible(False)
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)

    def _on_analysis_completed(self):
        """Handle analysis completion from state."""
        results = self.state.analysis.results_df
        if results is not None and len(results) > 0:
            self._populate_results(results)

    def _populate_results(self, df):
        """Populate the results table."""
        self._results_df = df  # Store for filtering
        self._display_results(df)

    def _display_results(self, df):
        """Display results in the table."""
        self.results_table.setRowCount(len(df))
        for i, (_, row) in enumerate(df.iterrows()):
            # Pin checkbox
            pin_cb = QCheckBox()
            pin_cb.stateChanged.connect(lambda state, idx=i: self._toggle_pin(idx, state))
            self.results_table.setCellWidget(i, 0, pin_cb)

            # Model
            model_name = str(row.get("model", row.get("model_type", "")))
            self.results_table.setItem(i, 1, QTableWidgetItem(model_name))

            # Preprocessing
            preprocess = str(row.get("preprocessing", row.get("preprocess", "")))
            self.results_table.setItem(i, 2, QTableWidgetItem(preprocess))

            # RMSECV
            rmsecv = row.get("rmsecv", row.get("mean_rmsecv", 0))
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{rmsecv:.4f}"))

            # R²
            r2 = row.get("r2", row.get("mean_r2", 0))
            self.results_table.setItem(i, 4, QTableWidgetItem(f"{r2:.4f}"))

            # Variables
            n_vars = row.get("n_variables", row.get("n_features", "all"))
            self.results_table.setItem(i, 5, QTableWidgetItem(str(n_vars)))

            # Score
            score = row.get("composite_score", row.get("score", 0))
            score_text = f"{score:.1f}"
            if i == 0:
                score_text += " 🥇"
            elif i == 1:
                score_text += " 🥈"
            elif i == 2:
                score_text += " 🥉"
            self.results_table.setItem(i, 6, QTableWidgetItem(score_text))

        self.results_count_label.setText(f"Showing: {len(df)} models")

    def _apply_filters(self):
        """Apply filters to results."""
        if not hasattr(self, '_results_df') or self._results_df is None:
            return

        df = self._results_df.copy()

        # Model filter
        model_filter = self.model_filter.currentText()
        if model_filter != "All Models":
            df = df[df.get("model", df.get("model_type", "")).str.lower().str.contains(model_filter.lower())]

        # Preprocessing filter
        preprocess_filter = self.preprocess_filter.currentText()
        if preprocess_filter != "All Preprocessing":
            df = df[df.get("preprocessing", df.get("preprocess", "")).str.lower() == preprocess_filter.lower()]

        # Score filter
        score_filter = self.score_filter.currentText()
        if score_filter != "Any":
            min_score = float(score_filter)
            df = df[df.get("composite_score", df.get("score", 0)) >= min_score]

        self._display_results(df)

    def _toggle_pin(self, index: int, state: int):
        """Toggle a model's pinned state."""
        is_pinned = self.state.toggle_pinned(index)
        n_pinned = len(self.state.pinned_indices)
        self.compare_btn.setText(f"Compare Pinned ({n_pinned})")
        self.compare_btn.setEnabled(n_pinned >= 2)

    def _on_result_double_click(self, row: int, col: int):
        """Handle double-click on result row."""
        # TODO: Show detailed model info
        if hasattr(self, '_results_df') and self._results_df is not None:
            model_info = self._results_df.iloc[row].to_dict()
            info_text = "\n".join(f"{k}: {v}" for k, v in model_info.items())
            QMessageBox.information(self, "Model Details", info_text)

    def _compare_pinned(self):
        """Show comparison of pinned models."""
        indices = self.state.pinned_indices
        if len(indices) < 2:
            return

        # TODO: Implement proper comparison view
        QMessageBox.information(
            self,
            "Compare Models",
            f"Comparing {len(indices)} models:\n"
            f"Indices: {indices}\n\n"
            "(Full comparison view coming in Phase 3)"
        )

    def _export_results(self):
        """Export results to CSV."""
        if not hasattr(self, '_results_df') or self._results_df is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "analysis_results.csv", "CSV Files (*.csv)"
        )
        if file_path:
            self._results_df.to_csv(file_path, index=False)
            QMessageBox.information(self, "Export", f"Results exported to:\n{file_path}")

    def _go_to_build(self):
        """Switch to Build mode with selected model."""
        from orchestration.state_store import AppMode
        self.state.set_mode(AppMode.BUILD)
