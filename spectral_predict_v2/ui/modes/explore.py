"""
Explore Mode - Automated model discovery.

Modern card-based layout with integrated data viewing and analysis.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QProgressBar, QFrame, QScrollArea, QSplitter,
    QFileDialog, QMessageBox, QSpinBox, QDialog, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, Slot
import numpy as np

from orchestration.state_store import StateStore, TaskType
from orchestration.job_runner import JobRunner, JobResult
from orchestration.config_manager import ConfigManager, AnalysisPreset
from engine.api import EngineAPI, AnalysisConfig

from ..theme.tokens import COLORS, SPACING, RADIUS, TYPOGRAPHY
from ..theme.icons import Icons
from ..components.cards import Card, CollapsibleCard
from ..components.buttons import PrimaryButton, SecondaryButton, GhostButton, OutlineButton
from ..components.inputs import (
    StyledComboBox, StyledSpinBox, StyledSlider, StyledCheckBox, LabeledInput
)
from ..widgets.file_drop import FileDropWidget
from ..widgets.column_config_dialog import ColumnConfigDialog
from ..widgets.data_grid import SpectralDataGrid
from ..widgets.plot_widget import SpectraOverlayPlot


class ExploreMode(QWidget):
    """
    Explore mode view for automated model discovery.

    Modern card-based layout:
    - Left panel: Data loading, configuration, run controls
    - Right panel: Data preview (grid + spectra plot)
    - Bottom: Results table with filtering
    """

    def __init__(self, state: StateStore, job_runner: JobRunner, config: ConfigManager, parent=None):
        super().__init__(parent)
        self.state = state
        self.job_runner = job_runner
        self.config = config
        self.engine = EngineAPI()

        self._current_job_id = None
        self._results_df = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACING["lg"], SPACING["md"], SPACING["lg"], SPACING["lg"])
        layout.setSpacing(SPACING["md"])

        # Main splitter: top (config + preview) and bottom (results)
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.setHandleWidth(SPACING["sm"])

        # Top section: config left, preview right
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(SPACING["md"])

        # Left panel: configuration
        left_panel = self._create_config_panel()
        left_panel.setMinimumWidth(400)
        top_layout.addWidget(left_panel)

        # Right panel: data preview
        right_panel = self._create_preview_panel()
        top_layout.addWidget(right_panel, 1)

        main_splitter.addWidget(top_widget)

        # Bottom section: results
        results_panel = self._create_results_panel()
        main_splitter.addWidget(results_panel)

        # Set splitter stretch factors (top gets more space)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 2)

        layout.addWidget(main_splitter)

    def _create_config_panel(self) -> QWidget:
        """Create the left configuration panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING["md"])

        # DATA Card
        data_card = Card("Load Data", parent=self)
        data_layout = data_card.content_layout

        # File drop
        self.file_drop = FileDropWidget()
        self.file_drop.file_selected.connect(self._on_file_selected)
        data_layout.addWidget(self.file_drop)

        # Target and task selection row
        target_row = QHBoxLayout()
        target_row.setSpacing(SPACING["sm"])

        # Target combo
        target_container = QWidget()
        target_inner = QVBoxLayout(target_container)
        target_inner.setContentsMargins(0, 0, 0, 0)
        target_inner.setSpacing(SPACING["xs"])
        target_label = QLabel("Target Variable")
        target_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: {TYPOGRAPHY['size_sm']}pt;")
        target_inner.addWidget(target_label)
        self.target_combo = StyledComboBox()
        self.target_combo.setEnabled(False)
        self.target_combo.currentIndexChanged.connect(self._on_target_changed)
        target_inner.addWidget(self.target_combo)
        target_row.addWidget(target_container, 1)

        # Task type combo
        task_container = QWidget()
        task_inner = QVBoxLayout(task_container)
        task_inner.setContentsMargins(0, 0, 0, 0)
        task_inner.setSpacing(SPACING["xs"])
        task_label = QLabel("Task Type")
        task_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: {TYPOGRAPHY['size_sm']}pt;")
        task_inner.addWidget(task_label)
        self.task_combo = StyledComboBox()
        self.task_combo.addItems(["Regression", "Classification"])
        self.task_combo.currentIndexChanged.connect(self._on_task_changed)
        task_inner.addWidget(self.task_combo)
        target_row.addWidget(task_container, 1)

        data_layout.addLayout(target_row)
        layout.addWidget(data_card)

        # ANALYSIS Card
        analysis_card = Card("Analysis Settings", parent=self)
        analysis_layout = analysis_card.content_layout

        # Preset row
        preset_row = QHBoxLayout()
        preset_row.setSpacing(SPACING["sm"])
        preset_label = QLabel("Preset:")
        preset_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        preset_row.addWidget(preset_label)
        self.preset_combo = StyledComboBox()
        self._populate_presets()
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        preset_row.addWidget(self.preset_combo, 1)
        analysis_layout.addLayout(preset_row)

        # Thoroughness slider
        thoroughness_container = QWidget()
        thor_layout = QVBoxLayout(thoroughness_container)
        thor_layout.setContentsMargins(0, 0, 0, 0)
        thor_layout.setSpacing(SPACING["xs"])

        thor_header = QHBoxLayout()
        thor_label = QLabel("Thoroughness")
        thor_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        thor_header.addWidget(thor_label)
        self.time_estimate_label = QLabel("~15 min")
        self.time_estimate_label.setStyleSheet(f"color: {COLORS['text_tertiary']};")
        thor_header.addStretch()
        thor_header.addWidget(self.time_estimate_label)
        thor_layout.addLayout(thor_header)

        self.thoroughness_slider = StyledSlider(Qt.Orientation.Horizontal)
        self.thoroughness_slider.setMinimum(0)
        self.thoroughness_slider.setMaximum(3)
        self.thoroughness_slider.setValue(1)
        self.thoroughness_slider.setTickPosition(StyledSlider.TickPosition.TicksBelow)
        self.thoroughness_slider.setTickInterval(1)
        self.thoroughness_slider.valueChanged.connect(self._update_time_estimate)
        thor_layout.addWidget(self.thoroughness_slider)

        # Slider labels
        slider_labels = QHBoxLayout()
        for text in ["Quick", "Standard", "Thorough", "Exhaustive"]:
            lbl = QLabel(text)
            lbl.setStyleSheet(f"color: {COLORS['text_tertiary']}; font-size: {TYPOGRAPHY['size_xs']}pt;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            slider_labels.addWidget(lbl)
        thor_layout.addLayout(slider_labels)

        analysis_layout.addWidget(thoroughness_container)
        layout.addWidget(analysis_card)

        # ADVANCED Card (collapsible)
        self.advanced_card = CollapsibleCard("Advanced Options", start_collapsed=True, parent=self)
        adv_layout = self.advanced_card.content_layout

        # Preprocessing options
        preprocess_label = QLabel("Preprocessing Methods")
        preprocess_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        adv_layout.addWidget(preprocess_label)

        preprocess_row = QHBoxLayout()
        preprocess_row.setSpacing(SPACING["sm"])
        self.preprocess_checks = {}
        for name in ["Raw", "SNV", "SG1", "SG2", "MSC"]:
            cb = StyledCheckBox(name)
            cb.setChecked(name in ["Raw", "SNV", "SG1"])
            self.preprocess_checks[name.lower()] = cb
            preprocess_row.addWidget(cb)
        preprocess_row.addStretch()
        adv_layout.addLayout(preprocess_row)

        # Model options
        models_label = QLabel("Model Types")
        models_label.setStyleSheet(f"color: {COLORS['text_secondary']}; margin-top: {SPACING['sm']}px;")
        adv_layout.addWidget(models_label)

        # Container for dynamic model checkboxes
        self.model_container = QWidget()
        self.model_layout = QHBoxLayout(self.model_container)
        self.model_layout.setContentsMargins(0, 0, 0, 0)
        self.model_layout.setSpacing(SPACING["sm"])
        self.model_checks = {}

        self.regression_models = ["PLS", "Ridge", "ElasticNet", "RandomForest", "XGBoost", "LightGBM", "SVR"]
        self.classification_models = ["PLS-DA", "RandomForest", "XGBoost", "LightGBM", "SVM", "LDA"]

        for name in self.regression_models:
            cb = StyledCheckBox(name)
            cb.setChecked(name in ["PLS", "Ridge", "ElasticNet"])
            key = name.lower().replace("-", "")
            self.model_checks[key] = cb
            self.model_layout.addWidget(cb)
        self.model_layout.addStretch()
        adv_layout.addWidget(self.model_container)

        # Bayesian options
        bayes_row = QHBoxLayout()
        bayes_row.setSpacing(SPACING["sm"])
        self.bayesian_check = StyledCheckBox("Bayesian Optimization")
        self.bayesian_check.setChecked(True)
        bayes_row.addWidget(self.bayesian_check)
        bayes_row.addWidget(QLabel("Trials:"))
        self.n_trials_spin = StyledSpinBox()
        self.n_trials_spin.setRange(10, 200)
        self.n_trials_spin.setValue(50)
        self.n_trials_spin.setFixedWidth(80)
        bayes_row.addWidget(self.n_trials_spin)
        bayes_row.addStretch()
        adv_layout.addLayout(bayes_row)

        layout.addWidget(self.advanced_card)

        # RUN section
        run_card = Card(parent=self)
        run_layout = run_card.content_layout

        # Progress (hidden initially)
        self.progress_container = QWidget()
        self.progress_container.setVisible(False)
        progress_inner = QVBoxLayout(self.progress_container)
        progress_inner.setContentsMargins(0, 0, 0, 0)
        progress_inner.setSpacing(SPACING["xs"])

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {COLORS["bg_elevated"]};
                border: 1px solid {COLORS["border_default"]};
                border-radius: {RADIUS["sm"]}px;
                height: 8px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS["accent_primary"]};
                border-radius: {RADIUS["sm"]}px;
            }}
        """)
        progress_inner.addWidget(self.progress_bar)

        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: {TYPOGRAPHY['size_sm']}pt;")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_inner.addWidget(self.progress_label)

        run_layout.addWidget(self.progress_container)

        # Button row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(SPACING["sm"])

        self.run_btn = PrimaryButton("Run Analysis", Icons.play(16, "#ffffff"))
        self.run_btn.setMinimumHeight(44)
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._run_analysis)
        btn_row.addWidget(self.run_btn, 1)

        self.cancel_btn = OutlineButton("Cancel")
        self.cancel_btn.setMinimumHeight(44)
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self._cancel_analysis)
        btn_row.addWidget(self.cancel_btn)

        run_layout.addLayout(btn_row)
        layout.addWidget(run_card)

        layout.addStretch()
        return panel

    def _create_preview_panel(self) -> QWidget:
        """Create the right data preview panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING["md"])

        # Preview splitter: grid top, plot bottom
        preview_splitter = QSplitter(Qt.Orientation.Vertical)
        preview_splitter.setHandleWidth(SPACING["xs"])

        # Data grid card
        grid_card = Card("Data Preview", parent=self)
        self.data_grid = SpectralDataGrid()
        grid_card.content_layout.addWidget(self.data_grid)
        preview_splitter.addWidget(grid_card)

        # Spectra plot card
        plot_card = Card("Spectra", parent=self)
        self.spectra_plot = SpectraOverlayPlot()
        plot_card.content_layout.addWidget(self.spectra_plot)
        preview_splitter.addWidget(plot_card)

        preview_splitter.setSizes([300, 200])
        layout.addWidget(preview_splitter)

        return panel

    def _create_results_panel(self) -> QWidget:
        """Create the bottom results panel."""
        card = Card("Analysis Results", parent=self)
        layout = card.content_layout

        # Filter row
        filter_row = QHBoxLayout()
        filter_row.setSpacing(SPACING["sm"])

        filter_label = QLabel("Filter:")
        filter_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        filter_row.addWidget(filter_label)

        self.model_filter = StyledComboBox()
        self.model_filter.addItems(["All Models", "PLS", "Ridge", "ElasticNet", "RandomForest", "XGBoost", "LightGBM"])
        self.model_filter.currentIndexChanged.connect(self._apply_filters)
        filter_row.addWidget(self.model_filter)

        self.preprocess_filter = StyledComboBox()
        self.preprocess_filter.addItems(["All Preprocessing", "Raw", "SNV", "SG1", "SG2", "MSC"])
        self.preprocess_filter.currentIndexChanged.connect(self._apply_filters)
        filter_row.addWidget(self.preprocess_filter)

        score_label = QLabel("Score >")
        score_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        filter_row.addWidget(score_label)

        self.score_filter = StyledComboBox()
        self.score_filter.addItems(["Any", "50", "70", "80", "90"])
        self.score_filter.currentIndexChanged.connect(self._apply_filters)
        filter_row.addWidget(self.score_filter)

        filter_row.addStretch()

        self.results_count_label = QLabel("No results yet")
        self.results_count_label.setStyleSheet(f"color: {COLORS['text_tertiary']};")
        filter_row.addWidget(self.results_count_label)

        layout.addLayout(filter_row)

        # Results grid (reusing SpectralDataGrid for now, will be specialized later)
        self.results_grid = SpectralDataGrid()
        self.results_grid.setMinimumHeight(150)
        layout.addWidget(self.results_grid)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(SPACING["sm"])

        self.compare_btn = SecondaryButton("Compare Pinned (0)")
        self.compare_btn.setEnabled(False)
        self.compare_btn.clicked.connect(self._compare_pinned)
        btn_row.addWidget(self.compare_btn)

        self.export_btn = OutlineButton("Export CSV", Icons.download(14, COLORS["text_primary"]))
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_results)
        btn_row.addWidget(self.export_btn)

        btn_row.addStretch()

        self.build_btn = PrimaryButton("Build Selected Model", Icons.chevron_right(14, "#ffffff"))
        self.build_btn.setEnabled(False)
        self.build_btn.clicked.connect(self._go_to_build)
        btn_row.addWidget(self.build_btn)

        layout.addLayout(btn_row)

        return card

    def _populate_presets(self):
        """Populate the preset dropdown."""
        self.preset_combo.clear()
        self.preset_combo.addItem("Custom", None)
        for key, name in self.config.list_presets():
            self.preset_combo.addItem(name, key)
        idx = self.preset_combo.findText("NIR Protein (Grain)")
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)

    def _connect_signals(self):
        """Connect signals."""
        self.state.data_changed.connect(self._on_data_changed)
        self.state.analysis_completed.connect(self._on_analysis_completed)
        self.job_runner.job_progress.connect(self._on_job_progress)
        self.job_runner.job_completed.connect(self._on_job_completed)

    def _on_file_selected(self, file_path: str):
        """Handle file selection - auto-detects format and loads appropriately."""
        from pathlib import Path

        try:
            path = Path(file_path)

            # Determine if this is a tabular file that needs column config
            # or a spectral file that can be loaded directly
            tabular_extensions = {'.csv', '.xlsx', '.xls', '.txt'}
            spectral_extensions = {'.asd', '.spc', '.jdx', '.dx', '.jcm', '.sp', '.dat', '.dpt'}

            is_directory = path.is_dir()
            extension = path.suffix.lower() if not is_directory else ''

            if is_directory or extension in spectral_extensions:
                # Load spectral files directly - no column config needed
                self._load_spectral_file(file_path)
            elif extension in tabular_extensions:
                # Show column config dialog for CSV/Excel
                self._load_tabular_file(file_path)
            else:
                # Try to auto-detect and load
                self._load_spectral_file(file_path)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Data",
                f"Could not load file:\n\n{str(e)}\n\nSupported formats:\n"
                "• CSV, Excel (.csv, .xlsx, .xls)\n"
                "• ASD directory or files\n"
                "• SPC files\n"
                "• JCAMP-DX (.jdx, .dx)\n"
                "• ASCII spectra (.txt, .dat)"
            )
            self.file_drop.clear()

    def _load_spectral_file(self, file_path: str):
        """Load ASD, SPC, JCAMP, or other spectral files directly."""
        from pathlib import Path

        path = Path(file_path)

        # If directory, check what's inside and give helpful feedback
        if path.is_dir():
            files = list(path.iterdir())
            extensions = set(f.suffix.lower() for f in files if f.is_file())

            # Check for supported spectral file types
            asd_files = [f for f in files if f.suffix.lower() in ['.asd', '.sig']]
            spc_files = [f for f in files if f.suffix.lower() == '.spc']
            jcamp_files = [f for f in files if f.suffix.lower() in ['.jdx', '.dx']]

            if not (asd_files or spc_files or jcamp_files):
                raise ValueError(
                    f"No spectral files found in folder.\n\n"
                    f"Folder: {path.name}\n"
                    f"Files found: {len(files)}\n"
                    f"Extensions: {', '.join(sorted(extensions)) if extensions else 'none'}\n\n"
                    f"Supported formats:\n"
                    f"• ASD files (.asd, .sig)\n"
                    f"• SPC files (.spc)\n"
                    f"• JCAMP-DX files (.jdx, .dx)\n\n"
                    f"For CSV/Excel files, use File > Open File instead."
                )

        data = self.engine.load_data(file_path, for_prediction=False)

        self.state.load_data(
            X=data.X,
            y=data.y if data.y is not None else None,
            wavelengths=data.wavelengths,
            file_path=file_path,
            target_column=data.target_column or "",
            sample_ids=data.sample_ids
        )

        # Show info about loaded data
        n_samples = data.X.shape[0]
        n_wavelengths = data.X.shape[1]
        wl_range = data.metadata.get("wavelength_range", (0, 0))
        file_format = data.metadata.get("file_format", "unknown")

        # Update target combo - spectral files usually don't have targets
        self.target_combo.clear()
        self.target_combo.addItem("(Load reference file for targets)")
        self.target_combo.setEnabled(False)
        self.run_btn.setEnabled(False)

        # Update preview
        self._update_preview(data.X, data.wavelengths, data.y, data.sample_ids)

        # Show success message
        QMessageBox.information(
            self,
            "Data Loaded",
            f"Successfully loaded {file_format.upper()} data:\n\n"
            f"• {n_samples} samples\n"
            f"• {n_wavelengths} wavelengths ({wl_range[0]:.0f} - {wl_range[1]:.0f} nm)\n\n"
            f"To run analysis, load a reference file with target values\n"
            f"using the Target dropdown menu."
        )

    def _load_tabular_file(self, file_path: str):
        """Load CSV/Excel with column configuration dialog."""
        dialog = ColumnConfigDialog(file_path, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self.file_drop.clear()
            return

        config = dialog.get_config()
        if not config:
            self.file_drop.clear()
            return

        data = self.engine.load_data_with_config(
            file_path,
            id_column=config.get('id_column'),
            target_column=config.get('target_column'),
            metadata_columns=config.get('metadata_columns', [])
        )

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

        self.run_btn.setEnabled(data.y is not None)

        # Auto-set task type
        if data.metadata.get("is_classification", False):
            self.task_combo.setCurrentIndex(1)
            self.state.set_task_type(TaskType.CLASSIFICATION)
        else:
            self.task_combo.setCurrentIndex(0)
            self.state.set_task_type(TaskType.REGRESSION)

        # Update preview panels
        self._update_preview(data.X, data.wavelengths, data.y, data.sample_ids)

    def _update_preview(self, X, wavelengths, y, sample_ids):
        """Update the data preview panels."""
        # Update data grid
        self.data_grid.set_data(X, wavelengths, sample_ids)

        # Update spectra plot
        self.spectra_plot.clear()
        n_preview = min(50, X.shape[0])
        sample_names = [f"Sample {i+1}" for i in range(n_preview)]
        self.spectra_plot.set_data(X[:n_preview], wavelengths, sample_ids=sample_names)

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
        if preset.task_type == "classification":
            self.task_combo.setCurrentIndex(1)
        else:
            self.task_combo.setCurrentIndex(0)

        for key, cb in self.preprocess_checks.items():
            cb.setChecked(key in preset.preprocessing.methods)

        for key, cb in self.model_checks.items():
            cb.setChecked(key in preset.models.model_types)

        self.bayesian_check.setChecked(preset.models.use_bayesian)
        self.n_trials_spin.setValue(preset.models.n_trials)

    def _on_target_changed(self, index: int):
        """Handle target selection change."""
        pass

    def _on_task_changed(self, index: int):
        """Handle task type change."""
        task = TaskType.CLASSIFICATION if index == 1 else TaskType.REGRESSION
        self.state.set_task_type(task)
        self._update_model_checkboxes(task)

    def _update_model_checkboxes(self, task: TaskType):
        """Update model checkboxes based on task type."""
        while self.model_layout.count() > 0:
            item = self.model_layout.takeAt(0)
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
            cb = StyledCheckBox(name)
            cb.setChecked(name in defaults)
            key = name.lower().replace("-", "")
            self.model_checks[key] = cb
            self.model_layout.addWidget(cb)

        self.model_layout.addStretch()

    def _update_time_estimate(self, value: int):
        """Update the time estimate based on thoroughness."""
        estimates = ["~2 min", "~15 min", "~45 min", "~2+ hours"]
        self.time_estimate_label.setText(estimates[value])

    def _get_analysis_config(self) -> AnalysisConfig:
        """Build analysis config from UI settings."""
        task_type = "classification" if self.task_combo.currentIndex() == 1 else "regression"

        preprocessing = [k for k, cb in self.preprocess_checks.items() if cb.isChecked()]
        if not preprocessing:
            preprocessing = ["raw"]

        models = [k for k, cb in self.model_checks.items() if cb.isChecked()]
        if not models:
            if task_type == "classification":
                models = ["plsda", "randomforest", "svm"]
            else:
                models = ["pls", "ridge"]

        thoroughness = self.thoroughness_slider.value()
        n_trials = self.n_trials_spin.value()

        if thoroughness == 0:
            n_trials = min(n_trials, 20)
            models = models[:2]
        elif thoroughness == 2:
            n_trials = max(n_trials, 100)
        elif thoroughness == 3:
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

        config = self._get_analysis_config()

        # Show progress UI
        self.run_btn.setVisible(False)
        self.cancel_btn.setVisible(True)
        self.progress_container.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting analysis...")

        self.state.start_analysis()

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

        self.run_btn.setVisible(True)
        self.cancel_btn.setVisible(False)
        self.progress_container.setVisible(False)

        if result.success:
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
        self.progress_container.setVisible(False)

    def _on_analysis_completed(self):
        """Handle analysis completion from state."""
        results = self.state.analysis.results_df
        if results is not None and len(results) > 0:
            self._populate_results(results)

    def _populate_results(self, df):
        """Populate the results table."""
        self._results_df = df
        self._display_results(df)

    def _display_results(self, df):
        """Display results in the grid."""
        if df is None or len(df) == 0:
            self.results_count_label.setText("No results")
            return

        # Convert DataFrame to numpy for display in grid
        # For now, just show key columns
        display_cols = ["model", "preprocessing", "rmsecv", "r2", "n_variables", "composite_score"]
        available_cols = [c for c in display_cols if c in df.columns or c.replace("_", "") in df.columns]

        if available_cols:
            display_df = df[available_cols].head(100)
            values = display_df.values
            wavelengths = np.arange(len(available_cols))
            self.results_grid.model().set_column_names([c.replace("_", " ").title() for c in available_cols])

        self.results_count_label.setText(f"Showing {len(df)} models")

    def _apply_filters(self):
        """Apply filters to results."""
        if self._results_df is None:
            return

        df = self._results_df.copy()

        model_filter = self.model_filter.currentText()
        if model_filter != "All Models":
            model_col = "model" if "model" in df.columns else "model_type"
            if model_col in df.columns:
                df = df[df[model_col].str.lower().str.contains(model_filter.lower())]

        preprocess_filter = self.preprocess_filter.currentText()
        if preprocess_filter != "All Preprocessing":
            prep_col = "preprocessing" if "preprocessing" in df.columns else "preprocess"
            if prep_col in df.columns:
                df = df[df[prep_col].str.lower() == preprocess_filter.lower()]

        score_filter = self.score_filter.currentText()
        if score_filter != "Any":
            score_col = "composite_score" if "composite_score" in df.columns else "score"
            if score_col in df.columns:
                df = df[df[score_col] >= float(score_filter)]

        self._display_results(df)

    def _compare_pinned(self):
        """Show comparison of pinned models."""
        indices = self.state.pinned_indices
        if len(indices) < 2:
            return
        QMessageBox.information(
            self, "Compare Models",
            f"Comparing {len(indices)} models:\nIndices: {indices}\n\n(Full comparison view coming soon)"
        )

    def _export_results(self):
        """Export results to CSV."""
        if self._results_df is None:
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
