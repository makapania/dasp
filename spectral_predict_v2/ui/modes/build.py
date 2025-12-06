"""
Build Mode - Train and refine models.

Modern card-based layout with integrated diagnostics.
"""

from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QScrollArea, QSplitter, QMessageBox, QFileDialog
)
from PySide6.QtCore import Qt, QThread, Signal
import numpy as np

from orchestration.state_store import StateStore, AppMode, TaskType
from orchestration.job_runner import JobRunner

from ..theme.tokens import COLORS, SPACING, RADIUS, TYPOGRAPHY
from ..theme.icons import Icons
from ..components.cards import Card, CollapsibleCard, StatusCard
from ..components.buttons import PrimaryButton, SecondaryButton, OutlineButton, GhostButton
from ..components.inputs import (
    StyledComboBox, StyledSpinBox, StyledDoubleSpinBox, LabeledInput
)
from ..widgets.plot_widget import DiagnosticsPanel


class TrainingWorker(QThread):
    """Background worker for model training."""

    finished = Signal(object)
    progress = Signal(str)

    def __init__(self, engine, X, y, config, wavelengths=None, task_type='regression'):
        super().__init__()
        self.engine = engine
        self.X = X
        self.y = y
        self.config = config
        self.wavelengths = wavelengths
        self.task_type = task_type

    def run(self):
        try:
            from sklearn.model_selection import cross_val_predict

            self.progress.emit("Applying preprocessing...")

            X_proc = self.engine.apply_preprocessing(
                self.X,
                self.config['preprocessing']
            )

            self.progress.emit("Training model...")

            trained = self.engine.train_model(
                self.X,
                self.y,
                model_type=self.config['model_type'],
                preprocessing=self.config['preprocessing'],
                n_components=self.config.get('n_components', 5),
            )

            self.progress.emit("Computing cross-validation metrics...")

            from src.spectral_predict.models import get_model
            model = get_model(
                self.config['model_type'],
                task_type=self.task_type,
                n_components=self.config.get('n_components', 5)
            )

            y_pred_cv = cross_val_predict(model, X_proc, self.y, cv=5)

            # Compute metrics based on task type
            if self.task_type == 'classification':
                from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

                accuracy = accuracy_score(self.y, y_pred_cv)
                # Use weighted average for multiclass
                f1 = f1_score(self.y, y_pred_cv, average='weighted', zero_division=0)
                precision = precision_score(self.y, y_pred_cv, average='weighted', zero_division=0)
                recall = recall_score(self.y, y_pred_cv, average='weighted', zero_division=0)

                trained.metrics['accuracy'] = accuracy
                trained.metrics['f1'] = f1
                trained.metrics['precision'] = precision
                trained.metrics['recall'] = recall
                trained.metrics['task_type'] = 'classification'
            else:
                from sklearn.metrics import mean_squared_error, r2_score
                rmse_cv = np.sqrt(mean_squared_error(self.y, y_pred_cv))
                r2_cv = r2_score(self.y, y_pred_cv)
                bias = np.mean(y_pred_cv - self.y)
                rpd = np.std(self.y) / rmse_cv if rmse_cv > 0 else 0

                trained.metrics['rmsecv'] = rmse_cv
                trained.metrics['r2_cv'] = r2_cv
                trained.metrics['bias'] = bias
                trained.metrics['rpd'] = rpd
                trained.metrics['task_type'] = 'regression'

            trained.config['y_pred_cv'] = y_pred_cv
            trained.config['y_true'] = self.y
            trained.config['task_type'] = self.task_type

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

    Modern card-based layout:
    - Left panel: Model selection, configuration, training controls
    - Right panel: Diagnostics plots and metrics
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
        layout.setContentsMargins(SPACING["lg"], SPACING["md"], SPACING["lg"], SPACING["lg"])
        layout.setSpacing(SPACING["md"])

        # Main splitter: config left, diagnostics right
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setHandleWidth(SPACING["sm"])

        # Left panel: configuration
        left_panel = self._create_config_panel()
        left_panel.setMaximumWidth(450)
        main_splitter.addWidget(left_panel)

        # Right panel: diagnostics
        right_panel = self._create_diagnostics_panel()
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

        # MODEL SELECTION Card
        selection_card = Card("Model Selection", parent=self)
        selection_layout = selection_card.content_layout

        # From explore results
        from_label = QLabel("From Explore Results")
        from_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: {TYPOGRAPHY['size_sm']}pt;")
        selection_layout.addWidget(from_label)

        self.model_combo = StyledComboBox()
        self.model_combo.addItem("(No results - run Explore first)")
        self.model_combo.setEnabled(False)
        self.model_combo.currentIndexChanged.connect(self._on_model_selected)
        selection_layout.addWidget(self.model_combo)

        # Pinned indicator
        self.pinned_label = QLabel("")
        self.pinned_label.setStyleSheet(f"color: {COLORS['accent_secondary']}; font-size: {TYPOGRAPHY['size_xs']}pt;")
        selection_layout.addWidget(self.pinned_label)

        # Or start fresh
        divider_row = QHBoxLayout()
        divider_row.setSpacing(SPACING["sm"])
        line1 = QFrame()
        line1.setFrameShape(QFrame.Shape.HLine)
        line1.setStyleSheet(f"background-color: {COLORS['border_subtle']};")
        divider_row.addWidget(line1, 1)
        or_label = QLabel("or")
        or_label.setStyleSheet(f"color: {COLORS['text_tertiary']};")
        divider_row.addWidget(or_label)
        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.HLine)
        line2.setStyleSheet(f"background-color: {COLORS['border_subtle']};")
        divider_row.addWidget(line2, 1)
        selection_layout.addLayout(divider_row)

        self.fresh_btn = OutlineButton("Start Fresh Configuration", Icons.plus(14, COLORS["text_primary"]))
        self.fresh_btn.clicked.connect(self._start_fresh)
        selection_layout.addWidget(self.fresh_btn)

        layout.addWidget(selection_card)

        # CONFIGURATION Card
        config_card = Card("Configuration", parent=self)
        config_layout = config_card.content_layout

        # Preprocessing
        preprocess_container = QWidget()
        preprocess_inner = QVBoxLayout(preprocess_container)
        preprocess_inner.setContentsMargins(0, 0, 0, 0)
        preprocess_inner.setSpacing(SPACING["xs"])
        preprocess_label = QLabel("Preprocessing")
        preprocess_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: {TYPOGRAPHY['size_sm']}pt;")
        preprocess_inner.addWidget(preprocess_label)
        self.preprocess_combo = StyledComboBox()
        self.preprocess_combo.addItems([
            "Raw", "SNV", "SG1 (1st Derivative)", "SG2 (2nd Derivative)", "MSC"
        ])
        self.preprocess_combo.setCurrentIndex(1)
        preprocess_inner.addWidget(self.preprocess_combo)
        config_layout.addWidget(preprocess_container)

        # Model type
        model_container = QWidget()
        model_inner = QVBoxLayout(model_container)
        model_inner.setContentsMargins(0, 0, 0, 0)
        model_inner.setSpacing(SPACING["xs"])
        model_label = QLabel("Model Type")
        model_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: {TYPOGRAPHY['size_sm']}pt;")
        model_inner.addWidget(model_label)
        self.model_type_combo = StyledComboBox()
        self.model_type_combo.addItems([
            "PLS", "Ridge", "Lasso", "ElasticNet",
            "RandomForest", "XGBoost", "SVR", "MLP"
        ])
        self.model_type_combo.currentTextChanged.connect(self._on_model_type_changed)
        model_inner.addWidget(self.model_type_combo)
        config_layout.addWidget(model_container)

        # Components and CV folds row
        params_row = QHBoxLayout()
        params_row.setSpacing(SPACING["sm"])

        # Components
        comp_container = QWidget()
        comp_inner = QVBoxLayout(comp_container)
        comp_inner.setContentsMargins(0, 0, 0, 0)
        comp_inner.setSpacing(SPACING["xs"])
        self.components_label = QLabel("Components")
        self.components_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: {TYPOGRAPHY['size_sm']}pt;")
        comp_inner.addWidget(self.components_label)
        self.n_components = StyledSpinBox()
        self.n_components.setRange(1, 50)
        self.n_components.setValue(5)
        comp_inner.addWidget(self.n_components)
        params_row.addWidget(comp_container, 1)

        # CV Folds
        cv_container = QWidget()
        cv_inner = QVBoxLayout(cv_container)
        cv_inner.setContentsMargins(0, 0, 0, 0)
        cv_inner.setSpacing(SPACING["xs"])
        cv_label = QLabel("CV Folds")
        cv_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: {TYPOGRAPHY['size_sm']}pt;")
        cv_inner.addWidget(cv_label)
        self.cv_folds = StyledSpinBox()
        self.cv_folds.setRange(2, 10)
        self.cv_folds.setValue(5)
        cv_inner.addWidget(self.cv_folds)
        params_row.addWidget(cv_container, 1)

        config_layout.addLayout(params_row)

        # Variables selection
        var_container = QWidget()
        var_inner = QVBoxLayout(var_container)
        var_inner.setContentsMargins(0, 0, 0, 0)
        var_inner.setSpacing(SPACING["xs"])
        var_label = QLabel("Variable Selection")
        var_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: {TYPOGRAPHY['size_sm']}pt;")
        var_inner.addWidget(var_label)
        self.variables_combo = StyledComboBox()
        self.variables_combo.addItems(["All Variables", "Top 10", "Top 20", "Top 50"])
        var_inner.addWidget(self.variables_combo)
        config_layout.addWidget(var_container)

        layout.addWidget(config_card)

        # ADVANCED Card (collapsible)
        self.advanced_card = CollapsibleCard("Advanced Options", start_collapsed=True, parent=self)
        adv_layout = self.advanced_card.content_layout

        # Alpha/Regularization
        alpha_row = QHBoxLayout()
        alpha_row.setSpacing(SPACING["sm"])
        alpha_label = QLabel("Alpha:")
        alpha_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        alpha_row.addWidget(alpha_label)
        self.alpha_spin = StyledDoubleSpinBox()
        self.alpha_spin.setRange(0.0001, 100.0)
        self.alpha_spin.setValue(1.0)
        self.alpha_spin.setDecimals(4)
        alpha_row.addWidget(self.alpha_spin)
        alpha_row.addStretch()
        adv_layout.addLayout(alpha_row)

        # L1 Ratio
        l1_row = QHBoxLayout()
        l1_row.setSpacing(SPACING["sm"])
        l1_label = QLabel("L1 Ratio:")
        l1_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        l1_row.addWidget(l1_label)
        self.l1_ratio_spin = StyledDoubleSpinBox()
        self.l1_ratio_spin.setRange(0.0, 1.0)
        self.l1_ratio_spin.setValue(0.5)
        self.l1_ratio_spin.setDecimals(2)
        l1_row.addWidget(self.l1_ratio_spin)
        l1_row.addStretch()
        adv_layout.addLayout(l1_row)

        layout.addWidget(self.advanced_card)

        # TRAIN Card
        train_card = Card(parent=self)
        train_layout = train_card.content_layout

        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {COLORS['text_tertiary']}; font-size: {TYPOGRAPHY['size_sm']}pt;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        train_layout.addWidget(self.status_label)

        # Train button
        self.train_btn = PrimaryButton("Train Model", Icons.play(16, "#ffffff"))
        self.train_btn.setMinimumHeight(48)
        self.train_btn.clicked.connect(self._train_model)
        train_layout.addWidget(self.train_btn)

        layout.addWidget(train_card)

        layout.addStretch()
        scroll.setWidget(panel)
        return scroll

    def _create_diagnostics_panel(self) -> QWidget:
        """Create the right diagnostics panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING["md"])

        # Diagnostics Card with plots
        diag_card = Card("Model Diagnostics", parent=self)
        diag_layout = diag_card.content_layout

        self.diagnostics_panel = DiagnosticsPanel()
        self.diagnostics_panel.setMinimumHeight(400)
        diag_layout.addWidget(self.diagnostics_panel)

        layout.addWidget(diag_card, 1)

        # Metrics Card
        metrics_card = Card("Performance Metrics", parent=self)
        metrics_layout = metrics_card.content_layout

        # Metrics grid
        metrics_grid = QHBoxLayout()
        metrics_grid.setSpacing(SPACING["lg"])

        # Metric 1: RMSECV (regression) / Accuracy (classification)
        self.metric1_container = self._create_metric_display("RMSECV", "--")
        self.metric1_label = self.metric1_container.findChild(QLabel, "label")
        self.rmsecv_value = self.metric1_container.findChild(QLabel, "value")
        metrics_grid.addWidget(self.metric1_container, 1)

        # Metric 2: R2 (regression) / F1 (classification)
        self.metric2_container = self._create_metric_display("R\u00b2 (CV)", "--")
        self.metric2_label = self.metric2_container.findChild(QLabel, "label")
        self.r2_value = self.metric2_container.findChild(QLabel, "value")
        metrics_grid.addWidget(self.metric2_container, 1)

        # Metric 3: RPD (regression) / Precision (classification)
        self.metric3_container = self._create_metric_display("RPD", "--")
        self.metric3_label = self.metric3_container.findChild(QLabel, "label")
        self.rpd_value = self.metric3_container.findChild(QLabel, "value")
        metrics_grid.addWidget(self.metric3_container, 1)

        # Metric 4: Bias (regression) / Recall (classification)
        self.metric4_container = self._create_metric_display("Bias", "--")
        self.metric4_label = self.metric4_container.findChild(QLabel, "label")
        self.bias_value = self.metric4_container.findChild(QLabel, "value")
        metrics_grid.addWidget(self.metric4_container, 1)

        metrics_layout.addLayout(metrics_grid)
        layout.addWidget(metrics_card)

        # Actions Card
        actions_card = Card(parent=self)
        actions_layout = actions_card.content_layout

        btn_row = QHBoxLayout()
        btn_row.setSpacing(SPACING["sm"])

        self.save_btn = PrimaryButton("Save Model", Icons.save(14, "#ffffff"))
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_model)
        btn_row.addWidget(self.save_btn)

        self.export_btn = OutlineButton("Export Report", Icons.download(14, COLORS["text_primary"]))
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_report)
        btn_row.addWidget(self.export_btn)

        btn_row.addStretch()

        self.predict_btn = SecondaryButton("Go to Predict", Icons.chevron_right(14, "#ffffff"))
        self.predict_btn.clicked.connect(self._go_to_predict)
        btn_row.addWidget(self.predict_btn)

        actions_layout.addLayout(btn_row)
        layout.addWidget(actions_card)

        return panel

    def _create_metric_display(self, label: str, value: str) -> QWidget:
        """Create a metric display widget."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(SPACING["md"], SPACING["sm"], SPACING["md"], SPACING["sm"])
        layout.setSpacing(SPACING["xs"])
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        label_widget = QLabel(label)
        label_widget.setObjectName("label")
        label_widget.setStyleSheet(f"""
            color: {COLORS['text_tertiary']};
            font-size: {TYPOGRAPHY['size_sm']}pt;
        """)
        label_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label_widget)

        value_widget = QLabel(value)
        value_widget.setObjectName("value")
        value_widget.setStyleSheet(f"""
            color: {COLORS['text_primary']};
            font-size: {TYPOGRAPHY['size_xl']}pt;
            font-weight: {TYPOGRAPHY['weight_semibold']};
        """)
        value_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(value_widget)

        container.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['bg_elevated']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: {RADIUS['md']}px;
            }}
        """)

        return container

    def _connect_signals(self):
        """Connect signals."""
        self.state.analysis_completed.connect(self._on_results_available)
        self.state.model_changed.connect(self._on_model_changed)
        self.state.data_changed.connect(self._on_data_changed)

    def _on_data_changed(self):
        """Handle data changes."""
        self._current_trained_model = None
        self.diagnostics_panel.clear_all()
        self._update_metrics(None)
        self.save_btn.setEnabled(False)
        self.export_btn.setEnabled(False)

    def _on_results_available(self):
        """Handle when analysis results are available from Explore."""
        if self.state.analysis.results_df is not None:
            self.model_combo.clear()
            df = self.state.analysis.results_df

            pinned = self.state.pinned_indices
            if pinned:
                self.pinned_label.setText(f"Pinned: {len(pinned)} model(s)")
                for idx in pinned:
                    if idx < len(df):
                        row = df.iloc[idx]
                        label = f"[Pinned] {row.get('model', '?')} + {row.get('preprocessing', '?')} (R\u00b2: {row.get('r2_cv', row.get('r2', 0)):.3f})"
                        self.model_combo.addItem(label, idx)
                self.model_combo.insertSeparator(len(pinned))
            else:
                self.pinned_label.setText("")

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

        row_idx = self.model_combo.itemData(index)
        if row_idx is None:
            return

        df = self.state.analysis.results_df
        if row_idx >= len(df):
            return

        row = df.iloc[row_idx]

        preprocess = str(row.get('preprocessing', 'raw')).lower()
        preprocess_map = {'raw': 0, 'none': 0, 'snv': 1, 'sg1': 2, 'sg2': 3, 'msc': 4}
        preprocess_idx = preprocess_map.get(preprocess, 0)
        if '1' in preprocess:
            preprocess_idx = 2
        elif '2' in preprocess:
            preprocess_idx = 3
        self.preprocess_combo.setCurrentIndex(preprocess_idx)

        model_name = str(row.get('model', 'PLS')).upper()
        for i in range(self.model_type_combo.count()):
            if self.model_type_combo.itemText(i).upper() == model_name:
                self.model_type_combo.setCurrentIndex(i)
                break

        if 'n_components' in row:
            self.n_components.setValue(int(row['n_components']))

    def _on_model_type_changed(self, model_type: str):
        """Update UI based on selected model type."""
        component_models = ['PLS', 'Ridge', 'Lasso', 'ElasticNet', 'SVR']
        show_components = model_type in component_models
        self.n_components.setVisible(show_components)
        self.components_label.setVisible(show_components)

        if model_type == 'PLS':
            self.components_label.setText("Components")
        else:
            self.components_label.setText("Max Iterations")

    def _on_model_changed(self):
        """Handle when a model is set in state."""
        if self.state.model.selected_model:
            self._update_metrics(self.state.model.metrics)
            self.save_btn.setEnabled(True)
            self.export_btn.setEnabled(True)

    def _update_metrics(self, metrics):
        """Update the metrics display based on task type."""
        if metrics is None:
            self.rmsecv_value.setText("--")
            self.r2_value.setText("--")
            self.rpd_value.setText("--")
            self.bias_value.setText("--")
            # Reset labels to regression defaults
            self._update_metric_labels(is_classification=False)
            for widget in [self.rmsecv_value, self.r2_value, self.rpd_value, self.bias_value]:
                widget.setStyleSheet(f"""
                    color: {COLORS['text_tertiary']};
                    font-size: {TYPOGRAPHY['size_xl']}pt;
                    font-weight: {TYPOGRAPHY['weight_semibold']};
                """)
        else:
            # Check if classification or regression
            is_classification = metrics.get('task_type') == 'classification'

            if is_classification:
                # Update labels for classification metrics
                self._update_metric_labels(is_classification=True)
                self.rmsecv_value.setText(f"{metrics.get('accuracy', 0)*100:.1f}%")
                self.r2_value.setText(f"{metrics.get('f1', 0):.4f}")
                self.rpd_value.setText(f"{metrics.get('precision', 0):.4f}")
                self.bias_value.setText(f"{metrics.get('recall', 0):.4f}")
            else:
                # Regression metrics
                self._update_metric_labels(is_classification=False)
                self.rmsecv_value.setText(f"{metrics.get('rmsecv', 0):.4f}")
                self.r2_value.setText(f"{metrics.get('r2_cv', 0):.4f}")
                self.rpd_value.setText(f"{metrics.get('rpd', 0):.2f}")
                self.bias_value.setText(f"{metrics.get('bias', 0):.4f}")

            for widget in [self.rmsecv_value, self.r2_value, self.rpd_value, self.bias_value]:
                widget.setStyleSheet(f"""
                    color: {COLORS['accent_primary']};
                    font-size: {TYPOGRAPHY['size_xl']}pt;
                    font-weight: {TYPOGRAPHY['weight_semibold']};
                """)

    def _update_metric_labels(self, is_classification: bool):
        """Update the metric labels based on task type."""
        if is_classification:
            self.metric1_label.setText("Accuracy")
            self.metric2_label.setText("F1 Score")
            self.metric3_label.setText("Precision")
            self.metric4_label.setText("Recall")
        else:
            self.metric1_label.setText("RMSECV")
            self.metric2_label.setText("R\u00b2 (CV)")
            self.metric3_label.setText("RPD")
            self.metric4_label.setText("Bias")

    def _start_fresh(self):
        """Start with a fresh model configuration."""
        if not self.state.has_data:
            QMessageBox.warning(self, "No Data", "Please load data first in Explore mode.")
            return

        self.preprocess_combo.setCurrentIndex(1)
        self.model_type_combo.setCurrentIndex(0)
        self.n_components.setValue(5)
        self.variables_combo.setCurrentIndex(0)

        self.diagnostics_panel.clear_all()
        self._current_trained_model = None
        self._update_metrics(None)
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

        preprocess_text = self.preprocess_combo.currentText()
        preprocess_map = {
            'Raw': 'raw', 'SNV': 'snv', 'SG1 (1st Derivative)': 'sg1',
            'SG2 (2nd Derivative)': 'sg2', 'MSC': 'msc'
        }
        preprocessing = preprocess_map.get(preprocess_text, 'raw')

        model_type = self.model_type_combo.currentText().lower()
        n_components = self.n_components.value()

        config = {
            'preprocessing': preprocessing,
            'model_type': model_type,
            'n_components': n_components,
        }

        self.train_btn.setEnabled(False)
        self.status_label.setText("Training...")
        self.status_label.setStyleSheet(f"color: {COLORS['accent_secondary']}; font-size: {TYPOGRAPHY['size_sm']}pt;")

        # Determine task type from state
        task_type = 'classification' if self.state.task_type == TaskType.CLASSIFICATION else 'regression'

        self._training_worker = TrainingWorker(
            engine=self._get_engine(),
            X=self.state.data.X,
            y=self.state.data.y,
            config=config,
            wavelengths=self.state.data.wavelengths,
            task_type=task_type
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
            self.status_label.setStyleSheet(f"color: {COLORS['accent_danger']}; font-size: {TYPOGRAPHY['size_sm']}pt;")
            QMessageBox.critical(self, "Training Failed", str(result))
            return

        self._current_trained_model = result
        self.status_label.setText("Training complete!")
        self.status_label.setStyleSheet(f"color: {COLORS['accent_primary']}; font-size: {TYPOGRAPHY['size_sm']}pt;")

        self.state.set_model(
            model=result.model,
            name=result.name,
            config=result.config,
            preprocessing=result.preprocessing,
            variable_mask=result.variable_mask,
            metrics=result.metrics
        )

        y_true = result.config.get('y_true')
        y_pred = result.config.get('y_pred_cv')
        wavelengths = result.config.get('wavelengths')
        loadings = result.config.get('loadings')
        coefficients = result.config.get('coefficients')

        # Update diagnostic plots
        if y_true is not None and y_pred is not None:
            sample_ids = self.state.data.sample_ids or [f"Sample {i+1}" for i in range(len(y_true))]
            self.diagnostics_panel.set_cv_results(y_true, y_pred, sample_ids)

        # Set loadings or coefficients
        if loadings is not None and wavelengths is not None:
            self.diagnostics_panel.set_loadings(loadings, wavelengths)
        elif coefficients is not None and wavelengths is not None:
            self.diagnostics_panel.get_loadings_plot().set_coefficients(coefficients, wavelengths)

        # Set spectra
        if wavelengths is not None and self.state.has_data:
            # Show first 10 spectra
            n_show = min(10, self.state.data.X.shape[0])
            sample_ids = self.state.data.sample_ids[:n_show] if self.state.data.sample_ids else [f"Sample {i+1}" for i in range(n_show)]
            self.diagnostics_panel.set_spectra(self.state.data.X[:n_show], wavelengths, sample_ids)

        self._update_metrics(result.metrics)
        self.save_btn.setEnabled(True)
        self.export_btn.setEnabled(True)

    def _save_model(self):
        """Save the trained model."""
        if self._current_trained_model is None:
            QMessageBox.warning(self, "No Model", "Please train a model first.")
            return

        default_name = f"{self.state.data.target_column}_{self._current_trained_model.name}.dasp"

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Model", default_name,
            "DASP Model Files (*.dasp);;All Files (*)"
        )

        if file_path:
            try:
                engine = self._get_engine()
                engine.save_model(self._current_trained_model, file_path)
                self.state.mark_model_saved(file_path)
                QMessageBox.information(self, "Model Saved", f"Model saved successfully to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Failed", str(e))

    def _export_report(self):
        """Export a model report."""
        if self._current_trained_model is None:
            QMessageBox.warning(self, "No Model", "Please train a model first.")
            return

        default_name = f"{self.state.data.target_column}_{self._current_trained_model.name}_report.md"

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Report", default_name,
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
- **R\u00b2 (CV):** {metrics.get('r2_cv', 0):.4f}
- **RPD:** {metrics.get('rpd', 0):.2f}
- **Bias:** {metrics.get('bias', 0):.4f}

## Calibration Metrics
- **RMSE (Cal):** {metrics.get('rmse', 0):.4f}
- **R\u00b2 (Cal):** {metrics.get('r2', 0):.4f}

---
*Generated by Spectral Predict 2.0*
"""
                Path(file_path).write_text(report)
                QMessageBox.information(self, "Report Exported", f"Report exported to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", str(e))

    def _go_to_predict(self):
        """Switch to Predict mode."""
        self.state.set_mode(AppMode.PREDICT)
