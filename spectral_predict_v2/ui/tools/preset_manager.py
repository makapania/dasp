"""
Preset Manager - Save, load, and manage custom presets.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QLineEdit, QTextEdit, QFrame,
    QListWidget, QListWidgetItem, QMessageBox, QInputDialog,
    QFormLayout, QCheckBox, QSpinBox
)
from PySide6.QtCore import Qt, Signal

from orchestration.config_manager import (
    ConfigManager, AnalysisPreset, PreprocessingConfig, ModelConfig,
    VariableSelectionConfig, ValidationConfig
)


class PresetManagerTool(QWidget):
    """
    Tool panel for managing analysis presets.

    Features:
    - View all built-in and custom presets
    - Create new presets from current settings
    - Edit custom presets
    - Delete custom presets
    - Export/import presets
    """

    preset_selected = Signal(str)  # Emits preset key when selected

    def __init__(self, config_manager: ConfigManager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self._current_preset_key = None

        self._setup_ui()
        self._load_preset_list()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Left side - preset list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        list_label = QLabel("Available Presets")
        list_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        left_layout.addWidget(list_label)

        self.preset_list = QListWidget()
        self.preset_list.currentRowChanged.connect(self._on_preset_selected)
        left_layout.addWidget(self.preset_list)

        # List buttons
        list_btn_row = QHBoxLayout()

        self.new_btn = QPushButton("New")
        self.new_btn.clicked.connect(self._create_new_preset)
        list_btn_row.addWidget(self.new_btn)

        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setEnabled(False)
        self.delete_btn.clicked.connect(self._delete_preset)
        list_btn_row.addWidget(self.delete_btn)

        list_btn_row.addStretch()
        left_layout.addLayout(list_btn_row)

        layout.addWidget(left_panel)

        # Right side - preset details/editor
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        detail_label = QLabel("Preset Details")
        detail_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        right_layout.addWidget(detail_label)

        # Details form
        form_widget = QWidget()
        form_layout = QFormLayout(form_widget)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Preset name")
        form_layout.addRow("Name:", self.name_edit)

        self.desc_edit = QTextEdit()
        self.desc_edit.setMaximumHeight(60)
        self.desc_edit.setPlaceholderText("Description...")
        form_layout.addRow("Description:", self.desc_edit)

        self.task_combo = QComboBox()
        self.task_combo.addItems(["Regression", "Classification"])
        form_layout.addRow("Task Type:", self.task_combo)

        right_layout.addWidget(form_widget)

        # Preprocessing section
        preproc_group = QGroupBox("Preprocessing")
        preproc_layout = QVBoxLayout(preproc_group)

        self.preproc_checks = {}
        preproc_row = QHBoxLayout()
        for method in ["raw", "snv", "sg1", "sg2", "msc", "baseline"]:
            cb = QCheckBox(method.upper())
            self.preproc_checks[method] = cb
            preproc_row.addWidget(cb)
        preproc_row.addStretch()
        preproc_layout.addLayout(preproc_row)

        right_layout.addWidget(preproc_group)

        # Models section
        models_group = QGroupBox("Models")
        models_layout = QVBoxLayout(models_group)

        self.model_checks = {}
        model_row1 = QHBoxLayout()
        for model in ["pls", "ridge", "lasso", "elasticnet"]:
            cb = QCheckBox(model.upper())
            self.model_checks[model] = cb
            model_row1.addWidget(cb)
        model_row1.addStretch()
        models_layout.addLayout(model_row1)

        model_row2 = QHBoxLayout()
        for model in ["randomforest", "xgboost", "lightgbm", "svr"]:
            cb = QCheckBox(model.upper() if model != "randomforest" else "RandomForest")
            self.model_checks[model] = cb
            model_row2.addWidget(cb)
        model_row2.addStretch()
        models_layout.addLayout(model_row2)

        tier_row = QHBoxLayout()
        tier_row.addWidget(QLabel("Tier:"))
        self.tier_combo = QComboBox()
        self.tier_combo.addItems(["Quick", "Standard", "Comprehensive", "Experimental"])
        tier_row.addWidget(self.tier_combo)
        tier_row.addSpacing(20)

        self.bayesian_check = QCheckBox("Use Bayesian Optimization")
        self.bayesian_check.setChecked(True)
        tier_row.addWidget(self.bayesian_check)

        tier_row.addStretch()
        models_layout.addLayout(tier_row)

        right_layout.addWidget(models_group)

        # Variable selection section
        varsel_group = QGroupBox("Variable Selection")
        varsel_layout = QHBoxLayout(varsel_group)

        self.varsel_enabled = QCheckBox("Enabled")
        varsel_layout.addWidget(self.varsel_enabled)

        varsel_layout.addWidget(QLabel("Methods:"))
        self.varsel_checks = {}
        for method in ["uve", "spa", "ipls"]:
            cb = QCheckBox(method.upper())
            self.varsel_checks[method] = cb
            varsel_layout.addWidget(cb)

        varsel_layout.addStretch()
        right_layout.addWidget(varsel_group)

        # Validation section
        val_row = QHBoxLayout()
        val_row.addWidget(QLabel("CV Folds:"))
        self.folds_spin = QSpinBox()
        self.folds_spin.setRange(2, 10)
        self.folds_spin.setValue(5)
        val_row.addWidget(self.folds_spin)
        val_row.addStretch()
        right_layout.addLayout(val_row)

        right_layout.addStretch()

        # Save button
        save_row = QHBoxLayout()
        save_row.addStretch()

        self.save_btn = QPushButton("Save Preset")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_preset)
        save_row.addWidget(self.save_btn)

        self.apply_btn = QPushButton("Apply to Current Analysis")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply_preset)
        save_row.addWidget(self.apply_btn)

        right_layout.addLayout(save_row)

        layout.addWidget(right_panel, 1)

    def _load_preset_list(self):
        """Load all presets into the list widget."""
        self.preset_list.clear()

        # Add section header for built-in presets
        builtin_header = QListWidgetItem("--- Built-in Presets ---")
        builtin_header.setFlags(Qt.ItemFlag.NoItemFlags)
        builtin_header.setForeground(Qt.GlobalColor.gray)
        self.preset_list.addItem(builtin_header)

        user_presets = []
        builtin_presets = []

        for key, name in self.config_manager.list_presets():
            # Check if it's a user preset
            user_file = self.config_manager.user_dir / f"{key}.yaml"
            if user_file.exists():
                user_presets.append((key, name))
            else:
                builtin_presets.append((key, name))

        # Add built-in presets
        for key, name in builtin_presets:
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setData(Qt.ItemDataRole.UserRole + 1, False)  # Not editable
            self.preset_list.addItem(item)

        # Add section header for user presets
        if user_presets:
            user_header = QListWidgetItem("--- Custom Presets ---")
            user_header.setFlags(Qt.ItemFlag.NoItemFlags)
            user_header.setForeground(Qt.GlobalColor.gray)
            self.preset_list.addItem(user_header)

            for key, name in user_presets:
                item = QListWidgetItem(f"* {name}")
                item.setData(Qt.ItemDataRole.UserRole, key)
                item.setData(Qt.ItemDataRole.UserRole + 1, True)  # Editable
                self.preset_list.addItem(item)

    def _on_preset_selected(self, row):
        """Handle preset selection."""
        item = self.preset_list.item(row)
        if item is None:
            return

        key = item.data(Qt.ItemDataRole.UserRole)
        if key is None:
            return

        is_editable = item.data(Qt.ItemDataRole.UserRole + 1)

        self._current_preset_key = key
        preset = self.config_manager.get_preset(key)

        if preset:
            self._populate_form(preset)

        # Enable/disable buttons
        self.delete_btn.setEnabled(is_editable)
        self.save_btn.setEnabled(True)
        self.apply_btn.setEnabled(True)

    def _populate_form(self, preset: AnalysisPreset):
        """Populate the form with preset values."""
        self.name_edit.setText(preset.name)
        self.desc_edit.setPlainText(preset.description)
        self.task_combo.setCurrentIndex(0 if preset.task_type == "regression" else 1)

        # Preprocessing
        for method, cb in self.preproc_checks.items():
            cb.setChecked(method in preset.preprocessing.methods)

        # Models
        for model, cb in self.model_checks.items():
            cb.setChecked(model in preset.models.model_types)

        tier_map = {"quick": 0, "standard": 1, "comprehensive": 2, "experimental": 3}
        self.tier_combo.setCurrentIndex(tier_map.get(preset.models.tier, 1))
        self.bayesian_check.setChecked(preset.models.use_bayesian)

        # Variable selection
        self.varsel_enabled.setChecked(preset.variable_selection.enabled)
        for method, cb in self.varsel_checks.items():
            cb.setChecked(method in preset.variable_selection.methods)

        # Validation
        self.folds_spin.setValue(preset.validation.n_folds)

    def _create_new_preset(self):
        """Create a new custom preset."""
        name, ok = QInputDialog.getText(
            self, "New Preset",
            "Enter preset name:",
            QLineEdit.EchoMode.Normal,
            "My Custom Preset"
        )

        if ok and name:
            # Generate key from name
            key = name.lower().replace(" ", "_").replace("-", "_")

            # Check if exists
            if self.config_manager.get_preset(key):
                QMessageBox.warning(
                    self, "Exists",
                    f"A preset with key '{key}' already exists."
                )
                return

            # Create new preset with defaults
            preset = AnalysisPreset(name=name)
            self.config_manager.save_preset(key, preset)

            # Reload list and select new item
            self._load_preset_list()

            # Find and select the new preset
            for i in range(self.preset_list.count()):
                item = self.preset_list.item(i)
                if item and item.data(Qt.ItemDataRole.UserRole) == key:
                    self.preset_list.setCurrentRow(i)
                    break

    def _delete_preset(self):
        """Delete the selected custom preset."""
        if self._current_preset_key is None:
            return

        reply = QMessageBox.question(
            self, "Delete Preset",
            f"Delete preset '{self._current_preset_key}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            if self.config_manager.delete_preset(self._current_preset_key):
                self._load_preset_list()
                self._current_preset_key = None
                self.delete_btn.setEnabled(False)
                self.save_btn.setEnabled(False)
                self.apply_btn.setEnabled(False)

    def _save_preset(self):
        """Save the current form values to the preset."""
        if self._current_preset_key is None:
            # Create new preset
            self._create_new_preset()
            return

        # Check if it's a built-in preset - need to save as new
        item = self.preset_list.currentItem()
        is_editable = item.data(Qt.ItemDataRole.UserRole + 1) if item else False

        if not is_editable:
            # Prompt for new name
            name, ok = QInputDialog.getText(
                self, "Save As",
                "Built-in presets cannot be modified.\nEnter a name for the new preset:",
                QLineEdit.EchoMode.Normal,
                self.name_edit.text() + " (Custom)"
            )
            if not ok or not name:
                return

            key = name.lower().replace(" ", "_").replace("-", "_")
        else:
            name = self.name_edit.text()
            key = self._current_preset_key

        # Build preset from form
        preproc_methods = [m for m, cb in self.preproc_checks.items() if cb.isChecked()]
        model_types = [m for m, cb in self.model_checks.items() if cb.isChecked()]
        varsel_methods = [m for m, cb in self.varsel_checks.items() if cb.isChecked()]

        tier_map = {0: "quick", 1: "standard", 2: "comprehensive", 3: "experimental"}

        preset = AnalysisPreset(
            name=name,
            description=self.desc_edit.toPlainText(),
            task_type="regression" if self.task_combo.currentIndex() == 0 else "classification",
            preprocessing=PreprocessingConfig(
                methods=preproc_methods if preproc_methods else ["raw"],
            ),
            models=ModelConfig(
                model_types=model_types if model_types else ["pls"],
                tier=tier_map.get(self.tier_combo.currentIndex(), "standard"),
                use_bayesian=self.bayesian_check.isChecked(),
            ),
            variable_selection=VariableSelectionConfig(
                enabled=self.varsel_enabled.isChecked(),
                methods=varsel_methods,
            ),
            validation=ValidationConfig(
                n_folds=self.folds_spin.value(),
            ),
        )

        self.config_manager.save_preset(key, preset)
        self._load_preset_list()

        QMessageBox.information(self, "Saved", f"Preset '{name}' saved successfully.")

    def _apply_preset(self):
        """Apply the selected preset to the current analysis."""
        if self._current_preset_key:
            self.preset_selected.emit(self._current_preset_key)
