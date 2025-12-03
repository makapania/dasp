"""
Column Configuration Dialog - Lets user specify column roles when loading data.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QListWidget, QListWidgetItem, QPushButton, QGroupBox,
    QMessageBox, QAbstractItemView
)
from PySide6.QtCore import Qt
import pandas as pd
from pathlib import Path


class ColumnConfigDialog(QDialog):
    """
    Dialog for configuring column roles when loading spectral data.

    Allows user to specify:
    - Sample ID column (row identifier)
    - Target column (y variable for prediction)
    - Metadata columns (to exclude from spectra)
    - Wavelength columns are auto-detected as remaining numeric columns
    """

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.df = None
        self.result = None

        self.setWindowTitle("Configure Columns")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)

        self._load_preview()
        self._setup_ui()

    def _load_preview(self):
        """Load first few rows to detect columns."""
        path = Path(self.file_path)
        try:
            if path.suffix.lower() in ['.xlsx', '.xls']:
                self.df = pd.read_excel(self.file_path, nrows=10)
            else:
                self.df = pd.read_csv(self.file_path, nrows=10)
        except Exception as e:
            raise ValueError(f"Could not read file: {e}")

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # File info
        file_label = QLabel(f"<b>File:</b> {Path(self.file_path).name}")
        layout.addWidget(file_label)

        cols_label = QLabel(f"<b>Columns detected:</b> {len(self.df.columns)}")
        layout.addWidget(cols_label)

        # Analyze columns
        self.numeric_cols = []
        self.string_cols = []
        self.wavelength_cols = []

        for col in self.df.columns:
            col_str = str(col)
            # Try to parse as wavelength (numeric)
            try:
                wl = float(col_str)
                if 100 < wl < 25000:  # Reasonable wavelength range
                    self.wavelength_cols.append(col)
                else:
                    self.numeric_cols.append(col)
            except ValueError:
                # Non-numeric column name
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.numeric_cols.append(col)
                else:
                    self.string_cols.append(col)

        # Show detected wavelengths
        if self.wavelength_cols:
            wl_min = min(float(str(c)) for c in self.wavelength_cols)
            wl_max = max(float(str(c)) for c in self.wavelength_cols)
            wl_label = QLabel(f"<b>Wavelengths detected:</b> {len(self.wavelength_cols)} columns ({wl_min:.0f} - {wl_max:.0f} nm)")
            wl_label.setStyleSheet("color: green;")
            layout.addWidget(wl_label)
        else:
            wl_label = QLabel("<b>Warning:</b> No wavelength columns detected")
            wl_label.setStyleSheet("color: orange;")
            layout.addWidget(wl_label)

        # Sample ID selection
        id_group = QGroupBox("Sample ID Column")
        id_layout = QHBoxLayout(id_group)
        id_layout.addWidget(QLabel("Use as row identifier:"))
        self.id_combo = QComboBox()
        self.id_combo.addItem("(Auto - use row index)", None)
        for col in self.string_cols + self.numeric_cols:
            self.id_combo.addItem(str(col), col)
        # Default to first string column if available
        if self.string_cols:
            idx = self.id_combo.findData(self.string_cols[0])
            if idx >= 0:
                self.id_combo.setCurrentIndex(idx)
        id_layout.addWidget(self.id_combo)
        id_layout.addStretch()
        layout.addWidget(id_group)

        # Target selection
        target_group = QGroupBox("Target Column (Property to Predict)")
        target_layout = QVBoxLayout(target_group)
        target_layout.addWidget(QLabel("Select target (numeric for regression, categorical for classification):"))
        self.target_combo = QComboBox()
        self.target_combo.addItem("(None - load spectra only)", None)

        # Add numeric columns (for regression)
        if self.numeric_cols:
            for col in self.numeric_cols:
                sample_vals = self.df[col].dropna().head(3).tolist()
                sample_str = ", ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in sample_vals)
                self.target_combo.addItem(f"{col}  [numeric] (e.g., {sample_str})", col)

        # Add string columns (for classification)
        if self.string_cols:
            for col in self.string_cols:
                unique_vals = self.df[col].dropna().unique()[:4]
                n_unique = self.df[col].nunique()
                sample_str = ", ".join(str(v) for v in unique_vals)
                if n_unique > 4:
                    sample_str += ", ..."
                self.target_combo.addItem(f"{col}  [class: {n_unique} categories] (e.g., {sample_str})", col)

        target_layout.addWidget(self.target_combo)
        layout.addWidget(target_group)

        # Metadata columns to exclude
        meta_group = QGroupBox("Metadata Columns (Exclude from Spectra)")
        meta_layout = QVBoxLayout(meta_group)
        meta_layout.addWidget(QLabel("Select columns to exclude (Ctrl+click for multiple):"))
        self.meta_list = QListWidget()
        self.meta_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        for col in self.string_cols + self.numeric_cols:
            item = QListWidgetItem(str(col))
            item.setData(Qt.ItemDataRole.UserRole, col)
            self.meta_list.addItem(item)
        self.meta_list.setMaximumHeight(100)
        meta_layout.addWidget(self.meta_list)
        layout.addWidget(meta_group)

        # Preview info
        preview_label = QLabel(f"<i>Preview: {len(self.df)} rows shown of file</i>")
        preview_label.setStyleSheet("color: #888;")
        layout.addWidget(preview_label)

        layout.addStretch()

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        ok_btn = QPushButton("Load Data")
        ok_btn.clicked.connect(self._on_accept)
        ok_btn.setDefault(True)
        btn_layout.addWidget(ok_btn)

        layout.addLayout(btn_layout)

    def _on_accept(self):
        """Validate and accept the configuration."""
        # Get selections
        id_col = self.id_combo.currentData()
        target_col = self.target_combo.currentData()

        # Get metadata columns
        meta_cols = []
        for item in self.meta_list.selectedItems():
            meta_cols.append(item.data(Qt.ItemDataRole.UserRole))

        # Validate
        if not self.wavelength_cols and not self.numeric_cols:
            QMessageBox.warning(
                self,
                "No Spectral Data",
                "Could not find any wavelength or numeric columns to use as spectra."
            )
            return

        # Store result
        self.result = {
            'id_column': id_col,
            'target_column': target_col,
            'metadata_columns': meta_cols,
            'wavelength_columns': self.wavelength_cols if self.wavelength_cols else None
        }

        self.accept()

    def get_config(self) -> dict:
        """Get the column configuration after dialog is accepted."""
        return self.result
