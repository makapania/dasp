"""
Column Configuration Dialog - Lets user specify column roles when loading data.

Redesigned for readability with proper theming and spacing.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QListWidget, QListWidgetItem, QPushButton, QFrame,
    QMessageBox, QAbstractItemView, QScrollArea, QWidget,
    QSizePolicy
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
import pandas as pd
from pathlib import Path

from ..theme.tokens import COLORS, SPACING, RADIUS, TYPOGRAPHY
from ..components.buttons import PrimaryButton, SecondaryButton


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

        self.setWindowTitle("Configure Data Columns")
        self.setMinimumWidth(700)
        self.setMinimumHeight(600)
        self.resize(750, 650)

        self._apply_theme()
        self._load_preview()
        self._setup_ui()

    def _apply_theme(self):
        """Apply dark theme styling to dialog."""
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {COLORS["bg_base"]};
                color: {COLORS["text_primary"]};
            }}
            QLabel {{
                color: {COLORS["text_primary"]};
                font-size: {TYPOGRAPHY["size_md"]}pt;
            }}
            QLabel[class="header"] {{
                font-size: {TYPOGRAPHY["size_lg"]}pt;
                font-weight: {TYPOGRAPHY["weight_semibold"]};
                color: {COLORS["text_primary"]};
            }}
            QLabel[class="subheader"] {{
                font-size: {TYPOGRAPHY["size_md"]}pt;
                color: {COLORS["text_secondary"]};
            }}
            QLabel[class="success"] {{
                color: {COLORS["accent_primary"]};
                font-weight: {TYPOGRAPHY["weight_medium"]};
            }}
            QLabel[class="warning"] {{
                color: {COLORS["accent_warning"]};
                font-weight: {TYPOGRAPHY["weight_medium"]};
            }}
            QComboBox {{
                background-color: {COLORS["bg_elevated"]};
                color: {COLORS["text_primary"]};
                border: 1px solid {COLORS["border_default"]};
                border-radius: {RADIUS["md"]}px;
                padding: 8px 12px;
                min-height: 20px;
                font-size: {TYPOGRAPHY["size_md"]}pt;
            }}
            QComboBox:hover {{
                border-color: {COLORS["border_emphasis"]};
            }}
            QComboBox:focus {{
                border-color: {COLORS["accent_secondary"]};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 24px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {COLORS["bg_elevated"]};
                color: {COLORS["text_primary"]};
                border: 1px solid {COLORS["border_default"]};
                border-radius: {RADIUS["md"]}px;
                padding: 4px;
                selection-background-color: {COLORS["accent_secondary"]};
                selection-color: white;
            }}
            QComboBox QAbstractItemView::item {{
                padding: 8px 12px;
                min-height: 24px;
            }}
            QListWidget {{
                background-color: {COLORS["bg_elevated"]};
                color: {COLORS["text_primary"]};
                border: 1px solid {COLORS["border_default"]};
                border-radius: {RADIUS["md"]}px;
                padding: 4px;
                font-size: {TYPOGRAPHY["size_md"]}pt;
            }}
            QListWidget::item {{
                padding: 8px 12px;
                border-radius: {RADIUS["sm"]}px;
            }}
            QListWidget::item:selected {{
                background-color: {COLORS["accent_secondary"]};
                color: white;
            }}
            QListWidget::item:hover {{
                background-color: {COLORS["bg_overlay"]};
            }}
            QFrame[class="section"] {{
                background-color: {COLORS["bg_surface"]};
                border: 1px solid {COLORS["border_subtle"]};
                border-radius: {RADIUS["lg"]}px;
            }}
        """)

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

    def _create_section(self, title: str, description: str = "") -> tuple:
        """Create a styled section with title and content area."""
        frame = QFrame()
        frame.setProperty("class", "section")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(SPACING["lg"], SPACING["lg"], SPACING["lg"], SPACING["lg"])
        layout.setSpacing(SPACING["sm"])

        # Title
        title_label = QLabel(title)
        title_label.setProperty("class", "header")
        layout.addWidget(title_label)

        # Description
        if description:
            desc_label = QLabel(description)
            desc_label.setProperty("class", "subheader")
            desc_label.setWordWrap(True)
            layout.addWidget(desc_label)

        return frame, layout

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(SPACING["xl"], SPACING["xl"], SPACING["xl"], SPACING["xl"])
        main_layout.setSpacing(SPACING["lg"])

        # === FILE INFO HEADER ===
        file_frame, file_layout = self._create_section(
            f"📁  {Path(self.file_path).name}",
            f"{len(self.df.columns)} columns detected"
        )

        # Analyze columns
        self.numeric_cols = []
        self.string_cols = []
        self.wavelength_cols = []

        for col in self.df.columns:
            col_str = str(col)
            try:
                wl = float(col_str)
                if 100 < wl < 25000:
                    self.wavelength_cols.append(col)
                else:
                    self.numeric_cols.append(col)
            except ValueError:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.numeric_cols.append(col)
                else:
                    self.string_cols.append(col)

        # Wavelength info
        if self.wavelength_cols:
            wl_min = min(float(str(c)) for c in self.wavelength_cols)
            wl_max = max(float(str(c)) for c in self.wavelength_cols)
            wl_label = QLabel(f"✓  {len(self.wavelength_cols)} wavelength columns detected ({wl_min:.0f} - {wl_max:.0f} nm)")
            wl_label.setProperty("class", "success")
            file_layout.addWidget(wl_label)
        else:
            wl_label = QLabel("⚠  No wavelength columns detected - will use numeric columns")
            wl_label.setProperty("class", "warning")
            file_layout.addWidget(wl_label)

        main_layout.addWidget(file_frame)

        # === SAMPLE ID SECTION ===
        id_frame, id_layout = self._create_section(
            "Sample ID Column",
            "Select a column to use as row identifier (optional)"
        )

        self.id_combo = QComboBox()
        self.id_combo.setMinimumHeight(36)
        self.id_combo.addItem("(Auto - use row index)", None)
        for col in self.string_cols + self.numeric_cols:
            display = self._format_column_name(col, show_sample=True)
            self.id_combo.addItem(display, col)

        if self.string_cols:
            idx = self.id_combo.findData(self.string_cols[0])
            if idx >= 0:
                self.id_combo.setCurrentIndex(idx)

        id_layout.addWidget(self.id_combo)
        main_layout.addWidget(id_frame)

        # === TARGET COLUMN SECTION ===
        target_frame, target_layout = self._create_section(
            "Target Column",
            "Select the property to predict (numeric for regression, categorical for classification)"
        )

        self.target_combo = QComboBox()
        self.target_combo.setMinimumHeight(36)
        self.target_combo.addItem("(None - load spectra only)", None)

        # Add numeric columns
        if self.numeric_cols:
            for col in self.numeric_cols:
                sample_vals = self.df[col].dropna().head(3).tolist()
                sample_str = ", ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in sample_vals)
                self.target_combo.addItem(f"{col}   •   numeric   •   e.g. {sample_str}", col)

        # Add string columns
        if self.string_cols:
            for col in self.string_cols:
                unique_vals = self.df[col].dropna().unique()[:3]
                n_unique = self.df[col].nunique()
                sample_str = ", ".join(str(v) for v in unique_vals)
                self.target_combo.addItem(f"{col}   •   {n_unique} classes   •   e.g. {sample_str}", col)

        target_layout.addWidget(self.target_combo)
        main_layout.addWidget(target_frame)

        # === METADATA SECTION ===
        meta_frame, meta_layout = self._create_section(
            "Exclude from Spectra",
            "Select metadata columns to exclude (click to select, Ctrl+click for multiple)"
        )

        self.meta_list = QListWidget()
        self.meta_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.meta_list.setMinimumHeight(150)
        self.meta_list.setMaximumHeight(200)

        for col in self.string_cols + self.numeric_cols:
            display = self._format_column_name(col, show_sample=True)
            item = QListWidgetItem(display)
            item.setData(Qt.ItemDataRole.UserRole, col)
            self.meta_list.addItem(item)

        meta_layout.addWidget(self.meta_list)
        main_layout.addWidget(meta_frame)

        # Add stretch
        main_layout.addStretch()

        # === BUTTONS ===
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(SPACING["md"])

        # Info label
        info_label = QLabel(f"Preview based on first {len(self.df)} rows")
        info_label.setProperty("class", "subheader")
        btn_layout.addWidget(info_label)

        btn_layout.addStretch()

        cancel_btn = SecondaryButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        ok_btn = PrimaryButton("Load Data")
        ok_btn.clicked.connect(self._on_accept)
        ok_btn.setDefault(True)
        btn_layout.addWidget(ok_btn)

        main_layout.addLayout(btn_layout)

    def _format_column_name(self, col, show_sample: bool = False) -> str:
        """Format column name with optional sample values."""
        col_str = str(col)
        if not show_sample:
            return col_str

        try:
            sample_vals = self.df[col].dropna().head(2).tolist()
            if sample_vals:
                sample_str = ", ".join(
                    f"{v:.2f}" if isinstance(v, float) else str(v)[:20]
                    for v in sample_vals
                )
                return f"{col_str}   •   e.g. {sample_str}"
        except Exception:
            pass

        return col_str

    def _on_accept(self):
        """Validate and accept the configuration."""
        id_col = self.id_combo.currentData()
        target_col = self.target_combo.currentData()

        meta_cols = []
        for item in self.meta_list.selectedItems():
            meta_cols.append(item.data(Qt.ItemDataRole.UserRole))

        if not self.wavelength_cols and not self.numeric_cols:
            QMessageBox.warning(
                self,
                "No Spectral Data",
                "Could not find any wavelength or numeric columns to use as spectra."
            )
            return

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
