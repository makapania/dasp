"""
Data Context Bar - Rich data summary display.

Shows key statistics and metadata about the currently loaded data.
"""

from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame,
    QGraphicsDropShadowEffect, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor

from ..theme.tokens import COLORS, SPACING, RADIUS, TYPOGRAPHY, SHADOWS
from ..theme.icons import Icons


class StatBadge(QWidget):
    """
    A compact badge showing a labeled statistic.

    Used for displaying data metrics like sample count, wavelength range, etc.
    """

    def __init__(
        self,
        label: str,
        value: str = "--",
        icon_method=None,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self._label = label
        self._value = value
        self._icon_method = icon_method

        self._setup_ui()

    def _setup_ui(self):
        """Set up the badge layout."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACING["sm"], SPACING["xs"], SPACING["sm"], SPACING["xs"])
        layout.setSpacing(SPACING["xs"])

        # Icon (optional)
        if self._icon_method:
            icon_label = QLabel()
            icon_label.setPixmap(self._icon_method(14, COLORS["text_tertiary"]).pixmap(14, 14))
            layout.addWidget(icon_label)

        # Label
        label = QLabel(f"{self._label}:")
        label.setStyleSheet(f"""
            color: {COLORS["text_tertiary"]};
            font-size: {TYPOGRAPHY["size_sm"]}pt;
        """)
        layout.addWidget(label)

        # Value
        self.value_label = QLabel(self._value)
        self.value_label.setStyleSheet(f"""
            color: {COLORS["text_primary"]};
            font-size: {TYPOGRAPHY["size_sm"]}pt;
            font-weight: {TYPOGRAPHY["weight_medium"]};
        """)
        layout.addWidget(self.value_label)

        # Container styling
        self.setStyleSheet(f"""
            StatBadge {{
                background-color: {COLORS["bg_elevated"]};
                border: 1px solid {COLORS["border_subtle"]};
                border-radius: {RADIUS["sm"]}px;
            }}
        """)

    def set_value(self, value: str):
        """Update the displayed value."""
        self._value = value
        self.value_label.setText(value)

    def set_highlight(self, highlight: bool):
        """Set highlight state for emphasis."""
        if highlight:
            self.value_label.setStyleSheet(f"""
                color: {COLORS["accent_primary"]};
                font-size: {TYPOGRAPHY["size_sm"]}pt;
                font-weight: {TYPOGRAPHY["weight_semibold"]};
            """)
        else:
            self.value_label.setStyleSheet(f"""
                color: {COLORS["text_primary"]};
                font-size: {TYPOGRAPHY["size_sm"]}pt;
                font-weight: {TYPOGRAPHY["weight_medium"]};
            """)


class DataContextBar(QWidget):
    """
    A rich context bar showing data summary information.

    Displays:
    - File name
    - Sample count
    - Wavelength count and range
    - Target column info
    - Data quality indicators
    """

    file_clicked = Signal()  # Emitted when file info is clicked

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()
        self.clear()

    def _setup_ui(self):
        """Set up the context bar layout."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACING["md"], SPACING["sm"], SPACING["md"], SPACING["sm"])
        layout.setSpacing(SPACING["md"])

        # Left section: File info
        file_section = QWidget()
        file_layout = QHBoxLayout(file_section)
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.setSpacing(SPACING["sm"])

        # File icon
        self.file_icon = QLabel()
        self.file_icon.setPixmap(Icons.folder_open(16, COLORS["text_secondary"]).pixmap(16, 16))
        file_layout.addWidget(self.file_icon)

        # File name
        self.file_label = QLabel("No data loaded")
        self.file_label.setStyleSheet(f"""
            color: {COLORS["text_secondary"]};
            font-size: {TYPOGRAPHY["size_md"]}pt;
        """)
        self.file_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.file_label.mousePressEvent = lambda e: self.file_clicked.emit()
        file_layout.addWidget(self.file_label)

        layout.addWidget(file_section)

        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.VLine)
        sep1.setStyleSheet(f"background-color: {COLORS['border_subtle']};")
        sep1.setFixedWidth(1)
        layout.addWidget(sep1)

        # Stats section
        stats_section = QWidget()
        stats_layout = QHBoxLayout(stats_section)
        stats_layout.setContentsMargins(0, 0, 0, 0)
        stats_layout.setSpacing(SPACING["sm"])

        self.samples_badge = StatBadge("Samples", "--")
        stats_layout.addWidget(self.samples_badge)

        self.wavelengths_badge = StatBadge("Wavelengths", "--")
        stats_layout.addWidget(self.wavelengths_badge)

        self.range_badge = StatBadge("Range", "--")
        stats_layout.addWidget(self.range_badge)

        layout.addWidget(stats_section)

        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setStyleSheet(f"background-color: {COLORS['border_subtle']};")
        sep2.setFixedWidth(1)
        layout.addWidget(sep2)

        # Target info section
        target_section = QWidget()
        target_layout = QHBoxLayout(target_section)
        target_layout.setContentsMargins(0, 0, 0, 0)
        target_layout.setSpacing(SPACING["sm"])

        self.target_badge = StatBadge("Target", "--")
        target_layout.addWidget(self.target_badge)

        self.target_range_badge = StatBadge("Values", "--")
        target_layout.addWidget(self.target_range_badge)

        layout.addWidget(target_section)

        # Spacer
        layout.addStretch()

        # Right section: Data quality indicators
        quality_section = QWidget()
        quality_layout = QHBoxLayout(quality_section)
        quality_layout.setContentsMargins(0, 0, 0, 0)
        quality_layout.setSpacing(SPACING["sm"])

        # Missing values indicator
        self.missing_indicator = QLabel()
        self.missing_indicator.setPixmap(Icons.success(14, COLORS["accent_primary"]).pixmap(14, 14))
        self.missing_indicator.setToolTip("No missing values")
        quality_layout.addWidget(self.missing_indicator)

        # Outliers indicator
        self.outliers_indicator = QLabel()
        self.outliers_indicator.setPixmap(Icons.success(14, COLORS["accent_primary"]).pixmap(14, 14))
        self.outliers_indicator.setToolTip("No flagged outliers")
        quality_layout.addWidget(self.outliers_indicator)

        layout.addWidget(quality_section)

        # Container styling
        self.setStyleSheet(f"""
            DataContextBar {{
                background-color: {COLORS["bg_surface"]};
                border-bottom: 1px solid {COLORS["border_default"]};
            }}
        """)

        self.setMinimumHeight(56)
        self.setMaximumHeight(72)

    def set_data(
        self,
        file_name: str,
        n_samples: int,
        n_wavelengths: int,
        wavelength_range: tuple[float, float],
        target_column: Optional[str] = None,
        target_range: Optional[tuple[float, float]] = None,
        missing_count: int = 0,
        outlier_count: int = 0
    ):
        """
        Update the context bar with data information.

        Args:
            file_name: Name of the loaded file
            n_samples: Number of samples/rows
            n_wavelengths: Number of wavelength columns
            wavelength_range: (min, max) wavelength values
            target_column: Name of the target variable column
            target_range: (min, max) of target values
            missing_count: Number of missing values
            outlier_count: Number of flagged outliers
        """
        # File info
        self.file_label.setText(file_name)
        self.file_label.setStyleSheet(f"""
            color: {COLORS["text_primary"]};
            font-size: {TYPOGRAPHY["size_md"]}pt;
            font-weight: {TYPOGRAPHY["weight_medium"]};
        """)
        self.file_icon.setPixmap(Icons.file(16, COLORS["accent_secondary"]).pixmap(16, 16))

        # Stats
        self.samples_badge.set_value(f"{n_samples:,}")
        self.wavelengths_badge.set_value(f"{n_wavelengths:,}")

        wl_min, wl_max = wavelength_range
        self.range_badge.set_value(f"{wl_min:.0f}-{wl_max:.0f} nm")

        # Target
        if target_column:
            self.target_badge.set_value(target_column)
            self.target_badge.set_highlight(True)
            if target_range:
                t_min, t_max = target_range
                self.target_range_badge.set_value(f"{t_min:.2f}-{t_max:.2f}")
        else:
            self.target_badge.set_value("Not set")
            self.target_badge.set_highlight(False)
            self.target_range_badge.set_value("--")

        # Quality indicators
        if missing_count > 0:
            self.missing_indicator.setPixmap(
                Icons.warning(14, COLORS["accent_warning"]).pixmap(14, 14)
            )
            self.missing_indicator.setToolTip(f"{missing_count:,} missing values")
        else:
            self.missing_indicator.setPixmap(
                Icons.success(14, COLORS["accent_primary"]).pixmap(14, 14)
            )
            self.missing_indicator.setToolTip("No missing values")

        if outlier_count > 0:
            self.outliers_indicator.setPixmap(
                Icons.warning(14, COLORS["accent_danger"]).pixmap(14, 14)
            )
            self.outliers_indicator.setToolTip(f"{outlier_count} flagged outliers")
        else:
            self.outliers_indicator.setPixmap(
                Icons.success(14, COLORS["accent_primary"]).pixmap(14, 14)
            )
            self.outliers_indicator.setToolTip("No flagged outliers")

    def clear(self):
        """Clear all data and reset to empty state."""
        self.file_label.setText("No data loaded")
        self.file_label.setStyleSheet(f"""
            color: {COLORS["text_tertiary"]};
            font-size: {TYPOGRAPHY["size_md"]}pt;
            font-style: italic;
        """)
        self.file_icon.setPixmap(Icons.folder_open(16, COLORS["text_tertiary"]).pixmap(16, 16))

        self.samples_badge.set_value("--")
        self.wavelengths_badge.set_value("--")
        self.range_badge.set_value("--")
        self.target_badge.set_value("--")
        self.target_badge.set_highlight(False)
        self.target_range_badge.set_value("--")

        self.missing_indicator.setPixmap(
            Icons.info(14, COLORS["text_tertiary"]).pixmap(14, 14)
        )
        self.missing_indicator.setToolTip("No data")

        self.outliers_indicator.setPixmap(
            Icons.info(14, COLORS["text_tertiary"]).pixmap(14, 14)
        )
        self.outliers_indicator.setToolTip("No data")


class CompactDataContextBar(QWidget):
    """
    A minimal context bar for space-constrained layouts.

    Shows only essential info: file name, samples, wavelengths.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Set up the compact bar."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACING["sm"], SPACING["xs"], SPACING["sm"], SPACING["xs"])
        layout.setSpacing(SPACING["md"])

        # File name
        self.file_label = QLabel("No data")
        self.file_label.setStyleSheet(f"""
            color: {COLORS["text_secondary"]};
            font-size: {TYPOGRAPHY["size_sm"]}pt;
        """)
        layout.addWidget(self.file_label)

        # Dot separator
        dot = QLabel("*")
        dot.setStyleSheet(f"color: {COLORS['text_tertiary']};")
        layout.addWidget(dot)

        # Sample count
        self.samples_label = QLabel("-- samples")
        self.samples_label.setStyleSheet(f"""
            color: {COLORS["text_tertiary"]};
            font-size: {TYPOGRAPHY["size_sm"]}pt;
        """)
        layout.addWidget(self.samples_label)

        # Dot separator
        dot2 = QLabel("*")
        dot2.setStyleSheet(f"color: {COLORS['text_tertiary']};")
        layout.addWidget(dot2)

        # Wavelength count
        self.wavelengths_label = QLabel("-- wavelengths")
        self.wavelengths_label.setStyleSheet(f"""
            color: {COLORS["text_tertiary"]};
            font-size: {TYPOGRAPHY["size_sm"]}pt;
        """)
        layout.addWidget(self.wavelengths_label)

        layout.addStretch()

        self.setStyleSheet(f"""
            CompactDataContextBar {{
                background-color: {COLORS["bg_elevated"]};
                border-radius: {RADIUS["sm"]}px;
            }}
        """)

    def set_data(self, file_name: str, n_samples: int, n_wavelengths: int):
        """Update with data info."""
        self.file_label.setText(file_name)
        self.file_label.setStyleSheet(f"""
            color: {COLORS["text_primary"]};
            font-size: {TYPOGRAPHY["size_sm"]}pt;
            font-weight: {TYPOGRAPHY["weight_medium"]};
        """)
        self.samples_label.setText(f"{n_samples:,} samples")
        self.wavelengths_label.setText(f"{n_wavelengths:,} wavelengths")

    def clear(self):
        """Reset to empty state."""
        self.file_label.setText("No data")
        self.file_label.setStyleSheet(f"""
            color: {COLORS["text_secondary"]};
            font-size: {TYPOGRAPHY["size_sm"]}pt;
        """)
        self.samples_label.setText("-- samples")
        self.wavelengths_label.setText("-- wavelengths")
