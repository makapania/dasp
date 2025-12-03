"""
Data Context Bar - Persistent data display across all modes.

Shows current dataset info: file name, samples, wavelengths, target, quality flags.
"""

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QPushButton, QFrame
)
from PySide6.QtCore import Qt, Signal
from orchestration.state_store import StateStore


class DataContextBar(QWidget):
    """
    Persistent bar showing current dataset information.

    Displayed at the top of the application, visible in all modes.
    """

    load_data_clicked = Signal()
    view_data_clicked = Signal()
    quality_check_clicked = Signal()

    def __init__(self, state: StateStore, parent=None):
        super().__init__(parent)
        self.state = state

        self._setup_ui()
        self._connect_signals()
        self._update_display()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(16)

        # File icon and name
        self.file_label = QLabel()
        self.file_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.file_label)

        # Separator
        layout.addWidget(self._create_separator())

        # Sample count
        self.samples_label = QLabel()
        layout.addWidget(self.samples_label)

        # Separator
        layout.addWidget(self._create_separator())

        # Wavelength range
        self.wavelength_label = QLabel()
        layout.addWidget(self.wavelength_label)

        # Separator
        layout.addWidget(self._create_separator())

        # Target info
        self.target_label = QLabel()
        layout.addWidget(self.target_label)

        # Separator
        layout.addWidget(self._create_separator())

        # Quality flags
        self.quality_label = QLabel()
        layout.addWidget(self.quality_label)

        # Spacer
        layout.addStretch()

        # Action buttons
        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.load_data_clicked.emit)
        layout.addWidget(self.load_btn)

        self.view_btn = QPushButton("View")
        self.view_btn.clicked.connect(self.view_data_clicked.emit)
        self.view_btn.setEnabled(False)
        layout.addWidget(self.view_btn)

        self.quality_btn = QPushButton("Quality Check")
        self.quality_btn.clicked.connect(self.quality_check_clicked.emit)
        self.quality_btn.setEnabled(False)
        layout.addWidget(self.quality_btn)

        # Style the bar
        self.setObjectName("dataContextBar")
        self.setStyleSheet("""
            #dataContextBar {
                background-color: #2d2d2d;
                border-bottom: 1px solid #444;
            }
            QLabel {
                color: #e0e0e0;
                padding: 2px 4px;
            }
            QPushButton {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 4px 12px;
                color: #e0e0e0;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
            QPushButton:disabled {
                color: #666;
                background-color: #2d2d2d;
            }
        """)

    def _create_separator(self) -> QFrame:
        """Create a vertical separator line."""
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #555;")
        return sep

    def _connect_signals(self):
        """Connect to state store signals."""
        self.state.data_changed.connect(self._update_display)

    def _update_display(self):
        """Update the display based on current state."""
        if not self.state.has_data:
            self.file_label.setText("No data loaded")
            self.samples_label.setText("")
            self.wavelength_label.setText("")
            self.target_label.setText("")
            self.quality_label.setText("")
            self.view_btn.setEnabled(False)
            self.quality_btn.setEnabled(False)
            return

        summary = self.state.get_data_summary()

        self.file_label.setText(f"📁 {summary.get('file_name', 'Unknown')}")
        self.samples_label.setText(f"{summary.get('n_samples', 0)} samples")
        self.wavelength_label.setText(
            f"{summary.get('n_wavelengths', 0)} λ ({summary.get('wavelength_range', '')})"
        )
        self.target_label.setText(
            f"Target: {summary.get('target', '?')} ({summary.get('target_range', '')})"
        )

        n_outliers = summary.get("n_outliers", 0)
        if n_outliers > 0:
            self.quality_label.setText(f"⚠️ {n_outliers} outliers")
            self.quality_label.setStyleSheet("color: #ffa500;")  # Orange
        else:
            self.quality_label.setText("✓ No issues")
            self.quality_label.setStyleSheet("color: #4caf50;")  # Green

        self.view_btn.setEnabled(True)
        self.quality_btn.setEnabled(True)
