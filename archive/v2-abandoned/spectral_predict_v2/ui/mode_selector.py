"""
Mode Selector - Switch between Explore, Build, and Predict modes.
"""

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QButtonGroup
)
from PySide6.QtCore import Signal
from orchestration.state_store import AppMode


class ModeSelector(QWidget):
    """
    Mode selection widget with three buttons: Explore, Build, Predict.
    """

    mode_changed = Signal(AppMode)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_mode = AppMode.EXPLORE

        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)

        # Mode definitions
        modes = [
            (AppMode.EXPLORE, "🔍 EXPLORE", "Find the best model automatically"),
            (AppMode.BUILD, "🔧 BUILD", "Train and refine models"),
            (AppMode.PREDICT, "📊 PREDICT", "Apply saved models to new data"),
        ]

        self.buttons: dict[AppMode, QPushButton] = {}

        for mode, label, tooltip in modes:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setToolTip(tooltip)
            btn.setMinimumWidth(150)
            btn.setMinimumHeight(40)

            self.button_group.addButton(btn)
            self.buttons[mode] = btn
            layout.addWidget(btn)

            # Connect click
            btn.clicked.connect(lambda checked, m=mode: self._on_button_clicked(m))

        # Select Explore by default
        self.buttons[AppMode.EXPLORE].setChecked(True)

        # Styling
        self.setStyleSheet("""
            QPushButton {
                background-color: #2d2d2d;
                border: 1px solid #444;
                border-radius: 0px;
                padding: 8px 16px;
                color: #aaa;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
                color: #ddd;
            }
            QPushButton:checked {
                background-color: #0078d4;
                border-color: #0078d4;
                color: white;
            }
            QPushButton:first-child {
                border-top-left-radius: 6px;
                border-bottom-left-radius: 6px;
            }
            QPushButton:last-child {
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
            }
        """)

    def _on_button_clicked(self, mode: AppMode):
        if mode != self._current_mode:
            self._current_mode = mode
            self.mode_changed.emit(mode)

    def set_mode(self, mode: AppMode):
        """Programmatically set the mode."""
        if mode in self.buttons:
            self.buttons[mode].setChecked(True)
            self._current_mode = mode

    def current_mode(self) -> AppMode:
        return self._current_mode
