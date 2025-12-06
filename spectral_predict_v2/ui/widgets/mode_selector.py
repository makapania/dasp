"""
Mode Selector Widget - Tab-style mode navigation.

A modern, themed tab selector for switching between application modes.
"""

from enum import Enum, auto
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QGraphicsDropShadowEffect, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve, Property
from PySide6.QtGui import QColor

from ..theme.tokens import COLORS, SPACING, RADIUS, TYPOGRAPHY, SHADOW_PARAMS
from ..theme.icons import Icons


class AppMode(Enum):
    """Application modes."""
    EXPLORE = auto()
    BUILD = auto()
    PREDICT = auto()


class ModeButton(QPushButton):
    """
    A styled button for mode selection.

    Features animated hover and active states.
    """

    def __init__(
        self,
        mode: AppMode,
        icon_name: str,
        label: str,
        description: str,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.mode = mode
        self._label = label
        self._description = description
        self._icon_name = icon_name
        self._is_active = False

        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(160)
        self.setMinimumHeight(90)

        self._setup_ui()
        self._apply_style()

    def _setup_ui(self):
        """Set up the button layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACING["md"], SPACING["sm"], SPACING["md"], SPACING["sm"])
        layout.setSpacing(SPACING["xs"])
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Icon
        self.icon_label = QLabel()
        icon = self._get_icon()
        if icon:
            self.icon_label.setPixmap(icon.pixmap(24, 24))
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.icon_label)

        # Label
        self.label = QLabel(self._label)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)

        # Description (smaller text)
        self.desc_label = QLabel(self._description)
        self.desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.desc_label.setWordWrap(True)
        layout.addWidget(self.desc_label)

    def _get_icon(self):
        """Get the appropriate icon."""
        icon_map = {
            "explore": Icons.chart,
            "build": Icons.settings,
            "predict": Icons.play,
        }
        method = icon_map.get(self._icon_name)
        if method:
            color = COLORS["accent_primary"] if self._is_active else COLORS["text_secondary"]
            return method(24, color)
        return None

    def _apply_style(self):
        """Apply styling based on state."""
        if self._is_active:
            # Active: Frosted glass with accent glow
            bg = "rgba(33, 38, 45, 0.9)"
            border = COLORS["accent_primary"]
            text_color = COLORS["text_primary"]
            desc_color = COLORS["text_secondary"]
        else:
            # Inactive: Subtle transparent
            bg = "rgba(45, 50, 60, 0.3)"
            border = "rgba(255, 255, 255, 0.06)"
            text_color = COLORS["text_secondary"]
            desc_color = COLORS["text_tertiary"]

        self.setStyleSheet(f"""
            ModeButton {{
                background-color: {bg};
                border: 2px solid {border};
                border-radius: {RADIUS["lg"]}px;
                min-height: 80px;
            }}
            ModeButton:hover {{
                background-color: rgba(48, 54, 61, 0.85);
                border-color: {COLORS["accent_primary"] if self._is_active else "rgba(255, 255, 255, 0.15)"};
            }}
        """)

        self.label.setStyleSheet(f"""
            color: {text_color};
            font-size: {TYPOGRAPHY["size_md"]}pt;
            font-weight: {TYPOGRAPHY["weight_semibold"]};
            background: transparent;
            border: none;
        """)

        self.desc_label.setStyleSheet(f"""
            color: {desc_color};
            font-size: {TYPOGRAPHY["size_sm"]}pt;
            background: transparent;
            border: none;
        """)

        # Update icon color
        icon = self._get_icon()
        if icon:
            self.icon_label.setPixmap(icon.pixmap(24, 24))

    def set_active(self, active: bool):
        """Set the active state."""
        self._is_active = active
        self.setChecked(active)
        self._apply_style()


class ModeSelector(QWidget):
    """
    Tab-style mode selector for switching between app modes.

    Displays three modes: Explore, Build, Predict with icons
    and descriptions. Emits signal when mode changes.
    """

    mode_changed = Signal(AppMode)

    MODE_CONFIG = {
        AppMode.EXPLORE: {
            "icon": "explore",
            "label": "Explore",
            "description": "View and edit data"
        },
        AppMode.BUILD: {
            "icon": "build",
            "label": "Build",
            "description": "Train models"
        },
        AppMode.PREDICT: {
            "icon": "predict",
            "label": "Predict",
            "description": "Apply models"
        }
    }

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._current_mode = AppMode.EXPLORE
        self._buttons: dict[AppMode, ModeButton] = {}

        self._setup_ui()

    def _setup_ui(self):
        """Set up the selector layout."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACING["md"], SPACING["md"], SPACING["md"], SPACING["md"])
        layout.setSpacing(SPACING["md"])

        # Create mode buttons
        for mode, config in self.MODE_CONFIG.items():
            btn = ModeButton(
                mode=mode,
                icon_name=config["icon"],
                label=config["label"],
                description=config["description"],
                parent=self
            )
            btn.clicked.connect(lambda checked, m=mode: self._on_mode_clicked(m))
            self._buttons[mode] = btn
            layout.addWidget(btn)

        # Set initial mode
        self._update_active_state()

        # Apply frosted glass container style
        self.setStyleSheet(f"""
            ModeSelector {{
                background-color: rgba(22, 27, 34, 0.85);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-top: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: {RADIUS["xl"]}px;
            }}
        """)

        # Add stronger shadow for floating glass effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(0, 0, 0, 80))
        self.setGraphicsEffect(shadow)

    def _on_mode_clicked(self, mode: AppMode):
        """Handle mode button click."""
        if mode != self._current_mode:
            self._current_mode = mode
            self._update_active_state()
            self.mode_changed.emit(mode)

    def _update_active_state(self):
        """Update active state of all buttons."""
        for mode, btn in self._buttons.items():
            btn.set_active(mode == self._current_mode)

    @property
    def current_mode(self) -> AppMode:
        """Get the current mode."""
        return self._current_mode

    def set_mode(self, mode: AppMode):
        """Programmatically set the mode."""
        if mode != self._current_mode:
            self._current_mode = mode
            self._update_active_state()
            self.mode_changed.emit(mode)


class CompactModeSelector(QWidget):
    """
    A compact horizontal mode selector for use in toolbars.

    Shows only icons with tooltips.
    """

    mode_changed = Signal(AppMode)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._current_mode = AppMode.EXPLORE
        self._buttons: dict[AppMode, QPushButton] = {}

        self._setup_ui()

    def _setup_ui(self):
        """Set up the compact selector."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACING["xs"], SPACING["xs"], SPACING["xs"], SPACING["xs"])
        layout.setSpacing(0)

        mode_info = [
            (AppMode.EXPLORE, Icons.chart, "Explore Data"),
            (AppMode.BUILD, Icons.settings, "Build Models"),
            (AppMode.PREDICT, Icons.play, "Predict"),
        ]

        for i, (mode, icon_method, tooltip) in enumerate(mode_info):
            btn = QPushButton()
            btn.setIcon(icon_method(18, COLORS["text_secondary"]))
            btn.setToolTip(tooltip)
            btn.setCheckable(True)
            btn.setFixedSize(36, 28)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked, m=mode: self._on_mode_clicked(m))

            # Determine border radius based on position
            if i == 0:
                radius = f"{RADIUS['md']}px 0 0 {RADIUS['md']}px"
            elif i == len(mode_info) - 1:
                radius = f"0 {RADIUS['md']}px {RADIUS['md']}px 0"
            else:
                radius = "0"

            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS["bg_elevated"]};
                    border: 1px solid {COLORS["border_default"]};
                    border-radius: {radius};
                    border-right: {"none" if i < len(mode_info) - 1 else f"1px solid {COLORS['border_default']}"};
                }}
                QPushButton:hover {{
                    background-color: {COLORS["bg_overlay"]};
                }}
                QPushButton:checked {{
                    background-color: {COLORS["accent_secondary"]};
                    border-color: {COLORS["accent_secondary"]};
                }}
            """)

            self._buttons[mode] = btn
            layout.addWidget(btn)

        self._update_active_state()

    def _on_mode_clicked(self, mode: AppMode):
        """Handle mode button click."""
        if mode != self._current_mode:
            self._current_mode = mode
            self._update_active_state()
            self.mode_changed.emit(mode)

    def _update_active_state(self):
        """Update active state of all buttons."""
        for mode, btn in self._buttons.items():
            btn.setChecked(mode == self._current_mode)

            # Update icon color
            icon_methods = {
                AppMode.EXPLORE: Icons.chart,
                AppMode.BUILD: Icons.settings,
                AppMode.PREDICT: Icons.play,
            }
            color = COLORS["text_primary"] if mode == self._current_mode else COLORS["text_secondary"]
            btn.setIcon(icon_methods[mode](18, color))

    @property
    def current_mode(self) -> AppMode:
        return self._current_mode

    def set_mode(self, mode: AppMode):
        """Programmatically set the mode."""
        if mode != self._current_mode:
            self._current_mode = mode
            self._update_active_state()
            self.mode_changed.emit(mode)
