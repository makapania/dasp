"""
Card Components - Spectral Predict v2

Card widgets with consistent styling, shadows, and optional collapse behavior.
"""

from typing import Optional
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QSizePolicy,
    QGraphicsDropShadowEffect,
)
from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QColor

from ..theme.tokens import COLORS, SPACING, RADIUS, TYPOGRAPHY, CARD
from ..theme.icons import Icons


class CardHeader(QWidget):
    """
    Card header with title and optional action buttons.

    Features:
    - Title with optional subtitle
    - Right-aligned action buttons
    - Collapse toggle button (optional)
    """

    collapse_toggled = Signal(bool)  # Emits True when expanded

    def __init__(
        self,
        title: str = "",
        subtitle: str = "",
        collapsible: bool = False,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._expanded = True
        self._collapsible = collapsible

        self._setup_ui(title, subtitle, collapsible)
        self._apply_style()

    def _setup_ui(self, title: str, subtitle: str, collapsible: bool):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING["sm"])

        # Title section
        title_layout = QVBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(2)

        self._title_label = QLabel(title)
        self._title_label.setStyleSheet(f"""
            font-size: {TYPOGRAPHY['size_lg']}pt;
            font-weight: {TYPOGRAPHY['weight_semibold']};
            color: {COLORS['text_primary']};
            background: transparent;
        """)
        title_layout.addWidget(self._title_label)

        if subtitle:
            self._subtitle_label = QLabel(subtitle)
            self._subtitle_label.setStyleSheet(f"""
                font-size: {TYPOGRAPHY['size_sm']}pt;
                color: {COLORS['text_secondary']};
                background: transparent;
            """)
            title_layout.addWidget(self._subtitle_label)
        else:
            self._subtitle_label = None

        layout.addLayout(title_layout)
        layout.addStretch()

        # Actions area (populated later)
        self._actions_layout = QHBoxLayout()
        self._actions_layout.setContentsMargins(0, 0, 0, 0)
        self._actions_layout.setSpacing(SPACING["xs"])
        layout.addLayout(self._actions_layout)

        # Collapse button
        if collapsible:
            self._collapse_btn = QPushButton()
            self._collapse_btn.setFixedSize(28, 28)
            self._collapse_btn.setIcon(Icons.chevron_up(16, COLORS["text_secondary"]))
            self._collapse_btn.clicked.connect(self._toggle_collapse)
            self._collapse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._collapse_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    border: 1px solid {COLORS["border_subtle"]};
                    border-radius: {RADIUS["sm"]}px;
                }}
                QPushButton:hover {{
                    background-color: {COLORS["bg_elevated"]};
                    border-color: {COLORS["border_default"]};
                }}
                QPushButton:pressed {{
                    background-color: {COLORS["bg_overlay"]};
                }}
            """)
            layout.addWidget(self._collapse_btn)
        else:
            self._collapse_btn = None

    def _apply_style(self):
        # Styles are now applied directly to labels in _setup_ui
        self.setStyleSheet("CardHeader { background-color: transparent; }")

    def _toggle_collapse(self):
        self._expanded = not self._expanded
        self._update_collapse_icon()
        self.collapse_toggled.emit(self._expanded)

    def _update_collapse_icon(self):
        if self._collapse_btn:
            if self._expanded:
                self._collapse_btn.setIcon(Icons.chevron_up(16, COLORS["text_secondary"]))
            else:
                self._collapse_btn.setIcon(Icons.chevron_down(16, COLORS["text_secondary"]))

    def set_title(self, title: str):
        self._title_label.setText(title)

    def set_subtitle(self, subtitle: str):
        if self._subtitle_label:
            self._subtitle_label.setText(subtitle)

    def add_action(self, widget: QWidget):
        """Add an action widget (button, etc.) to the header."""
        self._actions_layout.addWidget(widget)

    def is_expanded(self) -> bool:
        return self._expanded

    def set_expanded(self, expanded: bool):
        if self._expanded != expanded:
            self._expanded = expanded
            self._update_collapse_icon()
            self.collapse_toggled.emit(expanded)


class Card(QFrame):
    """
    A styled card component with shadow effect.

    Features:
    - Rounded corners
    - Drop shadow
    - Optional header
    - Customizable padding

    Usage:
        card = Card(title="My Card")
        card.add_widget(QLabel("Content"))
    """

    def __init__(
        self,
        title: str = "",
        subtitle: str = "",
        padding: int = CARD["padding"],
        elevated: bool = True,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._padding = padding

        # Prevent cards from being squished
        self.setMinimumHeight(80)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self._setup_ui(title, subtitle)
        self._apply_style(elevated)

        if elevated:
            self._apply_shadow()

    def _setup_ui(self, title: str, subtitle: str):
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(
            self._padding, self._padding, self._padding, self._padding
        )
        self._layout.setSpacing(CARD["gap"])

        # Header (optional)
        self._header = None
        if title:
            self._header = CardHeader(title, subtitle)
            self._layout.addWidget(self._header)
            self._add_separator()

        # Content area
        self._content_layout = QVBoxLayout()
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(SPACING["sm"])
        self._layout.addLayout(self._content_layout)

    def _add_separator(self):
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFixedHeight(1)
        separator.setStyleSheet(f"background-color: {COLORS['border_subtle']};")
        self._layout.addWidget(separator)

    def _apply_style(self, elevated: bool):
        # Frosted glass effect with semi-transparent backgrounds
        if elevated:
            bg_color = "rgba(33, 38, 45, 0.88)"
            border = "rgba(255, 255, 255, 0.08)"
            border_top = "rgba(255, 255, 255, 0.12)"
        else:
            bg_color = "rgba(22, 27, 34, 0.75)"
            border = "rgba(255, 255, 255, 0.05)"
            border_top = "rgba(255, 255, 255, 0.08)"

        self.setStyleSheet(f"""
            Card {{
                background-color: {bg_color};
                border: 1px solid {border};
                border-top: 1px solid {border_top};
                border-radius: {CARD["border_radius"]}px;
            }}
        """)

    def _apply_shadow(self):
        # Enhanced shadow for floating glass panel effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setXOffset(0)
        shadow.setYOffset(8)
        shadow.setColor(QColor(0, 0, 0, 80))
        self.setGraphicsEffect(shadow)

    def add_widget(self, widget: QWidget):
        """Add a widget to the card's content area."""
        self._content_layout.addWidget(widget)

    def add_layout(self, layout):
        """Add a layout to the card's content area."""
        self._content_layout.addLayout(layout)

    def add_stretch(self):
        """Add stretch to the content area."""
        self._content_layout.addStretch()

    def header(self) -> Optional[CardHeader]:
        """Get the card header (if any)."""
        return self._header

    @property
    def content_layout(self) -> QVBoxLayout:
        """Get the content layout for direct manipulation."""
        return self._content_layout

    def set_content_spacing(self, spacing: int):
        """Set spacing between content items."""
        self._content_layout.setSpacing(spacing)


class CollapsibleCard(Card):
    """
    A card that can be collapsed to show only the header.

    Features:
    - Animated collapse/expand
    - Remembers collapsed state
    - Shows collapse indicator in header
    """

    collapsed = Signal()
    expanded = Signal()

    def __init__(
        self,
        title: str = "",
        subtitle: str = "",
        padding: int = CARD["padding"],
        start_collapsed: bool = False,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(title, subtitle, padding, True, parent)

        self._is_expanded = not start_collapsed
        self._content_widget = QWidget()
        self._setup_collapsible()

        if start_collapsed:
            self._content_widget.setVisible(False)

    def _setup_ui(self, title: str, subtitle: str):
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(
            self._padding, self._padding, self._padding, self._padding
        )
        self._layout.setSpacing(CARD["gap"])

        # Header with collapse toggle
        self._header = CardHeader(title, subtitle, collapsible=True)
        self._header.collapse_toggled.connect(self._on_collapse_toggled)
        self._layout.addWidget(self._header)

        # Separator
        self._separator = QFrame()
        self._separator.setFrameShape(QFrame.Shape.HLine)
        self._separator.setFixedHeight(1)
        self._separator.setStyleSheet(f"background-color: {COLORS['border_subtle']};")
        self._layout.addWidget(self._separator)

        # Content area (to be wrapped in widget for animation)
        self._content_layout = QVBoxLayout()
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(SPACING["sm"])

    def _setup_collapsible(self):
        # Wrap content in a widget for animation
        self._content_widget.setLayout(self._content_layout)
        self._layout.addWidget(self._content_widget)

        # Set size policy
        self._content_widget.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Preferred
        )

    def _on_collapse_toggled(self, expanded: bool):
        self._is_expanded = expanded
        self._content_widget.setVisible(expanded)
        self._separator.setVisible(expanded)

        if expanded:
            self.expanded.emit()
        else:
            self.collapsed.emit()

    def is_expanded(self) -> bool:
        return self._is_expanded

    def set_expanded(self, expanded: bool):
        if self._is_expanded != expanded:
            self._header.set_expanded(expanded)

    def toggle(self):
        """Toggle the collapsed state."""
        self.set_expanded(not self._is_expanded)


class StatusCard(Card):
    """
    A card with a colored status indicator.

    Features:
    - Left border color based on status
    - Status badge in header
    """

    def __init__(
        self,
        title: str = "",
        status: str = "default",  # default, success, warning, danger, info
        parent: Optional[QWidget] = None,
    ):
        self._status = status
        super().__init__(title, parent=parent)
        self._apply_status_style()

    def _apply_status_style(self):
        status_colors = {
            "default": COLORS["border_default"],
            "success": COLORS["status_success"],
            "warning": COLORS["status_warning"],
            "danger": COLORS["status_error"],
            "info": COLORS["status_info"],
        }
        color = status_colors.get(self._status, COLORS["border_default"])

        # Frosted glass with colored left accent
        self.setStyleSheet(f"""
            StatusCard {{
                background-color: rgba(22, 27, 34, 0.8);
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-left: 3px solid {color};
                border-radius: {CARD["border_radius"]}px;
            }}
        """)

    def set_status(self, status: str):
        self._status = status
        self._apply_status_style()
