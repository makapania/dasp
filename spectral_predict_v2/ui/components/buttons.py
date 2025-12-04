"""
Button Components - Spectral Predict v2

Styled button variants with consistent appearance.
"""

from typing import Optional
from PySide6.QtWidgets import QPushButton, QWidget, QHBoxLayout, QSizePolicy
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon

from ..theme.tokens import COLORS, SPACING, RADIUS, TYPOGRAPHY, BUTTON


class BaseButton(QPushButton):
    """
    Base button class with common styling.

    All button variants inherit from this class.
    """

    def __init__(
        self,
        text: str = "",
        icon: Optional[QIcon] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(text, parent)

        if icon:
            self.setIcon(icon)
            self.setIconSize(QSize(BUTTON["icon_size"], BUTTON["icon_size"]))

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._size = "medium"
        self._apply_base_style()

    def _apply_base_style(self):
        """Apply base styling - override in subclasses."""
        pass

    def set_size(self, size: str):
        """
        Set button size variant.

        Args:
            size: "small", "medium", or "large"
        """
        self._size = size
        self.setProperty("size", size)
        self.style().unpolish(self)
        self.style().polish(self)

    def set_loading(self, loading: bool):
        """Set loading state (disables button, shows loading indicator)."""
        self.setEnabled(not loading)
        # TODO: Add loading spinner animation


class PrimaryButton(BaseButton):
    """
    Primary action button (green).

    Use for main actions like "Run", "Save", "Apply".
    """

    def __init__(
        self,
        text: str = "",
        icon: Optional[QIcon] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(text, icon, parent)
        self._apply_base_style()

    def _apply_base_style(self):
        self.setStyleSheet(f"""
            PrimaryButton {{
                background-color: {COLORS["accent_primary"]};
                color: #ffffff;
                border: 1px solid {COLORS["accent_primary"]};
                border-radius: {RADIUS["md"]}px;
                padding: 6px {BUTTON["padding_x_md"]}px;
                min-height: {BUTTON["height_md"] - 12}px;
                font-weight: {TYPOGRAPHY["weight_medium"]};
            }}
            PrimaryButton:hover {{
                background-color: {COLORS["accent_primary_hover"]};
                border-color: {COLORS["accent_primary_hover"]};
            }}
            PrimaryButton:pressed {{
                background-color: {COLORS["accent_primary_active"]};
            }}
            PrimaryButton:disabled {{
                background-color: {COLORS["accent_primary_muted"]};
                border-color: transparent;
                color: {COLORS["text_tertiary"]};
            }}
            PrimaryButton:focus {{
                outline: none;
                border-color: {COLORS["accent_secondary"]};
            }}
            PrimaryButton[size="small"] {{
                padding: 4px {BUTTON["padding_x_sm"]}px;
                min-height: {BUTTON["height_sm"] - 8}px;
                font-size: {TYPOGRAPHY["size_sm"]}pt;
            }}
            PrimaryButton[size="large"] {{
                padding: 8px {BUTTON["padding_x_lg"]}px;
                min-height: {BUTTON["height_lg"] - 16}px;
                font-size: {TYPOGRAPHY["size_lg"]}pt;
            }}
        """)


class SecondaryButton(BaseButton):
    """
    Secondary action button (blue).

    Use for secondary actions, links, navigation.
    """

    def __init__(
        self,
        text: str = "",
        icon: Optional[QIcon] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(text, icon, parent)
        self._apply_base_style()

    def _apply_base_style(self):
        self.setStyleSheet(f"""
            SecondaryButton {{
                background-color: {COLORS["accent_secondary"]};
                color: #ffffff;
                border: 1px solid {COLORS["accent_secondary"]};
                border-radius: {RADIUS["md"]}px;
                padding: 6px {BUTTON["padding_x_md"]}px;
                min-height: {BUTTON["height_md"] - 12}px;
                font-weight: {TYPOGRAPHY["weight_medium"]};
            }}
            SecondaryButton:hover {{
                background-color: {COLORS["accent_secondary_hover"]};
                border-color: {COLORS["accent_secondary_hover"]};
            }}
            SecondaryButton:pressed {{
                background-color: {COLORS["accent_secondary"]};
            }}
            SecondaryButton:disabled {{
                background-color: {COLORS["accent_secondary_muted"]};
                border-color: transparent;
                color: {COLORS["text_tertiary"]};
            }}
            SecondaryButton[size="small"] {{
                padding: 4px {BUTTON["padding_x_sm"]}px;
                min-height: {BUTTON["height_sm"] - 8}px;
                font-size: {TYPOGRAPHY["size_sm"]}pt;
            }}
            SecondaryButton[size="large"] {{
                padding: 8px {BUTTON["padding_x_lg"]}px;
                min-height: {BUTTON["height_lg"] - 16}px;
                font-size: {TYPOGRAPHY["size_lg"]}pt;
            }}
        """)


class DangerButton(BaseButton):
    """
    Danger/destructive action button (red).

    Use for delete, remove, or other destructive actions.
    """

    def __init__(
        self,
        text: str = "",
        icon: Optional[QIcon] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(text, icon, parent)
        self._apply_base_style()

    def _apply_base_style(self):
        self.setStyleSheet(f"""
            DangerButton {{
                background-color: {COLORS["accent_danger"]};
                color: #ffffff;
                border: 1px solid {COLORS["accent_danger"]};
                border-radius: {RADIUS["md"]}px;
                padding: 6px {BUTTON["padding_x_md"]}px;
                min-height: {BUTTON["height_md"] - 12}px;
                font-weight: {TYPOGRAPHY["weight_medium"]};
            }}
            DangerButton:hover {{
                background-color: {COLORS["accent_danger_hover"]};
                border-color: {COLORS["accent_danger_hover"]};
            }}
            DangerButton:pressed {{
                background-color: {COLORS["accent_danger"]};
            }}
            DangerButton:disabled {{
                background-color: {COLORS["accent_danger_muted"]};
                border-color: transparent;
                color: {COLORS["text_tertiary"]};
            }}
            DangerButton[size="small"] {{
                padding: 4px {BUTTON["padding_x_sm"]}px;
                min-height: {BUTTON["height_sm"] - 8}px;
                font-size: {TYPOGRAPHY["size_sm"]}pt;
            }}
            DangerButton[size="large"] {{
                padding: 8px {BUTTON["padding_x_lg"]}px;
                min-height: {BUTTON["height_lg"] - 16}px;
                font-size: {TYPOGRAPHY["size_lg"]}pt;
            }}
        """)


class GhostButton(BaseButton):
    """
    Ghost button (transparent background).

    Use for tertiary actions, toolbar buttons, subtle interactions.
    """

    def __init__(
        self,
        text: str = "",
        icon: Optional[QIcon] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(text, icon, parent)
        self._apply_base_style()

    def _apply_base_style(self):
        self.setStyleSheet(f"""
            GhostButton {{
                background-color: transparent;
                color: {COLORS["text_primary"]};
                border: none;
                border-radius: {RADIUS["md"]}px;
                padding: 6px {BUTTON["padding_x_md"]}px;
                min-height: {BUTTON["height_md"] - 12}px;
                font-weight: {TYPOGRAPHY["weight_medium"]};
            }}
            GhostButton:hover {{
                background-color: {COLORS["bg_elevated"]};
            }}
            GhostButton:pressed {{
                background-color: {COLORS["bg_overlay"]};
            }}
            GhostButton:disabled {{
                color: {COLORS["text_tertiary"]};
            }}
            GhostButton[size="small"] {{
                padding: 4px {BUTTON["padding_x_sm"]}px;
                min-height: {BUTTON["height_sm"] - 8}px;
                font-size: {TYPOGRAPHY["size_sm"]}pt;
            }}
            GhostButton[size="large"] {{
                padding: 8px {BUTTON["padding_x_lg"]}px;
                min-height: {BUTTON["height_lg"] - 16}px;
                font-size: {TYPOGRAPHY["size_lg"]}pt;
            }}
        """)


class OutlineButton(BaseButton):
    """
    Outline button (border only, no fill).

    Use for secondary actions that need more emphasis than ghost buttons.
    """

    def __init__(
        self,
        text: str = "",
        icon: Optional[QIcon] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(text, icon, parent)
        self._apply_base_style()

    def _apply_base_style(self):
        self.setStyleSheet(f"""
            OutlineButton {{
                background-color: transparent;
                color: {COLORS["text_primary"]};
                border: 1px solid {COLORS["border_default"]};
                border-radius: {RADIUS["md"]}px;
                padding: 6px {BUTTON["padding_x_md"]}px;
                min-height: {BUTTON["height_md"] - 12}px;
                font-weight: {TYPOGRAPHY["weight_medium"]};
            }}
            OutlineButton:hover {{
                background-color: {COLORS["bg_elevated"]};
                border-color: {COLORS["border_emphasis"]};
            }}
            OutlineButton:pressed {{
                background-color: {COLORS["bg_overlay"]};
            }}
            OutlineButton:disabled {{
                color: {COLORS["text_tertiary"]};
                border-color: {COLORS["border_subtle"]};
            }}
            OutlineButton[size="small"] {{
                padding: 4px {BUTTON["padding_x_sm"]}px;
                min-height: {BUTTON["height_sm"] - 8}px;
                font-size: {TYPOGRAPHY["size_sm"]}pt;
            }}
            OutlineButton[size="large"] {{
                padding: 8px {BUTTON["padding_x_lg"]}px;
                min-height: {BUTTON["height_lg"] - 16}px;
                font-size: {TYPOGRAPHY["size_lg"]}pt;
            }}
        """)


class IconButton(BaseButton):
    """
    Square icon-only button.

    Use for toolbar actions, close buttons, etc.
    """

    def __init__(
        self,
        icon: QIcon,
        tooltip: str = "",
        size: int = BUTTON["height_md"],
        parent: Optional[QWidget] = None,
    ):
        super().__init__("", icon, parent)
        self._button_size = size
        self.setFixedSize(size, size)
        self.setIconSize(QSize(size - 8, size - 8))

        if tooltip:
            self.setToolTip(tooltip)

        self._apply_base_style()

    def _apply_base_style(self):
        self.setStyleSheet(f"""
            IconButton {{
                background-color: transparent;
                color: {COLORS["text_primary"]};
                border: none;
                border-radius: {RADIUS["md"]}px;
                padding: 0px;
            }}
            IconButton:hover {{
                background-color: {COLORS["bg_elevated"]};
            }}
            IconButton:pressed {{
                background-color: {COLORS["bg_overlay"]};
            }}
            IconButton:disabled {{
                color: {COLORS["text_tertiary"]};
            }}
        """)


class ButtonGroup(QWidget):
    """
    A horizontal group of buttons with connected appearance.

    Use for toggle buttons, segmented controls.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._buttons: list[QPushButton] = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

    def add_button(self, button: QPushButton):
        """Add a button to the group."""
        self._buttons.append(button)
        self.layout().addWidget(button)
        self._update_styles()

    def _update_styles(self):
        """Update styles for connected appearance."""
        for i, btn in enumerate(self._buttons):
            # Determine position for border radius
            if len(self._buttons) == 1:
                radius = f"{RADIUS['md']}px"
            elif i == 0:
                radius = f"{RADIUS['md']}px 0 0 {RADIUS['md']}px"
            elif i == len(self._buttons) - 1:
                radius = f"0 {RADIUS['md']}px {RADIUS['md']}px 0"
            else:
                radius = "0"

            # Remove right border except for last button
            border_right = "none" if i < len(self._buttons) - 1 else f"1px solid {COLORS['border_default']}"

            btn.setStyleSheet(btn.styleSheet() + f"""
                QPushButton {{
                    border-radius: {radius};
                    border-right: {border_right};
                }}
            """)


class ToolbarButton(IconButton):
    """
    Specialized icon button for toolbars.

    Slightly smaller with no padding.
    """

    def __init__(
        self,
        icon: QIcon,
        tooltip: str = "",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(icon, tooltip, 28, parent)


class ToggleButton(BaseButton):
    """
    A button that toggles between on/off states.

    Visually indicates the current state.
    """

    def __init__(
        self,
        text: str = "",
        icon: Optional[QIcon] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(text, icon, parent)
        self.setCheckable(True)
        self._apply_base_style()

    def _apply_base_style(self):
        self.setStyleSheet(f"""
            ToggleButton {{
                background-color: {COLORS["bg_elevated"]};
                color: {COLORS["text_primary"]};
                border: 1px solid {COLORS["border_default"]};
                border-radius: {RADIUS["md"]}px;
                padding: 6px {BUTTON["padding_x_md"]}px;
                min-height: {BUTTON["height_md"] - 12}px;
                font-weight: {TYPOGRAPHY["weight_medium"]};
            }}
            ToggleButton:hover {{
                background-color: {COLORS["bg_overlay"]};
                border-color: {COLORS["border_emphasis"]};
            }}
            ToggleButton:checked {{
                background-color: {COLORS["accent_secondary"]};
                color: #ffffff;
                border-color: {COLORS["accent_secondary"]};
            }}
            ToggleButton:checked:hover {{
                background-color: {COLORS["accent_secondary_hover"]};
                border-color: {COLORS["accent_secondary_hover"]};
            }}
            ToggleButton:disabled {{
                background-color: {COLORS["bg_surface"]};
                color: {COLORS["text_tertiary"]};
                border-color: {COLORS["border_subtle"]};
            }}
        """)
