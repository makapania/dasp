"""
Style Generator - Spectral Predict v2

Generates Qt stylesheets (QSS) from design tokens.
Provides a centralized, maintainable approach to styling.
"""

from typing import Optional
from .tokens import (
    COLORS,
    COLORS_DARK,
    COLORS_LIGHT,
    SPACING,
    TYPOGRAPHY,
    RADIUS,
    SHADOWS,
    BUTTON,
    INPUT,
    CARD,
    TABLE,
    ThemeMode,
)


class StyleGenerator:
    """
    Generates QSS stylesheets from design tokens.

    Usage:
        generator = StyleGenerator(ThemeMode.DARK)
        stylesheet = generator.generate_app_stylesheet()
        app.setStyleSheet(stylesheet)
    """

    def __init__(self, mode: ThemeMode = ThemeMode.DARK):
        self.mode = mode
        self.colors = COLORS_DARK if mode == ThemeMode.DARK else COLORS_LIGHT

    def c(self, key: str) -> str:
        """Shorthand to get color value."""
        return self.colors.get(key, "#ff00ff")

    def generate_app_stylesheet(self) -> str:
        """Generate the complete application stylesheet."""
        parts = [
            self._base_styles(),
            self._button_styles(),
            self._input_styles(),
            self._combobox_styles(),
            self._spinbox_styles(),
            self._slider_styles(),
            self._checkbox_styles(),
            self._table_styles(),
            self._scrollbar_styles(),
            self._groupbox_styles(),
            self._tabwidget_styles(),
            self._menu_styles(),
            self._tooltip_styles(),
            self._progressbar_styles(),
            self._splitter_styles(),
        ]
        return "\n\n".join(parts)

    def _base_styles(self) -> str:
        """Base widget styles."""
        return f"""
/* ============================================
   BASE STYLES
   ============================================ */

QWidget {{
    background-color: {self.c("bg_base")};
    color: {self.c("text_primary")};
    font-family: {TYPOGRAPHY["font_family"]};
    font-size: {TYPOGRAPHY["size_md"]}pt;
}}

QMainWindow {{
    background-color: {self.c("bg_base")};
}}

QFrame {{
    background-color: transparent;
    border: none;
}}

QFrame[frameShape="4"] {{
    /* HLine */
    background-color: {self.c("border_subtle")};
    max-height: 1px;
}}

QFrame[frameShape="5"] {{
    /* VLine */
    background-color: {self.c("border_subtle")};
    max-width: 1px;
}}

QLabel {{
    background-color: transparent;
    color: {self.c("text_primary")};
    padding: 0px;
}}

QLabel[class="secondary"] {{
    color: {self.c("text_secondary")};
}}

QLabel[class="heading"] {{
    font-size: {TYPOGRAPHY["size_lg"]}pt;
    font-weight: {TYPOGRAPHY["weight_semibold"]};
}}

QLabel[class="title"] {{
    font-size: {TYPOGRAPHY["size_xl"]}pt;
    font-weight: {TYPOGRAPHY["weight_bold"]};
}}

QStatusBar {{
    background-color: {self.c("bg_surface")};
    border-top: 1px solid {self.c("border_subtle")};
    color: {self.c("text_secondary")};
    padding: 4px 8px;
}}

QStatusBar::item {{
    border: none;
}}
"""

    def _button_styles(self) -> str:
        """Button styles with variants."""
        return f"""
/* ============================================
   BUTTONS
   ============================================ */

QPushButton {{
    background-color: {self.c("bg_elevated")};
    color: {self.c("text_primary")};
    border: 1px solid {self.c("border_default")};
    border-radius: {RADIUS["md"]}px;
    padding: 6px {BUTTON["padding_x_md"]}px;
    min-height: {BUTTON["height_md"] - 12}px;
    font-weight: {TYPOGRAPHY["weight_medium"]};
}}

QPushButton:hover {{
    background-color: {self.c("bg_overlay")};
    border-color: {self.c("border_emphasis")};
}}

QPushButton:pressed {{
    background-color: {self.c("bg_elevated")};
}}

QPushButton:disabled {{
    background-color: {self.c("bg_surface")};
    color: {self.c("text_tertiary")};
    border-color: {self.c("border_subtle")};
}}

QPushButton:focus {{
    border-color: {self.c("accent_secondary")};
    outline: none;
}}

/* Primary button (green) */
QPushButton[class="primary"] {{
    background-color: {self.c("accent_primary")};
    color: {self.c("text_inverse") if self.mode == ThemeMode.LIGHT else "#ffffff"};
    border: 1px solid {self.c("accent_primary")};
}}

QPushButton[class="primary"]:hover {{
    background-color: {self.c("accent_primary_hover")};
    border-color: {self.c("accent_primary_hover")};
}}

QPushButton[class="primary"]:pressed {{
    background-color: {self.c("accent_primary_active")};
}}

QPushButton[class="primary"]:disabled {{
    background-color: {self.c("accent_primary_muted")};
    border-color: transparent;
    color: {self.c("text_tertiary")};
}}

/* Secondary button (blue) */
QPushButton[class="secondary"] {{
    background-color: {self.c("accent_secondary")};
    color: #ffffff;
    border: 1px solid {self.c("accent_secondary")};
}}

QPushButton[class="secondary"]:hover {{
    background-color: {self.c("accent_secondary_hover")};
    border-color: {self.c("accent_secondary_hover")};
}}

/* Danger button (red) */
QPushButton[class="danger"] {{
    background-color: {self.c("accent_danger")};
    color: #ffffff;
    border: 1px solid {self.c("accent_danger")};
}}

QPushButton[class="danger"]:hover {{
    background-color: {self.c("accent_danger_hover")};
    border-color: {self.c("accent_danger_hover")};
}}

/* Ghost button (transparent) */
QPushButton[class="ghost"] {{
    background-color: transparent;
    border: none;
}}

QPushButton[class="ghost"]:hover {{
    background-color: {self.c("bg_elevated")};
}}

/* Icon button (square) */
QPushButton[class="icon"] {{
    padding: 6px;
    min-width: {BUTTON["height_md"]}px;
    max-width: {BUTTON["height_md"]}px;
}}

/* Small button */
QPushButton[size="small"] {{
    padding: 4px {BUTTON["padding_x_sm"]}px;
    min-height: {BUTTON["height_sm"] - 8}px;
    font-size: {TYPOGRAPHY["size_sm"]}pt;
}}

/* Large button */
QPushButton[size="large"] {{
    padding: 8px {BUTTON["padding_x_lg"]}px;
    min-height: {BUTTON["height_lg"] - 16}px;
    font-size: {TYPOGRAPHY["size_lg"]}pt;
}}
"""

    def _input_styles(self) -> str:
        """Input field styles."""
        return f"""
/* ============================================
   TEXT INPUTS
   ============================================ */

QLineEdit {{
    background-color: {self.c("bg_elevated")};
    color: {self.c("text_primary")};
    border: 1px solid {self.c("border_default")};
    border-radius: {RADIUS["md"]}px;
    padding: 6px {INPUT["padding_x"]}px;
    min-height: {INPUT["height_md"] - 14}px;
    selection-background-color: {self.c("accent_secondary")};
    selection-color: #ffffff;
}}

QLineEdit:hover {{
    border-color: {self.c("border_emphasis")};
}}

QLineEdit:focus {{
    border-color: {self.c("accent_secondary")};
    background-color: {self.c("bg_surface")};
}}

QLineEdit:disabled {{
    background-color: {self.c("bg_surface")};
    color: {self.c("text_tertiary")};
    border-color: {self.c("border_subtle")};
}}

QLineEdit[readOnly="true"] {{
    background-color: {self.c("bg_surface")};
}}

QTextEdit, QPlainTextEdit {{
    background-color: {self.c("bg_elevated")};
    color: {self.c("text_primary")};
    border: 1px solid {self.c("border_default")};
    border-radius: {RADIUS["md"]}px;
    padding: 8px;
    selection-background-color: {self.c("accent_secondary")};
}}

QTextEdit:focus, QPlainTextEdit:focus {{
    border-color: {self.c("accent_secondary")};
}}
"""

    def _combobox_styles(self) -> str:
        """Combobox/dropdown styles."""
        return f"""
/* ============================================
   COMBOBOX / DROPDOWN
   ============================================ */

QComboBox {{
    background-color: {self.c("bg_elevated")};
    color: {self.c("text_primary")};
    border: 1px solid {self.c("border_default")};
    border-radius: {RADIUS["md"]}px;
    padding: 6px {INPUT["padding_x"]}px;
    padding-right: 30px;
    min-height: {INPUT["height_md"] - 14}px;
}}

QComboBox:hover {{
    border-color: {self.c("border_emphasis")};
}}

QComboBox:focus {{
    border-color: {self.c("accent_secondary")};
}}

QComboBox:disabled {{
    background-color: {self.c("bg_surface")};
    color: {self.c("text_tertiary")};
}}

QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 24px;
    border: none;
    background: transparent;
}}

QComboBox::down-arrow {{
    width: 12px;
    height: 12px;
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {self.c("text_secondary")};
    margin-right: 8px;
}}

QComboBox::down-arrow:hover {{
    border-top-color: {self.c("text_primary")};
}}

QComboBox QAbstractItemView {{
    background-color: {self.c("bg_overlay")};
    color: {self.c("text_primary")};
    border: 1px solid {self.c("border_default")};
    border-radius: {RADIUS["md"]}px;
    padding: 4px;
    selection-background-color: {self.c("accent_secondary")};
    selection-color: #ffffff;
    outline: none;
}}

QComboBox QAbstractItemView::item {{
    padding: 6px 12px;
    min-height: 24px;
    border-radius: {RADIUS["sm"]}px;
}}

QComboBox QAbstractItemView::item:hover {{
    background-color: {self.c("bg_elevated")};
}}

QComboBox QAbstractItemView::item:selected {{
    background-color: {self.c("accent_secondary")};
}}
"""

    def _spinbox_styles(self) -> str:
        """Spinbox styles."""
        return f"""
/* ============================================
   SPINBOX
   ============================================ */

QSpinBox, QDoubleSpinBox {{
    background-color: {self.c("bg_elevated")};
    color: {self.c("text_primary")};
    border: 1px solid {self.c("border_default")};
    border-radius: {RADIUS["md"]}px;
    padding: 6px {INPUT["padding_x"]}px;
    padding-right: 20px;
    min-height: {INPUT["height_md"] - 14}px;
}}

QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: {self.c("border_emphasis")};
}}

QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {self.c("accent_secondary")};
}}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background-color: transparent;
    border: none;
    width: 16px;
}}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: {self.c("bg_overlay")};
}}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    width: 8px;
    height: 8px;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 4px solid {self.c("text_secondary")};
}}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    width: 8px;
    height: 8px;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 4px solid {self.c("text_secondary")};
}}
"""

    def _slider_styles(self) -> str:
        """Slider styles."""
        return f"""
/* ============================================
   SLIDER
   ============================================ */

QSlider::groove:horizontal {{
    background-color: {self.c("bg_overlay")};
    height: 6px;
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background-color: {self.c("accent_secondary")};
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}}

QSlider::handle:horizontal:hover {{
    background-color: {self.c("accent_secondary_hover")};
}}

QSlider::sub-page:horizontal {{
    background-color: {self.c("accent_secondary")};
    border-radius: 3px;
}}

QSlider::groove:vertical {{
    background-color: {self.c("bg_overlay")};
    width: 6px;
    border-radius: 3px;
}}

QSlider::handle:vertical {{
    background-color: {self.c("accent_secondary")};
    width: 16px;
    height: 16px;
    margin: 0 -5px;
    border-radius: 8px;
}}

QSlider::add-page:vertical {{
    background-color: {self.c("accent_secondary")};
    border-radius: 3px;
}}
"""

    def _checkbox_styles(self) -> str:
        """Checkbox and radio button styles."""
        return f"""
/* ============================================
   CHECKBOX & RADIO
   ============================================ */

QCheckBox {{
    spacing: 8px;
    color: {self.c("text_primary")};
}}

QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {self.c("border_default")};
    border-radius: {RADIUS["sm"]}px;
    background-color: {self.c("bg_elevated")};
}}

QCheckBox::indicator:hover {{
    border-color: {self.c("accent_secondary")};
}}

QCheckBox::indicator:checked {{
    background-color: {self.c("accent_secondary")};
    border-color: {self.c("accent_secondary")};
    image: none;
}}

QCheckBox::indicator:checked:after {{
    /* Checkmark would go here - using border trick */
}}

QCheckBox:disabled {{
    color: {self.c("text_tertiary")};
}}

QCheckBox::indicator:disabled {{
    background-color: {self.c("bg_surface")};
    border-color: {self.c("border_subtle")};
}}

QRadioButton {{
    spacing: 8px;
    color: {self.c("text_primary")};
}}

QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {self.c("border_default")};
    border-radius: 8px;
    background-color: {self.c("bg_elevated")};
}}

QRadioButton::indicator:hover {{
    border-color: {self.c("accent_secondary")};
}}

QRadioButton::indicator:checked {{
    background-color: {self.c("accent_secondary")};
    border-color: {self.c("accent_secondary")};
}}
"""

    def _table_styles(self) -> str:
        """Table and tree view styles."""
        return f"""
/* ============================================
   TABLES & TREE VIEWS
   ============================================ */

QTableView, QTableWidget {{
    background-color: {self.c("bg_surface")};
    color: {self.c("text_primary")};
    border: 1px solid {self.c("border_default")};
    border-radius: {RADIUS["md"]}px;
    gridline-color: {self.c("border_subtle")};
    selection-background-color: {self.c("accent_secondary_muted")};
    selection-color: {self.c("text_primary")};
    alternate-background-color: {self.c("bg_elevated")};
}}

QTableView::item, QTableWidget::item {{
    padding: {TABLE["cell_padding_y"]}px {TABLE["cell_padding_x"]}px;
    border: none;
}}

QTableView::item:selected, QTableWidget::item:selected {{
    background-color: {self.c("accent_secondary_muted")};
    color: {self.c("text_primary")};
}}

QTableView::item:hover, QTableWidget::item:hover {{
    background-color: {self.c("bg_elevated")};
}}

QHeaderView {{
    background-color: {self.c("bg_surface")};
}}

QHeaderView::section {{
    background-color: {self.c("bg_elevated")};
    color: {self.c("text_secondary")};
    padding: 8px {TABLE["cell_padding_x"]}px;
    border: none;
    border-bottom: 1px solid {self.c("border_default")};
    border-right: 1px solid {self.c("border_subtle")};
    font-weight: {TYPOGRAPHY["weight_semibold"]};
}}

QHeaderView::section:hover {{
    background-color: {self.c("bg_overlay")};
    color: {self.c("text_primary")};
}}

QHeaderView::section:first {{
    border-top-left-radius: {RADIUS["md"]}px;
}}

QHeaderView::section:last {{
    border-right: none;
    border-top-right-radius: {RADIUS["md"]}px;
}}

QTreeView {{
    background-color: {self.c("bg_surface")};
    color: {self.c("text_primary")};
    border: 1px solid {self.c("border_default")};
    border-radius: {RADIUS["md"]}px;
    alternate-background-color: {self.c("bg_elevated")};
}}

QTreeView::item {{
    padding: 4px 8px;
    border-radius: {RADIUS["sm"]}px;
}}

QTreeView::item:hover {{
    background-color: {self.c("bg_elevated")};
}}

QTreeView::item:selected {{
    background-color: {self.c("accent_secondary_muted")};
}}

QTreeView::branch {{
    background-color: transparent;
}}

QListView, QListWidget {{
    background-color: {self.c("bg_surface")};
    color: {self.c("text_primary")};
    border: 1px solid {self.c("border_default")};
    border-radius: {RADIUS["md"]}px;
    outline: none;
}}

QListView::item, QListWidget::item {{
    padding: 8px 12px;
    border-radius: {RADIUS["sm"]}px;
}}

QListView::item:hover, QListWidget::item:hover {{
    background-color: {self.c("bg_elevated")};
}}

QListView::item:selected, QListWidget::item:selected {{
    background-color: {self.c("accent_secondary_muted")};
}}
"""

    def _scrollbar_styles(self) -> str:
        """Scrollbar styles."""
        return f"""
/* ============================================
   SCROLLBARS
   ============================================ */

QScrollBar:vertical {{
    background-color: {self.c("scrollbar_bg")};
    width: 12px;
    margin: 0;
    border-radius: 6px;
}}

QScrollBar::handle:vertical {{
    background-color: {self.c("scrollbar_thumb")};
    min-height: 30px;
    border-radius: 6px;
    margin: 2px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {self.c("scrollbar_thumb_hover")};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
    background: none;
}}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none;
}}

QScrollBar:horizontal {{
    background-color: {self.c("scrollbar_bg")};
    height: 12px;
    margin: 0;
    border-radius: 6px;
}}

QScrollBar::handle:horizontal {{
    background-color: {self.c("scrollbar_thumb")};
    min-width: 30px;
    border-radius: 6px;
    margin: 2px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {self.c("scrollbar_thumb_hover")};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
    background: none;
}}

QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: none;
}}

/* Thin scrollbars for certain widgets */
QScrollBar[class="thin"]:vertical {{
    width: 8px;
}}

QScrollBar[class="thin"]:horizontal {{
    height: 8px;
}}
"""

    def _groupbox_styles(self) -> str:
        """Group box styles."""
        return f"""
/* ============================================
   GROUP BOX
   ============================================ */

QGroupBox {{
    background-color: {self.c("bg_surface")};
    border: 1px solid {self.c("border_default")};
    border-radius: {RADIUS["lg"]}px;
    margin-top: 16px;
    padding: 16px;
    padding-top: 24px;
    font-weight: {TYPOGRAPHY["weight_semibold"]};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 16px;
    top: 4px;
    background-color: {self.c("bg_surface")};
    padding: 2px 8px;
    color: {self.c("text_secondary")};
}}
"""

    def _tabwidget_styles(self) -> str:
        """Tab widget styles."""
        return f"""
/* ============================================
   TAB WIDGET
   ============================================ */

QTabWidget::pane {{
    background-color: {self.c("bg_surface")};
    border: 1px solid {self.c("border_default")};
    border-radius: {RADIUS["md"]}px;
    padding: 8px;
}}

QTabBar::tab {{
    background-color: {self.c("bg_elevated")};
    color: {self.c("text_secondary")};
    border: 1px solid {self.c("border_default")};
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: {RADIUS["md"]}px;
    border-top-right-radius: {RADIUS["md"]}px;
}}

QTabBar::tab:hover {{
    background-color: {self.c("bg_overlay")};
    color: {self.c("text_primary")};
}}

QTabBar::tab:selected {{
    background-color: {self.c("bg_surface")};
    color: {self.c("text_primary")};
    border-bottom-color: {self.c("bg_surface")};
}}
"""

    def _menu_styles(self) -> str:
        """Menu and menubar styles."""
        return f"""
/* ============================================
   MENUS
   ============================================ */

QMenuBar {{
    background-color: {self.c("bg_surface")};
    color: {self.c("text_primary")};
    border-bottom: 1px solid {self.c("border_subtle")};
    padding: 2px;
}}

QMenuBar::item {{
    background-color: transparent;
    padding: 6px 12px;
    border-radius: {RADIUS["sm"]}px;
}}

QMenuBar::item:hover {{
    background-color: {self.c("bg_elevated")};
}}

QMenuBar::item:selected {{
    background-color: {self.c("bg_elevated")};
}}

QMenu {{
    background-color: {self.c("bg_overlay")};
    color: {self.c("text_primary")};
    border: 1px solid {self.c("border_default")};
    border-radius: {RADIUS["md"]}px;
    padding: 4px;
}}

QMenu::item {{
    padding: 8px 32px 8px 12px;
    border-radius: {RADIUS["sm"]}px;
}}

QMenu::item:hover {{
    background-color: {self.c("accent_secondary")};
    color: #ffffff;
}}

QMenu::item:disabled {{
    color: {self.c("text_tertiary")};
}}

QMenu::separator {{
    height: 1px;
    background-color: {self.c("border_subtle")};
    margin: 4px 8px;
}}

QMenu::indicator {{
    width: 16px;
    height: 16px;
    margin-left: 8px;
}}
"""

    def _tooltip_styles(self) -> str:
        """Tooltip styles."""
        return f"""
/* ============================================
   TOOLTIPS
   ============================================ */

QToolTip {{
    background-color: {self.c("bg_overlay")};
    color: {self.c("text_primary")};
    border: 1px solid {self.c("border_default")};
    border-radius: {RADIUS["sm"]}px;
    padding: 6px 10px;
    font-size: {TYPOGRAPHY["size_sm"]}pt;
}}
"""

    def _progressbar_styles(self) -> str:
        """Progress bar styles."""
        return f"""
/* ============================================
   PROGRESS BAR
   ============================================ */

QProgressBar {{
    background-color: {self.c("bg_elevated")};
    border: none;
    border-radius: {RADIUS["sm"]}px;
    height: 8px;
    text-align: center;
}}

QProgressBar::chunk {{
    background-color: {self.c("accent_secondary")};
    border-radius: {RADIUS["sm"]}px;
}}

/* Thick progress bar variant */
QProgressBar[class="thick"] {{
    height: 20px;
    font-size: {TYPOGRAPHY["size_sm"]}pt;
}}
"""

    def _splitter_styles(self) -> str:
        """Splitter styles."""
        return f"""
/* ============================================
   SPLITTER
   ============================================ */

QSplitter::handle {{
    background-color: {self.c("border_subtle")};
}}

QSplitter::handle:horizontal {{
    width: 2px;
}}

QSplitter::handle:vertical {{
    height: 2px;
}}

QSplitter::handle:hover {{
    background-color: {self.c("accent_secondary")};
}}
"""


def get_app_stylesheet(mode: ThemeMode = ThemeMode.DARK) -> str:
    """
    Generate complete application stylesheet.

    Args:
        mode: Theme mode (DARK or LIGHT)

    Returns:
        Complete QSS stylesheet string
    """
    generator = StyleGenerator(mode)
    return generator.generate_app_stylesheet()


# =============================================================================
# UTILITY FUNCTIONS FOR COMPONENT-LEVEL STYLING
# =============================================================================

def card_style(
    mode: ThemeMode = ThemeMode.DARK,
    elevated: bool = False,
    padding: int = CARD["padding"],
) -> str:
    """Generate inline style for a card component."""
    colors = COLORS_DARK if mode == ThemeMode.DARK else COLORS_LIGHT
    bg = colors["bg_elevated"] if elevated else colors["bg_surface"]

    return f"""
        background-color: {bg};
        border: 1px solid {colors["border_default"]};
        border-radius: {CARD["border_radius"]}px;
        padding: {padding}px;
    """


def badge_style(
    variant: str = "default",
    mode: ThemeMode = ThemeMode.DARK,
) -> str:
    """Generate inline style for a badge/chip component."""
    colors = COLORS_DARK if mode == ThemeMode.DARK else COLORS_LIGHT

    variants = {
        "default": (colors["bg_elevated"], colors["text_primary"]),
        "success": (colors["accent_primary_muted"], colors["status_success"]),
        "warning": (colors["accent_warning_muted"], colors["accent_warning"]),
        "danger": (colors["accent_danger_muted"], colors["accent_danger"]),
        "info": (colors["accent_secondary_muted"], colors["accent_secondary"]),
    }

    bg, fg = variants.get(variant, variants["default"])

    return f"""
        background-color: {bg};
        color: {fg};
        border-radius: {RADIUS["full"]}px;
        padding: 2px 8px;
        font-size: {TYPOGRAPHY["size_xs"]}pt;
        font-weight: {TYPOGRAPHY["weight_medium"]};
    """
