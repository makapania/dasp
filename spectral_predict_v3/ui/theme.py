"""
Theme configuration for Spectral Predict v3.

Modern theme system with multiple color schemes optimized for Dear PyGui.
Each theme features carefully selected colors with proper contrast ratios,
cohesive accent colors, and professional aesthetics.

Available dark themes:
- Classic: Dark theme with green/blue accents (GitHub-inspired)
- Sakura: Soft rose/pink accents on warm dark background
- Matcha: Earthy green tones with nature-inspired warmth
- Ocean: Deep blue tones for a calm, professional look
- Nord: Arctic, bluish-gray palette (popular modern theme)

Available light themes:
- Light Classic: Clean white/gray with green/blue accents
- Light Sakura: Soft cream/blush with rose accents
- Light Ocean: Light blue-gray with cyan accents
"""

import dearpygui.dearpygui as dpg
from typing import Dict, Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class ThemeColors:
    """Color palette for a theme."""
    # Backgrounds (darkest to lightest)
    bg_base: Tuple[int, int, int]
    bg_surface: Tuple[int, int, int]
    bg_elevated: Tuple[int, int, int]
    bg_hover: Tuple[int, int, int]

    # Text
    text_primary: Tuple[int, int, int]
    text_secondary: Tuple[int, int, int]
    text_muted: Tuple[int, int, int]

    # Accents
    accent_primary: Tuple[int, int, int]
    accent_secondary: Tuple[int, int, int]
    accent_warning: Tuple[int, int, int]
    accent_error: Tuple[int, int, int]

    # Borders
    border_default: Tuple[int, int, int]
    border_focus: Tuple[int, int, int]

    # Selection & highlights
    selection_bg: Tuple[int, int, int]
    row_highlight: Tuple[int, int, int]
    flagged_row: Tuple[int, int, int]

    # Chart colors (8 distinct colors for data visualization)
    chart_colors: Tuple[Tuple[int, int, int], ...]


# =============================================================================
# Theme Definitions
# =============================================================================

THEME_CLASSIC = ThemeColors(
    # GitHub dark-inspired backgrounds
    bg_base=(13, 17, 23),
    bg_surface=(22, 27, 34),
    bg_elevated=(33, 38, 45),
    bg_hover=(48, 54, 61),

    # High contrast text
    text_primary=(240, 246, 252),
    text_secondary=(176, 184, 193),
    text_muted=(118, 131, 144),

    # Green primary, blue secondary
    accent_primary=(63, 185, 80),
    accent_secondary=(88, 166, 255),
    accent_warning=(210, 153, 34),
    accent_error=(248, 81, 73),

    border_default=(48, 54, 61),
    border_focus=(88, 166, 255),

    selection_bg=(48, 79, 112),
    row_highlight=(35, 45, 60),
    flagged_row=(80, 40, 40),

    chart_colors=(
        (88, 166, 255),   # Blue
        (63, 185, 80),    # Green
        (247, 129, 102),  # Coral
        (191, 134, 252),  # Purple
        (255, 203, 107),  # Yellow
        (255, 121, 198),  # Pink
        (121, 192, 255),  # Light blue
        (163, 230, 53),   # Lime
    )
)

THEME_SAKURA = ThemeColors(
    # Warm dark backgrounds with slight warmth
    bg_base=(20, 18, 22),
    bg_surface=(30, 27, 33),
    bg_elevated=(42, 38, 46),
    bg_hover=(55, 50, 60),

    # Warm white text
    text_primary=(250, 245, 248),
    text_secondary=(200, 190, 195),
    text_muted=(140, 130, 138),

    # Rose/pink primary, soft lavender secondary
    accent_primary=(235, 134, 155),
    accent_secondary=(180, 150, 200),
    accent_warning=(230, 175, 100),
    accent_error=(235, 95, 110),

    border_default=(60, 55, 65),
    border_focus=(235, 134, 155),

    selection_bg=(70, 50, 65),
    row_highlight=(45, 38, 50),
    flagged_row=(80, 45, 50),

    chart_colors=(
        (235, 134, 155),  # Rose
        (180, 150, 200),  # Lavender
        (255, 180, 140),  # Peach
        (150, 200, 180),  # Sage
        (255, 210, 160),  # Cream
        (200, 160, 210),  # Orchid
        (170, 210, 220),  # Sky
        (220, 180, 190),  # Blush
    )
)

THEME_MATCHA = ThemeColors(
    # Earthy dark backgrounds
    bg_base=(18, 22, 18),
    bg_surface=(26, 32, 26),
    bg_elevated=(36, 44, 36),
    bg_hover=(48, 58, 48),

    # Warm cream text
    text_primary=(245, 245, 235),
    text_secondary=(190, 195, 180),
    text_muted=(130, 140, 125),

    # Matcha green primary, warm gold secondary
    accent_primary=(130, 180, 110),
    accent_secondary=(200, 170, 100),
    accent_warning=(210, 165, 80),
    accent_error=(200, 95, 85),

    border_default=(55, 65, 55),
    border_focus=(130, 180, 110),

    selection_bg=(50, 70, 50),
    row_highlight=(35, 48, 35),
    flagged_row=(70, 45, 40),

    chart_colors=(
        (130, 180, 110),  # Matcha
        (200, 170, 100),  # Gold
        (170, 140, 110),  # Mocha
        (140, 180, 160),  # Sage
        (180, 200, 130),  # Lime
        (160, 150, 130),  # Stone
        (120, 160, 140),  # Teal
        (200, 190, 150),  # Sand
    )
)

THEME_OCEAN = ThemeColors(
    # Deep blue-gray backgrounds
    bg_base=(15, 20, 28),
    bg_surface=(22, 30, 42),
    bg_elevated=(32, 42, 56),
    bg_hover=(45, 58, 75),

    # Cool white text
    text_primary=(235, 245, 255),
    text_secondary=(170, 190, 210),
    text_muted=(110, 135, 160),

    # Cyan primary, soft blue secondary
    accent_primary=(80, 200, 200),
    accent_secondary=(100, 150, 220),
    accent_warning=(220, 180, 90),
    accent_error=(220, 100, 100),

    border_default=(50, 65, 85),
    border_focus=(80, 200, 200),

    selection_bg=(40, 60, 90),
    row_highlight=(30, 42, 60),
    flagged_row=(70, 45, 50),

    chart_colors=(
        (80, 200, 200),   # Cyan
        (100, 150, 220),  # Blue
        (160, 200, 130),  # Seafoam
        (130, 180, 220),  # Sky
        (200, 180, 100),  # Sand
        (180, 140, 200),  # Violet
        (100, 180, 180),  # Teal
        (220, 160, 140),  # Coral
    )
)

THEME_NORD = ThemeColors(
    # Nord polar night backgrounds
    bg_base=(36, 41, 51),
    bg_surface=(46, 52, 64),
    bg_elevated=(59, 66, 82),
    bg_hover=(67, 76, 94),

    # Snow storm text
    text_primary=(236, 239, 244),
    text_secondary=(216, 222, 233),
    text_muted=(155, 165, 180),

    # Nord frost colors
    accent_primary=(136, 192, 208),
    accent_secondary=(129, 161, 193),
    accent_warning=(235, 203, 139),
    accent_error=(191, 97, 106),

    border_default=(67, 76, 94),
    border_focus=(136, 192, 208),

    selection_bg=(76, 86, 106),
    row_highlight=(59, 66, 82),
    flagged_row=(90, 60, 65),

    chart_colors=(
        (136, 192, 208),  # Frost blue
        (163, 190, 140),  # Aurora green
        (235, 203, 139),  # Aurora yellow
        (208, 135, 112),  # Aurora orange
        (180, 142, 173),  # Aurora purple
        (191, 97, 106),   # Aurora red
        (129, 161, 193),  # Frost dark
        (143, 188, 187),  # Frost teal
    )
)


# =============================================================================
# Light Theme Definitions
# =============================================================================

THEME_LIGHT_CLASSIC = ThemeColors(
    # Clean white/gray backgrounds
    bg_base=(250, 251, 252),
    bg_surface=(255, 255, 255),
    bg_elevated=(245, 246, 248),
    bg_hover=(235, 237, 240),

    # Dark text for contrast
    text_primary=(36, 41, 47),
    text_secondary=(87, 96, 106),
    text_muted=(139, 148, 158),

    # GitHub-inspired green/blue accents
    accent_primary=(45, 164, 78),
    accent_secondary=(9, 105, 218),
    accent_warning=(191, 135, 0),
    accent_error=(207, 34, 46),

    border_default=(216, 222, 228),
    border_focus=(9, 105, 218),

    selection_bg=(218, 235, 255),
    row_highlight=(246, 248, 250),
    flagged_row=(255, 235, 233),

    chart_colors=(
        (9, 105, 218),    # Blue
        (45, 164, 78),    # Green
        (207, 34, 46),    # Red
        (130, 80, 223),   # Purple
        (191, 135, 0),    # Yellow
        (219, 97, 162),   # Pink
        (17, 170, 170),   # Teal
        (247, 153, 57),   # Orange
    )
)

THEME_LIGHT_SAKURA = ThemeColors(
    # Soft cream/blush backgrounds
    bg_base=(255, 250, 250),
    bg_surface=(255, 255, 255),
    bg_elevated=(252, 245, 248),
    bg_hover=(248, 235, 240),

    # Warm dark text
    text_primary=(60, 45, 50),
    text_secondary=(100, 80, 88),
    text_muted=(150, 130, 138),

    # Rose/pink accents
    accent_primary=(219, 85, 120),
    accent_secondary=(180, 100, 150),
    accent_warning=(210, 150, 70),
    accent_error=(200, 60, 70),

    border_default=(235, 220, 225),
    border_focus=(219, 85, 120),

    selection_bg=(255, 230, 238),
    row_highlight=(255, 248, 250),
    flagged_row=(255, 225, 225),

    chart_colors=(
        (219, 85, 120),   # Rose
        (180, 100, 150),  # Mauve
        (250, 160, 130),  # Peach
        (120, 160, 140),  # Sage
        (230, 180, 140),  # Apricot
        (160, 120, 180),  # Lavender
        (130, 180, 190),  # Sky
        (200, 140, 160),  # Blush
    )
)

THEME_LIGHT_OCEAN = ThemeColors(
    # Light blue-gray backgrounds
    bg_base=(248, 251, 255),
    bg_surface=(255, 255, 255),
    bg_elevated=(242, 247, 252),
    bg_hover=(230, 240, 250),

    # Cool dark text
    text_primary=(30, 45, 60),
    text_secondary=(70, 95, 120),
    text_muted=(120, 145, 170),

    # Cyan/blue accents
    accent_primary=(0, 150, 160),
    accent_secondary=(55, 120, 190),
    accent_warning=(200, 155, 50),
    accent_error=(200, 75, 75),

    border_default=(210, 225, 240),
    border_focus=(0, 150, 160),

    selection_bg=(220, 240, 250),
    row_highlight=(245, 250, 255),
    flagged_row=(255, 235, 235),

    chart_colors=(
        (0, 150, 160),    # Cyan
        (55, 120, 190),   # Blue
        (100, 170, 130),  # Seafoam
        (100, 155, 200),  # Sky
        (190, 160, 90),   # Sand
        (150, 110, 180),  # Violet
        (60, 160, 160),   # Teal
        (210, 130, 110),  # Coral
    )
)


# Theme registry
THEMES: Dict[str, ThemeColors] = {
    # Dark themes
    "Classic": THEME_CLASSIC,
    "Sakura": THEME_SAKURA,
    "Matcha": THEME_MATCHA,
    "Ocean": THEME_OCEAN,
    "Nord": THEME_NORD,
    # Light themes
    "Light Classic": THEME_LIGHT_CLASSIC,
    "Light Sakura": THEME_LIGHT_SAKURA,
    "Light Ocean": THEME_LIGHT_OCEAN,
}

# Current active colors (starts with Classic)
COLORS: Dict[str, Tuple[int, int, int]] = {}

# Theme change callbacks
_theme_callbacks: list = []


def _update_colors_dict(theme: ThemeColors):
    """Update the global COLORS dict from a theme."""
    global COLORS
    COLORS.clear()
    COLORS.update({
        "bg_base": theme.bg_base,
        "bg_surface": theme.bg_surface,
        "bg_elevated": theme.bg_elevated,
        "bg_hover": theme.bg_hover,
        "text": theme.text_primary,
        "text_primary": theme.text_primary,
        "text_secondary": theme.text_secondary,
        "text_muted": theme.text_muted,
        "accent_primary": theme.accent_primary,
        "accent_secondary": theme.accent_secondary,
        "accent_warning": theme.accent_warning,
        "accent_error": theme.accent_error,
        "border_default": theme.border_default,
        "border_focus": theme.border_focus,
        "selection_bg": theme.selection_bg,
        "row_highlight": theme.row_highlight,
        "flagged_row": theme.flagged_row,
    })
    # Add chart colors
    for i, color in enumerate(theme.chart_colors):
        COLORS[f"chart_{i+1}"] = color


# Initialize with Classic theme
_update_colors_dict(THEME_CLASSIC)


def rgb_to_rgba(rgb: tuple, alpha: int = 255) -> tuple:
    """Convert RGB tuple to RGBA."""
    return (*rgb, alpha)


def get_available_themes() -> list:
    """Return list of available theme names."""
    return list(THEMES.keys())


def get_current_theme_name() -> str:
    """Return name of the currently active theme."""
    for name, theme in THEMES.items():
        if theme.bg_base == COLORS.get("bg_base"):
            return name
    return "Classic"


def on_theme_change(callback: Callable[[str], None]):
    """Register a callback for theme changes."""
    _theme_callbacks.append(callback)


def _notify_theme_change(theme_name: str):
    """Notify all registered callbacks of theme change."""
    for callback in _theme_callbacks:
        try:
            callback(theme_name)
        except Exception:
            pass


def create_theme(theme_name: str = "Classic") -> int:
    """
    Create and return a DPG theme for the specified theme.

    Parameters
    ----------
    theme_name : str
        Name of theme: 'Classic', 'Sakura', 'Matcha', 'Ocean', 'Nord'

    Returns
    -------
    int
        Theme ID to bind with dpg.bind_theme()
    """
    theme_colors = THEMES.get(theme_name, THEME_CLASSIC)
    _update_colors_dict(theme_colors)

    with dpg.theme() as global_theme:
        # Global component styles
        with dpg.theme_component(dpg.mvAll):
            # Window backgrounds
            dpg.add_theme_color(
                dpg.mvThemeCol_WindowBg,
                rgb_to_rgba(theme_colors.bg_base)
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ChildBg,
                rgb_to_rgba(theme_colors.bg_surface)
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_PopupBg,
                rgb_to_rgba(theme_colors.bg_elevated)
            )

            # Frame backgrounds (inputs, combos, etc.)
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBg,
                rgb_to_rgba(theme_colors.bg_elevated)
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBgHovered,
                rgb_to_rgba(theme_colors.bg_hover)
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBgActive,
                rgb_to_rgba(theme_colors.border_focus)
            )

            # Buttons - use secondary accent for modern look
            btn_base = theme_colors.accent_secondary
            btn_hover = tuple(min(c + 25, 255) for c in btn_base)
            btn_active = tuple(max(c - 20, 0) for c in btn_base)

            dpg.add_theme_color(
                dpg.mvThemeCol_Button,
                rgb_to_rgba(btn_base)
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ButtonHovered,
                rgb_to_rgba(btn_hover)
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ButtonActive,
                rgb_to_rgba(btn_active)
            )

            # Text
            dpg.add_theme_color(
                dpg.mvThemeCol_Text,
                rgb_to_rgba(theme_colors.text_primary)
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TextDisabled,
                rgb_to_rgba(theme_colors.text_muted)
            )

            # Headers (collapsing headers, tree nodes)
            dpg.add_theme_color(
                dpg.mvThemeCol_Header,
                rgb_to_rgba(theme_colors.bg_elevated)
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_HeaderHovered,
                rgb_to_rgba(theme_colors.bg_hover)
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_HeaderActive,
                rgb_to_rgba(theme_colors.accent_secondary)
            )

            # Borders
            dpg.add_theme_color(
                dpg.mvThemeCol_Border,
                rgb_to_rgba(theme_colors.border_default)
            )

            # Tabs
            dpg.add_theme_color(
                dpg.mvThemeCol_Tab,
                rgb_to_rgba(theme_colors.bg_surface)
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TabHovered,
                rgb_to_rgba(theme_colors.bg_hover)
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TabActive,
                rgb_to_rgba(theme_colors.bg_elevated)
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TabUnfocused,
                rgb_to_rgba(theme_colors.bg_surface)
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TabUnfocusedActive,
                rgb_to_rgba(theme_colors.bg_elevated)
            )

            # Scrollbar
            dpg.add_theme_color(
                dpg.mvThemeCol_ScrollbarBg,
                rgb_to_rgba(theme_colors.bg_base)
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ScrollbarGrab,
                rgb_to_rgba(theme_colors.bg_hover)
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ScrollbarGrabHovered,
                rgb_to_rgba(theme_colors.text_muted)
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ScrollbarGrabActive,
                rgb_to_rgba(theme_colors.text_secondary)
            )

            # Checkboxes and radio buttons
            dpg.add_theme_color(
                dpg.mvThemeCol_CheckMark,
                rgb_to_rgba(theme_colors.accent_primary)
            )

            # Sliders
            dpg.add_theme_color(
                dpg.mvThemeCol_SliderGrab,
                rgb_to_rgba(theme_colors.accent_secondary)
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_SliderGrabActive,
                rgb_to_rgba(theme_colors.accent_primary)
            )

            # Separator
            dpg.add_theme_color(
                dpg.mvThemeCol_Separator,
                rgb_to_rgba(theme_colors.border_default)
            )

            # Title bar
            dpg.add_theme_color(
                dpg.mvThemeCol_TitleBg,
                rgb_to_rgba(theme_colors.bg_base)
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TitleBgActive,
                rgb_to_rgba(theme_colors.bg_surface)
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TitleBgCollapsed,
                rgb_to_rgba(theme_colors.bg_base)
            )

            # Menu bar
            dpg.add_theme_color(
                dpg.mvThemeCol_MenuBarBg,
                rgb_to_rgba(theme_colors.bg_surface)
            )

            # Resize grip
            dpg.add_theme_color(
                dpg.mvThemeCol_ResizeGrip,
                rgb_to_rgba(theme_colors.bg_hover)
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ResizeGripHovered,
                rgb_to_rgba(theme_colors.accent_secondary)
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ResizeGripActive,
                rgb_to_rgba(theme_colors.accent_primary)
            )

            # Style variables - modern rounded corners
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 8)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 6)

            # Comfortable spacing
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 10, 7)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 10, 7)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 14, 14)
            dpg.add_theme_style(dpg.mvStyleVar_IndentSpacing, 20)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarSize, 14)
            dpg.add_theme_style(dpg.mvStyleVar_GrabMinSize, 14)

        # Table-specific styles
        with dpg.theme_component(dpg.mvTable):
            dpg.add_theme_color(
                dpg.mvThemeCol_TableRowBg,
                rgb_to_rgba(theme_colors.bg_surface)
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TableRowBgAlt,
                rgb_to_rgba(theme_colors.bg_elevated)
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TableHeaderBg,
                rgb_to_rgba(theme_colors.bg_base)
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TableBorderStrong,
                rgb_to_rgba(theme_colors.border_default)
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TableBorderLight,
                rgb_to_rgba(theme_colors.bg_hover)
            )

        # Plot-specific styles
        with dpg.theme_component(dpg.mvPlot):
            dpg.add_theme_color(
                dpg.mvPlotCol_PlotBg,
                rgb_to_rgba(theme_colors.bg_surface)
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_PlotBorder,
                rgb_to_rgba(theme_colors.border_default)
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_FrameBg,
                rgb_to_rgba(theme_colors.bg_base)
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_LegendBg,
                rgb_to_rgba(theme_colors.bg_elevated)
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_LegendBorder,
                rgb_to_rgba(theme_colors.border_default)
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_LegendText,
                rgb_to_rgba(theme_colors.text_secondary)
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_TitleText,
                rgb_to_rgba(theme_colors.text_primary)
            )
            # Axis colors (use AxisText for compatibility)
            if hasattr(dpg, 'mvPlotCol_XAxisText'):
                dpg.add_theme_color(
                    dpg.mvPlotCol_XAxisText,
                    rgb_to_rgba(theme_colors.text_muted)
                )
                dpg.add_theme_color(
                    dpg.mvPlotCol_YAxisText,
                    rgb_to_rgba(theme_colors.text_muted)
                )
            dpg.add_theme_color(
                dpg.mvPlotCol_Selection,
                rgb_to_rgba(theme_colors.selection_bg, 128)
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_Crosshairs,
                rgb_to_rgba(theme_colors.accent_secondary, 180)
            )

    _notify_theme_change(theme_name)
    return global_theme


def create_button_theme(color: str = "primary") -> int:
    """
    Create a button theme with specific color.

    Parameters
    ----------
    color : str
        'primary' (green), 'secondary' (blue), 'warning', 'error'

    Returns
    -------
    int
        Theme ID
    """
    color_map = {
        "primary": COLORS.get("accent_primary", (63, 185, 80)),
        "secondary": COLORS.get("accent_secondary", (88, 166, 255)),
        "warning": COLORS.get("accent_warning", (210, 153, 34)),
        "error": COLORS.get("accent_error", (248, 81, 73)),
    }

    base_color = color_map.get(color, color_map["secondary"])

    # Create hover and active variants
    hover_color = tuple(min(c + 25, 255) for c in base_color)
    active_color = tuple(max(c - 20, 0) for c in base_color)

    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, rgb_to_rgba(base_color))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, rgb_to_rgba(hover_color))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, rgb_to_rgba(active_color))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)

    return theme


def create_accent_text_theme(accent: str = "primary") -> int:
    """
    Create a theme for accent-colored text.

    Parameters
    ----------
    accent : str
        'primary', 'secondary', 'warning', 'error'

    Returns
    -------
    int
        Theme ID
    """
    color_map = {
        "primary": COLORS.get("accent_primary", (63, 185, 80)),
        "secondary": COLORS.get("accent_secondary", (88, 166, 255)),
        "warning": COLORS.get("accent_warning", (210, 153, 34)),
        "error": COLORS.get("accent_error", (248, 81, 73)),
    }

    text_color = color_map.get(accent, color_map["primary"])

    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvText):
            dpg.add_theme_color(dpg.mvThemeCol_Text, rgb_to_rgba(text_color))

    return theme
