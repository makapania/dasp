"""
Design Tokens - Spectral Predict v2

Centralized design values for colors, spacing, typography, and effects.
Inspired by GitHub's Primer design system with a rich dashboard aesthetic.
"""

from enum import Enum
from typing import Dict, Any


class ThemeMode(Enum):
    """Theme mode selection."""
    DARK = "dark"
    LIGHT = "light"


# =============================================================================
# COLOR PALETTE
# =============================================================================

# Dark mode colors (primary theme)
COLORS_DARK: Dict[str, str] = {
    # Background layers (darkest to lightest)
    "bg_base": "#0d1117",           # Deepest background (window)
    "bg_surface": "#161b22",        # Card/panel backgrounds
    "bg_elevated": "#21262d",       # Hover states, input backgrounds
    "bg_overlay": "#30363d",        # Modals, dropdowns, tooltips
    "bg_highlight": "#388bfd26",    # Selection highlight (semi-transparent)

    # Border colors
    "border_subtle": "#21262d",     # Subtle separators
    "border_default": "#30363d",    # Standard borders
    "border_emphasis": "#8b949e",   # Emphasized borders (focus states)

    # Text colors
    "text_primary": "#e6edf3",      # Primary text
    "text_secondary": "#8b949e",    # Secondary/muted text
    "text_tertiary": "#6e7681",     # Tertiary/disabled text
    "text_link": "#58a6ff",         # Links
    "text_inverse": "#0d1117",      # Text on light backgrounds

    # Accent colors - Success/Primary (green)
    "accent_primary": "#238636",    # Primary action (buttons)
    "accent_primary_hover": "#2ea043",
    "accent_primary_active": "#238636",
    "accent_primary_subtle": "#238636",
    "accent_primary_muted": "#23863633",  # Background tint

    # Accent colors - Secondary (blue)
    "accent_secondary": "#1f6feb",
    "accent_secondary_hover": "#388bfd",
    "accent_secondary_muted": "#1f6feb33",

    # Accent colors - Warning (yellow/orange)
    "accent_warning": "#d29922",
    "accent_warning_hover": "#e3b341",
    "accent_warning_muted": "#d2992233",

    # Accent colors - Danger (red)
    "accent_danger": "#f85149",
    "accent_danger_hover": "#ff6e69",
    "accent_danger_muted": "#f8514933",

    # Accent colors - Info (purple)
    "accent_info": "#a371f7",
    "accent_info_muted": "#a371f733",

    # Chart/visualization colors (optimized for dark backgrounds)
    "chart_1": "#58a6ff",           # Blue
    "chart_2": "#3fb950",           # Green
    "chart_3": "#f78166",           # Coral/Orange
    "chart_4": "#bc8cff",           # Purple
    "chart_5": "#39d353",           # Bright green
    "chart_6": "#f0883e",           # Orange
    "chart_7": "#79c0ff",           # Light blue
    "chart_8": "#ffa657",           # Amber

    # Special purpose
    "scrollbar_bg": "#21262d",
    "scrollbar_thumb": "#484f58",
    "scrollbar_thumb_hover": "#6e7681",

    # Status colors
    "status_success": "#3fb950",
    "status_warning": "#d29922",
    "status_error": "#f85149",
    "status_info": "#58a6ff",
}

# Light mode colors
COLORS_LIGHT: Dict[str, str] = {
    # Background layers
    "bg_base": "#ffffff",
    "bg_surface": "#f6f8fa",
    "bg_elevated": "#ffffff",
    "bg_overlay": "#ffffff",
    "bg_highlight": "#0969da1a",

    # Border colors
    "border_subtle": "#d0d7de",
    "border_default": "#d0d7de",
    "border_emphasis": "#8c959f",

    # Text colors
    "text_primary": "#1f2328",
    "text_secondary": "#656d76",
    "text_tertiary": "#8c959f",
    "text_link": "#0969da",
    "text_inverse": "#ffffff",

    # Accent colors - Success/Primary
    "accent_primary": "#1a7f37",
    "accent_primary_hover": "#218c3f",
    "accent_primary_active": "#1a7f37",
    "accent_primary_subtle": "#dafbe1",
    "accent_primary_muted": "#1a7f3722",

    # Accent colors - Secondary
    "accent_secondary": "#0969da",
    "accent_secondary_hover": "#0c82e8",
    "accent_secondary_muted": "#0969da22",

    # Accent colors - Warning
    "accent_warning": "#9a6700",
    "accent_warning_hover": "#b07d16",
    "accent_warning_muted": "#9a670022",

    # Accent colors - Danger
    "accent_danger": "#cf222e",
    "accent_danger_hover": "#df3b3b",
    "accent_danger_muted": "#cf222e22",

    # Accent colors - Info
    "accent_info": "#8250df",
    "accent_info_muted": "#8250df22",

    # Chart colors (optimized for light backgrounds)
    "chart_1": "#0969da",
    "chart_2": "#1a7f37",
    "chart_3": "#bc4c00",
    "chart_4": "#8250df",
    "chart_5": "#1a7f37",
    "chart_6": "#bf5700",
    "chart_7": "#0550ae",
    "chart_8": "#953800",

    # Special purpose
    "scrollbar_bg": "#f6f8fa",
    "scrollbar_thumb": "#afb8c1",
    "scrollbar_thumb_hover": "#8c959f",

    # Status colors
    "status_success": "#1a7f37",
    "status_warning": "#9a6700",
    "status_error": "#cf222e",
    "status_info": "#0969da",
}

# Current color scheme (default to dark)
COLORS = COLORS_DARK.copy()


def set_theme_mode(mode: ThemeMode) -> None:
    """Switch the global color palette to the specified theme mode."""
    global COLORS
    if mode == ThemeMode.DARK:
        COLORS.clear()
        COLORS.update(COLORS_DARK)
    else:
        COLORS.clear()
        COLORS.update(COLORS_LIGHT)


def get_color(key: str, mode: ThemeMode = None) -> str:
    """
    Get a color value by key.

    Args:
        key: Color key from the palette
        mode: Optional theme mode override

    Returns:
        Hex color string
    """
    if mode is not None:
        palette = COLORS_DARK if mode == ThemeMode.DARK else COLORS_LIGHT
        return palette.get(key, "#ff00ff")  # Magenta for missing colors
    return COLORS.get(key, "#ff00ff")


# =============================================================================
# SPACING SCALE
# =============================================================================

SPACING: Dict[str, int] = {
    "none": 0,
    "2xs": 2,
    "xs": 4,
    "sm": 8,
    "md": 12,
    "lg": 16,
    "xl": 24,
    "2xl": 32,
    "3xl": 48,
    "4xl": 64,
}


def get_spacing(key: str) -> int:
    """Get spacing value in pixels."""
    return SPACING.get(key, 16)


# =============================================================================
# TYPOGRAPHY
# =============================================================================

TYPOGRAPHY: Dict[str, Any] = {
    # Font families
    "font_family": "Segoe UI, -apple-system, BlinkMacSystemFont, sans-serif",
    "font_mono": "Cascadia Code, Consolas, Monaco, monospace",

    # Font sizes (in points for Qt)
    "size_2xs": 9,
    "size_xs": 10,
    "size_sm": 11,
    "size_md": 12,      # Base size
    "size_lg": 14,
    "size_xl": 16,
    "size_2xl": 20,
    "size_3xl": 24,
    "size_4xl": 32,

    # Font weights
    "weight_normal": 400,
    "weight_medium": 500,
    "weight_semibold": 600,
    "weight_bold": 700,

    # Line heights
    "line_height_tight": 1.2,
    "line_height_normal": 1.5,
    "line_height_relaxed": 1.75,
}


# =============================================================================
# BORDER RADIUS
# =============================================================================

RADIUS: Dict[str, int] = {
    "none": 0,
    "sm": 3,
    "md": 6,
    "lg": 8,
    "xl": 12,
    "2xl": 16,
    "full": 9999,
}


# =============================================================================
# SHADOWS (for card depth effect)
# =============================================================================

SHADOWS: Dict[str, str] = {
    # No shadow
    "none": "none",

    # Subtle shadow for cards
    "sm": "0 1px 2px rgba(0, 0, 0, 0.3)",

    # Default card shadow
    "md": "0 3px 6px rgba(0, 0, 0, 0.16), 0 1px 2px rgba(0, 0, 0, 0.23)",

    # Elevated elements (dropdowns, modals)
    "lg": "0 10px 20px rgba(0, 0, 0, 0.19), 0 3px 6px rgba(0, 0, 0, 0.23)",

    # Highly elevated (tooltips, popovers)
    "xl": "0 14px 28px rgba(0, 0, 0, 0.25), 0 5px 10px rgba(0, 0, 0, 0.22)",

    # Focus ring
    "focus": "0 0 0 3px rgba(31, 111, 235, 0.4)",

    # Inset shadow for inputs
    "inset": "inset 0 1px 2px rgba(0, 0, 0, 0.2)",
}

# Qt-compatible shadow parameters for QGraphicsDropShadowEffect
SHADOW_PARAMS: Dict[str, Dict[str, float]] = {
    "none": {"blur": 0, "y": 0, "alpha": 0.0},
    "sm": {"blur": 2, "y": 1, "alpha": 0.3},
    "md": {"blur": 6, "y": 3, "alpha": 0.16},
    "lg": {"blur": 20, "y": 10, "alpha": 0.19},
    "xl": {"blur": 28, "y": 14, "alpha": 0.25},
}


# =============================================================================
# ANIMATION / TRANSITIONS
# =============================================================================

TRANSITIONS: Dict[str, str] = {
    "fast": "100ms",
    "normal": "200ms",
    "slow": "300ms",
    "ease": "ease-in-out",
}


# =============================================================================
# Z-INDEX LAYERS
# =============================================================================

Z_INDEX: Dict[str, int] = {
    "base": 0,
    "dropdown": 100,
    "sticky": 200,
    "overlay": 300,
    "modal": 400,
    "popover": 500,
    "tooltip": 600,
}


# =============================================================================
# COMPONENT-SPECIFIC TOKENS
# =============================================================================

BUTTON: Dict[str, Any] = {
    "height_sm": 28,
    "height_md": 32,
    "height_lg": 40,
    "padding_x_sm": 12,
    "padding_x_md": 16,
    "padding_x_lg": 20,
    "icon_size": 16,
}

INPUT: Dict[str, Any] = {
    "height_sm": 28,
    "height_md": 32,
    "height_lg": 40,
    "padding_x": 12,
    "border_width": 1,
}

CARD: Dict[str, Any] = {
    "padding": 16,
    "header_padding": 12,
    "border_radius": 8,
    "gap": 12,
}

TABLE: Dict[str, Any] = {
    "header_height": 36,
    "row_height": 32,
    "cell_padding_x": 12,
    "cell_padding_y": 8,
}
