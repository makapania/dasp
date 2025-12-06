"""
Theme configuration for Spectral Predict v3.

Dark theme with professional color palette, adapted from v2's tokens.py.
"""

import dearpygui.dearpygui as dpg


# Color palette - Dark theme
COLORS = {
    # Backgrounds
    "bg_base": (13, 17, 23),           # #0d1117 - Deepest background
    "bg_surface": (22, 27, 34),        # #161b22 - Card/panel background
    "bg_elevated": (33, 38, 45),       # #21262d - Elevated elements
    "bg_hover": (48, 54, 61),          # #30363d - Hover states

    # Text
    "text_primary": (255, 255, 255),   # White - Primary text
    "text_secondary": (176, 184, 193), # #b0b8c1 - Secondary text
    "text_muted": (118, 131, 144),     # #768390 - Muted/disabled text

    # Accent colors
    "accent_primary": (63, 185, 80),   # #3fb950 - Green (success, primary action)
    "accent_secondary": (88, 166, 255), # #58a6ff - Blue (links, secondary)
    "accent_warning": (210, 153, 34),  # #d29922 - Yellow/Orange (warnings)
    "accent_error": (248, 81, 73),     # #f85149 - Red (errors, destructive)

    # Borders
    "border_default": (48, 54, 61),    # #30363d - Default borders
    "border_focus": (88, 166, 255),    # #58a6ff - Focus state

    # Chart/Plot colors
    "chart_1": (88, 166, 255),         # Blue
    "chart_2": (63, 185, 80),          # Green
    "chart_3": (247, 129, 102),        # Coral
    "chart_4": (191, 134, 252),        # Purple
    "chart_5": (255, 203, 107),        # Yellow
    "chart_6": (255, 121, 198),        # Pink
    "chart_7": (121, 192, 255),        # Light blue
    "chart_8": (163, 230, 53),         # Lime

    # Selection
    "selection_bg": (48, 79, 112),     # Blue tinted selection
    "row_highlight": (35, 45, 60),     # Subtle row highlight
    "flagged_row": (80, 40, 40),       # Red tinted for flagged rows
}


def rgb_to_rgba(rgb: tuple, alpha: int = 255) -> tuple:
    """Convert RGB tuple to RGBA."""
    return (*rgb, alpha)


def create_theme() -> int:
    """
    Create and return the main application theme.

    Returns
    -------
    int
        Theme ID to bind with dpg.bind_theme()
    """
    with dpg.theme() as global_theme:
        # Global component styles
        with dpg.theme_component(dpg.mvAll):
            # Window backgrounds
            dpg.add_theme_color(
                dpg.mvThemeCol_WindowBg,
                rgb_to_rgba(COLORS["bg_base"])
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ChildBg,
                rgb_to_rgba(COLORS["bg_surface"])
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_PopupBg,
                rgb_to_rgba(COLORS["bg_elevated"])
            )

            # Frame backgrounds (inputs, combos, etc.)
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBg,
                rgb_to_rgba(COLORS["bg_elevated"])
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBgHovered,
                rgb_to_rgba(COLORS["bg_hover"])
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_FrameBgActive,
                rgb_to_rgba(COLORS["border_focus"])
            )

            # Buttons
            dpg.add_theme_color(
                dpg.mvThemeCol_Button,
                rgb_to_rgba(COLORS["accent_secondary"])
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ButtonHovered,
                rgb_to_rgba((110, 185, 255))  # Lighter blue
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ButtonActive,
                rgb_to_rgba((60, 145, 230))  # Darker blue
            )

            # Text
            dpg.add_theme_color(
                dpg.mvThemeCol_Text,
                rgb_to_rgba(COLORS["text_primary"])
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TextDisabled,
                rgb_to_rgba(COLORS["text_muted"])
            )

            # Headers (collapsing headers, tree nodes)
            dpg.add_theme_color(
                dpg.mvThemeCol_Header,
                rgb_to_rgba(COLORS["bg_elevated"])
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_HeaderHovered,
                rgb_to_rgba(COLORS["bg_hover"])
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_HeaderActive,
                rgb_to_rgba(COLORS["accent_secondary"])
            )

            # Borders
            dpg.add_theme_color(
                dpg.mvThemeCol_Border,
                rgb_to_rgba(COLORS["border_default"])
            )

            # Tabs
            dpg.add_theme_color(
                dpg.mvThemeCol_Tab,
                rgb_to_rgba(COLORS["bg_surface"])
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TabHovered,
                rgb_to_rgba(COLORS["bg_hover"])
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TabActive,
                rgb_to_rgba(COLORS["bg_elevated"])
            )

            # Scrollbar
            dpg.add_theme_color(
                dpg.mvThemeCol_ScrollbarBg,
                rgb_to_rgba(COLORS["bg_base"])
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ScrollbarGrab,
                rgb_to_rgba(COLORS["bg_hover"])
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ScrollbarGrabHovered,
                rgb_to_rgba(COLORS["text_muted"])
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_ScrollbarGrabActive,
                rgb_to_rgba(COLORS["text_secondary"])
            )

            # Checkboxes and radio buttons
            dpg.add_theme_color(
                dpg.mvThemeCol_CheckMark,
                rgb_to_rgba(COLORS["accent_primary"])
            )

            # Sliders
            dpg.add_theme_color(
                dpg.mvThemeCol_SliderGrab,
                rgb_to_rgba(COLORS["accent_secondary"])
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_SliderGrabActive,
                rgb_to_rgba(COLORS["accent_primary"])
            )

            # Separator
            dpg.add_theme_color(
                dpg.mvThemeCol_Separator,
                rgb_to_rgba(COLORS["border_default"])
            )

            # Title bar
            dpg.add_theme_color(
                dpg.mvThemeCol_TitleBg,
                rgb_to_rgba(COLORS["bg_base"])
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TitleBgActive,
                rgb_to_rgba(COLORS["bg_surface"])
            )

            # Menu bar
            dpg.add_theme_color(
                dpg.mvThemeCol_MenuBarBg,
                rgb_to_rgba(COLORS["bg_surface"])
            )

            # Style variables
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 4)

            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 6)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 6)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 12, 12)

        # Table-specific styles
        with dpg.theme_component(dpg.mvTable):
            dpg.add_theme_color(
                dpg.mvThemeCol_TableRowBg,
                rgb_to_rgba(COLORS["bg_surface"])
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TableRowBgAlt,
                rgb_to_rgba(COLORS["bg_elevated"])
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TableHeaderBg,
                rgb_to_rgba(COLORS["bg_base"])
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TableBorderStrong,
                rgb_to_rgba(COLORS["border_default"])
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_TableBorderLight,
                rgb_to_rgba(COLORS["bg_hover"])
            )

        # Plot-specific styles
        with dpg.theme_component(dpg.mvPlot):
            dpg.add_theme_color(
                dpg.mvPlotCol_PlotBg,
                rgb_to_rgba(COLORS["bg_surface"])
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_PlotBorder,
                rgb_to_rgba(COLORS["border_default"])
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_FrameBg,
                rgb_to_rgba(COLORS["bg_base"])
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_LegendBg,
                rgb_to_rgba(COLORS["bg_elevated"])
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_LegendBorder,
                rgb_to_rgba(COLORS["border_default"])
            )

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
        "primary": COLORS["accent_primary"],
        "secondary": COLORS["accent_secondary"],
        "warning": COLORS["accent_warning"],
        "error": COLORS["accent_error"],
    }

    base_color = color_map.get(color, COLORS["accent_secondary"])

    # Create hover and active variants
    hover_color = tuple(min(c + 30, 255) for c in base_color)
    active_color = tuple(max(c - 30, 0) for c in base_color)

    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, rgb_to_rgba(base_color))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, rgb_to_rgba(hover_color))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, rgb_to_rgba(active_color))

    return theme
