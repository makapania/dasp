"""
Icon Management - Spectral Predict v2

Provides consistent icon access using Qt's built-in icons and SVG paths.
"""

from typing import Optional
from PySide6.QtWidgets import QStyle, QApplication
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor
from PySide6.QtCore import Qt, QSize
from PySide6.QtSvg import QSvgRenderer

from .tokens import COLORS, ThemeMode


class Icons:
    """
    Icon provider for the application.

    Uses Qt's built-in standard icons with optional SVG override paths.
    All icons are theme-aware and can adapt to dark/light modes.
    """

    # ==========================================================================
    # SVG ICON PATHS (inline SVG for custom icons)
    # ==========================================================================

    # These are simple, flat SVG icons optimized for 16x16 or 24x24 display
    SVG_ICONS = {
        "plus": '''<svg viewBox="0 0 16 16" fill="currentColor">
            <path d="M8 2a.75.75 0 0 1 .75.75v4.5h4.5a.75.75 0 0 1 0 1.5h-4.5v4.5a.75.75 0 0 1-1.5 0v-4.5h-4.5a.75.75 0 0 1 0-1.5h4.5v-4.5A.75.75 0 0 1 8 2Z"/>
        </svg>''',

        "minus": '''<svg viewBox="0 0 16 16" fill="currentColor">
            <path d="M2.75 7.25a.75.75 0 0 0 0 1.5h10.5a.75.75 0 0 0 0-1.5H2.75Z"/>
        </svg>''',

        "close": '''<svg viewBox="0 0 16 16" fill="currentColor">
            <path d="M3.72 3.72a.75.75 0 0 1 1.06 0L8 6.94l3.22-3.22a.749.749 0 0 1 1.275.326.749.749 0 0 1-.215.734L9.06 8l3.22 3.22a.749.749 0 0 1-.326 1.275.749.749 0 0 1-.734-.215L8 9.06l-3.22 3.22a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042L6.94 8 3.72 4.78a.75.75 0 0 1 0-1.06Z"/>
        </svg>''',

        "check": '''<svg viewBox="0 0 16 16" fill="currentColor">
            <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"/>
        </svg>''',

        "chevron_down": '''<svg viewBox="0 0 16 16" fill="currentColor">
            <path d="M12.78 5.22a.749.749 0 0 1 0 1.06l-4.25 4.25a.749.749 0 0 1-1.06 0L3.22 6.28a.749.749 0 1 1 1.06-1.06L8 8.939l3.72-3.719a.749.749 0 0 1 1.06 0Z"/>
        </svg>''',

        "chevron_up": '''<svg viewBox="0 0 16 16" fill="currentColor">
            <path d="M3.22 10.78a.749.749 0 0 1 0-1.06l4.25-4.25a.749.749 0 0 1 1.06 0l4.25 4.25a.749.749 0 1 1-1.06 1.06L8 7.061l-3.72 3.719a.749.749 0 0 1-1.06 0Z"/>
        </svg>''',

        "chevron_left": '''<svg viewBox="0 0 16 16" fill="currentColor">
            <path d="M9.78 12.78a.75.75 0 0 1-1.06 0L4.47 8.53a.75.75 0 0 1 0-1.06l4.25-4.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042L6.06 8l3.72 3.72a.75.75 0 0 1 0 1.06Z"/>
        </svg>''',

        "chevron_right": '''<svg viewBox="0 0 16 16" fill="currentColor">
            <path d="M6.22 3.22a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042L9.94 8 6.22 4.28a.75.75 0 0 1 0-1.06Z"/>
        </svg>''',

        "search": '''<svg viewBox="0 0 16 16" fill="currentColor">
            <path d="M10.68 11.74a6 6 0 0 1-7.922-8.982 6 6 0 0 1 8.982 7.922l3.04 3.04a.749.749 0 0 1-.326 1.275.749.749 0 0 1-.734-.215ZM11.5 7a4.499 4.499 0 1 0-8.997 0A4.499 4.499 0 0 0 11.5 7Z"/>
        </svg>''',

        "filter": '''<svg viewBox="0 0 16 16" fill="currentColor">
            <path d="M.75 3h14.5a.75.75 0 0 1 0 1.5H.75a.75.75 0 0 1 0-1.5ZM3 7.75A.75.75 0 0 1 3.75 7h8.5a.75.75 0 0 1 0 1.5h-8.5A.75.75 0 0 1 3 7.75Zm3 4a.75.75 0 0 1 .75-.75h2.5a.75.75 0 0 1 0 1.5h-2.5a.75.75 0 0 1-.75-.75Z"/>
        </svg>''',

        "copy": '''<svg viewBox="0 0 16 16" fill="currentColor">
            <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"/>
            <path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"/>
        </svg>''',

        "paste": '''<svg viewBox="0 0 16 16" fill="currentColor">
            <path d="M3.626 3.533a.249.249 0 0 0-.126.217v9.5c0 .138.112.25.25.25h8.5a.25.25 0 0 0 .25-.25v-9.5a.249.249 0 0 0-.126-.217.75.75 0 0 1 .752-1.298c.541.313.874.89.874 1.515v9.5A1.75 1.75 0 0 1 12.25 15h-8.5A1.75 1.75 0 0 1 2 13.25v-9.5c0-.625.333-1.202.874-1.515a.75.75 0 0 1 .752 1.298ZM5.75 1h4.5a.75.75 0 0 1 .75.75v3a.75.75 0 0 1-.75.75h-4.5A.75.75 0 0 1 5 4.75v-3A.75.75 0 0 1 5.75 1Zm.75 3h3V2.5h-3Z"/>
        </svg>''',

        "trash": '''<svg viewBox="0 0 16 16" fill="currentColor">
            <path d="M11 1.75V3h2.25a.75.75 0 0 1 0 1.5H2.75a.75.75 0 0 1 0-1.5H5V1.75C5 .784 5.784 0 6.75 0h2.5C10.216 0 11 .784 11 1.75ZM4.496 6.675l.66 6.6a.25.25 0 0 0 .249.225h5.19a.25.25 0 0 0 .249-.225l.66-6.6a.75.75 0 0 1 1.492.149l-.66 6.6A1.748 1.748 0 0 1 10.595 15h-5.19a1.75 1.75 0 0 1-1.741-1.575l-.66-6.6a.75.75 0 1 1 1.492-.15ZM6.5 1.75V3h3V1.75a.25.25 0 0 0-.25-.25h-2.5a.25.25 0 0 0-.25.25Z"/>
        </svg>''',

        "file": '''<svg viewBox="0 0 16 16" fill="currentColor">
            <path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"/>
        </svg>''',

        "folder": '''<svg viewBox="0 0 16 16" fill="currentColor">
            <path d="M1.75 1A1.75 1.75 0 0 0 0 2.75v10.5C0 14.216.784 15 1.75 15h12.5A1.75 1.75 0 0 0 16 13.25v-8.5A1.75 1.75 0 0 0 14.25 3H7.5a.25.25 0 0 1-.2-.1l-.9-1.2C6.07 1.26 5.55 1 5 1H1.75Z"/>
        </svg>''',

        "folder_open": '''<svg viewBox="0 0 16 16" fill="currentColor">
            <path d="M.513 1.513A1.75 1.75 0 0 1 1.75 1h3.5c.55 0 1.07.26 1.4.7l.9 1.2a.25.25 0 0 0 .2.1H13a1 1 0 0 1 1 1v.5H2.75a.75.75 0 0 0 0 1.5h11.978a1 1 0 0 1 .994 1.117L15 13.25A1.75 1.75 0 0 1 13.25 15H1.75A1.75 1.75 0 0 1 0 13.25V2.75c0-.464.184-.91.513-1.237Z"/>
        </svg>''',

        "save": '''<svg viewBox="0 0 16 16" fill="currentColor">
            <path d="M2.75 14A1.75 1.75 0 0 1 1 12.25v-8.5C1 2.784 1.784 2 2.75 2h5.586c.464 0 .909.184 1.237.513l3.914 3.914c.329.328.513.773.513 1.237v4.586A1.75 1.75 0 0 1 12.25 14H10v-2.5a1.5 1.5 0 0 0-1.5-1.5h-1A1.5 1.5 0 0 0 6 11.5V14Zm-.25-1.5h1v-2.5a3 3 0 0 1 3-3h1a3 3 0 0 1 3 3V14h1.75a.25.25 0 0 0 .25-.25V7.664a.25.25 0 0 0-.073-.177L8.263 3.573a.25.25 0 0 0-.177-.073H2.75a.25.25 0 0 0-.25.25v8.5c0 .138.112.25.25.25Z"/>
        </svg>''',

        "download": '''<svg viewBox="0 0 16 16" fill="currentColor">
            <path d="M2.75 14A1.75 1.75 0 0 1 1 12.25v-2.5a.75.75 0 0 1 1.5 0v2.5c0 .138.112.25.25.25h10.5a.25.25 0 0 0 .25-.25v-2.5a.75.75 0 0 1 1.5 0v2.5A1.75 1.75 0 0 1 13.25 14Z"/>
            <path d="M7.25 7.689V2a.75.75 0 0 1 1.5 0v5.689l1.97-1.969a.749.749 0 1 1 1.06 1.06l-3.25 3.25a.749.749 0 0 1-1.06 0L4.22 6.78a.749.749 0 1 1 1.06-1.06l1.97 1.969Z"/>
        </svg>''',

        "upload": '''<svg viewBox="0 0 16 16" fill="currentColor">
            <path d="M2.75 14A1.75 1.75 0 0 1 1 12.25v-2.5a.75.75 0 0 1 1.5 0v2.5c0 .138.112.25.25.25h10.5a.25.25 0 0 0 .25-.25v-2.5a.75.75 0 0 1 1.5 0v2.5A1.75 1.75 0 0 1 13.25 14Z"/>
            <path d="M11.78 4.72a.749.749 0 1 1-1.06 1.06L8.75 3.811V9.5a.75.75 0 0 1-1.5 0V3.811L5.28 5.78a.749.749 0 1 1-1.06-1.06l3.25-3.25a.749.749 0 0 1 1.06 0l3.25 3.25Z"/>
        </svg>''',

        "refresh": '''<svg viewBox="0 0 16 16" fill="currentColor">
            <path d="M1.705 8.005a.75.75 0 0 1 .834.656 5.5 5.5 0 0 0 9.592 2.97l-1.204-1.204a.25.25 0 0 1 .177-.427h3.646a.25.25 0 0 1 .25.25v3.646a.25.25 0 0 1-.427.177l-1.38-1.38A7.002 7.002 0 0 1 1.05 8.84a.75.75 0 0 1 .656-.834ZM8 2.5a5.487 5.487 0 0 0-4.131 1.869l1.204 1.204A.25.25 0 0 1 4.896 6H1.25A.25.25 0 0 1 1 5.75V2.104a.25.25 0 0 1 .427-.177l1.38 1.38A7.002 7.002 0 0 1 14.95 7.16a.75.75 0 0 1-1.49.178A5.5 5.5 0 0 0 8 2.5Z"/>
        </svg>''',

        "play": '''<svg viewBox="0 0 16 16" fill="currentColor">
            <path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0ZM1.5 8a6.5 6.5 0 1 0 13 0 6.5 6.5 0 0 0-13 0Zm4.879-2.773 4.264 2.559a.25.25 0 0 1 0 .428l-4.264 2.559A.25.25 0 0 1 6 10.559V5.442a.25.25 0 0 1 .379-.215Z"/>
        </svg>''',

        "stop": '''<svg viewBox="0 0 16 16" fill="currentColor">
            <path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0ZM1.5 8a6.5 6.5 0 1 0 13 0 6.5 6.5 0 0 0-13 0Zm4-2.25v4.5a.25.25 0 0 0 .25.25h4.5a.25.25 0 0 0 .25-.25v-4.5a.25.25 0 0 0-.25-.25h-4.5a.25.25 0 0 0-.25.25Z"/>
        </svg>''',

        "settings": '''<svg viewBox="0 0 16 16" fill="currentColor">
            <path d="M8 0a8.2 8.2 0 0 1 .701.031C6.444.095 4.851.758 4.851 1.75c0 .69.588 1.309 1.526 1.722.396.176.874.338 1.417.474.71.178 1.526.313 2.456.396.126.01.254.02.384.028.168 1.08.252 2.188.252 3.296 0 1.108-.084 2.216-.252 3.296-.13.008-.258.017-.384.028-.93.083-1.746.218-2.456.396-.543.136-1.021.298-1.417.474-.938.413-1.526 1.032-1.526 1.722 0 .993 1.593 1.656 3.85 1.719A8.2 8.2 0 0 1 8 16c-4.418 0-8-3.582-8-8s3.582-8 8-8Z"/>
            <path d="M8 0c4.418 0 8 3.582 8 8s-3.582 8-8 8a8.2 8.2 0 0 1-.701-.031c2.257-.063 3.85-.726 3.85-1.719 0-.69-.588-1.309-1.526-1.722a7.12 7.12 0 0 0-1.417-.474 18.45 18.45 0 0 0-2.456-.396 11.78 11.78 0 0 0-.384-.028C5.334 10.55 5.25 9.442 5.25 8.334c0-1.108.084-2.216.252-3.296.126-.011.254-.02.384-.028.93-.083 1.746-.218 2.456-.396.543-.136 1.021-.298 1.417-.474.938-.413 1.526-1.032 1.526-1.722 0-.993-1.593-1.656-3.85-1.719A8.2 8.2 0 0 1 8 0Z"/>
        </svg>''',

        "chart": '''<svg viewBox="0 0 16 16" fill="currentColor">
            <path d="M1.5 1.75V13.5h13.75a.75.75 0 0 1 0 1.5H.75a.75.75 0 0 1-.75-.75V1.75a.75.75 0 0 1 1.5 0Zm14.28 2.53-5.25 5.25a.75.75 0 0 1-1.06 0L7 7.06 4.28 9.78a.751.751 0 0 1-1.042-.018.751.751 0 0 1-.018-1.042l3.25-3.25a.75.75 0 0 1 1.06 0L10 7.94l4.72-4.72a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042Z"/>
        </svg>''',

        "table": '''<svg viewBox="0 0 16 16" fill="currentColor">
            <path d="M0 1.75C0 .784.784 0 1.75 0h12.5C15.216 0 16 .784 16 1.75v12.5A1.75 1.75 0 0 1 14.25 16H1.75A1.75 1.75 0 0 1 0 14.25ZM6.5 6.5v8h7.75a.25.25 0 0 0 .25-.25V6.5Zm8-1.5V1.75a.25.25 0 0 0-.25-.25H6.5V5ZM5 5V1.5H1.75a.25.25 0 0 0-.25.25V5Zm-3.5 1.5v7.75c0 .138.112.25.25.25H5v-8Z"/>
        </svg>''',

        "eye": '''<svg viewBox="0 0 16 16" fill="currentColor">
            <path d="M8 2c1.981 0 3.671.992 4.933 2.078 1.27 1.091 2.187 2.345 2.637 3.023a1.62 1.62 0 0 1 0 1.798c-.45.678-1.367 1.932-2.637 3.023C11.67 13.008 9.981 14 8 14c-1.981 0-3.671-.992-4.933-2.078C1.797 10.83.88 9.576.43 8.898a1.62 1.62 0 0 1 0-1.798c.45-.677 1.367-1.931 2.637-3.022C4.329 2.992 6.019 2 8 2ZM1.679 7.932a.12.12 0 0 0 0 .136c.411.622 1.241 1.75 2.366 2.717C5.176 11.758 6.527 12.5 8 12.5c1.473 0 2.825-.742 3.955-1.715 1.124-.967 1.954-2.096 2.366-2.717a.12.12 0 0 0 0-.136c-.412-.621-1.242-1.75-2.366-2.717C10.824 4.242 9.473 3.5 8 3.5c-1.473 0-2.825.742-3.955 1.715-1.124.967-1.954 2.096-2.366 2.717ZM8 10a2 2 0 1 1-.001-3.999A2 2 0 0 1 8 10Z"/>
        </svg>''',

        "eye_closed": '''<svg viewBox="0 0 16 16" fill="currentColor">
            <path d="M.143 2.31a.75.75 0 0 1 1.047-.167l14.5 10.5a.75.75 0 1 1-.88 1.214l-2.248-1.628C11.346 13.19 9.792 14 8 14c-1.981 0-3.67-.992-4.933-2.078C1.797 10.83.88 9.576.43 8.898a1.619 1.619 0 0 1 0-1.798c.39-.583 1.14-1.593 2.178-2.528L.976 3.357a.75.75 0 0 1-.833-1.047ZM4.138 6.582l1.687 1.222a2 2 0 0 0 2.376 2.376l1.687 1.223c-.657.413-1.39.597-1.888.597-1.473 0-2.824-.742-3.955-1.715C2.92 9.318 2.09 8.19 1.68 7.568a.12.12 0 0 1 0-.136c.337-.512.974-1.373 1.89-2.22l.569.411.001-.001ZM13.71 10.17l1.547 1.12c.95-.872 1.64-1.753 2.313-2.393a1.619 1.619 0 0 0 0-1.798c-.45-.678-1.367-1.932-2.637-3.023C13.67 2.992 11.98 2 10 2c-1.047 0-2.047.219-2.957.546L8.748 3.88C9.25 3.74 9.623 3.5 10 3.5c1.473 0 2.824.742 3.955 1.715 1.124.967 1.954 2.096 2.366 2.717a.12.12 0 0 1 0 .136c-.256.387-.693.995-1.316 1.645l.002.002-1.297.954Z"/>
        </svg>''',

        "info": '''<svg viewBox="0 0 16 16" fill="currentColor">
            <path d="M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8Zm8-6.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13ZM6.5 7.75A.75.75 0 0 1 7.25 7h1a.75.75 0 0 1 .75.75v2.75h.25a.75.75 0 0 1 0 1.5h-2a.75.75 0 0 1 0-1.5h.25v-2h-.25a.75.75 0 0 1-.75-.75ZM8 6a1 1 0 1 1 0-2 1 1 0 0 1 0 2Z"/>
        </svg>''',

        "warning": '''<svg viewBox="0 0 16 16" fill="currentColor">
            <path d="M6.457 1.047c.659-1.234 2.427-1.234 3.086 0l6.082 11.378A1.75 1.75 0 0 1 14.082 15H1.918a1.75 1.75 0 0 1-1.543-2.575Zm1.763.707a.25.25 0 0 0-.44 0L1.698 13.132a.25.25 0 0 0 .22.368h12.164a.25.25 0 0 0 .22-.368Zm.53 3.996v2.5a.75.75 0 0 1-1.5 0v-2.5a.75.75 0 0 1 1.5 0ZM9 11a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z"/>
        </svg>''',

        "error": '''<svg viewBox="0 0 16 16" fill="currentColor">
            <path d="M2.343 13.657A8 8 0 1 1 13.658 2.343 8 8 0 0 1 2.343 13.657ZM6.03 4.97a.751.751 0 0 0-1.042.018.751.751 0 0 0-.018 1.042L6.94 8 4.97 9.97a.749.749 0 0 0 .326 1.275.749.749 0 0 0 .734-.215L8 9.06l1.97 1.97a.749.749 0 0 0 1.275-.326.749.749 0 0 0-.215-.734L9.06 8l1.97-1.97a.749.749 0 0 0-.326-1.275.749.749 0 0 0-.734.215L8 6.94Z"/>
        </svg>''',

        "success": '''<svg viewBox="0 0 16 16" fill="currentColor">
            <path d="M8 16A8 8 0 1 1 8 0a8 8 0 0 1 0 16Zm3.78-9.72a.751.751 0 0 0-.018-1.042.751.751 0 0 0-1.042-.018L6.75 9.19 5.28 7.72a.751.751 0 0 0-1.042.018.751.751 0 0 0-.018 1.042l2 2a.75.75 0 0 0 1.06 0Z"/>
        </svg>''',

        "fill_down": '''<svg viewBox="0 0 16 16" fill="currentColor">
            <path d="M3.5 2.75a.75.75 0 0 1 .75-.75h7.5a.75.75 0 0 1 0 1.5h-7.5a.75.75 0 0 1-.75-.75ZM3.5 6.75a.75.75 0 0 1 .75-.75h7.5a.75.75 0 0 1 0 1.5h-7.5a.75.75 0 0 1-.75-.75ZM4.25 10a.75.75 0 0 0 0 1.5h7.5a.75.75 0 0 0 0-1.5h-7.5Z"/>
            <path d="M8 7.5a.75.75 0 0 1 .75.75v4.69l1.22-1.22a.749.749 0 0 1 1.275.326.749.749 0 0 1-.215.734l-2.5 2.5a.75.75 0 0 1-1.06 0l-2.5-2.5a.749.749 0 0 1 .326-1.275.749.749 0 0 1 .734.215l1.22 1.22V8.25A.75.75 0 0 1 8 7.5Z"/>
        </svg>''',

        "column_add": '''<svg viewBox="0 0 16 16" fill="currentColor">
            <path d="M14.25 0A1.75 1.75 0 0 1 16 1.75v12.5A1.75 1.75 0 0 1 14.25 16H9.5V0ZM0 1.75C0 .784.784 0 1.75 0H6.5v16H1.75A1.75 1.75 0 0 1 0 14.25Zm8 6.5a.75.75 0 0 0 0 1.5h1.75v1.75a.75.75 0 0 0 1.5 0V9.75H13a.75.75 0 0 0 0-1.5h-1.75V6.5a.75.75 0 0 0-1.5 0v1.75Z"/>
        </svg>''',
    }

    @classmethod
    def get_svg_icon(
        cls,
        name: str,
        size: int = 16,
        color: Optional[str] = None,
        mode: ThemeMode = ThemeMode.DARK,
    ) -> QIcon:
        """
        Get an icon from SVG data.

        Args:
            name: Icon name from SVG_ICONS
            size: Icon size in pixels
            color: Override color (hex string). If None, uses theme text color.
            mode: Theme mode for default color

        Returns:
            QIcon instance
        """
        if name not in cls.SVG_ICONS:
            return QIcon()

        svg_data = cls.SVG_ICONS[name]

        # Determine color
        if color is None:
            colors = COLORS
            color = colors.get("text_primary", "#e6edf3")

        # Replace currentColor with actual color
        svg_data = svg_data.replace('fill="currentColor"', f'fill="{color}"')

        # Render SVG to pixmap
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)

        renderer = QSvgRenderer(svg_data.encode())
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()

        return QIcon(pixmap)

    @classmethod
    def get_standard_icon(cls, icon_type: QStyle.StandardPixmap) -> QIcon:
        """
        Get a Qt standard icon.

        Args:
            icon_type: Qt standard icon type

        Returns:
            QIcon instance
        """
        app = QApplication.instance()
        if app:
            return app.style().standardIcon(icon_type)
        return QIcon()

    # ==========================================================================
    # CONVENIENCE METHODS FOR COMMON ICONS
    # ==========================================================================

    @classmethod
    def plus(cls, size: int = 16, color: str = None) -> QIcon:
        return cls.get_svg_icon("plus", size, color)

    @classmethod
    def minus(cls, size: int = 16, color: str = None) -> QIcon:
        return cls.get_svg_icon("minus", size, color)

    @classmethod
    def close(cls, size: int = 16, color: str = None) -> QIcon:
        return cls.get_svg_icon("close", size, color)

    @classmethod
    def check(cls, size: int = 16, color: str = None) -> QIcon:
        return cls.get_svg_icon("check", size, color)

    @classmethod
    def search(cls, size: int = 16, color: str = None) -> QIcon:
        return cls.get_svg_icon("search", size, color)

    @classmethod
    def filter(cls, size: int = 16, color: str = None) -> QIcon:
        return cls.get_svg_icon("filter", size, color)

    @classmethod
    def copy(cls, size: int = 16, color: str = None) -> QIcon:
        return cls.get_svg_icon("copy", size, color)

    @classmethod
    def paste(cls, size: int = 16, color: str = None) -> QIcon:
        return cls.get_svg_icon("paste", size, color)

    @classmethod
    def trash(cls, size: int = 16, color: str = None) -> QIcon:
        return cls.get_svg_icon("trash", size, color)

    @classmethod
    def file(cls, size: int = 16, color: str = None) -> QIcon:
        return cls.get_svg_icon("file", size, color)

    @classmethod
    def folder(cls, size: int = 16, color: str = None) -> QIcon:
        return cls.get_svg_icon("folder", size, color)

    @classmethod
    def folder_open(cls, size: int = 16, color: str = None) -> QIcon:
        return cls.get_svg_icon("folder_open", size, color)

    @classmethod
    def save(cls, size: int = 16, color: str = None) -> QIcon:
        return cls.get_svg_icon("save", size, color)

    @classmethod
    def download(cls, size: int = 16, color: str = None) -> QIcon:
        return cls.get_svg_icon("download", size, color)

    @classmethod
    def upload(cls, size: int = 16, color: str = None) -> QIcon:
        return cls.get_svg_icon("upload", size, color)

    @classmethod
    def refresh(cls, size: int = 16, color: str = None) -> QIcon:
        return cls.get_svg_icon("refresh", size, color)

    @classmethod
    def play(cls, size: int = 16, color: str = None) -> QIcon:
        return cls.get_svg_icon("play", size, color)

    @classmethod
    def stop(cls, size: int = 16, color: str = None) -> QIcon:
        return cls.get_svg_icon("stop", size, color)

    @classmethod
    def settings(cls, size: int = 16, color: str = None) -> QIcon:
        return cls.get_svg_icon("settings", size, color)

    @classmethod
    def chart(cls, size: int = 16, color: str = None) -> QIcon:
        return cls.get_svg_icon("chart", size, color)

    @classmethod
    def table(cls, size: int = 16, color: str = None) -> QIcon:
        return cls.get_svg_icon("table", size, color)

    @classmethod
    def eye(cls, size: int = 16, color: str = None) -> QIcon:
        return cls.get_svg_icon("eye", size, color)

    @classmethod
    def eye_closed(cls, size: int = 16, color: str = None) -> QIcon:
        return cls.get_svg_icon("eye_closed", size, color)

    @classmethod
    def info(cls, size: int = 16, color: str = None) -> QIcon:
        return cls.get_svg_icon("info", size, color)

    @classmethod
    def warning(cls, size: int = 16, color: str = None) -> QIcon:
        return cls.get_svg_icon("warning", size, color)

    @classmethod
    def error(cls, size: int = 16, color: str = None) -> QIcon:
        return cls.get_svg_icon("error", size, color)

    @classmethod
    def success(cls, size: int = 16, color: str = None) -> QIcon:
        return cls.get_svg_icon("success", size, color)

    @classmethod
    def fill_down(cls, size: int = 16, color: str = None) -> QIcon:
        return cls.get_svg_icon("fill_down", size, color)

    @classmethod
    def column_add(cls, size: int = 16, color: str = None) -> QIcon:
        return cls.get_svg_icon("column_add", size, color)

    @classmethod
    def chevron_down(cls, size: int = 16, color: str = None) -> QIcon:
        return cls.get_svg_icon("chevron_down", size, color)

    @classmethod
    def chevron_up(cls, size: int = 16, color: str = None) -> QIcon:
        return cls.get_svg_icon("chevron_up", size, color)

    @classmethod
    def chevron_left(cls, size: int = 16, color: str = None) -> QIcon:
        return cls.get_svg_icon("chevron_left", size, color)

    @classmethod
    def chevron_right(cls, size: int = 16, color: str = None) -> QIcon:
        return cls.get_svg_icon("chevron_right", size, color)
