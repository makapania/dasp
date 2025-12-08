"""
File Drop Widget - Drag-and-drop file loading with format detection.

Enhanced with theme system and modern styling.
"""

from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QGraphicsDropShadowEffect
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QColor

from ..theme.tokens import COLORS, SPACING, RADIUS, TYPOGRAPHY
from ..theme.icons import Icons


class FileDropWidget(QWidget):
    """
    A drag-and-drop widget for loading spectral data files.

    Features:
    - Drag and drop files
    - Click to browse
    - Visual feedback on drag hover
    - Shows selected file name
    - Theme-aware styling
    """

    file_selected = Signal(str)  # Emits the file path

    SUPPORTED_EXTENSIONS = [
        '.csv', '.xlsx', '.xls',  # Tabular
        '.asd', '.sig',           # ASD
        '.spc',                   # SPC
        '.jdx', '.dx', '.jcm',    # JCAMP-DX
        '.txt', '.dat', '.dpt',   # ASCII
        '.sp',                    # PerkinElmer
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._current_file = None
        self._is_hovering = False

        self._setup_ui()
        self._apply_style()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Drop zone container
        self.drop_zone = QWidget()
        self.drop_zone.setObjectName("dropZone")
        drop_layout = QVBoxLayout(self.drop_zone)
        drop_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_layout.setSpacing(SPACING["sm"])
        drop_layout.setContentsMargins(SPACING["xl"], SPACING["lg"], SPACING["xl"], SPACING["lg"])

        # Upload icon
        self.icon_label = QLabel()
        self.icon_label.setPixmap(Icons.upload(48).pixmap(48, 48))
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_layout.addWidget(self.icon_label)

        # Main text
        self.text_label = QLabel("Drop spectral file here")
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label.setStyleSheet(f"""
            color: {COLORS["text_primary"]};
            font-size: {TYPOGRAPHY["size_lg"]}pt;
            font-weight: {TYPOGRAPHY["weight_medium"]};
        """)
        drop_layout.addWidget(self.text_label)

        # Secondary text
        self.secondary_label = QLabel("or click to browse")
        self.secondary_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.secondary_label.setStyleSheet(f"""
            color: {COLORS["text_secondary"]};
            font-size: {TYPOGRAPHY["size_md"]}pt;
        """)
        drop_layout.addWidget(self.secondary_label)

        # Supported formats
        self.formats_label = QLabel("CSV, Excel, ASD, SPC, JCAMP-DX, TXT")
        self.formats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.formats_label.setStyleSheet(f"""
            color: {COLORS["text_tertiary"]};
            font-size: {TYPOGRAPHY["size_sm"]}pt;
            margin-top: {SPACING["sm"]}px;
        """)
        drop_layout.addWidget(self.formats_label)

        layout.addWidget(self.drop_zone)

        # Make clickable
        self.drop_zone.mousePressEvent = self._on_click
        self.drop_zone.setCursor(Qt.CursorShape.PointingHandCursor)

    def _apply_style(self):
        self._update_style(False)

    def _update_style(self, is_drag_over: bool = False, is_success: bool = False):
        """Update styling based on state."""
        if is_success:
            bg = COLORS["accent_primary_muted"]
            border_color = COLORS["accent_primary"]
        elif is_drag_over:
            bg = COLORS["accent_secondary_muted"]
            border_color = COLORS["accent_secondary"]
        else:
            bg = COLORS["bg_elevated"]
            border_color = COLORS["border_default"]

        self.drop_zone.setStyleSheet(f"""
            #dropZone {{
                background-color: {bg};
                border: 2px dashed {border_color};
                border-radius: {RADIUS["lg"]}px;
            }}
            #dropZone:hover {{
                border-color: {COLORS["accent_secondary"]};
                background-color: {COLORS["bg_overlay"]};
            }}
        """)

    def _on_click(self, event):
        """Handle click to browse."""
        self.browse()

    def browse(self):
        """Open file browser dialog."""
        extensions = " ".join(f"*{ext}" for ext in self.SUPPORTED_EXTENSIONS)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Spectral Data",
            "",
            f"Spectral Files ({extensions});;All Files (*)"
        )
        if file_path:
            self._handle_file(file_path)

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and self._is_valid_file(urls[0].toLocalFile()):
                event.acceptProposedAction()
                self._update_style(is_drag_over=True)
                return
        event.ignore()

    def dragLeaveEvent(self, event):
        """Handle drag leave."""
        self._update_style(False)

    def dropEvent(self, event: QDropEvent):
        """Handle file drop."""
        self._update_style(False)

        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            file_path = url.toLocalFile()
            if self._is_valid_file(file_path):
                self._handle_file(file_path)
                event.acceptProposedAction()
                return

        event.ignore()

    def _is_valid_file(self, file_path: str) -> bool:
        """Check if file has a supported extension."""
        if not file_path:
            return False
        path = Path(file_path)
        if path.is_dir():
            return True  # Directories can contain ASD/SPC files
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def _handle_file(self, file_path: str):
        """Process the selected file."""
        self._current_file = file_path
        file_name = Path(file_path).name

        # Update to success state
        self._update_style(is_success=True)

        # Update icon to checkmark
        self.icon_label.setPixmap(Icons.success(48, COLORS["accent_primary"]).pixmap(48, 48))

        # Update text
        self.text_label.setText(file_name)
        self.text_label.setStyleSheet(f"""
            color: {COLORS["text_primary"]};
            font-size: {TYPOGRAPHY["size_lg"]}pt;
            font-weight: {TYPOGRAPHY["weight_semibold"]};
        """)

        self.secondary_label.setText("Click to change file")
        self.formats_label.setVisible(False)

        # Emit signal
        self.file_selected.emit(file_path)

    def clear(self):
        """Clear the current selection."""
        self._current_file = None

        # Reset icon
        self.icon_label.setPixmap(Icons.upload(48).pixmap(48, 48))

        # Reset text
        self.text_label.setText("Drop spectral file here")
        self.text_label.setStyleSheet(f"""
            color: {COLORS["text_primary"]};
            font-size: {TYPOGRAPHY["size_lg"]}pt;
            font-weight: {TYPOGRAPHY["weight_medium"]};
        """)

        self.secondary_label.setText("or click to browse")
        self.formats_label.setVisible(True)

        self._update_style(False)

    @property
    def current_file(self) -> str:
        return self._current_file

    def set_file(self, file_path: str):
        """Programmatically set the file."""
        if file_path and Path(file_path).exists():
            self._handle_file(file_path)


class CompactFileDropWidget(QWidget):
    """
    A compact version of the file drop widget for use in cards.

    Shows a single row with icon, filename, and browse button.
    """

    file_selected = Signal(str)

    SUPPORTED_EXTENSIONS = FileDropWidget.SUPPORTED_EXTENSIONS

    def __init__(self, label: str = "Data file:", parent=None):
        super().__init__(parent)
        self._current_file = None
        self._label = label

        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING["sm"])

        # Label
        label = QLabel(self._label)
        label.setStyleSheet(f"""
            color: {COLORS["text_secondary"]};
            font-weight: {TYPOGRAPHY["weight_medium"]};
        """)
        layout.addWidget(label)

        # File name display
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet(f"""
            color: {COLORS["text_tertiary"]};
            padding: 4px 8px;
            background-color: {COLORS["bg_elevated"]};
            border: 1px solid {COLORS["border_default"]};
            border-radius: {RADIUS["sm"]}px;
        """)
        layout.addWidget(self.file_label, 1)

        # Browse button
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.setIcon(Icons.folder_open(14))
        self.browse_btn.clicked.connect(self.browse)
        self.browse_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS["bg_elevated"]};
                color: {COLORS["text_primary"]};
                border: 1px solid {COLORS["border_default"]};
                border-radius: {RADIUS["md"]}px;
                padding: 4px 12px;
            }}
            QPushButton:hover {{
                background-color: {COLORS["bg_overlay"]};
            }}
        """)
        layout.addWidget(self.browse_btn)

    def browse(self):
        """Open file browser dialog."""
        extensions = " ".join(f"*{ext}" for ext in self.SUPPORTED_EXTENSIONS)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Spectral Data",
            "",
            f"Spectral Files ({extensions});;All Files (*)"
        )
        if file_path:
            self._handle_file(file_path)

    def _handle_file(self, file_path: str):
        """Process the selected file."""
        self._current_file = file_path
        file_name = Path(file_path).name

        self.file_label.setText(file_name)
        self.file_label.setStyleSheet(f"""
            color: {COLORS["text_primary"]};
            padding: 4px 8px;
            background-color: {COLORS["bg_elevated"]};
            border: 1px solid {COLORS["accent_primary"]};
            border-radius: {RADIUS["sm"]}px;
        """)

        self.file_selected.emit(file_path)

    def clear(self):
        """Clear the current selection."""
        self._current_file = None
        self.file_label.setText("No file selected")
        self.file_label.setStyleSheet(f"""
            color: {COLORS["text_tertiary"]};
            padding: 4px 8px;
            background-color: {COLORS["bg_elevated"]};
            border: 1px solid {COLORS["border_default"]};
            border-radius: {RADIUS["sm"]}px;
        """)

    @property
    def current_file(self) -> str:
        return self._current_file

    def set_file(self, file_path: str):
        """Programmatically set the file."""
        if file_path and Path(file_path).exists():
            self._handle_file(file_path)
