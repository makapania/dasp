"""
File Drop Widget - Drag-and-drop file loading with format detection.
"""

from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent


class FileDropWidget(QWidget):
    """
    A drag-and-drop widget for loading spectral data files.

    Emits file_selected signal when a file is chosen via browse or drop.
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

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Drop zone
        self.drop_zone = QWidget()
        self.drop_zone.setObjectName("dropZone")
        drop_layout = QVBoxLayout(self.drop_zone)
        drop_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Icon and text
        self.icon_label = QLabel("📁")
        self.icon_label.setStyleSheet("font-size: 32px;")
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_layout.addWidget(self.icon_label)

        self.text_label = QLabel("Drop spectral file here\nor click to browse")
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label.setStyleSheet("color: #aaa; font-size: 14px;")
        drop_layout.addWidget(self.text_label)

        # Supported formats hint
        self.formats_label = QLabel("CSV, Excel, ASD, SPC, JCAMP...")
        self.formats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.formats_label.setStyleSheet("color: #666; font-size: 11px;")
        drop_layout.addWidget(self.formats_label)

        layout.addWidget(self.drop_zone)

        # Styling
        self.drop_zone.setMinimumHeight(100)
        self.drop_zone.setStyleSheet("""
            #dropZone {
                background-color: #2d2d2d;
                border: 2px dashed #555;
                border-radius: 8px;
            }
            #dropZone:hover {
                border-color: #0078d4;
                background-color: #333;
            }
        """)

        # Make the whole widget clickable
        self.drop_zone.mousePressEvent = self._on_click

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
                self.drop_zone.setStyleSheet("""
                    #dropZone {
                        background-color: #1a3a1a;
                        border: 2px dashed #4caf50;
                        border-radius: 8px;
                    }
                """)
                return
        event.ignore()

    def dragLeaveEvent(self, event):
        """Handle drag leave."""
        self._reset_style()

    def dropEvent(self, event: QDropEvent):
        """Handle file drop."""
        self._reset_style()

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

        # Update display
        self.icon_label.setText("✓")
        self.icon_label.setStyleSheet("font-size: 32px; color: #4caf50;")
        self.text_label.setText(file_name)
        self.text_label.setStyleSheet("color: #e0e0e0; font-size: 14px; font-weight: bold;")
        self.formats_label.setText("Click to change file")

        # Emit signal
        self.file_selected.emit(file_path)

    def _reset_style(self):
        """Reset to default style."""
        self.drop_zone.setStyleSheet("""
            #dropZone {
                background-color: #2d2d2d;
                border: 2px dashed #555;
                border-radius: 8px;
            }
            #dropZone:hover {
                border-color: #0078d4;
                background-color: #333;
            }
        """)

    def clear(self):
        """Clear the current selection."""
        self._current_file = None
        self.icon_label.setText("📁")
        self.icon_label.setStyleSheet("font-size: 32px;")
        self.text_label.setText("Drop spectral file here\nor click to browse")
        self.text_label.setStyleSheet("color: #aaa; font-size: 14px;")
        self.formats_label.setText("CSV, Excel, ASD, SPC, JCAMP...")
        self._reset_style()

    @property
    def current_file(self) -> str:
        return self._current_file

    def set_file(self, file_path: str):
        """Programmatically set the file."""
        if file_path and Path(file_path).exists():
            self._handle_file(file_path)
