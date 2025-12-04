"""
Excel-like Data Grid - Spectral Predict v2

A spreadsheet-style widget for viewing and editing spectral data.

Features:
- View spectral data (samples x wavelengths) in spreadsheet form
- Edit individual cells
- Fill down (Ctrl+D)
- Copy/paste with Excel compatibility (tab-separated)
- Add/delete columns
- Row selection for outlier marking
- Column sorting and filtering
"""

from typing import Optional, List, Any, Tuple, Set, Dict
import numpy as np
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableView,
    QHeaderView,
    QAbstractItemView,
    QMenu,
    QToolBar,
    QLabel,
    QLineEdit,
    QComboBox,
    QPushButton,
    QMessageBox,
    QInputDialog,
    QApplication,
    QStyledItemDelegate,
    QStyleOptionViewItem,
)
from PySide6.QtCore import (
    Qt,
    Signal,
    QAbstractTableModel,
    QModelIndex,
    QSortFilterProxyModel,
    QItemSelectionModel,
    QSize,
)
from PySide6.QtGui import (
    QAction,
    QKeySequence,
    QColor,
    QPainter,
    QBrush,
    QClipboard,
)

from ..theme.tokens import COLORS, SPACING, TYPOGRAPHY
from ..theme.icons import Icons
from ..components.buttons import IconButton, GhostButton


class SpectralDataModel(QAbstractTableModel):
    """
    Table model for spectral data backed by numpy arrays.

    Handles:
    - Spectral data matrix (samples x wavelengths)
    - Sample IDs
    - Target values (optional)
    - Additional metadata columns
    - Edit tracking for undo support
    """

    data_modified = Signal()
    cell_edited = Signal(int, int, object)  # row, col, new_value

    # Column type constants
    COL_TYPE_ID = "id"
    COL_TYPE_TARGET = "target"
    COL_TYPE_WAVELENGTH = "wavelength"
    COL_TYPE_METADATA = "metadata"

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Core data
        self._spectral_data: Optional[np.ndarray] = None  # (n_samples, n_wavelengths)
        self._wavelengths: Optional[np.ndarray] = None
        self._sample_ids: List[str] = []
        self._target_values: Optional[np.ndarray] = None
        self._target_name: str = "Target"

        # Additional columns (metadata)
        self._metadata_columns: Dict[str, List[Any]] = {}  # col_name -> values

        # Column layout: [ID, Target?, Wavelengths..., Metadata...]
        self._column_types: List[str] = []
        self._column_names: List[str] = []

        # Edit tracking
        self._edits: Dict[Tuple[int, int], Any] = {}  # (row, col) -> original_value
        self._modified_cells: Set[Tuple[int, int]] = set()

        # Selection state
        self._flagged_rows: Set[int] = set()  # Outlier/flagged rows

    def set_data(
        self,
        spectral_data: np.ndarray,
        wavelengths: np.ndarray,
        sample_ids: Optional[List[str]] = None,
        target_values: Optional[np.ndarray] = None,
        target_name: str = "Target",
        metadata: Optional[dict[str, List[Any]]] = None,
    ):
        """
        Set the data to display.

        Args:
            spectral_data: 2D array (n_samples, n_wavelengths)
            wavelengths: 1D array of wavelength values
            sample_ids: List of sample identifiers
            target_values: Optional 1D array of target values
            target_name: Name of the target column
            metadata: Optional dict of additional columns
        """
        self.beginResetModel()

        self._spectral_data = spectral_data.copy()
        self._wavelengths = wavelengths.copy()
        self._target_values = target_values.copy() if target_values is not None else None
        self._target_name = target_name
        self._metadata_columns = metadata.copy() if metadata else {}

        n_samples = spectral_data.shape[0]

        # Generate sample IDs if not provided
        if sample_ids is None:
            self._sample_ids = [f"Sample_{i+1}" for i in range(n_samples)]
        else:
            self._sample_ids = list(sample_ids)

        # Build column layout
        self._build_column_layout()

        # Clear edit tracking
        self._edits.clear()
        self._modified_cells.clear()
        self._flagged_rows.clear()

        self.endResetModel()

    def _build_column_layout(self):
        """Build the column types and names lists."""
        self._column_types = []
        self._column_names = []

        # ID column (always first)
        self._column_types.append(self.COL_TYPE_ID)
        self._column_names.append("Sample ID")

        # Target column (if present)
        if self._target_values is not None:
            self._column_types.append(self.COL_TYPE_TARGET)
            self._column_names.append(self._target_name)

        # Wavelength columns
        if self._wavelengths is not None:
            for wl in self._wavelengths:
                self._column_types.append(self.COL_TYPE_WAVELENGTH)
                self._column_names.append(f"{wl:.1f}")

        # Metadata columns
        for col_name in self._metadata_columns:
            self._column_types.append(self.COL_TYPE_METADATA)
            self._column_names.append(col_name)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid() or self._spectral_data is None:
            return 0
        return self._spectral_data.shape[0]

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._column_names)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None

        row, col = index.row(), index.column()

        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            return self._get_cell_value(row, col)

        elif role == Qt.ItemDataRole.BackgroundRole:
            # Highlight modified cells
            if (row, col) in self._modified_cells:
                return QColor(COLORS["accent_warning_muted"])
            # Highlight flagged rows
            if row in self._flagged_rows:
                return QColor(COLORS["accent_danger_muted"])
            return None

        elif role == Qt.ItemDataRole.ForegroundRole:
            col_type = self._column_types[col]
            if col_type == self.COL_TYPE_ID:
                return QColor(COLORS["text_secondary"])
            return QColor(COLORS["text_primary"])

        elif role == Qt.ItemDataRole.TextAlignmentRole:
            col_type = self._column_types[col]
            if col_type == self.COL_TYPE_ID:
                return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter

        return None

    def _get_cell_value(self, row: int, col: int) -> Any:
        """Get the value at a specific cell."""
        col_type = self._column_types[col]

        if col_type == self.COL_TYPE_ID:
            return self._sample_ids[row]

        elif col_type == self.COL_TYPE_TARGET:
            if self._target_values is not None:
                return f"{self._target_values[row]:.4f}"
            return ""

        elif col_type == self.COL_TYPE_WAVELENGTH:
            # Calculate wavelength index
            wl_offset = 1 + (1 if self._target_values is not None else 0)
            wl_idx = col - wl_offset
            if 0 <= wl_idx < self._spectral_data.shape[1]:
                return f"{self._spectral_data[row, wl_idx]:.6f}"
            return ""

        elif col_type == self.COL_TYPE_METADATA:
            col_name = self._column_names[col]
            if col_name in self._metadata_columns:
                values = self._metadata_columns[col_name]
                if row < len(values):
                    return str(values[row])
            return ""

        return ""

    def setData(
        self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole
    ) -> bool:
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False

        row, col = index.row(), index.column()
        col_type = self._column_types[col]

        # Track original value for undo
        if (row, col) not in self._edits:
            self._edits[(row, col)] = self._get_cell_value(row, col)

        try:
            if col_type == self.COL_TYPE_ID:
                self._sample_ids[row] = str(value)

            elif col_type == self.COL_TYPE_TARGET:
                if self._target_values is not None:
                    self._target_values[row] = float(value)

            elif col_type == self.COL_TYPE_WAVELENGTH:
                wl_offset = 1 + (1 if self._target_values is not None else 0)
                wl_idx = col - wl_offset
                if 0 <= wl_idx < self._spectral_data.shape[1]:
                    self._spectral_data[row, wl_idx] = float(value)

            elif col_type == self.COL_TYPE_METADATA:
                col_name = self._column_names[col]
                if col_name in self._metadata_columns:
                    self._metadata_columns[col_name][row] = value

            self._modified_cells.add((row, col))
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole])
            self.cell_edited.emit(row, col, value)
            self.data_modified.emit()
            return True

        except (ValueError, IndexError):
            return False

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags

        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

        # All columns are editable except ID (optionally)
        col_type = self._column_types[index.column()]
        if col_type != self.COL_TYPE_ID:
            flags |= Qt.ItemFlag.ItemIsEditable

        return flags

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                if 0 <= section < len(self._column_names):
                    return self._column_names[section]
            else:
                return str(section + 1)
        return None

    # =========================================================================
    # FILL DOWN
    # =========================================================================

    def fill_down(self, start_row: int, end_row: int, column: int) -> bool:
        """
        Fill cells from start_row down to end_row with the start_row value.

        Args:
            start_row: Source row
            end_row: Last row to fill (inclusive)
            column: Column to fill

        Returns:
            True if successful
        """
        if not (0 <= start_row < end_row <= self.rowCount()):
            return False

        source_value = self._get_cell_value(start_row, column)

        for row in range(start_row + 1, end_row + 1):
            index = self.index(row, column)
            self.setData(index, source_value)

        return True

    def fill_selection(self, selection: List[Tuple[int, int]]) -> bool:
        """
        Fill selected cells with the value from the first selected cell.

        Args:
            selection: List of (row, col) tuples

        Returns:
            True if successful
        """
        if not selection:
            return False

        # Get source value from first cell
        source_row, source_col = selection[0]
        source_value = self._get_cell_value(source_row, source_col)

        # Fill all other cells
        for row, col in selection[1:]:
            index = self.index(row, col)
            self.setData(index, source_value)

        return True

    # =========================================================================
    # COPY / PASTE
    # =========================================================================

    def copy_selection(self, selection: List[Tuple[int, int]]) -> str:
        """
        Copy selected cells to tab-separated string (Excel compatible).

        Args:
            selection: List of (row, col) tuples

        Returns:
            Tab-separated string suitable for clipboard
        """
        if not selection:
            return ""

        # Find bounding box
        min_row = min(r for r, c in selection)
        max_row = max(r for r, c in selection)
        min_col = min(c for r, c in selection)
        max_col = max(c for r, c in selection)

        # Create selection set for fast lookup
        selection_set = set(selection)

        # Build tab-separated output
        lines = []
        for row in range(min_row, max_row + 1):
            cells = []
            for col in range(min_col, max_col + 1):
                if (row, col) in selection_set:
                    cells.append(str(self._get_cell_value(row, col)))
                else:
                    cells.append("")
            lines.append("\t".join(cells))

        return "\n".join(lines)

    def paste_from_clipboard(
        self, start_row: int, start_col: int, clipboard_text: str
    ) -> int:
        """
        Paste tab-separated clipboard text starting at the given cell.

        Args:
            start_row: Starting row
            start_col: Starting column
            clipboard_text: Tab-separated text from clipboard

        Returns:
            Number of cells modified
        """
        if not clipboard_text:
            return 0

        lines = clipboard_text.strip().split("\n")
        cells_modified = 0

        for row_offset, line in enumerate(lines):
            row = start_row + row_offset
            if row >= self.rowCount():
                break

            cells = line.split("\t")
            for col_offset, cell_value in enumerate(cells):
                col = start_col + col_offset
                if col >= self.columnCount():
                    break

                index = self.index(row, col)
                if self.setData(index, cell_value):
                    cells_modified += 1

        return cells_modified

    # =========================================================================
    # COLUMN OPERATIONS
    # =========================================================================

    def add_column(self, name: str, position: int = -1, default_value: Any = 0.0) -> bool:
        """
        Add a new metadata column.

        Args:
            name: Column name
            position: Position to insert (-1 for end)
            default_value: Default value for all rows

        Returns:
            True if successful
        """
        if name in self._metadata_columns or name in self._column_names:
            return False

        n_rows = self.rowCount()

        # Add to metadata
        self._metadata_columns[name] = [default_value] * n_rows

        # Rebuild column layout
        self.beginResetModel()
        self._build_column_layout()
        self.endResetModel()

        self.data_modified.emit()
        return True

    def delete_columns(self, columns: List[int]) -> bool:
        """
        Delete specified columns (only metadata columns can be deleted).

        Args:
            columns: List of column indices

        Returns:
            True if any columns were deleted
        """
        deleted = False

        for col in sorted(columns, reverse=True):
            if col < 0 or col >= len(self._column_types):
                continue

            col_type = self._column_types[col]
            col_name = self._column_names[col]

            # Only metadata columns can be deleted
            if col_type == self.COL_TYPE_METADATA:
                if col_name in self._metadata_columns:
                    del self._metadata_columns[col_name]
                    deleted = True

        if deleted:
            self.beginResetModel()
            self._build_column_layout()
            self.endResetModel()
            self.data_modified.emit()

        return deleted

    # =========================================================================
    # ROW OPERATIONS
    # =========================================================================

    def delete_rows(self, rows: List[int]) -> bool:
        """
        Delete specified rows.

        Args:
            rows: List of row indices to delete

        Returns:
            True if any rows were deleted
        """
        if not rows:
            return False

        rows = sorted(set(rows), reverse=True)

        self.beginResetModel()

        for row in rows:
            if row < 0 or row >= self._spectral_data.shape[0]:
                continue

            # Delete from spectral data
            self._spectral_data = np.delete(self._spectral_data, row, axis=0)

            # Delete from sample IDs
            del self._sample_ids[row]

            # Delete from target values
            if self._target_values is not None:
                self._target_values = np.delete(self._target_values, row)

            # Delete from metadata columns
            for col_name, values in self._metadata_columns.items():
                del values[row]

            # Update flagged rows
            new_flagged = set()
            for flagged_row in self._flagged_rows:
                if flagged_row < row:
                    new_flagged.add(flagged_row)
                elif flagged_row > row:
                    new_flagged.add(flagged_row - 1)
            self._flagged_rows = new_flagged

        self.endResetModel()
        self.data_modified.emit()
        return True

    def flag_rows(self, rows: List[int], flagged: bool = True):
        """
        Flag or unflag rows (for outlier marking).

        Args:
            rows: List of row indices
            flagged: True to flag, False to unflag
        """
        for row in rows:
            if flagged:
                self._flagged_rows.add(row)
            else:
                self._flagged_rows.discard(row)

        # Update display
        for row in rows:
            top_left = self.index(row, 0)
            bottom_right = self.index(row, self.columnCount() - 1)
            self.dataChanged.emit(top_left, bottom_right, [Qt.ItemDataRole.BackgroundRole])

    def get_flagged_rows(self) -> List[int]:
        """Get list of flagged row indices."""
        return sorted(self._flagged_rows)

    # =========================================================================
    # DATA ACCESS
    # =========================================================================

    def get_spectral_data(self) -> Optional[np.ndarray]:
        """Get the spectral data matrix."""
        return self._spectral_data.copy() if self._spectral_data is not None else None

    def get_target_values(self) -> Optional[np.ndarray]:
        """Get the target values array."""
        return self._target_values.copy() if self._target_values is not None else None

    def get_wavelengths(self) -> Optional[np.ndarray]:
        """Get the wavelengths array."""
        return self._wavelengths.copy() if self._wavelengths is not None else None

    def get_sample_ids(self) -> List[str]:
        """Get the sample IDs list."""
        return self._sample_ids.copy()

    def has_unsaved_changes(self) -> bool:
        """Check if there are unsaved modifications."""
        return len(self._modified_cells) > 0

    def clear_modified_state(self):
        """Clear the modified cells tracking."""
        old_cells = list(self._modified_cells)
        self._modified_cells.clear()
        self._edits.clear()

        # Update display for previously modified cells
        for row, col in old_cells:
            index = self.index(row, col)
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.BackgroundRole])


class SpectralDataFilterProxy(QSortFilterProxyModel):
    """
    Proxy model for filtering and sorting spectral data.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._filter_text = ""
        self._filter_column = -1  # -1 means filter all columns

    def set_filter(self, text: str, column: int = -1):
        """
        Set the filter text and column.

        Args:
            text: Filter text (case-insensitive substring match)
            column: Column to filter (-1 for all)
        """
        self._filter_text = text.lower()
        self._filter_column = column
        self.invalidateFilter()

    def filterAcceptsRow(
        self, source_row: int, source_parent: QModelIndex
    ) -> bool:
        if not self._filter_text:
            return True

        model = self.sourceModel()

        if self._filter_column >= 0:
            # Filter single column
            index = model.index(source_row, self._filter_column)
            value = str(model.data(index, Qt.ItemDataRole.DisplayRole) or "")
            return self._filter_text in value.lower()
        else:
            # Filter all columns
            for col in range(model.columnCount()):
                index = model.index(source_row, col)
                value = str(model.data(index, Qt.ItemDataRole.DisplayRole) or "")
                if self._filter_text in value.lower():
                    return True
            return False


class SpectralDataGrid(QWidget):
    """
    Complete Excel-like data grid widget.

    Features:
    - Toolbar with common actions
    - Table view with model
    - Keyboard shortcuts (Ctrl+C, Ctrl+V, Ctrl+D)
    - Context menu
    - Status bar with selection info
    """

    selection_changed = Signal(list)  # List of selected row indices
    data_modified = Signal()
    rows_flagged = Signal(list)  # Flagged row indices

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Models
        self._model = SpectralDataModel(self)
        self._proxy = SpectralDataFilterProxy(self)
        self._proxy.setSourceModel(self._model)

        # Connect signals
        self._model.data_modified.connect(self.data_modified.emit)

        self._setup_ui()
        self._setup_shortcuts()
        self._setup_context_menu()
        self._apply_style()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        self._toolbar = self._create_toolbar()
        layout.addWidget(self._toolbar)

        # Filter bar
        self._filter_bar = self._create_filter_bar()
        layout.addWidget(self._filter_bar)

        # Table view
        self._table = QTableView()
        self._table.setModel(self._proxy)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self._table.setSortingEnabled(True)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._show_context_menu)

        # Header settings
        h_header = self._table.horizontalHeader()
        h_header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        h_header.setDefaultSectionSize(80)
        h_header.setMinimumSectionSize(50)
        h_header.setStretchLastSection(True)

        v_header = self._table.verticalHeader()
        v_header.setDefaultSectionSize(24)
        v_header.setMinimumSectionSize(20)

        # Selection changed
        self._table.selectionModel().selectionChanged.connect(self._on_selection_changed)

        layout.addWidget(self._table, 1)

        # Status bar
        self._status_bar = self._create_status_bar()
        layout.addWidget(self._status_bar)

    def _create_toolbar(self) -> QToolBar:
        toolbar = QToolBar()
        toolbar.setMovable(False)
        sizes = Icons.plus(16).availableSizes()
        toolbar.setIconSize(sizes[0] if sizes else QSize(16, 16))

        # Copy
        copy_action = QAction("Copy", self)
        copy_action.setIcon(Icons.copy(16))
        copy_action.setShortcut(QKeySequence.StandardKey.Copy)
        copy_action.triggered.connect(self._copy_selection)
        toolbar.addAction(copy_action)

        # Paste
        paste_action = QAction("Paste", self)
        paste_action.setIcon(Icons.paste(16))
        paste_action.setShortcut(QKeySequence.StandardKey.Paste)
        paste_action.triggered.connect(self._paste_clipboard)
        toolbar.addAction(paste_action)

        toolbar.addSeparator()

        # Fill Down
        fill_down_action = QAction("Fill Down", self)
        fill_down_action.setIcon(Icons.fill_down(16))
        fill_down_action.setShortcut(QKeySequence("Ctrl+D"))
        fill_down_action.triggered.connect(self._fill_down)
        toolbar.addAction(fill_down_action)

        toolbar.addSeparator()

        # Add Column
        add_col_action = QAction("Add Column", self)
        add_col_action.setIcon(Icons.column_add(16))
        add_col_action.triggered.connect(self._add_column)
        toolbar.addAction(add_col_action)

        # Delete Rows
        delete_rows_action = QAction("Delete Rows", self)
        delete_rows_action.setIcon(Icons.trash(16))
        delete_rows_action.triggered.connect(self._delete_selected_rows)
        toolbar.addAction(delete_rows_action)

        toolbar.addSeparator()

        # Flag/Unflag
        flag_action = QAction("Flag Selected", self)
        flag_action.setIcon(Icons.warning(16))
        flag_action.triggered.connect(self._flag_selected)
        toolbar.addAction(flag_action)

        unflag_action = QAction("Unflag Selected", self)
        unflag_action.setIcon(Icons.check(16))
        unflag_action.triggered.connect(self._unflag_selected)
        toolbar.addAction(unflag_action)

        return toolbar

    def _create_filter_bar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(SPACING["sm"], SPACING["xs"], SPACING["sm"], SPACING["xs"])
        layout.setSpacing(SPACING["sm"])

        # Filter icon/label
        filter_label = QLabel("Filter:")
        filter_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(filter_label)

        # Filter input
        self._filter_input = QLineEdit()
        self._filter_input.setPlaceholderText("Type to filter...")
        self._filter_input.textChanged.connect(self._on_filter_changed)
        self._filter_input.setMaximumWidth(200)
        layout.addWidget(self._filter_input)

        # Column selector
        self._filter_column = QComboBox()
        self._filter_column.addItem("All Columns", -1)
        self._filter_column.currentIndexChanged.connect(self._on_filter_column_changed)
        self._filter_column.setMaximumWidth(150)
        layout.addWidget(self._filter_column)

        layout.addStretch()

        # Row count
        self._row_count_label = QLabel("0 rows")
        self._row_count_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(self._row_count_label)

        return bar

    def _create_status_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(24)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(SPACING["sm"], 0, SPACING["sm"], 0)
        layout.setSpacing(SPACING["md"])

        # Selection info
        self._selection_label = QLabel("No selection")
        self._selection_label.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-size: {TYPOGRAPHY['size_sm']}pt;
        """)
        layout.addWidget(self._selection_label)

        layout.addStretch()

        # Modified indicator
        self._modified_label = QLabel("")
        self._modified_label.setStyleSheet(f"""
            color: {COLORS['accent_warning']};
            font-size: {TYPOGRAPHY['size_sm']}pt;
        """)
        layout.addWidget(self._modified_label)

        return bar

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        pass  # Already handled via QAction shortcuts

    def _setup_context_menu(self):
        """Setup the right-click context menu."""
        self._context_menu = QMenu(self)

        self._context_menu.addAction("Copy\tCtrl+C", self._copy_selection)
        self._context_menu.addAction("Paste\tCtrl+V", self._paste_clipboard)
        self._context_menu.addSeparator()
        self._context_menu.addAction("Fill Down\tCtrl+D", self._fill_down)
        self._context_menu.addSeparator()
        self._context_menu.addAction("Flag Selected", self._flag_selected)
        self._context_menu.addAction("Unflag Selected", self._unflag_selected)
        self._context_menu.addSeparator()
        self._context_menu.addAction("Delete Selected Rows", self._delete_selected_rows)

    def _apply_style(self):
        self.setStyleSheet(f"""
            SpectralDataGrid {{
                background-color: {COLORS["bg_surface"]};
                border: 1px solid {COLORS["border_default"]};
                border-radius: {SPACING["sm"]}px;
            }}
            QToolBar {{
                background-color: {COLORS["bg_elevated"]};
                border: none;
                border-bottom: 1px solid {COLORS["border_subtle"]};
                padding: 4px;
                spacing: 4px;
            }}
            QToolBar QToolButton {{
                background-color: transparent;
                border: none;
                border-radius: 4px;
                padding: 4px;
            }}
            QToolBar QToolButton:hover {{
                background-color: {COLORS["bg_overlay"]};
            }}
            QToolBar QToolButton:pressed {{
                background-color: {COLORS["bg_elevated"]};
            }}
        """)

    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================

    def set_data(
        self,
        spectral_data: np.ndarray,
        wavelengths: np.ndarray,
        sample_ids: Optional[List[str]] = None,
        target_values: Optional[np.ndarray] = None,
        target_name: str = "Target",
        metadata: Optional[dict] = None,
    ):
        """Set the data to display."""
        self._model.set_data(
            spectral_data, wavelengths, sample_ids,
            target_values, target_name, metadata
        )
        self._update_filter_columns()
        self._update_row_count()

    def get_model(self) -> SpectralDataModel:
        """Get the underlying data model."""
        return self._model

    def get_selected_rows(self) -> List[int]:
        """Get list of selected row indices (in source model)."""
        selection = self._table.selectionModel().selectedIndexes()
        rows = set()
        for index in selection:
            source_index = self._proxy.mapToSource(index)
            rows.add(source_index.row())
        return sorted(rows)

    def get_selected_cells(self) -> List[Tuple[int, int]]:
        """Get list of selected (row, col) tuples (in source model)."""
        selection = self._table.selectionModel().selectedIndexes()
        cells = []
        for index in selection:
            source_index = self._proxy.mapToSource(index)
            cells.append((source_index.row(), source_index.column()))
        return cells

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _show_context_menu(self, pos):
        self._context_menu.exec(self._table.mapToGlobal(pos))

    def _on_selection_changed(self):
        cells = self.get_selected_cells()
        rows = self.get_selected_rows()

        if not cells:
            self._selection_label.setText("No selection")
        elif len(cells) == 1:
            r, c = cells[0]
            self._selection_label.setText(f"Cell ({r+1}, {c+1})")
        else:
            self._selection_label.setText(f"{len(cells)} cells, {len(rows)} rows")

        self.selection_changed.emit(rows)

    def _on_filter_changed(self, text: str):
        col = self._filter_column.currentData()
        self._proxy.set_filter(text, col if col is not None else -1)
        self._update_row_count()

    def _on_filter_column_changed(self, index: int):
        text = self._filter_input.text()
        col = self._filter_column.currentData()
        self._proxy.set_filter(text, col if col is not None else -1)
        self._update_row_count()

    def _update_filter_columns(self):
        self._filter_column.clear()
        self._filter_column.addItem("All Columns", -1)

        for col in range(self._model.columnCount()):
            name = self._model.headerData(col, Qt.Orientation.Horizontal)
            self._filter_column.addItem(str(name), col)

    def _update_row_count(self):
        total = self._model.rowCount()
        visible = self._proxy.rowCount()
        if visible == total:
            self._row_count_label.setText(f"{total} rows")
        else:
            self._row_count_label.setText(f"{visible} / {total} rows")

    def _copy_selection(self):
        cells = self.get_selected_cells()
        if cells:
            text = self._model.copy_selection(cells)
            QApplication.clipboard().setText(text)

    def _paste_clipboard(self):
        cells = self.get_selected_cells()
        if not cells:
            return

        text = QApplication.clipboard().text()
        if text:
            start_row, start_col = cells[0]
            count = self._model.paste_from_clipboard(start_row, start_col, text)
            if count > 0:
                self._update_modified_indicator()

    def _fill_down(self):
        cells = self.get_selected_cells()
        if len(cells) < 2:
            return

        # Group by column
        by_column: Dict[int, List[int]] = {}
        for row, col in cells:
            if col not in by_column:
                by_column[col] = []
            by_column[col].append(row)

        # Fill down each column
        for col, rows in by_column.items():
            rows = sorted(rows)
            if len(rows) >= 2:
                self._model.fill_down(rows[0], rows[-1], col)

        self._update_modified_indicator()

    def _add_column(self):
        name, ok = QInputDialog.getText(
            self, "Add Column", "Column name:"
        )
        if ok and name:
            if self._model.add_column(name):
                self._update_filter_columns()
            else:
                QMessageBox.warning(
                    self, "Error", f"Column '{name}' already exists."
                )

    def _delete_selected_rows(self):
        rows = self.get_selected_rows()
        if not rows:
            return

        reply = QMessageBox.question(
            self, "Delete Rows",
            f"Delete {len(rows)} selected rows?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._model.delete_rows(rows)
            self._update_row_count()

    def _flag_selected(self):
        rows = self.get_selected_rows()
        if rows:
            self._model.flag_rows(rows, True)
            self.rows_flagged.emit(self._model.get_flagged_rows())

    def _unflag_selected(self):
        rows = self.get_selected_rows()
        if rows:
            self._model.flag_rows(rows, False)
            self.rows_flagged.emit(self._model.get_flagged_rows())

    def _update_modified_indicator(self):
        if self._model.has_unsaved_changes():
            self._modified_label.setText("Modified")
        else:
            self._modified_label.setText("")
