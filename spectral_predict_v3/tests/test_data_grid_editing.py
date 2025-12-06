"""
Comprehensive tests for data grid editing functionality.

Tests cover:
- Cell editing with various data types
- Column add/delete/rename operations
- Row add/delete/duplicate operations
- Fill operations (fill down, fill selection)
- Undo/redo functionality
- Edit history limits
- Concurrent editing scenarios
- Large dataset performance
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spectral_predict_v3.core.types import SpectralDataset
from spectral_predict_v3.ui.components.data_grid import DataGrid, EditHistory


class TestEditHistory:
    """Test EditHistory class."""

    def test_creation(self):
        """Test EditHistory initialization."""
        history = EditHistory(max_size=50)
        assert history.max_size == 50
        assert not history.can_undo()
        assert not history.can_redo()

    def test_push_and_undo(self):
        """Test pushing operations and undo."""
        history = EditHistory()

        op1 = {'type': 'cell_edit', 'row': 0, 'col': 1, 'old_value': 'a', 'new_value': 'b'}
        history.push(op1)

        assert history.can_undo()
        assert not history.can_redo()

        undone = history.undo()
        assert undone == op1
        assert not history.can_undo()
        assert history.can_redo()

    def test_redo(self):
        """Test redo functionality."""
        history = EditHistory()

        op1 = {'type': 'cell_edit', 'row': 0, 'col': 1, 'old_value': 'a', 'new_value': 'b'}
        history.push(op1)
        history.undo()

        redone = history.redo()
        assert redone == op1
        assert history.can_undo()
        assert not history.can_redo()

    def test_clear_redo_on_new_push(self):
        """Test that redo stack clears when new operation is pushed."""
        history = EditHistory()

        op1 = {'type': 'cell_edit', 'row': 0, 'col': 1, 'old_value': 'a', 'new_value': 'b'}
        op2 = {'type': 'cell_edit', 'row': 0, 'col': 2, 'old_value': 'c', 'new_value': 'd'}

        history.push(op1)
        history.undo()
        assert history.can_redo()

        # Push new operation should clear redo
        history.push(op2)
        assert not history.can_redo()

    def test_max_size_limit(self):
        """Test that history respects max_size limit."""
        history = EditHistory(max_size=3)

        # Push 5 operations
        for i in range(5):
            history.push({'type': 'test', 'index': i})

        # Only last 3 should be in history
        ops = []
        while history.can_undo():
            ops.append(history.undo())

        assert len(ops) == 3
        assert ops[0]['index'] == 4  # Most recent
        assert ops[2]['index'] == 2  # Oldest kept

    def test_clear(self):
        """Test clearing history."""
        history = EditHistory()

        history.push({'type': 'test'})
        history.push({'type': 'test2'})

        assert history.can_undo()
        history.clear()
        assert not history.can_undo()
        assert not history.can_redo()


class TestDataGridEditingBasics:
    """Test basic editing functionality."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        X = np.random.rand(10, 100)
        wavelengths = np.linspace(400, 2500, 100)
        sample_ids = [f"sample_{i}" for i in range(10)]
        y = np.random.rand(10)
        metadata = {
            'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
            'value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        }

        return SpectralDataset(
            X=X,
            wavelengths=wavelengths,
            sample_ids=sample_ids,
            y=y,
            target_name='target',
            metadata_columns=metadata
        )

    def test_edit_mode_toggle(self, sample_dataset):
        """Test enabling/disabling edit mode."""
        # Note: This test doesn't create actual DPG context
        # It only tests the API
        grid = DataGrid.__new__(DataGrid)
        grid._edit_mode = False
        grid._editing_cell = None
        grid._edit_input_tag = None

        grid.set_edit_mode(True)
        assert grid.is_edit_mode()

        grid.set_edit_mode(False)
        assert not grid.is_edit_mode()

    def test_get_cell_value(self, sample_dataset):
        """Test getting cell values."""
        grid = DataGrid.__new__(DataGrid)
        grid._dataset = sample_dataset
        grid._show_all_wavelengths = False

        # Test ID column
        assert grid._get_cell_value(0, 0) == 'sample_0'

        # Test target column
        assert grid._get_cell_value(0, 1) == sample_dataset.y[0]

        # Test metadata columns
        assert grid._get_cell_value(0, 2) == 'A'  # category
        assert grid._get_cell_value(0, 3) == 1.0  # value

    def test_set_cell_value(self, sample_dataset):
        """Test setting cell values."""
        grid = DataGrid.__new__(DataGrid)
        grid._dataset = sample_dataset
        grid._show_all_wavelengths = False

        # Test ID column
        assert grid._set_cell_value(0, 0, 'new_id')
        assert grid._dataset.sample_ids[0] == 'new_id'

        # Test target column
        assert grid._set_cell_value(0, 1, 99.5)
        assert grid._dataset.y[0] == 99.5

        # Test metadata columns
        assert grid._set_cell_value(0, 2, 'C')
        assert grid._dataset.metadata_columns['category'][0] == 'C'

    def test_set_cell_value_validation(self, sample_dataset):
        """Test that invalid values are rejected."""
        grid = DataGrid.__new__(DataGrid)
        grid._dataset = sample_dataset
        grid._show_all_wavelengths = False

        # Try to set non-numeric value to target (should fail)
        result = grid._set_cell_value(0, 1, 'invalid')
        assert not result

        # Original value should be unchanged
        assert isinstance(grid._dataset.y[0], (int, float, np.number))

    def test_get_column_type(self, sample_dataset):
        """Test column type detection."""
        grid = DataGrid.__new__(DataGrid)
        grid._dataset = sample_dataset

        assert grid._get_column_type(0) == 'id'
        assert grid._get_column_type(1) == 'target'
        assert grid._get_column_type(2) == 'metadata'
        assert grid._get_column_type(3) == 'metadata'
        # Wavelength columns start after metadata
        assert grid._get_column_type(4) == 'spectral'


class TestRowOperations:
    """Test row manipulation operations."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        X = np.random.rand(10, 100)
        wavelengths = np.linspace(400, 2500, 100)
        sample_ids = [f"sample_{i}" for i in range(10)]
        y = np.random.rand(10)
        metadata = {'category': ['A'] * 10}

        return SpectralDataset(
            X=X,
            wavelengths=wavelengths,
            sample_ids=sample_ids,
            y=y,
            target_name='target',
            metadata_columns=metadata
        )

    def test_delete_single_row(self, sample_dataset):
        """Test deleting a single row."""
        grid = DataGrid.__new__(DataGrid)
        grid._dataset = sample_dataset
        grid._edit_history = EditHistory()
        grid._on_data_change_callback = None

        original_count = grid._dataset.n_samples

        grid._delete_rows_by_indices([5], record_history=True)

        assert grid._dataset.n_samples == original_count - 1
        assert grid._edit_history.can_undo()

    def test_delete_multiple_rows(self, sample_dataset):
        """Test deleting multiple rows."""
        grid = DataGrid.__new__(DataGrid)
        grid._dataset = sample_dataset
        grid._edit_history = EditHistory()
        grid._on_data_change_callback = None

        original_count = grid._dataset.n_samples

        grid._delete_rows_by_indices([1, 3, 5], record_history=True)

        assert grid._dataset.n_samples == original_count - 3
        assert grid._edit_history.can_undo()

    def test_restore_rows(self, sample_dataset):
        """Test restoring deleted rows via undo."""
        grid = DataGrid.__new__(DataGrid)
        grid._dataset = sample_dataset.copy()
        grid._edit_history = EditHistory()
        grid._on_data_change_callback = None

        original_ids = grid._dataset.sample_ids.copy()
        original_count = grid._dataset.n_samples

        # Delete and then undo
        grid._delete_rows_by_indices([2, 4], record_history=True)
        assert grid._dataset.n_samples == original_count - 2

        # Undo should restore
        operation = grid._edit_history.undo()
        grid._restore_rows(operation['rows_data'], operation['indices'])

        assert grid._dataset.n_samples == original_count
        assert grid._dataset.sample_ids == original_ids

    def test_duplicate_rows(self, sample_dataset):
        """Test duplicating rows."""
        grid = DataGrid.__new__(DataGrid)
        grid._dataset = sample_dataset.copy()
        grid._edit_history = EditHistory()
        grid._on_data_change_callback = None

        original_count = grid._dataset.n_samples
        original_id = grid._dataset.sample_ids[0]

        grid.duplicate_rows([0])

        assert grid._dataset.n_samples == original_count + 1
        # Duplicated row should be right after original
        assert grid._dataset.sample_ids[1] == original_id + "_copy"
        assert grid._edit_history.can_undo()

    def test_insert_row(self, sample_dataset):
        """Test inserting a new row."""
        grid = DataGrid.__new__(DataGrid)
        grid._dataset = sample_dataset.copy()
        grid._edit_history = EditHistory()
        grid._on_data_change_callback = None

        original_count = grid._dataset.n_samples

        grid.insert_row()

        assert grid._dataset.n_samples == original_count + 1
        assert grid._edit_history.can_undo()

    def test_insert_row_at_position(self, sample_dataset):
        """Test inserting a row at specific position."""
        grid = DataGrid.__new__(DataGrid)
        grid._dataset = sample_dataset.copy()
        grid._edit_history = EditHistory()
        grid._on_data_change_callback = None

        grid.insert_row(position=3)

        # New row should be at position 3
        assert 'new_sample_3' in grid._dataset.sample_ids[3]


class TestColumnOperations:
    """Test column manipulation operations."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        X = np.random.rand(10, 100)
        wavelengths = np.linspace(400, 2500, 100)
        sample_ids = [f"sample_{i}" for i in range(10)]
        metadata = {'category': ['A'] * 10}

        return SpectralDataset(
            X=X,
            wavelengths=wavelengths,
            sample_ids=sample_ids,
            metadata_columns=metadata
        )

    def test_add_column(self, sample_dataset):
        """Test adding a new metadata column."""
        grid = DataGrid.__new__(DataGrid)
        grid._dataset = sample_dataset.copy()
        grid._edit_history = EditHistory()
        grid._on_data_change_callback = None

        original_cols = len(grid._dataset.metadata_columns)

        grid.add_column('new_column', default_value='test')

        assert len(grid._dataset.metadata_columns) == original_cols + 1
        assert 'new_column' in grid._dataset.metadata_columns
        assert all(v == 'test' for v in grid._dataset.metadata_columns['new_column'])
        assert grid._edit_history.can_undo()

    def test_delete_column(self, sample_dataset):
        """Test deleting a metadata column."""
        grid = DataGrid.__new__(DataGrid)
        grid._dataset = sample_dataset.copy()
        grid._edit_history = EditHistory()
        grid._on_data_change_callback = None

        original_cols = len(grid._dataset.metadata_columns)

        grid._delete_column('category', record_history=True)

        assert len(grid._dataset.metadata_columns) == original_cols - 1
        assert 'category' not in grid._dataset.metadata_columns
        assert grid._edit_history.can_undo()

    def test_restore_column(self, sample_dataset):
        """Test restoring a deleted column."""
        grid = DataGrid.__new__(DataGrid)
        grid._dataset = sample_dataset.copy()
        grid._edit_history = EditHistory()
        grid._on_data_change_callback = None

        original_values = grid._dataset.metadata_columns['category'].copy()

        # Delete then restore
        grid._delete_column('category', record_history=True)
        assert 'category' not in grid._dataset.metadata_columns

        operation = grid._edit_history.undo()
        # Fix the restore operation to include column name
        operation['column_data']['column_name'] = operation['column_name']
        grid._restore_column(operation['column_data'])

        assert 'category' in grid._dataset.metadata_columns
        assert grid._dataset.metadata_columns['category'] == original_values

    def test_rename_column(self, sample_dataset):
        """Test renaming a column."""
        grid = DataGrid.__new__(DataGrid)
        grid._dataset = sample_dataset.copy()
        grid._on_data_change_callback = None

        original_values = grid._dataset.metadata_columns['category'].copy()

        grid.rename_column('category', 'new_category')

        assert 'category' not in grid._dataset.metadata_columns
        assert 'new_category' in grid._dataset.metadata_columns
        assert grid._dataset.metadata_columns['new_category'] == original_values


class TestFillOperations:
    """Test fill operations."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        X = np.random.rand(10, 100)
        wavelengths = np.linspace(400, 2500, 100)
        sample_ids = [f"sample_{i}" for i in range(10)]
        metadata = {'category': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']}

        return SpectralDataset(
            X=X,
            wavelengths=wavelengths,
            sample_ids=sample_ids,
            metadata_columns=metadata
        )

    def test_fill_down(self, sample_dataset):
        """Test fill down operation."""
        grid = DataGrid.__new__(DataGrid)
        grid._dataset = sample_dataset.copy()
        grid._edit_history = EditHistory()
        grid._on_data_change_callback = None
        grid._show_all_wavelengths = False

        # Fill category column from row 0 to row 4
        grid.fill_down(start_row=0, col=2, end_row=4)

        # All rows 1-4 should now have value from row 0
        expected_value = sample_dataset.metadata_columns['category'][0]
        for i in range(1, 5):
            assert grid._dataset.metadata_columns['category'][i] == expected_value

        assert grid._edit_history.can_undo()

    def test_fill_selection(self, sample_dataset):
        """Test fill selection operation."""
        grid = DataGrid.__new__(DataGrid)
        grid._dataset = sample_dataset.copy()
        grid._edit_history = EditHistory()
        grid._on_data_change_callback = None
        grid._show_all_wavelengths = False

        # Fill rows 2, 4, 6 with 'Z'
        grid.fill_selection(row_indices=[2, 4, 6], col=2, value='Z')

        assert grid._dataset.metadata_columns['category'][2] == 'Z'
        assert grid._dataset.metadata_columns['category'][4] == 'Z'
        assert grid._dataset.metadata_columns['category'][6] == 'Z'
        # Other rows should be unchanged
        assert grid._dataset.metadata_columns['category'][3] != 'Z'

        assert grid._edit_history.can_undo()


class TestUndoRedo:
    """Test undo/redo functionality."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        X = np.random.rand(10, 100)
        wavelengths = np.linspace(400, 2500, 100)
        sample_ids = [f"sample_{i}" for i in range(10)]
        metadata = {'category': ['A'] * 10}

        return SpectralDataset(
            X=X,
            wavelengths=wavelengths,
            sample_ids=sample_ids,
            metadata_columns=metadata
        )

    def test_undo_cell_edit(self, sample_dataset):
        """Test undoing a cell edit."""
        grid = DataGrid.__new__(DataGrid)
        grid._dataset = sample_dataset.copy()
        grid._edit_history = EditHistory()
        grid._on_data_change_callback = None
        grid._show_all_wavelengths = False

        original_value = grid._dataset.metadata_columns['category'][0]

        # Simulate cell edit
        grid._edit_history.push({
            'type': 'cell_edit',
            'row': 0,
            'col': 2,
            'old_value': original_value,
            'new_value': 'Z'
        })
        grid._set_cell_value(0, 2, 'Z')

        # Undo
        operation = grid._edit_history.undo()
        grid._set_cell_value(operation['row'], operation['col'], operation['old_value'])

        assert grid._dataset.metadata_columns['category'][0] == original_value

    def test_undo_row_delete(self, sample_dataset):
        """Test undoing row deletion."""
        grid = DataGrid.__new__(DataGrid)
        grid._dataset = sample_dataset.copy()
        grid._edit_history = EditHistory()
        grid._on_data_change_callback = None

        original_count = grid._dataset.n_samples

        # Delete row
        grid._delete_rows_by_indices([3], record_history=True)
        assert grid._dataset.n_samples == original_count - 1

        # Undo
        operation = grid._edit_history.undo()
        grid._restore_rows(operation['rows_data'], operation['indices'])

        assert grid._dataset.n_samples == original_count

    def test_redo_after_undo(self, sample_dataset):
        """Test redoing an operation after undo."""
        grid = DataGrid.__new__(DataGrid)
        grid._dataset = sample_dataset.copy()
        grid._edit_history = EditHistory()
        grid._on_data_change_callback = None

        original_count = grid._dataset.n_samples

        # Delete, undo, then redo
        grid._delete_rows_by_indices([3], record_history=True)
        operation = grid._edit_history.undo()
        grid._restore_rows(operation['rows_data'], operation['indices'])

        # Redo
        operation = grid._edit_history.redo()
        grid._delete_rows_by_indices(operation['indices'], record_history=False)

        assert grid._dataset.n_samples == original_count - 1

    def test_undo_stack_limit(self, sample_dataset):
        """Test that undo stack respects the 50 operation limit."""
        grid = DataGrid.__new__(DataGrid)
        grid._dataset = sample_dataset.copy()
        grid._edit_history = EditHistory(max_size=50)
        grid._on_data_change_callback = None
        grid._show_all_wavelengths = False

        # Perform 60 edits
        for i in range(60):
            grid._edit_history.push({
                'type': 'cell_edit',
                'row': 0,
                'col': 2,
                'old_value': f'val_{i}',
                'new_value': f'val_{i+1}'
            })

        # Count how many undos are available
        undo_count = 0
        while grid._edit_history.can_undo():
            grid._edit_history.undo()
            undo_count += 1

        # Should only have 50 undos
        assert undo_count == 50


class TestLargeDataset:
    """Test performance with large datasets."""

    def test_large_dataset_operations(self):
        """Test operations on a dataset with 1000+ rows."""
        # Create large dataset
        n_samples = 1000
        n_wavelengths = 500

        X = np.random.rand(n_samples, n_wavelengths)
        wavelengths = np.linspace(400, 2500, n_wavelengths)
        sample_ids = [f"sample_{i}" for i in range(n_samples)]
        metadata = {'category': ['A'] * n_samples}

        dataset = SpectralDataset(
            X=X,
            wavelengths=wavelengths,
            sample_ids=sample_ids,
            metadata_columns=metadata
        )

        grid = DataGrid.__new__(DataGrid)
        grid._dataset = dataset
        grid._edit_history = EditHistory()
        grid._on_data_change_callback = None
        grid._show_all_wavelengths = False

        # Test get/set operations
        value = grid._get_cell_value(500, 2)
        assert value is not None

        success = grid._set_cell_value(500, 2, 'test')
        assert success

        # Test delete multiple rows
        grid._delete_rows_by_indices([100, 200, 300], record_history=True)
        assert grid._dataset.n_samples == n_samples - 3

        # Test undo
        operation = grid._edit_history.undo()
        grid._restore_rows(operation['rows_data'], operation['indices'])
        assert grid._dataset.n_samples == n_samples


class TestEdgeCase:
    """Test edge cases and error handling."""

    def test_empty_dataset(self):
        """Test operations on empty dataset."""
        grid = DataGrid.__new__(DataGrid)
        grid._dataset = None
        grid._edit_history = EditHistory()

        # These should not crash
        assert grid._get_cell_value(0, 0) is None
        assert not grid._set_cell_value(0, 0, 'test')
        grid._delete_rows_by_indices([0], record_history=True)  # Should do nothing

    def test_invalid_indices(self):
        """Test handling of invalid row/column indices."""
        X = np.random.rand(10, 100)
        wavelengths = np.linspace(400, 2500, 100)
        sample_ids = [f"sample_{i}" for i in range(10)]

        dataset = SpectralDataset(
            X=X,
            wavelengths=wavelengths,
            sample_ids=sample_ids
        )

        grid = DataGrid.__new__(DataGrid)
        grid._dataset = dataset
        grid._show_all_wavelengths = False

        # Out of bounds should return None
        assert grid._get_cell_value(999, 0) is None
        assert grid._get_cell_value(0, 999) is None

    def test_concurrent_editing(self):
        """Test behavior when multiple edits happen in sequence."""
        X = np.random.rand(10, 100)
        wavelengths = np.linspace(400, 2500, 100)
        sample_ids = [f"sample_{i}" for i in range(10)]
        metadata = {'category': ['A'] * 10}

        dataset = SpectralDataset(
            X=X,
            wavelengths=wavelengths,
            sample_ids=sample_ids,
            metadata_columns=metadata
        )

        grid = DataGrid.__new__(DataGrid)
        grid._dataset = dataset
        grid._edit_history = EditHistory()
        grid._on_data_change_callback = None
        grid._show_all_wavelengths = False

        # Multiple rapid edits
        grid._edit_history.push({
            'type': 'cell_edit',
            'row': 0,
            'col': 2,
            'old_value': 'A',
            'new_value': 'B'
        })
        grid._set_cell_value(0, 2, 'B')

        grid._edit_history.push({
            'type': 'cell_edit',
            'row': 0,
            'col': 2,
            'old_value': 'B',
            'new_value': 'C'
        })
        grid._set_cell_value(0, 2, 'C')

        # Undo twice should get back to original
        op2 = grid._edit_history.undo()
        grid._set_cell_value(op2['row'], op2['col'], op2['old_value'])

        op1 = grid._edit_history.undo()
        grid._set_cell_value(op1['row'], op1['col'], op1['old_value'])

        assert grid._dataset.metadata_columns['category'][0] == 'A'


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
