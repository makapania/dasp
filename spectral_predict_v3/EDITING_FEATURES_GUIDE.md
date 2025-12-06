# Data Grid Editing Features - Quick Reference Guide

## Overview
The Spectral Predict v3 data grid now supports comprehensive editing capabilities. This guide explains how to use each feature.

## Activating Edit Mode

1. Navigate to the **Data** tab
2. Check the **"Edit Mode"** checkbox in the toolbar
3. All editing buttons will become enabled

**Note:** Edit mode is disabled by default to prevent accidental changes.

## Editing Individual Cells

### Programmatic Method
```python
# From your code
data_grid.edit_cell(row=5, col=2)
```

This opens a modal dialog where you can:
- Type the new value
- Click **OK** or press **Enter** to confirm
- Click **Cancel** or press **Escape** to cancel

### Data Type Validation
- **ID columns:** Accept any string
- **Target columns:** Accept numbers (auto-converted to float)
- **Metadata columns:** Accept any value
- **Spectral columns:** Accept numbers only (validated)

## Row Operations

### Delete Rows
1. Enter row indices in the input field (e.g., `0,2,5`)
2. Click **"Delete Row"**
3. Confirm deletion in the dialog
4. Rows are removed and data is stored for undo

**Example:** To delete rows 0, 5, and 10:
```
Row indices: 0,5,10
```

### Duplicate Rows
1. Enter row indices to duplicate (e.g., `3,7`)
2. Click **"Duplicate Row"**
3. Duplicated rows appear immediately after originals
4. Sample IDs get "_copy" suffix

**Example:** Duplicating row 3 creates a new row 4 with ID "sample_3_copy"

### Insert Row
1. Click **"Insert Row"**
2. A new blank row is added at the end
3. Sample ID is auto-generated as "new_sample_N"
4. All spectral values are initialized to 0

**To insert at specific position (programmatic):**
```python
data_grid.insert_row(position=5)  # Insert at index 5
```

## Column Operations

### Add Column
1. Click **"Add Column"**
2. Enter column name in the dialog
3. Click **"Add"**
4. New metadata column is created with empty values

**Note:** Only metadata columns can be added (not spectral columns)

### Delete Column
1. Click **"Delete Column"**
2. Select column from dropdown
3. Click **"Delete"**
4. Column is removed (can be undone)

**Note:** Cannot delete ID, target, or spectral columns

### Rename Column (Programmatic)
```python
data_grid.rename_column(old_name="category", new_name="group")
```

## Fill Operations

### Fill Down (Programmatic)
```python
# Fill column 2 from row 0 down to row 10
data_grid.fill_down(start_row=0, col=2, end_row=10)

# Fill to end of dataset
data_grid.fill_down(start_row=0, col=2, end_row=-1)
```

This propagates the value from `start_row` to all rows below.

### Fill Selection (Programmatic)
```python
# Fill specific rows with a value
data_grid.fill_selection(row_indices=[1,3,5,7], col=2, value="Group A")
```

Useful for batch updates to non-contiguous rows.

## Undo/Redo

### Undo Last Operation
- Click **"Undo"** button
- Or programmatically: `data_grid.undo()`

### Redo Undone Operation
- Click **"Redo"** button
- Or programmatically: `data_grid.redo()`

### What Can Be Undone?
- Cell edits
- Row deletions
- Row insertions (duplicate, insert)
- Column additions
- Column deletions
- Fill operations

### History Limit
- Maximum 50 operations stored
- Oldest operations are automatically removed when limit is reached
- Memory-efficient design

## Tips and Best Practices

### 1. Use Undo Liberally
The undo system is robust - don't hesitate to experiment and use undo if needed.

### 2. Row Index Format
When entering multiple row indices:
- Use commas: `0,1,2,5`
- Spaces are ignored: `0, 1, 2, 5` works too
- Indices are 0-based (first row is 0)

### 3. Batch Operations
For multiple row operations, enter all indices at once instead of one at a time:
```
Good: 0,1,2,3,4,5
Avoid: Multiple single operations
```

### 4. Data Integrity
- All operations maintain data integrity across X, y, sample_ids, and metadata
- Spectral data is preserved during row operations
- Type validation prevents invalid data entry

### 5. Save Your Work
Edit mode changes are in-memory. Remember to save your dataset to file when done:
```python
# Save functionality will be added in future sprint
# For now, changes persist in the current session
```

## Keyboard Shortcuts (Future)
The following shortcuts are planned for future releases:
- `Ctrl+Z`: Undo
- `Ctrl+Y`: Redo
- `Ctrl+C`: Copy selection
- `Ctrl+V`: Paste
- `Delete`: Delete selected rows

## Examples

### Example 1: Clean Up Sample IDs
```python
# Suppose sample IDs need standardization
grid.edit_cell(0, 0)  # Opens dialog for ID column
# User types: "SAMPLE_001"
# Click OK
```

### Example 2: Mark QC Samples
```python
# Add a QC column
grid.add_column("QC_Status", default_value="Pass")

# Mark specific samples as failed
grid.fill_selection(
    row_indices=[5, 12, 23],
    col=3,  # QC_Status column
    value="Fail"
)
```

### Example 3: Remove Outliers
```python
# Delete outlier samples identified in PCA plot
# Row indices: 45, 67, 89
grid.delete_rows([45, 67, 89])

# If you deleted wrong samples
grid.undo()
```

### Example 4: Duplicate Calibration Standards
```python
# Duplicate standards for validation
grid.duplicate_rows([0, 1, 2])  # Duplicate first 3 samples

# Now have:
# Row 0: sample_0
# Row 1: sample_0_copy
# Row 2: sample_1
# Row 3: sample_1_copy
# ...
```

## Troubleshooting

### "No row indices specified"
- Make sure to enter row indices in the input field
- Format: comma-separated numbers (e.g., `0,1,2`)

### "Error: Invalid row indices format"
- Check for non-numeric characters
- Valid: `0,1,2,5`
- Invalid: `a,b,c`

### Edit buttons are disabled
- Make sure **Edit Mode** checkbox is checked
- Data must be loaded first

### Cell edit not working
- Ensure edit mode is enabled
- Use `grid.edit_cell(row, col)` method
- Double-click detection not yet implemented

### Column not appearing in delete dialog
- Only metadata columns can be deleted
- ID, target, and spectral columns are protected
- Make sure the column exists in the dataset

## API Reference

### DataGrid Methods

#### Edit Mode
```python
grid.set_edit_mode(True)      # Enable editing
grid.is_edit_mode()           # Check if enabled
```

#### Cell Operations
```python
grid.edit_cell(row, col)      # Open edit dialog
grid._get_cell_value(row, col)  # Get current value
grid._set_cell_value(row, col, value)  # Set value
```

#### Row Operations
```python
grid.delete_rows([0, 1, 2])   # Delete rows
grid.duplicate_rows([5, 6])   # Duplicate rows
grid.insert_row(position=3)   # Insert at position
grid.insert_row()             # Insert at end
```

#### Column Operations
```python
grid.add_column("name", default_value="")
grid.delete_column("name")
grid.rename_column("old", "new")
```

#### Fill Operations
```python
grid.fill_down(start_row=0, col=2, end_row=10)
grid.fill_selection([1,2,3], col=2, value="A")
```

#### Undo/Redo
```python
grid.undo()    # Undo last operation
grid.redo()    # Redo last undo
```

#### Callbacks
```python
def on_change():
    print("Data changed!")

grid.set_on_data_change(on_change)
```

## Known Limitations

1. **Double-click editing:** Not yet implemented - use `edit_cell()` method
2. **Keyboard shortcuts:** Use toolbar buttons instead of Ctrl+Z/Y
3. **Cell highlighting:** Modified cells not visually highlighted
4. **Multi-selection:** Cannot select multiple cells with mouse
5. **Auto-increment:** Not available for numeric sequences

These limitations are documented and will be addressed in future sprints.

## Getting Help

- Check the implementation report: `SPRINT0_IMPLEMENTATION_REPORT.md`
- Review tests for examples: `tests/test_data_grid_editing.py`
- Submit issues or feature requests to the development team
