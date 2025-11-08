# Tab 9 UX Improvements - Implementation Guide

**Agent 3 - UX Improvements for Calibration Transfer**

## Overview

This guide provides detailed instructions for implementing comprehensive UX improvements to Tab 9 (Calibration Transfer) in `spectral_predict_gui_optimized.py`.

---

## Changes Summary

| Section | Changes | Lines Affected |
|---------|---------|----------------|
| `__init__` | Add status tracking variables | ~148 |
| Helper Methods | Add 6 new methods for UX | New section |
| `_create_tab9_calibration_transfer` | Complete rewrite with UX | 5896-6130 |
| Tab 9 Methods | Update 5 methods with UX wrappers | 5504-5837 |

---

## STEP 1: Update `__init__` Method

**Location:** Line ~148, after existing Tab 9 variables

**Add the following code after line 148:**

```python
        # Tab 9 Section Status Tracking
        self.ct_section_a_complete = False  # Master model loaded
        self.ct_section_b_complete = False  # Instruments & paired spectra loaded
        self.ct_section_c_complete = False  # Transfer model built
        self.ct_section_d_complete = False  # Equalization done
        self.ct_section_e_complete = False  # Predictions made

        # Tab 9 UI References (for enable/disable control)
        self.ct_section_b_buttons = []  # Buttons to enable after section A
        self.ct_section_c_button = None  # Build button to enable after section B
        self.ct_section_d_buttons = []  # Equalization buttons
        self.ct_section_e_buttons = []  # Prediction buttons

        # Tab 9 Status Labels
        self.ct_status_labels = {}  # Dict of section -> status label widget
        self.ct_workflow_labels = {}  # Dict of step -> workflow label widget
```

---

## STEP 2: Add Helper Methods

**Location:** Add these methods before `_create_tab9_calibration_transfer` (before line 5896)

### Method 1: Create Help Button

```python
    def _create_help_button(self, parent, help_text, title="Help"):
        """Create a help button that shows info on click."""
        help_label = ttk.Label(parent, text="ℹ️", style='TLabel', cursor="hand2")
        help_label.bind("<Button-1>", lambda e: messagebox.showinfo(title, help_text))
        return help_label
```

### Method 2: Update Section Status

```python
    def _update_ct_section_status(self, section, complete):
        """Update status indicator for a calibration transfer section.

        Args:
            section: 'a', 'b', 'c', 'd', or 'e'
            complete: True if section is complete, False otherwise
        """
        # Update internal state
        if section == 'a':
            self.ct_section_a_complete = complete
        elif section == 'b':
            self.ct_section_b_complete = complete
        elif section == 'c':
            self.ct_section_c_complete = complete
        elif section == 'd':
            self.ct_section_d_complete = complete
        elif section == 'e':
            self.ct_section_e_complete = complete

        # Update status label
        if section in self.ct_status_labels:
            label = self.ct_status_labels[section]
            if complete:
                label.config(text="✓ Complete", foreground="#27AE60", font=('Segoe UI', 10, 'bold'))
            else:
                label.config(text="○ Pending", foreground="#95A5A6", font=('Segoe UI', 10))

        # Update workflow guide
        self._update_ct_workflow_guide()

        # Update button states
        self._update_ct_button_states()
```

### Method 3: Update Workflow Guide

```python
    def _update_ct_workflow_guide(self):
        """Update workflow guide colors based on section completion."""
        workflow_steps = {
            'a': self.ct_section_a_complete,
            'b': self.ct_section_b_complete,
            'c': self.ct_section_c_complete,
            'd': self.ct_section_d_complete,
            'e': self.ct_section_e_complete
        }

        for step, label in self.ct_workflow_labels.items():
            if workflow_steps.get(step, False):
                label.config(foreground="#27AE60", font=('Segoe UI', 9, 'bold'))
            elif step == 'a' or (step == 'b' and workflow_steps['a']) or \
                 (step == 'c' and workflow_steps['b']) or \
                 (step == 'e' and workflow_steps['a']):
                # Required step or next available step
                label.config(foreground="#E67E22", font=('Segoe UI', 9, 'bold'))
            else:
                label.config(foreground="#95A5A6", font=('Segoe UI', 9))
```

### Method 4: Update Button States

```python
    def _update_ct_button_states(self):
        """Enable/disable buttons based on section completion states."""
        # Section B buttons: enable only when section A complete
        for button in self.ct_section_b_buttons:
            if self.ct_section_a_complete:
                button.config(state='normal')
            else:
                button.config(state='disabled')

        # Section C button: enable only when section B complete
        if self.ct_section_c_button:
            if self.ct_section_b_complete:
                self.ct_section_c_button.config(state='normal')
            else:
                self.ct_section_c_button.config(state='disabled')

        # Section D buttons: enable only when instruments registered
        for button in self.ct_section_d_buttons:
            if self.instrument_profiles:
                button.config(state='normal')
            else:
                button.config(state='disabled')

        # Section E buttons: enable only when master model + transfer model loaded
        for button in self.ct_section_e_buttons:
            if self.ct_master_model_dict and self.ct_pred_transfer_model:
                button.config(state='normal')
            else:
                button.config(state='disabled')
```

### Method 5: Validate DS Lambda

```python
    def _validate_ct_ds_lambda(self, *args):
        """Validate DS Ridge Lambda parameter and show visual feedback."""
        try:
            value = float(self.ct_ds_lambda_var.get())
            if value < 0.0001 or value > 1.0:
                self.ct_ds_lambda_entry.config(foreground='#E74C3C')  # Red
                self.ct_ds_lambda_warning.config(
                    text="⚠ Recommended: 0.001-1.0",
                    foreground='#E67E22'
                )
            else:
                self.ct_ds_lambda_entry.config(foreground='#27AE60')  # Green
                self.ct_ds_lambda_warning.config(text="")
        except ValueError:
            self.ct_ds_lambda_entry.config(foreground='#E74C3C')
            self.ct_ds_lambda_warning.config(
                text="⚠ Invalid number",
                foreground='#E74C3C'
            )
```

### Method 6: Validate PDS Window

```python
    def _validate_ct_pds_window(self, *args):
        """Validate PDS Window parameter and show visual feedback."""
        try:
            value = int(self.ct_pds_window_var.get())
            if value < 5 or value > 101:
                self.ct_pds_window_entry.config(foreground='#E74C3C')
                self.ct_pds_window_warning.config(
                    text="⚠ Recommended: 5-101",
                    foreground='#E67E22'
                )
            elif value % 2 == 0:
                self.ct_pds_window_entry.config(foreground='#E67E22')
                self.ct_pds_window_warning.config(
                    text="⚠ Should be odd number",
                    foreground='#E67E22'
                )
            else:
                self.ct_pds_window_entry.config(foreground='#27AE60')
                self.ct_pds_window_warning.config(text="")
        except ValueError:
            self.ct_pds_window_entry.config(foreground='#E74C3C')
            self.ct_pds_window_warning.config(
                text="⚠ Invalid number",
                foreground='#E74C3C'
            )
```

---

## STEP 3: Replace `_create_tab9_calibration_transfer` Method

**Location:** Lines 5896-6130

**Action:** Replace the entire method with the version in `tab9_ux_improvements.py` (PART 3)

**Key additions in the new version:**

1. **Workflow Guide Frame** (new):
   - Visual guide showing A → B → C → D → E workflow
   - Color-coded based on completion status
   - Located at top of tab after title

2. **Section Status Indicators** (all sections):
   - Status label at top of each section (A-E)
   - Shows: ✓ Complete, ⚠ Required, or ○ Pending
   - Updates dynamically as user progresses

3. **Help Tooltips** (Section B):
   - Help button explaining paired spectra
   - Inline note about Tab 8 prerequisite

4. **Help Tooltips** (Section C):
   - Help button for transfer method selection
   - Help buttons for DS Lambda and PDS Window
   - Recommended ranges shown inline
   - Validation warnings below parameter inputs

5. **Button References** (all sections):
   - Buttons stored in lists for enable/disable control
   - Section B buttons stored in `self.ct_section_b_buttons`
   - Section C button stored in `self.ct_section_c_button`
   - Section D buttons stored in `self.ct_section_d_buttons`
   - Section E buttons stored in `self.ct_section_e_buttons`

6. **Parameter Validation Binding**:
   - Trace callbacks on DS Lambda and PDS Window
   - Real-time validation as user types

7. **Initial State Setup**:
   - Call `_update_ct_button_states()` at end
   - Call `_update_ct_workflow_guide()` at end

---

## STEP 4: Update Tab 9 Action Methods

### Method 1: `_load_ct_master_model`

**Location:** Line 5504-5537

**Action:** Replace with `_load_ct_master_model_ux` from `tab9_ux_improvements.py`

**Changes:**
- Add call to `self._update_ct_section_status('a', True)` on success
- Add call to `self._update_ct_section_status('a', False)` on error

### Method 2: `_load_ct_paired_spectra`

**Location:** Line 5557-5611

**Action:** Replace with `_load_ct_paired_spectra_ux`

**Changes:**
- Add call to `self._update_ct_section_status('b', True)` on success
- Add call to `self._update_ct_section_status('b', False)` on error

### Method 3: `_build_ct_transfer_model`

**Location:** Line 5613-5677

**Action:** Replace with `_build_ct_transfer_model_ux`

**Changes:**
- Add call to `self._update_ct_section_status('c', True)` on success
- Add call to `self._update_ct_section_status('c', False)` on error

### Method 4: `_load_ct_pred_transfer_model`

**Location:** Line 5739-5758

**Action:** Replace with `_load_ct_pred_transfer_model_ux`

**Changes:**
- Add call to `self._update_ct_button_states()` after loading
- This enables Section E buttons when both model and TM are loaded

### Method 5: `_load_and_predict_ct`

**Location:** Line 5766-5837

**Action:** Replace with `_load_and_predict_ct_ux`

**Changes:**
- Add import for `glob` and `Path`
- Extract sample IDs from filenames instead of generic "Sample_1", etc.
- Store actual sample IDs in `self.ct_pred_sample_ids`
- Display actual sample IDs in results
- Add call to `self._update_ct_section_status('e', True)` on success
- Add call to `self._update_ct_section_status('e', False)` on error

---

## Complete List of UX Features

### 1. Section Status Indicators
- **Location:** Top of each section (A, B, C, D, E)
- **States:**
  - ✓ Complete (green, bold)
  - ⚠ Required (orange, bold)
  - ○ Pending (gray, regular)
- **Updates:** Automatically when section completes

### 2. Workflow Guide
- **Location:** Top of Tab 9, below title
- **Content:** A → B → C → D → E with labels
- **Colors:**
  - Green: Completed steps
  - Orange: Required/available steps
  - Gray: Future steps

### 3. Help Tooltips
- **Section B - Paired Spectra:**
  - ℹ️ icon with detailed explanation
  - Inline note about Tab 8 prerequisite

- **Section C - Transfer Method:**
  - ℹ️ icon explaining DS vs PDS

- **Section C - DS Lambda:**
  - ℹ️ icon with range guidance
  - Inline recommended range text

- **Section C - PDS Window:**
  - ℹ️ icon with size guidance
  - Inline recommended range text

### 4. Parameter Validation
- **DS Ridge Lambda:**
  - Green text: Valid (0.001-1.0)
  - Red text: Out of range
  - Warning label: Shows when invalid

- **PDS Window:**
  - Green text: Valid odd number in range
  - Orange text: Valid but even number
  - Red text: Out of range or invalid
  - Warning label: Shows specific issue

### 5. Sample ID Improvements
- **Before:** Generic "Sample_1", "Sample_2", etc.
- **After:** Actual filenames (e.g., "soil_001", "leaf_sample_A")
- **Location:** Section E predictions display and CSV export
- **Implementation:** Parse filenames using `Path(file).stem`

### 6. Smart Button States
- **Section B buttons:**
  - Disabled until Section A complete
  - Includes: Refresh, Browse, Load Paired Spectra

- **Section C button:**
  - Disabled until Section B complete
  - Includes: Build Transfer Model

- **Section D buttons:**
  - Disabled until instruments registered in Tab 8
  - Includes: Load Multi-Instrument, Equalize & Export

- **Section E buttons:**
  - Disabled until master model AND transfer model loaded
  - Includes: Browse TM, Load TM, Browse Spectra, Load & Predict, Export

---

## Testing Checklist

### Test 1: Initial State
- [ ] All status indicators show "○ Pending" (except B shows "⚠ Required")
- [ ] Workflow guide shows all steps in gray except A (orange)
- [ ] Section B, C, D, E buttons are disabled
- [ ] Section A buttons are enabled

### Test 2: Load Master Model
- [ ] After loading, Section A status → "✓ Complete" (green)
- [ ] Workflow A → green, B → orange
- [ ] Section B buttons become enabled

### Test 3: Load Paired Spectra
- [ ] Help button works and shows paired spectra info
- [ ] After loading, Section B status → "✓ Complete"
- [ ] Workflow B → green, C → orange
- [ ] Section C button becomes enabled

### Test 4: Parameter Validation
- [ ] DS Lambda: Enter "5.0" → text turns red, warning shows
- [ ] DS Lambda: Enter "0.01" → text turns green, warning clears
- [ ] PDS Window: Enter "12" → text turns orange, warning about odd number
- [ ] PDS Window: Enter "11" → text turns green, warning clears

### Test 5: Build Transfer Model
- [ ] Help buttons work for all parameters
- [ ] After building, Section C status → "✓ Complete"
- [ ] Transfer model info displays correctly

### Test 6: Load Transfer Model for Prediction
- [ ] After loading TM, if master model also loaded, Section E buttons enable
- [ ] Workflow E → orange (available)

### Test 7: Make Predictions
- [ ] Sample IDs show actual filenames, not "Sample_1"
- [ ] After predicting, Section E status → "✓ Complete"
- [ ] Workflow E → green

### Test 8: Export Predictions
- [ ] CSV file contains actual sample IDs in first column
- [ ] Sample IDs match filenames from directory

---

## Before/After User Experience

### Before: Section A (Load Master Model)

```
A) Load Master Model
-------------------
Load a trained PLS/PCR model...

[Browse] [Load Model]
```

### After: Section A (Load Master Model)

```
A) Load Master Model
-------------------
○ Pending

Load a trained PLS/PCR model...

[Browse] [Load Model]

→ After loading:

✓ Complete (green, bold)
```

---

### Before: Section B (Paired Spectra)

```
B) Select Instruments & Load Paired Spectra
------------------------------------------
Select master and slave instruments...

Master: [____] Slave: [____] [Refresh]
[Browse] [Load Paired Spectra]
```

### After: Section B (Paired Spectra)

```
B) Select Instruments & Load Paired Spectra
------------------------------------------
⚠ Required (orange, bold)

ℹ️ "Paired spectra = identical samples..."
Note: Register instruments in Tab 8 first

Select master and slave instruments...

Master: [____] Slave: [____] [Refresh] (disabled until A complete)
[Browse] [Load Paired Spectra] (disabled until A complete)
```

---

### Before: Section C (Parameters)

```
DS Ridge Lambda: [0.001]
PDS Window: [11]
```

### After: Section C (Parameters)

```
DS Ridge Lambda: [0.001] ℹ️ (Recommended: 0.001-1.0) [warning label if invalid]
                 ↑ green if valid, red if invalid

PDS Window: [11] ℹ️ (Recommended: 11-51, must be odd) [warning label if invalid]
            ↑ green if valid, orange if even, red if out of range
```

---

### Before: Section E (Predictions)

```
Predictions (first 10):
  Sample_1: 45.234
  Sample_2: 52.891
  Sample_3: 38.776
```

### After: Section E (Predictions)

```
Predictions (first 10):
  soil_sample_001: 45.234
  soil_sample_002: 52.891
  leaf_A_replicate1: 38.776
```

---

## File Locations

- **Implementation Code:** `C:\Users\sponheim\git\dasp\tab9_ux_improvements.py`
- **This Guide:** `C:\Users\sponheim\git\dasp\TAB9_UX_IMPLEMENTATION_GUIDE.md`
- **Target File:** `C:\Users\sponheim\git\dasp\spectral_predict_gui_optimized.py`

---

## Implementation Notes

1. **Color Consistency:**
   - All colors match existing theme in `_configure_style()`
   - Green (#27AE60): Success/Complete
   - Orange (#E67E22): Warning/Required
   - Gray (#95A5A6): Pending/Disabled
   - Red (#E74C3C): Error/Invalid

2. **Help Button Pattern:**
   - Consistent ℹ️ icon throughout
   - Hand cursor on hover
   - Shows `messagebox.showinfo()` on click
   - Multi-line help text with \n\n for paragraphs

3. **Status Update Pattern:**
   - Call `_update_ct_section_status(section, True)` on success
   - Call `_update_ct_section_status(section, False)` on error
   - Method automatically updates labels, workflow, and buttons

4. **Button State Management:**
   - Store button references in instance variables
   - Call `_update_ct_button_states()` after state changes
   - Uses `button.config(state='normal'/'disabled')`

5. **Sample ID Extraction:**
   - Uses `Path(file).stem` to get filename without extension
   - Maintains order of files via `sorted()`
   - Works for .asd, .csv, and .spc files

---

## Success Criteria

✅ All 6 requirements from task description implemented:

1. ✅ **Section Status Indicators:** Complete with 5 states (A-E)
2. ✅ **Help Tooltips:** Added to paired spectra, methods, and parameters
3. ✅ **Workflow Guidance:** Visual guide at top with color coding
4. ✅ **Sample ID Improvements:** Real filenames instead of generic names
5. ✅ **Parameter Validation UI:** Real-time feedback with colors
6. ✅ **Disable/Enable Logic:** Smart button states based on prerequisites

---

## End of Implementation Guide
