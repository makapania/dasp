# Tab 9 Calibration Transfer - Validation Flow Diagram

## Complete User Flow with Validations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TAB 9: CALIBRATION TRANSFER                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ SECTION A: Load Master Model                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐                                                        │
│  │ Browse & Load   │ → (Existing validations - not modified)                │
│  │ Master Model    │                                                        │
│  └─────────────────┘                                                        │
│         ↓                                                                   │
│  Store: ct_master_model_dict                                                │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ SECTION B: Select Instruments & Load Paired Spectra                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │ User selects Master and Slave instruments           │                   │
│  └─────────────────────────────────────────────────────┘                   │
│         ↓                                                                   │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │ ✓ VALIDATION: Same Instrument Check                 │                   │
│  │   master_id == slave_id?                            │                   │
│  │   → ERROR: "Same Instrument Selected"               │                   │
│  └─────────────────────────────────────────────────────┘                   │
│         ↓ PASS                                                              │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │ Load spectra from directory                         │                   │
│  │  - wavelengths_master, X_master                     │                   │
│  │  - wavelengths_slave, X_slave                       │                   │
│  └─────────────────────────────────────────────────────┘                   │
│         ↓                                                                   │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │ ✓ VALIDATION: Sample Count Check                    │                   │
│  │   X_master.shape[0] == X_slave.shape[0]?            │                   │
│  │   → ERROR: "Sample Count Mismatch"                  │                   │
│  │              "Master has X, Slave has Y samples"    │                   │
│  └─────────────────────────────────────────────────────┘                   │
│         ↓ PASS                                                              │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │ ✓ VALIDATION: Minimum Sample Check                  │                   │
│  │   X_master.shape[0] >= 20?                          │                   │
│  │   → WARNING: "Few Samples" (with continue option)   │                   │
│  └─────────────────────────────────────────────────────┘                   │
│         ↓ PASS or CONTINUE                                                 │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │ Calculate wavelength overlap                        │                   │
│  │  overlap_start = max(master[0], slave[0])           │                   │
│  │  overlap_end = min(master[-1], slave[-1])           │                   │
│  └─────────────────────────────────────────────────────┘                   │
│         ↓                                                                   │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │ ✓ VALIDATION: Wavelength Overlap Check              │                   │
│  │   overlap_start < overlap_end?                      │                   │
│  │   → ERROR: "No Wavelength Overlap"                  │                   │
│  │              "Master: X-Y nm, Slave: A-B nm"        │                   │
│  └─────────────────────────────────────────────────────┘                   │
│         ↓ PASS                                                              │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │ Calculate overlap percentage                        │                   │
│  │  min_overlap_pct = (overlap_span / min_span) * 100  │                   │
│  └─────────────────────────────────────────────────────┘                   │
│         ↓                                                                   │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │ ✓ VALIDATION: Overlap Percentage Check              │                   │
│  │   min_overlap_pct >= 80%?                           │                   │
│  │   → WARNING: "Limited Wavelength Overlap"           │                   │
│  │              "Overlap is X% of instrument range"    │                   │
│  └─────────────────────────────────────────────────────┘                   │
│         ↓ PASS or CONTINUE                                                 │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │ Resample to common grid                             │                   │
│  │  ct_X_master_common                                 │                   │
│  │  ct_X_slave_common                                  │                   │
│  │  ct_wavelengths_common                              │                   │
│  └─────────────────────────────────────────────────────┘                   │
│         ↓                                                                   │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │ ✓ Display Info with Overlap %                       │                   │
│  │   "Loaded N paired spectra"                         │                   │
│  │   "Wavelength overlap: X.X%"                        │                   │
│  └─────────────────────────────────────────────────────┘                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ SECTION C: Build Transfer Model                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │ User clicks "Build DS/PDS Transfer Model"           │                   │
│  └─────────────────────────────────────────────────────┘                   │
│         ↓                                                                   │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │ ✓ VALIDATION: Data Loaded Check                     │                   │
│  │   hasattr(self, 'ct_X_master_common')?              │                   │
│  │   ct_X_master_common is not None?                   │                   │
│  │   → ERROR: "No Paired Spectra Loaded"               │                   │
│  │            "Please load spectra in Section B first" │                   │
│  └─────────────────────────────────────────────────────┘                   │
│         ↓ PASS                                                              │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │ ✓ VALIDATION: Different Instruments                 │                   │
│  │   master_id == slave_id?                            │                   │
│  │   → ERROR: "Same Instrument Selected"               │                   │
│  └─────────────────────────────────────────────────────┘                   │
│         ↓ PASS                                                              │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │ If method == 'ds':                                  │                   │
│  │   ┌───────────────────────────────────────────────┐ │                   │
│  │   │ ✓ VALIDATION: DS Lambda Parameter             │ │                   │
│  │   │   Try: float(lambda_entry.get())              │ │                   │
│  │   │   → ERROR: "must be a number" (ValueError)    │ │                   │
│  │   │   Range: 0 < lambda <= 100?                   │ │                   │
│  │   │   → ERROR: "must be between 0 and 100"        │ │                   │
│  │   └───────────────────────────────────────────────┘ │                   │
│  │         ↓ PASS                                      │                   │
│  │   ┌───────────────────────────────────────────────┐ │                   │
│  │   │ Build DS model: estimate_ds(...)              │ │                   │
│  │   └───────────────────────────────────────────────┘ │                   │
│  │                                                      │                   │
│  │ If method == 'pds':                                 │                   │
│  │   ┌───────────────────────────────────────────────┐ │                   │
│  │   │ ✓ VALIDATION: PDS Window Parameter            │ │                   │
│  │   │   Try: int(window_entry.get())                │ │                   │
│  │   │   → ERROR: "must be an integer" (ValueError)  │ │                   │
│  │   │   Range: 5 <= window <= 101?                  │ │                   │
│  │   │   → ERROR: "must be between 5 and 101"        │ │                   │
│  │   │   Is odd: window % 2 == 1?                    │ │                   │
│  │   │   → ERROR: "must be an odd number"            │ │                   │
│  │   └───────────────────────────────────────────────┘ │                   │
│  │         ↓ PASS                                      │                   │
│  │   ┌───────────────────────────────────────────────┐ │                   │
│  │   │ Build PDS model: estimate_pds(...)            │ │                   │
│  │   └───────────────────────────────────────────────┘ │                   │
│  └─────────────────────────────────────────────────────┘                   │
│         ↓                                                                   │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │ Store: ct_transfer_model                            │                   │
│  │ Display transfer model info                         │                   │
│  └─────────────────────────────────────────────────────┘                   │
│         ↓                                                                   │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │ Optional: Save Transfer Model                       │                   │
│  └─────────────────────────────────────────────────────┘                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ SECTION D: Multi-Instrument Equalization                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  (Existing validations - check subdirectories, minimum 2 instruments)      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ SECTION E: Predict with Transfer Model                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │ User clicks "Load and Predict"                      │                   │
│  └─────────────────────────────────────────────────────┘                   │
│         ↓                                                                   │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │ ✓ VALIDATION: Master Model Loaded                   │                   │
│  │   ct_master_model_dict is not None?                │                   │
│  │   → ERROR: "Master Model Not Loaded"                │                   │
│  │            "Load master model in Section A first"   │                   │
│  └─────────────────────────────────────────────────────┘                   │
│         ↓ PASS                                                              │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │ ✓ VALIDATION: Transfer Model Loaded                 │                   │
│  │   ct_pred_transfer_model is not None?              │                   │
│  │   → ERROR: "Transfer Model Not Loaded"              │                   │
│  │            "Load/build model in Section C first"    │                   │
│  └─────────────────────────────────────────────────────┘                   │
│         ↓ PASS                                                              │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │ Load new slave spectra                              │                   │
│  │  wavelengths_slave, X_slave_new                     │                   │
│  └─────────────────────────────────────────────────────┘                   │
│         ↓                                                                   │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │ ✓ VALIDATION: Wavelength Compatibility              │                   │
│  │   new_range covers transfer_range?                  │                   │
│  │   → WARNING: "Wavelength Range Mismatch"            │                   │
│  │              "New data has narrower coverage"       │                   │
│  └─────────────────────────────────────────────────────┘                   │
│         ↓ PASS or CONTINUE                                                 │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │ Resample to common grid                             │                   │
│  │ Apply transfer (DS or PDS)                          │                   │
│  │ Resample to master model wavelengths                │                   │
│  └─────────────────────────────────────────────────────┘                   │
│         ↓                                                                   │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │ ✓ VALIDATION: Extrapolation Check                   │                   │
│  │   wl_model within model training range?             │                   │
│  │   → WARNING: "Extrapolation Warning"                │                   │
│  │              "Predictions may be unreliable"        │                   │
│  └─────────────────────────────────────────────────────┘                   │
│         ↓ PASS or CONTINUE                                                 │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │ Apply preprocessing (if present)                    │                   │
│  │ Predict using master model                          │                   │
│  └─────────────────────────────────────────────────────┘                   │
│         ↓                                                                   │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │ Display predictions                                 │                   │
│  │ Store: ct_pred_y_pred                               │                   │
│  └─────────────────────────────────────────────────────┘                   │
│         ↓                                                                   │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │ Optional: Export Predictions                        │                   │
│  └─────────────────────────────────────────────────────┘                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
                               VALIDATION SUMMARY
═══════════════════════════════════════════════════════════════════════════════

┌──────────┬─────────────────────────────────────┬──────────┬────────────────┐
│ Section  │ Validation Check                    │ Type     │ Can Skip?      │
├──────────┼─────────────────────────────────────┼──────────┼────────────────┤
│ B        │ Same Instrument (pre-load)          │ ERROR    │ No             │
│ B        │ Sample Count Mismatch               │ ERROR    │ No             │
│ B        │ Few Samples (< 20)                  │ WARNING  │ Yes (choice)   │
│ B        │ No Wavelength Overlap               │ ERROR    │ No             │
│ B        │ Limited Overlap (< 80%)             │ WARNING  │ Yes (choice)   │
│ C        │ Data Not Loaded (hasattr)           │ ERROR    │ No             │
│ C        │ Data Not Loaded (None)              │ ERROR    │ No             │
│ C        │ Different Instruments               │ ERROR    │ No             │
│ C        │ DS Lambda Range                     │ ERROR    │ No             │
│ C        │ DS Lambda Type                      │ ERROR    │ No             │
│ C        │ PDS Window Range                    │ ERROR    │ No             │
│ C        │ PDS Window Odd                      │ ERROR    │ No             │
│ C        │ PDS Window Type                     │ ERROR    │ No             │
│ E        │ Master Model Not Loaded             │ ERROR    │ No             │
│ E        │ Transfer Model Not Loaded           │ ERROR    │ No             │
│ E        │ Wavelength Range Mismatch           │ WARNING  │ Yes (auto)     │
│ E        │ Extrapolation Warning               │ WARNING  │ Yes (auto)     │
└──────────┴─────────────────────────────────────┴──────────┴────────────────┘

Total: 17 checks (13 errors, 4 warnings)

═══════════════════════════════════════════════════════════════════════════════
                            VALIDATION DECISION TREE
═══════════════════════════════════════════════════════════════════════════════

Can I build a calibration transfer model?
│
├─ Have I loaded paired spectra? ───→ NO  → Load in Section B
│  └─ YES ↓
│
├─ Are master and slave different? ───→ NO  → Select different instruments
│  └─ YES ↓
│
├─ Do they have same sample count? ───→ NO  → Get same samples on both
│  └─ YES ↓
│
├─ Do they have overlapping wavelengths? ───→ NO  → Use compatible instruments
│  └─ YES ↓
│
├─ Are my parameters valid (Lambda/Window)? ───→ NO  → Fix parameters
│  └─ YES ↓
│
└─ ✓ Ready to build transfer model!


Can I predict on new slave data?
│
├─ Have I loaded master model? ───→ NO  → Load in Section A
│  └─ YES ↓
│
├─ Have I loaded/built transfer model? ───→ NO  → Load/build in Section C
│  └─ YES ↓
│
├─ Does new slave data have sufficient wavelength coverage?
│  ├─ NO  → ⚠ WARNING (can continue)
│  └─ YES ↓
│
├─ Will predictions be within training range?
│  ├─ NO  → ⚠ WARNING (can continue)
│  └─ YES ↓
│
└─ ✓ Ready to predict!

═══════════════════════════════════════════════════════════════════════════════

Legend:
  ✓ = Validation passed
  → ERROR = Blocks operation, user must fix
  → WARNING = User can choose to continue
  ⚠ = Warning shown but operation continues
