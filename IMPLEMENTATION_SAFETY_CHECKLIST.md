# Implementation Safety Checklist

## Critical Safety Rules

### ✅ MUST DO
1. **Default behavior MUST be unchanged** - All new features default to OFF/current behavior
2. **All changes are OPT-IN** - Users must explicitly enable new features
3. **Test in worktree FIRST** - Never touch main codebase until verified
4. **Validate all inputs** - Never trust user input
5. **Log all changes** - Clear progress messages for debugging

### ❌ MUST NOT DO
1. **Never change existing defaults** without explicit toggle
2. **Never remove working code** - only add new paths
3. **Never skip validation** - always check user inputs
4. **Never assume** - test every change
5. **Never commit broken code** - run tests first

---

## Change Summary

### Change 1: Option 1 - Add Column Name Collision Check
**Risk Level**: MINIMAL
**Lines Modified**: ~5 lines in spectral_predict_gui_optimized.py (~2575)
**Behavior Change**: None (only adds safety check)

**Safety Measures**:
- Only adds a check, doesn't modify existing logic
- If collision detected, renames column with suffix
- Logs warning to user

**Testing**:
- [ ] Test with normal column names (no collision)
- [ ] Test with wavelength column matching sample ID name (collision)
- [ ] Verify CSV exports correctly in both cases

---

### Change 2: Option 3 - Document Existing Feature
**Risk Level**: ZERO
**Lines Modified**: 0 (documentation only)
**Behavior Change**: None

**Action**: Just verify it works, no code changes

---

### Change 3: Option 4 - RF Hyperparameter Configuration
**Risk Level**: LOW (with safety measures)
**Lines Modified**: ~200 lines across 3 files
**Behavior Change**: NONE when advanced toggle OFF (default)

#### Safety Measures:

**1. Advanced Toggle Defaults to OFF**
```python
self.rf_enable_advanced = tk.BooleanVar(value=False)  # ✅ OFF by default
```

**2. Safe Defaults When Toggle OFF**
```python
# Current behavior: [None, 15, 30] = 6 configs per n_estimators
# NEW behavior:     [None, 30]     = 4 configs per n_estimators ✅ FASTER
if not self.rf_enable_advanced.get():
    rf_max_depth_list = [None, 30]  # Safe default
```

**3. Input Validation**
```python
# Validate max_depth
- Must be positive integer or None
- Must not exceed reasonable bounds (e.g., < 1000)
- Custom entry sanitized (strip whitespace, check type)
```

**4. Configuration Size Warning**
```python
# Warn if grid search too large
if total_configs > 30:
    show warning dialog → user can cancel
```

**5. Backward Compatibility**
```python
# models.py function signature maintains defaults
def get_model_grids(..., rf_max_depth_list=None):
    if rf_max_depth_list is None:
        rf_max_depth_list = [None, 30]  # Same as new default
```

#### Files Modified:
1. **spectral_predict_gui_optimized.py**
   - Add variables (lines ~232-240)
   - Add GUI controls (lines ~897-920)
   - Add parameter collection (lines ~2786-2810)
   - Add validation function

2. **src/spectral_predict/models.py**
   - Update function signature (line ~110)
   - Update RF grid generation (lines ~180-205)
   - Add defaults

3. **src/spectral_predict/search.py**
   - Update function signature (line ~20)
   - Pass parameters through (~line 140)

#### Testing Checklist:
- [ ] Test with advanced toggle OFF → verify uses [None, 30] defaults
- [ ] Test with advanced toggle ON + only defaults selected → same as OFF
- [ ] Test with custom max_depth values → verify they're used
- [ ] Test invalid inputs → verify validation catches them
- [ ] Test with current sample data → verify results match or improve
- [ ] Compare results with current version → no regression

---

### Change 4: Unified Complexity Score
**Risk Level**: MINIMAL
**Lines Modified**: ~100 lines (all new code)
**Behavior Change**: None (adds new column only)

#### Safety Measures:

**1. Purely Additive**
```python
# Add NEW column, don't modify existing CompositeScore
df["ComplexityScore"] = compute_unified_complexity(...)

# Existing ranking UNCHANGED
df["Rank"] = df["CompositeScore"].rank(...)  # Still uses CompositeScore
```

**2. No Dependencies**
```python
# Doesn't affect:
- Model training
- Result ranking
- CSV exports (unless explicitly included)
- Existing plots
```

**3. Graceful Degradation**
```python
# If calculation fails, set to NaN (don't break pipeline)
try:
    complexity = compute_unified_complexity(...)
except Exception as e:
    complexity = np.nan
    log_warning(f"Complexity calculation failed: {e}")
```

#### Testing Checklist:
- [ ] Verify scores are 0-100 range
- [ ] Verify simple models get low scores
- [ ] Verify complex models get high scores
- [ ] Verify ranking still uses CompositeScore (unchanged)
- [ ] Test with all model types (PLS, RF, MLP, etc.)

---

## Pre-Implementation Verification

### Current State Snapshot
```bash
# Save current state for comparison
git status
git diff
git log -1

# Verify we're in worktree
pwd  # Should be /home/user/dasp-add-columns-wt
```

### Backup Strategy
- Working in worktree (isolated from main)
- Branch: claude/add-columns-options-011CUwEFNqijV5S6r1hSWf8e-wt
- Can revert any change instantly with git

---

## Implementation Order

### Phase 1: Low-Risk Changes (Do First)
1. ✅ Change 1: Column collision check (5 lines)
2. ✅ Change 4: Unified complexity score (100 lines, additive)
3. Test both, commit if successful

### Phase 2: Medium-Risk Changes (Do After Phase 1 Works)
4. ⚠️ Change 3: RF hyperparameters (200 lines, with toggle)
5. Extensive testing, commit only if all tests pass

### Phase 3: Verification
6. Run full analysis with sample data
7. Compare with current version results
8. Verify no performance regression
9. Verify all defaults match or improve current behavior

---

## Rollback Plan

If anything breaks:
```bash
# In worktree
git reset --hard HEAD  # Undo uncommitted changes
git reset --hard <previous-commit>  # Undo last commit

# Worst case: delete worktree, start over
cd /home/user/dasp
git worktree remove ../dasp-add-columns-wt
```

---

## Success Criteria

### Must Pass ALL:
- [ ] All unit tests pass
- [ ] Integration test with sample data produces valid results
- [ ] Results with defaults match or improve current version
- [ ] No errors or warnings in console
- [ ] CSV exports correctly
- [ ] GUI responsive and no crashes
- [ ] Code review: no obvious bugs
- [ ] Documentation updated

### Performance:
- [ ] Analysis runtime not significantly slower (advanced OFF)
- [ ] Memory usage stable
- [ ] No infinite loops or hangs

---

## Final Pre-Commit Checklist

Before pushing:
- [ ] All tests passing
- [ ] Code reviewed line-by-line
- [ ] No debugging print statements left
- [ ] No commented-out code
- [ ] All TODOs addressed or documented
- [ ] Commit message descriptive
- [ ] Changes match design documents
- [ ] No secrets or sensitive data in code

---

## Validation Commands

```bash
# In worktree directory
cd /home/user/dasp-add-columns-wt

# Run Python syntax check
python -m py_compile spectral_predict_gui_optimized.py
python -m py_compile src/spectral_predict/models.py
python -m py_compile src/spectral_predict/search.py
python -m py_compile src/spectral_predict/scoring.py

# Run any unit tests
pytest tests/ -v

# Check for obvious issues
grep -r "TODO" .
grep -r "FIXME" .
grep -r "print(" src/  # Find debug prints
```

---

## Documentation Updates Needed

- [ ] Update START_HERE.md with new features
- [ ] Update CHANGELOG (if exists)
- [ ] Add comments to new code sections
- [ ] Update function docstrings
