# Ensemble Methods Implementation Plan
## Multi-Agent Development Strategy

**Project:** Intelligent Ensemble Methods for Spectral Predict
**Duration:** 16-19 hours (2-3 days focused work)
**Team Size:** 1 Lead + 4-5 Developers
**Last Updated:** 2025-11-09

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Team Roles & Responsibilities](#team-roles--responsibilities)
3. [Architecture Overview](#architecture-overview)
4. [Parallel Work Tracks](#parallel-work-tracks)
5. [Detailed Task Specifications](#detailed-task-specifications)
6. [Integration & Testing Strategy](#integration--testing-strategy)
7. [Dependencies & Coordination Points](#dependencies--coordination-points)
8. [Acceptance Criteria](#acceptance-criteria)
9. [Risk Mitigation](#risk-mitigation)

---

## Executive Summary

### Problem Statement
Current ensemble functionality (lines 3352-3394 in `spectral_predict_gui_optimized.py`) is configured in the GUI but not fully implemented. Models are selected arbitrarily (top 5 by CompositeScore), and ensembles cannot be saved/loaded.

### Key Insight
**Regional diversity matters more than arbitrary ranking.** Models ranked 6-30 may have complementary regional strengths (excelling at different value ranges) that simple top-5 selection misses.

### Solution Overview
Implement end-to-end ensemble pipeline:
1. **Smart selection** using regional performance diversity (Q1-Q4 RMSE)
2. **Model reconstruction** from results dataframe
3. **Ensemble training** during Tab 4 analysis
4. **Save/load infrastructure** for ensemble .dasp files
5. **Tab 7 integration** for ensemble predictions
6. **Visualization** of regional performance and ensemble weights

### Deliverables
- 3 new modules: `ensemble_selection.py`, `ensemble_io.py`, updated `model_io.py`
- Modified GUI: Tab 3 (config), Tab 4 (training), Tab 5 (results), Tab 7 (prediction)
- Comprehensive test suite
- Documentation and examples

---

## Team Roles & Responsibilities

### **Lead Agent (Orchestrator)**
**Responsibilities:**
- Overall architecture coordination
- Code review and integration
- Dependency management
- Testing strategy oversight
- Final acceptance testing
- Documentation review

**Key Tasks:**
- Define interfaces between modules
- Review all PRs before merge
- Coordinate integration milestones
- Run end-to-end integration tests
- Ensure coding standards compliance

**Time Commitment:** 4-5 hours (25% of total)

---

### **Agent 1: Model Selection Specialist**
**Focus:** Phase 1 - Smart Ensemble Selection Algorithm

**Deliverables:**
1. `src/spectral_predict/ensemble_selection.py` (new file)
   - `select_ensemble_candidates()` function
   - Regional diversity analysis
   - Architecture diversity scoring
   - Preprocessing diversity analysis
2. Unit tests for selection logic
3. Documentation with examples

**Time Commitment:** 3-4 hours

**Key Files to Modify:**
- CREATE: `src/spectral_predict/ensemble_selection.py`
- CREATE: `tests/test_ensemble_selection.py`

**Dependencies:**
- None (can start immediately)
- Provides interface for Agent 2 and Agent 3

---

### **Agent 2: Model Reconstruction Engineer**
**Focus:** Phase 2 - Rebuild Fitted Models from Results

**Deliverables:**
1. Model reconstruction function in `ensemble_selection.py`
   - `reconstruct_and_fit_models()` function
   - Hyperparameter parsing from string
   - Preprocessing pipeline rebuilding
   - Feature subset handling
2. Validation tests comparing reconstructed vs original models
3. Edge case handling (missing params, failed fits)

**Time Commitment:** 2-3 hours

**Key Files to Modify:**
- MODIFY: `src/spectral_predict/ensemble_selection.py`
- CREATE: `tests/test_model_reconstruction.py`

**Dependencies:**
- Parallel with Agent 1 (same file, different functions)
- Coordinate with Agent 1 on file structure

---

### **Agent 3: GUI Integration Developer**
**Focus:** Phase 3 - Tab 4 Analysis Integration

**Deliverables:**
1. Ensemble training in `_run_analysis_thread()` (lines 3352-3394)
   - Call selection algorithm
   - Reconstruct models
   - Train each enabled ensemble type
   - Evaluate ensemble performance
   - Log results with regional breakdown
2. Save ensemble option in Tab 5
3. Progress logging improvements

**Time Commitment:** 3-4 hours

**Key Files to Modify:**
- MODIFY: `spectral_predict_gui_optimized.py` (lines 3352-3394)
- MODIFY: Tab 5 results display (lines 3600-3654)

**Dependencies:**
- REQUIRES: Agent 1 and Agent 2 completion (selection + reconstruction)
- REQUIRES: Agent 4 progress (for saving ensemble)

---

### **Agent 4: Persistence Architect**
**Focus:** Phase 4 - Ensemble Save/Load Infrastructure

**Deliverables:**
1. `src/spectral_predict/ensemble_io.py` (new file)
   - `save_ensemble()` function
   - `load_ensemble()` function
   - Ensemble metadata schema
2. Extend `model_io.py` to detect ensemble files
3. .dasp format specification for ensembles
4. Comprehensive I/O tests

**Time Commitment:** 4-5 hours

**Key Files to Modify:**
- CREATE: `src/spectral_predict/ensemble_io.py`
- MODIFY: `src/spectral_predict/model_io.py` (lines 149-225)
- CREATE: `tests/test_ensemble_io.py`

**Dependencies:**
- Parallel with Agents 1-3 (can work independently)
- Coordinates with Agent 5 on Tab 7 loading

---

### **Agent 5: Prediction Integration Specialist**
**Focus:** Phase 5 - Tab 7 Ensemble Predictions

**Deliverables:**
1. Ensemble detection in model loading (lines 5456-5514)
2. Ensemble prediction logic (lines 5685-5786)
3. Ensemble metadata display
4. Handle mixed model types (individual + ensemble)
5. Update consensus logic to use trained ensembles

**Time Commitment:** 2-3 hours

**Key Files to Modify:**
- MODIFY: `spectral_predict_gui_optimized.py`
  - `_load_model_for_prediction()` (lines 5456-5514)
  - `_run_predictions()` (lines 5685-5786)
  - `_add_consensus_predictions()` (lines 5788-5915)

**Dependencies:**
- REQUIRES: Agent 4 completion (load_ensemble function)
- Can start UI mockups in parallel

---

### **Agent 6 (Optional): Visualization & Documentation**
**Focus:** Phase 6 - Reporting and User Experience

**Deliverables:**
1. Regional performance heatmaps
2. Ensemble weights visualization
3. Model specialization profiles
4. Updated user documentation
5. Example notebooks/scripts

**Time Commitment:** 2-3 hours

**Key Files to Modify:**
- MODIFY: `src/spectral_predict/ensemble_viz.py` (already exists)
- CREATE: `docs/ENSEMBLE_USAGE_GUIDE.md`
- CREATE: `examples/ensemble_example.py`

**Dependencies:**
- Can work in parallel (uses existing ensemble.py classes)
- Integrates into Tab 5 results after Agent 3 completion

---

## Architecture Overview

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TAB 4: ANALYSIS PROGRESS                         │
│                                                                      │
│  1. Run Individual Models (existing)                                │
│     ↓                                                                │
│  2. results_df with 300-1000 model evaluations                      │
│     ├── CompositeScore, RMSE, R2                                    │
│     ├── regional_rmse {Q1, Q2, Q3, Q4}                              │
│     ├── Model, Preprocess, Params                                   │
│     └── all_vars (wavelength subsets)                               │
│     ↓                                                                │
│  3. ┌──────────────────────────────────────────┐                    │
│     │ SMART SELECTION (Agent 1)                │                    │
│     │ - Analyze top 30 models                  │                    │
│     │ - Find Q1/Q2/Q3/Q4 specialists           │                    │
│     │ - Ensure architecture diversity          │                    │
│     │ - Return 5 complementary models          │                    │
│     └──────────────────────────────────────────┘                    │
│     ↓                                                                │
│  4. ┌──────────────────────────────────────────┐                    │
│     │ MODEL RECONSTRUCTION (Agent 2)           │                    │
│     │ - Parse Params column                    │                    │
│     │ - Rebuild preprocessing pipelines        │                    │
│     │ - Handle feature subsets                 │                    │
│     │ - Fit on full X_train, y_train           │                    │
│     └──────────────────────────────────────────┘                    │
│     ↓                                                                │
│  5. List of 5 fitted sklearn.pipeline.Pipeline objects              │
│     ↓                                                                │
│  6. ┌──────────────────────────────────────────┐                    │
│     │ ENSEMBLE TRAINING (Agent 3)              │                    │
│     │ - RegionAwareWeightedEnsemble            │                    │
│     │ - MixtureOfExpertsEnsemble               │                    │
│     │ - StackingEnsemble                       │                    │
│     │ - Evaluate performance                   │                    │
│     └──────────────────────────────────────────┘                    │
│     ↓                                                                │
│  7. Trained ensemble object + metadata                              │
│     ↓                                                                │
│  8. ┌──────────────────────────────────────────┐                    │
│     │ SAVE ENSEMBLE (Agent 4)                  │                    │
│     │ ensemble_model.dasp (ZIP):               │                    │
│     │   ├── metadata.json                      │                    │
│     │   ├── ensemble_weights.npz               │                    │
│     │   ├── base_model_0.dasp                  │                    │
│     │   └── base_model_1-4.dasp                │                    │
│     └──────────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    TAB 7: MODEL PREDICTION                          │
│                                                                      │
│  1. Load .dasp file(s)                                              │
│     ↓                                                                │
│  2. ┌──────────────────────────────────────────┐                    │
│     │ DETECT ENSEMBLE (Agent 5)                │                    │
│     │ - Check metadata.is_ensemble             │                    │
│     │ - If ensemble: load_ensemble()           │                    │
│     │ - If regular: load_model()               │                    │
│     └──────────────────────────────────────────┘                    │
│     ↓                                                                │
│  3. Load prediction data (X_new)                                    │
│     ↓                                                                │
│  4. ┌──────────────────────────────────────────┐                    │
│     │ ENSEMBLE PREDICTION (Agent 5)            │                    │
│     │ - ensemble.predict(X_new)                │                    │
│     │ - Applies regional weights dynamically   │                    │
│     └──────────────────────────────────────────┘                    │
│     ↓                                                                │
│  5. predictions_df with ensemble column                             │
└─────────────────────────────────────────────────────────────────────┘
```

### Module Dependencies

```
ensemble_selection.py (Agent 1 & 2)
├── select_ensemble_candidates()
│   ├── Uses: results_df['regional_rmse']
│   ├── Uses: results_df['Model', 'Preprocess']
│   └── Returns: DataFrame with selected model indices
└── reconstruct_and_fit_models()
    ├── Uses: spectral_predict.models.get_model()
    ├── Uses: spectral_predict.preprocess.build_preprocessing_pipeline()
    └── Returns: (models, model_names, preprocessors)

ensemble_io.py (Agent 4)
├── save_ensemble()
│   ├── Uses: model_io.save_model() for base models
│   ├── Uses: numpy.savez_compressed() for weights
│   └── Creates: ZIP file with embedded .dasp files
└── load_ensemble()
    ├── Uses: model_io.load_model() for base models
    ├── Uses: numpy.load() for weights
    └── Reconstructs: ensemble object from saved state

spectral_predict_gui_optimized.py (Agent 3 & 5)
├── Tab 4: _run_analysis_thread()
│   ├── Calls: ensemble_selection.select_ensemble_candidates()
│   ├── Calls: ensemble_selection.reconstruct_and_fit_models()
│   ├── Calls: ensemble.create_ensemble()
│   └── Calls: ensemble_io.save_ensemble()
└── Tab 7: _load_model_for_prediction()
    ├── Calls: ensemble_io.load_ensemble()
    └── Calls: ensemble.predict()

ensemble_viz.py (Agent 6)
├── plot_regional_performance()
├── plot_ensemble_weights()
└── plot_model_specialization_profile()
```

---

## Parallel Work Tracks

### Track A: Selection & Reconstruction (Days 1-2)
**Agents:** 1 & 2 (working in same file)

**Coordination:**
- Agent 1 starts first: File structure + selection algorithm
- Agent 2 joins after 1 hour: Reconstruction function
- Both work in `ensemble_selection.py`
- Daily sync on function signatures

**Milestones:**
- Hour 2: Selection algorithm complete, tested with sample results_df
- Hour 4: Reconstruction complete, tested with real model configs
- Hour 5: Integration test showing 5 fitted models from results_df

---

### Track B: Persistence Layer (Days 1-2, parallel)
**Agent:** 4

**Coordination:**
- Works independently on ensemble I/O
- Defines .dasp format spec (share with Lead)
- Can use mock ensemble objects for testing
- No blocking dependencies

**Milestones:**
- Hour 2: Format specification documented
- Hour 4: save_ensemble() complete with tests
- Hour 5: load_ensemble() complete with tests
- Hour 5: Round-trip test (save → load → predict) passes

---

### Track C: GUI Integration (Day 2, depends on A & B)
**Agent:** 3

**Coordination:**
- WAITS for Track A completion (hour 5)
- Can prep code structure in parallel
- Coordinates with Agent 4 on save interface
- Lead reviews integration points

**Milestones:**
- Hour 2: Code structure ready, imports stubbed
- Hour 5: Full integration complete
- Hour 6: End-to-end test: GUI analysis → ensemble saved
- Hour 7: Results display shows ensemble metrics

---

### Track D: Prediction Integration (Day 2-3, depends on B)
**Agent:** 5

**Coordination:**
- WAITS for Track B completion (hour 5)
- Can work on UI mockups in parallel
- Tests with saved ensemble from Track C
- Coordinates with Agent 3 on data format

**Milestones:**
- Hour 2: Ensemble detection logic complete
- Hour 3: Prediction logic complete
- Hour 4: Tab 7 integration tested with sample ensemble

---

### Track E: Visualization (Day 3, parallel)
**Agent:** 6 (Optional)

**Coordination:**
- Works independently
- Uses existing ensemble.py classes
- Creates standalone visualization functions
- Agent 3 integrates into GUI later

**Milestones:**
- Hour 2: Regional performance heatmap working
- Hour 3: All visualizations complete
- Hour 3: Documentation and examples complete

---

## Detailed Task Specifications

---

## AGENT 1: Model Selection Specialist

### Task 1.1: Create `ensemble_selection.py` File Structure
**Duration:** 30 minutes

**Specification:**
```python
"""
Intelligent model selection for ensemble methods.

This module implements smart model selection based on regional performance
diversity rather than arbitrary ranking by composite score.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any


def select_ensemble_candidates(
    results_df: pd.DataFrame,
    top_n: int = 30,
    ensemble_size: int = 5,
    ensure_diversity: bool = True
) -> pd.DataFrame:
    """
    Select models with complementary regional performance.

    Strategy:
    1. Filter to top N candidates by CompositeScore (proven performers)
    2. Analyze regional RMSE patterns (Q1, Q2, Q3, Q4)
    3. Select models that excel in DIFFERENT regions
    4. Ensure architectural diversity (linear, tree, neural)
    5. Ensure preprocessing diversity

    Parameters
    ----------
    results_df : pd.DataFrame
        Full results from analysis with columns:
        - CompositeScore
        - regional_rmse (dict with Q1, Q2, Q3, Q4)
        - Model
        - Preprocess
        - all other metadata
    top_n : int, default=30
        Number of top models to consider (by CompositeScore)
    ensemble_size : int, default=5
        Number of models to select for ensemble
    ensure_diversity : bool, default=True
        Whether to enforce architecture and preprocessing diversity

    Returns
    -------
    selected_df : pd.DataFrame
        Subset of results_df with selected models
        Sorted by selection criteria (not necessarily CompositeScore)

    Notes
    -----
    Selection priority:
    1. Best overall (lowest CompositeScore)
    2. Q1 specialist (best at predicting low values)
    3. Q4 specialist (best at predicting high values)
    4. Different architecture (if all above are same type)
    5. Different preprocessing (maximizes feature diversity)

    The function logs selection rationale for transparency.
    """
    pass


def _extract_regional_performance(
    results_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Extract regional RMSE into structured dataframe.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results with 'regional_rmse' column (dict)

    Returns
    -------
    regional_df : pd.DataFrame
        Columns: index, Q1_RMSE, Q2_RMSE, Q3_RMSE, Q4_RMSE, Model, Preprocess
    """
    pass


def _calculate_diversity_scores(
    candidates: pd.DataFrame,
    selected_indices: List[int]
) -> pd.Series:
    """
    Calculate diversity scores for remaining candidates.

    Scores based on:
    - Different Model type than selected
    - Different Preprocess method than selected
    - Complementary regional performance

    Parameters
    ----------
    candidates : pd.DataFrame
        Remaining candidates
    selected_indices : List[int]
        Already selected model indices

    Returns
    -------
    diversity_scores : pd.Series
        Higher score = more diverse from selected models
    """
    pass


def _classify_architecture(model_name: str) -> str:
    """
    Classify model into architecture family.

    Returns
    -------
    architecture : str
        One of: 'linear', 'tree', 'neural', 'kernel', 'pls'
    """
    architecture_map = {
        'PLS': 'pls',
        'Ridge': 'linear',
        'Lasso': 'linear',
        'ElasticNet': 'linear',
        'RandomForest': 'tree',
        'XGBoost': 'tree',
        'LightGBM': 'tree',
        'CatBoost': 'tree',
        'MLP': 'neural',
        'NeuralBoosted': 'neural',
        'SVR': 'kernel',
        'SVM': 'kernel'
    }
    return architecture_map.get(model_name, 'unknown')
```

**Acceptance Criteria:**
- [ ] File created with docstrings
- [ ] All function signatures defined
- [ ] Type hints included
- [ ] Architecture map complete

---

### Task 1.2: Implement Regional Performance Extraction
**Duration:** 30 minutes

**Code Location:** `ensemble_selection.py::_extract_regional_performance()`

**Implementation:**
```python
def _extract_regional_performance(results_df: pd.DataFrame) -> pd.DataFrame:
    """Extract regional RMSE into structured dataframe."""

    regional_data = []

    for idx, row in results_df.iterrows():
        regional_rmse = row.get('regional_rmse', None)

        # Handle missing regional data
        if regional_rmse is None or not isinstance(regional_rmse, dict):
            # Use overall RMSE as fallback
            overall_rmse = row.get('RMSE', np.nan)
            regional_rmse = {
                'Q1': overall_rmse,
                'Q2': overall_rmse,
                'Q3': overall_rmse,
                'Q4': overall_rmse
            }

        regional_data.append({
            'index': idx,
            'Q1_RMSE': regional_rmse.get('Q1', np.nan),
            'Q2_RMSE': regional_rmse.get('Q2', np.nan),
            'Q3_RMSE': regional_rmse.get('Q3', np.nan),
            'Q4_RMSE': regional_rmse.get('Q4', np.nan),
            'Model': row['Model'],
            'Preprocess': row['Preprocess'],
            'CompositeScore': row['CompositeScore']
        })

    return pd.DataFrame(regional_data)
```

**Test Cases:**
```python
def test_extract_regional_performance():
    # Mock results_df
    results = pd.DataFrame({
        'Model': ['PLS', 'XGBoost'],
        'Preprocess': ['snv', 'raw'],
        'CompositeScore': [0.5, 0.6],
        'RMSE': [0.2, 0.22],
        'regional_rmse': [
            {'Q1': 0.18, 'Q2': 0.20, 'Q3': 0.21, 'Q4': 0.23},
            {'Q1': 0.25, 'Q2': 0.22, 'Q3': 0.20, 'Q4': 0.21}
        ]
    })

    regional_df = _extract_regional_performance(results)

    assert len(regional_df) == 2
    assert 'Q1_RMSE' in regional_df.columns
    assert regional_df.loc[0, 'Q1_RMSE'] == 0.18

def test_extract_missing_regional_data():
    # Test fallback to overall RMSE
    results = pd.DataFrame({
        'Model': ['PLS'],
        'Preprocess': ['snv'],
        'CompositeScore': [0.5],
        'RMSE': [0.2],
        'regional_rmse': [None]  # Missing
    })

    regional_df = _extract_regional_performance(results)

    assert regional_df.loc[0, 'Q1_RMSE'] == 0.2  # Fallback
```

**Acceptance Criteria:**
- [ ] Handles valid regional_rmse dicts
- [ ] Handles None/missing regional data (fallback to RMSE)
- [ ] Returns DataFrame with expected columns
- [ ] All tests pass

---

### Task 1.3: Implement Core Selection Algorithm
**Duration:** 1.5 hours

**Code Location:** `ensemble_selection.py::select_ensemble_candidates()`

**Implementation:**
```python
def select_ensemble_candidates(
    results_df: pd.DataFrame,
    top_n: int = 30,
    ensemble_size: int = 5,
    ensure_diversity: bool = True
) -> pd.DataFrame:
    """Select models with complementary regional performance."""

    import logging
    logger = logging.getLogger(__name__)

    # Validate inputs
    if len(results_df) < ensemble_size:
        logger.warning(f"Only {len(results_df)} models available, need {ensemble_size}")
        return results_df.copy()

    # Step 1: Filter to top N candidates
    candidates = results_df.nsmallest(top_n, 'CompositeScore').copy()
    logger.info(f"Considering top {len(candidates)} models (by CompositeScore)")

    # Step 2: Extract regional performance
    regional_df = _extract_regional_performance(candidates)

    # Step 3: Initialize selection
    selected_indices = []
    selection_reasons = []

    # Step 3a: Best overall
    best_overall_idx = candidates['CompositeScore'].idxmin()
    selected_indices.append(best_overall_idx)
    selection_reasons.append(f"Best overall (CompositeScore: {candidates.loc[best_overall_idx, 'CompositeScore']:.4f})")

    # Step 3b: Q1 specialist (best at low values)
    if len(selected_indices) < ensemble_size:
        # Filter out already selected
        remaining_regional = regional_df[~regional_df['index'].isin(selected_indices)]

        if len(remaining_regional) > 0:
            q1_specialist_row = remaining_regional.nsmallest(1, 'Q1_RMSE').iloc[0]
            q1_specialist_idx = q1_specialist_row['index']
            selected_indices.append(q1_specialist_idx)
            selection_reasons.append(
                f"Q1 specialist (excels at low values, Q1 RMSE: {q1_specialist_row['Q1_RMSE']:.4f})"
            )

    # Step 3c: Q4 specialist (best at high values)
    if len(selected_indices) < ensemble_size:
        remaining_regional = regional_df[~regional_df['index'].isin(selected_indices)]

        if len(remaining_regional) > 0:
            q4_specialist_row = remaining_regional.nsmallest(1, 'Q4_RMSE').iloc[0]
            q4_specialist_idx = q4_specialist_row['index']
            selected_indices.append(q4_specialist_idx)
            selection_reasons.append(
                f"Q4 specialist (excels at high values, Q4 RMSE: {q4_specialist_row['Q4_RMSE']:.4f})"
            )

    # Step 3d: Q2/Q3 specialists if needed
    if len(selected_indices) < ensemble_size:
        for quartile in ['Q2', 'Q3']:
            if len(selected_indices) >= ensemble_size:
                break

            remaining_regional = regional_df[~regional_df['index'].isin(selected_indices)]
            if len(remaining_regional) > 0:
                specialist_row = remaining_regional.nsmallest(1, f'{quartile}_RMSE').iloc[0]
                specialist_idx = specialist_row['index']

                # Only add if not already selected
                if specialist_idx not in selected_indices:
                    selected_indices.append(specialist_idx)
                    selection_reasons.append(
                        f"{quartile} specialist ({quartile} RMSE: {specialist_row[f'{quartile}_RMSE']:.4f})"
                    )

    # Step 3e: Architecture diversity
    if ensure_diversity and len(selected_indices) < ensemble_size:
        selected_architectures = [
            _classify_architecture(candidates.loc[idx, 'Model'])
            for idx in selected_indices
        ]

        remaining = candidates[~candidates.index.isin(selected_indices)]

        for _, row in remaining.iterrows():
            if len(selected_indices) >= ensemble_size:
                break

            arch = _classify_architecture(row['Model'])
            if arch not in selected_architectures:
                selected_indices.append(row.name)
                selected_architectures.append(arch)
                selection_reasons.append(
                    f"Architecture diversity ({arch}, CompositeScore: {row['CompositeScore']:.4f})"
                )

    # Step 3f: Preprocessing diversity
    if ensure_diversity and len(selected_indices) < ensemble_size:
        selected_preprocs = [
            candidates.loc[idx, 'Preprocess']
            for idx in selected_indices
        ]

        remaining = candidates[~candidates.index.isin(selected_indices)]

        for _, row in remaining.iterrows():
            if len(selected_indices) >= ensemble_size:
                break

            if row['Preprocess'] not in selected_preprocs:
                selected_indices.append(row.name)
                selected_preprocs.append(row['Preprocess'])
                selection_reasons.append(
                    f"Preprocessing diversity ({row['Preprocess']}, CompositeScore: {row['CompositeScore']:.4f})"
                )

    # Step 3g: Fill remaining slots with next best
    if len(selected_indices) < ensemble_size:
        remaining = candidates[~candidates.index.isin(selected_indices)]
        needed = ensemble_size - len(selected_indices)

        for idx in remaining.nsmallest(needed, 'CompositeScore').index:
            selected_indices.append(idx)
            selection_reasons.append(
                f"Next best (CompositeScore: {candidates.loc[idx, 'CompositeScore']:.4f})"
            )

    # Step 4: Create selected dataframe
    selected_df = candidates.loc[selected_indices].copy()

    # Log selection rationale
    logger.info(f"\nSelected {len(selected_df)} models for ensemble:")
    for i, (idx, reason) in enumerate(zip(selected_indices, selection_reasons), 1):
        row = selected_df.loc[idx]
        regional = row.get('regional_rmse', {})
        logger.info(
            f"  {i}. {row['Model']} ({row['Preprocess']}) - {reason}\n"
            f"     Regional: Q1={regional.get('Q1', 'N/A'):.3f}, "
            f"Q2={regional.get('Q2', 'N/A'):.3f}, "
            f"Q3={regional.get('Q3', 'N/A'):.3f}, "
            f"Q4={regional.get('Q4', 'N/A'):.3f}"
        )

    return selected_df
```

**Test Cases:**
```python
def test_select_ensemble_candidates_basic():
    """Test basic selection with 30 candidates."""
    # Create mock results with regional performance
    np.random.seed(42)
    results = create_mock_results(n=50)

    selected = select_ensemble_candidates(results, top_n=30, ensemble_size=5)

    assert len(selected) == 5
    assert all(col in selected.columns for col in ['Model', 'Preprocess', 'regional_rmse'])

def test_select_ensemble_regional_diversity():
    """Test that selected models have diverse regional performance."""
    results = create_mock_results_with_specialists(n=50)

    selected = select_ensemble_candidates(results, top_n=30, ensemble_size=5)

    # Check that we have Q1 and Q4 specialists
    regional_df = _extract_regional_performance(selected)

    # Should have at least one model good at Q1
    q1_best = regional_df['Q1_RMSE'].min()
    assert q1_best < regional_df['Q1_RMSE'].median()

    # Should have at least one model good at Q4
    q4_best = regional_df['Q4_RMSE'].min()
    assert q4_best < regional_df['Q4_RMSE'].median()

def test_select_ensemble_architecture_diversity():
    """Test architecture diversity enforcement."""
    results = create_mock_results_varied_models(n=50)

    selected = select_ensemble_candidates(
        results, top_n=30, ensemble_size=5, ensure_diversity=True
    )

    architectures = [_classify_architecture(row['Model']) for _, row in selected.iterrows()]

    # Should have at least 2 different architectures
    assert len(set(architectures)) >= 2
```

**Acceptance Criteria:**
- [ ] Selects exactly `ensemble_size` models
- [ ] Best overall always selected first
- [ ] Q1 and Q4 specialists included (if different from best overall)
- [ ] Architecture diversity enforced when `ensure_diversity=True`
- [ ] Preprocessing diversity enforced when `ensure_diversity=True`
- [ ] Logs selection rationale with regional breakdown
- [ ] All tests pass
- [ ] Handles edge cases (< ensemble_size models, missing regional data)

---

### Task 1.4: Unit Tests
**Duration:** 1 hour

**File:** `tests/test_ensemble_selection.py`

**Test Coverage:**
- Regional performance extraction (normal, missing data)
- Architecture classification (all 11 model types)
- Selection algorithm (various scenarios)
- Edge cases (small datasets, identical models, missing columns)
- Diversity enforcement (on/off)

**Acceptance Criteria:**
- [ ] Test file created
- [ ] >90% code coverage for ensemble_selection.py
- [ ] All tests pass
- [ ] Tests use fixtures from conftest.py

---

## AGENT 2: Model Reconstruction Engineer

### Task 2.1: Hyperparameter Parsing
**Duration:** 30 minutes

**Code Location:** `ensemble_selection.py::_parse_model_params()`

**Specification:**
```python
def _parse_model_params(params_str: str) -> Dict[str, Any]:
    """
    Parse hyperparameter string into dictionary.

    The Params column stores model hyperparameters as a string representation
    of a dictionary, e.g., "{'n_components': 10, 'scale': True}"

    Parameters
    ----------
    params_str : str
        String representation of parameters dict

    Returns
    -------
    params : dict
        Parsed parameters ready for model instantiation

    Notes
    -----
    Uses ast.literal_eval for safe evaluation.
    Handles common variations:
    - Missing braces
    - Extra whitespace
    - Boolean capitalization

    Raises
    ------
    ValueError
        If params_str cannot be parsed
    """
    import ast
    import re

    # Clean string
    params_str = params_str.strip()

    # Handle empty
    if not params_str or params_str == '{}':
        return {}

    # Ensure braces
    if not params_str.startswith('{'):
        params_str = '{' + params_str + '}'

    try:
        params = ast.literal_eval(params_str)
        return params
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Cannot parse params string: {params_str}. Error: {e}")
```

**Test Cases:**
```python
def test_parse_model_params_valid():
    params_str = "{'n_components': 10, 'scale': True}"
    result = _parse_model_params(params_str)
    assert result == {'n_components': 10, 'scale': True}

def test_parse_model_params_empty():
    assert _parse_model_params("") == {}
    assert _parse_model_params("{}") == {}

def test_parse_model_params_no_braces():
    params_str = "'alpha': 1.0, 'fit_intercept': True"
    result = _parse_model_params(params_str)
    assert result == {'alpha': 1.0, 'fit_intercept': True}
```

---

### Task 2.2: Preprocessing Pipeline Rebuilding
**Duration:** 45 minutes

**Code Location:** `ensemble_selection.py::_rebuild_preprocessing()`

**Specification:**
```python
def _rebuild_preprocessing(
    preprocess: str,
    deriv: Optional[int],
    window: Optional[int],
    poly: Optional[int]
) -> List[Tuple[str, Any]]:
    """
    Rebuild preprocessing pipeline from metadata.

    Parameters
    ----------
    preprocess : str
        Preprocessing type: 'raw', 'snv', 'deriv', 'snv_deriv', 'deriv_snv'
    deriv : int or None
        Derivative order (1, 2, or None)
    window : int or None
        Savitzky-Golay window size
    poly : int or None
        Polynomial order

    Returns
    -------
    pipeline_steps : List[Tuple[str, estimator]]
        List of (name, transformer) tuples for sklearn.pipeline.Pipeline

    Notes
    -----
    Uses existing build_preprocessing_pipeline from preprocess.py if available,
    otherwise builds manually.
    """
    from spectral_predict.preprocess import (
        StandardNormalVariate,
        SavgolDerivative,
        build_preprocessing_pipeline
    )

    # Try using existing function first
    try:
        return build_preprocessing_pipeline(preprocess, deriv, window, poly)
    except Exception:
        # Build manually as fallback
        pass

    steps = []

    if preprocess == 'raw':
        # No preprocessing
        return steps

    elif preprocess == 'snv':
        steps.append(('snv', StandardNormalVariate()))

    elif preprocess == 'deriv':
        if deriv and window and poly:
            steps.append(('deriv', SavgolDerivative(
                window_length=window,
                polyorder=poly,
                deriv=deriv
            )))

    elif preprocess == 'snv_deriv':
        # SNV first, then derivative
        steps.append(('snv', StandardNormalVariate()))
        if deriv and window and poly:
            steps.append(('deriv', SavgolDerivative(
                window_length=window,
                polyorder=poly,
                deriv=deriv
            )))

    elif preprocess == 'deriv_snv':
        # Derivative first, then SNV
        if deriv and window and poly:
            steps.append(('deriv', SavgolDerivative(
                window_length=window,
                polyorder=poly,
                deriv=deriv
            )))
        steps.append(('snv', StandardNormalVariate()))

    return steps
```

**Test Cases:**
```python
def test_rebuild_preprocessing_raw():
    steps = _rebuild_preprocessing('raw', None, None, None)
    assert steps == []

def test_rebuild_preprocessing_snv():
    steps = _rebuild_preprocessing('snv', None, None, None)
    assert len(steps) == 1
    assert steps[0][0] == 'snv'

def test_rebuild_preprocessing_deriv():
    steps = _rebuild_preprocessing('deriv', 1, 11, 2)
    assert len(steps) == 1
    assert steps[0][0] == 'deriv'
    assert steps[0][1].deriv == 1
```

---

### Task 2.3: Feature Subset Handling
**Duration:** 45 minutes

**Code Location:** `ensemble_selection.py::_extract_wavelength_subset()`

**Specification:**
```python
def _extract_wavelength_subset(
    X_train: pd.DataFrame,
    subset_tag: str,
    all_vars: str
) -> pd.DataFrame:
    """
    Extract wavelength subset from full training data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Full training data with wavelength columns
    subset_tag : str
        Tag indicating subset type:
        - 'full': Use all wavelengths
        - 'subset_10', 'subset_20', etc.: Use selected wavelengths
        - 'region_X': Use regional wavelengths
    all_vars : str
        Comma-separated string of selected wavelength values
        e.g., "402.5,435.2,450.1,..."

    Returns
    -------
    X_subset : pd.DataFrame
        Training data with only selected wavelengths

    Notes
    -----
    Handles floating-point wavelength matching with 0.01 nm tolerance.
    """
    # Full spectrum case
    if subset_tag == 'full':
        return X_train

    # Parse wavelength list
    try:
        selected_wavelengths = [float(w.strip()) for w in all_vars.split(',')]
    except (ValueError, AttributeError) as e:
        raise ValueError(f"Cannot parse wavelengths from all_vars: {all_vars}. Error: {e}")

    # Match wavelengths with tolerance
    matched_columns = []
    tolerance = 0.01  # nm

    for target_wl in selected_wavelengths:
        # Try exact match first
        if target_wl in X_train.columns:
            matched_columns.append(target_wl)
            continue

        # Try tolerance matching
        for col in X_train.columns:
            if isinstance(col, (int, float)) and abs(col - target_wl) < tolerance:
                matched_columns.append(col)
                break
        else:
            # No match found
            import warnings
            warnings.warn(f"Wavelength {target_wl} not found in training data")

    if not matched_columns:
        raise ValueError(f"No wavelengths matched from subset: {all_vars}")

    return X_train[matched_columns]
```

**Test Cases:**
```python
def test_extract_wavelength_subset_full():
    X = pd.DataFrame(np.random.rand(10, 100), columns=np.arange(400, 500))
    result = _extract_wavelength_subset(X, 'full', '')
    assert result.equals(X)

def test_extract_wavelength_subset_selection():
    X = pd.DataFrame(np.random.rand(10, 100), columns=np.arange(400, 500))
    all_vars = "402.0,410.0,450.0"
    result = _extract_wavelength_subset(X, 'subset_10', all_vars)
    assert result.shape == (10, 3)
    assert list(result.columns) == [402.0, 410.0, 450.0]

def test_extract_wavelength_subset_tolerance():
    X = pd.DataFrame(np.random.rand(10, 100), columns=np.arange(400.0, 500.0, 1.0))
    all_vars = "402.005,410.003"  # Slight floating-point differences
    result = _extract_wavelength_subset(X, 'subset_10', all_vars)
    assert result.shape[1] == 2
```

---

### Task 2.4: Main Reconstruction Function
**Duration:** 1 hour

**Code Location:** `ensemble_selection.py::reconstruct_and_fit_models()`

**Implementation:**
```python
def reconstruct_and_fit_models(
    results_df: pd.DataFrame,
    selected_indices: List[int],
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Tuple[List[Any], List[str], List[List]]:
    """
    Rebuild and refit models from results dataframe.

    Parameters
    ----------
    results_df : pd.DataFrame
        Full analysis results with all metadata
    selected_indices : List[int]
        Indices of models to reconstruct (from select_ensemble_candidates)
    X_train : pd.DataFrame
        Full training data with wavelength columns
    y_train : pd.Series
        Training targets

    Returns
    -------
    models : List[sklearn.pipeline.Pipeline]
        List of fitted model pipelines
    model_names : List[str]
        Descriptive names for each model (Model_Preprocess format)
    preprocessors : List[List[Tuple]]
        Preprocessing steps for each model (for metadata)

    Raises
    ------
    ValueError
        If model reconstruction or fitting fails

    Notes
    -----
    Each model is fitted on the full training data (not CV splits).
    This ensures ensemble training uses all available data.

    Models are returned as sklearn.pipeline.Pipeline objects containing:
    - Preprocessing steps (if any)
    - Fitted model as final step
    """
    from spectral_predict.models import get_model
    from sklearn.pipeline import Pipeline
    import logging

    logger = logging.getLogger(__name__)

    models = []
    model_names = []
    preprocessors = []

    logger.info(f"\nReconstructing {len(selected_indices)} models...")

    for i, idx in enumerate(selected_indices, 1):
        try:
            result = results_df.loc[idx]

            # 1. Parse hyperparameters
            params_str = result.get('Params', '{}')
            params = _parse_model_params(params_str)

            logger.info(f"  {i}. {result['Model']} ({result['Preprocess']})")
            logger.info(f"     Params: {params}")

            # 2. Create model
            model = get_model(
                result['Model'],
                task_type=result.get('Task', 'regression'),
                **params
            )

            # 3. Build preprocessing pipeline
            prep_steps = _rebuild_preprocessing(
                result.get('Preprocess', 'raw'),
                result.get('Deriv', None),
                result.get('Window', None),
                result.get('Poly', None)
            )

            # 4. Handle feature subset
            subset_tag = result.get('SubsetTag', 'full')
            all_vars = result.get('all_vars', '')

            X_subset = _extract_wavelength_subset(X_train, subset_tag, all_vars)

            logger.info(f"     Features: {X_subset.shape[1]} wavelengths")

            # 5. Create and fit pipeline
            pipeline_steps = prep_steps + [('model', model)]
            pipeline = Pipeline(pipeline_steps)

            pipeline.fit(X_subset, y_train)

            # 6. Store results
            models.append(pipeline)
            model_names.append(f"{result['Model']}_{result['Preprocess']}")
            preprocessors.append(prep_steps)

            logger.info(f"     ✓ Fitted successfully")

        except Exception as e:
            logger.error(f"     ✗ Failed to reconstruct model at index {idx}: {e}")
            raise ValueError(f"Model reconstruction failed at index {idx}: {e}")

    logger.info(f"\n✓ All {len(models)} models reconstructed and fitted")

    return models, model_names, preprocessors
```

**Test Cases:**
```python
def test_reconstruct_and_fit_models_basic():
    """Test basic reconstruction with 3 models."""
    # Create mock data
    X_train = create_mock_spectra(n_samples=100, n_wavelengths=2000)
    y_train = pd.Series(np.random.rand(100))

    # Create mock results
    results_df = create_mock_results_with_metadata(n=10)
    selected_indices = [0, 2, 5]

    models, model_names, preprocessors = reconstruct_and_fit_models(
        results_df, selected_indices, X_train, y_train
    )

    assert len(models) == 3
    assert len(model_names) == 3
    assert all(isinstance(m, Pipeline) for m in models)

    # Test prediction
    predictions = models[0].predict(X_train)
    assert len(predictions) == 100

def test_reconstruct_and_fit_models_with_subsets():
    """Test reconstruction with feature subsets."""
    X_train = create_mock_spectra(n_samples=100, n_wavelengths=2000)
    y_train = pd.Series(np.random.rand(100))

    results_df = create_mock_results_with_subsets(n=10)
    selected_indices = [0, 1]  # One full, one subset

    models, model_names, _ = reconstruct_and_fit_models(
        results_df, selected_indices, X_train, y_train
    )

    # Both should predict successfully on appropriate data
    assert len(models) == 2

def test_reconstruct_and_fit_models_error_handling():
    """Test error handling for invalid data."""
    X_train = create_mock_spectra(n_samples=100, n_wavelengths=2000)
    y_train = pd.Series(np.random.rand(100))

    # Invalid results (missing required columns)
    results_df = pd.DataFrame({'Model': ['PLS']})

    with pytest.raises(ValueError):
        reconstruct_and_fit_models(results_df, [0], X_train, y_train)
```

**Acceptance Criteria:**
- [ ] Reconstructs models from all metadata
- [ ] Fits models on full training data
- [ ] Handles preprocessing correctly (raw, snv, deriv, combinations)
- [ ] Handles feature subsets correctly
- [ ] Returns sklearn Pipeline objects
- [ ] All test cases pass
- [ ] Error messages are informative

---

### Task 2.5: Validation Tests
**Duration:** 30 minutes

**File:** `tests/test_model_reconstruction.py`

**Specification:**
Test that reconstructed models produce identical predictions to original models.

**Approach:**
1. Train a model using normal pipeline
2. Store its metadata in results_df format
3. Reconstruct model using `reconstruct_and_fit_models()`
4. Compare predictions on test set

```python
def test_reconstruction_equivalence_pls():
    """Test PLS reconstruction produces same predictions."""
    from spectral_predict.models import get_model
    from sklearn.pipeline import Pipeline
    from spectral_predict.preprocess import StandardNormalVariate

    # Create data
    X_train, y_train, X_test, y_test = create_train_test_split()

    # Train original model
    original_model = get_model('PLS', n_components=10)
    original_pipeline = Pipeline([
        ('snv', StandardNormalVariate()),
        ('model', original_model)
    ])
    original_pipeline.fit(X_train, y_train)
    original_pred = original_pipeline.predict(X_test)

    # Create results_df entry
    results_df = pd.DataFrame([{
        'Model': 'PLS',
        'Preprocess': 'snv',
        'Deriv': None,
        'Window': None,
        'Poly': None,
        'Params': "{'n_components': 10}",
        'SubsetTag': 'full',
        'all_vars': '',
        'Task': 'regression'
    }])

    # Reconstruct
    reconstructed_models, _, _ = reconstruct_and_fit_models(
        results_df, [0], X_train, y_train
    )
    reconstructed_pred = reconstructed_models[0].predict(X_test)

    # Compare predictions (should be nearly identical)
    np.testing.assert_allclose(original_pred, reconstructed_pred, rtol=1e-10)
```

**Acceptance Criteria:**
- [ ] Tests for PLS, Ridge, XGBoost, MLP
- [ ] Tests for raw, snv, deriv preprocessing
- [ ] Tests for feature subsets
- [ ] Predictions match within numerical tolerance
- [ ] All tests pass

---

## AGENT 3: GUI Integration Developer

### Task 3.1: Modify Tab 4 Analysis Integration
**Duration:** 2 hours

**File:** `spectral_predict_gui_optimized.py`
**Lines:** 3352-3394 (current ensemble section)

**Specification:**
Replace current placeholder code with full ensemble training pipeline.

**Implementation:**
```python
# === Run Ensemble Methods (if enabled) ===
if self.enable_ensembles.get():
    try:
        self._log_progress(f"\n{'='*70}")
        self._log_progress(f"ENSEMBLE METHODS")
        self._log_progress(f"{'='*70}")

        from spectral_predict.ensemble_selection import (
            select_ensemble_candidates,
            reconstruct_and_fit_models
        )
        from spectral_predict.ensemble import create_ensemble
        from sklearn.metrics import mean_squared_error, r2_score
        import numpy as np

        # Step 1: Smart model selection
        self._log_progress(f"\n=== MODEL SELECTION ===")
        self._log_progress(f"Analyzing top 30 models for regional diversity...")

        selected_df = select_ensemble_candidates(
            results_df,
            top_n=min(30, len(results_df)),  # Use top 30 or all if fewer
            ensemble_size=5,
            ensure_diversity=True
        )

        self._log_progress(f"\nSelected {len(selected_df)} models:")
        for i, row in enumerate(selected_df.itertuples(), 1):
            regional = row.regional_rmse
            self._log_progress(
                f"  {i}. {row.Model} ({row.Preprocess})\n"
                f"     Overall: RMSE={row.RMSE:.4f}, R²={row.R2:.4f}\n"
                f"     Regional: Q1={regional['Q1']:.4f}, Q2={regional['Q2']:.4f}, "
                f"Q3={regional['Q3']:.4f}, Q4={regional['Q4']:.4f}"
            )

        # Step 2: Reconstruct and fit models
        self._log_progress(f"\n=== MODEL RECONSTRUCTION ===")

        try:
            models, model_names, preprocessors = reconstruct_and_fit_models(
                results_df,
                selected_df.index.tolist(),
                X_filtered,  # Available in scope
                y_filtered   # Available in scope
            )
        except Exception as e:
            self._log_progress(f"⚠️ Model reconstruction failed: {e}")
            self._log_progress(f"   Skipping ensemble training")
            raise

        # Step 3: Collect enabled ensemble methods
        ensemble_methods = []
        if self.ensemble_simple_average.get():
            ensemble_methods.append(('simple_average', 'Simple Average'))
        if self.ensemble_region_weighted.get():
            ensemble_methods.append(('region_weighted', 'Region-Aware Weighted'))
        if self.ensemble_mixture_experts.get():
            ensemble_methods.append(('mixture_experts', 'Mixture of Experts'))
        if self.ensemble_stacking.get():
            ensemble_methods.append(('stacking', 'Stacking'))
        if self.ensemble_stacking_region.get():
            ensemble_methods.append(('region_stacking', 'Region-Aware Stacking'))

        if not ensemble_methods:
            self._log_progress(f"\n⚠️ No ensemble methods selected, skipping...")
        else:
            # Step 4: Train each ensemble method
            self._log_progress(f"\n=== ENSEMBLE TRAINING ===")
            self._log_progress(f"Training {len(ensemble_methods)} ensemble methods...")
            self._log_progress(f"Number of regions: {self.ensemble_n_regions.get()}")

            ensemble_results = []

            for ensemble_type, ensemble_name in ensemble_methods:
                self._log_progress(f"\n--- {ensemble_name} ---")

                try:
                    # Create and fit ensemble
                    ensemble = create_ensemble(
                        models=models,
                        model_names=model_names,
                        X=X_filtered,
                        y=y_filtered,
                        ensemble_type=ensemble_type,
                        n_regions=self.ensemble_n_regions.get(),
                        cv=self.cv_folds.get()
                    )

                    # Evaluate on training data (CV within ensemble)
                    # Note: Ensemble methods use internal CV to avoid overfitting
                    train_predictions = ensemble.predict(X_filtered)
                    train_rmse = np.sqrt(mean_squared_error(y_filtered, train_predictions))
                    train_r2 = r2_score(y_filtered, train_predictions)

                    self._log_progress(f"  Training: RMSE={train_rmse:.4f}, R²={train_r2:.4f}")

                    # Compute regional performance
                    if hasattr(ensemble, 'analyzer_') and ensemble.analyzer_ is not None:
                        regions = ensemble.analyzer_.assign_regions(y_filtered)
                        regional_rmse = {}
                        for q_idx in range(self.ensemble_n_regions.get()):
                            mask = regions == q_idx
                            if np.sum(mask) > 0:
                                q_rmse = np.sqrt(mean_squared_error(
                                    y_filtered[mask],
                                    train_predictions[mask]
                                ))
                                regional_rmse[f'Q{q_idx+1}'] = q_rmse
                                self._log_progress(f"  Region {q_idx+1}: RMSE={q_rmse:.4f}")
                    else:
                        regional_rmse = {}

                    # Store results
                    ensemble_results.append({
                        'ensemble_type': ensemble_type,
                        'ensemble_name': ensemble_name,
                        'ensemble': ensemble,
                        'train_rmse': train_rmse,
                        'train_r2': train_r2,
                        'regional_rmse': regional_rmse,
                        'base_models': model_names,
                        'n_regions': self.ensemble_n_regions.get()
                    })

                    self._log_progress(f"  ✓ {ensemble_name} trained successfully")

                except Exception as e:
                    self._log_progress(f"  ✗ {ensemble_name} failed: {e}")
                    import traceback
                    self._log_progress(f"  Traceback: {traceback.format_exc()}")

            # Step 5: Summary
            if ensemble_results:
                self._log_progress(f"\n=== ENSEMBLE SUMMARY ===")
                self._log_progress(f"\nTrained {len(ensemble_results)} ensemble methods:")

                # Sort by R²
                ensemble_results.sort(key=lambda x: x['train_r2'], reverse=True)

                for i, result in enumerate(ensemble_results, 1):
                    self._log_progress(
                        f"  {i}. {result['ensemble_name']}: "
                        f"RMSE={result['train_rmse']:.4f}, R²={result['train_r2']:.4f}"
                    )

                # Store best ensemble for potential saving
                self.ensemble_results = ensemble_results
                self.best_ensemble = ensemble_results[0]

                self._log_progress(f"\n✓ Best ensemble: {self.best_ensemble['ensemble_name']}")
                self._log_progress(f"  Use Tab 5 to save the best ensemble model")
            else:
                self._log_progress(f"\n⚠️ No ensembles trained successfully")

    except Exception as e:
        self._log_progress(f"\n⚠️ Ensemble execution failed: {e}")
        self._log_progress(f"   Individual model results are still available")
        import traceback
        self._log_progress(f"   Traceback: {traceback.format_exc()}")
```

**Acceptance Criteria:**
- [ ] Calls selection algorithm correctly
- [ ] Calls reconstruction correctly
- [ ] Trains all enabled ensemble types
- [ ] Logs progress with regional breakdown
- [ ] Handles errors gracefully (doesn't crash analysis)
- [ ] Stores ensemble results for Tab 5 access
- [ ] Works with existing analysis flow

---

### Task 3.2: Add Save Ensemble Button to Tab 5
**Duration:** 1 hour

**File:** `spectral_predict_gui_optimized.py`
**Location:** Tab 5 results display (around line 3654)

**Specification:**
Add button to save best ensemble after analysis completes.

**Implementation:**
```python
# In _create_tab5_results() method, add after results table:

# Ensemble save section (only show if ensembles were trained)
self.ensemble_save_frame = ttk.LabelFrame(
    content_frame,
    text="Save Best Ensemble Model",
    padding="20"
)
self.ensemble_save_frame.grid(
    row=row, column=0, columnspan=2,
    sticky=(tk.W, tk.E), pady=10
)
row += 1

ttk.Label(
    self.ensemble_save_frame,
    text="After analysis, save the best ensemble model for later use in Tab 7.",
    style='Caption.TLabel'
).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))

button_frame = ttk.Frame(self.ensemble_save_frame)
button_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W)

self.save_ensemble_button = ttk.Button(
    button_frame,
    text="💾 Save Best Ensemble",
    command=self._save_best_ensemble,
    state='disabled'  # Enabled after ensemble training
)
self.save_ensemble_button.pack(side='left', padx=5)

self.ensemble_info_label = ttk.Label(
    self.ensemble_save_frame,
    text="No ensemble available",
    style='Caption.TLabel'
)
self.ensemble_info_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

# Initially hide the frame
self.ensemble_save_frame.grid_remove()
```

**Add method:**
```python
def _save_best_ensemble(self):
    """Save best ensemble model to .dasp file."""
    if not hasattr(self, 'best_ensemble') or self.best_ensemble is None:
        messagebox.showwarning("No Ensemble", "No ensemble model available to save.")
        return

    # Get save location
    from tkinter import filedialog
    import datetime

    default_name = f"ensemble_{self.best_ensemble['ensemble_type']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.dasp"

    filepath = filedialog.asksaveasfilename(
        title="Save Ensemble Model",
        defaultextension=".dasp",
        filetypes=[("DASP Model Files", "*.dasp"), ("All Files", "*.*")],
        initialfile=default_name
    )

    if not filepath:
        return

    try:
        from spectral_predict.ensemble_io import save_ensemble

        # Prepare metadata
        metadata = {
            'model_name': f"Ensemble_{self.best_ensemble['ensemble_name']}",
            'ensemble_type': self.best_ensemble['ensemble_type'],
            'task_type': 'regression',
            'n_regions': self.best_ensemble['n_regions'],
            'base_model_names': self.best_ensemble['base_models'],
            'performance': {
                'RMSE': self.best_ensemble['train_rmse'],
                'R2': self.best_ensemble['train_r2'],
                'regional_rmse': self.best_ensemble['regional_rmse']
            },
            'wavelengths': list(self.X_filtered.columns),
            'n_vars': len(self.X_filtered.columns),
            'target_column': self.target_column.get(),
            'created': datetime.datetime.now().isoformat(),
        }

        # Save ensemble
        save_ensemble(
            ensemble=self.best_ensemble['ensemble'],
            base_models=self.best_ensemble.get('base_model_objects', []),
            metadata=metadata,
            filepath=filepath
        )

        messagebox.showinfo(
            "Success",
            f"Ensemble saved successfully:\n{filepath}"
        )

    except Exception as e:
        messagebox.showerror(
            "Error",
            f"Failed to save ensemble:\n{e}"
        )
        import traceback
        print(traceback.format_exc())

def _update_ensemble_save_ui(self):
    """Update UI after ensemble training completes."""
    if hasattr(self, 'best_ensemble') and self.best_ensemble is not None:
        # Show the frame
        self.ensemble_save_frame.grid()

        # Enable button
        self.save_ensemble_button.config(state='normal')

        # Update info label
        info_text = (
            f"Best: {self.best_ensemble['ensemble_name']} | "
            f"RMSE: {self.best_ensemble['train_rmse']:.4f} | "
            f"R²: {self.best_ensemble['train_r2']:.4f}"
        )
        self.ensemble_info_label.config(text=info_text)
```

**Call in Task 3.1 after ensemble training:**
```python
# At end of ensemble training section:
if ensemble_results:
    # ... existing code ...

    # Update UI
    self.root.after(0, self._update_ensemble_save_ui)
```

**Acceptance Criteria:**
- [ ] Button appears only after ensemble training
- [ ] Shows best ensemble info (type, RMSE, R²)
- [ ] Calls save_ensemble() correctly
- [ ] Handles errors with user-friendly messages
- [ ] Default filename includes ensemble type and timestamp

---

### Task 3.3: Testing
**Duration:** 1 hour

**File:** `tests/test_gui_ensemble_integration.py`

**Test Cases:**
- Ensemble training triggered when checkbox enabled
- Ensemble training skipped when checkbox disabled
- Error handling doesn't crash main analysis
- Save button enabled after training
- Save button disabled before training

**Acceptance Criteria:**
- [ ] Tests cover happy path and error cases
- [ ] Tests use mocking to avoid full analysis
- [ ] All tests pass

---

## AGENT 4: Persistence Architect

### Task 4.1: Define Ensemble .dasp Format
**Duration:** 30 minutes

**Deliverable:** Document specification

**Format Specification:**

```
ensemble_model.dasp (ZIP archive)
├── metadata.json                    # Ensemble configuration and performance
├── ensemble_state.npz              # NumPy arrays (regional weights, etc.)
├── meta_model.pkl                  # Meta-learner (for stacking only)
└── base_models/
    ├── base_model_0.dasp          # Embedded individual model
    ├── base_model_1.dasp
    ├── base_model_2.dasp
    ├── base_model_3.dasp
    └── base_model_4.dasp
```

**metadata.json structure:**
```json
{
  "model_name": "Ensemble_RegionAwareWeighted",
  "is_ensemble": true,
  "ensemble_type": "region_weighted",
  "task_type": "regression",
  "n_base_models": 5,
  "base_model_names": ["PLS_snv", "XGBoost_raw", "MLP_deriv", ...],
  "n_regions": 5,
  "cv_folds": 5,
  "performance": {
    "RMSE": 0.234,
    "R2": 0.912,
    "RMSE_std": 0.023,
    "R2_std": 0.015,
    "regional_rmse": {
      "Q1": 0.201,
      "Q2": 0.223,
      "Q3": 0.245,
      "Q4": 0.267
    }
  },
  "wavelengths": [400.0, 401.0, ...],
  "n_vars": 2151,
  "target_column": "nitrogen",
  "created": "2025-11-09T14:23:45",
  "dasp_version": "2.0",
  "ensemble_config": {
    "soft_gating": true,  // For MixtureOfExperts
    "region_aware": true,  // For Stacking
    "meta_model_type": "Ridge"  // For Stacking
  }
}
```

**ensemble_state.npz contents:**
- `regional_weights`: (n_models, n_regions) array - for RegionAwareWeighted
- `expert_assignment`: (n_regions,) array - for MixtureOfExperts
- `expert_weights`: (n_models, n_regions) array - for MixtureOfExperts
- `region_boundaries`: (n_regions+1,) array - region quantile boundaries
- `region_analyzer_config`: JSON string with analyzer configuration

**Acceptance Criteria:**
- [ ] Format documented in `docs/ENSEMBLE_FORMAT.md`
- [ ] Handles all 5 ensemble types
- [ ] Backward compatible (doesn't break individual model loading)
- [ ] Includes version info for future compatibility

---

### Task 4.2: Implement `save_ensemble()`
**Duration:** 2 hours

**File:** CREATE `src/spectral_predict/ensemble_io.py`

**Implementation:**
```python
"""
Save and load ensemble models in .dasp format.

Ensemble .dasp files are ZIP archives containing:
- Ensemble metadata and configuration
- Ensemble-specific state (regional weights, expert assignments)
- Base models as embedded .dasp files
- Meta-model (for stacking ensembles)
"""

import json
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Union, Optional
import numpy as np
import joblib
import warnings

from spectral_predict.model_io import save_model, load_model, _json_serializer


def save_ensemble(
    ensemble: Any,
    base_models: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    filepath: Union[str, Path]
) -> None:
    """
    Save ensemble model to .dasp file.

    Parameters
    ----------
    ensemble : RegionAwareWeightedEnsemble | MixtureOfExpertsEnsemble | StackingEnsemble
        Fitted ensemble object from spectral_predict.ensemble
    base_models : List[Dict[str, Any]]
        List of base model dictionaries, each containing:
        - 'model': fitted model object
        - 'preprocessor': preprocessing pipeline
        - 'metadata': model metadata (optional)
    metadata : Dict[str, Any]
        Ensemble metadata with keys:
        - model_name (str)
        - ensemble_type (str): 'region_weighted', 'mixture_experts', 'stacking', etc.
        - task_type (str)
        - n_regions (int)
        - base_model_names (List[str])
        - performance (dict)
        - wavelengths (list)
        - Any other relevant metadata
    filepath : str or Path
        Output path for .dasp file

    Raises
    ------
    ValueError
        If ensemble type is unknown or data is invalid
    IOError
        If file cannot be written

    Examples
    --------
    >>> ensemble = RegionAwareWeightedEnsemble(models, names, n_regions=5)
    >>> ensemble.fit(X_train, y_train)
    >>> save_ensemble(
    ...     ensemble=ensemble,
    ...     base_models=model_dicts,
    ...     metadata={'model_name': 'MyEnsemble', ...},
    ...     filepath='ensemble.dasp'
    ... )
    """
    filepath = Path(filepath)

    # Validate inputs
    if not hasattr(ensemble, '__class__'):
        raise ValueError("ensemble must be a fitted ensemble object")

    ensemble_class = ensemble.__class__.__name__

    # Add ensemble-specific metadata
    metadata['is_ensemble'] = True
    metadata['ensemble_class'] = ensemble_class
    metadata['dasp_version'] = '2.0'

    if 'n_base_models' not in metadata:
        metadata['n_base_models'] = len(base_models)

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # 1. Save metadata
        metadata_path = tmppath / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=_json_serializer)

        # 2. Save ensemble-specific state
        ensemble_state = _extract_ensemble_state(ensemble, ensemble_class)
        if ensemble_state:
            np.savez_compressed(
                tmppath / 'ensemble_state.npz',
                **ensemble_state
            )

        # 3. Save meta-model (for stacking)
        if ensemble_class == 'StackingEnsemble' and hasattr(ensemble, 'meta_model'):
            joblib.dump(
                ensemble.meta_model,
                tmppath / 'meta_model.pkl',
                compress=3
            )

        # 4. Save base models
        base_models_dir = tmppath / 'base_models'
        base_models_dir.mkdir()

        for i, model_dict in enumerate(base_models):
            base_model_path = base_models_dir / f'base_model_{i}.dasp'

            # Prepare base model metadata
            base_metadata = model_dict.get('metadata', {}).copy()
            if 'model_name' not in base_metadata and i < len(metadata.get('base_model_names', [])):
                base_metadata['model_name'] = metadata['base_model_names'][i]

            # Save base model using existing save_model function
            try:
                save_model(
                    model=model_dict['model'],
                    preprocessor=model_dict.get('preprocessor', None),
                    metadata=base_metadata,
                    filepath=base_model_path
                )
            except Exception as e:
                warnings.warn(f"Failed to save base model {i}: {e}")
                raise

        # 5. Create ZIP archive
        with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add all files from temp directory
            for file in tmppath.rglob('*'):
                if file.is_file():
                    arcname = file.relative_to(tmppath)
                    zf.write(file, arcname=arcname)

    print(f"Ensemble saved: {filepath}")


def _extract_ensemble_state(ensemble: Any, ensemble_class: str) -> Dict[str, np.ndarray]:
    """
    Extract ensemble-specific state for saving.

    Returns
    -------
    state : Dict[str, np.ndarray]
        Dictionary of numpy arrays to save in ensemble_state.npz
    """
    state = {}

    # Common: Region boundaries
    if hasattr(ensemble, 'analyzer_') and ensemble.analyzer_ is not None:
        if hasattr(ensemble.analyzer_, 'region_boundaries'):
            state['region_boundaries'] = ensemble.analyzer_.region_boundaries

        # Store analyzer config
        analyzer_config = {
            'n_regions': ensemble.analyzer_.n_regions,
            'method': ensemble.analyzer_.method
        }
        state['analyzer_config'] = np.array([json.dumps(analyzer_config)])

    # RegionAwareWeightedEnsemble
    if ensemble_class == 'RegionAwareWeightedEnsemble':
        if hasattr(ensemble, 'regional_weights_'):
            state['regional_weights'] = ensemble.regional_weights_

    # MixtureOfExpertsEnsemble
    elif ensemble_class == 'MixtureOfExpertsEnsemble':
        if hasattr(ensemble, 'expert_assignment_'):
            state['expert_assignment'] = ensemble.expert_assignment_
        if hasattr(ensemble, 'expert_weights_'):
            state['expert_weights'] = ensemble.expert_weights_

    # StackingEnsemble
    elif ensemble_class == 'StackingEnsemble':
        # Meta-model saved separately as .pkl
        # Region-aware info in analyzer_config
        pass

    return state
```

**Test Cases:**
```python
def test_save_ensemble_region_weighted():
    """Test saving RegionAwareWeightedEnsemble."""
    # Create mock ensemble
    ensemble, base_models, metadata = create_mock_ensemble('region_weighted')

    with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
        filepath = f.name

    try:
        save_ensemble(ensemble, base_models, metadata, filepath)

        # Verify file exists
        assert Path(filepath).exists()

        # Verify it's a valid ZIP
        assert zipfile.is_zipfile(filepath)

        # Verify contents
        with zipfile.ZipFile(filepath, 'r') as zf:
            assert 'metadata.json' in zf.namelist()
            assert 'ensemble_state.npz' in zf.namelist()
            assert any('base_model' in name for name in zf.namelist())
    finally:
        Path(filepath).unlink()

def test_save_ensemble_all_types():
    """Test saving all ensemble types."""
    for etype in ['region_weighted', 'mixture_experts', 'stacking', 'region_stacking']:
        ensemble, base_models, metadata = create_mock_ensemble(etype)

        with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
            filepath = f.name

        try:
            save_ensemble(ensemble, base_models, metadata, filepath)
            assert Path(filepath).exists()
        finally:
            Path(filepath).unlink()
```

**Acceptance Criteria:**
- [ ] Saves all ensemble types correctly
- [ ] Creates valid ZIP archive
- [ ] Embeds base models as .dasp files
- [ ] Stores regional weights/expert assignments
- [ ] Stores meta-model for stacking
- [ ] All tests pass
- [ ] Error messages are informative

---

### Task 4.3: Implement `load_ensemble()`
**Duration:** 2 hours

**File:** `src/spectral_predict/ensemble_io.py`

**Implementation:**
```python
def load_ensemble(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load ensemble model from .dasp file.

    Parameters
    ----------
    filepath : str or Path
        Path to ensemble .dasp file

    Returns
    -------
    ensemble_dict : Dict[str, Any]
        Dictionary containing:
        - 'ensemble': Reconstructed ensemble object
        - 'base_models': List of loaded base model dicts
        - 'metadata': Ensemble metadata
        - 'filepath': str
        - 'filename': str

    Raises
    ------
    ValueError
        If file is not a valid ensemble .dasp file
    IOError
        If file cannot be read

    Examples
    --------
    >>> ensemble_dict = load_ensemble('ensemble.dasp')
    >>> predictions = ensemble_dict['ensemble'].predict(X_new)
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise IOError(f"File not found: {filepath}")

    if not zipfile.is_zipfile(filepath):
        raise ValueError(f"Not a valid .dasp file: {filepath}")

    # Create temporary directory for extraction
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Extract ZIP
        with zipfile.ZipFile(filepath, 'r') as zf:
            zf.extractall(tmppath)

        # 1. Load metadata
        metadata_path = tmppath / 'metadata.json'
        if not metadata_path.exists():
            raise ValueError("Missing metadata.json in ensemble file")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Verify it's an ensemble
        if not metadata.get('is_ensemble', False):
            raise ValueError("Not an ensemble model file")

        ensemble_class = metadata.get('ensemble_class')
        ensemble_type = metadata.get('ensemble_type')

        # 2. Load base models
        base_models_dir = tmppath / 'base_models'
        if not base_models_dir.exists():
            raise ValueError("Missing base_models directory")

        base_models = []
        base_model_files = sorted(base_models_dir.glob('base_model_*.dasp'))

        for base_model_path in base_model_files:
            try:
                base_model_dict = load_model(base_model_path)
                base_models.append(base_model_dict)
            except Exception as e:
                warnings.warn(f"Failed to load base model {base_model_path.name}: {e}")
                raise

        # Extract model and preprocessor from base models
        models = [bm['model'] for bm in base_models]
        model_names = metadata.get('base_model_names', [f"Model_{i}" for i in range(len(models))])

        # 3. Load ensemble state
        ensemble_state_path = tmppath / 'ensemble_state.npz'
        if ensemble_state_path.exists():
            ensemble_state = np.load(ensemble_state_path)
        else:
            ensemble_state = None

        # 4. Load meta-model (for stacking)
        meta_model = None
        meta_model_path = tmppath / 'meta_model.pkl'
        if meta_model_path.exists():
            meta_model = joblib.load(meta_model_path)

        # 5. Reconstruct ensemble object
        ensemble = _reconstruct_ensemble(
            ensemble_class=ensemble_class,
            ensemble_type=ensemble_type,
            models=models,
            model_names=model_names,
            ensemble_state=ensemble_state,
            meta_model=meta_model,
            metadata=metadata
        )

        # 6. Return dict
        return {
            'ensemble': ensemble,
            'base_models': base_models,
            'metadata': metadata,
            'filepath': str(filepath),
            'filename': filepath.name
        }


def _reconstruct_ensemble(
    ensemble_class: str,
    ensemble_type: str,
    models: List[Any],
    model_names: List[str],
    ensemble_state: Optional[np.lib.npyio.NpzFile],
    meta_model: Optional[Any],
    metadata: Dict[str, Any]
) -> Any:
    """
    Reconstruct ensemble object from saved state.

    Returns
    -------
    ensemble : RegionAwareWeightedEnsemble | MixtureOfExpertsEnsemble | StackingEnsemble
        Reconstructed ensemble object ready for prediction
    """
    from spectral_predict.ensemble import (
        RegionAwareWeightedEnsemble,
        MixtureOfExpertsEnsemble,
        StackingEnsemble,
        RegionBasedAnalyzer
    )

    n_regions = metadata.get('n_regions', 5)

    # Parse analyzer config
    analyzer = None
    if ensemble_state and 'analyzer_config' in ensemble_state:
        analyzer_config_str = str(ensemble_state['analyzer_config'][0])
        analyzer_config = json.loads(analyzer_config_str)

        analyzer = RegionBasedAnalyzer(
            n_regions=analyzer_config.get('n_regions', n_regions),
            method=analyzer_config.get('method', 'quantile')
        )

        # Restore region boundaries
        if 'region_boundaries' in ensemble_state:
            analyzer.region_boundaries = ensemble_state['region_boundaries']

    # Reconstruct based on type
    if ensemble_class == 'RegionAwareWeightedEnsemble':
        ensemble = RegionAwareWeightedEnsemble(
            models=models,
            model_names=model_names,
            n_regions=n_regions,
            cv=metadata.get('cv_folds', 5)
        )

        # Restore state
        if ensemble_state and 'regional_weights' in ensemble_state:
            ensemble.regional_weights_ = ensemble_state['regional_weights']

        if analyzer:
            ensemble.analyzer_ = analyzer

    elif ensemble_class == 'MixtureOfExpertsEnsemble':
        soft_gating = metadata.get('ensemble_config', {}).get('soft_gating', True)

        ensemble = MixtureOfExpertsEnsemble(
            models=models,
            model_names=model_names,
            n_regions=n_regions,
            soft_gating=soft_gating
        )

        # Restore state
        if ensemble_state:
            if 'expert_assignment' in ensemble_state:
                ensemble.expert_assignment_ = ensemble_state['expert_assignment']
            if 'expert_weights' in ensemble_state:
                ensemble.expert_weights_ = ensemble_state['expert_weights']

        if analyzer:
            ensemble.analyzer_ = analyzer

    elif ensemble_class == 'StackingEnsemble':
        region_aware = metadata.get('ensemble_config', {}).get('region_aware', False)

        ensemble = StackingEnsemble(
            models=models,
            model_names=model_names,
            meta_model=meta_model,
            region_aware=region_aware,
            n_regions=n_regions,
            cv=metadata.get('cv_folds', 5)
        )

        if analyzer and region_aware:
            ensemble.analyzer_ = analyzer

    else:
        raise ValueError(f"Unknown ensemble class: {ensemble_class}")

    return ensemble
```

**Test Cases:**
```python
def test_load_ensemble_round_trip():
    """Test save then load produces equivalent ensemble."""
    # Create and save
    ensemble, base_models, metadata = create_mock_ensemble('region_weighted')

    with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
        filepath = f.name

    try:
        save_ensemble(ensemble, base_models, metadata, filepath)

        # Load
        loaded_dict = load_ensemble(filepath)

        # Verify structure
        assert 'ensemble' in loaded_dict
        assert 'base_models' in loaded_dict
        assert 'metadata' in loaded_dict

        # Verify metadata
        assert loaded_dict['metadata']['is_ensemble'] == True
        assert loaded_dict['metadata']['ensemble_type'] == 'region_weighted'

        # Verify base models loaded
        assert len(loaded_dict['base_models']) == len(base_models)

    finally:
        Path(filepath).unlink()

def test_load_ensemble_predictions():
    """Test loaded ensemble produces same predictions."""
    # Create ensemble and test data
    ensemble, base_models, metadata = create_fitted_ensemble('region_weighted')
    X_test = create_mock_spectra(n_samples=20, n_wavelengths=2000)

    # Original predictions
    original_pred = ensemble.predict(X_test)

    # Save and load
    with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
        filepath = f.name

    try:
        save_ensemble(ensemble, base_models, metadata, filepath)
        loaded_dict = load_ensemble(filepath)

        # Loaded predictions
        loaded_pred = loaded_dict['ensemble'].predict(X_test)

        # Compare
        np.testing.assert_allclose(original_pred, loaded_pred, rtol=1e-10)

    finally:
        Path(filepath).unlink()
```

**Acceptance Criteria:**
- [ ] Loads all ensemble types correctly
- [ ] Reconstructs ensemble objects with correct state
- [ ] Loaded ensembles produce correct predictions
- [ ] Round-trip test passes (save → load → predict matches)
- [ ] Error handling for corrupt files
- [ ] All tests pass

---

### Task 4.4: Extend `model_io.py` for Ensemble Detection
**Duration:** 30 minutes

**File:** `src/spectral_predict/model_io.py`
**Location:** Modify `load_model()` function (lines 149-225)

**Implementation:**
```python
def load_model(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load model from .dasp file.

    Automatically detects ensemble vs individual models.

    ... existing docstring ...
    """
    filepath = Path(filepath)

    # ... existing validation ...

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        with zipfile.ZipFile(filepath, 'r') as zf:
            zf.extractall(tmppath)

        # Load metadata
        metadata_path = tmppath / 'metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # CHECK: Is this an ensemble?
        if metadata.get('is_ensemble', False):
            # Redirect to ensemble loader
            from spectral_predict.ensemble_io import load_ensemble
            return load_ensemble(filepath)

        # ... existing individual model loading code ...
```

**Acceptance Criteria:**
- [ ] Detects ensemble files automatically
- [ ] Redirects to `load_ensemble()` when appropriate
- [ ] Doesn't break existing individual model loading
- [ ] Tests pass for both individual and ensemble models

---

### Task 4.5: Comprehensive Testing
**Duration:** 1 hour

**File:** `tests/test_ensemble_io.py`

**Test Coverage:**
- Save/load all 5 ensemble types
- Round-trip predictions match
- Handles missing base models gracefully
- Handles corrupt files
- Handles version mismatches
- Large ensembles (10+ base models)
- Base models with different preprocessing

**Acceptance Criteria:**
- [ ] >90% code coverage for ensemble_io.py
- [ ] All tests pass
- [ ] Tests use fixtures and mocking appropriately

---

## AGENT 5: Prediction Integration Specialist

### Task 5.1: Add Ensemble Detection to Tab 7 Loading
**Duration:** 1 hour

**File:** `spectral_predict_gui_optimized.py`
**Location:** `_load_model_for_prediction()` (lines 5456-5514)

**Modification:**
```python
def _load_model_for_prediction(self):
    """Load model(s) for prediction - supports individual and ensemble models."""
    files = filedialog.askopenfilenames(
        title="Select Model File(s)",
        filetypes=[("DASP Model Files", "*.dasp"), ("All Files", "*.*")],
        initialdir=str(Path.cwd())
    )

    if not files:
        return

    for filepath in files:
        try:
            from spectral_predict.model_io import load_model

            # load_model now auto-detects ensemble vs individual
            model_dict = load_model(filepath)

            # Add filepath info
            model_dict['filepath'] = filepath
            model_dict['filename'] = Path(filepath).name

            # Check if ensemble
            is_ensemble = model_dict.get('metadata', {}).get('is_ensemble', False)
            model_dict['is_ensemble'] = is_ensemble

            self.loaded_models.append(model_dict)

        except Exception as e:
            messagebox.showerror(
                "Error Loading Model",
                f"Failed to load {Path(filepath).name}:\n{e}"
            )
            continue

    # Update display
    self._update_loaded_models_display()

def _update_loaded_models_display(self):
    """Update the loaded models text widget."""
    self.loaded_models_text.config(state='normal')
    self.loaded_models_text.delete(1.0, tk.END)

    if not self.loaded_models:
        self.loaded_models_text.insert(tk.END, "No models loaded.")
    else:
        for i, model_dict in enumerate(self.loaded_models, 1):
            metadata = model_dict.get('metadata', {})

            # Check if ensemble
            if model_dict.get('is_ensemble', False):
                # Ensemble display
                ensemble_type = metadata.get('ensemble_type', 'Unknown')
                n_base = metadata.get('n_base_models', 0)
                n_regions = metadata.get('n_regions', 0)

                self.loaded_models_text.insert(
                    tk.END,
                    f"{i}. ENSEMBLE: {ensemble_type}\n"
                )
                self.loaded_models_text.insert(
                    tk.END,
                    f"   Base models: {n_base}, Regions: {n_regions}\n"
                )

                # Performance
                perf = metadata.get('performance', {})
                if 'RMSE' in perf:
                    self.loaded_models_text.insert(
                        tk.END,
                        f"   RMSE: {perf['RMSE']:.4f}, R²: {perf.get('R2', 'N/A')}\n"
                    )

                # Base model names
                base_names = metadata.get('base_model_names', [])
                if base_names:
                    self.loaded_models_text.insert(
                        tk.END,
                        f"   Models: {', '.join(base_names)}\n"
                    )

                self.loaded_models_text.insert(tk.END, "\n")

            else:
                # Individual model display (existing code)
                model_name = metadata.get('model_name', 'Unknown')
                preprocess = metadata.get('preprocessing', 'Unknown')

                self.loaded_models_text.insert(
                    tk.END,
                    f"{i}. {model_name} ({preprocess})\n"
                )

                # ... rest of existing display code ...

    self.loaded_models_text.config(state='disabled')
```

**Acceptance Criteria:**
- [ ] Detects ensemble models automatically
- [ ] Displays ensemble metadata (type, base models, regions)
- [ ] Displays individual models as before
- [ ] Mixed loading (ensemble + individual) works
- [ ] Error handling doesn't crash GUI

---

### Task 5.2: Add Ensemble Prediction Logic
**Duration:** 1 hour

**File:** `spectral_predict_gui_optimized.py`
**Location:** `_run_predictions()` (lines 5685-5786)

**Modification:**
```python
def _run_predictions(self):
    """Run predictions using loaded models."""

    # ... existing validation code ...

    results_data = {'Sample_ID': sample_ids}
    successful_models = []

    for i, model_dict in enumerate(self.loaded_models):
        try:
            metadata = model_dict.get('metadata', {})

            # Determine model name for column
            if model_dict.get('is_ensemble', False):
                # Ensemble model
                ensemble_type = metadata.get('ensemble_type', 'ensemble')
                model_name = f"Ensemble_{ensemble_type}"

                # Use ensemble predict
                ensemble_obj = model_dict.get('ensemble')
                if ensemble_obj is None:
                    self._log_prediction_progress(f"⚠️ Skipping {model_name}: No ensemble object")
                    continue

                self._log_prediction_progress(f"Predicting with {model_name}...")

                # Get predictions
                predictions = ensemble_obj.predict(prediction_data)

            else:
                # Individual model (existing code)
                model_name = f"{metadata.get('model_name', 'Model')}_{metadata.get('preprocessing', 'raw')}"

                self._log_prediction_progress(f"Predicting with {model_name}...")

                from spectral_predict.model_io import predict_with_model
                predictions = predict_with_model(
                    model_dict,
                    prediction_data,
                    validate_wavelengths=True
                )

            # Handle duplicate column names
            original_name = model_name
            counter = 1
            while model_name in results_data:
                model_name = f"{original_name}_{counter}"
                counter += 1

            results_data[model_name] = predictions
            successful_models.append(model_name)

            self._log_prediction_progress(f"  ✓ {model_name} complete")

        except Exception as e:
            self._log_prediction_progress(f"  ✗ Failed: {e}")
            import traceback
            self._log_prediction_progress(traceback.format_exc())
            continue

    # ... rest of existing code (consensus, export) ...
```

**Acceptance Criteria:**
- [ ] Ensemble predictions work correctly
- [ ] Individual model predictions still work
- [ ] Mixed predictions (ensemble + individual) work
- [ ] Column naming handles duplicates
- [ ] Error handling is robust

---

### Task 5.3: Update Consensus Logic for Ensembles
**Duration:** 30 minutes

**File:** `spectral_predict_gui_optimized.py`
**Location:** `_add_consensus_predictions()` (lines 5788-5915)

**Note:**
Current consensus logic computes post-hoc weighted averages. With trained ensembles, this section can be simplified - just include ensemble predictions in the DataFrame without additional post-processing.

**Modification:**
```python
def _add_consensus_predictions(self, results_df, model_columns):
    """
    Add consensus predictions from multiple models.

    Note: If an ensemble model was used, it already provides optimal combination.
    Consensus methods are additional options for combining individual models.
    """

    # Filter to only individual model columns (exclude ensembles)
    individual_columns = []
    for col in model_columns:
        if not col.startswith('Ensemble_'):
            individual_columns.append(col)

    if len(individual_columns) < 2:
        self._log_prediction_progress(
            "\nConsensus requires 2+ individual models. Skipping consensus."
        )
        return results_df

    # ... existing consensus code using individual_columns ...

    # Note in log that ensemble predictions are already optimal
    if len(model_columns) > len(individual_columns):
        self._log_prediction_progress(
            "\nNote: Ensemble predictions already represent optimal model combinations."
        )
```

**Acceptance Criteria:**
- [ ] Consensus skips ensemble columns (already optimal)
- [ ] Consensus works with individual models
- [ ] Log messages clarify ensemble vs consensus

---

### Task 5.4: Testing
**Duration:** 1 hour

**File:** `tests/test_tab7_ensemble.py`

**Test Cases:**
- Load ensemble model
- Load mixed (ensemble + individual)
- Predict with ensemble
- Predict with mixed models
- Ensemble metadata display
- Error handling (corrupt ensemble file)

**Acceptance Criteria:**
- [ ] All test cases pass
- [ ] Tests use fixtures from Agent 4
- [ ] Integration test: save in Tab 4 → load in Tab 7 → predict

---

## AGENT 6 (Optional): Visualization & Documentation

### Task 6.1: Regional Performance Heatmap
**Duration:** 1 hour

**File:** `src/spectral_predict/ensemble_viz.py` (already exists)

**Add function:**
```python
def plot_ensemble_selection_heatmap(
    selected_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> Tuple[Figure, np.ndarray]:
    """
    Plot heatmap showing why each model was selected.

    Rows: Selected models
    Columns: Q1, Q2, Q3, Q4, Overall, Architecture, Preprocessing

    Color indicates performance (green = good, red = poor)
    """
    pass
```

---

### Task 6.2: User Documentation
**Duration:** 1 hour

**File:** CREATE `docs/ENSEMBLE_USAGE_GUIDE.md`

**Contents:**
- What are ensemble methods?
- When to use ensembles
- How to enable in GUI
- How to interpret results
- How to save/load ensembles
- Best practices
- Troubleshooting

---

### Task 6.3: Example Scripts
**Duration:** 30 minutes

**File:** CREATE `examples/ensemble_example.py`

**Show:**
- Loading results from analysis
- Smart model selection
- Training ensemble
- Saving ensemble
- Loading and predicting

---

## Integration & Testing Strategy

### Integration Milestones

**Milestone 1: Foundation (End of Day 1)**
- [ ] Agent 1: Selection algorithm complete
- [ ] Agent 2: Reconstruction complete
- [ ] Agent 4: Save/load format defined
- [ ] **Integration Test:** Select + Reconstruct pipeline works

**Milestone 2: Persistence (End of Day 2)**
- [ ] Agent 4: Save/load functions complete
- [ ] **Integration Test:** Round-trip (save → load → predict) works

**Milestone 3: GUI Integration (Middle of Day 3)**
- [ ] Agent 3: Tab 4 integration complete
- [ ] **Integration Test:** Full analysis → ensemble saved

**Milestone 4: Prediction (End of Day 3)**
- [ ] Agent 5: Tab 7 integration complete
- [ ] **Integration Test:** Load ensemble → predict on new data

**Milestone 5: Polish (Optional)**
- [ ] Agent 6: Visualization and docs
- [ ] **Final Test:** End-to-end user workflow

---

### Testing Pyramid

```
                    /\
                   /  \
                  / E2E \          1-2 tests
                 /______\
                /        \
               / Integration\      5-10 tests
              /____________\
             /              \
            /  Unit Tests    \    30-50 tests
           /__________________\
```

**Unit Tests (30-50 tests):**
- Each agent responsible for their module
- >90% code coverage
- Fast execution (<1 second each)

**Integration Tests (5-10 tests):**
- Lead agent coordinates
- Test inter-module communication
- Moderate execution time (<10 seconds each)

**End-to-End Tests (1-2 tests):**
- Lead agent creates
- Full workflow: GUI → analysis → save → load → predict
- Slower execution (1-2 minutes)

---

### Daily Coordination

**Daily Standup (15 minutes):**
1. What did you complete yesterday?
2. What are you working on today?
3. Any blockers or dependencies?

**Daily Integration (30 minutes):**
- Lead agent merges completed work
- Run integration tests
- Address any conflicts

**End-of-Day Review (15 minutes):**
- Demo completed features
- Update milestone checklist
- Plan next day's priorities

---

## Dependencies & Coordination Points

### Critical Dependencies

```
Agent 1 (Selection) ──────┬──────> Agent 3 (GUI Integration)
                          │
Agent 2 (Reconstruction) ─┘

Agent 4 (Persistence) ────────────> Agent 5 (Tab 7 Integration)
                         │
                         └────────> Agent 3 (Save button)
```

**Coordination Points:**

1. **Agents 1 & 2 → Agent 3:**
   - **Interface:** Function signatures for `select_ensemble_candidates()` and `reconstruct_and_fit_models()`
   - **Timing:** Agents 1 & 2 complete by hour 5, then Agent 3 starts integration
   - **Communication:** Share example usage code

2. **Agent 4 → Agent 3:**
   - **Interface:** `save_ensemble()` signature and metadata format
   - **Timing:** Agent 4 defines interface by hour 2, Agent 3 can prep code
   - **Communication:** Share metadata schema early

3. **Agent 4 → Agent 5:**
   - **Interface:** `load_ensemble()` return format
   - **Timing:** Agent 4 completes by hour 5, Agent 5 starts integration
   - **Communication:** Share example loaded ensemble dict

---

## Acceptance Criteria

### Phase 1: Model Selection (Agent 1)
- [ ] Selects models with regional diversity
- [ ] Ensures architecture diversity
- [ ] Ensures preprocessing diversity
- [ ] Logs selection rationale
- [ ] Handles edge cases (small datasets)
- [ ] Unit tests >90% coverage
- [ ] All tests pass

### Phase 2: Model Reconstruction (Agent 2)
- [ ] Reconstructs models from results_df
- [ ] Handles all model types (PLS, Ridge, XGBoost, etc.)
- [ ] Handles all preprocessing types
- [ ] Handles feature subsets
- [ ] Fitted models produce correct predictions
- [ ] Validation tests pass (predictions match)
- [ ] Unit tests >90% coverage

### Phase 3: GUI Integration (Agent 3)
- [ ] Ensemble training triggered when enabled
- [ ] All 5 ensemble types supported
- [ ] Progress logged with regional metrics
- [ ] Errors don't crash analysis
- [ ] Save button enabled after training
- [ ] Best ensemble identified and stored
- [ ] Integration tests pass

### Phase 4: Persistence (Agent 4)
- [ ] save_ensemble() works for all types
- [ ] load_ensemble() reconstructs correctly
- [ ] Round-trip test passes
- [ ] Format documented
- [ ] Backward compatible with individual models
- [ ] Unit tests >90% coverage
- [ ] All tests pass

### Phase 5: Prediction Integration (Agent 5)
- [ ] Ensemble models load in Tab 7
- [ ] Ensemble predictions work correctly
- [ ] Mixed loading (ensemble + individual) works
- [ ] Metadata displayed correctly
- [ ] Consensus logic updated
- [ ] Integration tests pass

### Phase 6: Visualization (Agent 6 - Optional)
- [ ] Regional heatmaps working
- [ ] Documentation complete
- [ ] Examples tested and working

### Overall Acceptance
- [ ] All unit tests pass (>90% coverage)
- [ ] All integration tests pass
- [ ] End-to-end test passes
- [ ] No regressions in existing functionality
- [ ] Code reviewed by Lead
- [ ] Documentation updated

---

## Risk Mitigation

### Risk 1: Integration Complexity
**Risk:** Agents' code doesn't integrate smoothly
**Mitigation:**
- Define clear interfaces early (Day 1 morning)
- Daily integration checkpoints
- Lead agent reviews all interfaces before implementation

### Risk 2: Ensemble State Serialization
**Risk:** Complex ensemble state (weights, assignments) doesn't serialize correctly
**Mitigation:**
- Agent 4 creates format spec early (Hour 1)
- Agent 4 creates round-trip test immediately (Hour 2)
- Use NumPy .npz format (proven, reliable)

### Risk 3: Model Reconstruction Accuracy
**Risk:** Reconstructed models don't match original predictions
**Mitigation:**
- Agent 2 creates validation test early (Task 2.5)
- Compare predictions on test set with numerical tolerance
- Use existing factories (get_model, build_preprocessing_pipeline)

### Risk 4: GUI Crashes
**Risk:** Ensemble training crashes the analysis
**Mitigation:**
- Wrap all ensemble code in try/except
- Log errors but continue analysis
- Test with intentionally broken data

### Risk 5: Large File Sizes
**Risk:** Ensemble .dasp files too large
**Mitigation:**
- Use compression (joblib compress=3, ZIP_DEFLATED)
- Monitor file sizes during testing
- Document expected sizes

### Risk 6: Dependency Hell
**Risk:** Agents blocked waiting for others
**Mitigation:**
- Clear dependency graph (see Dependencies section)
- Agents 1, 2, 4 work in parallel (no dependencies)
- Agent 3 preps code structure while waiting
- Agent 5 creates UI mockups while waiting

---

## Final Checklist

### Lead Agent Final Review
- [ ] All agent code merged to main branch
- [ ] All tests pass (unit + integration + E2E)
- [ ] Code review complete for all modules
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No regression in existing tests
- [ ] Performance acceptable (ensemble training <2 min)
- [ ] File sizes reasonable (< 100 MB per ensemble)
- [ ] Example scripts tested
- [ ] User-facing error messages clear
- [ ] Ready for merge to main

---

## Appendix: Code Templates

### Template: Unit Test File
```python
"""
Tests for ensemble_selection module.
"""

import pytest
import numpy as np
import pandas as pd
from spectral_predict.ensemble_selection import (
    select_ensemble_candidates,
    reconstruct_and_fit_models,
    _extract_regional_performance,
    _parse_model_params,
    _rebuild_preprocessing,
    _extract_wavelength_subset,
    _classify_architecture
)


@pytest.fixture
def mock_results_df():
    """Create mock results dataframe for testing."""
    # ... implementation ...
    pass


@pytest.fixture
def mock_training_data():
    """Create mock X_train, y_train."""
    # ... implementation ...
    pass


class TestEnsembleSelection:
    """Tests for model selection functions."""

    def test_select_basic(self, mock_results_df):
        """Test basic selection."""
        pass

    def test_select_regional_diversity(self, mock_results_df):
        """Test regional diversity."""
        pass

    # ... more tests ...


class TestModelReconstruction:
    """Tests for model reconstruction functions."""

    def test_parse_params_valid(self):
        """Test parameter parsing."""
        pass

    # ... more tests ...
```

### Template: Integration Test
```python
"""
Integration tests for ensemble pipeline.
"""

import pytest
import tempfile
from pathlib import Path
from spectral_predict.ensemble_selection import (
    select_ensemble_candidates,
    reconstruct_and_fit_models
)
from spectral_predict.ensemble_io import save_ensemble, load_ensemble
from spectral_predict.ensemble import create_ensemble


def test_full_ensemble_pipeline():
    """
    Test full pipeline: select → reconstruct → train → save → load → predict.
    """
    # 1. Create mock data
    results_df = create_mock_results(n=50)
    X_train, y_train = create_mock_training_data()
    X_test = create_mock_test_data()

    # 2. Select models
    selected_df = select_ensemble_candidates(results_df, top_n=30, ensemble_size=5)
    assert len(selected_df) == 5

    # 3. Reconstruct models
    models, model_names, _ = reconstruct_and_fit_models(
        results_df, selected_df.index.tolist(), X_train, y_train
    )
    assert len(models) == 5

    # 4. Train ensemble
    ensemble = create_ensemble(
        models=models,
        model_names=model_names,
        X=X_train,
        y=y_train,
        ensemble_type='region_weighted',
        n_regions=5
    )

    # 5. Original predictions
    original_pred = ensemble.predict(X_test)

    # 6. Save
    with tempfile.NamedTemporaryFile(suffix='.dasp', delete=False) as f:
        filepath = f.name

    try:
        base_models = [{'model': m, 'preprocessor': None} for m in models]
        metadata = {
            'model_name': 'TestEnsemble',
            'ensemble_type': 'region_weighted',
            'task_type': 'regression',
            'n_regions': 5,
            'base_model_names': model_names,
            'performance': {},
            'wavelengths': list(X_train.columns),
            'n_vars': len(X_train.columns)
        }

        save_ensemble(ensemble, base_models, metadata, filepath)

        # 7. Load
        loaded_dict = load_ensemble(filepath)

        # 8. Loaded predictions
        loaded_pred = loaded_dict['ensemble'].predict(X_test)

        # 9. Compare
        np.testing.assert_allclose(original_pred, loaded_pred, rtol=1e-10)

    finally:
        Path(filepath).unlink()
```

---

## Contact & Support

**Lead Agent:**
- Responsible for: Overall coordination, integration, final review
- Communication: Daily standups, integration reviews

**Agents 1-6:**
- Daily status updates
- Raise blockers immediately
- Request code reviews when tasks complete

**Resources:**
- Project repo: `C:\Users\sponheim\git\dasp`
- Documentation: `docs/`
- Tests: `tests/`
- Slack/Discord: #ensemble-implementation (if applicable)

---

**End of Implementation Plan**

*This plan provides a comprehensive roadmap for implementing intelligent ensemble methods with clear role definitions, detailed specifications, and robust testing strategy. Estimated completion: 16-19 hours (2-3 focused days).*
