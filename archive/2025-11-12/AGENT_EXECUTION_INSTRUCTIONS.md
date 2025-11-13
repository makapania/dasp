# AGENT EXECUTION INSTRUCTIONS

**Date**: 2025-11-12
**For**: New agent team implementing complete hyperparameter system
**Reference**: `HYPERPARAMETER_COMPLETE_IMPLEMENTATION_PLAN.md` (comprehensive specs)

---

## QUICK START

You are part of a 10-agent team implementing 39 missing hyperparameters across 12 models. This document contains:

1. **Your mission** - What needs to be done
2. **Agent assignments** - Which agent you are
3. **Execution order** - Dependencies and waves
4. **Ready-to-launch prompts** - Copy-paste instructions for each agent
5. **Validation checklist** - How to verify success

---

## MISSION OVERVIEW

**Goal**: Implement ALL 39 hyperparameters with single-value defaults (no grid explosion)

**Critical Context**:
- Previous "HYPERPARAMETER_IMPLEMENTATION_HANDOFF.md" was STALE - ignore it
- Use "HYPERPARAMETER_COMPLETE_IMPLEMENTATION_PLAN.md" as definitive reference
- Forensic analysis proved: PLS/Ridge/Lasso/ElasticNet were NEVER implemented
- RandomForest/LightGBM were implemented, then REMOVED
- User's vision: "Expose all with single-value defaults" = safe, no breaking changes

**Files to Modify**:
- `src/spectral_predict/models.py` - Grid generation (main work)
- `src/spectral_predict/search.py` - Parameter passing
- `src/spectral_predict/model_config.py` - Tier defaults
- `spectral_predict_gui_optimized.py` - GUI controls & extraction

---

## EXECUTION WAVES

### Wave 1: Backend Foundation (PARALLEL - 2.5 hours)
- Agent A1: Linear Models Backend
- Agent A2: Tree Models Backend
- Agent A3: Boosting Models Backend

**Dependencies**: None - START IMMEDIATELY

### Wave 2: Neural/SVM Backend (SEQUENTIAL - 1.5 hours)
- Agent B1: Neural & SVM Backend

**Dependencies**: Wait for Wave 1 complete (avoid merge conflicts)

### Wave 3: Search Integration (SEQUENTIAL - 1 hour)
- Agent C1: Search Integration

**Dependencies**: Wait for Waves 1 & 2 complete (needs all parameter names)

### Wave 4: Tab 4C GUI (PARALLEL - 2 hours)
- Agent D1: Tab 4C Linear & Tree GUI
- Agent D2: Tab 4C Boosting GUI
- Agent D3: Tab 4C Neural/SVM GUI

**Dependencies**: Wait for Wave 3 complete (backend must exist)

### Wave 5: Tab 4C Integration (SEQUENTIAL - 2.5 hours)
- Agent E1: Tab 4C Integration & Conditional Logic

**Dependencies**: Wait for Wave 4 complete (GUI controls must exist)

### Wave 6: Tab 7C Subtabs (PARALLEL - 3.5 hours)
- Agent F1: Tab 7C Architecture + First 6 Models
- Agent F2: Tab 7C Last 6 Models

**Dependencies**: Wait for Wave 3 complete (backend must exist)
**Note**: Can overlap with Waves 4 & 5

### Wave 7: Tab 7C Integration (SEQUENTIAL - 2 hours)
- Agent G1: Tab 7C Integration

**Dependencies**: Wait for Wave 6 complete (subtabs must exist)

### Wave 8: Testing (SEQUENTIAL - 3 hours)
- Agent H1: Testing & Validation

**Dependencies**: Wait for Waves 5 & 7 complete (all functionality must exist)

---

## AGENT PROMPTS

### Agent A1: Linear Models Backend Specialist

```
You are Agent A1: Linear Models Backend Specialist in a 10-agent hyperparameter implementation team.

CONTEXT:
Read HYPERPARAMETER_COMPLETE_IMPLEMENTATION_PLAN.md sections 1-4.1 for complete context. Key points:
- PLS, Ridge, Lasso, ElasticNet were NEVER implemented (not removed, never existed)
- You're implementing 11 new parameters across 4 models
- Single-value defaults maintain current grid sizes (no breaking changes)
- Complete forensic analysis in handoff doc proves this is safe

YOUR MISSION:
Implement complete hyperparameter support for 4 linear models in the backend.

MODELS & PARAMETERS:
1. PLS: Add 3 parameters (max_iter, tol, algorithm)
2. Ridge: Add 2 parameters (solver, tol)
3. Lasso: Add 3 parameters (selection, tol, max_iter)
4. ElasticNet: Add 3 parameters (selection, tol, max_iter)

FILES TO MODIFY:
1. src/spectral_predict/models.py
   - Function signature (line 223): Add 11 parameters
   - Default loading (after line 295): Add 11 get() calls
   - Grid generation (lines 397-439): Replace with nested loops
2. src/spectral_predict/model_config.py
   - Add parameters to all 3 tiers (standard/comprehensive/quick)
   - Use single-value defaults for standard/quick
   - Use 2-3 values for comprehensive

DETAILED SPECS:
See HYPERPARAMETER_COMPLETE_IMPLEMENTATION_PLAN.md section 4.1 for:
- Exact code to add/replace (complete nested loops provided)
- Line numbers for each modification
- Parameter types, defaults, help text
- model_config.py tier values

CRITICAL REQUIREMENTS:
- Use nested loops (not list comprehensions) for clarity
- Pass ALL parameters to model constructors
- Pass ALL parameters in params dict
- Single-value lists for standard/quick tiers: [500], [1e-6], ['nipals']
- Validate syntax after changes: python -m py_compile src/spectral_predict/models.py

DELIVERABLES:
1. Updated models.py with 4 models fully implemented
2. Updated model_config.py with all tier defaults
3. Syntax validation passed
4. Confirmation that grid sizes unchanged with defaults

REFERENCE CODE:
Section 8.1 of handoff doc has COMPLETE PLS implementation as example.

START: Begin immediately (no dependencies)
```

---

### Agent A2: Tree Models Backend Specialist

```
You are Agent A2: Tree Models Backend Specialist in a 10-agent hyperparameter implementation team.

CONTEXT:
Read HYPERPARAMETER_COMPLETE_IMPLEMENTATION_PLAN.md sections 1-4.1. Key points:
- RandomForest WAS implemented in commit cb1ead6, then REMOVED in d801c91
- You're RESTORING 6 parameters that were deliberately removed
- Hard-coded values were restored - you're removing them again
- Single-value defaults maintain current grid sizes

YOUR MISSION:
Restore complete RandomForest hyperparameter support (regression & classification).

MODEL & PARAMETERS:
RandomForest: Restore 6 parameters
- min_samples_split (default: [2])
- min_samples_leaf (default: [1])
- max_features (default: ['sqrt'])
- bootstrap (default: [True])
- max_leaf_nodes (default: [None])
- min_impurity_decrease (default: [0.0])

FILES TO MODIFY:
1. src/spectral_predict/models.py
   - Function signature (line 223): Add 6 parameters after rf_max_depth_list
   - Default loading (after line 302): Add 6 get() calls
   - Grid generation regression (lines 604-646): Replace 2D loop with 8D nested loop
   - Grid generation classification (lines 967-1003): Same 8D structure
2. src/spectral_predict/model_config.py
   - RandomForest section: Add 6 parameters to all 3 tiers

DETAILED SPECS:
See section 4.1.3 "RandomForest Regression" for COMPLETE replacement code.
- Current: 2D loop (n_estimators × max_depth)
- Target: 8D loop (adds 6 more nested loops)
- ALL 8 parameters passed to RandomForestRegressor constructor
- ALL 8 parameters in params dict

CRITICAL REQUIREMENTS:
- Update BOTH regression (line 604) and classification (line 967) sections
- Nested loops, not list comprehensions
- Single-value defaults for standard/quick tiers
- Syntax validation: python -m py_compile src/spectral_predict/models.py

DELIVERABLES:
1. RandomForest fully implemented with 8D grid
2. model_config.py updated with 6 new parameters
3. Syntax validation passed
4. Grid size proof: 2 n_est × 3 depth × 1×1×1×1×1×1 = 6 configs (UNCHANGED)

REFERENCE:
Section 4.1.3 has complete RandomForest grid generation code.

START: Begin immediately (no dependencies)
```

---

### Agent A3: Boosting Models Backend Specialist

```
You are Agent A3: Boosting Models Backend Specialist in a 10-agent hyperparameter implementation team.

CONTEXT:
Read HYPERPARAMETER_COMPLETE_IMPLEMENTATION_PLAN.md sections 1-4.1. Key points:
- LightGBM WAS implemented, then REMOVED - hard-coded values RESTORED
- XGBoost, CatBoost need additional parameters
- You're implementing 12 parameters across 3 models

YOUR MISSION:
Complete hyperparameter support for 3 boosting models.

MODELS & PARAMETERS:
1. LightGBM: Restore 6 parameters, REMOVE hard-coded values
   - max_depth, min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda
   - Currently hard-coded at lines 652-660: min_child_samples=5, subsample=0.8, etc.
   - Replace with parameters from lists
2. XGBoost: Add 2 parameters
   - min_child_weight (default: [1])
   - gamma (default: [0])
3. CatBoost: Add 4 parameters
   - l2_leaf_reg, border_count, bagging_temperature, random_strength

FILES TO MODIFY:
1. src/spectral_predict/models.py
   - Function signature: Add 12 parameters
   - Default loading: Add 12 get() calls
   - LightGBM (lines 647-680): REMOVE hard-coded values, 9D nested loop
   - XGBoost: Add 2 nested loops to existing structure
   - CatBoost: Add 4 nested loops to existing structure
2. src/spectral_predict/model_config.py
   - Add parameters to all 3 models, all 3 tiers

DETAILED SPECS:
Section 4.1.3 has COMPLETE LightGBM replacement code showing:
- 9D nested loop (3 original + 6 new)
- NO hard-coded values
- All parameters from lists

CRITICAL REQUIREMENTS:
- LightGBM: Remove hard-coded values at lines 652-660
- All parameters passed to constructors (LGBMRegressor, XGBRegressor, CatBoostRegressor)
- Single-value defaults for standard/quick tiers
- Update both regression and classification sections
- Syntax validation: python -m py_compile src/spectral_predict/models.py

DELIVERABLES:
1. LightGBM with 9D grid, NO hard-coded values
2. XGBoost with additional 2 parameters
3. CatBoost with additional 4 parameters
4. model_config.py updated for all 3 models
5. Syntax validation passed

REFERENCE:
Section 4.1.3 "LightGBM Regression" has complete replacement code.

START: Begin immediately (no dependencies)
```

---

### Agent B1: Neural & SVM Backend Specialist

```
You are Agent B1: Neural & SVM Backend Specialist in a 10-agent hyperparameter implementation team.

CONTEXT:
Read HYPERPARAMETER_COMPLETE_IMPLEMENTATION_PLAN.md sections 1-4.1.

YOUR MISSION:
Complete hyperparameter support for 3 models with conditional logic.

MODELS & PARAMETERS:
1. MLP: Add 5 parameters
   - activation, solver, batch_size, learning_rate_schedule, momentum
   - CONDITIONAL: momentum only when solver='sgd'
2. SVR: Add 4 parameters
   - epsilon, degree, coef0, shrinking
   - CONDITIONAL: degree only for kernel='poly', coef0 for 'poly'/'sigmoid'
3. NeuralBoosted: Add 1 parameter
   - subsample

FILES TO MODIFY:
1. src/spectral_predict/models.py
   - Function signature: Add 10 parameters
   - Default loading: Add 10 get() calls
   - MLP: Add nested loops with conditional momentum
   - SVR: Add nested loops with kernel-specific conditionals
   - NeuralBoosted: Add subsample nested loop
2. src/spectral_predict/neural_boosted.py (if needed)
   - Subsample may already be implemented, verify
3. src/spectral_predict/model_config.py
   - Add parameters to all 3 models

CONDITIONAL LOGIC:
MLP momentum:
```python
for solver in mlp_solver_list:
    if solver == 'sgd':
        for momentum in mlp_momentum_list:
            # Include momentum parameter
    else:
        # Exclude momentum parameter
```

SVR kernel-specific:
```python
for kernel in svr_kernels:
    if kernel == 'rbf':
        # Use gamma, not degree/coef0
    elif kernel == 'poly':
        # Use degree and coef0
    elif kernel == 'sigmoid':
        # Use coef0, not degree
    else:  # linear
        # No extra params
```

DETAILED SPECS:
See section 4.1.2 for default loading patterns.
See section 4.1.3 for grid generation (update existing sections).

DELIVERABLES:
1. MLP with 5 new parameters + conditional momentum
2. SVR with 4 new parameters + kernel conditionals
3. NeuralBoosted with subsample parameter
4. model_config.py updated
5. Syntax validation passed

DEPENDENCIES: Wait for Wave 1 (A1, A2, A3) to complete to avoid merge conflicts

START: After Wave 1 complete
```

---

### Agent C1: Search Integration Specialist

```
You are Agent C1: Search Integration Specialist in a 10-agent hyperparameter implementation team.

CONTEXT:
Read HYPERPARAMETER_COMPLETE_IMPLEMENTATION_PLAN.md section 4.2.

YOUR MISSION:
Update search.py to accept and pass all 39 new parameters.

SCOPE:
- 39 parameters total (11 linear + 6 RF + 12 boosting + 10 neural/SVM)
- 2 modifications: function signature + function call

FILES TO MODIFY:
1. src/spectral_predict/search.py
   - run_search() function signature (line 22): Add 39 parameters
   - get_model_grids() call (line 183): Pass all 39 parameters

DETAILED SPECS:
Section 4.2 has exact locations and parameter order.

Example additions:
```python
# After rf_max_depth_list=None:
rf_min_samples_split_list=None, rf_min_samples_leaf_list=None,
rf_max_features_list=None, rf_bootstrap_list=None,
rf_max_leaf_nodes_list=None, rf_min_impurity_decrease_list=None,

# New PLS parameters:
pls_max_iter_list=None, pls_tol_list=None, pls_algorithm_list=None,

# After ridge_alphas_list=None:
ridge_solver_list=None, ridge_tol_list=None,

# Continue for all 39 parameters...
```

CRITICAL REQUIREMENTS:
- Parameter ORDER must match models.py signature exactly
- Both function signature AND function call must be updated
- All parameters use =None default
- Syntax validation: python -m py_compile src/spectral_predict/search.py

DELIVERABLES:
1. run_search() signature updated with all 39 parameters
2. get_model_grids() call updated to pass all 39 parameters
3. Syntax validation passed

REFERENCE:
Section 8.1 shows example additions for PLS parameters.

DEPENDENCIES: Wait for Waves 1 & 2 (all backend agents) to get final parameter names

START: After Waves 1 & 2 complete
```

---

### Agent D1: Tab 4C Linear & Tree GUI Specialist

```
You are Agent D1: Tab 4C Linear & Tree GUI Specialist in a 10-agent hyperparameter implementation team.

CONTEXT:
Read HYPERPARAMETER_COMPLETE_IMPLEMENTATION_PLAN.md section 4.4.

YOUR MISSION:
Create GUI controls for 17 parameters across 5 models in Tab 4C.

MODELS & PARAMETERS:
1. PLS: 3 controls (max_iter, tol, algorithm)
2. Ridge: 2 controls (solver, tol)
3. Lasso: 3 controls (selection, tol, max_iter)
4. ElasticNet: 3 controls (selection, tol, max_iter)
5. RandomForest: 6 controls (min_samples_split, min_samples_leaf, max_features, bootstrap, max_leaf_nodes, min_impurity_decrease)

FILE TO MODIFY:
spectral_predict_gui_optimized.py (Tab 4C section, around lines 2200-2960)

PATTERN FOR EACH CONTROL:
```python
self.pls_max_iter_control = self._create_parameter_grid_control(
    parent=pls_section_frame,
    param_name='max_iter',
    param_label='Max Iterations',
    checkbox_values=[500, 1000, 2000, 5000],
    default_checked=[500],
    is_float=False,
    help_text='Maximum iterations for NIPALS algorithm'
)
```

DETAILED SPECS:
Section 4.4.1 has parameter specifications.
Section 8.1 has complete example for PLS controls.

LAYOUT:
- Create collapsible LabelFrame for each model
- Use existing patterns from Tab 4C
- Store each control as instance variable: self.{model}_{param}_control

CRITICAL REQUIREMENTS:
- Use _create_parameter_grid_control() helper (already exists)
- Checkbox values from handoff doc section 4.3 (model_config.py specs)
- Single value default checked (maintains grid size)
- is_float=True for numeric decimals, False for integers
- allow_string_values=True for strings like 'nipals', 'cyclic'

DELIVERABLES:
1. GUI controls for 17 parameters
2. Collapsible sections for each model
3. All controls stored as instance variables
4. Proper layout in Tab 4C

DEPENDENCIES: Wait for Wave 3 (backend must exist)

START: After Wave 3 complete
```

---

### Agent D2: Tab 4C Boosting GUI Specialist

```
You are Agent D2: Tab 4C Boosting GUI Specialist in a 10-agent hyperparameter implementation team.

CONTEXT:
Read HYPERPARAMETER_COMPLETE_IMPLEMENTATION_PLAN.md section 4.4.

YOUR MISSION:
Create GUI controls for 12 parameters across 3 boosting models in Tab 4C.

MODELS & PARAMETERS:
1. XGBoost: 2 controls (min_child_weight, gamma)
2. LightGBM: 6 controls (max_depth, min_child_samples, subsample, colsample_bytree, reg_alpha, reg_lambda)
3. CatBoost: 4 controls (l2_leaf_reg, border_count, bagging_temperature, random_strength)

FILE TO MODIFY:
spectral_predict_gui_optimized.py (Tab 4C section, around lines 2200-2960)

PATTERN:
Same as Agent D1 - use _create_parameter_grid_control() for each parameter.

DETAILED SPECS:
Section 4.4.1 for patterns.
Section 4.3 for parameter values and defaults.

Example:
```python
self.lgbm_max_depth_control = self._create_parameter_grid_control(
    parent=lgbm_section_frame,
    param_name='max_depth',
    param_label='Max Depth',
    checkbox_values=[-1, 5, 10, 20, 50],
    default_checked=[-1],
    is_float=False,
    help_text='-1 means no limit'
)
```

LAYOUT:
- Collapsible LabelFrame for each model
- Follow existing Tab 4C patterns

DELIVERABLES:
1. GUI controls for 12 parameters
2. Collapsible sections for 3 models
3. All controls stored as instance variables

DEPENDENCIES: Wait for Wave 3

START: After Wave 3 complete (parallel with D1, D3)
```

---

### Agent D3: Tab 4C Neural/SVM GUI Specialist

```
You are Agent D3: Tab 4C Neural/SVM GUI Specialist in a 10-agent hyperparameter implementation team.

CONTEXT:
Read HYPERPARAMETER_COMPLETE_IMPLEMENTATION_PLAN.md section 4.4.

YOUR MISSION:
Create GUI controls for 10 parameters across 3 models in Tab 4C.

MODELS & PARAMETERS:
1. MLP: 5 controls (activation, solver, batch_size, learning_rate_schedule, momentum)
2. SVR: 4 controls (epsilon, degree, coef0, shrinking)
3. NeuralBoosted: 1 control (subsample)

FILE TO MODIFY:
spectral_predict_gui_optimized.py (Tab 4C section, around lines 2200-2960)

PATTERN:
Same as D1/D2 - use _create_parameter_grid_control().

SPECIAL NOTES:
- MLP batch_size: Mix of string ('auto') and integers (32, 64, etc.) - use allow_string_values=True
- MLP momentum: Will need conditional logic in Agent E1 (just create control for now)
- SVR degree, coef0: Will need conditional logic in Agent E1

DELIVERABLES:
1. GUI controls for 10 parameters
2. Collapsible sections for 3 models
3. Instance variables stored

DEPENDENCIES: Wait for Wave 3

START: After Wave 3 complete (parallel with D1, D2)
```

---

### Agent E1: Tab 4C Integration & Conditional Logic Specialist

```
You are Agent E1: Tab 4C Integration & Conditional Logic Specialist in a 10-agent hyperparameter implementation team.

CONTEXT:
Read HYPERPARAMETER_COMPLETE_IMPLEMENTATION_PLAN.md sections 4.4.2, 4.4.3, 4.4.4.

YOUR MISSION:
1. Extract all 39 parameters in _run_analysis()
2. Update run_search() call with all parameters
3. Implement conditional logic for MLP momentum and SVR kernel-specific params

SCOPE:
- 39 parameter extractions
- 1 run_search() call update
- 2 conditional logic callbacks

FILE TO MODIFY:
spectral_predict_gui_optimized.py

LOCATIONS:
1. _run_analysis_thread() method (~line 6723): Add parameter extraction
2. run_search() call (~line 7344): Add all parameters
3. Conditional logic: Create callback methods, bind to controls

PARAMETER EXTRACTION PATTERN:
```python
# For each of 39 parameters:
pls_max_iter_list = self._extract_parameter_values(
    self.pls_max_iter_control, 'max_iter', is_float=False
) if hasattr(self, 'pls_max_iter_control') else None
```

run_search() CALL PATTERN:
```python
results_df = run_search(
    X, y, task_type,
    # ... existing params ...
    pls_max_iter_list=pls_max_iter_list,
    pls_tol_list=pls_tol_list,
    pls_algorithm_list=pls_algorithm_list,
    # ... continue for all 39 params ...
    tier=tier
)
```

CONDITIONAL LOGIC:

MLP momentum:
```python
def _on_mlp_solver_change(self):
    """Enable momentum controls only when SGD solver is selected"""
    sgd_checked = self.mlp_solver_sgd.get() if hasattr(self, 'mlp_solver_sgd') else False
    # Enable/disable momentum checkboxes based on sgd_checked
    # Update widget states

# Bind to solver checkboxes
self.mlp_solver_sgd.trace('w', lambda *args: self._on_mlp_solver_change())
```

SVR kernel-specific:
```python
def _on_svr_kernel_change(self):
    """Enable/disable SVR parameters based on kernel selection"""
    # Determine which kernels are checked
    # degree: Enable only for poly
    # coef0: Enable for poly OR sigmoid
    # gamma: Disable for linear only
    # Update widget states
```

DETAILED SPECS:
Section 4.4.2: Parameter extraction patterns
Section 4.4.3: run_search() call update
Section 4.4.4: Conditional logic implementations

DELIVERABLES:
1. All 39 parameters extracted in _run_analysis()
2. run_search() call updated with all parameters
3. MLP momentum conditional logic working
4. SVR kernel conditional logic working
5. Complete Tab 4C → backend parameter flow

DEPENDENCIES: Wait for Wave 4 (D1, D2, D3) - GUI controls must exist

START: After Wave 4 complete
```

---

### Agent F1: Tab 7C Architecture + First 6 Models

```
You are Agent F1: Tab 7C Architecture + First 6 Models in a 10-agent hyperparameter implementation team.

CONTEXT:
Read HYPERPARAMETER_COMPLETE_IMPLEMENTATION_PLAN.md sections 4.5.1, 4.5.2, 4.5.3.

YOUR MISSION:
1. Create Tab 7C notebook architecture
2. Implement auto-navigation
3. Create 6 model subtabs with all controls

ARCHITECTURE:
Replace existing Tab 7C Frame with ttk.Notebook containing 12 subtabs.

FILE TO MODIFY:
spectral_predict_gui_optimized.py (Tab 7 setup, around line 8800)

TASKS:

1. CREATE NOTEBOOK (section 4.5.1):
```python
# Replace Tab 7C Frame with Notebook
self.tab_7c = ttk.Frame(tab_7_notebook)
tab_7_notebook.add(self.tab_7c, text="C. Configuration")

self.tab_7c_model_notebook = ttk.Notebook(self.tab_7c)
self.tab_7c_model_notebook.pack(fill='both', expand=True, padx=10, pady=10)

# Create 12 subtab frames
self.tab_7c_pls = ttk.Frame(self.tab_7c_model_notebook)
self.tab_7c_model_notebook.add(self.tab_7c_pls, text="PLS")
# ... create all 12
```

2. IMPLEMENT AUTO-NAVIGATION (section 4.5.2):
```python
def _on_tab7_model_change(self, *args):
    """Switch Tab 7C subtab based on selected model"""
    model_name = self.model_type_var.get()
    model_to_tab_index = {
        'PLS': 0, 'Ridge': 1, 'Lasso': 2, 'ElasticNet': 3,
        'RandomForest': 4, 'XGBoost': 5, 'LightGBM': 6, 'CatBoost': 7,
        'SVR': 8, 'MLP': 9, 'NeuralBoosted': 10, 'PLS-DA': 11
    }
    if model_name in model_to_tab_index:
        self.tab_7c_model_notebook.select(model_to_tab_index[model_name])

# Bind to model dropdown
self.model_type_var.trace('w', self._on_tab7_model_change)
```

3. CREATE 6 SUBTABS WITH CONTROLS:
   - PLS: 4 parameters (n_components, max_iter, tol, algorithm)
   - Ridge: 3 parameters (alpha, solver, tol)
   - Lasso: 4 parameters (alpha, selection, tol, max_iter)
   - ElasticNet: 5 parameters (alpha, l1_ratio, selection, tol, max_iter)
   - RandomForest: 8 parameters (all)
   - XGBoost: 9 parameters (all)

PATTERN FOR EACH SUBTAB (section 4.5.3):
```python
# Create scrollable frame
pls_frame = ttk.Frame(self.tab_7c_pls)
pls_frame.pack(fill='both', expand=True, padx=10, pady=10)

# Create controls using _create_parameter_grid_control()
self.tab7c_pls_n_components_control = self._create_parameter_grid_control(
    parent=pls_frame,
    param_name='n_components',
    param_label='Number of Components',
    checkbox_values=[2, 4, 6, 8, 10, 12, 16, 20],
    default_checked=[10],
    is_float=False
)
# ... repeat for all parameters in model
```

CONTROL NAMING:
- Use prefix: tab7c_{model}_{param}_control
- Example: self.tab7c_pls_max_iter_control

DELIVERABLES:
1. Notebook architecture created
2. Auto-navigation working
3. 6 subtabs with all parameter controls
4. All controls stored as instance variables

DEPENDENCIES: Wait for Wave 3 (backend must exist)

START: After Wave 3 complete (can overlap with Waves 4 & 5)
```

---

### Agent F2: Tab 7C Last 6 Models

```
You are Agent F2: Tab 7C Last 6 Models in a 10-agent hyperparameter implementation team.

CONTEXT:
Read HYPERPARAMETER_COMPLETE_IMPLEMENTATION_PLAN.md section 4.5.3.

YOUR MISSION:
Create 6 model subtabs with all controls in Tab 7C.

MODELS & PARAMETERS:
7. LightGBM: 9 parameters
8. CatBoost: 7 parameters
9. SVR: 7 parameters
10. MLP: 8 parameters
11. NeuralBoosted: 5 parameters
12. PLS-DA: 8 parameters (PLS + LogisticRegression)

FILE TO MODIFY:
spectral_predict_gui_optimized.py (Tab 7C subtabs)

PATTERN:
Same as Agent F1 - create scrollable frame, add controls using _create_parameter_grid_control().

CONTROL NAMING:
- Prefix: tab7c_{model}_{param}_control
- Example: self.tab7c_mlp_activation_control

SPECIAL NOTES:
- MLP: Include all 8 parameters (conditional logic added by Agent E1 earlier)
- SVR: Include all 7 parameters (conditional logic added by Agent E1 earlier)
- PLS-DA: 3 PLS params + 5 LogisticRegression params (use lr_ prefix)

DELIVERABLES:
1. 6 subtabs with all parameter controls
2. All controls stored as instance variables

DEPENDENCIES: Wait for Wave 3

START: After Wave 3 complete (parallel with F1)
```

---

### Agent G1: Tab 7C Integration Specialist

```
You are Agent G1: Tab 7C Integration Specialist in a 10-agent hyperparameter implementation team.

CONTEXT:
Read HYPERPARAMETER_COMPLETE_IMPLEMENTATION_PLAN.md sections 4.5.4, 4.5.5.

YOUR MISSION:
1. Create parameter extraction helpers for all 12 models
2. Update _run_refined_model() to extract and apply params
3. Update Results → Tab 7C loading to populate controls

SCOPE:
- 12 extraction helper functions
- 1 _run_refined_model() update
- 12 population helper functions
- 1 _load_model_for_refinement() update

FILE TO MODIFY:
spectral_predict_gui_optimized.py

TASK 1: EXTRACTION HELPERS (section 4.5.4)

Create for each model:
```python
def _extract_tab7c_pls_params(self):
    """Extract PLS parameters from Tab 7C PLS subtab"""
    params = {}

    if hasattr(self, 'tab7c_pls_n_components_control'):
        vals = self._extract_parameter_values(
            self.tab7c_pls_n_components_control, 'n_components', is_float=False
        )
        if vals:
            params['n_components'] = vals[0]  # Use first value for single model

    # ... repeat for all parameters
    return params
```

Create 12 functions: _extract_tab7c_pls_params(), _extract_tab7c_ridge_params(), etc.

TASK 2: UPDATE _run_refined_model() (section 4.5.4)

```python
# Extract params based on model type
model_name = self.model_type_var.get()
params_to_apply = {}

if model_name == 'PLS':
    params_to_apply = self._extract_tab7c_pls_params()
elif model_name == 'Ridge':
    params_to_apply = self._extract_tab7c_ridge_params()
# ... etc for all 12 models

# Apply params to model
if params_to_apply:
    try:
        model.set_params(**params_to_apply)
        print(f"DEBUG: Applied Tab 7C parameters: {params_to_apply}")
    except Exception as e:
        print(f"WARNING: Failed to apply parameters: {e}")
```

TASK 3: POPULATION HELPERS (section 4.5.5)

Create for each model:
```python
def _populate_tab7c_pls_controls(self, params):
    """Populate PLS subtab controls from loaded parameters"""
    if 'n_components' in params and hasattr(self, 'tab7c_pls_n_components_control'):
        # Check appropriate checkbox or populate custom entry
        pass
    # ... repeat for all parameters
```

TASK 4: UPDATE _load_model_for_refinement() (section 4.5.5)

```python
def _load_model_for_refinement(self, config):
    # ... existing code ...

    # Parse Params column
    raw_params = config.get('Params', {})
    if isinstance(raw_params, str):
        try:
            params_dict = ast.literal_eval(raw_params)
        except:
            params_dict = {}

    # Auto-navigate to correct subtab
    self._on_tab7_model_change()

    # Populate controls
    model_name = config.get('Model')
    if model_name == 'PLS':
        self._populate_tab7c_pls_controls(params_dict)
    elif model_name == 'Ridge':
        self._populate_tab7c_ridge_controls(params_dict)
    # ... etc
```

DELIVERABLES:
1. 12 parameter extraction helpers
2. _run_refined_model() extracts and applies params
3. 12 population helpers
4. _load_model_for_refinement() populates Tab 7C
5. Complete Tab 7C → execution flow
6. Complete Results → Tab 7C loading flow

DEPENDENCIES: Wait for Wave 6 (F1, F2) - subtabs must exist

START: After Wave 6 complete
```

---

### Agent H1: Testing & Validation Specialist

```
You are Agent H1: Testing & Validation Specialist in a 10-agent hyperparameter implementation team.

CONTEXT:
Read HYPERPARAMETER_COMPLETE_IMPLEMENTATION_PLAN.md section 6.

YOUR MISSION:
Create comprehensive test suite and validate entire implementation.

TEST CATEGORIES:
1. Grid size validation (section 6.1)
2. Parameter flow integration (section 6.2)
3. Conditional logic (section 6.3)
4. Backward compatibility (section 6.4)

FILES TO CREATE:
- tests/test_grid_sizes.py
- tests/test_parameter_flow.py
- tests/test_conditional_logic.py
- tests/test_backward_compatibility.py
- tests/test_integration_full_workflow.py

TEST 1: GRID SIZE VALIDATION

For each of 12 models:
```python
def test_pls_grid_size_unchanged():
    """Verify PLS grid size unchanged with new params"""
    from src.spectral_predict.model_config import get_hyperparameters

    config = get_hyperparameters('PLS', 'standard')

    # Verify single-value defaults
    assert len(config['max_iter']) == 1
    assert len(config['tol']) == 1
    assert len(config['algorithm']) == 1

    # Verify grid size calculation
    # PLS: 12 components × 1 × 1 × 1 = 12 (unchanged)
    n_components = len(config['n_components'])
    grid_size = n_components * len(config['max_iter']) * len(config['tol']) * len(config['algorithm'])
    assert grid_size == n_components  # Grid size unchanged
```

TEST 2: PARAMETER FLOW

```python
def test_pls_parameter_flow():
    """Test PLS parameters flow through entire system"""
    from src.spectral_predict.search import run_search
    import numpy as np

    # Create mock data
    X = np.random.rand(100, 50)
    y = np.random.rand(100)

    # Call with custom params
    results = run_search(
        X, y, 'regression',
        models_to_test=['PLS'],
        pls_max_iter_list=[1000],
        pls_tol_list=[1e-5],
        pls_algorithm_list=['svd'],
        tier='standard'
    )

    # Verify params in results
    import ast
    params = ast.literal_eval(results.iloc[0]['Params'])
    assert params['max_iter'] == 1000
    assert params['tol'] == 1e-5
    assert params['algorithm'] == 'svd'
```

TEST 3: CONDITIONAL LOGIC

```python
def test_mlp_momentum_conditional():
    """Test momentum only applied when solver=sgd"""
    from src.spectral_predict.models import get_model_grids

    # Test 1: solver='adam', momentum should be ignored
    grids = get_model_grids(
        'regression', 100,
        mlp_solver_list=['adam'],
        mlp_momentum_list=[0.9],
        tier='standard',
        enabled_models=['MLP']
    )
    # Verify momentum NOT in params

    # Test 2: solver='sgd', momentum should be applied
    grids = get_model_grids(
        'regression', 100,
        mlp_solver_list=['sgd'],
        mlp_momentum_list=[0.9],
        tier='standard',
        enabled_models=['MLP']
    )
    # Verify momentum IS in params
```

TEST 4: BACKWARD COMPATIBILITY

```python
def test_load_old_results():
    """Test loading results from before param implementation"""
    old_result = {
        'Model': 'PLS',
        'Params': "{'n_components': 10}",  # No new params
        'RMSE': 0.5,
        # ... other fields
    }

    # Should load without errors, use defaults for missing params
    # Test in actual GUI context if possible
```

VALIDATION:

Run all syntax checks:
```bash
python -m py_compile src/spectral_predict/models.py
python -m py_compile src/spectral_predict/search.py
python -m py_compile src/spectral_predict/model_config.py
python -m py_compile spectral_predict_gui_optimized.py
```

Run all tests:
```bash
pytest tests/test_grid_sizes.py -v
pytest tests/test_parameter_flow.py -v
pytest tests/test_conditional_logic.py -v
pytest tests/test_backward_compatibility.py -v
pytest tests/test_integration_full_workflow.py -v
```

DELIVERABLES:
1. Complete test suite (5 test files)
2. All tests passing
3. Syntax validation passed for all files
4. Validation report documenting results
5. Any issues discovered with recommendations

DEPENDENCIES: Wait for Waves 5 & 7 (all functionality must be implemented)

START: After Waves 5 & 7 complete
```

---

## VALIDATION CHECKLIST

After all agents complete, verify:

### Backend
- [ ] All 39 parameters in models.py function signature
- [ ] All 39 parameters have default loading
- [ ] All 12 models have updated grid generation
- [ ] LightGBM hard-coded values removed
- [ ] search.py updated with all 39 parameters
- [ ] model_config.py has all 39 parameters in all tiers
- [ ] Single-value defaults for standard/quick tiers
- [ ] Syntax validation passes for all backend files

### GUI Tab 4C
- [ ] GUI controls for all 39 parameters
- [ ] Collapsible sections for each model
- [ ] Parameter extraction in _run_analysis() for all 39 params
- [ ] run_search() call passes all 39 parameters
- [ ] MLP momentum conditional logic working
- [ ] SVR kernel conditional logic working

### GUI Tab 7C
- [ ] ttk.Notebook with 12 subtabs
- [ ] Auto-navigation working
- [ ] All parameters have controls in respective subtabs
- [ ] _run_refined_model() extracts and applies params
- [ ] Results → Tab 7C loading populates controls

### Testing
- [ ] Grid size tests pass (sizes unchanged with defaults)
- [ ] Parameter flow tests pass
- [ ] Conditional logic tests pass
- [ ] Backward compatibility tests pass
- [ ] All syntax validations pass

---

## TROUBLESHOOTING

**If grid sizes explode**: Check model_config.py - ensure single-value defaults

**If parameters missing from results**: Trace flow: GUI → _run_analysis() → run_search() → get_model_grids()

**If syntax errors**: Run `python -m py_compile` on modified files

**If merge conflicts**: Use git stash/merge carefully, prefer sequential waves

**If conditional logic not working**: Add debug prints, verify callback bindings

---

## LAUNCH COMMAND

To start execution:

```bash
# Wave 1 - Launch 3 agents in parallel
# Copy Agent A1 prompt → Launch agent
# Copy Agent A2 prompt → Launch agent
# Copy Agent A3 prompt → Launch agent

# Wait for Wave 1 complete, then launch Wave 2
# Copy Agent B1 prompt → Launch agent

# Continue through all 8 waves...
```

---

**GOOD LUCK! You have everything you need to complete this implementation successfully.**
