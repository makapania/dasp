# Part VII: Ensemble Methods and Advanced Features

This section covers the advanced features of Spectral Predict V1, including intelligent ensemble methods, model development workflows, calibration transfer between instruments, interference removal techniques, spectral library management, and contaminant analysis.

---

## Chapter 1: Ensemble Methods

Ensemble methods combine multiple models to achieve better predictive performance than any single model alone. Spectral Predict implements several sophisticated ensemble strategies that go beyond simple averaging, including region-aware weighting and specialist model selection.

### 1.1 Overview of Ensemble Strategies

Spectral Predict provides five primary ensemble types:

| Ensemble Type | Description | Best For |
|---------------|-------------|----------|
| SimpleAverageEnsemble | Equal-weight averaging | Quick baseline |
| RegionAwareWeightedEnsemble | Quartile-based weighting | Heterogeneous target ranges |
| MixtureOfExpertsEnsemble | Best model per region | Strong specialists |
| StackingEnsemble | Ridge meta-learner | Complex model combinations |
| RegionSpecialistEnsemble | Per-quartile specialists | Quartile-optimized prediction |

### 1.2 SimpleAverageEnsemble

The SimpleAverageEnsemble provides the most straightforward ensemble approach: equal-weight averaging of predictions from multiple base models.

**Mathematical Formulation:**

$$\hat{y} = \frac{1}{K}\sum_{k=1}^{K} \hat{y}_k$$

Where:
- $\hat{y}$ is the final ensemble prediction
- $K$ is the number of base models
- $\hat{y}_k$ is the prediction from the $k$-th model

**When to Use:**
- Quick baseline ensemble for comparison
- When base models have similar overall performance
- When model diversity is already high

**Implementation Details:**

```python
class SimpleAverageEnsemble:
    def predict(self, X):
        predictions = []
        for i, model in enumerate(self.models):
            preprocessor = self._get_preprocessor(i)
            if preprocessor is not None:
                X_processed = preprocessor.transform(X)
            else:
                X_processed = X
            pred = model.predict(X_processed)
            predictions.append(pred.ravel())
        return np.mean(predictions, axis=0)
```

**Key Features:**
- Supports per-model preprocessing via `preprocessor_configs`
- Handles both fitted preprocessors and PreprocessorConfig objects
- Automatically flattens predictions from models like PLSRegression

### 1.3 RegionAwareWeightedEnsemble

The RegionAwareWeightedEnsemble assigns different weights to models based on their performance in different regions of the target space. This approach recognizes that some models may excel at predicting low concentrations while others perform better at high concentrations.

**Architecture Diagram:**

```
                    Input Sample
                         |
                         v
              +-------------------+
              |  Initial Routing  |
              |  (Avg Prediction) |
              +-------------------+
                         |
            +------------+------------+
            |            |            |
            v            v            v
       Region Q1    Region Q2    Region Q3/Q4
            |            |            |
            v            v            v
    +-------------+ +-------------+ +-------------+
    | Weights_Q1  | | Weights_Q2  | | Weights_Q3  |
    | [0.4, 0.3,  | | [0.2, 0.5,  | | [0.1, 0.2,  |
    |  0.2, 0.1]  | |  0.2, 0.1]  | |  0.3, 0.4]  |
    +-------------+ +-------------+ +-------------+
            |            |            |
            v            v            v
         Weighted     Weighted     Weighted
         Average      Average      Average
```

**Weight Computation Algorithm:**

1. **Generate Out-of-Fold (OOF) Predictions:**
   - Use K-Fold cross-validation to generate predictions for all samples
   - This prevents data leakage in weight computation

2. **Define Region Boundaries:**
   - Compute quartile boundaries from TRUE Y values:
   $$\text{boundaries} = [Q_0, Q_{25}, Q_{50}, Q_{75}, Q_{100}]$$

3. **Compute Regional Performance:**
   - For each model $k$ and region $r$, compute RMSE:
   $$\text{RMSE}_{k,r} = \sqrt{\frac{1}{n_r}\sum_{i \in r}(y_i - \hat{y}_{k,i})^2}$$

4. **Convert to Weights (Inverse Error):**
   $$w_{k,r} = \frac{1}{\text{RMSE}_{k,r} + \epsilon}$$

5. **Normalize Per Region:**
   $$w_{k,r}^{norm} = \frac{w_{k,r}}{\sum_{k=1}^{K} w_{k,r}}$$

**Final Prediction:**

For a sample routed to region $r$:
$$\hat{y} = \sum_{k=1}^{K} w_{k,r}^{norm} \cdot \hat{y}_k$$

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_regions` | int | 5 | Number of regions to analyze |
| `cv` | int | 5 | Cross-validation folds for OOF predictions |
| `y_percentiles` | array | None | Pre-computed TRUE Y boundaries |
| `refit_base_models` | bool | True | Whether to refit models on CV folds |

**Model Profiles:**

The ensemble tracks specialization for each model:

```python
profiles = ensemble.get_model_profiles()
# Returns:
# {
#   'PLS_SNV': {
#     'weights': [0.35, 0.28, 0.22, 0.15],
#     'specialization': 'specialist',
#     'best_regions': [0, 1],  # Q1, Q2
#     'weight_variance': 0.15
#   },
#   'XGBoost_SG1': {
#     'weights': [0.18, 0.24, 0.28, 0.30],
#     'specialization': 'generalist',
#     'best_regions': [2, 3],  # Q3, Q4
#     'weight_variance': 0.08
#   }
# }
```

### 1.4 MixtureOfExpertsEnsemble

The MixtureOfExpertsEnsemble implements a gating mechanism that selects the best model (or weighted combination) for each region of the target space.

**Hard Gating vs Soft Gating:**

- **Hard Gating:** Uses only the best-performing model for each region
- **Soft Gating:** Uses weighted combination with error-based weights (default)

**Mathematical Formulation (Soft Gating):**

$$\hat{y} = \sum_{k=1}^{K} g_k(r) \cdot \hat{y}_k$$

Where $g_k(r)$ is the gating weight for model $k$ in region $r$:

$$g_k(r) = \frac{\exp(-\text{error}_{k,r} / \tau)}{\sum_{j=1}^{K} \exp(-\text{error}_{j,r} / \tau)}$$

**Expert Assignment:**

The ensemble provides expert assignment information:

```python
assignments = ensemble.get_expert_assignments()
# Returns:
# {
#   'Region 0': {
#     'primary_expert': 'PLS_SNV',
#     'weights': {'PLS_SNV': 0.45, 'Ridge_SG1': 0.30, 'XGBoost': 0.25}
#   },
#   'Region 1': {
#     'primary_expert': 'Ridge_SG1',
#     'weights': {'PLS_SNV': 0.28, 'Ridge_SG1': 0.42, 'XGBoost': 0.30}
#   },
#   ...
# }
```

**Configuration:**

```python
ensemble = MixtureOfExpertsEnsemble(
    models=fitted_models,
    model_names=['PLS', 'Ridge', 'XGBoost'],
    n_regions=4,
    soft_gating=True,  # Use weighted combination
    cv=5,
    y_percentiles=np.percentile(y_train, [0, 25, 50, 75, 100])
)
```

### 1.5 StackingEnsemble

The StackingEnsemble implements a two-level architecture where base model predictions are combined using a Ridge regression meta-learner.

**Architecture:**

```
Level 0 (Base Models):
    Model 1 ─────┐
    Model 2 ─────┼──► OOF Predictions ──► Meta Features
    Model 3 ─────┤
    ...          │
    Model K ─────┘

Level 1 (Meta-Learner):
    Meta Features ──► Ridge Regression ──► Final Prediction
```

**Training Process:**

1. **Generate OOF Predictions:**
   - For each base model, generate out-of-fold predictions using K-fold CV
   - Stack predictions into meta-feature matrix: shape $(n_{samples}, K)$

2. **Add Region Features (if region_aware=True):**
   - One-hot encode region assignments
   - Append average prediction as additional feature
   - Meta-features: $(n_{samples}, K + n_{regions} + 1)$

3. **Fit Meta-Learner:**
   - Train Ridge regression on meta-features
   - Ridge weights become implicit model weights

**Mathematical Formulation:**

The meta-learner learns optimal weights $\mathbf{w}$ by solving:

$$\mathbf{w}^* = \arg\min_{\mathbf{w}} \|\mathbf{y} - \mathbf{M}\mathbf{w}\|_2^2 + \alpha\|\mathbf{w}\|_2^2$$

Where $\mathbf{M}$ is the meta-feature matrix.

Final prediction:
$$\hat{y}_{stack} = \sum_k w_k \hat{y}_k + \text{region\_contributions}$$

**Configuration:**

```python
ensemble = StackingEnsemble(
    models=fitted_models,
    model_names=model_names,
    meta_model=Ridge(alpha=1.0),  # Default meta-learner
    region_aware=True,  # Include region features
    n_regions=5,
    cv=5
)
```

### 1.6 RegionSpecialistEnsemble

The RegionSpecialistEnsemble assigns specific specialist models to each quartile. Unlike MixtureOfExperts, specialists are pre-determined from regional ranking analysis.

**Design Philosophy:**

For a Q1 sample (low concentration range), only Q1's top-performing models contribute to the prediction. This provides:
- Cleaner specialization boundaries
- Reduced noise from poorly-performing models in each region
- Interpretable model assignments

**Workflow:**

1. **Regional Ranking Analysis:**
   - Compute per-quartile RMSE from `regional_rmse` column in results
   - Rank models within each quartile

2. **Specialist Selection:**
   - Select top-N models for each quartile (Q1, Q2, Q3, Q4)
   - Same model may be specialist in multiple quartiles

3. **Prediction Routing:**
   - Initial prediction determines sample's quartile
   - Only that quartile's specialists contribute to final prediction

**Prediction Algorithm:**

```python
def predict(self, X):
    # First pass: determine routing
    all_preds = [model.predict(X) for model, _ in all_models]
    initial_pred = np.mean(all_preds, axis=0)

    # Second pass: use specialists
    predictions = np.zeros(len(X))
    for i in range(len(X)):
        region = self._assign_region(initial_pred[i])  # Q1, Q2, Q3, or Q4
        region_models = self.models_per_region[region]

        # Average specialist predictions
        region_preds = [model.predict(X[i:i+1]) for model, _ in region_models]
        predictions[i] = np.mean(region_preds)

    return predictions
```

**Specialist Information:**

```python
info = ensemble.get_specialist_info()
# Returns:
# {
#   'Q1': ['PLS_SNV', 'Ridge_SG1'],
#   'Q2': ['Ridge_SG1', 'ElasticNet_SNV'],
#   'Q3': ['XGBoost_raw', 'RandomForest_SG2'],
#   'Q4': ['XGBoost_raw', 'LightGBM_SNV']
# }
```

### 1.7 Auto-Ensemble Creation

Spectral Predict automatically creates three ensemble configurations at the end of every search run:

| Ensemble | Top Models per Quartile | Typical Model Count |
|----------|------------------------|---------------------|
| Ensemble-Top1 | 1 | 4 (may overlap) |
| Ensemble-Top2 | 2 | 5-8 |
| Ensemble-Top3 | 3 | 8-12 |

**Selection Algorithm:**

1. **Compute Regional Rankings:**
   ```python
   rankings = compute_regional_rankings(results_df, top_n=3)
   # Returns rankings per quartile based on regional_rmse column
   ```

2. **Select Specialists:**
   ```python
   selection = select_top_models_per_region(
       results_df, top_n=3, task_type='regression',
       reconstruct_func, X_train, y_train, all_wavelengths
   )
   ```

3. **Build RegionSpecialistEnsemble:**
   - Compute TRUE Y quartile boundaries
   - Fit ensemble with selected specialists
   - Compute CV-based metrics for fair comparison

**Diversity-Based Model Selection:**

The quartile-based selection naturally promotes diversity:
- Models that excel at low concentrations (Q1) may differ from high-concentration specialists (Q4)
- Avoids selecting similar models that would provide redundant predictions

**Performance Threshold Filtering:**

Models must meet minimum performance thresholds to be included:
- Sufficient samples in regional_rmse computation
- Non-NaN regional metrics
- At least 2 unique models selected

### 1.8 Ensemble Comparison Workflow

**Recommended Workflow:**

1. Run comprehensive search to generate ranked results
2. Examine auto-ensemble performance in Results tab
3. Compare ensemble metrics to best individual model
4. If ensemble underperforms, investigate:
   - Are base models too similar?
   - Is there sufficient regional specialization?
   - Try different ensemble types

**Diagnostic Output (with SPECTRAL_PREDICT_DEBUG=1):**

```
=== DIAGNOSTIC: Ensemble-Top1 ===
  Q1 model 0 (PLS_SNV): R2=0.9234, RMSE=1.45
  Q2 model 0 (Ridge_SG1): R2=0.9156, RMSE=1.52
  Q3 model 0 (XGBoost_raw): R2=0.9312, RMSE=1.38
  Q4 model 0 (XGBoost_raw): R2=0.9312, RMSE=1.38
  ENSEMBLE Ensemble-Top1 (RegionSpecialist): R2=0.9287, RMSE=1.41
```

---

## Chapter 2: Model Development Tab (Tab 7)

The Model Development tab provides interactive tools for refining models selected from the Results tab. This allows fine-tuning of preprocessing, wavelength selection, and hyperparameters.

### 2.1 Overview

The Model Development tab contains four subtabs:

1. **Selection (7A):** Load models from Results or start fresh
2. **Features (7B):** Wavelength and preprocessing configuration
3. **Configuration (7C):** Model type and hyperparameter settings
4. **Results (7D):** Diagnostics and visualization

### 2.2 Loading Models from Results

**Double-Click Workflow:**
1. In the Results tab, double-click any result row
2. The model configuration is automatically loaded into Model Development
3. All preprocessing, wavelengths, and hyperparameters are populated

**Mode Status:**
- "Mode: Refining [Model Name]" - Loaded from Results
- "Mode: Fresh Development" - Starting from defaults

### 2.3 Wavelength Subset Configuration

**Quick Presets:**

| Preset | Wavelength Range | Description |
|--------|-----------------|-------------|
| All | Full spectrum | Use all available wavelengths |
| NIR Only | 700-2500 nm | Near-infrared region |
| Visible | 350-700 nm | Visible spectrum |
| Custom Range | User-defined | Enter specific ranges |

**Custom Wavelength Specification:**

Enter wavelengths as comma-separated values or ranges:
```
1920, 1930-1940, 1950, 2100-2200
```

**Real-Time Wavelength Count:**
The interface shows "Wavelengths: N selected" updated as you type.

### 2.4 Preprocessing Configuration

**Available Methods:**

| Code | Description |
|------|-------------|
| `raw` | No preprocessing |
| `snv` | Standard Normal Variate |
| `sg1` | 1st derivative (Savitzky-Golay) |
| `sg2` | 2nd derivative |
| `sg3` | 3rd derivative |
| `sg4` | 4th derivative |
| `snv_sg1` | SNV followed by 1st derivative |
| `snv_sg2` | SNV followed by 2nd derivative |
| `deriv_snv` | Derivative followed by SNV |
| `ga_optimized` | Genetic algorithm optimized |

**Window Size Options:**
7, 11, 17, 23, 31 (for Savitzky-Golay derivatives)

### 2.5 Manual Hyperparameter Adjustment

Each model type exposes relevant hyperparameters:

**PLS:**
- `n_components`: Number of latent variables (1-50)

**Ridge/Lasso/ElasticNet:**
- `alpha`: Regularization strength
- `l1_ratio`: ElasticNet mixing parameter (0-1)

**RandomForest/XGBoost/LightGBM:**
- `n_estimators`: Number of trees
- `max_depth`: Maximum tree depth
- `learning_rate`: Step size (boosting methods)

**SVM:**
- `C`: Regularization parameter
- `kernel`: Kernel type (rbf, linear, poly)
- `gamma`: Kernel coefficient

### 2.6 Predicted vs Actual Plots

After running a refined model, the Results subtab displays:

**Predicted vs Actual Scatter Plot:**
- X-axis: Reference (actual) values
- Y-axis: Predicted values
- Diagonal line represents perfect prediction
- Points colored by prediction error

**Key Metrics Displayed:**
- R^2 (coefficient of determination)
- RMSE (root mean square error)
- RPD (residual predictive deviation)
- Bias

### 2.7 Residual Analysis

**Residual Plot:**
- X-axis: Predicted values
- Y-axis: Residuals (actual - predicted)
- Horizontal line at y=0

**What to Look For:**
- Random scatter around zero: Good model
- Patterns (curved, funnel): Model inadequacy
- Outliers: Potential measurement errors

### 2.8 Leave-One-Out Analysis

The LOO analysis identifies influential samples:

1. For each sample $i$:
   - Train model on all samples except $i$
   - Predict sample $i$
   - Record prediction error

2. **Leverage Values:**
   $$h_{ii} = \mathbf{x}_i^T (\mathbf{X}^T\mathbf{X})^{-1} \mathbf{x}_i$$

3. **Cook's Distance:**
   $$D_i = \frac{e_i^2}{p \cdot MSE} \cdot \frac{h_{ii}}{(1-h_{ii})^2}$$

**Interpretation:**
- High leverage + high residual = Influential outlier
- Consider investigating or removing

---

## Chapter 3: Model Prediction Tab (Tab 8)

The Model Prediction tab enables loading saved .dasp models and applying them to new spectral data.

### 3.1 Loading Saved Models

**Supported Formats:**
- Individual models: `.dasp` files
- Ensemble models: `.dasp` files with `ensemble_config.json`

**Loading Process:**
1. Click "Load Model File(s)"
2. Select one or more .dasp files
3. Models appear in the loaded models list

**Loaded Model Display:**
```
1. [Model] soil_nitrogen_pls.dasp
   Model: PLSRegression (n_components=8)
   Preprocessing: snv_sg1, Window=17
   Performance: R²=0.9234, RMSE=1.45

2. [Ensemble] top3_ensemble.dasp
   Type: RegionSpecialist
   Base Models: PLS_SNV, Ridge_SG1, XGBoost_raw
   Performance: R²=0.9312, RMSE=1.38
```

### 3.2 Model File Format (.dasp)

The `.dasp` file is a ZIP archive containing:

```
model.dasp/
├── model.pkl           # Serialized sklearn model
├── preprocessor.pkl    # Fitted preprocessing pipeline
├── metadata.json       # Model metadata
├── wavelengths.npy     # Expected wavelengths
└── [ensemble_config.json]  # For ensembles only
```

**metadata.json Structure:**
```json
{
  "model_type": "PLSRegression",
  "preprocessing": "snv_sg1",
  "window": 17,
  "n_components": 8,
  "r2": 0.9234,
  "rmse": 1.45,
  "target_column": "Nitrogen",
  "created_date": "2024-01-15T14:30:00",
  "spectral_predict_version": "1.5.0"
}
```

### 3.3 Data Sources for Prediction

**Directory (ASD/SPC):**
- Browse to folder containing spectral files
- Automatically reads all .asd or .spc files
- Sample IDs from filenames

**CSV/Excel File:**
- Standard format: wavelengths as columns
- First column: sample identifiers
- Subsequent columns: spectral values

**Pre-Selected Validation Set:**
- Uses validation samples excluded during model building
- Enables comparison of predicted vs actual values
- Reference vs Predicted plots available

### 3.4 Batch Prediction Workflow

1. **Load Models:** Select multiple .dasp files
2. **Load Data:** Choose data source and load
3. **Run Predictions:** Click "Run All Models"
4. **View Results:** Predictions table and statistics

**Predictions Table Columns:**
- Sample ID
- Model 1 Prediction
- Model 2 Prediction
- ...
- Consensus (average)
- Std Dev (uncertainty estimate)

### 3.5 Prediction Confidence Intervals

**For Regression Models:**
- Bootstrap-based intervals
- Standard deviation across models (if multiple loaded)
- LOO-based uncertainty from training

**For Classification Models:**
- Class probabilities from predict_proba
- Confidence = max(probabilities)
- Flag low-confidence predictions

**Uncertainty Display:**
```
Sample    Prediction    95% CI Low    95% CI High    Confidence
S001      15.23        14.12         16.34          High
S002      12.87        11.45         14.29          High
S003       8.92         6.21         11.63          Low*
```

---

## Chapter 4: Calibration Transfer Tab (Tab 10)

Calibration transfer enables applying models developed on one instrument (primary) to spectra measured on a different instrument (satellite).

### 4.1 Overview

**When Calibration Transfer is Needed:**
- Using models on a different spectrometer
- After instrument maintenance/parts replacement
- Comparing data across laboratories

**Method Selection Guide:**

| Scenario | Recommended Method |
|----------|-------------------|
| Same wavelength range + 10+ standards | PDS |
| Same wavelength range + <10 standards | DS |
| Different wavelength ranges | CTAI |
| No transfer standards available | Feature-based |

### 4.2 Direct Standardization (DS)

DS computes a global linear transformation matrix to map satellite spectra to primary space.

**Mathematical Formulation:**

$$\mathbf{S}_{new} = \mathbf{A} \cdot \mathbf{S}_{old}$$

Where the transfer matrix $\mathbf{A}$ is computed as:

$$\mathbf{A} = \mathbf{S}_{master}^+ \cdot \mathbf{S}_{slave}$$

Here $\mathbf{S}_{master}^+$ is the Moore-Penrose pseudoinverse of the master (primary) spectra matrix.

**With Regularization:**

$$\mathbf{A} = (\mathbf{X}_{sat}^T \mathbf{X}_{sat} + \lambda\mathbf{I})^{-1} \mathbf{X}_{sat}^T \mathbf{X}_{pri}$$

**Parameters:**
- `lambda`: Ridge regularization (default: 0.001)

**Transfer Sample Requirements:**
- Minimum: 5-10 samples
- Recommended: 10-30 samples
- Same samples measured on both instruments

### 4.3 Piecewise Direct Standardization (PDS)

PDS computes local transfer coefficients for each wavelength using a sliding window, better handling non-linear instrument differences.

**Mathematical Formulation:**

For each wavelength $\lambda_i$, compute local regression coefficients:

$$x_{primary,i} = \sum_{j \in window(i)} b_{i,j} \cdot x_{satellite,j}$$

**Algorithm:**

```python
def estimate_pds(X_primary, X_satellite, window=11):
    B = np.zeros((n_wavelengths, window))
    half_window = window // 2

    for i in range(n_wavelengths):
        start = max(0, i - half_window)
        end = min(n_wavelengths, i + half_window + 1)

        X_window = X_satellite[:, start:end]
        y_target = X_primary[:, i]

        # Least squares fit
        b = np.linalg.lstsq(X_window, y_target, rcond=None)[0]
        B[i, offset:offset+len(b)] = b

    return B
```

**Parameters:**
- `window`: Window size (odd integer, default: 11)

**When to Use PDS over DS:**
- Non-linear spectral shifts between instruments
- Wavelength-dependent response differences
- Better performance with sufficient transfer samples

### 4.4 Transfer Sample Regression (TSR / Shenk-Westerhaus)

TSR estimates per-wavelength slope and bias corrections using a small set of transfer samples.

**Mathematical Formulation:**

For each wavelength $\lambda$:
$$x_{primary,\lambda} = slope_\lambda \cdot x_{satellite,\lambda} + bias_\lambda$$

**Algorithm:**

1. Select transfer samples (12-13 recommended)
2. For each wavelength, fit linear regression
3. Store slope and bias arrays
4. Apply transformation: `X_new = X_old * slope + bias`

**Quality Metrics:**
- Per-wavelength R^2
- Mean R^2 across wavelengths

**Advantages:**
- Simple and fast
- Works well with Kennard-Stone sample selection
- Can achieve near-recalibration accuracy with optimal samples

### 4.5 CTAI (Calibration Transfer based on Affine Invariance)

CTAI is a transfer-standard-free method that leverages affine invariance properties.

**Key Innovation:** No transfer samples required!

**Mathematical Formulation:**

Estimate affine transformation:
$$\mathbf{X}_{primary} \approx \mathbf{X}_{satellite} \cdot \mathbf{M} + \mathbf{T}$$

Where:
- $\mathbf{M}$: Transformation matrix
- $\mathbf{T}$: Translation vector

**Algorithm:**

1. Mean-center both datasets
2. Compute SVD of satellite data
3. Find optimal transformation in reduced-rank space
4. Project back to full wavelength space

**Requirements:**
- Paired samples (same samples on both instruments)
- Sufficient spectral diversity in both datasets

### 4.6 Validation of Transfer Quality

**Transfer Validation Workflow:**

1. **Before Transfer:**
   - Apply primary model to raw satellite spectra
   - Record prediction error (RMSEP_before)

2. **After Transfer:**
   - Transform satellite spectra
   - Apply primary model
   - Record prediction error (RMSEP_after)

3. **Quality Metrics:**
   - Improvement ratio: RMSEP_before / RMSEP_after
   - Spectral reconstruction RMSE
   - Per-wavelength correlation

**Visualization:**
- Overlay transferred vs primary spectra
- Difference spectrum before/after transfer
- Prediction correlation plots

---

## Chapter 5: Interference Removal Tab (Tab 11)

The Interference Removal tab provides advanced methods for removing systematic interference (moisture, temperature, particle size) from spectral data.

### 5.1 Overview of Interference Methods

| Method | Requires Y | Requires Library | Best For |
|--------|-----------|-----------------|----------|
| OSC | Yes | No | General Y-orthogonal noise |
| EPO | Optional | Yes | Known interferents |
| DOSC | Yes | No | Stable alternative to OSC |
| GLSW | Optional | No | Heteroscedastic noise |
| Wavelength Exclusion | No | No | Known bad regions |

### 5.2 EPO (External Parameter Orthogonalization)

EPO removes specific interferent effects using a reference library of interferent spectra.

**Algorithm:**

1. **Build Interferent Subspace:**
   - Load interferent library (e.g., moisture at different levels)
   - Center the library
   - Compute SVD to find principal directions

2. **Create Orthogonal Projector:**
   $$\mathbf{P}_{orth} = \mathbf{I} - \mathbf{V}\mathbf{V}^T$$
   Where $\mathbf{V}$ contains the first $n$ principal components

3. **Apply Correction:**
   $$\mathbf{X}_{corrected} = (\mathbf{X} - \bar{\mathbf{X}}) \cdot \mathbf{P}_{orth}$$

**Building Interferent Libraries:**

For moisture interference:
1. Measure same samples at different moisture levels
2. Stack into matrix: shape (n_levels, n_wavelengths)
3. Save as CSV or NPY file

**Parameters:**
- `n_components`: Number of interferent directions to remove (1-10)
- `center`: Mean-center data before EPO (default: True)
- `svd_tol`: Numerical tolerance (default: 1e-8)

**Practical Example (Moisture):**

```python
# Load moisture library (spectra at 5%, 10%, 15%, 20% moisture)
X_moisture = np.loadtxt('moisture_library.csv', delimiter=',')

# Fit EPO
epo = EPO(n_components=3)
epo.fit(X_train, y_train, X_interferents=X_moisture)

# Transform spectra
X_corrected = epo.transform(X_train)
X_test_corrected = epo.transform(X_test)
```

### 5.3 DOSC (Direct Orthogonal Signal Correction)

DOSC is a simplified, more stable variant of OSC that directly computes the Y-orthogonal subspace.

**Algorithm:**

1. Fit PLS model to find Y-relevant directions
2. Compute residuals orthogonal to Y-predictive space:
   $$\mathbf{E}_{orth} = \mathbf{X} - \mathbf{T}_{PLS}\mathbf{P}_{PLS}^T$$
3. Extract principal components of residuals
4. Build orthogonal projector from these components

**Advantages over OSC:**
- No iterative deflation (more stable)
- Computationally efficient
- Better handling of noisy data

**Parameters:**
- `n_components`: Y-orthogonal components to remove (1-3 typical)
- `center`: Mean-center data (default: True)
- `n_pls_components`: PLS components for Y-space ('auto' or int)

### 5.4 GLSW (Generalized Least Squares Weighting)

GLSW down-weights wavelength regions with high noise/interference while preserving informative regions.

**Concept:**
- Wavelengths with high variance get lower weights
- Equivalent to weighted least squares regression

**Weighting Methods:**

1. **Covariance Method:**
   $$w_\lambda = \frac{1}{\text{Var}(X_\lambda) + \epsilon}$$

2. **Residual Method:**
   - Fit PLS model
   - Weight by inverse of residual variance

**Application:**
```python
glsw = GLSW(method='covariance', regularization=1e-6)
X_weighted = glsw.fit_transform(X_train)
```

**Feature Weights:**
```python
weights = glsw.get_feature_weights()
# Array of weights for each wavelength
# Higher weight = more informative
```

### 5.5 Wavelength Exclusion

Simple but effective approach: remove wavelength regions known to be dominated by interference.

**Common NIR Exclusion Regions:**

| Region (nm) | Source |
|-------------|--------|
| 1400-1500 | O-H stretch (water) |
| 1900-2000 | O-H combination (water) |
| 2300-2400 | CO2 absorption |

**Usage:**
```python
from spectral_predict.interference import WavelengthExcluder

excluder = WavelengthExcluder(
    wavelengths=wavelengths,
    exclude_ranges=[(1400, 1500), (1900, 2000)]
)
X_filtered = excluder.fit_transform(X)
```

---

## Chapter 6: Spectral Library Tab (Tab 12)

The Spectral Library tab enables building and searching a persistent library of reference spectra.

### 6.1 Building Reference Libraries

**Adding Spectra:**
1. Load data in the Data Upload tab
2. Navigate to Spectral Library tab
3. Enter category name (e.g., "Soil Samples")
4. Click "Add Current Data to Library"

**Library Storage:**
- Persistent SQLite database in user data directory
- Automatic duplicate detection via spectral fingerprinting
- Categories for organization

**Library Statistics:**
```
Current Library:
- Total spectra: 1,247
- Categories: 5
- Wavelength range: 350-2500 nm
- Resolution: 1 nm
```

### 6.2 Similarity Search Algorithms

**Available Metrics:**

1. **HQI (Hit Quality Index):**
   $$HQI = r^2 = \left(\frac{\sum(x - \bar{x})(y - \bar{y})}{\sqrt{\sum(x-\bar{x})^2 \sum(y-\bar{y})^2}}\right)^2$$
   - Industry standard (0-1 scale)
   - Insensitive to intensity scaling

2. **SAM (Spectral Angle Mapper):**
   $$SAM = \cos^{-1}\left(\frac{\mathbf{x} \cdot \mathbf{y}}{|\mathbf{x}||\mathbf{y}|}\right)$$
   - Angle between spectra as vectors (radians)
   - Completely insensitive to intensity

3. **Euclidean Distance:**
   $$d = \sqrt{\sum_\lambda (x_\lambda - y_\lambda)^2}$$
   - Simple L2 norm
   - Sensitive to intensity differences

4. **Cosine Similarity:**
   $$\cos\theta = \frac{\mathbf{x} \cdot \mathbf{y}}{|\mathbf{x}||\mathbf{y}|}$$
   - -1 to 1 scale
   - Related to SAM

5. **Derivative Correlations:**
   - 1st derivative HQI: Robust to baseline shifts
   - 2nd derivative HQI: More aggressive baseline removal

6. **SID (Spectral Information Divergence):**
   $$SID = D_{KL}(p||q) + D_{KL}(q||p)$$
   - Based on KL divergence
   - Treats spectra as probability distributions

### 6.3 Library Matching Workflow

1. **Select Query:**
   - From current loaded data, OR
   - From existing library entry

2. **Configure Search:**
   - Select similarity metric
   - Set top-K results (1-50)
   - Optionally filter by category

3. **Run Search:**
   - Returns ranked list of matches
   - Score interpretation depends on metric

4. **Compare Results:**
   - Overlay query with matches
   - Examine difference spectra
   - Export results to CSV

**Search Results Table:**
| Rank | Sample ID | Score | Category | Source File |
|------|-----------|-------|----------|-------------|
| 1 | REF_042 | 0.9876 | Soil | soil_library.csv |
| 2 | REF_089 | 0.9654 | Soil | soil_library.csv |
| 3 | REF_017 | 0.9432 | Compost | compost_set.csv |

---

## Chapter 7: Contaminant Analysis Tab (Tab 13)

The Contaminant Analysis tab provides methods for identifying and removing contaminant influences when pure contaminant spectra are unavailable.

### 7.1 Overview

**Use Case:**
Bone samples with and without consolidants (like Glyptal). Goal is to predict % collagen accurately, but cannot scan pure contaminant due to probe geometry.

**Available Methods:**
- DifferenceAnalyzer: Exploratory analysis
- EstimatedEPO: EPO with estimated interferent library
- ContaminantOPLSDA: OPLS-DA based detection
- ContaminantGLSW: Contaminant-aware weighting

### 7.2 Defining Contaminant Groups

**Group Definition:**
1. Load clean (uncontaminated) samples
2. Load contaminated samples
3. Define group membership

**Group Requirements:**
- At least 5 samples per group recommended
- Groups should be clearly distinguishable
- Same measurement conditions for both groups

### 7.3 Difference Spectrum Analysis

The DifferenceAnalyzer computes and visualizes differences between contaminated and uncontaminated groups.

**Algorithm:**

1. Optionally normalize spectra (SNV)
2. Compute representative spectrum for each group (mean/median/PCA)
3. Calculate difference: contaminated - uncontaminated
4. Identify peak regions above threshold

**Methods for Group Representative:**
- `mean`: Mean spectrum (default)
- `median`: More robust to outliers
- `pca`: First principal component direction

**Peak Region Identification:**
```python
analyzer = DifferenceAnalyzer(normalize=True, method='mean')
analyzer.fit(X_contaminated, X_uncontaminated)

# Identify regions with >50% of maximum influence
peaks = analyzer.identify_peak_regions(wavelengths, threshold=0.5, min_width=3)
# Returns: [(1456, 1478, 0.87), (1923, 1952, 0.92)]
```

**Confidence Intervals:**
```python
lower, upper = analyzer.get_confidence_interval(confidence=0.95)
# Statistical bounds on difference spectrum
```

### 7.4 PCA-Based Detection

PCA can separate contaminated from uncontaminated samples:

1. **Fit PCA on combined data**
2. **Plot PC1 vs PC2:**
   - Contaminated samples cluster separately
   - PC loadings indicate wavelengths driving separation

3. **Use PC scores for screening:**
   - Define threshold on discriminating PC
   - Classify new samples as contaminated/clean

### 7.5 Automated Detection Methods

**EstimatedEPO:**

When pure contaminant spectra are unavailable, EstimatedEPO builds a "pseudo-interferent library" from group differences.

**Algorithm:**
1. Compute difference vectors between groups
2. Use PCA or bootstrap to build interferent library
3. Apply standard EPO with estimated library

**Estimation Methods:**
- `mean_diff`: Single difference between group means
- `pca_diff`: PCA on concatenated groups
- `bootstrap`: Multiple bootstrap difference vectors

```python
epo = EstimatedEPO(n_components=2, estimation_method='pca_diff')
epo.fit_groups(X_contaminated, X_uncontaminated)
X_corrected = epo.transform(X_all)
```

**ContaminantGLSW:**

Weighted approach that down-weights contaminant-influenced wavelengths:

1. Identify wavelengths with high group differences
2. Assign lower weights to these wavelengths
3. Apply weighted regression

### 7.6 Threshold Setting

**Manual Threshold:**
- Based on domain knowledge
- Visual inspection of difference spectrum
- Known contaminant absorption bands

**Statistical Threshold:**
- Confidence interval on difference
- Wavelengths outside CI are significant
- Multiple testing correction (Bonferroni)

**Data-Driven Threshold:**
- Cross-validation to optimize prediction accuracy
- Grid search over threshold values
- Select threshold minimizing RMSECV

### 7.7 Choosing Between Methods (Practical Guide)

This section provides practical guidance for using the contaminant analysis tools effectively. It answers common questions about why methods produce different results and which settings to use.

#### 7.7.1 Why Difference Analysis Excludes More Than Automated Detection

New users often notice that **Difference Analysis** (Section 7.3) flags more wavelengths for exclusion than **Estimated EPO** or **OPLS-DA** (Section 7.5). This is expected behavior, not a bug.

**How Difference Analysis works:**
The DifferenceAnalyzer computes a raw subtraction between the mean (or median) contaminated and uncontaminated spectra. Its influence score at each wavelength is simply the absolute difference, normalized to 0–1. This means *any* difference between the two groups shows up — including:
- True contaminant absorption bands
- Normal sample-to-sample variation
- Baseline drift between measurement sessions
- Noise amplified by normalization

The resulting influence curve tends to be **broad and bumpy**, with many wavelengths exceeding the threshold.

**How statistical methods work:**
EstimatedEPO and OPLS-DA use multivariate decomposition (PCA or PLS) to separate the *systematic* contamination signal from noise. Their influence curves are computed from squared loadings (EPO) or VIP scores (OPLS-DA), which concentrate influence at wavelengths where the contaminant has a *structured, repeatable* effect.

The resulting influence curve is **peaky and selective**, typically flagging only the wavelengths with genuine contaminant absorption.

**Rule of thumb:** If Difference Analysis flags 30–50% of your wavelength range but EPO flags 5–15%, the statistical methods are likely more accurate. Difference Analysis is best used as an exploratory tool to visually confirm where contamination occurs, not as the primary method for deciding which wavelengths to exclude.

#### 7.7.2 Recommended Method for Model Building

For most users building predictive models, **Estimated EPO** with the `pca_diff` estimation method is the best general-purpose choice.

| Method | Needs Y? | Sensitivity | Best For |
|--------|----------|-------------|----------|
| DifferenceAnalyzer | No | High (over-inclusive) | Exploratory visualization, confirming contaminant location |
| EstimatedEPO | No | Moderate (selective) | **General-purpose model building** (recommended) |
| ContaminantOPLSDA | No | Moderate (selective) | When you want VIP-based importance ranking |
| ContaminantGLSW | No | Soft (no hard exclusion) | When you want to down-weight rather than exclude |
| RegionExcluder | **Yes** | Data-driven | When you have Y and want iPLS-optimized regions |

**Why EPO is recommended:**
1. It does not require Y values, so it works even before you have reference measurements
2. It uses PCA to build an interferent library, capturing the dominant contamination pattern
3. It can *transform* new data (project out the interferent), not just identify regions
4. The `pca_diff` estimation method handles subtle, distributed contamination better than `mean_diff`

**When to use OPLS-DA instead:**
- When you want a ranked importance score (VIP) for each wavelength
- When contamination affects many small, scattered wavelength regions
- When you need the discriminant direction (which wavelengths go up vs. down with contamination)

**When to use GLSW instead:**
- When you want a "soft" correction that reduces contamination influence without fully removing wavelengths
- When you are unsure about the threshold and want a more forgiving approach
- GLSW assigns a weight between `min_weight` (default 0.1) and 1.0 to each wavelength, so no data is completely discarded

#### 7.7.3 Handling Multiple Contaminants

When samples may contain different contaminant types (e.g., Glyptal, Paraloid B-72, or both), use the **MultiContaminantAnalyzer** class.

**Workflow:**
1. Define your clean (uncontaminated) reference group
2. Define separate contaminated groups — one per contaminant type
3. The analyzer runs EPO separately for each contaminant group
4. Influences are combined using an aggregation method

**Aggregation options:**

| Aggregation | Formula | Effect | Use When |
|-------------|---------|--------|----------|
| `max` (default) | Highest influence across contaminants | Conservative — excludes a wavelength if *any* contaminant affects it | You want to be safe and remove all contamination risk |
| `mean` | Average influence across contaminants | Balanced — wavelengths need to be affected by multiple contaminants | You have many contaminant types and want to avoid over-exclusion |
| `sum` | Sum of all influences | Aggressive — accumulates evidence from all contaminants | Contaminants have subtle effects that individually fall below threshold |

**Recommended approach for multiple contaminants:**
Use `aggregation='max'` (the default). This ensures that a wavelength affected by even one contaminant type is flagged. Since different contaminants typically affect different spectral regions, `max` does not over-exclude in practice.

**Example usage:**
```python
from spectral_predict.contaminant_analysis import MultiContaminantAnalyzer

analyzer = MultiContaminantAnalyzer(
    n_epo_components=2,
    estimation_method='pca_diff',
    aggregation='max'
)

contaminant_groups = {
    'Glyptal': X_glyptal,
    'Paraloid': X_paraloid,
    'Both': X_both
}

analyzer.fit(X_clean, contaminant_groups)
regions = analyzer.get_exclusion_regions(wavelengths, threshold=0.5)
```

#### 7.7.4 Practical Settings Reference

**Threshold:**

| Scenario | Recommended Threshold | Rationale |
|----------|----------------------|-----------|
| Strong, localized contaminant | 0.5–0.7 | Clear peaks make higher thresholds safe |
| Subtle, distributed contaminant | 0.3–0.5 | Lower threshold catches weaker effects |
| Multiple contaminants with `max` aggregation | 0.5 | Good balance between sensitivity and specificity |
| Exploratory / first pass | 0.5 (default) | Start here and adjust based on results |

**EPO components (`n_components`):**

| Scenario | Recommended Components | Rationale |
|----------|----------------------|-----------|
| Single known contaminant | 1–2 | Contamination is low-rank; 1–2 components capture it |
| Multiple contaminants or unknown | 2–3 | Extra components capture secondary effects |
| Very complex contamination | 3–5 | Rarely needed; risk of removing real signal |

**General guidance:**
- Start with `n_components=2` and `threshold=0.5` — these defaults work well for most cases
- Increase components only if the influence plot shows clear contamination that is not being captured
- Lower the threshold if known contaminant regions are not being flagged
- Raise the threshold if the model is losing too many informative wavelengths
- Always compare the influence plots from DifferenceAnalyzer (exploratory) and EstimatedEPO (statistical) side by side to build confidence in your exclusion decisions

---

## Appendix A: Ensemble Formula Summary

### A.1 Simple Average
$$\hat{y} = \frac{1}{K}\sum_{k=1}^{K} \hat{y}_k$$

### A.2 Region-Aware Weighted
$$\hat{y} = \sum_{k=1}^{K} w_{k,r(x)} \cdot \hat{y}_k$$

Where $r(x)$ is the region assigned to sample $x$.

### A.3 Stacking Meta-Learner
$$\hat{y}_{stack} = \mathbf{w}^T \cdot [\hat{y}_1, \hat{y}_2, ..., \hat{y}_K, \mathbf{r}, \bar{y}]$$

Where $\mathbf{r}$ is one-hot region encoding and $\bar{y}$ is average prediction.

### A.4 Mixture of Experts (Soft Gating)
$$\hat{y} = \sum_{k=1}^{K} \frac{e^{-\epsilon_k/\tau}}{\sum_j e^{-\epsilon_j/\tau}} \cdot \hat{y}_k$$

---

## Appendix B: Calibration Transfer Formula Summary

### B.1 Direct Standardization (DS)
$$\mathbf{A} = (\mathbf{X}_{sat}^T \mathbf{X}_{sat} + \lambda\mathbf{I})^{-1} \mathbf{X}_{sat}^T \mathbf{X}_{pri}$$

$$\mathbf{X}_{transferred} = \mathbf{X}_{sat} \cdot \mathbf{A}$$

### B.2 Piecewise Direct Standardization (PDS)
$$x_{pri,i} = \sum_{j \in W_i} b_{i,j} \cdot x_{sat,j}$$

Where $W_i$ is the window around wavelength $i$.

### B.3 Transfer Sample Regression (TSR)
$$x_{pri,\lambda} = slope_\lambda \cdot x_{sat,\lambda} + bias_\lambda$$

### B.4 CTAI
$$\mathbf{X}_{transferred} = \mathbf{X}_{sat} \cdot \mathbf{M} + \mathbf{T}$$

---

## Appendix C: Interference Removal Formula Summary

### C.1 EPO Projection
$$\mathbf{P}_{orth} = \mathbf{I} - \mathbf{V}_k\mathbf{V}_k^T$$

$$\mathbf{X}_{corrected} = \mathbf{X}_{centered} \cdot \mathbf{P}_{orth}$$

### C.2 GLSW Weighting
$$w_\lambda = \frac{1}{Var(X_\lambda) + \epsilon}$$

$$\mathbf{X}_{weighted} = \mathbf{X} \cdot \text{diag}(\sqrt{\mathbf{w}})$$

### C.3 OSC Component Removal
$$\mathbf{X}_{corrected} = \mathbf{X} - \mathbf{t}_{orth}\mathbf{p}_{orth}^T$$

Where $\mathbf{t}_{orth}$ is the Y-orthogonal score vector.

---

## Appendix D: Troubleshooting Guide

### D.1 Ensemble Performance Issues

**Problem:** Ensemble performs worse than best individual model

**Solutions:**
1. Check model diversity - are base models too similar?
2. Reduce number of base models
3. Try different ensemble type (stacking vs region-weighted)
4. Verify regional rankings are meaningful

### D.2 Calibration Transfer Failures

**Problem:** High RMSEP after transfer

**Solutions:**
1. Increase number of transfer samples
2. Try different method (PDS if using DS)
3. Check wavelength alignment between instruments
4. Verify transfer samples span concentration range

### D.3 Interference Removal Side Effects

**Problem:** EPO removes analyte signal

**Solutions:**
1. Reduce n_components
2. Verify interferent library doesn't overlap with analyte
3. Use GLSW instead for subtle corrections
4. Validate with held-out test set

---

*Part VII of the Spectral Predict V1 User Guide*

*Document Version: 1.0*
*Last Updated: January 2026*
