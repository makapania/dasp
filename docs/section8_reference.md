# Part VIII: Reference

This comprehensive reference section provides detailed documentation on all metrics, file formats, and terminology used throughout Spectral Predict. Whether you need to interpret model results, troubleshoot file import issues, or understand technical terminology, this section serves as your definitive guide.

---

## Chapter 22: Regression Metrics

Regression metrics quantify how well a model predicts continuous target values (e.g., chemical concentrations, physical properties). Understanding these metrics is essential for model selection and validation.

### 22.1 Root Mean Square Error (RMSE)

**Formula:**

$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

Where:
- $y_i$ = actual (observed) value for sample $i$
- $\hat{y}_i$ = predicted value for sample $i$
- $n$ = number of samples

**Interpretation:**
- RMSE represents the standard deviation of prediction errors (residuals)
- Expressed in the same units as the target variable
- Lower values indicate better model performance
- Sensitive to outliers due to squaring of errors

**Example:** An RMSE of 0.5% for nitrogen prediction means predictions typically deviate from actual values by approximately 0.5 percentage points.

**Advantages:**
- Intuitive interpretation in original units
- Penalizes large errors more heavily than small errors
- Industry standard metric in chemometrics

**Limitations:**
- Cannot compare across different target scales
- Outliers can disproportionately influence the value

---

### 22.2 RMSE Variants

Spectral Predict reports three RMSE variants based on the dataset used for calculation:

| Metric | Dataset | Purpose | Symbol |
|--------|---------|---------|--------|
| **RMSEC** | Training (Calibration) | Assesses model fit to training data | $RMSE_{cal}$ |
| **RMSEcv** | Cross-Validation | Estimates generalization performance | $RMSE_{cv}$ |
| **RMSEP** | External Prediction | True out-of-sample performance | $RMSE_{pred}$ |

#### RMSEC (RMSE Calibration)

$$RMSEC = \sqrt{\frac{1}{n_{cal}}\sum_{i=1}^{n_{cal}}(y_i - \hat{y}_i)^2}$$

Calculated using predictions on the same data used for training. RMSEC is always optimistic (lower than true performance) due to overfitting. Use for:
- Checking model fit
- Detecting underfitting (if RMSEC is high, model cannot fit training data)

**Warning:** RMSEC alone should never be used for model selection, as complex models will always achieve lower RMSEC by memorizing training data.

#### RMSEcv (RMSE Cross-Validation)

$$RMSEcv = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_{i,cv})^2}$$

Where $\hat{y}_{i,cv}$ is the prediction for sample $i$ when it was in the validation fold (i.e., the model never saw that sample during training for that prediction).

**Key Insight:** RMSEcv aggregates predictions from all CV folds, computing RMSE on the combined set. This provides an unbiased estimate of generalization performance.

**Best Practice:** RMSEcv is the primary metric for model selection in Spectral Predict because it:
- Provides unbiased performance estimate
- Uses all available data efficiently
- Accounts for model complexity through validation

#### RMSEP (RMSE Prediction)

$$RMSEP = \sqrt{\frac{1}{n_{pred}}\sum_{i=1}^{n_{pred}}(y_i - \hat{y}_i)^2}$$

Calculated on a completely independent test set that was never used during model development. RMSEP represents true out-of-sample performance.

**When to Use:** RMSEP should be reported when:
- Publishing research results
- Validating models for production deployment
- Comparing against literature values

---

### 22.3 Coefficient of Determination (R-squared)

**Formula:**

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

Where:
- $SS_{res}$ = residual sum of squares (sum of squared prediction errors)
- $SS_{tot}$ = total sum of squares (variance in the target variable)
- $\bar{y}$ = mean of actual values

**Interpretation:**
- R-squared represents the proportion of variance explained by the model
- Range: typically 0 to 1 (can be negative for very poor models)
- Higher values indicate better model performance
- R-squared = 0.95 means the model explains 95% of the variance in the target

**R-squared Interpretation Guide:**

| R-squared | Interpretation | Typical Application |
|-----------|----------------|---------------------|
| > 0.99 | Excellent | Laboratory calibrations, pure compounds |
| 0.95 - 0.99 | Very Good | Most quantitative spectroscopy applications |
| 0.90 - 0.95 | Good | Field applications, heterogeneous samples |
| 0.80 - 0.90 | Moderate | Screening applications, complex matrices |
| 0.70 - 0.80 | Acceptable | Qualitative assessment, trend monitoring |
| < 0.70 | Poor | Model may not be suitable for predictions |

**R-squared Variants:**

| Metric | Dataset | Formula Modification |
|--------|---------|---------------------|
| **R-squared (cal)** | Training | Standard formula on calibration data |
| **R-squared cv** | Cross-Validation | Computed from aggregated CV predictions |
| **R-squared pred** | External Test | Computed on independent test set |

---

### 22.4 Mean Absolute Error (MAE)

**Formula:**

$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

**Interpretation:**
- Average absolute difference between predicted and actual values
- Expressed in original units
- Less sensitive to outliers than RMSE

**Comparison with RMSE:**

| Aspect | RMSE | MAE |
|--------|------|-----|
| Outlier sensitivity | High (squared errors) | Low (absolute errors) |
| Interpretability | Standard deviation of errors | Average error magnitude |
| Optimization | Penalizes large errors | Equal weight to all errors |
| Typical relationship | RMSE >= MAE (always) | --- |

**Rule of Thumb:** If RMSE >> MAE, the model has some large errors (outliers). If RMSE is close to MAE, errors are relatively uniform.

---

### 22.5 Regional RMSE (Q1-Q4)

Regional RMSE breaks down model performance across different ranges of the target variable, revealing whether the model performs uniformly or has concentration-dependent performance.

**Quartile Definitions:**

| Region | Range | Description |
|--------|-------|-------------|
| Q1 | 0-25th percentile | Low concentration samples |
| Q2 | 25-50th percentile | Below-median samples |
| Q3 | 50-75th percentile | Above-median samples |
| Q4 | 75-100th percentile | High concentration samples |

**Formula (for each quartile):**

$$RMSE_{Qk} = \sqrt{\frac{1}{n_k}\sum_{i \in Q_k}(y_i - \hat{y}_i)^2}$$

**Interpretation:**
- Uniform regional RMSE indicates consistent model performance
- Higher RMSE in Q1 or Q4 may indicate extrapolation issues
- Higher RMSE in specific regions may indicate:
  - Non-linear relationships in that range
  - Fewer training samples in that range
  - Different spectral characteristics at extreme concentrations

**Use Case:** Regional analysis is essential for ensemble methods, as different models often excel in different concentration ranges. Spectral Predict's region-aware ensembles leverage this specialization.

---

### 22.6 Ratio of Performance to Deviation (RPD)

**Formula:**

$$RPD = \frac{SD_y}{RMSEP}$$

Where:
- $SD_y$ = standard deviation of the reference (actual) values
- $RMSEP$ = root mean square error of prediction

**Alternative Form (using SEP):**

$$RPD = \frac{SD_y}{SEP}$$

Where SEP (Standard Error of Prediction) is equivalent to RMSEP when bias is negligible.

**Interpretation:**

| RPD Value | Interpretation | Recommended Use |
|-----------|----------------|-----------------|
| < 1.5 | Very poor | Not suitable for any application |
| 1.5 - 2.0 | Poor | May distinguish high/low values |
| 2.0 - 2.5 | Acceptable | Rough screening |
| 2.5 - 3.0 | Good | Screening and approximate quantification |
| 3.0 - 5.0 | Very good | Quality control applications |
| 5.0 - 8.0 | Excellent | Process control applications |
| > 8.0 | Outstanding | Reference method replacement |

**Advantages:**
- Dimensionless (allows comparison across different analytes)
- Accounts for natural variability in the dataset
- Widely used in NIR spectroscopy literature

**Limitations:**
- Depends on dataset variability (narrow-range datasets yield lower RPD)
- Should not be used alone for model assessment
- Some statisticians prefer R-squared as it does not depend on sample selection

---

### 22.7 Bias

**Formula:**

$$Bias = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i) = \bar{y} - \bar{\hat{y}}$$

**Interpretation:**
- Bias measures systematic over- or under-prediction
- Positive bias: model systematically under-predicts (actual > predicted)
- Negative bias: model systematically over-predicts (actual < predicted)
- Ideal value: 0 (no systematic error)

**Bias Correction:**
If a model shows consistent bias, predictions can be corrected:

$$\hat{y}_{corrected} = \hat{y} + Bias$$

**Statistical Significance:**
To determine if bias is statistically significant, compare to its standard error:

$$SE_{bias} = \frac{SD_{residuals}}{\sqrt{n}}$$

If $|Bias| > 2 \times SE_{bias}$, the bias is likely statistically significant at p < 0.05.

---

### 22.8 Generalization Gap

**Formula:**

$$Gap = R^2_{cal} - R^2_{cv}$$

Or equivalently:

$$Gap_{RMSE} = RMSE_{cv} - RMSE_{cal}$$

**Interpretation:**

| Gap Value | Interpretation | Action |
|-----------|----------------|--------|
| < 0.02 | Excellent generalization | Model is not overfitting |
| 0.02 - 0.05 | Good generalization | Acceptable for most applications |
| 0.05 - 0.10 | Moderate overfitting | Consider reducing model complexity |
| > 0.10 | Significant overfitting | Reduce components/regularize/simplify |

**Causes of Large Gaps:**
1. Too many latent variables in PLS
2. Insufficient regularization in Ridge/Lasso/ElasticNet
3. Tree-based models too deep
4. Small training dataset relative to model complexity
5. Outliers in training data

---

### 22.9 Metric Summary Table

| Metric | Formula | Range | Optimal | Units | Primary Use |
|--------|---------|-------|---------|-------|-------------|
| RMSE | $\sqrt{\frac{1}{n}\sum(y-\hat{y})^2}$ | [0, infinity) | Lower | Same as target | Error magnitude |
| R-squared | $1 - \frac{SS_{res}}{SS_{tot}}$ | (-infinity, 1] | Higher | Dimensionless | Variance explained |
| MAE | $\frac{1}{n}\sum|y-\hat{y}|$ | [0, infinity) | Lower | Same as target | Robust error |
| RPD | $\frac{SD_y}{RMSEP}$ | [0, infinity) | Higher | Dimensionless | Model utility |
| Bias | $\bar{y} - \bar{\hat{y}}$ | (-infinity, infinity) | 0 | Same as target | Systematic error |

---

## Chapter 23: Classification Metrics

Classification metrics assess model performance when predicting categorical outcomes (e.g., species identification, quality grades, pass/fail). These metrics are fundamentally different from regression metrics.

### 23.1 Accuracy

**Formula:**

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}$$

Where:
- $TP$ = True Positives (correctly predicted positive class)
- $TN$ = True Negatives (correctly predicted negative class)
- $FP$ = False Positives (incorrectly predicted positive)
- $FN$ = False Negatives (incorrectly predicted negative)

**Interpretation:**
- Proportion of correct predictions
- Range: 0 to 1 (often expressed as percentage)
- Higher values indicate better performance

**Accuracy Variants:**

| Metric | Dataset | Description |
|--------|---------|-------------|
| **Accuracy (cal)** | Training | Accuracy on calibration data (optimistic) |
| **Accuracy_cv** | Cross-Validation | Unbiased estimate via CV |
| **Accuracy (pred)** | External Test | True out-of-sample accuracy |

**Limitations:**
Accuracy can be misleading with imbalanced datasets:

*Example:* In a dataset with 95% Class A and 5% Class B, a model that always predicts Class A achieves 95% accuracy but is useless for detecting Class B.

**When to Use:** Accuracy is appropriate when:
- Classes are approximately balanced
- All types of errors are equally costly
- Simple interpretability is needed

---

### 23.2 Precision

**Formula:**

$$Precision = \frac{TP}{TP + FP}$$

**Interpretation:**
- Proportion of positive predictions that are correct
- Answers: "When the model predicts positive, how often is it right?"
- Range: 0 to 1

**High Precision Use Cases:**
- Spam detection (minimize false positives that filter legitimate emails)
- Quality acceptance (minimize accepting defective products)
- Drug discovery (minimize false leads that waste resources)

---

### 23.3 Recall (Sensitivity)

**Formula:**

$$Recall = \frac{TP}{TP + FN} = Sensitivity = True\ Positive\ Rate$$

**Interpretation:**
- Proportion of actual positives correctly identified
- Answers: "Of all actual positive cases, how many did the model find?"
- Range: 0 to 1

**High Recall Use Cases:**
- Disease screening (minimize missing actual cases)
- Fraud detection (minimize undetected fraud)
- Safety-critical applications (minimize missed defects)

---

### 23.4 Specificity

**Formula:**

$$Specificity = \frac{TN}{TN + FP} = True\ Negative\ Rate$$

**Interpretation:**
- Proportion of actual negatives correctly identified
- Answers: "Of all actual negative cases, how many did the model correctly identify?"
- Range: 0 to 1

**Relationship with Recall:**
- Recall focuses on positive class detection
- Specificity focuses on negative class detection
- Both are important for balanced assessment

---

### 23.5 F1 Score

**Formula:**

$$F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

**Interpretation:**
- Harmonic mean of precision and recall
- Balances both metrics in a single score
- Range: 0 to 1

**Properties:**
- F1 = 1 only when both precision and recall are perfect
- F1 is low when either precision or recall is low
- Harmonic mean penalizes extreme imbalances more than arithmetic mean

**F1 Variants for Multi-class:**

| Variant | Calculation | Use Case |
|---------|-------------|----------|
| **Micro F1** | Global TP, FP, FN across all classes | When class imbalance exists |
| **Macro F1** | Average F1 across classes (unweighted) | When all classes equally important |
| **Weighted F1** | Average F1 weighted by class frequency | General purpose |

---

### 23.6 ROC-AUC

**Definition:**
ROC-AUC (Receiver Operating Characteristic - Area Under Curve) measures the probability that the model ranks a randomly chosen positive instance higher than a randomly chosen negative instance.

**ROC Curve:**
- X-axis: False Positive Rate = 1 - Specificity
- Y-axis: True Positive Rate = Recall
- Each point represents a different classification threshold

**AUC Interpretation:**

| AUC Value | Interpretation |
|-----------|----------------|
| 1.0 | Perfect classification |
| 0.9 - 1.0 | Excellent |
| 0.8 - 0.9 | Good |
| 0.7 - 0.8 | Fair |
| 0.6 - 0.7 | Poor |
| 0.5 | Random guessing |
| < 0.5 | Worse than random (model is inverted) |

**Advantages:**
- Threshold-independent (evaluates model across all thresholds)
- Invariant to class imbalance
- Widely used and well-understood

**Multi-class Extension:**
For multi-class problems, Spectral Predict uses macro-averaged ROC-AUC (One-vs-Rest with equal weight per class), which treats all classes equally regardless of frequency.

---

### 23.7 Balanced Accuracy

**Formula:**

$$Balanced\ Accuracy = \frac{1}{K}\sum_{k=1}^{K} Recall_k$$

Where:
- $K$ = number of classes
- $Recall_k$ = recall for class $k$

**Interpretation:**
- Average recall across all classes
- Treats all classes equally regardless of frequency
- Range: 0 to 1

**Comparison with Standard Accuracy:**

| Scenario | Standard Accuracy | Balanced Accuracy |
|----------|-------------------|-------------------|
| Balanced classes | Same | Same |
| Imbalanced (90/10) | Can be misleading | More informative |
| Minority class errors | Less sensitive | Equally weighted |

**When to Use:**
Balanced accuracy should be preferred when:
- Classes have unequal frequencies
- Minority class performance is important
- All classes should contribute equally to evaluation

---

### 23.8 Confusion Matrix

A confusion matrix is a tabular summary of prediction results, showing the count of true vs. predicted class labels.

**Binary Classification Example:**

|  | Predicted Positive | Predicted Negative |
|--|--------------------|--------------------|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

**Multi-class Example (3 classes: A, B, C):**

|  | Predicted A | Predicted B | Predicted C |
|--|-------------|-------------|-------------|
| **Actual A** | 45 | 3 | 2 |
| **Actual B** | 5 | 38 | 7 |
| **Actual C** | 1 | 4 | 45 |

**Reading the Matrix:**
- Diagonal elements: correct predictions
- Off-diagonal elements: misclassifications
- Row sums: actual class frequencies
- Column sums: predicted class frequencies

**Key Insights from Confusion Matrix:**
1. **High diagonal values**: Good classification performance
2. **Specific off-diagonal patterns**: Reveals which classes are confused
3. **Row-wise analysis**: Reveals recall for each class
4. **Column-wise analysis**: Reveals precision for each class

**Derived Metrics (from confusion matrix):**
- Accuracy = (sum of diagonal) / (total samples)
- Per-class Precision = diagonal[k] / column_sum[k]
- Per-class Recall = diagonal[k] / row_sum[k]

---

### 23.9 Per-Class Metrics

For multi-class problems, metrics can be computed for each class individually:

**Per-Class Precision:**

$$Precision_k = \frac{TP_k}{TP_k + FP_k}$$

**Per-Class Recall:**

$$Recall_k = \frac{TP_k}{TP_k + FN_k}$$

**Per-Class F1:**

$$F1_k = \frac{2 \cdot Precision_k \cdot Recall_k}{Precision_k + Recall_k}$$

**Example Report:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Class A | 0.90 | 0.93 | 0.91 | 50 |
| Class B | 0.85 | 0.76 | 0.80 | 50 |
| Class C | 0.92 | 0.90 | 0.91 | 50 |
| **Macro avg** | 0.89 | 0.86 | 0.87 | 150 |
| **Weighted avg** | 0.89 | 0.86 | 0.87 | 150 |

---

### 23.10 Classification Metric Summary Table

| Metric | Formula | Range | Optimal | Handles Imbalance |
|--------|---------|-------|---------|-------------------|
| Accuracy | $\frac{TP+TN}{Total}$ | [0, 1] | Higher | No |
| Precision | $\frac{TP}{TP+FP}$ | [0, 1] | Higher | Per-class |
| Recall | $\frac{TP}{TP+FN}$ | [0, 1] | Higher | Per-class |
| Specificity | $\frac{TN}{TN+FP}$ | [0, 1] | Higher | Per-class |
| F1 Score | $\frac{2 \cdot P \cdot R}{P+R}$ | [0, 1] | Higher | Depends on averaging |
| ROC-AUC | Area under ROC curve | [0, 1] | Higher | Yes (threshold-independent) |
| Balanced Accuracy | $\frac{1}{K}\sum Recall_k$ | [0, 1] | Higher | Yes |

---

## Chapter 24: Model Quality Indicators

Beyond individual metrics, Spectral Predict provides composite scores and quality indicators to assist with model selection and interpretation.

### 24.1 Composite Score

The composite score combines multiple metrics into a single ranking value, allowing automated model comparison.

**Default Formula (Regression, no penalties):**

$$CompositeScore = -R^2_{cv}$$

When penalties are enabled:

$$CompositeScore = PerformanceScore + VariablePenalty + ComplexityPenalty$$

Where:
- **PerformanceScore** = $0.5 \times z(RMSE_{cv}) - 0.5 \times z(R^2_{cv})$
- $z()$ denotes z-score normalization

**Lower composite scores indicate better models.**

**Classification Formula:**

$$CompositeScore = -z(Accuracy_{cv}) - 0.3 \times z(F1_{cv})$$

**Interpretation:**
- Primary ranking by cross-validated performance metrics
- Secondary consideration of variable count and model complexity
- Configurable penalty weights (0-10 scale)

---

### 24.2 Variable Count Penalty

**Purpose:** Penalizes models that use many wavelengths, encouraging parsimony.

**Formula:**

$$VariablePenalty = \left(\frac{penalty}{10}\right)^2 \times \frac{n_{vars}}{n_{full}}$$

Where:
- $penalty$ = user-specified penalty (0-10 scale)
- $n_{vars}$ = number of variables used by the model
- $n_{full}$ = total number of available variables

**Effect:**
| Penalty Setting | Effect on Ranking |
|-----------------|-------------------|
| 0 (default) | Variable count ignored |
| 2-3 | Slight preference for fewer variables |
| 5 | Moderate preference for fewer variables |
| 8-10 | Strong preference for parsimonious models |

**Use Cases:**
- Model interpretability (fewer variables = easier to explain)
- Identifying key spectral regions
- Reducing measurement requirements

---

### 24.3 Complexity Penalty

**Purpose:** Penalizes complex models, favoring simpler approaches.

**Formula:**

$$ComplexityPenalty = \left(\frac{penalty}{10}\right)^2 \times ComplexityFraction$$

**Complexity Scores by Model:**

| Model | Base Complexity | Notes |
|-------|-----------------|-------|
| PLS | $\frac{LVs}{25}$ | Based on number of latent variables |
| Ridge | 0.25 | Low complexity |
| Lasso | 0.30 | Low complexity |
| ElasticNet | 0.28 | Low complexity |
| KNN | 0.45 | Medium complexity |
| SVM | 0.55 | Medium complexity |
| RandomForest | 0.60 | Medium-high complexity |
| LightGBM | 0.63 | Medium-high complexity |
| CatBoost | 0.64 | Medium-high complexity |
| XGBoost | 0.65 | Medium-high complexity |
| MLP | 0.80 | High complexity |
| NeuralBoosted | 0.85 | High complexity |

---

### 24.4 Unified Complexity Score

**Formula:**

$$ComplexityScore = 0.25 \times Model + 0.30 \times Variables + 0.25 \times LVs + 0.20 \times Preprocessing$$

**Components (0-100 scale):**

1. **Model Type (25%):** Intrinsic algorithm complexity
2. **Variables (30%):** $\min(100, \sqrt{n_{vars}} \times 4.5)$ - nonlinear penalty
3. **Latent Variables (25%):** $\min(100, \frac{LVs - 2}{23} \times 100)$ for PLS; 50 for non-PLS
4. **Preprocessing (20%):** Based on derivative order and SNV application

**Interpretation:**
| Score | Complexity Level |
|-------|------------------|
| 0-25 | Low (simple PLS, few variables) |
| 25-50 | Medium-low |
| 50-75 | Medium-high |
| 75-100 | High (MLP, many variables, complex preprocessing) |

---

### 24.5 Generalization Gap

As introduced in Section 22.8, the generalization gap indicates potential overfitting:

$$Gap = R^2_{cal} - R^2_{cv}$$

**Quality Thresholds:**

| Gap | Generalization Quality | Recommended Action |
|-----|------------------------|-------------------|
| < 0.02 | Excellent | Model is well-generalized |
| 0.02-0.05 | Good | Acceptable for production |
| 0.05-0.10 | Concerning | Consider simplifying model |
| > 0.10 | Poor | Model is likely overfitting |

---

## Chapter 25: Supported File Formats

Spectral Predict supports a wide variety of spectral file formats from different instrument manufacturers and data standards. This chapter provides detailed specifications for each supported format.

### 25.1 CSV (Comma-Separated Values)

**Extensions:** `.csv`

**Structure:**

**Wide Format (Preferred):**
```
sample_id,350.0,351.0,352.0,...,2500.0
sample001,0.245,0.248,0.251,...,0.156
sample002,0.312,0.315,0.318,...,0.201
```

- First column: Sample identifier
- Remaining columns: Wavelengths as numeric headers
- Each row: One spectrum

**Long Format:**
```
wavelength,value
350.0,0.245
351.0,0.248
...
```

- Requires `wavelength` and `value` columns
- Single spectrum per file
- Filename becomes sample ID

**Combined Format:**
```
specimen_id,400.0,401.0,...,2400.0,collagen
A-53,0.245,0.248,...,0.156,6.4
B-12,0.312,0.315,...,0.201,7.9
```

- Includes both spectra and target variable
- Supports optional metadata columns
- Specimen ID column optional (auto-generated if absent)

**Requirements:**
- Minimum 100 wavelength columns
- Wavelengths must be strictly increasing
- Numeric spectral values

**Metadata Extracted:**
- Number of spectra
- Wavelength range
- Data type (auto-detected)

---

### 25.2 Excel (.xlsx, .xls)

**Extensions:** `.xlsx`, `.xls`

**Structure:** Same formats as CSV (wide, long, or combined)

**Notes:**
- Uses first sheet by default
- Same column requirements as CSV
- Supports formulas (evaluated values used)

**Limitations:**
- Large files may load slowly
- Complex formatting ignored
- Multiple sheets not supported automatically

---

### 25.3 ASD (Analytical Spectral Devices)

**Extensions:** `.asd`, `.sig`

**Description:** Native format for ASD FieldSpec and related portable spectrometers, commonly used for vegetation, soil, and mineral analysis.

**ASCII Format (.sig or ASCII .asd):**
```
Wavelength  Reflectance  Reference
350.0       0.245        12500
351.0       0.248        12480
...
```

**Binary Format (.asd):**
- Contains binary header with metadata
- Spectral data in binary format
- Requires SpecDAL library for reading

**Typical Wavelength Range:** 350-2500 nm

**Data Content:**
- Reflectance values (typically 0-1 or percentage)
- Optional: Reference spectrum, white reference

**Metadata Available:**
- Instrument serial number
- GPS coordinates
- Date/time
- Integration time
- Target/panel type

**Installation Note:**
For binary ASD files, install SpecDAL:
```bash
pip install specdal
```

---

### 25.4 SPC (Thermo/Galactic GRAMS)

**Extensions:** `.spc`

**Description:** Standard format for GRAMS spectroscopy software, widely used in FT-IR, Raman, and UV-Vis spectroscopy.

**Structure:**
- Binary file format
- Can contain multiple spectra (subfiles)
- Extensive metadata support

**Data Organization:**
- X-axis: Wavelength, wavenumber, or other units
- Y-axis: Absorbance, transmittance, or intensity
- Multiple subfiles: Time series, depth profiles, etc.

**Metadata Available:**
- Sample description
- Instrument parameters
- Collection date/time
- Experiment details
- Custom fields

**Installation Note:**
Requires spc-io library:
```bash
pip install spc-io
```

---

### 25.5 JCAMP-DX (.jdx, .dx, .jcm)

**Extensions:** `.jdx`, `.dx`, `.jcm`

**Description:** IUPAC/JCAMP standard text-based exchange format for spectroscopic data. Designed for data interchange between different software systems.

**Structure:**
```
##TITLE=Sample Name
##JCAMP-DX=5.00
##DATA TYPE=INFRARED SPECTRUM
##XUNITS=1/CM
##YUNITS=ABSORBANCE
##FIRSTX=4000.0
##LASTX=400.0
##NPOINTS=3601
##XYDATA=(X++(Y..Y))
4000.0 0.0234 0.0235 0.0237
3999.0 0.0238 0.0240 0.0242
...
##END=
```

**Header Fields:**
- `##TITLE=`: Sample name
- `##JCAMP-DX=`: Format version
- `##DATA TYPE=`: Spectrum type
- `##XUNITS=`: X-axis units (1/CM, NANOMETERS, etc.)
- `##YUNITS=`: Y-axis units (ABSORBANCE, TRANSMITTANCE, etc.)
- `##FIRSTX=`, `##LASTX=`: X-axis range
- `##NPOINTS=`: Number of data points
- `##XFACTOR=`, `##YFACTOR=`: Scaling factors

**Data Formats:**
- `(X++(Y..Y))`: Fixed X increment, Y values only
- `(XY..XY)`: X,Y pairs
- Compressed formats available

**Metadata Extracted:**
- Title, units, date
- Instrument information
- Sample details
- Custom fields

**Installation Note:**
Requires jcamp library:
```bash
pip install jcamp
```

---

### 25.6 Bruker OPUS

**Extensions:** `.0`, `.1`, `.2`, ... `.999` (numbered extensions)

**Description:** Proprietary binary format used by Bruker FT-IR instruments. One of the most common formats in infrared spectroscopy.

**Data Types Available:**
| Code | Data Type |
|------|-----------|
| `a` | Absorbance spectrum |
| `t` | Transmittance spectrum |
| `sm` | Sample spectrum (single beam) |
| `rf` | Reference spectrum |

**Priority Order:** Spectral Predict reads absorbance > transmittance > sample > reference

**Typical Wavenumber Range:** 400-4000 cm-1 (mid-IR) or 4000-12000 cm-1 (NIR)

**Metadata Available:**
- Sample name
- Sample form
- Instrument parameters
- Acquisition date/time

**Installation Note:**
Requires brukeropus library:
```bash
pip install brukeropus
```

**Wavelength Conversion:**
OPUS files store wavenumbers (cm-1). To convert to wavelength:
$$\lambda (nm) = \frac{10^7}{\tilde{\nu} (cm^{-1})}$$

---

### 25.7 PerkinElmer (.sp)

**Extensions:** `.sp`

**Description:** Binary format from PerkinElmer infrared spectrometers (Spectrum series).

**Data Content:**
- Absorbance or transmittance spectra
- Wavenumber or wavelength axis
- Single spectrum per file

**Typical Range:** 400-4000 cm-1 (mid-IR)

**Metadata Available:**
- Sample information
- Instrument settings
- Acquisition parameters

**Installation Note:**
Requires specio library (Python 3.10+ compatible version):
```bash
pip install specio-py310
```

---

### 25.8 Agilent Formats

**Extensions:** `.seq`, `.dmt`, `.asp`, `.bsw`

**Description:** Formats from Agilent infrared instruments, including hyperspectral imaging data.

**Format Types:**

| Extension | Description |
|-----------|-------------|
| `.seq` | Single-tile hyperspectral image |
| `.dmt` | Multi-tile mosaic hyperspectral image |
| `.asp` | Single spectrum file |
| `.bsw` | Batch file (limited support) |

**Hyperspectral Data:**
- 3D data cube (height x width x spectral points)
- Multiple pixels per sample
- Extraction modes: total (sum), mean, first pixel

**Installation Note:**
Requires agilent-ir-formats library:
```bash
pip install agilent-ir-formats
```

---

### 25.9 ASCII Text Variants

**Extensions:** `.txt`, `.dat`, `.dpt`, `.asc`

**Description:** Generic text-based spectral files with various delimiters.

**Supported Delimiters:**
- Tab (`\t`)
- Space
- Comma (`,`)
- Semicolon (`;`)

**Format Detection:**
- Automatically detects delimiter
- Skips comment lines (starting with `#` or `%`)
- Identifies X,Y pair columns

**Example (.dpt - Bruker data point table):**
```
4000.0	0.0234
3999.5	0.0235
3999.0	0.0237
...
```

---

### 25.10 File Format Summary Table

| Format | Extension(s) | Type | Library Required | Typical Application |
|--------|--------------|------|------------------|---------------------|
| CSV | .csv | Text | None | Universal exchange |
| Excel | .xlsx, .xls | Binary | openpyxl/xlrd | Universal exchange |
| ASD | .asd, .sig | Binary/Text | specdal (binary) | VIS-NIR field spectrometers |
| SPC | .spc | Binary | spc-io | FT-IR, Raman, UV-Vis |
| JCAMP-DX | .jdx, .dx, .jcm | Text | jcamp | Standard exchange |
| OPUS | .0-.999 | Binary | brukeropus | Bruker FT-IR |
| PerkinElmer | .sp | Binary | specio | PerkinElmer IR |
| Agilent | .seq, .dmt, .asp | Binary | agilent-ir-formats | Agilent IR/imaging |
| ASCII | .txt, .dat, .dpt, .asc | Text | None | Generic text spectra |

---

## Chapter 26: Glossary of Terms

This comprehensive glossary defines all technical terms used throughout Spectral Predict and in the broader fields of spectroscopy and chemometrics. Terms are arranged alphabetically.

---

### A

**Absorbance**
A measure of light absorbed by a sample, calculated as $A = -\log_{10}(T)$ where T is transmittance. Absorbance is additive with concentration (Beer-Lambert Law), making it preferred for quantitative analysis. Typical range: 0-3 absorbance units (AU).

**Accuracy**
The proportion of correct predictions among all predictions. For classification: $Accuracy = \frac{Correct}{Total}$.

**Alpha**
Regularization parameter in Ridge, Lasso, and ElasticNet regression. Higher alpha increases regularization strength, shrinking coefficients toward zero.

**ASD (Analytical Spectral Devices)**
Manufacturer of portable VIS-NIR spectrometers (FieldSpec series). Also refers to their proprietary file format (.asd).

---

### B

**Baseline Correction**
Preprocessing technique to remove spectral baseline drift caused by scattering or instrumental effects. Common methods include polynomial fitting, asymmetric least squares, and rubber band correction.

**Bias**
Systematic error in predictions, calculated as the mean difference between actual and predicted values. Non-zero bias indicates the model consistently over- or under-predicts.

**Binary Classification**
Classification problem with exactly two classes (e.g., pass/fail, present/absent).

---

### C

**Calibration**
The process of building a mathematical relationship between spectra (X) and reference values (y). Also refers to the training dataset used for this purpose.

**CARS (Competitive Adaptive Reweighted Sampling)**
Variable selection algorithm that uses adaptive reweighted sampling to identify important wavelengths. Combines Monte Carlo sampling with PLS modeling.

**CatBoost**
Gradient boosting algorithm developed by Yandex featuring ordered boosting and native handling of categorical features.

**Chemometrics**
The science of extracting information from chemical systems using mathematical and statistical methods. Encompasses multivariate analysis, experimental design, and signal processing.

**Classification**
Supervised learning task where the target variable is categorical (discrete classes) rather than continuous.

**Composite Score**
Combined metric used for ranking models in Spectral Predict. Lower scores indicate better performance.

**Confusion Matrix**
Table showing the distribution of actual vs. predicted class labels, allowing detailed error analysis in classification.

**Cross-Validation (CV)**
Model validation technique that partitions data into complementary subsets, training on some and testing on others. K-fold CV uses K partitions; leave-one-out CV tests each sample individually.

---

### D

**Derivative (Spectral)**
Mathematical transformation of spectra using differentiation. First derivative removes baseline offset; second derivative removes both offset and linear baseline drift. Implemented via Savitzky-Golay filtering.

**DS (Direct Standardization)**
Calibration transfer method that computes a transformation matrix to convert spectra from a secondary instrument to match a primary instrument.

---

### E

**ElasticNet**
Regularized regression combining L1 (Lasso) and L2 (Ridge) penalties. The l1_ratio parameter controls the balance between the two.

**Ensemble**
Method combining multiple models to improve prediction accuracy. Types include averaging, stacking, and region-aware weighted combinations.

---

### F

**F1 Score**
Harmonic mean of precision and recall: $F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$. Balances both metrics in a single score.

**False Negative (FN)**
Prediction error where a positive class sample is incorrectly classified as negative.

**False Positive (FP)**
Prediction error where a negative class sample is incorrectly classified as positive.

**Feature**
Input variable to a model. In spectroscopy, features are typically wavelengths or wavenumbers.

**Feature Importance**
Measure of how much each feature (wavelength) contributes to model predictions. Methods include VIP scores (PLS), coefficient magnitude (linear models), and gain-based importance (tree models).

**Feature Selection**
See Variable Selection.

**FT-IR (Fourier Transform Infrared)**
Infrared spectroscopy technique using interferometry and Fourier transformation. Produces spectra with high resolution and signal-to-noise ratio.

---

### G

**GA-PLS (Genetic Algorithm PLS)**
Variable selection method using genetic algorithms to evolve optimal wavelength subsets for PLS modeling.

**Generalization**
A model's ability to perform well on new, unseen data (not used during training).

**Gradient Boosting**
Ensemble method that builds models sequentially, with each new model correcting errors from previous ones. Examples: XGBoost, LightGBM, CatBoost.

---

### H

**Hyperparameter**
Model configuration parameter set before training (not learned from data). Examples: number of PLS components, regularization strength, learning rate.

**Hyperspectral**
Imaging technique capturing hundreds of spectral bands per pixel, creating a 3D data cube (spatial x spatial x spectral).

---

### I

**iPLS (Interval PLS)**
Variable selection method that divides the spectrum into intervals and identifies which intervals contribute most to prediction.

---

### J

**JCAMP-DX**
IUPAC standard text format for spectroscopic data exchange. Human-readable with embedded metadata.

---

### K

**K-Fold Cross-Validation**
CV method dividing data into K equal parts (folds). Each fold serves once as validation set while others train the model. Final performance is averaged across folds.

**KNN (K-Nearest Neighbors)**
Classification/regression algorithm predicting based on the K most similar training samples.

---

### L

**L1 Regularization**
Penalty term equal to the sum of absolute coefficient values: $\lambda \sum |w_i|$. Promotes sparsity (drives some coefficients to exactly zero). Used in Lasso regression.

**L2 Regularization**
Penalty term equal to the sum of squared coefficient values: $\lambda \sum w_i^2$. Shrinks all coefficients toward zero without eliminating any. Used in Ridge regression.

**Lasso (Least Absolute Shrinkage and Selection Operator)**
Linear regression with L1 regularization. Performs automatic feature selection by setting some coefficients to zero.

**Latent Variable (LV)**
Unobserved variable derived from measured variables. In PLS, latent variables capture the maximum covariance between X and y.

**Leave-One-Out Cross-Validation (LOOCV)**
CV method where each sample is used once as the validation set. Equivalent to K-fold CV where K equals the number of samples.

**LightGBM**
Microsoft's gradient boosting implementation using histogram-based learning for speed and efficiency.

**Loading**
In PLS/PCA, the coefficients relating original variables to latent variables. Loadings reveal which wavelengths contribute most to each component.

---

### M

**MAE (Mean Absolute Error)**
Average absolute difference between predicted and actual values: $MAE = \frac{1}{n}\sum |y_i - \hat{y}_i|$.

**MLP (Multi-Layer Perceptron)**
Feed-forward neural network with one or more hidden layers. Universal function approximator capable of learning complex patterns.

**MSC (Multiplicative Scatter Correction)**
Preprocessing technique to correct for scatter effects in spectra. Transforms each spectrum to minimize differences from a reference spectrum (usually the mean).

**Multicollinearity**
High correlation among predictor variables. Common in spectral data where adjacent wavelengths are highly correlated.

**Multi-class Classification**
Classification problem with more than two classes (e.g., species identification with 5 species).

---

### N

**NIR (Near-Infrared)**
Spectral region from approximately 700-2500 nm. NIR spectroscopy is widely used for non-destructive analysis of organic compounds.

**NeuralBoosted**
Gradient boosting algorithm using small neural networks as weak learners. Provides interpretable non-linearity.

**Normalization**
See SNV (Standard Normal Variate) or StandardScaler.

---

### O

**OPUS**
Bruker's proprietary spectroscopy software and file format for FT-IR instruments.

**Overfitting**
When a model learns noise in training data rather than true patterns, resulting in poor generalization to new data. Indicated by large gap between calibration and validation performance.

---

### P

**PCA (Principal Component Analysis)**
Unsupervised dimensionality reduction technique finding orthogonal directions of maximum variance. Used for visualization, outlier detection, and preprocessing.

**PDS (Piecewise Direct Standardization)**
Calibration transfer method using local transformations in wavelength windows rather than a single global transformation.

**PLS (Partial Least Squares)**
Supervised method finding latent variables that maximize covariance between predictors and response. Standard algorithm for spectroscopic calibration.

**PLS-DA (PLS Discriminant Analysis)**
PLS adapted for classification by encoding class labels as binary dummy variables.

**Polynomial Order**
In Savitzky-Golay filtering, the degree of polynomial fitted to local data windows. Higher orders preserve more spectral features but may introduce artifacts.

**Precision**
Proportion of positive predictions that are correct: $Precision = \frac{TP}{TP + FP}$.

**Preprocessing**
Spectral transformations applied before modeling to reduce noise, remove baseline effects, or normalize spectra. Common methods: SNV, derivatives, smoothing.

**Prediction**
Using a trained model to estimate target values for new samples.

---

### Q

**Quantile**
Value dividing a distribution into equal probability intervals. Q1 (25th percentile), Q2/median (50th), Q3 (75th) divide data into quarters.

---

### R

**R-squared (R2, Coefficient of Determination)**
Proportion of variance explained by the model: $R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$. Range typically 0-1.

**Random Forest**
Ensemble of decision trees trained on bootstrap samples with random feature subsets. Robust and interpretable.

**Recall (Sensitivity)**
Proportion of actual positives correctly identified: $Recall = \frac{TP}{TP + FN}$.

**Reflectance**
Ratio of reflected to incident light intensity. Range: 0-1 (or 0-100%). Reflectance spectra show absorption features as dips/valleys.

**Regression**
Supervised learning task predicting continuous target values (e.g., concentration, temperature).

**Regularization**
Technique to prevent overfitting by adding penalty terms to the optimization objective. Types: L1 (Lasso), L2 (Ridge), ElasticNet (combined).

**Ridge Regression**
Linear regression with L2 regularization. Shrinks all coefficients toward zero but doesn't eliminate any.

**RMSE (Root Mean Square Error)**
Square root of mean squared prediction error: $RMSE = \sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$.

**RMSEC, RMSEcv, RMSEP**
RMSE calculated on calibration, cross-validation, and prediction (external test) sets, respectively.

**ROC (Receiver Operating Characteristic)**
Curve plotting True Positive Rate vs. False Positive Rate across classification thresholds.

**ROC-AUC**
Area Under the ROC Curve. Measures classifier's ability to rank positive instances higher than negative ones.

**RPD (Ratio of Performance to Deviation)**
Ratio of reference standard deviation to RMSEP: $RPD = \frac{SD_y}{RMSEP}$. Measures model utility on a standardized scale.

---

### S

**Savitzky-Golay Filter**
Smoothing/differentiation method fitting local polynomials to data windows. Preserves spectral features better than simple moving averages.

**Score (PLS/PCA)**
The coordinates of samples in the latent variable space. Scores reveal sample similarities and groupings.

**Sensitivity**
See Recall.

**SNV (Standard Normal Variate)**
Preprocessing transformation that centers and scales each spectrum individually: $x_{snv} = \frac{x - \bar{x}}{s_x}$. Corrects for path length and scatter variations.

**SPA (Successive Projections Algorithm)**
Variable selection method that iteratively selects wavelengths with minimum collinearity.

**SPC**
Thermo/Galactic GRAMS spectral file format.

**Specificity**
Proportion of actual negatives correctly identified: $Specificity = \frac{TN}{TN + FP}$.

**StandardScaler**
Scikit-learn transformer that centers and scales features to zero mean and unit variance across samples (column-wise). Different from SNV which operates row-wise.

**Stratified Cross-Validation**
CV method ensuring each fold has approximately the same class distribution as the full dataset. Essential for imbalanced classification.

**Supervised Learning**
Machine learning using labeled data (known target values) to train models.

**Support Vector Machine (SVM)**
Algorithm finding optimal hyperplane to separate classes or fit regression. Uses kernel functions for non-linear problems.

**SVR (Support Vector Regression)**
SVM adapted for regression tasks.

---

### T

**Test Set**
Data reserved exclusively for final model evaluation, never used during training or hyperparameter tuning.

**Tier**
In Spectral Predict, a configuration level (Quick, Standard, Comprehensive) balancing analysis thoroughness with computation time.

**Training Set**
Data used to fit model parameters.

**Transmittance**
Ratio of transmitted to incident light intensity: $T = \frac{I}{I_0}$. Related to absorbance: $A = -\log_{10}(T)$.

**True Negative (TN)**
Correct prediction of negative class.

**True Positive (TP)**
Correct prediction of positive class.

---

### U

**Underfitting**
When a model is too simple to capture underlying patterns, resulting in poor performance on both training and test data.

**UVE (Uninformative Variable Elimination)**
Variable selection method comparing real variable importance to artificial noise variables.

---

### V

**Validation**
Process of assessing model performance on data not used for training. Internal validation uses CV; external validation uses independent test sets.

**Variable Selection**
Methods to identify and retain only the most informative wavelengths for modeling. Reduces noise, improves interpretability, and can improve predictions.

**Variance Explained**
Proportion of total variance captured by a model or component. In PCA, cumulative variance explained indicates how many components are needed.

**VIP (Variable Importance in Projection)**
PLS-based measure of each variable's contribution to the model. VIP > 1 indicates above-average importance.

**VIS (Visible)**
Spectral region from approximately 400-700 nm.

**VIS-NIR**
Combined visible and near-infrared spectral region (350-2500 nm), common range for ASD instruments.

---

### W

**Wavenumber**
Reciprocal of wavelength, typically in cm-1: $\tilde{\nu} = \frac{10^7}{\lambda(nm)}$. Preferred unit in IR spectroscopy as it is proportional to energy.

**Wavelength**
Distance between wave peaks, typically in nm or um for optical spectroscopy.

**Weighted Average**
Average where each component has a different weight. In ensemble methods, weights reflect model confidence or regional expertise.

**Window Size**
In Savitzky-Golay filtering, the number of points in the local fitting window. Larger windows smooth more but may blur features.

---

### X

**XGBoost (Extreme Gradient Boosting)**
Highly optimized gradient boosting implementation with regularization and advanced features. Industry standard for structured data.

---

### Z

**Z-score**
Standardized value: $z = \frac{x - \mu}{\sigma}$. Measures how many standard deviations a value is from the mean.

---

## Chapter 27: Quick Reference Tables

### 27.1 Regression Metrics Quick Reference

| Metric | Formula | Optimal | Interpretation |
|--------|---------|---------|----------------|
| RMSE | $\sqrt{\frac{1}{n}\sum(y-\hat{y})^2}$ | Lower | Typical prediction error (same units as y) |
| R-squared | $1 - \frac{SS_{res}}{SS_{tot}}$ | Higher (max 1) | Proportion of variance explained |
| MAE | $\frac{1}{n}\sum|y-\hat{y}|$ | Lower | Average absolute error |
| RPD | $\frac{SD_y}{RMSEP}$ | Higher (>3 good) | Model utility ratio |
| Bias | $\bar{y} - \bar{\hat{y}}$ | 0 | Systematic prediction error |

### 27.2 Classification Metrics Quick Reference

| Metric | Formula | Optimal | Handles Imbalance |
|--------|---------|---------|-------------------|
| Accuracy | $\frac{TP+TN}{Total}$ | Higher (max 1) | No |
| Precision | $\frac{TP}{TP+FP}$ | Higher (max 1) | Per-class |
| Recall | $\frac{TP}{TP+FN}$ | Higher (max 1) | Per-class |
| F1 Score | $\frac{2 \cdot P \cdot R}{P+R}$ | Higher (max 1) | Depends |
| ROC-AUC | Area under ROC | Higher (max 1) | Yes |
| Balanced Accuracy | $\frac{1}{K}\sum Recall_k$ | Higher (max 1) | Yes |

### 27.3 File Format Quick Reference

| Format | Extensions | Type | Typical Application |
|--------|------------|------|---------------------|
| CSV | .csv | Text | Universal exchange |
| Excel | .xlsx, .xls | Binary | Universal exchange |
| ASD | .asd, .sig | Binary/Text | Field VIS-NIR |
| SPC | .spc | Binary | FT-IR, Raman |
| JCAMP-DX | .jdx, .dx | Text | Standard exchange |
| OPUS | .0-.999 | Binary | Bruker FT-IR |
| PerkinElmer | .sp | Binary | PerkinElmer IR |
| Agilent | .seq, .dmt | Binary | Agilent imaging |

### 27.4 Model Complexity Reference

| Model | Complexity | Requires Scaling | Feature Importance Method |
|-------|------------|------------------|---------------------------|
| PLS | Low | No (built-in) | VIP scores |
| Ridge | Low | Yes | Coefficient magnitude |
| Lasso | Low | Yes | Coefficient magnitude |
| ElasticNet | Low | Yes | Coefficient magnitude |
| Random Forest | Medium | No | Gini importance |
| XGBoost | Medium | No | Gain-based |
| LightGBM | Medium | No | Split-based |
| CatBoost | Medium | No | PredictionValuesChange |
| SVR | Medium | Yes | Kernel-dependent |
| MLP | High | Yes | First-layer weights |
| NeuralBoosted | High | Yes | Aggregated weights |

### 27.5 Preprocessing Reference

| Method | Effect | Use When |
|--------|--------|----------|
| Raw | None | Baseline for comparison |
| SNV | Corrects scatter, normalizes | Path length variations |
| 1st Derivative | Removes offset, enhances peaks | Baseline shift |
| 2nd Derivative | Removes linear baseline | Sloping baseline |
| SNV + Derivative | Combined correction | Multiple issues |
| Smoothing (SG) | Reduces noise | Noisy spectra |

### 27.6 RPD Interpretation Reference

| RPD | Quality | Application |
|-----|---------|-------------|
| < 1.5 | Very poor | Not suitable |
| 1.5-2.0 | Poor | Rough screening only |
| 2.0-2.5 | Acceptable | Screening |
| 2.5-3.0 | Good | Screening, approximate quant |
| 3.0-5.0 | Very good | Quality control |
| 5.0-8.0 | Excellent | Process control |
| > 8.0 | Outstanding | Reference method quality |

---

## Chapter 28: Common Abbreviations

| Abbreviation | Full Term |
|--------------|-----------|
| ASD | Analytical Spectral Devices |
| AUC | Area Under Curve |
| CARS | Competitive Adaptive Reweighted Sampling |
| CV | Cross-Validation |
| DS | Direct Standardization |
| FN | False Negative |
| FP | False Positive |
| FT-IR | Fourier Transform Infrared |
| GA-PLS | Genetic Algorithm PLS |
| iPLS | Interval PLS |
| LV | Latent Variable |
| MAE | Mean Absolute Error |
| MLP | Multi-Layer Perceptron |
| MSC | Multiplicative Scatter Correction |
| NIR | Near-Infrared |
| OPUS | Bruker software/format |
| PCA | Principal Component Analysis |
| PDS | Piecewise Direct Standardization |
| PLS | Partial Least Squares |
| PLS-DA | PLS Discriminant Analysis |
| RMSE | Root Mean Square Error |
| RMSEC | RMSE Calibration |
| RMSEcv | RMSE Cross-Validation |
| RMSEP | RMSE Prediction |
| ROC | Receiver Operating Characteristic |
| RPD | Ratio of Performance to Deviation |
| SG | Savitzky-Golay |
| SNV | Standard Normal Variate |
| SPA | Successive Projections Algorithm |
| SPC | Thermo/Galactic format |
| SVM | Support Vector Machine |
| SVR | Support Vector Regression |
| TN | True Negative |
| TP | True Positive |
| UVE | Uninformative Variable Elimination |
| VIP | Variable Importance in Projection |
| VIS | Visible |
| XGBoost | Extreme Gradient Boosting |

---

*This reference section is designed to be used alongside the main user guide. For practical tutorials and step-by-step instructions, refer to Parts I through VII. For the most current information on supported file formats and metrics, consult the online documentation.*
