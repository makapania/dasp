# Part I: Getting Started

## Chapter 1: Introduction to Spectral Predict

### 1.1 What is Spectral Predict?

Spectral Predict is an automated spectral modeling application designed to streamline the development of quantitative and qualitative calibration models from spectroscopic data. The software combines advanced chemometric algorithms with an intuitive graphical interface, enabling researchers and practitioners to rapidly evaluate multiple preprocessing and modeling approaches to identify optimal spectral calibrations.

#### Target Applications

Spectral Predict is designed for professionals working in:

- **Near-Infrared (NIR) Spectroscopy**: Agricultural analysis, food composition, pharmaceutical quality control
- **Mid-Infrared (MIR) Spectroscopy**: Chemical identification, material characterization
- **Process Analytical Technology (PAT)**: Real-time process monitoring and control
- **Quality Control**: Routine analysis with established calibration models
- **Research and Development**: Method development and optimization

#### Core Capabilities

The software provides comprehensive functionality across the spectral modeling workflow:

**Data Import and Management**
- Support for 10+ spectral file formats (ASD, SPC, OPUS, JCAMP-DX, CSV, Excel, and more)
- Automatic data type detection (reflectance vs. absorbance)
- Flexible ID matching between spectra and reference data
- Wavelength range selection and configuration

**Preprocessing Pipeline**
- Standard Normal Variate (SNV) normalization
- Savitzky-Golay derivatives (1st and 2nd order)
- Multiple smoothing window sizes
- Baseline correction methods
- Automatic preprocessing combination search

**Modeling Algorithms**
- Partial Least Squares (PLS) regression and discriminant analysis (PLS-DA)
- Ridge, Lasso, and ElasticNet regularized regression
- Random Forest, Gradient Boosting (XGBoost, LightGBM, CatBoost)
- Support Vector Machines (SVM/SVR)
- Multi-Layer Perceptron (MLP) neural networks

**Variable Selection**
- Uninformative Variable Elimination (UVE)
- Successive Projections Algorithm (SPA)
- Interval PLS (iPLS)
- Competitive Adaptive Reweighted Sampling (CARS)
- Genetic Algorithm PLS (GA-PLS)

**Quality Assurance**
- PCA-based outlier detection
- Hotelling T-squared statistics
- Q-residuals (Squared Prediction Error)
- Mahalanobis distance analysis
- Cross-validation with multiple fold configurations

---

### 1.2 System Requirements

#### Minimum Requirements
- Python 3.8 or higher
- 8 GB RAM
- 500 MB disk space
- Windows 10/11, macOS 10.14+, or Linux

#### Recommended Requirements
- Python 3.10 or higher
- 16 GB RAM (for large datasets)
- SSD storage
- Multi-core processor for parallel model training

#### Required Python Packages
```
numpy >= 1.20
pandas >= 1.3
scikit-learn >= 1.0
scipy >= 1.7
matplotlib >= 3.4
tkinter (included with Python)
```

#### Optional Packages (for specific file formats)
```
spc-io          # SPC file support
jcamp           # JCAMP-DX file support
brukeropus      # Bruker OPUS file support
specdal         # Binary ASD file support
specio          # PerkinElmer file support
agilent-ir      # Agilent format support
```

---

### 1.3 Installation

#### From PyPI
```bash
pip install spectral-predict
```

#### From Source
```bash
git clone https://github.com/your-repo/spectral-predict.git
cd spectral-predict
pip install -e .
```

#### Launching the Application
```bash
python spectral_predict_gui_optimized.py
```

---

## Chapter 2: Quick Start Tutorial

This tutorial walks through a complete analysis workflow in approximately 10 minutes, demonstrating the core capabilities of Spectral Predict.

### 2.1 Scenario Overview

You have collected NIR spectra from 100 soil samples and want to build a calibration model to predict nitrogen content. Your data consists of:
- A folder containing 100 ASD spectral files (`spectra/`)
- A CSV file with reference nitrogen values (`reference.csv`)

### 2.2 Step-by-Step Workflow

#### Step 1: Load Spectral Data

1. Launch Spectral Predict
2. In the **Import & Preview** tab, click **Browse...** next to "Spectral File Directory"
3. Navigate to and select your `spectra/` folder
4. The status will show: "Detected 100 ASD files"

#### Step 2: Load Reference Data

1. Click **Browse...** next to "Reference CSV/Excel"
2. Select your `reference.csv` file
3. The column dropdowns will populate automatically

#### Step 3: Configure Data Alignment

1. Set **Specimen ID** to the column matching your file names (e.g., "sample_id")
2. Set **Target Variable** to your target column (e.g., "nitrogen")
3. Click **Load Data & Generate Plots**

The system will:
- Read all spectral files
- Match spectra to reference values by ID
- Detect data type (reflectance/absorbance)
- Generate spectral overlay plots

#### Step 4: Review Data Quality (Optional)

1. Switch to the **Data Quality Check** tab
2. Click **Run Outlier Detection**
3. Review the PCA scores plot and outlier summary
4. If outliers are identified, select them and click **Mark Selected for Exclusion**

#### Step 5: Configure and Run Analysis

1. Go to the **Analysis Configuration** tab
2. Select your model tier:
   - **Quick**: Fast evaluation with 3 core models (~2 min)
   - **Standard**: Balanced evaluation with 6 models (~10 min)
   - **Comprehensive**: Full evaluation with all models (~30 min)
3. Click **Run Analysis**

#### Step 6: Review Results

1. Navigate to the **Results** tab
2. Examine the ranked model list sorted by cross-validated performance
3. Click on any model row to view:
   - Predicted vs. Actual plot
   - Residual distribution
   - Performance metrics (R-squared, RMSECV, RPD)

#### Step 7: Export Best Model

1. Select your preferred model from the results table
2. Click **Export Model**
3. Choose export options:
   - Trained model file (`.pkl`)
   - Prediction script
   - Model report (PDF)

### 2.3 Quick Start Summary

```
                    QUICK START WORKFLOW
    +--------------------------------------------------+
    |                                                  |
    |   1. Import Tab      Load spectra + reference    |
    |         |                                        |
    |         v                                        |
    |   2. Quality Tab     Check for outliers          |
    |         |                                        |
    |         v                                        |
    |   3. Config Tab      Select models + settings    |
    |         |                                        |
    |         v                                        |
    |   4. Run Analysis    Automated model search      |
    |         |                                        |
    |         v                                        |
    |   5. Results Tab     Review ranked models        |
    |         |                                        |
    |         v                                        |
    |   6. Export          Save model for deployment   |
    |                                                  |
    +--------------------------------------------------+
```

---

## Chapter 3: The Import & Preview Tab

The Import & Preview tab is the starting point for any analysis in Spectral Predict. This chapter provides comprehensive documentation of all features and options available for loading and configuring spectral data.

### 3.1 Tab Overview

The Import & Preview tab is organized into two subtabs:
- **Data**: File selection, column mapping, and configuration options
- **Plots**: Spectral visualization and exploration

```
+------------------------------------------------------------------+
|  Import & Preview                                                |
|  +--------+--------+                                             |
|  |  Data  | Plots  |                                             |
|  +--------+--------+---------------------------------------------+
|                                                                  |
|  1. INPUT DATA                                                   |
|  +------------------------------------------------------------+  |
|  | Spectral File Directory: [________________________] Browse  |  |
|  | Reference CSV/Excel:     [________________________] Browse  |  |
|  +------------------------------------------------------------+  |
|                                                                  |
|  [Load Data & Generate Plots]  [ ] Append to existing data       |
|                                                                  |
|  2. ADVANCED CONFIGURATION (Optional)                            |
|  +------------------------------------------------------------+  |
|  | Column Mapping:                                             |  |
|  |   Spectral File: [dropdown]    Specimen ID: [dropdown]     |  |
|  |   Target Variable: [dropdown]  [Auto-Detect]               |  |
|  |                                                             |  |
|  | Analysis Type:  ( ) Auto-detect  ( ) Regression  ( ) Class |  |
|  | Wavelength Range: [____] to [____] nm                      |  |
|  | Data Type: Reflectance/Absorbance [Convert to Absorbance]  |  |
|  +------------------------------------------------------------+  |
|                                                                  |
+------------------------------------------------------------------+
```

### 3.2 Loading Spectral Files

#### Spectral File Directory

The primary input field accepts either:
- **A directory path**: Contains multiple spectral files (ASD, SPC, JCAMP, etc.)
- **A single file path**: A combined CSV/Excel file with all spectra

When you select a directory, Spectral Predict automatically:
1. Scans for supported file types
2. Counts the number of spectral files found
3. Displays detection status (e.g., "Found 150 ASD files")

**Supported Input Modes:**

| Mode | Description | Best For |
|------|-------------|----------|
| Directory of files | One file per spectrum | ASD, SPC, OPUS instruments |
| Single combined file | All spectra in one table | Exported datasets, CSV/Excel |

#### Browse Button Behavior

Clicking **Browse...** opens a folder selection dialog. The system then:
1. Checks if the selected path is a directory or file
2. For directories: scans for spectral files matching supported extensions
3. For files: detects combined format (CSV/Excel with embedded spectra + targets)

### 3.3 Supported File Formats

Spectral Predict supports a wide range of spectral file formats through its unified I/O architecture.

#### Native Format Support

| Format | Extensions | Description | Notes |
|--------|------------|-------------|-------|
| **CSV** | `.csv` | Comma-separated values | Wide format (wavelengths as columns) or long format |
| **Excel** | `.xlsx`, `.xls` | Microsoft Excel | Same format options as CSV |
| **ASD** | `.asd`, `.sig` | ASD FieldSpec | ASCII format; binary requires `specdal` package |
| **SPC** | `.spc` | GRAMS/Thermo Galactic | Binary format; requires `spc-io` package |
| **JCAMP-DX** | `.jdx`, `.dx`, `.jcm` | Joint Committee format | Text-based with embedded metadata |
| **ASCII** | `.txt`, `.dat`, `.dpt`, `.asc` | Generic text files | Auto-detects column structure |

#### Extended Format Support (Optional Packages)

| Format | Extensions | Required Package | Description |
|--------|------------|------------------|-------------|
| **Bruker OPUS** | `.0`, `.1`, `.2`, etc. | `brukeropus` | FTIR instrument files |
| **PerkinElmer** | `.sp` | `specio` | PerkinElmer spectrometers |
| **Agilent** | `.seq`, `.dmt`, `.asp`, `.bsw` | `agilent-ir-formats` | Agilent FTIR systems |

#### Format Auto-Detection

The software automatically detects file formats using:
1. **File extension mapping**: Primary detection method
2. **Magic bytes analysis**: For files without standard extensions
3. **Content inspection**: For ambiguous cases

```python
# Format detection priority:
# 1. Extension-based (.csv -> csv, .asd -> asd, etc.)
# 2. Content-based (OPUS magic bytes, JCAMP headers, etc.)
# 3. Directory analysis (scan for predominant file types)
```

### 3.4 CSV and Excel Format Specifications

#### Wide Format (Recommended)

The standard format has samples as rows and wavelengths as columns:

```
sample_id,350,351,352,...,2500,nitrogen
Sample_001,0.245,0.248,0.251,...,0.156,6.4
Sample_002,0.312,0.315,0.318,...,0.201,7.9
Sample_003,0.189,0.192,0.195,...,0.134,5.2
```

**Requirements:**
- First column: Sample identifier
- Wavelength columns: Numeric column names (parsed as floats)
- Last column(s): Target variable(s) with non-numeric names
- Minimum 100 wavelength columns

#### Long Format (Single Spectrum)

For files containing a single spectrum in two-column format:

```
wavelength,value
350,0.245
351,0.248
352,0.251
...
2500,0.156
```

The sample ID is automatically derived from the filename.

#### Combined Format Detection

For directories containing a single CSV/Excel file, Spectral Predict treats it as a combined format where:
- Wavelength columns are identified by numeric column names
- ID and target columns are auto-detected from non-numeric columns
- Missing ID columns result in auto-generated sample IDs (Sample_1, Sample_2, etc.)

### 3.5 Loading Reference Data

The reference file contains the target values (Y) for calibration model development.

#### Supported Reference Formats

| Format | Extensions | Notes |
|--------|------------|-------|
| CSV | `.csv` | Standard comma-separated |
| Excel | `.xlsx`, `.xls` | First sheet by default |

#### Reference File Structure

```
sample_id,nitrogen,moisture,organic_matter,collection_date
Sample_001,6.4,12.3,8.5,2024-01-15
Sample_002,7.9,10.8,9.2,2024-01-15
Sample_003,5.2,14.1,7.8,2024-01-16
```

**Key Requirements:**
- Must contain an ID column that matches spectral file names
- Target variable column(s) for modeling
- Additional metadata columns are preserved but not used in modeling

### 3.6 ID Matching and Alignment

Spectral Predict uses intelligent ID matching to align spectral data with reference values.

#### Matching Algorithm

```
                    ID MATCHING WORKFLOW
    +--------------------------------------------------+
    |                                                  |
    |   Spectral IDs        Reference IDs              |
    |   (from filenames)    (from ID column)           |
    |         |                   |                    |
    |         v                   v                    |
    |   +-------------+   +-------------+              |
    |   | Sample_001  |   | Sample_001  |  EXACT      |
    |   | sample.asd  |   | sample      |  MATCH      |
    |   | SAMPLE_002  |   | sample_002  |             |
    |   +-------------+   +-------------+              |
    |         |                   |                    |
    |         +-------+   +-------+                    |
    |                 |   |                            |
    |                 v   v                            |
    |         [  Normalization  ]                      |
    |         - Remove extensions (.asd, .sig, etc.)   |
    |         - Remove spaces                          |
    |         - Convert to lowercase                   |
    |                 |                                |
    |                 v                                |
    |         [  Match Results  ]                      |
    |         - Matched: 98 samples                    |
    |         - Unmatched spectra: 2                   |
    |         - Missing reference: 0                   |
    |                                                  |
    +--------------------------------------------------+
```

#### Flexible Matching Rules

1. **Exact matching** is attempted first
2. If no exact matches, **normalized matching** is applied:
   - File extensions removed (`.asd`, `.sig`, `.csv`, `.txt`, `.spc`)
   - Spaces removed
   - Case-insensitive comparison

#### Handling Mismatches

The system reports:
- Number of matched samples
- Samples with spectra but no reference (excluded from analysis)
- Reference entries with no corresponding spectra (ignored)
- Samples dropped due to missing target values (NaN)

### 3.7 Data Type Detection

Spectral Predict automatically determines whether loaded data represents reflectance or absorbance measurements.

#### Detection Algorithm

The detection uses multiple criteria with weighted scoring:

| Criterion | Weight | Reflectance Indicator | Absorbance Indicator |
|-----------|--------|----------------------|---------------------|
| Value bounds | 40% | Values in 0-1 range | Values > 1 or negative |
| Mean value | 30% | Mean 0.3-0.9 | Mean > 1.0 |
| Peak analysis | 20% | Upward peaks at known wavelengths | Downward peaks |
| Metadata hints | 10% | Column names contain "reflectance" | Column names contain "absorbance" |

#### Detection Output

After loading data, the system displays:
```
Detected data type: Reflectance (confidence: 87.5%)
```

**Confidence levels:**
- **High (>80%)**: Strong evidence for detected type
- **Medium (60-80%)**: Reasonable confidence, verify visually
- **Low (<60%)**: Ambiguous data, manual verification recommended

#### Manual Override

Users can override the automatic detection:
1. Select **Reflectance** or **Absorbance** radio button
2. Use **Convert to Absorbance** button if conversion is needed

The absorbance conversion applies the logarithmic transformation:

$$A = \log_{10}\left(\frac{1}{R}\right)$$

where $A$ is absorbance and $R$ is reflectance.

### 3.8 Wavelength Range Configuration

#### Automatic Range Detection

Upon loading data, wavelength limits are automatically populated:
- **Minimum wavelength**: Smallest wavelength in the dataset
- **Maximum wavelength**: Largest wavelength in the dataset

#### Manual Range Adjustment

Users can restrict the wavelength range for analysis:
1. Enter desired minimum in the "from" field
2. Enter desired maximum in the "to" field
3. Click **Update Plots** to apply

**Best Practice**: Import the full spectral range initially. Wavelength selection can be refined during analysis configuration to ensure preprocessing algorithms have access to complete spectral information.

### 3.9 Preview Functionality

The **Plots** subtab provides immediate visualization of loaded data.

#### Available Plot Types

| Plot | Description | Purpose |
|------|-------------|---------|
| Raw Spectra | Overlay of all loaded spectra | Identify gross anomalies, verify data quality |
| 1st Derivative | Savitzky-Golay first derivative | Enhance spectral features, baseline removal |
| 2nd Derivative | Savitzky-Golay second derivative | Peak resolution, further baseline removal |
| Target Distribution | Histogram of reference values | Check for outliers, class balance |

#### Interactive Features

- **Hover**: Display sample ID and value at cursor position
- **Click**: Toggle spectrum exclusion (highlighted in red)
- **Zoom**: Mouse scroll or click-drag to zoom
- **Pan**: Right-click drag to pan view

### 3.10 Import Process Flow

The complete import workflow follows this sequence:

```
                        IMPORT PROCESS FLOW
    +------------------------------------------------------------------+
    |                                                                  |
    |   User Input                                                     |
    |   [Spectral Dir] + [Reference File]                              |
    |         |                                                        |
    |         v                                                        |
    |   +------------------+                                           |
    |   | Format Detection |                                           |
    |   | - Scan directory |                                           |
    |   | - Identify types |                                           |
    |   +------------------+                                           |
    |         |                                                        |
    |         v                                                        |
    |   +------------------+     +------------------+                   |
    |   | Read Spectra     |     | Read Reference   |                   |
    |   | - Parse files    |     | - Load CSV/Excel |                   |
    |   | - Extract values |     | - Parse columns  |                   |
    |   +------------------+     +------------------+                   |
    |         |                         |                              |
    |         v                         v                              |
    |   +------------------------------------------+                   |
    |   |           ID Alignment                    |                   |
    |   | - Match spectra to reference by ID        |                   |
    |   | - Report unmatched samples                |                   |
    |   | - Remove samples with missing Y values    |                   |
    |   +------------------------------------------+                   |
    |         |                                                        |
    |         v                                                        |
    |   +------------------+                                           |
    |   | Data Type        |                                           |
    |   | Detection        |                                           |
    |   | - Analyze values |                                           |
    |   | - Score criteria |                                           |
    |   +------------------+                                           |
    |         |                                                        |
    |         v                                                        |
    |   +------------------+                                           |
    |   | Validation       |                                           |
    |   | - Check min WLs  |                                           |
    |   | - Verify order   |                                           |
    |   +------------------+                                           |
    |         |                                                        |
    |         v                                                        |
    |   +------------------+                                           |
    |   | Generate Plots   |                                           |
    |   | - Spectral view  |                                           |
    |   | - Derivatives    |                                           |
    |   | - Distribution   |                                           |
    |   +------------------+                                           |
    |         |                                                        |
    |         v                                                        |
    |   [Data Ready for Analysis]                                      |
    |                                                                  |
    +------------------------------------------------------------------+
```

### 3.11 Troubleshooting Import Issues

#### Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| "No matching IDs" | Spectral and reference IDs don't align | Check ID column selection; verify file naming conventions |
| "Expected at least 100 wavelengths" | Too few spectral points detected | Verify file format; check for header issues |
| "Wavelengths must be strictly increasing" | Non-monotonic wavelength values | Sort data or check for duplicate wavelengths |
| "Empty CSV file" | File contains no data rows | Verify file content; check encoding |

#### Diagnostic Tips

1. **Preview file contents**: Open CSV/Excel files to verify structure
2. **Check file names**: Ensure spectral filenames match reference IDs
3. **Verify column headers**: Wavelength columns must be numeric
4. **Check for special characters**: Avoid non-ASCII characters in IDs

---

## Chapter 4: Data Quality Check Tab

The Data Quality Check tab provides comprehensive outlier detection capabilities using established chemometric methods. This chapter documents the statistical foundations and practical usage of each detection method.

### 4.1 Tab Overview

The Data Quality Check tab enables systematic identification of anomalous samples that may compromise model performance.

```
+------------------------------------------------------------------+
|  Data Quality Check                                              |
|                                                                  |
|  1. OUTLIER DETECTION PARAMETERS                                 |
|  +------------------------------------------------------------+  |
|  | PCA Components: [5]  Y Range: Min [____] Max [____]        |  |
|  | [Run Outlier Detection]  [Export Report]                   |  |
|  +------------------------------------------------------------+  |
|                                                                  |
|  2. OUTLIER DETECTION PLOTS                                      |
|  +----------------------------------------------------------+    |
|  | PCA Scores | Hotelling T2 | Q-Residuals | Mahalanobis | Y |    |
|  +----------------------------------------------------------+    |
|  |                                                          |    |
|  |              [Interactive Plot Area]                     |    |
|  |                                                          |    |
|  +----------------------------------------------------------+    |
|                                                                  |
|  3. OUTLIER SUMMARY                                              |
|  +------------------------------------------------------------+  |
|  | Sample | Y Value | T2 | Q | Maha | Y | Flags |             |  |
|  |--------|---------|----|----|------|---|-------|             |  |
|  | S_001  |   6.4   | No | No | No   |No |   0   |             |  |
|  | S_042  |  15.2   |YES |YES | YES  |YES|   4   |  <- Outlier |  |
|  +------------------------------------------------------------+  |
|                                                                  |
|  4. SAMPLE SELECTION & EXCLUSION                                 |
|  [ ] Select all flagged  [ ] High confidence (3+)               |
|  [Mark Selected for Exclusion]                                   |
|                                                                  |
+------------------------------------------------------------------+
```

### 4.2 Principal Component Analysis for Outlier Detection

PCA-based outlier detection projects high-dimensional spectral data into a lower-dimensional space where outliers become visually apparent and statistically quantifiable.

#### The PCA Model

Principal Component Analysis finds orthogonal directions (principal components) that capture maximum variance in the data:

$$\mathbf{X} = \mathbf{T}\mathbf{P}^T + \mathbf{E}$$

where:
- $\mathbf{X}$ is the centered spectral data matrix ($n \times p$)
- $\mathbf{T}$ is the scores matrix ($n \times k$)
- $\mathbf{P}$ is the loadings matrix ($p \times k$)
- $\mathbf{E}$ is the residual matrix
- $k$ is the number of components retained

#### Selecting the Number of Components

The **PCA Components** parameter determines how many principal components are used for outlier detection:

| Components | Typical Variance Explained | Best For |
|------------|---------------------------|----------|
| 2-3 | 60-80% | Quick visualization |
| 5 (default) | 80-95% | Balanced detection |
| 8-10 | 95-99% | Comprehensive analysis |

**Recommendation**: Start with the default (5 components). Increase if the score plots show clear structure not captured by the first few PCs.

### 4.3 Hotelling T-squared Statistic

The Hotelling $T^2$ statistic measures the distance of each sample from the center of the PCA model in the score space.

#### Mathematical Definition

$$T^2_i = \sum_{a=1}^{A} \frac{t_{ia}^2}{\lambda_a}$$

where:
- $T^2_i$ is the Hotelling statistic for sample $i$
- $t_{ia}$ is the score of sample $i$ on component $a$
- $\lambda_a$ is the eigenvalue (variance) of component $a$
- $A$ is the number of components

In matrix form for all samples:

$$T^2 = \text{diag}(\mathbf{T} \mathbf{\Lambda}^{-1} \mathbf{T}^T)$$

where $\mathbf{\Lambda}$ is the diagonal matrix of eigenvalues.

#### Statistical Threshold

The 95% confidence threshold is computed using the F-distribution:

$$T^2_{crit} = \frac{A(n-1)}{n-A} \cdot F_{\alpha, A, n-A}$$

where:
- $A$ = number of principal components
- $n$ = number of samples
- $\alpha$ = significance level (typically 0.05)
- $F_{\alpha, A, n-A}$ = F-distribution critical value

#### Interpretation

| $T^2$ Value | Interpretation |
|-------------|----------------|
| Below threshold | Sample within normal variation |
| 1-2x threshold | Moderate outlier, investigate |
| >2x threshold | Strong outlier, likely anomalous |

**Visual**: The Hotelling T-squared plot displays bar charts of $T^2$ values with a horizontal threshold line. Bars exceeding the threshold are highlighted in red.

### 4.4 Q-Residuals (Squared Prediction Error)

Q-residuals measure how well each sample is represented by the PCA model, detecting samples that are structurally different from the majority.

#### Mathematical Definition

$$Q_i = \sum_{j=1}^{p} e_{ij}^2 = \sum_{j=1}^{p} (x_{ij} - \hat{x}_{ij})^2$$

where:
- $Q_i$ is the Q-residual for sample $i$
- $e_{ij}$ is the residual for sample $i$ at wavelength $j$
- $x_{ij}$ is the original spectrum value
- $\hat{x}_{ij}$ is the PCA-reconstructed value

The reconstructed spectrum is:

$$\hat{\mathbf{x}}_i = \sum_{a=1}^{A} t_{ia} \mathbf{p}_a$$

#### Threshold Determination

The 95th percentile of Q-residuals serves as the threshold:

$$Q_{crit} = Q_{95\%}$$

This empirical threshold is robust and does not assume a specific distribution.

#### Complementary to Hotelling $T^2$

| Metric | Detects | Misses |
|--------|---------|--------|
| Hotelling $T^2$ | Samples far from center in PC space | Samples with unusual spectral features not captured by PCs |
| Q-residuals | Samples poorly described by the PCA model | Samples that are extreme but well-modeled |

**Best Practice**: Use both metrics together. Samples flagged by both have the strongest evidence for being outliers.

### 4.5 Mahalanobis Distance

The Mahalanobis distance provides a scale-invariant measure of how far each sample is from the multivariate center of the data.

#### Mathematical Definition

$$D_i = \sqrt{(\mathbf{t}_i - \bar{\mathbf{t}})^T \mathbf{S}^{-1} (\mathbf{t}_i - \bar{\mathbf{t}})}$$

where:
- $D_i$ is the Mahalanobis distance for sample $i$
- $\mathbf{t}_i$ is the score vector for sample $i$
- $\bar{\mathbf{t}}$ is the mean score vector
- $\mathbf{S}$ is the covariance matrix of scores

#### Robust Threshold Using MAD

The threshold uses the Median Absolute Deviation (MAD) for robustness:

$$D_{crit} = \text{median}(D) + 3 \times \text{MAD}(D)$$

where:

$$\text{MAD} = \text{median}(|D_i - \text{median}(D)|)$$

#### Advantages Over Euclidean Distance

| Euclidean | Mahalanobis |
|-----------|-------------|
| Treats all dimensions equally | Accounts for variance in each dimension |
| Ignores correlations | Accounts for correlations between variables |
| Sensitive to scale | Scale-invariant |

### 4.6 Y-Value Consistency Checks

Reference value outliers can indicate data entry errors, mislabeled samples, or samples outside the calibration range.

#### Statistical Detection

**Z-score method**: Samples with $|z| > 3$ are flagged:

$$z_i = \frac{y_i - \bar{y}}{s_y}$$

where $\bar{y}$ is the mean and $s_y$ is the standard deviation.

#### Range-Based Detection

Users can specify chemically reasonable bounds:
- **Lower bound**: Minimum plausible value (e.g., 0% for concentration)
- **Upper bound**: Maximum plausible value (e.g., 100% for percentage)

Samples outside these bounds are flagged regardless of statistical considerations.

### 4.7 Combined Outlier Assessment

The outlier summary table integrates all detection methods with confidence levels.

#### Confidence Levels

| Flags | Confidence | Recommendation |
|-------|------------|----------------|
| 0 | Normal | Include in analysis |
| 1 | Low | Monitor but likely acceptable |
| 2 | Moderate | Investigate; consider exclusion |
| 3+ | High | Strong evidence of outlier; recommend exclusion |

#### Sample Selection Workflow

1. **Run Outlier Detection**: Computes all metrics
2. **Review Plots**: Visually inspect each diagnostic plot
3. **Auto-Select**: Use checkboxes to select by confidence level
4. **Manual Refinement**: Click individual rows to toggle selection
5. **Mark for Exclusion**: Apply exclusions to selected samples

### 4.8 Score Plot Interpretation

The PCA Scores plot is a powerful diagnostic tool for understanding data structure and identifying outliers.

```
                    PCA SCORE PLOT
        ^
        |           o
        |      o   o o    [Cluster 2]
   PC2  |    o  o o o
        |   o o  o  o
        |  [Cluster 1]         X   <- Outlier
        |    o o o o
        |       o o
        +-------------------------------->
                     PC1

    Legend:
    o  = Normal samples
    X  = Flagged outliers
```

#### Typical Patterns

| Pattern | Interpretation | Action |
|---------|---------------|--------|
| Single dense cluster | Homogeneous dataset | Outliers are clear deviations |
| Multiple clusters | Subgroups in data | May indicate batch effects or categorical differences |
| Gradient along PC1 | Continuous variation | Related to target variable; normal |
| Scattered points | High heterogeneity | Consider stratification or subset modeling |

### 4.9 Typical Outlier Thresholds

The following table summarizes default thresholds used by Spectral Predict:

| Metric | Default Threshold | Basis |
|--------|------------------|-------|
| Hotelling $T^2$ | F-distribution, 95% | Statistical theory |
| Q-residuals | 95th percentile | Empirical |
| Mahalanobis | Median + 3*MAD | Robust statistics |
| Y z-score | $|z| > 3$ | Normal distribution (99.7% interval) |

### 4.10 Practical Guidelines for Outlier Handling

#### When to Exclude Outliers

**Exclude** when outliers are due to:
- Measurement errors (instrument malfunction, sample preparation issues)
- Data entry errors (transcription mistakes)
- Contaminated or damaged samples
- Samples outside the intended calibration scope

**Retain** when outliers represent:
- Natural variation within the population
- Rare but valid sample types
- Extremes needed for calibration range coverage

#### Documentation Best Practices

1. Record which samples were excluded and why
2. Save the outlier report for reference
3. Document threshold settings used
4. Note any manual overrides of automatic detection

### 4.11 Exporting Outlier Reports

Click **Export Report** to generate a CSV file containing:
- Sample identifiers
- All computed metrics ($T^2$, Q, Mahalanobis, z-scores)
- Outlier flags for each method
- Total flag count
- Final inclusion/exclusion status

This report provides a complete audit trail for quality assurance and regulatory compliance.

---

## Chapter 5: Understanding Spectral Data Types

### 5.1 Reflectance vs. Absorbance

Spectroscopic measurements can be expressed as either reflectance or absorbance, and understanding the difference is critical for proper data handling.

#### Reflectance

Reflectance ($R$) is the ratio of reflected light to incident light:

$$R = \frac{I_{reflected}}{I_{incident}}$$

- Typically ranges from 0 to 1 (or 0% to 100%)
- Higher values indicate more light reflected (lighter materials)
- Common output format for NIR and VIS-NIR spectrometers

#### Absorbance

Absorbance ($A$) follows the Beer-Lambert law:

$$A = -\log_{10}(T) = -\log_{10}(1-R) \approx \log_{10}\left(\frac{1}{R}\right)$$

- Theoretically unbounded but typically 0-3 AU
- Proportional to analyte concentration (Beer's Law)
- Preferred for quantitative analysis

### 5.2 When to Convert

| Starting Data | Target Application | Recommendation |
|--------------|-------------------|----------------|
| Reflectance | Quantitative analysis | Convert to absorbance |
| Reflectance | Pattern recognition/classification | Either works |
| Absorbance | All applications | Use as-is |

**Note**: Many preprocessing methods (SNV, derivatives) work well with either data type, but concentration predictions follow Beer's Law more directly with absorbance data.

---

## Summary

Part I has covered the fundamental concepts and initial steps for working with Spectral Predict:

1. **Introduction**: Understanding the software's purpose, capabilities, and target applications
2. **Quick Start**: A rapid walkthrough of the complete analysis workflow
3. **Import & Preview Tab**: Comprehensive data loading, format support, and ID alignment
4. **Data Quality Check Tab**: PCA-based outlier detection using Hotelling $T^2$, Q-residuals, Mahalanobis distance, and Y-value consistency checks

With data properly imported and quality-checked, you are ready to proceed to Part II: Analysis Configuration, which covers preprocessing options, model selection, and variable selection methods.

---

*Spectral Predict V1 User Guide - Part I: Getting Started*
*Document Version 1.0*
