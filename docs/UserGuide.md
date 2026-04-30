# Spectral Predict V1 by Matt Sponheimer 

User Guide

**Comprehensive Reference for Automated Spectral Modeling**

---

**Version 1.0 | January 2026**

**Copyright 2026 - All Rights Reserved**

---

## Table of Contents

- [Part I: Getting Started](#part-i-getting-started)
  - [Chapter 1: Introduction to Spectral Predict](#chapter-1-introduction-to-spectral-predict)
  - [Chapter 2: Quick Start Tutorial](#chapter-2-quick-start-tutorial)
  - [Chapter 3: The Import & Preview Tab](#chapter-3-the-import--preview-tab)
  - [Chapter 4: Data Quality Check Tab](#chapter-4-data-quality-check-tab)
- [Part II: Running Analyses](#part-ii-running-analyses)
  - [Chapter 4: Analysis Configuration Tab](#chapter-4-analysis-configuration-tab)
  - [Chapter 5: Analysis Progress Tab](#chapter-5-analysis-progress-tab)
  - [Chapter 6: Results Tab](#chapter-6-results-tab)
- [Part III: Machine Learning Models Reference](#part-iii-machine-learning-models-reference)
- [Part IV: Spectral Preprocessing Reference](#part-iv-spectral-preprocessing-reference)
- [Part V: Variable Selection Reference](#part-v-variable-selection-reference)
- [Part VI: Hyperparameter Optimization Reference](#part-vi-hyperparameter-optimization-reference)
- [Part VII: Ensemble Methods and Advanced Features](#part-vii-ensemble-methods-and-advanced-features)
- [Part VIII: Reference](#part-viii-reference)
- [Appendices](#appendices)

---

## About This Guide

This comprehensive user guide documents all features of Spectral Predict V1, an automated spectral modeling application for developing quantitative and qualitative calibration models from spectroscopic data.

**What This Guide Covers:**

- Complete documentation of all 14 GUI tabs
- Detailed explanations of 13+ machine learning models
- Mathematical foundations for all preprocessing methods
- Step-by-step algorithms for variable selection techniques
- Complete hyperparameter optimization reference
- Metrics glossary with formulas
- Supported file format specifications

**Target Audience:**

- Spectroscopists developing calibration models
- Researchers in chemometrics and analytical chemistry
- Quality control professionals using NIR/MIR spectroscopy
- Scientists in food, pharmaceutical, and agricultural industries

**Document Conventions:**

- Mathematical formulas are presented in LaTeX notation
- ASCII diagrams illustrate key concepts
- Tables summarize parameters and their effects
- Code examples show implementation details

---

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
# Part II: Running Analyses

This section provides comprehensive documentation for configuring and executing spectral analyses in Spectral Predict V1. You will learn how to set up analysis parameters, monitor progress, and interpret results.

---

## Chapter 4: Analysis Configuration Tab

The Analysis Configuration tab (Tab 4) is the central control panel for all analysis settings. It is organized into five sub-tabs to help you navigate the many configuration options efficiently.

```
+------------------------------------------------------------------+
|  Analysis Configuration                                           |
|  +--------------------------------------------------------------+|
|  | Basic Settings | Variable Selection | Model Config |          ||
|  |                | Ensemble Methods   | Validation   |          ||
|  +--------------------------------------------------------------+|
|                                                                   |
|  [Run Analysis Button - Available on all sub-tabs]               |
|                                                                   |
+------------------------------------------------------------------+
```

### 4.1 Basic Settings Sub-Tab

The Basic Settings sub-tab contains fundamental configuration options that affect all analyses.

#### 4.1.1 Target Variable Selection

The target variable dropdown allows you to select which column from your reference data should be used as the prediction target.

| Setting | Description |
|---------|-------------|
| Target Variable | The column containing values to predict (e.g., protein content, moisture, species class) |
| Auto-Detection | If only one numeric column exists, it is automatically selected |

**Important**: For classification tasks, ensure your target variable contains categorical values (text labels or discrete integers). For regression tasks, use continuous numeric values.

#### 4.1.2 Cross-Validation Configuration

Cross-validation (CV) is the primary method for evaluating model performance. The CV folds setting determines how many subsets your data is split into for training and validation.

| CV Folds | Recommended For | Description |
|----------|-----------------|-------------|
| 3 | Small datasets (<30 samples) | Minimal but fast |
| 5 | Medium datasets (30-100 samples) | Standard default |
| 10 | Large datasets (>100 samples) | More robust estimates |
| Leave-One-Out | Very small datasets (<20 samples) | Each sample used as test once |

**Guidelines for CV Fold Selection**:
- 50+ samples: Use 5-fold cross-validation
- 100+ samples: Use 10-fold cross-validation
- <50 samples: Consider Leave-One-Out (LOO)

#### 4.1.3 Penalty Settings

Spectral Predict uses a composite scoring system that can incorporate penalties for model complexity and variable count. Both penalties use a 0-10 scale.

**Variable Count Penalty (0-10)**

| Value | Effect |
|-------|--------|
| 0 | Only performance (R^2/Accuracy) matters - no penalty for using many wavelengths |
| 1-3 | Light penalty - exploration-friendly, slightly favors parsimonious models |
| 4-6 | Moderate penalty - balances performance with simplicity |
| 7-10 | Strong penalty - strongly prefers models using fewer wavelengths |

**Model Complexity Penalty (0-10)**

| Value | Effect |
|-------|--------|
| 0 | Only performance matters - no penalty for complex models |
| 1-3 | Light penalty - allows complex models with slight preference for simpler ones |
| 4-6 | Moderate penalty - balances performance with interpretability |
| 7-10 | Strong penalty - strongly prefers simpler models (fewer latent variables, shallower trees) |

**Scoring Formula**:
```
CompositeScore = PerformanceScore + VariablePenaltyTerm + ComplexityPenaltyTerm

Where:
- PerformanceScore (penalties=0): -R2cv (lower is better)
- PerformanceScore (penalties>0): 0.5*z(RMSEcv) - 0.5*z(R2cv)
- VariablePenaltyTerm = (penalty/10)^2 * (n_vars/full_vars)
- ComplexityPenaltyTerm = (penalty/10)^2 * complexity_fraction
```

#### 4.1.4 Wavelength Restriction

You can restrict the wavelength range used for analysis without affecting your imported data.

| Option | Description |
|--------|-------------|
| Enable Restriction | Checkbox to activate wavelength filtering |
| Analysis Range | Min and max wavelength values (e.g., 800-2500 nm) |
| Custom Regions | Specify multiple regions (e.g., "2000-2100, 2200-2300") |

**Preset Buttons**:
- UV (10-400 nm)
- VIS (400-800 nm)
- NIR (800-2500 nm)
- MIR (2500-25000 nm)

#### 4.1.5 Preprocessing Methods

Select which preprocessing transformations to test during analysis.

| Method | Description | Best For |
|--------|-------------|----------|
| Raw | No preprocessing | Baseline comparison |
| SNV | Standard Normal Variate | Scatter correction, solid samples |
| SG1 (1st derivative) | First derivative with Savitzky-Golay smoothing | Baseline offset removal |
| SG2 (2nd derivative) | Second derivative with Savitzky-Golay smoothing | Peak enhancement, overlapping peaks |
| SG3 (3rd derivative) | Third derivative | Complex spectral features |
| SG4 (4th derivative) | Fourth derivative | Rarely needed |
| deriv_snv | Derivative then SNV | Combined baseline issues |

**Derivative Window Sizes**:

| Window Size | Characteristics |
|-------------|-----------------|
| 7 | Preserves sharp features, more noise |
| 11 | Balance of smoothing and feature preservation |
| 17 | Standard default - good for most spectral data |
| 23 | More smoothing |
| 31 | Heavy smoothing |

**Tip**: Select Window=17 as the default starting point. Enable multiple window sizes only for comprehensive analyses.

#### 4.1.6 Pre-Processing Steps (Optional)

These optional steps are applied before the main transformations:

**Baseline Correction**:
- Polynomial: Fits and removes polynomial baseline (degree 1-5)
- AsLS: Asymmetric Least Squares baseline correction

**Smoothing**:
- Savitzky-Golay smoothing applied before derivative/SNV transforms
- Configure window size (7-25) and polynomial order (2-4)

#### 4.1.7 Smart Preprocessing Discovery

Enable automated preprocessing optimization using NSGA-II-style intelligence.

| Setting | Description | Default |
|---------|-------------|---------|
| Enable Discovery | Activates exhaustive preprocessing testing | Off |
| Importance Method | cars_tree, model_specific, lightgbm, or vip | model_specific |
| Top Configs | Number of preprocessing configurations to pass to grid search | 7 |

**What It Tests**: 14 preprocessing types x 6 window sizes = 84 combinations
**Runtime**: Approximately 3-4 minutes

---

### 4.2 Variable Selection Sub-Tab

The Variable Selection sub-tab configures which wavelengths to include in your models.

#### 4.2.1 Subset Analysis

**Top-N Variable Analysis**:
Test models using only the N most important wavelengths.

| N Value | Description |
|---------|-------------|
| N=10 | Minimal feature set |
| N=20 | Small subset |
| N=50 | Moderate subset |
| N=100 | Standard subset |
| N=250 | Large subset |
| N=500 | Very large subset |
| N=1000 | Nearly full spectrum |

**Spectral Region Analysis**:
Test models using auto-detected spectral regions.

| Depth | Regions Tested |
|-------|----------------|
| Shallow | 5 regions |
| Medium | 10 regions |
| Deep | 15 regions |
| Thorough | 20 regions |

**Advanced Region Options**:
- Test all regions individually (default tests approximately half)
- Test pairwise combinations (e.g., 45 pairs for top 10 regions)

#### 4.2.2 Variable Selection Methods

Select one or more sophisticated wavelength selection techniques:

| Method | Speed | Best For |
|--------|-------|----------|
| Feature Importance | Fast | Quick ranking, interpretable results |
| SPA (Successive Projections) | Fast | Minimizing collinearity |
| UVE (Uninformative Variable Elimination) | Fast | Filtering noisy variables |
| UVE-SPA Hybrid | Fast | Combined noise + collinearity reduction |
| iPLS (Interval PLS) | Medium | Region-based interpretation |
| Forward iPLS | Medium | Interval combination |
| Backward iPLS | Medium | Interval elimination |
| CARS (PLS-based) | Slow | Best for linear models |
| CARS-Tree | Slow | Best for tree models (RF, LightGBM, XGBoost) |
| VCPA-IRIV | Slowest | Best compact subset |
| GA (Genetic Algorithm) | Slowest | Comprehensive evolutionary search |

**Method Parameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| UVE Cutoff | 1.0 | Threshold multiplier (0.7-1.5) |
| UVE Components | Auto | Number of PLS components |
| iPLS Intervals | 20 | Number of spectral intervals |
| Fwd iPLS Max Combine | 5 | Maximum intervals to combine |
| GA Population | 64 | Genetic algorithm population size |
| GA Generations | 100 | Number of evolution generations |
| GA Runs | 5 | Independent runs for stability |

---

### 4.3 Model Configuration Sub-Tab

The Model Configuration sub-tab controls which machine learning models to test and their hyperparameters.

#### 4.3.1 Model Tiers

Select a preset tier to automatically configure model selections:

| Tier | Models Included | Recommended For |
|------|-----------------|-----------------|
| Quick | PLS, RandomForest | Rapid testing, daily QC, preliminary analysis |
| Standard | PLS, Ridge, Lasso, ElasticNet, RandomForest, LightGBM | Most users, routine work, daily analysis |
| Comprehensive | Standard + XGBoost, CatBoost, NeuralBoosted | Thorough analysis, research, publications |
| Experimental | All models including SVR, MLP | Exploration, method comparison, no time constraints |
| Custom | User-selected | When you need specific model combinations |

**Classification Tiers**:

| Tier | Classification Models |
|------|----------------------|
| Quick | PLS-DA, RandomForest |
| Standard | PLS-DA, RandomForest, LightGBM, XGBoost, CatBoost |
| Comprehensive | Standard + SVM, MLP, NeuralBoosted |
| Experimental | All classification models |

#### 4.3.2 Optimization Method

Choose how hyperparameters are searched:

| Method | Description | When to Use |
|--------|-------------|-------------|
| Grid Search | Tests all parameter combinations exhaustively | When thoroughness matters more than speed |
| Bayesian Optimization | Intelligent sampling of parameter space | Recommended for most analyses |
| NSGA-II Multi-Objective | Pareto optimization for error, parsimony, and complexity | When you need trade-off exploration |

**Bayesian Optimization Parameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Trials | 500 | Number of optimization iterations |

**NSGA-II Parameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Population | 60 | Size of each generation |
| Generations | 120 | Number of evolutionary cycles |
| Selection | Min Error | How to select final solution (Min Error, Balanced, Knee Point) |
| Mode | NSGA-II w/Guidance | Standard NSGA-II or guided variant |

#### 4.3.3 Available Models

**Core Models (Linear)**:

| Model | Description | Key Hyperparameters |
|-------|-------------|---------------------|
| PLS | Partial Least Squares | n_components (latent variables), max_iter, tol |
| PLS-DA | PLS Discriminant Analysis (classification) | n_components, C (LogisticRegression) |
| Ridge | L2 regularized linear regression | alpha (regularization strength), solver, tol |
| Lasso | L1 regularized linear regression (sparse) | alpha, selection, tol |
| ElasticNet | Combined L1+L2 regularization | alpha, l1_ratio, selection, tol |

**Tree-Based Models**:

| Model | Description | Key Hyperparameters |
|-------|-------------|---------------------|
| RandomForest | Ensemble of decision trees | n_estimators, max_depth, min_samples_split |
| XGBoost | Extreme Gradient Boosting | n_estimators, learning_rate, max_depth, subsample |
| LightGBM | Light Gradient Boosting Machine | n_estimators, learning_rate, num_leaves, max_depth |
| CatBoost | Categorical Boosting | iterations, learning_rate, depth, l2_leaf_reg |

**Other Models**:

| Model | Description | Key Hyperparameters |
|-------|-------------|---------------------|
| MLP | Multi-Layer Perceptron (neural network) | hidden_layer_sizes, alpha, learning_rate_init |
| SVR/SVM | Support Vector Regression/Machine | kernel, C, gamma, epsilon |
| NeuralBoosted | Gradient boosting with neural networks | n_estimators, learning_rate, hidden_layer_size |

#### 4.3.4 Advanced Model Options

Each model type has collapsible sections for detailed hyperparameter configuration:

**Example: PLS Hyperparameters**
- Max Latent Variables: 2-100 (default: 8)
- Max Iterations: 500, 1000, or custom
- Tolerance: 1e-7, 1e-6 (default), 1e-5

**Example: Random Forest Hyperparameters**
- Number of Trees: 100, 200, 500 (default), custom
- Maximum Depth: None (unlimited), 30, custom
- Min Samples Split: 2, 5, 10, 20
- Max Features: sqrt (default), log2, None (all)
- Bootstrap: True (default), False

---

### 4.4 Ensemble Methods Sub-Tab

Ensemble methods combine predictions from multiple models to improve overall performance. This feature is available for regression tasks only.

#### 4.4.1 Enabling Ensembles

Check "Enable Ensemble Methods" to activate ensemble training after the main analysis completes.

#### 4.4.2 Available Ensemble Methods

| Method | Description | Best For |
|--------|-------------|----------|
| Simple Average | Equal weight to all models | Baseline ensemble, stable predictions |
| Region-Aware Weighted | Dynamic weights based on prediction region | When models excel in different Y-value ranges |
| Mixture of Experts | Selects best model per region | Highly specialized model combination |
| Stacking Ensemble | Meta-learner combines predictions | Complex prediction scenarios |
| Stacking + Region Features | Stacking with region-aware meta-features | Advanced region-dependent optimization |

#### 4.4.3 Ensemble Configuration

| Setting | Options | Description |
|---------|---------|-------------|
| Number of Regions | 3, 5 (default), 7, 10 | How many Y-value ranges to divide data into |
| Top N Models | 3-20 (default: 10) | Number of models to include from ranked list |
| Models per Quartile | 1-10 (default: 5) | For quartile-based specialist selection |

**Workflow**:
```
1. Run main analysis
2. Top N models automatically selected
3. Each ensemble method trained
4. Results compared in Ensemble Results section
5. Best ensemble identified and available for saving
```

---

### 4.5 Validation Sub-Tab

The Validation sub-tab configures holdout validation for independent model testing.

#### 4.5.1 Enabling Validation

Check "Enable Validation Set" to hold out samples from training for independent evaluation.

#### 4.5.2 Validation Set Size

Use the slider to select what percentage of your data to reserve for validation:

| Percentage | Description |
|------------|-------------|
| 5-10% | Minimal validation, more data for training |
| 15-25% | Standard range for most applications |
| 30-40% | Large validation set for high-confidence estimates |

**Warning**: Validation sets larger than 40% are generally not recommended.

#### 4.5.3 Selection Algorithms

| Algorithm | Description | Recommended For |
|-----------|-------------|-----------------|
| Kennard-Stone | Maximizes spectral diversity only | When Y distribution doesn't matter |
| SPXY | Balances spectral and target diversity | Most applications (recommended) |
| Random | Simple random selection | Quick testing, baseline comparison |
| Stratified | Ensures balanced target distribution | Classification or highly skewed regression |
| Manual | Select samples from Data Viewer | Expert-guided validation design |

#### 4.5.4 Creating the Validation Set

1. Configure size and algorithm
2. Click "Create Validation Set"
3. Status shows: "Calibration: N samples | Validation: M samples"

**For Manual Selection**:
1. Select algorithm = Manual
2. Go to Data Viewer tab and select rows
3. Return to Validation tab and click "Use Selected Rows from Data Viewer"

#### 4.5.5 Validation Metrics Configuration

| Setting | Description |
|---------|-------------|
| Show validation metrics in results | Display RMSEP and R^2pred columns |
| Calculate for top N models | Limit validation calculation to top performers (50-500) |

---

## Chapter 5: Analysis Progress Tab

The Analysis Progress tab (Tab 5) provides real-time monitoring of running analyses.

### 5.1 Progress Display

```
+------------------------------------------------------------------+
|  Analysis Progress                                                |
|                                                                   |
|  [Animated Running Figure]              Best Model So Far:        |
|                                         PLS snv W=17 (R2=0.9523)  |
|                                                                   |
|  Progress: Testing model 45/120         Est. remaining: 3m 20s    |
|                                                                   |
|  [Pause] [Resume] [Stop]                                          |
|                                                                   |
|  +--------------------------------------------------------------+|
|  | Progress Log:                                                 ||
|  | > Starting analysis...                                        ||
|  | > Testing PLS with raw preprocessing...                       ||
|  | > Completed: PLS raw (R2cv=0.8234, RMSE=1.234)               ||
|  | > Testing PLS with SNV preprocessing...                       ||
|  | ...                                                           ||
|  +--------------------------------------------------------------+|
|                                                                   |
|  Status: Analysis running...                                      |
+------------------------------------------------------------------+
```

### 5.2 Progress Information

| Element | Description |
|---------|-------------|
| Best Model So Far | Updates with the current top performer as analysis progresses |
| Progress Counter | Shows current model number out of total |
| Time Estimate | Estimated remaining time based on elapsed rate |
| Progress Log | Scrollable text area with detailed status messages |

### 5.3 Control Buttons

| Button | Action |
|--------|--------|
| Pause | Suspends analysis after current model completes |
| Resume | Continues paused analysis |
| Stop | Terminates analysis and preserves results collected so far |

### 5.4 Status Messages

The progress log shows:
- Analysis initialization
- Each preprocessing/model combination being tested
- Completion metrics (R^2cv, RMSEcv) for each test
- Variable selection progress (if enabled)
- Ensemble training progress (if enabled)
- Final completion status

**Example Progress Log**:
```
> Starting analysis with 5-fold cross-validation
> Dataset: 150 samples, 500 wavelengths
> Models: PLS, Ridge, RandomForest, LightGBM
> Preprocessing: raw, snv, deriv1 W=17, deriv2 W=17

> Testing PLS with raw preprocessing...
  - Full spectrum: R2cv=0.8523, RMSEcv=2.134
  - Top 50 variables: R2cv=0.8612, RMSEcv=2.089

> Testing PLS with SNV preprocessing...
  - Full spectrum: R2cv=0.9234, RMSEcv=1.523

> Testing RandomForest with raw preprocessing...
...

> Analysis complete! 120 models tested.
> Best model: PLS snv W=17 (R2cv=0.9412, RMSEcv=1.234)
> Results saved to: outputs/analysis_results_20250122_143025.csv
```

---

## Chapter 6: Results Tab

The Results tab (Tab 6) displays all tested model configurations ranked by performance.

### 6.1 Results Table Overview

```
+------------------------------------------------------------------+
|  Model Performance Results                                        |
|  All tested models ranked by performance. Click on any result     |
|  row to load it into the 'Model Development' tab for tuning.      |
|                                                                   |
|  [Legend: Q1 Low Y | Q2 | Q3 | Q4 High Y] [Rank Symbols: * = Expert Choice]|
|                                                                   |
|  +--------------------------------------------------------------+|
|  | Rank | Model | Preprocess | R2   | RMSE | R2cv | RMSEcv |... ||
|  |------|-------|------------|------|------|------|--------|----||
|  | * 1  | PLS   | snv W=17   | 0.95 | 1.12 | 0.93 | 1.34   |... ||
|  | * 2  | PLS   | deriv1 W=17| 0.94 | 1.18 | 0.92 | 1.45   |... ||
|  |   3  | Ridge | snv W=17   | 0.93 | 1.25 | 0.91 | 1.52   |... ||
|  | ...  | ...   | ...        | ...  | ...  | ...  | ...    |... ||
|  +--------------------------------------------------------------+|
|                                                                   |
|  [Export Results to CSV] [Select Top by Region] Top: [3] From: [All]|
|                                                                   |
|  Displaying 120 results. 15 highlighted as top performers by region|
+------------------------------------------------------------------+
```

### 6.2 Results Table Columns

#### 6.2.1 Regression Results Columns

| Column | Description |
|--------|-------------|
| Rank | Overall ranking (1 = best based on composite score) |
| Select | Checkbox for ensemble training selection |
| Model | Algorithm name (PLS, Ridge, RandomForest, etc.) |
| Preprocess | Preprocessing applied (raw, snv, deriv1, deriv2, etc.) |
| Deriv | Derivative order (1, 2, or blank for non-derivative) |
| Window | Savitzky-Golay window size |
| Poly | Polynomial order for derivatives |
| LVs | Latent variables (for PLS models) |
| n_vars | Number of wavelengths used |
| full_vars | Total wavelengths available |
| SubsetTag | Variable subset identifier (e.g., "Top50", "Region_2000-2100") |
| R2 | Calibration R-squared (coefficient of determination) |
| RMSE | Calibration Root Mean Square Error |
| R2cv | Cross-validation R-squared (primary ranking metric) |
| RMSEcv | Cross-validation RMSE |
| RMSEP | Validation set RMSE (if validation enabled) |
| R2pred | Validation set R-squared (if validation enabled) |
| BestRegion | Y-value region where model excels (Q1-Q4) |
| CompositeScore | Final ranking score (lower is better) |
| ComplexityScore | Unified complexity measure (0-100) |
| top_vars | Most important wavelengths |

#### 6.2.2 Classification Results Columns

| Column | Description |
|--------|-------------|
| Rank | Overall ranking |
| Model | Algorithm name |
| Preprocess | Preprocessing applied |
| Accuracy | Calibration accuracy |
| ROC_AUC | Calibration ROC Area Under Curve |
| F1 | Calibration F1 score |
| Precision | Calibration precision |
| Recall | Calibration recall |
| Accuracycv | Cross-validation accuracy (primary ranking metric) |
| ROC_AUCcv | Cross-validation ROC AUC |
| F1cv | Cross-validation F1 score |
| Precisioncv | Cross-validation precision |
| Recallcv | Cross-validation recall |
| BestClass | Class where model excels |

### 6.3 Composite Scoring Formula

The composite score determines overall ranking. Lower scores are better.

**For Regression (with penalties = 0)**:
```
CompositeScore = -R2cv
```

**For Regression (with penalties > 0)**:
```
PerformanceScore = 0.5 * z_score(RMSEcv) - 0.5 * z_score(R2cv)
VariablePenalty = (variable_penalty/10)^2 * (n_vars/full_vars)
ComplexityPenalty = (complexity_penalty/10)^2 * complexity_fraction

CompositeScore = PerformanceScore + VariablePenalty + ComplexityPenalty
```

**For Classification (with penalties = 0)**:
```
CompositeScore = -Accuracycv - 0.0001 * F1cv
```

**For Classification (with penalties > 0)**:
```
PerformanceScore = -z_score(Accuracycv) - 0.3 * z_score(F1cv)
CompositeScore = PerformanceScore + VariablePenalty + ComplexityPenalty
```

### 6.4 Quartile Highlighting

Results are color-coded based on which Y-value region the model excels in:

| Color | Region | Description |
|-------|--------|-------------|
| Light Blue | Q1 | Low Y values (bottom 25%) |
| Light Green | Q2 | Low-mid Y values (25-50%) |
| Light Orange | Q3 | Mid-high Y values (50-75%) |
| Light Pink | Q4 | High Y values (top 25%) |

**How Quartile Performance is Calculated**:
1. Training data Y values are divided into quartiles
2. For each model, regional RMSE is computed for each quartile
3. Models are ranked within each region
4. Highlighting shows which region(s) each model excels in

### 6.5 Expert's Choice Indicators

Special symbols in the Rank column indicate model quality:

| Symbol | Meaning |
|--------|---------|
| (star) | Expert Choice - Well-generalized, minimal overfitting |
| (bullet) | Good Fit - Acceptable but some overfitting risk |
| Strikethrough | Overfit Warning - Calibration metrics much better than CV |

**Expert Choice Criteria**:
- Model must be in top 3 for its model type
- Calibration R^2 should not exceed CV R^2 by more than threshold
- Cross-validation metrics show good generalization

### 6.6 Sorting and Filtering

**Sorting**:
- Click any column header to sort by that column
- Click again to reverse sort order
- Sort indicators: (up arrow) = ascending, (down arrow) = descending

**Filtering Options**:
- Use "Select Top by Region" to auto-select top N models per quartile
- Configure "Top" spinbox (1-15) and "From" dropdown (All, Q1, Q2, Q3, Q4)

### 6.7 Exporting Results

**Export to CSV/Excel**:
1. Click "Export Results to CSV"
2. Choose save location and filename
3. Select format: CSV (.csv) or Excel (.xlsx)

**Exported File Contains**:
- All columns from results table
- All rows (not just visible/selected)
- Full precision numeric values

### 6.8 Saving Model Files

**From Results Tab**:
1. Double-click a result row to load into Model Development tab
2. In Model Development, click "Save Model (.dasp)"
3. Choose save location

**.dasp File Contents**:
- Trained model object (sklearn compatible)
- Preprocessing configuration
- Wavelength selection
- Training metadata
- Performance metrics

---

## Chapter 7: Ensemble Results Section

The Ensemble Results section appears at the bottom of the Results tab when ensembles are enabled.

### 7.1 Ensemble Results Table

| Column | Description |
|--------|-------------|
| Rank | Ensemble ranking (1 = best) |
| Method | Ensemble type (SimpleAverage, RegionAware, etc.) |
| RMSE | Calibration RMSE |
| R2 | Calibration R-squared |
| RMSEP | Validation RMSE (if validation set used) |
| R2pred | Validation R-squared (if validation set used) |
| RMSEcv | Cross-validation RMSE |
| R2cv | Cross-validation R-squared |
| MAEcv | Cross-validation Mean Absolute Error |
| RPD | Residual Prediction Deviation |
| vs Best Individual | Performance comparison to best single model |

### 7.2 Ensemble Action Buttons

| Button | Function |
|--------|----------|
| Save Selected Ensemble | Export selected ensemble as .dasp file |
| Train Ensemble | Manually train ensemble with custom model selection |
| Show Regional Performance | Display performance breakdown by Y-value region |
| Show Ensemble Weights | View model weights for weighted ensemble methods |
| Show Specialization | See which models are experts for which regions |

### 7.3 Understanding Ensemble Performance

**vs Best Individual Column**:
- Positive value (+0.0234 >) indicates ensemble outperforms best single model
- Negative value (-0.0156 [X]) indicates best single model is better
- "Same" indicates equivalent performance

**RPD (Residual Prediction Deviation)**:
| RPD Value | Interpretation |
|-----------|----------------|
| < 1.5 | Poor predictive ability |
| 1.5 - 2.0 | Acceptable for rough screening |
| 2.0 - 2.5 | Good predictive ability |
| 2.5 - 3.0 | Very good predictive ability |
| > 3.0 | Excellent predictive ability |

---

## Chapter 8: Analysis Workflow Summary

### 8.1 Recommended Analysis Workflow

```
+-------------------+
| 1. Import Data    |
+-------------------+
         |
         v
+-------------------+
| 2. Configure      |
| Basic Settings    |
| - Target variable |
| - CV folds        |
| - Penalties       |
+-------------------+
         |
         v
+-------------------+
| 3. Select         |
| Preprocessing     |
| - Methods         |
| - Window sizes    |
+-------------------+
         |
         v
+-------------------+
| 4. Configure      |
| Variable Selection|
| (optional)        |
+-------------------+
         |
         v
+-------------------+
| 5. Select Models  |
| - Choose tier or  |
| - Custom selection|
+-------------------+
         |
         v
+-------------------+
| 6. Configure      |
| Ensembles         |
| (optional)        |
+-------------------+
         |
         v
+-------------------+
| 7. Set Up         |
| Validation        |
| (recommended)     |
+-------------------+
         |
         v
+-------------------+
| 8. Run Analysis   |
+-------------------+
         |
         v
+-------------------+
| 9. Review Results |
| - Compare models  |
| - Check ensembles |
+-------------------+
         |
         v
+-------------------+
| 10. Export/Save   |
| - Export CSV      |
| - Save best model |
+-------------------+
```

### 8.2 Quick Start Configuration

For a rapid first analysis:

1. **Basic Settings**: 5-fold CV, penalties = 0
2. **Preprocessing**: Raw, SNV, SG1 W=17
3. **Variable Selection**: Disabled (use full spectrum)
4. **Models**: Quick tier (PLS, RandomForest)
5. **Ensembles**: Disabled
6. **Validation**: Optional

**Expected Runtime**: 1-3 minutes depending on dataset size

### 8.3 Comprehensive Analysis Configuration

For thorough method comparison:

1. **Basic Settings**: 10-fold CV, penalties = 2/2
2. **Preprocessing**: All methods, windows 11/17/23
3. **Variable Selection**: Importance + SPA, Top-N (10, 50, 100, 250)
4. **Models**: Comprehensive tier
5. **Ensembles**: Enable all methods, 5 regions
6. **Validation**: 20% SPXY selection

**Expected Runtime**: 30-60 minutes depending on dataset size

### 8.4 Common Configuration Mistakes

| Mistake | Consequence | Solution |
|---------|-------------|----------|
| Too few CV folds | Unreliable performance estimates | Use 5+ folds for >50 samples |
| No validation set | Cannot assess true generalization | Enable 15-25% holdout |
| All models on small data | Overfitting, long runtime | Use Quick or Standard tier |
| Complex preprocessing on simple spectra | Introduces artifacts | Start with Raw/SNV |
| High penalties on exploration | Suppresses good complex models | Start with penalties = 0 |

---

## Chapter 9: Interpreting Results

### 9.1 Calibration vs Cross-Validation Metrics

| Metric Type | What It Shows | Caution |
|-------------|---------------|---------|
| Calibration (R2, RMSE) | Fit to training data | Can be overfit |
| Cross-Validation (R2cv, RMSEcv) | Estimated generalization | More reliable |
| Validation (R2pred, RMSEP) | True independent test | Most reliable |

**Overfitting Indicators**:
- R2 >> R2cv (calibration much better than CV)
- RMSE << RMSEcv (calibration much lower than CV)
- Results table shows strikethrough for overfit models

### 9.2 Comparing Models

**Key Metrics for Regression**:
1. R2cv - Higher is better (target > 0.90 for most applications)
2. RMSEcv - Lower is better (in target units)
3. RPD - Higher is better (target > 2.5 for quantitative applications)

**Key Metrics for Classification**:
1. Accuracycv - Higher is better (target depends on application)
2. F1cv - Higher is better (balances precision and recall)
3. ROC_AUCcv - Higher is better (target > 0.90)

### 9.3 Regional Performance

Understanding quartile-specific performance helps identify:
- Models that excel across all Y-value ranges (universal performers)
- Models that specialize in low or high Y values (specialists)
- Optimal ensemble compositions for different applications

### 9.4 Model Selection Guidelines

| Scenario | Recommended Model |
|----------|-------------------|
| Simple, interpretable model needed | PLS or Ridge |
| Highest accuracy, complexity acceptable | LightGBM or XGBoost |
| Robust to noise and outliers | RandomForest |
| Limited data (<50 samples) | PLS with few components |
| Classification with imbalanced classes | LightGBM with class weights |
| Need prediction uncertainty | RandomForest (can output variance) |

---

## Chapter 10: Troubleshooting Analysis Issues

### 10.1 Analysis Won't Start

| Issue | Cause | Solution |
|-------|-------|----------|
| "No data loaded" | Data not imported | Import data first |
| "No target variable" | Target column not selected | Select target in Basic Settings |
| "No models selected" | All model checkboxes unchecked | Select a tier or check models manually |
| "No preprocessing selected" | All preprocessing unchecked | Check at least Raw |

### 10.2 Poor Results

| Symptom | Possible Cause | Solution |
|---------|----------------|----------|
| All models R2cv < 0.5 | Weak signal in data | Check data quality, try different wavelength ranges |
| R2 high but R2cv low | Overfitting | Increase CV folds, add regularization |
| Large RMSE differences between models | Inconsistent preprocessing | Use consistent preprocessing across comparison |
| Classification accuracy at chance level | Misaligned labels | Verify target column values |

### 10.3 Slow Analysis

| Cause | Solution |
|-------|----------|
| Too many models | Use Quick or Standard tier |
| Many preprocessing combinations | Reduce window size options |
| Large dataset | Enable wavelength restriction |
| Variable selection enabled | Use faster methods (Importance, SPA) |
| NSGA-II/Bayesian with many trials | Reduce trials/generations |

### 10.4 Memory Issues

| Symptom | Solution |
|---------|----------|
| Application crashes during analysis | Reduce number of concurrent models |
| Out of memory errors | Close other applications, reduce dataset size |
| Slow scrolling in results | Export to CSV and view in spreadsheet application |

---

This concludes Part II: Running Analyses. Continue to Part III for detailed coverage of Model Development, Predictions, and Export functionality.
# Part III: Machine Learning Models Reference

This section provides comprehensive documentation of all machine learning models available in Spectral Predict V1. Each model is documented with its mathematical foundation, algorithm explanation, hyperparameters, and practical guidance for spectroscopy applications.

---

## Table of Contents

1. [Model Overview and Comparison](#model-overview-and-comparison)
2. [PLS (Partial Least Squares Regression)](#1-pls-partial-least-squares-regression)
3. [PLS-DA (Partial Least Squares Discriminant Analysis)](#2-pls-da-partial-least-squares-discriminant-analysis)
4. [Ridge Regression](#3-ridge-regression-l2-regularization)
5. [Lasso Regression](#4-lasso-regression-l1-regularization)
6. [ElasticNet](#5-elasticnet-combined-l1l2-regularization)
7. [Random Forest](#6-random-forest)
8. [XGBoost](#7-xgboost-extreme-gradient-boosting)
9. [LightGBM](#8-lightgbm-light-gradient-boosting-machine)
10. [CatBoost](#9-catboost-categorical-boosting)
11. [SVR/SVC (Support Vector Machines)](#10-svrsvc-support-vector-machines)
12. [MLP (Multi-Layer Perceptron)](#11-mlp-multi-layer-perceptron)
13. [NeuralBoosted](#12-neuralboosted)
14. [Model Selection Guidelines](#model-selection-guidelines)
15. [Hyperparameter Tuning Strategies](#hyperparameter-tuning-strategies)

---

## Model Overview and Comparison

### Model Categories

Spectral Predict organizes models into four categories based on their algorithmic approach:

| Category | Models | Characteristics |
|----------|--------|-----------------|
| **Latent Variable** | PLS, PLS-DA | Dimensionality reduction with maximum covariance |
| **Linear Regularized** | Ridge, Lasso, ElasticNet | Linear regression with penalty terms |
| **Ensemble Tree** | RandomForest, XGBoost, LightGBM, CatBoost | Decision tree ensembles |
| **Neural Network** | MLP, NeuralBoosted | Multilayer perceptrons and boosted networks |
| **Kernel Methods** | SVR, SVC | Support vector machines with kernel transformations |

### Model Tiers in Spectral Predict

Models are organized into tiers for different use cases:

| Tier | Regression Models | Classification Models | Use Case |
|------|-------------------|----------------------|----------|
| **Quick** | PLS, RandomForest | PLS-DA, RandomForest | Rapid testing (3-5 min) |
| **Standard** | PLS, Ridge, Lasso, ElasticNet, RandomForest, LightGBM | PLS-DA, RandomForest, LightGBM, XGBoost, CatBoost | Daily analysis (10-15 min) |
| **Comprehensive** | PLS, Ridge, ElasticNet, RandomForest, LightGBM, XGBoost, CatBoost, NeuralBoosted | PLS-DA, RandomForest, LightGBM, XGBoost, CatBoost, SVM, MLP, NeuralBoosted | Research (20-30 min) |
| **Experimental** | All models | All models | Method exploration (45+ min) |

### Quick Comparison Table

| Model | Type | Interpretable | Handles Collinearity | Feature Selection | Training Speed | Prediction Speed |
|-------|------|---------------|---------------------|-------------------|----------------|------------------|
| PLS | Regression | High | Excellent | Indirect (VIP) | Fast | Very Fast |
| PLS-DA | Classification | High | Excellent | Indirect (VIP) | Fast | Fast |
| Ridge | Regression | High | Good | No | Very Fast | Very Fast |
| Lasso | Regression | High | Good | Yes (Sparse) | Fast | Very Fast |
| ElasticNet | Regression | High | Very Good | Yes (Sparse) | Fast | Very Fast |
| RandomForest | Both | Medium | Good | Yes | Medium | Fast |
| XGBoost | Both | Medium | Good | Yes | Medium | Fast |
| LightGBM | Both | Medium | Good | Yes | Fast | Fast |
| CatBoost | Both | Medium | Good | Yes | Medium | Fast |
| SVR/SVC | Both | Low | Excellent | No | Slow | Medium |
| MLP | Both | Low | Good | No | Medium | Fast |
| NeuralBoosted | Both | Low | Good | Yes | Medium | Medium |

---

## 1. PLS (Partial Least Squares Regression)

### Full Name and Abbreviation

**Partial Least Squares Regression (PLSR)**, also known as **Projection to Latent Structures**

### Mathematical Foundation

PLS is a dimensionality reduction technique that finds latent variables (components) that maximize the covariance between the predictor matrix $X$ and the response $Y$. Unlike Principal Component Analysis (PCA), which only considers variance in $X$, PLS explicitly models the relationship between $X$ and $Y$.

The fundamental decomposition is:

$$X = TP^T + E$$

$$Y = UQ^T + F$$

Where:
- $X$ is the $(n \times p)$ predictor matrix (spectra)
- $Y$ is the $(n \times 1)$ response vector (target property)
- $T$ is the $(n \times A)$ X-scores matrix
- $P$ is the $(p \times A)$ X-loadings matrix
- $U$ is the $(n \times A)$ Y-scores matrix
- $Q$ is the $(1 \times A)$ Y-loadings matrix
- $E$ and $F$ are residual matrices
- $A$ is the number of latent variables (components)

The algorithm maximizes:

$$\max_{w, c} \text{cov}(Xw, Yc)^2$$

subject to $||w|| = ||c|| = 1$

### NIPALS Algorithm

Spectral Predict uses the NIPALS (Nonlinear Iterative Partial Least Squares) algorithm:

1. **Initialize**: Set $u = y$ (first column of $Y$)
2. **Calculate X-weights**: $w = X^T u / (u^T u)$, normalize $||w|| = 1$
3. **Calculate X-scores**: $t = Xw$
4. **Calculate Y-loadings**: $q = Y^T t / (t^T t)$, normalize $||q|| = 1$
5. **Calculate Y-scores**: $u = Yq / (q^T q)$
6. **Check convergence**: If $||t_{new} - t_{old}|| < \text{tol}$, continue to step 7
7. **Deflate matrices**: $X = X - tp^T$ where $p = X^T t / (t^T t)$
8. **Repeat** for next component

### Key Hyperparameters

| Parameter | Type | Range | Default | Effect |
|-----------|------|-------|---------|--------|
| `n_components` | int | 1-20 | Auto | Number of latent variables to extract |
| `max_iter` | int | 100-1000 | 500 | Maximum NIPALS iterations per component |
| `tol` | float | 1e-8 to 1e-4 | 1e-6 | Convergence tolerance |
| `scale` | bool | True/False | False | Whether to scale X to unit variance |

**Selecting Optimal n_components:**

The number of components is the most critical hyperparameter. Guidelines:

1. **Cross-validation**: Use the value that minimizes RMSECV
2. **Parsimony rule**: If two models have similar RMSECV, prefer fewer components
3. **Practical limits**: Rarely need more than 10-15 components for spectroscopy
4. **Information criterion**: Look for "elbow" in explained variance plot

$$\text{Optimal } A = \arg\min_a \text{RMSECV}(a)$$

### Variable Importance in Projection (VIP)

VIP scores quantify each variable's contribution to the PLS model:

$$VIP_j = \sqrt{p \sum_{a=1}^{A} \frac{w_{aj}^2 \cdot SSY_a}{SSY_{total}}}$$

Where:
- $p$ = number of predictor variables (wavelengths)
- $w_{aj}$ = weight of variable $j$ in component $a$
- $SSY_a$ = sum of squares of $Y$ explained by component $a$
- $SSY_{total}$ = total sum of squares of $Y$

**Interpretation:**
- VIP > 1: Variable is important for the model
- VIP < 0.8: Variable may be a candidate for removal
- VIP = 1: Average contribution

### When to Use PLS

**Strengths:**
- Designed specifically for spectroscopy applications
- Handles severe multicollinearity in spectral data
- Works when $p >> n$ (more wavelengths than samples)
- Provides interpretable latent variables
- Fast training and prediction
- VIP scores enable variable selection

**Weaknesses:**
- Assumes linear relationships between spectra and properties
- May underperform for highly non-linear relationships
- Requires careful selection of n_components
- Sensitive to outliers in training data

**Best for:**
- First-pass modeling of any spectroscopy problem
- When interpretability is important
- Small sample sizes (n < 100)
- Real-time predictions where speed matters

### Spectroscopy-Specific Considerations

1. **Preprocessing**: PLS often works best with SNV or derivatives to remove baseline effects
2. **Number of components**: Start with cross-validation across 1-15 components
3. **Overfitting risk**: More components = better training fit but potential overfitting
4. **Wavelength selection**: Use VIP scores to identify important spectral regions
5. **Outlier sensitivity**: Consider robust PLS variants for contaminated data

### Implementation Details

```python
# Spectral Predict uses sklearn's PLSRegression
from sklearn.cross_decomposition import PLSRegression

model = PLSRegression(
    n_components=n_components,  # Tested 1 to max_n_components
    max_iter=500,               # Default
    tol=1e-6,                   # Default
    scale=False                 # Preprocessing handled separately
)
```

---

## 2. PLS-DA (Partial Least Squares Discriminant Analysis)

### Full Name and Abbreviation

**Partial Least Squares Discriminant Analysis (PLS-DA)**

### Mathematical Foundation

PLS-DA extends PLS for classification by treating class labels as a continuous response variable. The approach uses regression to predict class membership, then applies a decision rule.

For binary classification with classes $\{0, 1\}$:

$$Y = TP^T + E$$

Where $Y$ is encoded as 0 or 1, and the prediction $\hat{y}$ is converted to class labels:

$$\hat{c} = \begin{cases} 1 & \text{if } \hat{y} > \tau \\ 0 & \text{otherwise} \end{cases}$$

The threshold $\tau$ is typically 0.5 but can be optimized for imbalanced classes.

### Algorithm Explanation

Spectral Predict implements PLS-DA as a two-stage pipeline:

**Stage 1: PLS Transformation**
- Extract latent variables that maximize covariance with class labels
- Transform high-dimensional spectra to low-dimensional scores

**Stage 2: Logistic Regression Classification**
- Use PLS scores as input features
- Apply logistic regression for probabilistic classification

$$P(y=1|T) = \frac{1}{1 + e^{-(\beta_0 + \beta^T T)}}$$

This approach provides:
- Probability estimates for each class
- Better handling of multiclass problems
- More robust decision boundaries

### Key Hyperparameters

**PLS Transformation Parameters:**

| Parameter | Type | Range | Default | Effect |
|-----------|------|-------|---------|--------|
| `n_components` | int | 1-20 | Auto | Number of PLS components |
| `max_iter` | int | 100-1000 | 500 | NIPALS iterations |
| `tol` | float | 1e-8 to 1e-4 | 1e-6 | Convergence tolerance |

**Logistic Regression Parameters:**

| Parameter | Type | Range | Default | Effect |
|-----------|------|-------|---------|--------|
| `lr_C` | float | 0.001-1000 | 1.0 | Inverse regularization strength |
| `lr_solver` | str | lbfgs, saga | lbfgs | Optimization algorithm |
| `lr_max_iter` | int | 100-5000 | 1000 | Maximum iterations |

### Threshold Setting

For imbalanced classes, optimizing the classification threshold can improve performance:

$$\tau_{opt} = \arg\max_\tau F_1(\tau)$$

Or using Youden's J statistic:

$$\tau_{opt} = \arg\max_\tau (\text{Sensitivity} + \text{Specificity} - 1)$$

### When to Use PLS-DA

**Strengths:**
- Excellent for spectroscopic classification
- Handles multicollinearity naturally
- Works with $p >> n$
- Provides class probabilities
- VIP scores identify discriminating wavelengths

**Weaknesses:**
- Assumes linear class boundaries in latent space
- Sensitive to class imbalance
- May struggle with highly non-linear class separations
- Requires careful component selection

**Best for:**
- Material identification
- Authenticity/adulteration detection
- Quality grade classification
- Any classification where interpretability matters

### Spectroscopy-Specific Considerations

1. **Class balance**: Check class distribution; use stratified sampling
2. **Component selection**: Cross-validate accuracy, not RMSE
3. **Multiclass**: Use one-vs-rest or softmax extension
4. **Preprocessing**: Same considerations as PLS regression
5. **Validation**: Use confusion matrix, not just accuracy

---

## 3. Ridge Regression (L2 Regularization)

### Full Name and Abbreviation

**Ridge Regression**, also known as **Tikhonov Regularization** or **L2-Penalized Regression**

### Mathematical Foundation

Ridge regression adds an L2 penalty to ordinary least squares to control model complexity:

$$\min_w ||Xw - y||_2^2 + \alpha||w||_2^2$$

The closed-form solution is:

$$\hat{w} = (X^T X + \alpha I)^{-1} X^T y$$

Where:
- $X$ is the $(n \times p)$ design matrix
- $y$ is the $(n \times 1)$ response vector
- $w$ is the $(p \times 1)$ coefficient vector
- $\alpha$ is the regularization parameter
- $I$ is the identity matrix

### Algorithm Explanation

The L2 penalty has several effects:

1. **Shrinkage**: Coefficients are shrunk toward zero
2. **Stabilization**: $(X^T X + \alpha I)$ is always invertible
3. **Bias-variance tradeoff**: Adds bias but reduces variance

The amount of shrinkage depends on eigenvalue structure:

$$\hat{w}_j^{ridge} = \frac{d_j^2}{d_j^2 + \alpha} \hat{w}_j^{OLS}$$

Where $d_j$ are singular values of $X$. Small eigenvalues (multicollinearity) are heavily regularized.

### Key Hyperparameters

| Parameter | Type | Range | Default | Effect |
|-----------|------|-------|---------|--------|
| `alpha` | float | 0.001-1000 | 1.0 | Regularization strength |
| `solver` | str | auto, svd, cholesky, lsqr, sparse_cg, sag, saga | auto | Optimization algorithm |
| `tol` | float | 1e-6 to 1e-3 | 1e-4 | Convergence tolerance |

**Alpha Selection:**

| Alpha Range | Effect | Best When |
|-------------|--------|-----------|
| 0.001-0.1 | Minimal regularization | Good signal-to-noise |
| 0.1-10 | Moderate regularization | Typical spectroscopy |
| 10-1000 | Heavy regularization | Severe multicollinearity |

### When to Use Ridge Regression

**Strengths:**
- Very fast training (closed-form solution)
- Handles multicollinearity well
- All features retained (no variable selection)
- Stable predictions
- Interpretable coefficients

**Weaknesses:**
- No automatic feature selection
- Coefficients never exactly zero
- Assumes linear relationship
- All wavelengths contribute to prediction

**Best for:**
- When all spectral features are potentially relevant
- Fast initial modeling
- When feature selection is done separately
- Baseline model for comparison

### Spectroscopy-Specific Considerations

1. **Scaling**: Ridge assumes features are on similar scales; standardize if not preprocessing
2. **Wavelength importance**: Use absolute coefficient values as importance proxy
3. **Comparison to PLS**: Ridge often performs similarly but without latent variable interpretation
4. **Large p**: Solver='svd' is numerically stable for high-dimensional data

---

## 4. Lasso Regression (L1 Regularization)

### Full Name and Abbreviation

**Lasso Regression** (Least Absolute Shrinkage and Selection Operator), also known as **L1-Penalized Regression**

### Mathematical Foundation

Lasso uses an L1 penalty that induces sparsity in coefficients:

$$\min_w \frac{1}{2n}||Xw - y||_2^2 + \alpha||w||_1$$

Where:

$$||w||_1 = \sum_{j=1}^{p} |w_j|$$

The L1 penalty creates a non-smooth optimization problem with no closed-form solution.

### Algorithm Explanation

Lasso's key property is **automatic feature selection**:

1. The L1 penalty creates corners in the constraint region
2. Optimal solutions often lie at corners
3. At corners, some coefficients are exactly zero

**Coordinate Descent Solution:**

For each coefficient $w_j$:

$$w_j = S_{\alpha/n}\left(\frac{1}{n}\sum_{i=1}^{n} x_{ij}(y_i - \hat{y}_i^{(-j)})\right)$$

Where $S_\lambda$ is the soft-thresholding operator:

$$S_\lambda(z) = \begin{cases} z - \lambda & \text{if } z > \lambda \\ 0 & \text{if } |z| \leq \lambda \\ z + \lambda & \text{if } z < -\lambda \end{cases}$$

### Key Hyperparameters

| Parameter | Type | Range | Default | Effect |
|-----------|------|-------|---------|--------|
| `alpha` | float | 0.001-10 | 1.0 | Regularization strength |
| `selection` | str | cyclic, random | cyclic | Coordinate selection method |
| `tol` | float | 1e-6 to 1e-3 | 1e-4 | Convergence tolerance |
| `max_iter` | int | 100-5000 | 500 | Maximum iterations |

**Alpha Effect on Sparsity:**

| Alpha | Typical Sparsity | Effect |
|-------|------------------|--------|
| 0.001 | ~10% zeros | Minimal selection |
| 0.01 | ~30% zeros | Light selection |
| 0.1 | ~60% zeros | Moderate selection |
| 1.0 | ~90% zeros | Heavy selection |

### When to Use Lasso Regression

**Strengths:**
- Automatic wavelength selection
- Sparse, interpretable models
- Identifies most important spectral regions
- Handles multicollinearity (by selection)
- Fast prediction with sparse model

**Weaknesses:**
- Unstable selection with correlated features
- May select only one of correlated wavelengths arbitrarily
- Can over-regularize important features
- No closed-form solution (iterative)

**Best for:**
- When you want to identify key wavelengths
- Building parsimonious models
- Understanding which spectral regions matter
- Reducing model complexity for deployment

### Spectroscopy-Specific Considerations

1. **Correlated wavelengths**: Lasso may arbitrarily select one of correlated bands
2. **Group selection**: Adjacent wavelengths often correlate; consider group lasso
3. **Stability selection**: Run Lasso multiple times on bootstrap samples
4. **Comparison to VIP**: Different selection mechanisms; compare results
5. **Physical interpretation**: Selected wavelengths should match known absorption bands

---

## 5. ElasticNet (Combined L1/L2 Regularization)

### Full Name and Abbreviation

**Elastic Net Regularization**, combining L1 (Lasso) and L2 (Ridge) penalties

### Mathematical Foundation

ElasticNet combines both penalties to achieve the benefits of each:

$$\min_w \frac{1}{2n}||Xw - y||_2^2 + \alpha \rho ||w||_1 + \frac{\alpha(1-\rho)}{2}||w||_2^2$$

Where:
- $\alpha$ controls overall regularization strength
- $\rho$ (l1_ratio) controls the mix: $\rho=1$ is Lasso, $\rho=0$ is Ridge

The naive elastic net solution is:

$$\hat{w}^{enet} = \frac{1}{1 + \alpha(1-\rho)} \hat{w}^{naive}$$

### Algorithm Explanation

ElasticNet provides a "grouping effect" not present in pure Lasso:

1. **Group selection**: Correlated features tend to be selected together
2. **Stability**: More stable than Lasso with correlated predictors
3. **Shrinkage + Selection**: Combines both mechanisms

The optimization uses coordinate descent with modified soft-thresholding:

$$w_j = \frac{S_{\alpha\rho}(z_j)}{1 + \alpha(1-\rho)}$$

Where $z_j$ is the partial residual correlation.

### Key Hyperparameters

| Parameter | Type | Range | Default | Effect |
|-----------|------|-------|---------|--------|
| `alpha` | float | 0.01-10 | 1.0 | Overall regularization strength |
| `l1_ratio` | float | 0.0-1.0 | 0.5 | Mix between L1 and L2 |
| `selection` | str | cyclic, random | cyclic | Coordinate selection |
| `tol` | float | 1e-6 to 1e-3 | 1e-4 | Convergence tolerance |

**l1_ratio Guidelines:**

| l1_ratio | Behavior | Best When |
|----------|----------|-----------|
| 0.1-0.3 | Ridge-like, grouped selection | Strong multicollinearity |
| 0.4-0.6 | Balanced | General spectroscopy |
| 0.7-0.9 | Lasso-like, sparse | Feature selection emphasis |

### When to Use ElasticNet

**Strengths:**
- Best of both Ridge and Lasso
- Selects groups of correlated features
- More stable than Lasso
- Handles multicollinearity better than Lasso
- Still provides sparse solutions

**Weaknesses:**
- Two hyperparameters to tune
- Computationally slower than Ridge
- Less interpretable than pure Lasso

**Best for:**
- Spectral data with correlated wavelength groups
- When Lasso is unstable but selection is desired
- Balancing prediction accuracy and interpretability

### Spectroscopy-Specific Considerations

1. **Adjacent wavelengths**: ElasticNet tends to select bands of consecutive wavelengths
2. **Chemical interpretation**: Grouped selection often matches actual absorption bands
3. **Preprocessing interaction**: Derivatives may reduce the advantage over Lasso
4. **Hyperparameter search**: Use 2D grid search over (alpha, l1_ratio)
5. **Default recommendation**: Start with l1_ratio=0.5, alpha=0.1

---

## 6. Random Forest

### Full Name and Abbreviation

**Random Forest (RF)**, an ensemble of decision trees with bagging and feature randomization

### Mathematical Foundation

Random Forest builds an ensemble of decorrelated decision trees:

$$\hat{f}(x) = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$

Where:
- $B$ = number of trees (n_estimators)
- $T_b(x)$ = prediction of tree $b$

Each tree is trained on:
1. **Bootstrap sample**: Random sample with replacement
2. **Random feature subset**: $m$ features at each split ($m \approx \sqrt{p}$ for classification, $m \approx p/3$ for regression)

### Algorithm Explanation

**Building a Single Tree:**

1. Draw bootstrap sample of size $n$ from training data
2. At each node:
   - Randomly select $m$ features
   - Find best split among these $m$ features
   - Split node into two children
3. Grow tree to maximum depth (no pruning)

**Split Criterion (Regression):**

$$\text{MSE} = \frac{1}{n_L} \sum_{i \in L} (y_i - \bar{y}_L)^2 + \frac{1}{n_R} \sum_{i \in R} (y_i - \bar{y}_R)^2$$

**Feature Importance:**

Mean Decrease in Impurity (MDI):

$$\text{Importance}_j = \frac{1}{B} \sum_{b=1}^{B} \sum_{t \in T_b} \mathbb{1}(j_t = j) \cdot \Delta\text{Impurity}_t$$

### Key Hyperparameters

| Parameter | Type | Range | Default | Effect |
|-----------|------|-------|---------|--------|
| `n_estimators` | int | 100-1000 | 200 | Number of trees |
| `max_depth` | int/None | 5-None | None | Maximum tree depth |
| `max_features` | str/float | sqrt, log2, 0.1-1.0 | sqrt | Features per split |
| `min_samples_split` | int | 2-20 | 2 | Minimum samples to split |
| `min_samples_leaf` | int | 1-10 | 1 | Minimum samples in leaf |
| `bootstrap` | bool | True/False | True | Use bootstrap sampling |
| `max_leaf_nodes` | int/None | 10-None | None | Maximum leaf nodes |

**Effect of Key Parameters:**

| Parameter | Low Value | High Value |
|-----------|-----------|------------|
| n_estimators | Faster, less stable | Slower, more stable |
| max_depth | Underfitting | Overfitting |
| max_features | More diversity | Better individual trees |
| min_samples_split | More splits, overfitting | Fewer splits, underfitting |

### When to Use Random Forest

**Strengths:**
- Handles non-linear relationships
- Robust to outliers
- Built-in feature importance
- Minimal hyperparameter tuning needed
- Works well out-of-box
- Parallelizable

**Weaknesses:**
- Less interpretable than linear models
- Can overfit with deep trees
- Memory intensive with many trees
- May struggle with extrapolation
- Feature importance can be biased

**Best for:**
- Non-linear spectral responses
- Quick baseline with good performance
- When interpretability is secondary
- Large datasets with complex patterns

### Spectroscopy-Specific Considerations

1. **max_features**: Use 'sqrt' for classification, 1.0 (all) for regression with spectral data
2. **Feature importance**: Provides wavelength importance, but may be biased toward continuous features
3. **Extrapolation**: RF cannot extrapolate beyond training range; be cautious with predictions outside calibration range
4. **High dimensions**: Works well with thousands of wavelengths
5. **Preprocessing**: Often performs well without preprocessing; experiment with raw vs. preprocessed

---

## 7. XGBoost (Extreme Gradient Boosting)

### Full Name and Abbreviation

**XGBoost (eXtreme Gradient Boosting)**, a regularized gradient boosting framework

### Mathematical Foundation

XGBoost optimizes a regularized objective function:

$$\mathcal{L} = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)$$

Where the regularization term is:

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda ||w||^2 + \alpha||w||_1$$

Where:
- $l(y_i, \hat{y}_i)$ = loss function (MSE for regression, log-loss for classification)
- $T$ = number of leaves in tree
- $w$ = leaf weights
- $\gamma$ = complexity penalty for tree structure
- $\lambda$ = L2 regularization on leaf weights
- $\alpha$ = L1 regularization on leaf weights

### Algorithm Explanation

XGBoost builds trees sequentially, each fitting the gradient of the loss:

**Gradient Boosting Update:**

$$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta \cdot f_t(x_i)$$

Where $\eta$ is the learning rate and $f_t$ is the new tree.

**Newton-Raphson Optimization:**

XGBoost uses second-order gradients for better optimization:

$$g_i = \frac{\partial l(y_i, \hat{y}_i)}{\partial \hat{y}_i}, \quad h_i = \frac{\partial^2 l(y_i, \hat{y}_i)}{\partial \hat{y}_i^2}$$

**Optimal Leaf Weight:**

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

**Split Gain:**

$$\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma$$

### Key Hyperparameters

| Parameter | Type | Range | Default | Effect |
|-----------|------|-------|---------|--------|
| `n_estimators` | int | 50-500 | 100 | Number of boosting rounds |
| `learning_rate` | float | 0.01-0.3 | 0.1 | Step size shrinkage |
| `max_depth` | int | 3-10 | 6 | Maximum tree depth |
| `subsample` | float | 0.5-1.0 | 0.8 | Row sampling ratio |
| `colsample_bytree` | float | 0.5-1.0 | 0.6 | Column sampling per tree |
| `reg_alpha` | float | 0-1.0 | 0.2 | L1 regularization |
| `reg_lambda` | float | 0.5-2.0 | 1.5 | L2 regularization |
| `min_child_weight` | float | 1-10 | 1 | Minimum sum of instance weight in child |
| `gamma` | float | 0-0.5 | 0 | Minimum loss reduction to split |

**Parameter Interactions:**

| Overfitting | Underfitting |
|-------------|--------------|
| Decrease learning_rate | Increase learning_rate |
| Increase reg_alpha, reg_lambda | Decrease regularization |
| Decrease max_depth | Increase max_depth |
| Increase subsample | Decrease subsample |

### When to Use XGBoost

**Strengths:**
- State-of-the-art accuracy for tabular data
- Built-in regularization
- Handles missing values
- Feature importance measures
- Fast with histogram-based splitting
- Handles imbalanced data

**Weaknesses:**
- Many hyperparameters to tune
- Can overfit without careful tuning
- Less interpretable than linear models
- Sensitive to hyperparameter settings

**Best for:**
- Maximum predictive accuracy
- Complex non-linear relationships
- When regularization is important
- Kaggle-style competitive modeling

### Spectroscopy-Specific Considerations

1. **colsample_bytree**: Lower values (0.6-0.8) help with highly correlated wavelengths
2. **reg_alpha**: L1 regularization helps with feature selection
3. **tree_method**: 'hist' is faster for high-dimensional spectral data
4. **max_depth**: Keep moderate (3-6) to prevent overfitting
5. **Early stopping**: Monitor validation loss to prevent overfitting

---

## 8. LightGBM (Light Gradient Boosting Machine)

### Full Name and Abbreviation

**LightGBM (Light Gradient Boosting Machine)**, a fast gradient boosting implementation by Microsoft

### Mathematical Foundation

LightGBM uses the same gradient boosting framework as XGBoost but with key algorithmic innovations:

$$\hat{y}^{(t)} = \hat{y}^{(t-1)} + \eta \cdot f_t(x)$$

**Gradient-based One-Side Sampling (GOSS):**

Keeps all instances with large gradients, randomly samples those with small gradients:

$$\text{Weight amplification} = \frac{1 - a}{b}$$

Where $a$ = fraction of large gradient instances, $b$ = fraction of small gradient instances.

### Algorithm Explanation

**Leaf-wise (Best-first) Tree Growth:**

Unlike XGBoost's level-wise growth, LightGBM grows leaf-wise:
- Choose the leaf with maximum delta loss
- Split that leaf
- Continue until num_leaves reached

This is more efficient but requires `num_leaves` control to prevent overfitting.

**Histogram-based Splitting:**

1. Bucket continuous features into discrete bins
2. Compute histograms for gradient statistics
3. Find best split from histograms

Benefits:
- $O(data \times features)$ to $O(data + 2^{bins} \times features)$
- Cache-efficient memory access
- Faster than exact algorithms

### Key Hyperparameters

| Parameter | Type | Range | Default | Effect |
|-----------|------|-------|---------|--------|
| `n_estimators` | int | 50-500 | 100 | Number of boosting rounds |
| `learning_rate` | float | 0.01-0.3 | 0.1 | Step size shrinkage |
| `num_leaves` | int | 15-127 | 15 | Maximum number of leaves per tree |
| `max_depth` | int | -1 to 15 | 10 | Maximum tree depth (-1 = unlimited) |
| `min_child_samples` | int | 5-100 | 5 | Minimum samples in leaf |
| `subsample` | float | 0.5-1.0 | 0.8 | Row sampling ratio |
| `colsample_bytree` | float | 0.5-1.0 | 0.8 | Column sampling per tree |
| `reg_alpha` | float | 0-1.0 | 0.3 | L1 regularization |
| `reg_lambda` | float | 0.5-2.0 | 1.5 | L2 regularization |

**num_leaves vs max_depth:**

Unlike XGBoost, LightGBM primarily controls complexity via `num_leaves`:

$$\text{num\_leaves} \approx 2^{\text{max\_depth}}$$

For spectral data, use `num_leaves = 15-31` to prevent overfitting.

### When to Use LightGBM

**Strengths:**
- Fastest gradient boosting implementation
- Memory efficient
- Excellent accuracy
- Handles large datasets well
- Good for high-dimensional data
- Built-in categorical feature handling

**Weaknesses:**
- Can overfit with small datasets
- num_leaves requires careful tuning
- Less mature than XGBoost
- Sensitive to outliers

**Best for:**
- Large spectral datasets
- When training speed matters
- High-dimensional data
- Production deployment

### Spectroscopy-Specific Considerations

1. **num_leaves**: Keep low (15-31) for spectral data to prevent overfitting
2. **min_child_samples**: Set higher (5-20) for small spectral datasets
3. **Speed advantage**: Most pronounced with >1000 samples and >500 wavelengths
4. **Bagging**: Enable subsample with bagging_freq=1 for regularization
5. **verbosity=-1**: Suppress warnings during grid search

---

## 9. CatBoost (Categorical Boosting)

### Full Name and Abbreviation

**CatBoost (Categorical Boosting)**, gradient boosting with ordered boosting by Yandex

### Mathematical Foundation

CatBoost introduces "ordered boosting" to prevent target leakage:

**Ordered Boosting:**

For each sample $i$, residuals are calculated using a model trained only on samples that come before $i$ in a random permutation:

$$r_i^{(t)} = y_i - M^{(t)}_{i-1}(x_i)$$

Where $M^{(t)}_{i-1}$ is trained on samples $\{1, ..., i-1\}$ in the current permutation.

This prevents the gradient bias that can occur in standard gradient boosting.

### Algorithm Explanation

**Symmetric Trees:**

CatBoost uses oblivious (symmetric) decision trees:
- Same splitting criterion at each level
- All nodes at same depth use identical split
- Creates balanced trees with $2^d$ leaves for depth $d$

Benefits:
- Faster inference (SIMD-friendly)
- Natural regularization
- Interpretable structure

**Gradient Estimation:**

Uses ordered statistics to compute unbiased gradient estimates, reducing overfitting.

### Key Hyperparameters

| Parameter | Type | Range | Default | Effect |
|-----------|------|-------|---------|--------|
| `iterations` | int | 50-500 | 100 | Number of trees |
| `learning_rate` | float | 0.01-0.3 | 0.1 | Step size shrinkage |
| `depth` | int | 4-10 | 5 | Tree depth |
| `l2_leaf_reg` | float | 1-10 | 4.0 | L2 regularization on leaves |
| `min_data_in_leaf` | int | 1-20 | 1 | Minimum samples in leaf |
| `bootstrap_type` | str | Bayesian, Bernoulli, MVS | Bayesian | Sampling method |
| `bagging_temperature` | float | 0-10 | 1.0 | Bayesian bootstrap parameter |
| `random_strength` | float | 0-10 | 1.0 | Score randomization strength |
| `border_count` | int | 32-255 | 128 | Number of splits for numerical features |

**bootstrap_type Options:**

| Type | Description | Best For |
|------|-------------|----------|
| Bayesian | Weights from exponential distribution | Default, good regularization |
| Bernoulli | Random subset of samples | Large datasets |
| MVS | Minimum variance sampling | Small datasets |

### When to Use CatBoost

**Strengths:**
- Best handling of categorical features
- Ordered boosting reduces overfitting
- Symmetric trees are fast at inference
- Robust to hyperparameter settings
- GPU acceleration available
- Built-in cross-validation

**Weaknesses:**
- Slower training than LightGBM
- Memory intensive
- Less flexible tree structure

**Best for:**
- Mixed numerical/categorical data
- When overfitting is a concern
- Production deployment (fast inference)
- When minimal tuning time available

### Spectroscopy-Specific Considerations

1. **depth**: Use 4-6 for spectral data; deeper trees overfit
2. **Categorical handling**: Useful if metadata (sample type, operator) included
3. **Ordered boosting**: Provides natural regularization for small datasets
4. **l2_leaf_reg**: Increase for more regularization (try 3.0-5.0)
5. **silent mode**: Set verbose=False to suppress training output

---

## 10. SVR/SVC (Support Vector Machines)

### Full Name and Abbreviation

**Support Vector Regression (SVR)** and **Support Vector Classification (SVC)**

### Mathematical Foundation

SVMs find a hyperplane that maximizes the margin while minimizing error:

**Classification (SVC):**

$$\min_{w, b} \frac{1}{2}||w||^2 + C\sum_{i=1}^{n} \xi_i$$

Subject to: $y_i(w^T x_i + b) \geq 1 - \xi_i$, $\xi_i \geq 0$

**Regression (SVR):**

$$\min_{w, b} \frac{1}{2}||w||^2 + C\sum_{i=1}^{n} (\xi_i + \xi_i^*)$$

Subject to: $|y_i - (w^T x_i + b)| \leq \epsilon + \xi_i$

### The Kernel Trick

SVMs use kernels to operate in high-dimensional spaces without explicit computation:

$$K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$$

**Common Kernels:**

**Linear:**
$$K(x_i, x_j) = x_i^T x_j$$

**Radial Basis Function (RBF):**
$$K(x_i, x_j) = \exp(-\gamma||x_i - x_j||^2)$$

**Polynomial:**
$$K(x_i, x_j) = (\gamma x_i^T x_j + r)^d$$

### Algorithm Explanation

The dual formulation uses Lagrange multipliers:

$$\max_\alpha \sum_{i=1}^{n} \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$

Subject to: $0 \leq \alpha_i \leq C$, $\sum_i \alpha_i y_i = 0$

Predictions use support vectors (samples with $\alpha_i > 0$):

$$f(x) = \sum_{i \in SV} \alpha_i y_i K(x_i, x) + b$$

### Key Hyperparameters

| Parameter | Type | Range | Default | Effect |
|-----------|------|-------|---------|--------|
| `C` | float | 0.1-1000 | 1.0 | Regularization (inverse) |
| `kernel` | str | rbf, linear, poly | rbf | Kernel function |
| `gamma` | float/str | 0.001-1.0 or 'scale' | 'scale' | RBF kernel width |
| `epsilon` | float | 0.01-0.5 | 0.1 | SVR tube width |
| `degree` | int | 2-5 | 3 | Polynomial degree |
| `coef0` | float | 0-1 | 0 | Polynomial/sigmoid intercept |
| `shrinking` | bool | True/False | True | Use shrinking heuristic |
| `max_iter` | int | 1000-10000 | 5000 | Maximum iterations |

**Kernel Selection:**

| Kernel | Complexity | Best For |
|--------|------------|----------|
| linear | Low | High-dimensional, linearly separable |
| rbf | Medium | General non-linear, default choice |
| poly | High | Polynomial relationships |

### When to Use SVM

**Strengths:**
- Effective in high dimensions
- Memory efficient (uses support vectors only)
- Versatile through kernel selection
- Robust to overfitting with proper C
- Works well with clear margin of separation

**Weaknesses:**
- Slow training for large datasets
- Sensitive to feature scaling
- Kernel selection requires tuning
- No probability estimates without extra computation (classification)
- Poor with noisy data

**Best for:**
- Small to medium datasets (n < 1000)
- Clear class separation
- When memory is limited
- Non-linear relationships with RBF

### Spectroscopy-Specific Considerations

1. **Scaling**: Essential! Use StandardScaler or SNV preprocessing
2. **Kernel choice**: RBF works well for most spectral problems
3. **gamma='scale'**: Good default (1 / (n_features * X.var()))
4. **Training time**: Slow for large spectral datasets; consider subsampling
5. **max_iter**: Increase to 5000+ to ensure convergence

---

## 11. MLP (Multi-Layer Perceptron)

### Full Name and Abbreviation

**Multi-Layer Perceptron (MLP)**, a feedforward artificial neural network

### Mathematical Foundation

An MLP with $L$ hidden layers computes:

**Forward Propagation:**

$$a^{(0)} = x$$
$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$$
$$a^{(l)} = g(z^{(l)})$$
$$\hat{y} = a^{(L+1)}$$

Where:
- $W^{(l)}$ = weight matrix for layer $l$
- $b^{(l)}$ = bias vector for layer $l$
- $g(\cdot)$ = activation function

**Loss Function (Regression):**

$$\mathcal{L} = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \frac{\alpha}{2}\sum_l ||W^{(l)}||_F^2$$

### Algorithm Explanation

**Activation Functions:**

**ReLU (Rectified Linear Unit):**
$$g(z) = \max(0, z)$$

**Tanh:**
$$g(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

**Logistic (Sigmoid):**
$$g(z) = \frac{1}{1 + e^{-z}}$$

**Backpropagation:**

Gradients are computed by chain rule:

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T$$

Where $\delta^{(l)}$ is the error term for layer $l$.

**Optimization (Adam):**

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = m_t / (1 - \beta_1^t)$$
$$\hat{v}_t = v_t / (1 - \beta_2^t)$$
$$\theta_t = \theta_{t-1} - \eta \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$$

### Key Hyperparameters

| Parameter | Type | Range | Default | Effect |
|-----------|------|-------|---------|--------|
| `hidden_layer_sizes` | tuple | (16,) to (256, 128) | (64,) | Network architecture |
| `alpha` | float | 1e-5 to 0.1 | 0.001 | L2 regularization |
| `learning_rate_init` | float | 1e-4 to 0.1 | 0.001 | Initial learning rate |
| `activation` | str | relu, tanh, logistic | relu | Activation function |
| `solver` | str | adam, sgd, lbfgs | adam | Optimization algorithm |
| `batch_size` | int/str | 32-256 or 'auto' | 'auto' | Mini-batch size |
| `learning_rate` | str | constant, invscaling, adaptive | constant | Learning rate schedule |
| `momentum` | float | 0-0.99 | 0.9 | SGD momentum |
| `max_iter` | int | 100-1000 | 500 | Maximum epochs |
| `early_stopping` | bool | True/False | True | Use validation for stopping |

**Architecture Guidelines:**

| Dataset Size | Recommended Architecture |
|--------------|-------------------------|
| < 100 samples | (16,) or (32,) |
| 100-500 samples | (64,) or (32, 32) |
| 500-1000 samples | (128,) or (64, 32) |
| > 1000 samples | (128, 64) or (256, 128) |

### When to Use MLP

**Strengths:**
- Universal function approximator
- Can learn complex non-linear patterns
- Flexible architecture
- Good for large datasets
- Transfer learning potential

**Weaknesses:**
- Many hyperparameters
- Prone to overfitting
- Requires careful tuning
- Not interpretable
- Sensitive to initialization

**Best for:**
- Large datasets (n > 500)
- Complex non-linear relationships
- When other methods plateau
- When interpretability not needed

### Spectroscopy-Specific Considerations

1. **Input scaling**: Essential! Use StandardScaler or SNV
2. **Architecture**: Start simple (one hidden layer)
3. **early_stopping=True**: Critical to prevent overfitting
4. **alpha regularization**: Use 0.001-0.01 for spectral data
5. **Learning rate**: Adam usually works; try 0.001 first
6. **Validation**: Use 10-20% for early stopping

---

## 12. NeuralBoosted

### Full Name and Abbreviation

**Neural Boosted (NB)**, gradient boosting with neural network weak learners

### Mathematical Foundation

NeuralBoosted combines gradient boosting with small neural networks:

$$F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x)$$

Where:
- $F_m$ = ensemble prediction after $m$ rounds
- $\nu$ = learning rate (shrinkage)
- $h_m$ = neural network weak learner fitted to residuals

**Regression Loss:**

$$\mathcal{L} = \frac{1}{n}\sum_{i=1}^{n} (y_i - F_m(x_i))^2$$

Or with Huber loss for robustness:

$$\mathcal{L}_\delta(r) = \begin{cases} \frac{1}{2}r^2 & |r| \leq \delta \\ \delta(|r| - \frac{\delta}{2}) & |r| > \delta \end{cases}$$

### Algorithm Explanation

**Training Procedure:**

1. **Initialize**: $F_0(x) = 0$
2. **For $m = 1$ to $M$ (n_estimators):**
   - Compute residuals: $r_i = y_i - F_{m-1}(x_i)$
   - Fit small MLP $h_m$ to residuals
   - Update ensemble: $F_m = F_{m-1} + \nu \cdot h_m$
   - Check early stopping criterion
3. **Return**: $F_M(x)$

**Weak Learner Architecture:**

Each weak learner is a small MLP:
- Single hidden layer
- 3-5 hidden nodes (intentionally weak)
- tanh activation (smooth gradients)

**Feature Importance:**

Aggregated from input layer weights:

$$\text{Importance}_j = \frac{1}{M}\sum_{m=1}^{M} \frac{1}{H}\sum_{h=1}^{H} |W_{jh}^{(m)}|$$

Where $W_{jh}^{(m)}$ is the weight from feature $j$ to hidden node $h$ in learner $m$.

### Key Hyperparameters

| Parameter | Type | Range | Default | Effect |
|-----------|------|-------|---------|--------|
| `n_estimators` | int | 50-200 | 100 | Number of weak learners |
| `learning_rate` | float | 0.05-0.3 | 0.1 | Shrinkage parameter |
| `hidden_layer_size` | int | 3-8 | 3 (reg), 5 (clf) | Hidden nodes per weak learner |
| `activation` | str | tanh, relu, identity | tanh | Activation function |
| `alpha` | float | 1e-5 to 0.001 | 0.0001 | L2 regularization |
| `max_iter` | int | 50-200 | 100 | Max iterations per weak learner |
| `early_stopping` | bool | True/False | True | Use validation for stopping |
| `validation_fraction` | float | 0.1-0.2 | 0.15 | Validation set size |
| `n_iter_no_change` | int | 5-20 | 10 | Patience for early stopping |
| `loss` | str | mse, huber | mse | Loss function (regression) |

**Classification-Specific:**

| Parameter | Type | Range | Default | Effect |
|-----------|------|-------|---------|--------|
| `early_stopping_metric` | str | accuracy, log_loss | accuracy | Validation metric |
| `class_weight` | str/dict | None, 'balanced' | None | Handle imbalanced classes |

### When to Use NeuralBoosted

**Strengths:**
- Combines boosting stability with neural network flexibility
- Handles non-linear patterns
- Feature importance scores
- More interpretable than deep MLPs
- Robust to overfitting with early stopping

**Weaknesses:**
- Slower than tree-based boosting
- More hyperparameters than RF
- Less established than XGBoost/LightGBM
- Requires more memory

**Best for:**
- Non-linear relationships
- When tree-based methods plateau
- Moderate-sized datasets
- Alternative to pure MLP

### Spectroscopy-Specific Considerations

1. **hidden_layer_size**: Keep small (3-5) for spectral data
2. **activation='tanh'**: Often works best for spectroscopy
3. **learning_rate**: Start with 0.1, decrease if overfitting
4. **early_stopping**: Essential for spectral data
5. **Comparison**: Test against XGBoost and LightGBM

---

## Model Selection Guidelines

### Decision Framework

Use this flowchart to select an appropriate model:

```
START
  |
  v
[Sample size < 100?] --YES--> PLS or PLS-DA
  |
  NO
  |
  v
[Interpretability critical?] --YES--> PLS, Ridge, Lasso, or ElasticNet
  |
  NO
  |
  v
[Training time limited?] --YES--> RandomForest or LightGBM
  |
  NO
  |
  v
[Need maximum accuracy?] --YES--> XGBoost, LightGBM, or CatBoost (ensemble)
  |
  NO
  |
  v
[Complex non-linear patterns?] --YES--> MLP or NeuralBoosted
  |
  NO
  |
  v
Default: Start with PLS, then try RandomForest, then gradient boosting
```

### Model Selection by Problem Type

| Problem Type | Primary Recommendation | Alternative |
|--------------|----------------------|-------------|
| Quantitative analysis | PLS | Ridge, XGBoost |
| Qualitative analysis | PLS-DA | RandomForest, LightGBM |
| Small samples (< 50) | PLS, Ridge | Lasso |
| Large samples (> 1000) | LightGBM, XGBoost | RandomForest |
| Real-time prediction | PLS, Ridge | Sparse Lasso |
| Feature selection | Lasso, ElasticNet | VIP (PLS) |
| Non-linear | XGBoost, LightGBM | MLP, NeuralBoosted |
| Regulatory compliance | PLS | Ridge |

### Model Performance Expectations

**Typical R2 Ranges by Model (well-optimized):**

| Model | Low R2 Data | Medium R2 Data | High R2 Data |
|-------|-------------|----------------|--------------|
| PLS | 0.6-0.7 | 0.8-0.85 | 0.9-0.95 |
| Ridge | 0.55-0.65 | 0.75-0.82 | 0.88-0.93 |
| RandomForest | 0.5-0.6 | 0.75-0.85 | 0.88-0.94 |
| XGBoost | 0.55-0.7 | 0.8-0.88 | 0.9-0.96 |
| LightGBM | 0.55-0.7 | 0.8-0.88 | 0.9-0.96 |

---

## Hyperparameter Tuning Strategies

### General Principles

1. **Start simple**: Use default hyperparameters first
2. **One at a time**: Understand each parameter's effect
3. **Use cross-validation**: Always validate on held-out data
4. **Grid vs. random**: Random search often more efficient
5. **Early stopping**: Prevent overfitting during tuning

### Recommended Search Order

**For PLS:**
1. n_components (most important)
2. Preprocessing method
3. max_iter/tol (rarely needed)

**For Ridge/Lasso/ElasticNet:**
1. alpha (regularization strength)
2. l1_ratio (ElasticNet only)
3. Preprocessing method

**For RandomForest:**
1. n_estimators (more is usually better)
2. max_depth
3. max_features
4. min_samples_split

**For XGBoost/LightGBM:**
1. learning_rate and n_estimators together
2. max_depth / num_leaves
3. subsample and colsample_bytree
4. reg_alpha and reg_lambda

**For MLP/NeuralBoosted:**
1. hidden_layer_sizes / hidden_layer_size
2. learning_rate_init / learning_rate
3. alpha (regularization)
4. Activation function

### Spectral Predict's Tier-Based Defaults

Spectral Predict provides optimized defaults for each tier:

**Standard Tier Defaults:**

| Model | Key Parameters |
|-------|----------------|
| PLS | n_components: 1-8, max_iter: 500 |
| Ridge | alpha: [0.001, 0.01, 0.1, 1.0, 10.0] |
| Lasso | alpha: [0.001, 0.01, 0.1, 1.0] |
| ElasticNet | alpha: [0.01, 0.1, 1.0], l1_ratio: [0.3, 0.5, 0.7] |
| RandomForest | n_estimators: [100], max_depth: [None, 30] |
| XGBoost | n_estimators: [100, 200], learning_rate: [0.05, 0.1], max_depth: [3, 6] |
| LightGBM | n_estimators: [100, 200], num_leaves: [15, 31], max_depth: [5, 10, -1] |
| CatBoost | iterations: [100, 200], depth: [4, 5, 6], l2_leaf_reg: [3.0, 5.0] |

These defaults are optimized for typical spectroscopy datasets and provide a good starting point for most applications.

---

## Appendix: Mathematical Notation Reference

| Symbol | Meaning |
|--------|---------|
| $X$ | Predictor matrix (n samples x p features) |
| $y$ | Response vector (n samples) |
| $w$ | Weight/coefficient vector |
| $\alpha$ | Regularization parameter |
| $\rho$ | L1/L2 mixing ratio (ElasticNet) |
| $\eta$ | Learning rate |
| $T$ | PLS scores matrix |
| $P$ | PLS loadings matrix |
| $K(\cdot, \cdot)$ | Kernel function |
| $\gamma$ | RBF kernel width / tree complexity penalty |
| $C$ | SVM regularization parameter |
| $\epsilon$ | SVR tube width |

---

## References

1. Wold, S., Sjostrom, M., & Eriksson, L. (2001). PLS-regression: a basic tool of chemometrics. Chemometrics and Intelligent Laboratory Systems, 58(2), 109-130.

2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning (2nd ed.). Springer.

3. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD '16.

4. Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NIPS 2017.

5. Prokhorenkova, L., et al. (2018). CatBoost: unbiased boosting with categorical features. NeurIPS 2018.

6. Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. Annals of Statistics, 29(5), 1189-1232.

---

*Document Version: 1.0*
*Last Updated: January 2026*
*Spectral Predict V1 User Guide - Part III*
# Part IV: Spectral Preprocessing Reference

**Comprehensive Guide to Preprocessing Methods for Spectral Analysis**

---

## Table of Contents

1. [Introduction to Spectral Preprocessing](#1-introduction-to-spectral-preprocessing)
2. [Standard Normal Variate (SNV)](#2-standard-normal-variate-snv)
3. [Savitzky-Golay Derivatives](#3-savitzky-golay-derivatives)
4. [Savitzky-Golay Smoothing](#4-savitzky-golay-smoothing)
5. [Baseline Correction - Polynomial](#5-baseline-correction---polynomial)
6. [Baseline Correction - ALS](#6-baseline-correction---als-asymmetric-least-squares)
7. [Baseline Correction - airPLS](#7-baseline-correction---airpls)
8. [Multiplicative Scatter Correction (MSC)](#8-multiplicative-scatter-correction-msc)
9. [Orthogonal Signal Correction (OSC)](#9-orthogonal-signal-correction-osc)
10. [External Parameter Orthogonalization (EPO)](#10-external-parameter-orthogonalization-epo)
11. [Direct Orthogonal Signal Correction (DOSC)](#11-direct-orthogonal-signal-correction-dosc)
12. [Generalized Least Squares Weighting (GLSW)](#12-generalized-least-squares-weighting-glsw)
13. [Wavelength Exclusion](#13-wavelength-exclusion)
14. [Combined Preprocessing Options](#14-combined-preprocessing-options)
15. [Preprocessing Selection Guide](#15-preprocessing-selection-guide)
16. [Processing Order and Pipeline Architecture](#16-processing-order-and-pipeline-architecture)

---

## 1. Introduction to Spectral Preprocessing

### 1.1 Why Preprocess Spectral Data?

Raw spectral data contains both useful chemical information and unwanted artifacts. Preprocessing removes or reduces these artifacts while preserving the analytical signal. Common artifacts include:

- **Baseline drift**: Gradual shifts in spectral intensity
- **Multiplicative scatter**: Light scattering effects varying by sample
- **Additive noise**: Random electronic and detector noise
- **Systematic interference**: Temperature, moisture, or particle size effects
- **Peak overlap**: Adjacent spectral features that obscure analyte signals

### 1.2 The Preprocessing Pipeline

Spectral Predict applies preprocessing in a specific order to maximize effectiveness:

```
Raw Spectra
    |
    v
[1. Interference Removal (optional)]
    - Wavelength exclusion
    - MSC
    - OSC/EPO/DOSC/GLSW
    |
    v
[2. Baseline Correction (optional)]
    - Polynomial
    - ALS
    - airPLS
    |
    v
[3. Smoothing (optional)]
    - Savitzky-Golay smoothing
    |
    v
[4. Spectral Transformation]
    - SNV
    - Derivatives
    - SNV + Derivatives (in either order)
    |
    v
Preprocessed Spectra --> Modeling
```

### 1.3 Key Principles

1. **Order matters**: The sequence of preprocessing steps affects results
2. **Less is often more**: Excessive preprocessing can remove useful information
3. **Match to problem**: Different spectroscopy types benefit from different preprocessing
4. **Validate with cross-validation**: Always evaluate preprocessing choices using proper validation

---

## 2. Standard Normal Variate (SNV)

### 2.1 Full Name and Purpose

**Full Name:** Standard Normal Variate Transformation

**Purpose:** Removes multiplicative scatter effects and baseline offsets by normalizing each spectrum to have zero mean and unit variance. SNV is particularly effective for correcting particle size effects and path length variations in diffuse reflectance spectroscopy.

### 2.2 Theory

In diffuse reflectance spectroscopy, light scattering causes multiplicative effects that scale the entire spectrum. Different particle sizes, packing densities, or sample-to-detector distances produce spectra that are scaled versions of each other. SNV removes these effects by row-normalizing each spectrum independently.

### 2.3 Mathematical Formula

For a single spectrum $\mathbf{x}$ with $p$ wavelengths:

$$x_{snv,j} = \frac{x_j - \bar{x}}{\sigma_x}$$

Where:
- $x_j$ is the intensity at wavelength $j$
- $\bar{x} = \frac{1}{p}\sum_{j=1}^{p} x_j$ is the mean of the spectrum
- $\sigma_x = \sqrt{\frac{1}{p-1}\sum_{j=1}^{p}(x_j - \bar{x})^2}$ is the standard deviation

In matrix form, for data matrix $\mathbf{X}$ with $n$ samples and $p$ wavelengths:

$$\mathbf{X}_{snv} = \text{diag}\left(\frac{1}{\sigma_1}, \ldots, \frac{1}{\sigma_n}\right) \cdot (\mathbf{X} - \mathbf{M})$$

Where $\mathbf{M}$ is the matrix of row means.

### 2.4 Parameters

SNV has **no user-adjustable parameters**. The transformation is fully determined by each spectrum's statistics.

### 2.5 Visual Representation

```
BEFORE SNV:                          AFTER SNV:
Intensity                            Intensity
    |    ___                             |      ___
    |   /   \   Sample A (high)          |     /   \
    |  /     \__                         |    /     \__
    | /   ___                            |   /   ___
    |    /   \   Sample B (med)          |       /   \
    |   /     \__                        |      /     \__
    |  /   ___                           |     /   ___
    |     /   \   Sample C (low)         |        /   \
    |    /     \__                       |       /     \__
    +------------------------> nm        +------------------------> nm

    [Different baseline offsets          [Spectra now overlap -
     and scaling factors]                 same relative shape]
```

### 2.6 When to Use

**Ideal for:**
- NIR diffuse reflectance spectroscopy
- Powder samples with varying particle sizes
- Samples with varying packing density
- Situations where path length varies between measurements
- Solid samples measured with fiber optics

**Spectroscopy applications:**
| Application | Effectiveness | Notes |
|-------------|---------------|-------|
| NIR reflectance | Excellent | Primary use case |
| NIR transmission | Good | Useful for path length correction |
| Mid-IR (ATR) | Good | Corrects contact pressure variations |
| Raman | Limited | Better alternatives exist |
| UV-Vis | Limited | Usually not needed |

**Avoid when:**
- Spectra have important intensity information (e.g., quantitative transmission)
- Analyzing very noisy spectra (noise can dominate std calculation)
- Spectra contain constant regions (can cause division by near-zero)

### 2.7 Implementation Notes

```python
from spectral_predict.preprocess import SNV

# Create transformer
snv = SNV()

# Apply to data
X_snv = snv.fit_transform(X)

# Safe handling of zero variance
# The implementation sets std=1.0 when std==0 to avoid division by zero
```

---

## 3. Savitzky-Golay Derivatives

### 3.1 Full Name and Purpose

**Full Name:** Savitzky-Golay Derivative Transformation

**Purpose:** Computes derivatives of spectral data using polynomial smoothing within a moving window. Derivatives remove baseline effects, enhance peak resolution, and emphasize rate-of-change information while reducing noise through the inherent smoothing of the polynomial fitting.

### 3.2 Theory

The Savitzky-Golay method fits a polynomial of specified order to data points within a moving window, then evaluates the derivative of the fitted polynomial at the central point. This provides smoothed derivatives without the noise amplification typical of simple finite difference methods.

For a window of $2m+1$ points centered at position $j$, a polynomial of order $k$ is fitted:

$$p(x) = \sum_{i=0}^{k} a_i x^i$$

The $n$-th derivative at the center point is then:

$$\frac{d^n p}{dx^n}\bigg|_{x=0} = n! \cdot a_n$$

### 3.3 Mathematical Formulas

#### 3.3.1 First Derivative (deriv=1, SG1)

The first derivative represents the rate of change of spectral intensity:

$$\frac{d\mathbf{x}}{d\lambda} \approx \mathbf{C}_1 \cdot \mathbf{x}$$

Where $\mathbf{C}_1$ is the convolution coefficient matrix for first derivative.

**Effect:** Removes constant (additive) baseline offsets.

#### 3.3.2 Second Derivative (deriv=2, SG2)

The second derivative represents curvature:

$$\frac{d^2\mathbf{x}}{d\lambda^2} \approx \mathbf{C}_2 \cdot \mathbf{x}$$

**Effect:** Removes both constant and linear (sloping) baseline components.

#### 3.3.3 Higher Derivatives (3rd, 4th)

Third derivative:
$$\frac{d^3\mathbf{x}}{d\lambda^3} \approx \mathbf{C}_3 \cdot \mathbf{x}$$

Fourth derivative:
$$\frac{d^4\mathbf{x}}{d\lambda^4} \approx \mathbf{C}_4 \cdot \mathbf{x}$$

**Effect:** Remove progressively higher-order polynomial baselines.

### 3.4 Parameters and Their Effects

| Parameter | Values | Default | Effect |
|-----------|--------|---------|--------|
| `deriv` | 0, 1, 2, 3, 4 | 1 | Derivative order |
| `window` | 5, 7, 11, 17, 19, 23, 31 | 7 | Window size (must be odd) |
| `polyorder` | 1, 2, 3, 4, 5 | deriv+1 | Polynomial fitting order |

#### 3.4.1 Window Size Effects

```
Window Size Trade-off:

SMALLER WINDOW (5-7):              LARGER WINDOW (17-31):
 + Better spectral resolution       + More noise reduction
 + Preserves sharp peaks           + Smoother output
 - More noise in output            - Broader peaks (resolution loss)
 - Sensitive to high-freq noise    - May merge adjacent peaks

Conceptual Diagram:

Small Window (5):                  Large Window (17):
    |  ___                             |  ___
    | /   \  Sharp peak                |  /   \  Broadened peak
    |/     \                           | /     \
    +--------> nm                      +-----------> nm
    [Noise visible]                    [Smooth, wider]
```

#### 3.4.2 Polynomial Order Effects

The polynomial order must be at least equal to the derivative order plus one:

- `polyorder` >= `deriv` + 1 (minimum requirement)
- Higher `polyorder` = less smoothing, better preservation of sharp features
- Lower `polyorder` = more smoothing, potential distortion of sharp peaks

**Common configurations:**

| Derivative | Recommended polyorder | Notes |
|------------|----------------------|-------|
| 1st | 2 or 3 | 2 for smooth spectra, 3 for sharper peaks |
| 2nd | 3 or 4 | 3 is standard, 4 for high-resolution data |
| 3rd | 4 | Rarely needed |
| 4th | 5 | Very specialized applications |

#### 3.4.3 Constraint: Window Must Accommodate Polynomial

The window size must satisfy: `window >= polyorder + 2`

| polyorder | Minimum window |
|-----------|----------------|
| 2 | 5 |
| 3 | 5 |
| 4 | 7 |
| 5 | 7 |

### 3.5 Visual Representation

```
ORIGINAL SPECTRUM:
Intensity
    |
    |     /\      /\
    |    /  \    /  \     Peaks on sloping baseline
    |   /    \  /    \
    |  /      \/      \
    | /________________\____
    +-------------------------> Wavelength


FIRST DERIVATIVE:
dI/d(lambda)
    |     _
    |    / \     _
    |   /   \   / \
    +--|-----|--|---|-------> Wavelength
    |       \ /   \ /
    |        v     v

    [Constant baseline removed]
    [Peaks become bipolar]


SECOND DERIVATIVE:
d2I/d(lambda)2
    |        _     _
    |       / \   / \
    +------|---|---|---|-----> Wavelength
    |     \_/   \_/
    |      v     v

    [Linear baseline removed]
    [Peaks become inverted]
    [Better peak resolution]
```

### 3.6 When to Use

#### 3.6.1 First Derivative (SG1)

**Ideal for:**
- Removing additive baseline offsets
- Emphasizing peak shoulders
- NIR spectroscopy with baseline drift
- Protein secondary structure analysis

**Spectroscopy applications:**
| Application | Effectiveness | Recommended Window |
|-------------|---------------|-------------------|
| NIR food analysis | Excellent | 11-17 |
| NIR pharmaceutical | Excellent | 7-11 |
| Mid-IR transmission | Good | 5-9 |
| Raman | Limited | 5-7 |

#### 3.6.2 Second Derivative (SG2)

**Ideal for:**
- Removing sloping baselines
- Resolving overlapping peaks
- Enhancing weak shoulders
- Qualitative peak identification

**Spectroscopy applications:**
| Application | Effectiveness | Recommended Window |
|-------------|---------------|-------------------|
| NIR reflectance | Excellent | 17-23 |
| Pharmaceutical analysis | Excellent | 11-17 |
| Protein conformational analysis | Excellent | 5-9 |
| Peak identification | Excellent | 7-11 |

### 3.7 Implementation Notes

```python
from spectral_predict.preprocess import SavgolDerivative

# First derivative with window=11, polyorder=2
deriv1 = SavgolDerivative(deriv=1, window=11, polyorder=2)
X_deriv1 = deriv1.fit_transform(X)

# Second derivative with window=17, polyorder=3
deriv2 = SavgolDerivative(deriv=2, window=17, polyorder=3)
X_deriv2 = deriv2.fit_transform(X)

# Automatic window adjustment for small datasets
# If window > n_features, the implementation automatically
# reduces window to the largest valid odd value
```

---

## 4. Savitzky-Golay Smoothing

### 4.1 Full Name and Purpose

**Full Name:** Savitzky-Golay Smoothing (Zero-Order Derivative)

**Purpose:** Reduces random noise in spectral data while preserving peak shape and position. Unlike derivatives, smoothing maintains the original spectral scale and baseline, making it ideal for noise reduction without baseline alteration.

### 4.2 Theory

Savitzky-Golay smoothing is equivalent to computing the zero-order derivative (i.e., the polynomial value itself, not its derivative). A polynomial is fitted to points within the moving window, and the fitted value at the center replaces the original value. This provides optimal smoothing in the least-squares sense while preserving moment information.

### 4.3 Mathematical Formula

For each point $j$ in the spectrum, the smoothed value is:

$$x_{smooth,j} = \sum_{i=-m}^{m} c_i \cdot x_{j+i}$$

Where:
- $c_i$ are the Savitzky-Golay convolution coefficients for smoothing (deriv=0)
- $m = \frac{\text{window} - 1}{2}$ is the half-window size

The coefficients $c_i$ are derived from the normal equations of polynomial fitting and depend on the window size and polynomial order.

### 4.4 Parameters and Their Effects

| Parameter | Values | Default | Effect |
|-----------|--------|---------|--------|
| `window_length` | 5, 7, 11, 17, 19, 23, 31 | 17 | Smoothing window size |
| `polyorder` | 1, 2, 3, 4 | 2 | Polynomial order |

#### 4.4.1 Window Size Selection

```
Noise Level vs. Window Size:

HIGH NOISE:                        LOW NOISE:
  Use larger window (17-31)          Use smaller window (5-11)
  + Maximum smoothing                + Preserves fine features
  - May broaden peaks                - Less noise reduction

Diagram - Effect of Window Size on Peak Shape:

Original (noisy):          Small window (7):        Large window (21):
    ^^^^^                     ^^^^                      ___
   ^/   \^^                  ^/  \^                    /   \
  ^/     \^                  /    \                   /     \
 ^/       \^^               /      \                 /       \
-----------               -----------              -----------
[Noisy peak]              [Some smoothing]         [Heavy smoothing]
```

#### 4.4.2 Polynomial Order Guidelines

- **polyorder=1**: Linear fitting, maximum smoothing but can distort peaks
- **polyorder=2**: Quadratic fitting, good balance (default)
- **polyorder=3**: Cubic fitting, better peak preservation
- **polyorder=4**: Quartic fitting, minimal distortion, less smoothing

### 4.5 When to Use

**Ideal for:**
- Reducing random noise before other processing
- Improving visual appearance of spectra
- Pre-processing before peak detection
- Raman spectroscopy with high baseline noise

**Spectroscopy applications:**
| Application | Recommended Settings | Notes |
|-------------|---------------------|-------|
| High-resolution NIR | window=7-11, poly=2 | Preserve fine structure |
| Raman (noisy) | window=17-23, poly=2 | Significant smoothing |
| Low-noise spectra | window=5-7, poly=2-3 | Minimal smoothing |
| Baseline determination | window=23-31, poly=2 | Heavy smoothing |

### 4.6 Implementation Notes

```python
from spectral_predict.preprocess import SavgolSmooth

# Standard smoothing
smooth = SavgolSmooth(window_length=17, polyorder=2)
X_smooth = smooth.fit_transform(X)

# Light smoothing for high-resolution data
smooth_light = SavgolSmooth(window_length=7, polyorder=3)
X_smooth = smooth_light.fit_transform(X)
```

---

## 5. Baseline Correction - Polynomial

### 5.1 Full Name and Purpose

**Full Name:** Polynomial Baseline Correction

**Purpose:** Removes baseline drift by fitting a polynomial curve to the spectrum and subtracting it. This is a simple, fast method suitable for spectra with smooth, gradual baseline variations.

### 5.2 Theory

A polynomial of specified degree is fitted to the entire spectrum using least-squares regression. The fitted polynomial represents the estimated baseline, which is then subtracted from the original spectrum. This assumes the baseline can be approximated by a smooth polynomial curve.

### 5.3 Mathematical Formula

For a spectrum $\mathbf{x}$ with wavelength indices $\lambda_1, \lambda_2, \ldots, \lambda_p$:

**Step 1: Fit polynomial baseline**
$$b(\lambda) = a_0 + a_1\lambda + a_2\lambda^2 + \ldots + a_d\lambda^d$$

Where coefficients $\{a_0, \ldots, a_d\}$ are found by minimizing:
$$\sum_{j=1}^{p} (x_j - b(\lambda_j))^2$$

**Step 2: Subtract baseline**
$$x_{corrected,j} = x_j - b(\lambda_j)$$

### 5.4 Parameters and Their Effects

| Parameter | Values | Default | Effect |
|-----------|--------|---------|--------|
| `degree` | 1, 2, 3, 4, 5 | 3 | Polynomial degree |

#### 5.4.1 Degree Selection Guide

```
Degree 1 (Linear):                 Degree 2-3 (Curved):
Baseline: ___________              Baseline: ___________
           \                                   \___/
Removes: Tilted baseline           Removes: Bowl-shaped baseline

Degree 4-5 (Complex):              WARNING: Too High Degree:
Baseline: __/\__                   Baseline: ~~~
              \_/                  May fit peaks, not baseline!
Removes: Multiple dips/rises       Risk of removing real signal
```

| Degree | Baseline Type | Use Case |
|--------|--------------|----------|
| 1 | Linear (tilted) | Simple drift, transmission spectra |
| 2 | Quadratic (curved) | Bowl-shaped baselines |
| 3 | Cubic | S-shaped or single inflection |
| 4-5 | Complex curves | Multiple baseline features |

### 5.5 Visual Representation

```
BEFORE POLYNOMIAL BASELINE CORRECTION:
Intensity
    |
    |        /\
    |       /  \  /\
    |      /    \/  \
    |_____/          \____  <- Actual spectrum
    |
    |______________________ <- Polynomial fit (degree 2)
    +-------------------------> Wavelength


AFTER CORRECTION (degree=2):
Intensity
    |     /\
    |    /  \  /\
    |   /    \/  \
    |  /          \
    +-------------------> Wavelength

    [Baseline now flat]
```

### 5.6 When to Use

**Ideal for:**
- Simple, gradually varying baselines
- Quick baseline correction when speed matters
- Spectra where baseline doesn't interfere with peaks
- Preprocessing before more advanced methods

**Spectroscopy applications:**
| Application | Recommended Degree | Notes |
|-------------|-------------------|-------|
| UV-Vis absorption | 1-2 | Usually linear drift |
| Transmission IR | 1-2 | Path length effects |
| Low-background NIR | 2-3 | Gradual variations |
| Quick preprocessing | 2 | General purpose |

**Avoid when:**
- Baseline has sharp features
- Baseline changes near peak positions
- High-degree polynomial needed (consider ALS instead)
- Peaks are broad relative to spectrum width

### 5.7 Implementation Notes

```python
from spectral_predict.baseline import BaselinePolynomial

# Linear baseline correction
baseline_linear = BaselinePolynomial(degree=1)
X_corrected = baseline_linear.fit_transform(X)

# Quadratic baseline correction (default)
baseline_quad = BaselinePolynomial(degree=2)
X_corrected = baseline_quad.fit_transform(X)
```

---

## 6. Baseline Correction - ALS (Asymmetric Least Squares)

### 6.1 Full Name and Purpose

**Full Name:** Asymmetric Least Squares Baseline Correction

**Purpose:** Iteratively estimates and removes baseline using penalized least squares with asymmetric weighting. ALS is highly effective for spectra where peaks protrude above the baseline, as it preferentially fits below the peaks.

**Reference:** Eilers, P. H. C., & Boelens, H. F. M. (2005). "Baseline correction with asymmetric least squares smoothing." Leiden University Medical Centre Report.

### 6.2 Theory

ALS solves a penalized least squares problem where:
1. The baseline $\mathbf{z}$ should be close to the data $\mathbf{y}$
2. The baseline should be smooth (penalized second derivative)
3. The fitting is asymmetric: points above the baseline are weighted less

The key insight is that spectral peaks typically protrude above the baseline. By iteratively down-weighting points where the data exceeds the current baseline estimate, ALS naturally settles below the peaks.

### 6.3 Mathematical Formula

**Objective function:**

$$\min_{\mathbf{z}} \left[ (\mathbf{y} - \mathbf{z})^T \mathbf{W} (\mathbf{y} - \mathbf{z}) + \lambda \mathbf{z}^T \mathbf{D}^T \mathbf{D} \mathbf{z} \right]$$

Where:
- $\mathbf{y}$ is the original spectrum
- $\mathbf{z}$ is the estimated baseline
- $\mathbf{W}$ is the diagonal weight matrix
- $\lambda$ is the smoothness parameter
- $\mathbf{D}$ is the second-difference matrix

**Weight update (asymmetric):**

$$w_i = \begin{cases} p & \text{if } y_i > z_i \text{ (point above baseline)} \\ 1-p & \text{if } y_i \leq z_i \text{ (point below baseline)} \end{cases}$$

Where $p$ is the asymmetry parameter (typically $p << 0.5$).

**Solution at each iteration:**

$$(\mathbf{W} + \lambda \mathbf{D}^T \mathbf{D}) \mathbf{z} = \mathbf{W} \mathbf{y}$$

### 6.4 Parameters and Their Effects

| Parameter | Symbol | Range | Default | Effect |
|-----------|--------|-------|---------|--------|
| `lambda_` | $\lambda$ | $10^2$ to $10^9$ | $10^5$ | Smoothness |
| `p` | $p$ | 0.001 to 0.1 | 0.001 | Asymmetry |
| `niter` | - | 5-30 | 10 | Iterations |

#### 6.4.1 Lambda (Smoothness) Parameter

```
LAMBDA EFFECT ON BASELINE:

Lambda = 1e3 (Low):              Lambda = 1e7 (High):
    _____                            _______
   /     \_                         /       \
  /        \___                    /         \____
 Follows local features           Very smooth, ignores local features

Guideline:
- 1e2-1e4: Less smooth, follows signal more closely (risk: may enter peaks)
- 1e5-1e6: Good for most spectra (default range)
- 1e7-1e9: Very smooth, good for broad featureless baselines
```

#### 6.4.2 P (Asymmetry) Parameter

```
P EFFECT ON BASELINE POSITION:

p = 0.001 (Strong asymmetry):    p = 0.1 (Weak asymmetry):
    ___                              ___
   /   \  <- Peak                   /   \  <- Peak
  /     \                          /     \
 ____     \___  <- Baseline       /____   \___  <- Baseline rises
     \___/                        (may enter peaks)

Guideline:
- 0.001-0.01: Strong asymmetry, baseline stays well below peaks
- 0.01-0.05: Moderate asymmetry
- 0.05-0.1: Less asymmetry, baseline may rise into peak regions
```

### 6.5 Visual Representation

```
ITERATION PROCESS:

Iteration 1:                     Iteration 5:                    Final:
    ___                              ___                             ___
   /   \                            /   \                           /   \
  /     \                          /     \                         /     \
 /       \____                    /       \____                   /       \____
________________  Initial fit    _______        Converging         _____
                                      \_______/                        \_____/

[Baseline starts high]         [Descending below peaks]         [Stable solution]
```

### 6.6 When to Use

**Ideal for:**
- Raman spectroscopy (fluorescence background removal)
- NIR with strong baseline drift
- Spectra with distinct peaks above baseline
- Situations where polynomial fitting fails

**Spectroscopy applications:**
| Application | Lambda | p | Notes |
|-------------|--------|---|-------|
| Raman spectroscopy | 1e5-1e6 | 0.001 | Fluorescence removal |
| NIR reflectance | 1e5-1e7 | 0.001-0.01 | General baseline |
| Mid-IR (broad features) | 1e6-1e8 | 0.001 | Very smooth baseline |
| ATR-IR | 1e5-1e6 | 0.01 | Contact effects |

### 6.7 Implementation Notes

```python
from spectral_predict.baseline import BaselineALS

# Standard ALS correction
als = BaselineALS(lambda_=1e5, p=0.001, niter=10)
X_corrected = als.fit_transform(X)

# Very smooth baseline for Raman fluorescence
als_smooth = BaselineALS(lambda_=1e7, p=0.001, niter=15)
X_corrected = als_smooth.fit_transform(X_raman)
```

---

## 7. Baseline Correction - airPLS

### 7.1 Full Name and Purpose

**Full Name:** Adaptive Iteratively Reweighted Penalized Least Squares

**Purpose:** An improved version of ALS that adaptively determines weights based on fitting residuals, providing better convergence and more robust baseline estimation, especially for spectra with varying peak heights.

**Reference:** Zhang, Z. M., Chen, S., & Liang, Y. Z. (2010). "Baseline correction using adaptive iteratively reweighted penalized least squares." Analyst, 135(5), 1138-1146.

### 7.2 Theory

airPLS improves on standard ALS by using an adaptive weighting scheme. Instead of a fixed asymmetric weight, airPLS computes weights based on the exponential decay of positive residuals (points above the current baseline estimate). This provides:

1. **Automatic adaptation** to different peak intensities
2. **Better convergence** compared to fixed-weight ALS
3. **Robustness** to different spectral shapes

### 7.3 Mathematical Formula

The penalty objective is similar to ALS:

$$\min_{\mathbf{z}} \left[ (\mathbf{y} - \mathbf{z})^T \mathbf{W} (\mathbf{y} - \mathbf{z}) + \lambda \mathbf{z}^T \mathbf{D}^T \mathbf{D} \mathbf{z} \right]$$

**Adaptive weight update:**

Let $d_i = y_i - z_i$ be the residual at point $i$.

$$w_i = \begin{cases} 1 & \text{if } d_i \leq 0 \\ \exp\left(-\frac{|d_i|}{2m}\right) & \text{if } d_i > 0 \end{cases}$$

Where $m = \text{mean}(|d_i| \text{ for } d_i < 0)$ is the mean absolute negative residual.

**Convergence criterion:**

$$\frac{\sum_i |w_i^{(k)} - w_i^{(k-1)}|}{\sum_i w_i^{(k-1)}} < \text{tol}$$

### 7.4 Parameters and Their Effects

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| `lam` | $10^2$ to $10^9$ | $10^5$ | Smoothness (same as ALS lambda) |
| `max_iter` | 5-30 | 15 | Maximum iterations |
| `tol` | $10^{-4}$ to $10^{-2}$ | $10^{-3}$ | Convergence tolerance |

### 7.5 Comparison with Standard ALS

```
ALS vs. airPLS - Peak Handling:

                    ALS (fixed p=0.001):        airPLS (adaptive):

Tall peak:              ___                          ___
                       /   \                        /   \
                      /     \                      /     \
                     /       \                    /       \
Baseline:           ______    \___               ______    \___
                          \__/                        \__/

Short peak:             _                            _
                       / \                          / \
                      /   \                        /   \
Baseline:           _______\___                  ______\___
                                                       ^
                    [Same weight everywhere]      [Adaptive to peak height]
```

**Advantages of airPLS:**
- Better handling of peaks with varying heights
- More robust convergence
- Less sensitivity to initial parameter choices
- Particularly effective for Raman spectroscopy

### 7.6 When to Use

**Ideal for:**
- Raman spectroscopy with fluorescence
- Spectra with peaks of varying intensities
- When standard ALS convergence is poor
- Automated processing of diverse spectra

**Spectroscopy applications:**
| Application | Lambda | Notes |
|-------------|--------|-------|
| Raman with fluorescence | 1e5-1e6 | Primary use case |
| NIR with variable peaks | 1e5-1e6 | Better than ALS |
| Automated pipelines | 1e5 | Robust default |

### 7.7 Implementation Notes

```python
from spectral_predict.baseline import BaselineAirPLS

# Standard airPLS correction
airpls = BaselineAirPLS(lam=1e6, max_iter=15, tol=1e-3)
X_corrected = airpls.fit_transform(X)
```

---

## 8. Multiplicative Scatter Correction (MSC)

### 8.1 Full Name and Purpose

**Full Name:** Multiplicative Scatter Correction

**Purpose:** Removes multiplicative scatter effects by fitting each spectrum to a reference spectrum (typically the mean of the calibration set). MSC is an alternative to SNV that uses a common reference, making it more suitable when spectra should be comparable to a specific standard.

**Reference:** Geladi, P., McDougall, D., & Martens, H. (1985). "Linearization and scatter-correction for near-infrared reflectance spectra of meat." Applied Spectroscopy, 39(3), 491-500.

### 8.2 Theory

MSC assumes that scatter effects cause both multiplicative scaling and additive offset:

$$\mathbf{x}_i = a_i + b_i \cdot \mathbf{x}_{ref} + \mathbf{e}_i$$

Where:
- $\mathbf{x}_i$ is the measured spectrum of sample $i$
- $\mathbf{x}_{ref}$ is the reference spectrum
- $a_i$ is the additive offset (baseline shift)
- $b_i$ is the multiplicative factor (scaling)
- $\mathbf{e}_i$ is the residual (chemical information + noise)

By fitting $a_i$ and $b_i$ via ordinary least squares (OLS) and correcting, we obtain scatter-corrected spectra.

### 8.3 Mathematical Formula

**Step 1: Define reference spectrum**
$$\mathbf{x}_{ref} = \frac{1}{n}\sum_{i=1}^{n} \mathbf{x}_i \quad \text{(mean spectrum)}$$

**Step 2: Fit each spectrum to reference**

For each spectrum $\mathbf{x}_i$, find $a_i$ and $b_i$ by minimizing:
$$\sum_{j=1}^{p} (x_{ij} - a_i - b_i \cdot x_{ref,j})^2$$

**Solution (OLS):**
$$b_i = \frac{\sum_j (x_{ij} - \bar{x}_i)(x_{ref,j} - \bar{x}_{ref})}{\sum_j (x_{ref,j} - \bar{x}_{ref})^2}$$
$$a_i = \bar{x}_i - b_i \cdot \bar{x}_{ref}$$

**Step 3: Apply correction**
$$\mathbf{x}_{msc,i} = \frac{\mathbf{x}_i - a_i}{b_i}$$

### 8.4 Parameters

| Parameter | Options | Default | Effect |
|-----------|---------|---------|--------|
| `reference` | 'mean', 'median', or array | 'mean' | Reference spectrum source |

- **'mean'**: Use mean spectrum of training data (most common)
- **'median'**: Use median spectrum (robust to outliers)
- **array**: Use provided external reference spectrum

### 8.5 Visual Representation

```
MSC TRANSFORMATION:

Before MSC:                          After MSC:
Intensity                            Intensity
    |                                    |
    |   ___    Sample A (scaled up)      |    ___
    |  /   \                             |   /   \
    | /     \                            |  /     \    All samples
    |/  ___                              | /  ___       now aligned
    |  /   \   Sample B (reference)      |    /   \     to reference
    | /     \                            |   /     \
    |___                                 |  ___
    |/  \   Sample C (scaled down)       |  /   \
    +------------------------> nm        +------------------------> nm

    [Different scales due to scatter]    [Corrected to reference]
```

### 8.6 SNV vs. MSC Comparison

```
                        SNV                     MSC

Reference:              Each spectrum's         Common reference
                        own statistics          (typically mean)

Information used:       Row-wise only           Cross-sample

New samples:            Can transform           Can transform using
                        independently           stored reference

Interpretation:         Standardized units      Units of reference
                        (mean=0, std=1)         spectrum

When to prefer SNV:     Unknown reference
                        Fast computation

When to prefer MSC:     Known reference
                        Comparing to standard
                        Consistent calibration
```

### 8.7 When to Use

**Ideal for:**
- When you have a known reference spectrum
- Calibration transfer between instruments
- Building consistent calibration models
- NIR diffuse reflectance with known standards

**Spectroscopy applications:**
| Application | Reference Type | Notes |
|-------------|----------------|-------|
| NIR calibration | Mean of training set | Standard approach |
| Instrument standardization | Master instrument spectrum | Transfer calibration |
| Quality control | Reference standard spectrum | Comparison to standard |

### 8.8 Implementation Notes

```python
from spectral_predict.interference import MSC

# MSC with mean reference (standard)
msc = MSC(reference='mean')
X_train_msc = msc.fit_transform(X_train)  # Computes mean, transforms
X_test_msc = msc.transform(X_test)        # Uses stored mean

# MSC with median reference (robust)
msc_robust = MSC(reference='median')
X_msc = msc_robust.fit_transform(X)

# MSC with external reference
reference_spectrum = np.load('reference.npy')
msc_external = MSC(reference=reference_spectrum)
X_msc = msc_external.fit_transform(X)
```

---

## 9. Orthogonal Signal Correction (OSC)

### 9.1 Full Name and Purpose

**Full Name:** Orthogonal Signal Correction

**Purpose:** Removes systematic variation in spectral data (X) that is orthogonal to (uncorrelated with) the target variable (y). OSC is particularly effective for removing temperature effects, moisture variations, and other interference that does not correlate with the analyte of interest.

**Reference:** Wold, S., Antti, H., Lindgren, F., & Ohman, J. (1998). "Orthogonal signal correction of near-infrared spectra." Chemometrics and Intelligent Laboratory Systems, 44(1-2), 175-185.

### 9.2 Theory

OSC operates on a key principle: spectral variance can be decomposed into:
1. **Y-relevant variance**: Information correlated with the target property
2. **Y-orthogonal variance**: Systematic effects not correlated with target

By building a PLS model between X and y, we can identify the Y-relevant subspace. The orthogonal complement contains systematic interference that can be safely removed without losing predictive information.

### 9.3 Mathematical Formula

**Step 1: Center the data**
$$\mathbf{X}_c = \mathbf{X} - \bar{\mathbf{X}}$$
$$\mathbf{y}_c = \mathbf{y} - \bar{y}$$

**Step 2: Build PLS model to find Y-relevant directions**

PLS finds weights $\mathbf{w}$ that maximize covariance between $\mathbf{X}_c \mathbf{w}$ and $\mathbf{y}_c$.

**Step 3: Extract orthogonal component**

From the PLS model, extract score vector $\mathbf{t}$ and loading vector $\mathbf{p}$:
$$\mathbf{t}_{osc} = \mathbf{X}_c \mathbf{w}_{osc}$$

Where $\mathbf{w}_{osc}$ is orthogonalized to the Y-relevant direction.

**Step 4: Remove orthogonal component**
$$\mathbf{X}_{corrected} = \mathbf{X}_c - \mathbf{t}_{osc} \mathbf{p}_{osc}^T$$

**Step 5: Iterate for multiple components**

Repeat Steps 2-4 for each orthogonal component to remove.

### 9.4 Parameters and Their Effects

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| `n_components` | 1-5 | 1 | Number of orthogonal components to remove |
| `tol` | $10^{-8}$ to $10^{-4}$ | $10^{-6}$ | Convergence tolerance |
| `max_iter` | 50-200 | 100 | Maximum iterations |

#### 9.4.1 n_components Selection

```
Number of OSC Components:

1 component:                      3+ components:
  Removes dominant                  Removes multiple
  orthogonal effect                 interference effects

  SAFE - Minimal risk              CAUTION - Risk of
  of removing Y-info               removing useful signal

Diagnostic:
- Check explained variance by each component
- Component with low variance removal may be unnecessary
- Stop if model performance decreases
```

### 9.5 Visual Representation

```
OSC CONCEPT:

Original X-space:                  After OSC:

      Y-direction                       Y-direction
          |                                 |
          |    x                            |    x
          |   x x                           |   x x
          |  x   x  <- Data cloud           |  x x x  <- Compressed
          | x  x  x                         | x x x
          |x   x                            |x x
   -------|-------> Orthogonal              |
          |         direction              No orthogonal spread

  [Variance in both directions]      [Only Y-relevant variance remains]
```

### 9.6 When to Use

**Ideal for:**
- Removing temperature effects in NIR
- Removing moisture variations
- Removing physical property interference
- When interference is systematic but uncorrelated with analyte

**Spectroscopy applications:**
| Application | n_components | Notes |
|-------------|--------------|-------|
| Temperature effects | 1-2 | Common NIR interference |
| Moisture in solids | 1-2 | Water band variation |
| Particle size effects | 1-3 | Physical property variation |
| Instrument drift | 1-2 | Systematic temporal effects |

**Important considerations:**
- OSC requires target variable (y) during fitting
- Only training data should be used for fitting (avoid data leakage)
- Transformed data is mean-centered (training mean subtracted)
- More aggressive than SNV/MSC - use carefully

### 9.7 Implementation Notes

```python
from spectral_predict.interference import OSC

# Remove 1 orthogonal component
osc = OSC(n_components=1)
X_train_osc = osc.fit_transform(X_train, y_train)  # Requires y
X_test_osc = osc.transform(X_test)                  # y not needed

# Remove multiple components (use with caution)
osc_multi = OSC(n_components=2)
X_corrected = osc_multi.fit_transform(X_train, y_train)

# Check variance removed
print(f"Variance removed: {osc.variance_removed_}")
```

---

## 10. External Parameter Orthogonalization (EPO)

### 10.1 Full Name and Purpose

**Full Name:** External Parameter Orthogonalization

**Purpose:** Removes specific known interference effects using a reference library of interferent spectra. Unlike OSC (which finds orthogonal components automatically), EPO uses explicitly provided interferent spectra to define the unwanted subspace.

**Reference:** Roger, J. M., Chauchard, F., & Bellon-Maurel, V. (2003). "EPO-PLS external parameter orthogonalisation of PLS application to temperature-independent measurement of sugar content of intact fruits." Chemometrics and Intelligent Laboratory Systems, 66(2), 191-204.

### 10.2 Theory

EPO requires a library of interferent spectra that characterize the unwanted variation. For example, to remove moisture interference, you provide spectra of samples with varying moisture content. EPO then:

1. Builds a subspace from the interferent library using PCA/SVD
2. Projects new data orthogonal to this subspace
3. The projected data is free from the interferent effect

This is more targeted than OSC because you explicitly specify what to remove, rather than removing anything orthogonal to Y.

### 10.3 Mathematical Formula

**Step 1: Center interferent library**
$$\mathbf{X}_{int,c} = \mathbf{X}_{int} - \bar{\mathbf{X}}_{int}$$

**Step 2: Compute interferent subspace via SVD**
$$\mathbf{X}_{int,c} = \mathbf{U} \mathbf{S} \mathbf{V}^T$$

Take first $k$ columns of $\mathbf{V}$ as interferent basis:
$$\mathbf{V}_k = [\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k]$$

**Step 3: Build orthogonal projection matrix**
$$\mathbf{P}_{orth} = \mathbf{I} - \mathbf{V}_k \mathbf{V}_k^T$$

**Step 4: Apply to new data**
$$\mathbf{X}_{corrected} = (\mathbf{X} - \bar{\mathbf{X}}_{train}) \mathbf{P}_{orth}$$

### 10.4 Parameters and Their Effects

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| `n_components` | 1-10 | 2 | Number of interferent components |
| `center` | True/False | True | Center data before EPO |
| `svd_tol` | $10^{-10}$ to $10^{-6}$ | $10^{-8}$ | SVD truncation tolerance |

#### 10.4.1 n_components Selection

```
EPO Components vs. Interferent Library:

If interferent library has N spectra:
- Maximum meaningful components = N-1 (after centering)
- Typical: 2-5 components captures most interferent variation
- Start low, increase if interference remains

WARNING: Too many components risk removing analyte signal
```

### 10.5 Visual Representation

```
EPO WORKFLOW:

1. Interferent Library:              2. Build Subspace:
   [Moisture level spectra]             [SVD of library]

   5% ----~~~----                       Component 1: ----/\/----
   10% ----~~~----                      Component 2: ----\/\----
   15% ----~~~----
   20% ----~~~----

3. Project Data Orthogonal:          4. Result:

   Original data                        Corrected data
      |                                    |
      |  . .                               |  . .
      |  ...  <- With moisture             | ....  <- Moisture
      | .. .      variation                |....       removed
      +--------> Moisture                  +-------->
                 direction                 (collapsed)
```

### 10.6 When to Use

**Ideal for:**
- Known interference sources with available reference spectra
- Temperature effects (measure at multiple temperatures)
- Moisture effects (measure at multiple moisture levels)
- Particle size effects (measure same sample with different grinding)

**Requirements:**
- Must have interferent library spectra
- Library should span the expected range of interference
- Interferent spectra should be measured under same conditions as target spectra

**Spectroscopy applications:**
| Interferent | Library Examples | n_components |
|-------------|------------------|--------------|
| Moisture | Same sample at 5%, 10%, 15%, 20% moisture | 2-3 |
| Temperature | Same sample at 20C, 30C, 40C, 50C | 2-3 |
| Particle size | Same material at different grind sizes | 2-4 |
| Instrument variation | Same sample on different instruments | 2-5 |

### 10.7 Implementation Notes

```python
from spectral_predict.interference import EPO

# Load interferent library (e.g., moisture spectra)
X_moisture_library = np.load('moisture_library.npy')
# Shape: (n_moisture_levels, n_wavelengths)

# Fit EPO with interferent library
epo = EPO(n_components=2)
epo.fit(X_train, y_train, X_interferents=X_moisture_library)

# Transform data
X_train_corrected = epo.transform(X_train)
X_test_corrected = epo.transform(X_test)

# Inspect explained variance
print(f"Interferent variance by component: {epo.explained_variance_}")
```

---

## 11. Direct Orthogonal Signal Correction (DOSC)

### 11.1 Full Name and Purpose

**Full Name:** Direct Orthogonal Signal Correction

**Purpose:** A simplified variant of OSC that directly computes the Y-orthogonal subspace using PLS residuals. DOSC is more computationally efficient than iterative OSC and provides stable results.

**Reference:** Westerhuis, J. A., de Jong, S., & Smilde, A. K. (2001). "Direct orthogonal signal correction." Chemometrics and Intelligent Laboratory Systems, 56(1), 13-25.

### 11.2 Theory

DOSC works by:
1. Building a PLS model between X and y
2. Computing residuals: the part of X not explained by PLS
3. Performing PCA on these residuals to find Y-orthogonal directions
4. Projecting data orthogonal to these directions

The key difference from standard OSC is the "direct" computation via PLS residuals, avoiding iterative deflation.

### 11.3 Mathematical Formula

**Step 1: Fit PLS model**
$$\mathbf{X}_c = \mathbf{T}_{pls} \mathbf{P}_{pls}^T + \mathbf{E}_{pls}$$

Where $\mathbf{T}_{pls}$ are PLS scores and $\mathbf{P}_{pls}$ are PLS loadings.

**Step 2: Extract PLS residuals (Y-orthogonal part)**
$$\mathbf{E}_{orth} = \mathbf{X}_c - \mathbf{T}_{pls} \mathbf{P}_{pls}^T$$

**Step 3: PCA on residuals**
$$\mathbf{E}_{orth} = \mathbf{U} \mathbf{S} \mathbf{V}^T$$

Take first $k$ columns of $\mathbf{V}$ as DOSC components:
$$\mathbf{V}_{dosc} = [\mathbf{v}_1, \ldots, \mathbf{v}_k]$$

**Step 4: Build projection matrix**
$$\mathbf{P}_{orth} = \mathbf{I} - \mathbf{V}_{dosc} \mathbf{V}_{dosc}^T$$

**Step 5: Apply to data**
$$\mathbf{X}_{corrected} = (\mathbf{X} - \bar{\mathbf{X}}) \mathbf{P}_{orth}$$

### 11.4 Parameters and Their Effects

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| `n_components` | 1-5 | 1 | Y-orthogonal components to remove |
| `center` | True/False | True | Center data |
| `n_pls_components` | 'auto' or int | 'auto' | PLS components for Y-subspace |

#### 11.4.1 n_pls_components Selection

- **'auto'**: Uses min(10, n_samples-1, n_features)
- **Integer**: Specific number of PLS components
- More PLS components = better approximation of Y-space
- Fewer PLS components = faster, more robust

### 11.5 DOSC vs. OSC Comparison

```
                        OSC                     DOSC

Algorithm:              Iterative deflation     Direct PLS residuals

Computation:            Slower, iterative       Faster, single pass

Stability:              May not converge        Always converges

Results:                Similar                 Similar

When to use:            Classic approach        Modern, recommended
```

### 11.6 When to Use

**Ideal for:**
- Same applications as OSC but with better stability
- Automated processing pipelines (reliable convergence)
- When OSC has convergence issues
- Large datasets (faster computation)

**Spectroscopy applications:**
| Application | n_components | Notes |
|-------------|--------------|-------|
| Temperature correction | 1-2 | Same as OSC |
| Moisture correction | 1-2 | Same as OSC |
| Automated pipelines | 1 | Reliable default |

### 11.7 Implementation Notes

```python
from spectral_predict.interference import DOSC

# Standard DOSC
dosc = DOSC(n_components=2)
dosc.fit(X_train, y_train)  # Requires y

# Transform
X_train_corrected = dosc.transform(X_train)
X_test_corrected = dosc.transform(X_test)

# Check variance explained
print(f"Variance removed: {dosc.explained_variance_}")

# Custom PLS components
dosc_custom = DOSC(n_components=2, n_pls_components=5)
dosc_custom.fit(X_train, y_train)
```

---

## 12. Generalized Least Squares Weighting (GLSW)

### 12.1 Full Name and Purpose

**Full Name:** Generalized Least Squares Weighting

**Purpose:** Down-weights wavelength regions dominated by noise or interference while up-weighting informative regions. GLSW provides optimal wavelength weighting for heteroscedastic (non-uniform) noise.

**Reference:** Seasholtz, M. B., & Kowalski, B. R. (1993). "The parsimony principle applied to multivariate calibration." Analytica Chimica Acta, 277(2), 165-177.

### 12.2 Theory

Different spectral regions often have different noise levels. Water absorption bands in NIR are typically noisier than other regions. GLSW computes optimal weights for each wavelength based on:

1. **Covariance method**: Weights inversely proportional to variance
2. **Residual method**: Weights based on PLS model residuals

The weighted data can then be used with standard regression methods, effectively performing weighted least squares.

### 12.3 Mathematical Formula

**Covariance Method:**

Compute per-wavelength variance:
$$\sigma_j^2 = \text{Var}(X_{:,j})$$

Compute weights (inverse variance):
$$w_j = \frac{1}{\sigma_j^2 + \epsilon}$$

Where $\epsilon$ is a regularization term.

**Residual Method:**

Fit PLS model and compute residual variance per wavelength:
$$r_j^2 = \text{Var}(X_{:,j} - \hat{X}_{:,j})$$

Compute weights:
$$w_j = \frac{1}{r_j^2 + \epsilon}$$

**Apply to data:**

The transformation applies square root of weights:
$$X_{weighted,j} = X_j \cdot \sqrt{w_j}$$

This is equivalent to weighted least squares: $\min \|\mathbf{W}^{1/2}(\mathbf{X}\beta - \mathbf{y})\|^2$

### 12.4 Parameters and Their Effects

| Parameter | Options | Default | Effect |
|-----------|---------|---------|--------|
| `method` | 'covariance', 'residual' | 'covariance' | Weight computation method |
| `regularization` | $10^{-8}$ to $10^{-4}$ | $10^{-6}$ | Prevents division by zero |
| `n_components` | int or None | None | PLS components (residual method) |

### 12.5 Visual Representation

```
GLSW CONCEPT:

Spectral Variance:                   GLSW Weights:
Variance                             Weight
    |                                    |
    |   ____                             |________
    |__/    \                            |        \____
    |        \___                        |             \
    |            \_____                  |              \_____
    +--------------------> nm            +--------------------> nm
    [High var in water band]             [Low weight in water band]

Before GLSW:                         After GLSW:
    |                                    |
    |    /\                              |    /\
    |   /  \   /\~~~                     |   /  \   /\
    |  /    \/    ~~                     |  /    \/  \
    +--------------------> nm            +--------------------> nm
    [Noisy water band]                   [Weighted down]
```

### 12.6 When to Use

**Ideal for:**
- Spectra with known noisy regions (e.g., water bands)
- Heteroscedastic noise (different noise levels across spectrum)
- When certain wavelengths have high interference
- Optimizing wavelength weighting before regression

**Spectroscopy applications:**
| Application | Method | Notes |
|-------------|--------|-------|
| NIR with water bands | covariance | Down-weight 1400-1500, 1900-2000 nm |
| Unequal noise levels | covariance | General purpose |
| Complex interference | residual | Uses PLS to identify non-informative regions |

### 12.7 Implementation Notes

```python
from spectral_predict.interference import GLSW

# Covariance-based weighting
glsw = GLSW(method='covariance')
glsw.fit(X_train)  # y not needed
X_weighted = glsw.transform(X_train)

# Residual-based weighting (requires y)
glsw_residual = GLSW(method='residual', n_components=5)
glsw_residual.fit(X_train, y_train)
X_weighted = glsw_residual.transform(X_train)

# Get weights for analysis
weights = glsw.get_feature_weights()
```

---

## 13. Wavelength Exclusion

### 13.1 Full Name and Purpose

**Full Name:** Wavelength Range Exclusion

**Purpose:** Physically removes specified wavelength ranges from the data, eliminating regions dominated by noise, interference, or uninformative signal. This is the most direct approach to handling problematic spectral regions.

### 13.2 Theory

Some spectral regions contain no useful information and only add noise to models:
- **Water absorption bands**: 1400-1500 nm and 1900-2000 nm in NIR
- **Detector cutoff regions**: Low signal at spectrum edges
- **Saturated regions**: Where detector saturates

Rather than down-weighting these regions (GLSW) or correcting them (baseline), wavelength exclusion simply removes them from the data matrix.

### 13.3 Parameters

| Parameter | Type | Default | Effect |
|-----------|------|---------|--------|
| `wavelengths` | array | Required | Wavelength values for each column |
| `exclude_ranges` | list of tuples | [(1400,1500), (1900,2000)] | Ranges to exclude |
| `invert` | bool | False | If True, KEEP only specified ranges |

### 13.4 Visual Representation

```
WAVELENGTH EXCLUSION:

Original (1000-2500 nm):
    |                                  |
    |      /\        /\       /\       |
    |     /  \      /~~\     /  \      |
    |____/    \____/~~~~\___/    \_____|
    1000      1400  1500 1900  2000   2500
                |______|    |______|
                Water bands (exclude)

After Exclusion:
    |                        |
    |      /\       /\       |
    |     /  \     /  \      |
    |____/    \___/    \_____|
    1000      1400  1500   2000   2500
              (gap)  (gap)

Feature count reduced: e.g., 1500 -> 1300 wavelengths
```

### 13.5 When to Use

**Ideal for:**
- Removing known uninformative regions
- Water absorption bands in NIR (1400-1500, 1900-2000 nm)
- Detector noise regions at spectrum edges
- Regions with strong interference that cannot be corrected

**Common exclusion ranges:**
| Spectroscopy | Region | Wavelengths | Reason |
|--------------|--------|-------------|--------|
| NIR | Water O-H 1st | 1400-1500 nm | Strong water absorption |
| NIR | Water O-H 2nd | 1900-2000 nm | Strong water absorption |
| NIR | CO2 | 2300-2400 nm | Atmospheric CO2 |
| Visible-NIR | Edge noise | <400 nm, >2500 nm | Detector limits |

### 13.6 Implementation Notes

```python
from spectral_predict.interference import WavelengthExcluder

# Get wavelength array from your data
wavelengths = np.linspace(1000, 2500, 1500)  # Example

# Exclude water bands
excluder = WavelengthExcluder(
    wavelengths,
    exclude_ranges=[(1400, 1500), (1900, 2000)]
)
X_filtered = excluder.fit_transform(X)

# Get remaining wavelengths
wavelengths_out = excluder.get_feature_names_out()
print(f"Reduced from {X.shape[1]} to {X_filtered.shape[1]} wavelengths")

# Invert mode: keep ONLY specified ranges
excluder_keep = WavelengthExcluder(
    wavelengths,
    exclude_ranges=[(1100, 1350), (1550, 1850)],
    invert=True  # Keep only these ranges
)
```

---

## 14. Combined Preprocessing Options

### 14.1 Overview

Spectral Predict allows combining multiple preprocessing methods in a single pipeline. The most common combinations involve SNV and derivatives in different orders.

### 14.2 Available Combinations

| Name | Steps | Order |
|------|-------|-------|
| `raw` | None | No preprocessing |
| `snv` | SNV only | SNV |
| `deriv` | Derivative only | Derivative |
| `snv_deriv` | SNV then Derivative | SNV -> Deriv |
| `deriv_snv` | Derivative then SNV | Deriv -> SNV |

### 14.3 Order Effects

The order of SNV and derivatives matters and produces different results:

```
SNV THEN DERIVATIVE (snv_deriv):              DERIVATIVE THEN SNV (deriv_snv):

Step 1: Apply SNV                             Step 1: Apply Derivative
   - Row normalization                           - Computes d/d(lambda)
   - Mean=0, Std=1 per row                       - Removes baseline

Step 2: Apply Derivative                      Step 2: Apply SNV
   - Takes derivative of SNV result              - Normalizes derivative spectrum
   - Derivative of normalized spectrum           - Row normalization of derivatives

Result:                                       Result:
   - SNV scatter correction first                - Baseline removal first
   - Then baseline removal                       - Then amplitude normalization
   - Derivative of scatter-corrected signal      - SNV of derivative shape
```

### 14.4 When to Use Each Combination

| Combination | Best For | Notes |
|-------------|----------|-------|
| `raw` | Clean data, testing baseline performance | No preprocessing overhead |
| `snv` | Scatter correction only | When baseline is acceptable |
| `deriv` | Baseline removal only | When scatter is not an issue |
| `snv_deriv` | **Most common NIR choice** | Scatter then baseline removal |
| `deriv_snv` | Alternative approach | Baseline then scatter normalization |

### 14.5 Experimental Comparison

For most NIR spectroscopy applications, `snv_deriv1` (SNV followed by first derivative) performs best. However, the optimal choice depends on:

1. **Nature of scatter effects**: If scatter dominates, apply SNV first
2. **Nature of baseline**: If baseline is the main issue, apply derivative first
3. **Empirical testing**: Always validate with cross-validation

### 14.6 Full Preprocessing Grid

Spectral Predict's search function tests multiple combinations automatically:

```
PREPROCESSING SEARCH GRID:

Basic options:
- raw
- snv

Derivative options (window=5,7,11,17,19,23,31):
- deriv1 (1st derivative)
- deriv2 (2nd derivative)

Combined options:
- snv_deriv1 (SNV + 1st derivative)
- snv_deriv2 (SNV + 2nd derivative)
- deriv1_snv (1st derivative + SNV)
- deriv2_snv (2nd derivative + SNV)

Additional options (when enabled):
- With baseline correction (polynomial, ALS, airPLS)
- With smoothing
- With interference removal (OSC, EPO, DOSC, GLSW)
```

### 14.7 Implementation Example

```python
from spectral_predict.preprocess import build_preprocessing_pipeline
from sklearn.pipeline import Pipeline

# Build SNV + 1st derivative pipeline
steps = build_preprocessing_pipeline(
    preprocess_name='snv_deriv',
    deriv=1,
    window=11,
    polyorder=2
)

# Create sklearn Pipeline
pipeline = Pipeline(steps)

# Apply
X_preprocessed = pipeline.fit_transform(X)

# Full pipeline with baseline and smoothing
steps_full = build_preprocessing_pipeline(
    preprocess_name='snv_deriv',
    deriv=1,
    window=11,
    polyorder=2,
    baseline_method='asls',
    baseline_params={'lam': 1e5, 'p': 0.001},
    smoothing=True,
    smoothing_window=17
)
```

---

## 15. Preprocessing Selection Guide

### 15.1 Decision Tree

```
START
  |
  v
Is baseline drift a major issue?
  |
  +--YES--> Apply baseline correction
  |           |
  |           +-- Gradual drift? --> Polynomial (degree 2-3)
  |           +-- Complex shape? --> ALS (lambda=1e5)
  |           +-- Raman fluorescence? --> airPLS (lam=1e6)
  |
  +--NO---> Continue
  |
  v
Is scatter variation present?
  |
  +--YES--> Apply scatter correction
  |           |
  |           +-- Unknown reference? --> SNV
  |           +-- Known reference? --> MSC
  |
  +--NO---> Continue
  |
  v
Are there overlapping peaks?
  |
  +--YES--> Apply derivatives
  |           |
  |           +-- Constant baseline? --> 1st derivative
  |           +-- Sloping baseline? --> 2nd derivative
  |
  +--NO---> Continue
  |
  v
Is there systematic interference?
  |
  +--YES--> Apply interference correction
  |           |
  |           +-- Known interferent spectra? --> EPO
  |           +-- Unknown but Y-orthogonal? --> OSC/DOSC
  |           +-- Heteroscedastic noise? --> GLSW
  |           +-- Known bad regions? --> Wavelength exclusion
  |
  +--NO---> Continue
  |
  v
DONE - Run cross-validation to select final preprocessing
```

### 15.2 Recommendations by Spectroscopy Type

| Spectroscopy Type | Primary Method | Secondary | Avoid |
|-------------------|----------------|-----------|-------|
| **NIR Reflectance** | SNV + deriv1 | MSC, ALS | High-order derivatives |
| **NIR Transmission** | SNV | deriv1 | Heavy smoothing |
| **Mid-IR (ATR)** | SNV + deriv2 | ALS | MSC (contact issues) |
| **Raman** | airPLS + smoothing | polynomial baseline | SNV (not scatter) |
| **UV-Vis** | polynomial baseline | None | Derivatives (broad peaks) |
| **Fluorescence** | None or baseline | smoothing | Derivatives |

### 15.3 Recommendations by Sample Type

| Sample Type | Key Issues | Recommended Preprocessing |
|-------------|------------|---------------------------|
| **Powders** | Scatter, particle size | SNV or MSC |
| **Liquids** | Path length | SNV, baseline correction |
| **Intact fruit** | Scatter, moisture | SNV + deriv1, OSC |
| **Pharmaceuticals** | Baseline, scatter | SNV + deriv2 |
| **Biological tissue** | Scatter, fluorescence | SNV + airPLS |

### 15.4 Parameter Selection Quick Reference

**Savitzky-Golay Window Size:**
| Data Resolution | Recommended Window |
|-----------------|-------------------|
| Very high (>5000 points) | 17-31 |
| High (2000-5000 points) | 11-19 |
| Medium (500-2000 points) | 7-11 |
| Low (<500 points) | 5-7 |

**ALS Lambda:**
| Baseline Type | Recommended Lambda |
|---------------|-------------------|
| Smooth, gradual | 1e6 - 1e8 |
| Moderate curvature | 1e5 - 1e6 |
| Sharp features | 1e3 - 1e5 |

**OSC/DOSC Components:**
| Interference Level | Recommended n_components |
|--------------------|-------------------------|
| Single dominant effect | 1 |
| Multiple effects | 2-3 |
| Complex interference | 3-5 (use caution) |

---

## 16. Processing Order and Pipeline Architecture

### 16.1 Recommended Processing Order

Spectral Predict applies preprocessing in a specific, carefully designed order:

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                    RAW SPECTRAL DATA                        â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                              â”‚
                              â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚           STEP 1: INTERFERENCE REMOVAL (Optional)           â”‚
â”‚                                                             â”‚
â”‚   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                â”‚
â”‚   â”‚ Wavelength      â”‚   â”‚ MSC             â”‚                â”‚
â”‚   â”‚ Exclusion       â”‚   â”‚                 â”‚                â”‚
â”‚   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                â”‚
â”‚                                                             â”‚
â”‚   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                â”‚
â”‚   â”‚ OSC / DOSC      â”‚   â”‚ EPO             â”‚                â”‚
â”‚   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                â”‚
â”‚                                                             â”‚
â”‚   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                                      â”‚
â”‚   â”‚ GLSW            â”‚                                      â”‚
â”‚   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                                      â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                              â”‚
                              â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚           STEP 2: BASELINE CORRECTION (Optional)            â”‚
â”‚                                                             â”‚
â”‚   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                â”‚
â”‚   â”‚ Polynomial      â”‚   â”‚ ALS             â”‚                â”‚
â”‚   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                â”‚
â”‚                                                             â”‚
â”‚   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                                      â”‚
â”‚   â”‚ airPLS          â”‚                                      â”‚
â”‚   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                                      â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                              â”‚
                              â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚              STEP 3: SMOOTHING (Optional)                   â”‚
â”‚                                                             â”‚
â”‚   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                                      â”‚
â”‚   â”‚ Savitzky-Golay  â”‚                                      â”‚
â”‚   â”‚ Smoothing       â”‚                                      â”‚
â”‚   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                                      â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                              â”‚
                              â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚         STEP 4: SPECTRAL TRANSFORMATION (Required)          â”‚
â”‚                                                             â”‚
â”‚   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                â”‚
â”‚   â”‚ raw             â”‚   â”‚ snv             â”‚                â”‚
â”‚   â”‚ (no transform)  â”‚   â”‚                 â”‚                â”‚
â”‚   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                â”‚
â”‚                                                             â”‚
â”‚   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                â”‚
â”‚   â”‚ deriv           â”‚   â”‚ snv_deriv       â”‚                â”‚
â”‚   â”‚ (derivative)    â”‚   â”‚ (SNV+derivative)â”‚                â”‚
â”‚   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                â”‚
â”‚                                                             â”‚
â”‚   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                                      â”‚
â”‚   â”‚ deriv_snv       â”‚                                      â”‚
â”‚   â”‚ (derivative+SNV)â”‚                                      â”‚
â”‚   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                                      â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                              â”‚
                              â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                   PREPROCESSED DATA                         â”‚
â”‚                         â”‚                                   â”‚
â”‚                         â–¼                                   â”‚
â”‚                â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                         â”‚
â”‚                â”‚ Machine Learningâ”‚                         â”‚
â”‚                â”‚ Model           â”‚                         â”‚
â”‚                â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                         â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### 16.2 Rationale for Processing Order

**1. Interference removal first:**
- Removes known problematic regions/effects before other processing
- MSC needs to see original scale for reference computation
- OSC/EPO need to identify orthogonal components in original space

**2. Baseline correction second:**
- Works best on interference-cleaned data
- Baseline estimation is more accurate without systematic artifacts

**3. Smoothing third:**
- Reduces noise before derivatives (which amplify noise)
- Should not affect baseline or interference correction

**4. Spectral transformation last:**
- SNV and derivatives work on cleaned, smoothed data
- Final transformation prepares data for modeling

### 16.3 sklearn Pipeline Integration

```python
from sklearn.pipeline import Pipeline
from spectral_predict.preprocess import build_preprocessing_pipeline, SNV, SavgolDerivative
from spectral_predict.baseline import BaselineALS
from spectral_predict.interference import OSC

# Manual pipeline construction
pipeline = Pipeline([
    ('osc', OSC(n_components=1)),           # Step 1: Interference
    ('baseline', BaselineALS(lambda_=1e5)), # Step 2: Baseline
    ('snv', SNV()),                         # Step 4a: SNV
    ('deriv', SavgolDerivative(deriv=1))    # Step 4b: Derivative
])

# Fit and transform
pipeline.fit(X_train, y_train)  # OSC needs y
X_preprocessed = pipeline.transform(X_test)

# Using build_preprocessing_pipeline helper
steps = build_preprocessing_pipeline(
    preprocess_name='snv_deriv',
    deriv=1,
    window=11,
    baseline_method='asls',
    baseline_params={'lam': 1e5}
)
auto_pipeline = Pipeline(steps)
```

### 16.4 Important Considerations

**Data leakage prevention:**
- Fit preprocessing on training data only
- Transform test data using fitted parameters
- Methods like MSC, OSC, EPO store parameters during fit

**Reproducibility:**
- Set random_state for methods with randomness
- Document exact preprocessing parameters
- Save fitted pipelines for later use

**Memory efficiency:**
- Preprocessing creates copies of data
- For large datasets, apply transformations in chunks
- Consider using `float32` instead of `float64`

---

## Summary

This reference guide covers all preprocessing methods available in Spectral Predict V1. The key points to remember:

1. **Start simple**: Try `snv_deriv1` first for most NIR applications
2. **Validate empirically**: Use cross-validation to select preprocessing
3. **Understand the theory**: Know what each method does to interpret results
4. **Consider the order**: Processing sequence affects results
5. **Document choices**: Record preprocessing parameters for reproducibility

For questions about specific preprocessing methods or their application to your data, consult the relevant sections above or reach out to the Spectral Predict community.

---

*Part IV: Spectral Preprocessing Reference - Spectral Predict V1 User Guide*
# Part V: Variable Selection Reference

## Overview

Variable selection (also called wavelength selection or feature selection) is a critical step in spectroscopic modeling that identifies the most informative spectral variables while removing noise, redundant information, and uninformative regions. Effective variable selection can:

- **Improve prediction performance** by focusing on relevant spectral features
- **Reduce model complexity** and prevent overfitting
- **Enhance interpretability** by identifying key spectral regions
- **Decrease computation time** for model training and prediction
- **Improve calibration transfer** between instruments

Spectral Predict implements a comprehensive suite of variable selection methods, ranging from simple importance-based approaches to sophisticated iterative algorithms.

---

## 1. Importance-Based Selection

Importance-based selection uses model-derived importance scores to rank and select the most informative variables. This approach is intuitive, computationally efficient, and applicable to any model that provides feature importance measures.

### 1.1 Model-Specific Importance Methods

#### 1.1.1 VIP (Variable Importance in Projection) for PLS

**Full Name:** Variable Importance in Projection

VIP scores quantify each variable's contribution to the PLS model by measuring how much the variable participates in explaining both the predictor (X) and response (Y) variance.

**Mathematical Formula:**

For a PLS model with $A$ components, the VIP score for variable $j$ is:

$$VIP_j = \sqrt{\frac{p \sum_{a=1}^{A} SS_{Y,a} \cdot w_{ja}^2}{\sum_{a=1}^{A} SS_{Y,a}}}$$

Where:
- $p$ = total number of variables
- $A$ = number of PLS components
- $SS_{Y,a}$ = sum of squares of $Y$ explained by component $a$
- $w_{ja}$ = weight of variable $j$ for component $a$

**Algorithm Steps:**

```
1. Fit PLS model with A components
2. Extract X-weights matrix W (p x A)
3. Extract X-scores matrix T (n x A)
4. For each component a = 1 to A:
   a. Compute SS_Y,a = sum(T[:,a]^2) * var(y)
5. Compute total SS_Y = sum(SS_Y,a for all a)
6. For each variable j = 1 to p:
   a. Compute weighted sum = sum(w[j,a]^2 * SS_Y,a for all a)
   b. VIP[j] = sqrt(p * weighted_sum / total_SS_Y)
7. Return VIP scores
```

**Interpretation:**
- VIP > 1.0: Variable is highly important (rule of thumb threshold)
- VIP between 0.8-1.0: Moderately important
- VIP < 0.8: Less important, candidate for elimination

**Implementation in Spectral Predict:**

```python
def compute_vip(pls_model, X, y):
    W = pls_model.x_weights_    # (n_features, n_components)
    T = pls_model.x_scores_     # (n_samples, n_components)

    # SSY for each component
    y = np.asarray(y).reshape(-1, 1)
    ssy_comp = np.sum(T**2, axis=0) * np.var(y, axis=0)
    ssy_total = np.sum(ssy_comp)

    # VIP calculation (vectorized)
    n_features = W.shape[0]
    weight = np.sum((W ** 2) * ssy_comp, axis=1)
    vip_scores = np.sqrt(n_features * weight / ssy_total)

    return vip_scores
```

---

#### 1.1.2 MDI (Mean Decrease in Impurity) for Tree-Based Models

**Full Name:** Mean Decrease in Impurity (also called Gini Importance)

MDI measures feature importance by computing the total reduction in node impurity (Gini index or entropy) brought by each feature across all trees in the ensemble.

**Mathematical Formula:**

For feature $j$ in a tree ensemble with $T$ trees:

$$MDI_j = \frac{1}{T} \sum_{t=1}^{T} \sum_{v \in \mathcal{V}_t^j} \frac{n_v}{n} \cdot \Delta I(v)$$

Where:
- $\mathcal{V}_t^j$ = set of nodes in tree $t$ that split on feature $j$
- $n_v$ = number of samples reaching node $v$
- $n$ = total number of samples
- $\Delta I(v)$ = impurity decrease at node $v$

**Impurity Decrease:**

$$\Delta I(v) = I(v) - \frac{n_{v,L}}{n_v} I(v_L) - \frac{n_{v,R}}{n_v} I(v_R)$$

Where $v_L$ and $v_R$ are the left and right child nodes.

**Applicability:**
- RandomForest: Uses `feature_importances_` attribute (MDI by default)
- XGBoost: Uses `feature_importances_` (gain-based by default)
- LightGBM: Uses `feature_importances_` (split-count or gain)
- CatBoost: Uses `feature_importances_`

**Strengths:**
- Computationally efficient (computed during training)
- Available for all tree ensemble models
- Handles non-linear relationships

**Limitations:**
- Biased toward high-cardinality features
- Can be misleading with correlated features
- Does not account for feature interactions directly

---

#### 1.1.3 Coefficient-Based Importance for Linear Models

For linear models (Ridge, Lasso, ElasticNet), feature importance is derived from the absolute value of regression coefficients:

$$Importance_j = |c_j|$$

Where $c_j$ is the fitted coefficient for feature $j$.

**Notes:**
- Features should be standardized for meaningful comparison
- Lasso naturally produces sparse coefficients (built-in selection)
- ElasticNet balances L1 (sparsity) and L2 (stability) regularization

---

#### 1.1.4 Permutation Importance

**Full Name:** Permutation Feature Importance

Permutation importance measures how much model performance decreases when a feature's values are randomly shuffled, breaking its relationship with the target.

**Algorithm:**

```
1. Fit model and compute baseline performance metric M_baseline
2. For each feature j = 1 to p:
   a. Create permuted dataset by shuffling column j
   b. Compute performance M_permuted on permuted data
   c. Importance[j] = M_baseline - M_permuted (or ratio)
3. Optionally repeat K times and average
4. Return importance scores
```

**Strengths:**
- Model-agnostic (works with any model)
- Accounts for feature interactions
- Uses actual model predictions

**Limitations:**
- Computationally expensive (requires multiple model evaluations)
- Can underestimate importance of correlated features
- Sensitive to randomness (multiple repetitions recommended)

---

### 1.2 Top-N Selection

Top-N selection is the simplest variable selection strategy: compute importance scores and keep the N most important variables.

**Standard N Values:**

| N | Use Case |
|---|----------|
| 10 | Very aggressive reduction, key wavelengths only |
| 20 | Aggressive reduction, sparse models |
| 50 | Moderate reduction, balanced |
| 100 | Conservative reduction, retains more information |
| 250 | Mild reduction, baseline for comparison |
| 500 | Light reduction, removes obvious noise |
| 1000 | Minimal reduction, large feature sets |

**Algorithm:**

```
1. Compute importance scores for all p variables
2. Sort variables by importance (descending)
3. Select top N variables with highest importance
4. Return selected indices or boolean mask
```

**Selection by Threshold vs Fixed Count:**

| Approach | Description | When to Use |
|----------|-------------|-------------|
| **Fixed Count (Top-N)** | Select exactly N variables | When model complexity must be controlled |
| **Threshold-Based** | Select variables where importance > threshold | When natural grouping exists (e.g., VIP > 1.0) |
| **Percentile-Based** | Select top X% of variables | When relative importance matters |

---

## 2. UVE (Uninformative Variable Elimination)

**Full Name:** Uninformative Variable Elimination

UVE is a statistical method that distinguishes informative variables from noise by comparing real variables against artificial noise variables with known zero information content.

### 2.1 Algorithm

**Detailed Steps:**

```
1. AUGMENTATION
   a. Generate noise matrix N of same size as X
   b. N[i,j] ~ Normal(0, 10^-10) for small noise amplitude
   c. Create augmented matrix X_aug = [X | N] (concatenate)

2. CROSS-VALIDATION MODELING
   For each CV fold k = 1 to K:
   a. Split X_aug into train/test sets
   b. Fit PLS model on augmented training data
   c. Extract regression coefficients c_k for all variables
   d. Store coefficients from each fold

3. RELIABILITY CALCULATION
   For each variable j = 1 to 2p (real + noise):
   a. Compute mean absolute coefficient: mean_j = mean(|c_j|) across folds
   b. Compute standard deviation: std_j = std(c_j) across folds
   c. Reliability score: R_j = mean_j / std_j

4. THRESHOLD DETERMINATION
   a. Extract reliability scores for noise variables: R_noise
   b. Compute threshold: T = max(R_noise) * cutoff_multiplier

5. VARIABLE SELECTION
   a. For real variables (j = 1 to p):
      - If R_j > T: variable is informative
      - If R_j <= T: variable is uninformative
   b. Return importances (reliability scores for real variables)
```

### 2.2 Mathematical Formulas

**Reliability Score:**

$$R_j = \frac{\overline{|c_j|}}{\sigma_{c_j}}$$

Where:
- $\overline{|c_j|}$ = mean of absolute coefficients across CV folds
- $\sigma_{c_j}$ = standard deviation of coefficients across CV folds

**Noise Threshold:**

$$T = \max_{j \in \text{noise}} (R_j) \times \text{cutoff\_multiplier}$$

**Selection Criterion:**

Variable $j$ is selected if: $R_j > T$

### 2.3 Parameters

| Parameter | Default | Description | Effect |
|-----------|---------|-------------|--------|
| `cutoff_multiplier` | 1.0 | Multiplier for noise threshold | >1.0: more conservative (keep more); <1.0: more aggressive (eliminate more) |
| `n_components` | auto | Number of PLS components | Higher values capture more variance but risk overfitting |
| `cv_folds` | 5 | Number of CV folds | More folds = more stable but slower |
| `random_state` | 42 | Random seed for noise generation | Ensures reproducibility |

### 2.4 When to Use

**Strengths:**
- Statistically principled approach
- Automatic threshold determination
- No hyperparameter tuning for threshold
- Effective for removing pure noise variables

**Limitations:**
- Computationally expensive (multiple PLS fits)
- Assumes PLS is appropriate for the data
- May not work well with very few samples
- Noise amplitude selection can affect results

**Computational Considerations:**
- Time complexity: O(K * n * p) for K folds
- Memory: Requires storing augmented matrix (2p columns)
- Minimum features: 3 (for meaningful n_components)

**Reference:**
> Centner, V., Massart, D. L., de Noord, O. E., de Jong, S., Vandeginste, B. M., & Sterna, C. (1996). Elimination of uninformative variables for multivariate calibration. *Analytical Chemistry*, 68(21), 3851-3858.

---

## 3. SPA (Successive Projections Algorithm)

**Full Name:** Successive Projections Algorithm

SPA is a forward selection method that minimizes multicollinearity by iteratively selecting variables that are maximally uncorrelated with the already-selected set.

### 3.1 Algorithm

**Detailed Steps:**

```
1. INITIALIZATION
   a. Normalize X to zero mean and unit variance
   b. Compute correlation of each variable with y
   c. Select initial variable: j_0 = argmax(|corr(X[:,j], y)|)
   d. Initialize selected set S = {j_0}

2. ITERATIVE SELECTION
   For iteration i = 1 to n_features - 1:
   a. Extract selected subspace: X_selected = X[:, S]
   b. Compute QR decomposition: Q, R = qr(X_selected)
   c. For each remaining variable j not in S:
      i. Compute projection onto orthogonal complement:
         x_orth = X[:,j] - Q @ (Q^T @ X[:,j])
      ii. Compute projection norm: norm_j = ||x_orth||
   d. Select variable with maximum projection norm:
      j_next = argmax(norm_j) for j not in S
   e. Add to selected set: S = S âˆª {j_next}

3. QUALITY EVALUATION (canonical seed enumeration, Araújo 2001)
   For each candidate first variable k = 1 to J (every variable in turn):
   a. Run forward selection chain from variable k
   b. Build PLS model on selected variables
   c. Compute CV RÂ² score
   d. Track best chain across all J seeds

4. IMPORTANCE SCORING
   a. Variables selected earlier get higher scores
   b. Score = n_features - rank (first selected = highest)
```

### 3.2 Mathematical Formulas

**Projection onto Orthogonal Complement:**

Given selected variables $\mathbf{X}_S$ with QR decomposition $\mathbf{Q}\mathbf{R}$:

$$\mathbf{x}_j^{\perp} = \mathbf{x}_j - \mathbf{Q}\mathbf{Q}^T\mathbf{x}_j$$

**Selection Criterion:**

$$j^* = \arg\max_{j \notin S} ||\mathbf{x}_j^{\perp}||_2$$

**Correlation (for initialization):**

$$\text{corr}(\mathbf{x}_j, \mathbf{y}) = \frac{\mathbf{x}_j^T \mathbf{y}}{||\mathbf{x}_j|| \cdot ||\mathbf{y}||}$$

### 3.3 Parameters

| Parameter | Default | Description | Effect |
|-----------|---------|-------------|--------|
| `n_features` | required | Number of features to select | Directly controls output size |
| `cv_folds` | 5 | CV folds for quality evaluation | More folds = more stable scores |

SPA is fully deterministic — every variable is enumerated as a candidate
first variable per Araújo 2001, and the chain with the best CV criterion is
returned. There is no random initialization and no `random_state`.

### 3.4 When to Use

**Strengths:**
- Explicitly minimizes multicollinearity
- Fully deterministic (canonical Araújo 2001 enumeration)
- Works well for spectroscopic data with high collinearity

**Limitations:**
- Does not directly optimize prediction performance
- Sequential nature may miss globally optimal subsets
- Initial variable choice affects final selection
- Assumes linear relationships

**Computational Considerations:**
- Time complexity: O(n_starts * n_features * p^2)
- Memory: O(n * p) for QR decomposition
- Minimum features: 3 for meaningful selection

**Reference:**
> Araujo, M. C. U., et al. (2001). The successive projections algorithm for variable selection in spectroscopic multicomponent analysis. *Chemometrics and Intelligent Laboratory Systems*, 57(2), 65-73.

---

## 4. UVE-SPA Hybrid

**Full Name:** Uninformative Variable Elimination followed by Successive Projections Algorithm

The UVE-SPA hybrid combines the noise filtering capability of UVE with the collinearity reduction of SPA in a two-stage approach.

### 4.1 Algorithm

**Stage 1: UVE Filtering**
```
1. Apply UVE algorithm to full variable set
2. Compute reliability scores for all variables
3. Filter out variables below noise threshold
4. Keep informative variables only
```

**Stage 2: SPA Selection**
```
1. Apply SPA to UVE-filtered variables
2. Select n_features from the reduced set
3. Minimize collinearity among informative variables
```

**Combined Steps:**

```
1. Run UVE on X, y with cutoff_multiplier
2. Get UVE mask: variables with reliability > threshold
3. Create reduced matrix: X_uve = X[:, uve_mask]
4. Adjust n_features if UVE kept fewer variables
5. Run SPA on X_uve, y with adjusted n_features
6. Map SPA importances back to original indices
7. Return combined importances (0 for eliminated, SPA scores for kept)
```

### 4.2 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_features` | required | Target number of final features |
| `cutoff_multiplier` | 1.0 | UVE threshold multiplier |
| `uve_n_components` | auto | PLS components for UVE |
| `uve_cv_folds` | 5 | CV folds for UVE |
| `spa_cv_folds` | 5 | CV folds for SPA evaluation |
| `random_state` | 42 | Random seed (used by UVE only; SPA is deterministic) |

### 4.3 When to Use

**Strengths:**
- Combines noise filtering with collinearity reduction
- Produces stable, interpretable selections
- Two-stage approach prevents noise from influencing SPA

**Limitations:**
- More computationally expensive than either method alone
- UVE may over-filter before SPA can optimize
- Requires tuning parameters for both stages

**Best Practice:**
- Use when data has both noise and high collinearity
- Consider running UVE and SPA separately first to understand data characteristics

---

## 5. iPLS (Interval Partial Least Squares)

**Full Name:** Interval Partial Least Squares

iPLS divides the spectrum into contiguous intervals and evaluates each interval's predictive ability independently, identifying the most informative spectral regions.

### 5.1 Algorithm

**Basic iPLS:**

```
1. INTERVAL CREATION
   a. Divide spectrum into n_intervals equal segments
   b. For n_features variables and n_intervals:
      interval_size = n_features / n_intervals
   c. Create interval boundaries: [(0, size), (size, 2*size), ...]

2. INTERVAL EVALUATION
   For each interval i = 1 to n_intervals:
   a. Extract interval data: X_i = X[:, start_i:end_i]
   b. Auto-select PLS components based on interval size
   c. Build PLS model with cross-validation
   d. Compute RMSECV and RÂ² for interval
   e. Store performance metrics

3. IMPORTANCE ASSIGNMENT
   For each variable j:
   a. Find interval containing variable j
   b. Assign interval's RÂ² score as importance
   c. Better intervals = higher importance

4. RETURN
   a. Best interval (lowest RMSECV)
   b. All interval scores
   c. Variable importances based on interval membership
```

### 5.2 Mathematical Formulas

**Interval Boundaries:**

For interval $i$ (0-indexed):
$$\text{start}_i = i \times \lfloor p / n \rfloor$$
$$\text{end}_i = \begin{cases} (i+1) \times \lfloor p / n \rfloor & i < n-1 \\ p & i = n-1 \end{cases}$$

Where $p$ = total variables, $n$ = number of intervals.

**Cross-Validated RMSE:**

$$RMSECV = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_{i,-k(i)})^2}$$

Where $\hat{y}_{i,-k(i)}$ is the prediction for sample $i$ from the model trained without fold $k(i)$.

### 5.3 Parameters

| Parameter | Default | Description | Effect |
|-----------|---------|-------------|--------|
| `n_intervals` | 20 | Number of intervals | 10-40 typical; more intervals = finer resolution |
| `n_components` | auto | PLS components per interval | Auto-adjusted based on interval size |
| `cv_folds` | 5 | CV folds for evaluation | More folds = more stable ranking |
| `random_state` | 42 | Random seed | Ensures reproducibility |

### 5.4 When to Use

**Strengths:**
- Preserves spectral region structure
- Easy to interpret (identifies spectral bands)
- Computationally efficient
- Good for identifying relevant spectral regions

**Limitations:**
- Fixed interval boundaries may miss optimal regions
- Cannot select non-contiguous variables within intervals
- Performance depends on interval count selection

**Reference:**
> Norgaard, L., et al. (2000). Interval partial least-squares regression (iPLS): A comparative chemometric study with an example from near-infrared spectroscopy. *Applied Spectroscopy*, 54(3), 413-419.

---

## 6. iPLS Forward (Forward Interval Selection)

**Full Name:** Forward Interval Partial Least Squares

Forward iPLS starts with an empty selection and iteratively adds the interval that provides the greatest improvement in model performance.

### 6.1 Algorithm

```
1. INITIALIZATION
   a. Divide spectrum into n_intervals
   b. Evaluate each interval independently
   c. Rank intervals by RMSECV (lower = better)
   d. Initialize selected set S = {best_interval}

2. FORWARD SELECTION
   While |S| < max_combine:
   a. For each remaining interval i not in S:
      i. Create combined subset: S_test = S âˆª {i}
      ii. Extract combined variables: X_combined
      iii. Build PLS model and compute RMSECV
   b. Find best addition: i* = argmin(RMSECV over S_test)
   c. If RMSECV(S âˆª {i*}) < RMSECV(S):
      - Add interval: S = S âˆª {i*}
   d. Else:
      - Stop (no improvement)

3. OUTPUT
   a. All individual intervals with scores
   b. Combined subsets from forward selection
   c. Tag each subset with wavelength ranges
```

### 6.2 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_intervals` | 20 | Number of intervals to divide spectrum |
| `max_combine` | 5 | Maximum intervals to combine |
| `cv_folds` | 5 | CV folds for evaluation |
| `random_state` | 42 | Random seed |

### 6.3 When to Use

**Strengths:**
- Systematically explores interval combinations
- Stops when no improvement (avoids overfitting)
- Produces interpretable multi-region selections

**Limitations:**
- May not find globally optimal combination
- Computational cost increases with n_intervals
- Order of addition affects final selection

---

## 7. iPLS Backward (biPLS)

**Full Name:** Backward Interval Partial Least Squares

Backward iPLS starts with all intervals and iteratively removes the interval whose removal causes the least degradation (or most improvement) in model performance.

### 7.1 Algorithm

```
1. INITIALIZATION
   a. Divide spectrum into n_intervals
   b. Start with all intervals: S = {1, 2, ..., n_intervals}
   c. Evaluate full model as baseline

2. BACKWARD ELIMINATION
   While |S| > min_intervals:
   a. For each interval i in S:
      i. Create test subset: S_test = S \ {i}
      ii. Extract combined variables: X_test
      iii. Build PLS model and compute RMSECV
   b. Find best removal: i* = argmin(RMSECV after removing i)
   c. Remove interval: S = S \ {i*}
   d. Record RMSECV at this step

3. OPTIMAL SELECTION
   a. Find step with lowest RMSECV along elimination path
   b. Mark as optimal subset
   c. Return all steps + optimal

4. OUTPUT
   a. Elimination path with RMSECV at each step
   b. Optimal subset (best RMSECV along path)
   c. Tags indicating retained wavelength ranges
```

### 7.2 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_intervals` | 20 | Number of initial intervals |
| `min_intervals` | 1 | Minimum intervals to keep |
| `cv_folds` | 5 | CV folds for evaluation |
| `random_state` | 42 | Random seed |

### 7.3 When to Use

**Strengths:**
- Explores full elimination path
- Automatically finds optimal subset size
- Good when most regions are informative

**Limitations:**
- Computationally expensive for many intervals
- May not recover from bad early eliminations
- Assumes global optimum lies along elimination path

**Reference:**
> Leardi, R. & Norgaard, L. (2004). Sequential application of backward interval PLS and genetic algorithms for the selection of relevant spectral regions. *Journal of Chemometrics*, 18(11), 486-497.

---

## 8. CARS (Competitive Adaptive Reweighted Sampling)

**Full Name:** Competitive Adaptive Reweighted Sampling

CARS is a Monte Carlo-based method that uses adaptive reweighted sampling combined with exponential decay to iteratively select optimal variables while balancing exploration and exploitation.

### 8.1 Algorithm

**Detailed Steps:**

```
1. INITIALIZATION
   a. Initialize weights: w_j = 1 for all j = 1 to p
   b. Initialize history arrays for tracking

2. MONTE CARLO ITERATIONS
   For iteration i = 1 to n_iterations:

   a. COMPUTE SAMPLING RATIO
      r_i = 0.8 * exp(-2i / n_iterations)
      n_sample = max(floor(r_i * p * monte_carlo_percent/100),
                     pls_components + 1)

   b. WEIGHTED SAMPLING
      - Normalize weights: prob_j = w_j / sum(w)
      - Sample n_sample variables without replacement
      - Probability proportional to weights

   c. MODEL BUILDING AND EVALUATION
      - Extract sampled variables: X_subset
      - Build PLS model with cross-validation
      - Compute RMSECV for this iteration

   d. WEIGHT UPDATE (Adaptive Reweighting)
      - Fit PLS on full sampled subset
      - Extract coefficients c_j for sampled variables
      - Update weights: w_j = |c_j| / max(|c|)
      - Normalize weights to sum to 1

   e. RECORD HISTORY
      - Store RMSECV, n_selected, selected indices

3. OPTIMAL SELECTION
   a. Find iteration with minimum RMSECV
   b. Extract selected variables from best iteration

4. IMPORTANCE SCORING
   a. Variables in best iteration get accumulated weights
   b. Variables not in best iteration get zero importance
```

### 8.2 Mathematical Formulas

**Exponential Decay Function:**

$$r_i = 0.8 \times \exp\left(-\frac{2i}{n_{iter}}\right)$$

Where $i$ is the current iteration and $n_{iter}$ is total iterations.

**Number of Variables to Sample:**

$$n_{sample} = \max\left(\lfloor r_i \times p \times \frac{mc\%}{100} \rfloor, n_{comp} + 1\right)$$

**Sampling Probability:**

$$P(j) = \frac{w_j}{\sum_{k=1}^{p} w_k}$$

**Weight Update:**

$$w_j^{(i+1)} = \frac{|c_j^{(i)}|}{\max_k |c_k^{(i)}|}$$

Where $c_j^{(i)}$ is the PLS coefficient for variable $j$ at iteration $i$.

### 8.3 Parameters

| Parameter | Default | Description | Typical Range |
|-----------|---------|-------------|---------------|
| `n_iterations` | 50 | Monte Carlo iterations | 50-100 |
| `monte_carlo_samples` | 80 | Initial sampling percentage | 70-90 |
| `pls_components` | 5 | PLS components for evaluation | 5-15 |
| `cv_folds` | 5 | CV folds | 5-10 |
| `random_state` | 42 | Random seed | - |
| `model_type` | None | Model for tree-based CARS | - |
| `use_hybrid_importance` | False | Enable CARS-Tree mode | - |
| `hybrid_importance_weight` | 0.5 | Split/gain blend weight | 0.3-0.7 |

### 8.4 When to Use

**Strengths:**
- Balances exploration (random sampling) and exploitation (weight-based)
- Exponential decay forces progressive elimination
- Often produces very compact variable sets
- Cross-validated selection prevents overfitting

**Limitations:**
- Computationally expensive (many model fits)
- Results can vary with random seed
- Requires careful selection of pls_components
- May converge to local optima

**Computational Considerations:**
- Time complexity: O(n_iter * cv_folds * n * n_sample)
- Memory: O(n * p) for data + O(n_iter) for history
- Minimum features: 7 for meaningful selection

**Reference:**
> Li, H. D., et al. (2009). Key wavelengths screening using competitive adaptive reweighted sampling method for multivariate calibration. *Analytica Chimica Acta*, 648(1), 77-84.

---

## 9. CARS-Tree Hybrid

**Full Name:** Competitive Adaptive Reweighted Sampling with Tree-Based Importance

CARS-Tree extends CARS by using LightGBM-based importance instead of PLS coefficients, with a hybrid importance measure that blends split-based and gain-based importance.

### 9.1 Algorithm Modifications

The CARS-Tree variant modifies the standard CARS algorithm in the weight update step:

```
WEIGHT UPDATE (Tree-Based, Hybrid Mode):
1. Fit LightGBM model on sampled variables
2. Extract split importance: split_imp = booster.feature_importance('split')
3. Extract gain importance: gain_imp = booster.feature_importance('gain')
4. Normalize each:
   - split_norm = split_imp / sum(split_imp)
   - gain_norm = gain_imp / sum(gain_imp)
5. Compute hybrid importance:
   - hybrid_imp = weight * split_norm + (1 - weight) * gain_norm
6. Update weights: w_j = hybrid_imp_j
7. Add floor to prevent complete elimination: w_j = max(w_j, 1e-6)
```

### 9.2 Hybrid Importance Formula

$$I_{hybrid,j} = \alpha \times I_{split,j}^{norm} + (1-\alpha) \times I_{gain,j}^{norm}$$

Where:
- $\alpha$ = `hybrid_importance_weight` (default 0.5)
- $I_{split}^{norm}$ = normalized split count importance
- $I_{gain}^{norm}$ = normalized gain importance

### 9.3 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_type` | required | Must be tree model (RandomForest, XGBoost, LightGBM, CatBoost) |
| `use_hybrid_importance` | True | Enable hybrid split+gain importance |
| `hybrid_importance_weight` | 0.5 | Balance between split (1.0) and gain (0.0) |
| `task_type` | 'regression' | 'regression' or 'classification' |

### 9.4 When to Use

**Strengths:**
- Better suited when final model is tree-based
- Hybrid importance produces denser distributions than split-only
- Handles non-linear relationships naturally

**Limitations:**
- Computationally heavier than PLS-based CARS
- May not be optimal for linear final models
- LightGBM parameters affect results

---

## 10. GA-PLS (Genetic Algorithm PLS)

**Full Name:** Genetic Algorithm for Partial Least Squares Variable Selection

GA-PLS uses genetic algorithm optimization to search for the optimal subset of variables, where each candidate solution (chromosome) is a binary mask indicating which variables are selected.

### 10.1 Algorithm

**Detailed Steps:**

```
1. CHROMOSOME REPRESENTATION
   - Binary string of length p (number of variables)
   - Gene j = 1: variable j is selected
   - Gene j = 0: variable j is excluded

2. POPULATION INITIALIZATION
   For i = 1 to population_size:
   a. Generate random binary chromosome
   b. Ensure minimum wavelengths selected (constraint repair)
   c. Optionally seed some chromosomes with prior importances

3. FITNESS EVALUATION
   For each chromosome:
   a. Extract selected variables: X_subset = X[:, chromosome == 1]
   b. Build PLS model with cross-validation
   c. Compute RMSECV
   d. Fitness = -RMSECV (maximize fitness = minimize error)

4. GENETIC OPERATORS
   a. SELECTION (Tournament)
      - Select tournament_size individuals randomly
      - Winner = individual with best fitness
      - Repeat until new population filled

   b. CROSSOVER (Two-Point)
      - With probability crossover_rate:
        * Select two crossover points
        * Exchange segments between parents
        * Create two children

   c. MUTATION (Bit-Flip)
      - For each gene with probability mutation_rate:
        * Flip bit (0 -> 1 or 1 -> 0)
      - Repair if minimum wavelengths violated

5. ELITISM
   - Keep top 2 individuals unchanged
   - Replace rest with offspring

6. TERMINATION
   - Stop after n_generations
   - Or early stopping if no improvement for early_stopping generations

7. MULTI-RUN AGGREGATION
   - Run GA n_runs times with different seeds
   - Compute selection frequency for each variable
   - Importance = frequency of selection across runs
```

### 10.2 Genetic Operators in Detail

**Tournament Selection:**

```
def tournament_selection(population, fitness, tournament_size=3):
    competitors = random_sample(population, tournament_size)
    winner = competitor with max(fitness)
    return winner
```

**Two-Point Crossover:**

```
def two_point_crossover(parent1, parent2, rate=0.6):
    if random() > rate:
        return parent1.copy(), parent2.copy()

    points = sorted(random_choice(length, 2))
    child1 = parent1.copy()
    child2 = parent2.copy()
    child1[points[0]:points[1]] = parent2[points[0]:points[1]]
    child2[points[0]:points[1]] = parent1[points[0]:points[1]]

    return child1, child2
```

**Bit-Flip Mutation:**

```
def bit_flip_mutation(chromosome, rate=0.01, min_wavelengths=5):
    mutated = chromosome.copy()
    mask = random(length) < rate
    mutated[mask] = 1 - mutated[mask]

    # Repair if needed
    if sum(mutated) < min_wavelengths:
        add_random_ones(mutated, min_wavelengths - sum(mutated))

    return mutated
```

### 10.3 Parameters

| Parameter | Default | Description | Typical Range |
|-----------|---------|-------------|---------------|
| `population_size` | 64 | Number of individuals | 50-200 |
| `n_generations` | 100 | Maximum generations | 50-200 |
| `crossover_rate` | 0.6 | Crossover probability | 0.4-0.9 |
| `mutation_rate` | 0.01 | Per-gene mutation rate | 0.005-0.05 |
| `n_runs` | 5 | Independent GA runs | 3-10 |
| `selection_threshold` | 0.6 | Min frequency for stable selection | 0.5-0.8 |
| `min_wavelengths` | 5 | Minimum selected variables | 5-20 |
| `early_stopping` | 20 | Generations without improvement | 10-30 |
| `n_components` | 10 | PLS components for fitness | 5-15 |
| `cv` | 5 | CV folds for fitness | 5-10 |

### 10.4 When to Use

**Strengths:**
- Global search capability (explores large solution space)
- No assumptions about variable relationships
- Multi-run aggregation provides stability
- Can escape local optima

**Limitations:**
- Computationally expensive (many fitness evaluations)
- Parameter-sensitive (many hyperparameters to tune)
- No guarantee of global optimum
- Results vary with random seed

**Computational Considerations:**
- Time complexity: O(n_runs * n_gen * pop_size * cv_folds * n * p)
- For 2000 wavelengths: single run ~2-5 minutes, multi-run ~10-25 minutes
- Consider "quick" settings for exploration: pop_size=32, n_gen=50, n_runs=3

**Reference:**
> Leardi, R. & Lupianez Gonzalez, A. (1998). Genetic algorithms applied to feature selection in PLS regression. *Chemometrics and Intelligent Laboratory Systems*, 41(2), 195-207.

---

## 11. VCPA-IRIV

**Full Name:** Variable Combination Population Analysis - Iteratively Retains Informative Variables

VCPA-IRIV is an advanced iterative method that classifies variables into categories based on statistical tests comparing model performance when variables are included versus excluded.

### 11.1 Algorithm

**Detailed Steps:**

```
1. INITIALIZATION
   - Start with all p variables as candidates
   - active_indices = [0, 1, 2, ..., p-1]

2. OUTER ITERATION LOOP
   For iteration = 1 to n_outer_iterations:

   a. BINARY MATRIX GENERATION
      - Create BM with binary_matrix_samples rows
      - Each row: random subset (~50% inclusion probability)
      - Ensure each row has minimum variables for model

   b. INCLUDE/EXCLUDE COMPARISON
      For each row in BM:
      i. Extract selected variables
      ii. Build model (PLS or LightGBM) with CV
      iii. Record RMSECV

      For each variable j:
      - Collect RMSECV_include: errors when j is included
      - Collect RMSECV_exclude: errors when j is excluded

   c. STATISTICAL CLASSIFICATION
      For each variable j:
      i. Compute DMEAN = mean(exclude) - mean(include)
         - Positive DMEAN: including improves performance
         - Negative DMEAN: including worsens performance

      ii. Apply Mann-Whitney U test (p < 0.05 threshold)
          H = 1 if significant, 0 otherwise

      iii. Classify variable:
           - H=1, DMEAN>0: STRONGLY INFORMATIVE (keep)
           - H=0, DMEAN>0: WEAKLY INFORMATIVE (keep)
           - H=0, DMEANâ‰¤0: UNINFORMATIVE (remove)
           - H=1, DMEAN<0: INTERFERING (remove)

   d. VARIABLE ELIMINATION
      - Remove uninformative and interfering variables
      - Update active_indices

   e. CONVERGENCE CHECK
      - Stop if no variables removed
      - Stop if too few variables remain

3. OUTPUT
   - selected_indices: final selected variables
   - variable_categories: classification for each variable
   - importance_scores: 1.0 (strong), 0.5 (weak), 0.25 (other)
   - convergence_history: RMSECV at each iteration
```

### 11.2 Mathematical Formulas

**DMEAN (Difference in Means):**

$$DMEAN_j = \overline{RMSECV}_{exclude,j} - \overline{RMSECV}_{include,j}$$

**Interpretation:**
- $DMEAN_j > 0$: Including variable $j$ reduces error (informative)
- $DMEAN_j \leq 0$: Including variable $j$ increases error (uninformative/interfering)

**Mann-Whitney U Test:**

Tests whether the distribution of RMSECV values when variable $j$ is included differs significantly from when it is excluded.

$$H_0: \text{Distribution}_{include} = \text{Distribution}_{exclude}$$

$$H = \begin{cases} 1 & \text{if } p < 0.05 \\ 0 & \text{otherwise} \end{cases}$$

**Variable Classification:**

| H | DMEAN | Category | Action |
|---|-------|----------|--------|
| 1 | > 0 | Strongly Informative | Keep |
| 0 | > 0 | Weakly Informative | Keep |
| 0 | â‰¤ 0 | Uninformative | Remove |
| 1 | < 0 | Interfering | Remove |

### 11.3 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_outer_iterations` | 10 | Maximum elimination rounds |
| `n_inner_iterations` | 50 | Deprecated (use binary_matrix_samples) |
| `binary_matrix_samples` | 100 | Number of random combinations per iteration |
| `pls_components` | 5 | PLS components (when using PLS) |
| `cv_folds` | 5 | CV folds for model evaluation |
| `model_type` | None | 'PLS' (default) or tree model name |
| `importance_threshold` | 0.5 | Deprecated (statistical test used instead) |
| `random_state` | None | Random seed |

### 11.4 When to Use

**Strengths:**
- Statistically rigorous classification
- Identifies interfering variables (unique capability)
- Iterative refinement improves robustness
- No arbitrary threshold selection

**Limitations:**
- Computationally expensive (many model evaluations)
- Requires sufficient samples for reliable statistics
- May converge prematurely with aggressive settings

**Computational Considerations:**
- Time complexity: O(n_outer * binary_samples * cv_folds * n * p)
- Adaptive binary_matrix_samples based on variable count
- Consider model_type='LightGBM' for tree-based final models

**Reference:**
> Yun, Y. H., et al. (2014). A strategy that iteratively retains informative variables for selecting optimal variable subset in multivariate calibration. *Analytica Chimica Acta*, 807, 36-43.

---

## Method Comparison Table

| Method | Type | Supervision | Collinearity | Noise Filtering | Computation | Variables Selected |
|--------|------|-------------|--------------|-----------------|-------------|-------------------|
| **VIP/Top-N** | Importance | Yes | No | Partial | Fast | Fixed N |
| **UVE** | Statistical | Yes | No | Yes | Medium | Auto (threshold) |
| **SPA** | Projection | Partial | Yes | No | Fast | Fixed N |
| **UVE-SPA** | Hybrid | Yes | Yes | Yes | Medium | Fixed N |
| **iPLS** | Interval | Yes | Partial | Partial | Medium | Interval-based |
| **iPLS Forward** | Interval | Yes | Partial | Partial | Medium | Multi-interval |
| **iPLS Backward** | Interval | Yes | Partial | Partial | Medium-High | Auto (optimal) |
| **CARS** | Monte Carlo | Yes | No | Partial | High | Auto (best iteration) |
| **CARS-Tree** | Monte Carlo | Yes | No | Partial | High | Auto (best iteration) |
| **GA-PLS** | Evolutionary | Yes | No | No | Very High | Frequency-based |
| **VCPA-IRIV** | Statistical | Yes | No | Yes | Very High | Auto (convergence) |

---

## Selection Guide

### By Data Characteristics

| Situation | Recommended Methods |
|-----------|-------------------|
| High collinearity | SPA, UVE-SPA |
| High noise levels | UVE, VCPA-IRIV |
| Unknown optimal size | CARS, iPLS Backward, VCPA-IRIV |
| Need interpretable regions | iPLS variants |
| Large variable count (>2000) | Top-N first, then refine |
| Small sample size (<50) | UVE, SPA (simpler methods) |
| Tree-based final model | CARS-Tree, VCPA-IRIV with model_type |

### By Computational Budget

| Time Available | Recommended Methods |
|----------------|-------------------|
| < 1 minute | Top-N, SPA |
| 1-5 minutes | UVE, iPLS |
| 5-30 minutes | CARS, UVE-SPA |
| 30+ minutes | GA-PLS, VCPA-IRIV |

### By Goal

| Goal | Recommended Methods |
|------|-------------------|
| Quick screening | Top-N, iPLS |
| Maximum performance | GA-PLS, VCPA-IRIV |
| Identify spectral bands | iPLS variants |
| Remove noise only | UVE |
| Reduce collinearity only | SPA |
| Both noise and collinearity | UVE-SPA |
| Statistical rigor | UVE, VCPA-IRIV |

---

## Best Practices

### 1. Preprocessing Before Selection

- Apply spectral preprocessing (SNV, derivatives) before variable selection
- Variable selection operates on preprocessed data
- Preprocessing affects which variables appear informative

### 2. Cross-Validation Strategy

- Always use cross-validation within selection methods
- External validation set should not influence selection
- Consider stratified CV for classification tasks

### 3. Method Combination

- Start with fast methods (Top-N, SPA) for exploration
- Refine with more sophisticated methods (CARS, GA-PLS)
- Compare multiple methods and look for consensus

### 4. Parameter Tuning

- Use default parameters initially
- Tune based on CV performance, not test performance
- Document parameter choices for reproducibility

### 5. Stability Assessment

- Run stochastic methods (CARS, GA-PLS) multiple times
- Look for variables consistently selected across runs
- Selection frequency indicates variable importance reliability

### 6. Interpretation

- Map selected indices back to wavelengths
- Verify selected regions make spectroscopic sense
- Consult domain knowledge about expected absorption bands

---

## Glossary

| Term | Definition |
|------|------------|
| **Variable** | A single spectral measurement point (wavelength/wavenumber) |
| **Feature** | Synonym for variable in machine learning context |
| **Wavelength** | Spectral position in nm or Î¼m |
| **Wavenumber** | Spectral position in cm^-1 |
| **Collinearity** | High correlation between variables |
| **RMSECV** | Root Mean Square Error of Cross-Validation |
| **CV** | Cross-Validation |
| **PLS** | Partial Least Squares |
| **VIP** | Variable Importance in Projection |
| **MDI** | Mean Decrease in Impurity |
| **Binary Matrix** | Random 0/1 matrix for variable combination testing |
| **Chromosome** | Binary representation of variable selection in GA |
| **Fitness** | Quality measure for GA selection (negative RMSECV) |

---

## References

1. AraÃºjo, M. C. U., et al. (2001). The successive projections algorithm for variable selection in spectroscopic multicomponent analysis. *Chemometrics and Intelligent Laboratory Systems*, 57(2), 65-73.

2. Centner, V., et al. (1996). Elimination of uninformative variables for multivariate calibration. *Analytical Chemistry*, 68(21), 3851-3858.

3. Li, H. D., et al. (2009). Key wavelengths screening using competitive adaptive reweighted sampling method for multivariate calibration. *Analytica Chimica Acta*, 648(1), 77-84.

4. Leardi, R. & LupiÃ¡Ã±ez GonzÃ¡lez, A. (1998). Genetic algorithms applied to feature selection in PLS regression. *Chemometrics and Intelligent Laboratory Systems*, 41(2), 195-207.

5. NÃ¸rgaard, L., et al. (2000). Interval partial least-squares regression (iPLS): A comparative chemometric study with an example from near-infrared spectroscopy. *Applied Spectroscopy*, 54(3), 413-419.

6. Leardi, R. & NÃ¸rgaard, L. (2004). Sequential application of backward interval PLS and genetic algorithms for the selection of relevant spectral regions. *Journal of Chemometrics*, 18(11), 486-497.

7. Yun, Y. H., et al. (2014). A strategy that iteratively retains informative variables for selecting optimal variable subset in multivariate calibration. *Analytica Chimica Acta*, 807, 36-43.

8. Stefansson, P., et al. (2020). Fast method for GA-PLS with simultaneous feature selection and identification of optimal preprocessing technique. *Journal of Chemometrics*, 34(3), e3195.
# Part VI: Hyperparameter Optimization Reference

This section provides a comprehensive reference for the three hyperparameter optimization methods available in Spectral Predict V1: Grid Search, Bayesian Optimization (TPE), and NSGA-II Multi-Objective Optimization. Understanding the strengths and appropriate use cases for each method is essential for building optimal spectroscopic models.

---

## Table of Contents

1. [Introduction to Hyperparameter Optimization](#1-introduction-to-hyperparameter-optimization)
2. [Grid Search](#2-grid-search)
3. [Bayesian Optimization (TPE)](#3-bayesian-optimization-tpe)
4. [NSGA-II Multi-Objective Optimization](#4-nsga-ii-multi-objective-optimization)
5. [Choosing an Optimization Method](#5-choosing-an-optimization-method)
6. [Computational Considerations](#6-computational-considerations)
7. [Best Practices](#7-best-practices)

---

## 1. Introduction to Hyperparameter Optimization

### 1.1 Why Hyperparameter Optimization Matters

Machine learning models have two types of parameters:

1. **Model Parameters**: Learned from data during training (e.g., regression coefficients, neural network weights)
2. **Hyperparameters**: Set before training and control the learning process (e.g., number of PLS components, regularization strength, tree depth)

Hyperparameter optimization is the process of finding the best combination of hyperparameters that maximizes model performance on unseen data. In spectroscopy, this includes not just model hyperparameters but also:

- **Preprocessing choices**: SNV, derivatives, window sizes
- **Variable selection**: Which wavelengths to include
- **Model architecture**: Number of latent variables, tree depth, learning rate

### 1.2 The Search Space Challenge

For a typical spectroscopic analysis, the search space can be enormous:

```
Example Search Space:
- Preprocessing methods: 6 options (raw, snv, deriv1, deriv2, snv_deriv1, snv_deriv2)
- Window sizes: 7 options (7, 11, 15, 19, 23, 27, 31)
- Model types: 8 options (PLS, Ridge, LightGBM, XGBoost, etc.)
- Model hyperparameters: 10-50 combinations per model
- Variable subsets: 10-20 options

Total combinations: 6 x 7 x 8 x 30 x 15 = 151,200+ configurations
```

Exhaustively evaluating all combinations is computationally prohibitive, which motivates the use of intelligent search strategies.

### 1.3 Overview of Optimization Methods

| Method | Strategy | Best For | Time |
|--------|----------|----------|------|
| **Grid Search** | Exhaustive enumeration | Complete coverage, reproducibility | Medium |
| **Bayesian (TPE)** | Probabilistic modeling | Joint optimization, continuous params | Medium |
| **NSGA-II** | Evolutionary multi-objective | Pareto analysis, parsimony | Long |

---

## 2. Grid Search

### 2.1 Full Name and Abbreviation

**Grid Search** (also known as **Exhaustive Search** or **Parameter Sweep**)

### 2.2 Algorithm Explanation

Grid Search is the simplest hyperparameter optimization strategy. It defines a discrete grid of hyperparameter values and evaluates every possible combination systematically.

```
Algorithm: Grid Search
-----------------------
Input: Parameter grids {P1: [v1, v2, ...], P2: [w1, w2, ...], ...}
Output: Best parameter combination

1. Generate all combinations: C = P1 x P2 x P3 x ...
2. For each combination c in C:
   a. Train model with parameters c
   b. Evaluate via cross-validation
   c. Store score
3. Return combination with best score
```

**Advantages:**
- Guaranteed to find the best combination within the grid
- Highly parallelizable
- Results are reproducible and deterministic
- Easy to understand and interpret

**Disadvantages:**
- Computationally expensive (exponential scaling)
- Cannot explore continuous parameter spaces
- May miss optimal values between grid points
- Wastes computation on poor regions of the space

### 2.3 How Grid Search Works in Spectral Predict

Spectral Predict implements Grid Search with several enhancements:

1. **Tiered Parameter Grids**: Four predefined tiers control search thoroughness
2. **Preprocessing Integration**: Each preprocessing method is tested independently
3. **Variable Selection**: Top-N wavelength subsets based on feature importance
4. **Parallel Execution**: Uses joblib for multi-core parallelization
5. **Progress Tracking**: Real-time updates with best model so far

```
Spectral Predict Grid Search Flow:
==================================

[Data] --> [Preprocessing Methods]
                    |
                    v
           +------------------+
           | For each method: |
           |  - raw           |
           |  - snv           |
           |  - deriv1        |
           |  - deriv2        |
           |  - snv_deriv1    |
           |  - snv_deriv2    |
           +------------------+
                    |
                    v
           [Model Grid Search]
                    |
                    v
    +-------------------------------+
    | For each model type:          |
    |   For each hyperparameter set:|
    |     - Cross-validate          |
    |     - Record metrics          |
    +-------------------------------+
                    |
                    v
           [Variable Selection]
                    |
                    v
    +-------------------------------+
    | For each subset size:         |
    |   - Compute importances       |
    |   - Select top-N wavelengths  |
    |   - Retrain and evaluate      |
    +-------------------------------+
                    |
                    v
           [Ranked Results]
```

### 2.4 The Tier System

Spectral Predict uses a tiered system to balance thoroughness with computational cost.

#### 2.4.1 Quick Tier

**Purpose**: Rapid preliminary analysis, daily QC, quick exploration

**Models Included** (Regression):
- PLS
- RandomForest

**Models Included** (Classification):
- PLS-DA
- RandomForest

**Preprocessing**:
- Window sizes: [11]
- Methods: raw, snv

**Characteristics**:
- Minimal hyperparameter combinations
- Fast execution (minutes)
- Good for initial data exploration

#### 2.4.2 Standard Tier

**Purpose**: Most users, routine analysis, daily work

**Models Included** (Regression):
- PLS, Ridge, Lasso, ElasticNet, RandomForest, LightGBM

**Models Included** (Classification):
- PLS-DA, RandomForest, LightGBM, XGBoost, CatBoost

**Preprocessing**:
- Window sizes: [7, 19]
- Methods: raw, snv, deriv1, deriv2

**Characteristics**:
- Balanced coverage
- Moderate execution time (10-30 minutes)
- Recommended for most applications

#### 2.4.3 Comprehensive Tier

**Purpose**: Thorough analysis, research, publications

**Models Included** (Regression):
- PLS, Ridge, ElasticNet, RandomForest, LightGBM, XGBoost, CatBoost, NeuralBoosted

**Models Included** (Classification):
- PLS-DA, RandomForest, LightGBM, XGBoost, CatBoost, SVM, MLP, NeuralBoosted

**Preprocessing**:
- Window sizes: [7, 15, 25]
- Methods: raw, snv, deriv1, deriv2, snv_deriv1, snv_deriv2

**Characteristics**:
- Extended hyperparameter ranges
- Longer execution (30-60 minutes)
- Suitable for publication-quality models

#### 2.4.4 Experimental Tier

**Purpose**: Research exploration, method comparison, no time constraints

**Models Included** (Regression):
- All available models including SVR, MLP

**Models Included** (Classification):
- All available models

**Preprocessing**:
- All window sizes and methods

**Characteristics**:
- Maximum coverage
- Long execution (hours)
- For comprehensive method comparison

### 2.5 Model-Specific Parameter Grids

#### PLS / PLS-DA

| Parameter | Quick | Standard | Comprehensive |
|-----------|-------|----------|---------------|
| n_components | 1-8 | 1-15 | 1-20 |
| max_iter | [500] | [500] | [500] |
| tol | [1e-6] | [1e-6] | [1e-6] |

#### Ridge

| Parameter | Quick | Standard | Comprehensive |
|-----------|-------|----------|---------------|
| alpha | [1.0] | [0.001, 0.01, 0.1, 1.0, 10.0] | Extended range |
| solver | ['auto'] | ['auto'] | ['auto'] |
| tol | [1e-4] | [1e-4] | [1e-4] |

#### Lasso

| Parameter | Quick | Standard | Comprehensive |
|-----------|-------|----------|---------------|
| alpha | [0.1] | [0.001, 0.01, 0.1, 1.0] | Extended range |
| selection | ['cyclic'] | ['cyclic'] | ['cyclic'] |

#### ElasticNet

| Parameter | Quick | Standard | Comprehensive |
|-----------|-------|----------|---------------|
| alpha | [0.1] | [0.01, 0.1, 1.0] | Extended range |
| l1_ratio | [0.5] | [0.3, 0.5, 0.7] | [0.1, 0.3, 0.5, 0.7, 0.9] |

#### RandomForest

| Parameter | Quick | Standard | Comprehensive |
|-----------|-------|----------|---------------|
| n_estimators | [100] | [100] | [100, 200] |
| max_depth | [None] | [None, 30] | [None, 10, 20, 30] |
| min_samples_split | [2] | [2] | [2, 5] |
| max_features | ['sqrt'] | ['sqrt'] | ['sqrt', 'log2'] |

#### LightGBM

| Parameter | Quick | Standard | Comprehensive |
|-----------|-------|----------|---------------|
| n_estimators | [100] | [100, 200] | [100, 200, 300] |
| learning_rate | [0.1] | [0.05, 0.1] | [0.01, 0.05, 0.1] |
| num_leaves | [31] | [15, 31] | [15, 31, 63] |
| max_depth | [-1] | [5, 10, -1] | [5, 10, 15, -1] |
| reg_alpha | [0.1] | [0.1, 0.5] | [0.1, 0.5, 1.0] |
| reg_lambda | [1.0] | [1.0, 2.0] | [1.0, 2.0, 5.0] |

#### XGBoost

| Parameter | Quick | Standard | Comprehensive |
|-----------|-------|----------|---------------|
| n_estimators | [100] | [100, 200] | [100, 200, 300] |
| learning_rate | [0.1] | [0.05, 0.1] | [0.01, 0.05, 0.1] |
| max_depth | [6] | [3, 6] | [3, 6, 9] |
| subsample | [0.8] | [0.8] | [0.6, 0.8, 1.0] |
| colsample_bytree | [0.8] | [0.6, 0.8] | [0.5, 0.7, 0.9] |
| reg_alpha | [0.1] | [0.1, 0.2] | [0.1, 0.2, 0.5] |
| reg_lambda | [1.0] | [1.0, 2.0] | [1.0, 2.0, 5.0] |

#### CatBoost

| Parameter | Quick | Standard | Comprehensive |
|-----------|-------|----------|---------------|
| iterations | [100] | [100, 200] | [100, 200, 300] |
| learning_rate | [0.1] | [0.05, 0.1] | [0.01, 0.05, 0.1] |
| depth | [6] | [4, 5, 6] | [4, 5, 6, 8] |
| l2_leaf_reg | [3.0] | [3.0, 5.0] | [1.0, 3.0, 5.0, 10.0] |

#### SVR / SVM

| Parameter | Quick | Standard | Comprehensive |
|-----------|-------|----------|---------------|
| kernel | ['rbf'] | ['rbf', 'linear'] | ['rbf', 'linear', 'poly'] |
| C | [1.0] | [1.0, 10.0] | [0.1, 1.0, 10.0, 100.0] |
| gamma | ['scale'] | ['scale'] | ['scale', 'auto'] |
| epsilon (SVR) | [0.1] | [0.1] | [0.01, 0.1, 0.2] |

#### MLP

| Parameter | Quick | Standard | Comprehensive |
|-----------|-------|----------|---------------|
| hidden_layer_sizes | [(64,)] | [(64,), (128, 64)] | [(64,), (128,), (128, 64), (256, 128)] |
| alpha | [0.001] | [0.001] | [0.0001, 0.001, 0.01] |
| learning_rate_init | [0.001] | [0.001] | [0.0001, 0.001, 0.01] |
| activation | ['relu'] | ['relu'] | ['relu', 'tanh'] |

### 2.6 Total Combinations Calculation

The total number of configurations tested in Grid Search is:

$$N_{total} = N_{preprocess} \times N_{models} \times \prod_{i} |P_i| \times N_{subsets}$$

Where:
- $N_{preprocess}$ = Number of preprocessing methods
- $N_{models}$ = Number of model types
- $|P_i|$ = Size of parameter grid for parameter $i$
- $N_{subsets}$ = Number of variable subsets tested

**Example (Standard Tier)**:
- Preprocessing: 4 methods x 2 windows = 8
- Models: 6 types
- Average params per model: ~10 combinations
- Variable subsets: 5 sizes

Total: $8 \times 6 \times 10 \times 5 = 2,400$ configurations

### 2.7 Parallelization via Joblib

Spectral Predict uses joblib for parallel execution:

```python
# Parallel model evaluation
from joblib import Parallel, delayed

results = Parallel(n_jobs=-1)(
    delayed(evaluate_model)(config)
    for config in configurations
)
```

**Parallelization benefits**:
- `n_jobs=-1`: Uses all available CPU cores
- Near-linear speedup for embarrassingly parallel workloads
- Automatic memory management and task distribution

---

## 3. Bayesian Optimization (TPE)

### 3.1 Full Name and Abbreviation

**Bayesian Optimization** using **Tree-structured Parzen Estimator (TPE)**

### 3.2 Algorithm Explanation

Bayesian Optimization is a sequential model-based optimization strategy that builds a probabilistic model of the objective function and uses it to intelligently select the next points to evaluate.

#### 3.2.1 The TPE Algorithm

TPE is a specific type of Bayesian optimization that models the search differently from traditional approaches:

**Traditional Bayesian Optimization**:
- Models $p(y|x)$ - probability of score given parameters
- Uses Gaussian Process surrogate

**TPE Approach**:
- Models $p(x|y)$ - probability of parameters given score
- Splits observations into "good" (top $\gamma$%) and "bad"
- Models each group separately using kernel density estimation

**Mathematical Formulation**:

Given observations $D = \{(x_1, y_1), ..., (x_n, y_n)\}$, TPE defines:

$$l(x) = p(x | y < y^*)$$
$$g(x) = p(x | y \geq y^*)$$

Where $y^*$ is the $\gamma$-quantile of observed scores (typically $\gamma = 0.25$).

The **Expected Improvement (EI)** acquisition function becomes:

$$EI(x) \propto \frac{l(x)}{g(x)}$$

This ratio is high when:
- $l(x)$ is large: parameters likely to produce good results
- $g(x)$ is small: parameters unlikely to produce bad results

```
TPE Algorithm:
==============

1. Initialize: Evaluate n_startup random configurations
2. For trial t = n_startup to n_trials:
   a. Split observations into good (y < quantile) and bad (y >= quantile)
   b. Fit KDE l(x) to good observations
   c. Fit KDE g(x) to bad observations
   d. For each candidate x:
      - Compute l(x)/g(x) ratio
   e. Select x* with highest ratio
   f. Evaluate f(x*) and add to observations
3. Return best observed configuration
```

#### 3.2.2 Handling Conditional Parameters

TPE naturally handles **tree-structured** (conditional) parameter spaces:

```
Example: Window size only relevant when derivative is enabled

if preprocessing in ['deriv1', 'deriv2']:
    window = suggest_int('window', 5, 51, step=2)
else:
    window = None  # Not applicable
```

This is why TPE is particularly well-suited for spectroscopic optimization where preprocessing choices affect which downstream parameters are relevant.

### 3.3 How Bayesian Optimization Works in Spectral Predict

Spectral Predict's Bayesian optimization (implemented in `unified_bayesian.py`) provides **joint optimization** of:

1. **Preprocessing**: Method selection and parameters
2. **Model Hyperparameters**: All relevant parameters for the chosen model
3. **Variable Selection**: Method and subset size
4. **Regional Subsets**: Dynamically computed on preprocessed data

```
Unified Bayesian Optimization Flow:
===================================

            +-------------------+
            |   TPE Sampler     |
            | (n_startup=20)    |
            +-------------------+
                    |
                    v
    +-------------------------------+
    | Suggest Configuration:        |
    |  - Preprocessing type         |
    |  - Window size (if deriv)     |
    |  - Model hyperparameters      |
    |  - Variable selection method  |
    |  - Subset size                |
    +-------------------------------+
                    |
                    v
    +-------------------------------+
    | Apply Preprocessing           |
    |  raw -> SNV -> Derivative     |
    +-------------------------------+
                    |
                    v
    +-------------------------------+
    | Compute Variable Selection    |
    |  - importance (model-based)   |
    |  - CARS                       |
    |  - regional (dynamic)         |
    +-------------------------------+
                    |
                    v
    +-------------------------------+
    | Cross-Validate Model          |
    |  RMSE (regression)            |
    |  1-Accuracy (classification)  |
    +-------------------------------+
                    |
                    v
    +-------------------------------+
    | Return Score to TPE           |
    |  (subset score, not full)     |
    +-------------------------------+
                    |
                    v
           [Update TPE Model]
                    |
            +-------+-------+
            |               |
        [Good Set]     [Bad Set]
            |               |
            v               v
         l(x) KDE        g(x) KDE
```

### 3.4 Parameters and Settings

#### 3.4.1 Optuna Integration

Spectral Predict uses the **Optuna** library for TPE optimization:

```python
from optuna.samplers import TPESampler

sampler = TPESampler(
    seed=random_state,
    n_startup_trials=20,    # Random exploration first
    n_ei_candidates=32,     # Candidates per EI evaluation
    multivariate=True,      # Model parameter interactions
    consider_endpoints=True # Include boundary values
)
```

#### 3.4.2 Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| n_trials | 300 | Total number of configurations to evaluate |
| n_startup_trials | 20 | Random trials before TPE kicks in |
| cv_folds | 5 | Cross-validation folds |
| n_top_regions | 10 | Regional subsets to explore |

#### 3.4.3 Preprocessing Options

```python
PREPROCESSING_OPTIONS = [
    'raw', 'snv',
    'deriv1', 'deriv2', 'deriv3', 'deriv4',
    'snv_deriv1', 'snv_deriv2', 'snv_deriv3', 'snv_deriv4',
    'deriv1_snv', 'deriv2_snv', 'deriv3_snv', 'deriv4_snv'
]
```

#### 3.4.4 Variable Selection Methods

```python
VAR_METHODS = ['importance', 'cars', 'region']

SUBSET_SIZES = ['full', 10, 20, 50, 100, 250, 500, 1000]
```

### 3.5 Model-Specific Hyperparameter Ranges

Bayesian optimization uses **continuous** ranges rather than discrete grids:

#### PLS

| Parameter | Range | Scale |
|-----------|-------|-------|
| n_components | 2-20 | Integer |

#### Ridge / Lasso

| Parameter | Range | Scale |
|-----------|-------|-------|
| alpha | 1e-4 to 100 | Log |

#### ElasticNet

| Parameter | Range | Scale |
|-----------|-------|-------|
| alpha | 1e-4 to 100 | Log |
| l1_ratio | 0.1 to 0.9 | Linear |

#### LightGBM

| Parameter | Range | Scale |
|-----------|-------|-------|
| n_estimators | 50-500 | Integer |
| learning_rate | 0.01-0.3 | Log |
| num_leaves | 15-127 | Integer |
| max_depth | -1 to 15 | Integer |

#### XGBoost

| Parameter | Range | Scale |
|-----------|-------|-------|
| n_estimators | 50-500 | Integer |
| learning_rate | 0.01-0.3 | Log |
| max_depth | 3-12 | Integer |
| subsample | 0.6-1.0 | Linear |

### 3.6 Why Bayesian Beats Grid Search

| Aspect | Grid Search | Bayesian (TPE) |
|--------|-------------|----------------|
| **Continuous params** | Only discrete values | Any value in range |
| **Example** | alpha in [0.1, 0.2, 0.3] | alpha = 0.127 (optimal) |
| **Joint optimization** | Independent grids | Learns interactions |
| **Exploration** | Exhaustive (wastes time) | Focuses on promising regions |
| **Window sizes** | [7, 19] (2 values) | [5, 7, ..., 51] (24 values) |
| **Search space** | ~3,000 configs | 140,000+ intelligently explored |

### 3.7 When to Use Bayesian Optimization

**Best for**:
- Production modeling where you want the absolute best configuration
- When continuous hyperparameters matter (regularization, learning rates)
- Joint optimization of preprocessing + model + variables
- Limited computational budget (smarter than Grid Search)

**Not ideal for**:
- When you need complete reproducibility of all tested configurations
- When you want to understand the full parameter landscape
- When Pareto analysis of multiple objectives is needed

---

## 4. NSGA-II Multi-Objective Optimization

### 4.1 Full Name and Abbreviation

**NSGA-II**: Non-dominated Sorting Genetic Algorithm II

### 4.2 Algorithm Explanation

NSGA-II is an evolutionary algorithm for multi-objective optimization. Unlike single-objective methods that find one "best" solution, NSGA-II finds the **Pareto front** - a set of solutions representing the optimal tradeoffs between competing objectives.

#### 4.2.1 Multi-Objective Optimization Concepts

**Pareto Dominance**: Solution A dominates solution B if:
1. A is at least as good as B in all objectives
2. A is strictly better than B in at least one objective

**Pareto Front**: The set of all non-dominated solutions (no solution in the front dominates another).

```
Pareto Front Visualization (2 Objectives):
==========================================

Error (minimize)
     ^
     |
  1.0+  x  x
     |      x  x
     |         x  x
     |            x  x
  0.5+              x  x  <- Pareto Front
     |                 x
     |                   x
     |                     x
  0.0+-------------------------> Wavelengths (minimize)
     0    200   400   600   800

Legend:
x = Pareto-optimal solution
  = Dominated solution (not shown)
```

#### 4.2.2 The NSGA-II Algorithm

```
NSGA-II Algorithm:
==================

1. Initialize population P0 of size N
2. For generation g = 0 to G:

   a. CREATE OFFSPRING:
      - Select parents via binary tournament
      - Apply crossover (SBX)
      - Apply mutation (SmartMutation)
      - Create offspring population Q

   b. COMBINE: R = P U Q (size 2N)

   c. NON-DOMINATED SORTING:
      - Identify rank-0 (Pareto front)
      - Remove, repeat for rank-1, rank-2, ...

   d. SELECTION:
      - Fill next generation from lowest ranks
      - Within same rank, prefer diverse solutions
        (crowding distance)

3. Return final Pareto front
```

**Key Components**:

1. **Non-dominated Sorting**: O(MN^2) algorithm to rank solutions
2. **Crowding Distance**: Maintains diversity on the Pareto front
3. **Binary Tournament**: Selects parents based on rank and crowding
4. **Elitism**: Best solutions always survive to next generation

### 4.3 The Three Objectives in Spectral Predict

NSGA-II in Spectral Predict optimizes three competing objectives simultaneously:

#### Objective 1: Prediction Error (Minimize)

**Regression**: Root Mean Square Error (RMSE)
$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

**Classification**: 1 - Accuracy
$$Error = 1 - \frac{\text{Correct Predictions}}{\text{Total Predictions}}$$

#### Objective 2: Wavelength Count (Minimize - Parsimony)

Encourages sparse, interpretable models:

$$f_2 = \left(\frac{n_{selected}}{n_{total}}\right)^2$$

The quadratic penalty means:
- 400 wavelengths: penalty = (400/800)^2 = 0.25
- 800 wavelengths: penalty = (800/800)^2 = 1.00

This creates 4x more pressure to reduce from 800 to 400 than from 400 to 200.

#### Objective 3: Model Complexity (Minimize)

Combines model and preprocessing complexity:

$$f_3 = 0.7 \times C_{model} + 0.3 \times C_{preproc}$$

**Model complexity scores**:
| Model | Complexity Range |
|-------|------------------|
| PLS | 0.00 - 0.25 |
| Ridge | 0.00 - 0.20 |
| Lasso | 0.075 |
| ElasticNet | 0.10 |
| LightGBM | 0.15 - 0.40 |
| XGBoost | 0.15 - 0.40 |
| RandomForest | 0.20 - 0.40 |
| CatBoost | 0.175 - 0.40 |
| SVR | 0.25 - 0.40 |
| MLP | 0.25 - 0.45 |

**Preprocessing complexity scores**:
| Preprocessing | Complexity |
|---------------|------------|
| raw | 0.0 |
| snv | 0.2 |
| deriv1 | 0.4 |
| snv_deriv1 | 0.5 |
| deriv2 | 0.6 |
| snv_deriv2 | 0.7 |
| deriv3, deriv4 | 0.8 |

### 4.4 Pareto Front Concept with Diagram

```
3D Pareto Front (3 Objectives):
===============================

                Complexity
                    ^
                   /|
                  / |
                 /  |     * * *
                /   |   *       *
               /    | *           *
              /     |*             *
             /      +----------------> Wavelengths
            /      /
           /      /
          /      / * * *
         /      / *     *
        v      / *       *
      Error   /  *        *
             /

Each point (*) is a Pareto-optimal solution.
No solution dominates another - they represent
different tradeoffs between the objectives.
```

### 4.5 Non-dominated Sorting Algorithm

```
Non-dominated Sorting:
======================

Input: Population P with objective values
Output: Fronts F0, F1, F2, ...

1. For each solution p in P:
   - Sp = solutions dominated by p
   - np = count of solutions dominating p

2. F0 = {p : np = 0}  # First Pareto front

3. i = 0
4. While Fi is not empty:
   - Q = empty set
   - For each p in Fi:
     - For each q in Sp:
       - nq = nq - 1
       - If nq == 0: Q = Q U {q}
   - i = i + 1
   - Fi = Q

Result: F0, F1, F2, ... (non-dominated fronts)
```

### 4.6 Crowding Distance for Diversity

Crowding distance measures how "crowded" a solution is on the Pareto front. Solutions in less crowded regions are preferred to maintain diversity.

$$CD_i = \sum_{m=1}^{M} \frac{f_m^{i+1} - f_m^{i-1}}{f_m^{max} - f_m^{min}}$$

Where:
- $M$ = number of objectives
- $f_m^{i+1}$, $f_m^{i-1}$ = objective values of neighboring solutions
- Boundary solutions receive infinite distance (always preserved)

### 4.7 Selection Strategies

Spectral Predict provides three strategies for selecting from the Pareto front:

#### 4.7.1 Min Error (selection_bias = 0)

**Behavior**: Selects the solution with the absolute lowest prediction error, regardless of wavelength count or complexity.

**Use when**:
- Accuracy is the only priority
- You don't care about model interpretability
- Hardware can handle any model complexity

**Selection**: Searches **all evaluated solutions** (not just Pareto front) for minimum error.

#### 4.7.2 Balanced (selection_bias = 1)

**Behavior**: 50/50 compromise between minimum error and knee point.

**Use when**:
- You want some parsimony but prioritize accuracy
- Willing to trade a small amount of accuracy for fewer wavelengths

**Selection**:
$$score_i = 0.5 \times rank_{error} + 0.5 \times (1 - knee\_score)$$

#### 4.7.3 Knee Point (selection_bias = 2)

**Behavior**: Finds the point of maximum curvature on the Pareto front - where improving one objective starts requiring significant sacrifices in others.

**Use when**:
- You want the optimal tradeoff
- Interpretability and parsimony matter
- Research applications where understanding matters

**Selection Algorithm** (for 2D):
1. Draw line from first to last Pareto point
2. For each point, compute perpendicular distance to line
3. Point with maximum distance is the knee

```
Knee Point Detection:
=====================

Error
  ^
  |
  | *
  |  \
  |   *
  |    \
  |     * <- Knee point (max distance to line)
  |      \
  |       *
  |         \
  +----------*------> Wavelengths
```

### 4.8 SmartMutation: Importance-Biased Selection

NSGA-II in Spectral Predict uses **SmartMutation**, which leverages CARS-Tree importance scores to guide wavelength selection:

```
SmartMutation Logic:
====================

For each wavelength w with importance I(w):

If currently SELECTED (value = 1):
  P(drop) = base_prob x sparsity_bias x (1 - I(w))

  High importance -> Low drop probability
  sparsity_bias = 1.4 increases drop pressure

If currently NOT SELECTED (value = 0):
  P(add) = base_prob x (1/sparsity_bias) x I(w)

  High importance -> High add probability
  1/sparsity_bias reduces add likelihood
```

This creates evolutionary pressure toward:
- Keeping important wavelengths
- Dropping unimportant wavelengths
- Overall parsimony (sparsity_bias > 1)

### 4.9 Chromosome Representation

NSGA-II encodes each solution as an integer chromosome:

```
Chromosome Structure (13 + N_wavelengths genes):
================================================

[0]  : Preprocessing type (0-13)
[1]  : Window size index (0-13)
[2]  : Model type (0-10)
[3]  : Model parameter (0-14)
[4]  : Learning rate gene (0-14)
[5]  : L1 regularization gene (0-14)
[6]  : L2 regularization gene (0-14)
[7]  : ElasticNet l1_ratio gene (0-14)
[8]  : Subsample gene (0-14)
[9]  : Colsample gene (0-14)
[10] : Min samples gene (0-14)
[11] : Gamma gene (0-14)
[12] : Max features gene (0-14)
[13+]: Wavelength selection (0 or 1)

Example (1000 wavelengths):
[7, 6, 2, 8, 10, 12, 14, 7, 10, 10, 2, 0, 7, 0, 1, 1, 0, ...]
 |  |  |  |                                     |  |  |  |
 |  |  |  |                                     wavelengths
 |  |  |  model_param=8 (e.g., 9 PLS components)
 |  |  model=Ridge
 |  window=17
 preproc=snv_deriv2
```

### 4.10 Default Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| population_size | 60 | Solutions per generation |
| n_generations | 120 | Number of evolutionary cycles |
| cv_folds | 5 | Cross-validation folds |
| min_wavelengths | 10 | Constraint: minimum selected |
| selection_bias | 0.0 | 0=min error, 1=balanced, 2=knee |
| use_guidance | True | Enable CARS-Tree importance guidance |

### 4.11 When to Use NSGA-II

**Best for**:
- Research requiring Pareto analysis
- Understanding tradeoffs between accuracy and parsimony
- Building interpretable models with minimal wavelengths
- Exploring the full spectrum of optimal solutions

**Not ideal for**:
- Quick analysis (long runtime)
- When you only care about accuracy (use Bayesian)
- Simple datasets where Grid Search suffices

---

## 5. Choosing an Optimization Method

### 5.1 Decision Flowchart

```
                    START
                      |
                      v
            +-------------------+
            | Time constraint?  |
            +-------------------+
                  |       |
                < 30 min  > 30 min
                  |       |
                  v       v
          +----------+  +------------------+
          | Quick    |  | Need Pareto      |
          | Grid     |  | analysis?        |
          | Search   |  +------------------+
          +----------+        |       |
                            Yes      No
                              |       |
                              v       v
                      +----------+  +-----------+
                      | NSGA-II  |  | Bayesian  |
                      | (knee/   |  | (TPE)     |
                      | balanced)|  +-----------+
                      +----------+
```

### 5.2 Method Comparison Table

| Criterion | Grid Search | Bayesian (TPE) | NSGA-II |
|-----------|-------------|----------------|---------|
| **Speed** | Medium | Medium | Slow |
| **Coverage** | Exhaustive (grid) | Intelligent sampling | Multi-objective |
| **Reproducibility** | Excellent | Good | Good |
| **Continuous params** | No | Yes | Yes |
| **Joint optimization** | Limited | Excellent | Excellent |
| **Interpretability** | High | Medium | High (Pareto) |
| **Parsimony** | Manual | Automatic | Built-in objective |
| **Best single model** | Good | Excellent | Good |
| **Understanding tradeoffs** | Poor | Poor | Excellent |

### 5.3 Use Case Recommendations

#### Quick Exploration
**Method**: Grid Search (Quick tier)
**Time**: 5-15 minutes
**When**: Initial data exploration, daily QC

#### Production Modeling
**Method**: Bayesian Optimization
**Time**: 20-45 minutes
**When**: Building deployment models, best accuracy needed

#### Research / Publications
**Method**: NSGA-II with Balanced selection
**Time**: 1-3 hours
**When**: Understanding model tradeoffs, interpretability matters

#### Complete Analysis
**Method**: All three in sequence
**Time**: 3-5 hours
**When**: Comprehensive method comparison, publications

### 5.4 Dataset Size Recommendations

| Dataset Size | Recommended Method | Notes |
|--------------|-------------------|-------|
| < 50 samples | Grid Search (Quick) | Limited data, simple models preferred |
| 50-200 samples | Bayesian or Grid (Standard) | Good balance of methods |
| 200-1000 samples | Bayesian | Enough data for complex models |
| > 1000 samples | NSGA-II or Bayesian | Can afford longer optimization |

---

## 6. Computational Considerations

### 6.1 Time Complexity

| Method | Complexity | Typical Time (100 samples, 1000 wavelengths) |
|--------|------------|---------------------------------------------|
| Grid Search (Quick) | O(n_configs) | 5-15 min |
| Grid Search (Standard) | O(n_configs) | 15-45 min |
| Bayesian (300 trials) | O(n_trials x n_cv) | 20-60 min |
| NSGA-II (60 pop, 120 gen) | O(pop x gen x n_cv) | 1-3 hours |

### 6.2 Memory Requirements

| Method | Memory Usage | Notes |
|--------|--------------|-------|
| Grid Search | Low-Medium | Results stored in DataFrame |
| Bayesian | Medium | Optuna study stores trial history |
| NSGA-II | Medium-High | Population + Pareto front storage |

### 6.3 Parallelization

| Method | Parallelization | Efficiency |
|--------|-----------------|------------|
| Grid Search | Excellent (joblib) | Near-linear with cores |
| Bayesian | Limited (sequential) | TPE is inherently sequential |
| NSGA-II | Medium (per-generation) | Population evaluated in parallel |

### 6.4 Early Stopping and Pruning

**Bayesian Optimization**:
- Optuna supports pruning via median stopping rule
- Trials performing worse than median are stopped early

**NSGA-II**:
- No early stopping (evolutionary process requires full generations)
- Controller allows manual cancellation with partial results

---

## 7. Best Practices

### 7.1 Before Optimization

1. **Clean your data**: Remove outliers and verify data quality
2. **Split validation set**: Keep 10-20% for final evaluation
3. **Understand your constraints**: Time, interpretability, deployment requirements

### 7.2 During Optimization

1. **Start simple**: Begin with Grid Search (Quick) to verify pipeline
2. **Scale up gradually**: Move to Standard, then Bayesian/NSGA-II
3. **Monitor progress**: Check for convergence and overfitting

### 7.3 After Optimization

1. **Validate results**: Test top models on held-out validation set
2. **Check for overfitting**: Compare CV scores to validation scores
3. **Consider ensemble**: Combine top-performing models if appropriate

### 7.4 Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Overfitting to CV | Use separate validation set |
| Too many hyperparameters | Start with defaults, tune important ones |
| Ignoring preprocessing | Test multiple preprocessing methods |
| Computational timeout | Start with Quick tier, scale up |
| Selecting only by accuracy | Consider parsimony and interpretability |

### 7.5 Recommended Workflow

```
Recommended Optimization Workflow:
==================================

1. EXPLORE (10 min)
   - Grid Search (Quick tier)
   - Identify promising models and preprocessing

2. OPTIMIZE (30-60 min)
   - Bayesian Optimization (300 trials)
   - Joint preprocessing + hyperparameter tuning

3. ANALYZE (optional, 1-3 hours)
   - NSGA-II with Balanced selection
   - Understand accuracy vs parsimony tradeoffs

4. VALIDATE (10 min)
   - Test top 3-5 models on validation set
   - Select final model based on validation performance

5. DEPLOY
   - Export selected model
   - Document preprocessing pipeline
```

---

## Summary

Spectral Predict V1 provides three complementary hyperparameter optimization methods:

1. **Grid Search**: Exhaustive, reproducible, tiered coverage
2. **Bayesian (TPE)**: Intelligent, continuous, joint optimization
3. **NSGA-II**: Multi-objective, Pareto analysis, parsimony-aware

The choice depends on your constraints:
- **Time-limited**: Grid Search (Quick/Standard)
- **Best accuracy**: Bayesian Optimization
- **Interpretability**: NSGA-II with Balanced selection

For most users, the recommended approach is:
1. Quick Grid Search for exploration
2. Bayesian Optimization for final model selection
3. NSGA-II when parsimony and tradeoff analysis matter

---

*Part VI of the Spectral Predict V1 User Guide*
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
    Model 1 â”€â”€â”€â”€â”€â”
    Model 2 â”€â”€â”€â”€â”€â”¼â”€â”€â–º OOF Predictions â”€â”€â–º Meta Features
    Model 3 â”€â”€â”€â”€â”€â”¤
    ...          â”‚
    Model K â”€â”€â”€â”€â”€â”˜

Level 1 (Meta-Learner):
    Meta Features â”€â”€â–º Ridge Regression â”€â”€â–º Final Prediction
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
   Performance: RÂ²=0.9234, RMSE=1.45

2. [Ensemble] top3_ensemble.dasp
   Type: RegionSpecialist
   Base Models: PLS_SNV, Ridge_SG1, XGBoost_raw
   Performance: RÂ²=0.9312, RMSE=1.38
```

### 3.2 Model File Format (.dasp)

The `.dasp` file is a ZIP archive containing:

```
model.dasp/
â”œâ”€â”€ model.pkl           # Serialized sklearn model
â”œâ”€â”€ preprocessor.pkl    # Fitted preprocessing pipeline
â”œâ”€â”€ metadata.json       # Model metadata
â”œâ”€â”€ wavelengths.npy     # Expected wavelengths
â””â”€â”€ [ensemble_config.json]  # For ensembles only
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

---

# Appendices

## Appendix A: Quick Reference Cards

### A.1 Preprocessing Quick Reference

| Method | Purpose | Key Parameters | Best For |
|--------|---------|----------------|----------|
| SNV | Scatter correction | None | NIR reflectance |
| 1st Derivative | Baseline removal | window, polyorder | Baseline offset |
| 2nd Derivative | Peak resolution | window, polyorder | Overlapping peaks |
| ALS | Baseline correction | lambda, p | Complex baselines |
| MSC | Scatter correction | reference | Known standards |

### A.2 Model Selection Quick Reference

| Scenario | Primary Choice | Alternative |
|----------|----------------|-------------|
| Small samples (<50) | PLS | Ridge |
| Interpretability needed | PLS, Ridge | Lasso |
| Maximum accuracy | LightGBM, XGBoost | CatBoost |
| Non-linear patterns | XGBoost | RandomForest |
| Classification | PLS-DA | RandomForest |

### A.3 Hyperparameter Defaults

| Model | Key Parameter | Default | Range |
|-------|---------------|---------|-------|
| PLS | n_components | Auto (1-8) | 1-20 |
| Ridge | alpha | 1.0 | 0.001-1000 |
| RandomForest | n_estimators | 200 | 100-1000 |
| XGBoost | learning_rate | 0.1 | 0.01-0.3 |
| LightGBM | num_leaves | 15 | 15-127 |

---

## Appendix B: Keyboard Shortcuts

### B.1 General Navigation

| Shortcut | Action |
|----------|--------|
| Ctrl+Tab | Next tab |
| Ctrl+Shift+Tab | Previous tab |
| F5 | Refresh current view |
| Ctrl+S | Save current model/results |
| Ctrl+O | Open file dialog |

### B.2 Results Tab

| Shortcut | Action |
|----------|--------|
| Up/Down Arrow | Navigate results |
| Enter | Load selected model |
| Ctrl+E | Export results |
| Ctrl+A | Select all |

### B.3 Plot Interactions

| Action | Mouse/Key |
|--------|-----------|
| Zoom in | Scroll up / Click+drag |
| Zoom out | Scroll down |
| Pan | Right-click + drag |
| Reset view | Double-click |
| Save plot | Right-click -> Save |

---

## Appendix C: Troubleshooting Guide

### C.1 Common Issues and Solutions

| Issue | Possible Cause | Solution |
|-------|----------------|----------|
| Import fails | File format not recognized | Check file extension, try CSV export |
| No ID matches | ID format mismatch | Check for spaces, extensions in IDs |
| Analysis crashes | Memory limit | Reduce dataset size or wavelengths |
| Poor R2 values | Weak spectral signal | Try different preprocessing |
| Overfitting | Too many components | Reduce complexity, add regularization |

### C.2 Memory Optimization

For large datasets:
1. Enable wavelength restriction
2. Use Quick or Standard tier
3. Disable ensemble methods initially
4. Close other applications

### C.3 Performance Tips

1. Start with Quick tier for initial evaluation
2. Use validation set for true generalization estimate
3. Enable preprocessing discovery for comprehensive search
4. Consider Bayesian optimization for efficient search

---

## Index

### A
- Accuracy, see Classification Metrics
- ALS (Asymmetric Least Squares), see Baseline Correction
- Analysis Configuration Tab, Part II
- airPLS, see Baseline Correction

### B
- Baseline Correction, Part IV
- Bayesian Optimization, Part VI

### C
- CARS (Competitive Adaptive Reweighted Sampling), Part V
- CatBoost, Part III
- Classification Metrics, Part VIII
- Composite Score, Part II
- Cross-Validation, Part II

### D
- Derivatives (Savitzky-Golay), Part IV
- DOSC (Direct Orthogonal Signal Correction), Part IV

### E
- ElasticNet, Part III
- Ensemble Methods, Part VII
- EPO (External Parameter Orthogonalization), Part IV

### G
- GA-PLS (Genetic Algorithm PLS), Part V
- GLSW (Generalized Least Squares Weighting), Part IV
- Grid Search, Part VI

### H
- Hotelling T-squared, Part I
- Hyperparameter Optimization, Part VI

### I
- Import & Preview Tab, Part I
- iPLS (Interval PLS), Part V

### L
- Lasso, Part III
- LightGBM, Part III

### M
- Mahalanobis Distance, Part I
- MLP (Multi-Layer Perceptron), Part III
- MSC (Multiplicative Scatter Correction), Part IV

### N
- NeuralBoosted, Part III
- NSGA-II, Part VI

### O
- OSC (Orthogonal Signal Correction), Part IV
- Outlier Detection, Part I

### P
- PLS (Partial Least Squares), Part III
- PLS-DA, Part III
- Preprocessing, Part IV

### Q
- Q-Residuals, Part I
- Quartile Highlighting, Part II

### R
- Random Forest, Part III
- R-squared (R2), Part VIII
- Ridge Regression, Part III
- RMSE, Part VIII
- RPD, Part VIII

### S
- SNV (Standard Normal Variate), Part IV
- SPA (Successive Projections Algorithm), Part V
- Stacking Ensemble, Part VII
- SVR/SVC, Part III

### U
- UVE (Uninformative Variable Elimination), Part V

### V
- Variable Selection, Part V
- VIP (Variable Importance in Projection), Part III

### X
- XGBoost, Part III

---

*Spectral Predict V1 User Guide*
*Version 1.0 | January 2026*
