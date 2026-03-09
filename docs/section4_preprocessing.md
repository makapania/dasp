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
┌─────────────────────────────────────────────────────────────┐
│                    RAW SPECTRAL DATA                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           STEP 1: INTERFERENCE REMOVAL (Optional)           │
│                                                             │
│   ┌─────────────────┐   ┌─────────────────┐                │
│   │ Wavelength      │   │ MSC             │                │
│   │ Exclusion       │   │                 │                │
│   └─────────────────┘   └─────────────────┘                │
│                                                             │
│   ┌─────────────────┐   ┌─────────────────┐                │
│   │ OSC / DOSC      │   │ EPO             │                │
│   └─────────────────┘   └─────────────────┘                │
│                                                             │
│   ┌─────────────────┐                                      │
│   │ GLSW            │                                      │
│   └─────────────────┘                                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           STEP 2: BASELINE CORRECTION (Optional)            │
│                                                             │
│   ┌─────────────────┐   ┌─────────────────┐                │
│   │ Polynomial      │   │ ALS             │                │
│   └─────────────────┘   └─────────────────┘                │
│                                                             │
│   ┌─────────────────┐                                      │
│   │ airPLS          │                                      │
│   └─────────────────┘                                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              STEP 3: SMOOTHING (Optional)                   │
│                                                             │
│   ┌─────────────────┐                                      │
│   │ Savitzky-Golay  │                                      │
│   │ Smoothing       │                                      │
│   └─────────────────┘                                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│         STEP 4: SPECTRAL TRANSFORMATION (Required)          │
│                                                             │
│   ┌─────────────────┐   ┌─────────────────┐                │
│   │ raw             │   │ snv             │                │
│   │ (no transform)  │   │                 │                │
│   └─────────────────┘   └─────────────────┘                │
│                                                             │
│   ┌─────────────────┐   ┌─────────────────┐                │
│   │ deriv           │   │ snv_deriv       │                │
│   │ (derivative)    │   │ (SNV+derivative)│                │
│   └─────────────────┘   └─────────────────┘                │
│                                                             │
│   ┌─────────────────┐                                      │
│   │ deriv_snv       │                                      │
│   │ (derivative+SNV)│                                      │
│   └─────────────────┘                                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   PREPROCESSED DATA                         │
│                         │                                   │
│                         ▼                                   │
│                ┌─────────────────┐                         │
│                │ Machine Learning│                         │
│                │ Model           │                         │
│                └─────────────────┘                         │
└─────────────────────────────────────────────────────────────┘
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
