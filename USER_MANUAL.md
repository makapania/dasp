# DASP User Manual

## 1. Introduction

Welcome to DASP (Directional-agnostic spectral-preprocessing), a powerful graphical application for automated spectral modeling. DASP is designed to streamline the process of building robust machine learning models from spectral data.

The typical workflow for building spectral models can be tedious, involving manual testing of multiple preprocessing methods, different machine learning algorithms, and extensive hyperparameter tuning. DASP automates this entire process.

### Key Features:

*   **Automated Model Building:** Automatically search through a grid of preprocessing techniques and machine learning models to find the best combination for your data.
*   **Advanced Preprocessing:** Includes a wide range of preprocessing methods like Standard Normal Variate (SNV), Savitzky-Golay (SG) smoothing and derivatives, and more.
*   **Multiple Model Support:** Test and compare various regression and classification models, including PLS, Random Forest, Ridge, Lasso, MLP, and Neural Boosted models.
*   **Variable Selection:** Employ powerful techniques like Successive Projections Algorithm (SPA), Uninformative Variable Elimination (UVE), and Interval PLS (iPLS) to identify the most important wavelengths.
*   **Calibration Transfer:** Standardize spectra from different instruments or conditions.
*   **Interactive UI:** An intuitive graphical user interface allows for easy data loading, analysis configuration, and results visualization.
*   **Rich Visualization:** Interactive plots for predictions, residuals, and outlier detection.
*   **Flexible Data Handling:** Supports a wide variety of common spectral data formats.

This manual will guide you through the installation, features, and workflows of the DASP application.

## 2. Installation

DASP is a Python application and requires a Python environment. The following steps will guide you through setting up the necessary environment and installing the application.

### 2.1. Prerequisites

*   Python 3.10 or higher.
*   `git` for cloning the repository.

### 2.2. Setup Steps

1.  **Clone the Repository:**
    Open a terminal or command prompt and clone the project repository:
    ```bash
    git clone <repository_url>
    cd dasp
    ```
    *(Replace `<repository_url>` with the actual URL of the git repository)*

2.  **Create a Virtual Environment:**
    It is highly recommended to use a Python virtual environment to manage dependencies and avoid conflicts with other projects.
    ```bash
    python -m venv .venv
    ```

3.  **Activate the Virtual Environment:**
    *   On **Windows**:
        ```bash
        .venv\Scripts\activate
        ```
    *   On **macOS and Linux**:
        ```bash
        source .venv/bin/activate
        ```
    Your terminal prompt should now be prefixed with `(.venv)`, indicating that the virtual environment is active.

4.  **Install Core Dependencies:**
    Install the main application and its core dependencies using `pip`:
    ```bash
    pip install -e .[dev]
    ```

### 2.3. Installing Optional File Format Support

DASP can be extended to support various proprietary and binary file formats by installing optional dependencies.

*   **For Binary ASD files (`.asd`):**
    This is common for data from ASD (Malvern Panalytical) instruments.
    ```bash
    pip install specdal
    ```
    Alternatively, you can install it using the project's extras syntax:
    ```bash
    pip install -e ".[asd]"
    ```

*   **For Other Vendor Formats:**
    You can install support for other formats like Bruker OPUS, PerkinElmer, etc.
    ```bash
    # For Bruker OPUS files (.0, .1, etc.)
    pip install -e ".[opus]"

    # For PerkinElmer files (.sp)
    pip install -e ".[perkinelmer]"

    # For Agilent files (.seq)
    pip install -e ".[agilent]"
    ```

    # To Install All Supported Formats:
    To install all available file format handlers at once, use:
    ```bash
    pip install -e ".[all-formats]"
    ```

## 3. Running the Application

Once the installation is complete, you can run the DASP graphical user interface.

1.  **Activate the Virtual Environment:**
    If it's not already active, navigate to the project directory and activate the virtual environment:
    *   On **Windows**:
        ```bash
        .venv\Scripts\activate
        ```
    *   On **macOS and Linux**:
        ```bash
        source .venv/bin/activate
        ```

2.  **Launch the GUI:**
    Run the main application script from the root of the project directory:
    ```bash
    python spectral_predict_gui_optimized.py
    ```
    The main application window should appear on your screen.

## 4. GUI Overview

The DASP application is organized into a series of tabs, each dedicated to a specific stage of the model-building workflow. You can switch between themes using the buttons at the top right of the window.

Here is a brief overview of each tab:

*   **📁 Import & Preview:** This is the starting point. Here you will load your spectral data and associated reference values. It includes sub-tabs for data configuration and for visualizing the raw and preprocessed spectra.

*   **📋 Data Viewer:** Provides a spreadsheet-like view of your loaded data, allowing you to inspect the raw values of your spectra and reference data.

*   **🔍 Data Quality Check:** This tab is for identifying and excluding outliers from your dataset. You can use methods based on PCA, leverage, and residuals to ensure your data quality.

*   **⚙️ Analysis Configuration:** The core of the setup process. In this tab, you configure the entire analysis pipeline, including:
    *   Selecting machine learning models.
    *   Choosing preprocessing methods.
    *   Defining variable and region subset analyses.
    *   Setting advanced hyperparameters for each model.

*   **📈 Analysis Progress:** Monitor the status of your analysis in real-time. This tab shows progress bars and logs as the application works through the configured model and preprocessing combinations.

*   **🏆 Results:** Once the analysis is complete, this tab displays a ranked table of all the tested models. You can sort the results by various metrics (like R², RMSE) to find the best-performing models.

*   **🔬 Model Development:** Interactively refine a model selected from the Results tab. You can tweak its hyperparameters, view detailed performance plots (like prediction vs. actual), and save the refined model for future use.

*   **🔮 Model Prediction:** Load a previously saved `.dasp` model and use it to make predictions on new, unseen spectral data.

*   **🔄 Calibration Transfer:** A powerful suite of tools for aligning spectra from different instruments. This includes features for building, applying, and managing transfer models (e.g., Direct Standardization, Piecewise Direct Standardization).

## 5. Detailed GUI Workflow

This section provides a detailed walkthrough of each tab in the DASP application.

### 5.1. Tab 1: Import & Preview

This is the first and most critical step in the workflow. This tab is divided into two sub-tabs: **Data** and **Plots**.

#### 5.1.1. Data Sub-Tab: Loading and Configuration

This is where you specify your input files and configure how the data should be read.

**1. Input Data Files:**

*   **Spectral File Directory:** Click **"Browse..."** to select the location of your spectral data. The application supports several modes of data loading:
    *   **Directory of individual files:** Select a folder containing multiple spectral files (e.g., `.asd`, `.spc`).
    *   **Single CSV/Excel file (Spectra Only):** Select a single file where each row is a sample and columns are wavelengths.
    *   **Single Combined CSV/Excel file:** Select a single file that contains both spectral data and target/reference values in the same sheet. The application will automatically detect this format.
    *   A status label will indicate the type of data detected (e.g., "Detected ASD directory", "Detected Combined CSV").

*   **Reference CSV/Excel:** Click **"Browse..."** to select the file containing your reference values (the "y" variables you want to predict). This file should contain a column that identifies the sample and columns for the target variables.
    *   *Note: This field is optional if you are using a "Combined" file format that already includes the target variables.*

**2. Load Data:**

*   Once you have selected your files, click the **"📊 Load Data & Generate Plots"** button.
*   The application will read the data, match spectra to their reference values, and generate the initial plots in the "Plots" sub-tab.
*   A status label below the button will show progress and indicate when loading is complete.

**3. Advanced Configuration (Optional):**

This section allows you to fine-tune how your data is interpreted.

*   **Column Mapping:**
    *   After loading data, the dropdown menus for **Spectral File**, **Specimen ID**, and **Target Variable** will be populated with the column headers found in your reference file.
    *   **Spectral File:** The column in your reference file that contains the filenames or identifiers for each spectrum.
    *   **Specimen ID:** The column that uniquely identifies each sample.
    *   **Target Variable:** The specific variable from your reference file that you want to model and predict.
    *   You can click **"🔍 Auto-Detect"** to have the application attempt to guess the correct columns based on common naming conventions.

*   **Analysis Type:**
    *   Choose the type of modeling task.
    *   **Auto-detect:** The application will automatically determine if it's a `regression` (continuous numeric target) or `classification` (categorical/text target) task based on the selected target variable. A label will confirm the detected type.
    *   **Regression/Classification:** You can manually override the task type if needed.

*   **Wavelength Range:**
    *   After data is loaded, the `min` and `max` wavelength values will be auto-filled.
    *   You can enter new values to narrow the analysis to a specific spectral range.
    *   Click **"Update Plots"** to visualize the effect of the new range.

*   **Data Type:**
    *   The application attempts to detect if the data is `Reflectance` or `Absorbance`. The detected type and confidence will be displayed.
    *   If your data is in Reflectance, you can convert it to Absorbance (`log10(1/R)`) by clicking the **"Convert to Absorbance"** button. This will update the plots.

*   **Spectrum Selection:**
    *   This section tracks spectra that you have manually excluded.
    *   To exclude a spectrum, go to the **Plots** sub-tab and click on its line in any of the plots. The line will be grayed out, and the counter here will update.
    *   Click **"Reset Exclusions"** to include all spectra in the analysis again.

#### 5.1.2. Plots Sub-Tab: Visualization

This sub-tab will populate with plots after you click the "Load Data & Generate Plots" button. It provides a visual check of your data. You can interact with the plots:

*   **Click on a spectrum line:** Toggles the exclusion of that spectrum from the analysis. The line will turn gray, and it will be removed from model training. Click it again to re-include it.
*   **Use the toolbar:** Pan, zoom, and save the plots using the standard Matplotlib toolbar at the bottom of the plots.

### 5.2. Tab 2: Data Viewer

Once your data is loaded, the **Data Viewer** tab provides a familiar, spreadsheet-like interface to inspect your entire dataset.

*   **Data Grid:** The main part of the tab is a table showing all your data, including the sample identifiers, the target variable, and all the spectral wavelength values for each sample.
*   **Interactions:** You can interact with the grid similarly to a standard spreadsheet:
    *   Scroll vertically and horizontally to see all data.
    *   Resize columns by dragging the headers.
    *   Select rows and copy data.
*   **Show Excluded Samples:** By default, samples you excluded in the "Import & Preview" tab are hidden. Check the **"Show excluded samples"** box to display them in the grid (they will still be excluded from analysis).
*   **Export Data:** Click the **"📥 Export All Data to CSV"** button to save the currently displayed data (including the target variable and all spectral data) to a single CSV file. This is useful for external analysis or record-keeping.

### 5.3. Tab 3: Data Quality Check

Ensuring high data quality is essential for building reliable models. This tab provides tools to identify and remove potential outliers from your dataset before running the main analysis.

**1. Configure Outlier Detection:**

*   **Number of PCA Components:** Specify the number of Principal Components to use for the analysis. PCA is used to reduce the dimensionality of the spectral data to identify samples that are spectrally different from the majority. A value between 3 and 10 is typically a good starting point.
*   **Y Variable Min/Max Bounds:** You can filter outliers based on the target variable itself. Enter a minimum or maximum value to flag any samples that fall outside this expected range.

**2. Run Analysis:**

*   Click the **"Run Outlier Detection"** button to start the analysis.
*   The application will perform PCA on the spectra and calculate various outlier metrics.

**3. Interpret the Results:**

Once the analysis is complete, a series of plots and a report will be generated.

*   **Outlier Plots:**
    *   **PCA Scores Plot:** Visualizes the samples in the PCA space. Samples far from the main cluster are potential outliers.
    *   **Hotelling's T² vs. Q Residuals:** A standard plot for outlier detection. Points in the top right corner are strong outlier candidates.
    *   **Leverage vs. Studentized Residuals:** Helps identify influential outliers.
*   **Outlier Report:** A table will appear listing all samples and flagging them as potential outliers based on different metrics.
*   **Interacting with Plots:** You can click on points in the plots to select them.
*   **Exclude Selected Samples:** After selecting potential outliers, click the **"Exclude Selected Samples"** button to remove them from the dataset for the main analysis. The exclusion status will be updated across the application.

### 5.4. Tab 4: Analysis Configuration

This tab is the control center for defining your entire automated modeling workflow. It is organized into five sub-tabs to guide you through the process. A **"▶ Run Analysis"** button is conveniently located at the top of each sub-tab to start the process at any time.

#### 5.4.1. Basic Settings Sub-Tab

This sub-tab contains the fundamental parameters for the analysis.

*   **Analysis Options:**
    *   **CV Folds:** The number of folds to use in cross-validation. A value of 5 or 10 is standard.
    *   **Variable Count Penalty (0-10):** A penalty applied to the model ranking score based on the number of variables (wavelengths) it uses. Higher values will more strongly favor models that use fewer variables.
    *   **Model Complexity Penalty (0-10):** A penalty applied based on the inherent complexity of the model (e.g., the number of latent variables in PLS). Higher values favor simpler models.
    *   **Output Directory:** The folder where all results, reports, and saved models will be stored.
    *   **Show live progress monitor:** Toggles whether the "Analysis Progress" tab will be automatically selected and updated when an analysis is running.

*   **Preprocessing Methods:**
    *   Select one or more preprocessing techniques to apply to the spectra before modeling. The analysis will run a separate set of models for each selected method.
    *   **Methods:** Raw (none), SNV, SG1 (1st Derivative), SG2 (2nd Derivative), deriv_snv (Derivative then SNV).
    *   **Derivative Window Sizes:** For derivative methods (SG1, SG2), you can select one or more Savitzky-Golay window sizes to test.

#### 5.4.2. Variable Selection Sub-Tab

This sub-tab lets you configure analyses that use only a subset of the available wavelengths, which can lead to simpler, faster, and sometimes more robust models.

*   **Subset Analysis:**
    *   **Enable Top-N Variable Analysis:** If checked, the analysis will include runs that use only the top N most important variables (e.g., top 10, top 50). You can select which values of N to test.
    *   **Enable Spectral Region Analysis:** If checked, the analysis will automatically identify and test models on specific, highly informative spectral regions.

*   **Advanced Variable Selection Methods:**
    *   Here you can select sophisticated algorithms for wavelength selection. The analysis will run a separate modeling pipeline for each selected method.
    *   **Feature Importance (default):** Uses model-specific scores (e.g., VIP for PLS).
    *   **SPA (Successive Projections Algorithm):** Reduces collinearity between wavelengths.
    *   **UVE (Uninformative Variable Elimination):** Filters out noisy, non-informative variables.
    *   **UVE-SPA Hybrid:** Combines the strengths of both UVE and SPA.
    *   **iPLS (Interval PLS):** Identifies the most informative spectral *regions*.
    *   You can also configure parameters for these methods, such as the UVE cutoff threshold.

#### 5.4.3. Model Config Sub-Tab

This is where you choose which machine learning algorithms to test and configure their hyperparameters.

*   **Model Tier (Quick Presets):**
    *   To simplify configuration, you can select a tier, which automatically checks or unchecks the models and hyperparameters below.
    *   **⚡ Quick:** A very fast run with a limited set of simple, fast models.
    *   **⭐ Standard:** A balanced selection of common and effective models.
    *   **🔬 Comprehensive:** A much longer run that includes more models and more hyperparameter combinations.
    *   **🧪 Experimental:** Includes models that may be very slow or are still under development.
    *   **⚙️ Custom:** The tier automatically switches to "Custom" if you manually change any setting, allowing for full control.

*   **Select Models:**
    *   A list of all available models is provided, grouped into Core, Advanced, and Modern Gradient Boosting. Check the box next to each model you want to include in the analysis.
    *   **Models include:** PLS, PLS-DA, Ridge, Lasso, ElasticNet, Random Forest, MLP, SVR, XGBoost, LightGBM, CatBoost, and NeuralBoosted.

*   **Advanced Model Options:**
    *   Below the model selection, there are collapsible sections for each complex model type (e.g., "Random Forest Hyperparameters", "XGBoost Hyperparameters").
    *   Click on a section to expand it and see the detailed grid of hyperparameters that will be tested for that model. You can enable or disable specific values to customize the search space. For example, for Random Forest, you can select which values of "Number of Trees" and "Maximum Tree Depth" to test.

#### 5.4.4. Ensemble Methods Sub-Tab

Ensemble methods combine the predictions from several of the best-performing individual models, which can often lead to more accurate and robust results.

*   **Enable Ensemble Methods:** Check this box to run ensemble methods *after* the main analysis is complete. The ensembles will be built using the top models found in the initial run.
*   **Select Ensemble Methods:**
    *   **Simple Average:** A baseline that simply averages the predictions of the top models.
    *   **Region-Aware Weighted:** Dynamically weights model predictions based on their performance in different regions of the prediction space (e.g., low vs. high values).
    *   **Mixture of Experts:** Selects the single best model for each prediction region.
    *   **Stacking Ensemble:** Trains a "meta-learner" that learns how to best combine the predictions from the base models.

#### 5.4.5. Validation Sub-Tab

This sub-tab allows you to create a **holdout validation set**. These samples are completely excluded from all model training and cross-validation and are used only at the very end to get a truly independent measure of the final model's performance.

*   **Enable Validation Set:** Check this to activate the feature.
*   **Validation Set Size:** Use the slider to determine the percentage of your data to hold out (e.g., 20%).
*   **Selection Algorithm:** Choose the algorithm used to select the holdout samples:
    *   **Kennard-Stone:** Selects a spectrally diverse set of samples.
    *   **SPXY ⭐:** Balances both spectral (X) and target (Y) diversity. This is the recommended default.
    *   **Random:** Simple random selection.
    *   **Stratified:** Ensures the distribution of the target variable is similar in both the training and validation sets.
*   **Create Validation Set:** After configuring, click this button to perform the split. A status message will confirm how many samples were allocated to the validation set.

### 5.5. Tab 5: Analysis Progress

Once you click the **"▶ Run Analysis"** button, this tab becomes active (if enabled in settings). It allows you to monitor the analysis in real-time.

*   **Progress Log:** The main text area displays a detailed log of the operations being performed. It shows which model, preprocessing method, and variable subset is currently being tested, along with intermediate results.
*   **Best Model So Far:** A summary at the top right keeps track of the best-performing model found so far during the run, giving you an early indication of the leading candidates.
*   **Progress Info & Time Estimate:** Provides an overview of the overall progress and an estimated time remaining until the analysis is complete.
*   **Completion Chime:** A pleasant chime will sound when the analysis is finished.

### 5.6. Tab 6: Results

After the analysis is complete, this tab will be populated with a comprehensive table of every model combination that was tested.

*   **Results Table:** The main table shows all results, ranked by a composite score that balances performance (e.g., R² or Accuracy) with model simplicity.
    *   **Sorting:** You can click on any column header (e.g., `RMSE`, `R2`, `Rank`) to sort the entire table by that metric. Click again to reverse the sort order.
    *   **Columns:** The table includes detailed information for each run, such as the model name, hyperparameters, preprocessing method, variable selection method, number of variables, and key performance metrics.
*   **Export Results:** Click the **"📥 Export Results to CSV"** button to save the entire results table to a CSV file for your records or for further analysis in other software.
*   **Ensemble Model Results:** If you enabled ensembles, a separate table at the bottom will show the performance of the combined models.
*   **Next Step:** The primary action on this tab is to find a promising model and **double-click on its row**. This will load the model's exact configuration into the "Model Development" tab for interactive refinement.

### 5.7. Tab 7: Model Development

This tab is an interactive workshop for fine-tuning a single, promising model selected from the Results tab. It is organized into its own set of sub-tabs.

**Workflow:**
1.  Go to the **Results** tab and double-click on a model row.
2.  The configuration of that model is automatically loaded into this **Model Development** tab.
3.  Adjust settings in the sub-tabs as described below.
4.  Click **"▶ Run Model"** to train and evaluate this single configuration.
5.  Analyze the output in the **Results & Diagnostics** sub-tab.
6.  If satisfied, click **"💾 Save Model"** to create a portable `.dasp` model file.

#### 5.7.1. Selection Sub-Tab

This sub-tab shows the configuration of the currently loaded model and allows you to precisely control the wavelengths used.

*   **Current Configuration:** A text box displays the full configuration (model type, preprocessing, hyperparameters) of the model you loaded from the results table.
*   **Wavelength Selection:**
    *   A text box allows you to manually define the exact wavelengths or wavelength ranges to be used in the model. You can use this to force the model to focus on specific known spectral features.
    *   You can use presets like "All", "NIR Only", or "Visible".
    *   The number of selected wavelengths is counted in real-time.

#### 5.7.2. Configuration Sub-Tab

This sub-tab provides a focused interface to modify the most important hyperparameters for the currently selected model type (e.g., PLS, Random Forest). This allows you to manually test variations that may not have been part of the initial automated search.

#### 5.7.3. Results & Diagnostics Sub-Tab

After you click **"▶ Run Model"**, this tab populates with detailed results and diagnostic plots for your refined model.

*   **Performance Metrics:** Displays key metrics like R², RMSE, and more for the cross-validation run.
*   **Prediction Plot:** A scatter plot of the cross-validated predictions vs. the actual (true) values. A perfect model would have all points on a 1:1 line.
*   **Residuals Plot:** A plot of the model's errors (residuals). This is useful for identifying patterns in the model's errors or potential outliers.
*   **Save Model:** The **"💾 Save Model"** button is located here. Clicking it will save the trained model, including its specific preprocessing pipeline and selected wavelengths, into a single `.dasp` file that can be used later in the "Model Prediction" tab.

### 5.8. Tab 8: Model Prediction

This tab allows you to use your saved models to make predictions on new, unseen data.

**Workflow:**

1.  **Load Model(s):**
    *   Click the **"Load Model File(s)"** button.
    *   You can select one or more `.dasp` model files that you previously saved from the "Model Development" tab. The loaded models will be listed.

2.  **Load New Data:**
    *   Click the **"Load New Spectral Data"** button.
    *   Select the directory or file containing the new spectra you want to make predictions on. This follows the same loading logic as the "Import & Preview" tab.

3.  **Run Predictions:**
    *   Click the **"Run Predictions"** button.
    *   The application will automatically apply the correct preprocessing for each loaded model to the new data and generate predictions.

4.  **View Results:**
    *   A table will appear showing the predictions.
    *   Each loaded model will have its own column with its predictions for each new sample.
    *   **Consensus Prediction:** An additional column provides a consensus prediction, which is the average of all the individual model predictions. This can often provide a more stable and reliable estimate.
    *   You can export the prediction results to a CSV file using the **"Export Predictions"** button.

### 5.9. Tab 9: Calibration Transfer

This advanced tab provides tools to correct for instrumental differences, allowing a calibration model built on a "Master" instrument to be applied to data from a "Slave" instrument. The process is guided by a wizard-like interface.

**Concept:**
If you have two instruments, you can't always apply a model from one to the other due to small physical differences. Calibration transfer creates a mathematical transformation to make the data from the slave instrument look like it came from the master instrument.

**Workflow:**

1.  **Load Master and Slave Data:**
    *   In the **"Master Instrument"** and **"Slave Instrument"** sections, load the spectral data and reference files for each instrument. This requires a set of "transfer samples" that have been measured on *both* instruments.
    *   The loading process is identical to the "Import & Preview" tab.

2.  **Build Transfer Model:**
    *   Once both datasets are loaded, select a **Transfer Method**, such as `DS` (Direct Standardization) or `PDS` (Piecewise Direct Standardization).
    *   Click the **"Build Transfer Model"** button. The application will compute the transformation required to map the slave data to the master data.
    *   You can save this transfer model for later use.

3.  **Apply Transfer Model:**
    *   After building or loading a transfer model, you can apply it to a new set of data from the slave instrument.
    *   Load the **"New Slave Data"** you wish to transform.
    *   Click **"Apply Transfer Model"**.

4.  **Generate Outputs:**
    *   The transformed spectra will be displayed in a plot.
    *   You have two options:
        *   **Export Transformed Spectra:** Save the corrected slave spectra to a new file. You can then use this corrected data in the main analysis workflow.
        *   **Predict with Master Model:** Load a `.dasp` model that was built using the original master instrument's data and use it to make predictions directly on the newly transformed slave data.

## 6. Supported File Formats

DASP supports a wide variety of spectral file formats through a unified I/O system that attempts to automatically detect the format.

### 6.1. Format Support Matrix

| Format          | Extensions        | Read | Write | Auto-Detect | Dependencies           |
| --------------- | ----------------- | :--: | :---: | :---------: | ---------------------- |
| CSV             | .csv              |  ✅  |  ✅   |     ✅      | Built-in               |
| Excel           | .xlsx, .xls       |  ✅  |  ✅   |     ✅      | `openpyxl`, `xlsxwriter` |
| ASD (ASCII)     | .asd, .sig        |  ✅  |  ❌   |     ✅      | Built-in               |
| ASD (Binary)    | .asd              |  ✅  |  ❌   |     ✅      | `specdal` (optional)   |
| SPC             | .spc              |  ✅  |  ✅   |     ✅      | `spc-io`               |
| JCAMP-DX        | .jdx, .dx, .jcm   |  ✅  |  ✅   |     ✅      | `jcamp`                |
| ASCII Text      | .txt, .dat        |  ✅  |  ✅   |     ✅      | Built-in               |
| Bruker OPUS     | .0, .1, .2, etc.  |  ✅  |  ❌   |     ✅      | `brukeropus` (optional)|
| PerkinElmer     | .sp               |  ✅  |  ❌   |     ✅      | `specio` (optional)    |

*Refer to the **Installation** section for instructions on installing optional format support.*

### 6.2. Data Layouts

#### Combined CSV/Excel Format (Recommended)

The simplest way to provide data is a single CSV or Excel file that contains everything: sample identifiers, target variables, and all spectral data. The application will automatically detect this format if you select a single file within a directory.

The columns can be in any order. The first row must be a header.

**Example:**
```csv
specimen_id,nitrogen,protein,400.0,401.0,...,2400.0
S001,2.45,15.3,0.123,0.125,...,0.456
S002,2.78,17.4,0.134,0.136,...,0.467
```

#### Separate Spectra and Reference Files

You can also provide your data in separate files.

**1. Spectra File (CSV/Excel "Wide" Format):**

The first column should be a unique sample ID, and the remaining columns should be the spectral intensity at each wavelength.

*Example (`spectra.csv`):*
```csv
sample_id,400.0,401.0,402.0,...,2400.0
S001,0.123,0.125,0.127,...,0.456
S002,0.134,0.136,0.138,...,0.467
```

**2. Reference File (CSV/Excel):**

This file maps the sample IDs from your spectra file to the target variables you want to predict.

*Example (`reference.csv`):*
```csv
sample_id,nitrogen,carbon,protein
S001,2.45,45.2,15.3
S002,2.78,43.8,17.4
```
The software intelligently matches IDs, so it can handle small differences in naming (e.g., "Sample_01" in the reference file will match "Sample_01.asd" in the spectral data directory).

#### Directory of Individual Files

For formats that save one file per spectrum (like `.asd`, `.spc`, `.jdx`), simply place all the files in a single directory and select that directory as the "Spectral File Directory". You will then need to provide a separate Reference File that maps the filenames to the target variables.

## 7. Conclusion

This manual has provided a comprehensive overview of the DASP application, from installation to advanced features like calibration transfer. With its automated workflows, interactive interface, and powerful modeling capabilities, DASP is designed to significantly accelerate your spectral analysis research.

We hope this guide helps you unlock the full potential of your spectral data. Happy modeling!


