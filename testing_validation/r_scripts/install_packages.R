# ==============================================================================
# DASP Validation Testing - R Package Installation
# ==============================================================================
#
# This script installs all R packages needed for comparing DASP results
# against standard R implementations.
#
# Usage: Rscript install_packages.R
#
# ==============================================================================

cat("================================================================================\n")
cat("DASP Validation Testing - Installing R Packages\n")
cat("================================================================================\n\n")

# Set CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org"))

# List of required packages
packages <- c(
  # Core modeling packages
  "pls",              # PLS regression
  "glmnet",           # Ridge, Lasso, ElasticNet
  "randomForest",     # Random Forest
  "xgboost",          # XGBoost
  "lightgbm",         # LightGBM (if available)
  "e1071",            # SVM

  # Preprocessing
  "prospectr",        # Spectral preprocessing (SNV, Savitzky-Golay)

  # Utilities
  "jsonlite",         # JSON export
  "caret",            # ML utilities, PLS-DA

  # Variable selection
  # Note: mdatools may not be on CRAN, install separately if needed

  # Data manipulation
  "dplyr",
  "tidyr",
  "readr"
)

# Function to install if not already installed
install_if_missing <- function(pkg) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat(sprintf("Installing %s...\n", pkg))
    install.packages(pkg, dependencies = TRUE)
    return(TRUE)
  } else {
    cat(sprintf("%s is already installed (version %s)\n",
                pkg, packageVersion(pkg)))
    return(FALSE)
  }
}

# Install packages
cat("\nChecking and installing packages...\n\n")
newly_installed <- sapply(packages, install_if_missing)

# Special handling for mdatools (may need to install from GitHub)
cat("\nChecking for mdatools (for SPA, iPLS)...\n")
if (!require("mdatools", quietly = TRUE)) {
  cat("mdatools not found. Attempting to install from CRAN...\n")
  tryCatch({
    install.packages("mdatools")
  }, error = function(e) {
    cat("Warning: mdatools installation failed. You may need to install from GitHub:\n")
    cat("  devtools::install_github('svkucheryavski/mdatools')\n")
  })
}

# Special handling for lightgbm (may require manual installation)
cat("\nChecking for lightgbm...\n")
if (!require("lightgbm", quietly = TRUE)) {
  cat("lightgbm not found. Attempting to install...\n")
  tryCatch({
    install.packages("lightgbm", repos = "https://cloud.r-project.org")
  }, error = function(e) {
    cat("Warning: lightgbm installation failed. This is optional.\n")
    cat("  See: https://lightgbm.readthedocs.io/en/latest/R/index.html\n")
  })
}

# Verify installations
cat("\n================================================================================\n")
cat("Verifying installations...\n")
cat("================================================================================\n\n")

verification_results <- sapply(packages, function(pkg) {
  if (require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat(sprintf("[OK] %s %s\n", pkg, packageVersion(pkg)))
    return(TRUE)
  } else {
    cat(sprintf("[FAILED] %s\n", pkg))
    return(FALSE)
  }
})

# Summary
cat("\n================================================================================\n")
cat("Installation Summary\n")
cat("================================================================================\n\n")

total_packages <- length(packages)
installed_packages <- sum(verification_results)
failed_packages <- total_packages - installed_packages

cat(sprintf("Total packages: %d\n", total_packages))
cat(sprintf("Successfully installed: %d\n", installed_packages))
cat(sprintf("Failed: %d\n", failed_packages))

if (failed_packages > 0) {
  cat("\nWarning: Some packages failed to install. Check the output above.\n")
} else {
  cat("\nAll packages installed successfully!\n")
}

cat("\n================================================================================\n")
cat("Next steps:\n")
cat("  1. Run regression testing: Rscript regression_models.R\n")
cat("  2. Run classification testing: Rscript classification_models.R\n")
cat("  3. Compare results with DASP\n")
cat("================================================================================\n")
