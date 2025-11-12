# Install required R packages for validation testing

cat("Installing required R packages for validation testing...\n")

# Setup user library if it doesn't exist
user_lib <- Sys.getenv("R_LIBS_USER")
if (!dir.exists(user_lib)) {
  dir.create(user_lib, recursive = TRUE)
  cat(sprintf("Created user library at: %s\n", user_lib))
}

# Add user library to library paths
.libPaths(c(user_lib, .libPaths()))
cat(sprintf("Using library: %s\n", user_lib))

# List of required packages
packages <- c("pls", "randomForest", "xgboost", "glmnet", "jsonlite")

# Install packages that are not already installed
for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("Installing %s...\n", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org/", lib = user_lib)
  } else {
    cat(sprintf("%s is already installed.\n", pkg))
  }
}

cat("\nPackage installation complete!\n")
cat("\nVerifying installations:\n")

# Verify each package can be loaded
for (pkg in packages) {
  tryCatch({
    library(pkg, character.only = TRUE)
    cat(sprintf("✓ %s\n", pkg))
  }, error = function(e) {
    cat(sprintf("✗ %s - ERROR: %s\n", pkg, e$message))
  })
}
