"""Global constants for Spectral Predict."""

# Random state used throughout the codebase for reproducibility.
# This value is hardcoded in 65+ locations across the codebase.
# Changing this will NOT automatically update all usages - each
# file would need to import and use this constant.
RANDOM_STATE = 42
