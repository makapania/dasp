#!/usr/bin/env python3
"""
Replace the Adam-based train_weak_learner! function with LBFGS-based version.
"""

import re

# Read the current neural_boosted.jl file
with open('julia_port/SpectralPredict/src/neural_boosted.jl', 'r', encoding='utf-8') as f:
    content = f.read()

# Read the new function implementation
with open('replace_train_weak_learner.jl', 'r', encoding='utf-8') as f:
    new_function = f.read()

# Find and replace the old train_weak_learner! function (lines 248-345)
# Pattern: from the docstring """    train_weak_learner! to the end function
pattern = r'"""(\s+)train_weak_learner!\(model, X, y, max_iter, alpha, verbose\).*?^end$'

# Replacement
replacement = new_function.strip()

# Perform the replacement
new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)

# Write back
with open('julia_port/SpectralPredict/src/neural_boosted.jl', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("✓ Successfully replaced train_weak_learner!() function with LBFGS implementation")
print("✓ Old function: Adam optimizer (lines 248-345)")
print("✓ New function: LBFGS optimizer from Optim.jl")
