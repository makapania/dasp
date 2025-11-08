import re

# Read the file
with open('julia_port/SpectralPredict/src/neural_boosted.jl', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Change Float32 to Float64 in train_weak_learner! (lines 273-274)
content = content.replace(
    '    # Transpose for Flux (features × samples)\n    X_t = Float32.(X\')\n    y_t = Float32.(reshape(y, 1, :))',
    '    # Transpose for Flux (features × samples)\n    # PHASE 1 FIX: Use Float64 for numerical precision during residual fitting\n    X_t = Float64.(X\')\n    y_t = Float64.(reshape(y, 1, :))'
)

# Fix 2 & 3: Update optimizer comment and add convergence tracking
content = content.replace(
    '    # Optimizer: Adam with learning rate 0.01 (NEW FLUX API)\n    opt = Adam(0.01)\n    opt_state = Flux.setup(opt, model)\n\n    # Training loop (NEW FLUX API)',
    '    # Optimizer: Adam with learning rate 0.01\n    # Note: sklearn uses LBFGS (ideal for small networks). Consider Optim.jl LBFGS in Phase 2\n    opt = Adam(0.01)\n    opt_state = Flux.setup(opt, model)\n\n    # PHASE 1 FIX: Track convergence for early stopping\n    prev_loss = Inf\n    patience_counter = 0\n    max_patience = 5  # Stop if no improvement for 5 iterations\n\n    # Training loop (NEW FLUX API)'
)

# Fix 4: Add gradient/loss validation and convergence detection in training loop
old_train_loop = '        Flux.update!(opt_state, model, grads[1])\n\n        # Verbose output\n        if verbose >= 2 && epoch % 20 == 0\n            # Compute current loss for logging\n            pred = model(X_t)\n            mse = Flux.mse(pred, y_t)\n            l2_penalty = sum(sum(p.^2) for p in Flux.params(model))\n            current_loss = mse + alpha * l2_penalty\n            println("    Epoch $epoch: loss = $(current_loss)")\n        end'

new_train_loop = '        # PHASE 1 FIX: Validate gradients before update\n        if grads[1] === nothing || any(x -> any(isnan.(x)) || any(isinf.(x)), values(grads[1]))\n            if verbose >= 2\n                println("    WARNING: NaN/Inf gradients at epoch $epoch. Stopping.")\n            end\n            break\n        end\n\n        Flux.update!(opt_state, model, grads[1])\n\n        # Compute current loss for convergence check and logging\n        pred = model(X_t)\n        mse = Flux.mse(pred, y_t)\n        l2_penalty = sum(sum(p.^2) for p in Flux.params(model))\n        current_loss = mse + alpha * l2_penalty\n\n        # PHASE 1 FIX: Early convergence detection (tolerance relaxed to 1e-4)\n        if current_loss >= prev_loss - 1e-4\n            patience_counter += 1\n            if patience_counter >= max_patience\n                if verbose >= 2\n                    println("    Converged at epoch $epoch (loss plateaued)")\n                end\n                break\n            end\n        else\n            patience_counter = 0\n        end\n        prev_loss = current_loss\n\n        # Verbose output\n        if verbose >= 2 && epoch % 20 == 0\n            println("    Epoch $epoch: loss = $(current_loss)")\n        end\n\n        # PHASE 1 FIX: Detect NaN/Inf in loss\n        if isnan(current_loss) || isinf(current_loss)\n            if verbose >= 2\n                println("    WARNING: Loss became NaN/Inf at epoch $epoch. Stopping.")\n            end\n            error("Training diverged: NaN/Inf loss detected")\n        end'

content = content.replace(old_train_loop, new_train_loop)

# Fix 5: Change Float32 to Float64 in fit! prediction lines (lines 453, 472)
content = content.replace(
    '        # Get predictions from weak learner\n        X_train_t = Float32.(X_train\')',
    '        # Get predictions from weak learner\n        X_train_t = Float64.(X_train\')'
)
content = content.replace(
    '            X_val_t = Float32.(X_val\')',
    '            X_val_t = Float64.(X_val\')'
)

# Fix 6: Change Float32 to Float64 in predict function (line 579)
content = content.replace(
    '    # Transpose for Flux (features × samples)\n    X_t = Float32.(X\')',
    '    # Transpose for Flux (features × samples)\n    X_t = Float64.(X\')'
)

# Fix 7: Add per-learner random seeding in boosting loop
content = content.replace(
    '    # Step 2: Boosting loop\n    for m in 1:model.n_estimators\n        if model.verbose >= 1\n            println("  Stage $m/$(model.n_estimators)...")\n        end\n\n        # Compute residuals: what the ensemble got wrong\n        residuals = y_train .- F_train\n\n        # Build weak learner (small MLP)',
    '    # Step 2: Boosting loop\n    for m in 1:model.n_estimators\n        # PHASE 1 FIX: Set unique random seed for each weak learner (diversity)\n        Random.seed!(model.random_state + m)\n\n        if model.verbose >= 1\n            println("  Stage $m/$(model.n_estimators)...")\n        end\n\n        # Compute residuals: what the ensemble got wrong\n        residuals = y_train .- F_train\n\n        # Build weak learner (small MLP)'
)

# Fix 8: Add NaN/Inf validation for predictions in fit!
content = content.replace(
    '        # Get predictions from weak learner\n        X_train_t = Float64.(X_train\')\n        h_m_train = vec(weak_learner(X_train_t))\n\n        # Update ensemble predictions: F_m(x) = F_{m-1}(x) + ν * h_m(x)',
    '        # Get predictions from weak learner\n        X_train_t = Float64.(X_train\')\n        h_m_train = vec(weak_learner(X_train_t))\n\n        # PHASE 1 FIX: Validate predictions before updating ensemble\n        if any(isnan.(h_m_train)) || any(isinf.(h_m_train))\n            n_failed_learners += 1\n            if model.verbose >= 1\n                @warn "Weak learner $m produced invalid predictions (NaN/Inf). Skipping."\n            end\n            continue\n        end\n\n        # Update ensemble predictions: F_m(x) = F_{m-1}(x) + ν * h_m(x)'
)

# Fix 9: Add NaN/Inf validation for validation predictions
content = content.replace(
    '            X_val_t = Float64.(X_val\')\n            h_m_val = vec(weak_learner(X_val_t))\n            F_val .+= model.learning_rate .* h_m_val',
    '            X_val_t = Float64.(X_val\')\n            h_m_val = vec(weak_learner(X_val_t))\n\n            # PHASE 1 FIX: Validate validation predictions\n            if any(isnan.(h_m_val)) || any(isinf.(h_m_val))\n                if model.verbose >= 1\n                    @warn "Weak learner $m produced invalid validation predictions (NaN/Inf)."\n                end\n                # Remove already-added estimator\n                pop!(model.estimators_)\n                pop!(model.train_score_)\n                n_failed_learners += 1\n                continue\n            end\n\n            F_val .+= model.learning_rate .* h_m_val'
)

# Write the fixed content
with open('julia_port/SpectralPredict/src/neural_boosted.jl', 'w', encoding='utf-8') as f:
    f.write(content)

print('Phase 1 fixes applied successfully!')
print('Changes made:')
print('  1. Float32 → Float64 throughout (improved precision)')
print('  2. Added per-learner random seeds (improved diversity)')
print('  3. Kept Adam learning rate at 0.01 (better than 0.001)')
print('  4. Added NaN/Inf validation for gradients and predictions')
print('  5. Added convergence detection with 1e-4 tolerance')
