"""
Tooltip infrastructure for Spectral Predict v3.

Provides centralized tooltip content and helper functions for adding
themed tooltips to Dear PyGui elements.

Ported from V1's TOOLTIP_CONTENT dictionary with additions for V3-specific features.
"""

import dearpygui.dearpygui as dpg
from typing import Optional

# Lazy import to avoid circular dependency
_COLORS = None


def _get_colors():
    """Lazy load COLORS to avoid circular import."""
    global _COLORS
    if _COLORS is None:
        from .theme import COLORS
        _COLORS = COLORS
    return _COLORS


# =============================================================================
# TOOLTIP CONTENT DICTIONARY
# =============================================================================
# Ported from V1 (spectral_predict_gui_optimized.py lines 212-744)
# Plus V3-specific additions

TOOLTIP_CONTENT = {
    # ===== MODEL DESCRIPTIONS (from V1) =====
    'models': {
        'PLS': (
            "Partial Least Squares (PLS) is a linear regression method that finds latent variables "
            "maximizing covariance between spectral data and target values. Excellent for high-dimensional "
            "data with multicollinearity (correlated wavelengths). Fast, interpretable, and works well "
            "with more variables than samples. Ideal for quantitative spectroscopy."
        ),
        'PLS-DA': (
            "PLS Discriminant Analysis extends PLS for classification tasks by treating categorical "
            "labels as continuous variables. Uses latent variables to maximize class separation. "
            "Effective for spectral classification with multiple correlated features. Returns probability "
            "scores for each class."
        ),
        'Ridge': (
            "Ridge Regression is linear regression with L2 regularization that shrinks coefficient "
            "magnitudes toward zero without eliminating any. Prevents overfitting by penalizing large "
            "coefficients. Good for spectral data with multicollinearity. Controlled by alpha parameter - "
            "higher alpha = more regularization."
        ),
        'Lasso': (
            "Lasso Regression uses L1 regularization that can shrink coefficients exactly to zero, "
            "performing automatic feature selection. Creates sparse models by identifying the most "
            "important wavelengths. Useful when you want to know which specific wavelengths matter most. "
            "Higher alpha = more sparsity."
        ),
        'ElasticNet': (
            "ElasticNet combines Ridge (L2) and Lasso (L1) regularization, getting benefits of both. "
            "Can select groups of correlated features (like Lasso) while maintaining stability (like Ridge). "
            "Controlled by alpha (regularization strength) and l1_ratio (0=Ridge only, 0.5=balanced, 1=Lasso only). "
            "Excellent for spectral data with highly correlated wavelengths."
        ),
        'RandomForest': (
            "Random Forest builds many decision trees on random subsets of data and features, then "
            "averages their predictions. Nonlinear, handles outliers well, and resistant to overfitting. "
            "No assumptions about data distribution. Can capture complex spectral patterns. "
            "Slower than linear models but very robust and versatile."
        ),
        'MLP': (
            "Multi-Layer Perceptron is a feedforward neural network with hidden layers that can learn "
            "complex nonlinear relationships. Uses backpropagation for training. Can approximate any "
            "continuous function given enough neurons. Good for spectral data with nonlinear relationships "
            "but requires more samples and careful tuning to avoid overfitting."
        ),
        'SVR': (
            "Support Vector Regression finds a hyperplane that best fits the data while allowing "
            "a margin of tolerance (epsilon). Can use kernels (linear, RBF, poly) to capture nonlinear "
            "patterns. Effective for small to medium datasets. RBF kernel good for complex spectral patterns. "
            "Memory-intensive for large datasets."
        ),
        'SVM': (
            "Support Vector Machine for classification. Finds optimal hyperplane separating classes. "
            "Can use kernels (linear, RBF, poly) to capture nonlinear patterns. Effective for small to "
            "medium datasets. RBF kernel works well for complex spectral patterns."
        ),
        'XGBoost': (
            "XGBoost is gradient boosting that builds trees sequentially, each correcting errors of "
            "previous trees. Industry-leading performance with built-in regularization (L1/L2). "
            "Handles missing data, reduces overfitting, and excellent for spectral data. Fast parallel "
            "processing. Highly tunable with many hyperparameters for optimal performance."
        ),
        'LightGBM': (
            "LightGBM is Microsoft's gradient boosting framework optimized for speed and memory efficiency. "
            "Uses leaf-wise tree growth (vs level-wise) for faster training. Excellent for large datasets "
            "with many features (like spectral data). Faster than XGBoost with similar performance. "
            "Great for high-dimensional spectroscopy applications."
        ),
        'CatBoost': (
            "CatBoost is Yandex's gradient boosting that handles categorical features automatically "
            "and reduces overfitting with ordered boosting. Very robust with minimal tuning required - "
            "good default hyperparameters. Slower than LightGBM but often achieves better accuracy out-of-box. "
            "Note: Requires Visual Studio 2022 Build Tools on Windows."
        ),
        'NeuralBoosted': (
            "Neural Boosted combines gradient boosting with neural networks as base learners instead of "
            "decision trees. Each boosting round adds a small neural network. Can capture very complex "
            "nonlinear patterns in spectral data. More powerful than standard tree-based boosting but "
            "slower to train. Best for complex spectroscopy problems with sufficient data."
        ),
    },

    # ===== PREPROCESSING METHOD DESCRIPTIONS (from V1) =====
    'preprocessing': {
        'Raw': (
            "Raw (no preprocessing) uses the original, unprocessed spectral data exactly as measured. "
            "No transformations applied. Useful as a baseline comparison to see if preprocessing helps. "
            "Works well when spectra are already clean and consistent, or when you want to preserve "
            "absolute intensity information. Start here to establish baseline performance."
        ),
        'SNV': (
            "Standard Normal Variate (SNV) corrects for scatter effects and baseline variations by "
            "normalizing each spectrum to zero mean and unit variance. Removes multiplicative and "
            "additive effects from particle size, path length differences, and light scattering. "
            "Very effective for solid samples and diffuse reflectance spectroscopy. "
            "One of the most commonly used preprocessing methods in NIR spectroscopy."
        ),
        'SG1': (
            "Savitzky-Golay 1st Derivative calculates the first derivative of spectra using polynomial "
            "smoothing. Removes baseline offset and linear trends while preserving peak shapes. "
            "Highlights regions of spectral change and reduces baseline drift effects. Enhances "
            "differences between spectra. Derivative window size controls smoothing - larger windows = "
            "more smoothing but may miss fine details."
        ),
        'SG2': (
            "Savitzky-Golay 2nd Derivative calculates the second derivative of spectra. "
            "Removes both baseline offset and linear drift, and enhances peak shapes by showing "
            "inflection points and hidden shoulders. Very sensitive to subtle spectral features. "
            "Provides excellent peak resolution but amplifies noise more than 1st derivative. "
            "Best for overlapping peaks and complex spectral patterns."
        ),
        'SG3': (
            "Savitzky-Golay 3rd Derivative provides even higher resolution than 2nd derivative. "
            "Very sensitive to subtle spectral features but significantly amplifies noise. "
            "Use with caution - may require larger smoothing windows to control noise."
        ),
        'SG4': (
            "Savitzky-Golay 4th Derivative provides maximum resolution but extreme noise amplification. "
            "Rarely used in practice. Only consider for very high SNR spectra with overlapping features."
        ),
        'deriv_snv': (
            "Derivative then SNV applies Savitzky-Golay derivative first, followed by SNV normalization. "
            "This advanced combination removes baseline effects (derivative) then normalizes the "
            "derivative spectra (SNV). Less common than SNV alone or derivative alone. "
            "Can be useful for very noisy data or when both scatter correction and baseline removal "
            "are needed, but may over-process in some cases."
        ),
        'snv_deriv': (
            "SNV then Derivative applies SNV normalization first, followed by Savitzky-Golay derivative. "
            "Normalizes spectra to remove scatter effects, then takes derivative to remove baseline. "
            "The order matters - this preserves derivative shape better than Deriv+SNV for some data."
        ),
        'window_size': (
            "Derivative window size controls the Savitzky-Golay smoothing window (must be odd number). "
            "Smaller windows (7-11) = less smoothing, preserves fine details, more noise. "
            "Medium windows (13-17) = balanced smoothing, good for most applications (default: 17). "
            "Larger windows (19-25) = more smoothing, removes noise, may lose sharp features. "
            "Choose based on your spectral resolution and noise level. Can test multiple values."
        ),
    },

    # ===== HYPERPARAMETER DESCRIPTIONS (from V1) =====
    'hyperparameters': {
        # Neural Boosted
        'neuralboosted_n_estimators': (
            "Number of boosting rounds (sequential neural networks to train). Each round adds a small "
            "neural network that corrects errors from previous rounds. More estimators = better fit but "
            "slower training and risk of overfitting. Typical range: 50-200 for spectral data."
        ),
        'neuralboosted_learning_rate': (
            "Controls step size when updating the model (how much each boosting round contributes). "
            "Lower values (0.05-0.1) = more conservative, less overfitting, but needs more estimators. "
            "Higher values (0.2-0.3) = faster learning but risk overfitting. "
            "Range 0.1-0.3 works well for most spectroscopy applications."
        ),
        'neuralboosted_hidden_layer_size': (
            "Number of neurons in the hidden layer of each weak learner neural network. "
            "Keep small (3-5) to maintain weak learner properties and ensemble diversity. "
            "Larger values (7-10) create stronger learners but may reduce benefit of boosting. "
            "Default: 3 and 5 (grid search explores both options)."
        ),
        'neuralboosted_activation': (
            "Activation function for the hidden layer in weak learner neural networks. "
            "tanh = Smooth sigmoid-like nonlinear function (default, good for spectral data). "
            "identity = Linear activation (fast, simple baseline). "
            "relu = Rectified Linear Unit (standard deep learning activation). "
            "logistic = Sigmoid function (0 to 1 output range)."
        ),
        'neuralboosted_max_iter': (
            "Maximum iterations for each neural base learner during training. Controls how long each "
            "small neural network trains. Higher = better fit per network but slower overall. "
            "100-200 typically sufficient for spectral features. Reduced from default 500 for speed optimization."
        ),

        # Random Forest
        'rf_n_estimators': (
            "Number of decision trees in the forest. More trees = better performance and stability but "
            "slower training and prediction. Returns typically diminish after 100-200 trees. "
            "Typical range: 100-500 for spectroscopy. Can use 1000+ if you have time."
        ),
        'rf_max_depth': (
            "Maximum depth each tree can grow. None = unlimited (trees grow until leaves are pure or "
            "reach min_samples_split). Lower values (10-30) prevent overfitting by limiting tree complexity. "
            "Higher values or None allow capturing complex patterns but risk overfitting."
        ),
        'rf_min_samples_split': "Minimum samples required to split an internal node.",
        'rf_min_samples_leaf': "Minimum samples required in each leaf node.",
        'rf_max_features': "sqrt = square root of features, log2 = log2 of features, None = all features.",
        'rf_bootstrap': "Whether to use bootstrap samples when building trees.",
        'rf_max_leaf_nodes': "Limits tree growth, None = unlimited.",
        'rf_min_impurity_decrease': "Minimum impurity decrease required to split a node.",

        # Ridge
        'ridge_alpha': (
            "Regularization strength (penalty for large coefficients). Controls the bias-variance tradeoff. "
            "Lower values (0.001-0.1) = less regularization, closer to ordinary least squares. "
            "Higher values (1-10+) = more shrinkage, simpler model. Optimal value depends on data complexity "
            "and noise level. Cross-validation finds the best alpha."
        ),

        # Lasso
        'lasso_alpha': (
            "Regularization strength that controls sparsity (how many coefficients become zero). "
            "Lower values (0.001-0.01) = keep more wavelengths, less feature selection. "
            "Higher values (0.1-1+) = aggressive feature selection, very sparse model. "
            "Useful for identifying the most important wavelengths in your spectra."
        ),

        # ElasticNet
        'elasticnet_alpha': (
            "Overall regularization strength combining L1 and L2 penalties. Similar to Ridge/Lasso alpha. "
            "Lower = less regularization. Higher = more regularization. Works with l1_ratio to determine "
            "the mix of L1 (sparsity) vs L2 (coefficient shrinkage). Typical range: 0.01-1.0."
        ),
        'elasticnet_l1_ratio': (
            "Controls the mix of L1 (Lasso) vs L2 (Ridge) regularization. "
            "0.0 = Ridge only (no feature selection, all wavelengths kept). "
            "0.5 = balanced mix of L1 and L2. "
            "1.0 = Lasso only (aggressive feature selection). "
            "Values 0.3-0.7 often work well for spectral data with correlated features."
        ),

        # PLS
        'pls_max_n_components': (
            "Maximum number of latent variables (components) to evaluate during model optimization. "
            "PLS extracts these components from spectral data to predict target values. More components "
            "can capture more variance but risk overfitting. Cross-validation automatically finds the "
            "optimal number <= this maximum. Typical range: 5-20 for spectroscopy."
        ),
        'pls_max_iter': (
            "Maximum iterations for the PLS algorithm to converge when extracting each component. "
            "PLS uses an iterative algorithm (NIPALS) to find latent variables. Most spectral data "
            "converges quickly (<100 iterations). 500-1000 is safe for all cases."
        ),
        'pls_tol': (
            "Convergence tolerance for the PLS iterative algorithm. Smaller = more precision but "
            "longer computation. Algorithm stops when change between iterations < tolerance. "
            "1e-6 (0.000001) is standard and works well for spectral data."
        ),
        'pls_scale': "Whether to scale X and Y data before fitting. Usually True for spectral data.",

        # XGBoost
        'xgb_n_estimators': (
            "Number of boosting rounds (trees) to build sequentially. Each tree corrects errors from "
            "previous trees. More estimators = better fit but slower and risk overfitting (use with "
            "lower learning_rate). Typical range: 100-200 for spectral data."
        ),
        'xgb_learning_rate': (
            "Step size shrinkage to prevent overfitting (also called eta). Lower values make model more "
            "conservative by reducing the contribution of each tree. "
            "0.01-0.05 = very conservative (use with many trees). "
            "0.1 = balanced (standard default). "
            "0.2-0.3 = aggressive (faster but risk overfitting)."
        ),
        'xgb_max_depth': (
            "Maximum depth for each decision tree. Controls model complexity and overfitting. "
            "Shallow trees (3-6) = simple patterns, less overfitting, faster. "
            "Deep trees (9-12) = complex patterns, more overfitting risk. "
            "For spectroscopy with many features, 3-6 usually optimal."
        ),
        'xgb_subsample': (
            "Fraction of training samples used for each tree (row sampling). Adds randomness to prevent "
            "overfitting. 1.0 = use all samples. 0.8 = use 80% (recommended for robustness). "
            "0.5-0.7 = more aggressive sampling (helps with very noisy data)."
        ),
        'xgb_colsample_bytree': (
            "Fraction of features (wavelengths) used for each tree. Critical for high-dimensional spectral "
            "data (1000+ wavelengths). 1.0 = use all wavelengths. 0.8 = use 80% (recommended). "
            "0.5-0.7 = aggressive feature sampling (increases tree diversity)."
        ),
        'xgb_reg_alpha': (
            "L1 regularization (Lasso-style) on tree leaf weights. Encourages sparsity in leaf values. "
            "0 = no L1 penalty (default). 0.1-0.5 = light regularization (recommended for high-dim data). "
            "1-5 = strong regularization (very sparse model)."
        ),
        'xgb_reg_lambda': (
            "L2 regularization (Ridge-style) on tree leaf weights. Smooths leaf values without forcing "
            "sparsity. 1.0 = default (light regularization). 5-10 = moderate. "
            "Higher values make model more conservative."
        ),
        'xgb_min_child_weight': (
            "Minimum sum of instance weight needed in a child node. Controls overfitting by "
            "requiring minimum data in each leaf. 1 = allow small leaves. "
            "3-5 = moderate constraint. 7-10 = conservative."
        ),
        'xgb_gamma': (
            "Minimum loss reduction required to make a split. Regularization parameter that makes the "
            "algorithm more conservative. 0 = split freely (default). 0.1-0.5 = light regularization. "
            "1-5 = strong regularization."
        ),

        # LightGBM
        'lightgbm_n_estimators': "Number of boosting rounds. More = better but slower.",
        'lightgbm_learning_rate': "Step size for each boosting round. Lower = more conservative.",
        'lightgbm_num_leaves': "Maximum number of leaves per tree. Controls complexity.",
        'lightgbm_max_depth': (
            "Maximum tree depth. -1 = no limit (controlled by num_leaves and min_data_in_leaf instead). "
            "5-10 = shallow to moderate trees. LightGBM uses leaf-wise growth, so depth matters less."
        ),
        'lightgbm_min_child_samples': (
            "Minimum samples required in each leaf node. 5 = allow small leaves (may overfit). "
            "20-50 = moderate constraint. 100+ = conservative (smoother model)."
        ),
        'lightgbm_subsample': (
            "Fraction of training samples used per tree. "
            "1.0 = use all samples. 0.8-0.9 = recommended (adds robustness)."
        ),
        'lightgbm_colsample_bytree': (
            "Fraction of features used per tree. Important for spectral data with many wavelengths. "
            "1.0 = use all wavelengths. 0.8-0.9 = recommended."
        ),
        'lightgbm_reg_alpha': "L1 regularization on leaf weights. 0 = none, 0.1-0.5 = light.",
        'lightgbm_reg_lambda': "L2 regularization on leaf weights. 0 = none, 0.5-1.0 = light.",

        # MLP
        'mlp_activation': (
            "Activation function for hidden layers. "
            "relu = Rectified Linear Unit - fast, works well, standard choice. "
            "tanh = smooth, outputs -1 to 1, good for normalized data. "
            "logistic = Sigmoid (0 to 1) - smooth, slower than relu."
        ),
        'mlp_solver': (
            "Optimization algorithm for training. "
            "adam = Adaptive learning rate optimizer - robust, fast (recommended). "
            "lbfgs = Quasi-Newton - good for small datasets. "
            "sgd = Stochastic Gradient Descent - requires careful tuning."
        ),
        'mlp_alpha': (
            "L2 regularization parameter. "
            "0.0001-0.001 = less regularization (risk overfitting). "
            "0.01-0.1 = balanced. 1-10 = strong (may underfit)."
        ),
        'mlp_hidden_layer_sizes': "Tuple of layer sizes, e.g., (100, 50) = two layers with 100 and 50 neurons.",

        # SVR/SVM
        'svr_C': (
            "Regularization parameter. Smaller C = more regularization (smoother). "
            "Larger C = less regularization (fit training data closely). "
            "Typical range: 0.1-100."
        ),
        'svr_kernel': (
            "Kernel function. "
            "linear = Linear kernel - fast, good for linearly separable data. "
            "rbf = Radial Basis Function - captures nonlinear patterns (recommended). "
            "poly = Polynomial - specific nonlinear patterns."
        ),
        'svr_gamma': (
            "Kernel coefficient for rbf, poly, sigmoid. Defines influence reach of training examples. "
            "Small gamma = far reach (smooth). Large gamma = close reach (complex). "
            "scale = 1/(n_features * X.var()) - good default."
        ),
        'svr_epsilon': "Epsilon-tube width. Points within epsilon of true value have zero loss.",

        # CatBoost
        'catboost_iterations': "Number of boosting iterations. More = better but slower.",
        'catboost_learning_rate': "Step size for each iteration. Lower = more conservative.",
        'catboost_depth': "Depth of trees. 4-10 typical range.",
        'catboost_l2_leaf_reg': (
            "L2 regularization for leaf values. "
            "1.0 = light. 3.0 = default. 10-30 = strong."
        ),
        'catboost_border_count': (
            "Number of splits for numerical features. More = more precision but slower. "
            "32-64 = fast. 128 = balanced. 254 = maximum precision."
        ),
        'catboost_bagging_temperature': (
            "Controls Bayesian bootstrap intensity. "
            "0.0 = no bagging. 1.0 = default. 3-10 = aggressive."
        ),
        'catboost_random_strength': (
            "Randomness for scoring splits. "
            "0.0 = deterministic. 1.0 = default. 2-5 = more random."
        ),
    },

    # ===== RANKING PENALTIES (from V1) =====
    'ranking': {
        'variable_penalty': (
            "Controls how much using many wavelengths affects model ranking. Uses cubic scaling "
            "for gentle impact at low values (exploration-friendly). "
            "0 = ignore variable count, rank only by performance (R2 or Accuracy). "
            "2 = minimal penalty (~1% impact for using all wavelengths). "
            "5 = balanced penalty favoring parsimony without dominating performance. "
            "10 = strong preference for fewer wavelengths. "
            "Recommended: 2 for exploration, 5-7 for deployment model selection."
        ),
        'complexity_penalty': (
            "Controls how much model complexity (latent variables, tree depth) affects ranking. "
            "Uses cubic scaling for gentle impact at low values. "
            "0 = ignore complexity, rank only by performance. "
            "2 = minimal penalty for complex models. "
            "5 = balanced penalty preferring simpler models when performance is similar. "
            "10 = strong preference for simple, interpretable models. "
            "Recommended: 2 for exploration, 5-7 for interpretable models."
        ),
    },

    # ===== CALIBRATION TRANSFER METHODS (from V1) =====
    'calibration_transfer': {
        # Transfer Methods
        'method_DS': (
            "Direct Standardization (DS) is a simple pairwise calibration transfer method. "
            "Builds a linear transformation matrix F that directly maps slave spectra to master spectra: "
            "X_master = X_slave x F. Fast and straightforward, works well when master and slave "
            "instruments have similar wavelength grids. Best for simple spectral differences. "
            "Requires paired samples measured on both instruments."
        ),
        'method_PDS': (
            "Piecewise Direct Standardization (PDS) is a local version of DS that models each master "
            "wavelength independently using a sliding window of neighboring slave wavelengths. "
            "More flexible than global DS, better at handling nonlinear wavelength dependencies. "
            "Window size controls how many neighboring wavelengths are used (typical: 7-15). "
            "Good for instruments with slight wavelength misalignments."
        ),
        'method_TSR': (
            "Transfer by Sample Regression (TSR) selects a subset of representative transfer samples "
            "that span the spectral space, then uses only these samples to build the transformation. "
            "More efficient than using all transfer samples, reduces overfitting. Sample selection uses "
            "Kennard-Stone algorithm to ensure good coverage of spectral diversity."
        ),
        'method_CTAI': (
            "Calibration Transfer via Adaptive Integration (CTAI) combines spectral standardization "
            "with adaptive selection of informative wavelengths. Uses an iterative algorithm to identify "
            "and weight wavelengths that transfer well between instruments while down-weighting problematic "
            "regions (e.g., noise, nonlinear response). Generally provides robust transfer with minimal tuning."
        ),
        'method_NSPFCE': (
            "Null-Space Projection followed by Feature Correlation Enhancement (NS-PFCE) is an advanced "
            "method that removes instrument-specific variance while preserving chemical information. "
            "Uses wavelength selection algorithms (VCPA-IRIV, CARS, SPA) to identify informative features, "
            "then projects out instrument-specific interference using null-space operations. "
            "Slower than other methods but very effective for complex instrumental differences."
        ),
        'method_JYPLS': (
            "Joint-Y Partial Least Squares Inverse (JYPLS-inv) uses PLS regression to model the "
            "master-slave relationship, treating master spectra as 'Y' and slave spectra as 'X'. "
            "The PLS model learns latent variables capturing the systematic spectral differences. "
            "More flexible than DS for nonlinear relationships, but requires more transfer samples (30+)."
        ),

        # Transfer Parameters
        'param_ds_lambda': (
            "Regularization strength for Direct Standardization (DS). Controls the bias-variance tradeoff. "
            "Smaller values (0.0001-0.001) = less regularization, fits transfer samples more closely. "
            "Larger values (0.01-0.1) = more regularization, smoother transformation. "
            "Default 0.001 works well for most cases."
        ),
        'param_pds_window': (
            "Window size for Piecewise Direct Standardization (PDS). Must be odd number. "
            "Small windows (5-9) = more local, captures fine spectral details, may be noisy. "
            "Medium windows (11-15) = balanced, recommended for most applications (default: 11). "
            "Large windows (17-25) = more global, smoother, approaches regular DS."
        ),
        'param_tsr_samples': (
            "Number of representative samples selected for Transfer by Sample Regression (TSR). "
            "Fewer samples (8-12) = faster, simpler model. "
            "More samples (20-30) = more comprehensive, better coverage. "
            "Rule of thumb: 10-20% of total transfer samples, minimum 10."
        ),
        'param_jypls_samples': (
            "Number of representative samples selected for JYPLS-inv calibration transfer. "
            "Uses Kennard-Stone algorithm to select diverse subset. "
            "Typical range: 10-30 samples (default: 12). Should be at least 2-3x the number of PLS components."
        ),
        'param_jypls_components': (
            "Number of PLS latent variables for JYPLS-inv transfer model. "
            "'Auto' = automatic optimization via cross-validation (recommended). "
            "3-5 = simple spectral differences. "
            "8-12 = moderate complexity. "
            "15-20 = complex instrumental differences."
        ),
        'param_nspfce_max_iter': (
            "Maximum iterations for NS-PFCE optimization algorithm. "
            "50-100 = fast, usually sufficient for simple cases (default: 100). "
            "200-500 = thorough optimization for complex instrumental differences."
        ),
        'param_nspfce_wavelength_selection': (
            "Enable wavelength selection for NS-PFCE to identify informative features. "
            "When checked: Uses feature selection algorithm to find wavelengths that transfer reliably. "
            "Recommended: CHECKED for most applications."
        ),
        'param_nspfce_selector': (
            "Wavelength selection algorithm for NS-PFCE method. "
            "vcpa-iriv = Most comprehensive, identifies stable informative wavelengths (recommended). "
            "cars = Fast, uses competitive mechanism. Good for large datasets. "
            "spa = Very fast, projects orthogonal variables. Good for highly collinear data."
        ),
    },

    # ===== V3-SPECIFIC: SEARCH MODES =====
    'search_modes': {
        'grid_search': (
            "Grid Search exhaustively tests all combinations of hyperparameters in the search grid. "
            "Guarantees finding the best combination within the grid but can be slow with many parameters. "
            "Best for: Smaller search spaces, when you want deterministic results."
        ),
        'bayesian': (
            "Bayesian Optimization (Optuna) uses probabilistic modeling to intelligently explore "
            "hyperparameter space. Learns from previous trials to focus on promising regions. "
            "More efficient than grid search for large search spaces. "
            "Best for: Large search spaces, limited computational budget."
        ),
        'nsga2': (
            "NSGA-II (Non-dominated Sorting Genetic Algorithm II) is a multi-objective optimization "
            "algorithm that finds Pareto-optimal solutions balancing multiple objectives: "
            "prediction error, number of wavelengths, and model complexity. "
            "Returns a set of trade-off solutions rather than a single best. "
            "Best for: Finding interpretable models, variable selection, understanding trade-offs."
        ),
    },

    # ===== V3-SPECIFIC: MODEL TIERS =====
    'model_tiers': {
        'quick': (
            "Quick tier: PLS only. Fastest option for initial exploration. "
            "Good for getting a baseline model quickly."
        ),
        'standard': (
            "Standard tier: PLS, Ridge, ElasticNet, RandomForest, LightGBM. "
            "Balanced coverage of linear and tree-based models. Recommended for most use cases."
        ),
        'comprehensive': (
            "Comprehensive tier: All available models. Most thorough search but slowest. "
            "Use when you have time and want to ensure the best model is found."
        ),
        'custom': (
            "Custom tier: Select specific models to include in the search. "
            "Use when you know which model types work best for your data."
        ),
    },

    # ===== V3-SPECIFIC: VARIABLE SELECTION =====
    'variable_selection': {
        'feature_importance': (
            "Uses Random Forest feature importance scores to rank and select wavelengths. "
            "Fast and effective for identifying important spectral regions."
        ),
        'vip': (
            "Variable Importance in Projection (VIP) scores from PLS. "
            "Measures each wavelength's contribution to the PLS model. "
            "VIP > 1 indicates important variables. Standard method in chemometrics."
        ),
        'spa': (
            "Successive Projections Algorithm (SPA) selects wavelengths with minimum collinearity. "
            "Projects out selected variables to find orthogonal features. "
            "Produces compact, uncorrelated variable sets."
        ),
        'uve': (
            "Uninformative Variable Elimination (UVE) removes variables with low stability. "
            "Uses noise variables as reference to identify reliable wavelengths. "
            "Good for eliminating clearly uninformative spectral regions."
        ),
        'uve_spa': (
            "Hybrid UVE-SPA: First applies UVE to remove uninformative variables, "
            "then uses SPA to select orthogonal subset from remaining variables. "
            "Combines strengths of both methods."
        ),
        'ga_pls': (
            "Genetic Algorithm for PLS variable selection. "
            "Evolves populations of variable subsets to optimize PLS performance. "
            "Can find complex variable combinations that other methods miss. "
            "Multiple runs with consensus voting improves robustness."
        ),
        'ipls': (
            "Interval PLS (iPLS) analyzes spectral regions systematically. "
            "Divides spectrum into intervals and evaluates each region's predictive power. "
            "Forward iPLS adds best intervals iteratively. "
            "Backward iPLS removes worst intervals iteratively. "
            "Good for understanding which spectral regions contain information."
        ),
        'region_analysis': (
            "Analyzes selected wavelengths to identify spectral regions. "
            "Groups adjacent selected variables and ranks regions by importance. "
            "Helps interpret variable selection results in terms of spectral features."
        ),
    },

    # ===== V3-SPECIFIC: ADVANCED PREPROCESSING =====
    'advanced_preprocessing': {
        'msc': (
            "Multiplicative Scatter Correction (MSC) removes scatter effects using a reference spectrum. "
            "Each spectrum is regressed against the reference, then corrected using the slope and intercept. "
            "More physically meaningful than SNV for some applications. "
            "Reference can be mean or median of training spectra."
        ),
        'osc': (
            "Orthogonal Signal Correction (OSC) removes variation in X that is orthogonal to Y. "
            "Identifies and removes spectral variance not related to the target variable. "
            "Can improve model performance but may remove useful information if overused. "
            "Number of components controls how much variance is removed."
        ),
        'epo': (
            "External Parameter Orthogonalization (EPO) removes known interferent effects. "
            "Requires an interferent library (spectra of interferents at different levels). "
            "Projects out the interferent subspace from calibration and prediction data. "
            "Useful when specific interferences are known (e.g., temperature, moisture)."
        ),
        'glsw': (
            "Generalized Least Squares Weighting (GLSW) down-weights noisy or uninformative wavelengths. "
            "Estimates wavelength-specific noise and applies inverse weighting. "
            "Covariance method uses spectral covariance. Residual method uses model residuals. "
            "Improves model performance when noise varies across wavelengths."
        ),
        'baseline_none': "No baseline correction applied.",
        'baseline_poly': (
            "Polynomial baseline correction fits and subtracts a polynomial baseline. "
            "Degree controls polynomial complexity (1=linear, 2=quadratic, etc.). "
            "Segments allow piecewise fitting for complex baselines. "
            "Good for Raman spectroscopy and fluorescence removal."
        ),
        'baseline_asls': (
            "Asymmetric Least Squares (AsLS) baseline correction. "
            "Iteratively fits a smooth baseline using asymmetric weighting. "
            "Smoothness parameter controls baseline flexibility. "
            "Asymmetry parameter controls how peaks are handled. "
            "Effective for spectra with broad baseline drift."
        ),
        'baseline_airpls': (
            "Adaptive Iteratively Reweighted Penalized Least Squares (airPLS). "
            "Automatically adapts weights based on signal intensity. "
            "More robust than AsLS for spectra with varying peak heights. "
            "Smoothness parameter controls baseline flexibility."
        ),
        'smoothing_sg': (
            "Savitzky-Golay smoothing reduces noise while preserving peak shapes. "
            "Window length must be odd. Larger window = more smoothing. "
            "Polynomial order controls how well peaks are preserved. "
            "Applied after preprocessing, before model building."
        ),
    },

    # ===== V3-SPECIFIC: GA PREPROCESSING =====
    'ga_preprocessing': {
        'enable': (
            "Uses Genetic Algorithm to find optimal preprocessing combination. "
            "Searches across preprocessing methods, derivative orders, and parameters. "
            "Can discover preprocessing pipelines you wouldn't try manually."
        ),
        'population': (
            "Number of candidate preprocessing pipelines in each generation. "
            "Larger population = more diversity but slower. "
            "16-32 for quick search, 48-64 for thorough search."
        ),
        'generations': (
            "Number of evolutionary iterations. "
            "More generations = better optimization but longer runtime. "
            "20-50 for quick search, 80-100 for thorough search."
        ),
        'cv_folds': "Cross-validation folds for evaluating each preprocessing pipeline.",
    },

    # ===== V3-SPECIFIC: NSGA-II PARAMETERS =====
    'nsga2': {
        'population_size': (
            "Number of solutions in each generation. "
            "Larger population = more diversity in Pareto front. "
            "20-50 for quick search, 80-100 for thorough exploration."
        ),
        'generations': (
            "Number of evolutionary generations. "
            "More generations = better convergence to Pareto front. "
            "30-100 for quick search, 200-300 for thorough optimization."
        ),
        'min_wavelengths': (
            "Minimum number of wavelengths in any solution. "
            "Prevents degenerate solutions with too few variables. "
            "5-10 is usually sufficient for spectral data."
        ),
    },

    # ===== V3-SPECIFIC: DATA MANAGEMENT =====
    'data_management': {
        'merge_intersection': (
            "Uses only wavelengths common to all data sources. "
            "Safest option - no interpolation needed. "
            "May lose wavelength range if sources have different coverage."
        ),
        'merge_union': (
            "Includes all wavelengths from all sources. "
            "Missing values filled with NaN or interpolated. "
            "Preserves full spectral range from each source."
        ),
        'merge_interpolation': (
            "Interpolates all sources to a common wavelength grid. "
            "Requires specifying the target wavelength step. "
            "Most flexible but may introduce interpolation artifacts."
        ),
        'duplicate_error': "Raise error if duplicate sample IDs are found.",
        'duplicate_keep_first': "Keep first occurrence of duplicate sample IDs.",
        'duplicate_keep_last': "Keep last occurrence of duplicate sample IDs.",
        'duplicate_rename': "Rename duplicates by appending source name or number.",
        'interp_step': (
            "Wavelength interval for interpolated grid. "
            "Smaller step = higher resolution but larger data. "
            "Match to your instrument's native resolution."
        ),
    },

    # ===== V3-SPECIFIC: OUTLIER DETECTION =====
    'outlier_detection': {
        't2': (
            "Hotelling T-squared statistic measures distance from PCA center in score space. "
            "High T2 indicates samples far from the model center. "
            "Uses chi-square distribution for threshold calculation."
        ),
        'q_residuals': (
            "Q-Residuals (SPE) measure reconstruction error from PCA model. "
            "High Q indicates samples not well explained by the model. "
            "Sensitive to unusual spectral features not in training data."
        ),
        'mahalanobis': (
            "Mahalanobis distance accounts for correlations between variables. "
            "More sensitive than Euclidean distance for correlated spectral data."
        ),
        'y_consistency': (
            "Y-consistency check identifies samples with unusual target values. "
            "Uses Z-score of prediction residuals to flag potential target outliers."
        ),
        'confidence_level': (
            "Confidence level for outlier threshold. "
            "95% = flags ~5% of normal data as outliers. "
            "99% = more conservative, flags only extreme outliers. "
            "99.9% = very conservative, only flags severe outliers."
        ),
        'n_components': (
            "Number of PCA components for T2 and Q calculation. "
            "More components capture more variance but may include noise. "
            "Typically 3-10 components depending on spectral complexity."
        ),
    },

    # ===== V3-SPECIFIC: VALIDATION =====
    'validation': {
        'holdout': (
            "Reserve a portion of data for independent validation. "
            "Not used during model training or cross-validation. "
            "Provides unbiased estimate of model performance."
        ),
        'holdout_percent': (
            "Percentage of samples to hold out for validation. "
            "10-25% typical. Balance between training data size and validation reliability."
        ),
        'spxy': (
            "SPXY (Sample set Partitioning based on joint X-Y distances) ensures coverage "
            "of both spectral (X) and target (Y) space in the selected subset."
        ),
        'duplex': (
            "DUPLEX alternates between calibration and validation sets when selecting samples. "
            "Ensures both sets have similar coverage of the sample space."
        ),
        'kennard_stone': (
            "Kennard-Stone algorithm selects samples covering maximum spectral diversity. "
            "Starts with the two most distant samples, then iteratively adds the sample "
            "most distant from those already selected."
        ),
        'random': "Random selection of validation samples. Simple but may miss extremes.",
    },

    # ===== V3-SPECIFIC: EXPORT =====
    'export': {
        'format_python': "Export as Python script (.py) for command-line execution.",
        'format_notebook': "Export as Jupyter Notebook (.ipynb) for interactive use.",
        'include_data_loading': "Include example code for loading spectral data files.",
        'include_preprocessing': "Include preprocessing function definitions.",
        'include_variable_selection': "Include variable selection code and selected indices.",
        'include_cv': "Include cross-validation code for reproducibility.",
        'include_visualization': "Include matplotlib code for plotting results.",
        'include_prediction': "Include template for making predictions on new data.",
    },

    # ===== V3-SPECIFIC: UI CONTROLS =====
    'ui': {
        'cv_folds': (
            "Number of cross-validation folds. Higher = more robust estimate but slower. "
            "5-fold is standard. 10-fold for smaller datasets. Leave-one-out for very small datasets."
        ),
        'target_variable': "Column containing the values to predict (Y variable).",
        'task_type_regression': "Regression: Predict continuous numerical values.",
        'task_type_classification': "Classification: Predict categorical class labels.",
        'color_bins': (
            "Number of color bins for target-based coloring in plots. "
            "More bins = finer color gradation. 1 = no binning (continuous color)."
        ),
        'edit_mode': "Enable editing of data values directly in the grid.",
        'fill_down': "Copy the first selected cell's value to all selected cells.",
        'undo': "Undo the last edit operation.",
        'redo': "Redo the last undone operation.",
        'delete_row': "Remove selected rows from the dataset.",
        'duplicate_row': "Create copies of selected rows.",
        'insert_row': "Insert a new empty row.",
        'add_column': "Add a new metadata column.",
        'show_all_wavelengths': "Show all wavelength columns (may be slow for large datasets).",
    },

    # ===== V3-SPECIFIC: WAVELENGTH RANGE =====
    'wavelength_range': {
        'enable': "Restrict model training to a specific wavelength range.",
        'from_nm': "Starting wavelength in nanometers.",
        'to_nm': "Ending wavelength in nanometers.",
        'preset_full': "Use full available wavelength range.",
        'preset_uv': "UV region: 200-400 nm.",
        'preset_vis': "Visible region: 400-700 nm.",
        'preset_nir': "Near-infrared region: 700-2500 nm.",
        'preset_swir': "Short-wave infrared: 1000-2500 nm.",
        'preset_mir': "Mid-infrared region: 2500-25000 nm.",
    },
}


# =============================================================================
# TOOLTIP THEME
# =============================================================================

_tooltip_theme = None


def get_tooltip_theme() -> int:
    """
    Create and cache a tooltip theme using V3 colors.

    Returns themed popup with:
    - bg_elevated background
    - text_primary text
    - accent_secondary border
    """
    global _tooltip_theme

    if _tooltip_theme is not None and dpg.does_item_exist(_tooltip_theme):
        return _tooltip_theme

    colors = _get_colors()

    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvAll):
            # Tooltip background (elevated surface)
            bg = colors.get("bg_elevated", (33, 38, 45))
            dpg.add_theme_color(
                dpg.mvThemeCol_PopupBg,
                (*bg, 245)  # Slight transparency
            )
            # Text color
            text = colors.get("text_primary", (240, 246, 252))
            dpg.add_theme_color(
                dpg.mvThemeCol_Text,
                (*text, 255)
            )
            # Border
            border = colors.get("accent_secondary", (88, 166, 255))
            dpg.add_theme_color(
                dpg.mvThemeCol_Border,
                (*border, 180)
            )
            # Rounded corners
            dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 12, 8)

    _tooltip_theme = theme
    return _tooltip_theme


def reset_tooltip_theme():
    """Reset the tooltip theme cache. Call after theme changes."""
    global _tooltip_theme
    _tooltip_theme = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def add_tooltip(parent_tag: str, text: str, wrap_width: int = 400) -> Optional[int]:
    """
    Add a themed tooltip to a widget.

    Parameters
    ----------
    parent_tag : str
        Tag of the widget to attach tooltip to
    text : str
        Tooltip text content
    wrap_width : int
        Text wrap width in pixels (default 400)

    Returns
    -------
    int or None
        Tooltip ID if created, None if parent doesn't exist

    Example
    -------
    >>> dpg.add_button(label="Run", tag="run_btn")
    >>> add_tooltip("run_btn", "Click to start the model search")
    """
    if not dpg.does_item_exist(parent_tag):
        return None

    with dpg.tooltip(parent=parent_tag) as tooltip_id:
        dpg.add_text(text, wrap=wrap_width)

    # Apply themed styling
    dpg.bind_item_theme(tooltip_id, get_tooltip_theme())

    return tooltip_id


def add_tooltip_from_dict(
    parent_tag: str,
    category: str,
    key: str,
    wrap_width: int = 400
) -> Optional[int]:
    """
    Add a tooltip from the TOOLTIP_CONTENT dictionary.

    Parameters
    ----------
    parent_tag : str
        Tag of the widget to attach tooltip to
    category : str
        Category in TOOLTIP_CONTENT (e.g., 'models', 'hyperparameters')
    key : str
        Key within the category (e.g., 'PLS', 'ridge_alpha')
    wrap_width : int
        Text wrap width in pixels

    Returns
    -------
    int or None
        Tooltip ID if created, None if key not found or parent doesn't exist

    Example
    -------
    >>> dpg.add_combo(items=["PLS", "Ridge"], tag="model_combo")
    >>> add_tooltip_from_dict("model_combo", "models", "PLS")
    """
    content = TOOLTIP_CONTENT.get(category, {}).get(key)
    if content is None:
        return None
    return add_tooltip(parent_tag, content, wrap_width)
