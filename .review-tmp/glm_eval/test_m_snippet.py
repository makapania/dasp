from spectral_predict.scoring import score_model_results
from spectral_predict.contamination import run_one_class_cv


def evaluate_top_n_one_class(
    results_df,
    X_val,
    y_val_oc,
    n: int = 10,
):
    """Re-score the top N one-class models on an external validation set.

    Args:
        results_df: DataFrame from a one-class search, sorted by cv_score.
        X_val: Validation spectra (n_samples, n_wavelengths).
        y_val_oc: Validation labels in {-1, +1}.
        n: How many top models to evaluate.

    Returns:
        DataFrame with the top N rows plus added 'val_score' column,
        sorted by val_score descending.
    """
    top = results_df.head(n).copy()
    val_scores = []
    for _, row in top.iterrows():
        cv_result = run_one_class_cv(
            X=X_val,
            y_oc=y_val_oc,
            model_name=row['model_name'],
            params=row['params'],
            n_folds=5,
        )
        val_scores.append(score_model_results(cv_result))
    top['val_score'] = val_scores
    return top.sort_values('val_score', ascending=False)
