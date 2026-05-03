"""Comprehensive tests for spectral_predict.report module.

Tests cover:
- write_markdown_report: regression result reports
- write_markdown_report: classification result reports
- Empty/minimal result handling
- File creation and content verification
- Markdown structure validation (headers, tables)
"""

import numpy as np
import pandas as pd
import pytest

from spectral_predict.report import write_markdown_report


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def regression_results():
    """Synthetic ranked regression results DataFrame mimicking search.py output."""
    n_models = 8
    return pd.DataFrame({
        "Rank": list(range(1, n_models + 1)),
        "Model": ["PLS", "Ridge", "Lasso", "ElasticNet", "PLS", "Ridge", "RandomForest", "LightGBM"],
        "Preprocess": ["snv", "raw", "snv_deriv1", "deriv2", "raw", "snv", "deriv1", "raw"],
        "SubsetTag": ["full", "full", "SPA_50", "CARS_100", "full", "UVE_200", "full", "GA_150"],
        "LVs": [5, np.nan, np.nan, np.nan, 8, np.nan, np.nan, np.nan],
        "n_vars": [200, 200, 50, 100, 200, 200, 200, 150],
        "full_vars": [200] * n_models,
        "RMSE": [0.1234, 0.1456, 0.1567, 0.1678, 0.1789, 0.1890, 0.1950, 0.2010],
        "R2": [0.9432, 0.9321, 0.9210, 0.9100, 0.8990, 0.8880, 0.8770, 0.8660],
        "CompositeScore": [0.95, 0.93, 0.91, 0.89, 0.87, 0.85, 0.83, 0.81],
        "Deriv": [np.nan, np.nan, 1.0, 2.0, np.nan, np.nan, 1.0, np.nan],
        "Window": [np.nan, np.nan, 11.0, 17.0, np.nan, np.nan, 11.0, np.nan],
        "Poly": [np.nan, np.nan, 2.0, 3.0, np.nan, np.nan, 2.0, np.nan],
        "Task": ["regression"] * n_models,
    })


@pytest.fixture
def classification_results():
    """Synthetic ranked classification results DataFrame."""
    n_models = 5
    return pd.DataFrame({
        "Rank": list(range(1, n_models + 1)),
        "Model": ["PLS-DA", "RandomForest", "LightGBM", "SVM", "Ridge"],
        "Preprocess": ["snv", "raw", "deriv1", "snv_deriv2", "raw"],
        "SubsetTag": ["full", "CARS_100", "full", "SPA_50", "full"],
        "LVs": [5, np.nan, np.nan, np.nan, np.nan],
        "n_vars": [200, 100, 200, 50, 200],
        "full_vars": [200] * n_models,
        "Accuracy": [0.9600, 0.9400, 0.9200, 0.9000, 0.8800],
        "ROC_AUC": [0.9800, 0.9600, 0.9400, 0.9200, np.nan],
        "CompositeScore": [0.97, 0.95, 0.93, 0.91, 0.88],
        "Deriv": [np.nan, np.nan, 1.0, 2.0, np.nan],
        "Window": [np.nan, np.nan, 11.0, 17.0, np.nan],
        "Poly": [np.nan, np.nan, 2.0, 3.0, np.nan],
        "Task": ["classification"] * n_models,
    })


@pytest.fixture
def empty_results():
    """Empty DataFrame with expected columns."""
    return pd.DataFrame(columns=[
        "Rank", "Model", "Preprocess", "SubsetTag", "LVs",
        "n_vars", "full_vars", "RMSE", "R2", "CompositeScore",
        "Deriv", "Window", "Poly", "Task"
    ])


# =============================================================================
# Regression report tests
# =============================================================================


def test_regression_report_file_created(tmp_path, regression_results):
    """Report file should be created at the expected path."""
    report_path = write_markdown_report("Collagen", regression_results, tmp_path)

    assert report_path.exists()
    assert report_path.name == "Collagen.md"


def test_regression_report_contains_title(tmp_path, regression_results):
    """Report should contain the target name in the title."""
    report_path = write_markdown_report("Collagen", regression_results, tmp_path)
    content = report_path.read_text()

    assert "# Spectral Predict Report: Collagen" in content


def test_regression_report_contains_task_type(tmp_path, regression_results):
    """Report should contain the task type."""
    report_path = write_markdown_report("Collagen", regression_results, tmp_path)
    content = report_path.read_text()

    assert "Regression" in content


def test_regression_report_contains_top5(tmp_path, regression_results):
    """Report should contain sections for the top 5 models."""
    report_path = write_markdown_report("Collagen", regression_results, tmp_path)
    content = report_path.read_text()

    # Should have Rank 1 through Rank 5
    for rank in range(1, 6):
        assert f"Rank {rank}" in content

    # Should NOT have Rank 6+ in headers
    assert "### Rank 6" not in content


def test_regression_report_contains_rmse_r2(tmp_path, regression_results):
    """Report should contain RMSE and R-squared metrics."""
    report_path = write_markdown_report("Collagen", regression_results, tmp_path)
    content = report_path.read_text()

    assert "RMSE" in content
    assert "R\u00b2" in content or "R2" in content


def test_regression_report_contains_summary_table(tmp_path, regression_results):
    """Report should contain a summary table section."""
    report_path = write_markdown_report("Collagen", regression_results, tmp_path)
    content = report_path.read_text()

    assert "## Summary Table" in content
    # Markdown tables use | as delimiters
    assert "|" in content


def test_regression_report_total_models_count(tmp_path, regression_results):
    """Report should show total number of models evaluated."""
    report_path = write_markdown_report("Collagen", regression_results, tmp_path)
    content = report_path.read_text()

    assert "8" in content  # 8 models in fixture


# =============================================================================
# Classification report tests
# =============================================================================


def test_classification_report_contains_accuracy(tmp_path, classification_results):
    """Classification report should contain accuracy metrics."""
    report_path = write_markdown_report("CollagenCat", classification_results, tmp_path)
    content = report_path.read_text()

    assert "Accuracy" in content


def test_classification_report_contains_roc_auc(tmp_path, classification_results):
    """Classification report should contain ROC AUC where available."""
    report_path = write_markdown_report("CollagenCat", classification_results, tmp_path)
    content = report_path.read_text()

    assert "ROC AUC" in content


# =============================================================================
# Edge cases
# =============================================================================


def test_empty_results_report(tmp_path, empty_results):
    """Empty results should produce a report indicating no models."""
    report_path = write_markdown_report("NoData", empty_results, tmp_path)
    content = report_path.read_text()

    assert report_path.exists()
    assert "No models completed successfully" in content


def test_report_creates_output_directory(tmp_path, regression_results):
    """Report should create the output directory if it does not exist."""
    nested_dir = tmp_path / "subdir" / "reports"
    report_path = write_markdown_report("Test", regression_results, nested_dir)

    assert report_path.exists()
    assert nested_dir.exists()


def test_report_footer_present(tmp_path, regression_results):
    """Report should contain footer with version info."""
    report_path = write_markdown_report("Collagen", regression_results, tmp_path)
    content = report_path.read_text()

    assert "Generated by Spectral Predict" in content


def test_t14_report_version_matches_package_version(tmp_path, regression_results):
    """T-14: the report footer must reflect the actual package __version__,
    not a hardcoded stale value (the prior bug was a hardcoded 'v0.4.0' that
    drifted away from the real 0.5.0b1)."""
    from spectral_predict import __version__

    report_path = write_markdown_report("Collagen", regression_results, tmp_path)
    content = report_path.read_text()

    assert f"v{__version__}" in content, (
        f"Report footer must contain v{__version__}; got footer text: "
        f"{[line for line in content.splitlines() if 'Generated' in line]!r}"
    )

    # T-14 fix-of-fixes (DeepSeek MEDIUM-2): also assert that known stale
    # version strings are NOT present. Without this, a future dev could
    # re-hardcode the *current* canonical value (v0.5.0b1) and the
    # positive assertion above would still pass — drift would only show
    # up after the next bump. Pinning known-stale values fails fast.
    for stale in ("v0.4.0", "v0.3.0", "v0.2.0", "v0.5.0b1"):
        assert stale not in content, (
            f"Report footer contains stale version string {stale!r}; "
            "version drift has regressed"
        )


def test_t14_empty_results_report_includes_version_footer(tmp_path, empty_results):
    """T-14 fix-of-fixes (GLM MEDIUM-1): the empty-results early-return
    path was missing the footer entirely. Fix added a footer using the
    same __version__ source so an empty-results report doesn't drift back
    into "Generated by ?" indeterminacy."""
    from spectral_predict import __version__

    report_path = write_markdown_report("NoData", empty_results, tmp_path)
    content = report_path.read_text()

    assert f"v{__version__}" in content
    assert "Generated by Spectral Predict" in content


def test_report_with_score_column(tmp_path, regression_results):
    """Report should handle both 'Score' and 'CompositeScore' column names."""
    # Rename CompositeScore to Score (Bayesian search format)
    df = regression_results.rename(columns={"CompositeScore": "Score"})
    report_path = write_markdown_report("Test", df, tmp_path)
    content = report_path.read_text()

    assert "Composite Score" in content
    assert report_path.exists()
