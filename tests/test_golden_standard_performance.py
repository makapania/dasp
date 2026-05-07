"""Golden standard test: exact result reproduction before/after optimization.

Run BEFORE any performance changes to capture baseline R²/RMSE values.
After each optimization change, rerun to verify identical results.

Uses real example data (BoneCollagen ASD spectra + CSV targets).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixture: load example BoneCollagen data (ASD spectra + CSV targets)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def example_data():
    """Load ASD spectra and align with BoneCollagen CSV targets."""
    from spectral_predict.io import read_asd_dir

    example_dir = Path(__file__).parent.parent / "example"
    csv_path = example_dir / "BoneCollagen.csv"

    if not csv_path.exists():
        pytest.skip("BoneCollagen.csv not found in example/")

    # Load ASD spectra (rows=filename stems, columns=wavelengths)
    X, _ = read_asd_dir(str(example_dir))

    # Load reference CSV
    ref = pd.read_csv(csv_path, encoding="utf-8-sig")

    # Align: CSV "File Number" like "Spectrum 00001" -> ASD stem "Spectrum00001"
    ref["_stem"] = ref["File Number"].str.replace(" ", "")
    ref = ref.set_index("_stem")

    # Keep only rows present in both
    common = X.index.intersection(ref.index)
    if len(common) < 10:
        pytest.skip(f"Only {len(common)} matching samples found")

    X = X.loc[common].sort_index()
    y = ref.loc[common, "%Collagen"].sort_index().astype(float)

    # Convert column names to strings (wavelengths)
    X.columns = [str(c) for c in X.columns]

    return X, y


# ---------------------------------------------------------------------------
# PLS golden standard
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_pls_golden_standard(example_data):
    """PLS with raw preprocessing, 5-fold CV, quick tier.

    Captures exact R²/RMSE for regression testing after optimization changes.
    """
    from spectral_predict.search import run_search

    X, y = example_data

    results_df, _ = run_search(
        X, y,
        task_type="regression",
        folds=5,
        models_to_test=["PLS"],
        preprocessing_methods={"raw": True},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        tier="quick",
    )

    assert len(results_df) > 0, "PLS should produce results"
    best = results_df.iloc[0]

    assert not np.isnan(best["R2"]), "R² should not be NaN"
    assert best["RMSE"] > 0, "RMSE should be positive"

    # Print values for initial capture
    print(f"\n  PLS Golden: R2={best['R2']:.15f}, RMSE={best['RMSE']:.15f}")
    print(f"  PLS Model: {best['Model']}, Preprocess: {best['Preprocess']}")

    # ---- GOLDEN VALUES (captured 2026-03-22 before optimization changes) ----
    GOLDEN_R2 = 0.958248707573222
    GOLDEN_RMSE = 1.396669152643774
    np.testing.assert_allclose(best["R2"], GOLDEN_R2, rtol=1e-6,
        err_msg="PLS R² changed after optimization!")
    np.testing.assert_allclose(best["RMSE"], GOLDEN_RMSE, rtol=1e-6,
        err_msg="PLS RMSE changed after optimization!")


# ---------------------------------------------------------------------------
# LightGBM golden standard
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_lightgbm_golden_standard(example_data):
    """LightGBM with raw preprocessing, 5-fold CV, standard tier.

    Captures exact R²/RMSE for regression testing after optimization changes.
    LightGBM requires standard tier (not available in quick).
    """
    from spectral_predict.search import run_search

    X, y = example_data

    results_df, _ = run_search(
        X, y,
        task_type="regression",
        folds=5,
        models_to_test=["LightGBM"],
        preprocessing_methods={"raw": True},
        enable_variable_subsets=False,
        enable_region_subsets=False,
        tier="standard",
    )

    assert len(results_df) > 0, "LightGBM should produce results"
    best = results_df.iloc[0]

    assert not np.isnan(best["R2"]), "R² should not be NaN"
    assert best["RMSE"] > 0, "RMSE should be positive"

    # Print values for initial capture
    print(f"\n  LightGBM Golden: R2={best['R2']:.15f}, RMSE={best['RMSE']:.15f}")
    print(f"  LightGBM Model: {best['Model']}, Preprocess: {best['Preprocess']}")

    # ---- GOLDEN VALUES (captured 2026-03-22 before optimization changes) ----
    GOLDEN_R2 = 0.998707473227969
    GOLDEN_RMSE = 0.245741414101907
    np.testing.assert_allclose(best["R2"], GOLDEN_R2, rtol=1e-6,
        err_msg="LightGBM R² changed after optimization!")
    np.testing.assert_allclose(best["RMSE"], GOLDEN_RMSE, rtol=1e-6,
        err_msg="LightGBM RMSE changed after optimization!")


# ---------------------------------------------------------------------------
# Variable selection caching verification
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_variable_selection_spa_correctness(example_data):
    """Run PLS + LightGBM with SPA variable selection.

    SPA is model-independent (uses its own internal PLS), so results for
    both models should use the same selected wavelengths. This test verifies
    that caching won't break correctness.
    """
    from spectral_predict.search import run_search

    X, y = example_data

    results_df, _ = run_search(
        X, y,
        task_type="regression",
        folds=3,
        models_to_test=["PLS", "LightGBM"],
        preprocessing_methods={"raw": True},
        enable_variable_subsets=True,
        variable_counts=[20],
        variable_selection_methods=["spa"],
        enable_region_subsets=False,
        tier="standard",  # LightGBM requires standard tier
    )

    assert len(results_df) > 0, "Should produce results with SPA variable selection"

    # Print all results for inspection
    print(f"\n  Total results: {len(results_df)}")
    for i, row in results_df.iterrows():
        tag = row.get("SubsetTag", "full")
        print(f"  [{i}] {row['Model']} | {row.get('Preprocess', 'raw')} | "
              f"R2={row['R2']:.6f} | RMSE={row['RMSE']:.6f} | Tag={tag}")

    # Both PLS and LightGBM should produce SPA subset results
    assert "SubsetTag" in results_df.columns, "Results should have SubsetTag column"
    spa_results = results_df[results_df["SubsetTag"].str.contains("spa", na=False)]
    pls_spa = spa_results[spa_results["Model"] == "PLS"]
    lgbm_spa = spa_results[spa_results["Model"] == "LightGBM"]
    assert len(pls_spa) > 0, "PLS should have SPA subset results"
    assert len(lgbm_spa) > 0, "LightGBM should have SPA subset results"

    # SPA is model-independent: both models should use the same subset tag
    pls_tags = set(pls_spa["SubsetTag"].values)
    lgbm_tags = set(lgbm_spa["SubsetTag"].values)
    assert pls_tags == lgbm_tags, (
        f"SPA subsets should be identical for PLS and LightGBM: {pls_tags} vs {lgbm_tags}"
    )

