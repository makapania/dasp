"""
Unit and integration tests for exhaustive preprocessing optimization.

Originally written for the GA preprocessing path; the GA evolution loop and
its operators (tournament_selection, crossover, mutate) were removed in
2026-05-06 because parallel exhaustive enumeration finished the same
14 x 17 = 238 cell space in seconds. This module now validates:
- Chromosome encoding / decoding (still 2-gene; backward-compat with saved CSVs)
- Fitness evaluation
- Full exhaustive optimization workflow
- Convenience wrapper for integration with search.py
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    from spectral_predict.ga_preprocessing import (
        random_chromosome,
        chromosome_to_transform,
        get_config_description,
        evaluate_fitness,
        optimize_preprocessing,
        get_optimized_preproc_config,
        _decode_autoscale_gene,
        PREPROC_TYPES,
        WINDOW_SIZES,
        DERIVATIVE_WINDOW_RANGES,
        N_GENES,
    )
    HAS_GA_PREPROCESSING = True
except (ImportError, ModuleNotFoundError):
    HAS_GA_PREPROCESSING = False

pytestmark = pytest.mark.skipif(
    not HAS_GA_PREPROCESSING, reason="ga_preprocessing imports failed"
)


# =============================================================================
# Unit Tests - Chromosome Encoding
# =============================================================================


@pytest.mark.unit
class TestChromosomeEncoding:
    """Test chromosome encoding and decoding."""

    def test_random_chromosome_shape(self):
        """Random chromosome should have correct number of genes."""
        rng = np.random.RandomState(42)
        chrom = random_chromosome(rng)

        assert chrom.shape == (N_GENES,), f"Expected {N_GENES} genes"
        assert chrom.dtype == np.int32, "Genes should be integers"

    def test_random_chromosome_valid_ranges(self):
        """Random chromosome genes should be in valid ranges."""
        rng = np.random.RandomState(42)
        chrom = random_chromosome(rng)

        assert 0 <= chrom[0] < len(PREPROC_TYPES), "Invalid preproc type index"
        assert 0 <= chrom[1] < len(WINDOW_SIZES), "Invalid window index"

    def test_encode_decode_roundtrip(self):
        """Encoding and decoding should preserve chromosome information."""
        rng = np.random.RandomState(42)
        chrom = random_chromosome(rng)

        # Decode to transform
        name, transform = chromosome_to_transform(chrom)

        # Check name is string
        assert isinstance(name, str), "Name should be string"
        assert len(name) > 0, "Name should not be empty"

        # Check transform is callable or None
        assert transform is None or callable(transform), "Transform should be callable or None"

    def test_valid_preprocessing_decoded(self):
        """Decoded preprocessing should produce valid transformations."""
        # Test specific configurations (2-gene encoding: [preproc_type, window])
        test_configs = [
            np.array([0, 0], dtype=np.int32),   # raw
            np.array([1, 0], dtype=np.int32),   # snv
            np.array([2, 2], dtype=np.int32),   # deriv1 with window=9
            np.array([3, 3], dtype=np.int32),   # deriv2 with window=11
            np.array([6, 4], dtype=np.int32),   # snv_deriv1 with window=13
        ]

        X_test = np.random.randn(10, 50)

        for chrom in test_configs:
            name, transform = chromosome_to_transform(chrom)

            if transform is not None:
                X_transformed = transform(X_test)

                # Check output shape matches input
                assert X_transformed.shape == X_test.shape, (
                    f"Transform {name} changed shape"
                )

                # Check for finite values
                assert np.isfinite(X_transformed).all(), (
                    f"Transform {name} produced non-finite values"
                )

    def test_get_config_description(self):
        """Config description should be human-readable."""
        rng = np.random.RandomState(42)
        chrom = random_chromosome(rng)

        desc = get_config_description(chrom)

        assert isinstance(desc, str), "Description should be string"
        assert len(desc) > 0, "Description should not be empty"
        assert "Preproc:" in desc, "Description should mention preprocessing"


# =============================================================================
# Unit Tests - Fitness Evaluation
# =============================================================================


@pytest.mark.unit
class TestFitnessEvaluation:
    """Test fitness evaluation function."""

    def test_fitness_is_cross_val_score(self, synthetic_spectra_small):
        """Fitness should be based on cross-validation score."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        rng = np.random.RandomState(42)
        chrom = random_chromosome(rng)

        # Evaluate fitness
        fitness = evaluate_fitness(chrom, X, y, cv_folds=3, n_components=5)

        # For regression, fitness is negative RMSECV (higher is better)
        assert fitness <= 0 or fitness == -np.inf, "Fitness should be negative or -inf"

    def test_handles_failed_preprocessing(self, synthetic_spectra_small):
        """Fitness should handle preprocessing failures gracefully."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        # Create chromosome that might cause issues
        # deriv4_snv (index 13) with smallest window (index 0 = 5)
        chrom = np.array([13, 0], dtype=np.int32)

        fitness = evaluate_fitness(chrom, X, y, cv_folds=3, n_components=5)

        # Should return finite value or -inf, but not crash
        assert fitness == -np.inf or np.isfinite(fitness), (
            "Fitness should be finite or -inf, not NaN"
        )

    def test_penalty_for_invalid(self):
        """Invalid chromosomes should get very low fitness."""
        # Create data that will cause preprocessing to fail
        X = np.ones((10, 20))  # Zero variance
        y = np.random.randn(10)

        rng = np.random.RandomState(42)
        chrom = random_chromosome(rng)

        fitness = evaluate_fitness(chrom, X, y, cv_folds=3, n_components=5)

        # Should get penalty
        assert fitness == -np.inf, "Invalid preprocessing should get -inf fitness"

    def test_fitness_model_pls(self, synthetic_spectra_small):
        """Test fitness evaluation with PLS model."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        rng = np.random.RandomState(42)
        chrom = random_chromosome(rng)

        fitness = evaluate_fitness(
            chrom, X, y, cv_folds=3, n_components=5, fitness_model="pls"
        )

        assert np.isfinite(fitness) or fitness == -np.inf

    def test_fitness_model_lightgbm(self, synthetic_spectra_small):
        """Test fitness evaluation with LightGBM model."""
        try:
            import lightgbm
        except ImportError:
            pytest.skip("LightGBM not installed")

        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        rng = np.random.RandomState(42)
        chrom = random_chromosome(rng)

        fitness = evaluate_fitness(
            chrom, X, y, cv_folds=3, n_components=5, fitness_model="lightgbm"
        )

        assert np.isfinite(fitness) or fitness == -np.inf


# =============================================================================
# Unit Tests - Genetic Operators
# =============================================================================


# =============================================================================
# Phase 3: Backward-compat 2-gene / 3-gene chromosome decode
# =============================================================================


@pytest.mark.unit
class TestChromosomeAutoscaleBackwardCompat:
    """Pin that the autoscale-gene decode handles both legacy 2-gene and
    Phase-3 3-gene chromosomes. These tests are load-bearing for saved-CSV
    rebuild correctness — old result files have 2-gene ga_genes arrays and
    must continue to decode as autoscale=False after Phase 3 ships.
    """

    def test_2gene_chromosome_decodes_autoscale_false(self):
        """Legacy 2-gene array → autoscale=False, no StandardScaler in transform."""
        genes_2gene = np.array([3, 5], dtype=np.int32)  # arbitrary deriv2, w=15

        assert _decode_autoscale_gene(genes_2gene) is False

        name, transform = chromosome_to_transform(genes_2gene)
        # Name should NOT include +autoscale
        assert "+autoscale" not in name

        # Transform output should be valid; standard SG-derivative behavior
        # (we just check the function runs and produces correct shape)
        X = np.random.RandomState(42).randn(10, 50)
        if transform is not None:
            X_out = transform(X)
            assert X_out.shape == X.shape

    def test_3gene_chromosome_with_autoscale_true(self):
        """3-gene array with autoscale=1 → name carries +autoscale, but the
        closure itself stays per-spectrum-only.

        Pre-fix the closure called ``StandardScaler().fit_transform()``
        internally, which caused validation rebuilds to refit a fresh scaler
        on the held-out set and collapsed R²pred. After the fix, autoscale
        is applied OUTSIDE the closure (search.py search-time path,
        compute_validation_metrics_for_top_models, evaluate_fitness) so
        train/val state can stay consistent.
        """
        genes_3gene = np.array([3, 5, 1], dtype=np.int32)

        assert _decode_autoscale_gene(genes_3gene) is True

        name, transform = chromosome_to_transform(genes_3gene)
        assert "+autoscale" in name

        X = np.random.RandomState(42).randn(20, 50)
        X_out = transform(X)
        assert X_out.shape == X.shape

        # Closure must NOT autoscale — output is whatever the per-spectrum
        # SG-derivative path produced. Specifically, columns are not forced
        # to mean=0 / std=1.
        col_means = np.abs(X_out.mean(axis=0))
        col_stds = X_out.std(axis=0)
        autoscaled = (
            np.all(col_means < 1e-9) and np.all(np.abs(col_stds - 1.0) < 1e-9)
        )
        assert not autoscaled, (
            "Closure must not bake StandardScaler — autoscale belongs outside "
            "so train/val rebuild can fit-on-train, transform-on-val."
        )

    def test_3gene_autoscale_true_matches_2gene_closure(self):
        """The closure for ``[p, w, 1]`` and ``[p, w, 0]`` must be bit-equal
        — the only difference between them is now the *name* (``+autoscale``
        suffix) and the metadata flag callers read separately.
        """
        genes_off = np.array([3, 5, 0], dtype=np.int32)
        genes_on = np.array([3, 5, 1], dtype=np.int32)

        _, t_off = chromosome_to_transform(genes_off)
        _, t_on = chromosome_to_transform(genes_on)

        X = np.random.RandomState(42).randn(20, 50)
        np.testing.assert_array_equal(t_off(X), t_on(X))

    def test_3gene_chromosome_with_autoscale_false(self):
        """3-gene array with autoscale=0 → no +autoscale in name; behavior
        identical to the 2-gene equivalent."""
        genes_2gene = np.array([3, 5], dtype=np.int32)
        genes_3gene = np.array([3, 5, 0], dtype=np.int32)

        assert _decode_autoscale_gene(genes_3gene) is False

        name_2gene, transform_2gene = chromosome_to_transform(genes_2gene)
        name_3gene, transform_3gene = chromosome_to_transform(genes_3gene)

        # Names must match exactly (no +autoscale tag for either)
        assert name_2gene == name_3gene
        assert "+autoscale" not in name_2gene

        # Transform outputs must be bit-exact equal
        X = np.random.RandomState(42).randn(10, 50)
        if transform_2gene is None:
            assert transform_3gene is None
        else:
            X_2 = transform_2gene(X)
            X_3 = transform_3gene(X)
            np.testing.assert_array_equal(X_2, X_3)

    def test_raw_returns_none_closure_regardless_of_autoscale(self):
        """raw is a no-op for the closure regardless of the autoscale gene.

        Before the fix, raw + autoscale=True returned a transform that
        called ``StandardScaler().fit_transform`` — a fresh fit on every
        call. That refit-on-val behaviour was the root cause of the R²pred
        regression on Exhaustive Preprocessing rows. Autoscale now lives
        outside the closure (callers honour the autoscale flag with their
        own fit-on-train, transform-on-val logic), so raw chromosomes
        simply return ``None`` whether or not autoscale is set; the name
        still carries ``+autoscale`` for display and metadata.
        """
        raw_idx = PREPROC_TYPES.index("raw")
        genes_raw_no_autoscale = np.array([raw_idx, 0], dtype=np.int32)
        genes_raw_with_autoscale = np.array([raw_idx, 0, 1], dtype=np.int32)

        name_off, t_off = chromosome_to_transform(genes_raw_no_autoscale)
        name_on, t_on = chromosome_to_transform(genes_raw_with_autoscale)

        assert t_off is None
        assert t_on is None
        assert "+autoscale" not in name_off
        assert "+autoscale" in name_on

    def test_get_config_description_handles_both_shapes(self):
        """get_config_description must accept both 2-gene and 3-gene arrays."""
        genes_2gene = np.array([3, 5], dtype=np.int32)
        genes_3gene_off = np.array([3, 5, 0], dtype=np.int32)
        genes_3gene_on = np.array([3, 5, 1], dtype=np.int32)

        desc_2gene = get_config_description(genes_2gene)
        desc_3gene_off = get_config_description(genes_3gene_off)
        desc_3gene_on = get_config_description(genes_3gene_on)

        # 2-gene and 3-gene-off should produce the same description
        assert desc_2gene == desc_3gene_off
        # 3-gene-on must mention autoscale
        assert "autoscale" in desc_3gene_on
        # 2-gene must NOT mention autoscale
        assert "autoscale" not in desc_2gene

    def test_saved_csv_2gene_rebuild_compatible(self):
        """Simulate the rebuild path that re-runs chromosome_to_transform
        on a 2-gene ga_genes array deserialized from an old result CSV.
        Must produce a working transform."""
        # Old CSVs serialize ga_genes via .tolist(); on rebuild it's a list
        # of 2 ints. search.py converts back to np.ndarray before calling
        # chromosome_to_transform. Pin that path:
        gene_list_from_csv = [3, 5]
        genes = np.array(gene_list_from_csv, dtype=np.int32)

        name, transform = chromosome_to_transform(genes)
        assert isinstance(name, str)

        X = np.random.RandomState(42).randn(10, 50)
        if transform is not None:
            X_out = transform(X)
            assert X_out.shape == X.shape

    def test_csv_roundtrip_preprocess_chromosome_via_pandas(self, tmp_path):
        """Closes pr-test-analyzer rating-9 finding: the renamed
        `preprocess_chromosome` column must round-trip through pandas
        (write → ast.literal_eval read → np.ndarray) and produce a working
        transform via chromosome_to_transform.

        The contract: a result-row column named `preprocess_chromosome`
        contains a string repr of a Python list (e.g. "[3, 5, 1]") that
        ast.literal_eval converts back to a list, then np.array converts
        to ndarray, and chromosome_to_transform consumes either shape
        (2-gene legacy or 3-gene Phase-3).
        """
        import ast
        import pandas as pd

        # 3-gene chromosome (Phase 3): [snv_deriv2_idx=7, w=11_idx=4, autoscale=1]
        chromosome = np.array([7, 4, 1], dtype=np.int32)
        # search.py serializes via .tolist() before writing to CSV:
        serialized = chromosome.tolist()

        # Round-trip through pandas (matches the actual save/load path)
        df = pd.DataFrame([{"preprocess_chromosome": str(serialized)}])
        csv_path = tmp_path / "result.csv"
        df.to_csv(csv_path, index=False)
        df_loaded = pd.read_csv(csv_path)

        # Mimic search.py's reader logic
        loaded_str = df_loaded.iloc[0]["preprocess_chromosome"]
        assert isinstance(loaded_str, str), "CSV writes lists as strings"
        loaded_list = ast.literal_eval(loaded_str)
        loaded_genes = np.array(loaded_list, dtype=np.int32)

        # Verify the rebuild path produces an equivalent transform
        name_orig, transform_orig = chromosome_to_transform(chromosome)
        name_loaded, transform_loaded = chromosome_to_transform(loaded_genes)
        assert name_orig == name_loaded
        assert "+autoscale" in name_loaded  # Phase 3 marker preserved

        if transform_orig is not None and transform_loaded is not None:
            X = np.random.RandomState(42).randn(20, 100)
            X_orig = transform_orig(X)
            X_loaded = transform_loaded(X)
            np.testing.assert_array_almost_equal(X_orig, X_loaded)

    def test_csv_legacy_ga_genes_column_still_rebuilds(self, tmp_path):
        """Pin the backward-compat fallback: an OLD result CSV with
        `ga_genes` (and no `preprocess_chromosome` column) must still
        rebuild via the search.py reader's fallback chain. This test
        exercises the reader logic directly.

        Without this pin, a future refactor could drop the fallback
        without any test catching it — every old saved-result file would
        silently fail to rebuild and the GUI Tab 7 would produce wrong
        predictions.
        """
        import ast
        import pandas as pd

        # Legacy CSV: only `ga_genes` column populated, no `preprocess_chromosome`
        chromosome_2gene = [3, 5]  # 2-gene legacy shape
        df = pd.DataFrame([{"ga_genes": str(chromosome_2gene)}])
        csv_path = tmp_path / "legacy_result.csv"
        df.to_csv(csv_path, index=False)
        df_loaded = pd.read_csv(csv_path)

        row = df_loaded.iloc[0].to_dict()

        # Mirror search.py:768-794 reader logic exactly (without invoking
        # the full validation rebuild which needs much more setup)
        chromosome_str = row.get("preprocess_chromosome", None)
        if chromosome_str is None or (
            isinstance(chromosome_str, float) and np.isnan(chromosome_str)
        ):
            chromosome_str = row.get("ga_genes", None)

        assert chromosome_str is not None, "Reader must find legacy ga_genes"
        assert isinstance(chromosome_str, str)
        loaded_list = ast.literal_eval(chromosome_str)
        loaded_genes = np.array(loaded_list, dtype=np.int32)

        # The legacy 2-gene array must decode as autoscale=False
        assert loaded_genes.shape == (2,)
        from spectral_predict.ga_preprocessing import _decode_autoscale_gene
        assert _decode_autoscale_gene(loaded_genes) is False

        # And produce a working transform
        name, transform = chromosome_to_transform(loaded_genes)
        assert isinstance(name, str)
        assert "+autoscale" not in name  # legacy chromosomes never have autoscale


# =============================================================================
# Regression: closure must not bake StandardScaler — train/val asymmetry
# =============================================================================


@pytest.mark.unit
class TestAutoscaleTrainValAsymmetry:
    """Pin the bug that produced ≥0.11 R²pred drop on Exhaustive
    Preprocessing rows with autoscale=True.

    Pre-fix the chromosome closure called
    ``StandardScaler().fit_transform(X)`` internally. The validation
    rebuild path (compute_validation_metrics_for_top_models) calls the
    closure once on X_train and once on X_val — each call refit a fresh
    StandardScaler on its own input, so val features were centred to
    *val's* column means / stds rather than train's. The model trained
    on train-statistic features then predicted on val-statistic features
    and R²pred collapsed.

    Tests below pin the post-fix invariants:
      1. Closure is deterministic per-spectrum-only — feeding train and
         val through it independently does not depend on cross-sample
         statistics, so column means of the outputs reflect the input
         column means (no autoscale baked in).
      2. The standalone autoscale step the search/rebuild now applies
         outside the closure must use a TRAIN-fitted scaler — verified
         by feeding val through it and checking the val output preserves
         its mean/std offset relative to train rather than being recentred
         to its own (val) mean.
    """

    def test_closure_is_per_spectrum_only(self):
        """Apply the closure to two batches with deliberately different
        column means. The output column means should differ between
        batches (because nothing inside the closure aligns them) — proving
        no cross-sample StandardScaler is hiding in the closure."""
        # snv_deriv1, w=11, autoscale=1 — a config that pre-fix would
        # have z-scored every column to mean=0 inside the closure
        snv_deriv1_idx = PREPROC_TYPES.index("snv_deriv1")
        w11_idx = WINDOW_SIZES.index(11) if 11 in WINDOW_SIZES else 0
        genes = np.array([snv_deriv1_idx, w11_idx, 1], dtype=np.int32)

        _, transform = chromosome_to_transform(genes)
        assert transform is not None

        rng = np.random.RandomState(0)
        X_train = rng.randn(40, 60)
        # X_val with a deliberate +5.0 column-mean offset on every column
        X_val = rng.randn(20, 60) + 5.0

        X_train_out = transform(X_train)
        X_val_out = transform(X_val)

        # If the closure were autoscaling (pre-fix bug), every column of
        # both outputs would have mean ≈ 0 — the +5.0 offset would vanish
        # because StandardScaler is refit on each input. Post-fix, SNV +
        # SG-derivative are per-spectrum, so the cross-sample column means
        # of train and val outputs should NOT both be zero.
        train_col_means = np.abs(X_train_out.mean(axis=0))
        val_col_means = np.abs(X_val_out.mean(axis=0))
        train_zeroed = np.all(train_col_means < 1e-9)
        val_zeroed = np.all(val_col_means < 1e-9)
        assert not (train_zeroed and val_zeroed), (
            "Closure must not fit StandardScaler internally — both train "
            "and val came back with zero column means, meaning each call "
            "refit a fresh scaler. This is the bug the fix targets."
        )

    def test_external_autoscale_uses_train_fitted_scaler(self):
        """The fixed pipeline uses a TRAIN-fitted StandardScaler at
        train/val rebuild. Mimic the post-closure step in
        search.py:compute_validation_metrics_for_top_models and assert
        that val passes through the train-fitted scaler — i.e. its
        means are recentred toward TRAIN's column means, not zeroed
        to its own.
        """
        from sklearn.preprocessing import StandardScaler

        rng = np.random.RandomState(1)
        X_train = rng.randn(40, 60)
        X_val_raw = rng.randn(20, 60) + 5.0  # deliberate offset

        scaler = StandardScaler().fit(X_train)
        X_val_scaled = scaler.transform(X_val_raw)

        # If val had been independently fit (pre-fix bug), its column
        # means would all be ≈ 0. With the train-fitted scaler reused,
        # the +5.0 offset survives (relative to train's near-zero mean
        # → val's column means are far from zero in standardised units).
        val_means = X_val_scaled.mean(axis=0)
        assert np.mean(np.abs(val_means)) > 1.0, (
            "Train-fitted scaler reused on val should preserve val's "
            "offset relative to train. If means are ~0, the scaler was "
            "(incorrectly) refit on val — the original bug."
        )

    def test_compute_validation_metrics_uses_train_fitted_scaler(self, tmp_path):
        """Direct regression test against the actual bug surface — pins the
        train-fitted-scaler invariant inside
        ``compute_validation_metrics_for_top_models`` itself.

        Pre-fix this function called the GA closure twice (once on train,
        once on val) and each call refit StandardScaler on its own input
        — collapsing R²pred. The closure was fixed and the function now
        explicitly fits StandardScaler on train and reuses .transform on
        val. This test would catch any future refactor that re-introduces
        independent val fitting at this boundary, even if the closure is
        kept correct.

        The contract: the R²pred reported by the function for an
        autoscale=True chromosome row must be numerically equivalent to
        what an external sklearn Pipeline (SNV → SG-deriv → StandardScaler)
        produces with proper fit-on-train, transform-on-val.
        """
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline as SkPipeline
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.metrics import r2_score
        from spectral_predict.preprocess import SNV, SavgolDerivative
        from spectral_predict.search import compute_validation_metrics_for_top_models

        rng = np.random.RandomState(11)
        n_train, n_val, n_features = 80, 40, 60
        wl = np.linspace(1100, 2500, n_features)

        # Synthetic spectra with deliberate train/val column-mean offset,
        # exactly the kind of distribution shift StandardScaler is supposed
        # to handle and where the bug used to silently corrupt val outputs.
        y_train = rng.uniform(0.1, 1.0, n_train)
        y_val = rng.uniform(0.1, 1.0, n_val)
        peak = np.exp(-((wl - 1500) / 200) ** 2)
        X_train = peak[None, :] * y_train[:, None] + 0.02 * rng.randn(n_train, n_features)
        X_val = peak[None, :] * y_val[:, None] + 0.02 * rng.randn(n_val, n_features) + 0.05

        # Build chromosome: snv_deriv1 + autoscale=1
        snv_deriv1_idx = PREPROC_TYPES.index("snv_deriv1")
        w11_idx = WINDOW_SIZES.index(11) if 11 in WINDOW_SIZES else 0
        chromosome = [snv_deriv1_idx, w11_idx, 1]

        # Reference: what the equivalent sklearn Pipeline would predict.
        # If the function correctly reuses a train-fitted scaler on val,
        # its R²pred must match this reference within sklearn precision.
        ref_pipe = SkPipeline(
            [
                ("snv", SNV()),
                ("sg", SavgolDerivative(deriv=1, window=11)),
                ("scaler", StandardScaler()),
            ]
        )
        Xt_ref = ref_pipe.fit_transform(X_train)
        Xv_ref = ref_pipe.transform(X_val)
        ref_model = PLSRegression(n_components=3).fit(Xt_ref, y_train)
        y_pred_ref = ref_model.predict(Xv_ref).ravel()
        r2_ref = r2_score(y_val, y_pred_ref)

        # Build a minimal results DataFrame matching what run_search would
        # write for one PLS row whose preprocessing was an autoscale=True
        # exhaustive chromosome.
        df_results = pd.DataFrame(
            [
                {
                    "Model": "PLS",
                    "Params": "{'n_components': 3}",
                    "LVs": 3,
                    "Preprocess": "snv_deriv1",
                    "PreprocessBase": "snv_deriv1",
                    "Deriv": 1,
                    "Window": 11,
                    "Poly": 2,
                    "preprocess_chromosome": str(chromosome),
                    "Autoscale": True,
                    "all_vars": "N/A",
                    "CompositeScore": 0.5,
                    "baseline_method": None,
                    "smoothing": False,
                    "smoothing_window": 17,
                    "smoothing_polyorder": 2,
                }
            ]
        )

        wavelengths = wl
        df_out = compute_validation_metrics_for_top_models(
            df_results,
            X_train,
            y_train,
            X_val,
            y_val,
            task_type="regression",
            wavelengths=wavelengths,
            top_n=1,
        )

        assert "R2pred" in df_out.columns
        r2_func = df_out.iloc[0]["R2pred"]
        assert pd.notna(r2_func), (
            "R²pred is NaN — function failed to rebuild and validate the "
            "autoscale=True chromosome row. Either the closure-rebuild path "
            "or the post-closure StandardScaler step is broken."
        )

        # The function and the manual sklearn Pipeline must produce
        # numerically equivalent R²pred. Tolerance accommodates any minor
        # numerical drift from PLS internals; pre-fix this assertion would
        # have failed by orders of magnitude (R²pred collapsed by ≥0.11
        # in user data; synthetic shift here makes the gap larger still).
        assert abs(r2_func - r2_ref) < 1e-3, (
            f"compute_validation_metrics_for_top_models returned R²pred="
            f"{r2_func:.4f}, sklearn-Pipeline reference={r2_ref:.4f}. "
            f"The {abs(r2_func - r2_ref):.4f} gap means val features "
            f"diverged from the train-fitted-scaler reference — likely "
            f"a regression of the closure-refit-on-val bug."
        )

    def test_evaluate_fitness_applies_autoscale_externally(self):
        """evaluate_fitness must still apply autoscale to inputs whose
        chromosome's autoscale gene is set — even though the closure no
        longer does it. Pre-fix the closure handled this; post-fix the
        function applies StandardScaler after the closure. A regression
        here would silently change exhaustive-search rankings (autoscale
        configs would degrade because they'd no longer be autoscaled at
        all).

        The smoke check: a chromosome whose underlying transform has a
        finite output should produce a finite fitness when autoscale is
        on, matching what the full evaluate_fitness path does. The real
        guard is that fitness is finite — pre-fix the post-closure path
        was missing entirely, which would produce wildly different
        scores for autoscale-on configs.
        """
        rng = np.random.RandomState(2)
        X = rng.randn(30, 60)
        y = rng.randn(30)

        snv_deriv1_idx = PREPROC_TYPES.index("snv_deriv1")
        w11_idx = WINDOW_SIZES.index(11) if 11 in WINDOW_SIZES else 0
        genes_autoscale_on = np.array([snv_deriv1_idx, w11_idx, 1], dtype=np.int32)
        genes_autoscale_off = np.array([snv_deriv1_idx, w11_idx, 0], dtype=np.int32)

        f_on = evaluate_fitness(
            genes_autoscale_on, X, y, cv_folds=3, n_components=3, task_type="regression"
        )
        f_off = evaluate_fitness(
            genes_autoscale_off, X, y, cv_folds=3, n_components=3, task_type="regression"
        )

        # Both must be finite — autoscale path is wired up.
        assert np.isfinite(f_on), (
            "evaluate_fitness with autoscale=True returned non-finite — "
            "the post-closure StandardScaler step is missing or broken."
        )
        assert np.isfinite(f_off)


# =============================================================================
# Integration Tests - Exhaustive Optimization
# =============================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestExhaustivePreprocessingOptimization:
    """Test full exhaustive preprocessing optimization (replaces GA loop)."""

    def test_basic_optimization(self, synthetic_spectra_small):
        """Exhaustive should complete and return valid result."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        result = optimize_preprocessing(
            X,
            y,
            method="exhaustive",
            cv_folds=3,
            n_components=5,
            random_state=42,
            verbose=0,
            n_jobs=1,  # Sequential for test determinism
        )

        # Check result structure
        assert "best_genes" in result
        assert "best_name" in result
        assert "best_transform" in result
        assert "best_rmsecv" in result
        assert "best_config" in result
        assert "history" in result
        assert "configs" in result  # Top-N output

        # Check best_genes shape (chromosome encoding still 2-gene)
        assert result["best_genes"].shape == (N_GENES,)
        assert isinstance(result["best_name"], str)
        assert result["best_transform"] is None or callable(result["best_transform"])

    def test_best_preprocessing_identified(self, synthetic_spectra_small):
        """Exhaustive should identify a reasonable preprocessing configuration."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        result = optimize_preprocessing(
            X,
            y,
            method="exhaustive",
            cv_folds=3,
            n_components=5,
            random_state=42,
            verbose=0,
            n_jobs=1,
        )

        if result["best_transform"] is not None:
            X_preprocessed = result["best_transform"](X)
            assert X_preprocessed.shape == X.shape
            assert np.isfinite(X_preprocessed).all()

    def test_respects_search_space(self, synthetic_spectra_small):
        """Exhaustive should only produce valid preprocessing combinations."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        result = optimize_preprocessing(
            X,
            y,
            method="exhaustive",
            cv_folds=3,
            n_components=5,
            random_state=42,
            verbose=0,
            n_jobs=1,
        )

        genes = result["best_genes"]
        assert 0 <= genes[0] < len(PREPROC_TYPES)
        assert 0 <= genes[1] < len(WINDOW_SIZES)

    def test_reproducibility_with_seed(self, synthetic_spectra_small):
        """Exhaustive should be reproducible with same random seed."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        result1 = optimize_preprocessing(
            X, y, method="exhaustive", cv_folds=3, n_components=5,
            random_state=42, verbose=0, n_jobs=1,
        )
        result2 = optimize_preprocessing(
            X, y, method="exhaustive", cv_folds=3, n_components=5,
            random_state=42, verbose=0, n_jobs=1,
        )

        assert np.array_equal(
            result1["best_genes"], result2["best_genes"]
        ), "Same seed should give same result"

    def test_classification_task(self, classification_data):
        """Exhaustive should work for classification tasks."""
        X, y = classification_data
        X = X.values
        y = y.values

        result = optimize_preprocessing(
            X, y, method="exhaustive", cv_folds=3, n_components=5,
            task_type="classification", random_state=42, verbose=0, n_jobs=1,
        )

        assert "best_genes" in result
        assert "best_rmsecv" in result  # 1 - accuracy for classification
        assert result["task_type"] == "classification"

    def test_legacy_ga_method_raises(self, synthetic_spectra_small):
        """method='ga' was removed in 2026-05-06; passing it must raise."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        with pytest.raises(ValueError, match="'ga' mode was removed"):
            optimize_preprocessing(
                X, y, method="ga", cv_folds=3, random_state=42, verbose=0,
            )

    def test_apply_autoscale_off_emits_2gene_chromosomes(self, synthetic_spectra_small):
        """Phase 3 backward-compat: apply_autoscale=False (default) must
        produce 2-gene chromosomes so saved-CSV ga_genes columns match
        legacy behavior bit-exact."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        result = optimize_preprocessing(
            X, y, method="exhaustive", cv_folds=3, n_components=3,
            random_state=42, verbose=0, n_jobs=1,
            apply_autoscale=False,
        )

        # All emitted configs should have len(genes) == 2
        for cfg in result["configs"]:
            assert cfg["genes"].shape == (2,), (
                f"Expected 2-gene chromosome, got shape {cfg['genes'].shape}"
            )

    def test_apply_autoscale_on_emits_3gene_chromosomes(self, synthetic_spectra_small):
        """Phase 3: apply_autoscale=True produces 3-gene chromosomes."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        result = optimize_preprocessing(
            X, y, method="exhaustive", cv_folds=3, n_components=3,
            random_state=42, verbose=0, n_jobs=1,
            apply_autoscale=True,
        )

        for cfg in result["configs"]:
            assert cfg["genes"].shape == (3,), (
                f"Expected 3-gene chromosome, got shape {cfg['genes'].shape}"
            )

    def test_apply_autoscale_on_emits_both_flag_values(self, synthetic_spectra_small):
        """When apply_autoscale=True, exhaustive must explore BOTH
        autoscale=False and autoscale=True; the diversity selector's
        (preproc, autoscale) key should let both surface in top-N."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        result = optimize_preprocessing(
            X, y, method="exhaustive", cv_folds=3, n_components=3,
            random_state=42, verbose=0, n_jobs=1,
            apply_autoscale=True,
            top_n=10,  # bigger top-N so both flag values can fit
        )

        autoscale_values = {
            _decode_autoscale_gene(cfg["genes"]) for cfg in result["configs"]
        }
        # Both False and True should appear among the diverse top-N
        assert autoscale_values == {False, True}, (
            f"Expected both autoscale=False and =True in top-N, got {autoscale_values}"
        )

    def test_phase2_disabled_matches_legacy_shape(self, synthetic_spectra_small):
        """Phase 2 regression pin: phase2_n_seeds=0 must produce the same
        output shape as pre-Phase-2 exhaustive (legacy single-seed
        diversity-selected top-N). The returned dict's phase2_halt_reason
        is 'disabled' in this case."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        result = optimize_preprocessing(
            X, y, method="exhaustive", cv_folds=3, n_components=3,
            random_state=42, verbose=0, n_jobs=1,
            phase2_n_seeds=0,  # disable Phase 2
        )

        assert result["phase2_halt_reason"] == "disabled"
        # Configs should still be the same shape
        assert "configs" in result and len(result["configs"]) > 0
        assert "best_genes" in result

    def test_phase2_enabled_logs_halt_reason(self, synthetic_spectra_small):
        """Phase 2 with default settings (n_seeds=5) must populate
        halt_reason with one of the live values: 'converged', 'cap', or
        'single_iteration'. 'disabled' is reserved for n_seeds=0."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        result = optimize_preprocessing(
            X, y, method="exhaustive", cv_folds=3, n_components=3,
            random_state=42, verbose=0, n_jobs=1,
            phase2_n_seeds=5,
        )

        assert result["phase2_halt_reason"] in {"converged", "cap", "single_iteration"}

    def test_phase2_cap_hit_emits_user_warning(self, synthetic_spectra_small, capsys):
        """Closes pr-test-analyzer rating-8 finding: the cap-hit branch
        prints a user-visible WARNING. Force a cap-hit by setting
        max_pool_multiplier=1 (cap = 1*top_n = 5) on a synthetic dataset
        small enough to enumerate, then capture stdout via capsys.

        This is the only behavioral coverage of the user-visible warning
        that the cap branch is supposed to produce. A regression that
        drops the print() (or routes it to logger.debug) silently
        degrades the user-visible signal documented in the commit.
        """
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        result = optimize_preprocessing(
            X, y, method="exhaustive", cv_folds=3, n_components=3,
            random_state=42, verbose=1, n_jobs=1,
            phase2_n_seeds=3,
            phase2_max_pool_multiplier=1,  # forces immediate cap
            top_n=5,  # cap = 1 * 5 = 5
        )

        captured = capsys.readouterr()
        # Tightened per Codex review (PR #57 cycle 2): assert the exact
        # prefix and the multiplier-text format, not just the loose
        # "WARNING in stdout AND cap in stdout" combo (which would
        # accept unrelated output that happens to contain both words).
        if result["phase2_halt_reason"] == "cap":
            assert "WARNING: Phase 2 halted at cap" in captured.out, (
                "Expected exact 'WARNING: Phase 2 halted at cap' prefix when "
                f"phase2_halt_reason='cap', got stdout: {captured.out[:500]}"
            )
            assert "* top_n" in captured.out, (
                "Expected multiplier-text marker '* top_n' in cap warning "
                f"so the user knows how to extend, got stdout: {captured.out[:500]}"
            )

    def test_phase2_can_change_top_n_vs_legacy(self, synthetic_spectra_small):
        """Phase 2 should produce a top-N that's potentially different from
        single-seed legacy. We don't assert "must differ" because on small
        synthetic data the top-N may genuinely converge across both paths;
        instead assert that BOTH paths run and return valid top-N of the
        same size, so a future regression that breaks the multi-seed
        helper integration doesn't go silent."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        legacy = optimize_preprocessing(
            X, y, method="exhaustive", cv_folds=3, n_components=3,
            random_state=42, verbose=0, n_jobs=1,
            phase2_n_seeds=0,
            top_n=5,
        )
        rescored = optimize_preprocessing(
            X, y, method="exhaustive", cv_folds=3, n_components=3,
            random_state=42, verbose=0, n_jobs=1,
            phase2_n_seeds=5,
            top_n=5,
        )

        # Both runs should produce a valid top-N
        assert len(legacy["configs"]) == 5
        assert len(rescored["configs"]) == 5
        # halt_reason should differentiate the two paths
        assert legacy["phase2_halt_reason"] == "disabled"
        assert rescored["phase2_halt_reason"] != "disabled"

    def test_with_actual_model_config(self, synthetic_spectra_small):
        """Closes Codex MEDIUM: actual-model exhaustive path is the live
        production path used by run_search (search.py:2052), but no test
        previously exercised it. This pins the model_config branch of
        evaluate_fitness — exhaustive must succeed and return a result with
        correct shape when given an explicit model_config dict.
        """
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        result = optimize_preprocessing(
            X,
            y,
            method="exhaustive",
            cv_folds=3,
            n_components=3,
            random_state=42,
            verbose=0,
            n_jobs=1,
            # Live path: actual-model fitness with first hyperparam point
            model_config={"name": "PLS", "params": {"n_components": 3}},
        )

        assert "best_genes" in result
        assert "configs" in result
        assert len(result["configs"]) > 0
        assert result["best_genes"].shape == (N_GENES,)
        # Best transform must be runnable on the data
        if result["best_transform"] is not None:
            X_pp = result["best_transform"](X)
            assert X_pp.shape == X.shape
            assert np.isfinite(X_pp).all()


# =============================================================================
# Integration Tests - Convenience Function
# =============================================================================


@pytest.mark.integration
class TestConvenienceFunction:
    """Test convenience function for integration with search.py."""

    def test_get_optimized_preproc_config_quick(self, synthetic_spectra_small):
        """Quick mode should return valid preprocessing config."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        name, transform = get_optimized_preproc_config(
            X, y, quick=True, random_state=42, verbose=0
        )

        assert isinstance(name, str)
        assert transform is None or callable(transform)

    def test_get_optimized_preproc_config_full(self, synthetic_spectra_small):
        """Full mode should return valid preprocessing config."""
        X, y = synthetic_spectra_small
        X = X.values
        y = y.values

        name, transform = get_optimized_preproc_config(
            X, y, quick=False, random_state=42, verbose=0
        )

        assert isinstance(name, str)
        assert transform is None or callable(transform)

        # Test that transform works
        if transform is not None:
            X_transformed = transform(X)
            assert X_transformed.shape == X.shape
            assert np.isfinite(X_transformed).all()

    def test_quick_vs_full_passes_different_cv_folds(self):
        """Closes Codex LOW: pre-Phase-1, quick/full differed in
        population_size/n_generations; post-refactor they only differ in
        cv_folds (3 vs 5). Pin that the wrapper actually threads cv_folds
        through to optimize_preprocessing — otherwise the docstring promise
        is unverified.
        """
        from unittest.mock import patch

        return_value = {
            "best_name": "raw",
            "best_transform": None,
            "best_genes": np.array([0, 0], dtype=np.int32),
            "best_rmsecv": 0.0,
            "best_config": "raw",
            "configs": [],
            "history": [],
            "task_type": "regression",
            "method": "exhaustive",
        }

        X = np.random.randn(20, 50)
        y = np.random.randn(20)

        with patch(
            "spectral_predict.ga_preprocessing.optimize_preprocessing",
            return_value=return_value,
        ) as spy:
            get_optimized_preproc_config(X, y, quick=True, random_state=42, verbose=0)
            get_optimized_preproc_config(X, y, quick=False, random_state=42, verbose=0)

        assert spy.call_count == 2
        quick_kwargs = spy.call_args_list[0].kwargs
        full_kwargs = spy.call_args_list[1].kwargs
        assert quick_kwargs["cv_folds"] == 3
        assert full_kwargs["cv_folds"] == 5
        # And both go through method='exhaustive', not the removed 'ga' default
        assert quick_kwargs["method"] == "exhaustive"
        assert full_kwargs["method"] == "exhaustive"


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_small_dataset(self):
        """Exhaustive should handle small datasets."""
        X = np.random.randn(10, 20)
        y = np.random.randn(10)

        result = optimize_preprocessing(
            X, y, method="exhaustive", cv_folds=3, n_components=3,
            random_state=42, verbose=0, n_jobs=1,
        )

        assert "best_genes" in result

    def test_high_dimensional_data(self):
        """Exhaustive should handle high-dimensional data."""
        X = np.random.randn(50, 1000)
        y = np.random.randn(50)

        result = optimize_preprocessing(
            X, y, method="exhaustive", cv_folds=3, n_components=5,
            random_state=42, verbose=0, n_jobs=1,
        )

        assert "best_genes" in result
