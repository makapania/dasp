"""
Example workflow for calibration transfer & equalization.

This assumes the following modules are implemented:

- spectral_predict.instrument_profiles
- spectral_predict.calibration_transfer
- spectral_predict.equalization

The goal is to show how the pieces are *supposed* to be used, so the
agent can implement the missing function bodies to make this script work.
"""

import numpy as np

from spectral_predict.instrument_profiles import (
    InstrumentProfile,
    characterize_instrument,
    rank_instruments_by_detail,
    estimate_smoothing_between_instruments,
)
from spectral_predict.calibration_transfer import (
    TransferModel,
    resample_to_grid,
    estimate_ds,
    apply_ds,
    estimate_tsr,
    apply_tsr,
    estimate_ctai,
    apply_ctai,
    estimate_jypls_inv,
    apply_jypls_inv,
    estimate_nspfce,
    apply_nspfce,
    save_transfer_model,
    load_transfer_model,
)
from spectral_predict.sample_selection import kennard_stone
from spectral_predict.equalization import (
    choose_common_grid,
    build_equalization_mapping_for_instrument,
    equalize_dataset,
)


def make_synthetic_gaussian_spectra(wavelengths, centers, widths, n_samples=20, noise=0.0):
    """
    Generate simple synthetic spectra: sum of Gaussians + optional noise.
    """
    wl = wavelengths[None, :]  # (1, p)
    X = np.zeros((n_samples, wl.shape[1]), dtype=float)
    for i in range(n_samples):
        spec = np.zeros_like(wl, dtype=float)
        for c, w in zip(centers, widths):
            spec += np.exp(-0.5 * ((wl - c) / w) ** 2)
        if noise > 0:
            spec += noise * np.random.randn(*spec.shape)
        X[i, :] = spec
    return X


def demo_backend_workflow():
    # 1. Create synthetic instruments A (high-res) and B (low-res)
    wl_A = np.arange(1000.0, 2500.0, 1.0)   # ~1 nm spacing
    wl_B = np.arange(1000.0, 2500.0, 5.0)   # ~5 nm spacing

    X_A = make_synthetic_gaussian_spectra(wl_A, centers=[1450, 1930], widths=[8, 10], n_samples=30, noise=0.01)
    X_B = make_synthetic_gaussian_spectra(wl_B, centers=[1450, 1930], widths=[10, 12], n_samples=30, noise=0.01)

    # 2. Build instrument profiles from synthetic data
    prof_A = characterize_instrument(
        instrument_id="Instrument_A",
        wavelengths=wl_A,
        spectra=X_A,
        vendor="SyntheticCo",
        model="HighRes",
        description="Synthetic high-resolution instrument",
    )
    prof_B = characterize_instrument(
        instrument_id="Instrument_B",
        wavelengths=wl_B,
        spectra=X_B,
        vendor="SyntheticCo",
        model="LowRes",
        description="Synthetic low-resolution instrument",
    )

    profiles = {
        prof_A.instrument_id: prof_A,
        prof_B.instrument_id: prof_B,
    }

    # 3. Rank instruments by detail (higher detail_score = higher effective resolution)
    ranked = rank_instruments_by_detail(profiles)
    print("Instruments ranked by detail_score (desc):", ranked)

    # 4. Choose a common grid for transfer/equalization
    common_wl = choose_common_grid(profiles, instrument_ids=list(profiles.keys()))
    print("Common wavelength grid size:", common_wl.shape[0])

    # 5. Resample A and B onto the common grid
    X_A_common = resample_to_grid(X_A, wl_A, common_wl)
    X_B_common = resample_to_grid(X_B, wl_B, common_wl)

    # 6. Optional: estimate smoothing needed to map high-res to low-res
    # (here we pretend Instrument_A is the high-detail one)
    sigma_candidates = [0.0, 1.0, 2.0, 4.0, 6.0]
    best_sigma = estimate_smoothing_between_instruments(
        wavelengths_high=wl_A,
        X_high=X_A,
        wavelengths_low=wl_B,
        X_low=X_B,
        sigma_candidates=sigma_candidates,
    )
    print("Estimated best smoothing sigma (high -> low):", best_sigma)

    # 7. Build a simple Direct Standardization transfer model B -> A on the common grid
    #    (here we treat A as the master, B as the slave)
    A_ds = estimate_ds(X_master=X_A_common, X_slave=X_B_common, lam=1e-3)

    tm = TransferModel(
        master_id="Instrument_A",
        slave_id="Instrument_B",
        method="ds",
        wavelengths_common=common_wl,
        params={"A": A_ds},
        meta={"note": "Synthetic DS demo"},
    )

    # 8. Apply transfer to new slave spectra (here just reuse X_B_common for demo)
    X_B_to_A = apply_ds(X_B_common, A_ds)
    print("DS transformed slave spectra shape:", X_B_to_A.shape)

    # 9. Save and reload the transfer model
    save_prefix = save_transfer_model(tm, directory="calibration_transfer_demo", name="synthetic_A_from_B_ds")
    print("Saved TransferModel prefix:", save_prefix)

    tm_loaded = load_transfer_model(save_prefix)
    print("Loaded TransferModel:", tm_loaded.master_id, tm_loaded.slave_id, tm_loaded.method)

    # 9b. NEW: Test TSR (Transfer Sample Regression) - requires only 12-13 samples!
    print("\n--- TSR (Transfer Sample Regression) Demo ---")
    # Select 12 transfer samples using Kennard-Stone algorithm
    transfer_indices = kennard_stone(X_A_common, n_samples=12)
    print(f"Selected {len(transfer_indices)} transfer samples using Kennard-Stone")

    # Estimate TSR model
    tsr_params = estimate_tsr(X_A_common, X_B_common, transfer_indices)
    print(f"TSR model estimated - Mean R²: {tsr_params['mean_r_squared']:.4f}")

    # Apply TSR
    X_B_to_A_tsr = apply_tsr(X_B_common, tsr_params)
    print(f"TSR transformed spectra shape: {X_B_to_A_tsr.shape}")

    # Compare TSR to DS
    rmse_ds = np.sqrt(np.mean((X_B_to_A - X_A_common) ** 2))
    rmse_tsr = np.sqrt(np.mean((X_B_to_A_tsr - X_A_common) ** 2))
    print(f"Comparison - DS RMSE: {rmse_ds:.6f}, TSR RMSE: {rmse_tsr:.6f}")

    # Create TSR TransferModel
    tm_tsr = TransferModel(
        master_id="Instrument_A",
        slave_id="Instrument_B",
        method="tsr",
        wavelengths_common=common_wl,
        params=tsr_params,
        meta={"note": "Synthetic TSR demo", "n_transfer_samples": 12}
    )

    # 9c. NEW: Test CTAI (Affine Invariance) - NO transfer samples needed!
    print("\n--- CTAI (Affine Invariance) Demo ---")
    print("CTAI requires NO transfer samples - uses statistical properties!")

    # Estimate CTAI model
    ctai_params = estimate_ctai(X_A_common, X_B_common)
    print(f"CTAI model estimated:")
    print(f"  - Components: {ctai_params['n_components']}")
    print(f"  - Explained variance: {ctai_params['explained_variance']:.4f}")
    print(f"  - Reconstruction error: {ctai_params['reconstruction_error']:.6f}")

    # Apply CTAI
    X_B_to_A_ctai = apply_ctai(X_B_common, ctai_params)
    print(f"CTAI transformed spectra shape: {X_B_to_A_ctai.shape}")

    # Compare all methods
    rmse_ctai = np.sqrt(np.mean((X_B_to_A_ctai - X_A_common) ** 2))
    print(f"\nMethod Comparison (RMSE):")
    print(f"  - DS (full): {rmse_ds:.6f}")
    print(f"  - TSR (12 samples): {rmse_tsr:.6f}")
    print(f"  - CTAI (0 samples): {rmse_ctai:.6f}")

    # Create CTAI TransferModel
    tm_ctai = TransferModel(
        master_id="Instrument_A",
        slave_id="Instrument_B",
        method="ctai",
        wavelengths_common=common_wl,
        params=ctai_params,
        meta={"note": "Synthetic CTAI demo - no standards required!"}
    )

    # 9c. NEW: Test JYPLS-inv (Joint-Y PLS) - 12 transfer samples + PLS modeling
    print("\n--- JYPLS-inv (Joint-Y PLS with Inversion) Demo ---")
    print("JYPLS-inv: Uses PLS modeling with 12-13 selected transfer samples")

    # Select 12 transfer samples using Kennard-Stone
    jypls_transfer_idx = kennard_stone(X_A_common, n_samples=12)
    # Generate pseudo-Y values (in practice, these would be reference measurements)
    y_jypls_transfer = X_A_common[jypls_transfer_idx].mean(axis=1)

    # Estimate JYPLS-inv model with automatic PLS component selection
    jypls_params = estimate_jypls_inv(
        X_A_common, X_B_common, y_jypls_transfer, jypls_transfer_idx,
        n_components=None  # Auto-select via cross-validation
    )
    print(f"JYPLS-inv model estimated:")
    print(f"  - Transfer samples: 12 (Kennard-Stone)")
    print(f"  - PLS components: {jypls_params['n_components']} (auto-selected)")
    print(f"  - CV RMSE: {jypls_params['cv_rmse']:.6f}")
    print(f"  - Explained variance: {jypls_params['explained_variance_ratio']:.4f}")

    # Apply JYPLS-inv
    X_B_to_A_jypls = apply_jypls_inv(X_B_common, jypls_params)
    print(f"JYPLS-inv transformed spectra shape: {X_B_to_A_jypls.shape}")

    rmse_jypls = np.sqrt(np.mean((X_B_to_A_jypls - X_A_common) ** 2))

    # Create JYPLS-inv TransferModel
    tm_jypls = TransferModel(
        master_id="Instrument_A",
        slave_id="Instrument_B",
        method="jypls-inv",
        wavelengths_common=common_wl,
        params=jypls_params,
        meta={"note": "Synthetic JYPLS-inv demo - PLS-based transfer"}
    )

    # 9d. NEW: Test NS-PFCE (Non-supervised Parameter-Free) - NO transfer samples + wavelength selection!
    print("\n--- NS-PFCE (Non-supervised Parameter-Free Calibration Enhancement) Demo ---")
    print("NS-PFCE: NO transfer samples + automatic wavelength selection (VCPA-IRIV)!")

    # Estimate NS-PFCE model with wavelength selection
    nspfce_params = estimate_nspfce(
        X_A_common, X_B_common, common_wl,
        use_wavelength_selection=True,
        wavelength_selector='vcpa-iriv',
        max_iterations=100
    )
    print(f"NS-PFCE model estimated:")
    print(f"  - Iterations: {nspfce_params['n_iterations']} / 100")
    print(f"  - Converged: {nspfce_params['converged']}")
    if nspfce_params.get('selected_wavelength_indices') is not None:
        n_selected = len(nspfce_params['selected_wavelength_indices'])
        n_total = len(common_wl)
        print(f"  - Wavelength selection: {n_selected} / {n_total} ({100*n_selected/n_total:.1f}%)")

    # Apply NS-PFCE
    X_B_to_A_nspfce = apply_nspfce(X_B_common, nspfce_params)
    print(f"NS-PFCE transformed spectra shape: {X_B_to_A_nspfce.shape}")

    # Compare all methods
    rmse_nspfce = np.sqrt(np.mean((X_B_to_A_nspfce - X_A_common) ** 2))
    print(f"\nMethod Comparison (RMSE):")
    print(f"  - DS (full): {rmse_ds:.6f}")
    print(f"  - TSR (12 samples): {rmse_tsr:.6f}")
    print(f"  - CTAI (0 samples): {rmse_ctai:.6f}")
    print(f"  - JYPLS-inv (12 samples + PLS): {rmse_jypls:.6f}")
    print(f"  - NS-PFCE (0 samples + wavelength selection): {rmse_nspfce:.6f}")

    # Create NS-PFCE TransferModel
    tm_nspfce = TransferModel(
        master_id="Instrument_A",
        slave_id="Instrument_B",
        method="nspfce",
        wavelengths_common=common_wl,
        params=nspfce_params,
        meta={"note": "Synthetic NS-PFCE demo - no standards + automatic wavelength selection!"}
    )

    # 10. Build equalization mappings for multi-instrument dataset
    spectra_by_instrument = {
        "Instrument_A": (wl_A, X_A),
        "Instrument_B": (wl_B, X_B),
    }

    wl_common_eq, X_common_eq = equalize_dataset(
        spectra_by_instrument=spectra_by_instrument,
        profiles=profiles,
    )
    print("Equalized spectra shape:", X_common_eq.shape)
    print("Equalized wavelength grid length:", wl_common_eq.shape[0])


if __name__ == "__main__":
    demo_backend_workflow()
