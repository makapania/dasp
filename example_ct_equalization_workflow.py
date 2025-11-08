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
    save_transfer_model,
    load_transfer_model,
)
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
