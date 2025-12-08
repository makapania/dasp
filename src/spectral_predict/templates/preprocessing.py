"""
Preprocessing function templates for generated scripts.

All templates produce standalone functions using only numpy and scipy.
"""

SNV_TEMPLATE = '''
def apply_snv(spectra):
    """
    Standard Normal Variate (SNV) transformation.

    Normalizes each spectrum (row) by subtracting its mean and dividing
    by its standard deviation. This removes multiplicative scatter effects.

    Parameters
    ----------
    spectra : np.ndarray
        Spectral data, shape (n_samples, n_wavelengths)

    Returns
    -------
    np.ndarray
        SNV-transformed spectra, same shape as input

    Reference
    ---------
    Barnes, R.J., Dhanoa, M.S., & Lister, S.J. (1989). Standard Normal Variate
    Transformation and De-trending of Near-Infrared Diffuse Reflectance Spectra.
    Applied Spectroscopy, 43(5), 772-777.
    """
    spectra = np.asarray(spectra, dtype=np.float64)
    mean = spectra.mean(axis=1, keepdims=True)
    std = spectra.std(axis=1, keepdims=True)
    # Avoid division by zero
    std[std == 0] = 1.0
    return (spectra - mean) / std
'''

SAVGOL_DERIVATIVE_TEMPLATE = '''
def apply_savgol_derivative(spectra, derivative=1, window_length={window}, polyorder={polyorder}):
    """
    Savitzky-Golay derivative transformation.

    Applies smoothing and computes the derivative in a single step.
    This enhances spectral features while reducing noise.

    Parameters
    ----------
    spectra : np.ndarray
        Spectral data, shape (n_samples, n_wavelengths)
    derivative : int
        Derivative order (1 for 1st derivative, 2 for 2nd derivative)
    window_length : int
        Window size for Savitzky-Golay filter (must be odd)
    polyorder : int
        Polynomial order for Savitzky-Golay filter

    Returns
    -------
    np.ndarray
        Derivative spectra, same shape as input

    Reference
    ---------
    Savitzky, A., & Golay, M.J.E. (1964). Smoothing and Differentiation of Data
    by Simplified Least Squares Procedures. Analytical Chemistry, 36(8), 1627-1639.
    """
    from scipy.signal import savgol_filter

    spectra = np.asarray(spectra, dtype=np.float64)

    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1

    return savgol_filter(
        spectra,
        window_length=window_length,
        polyorder=polyorder,
        deriv=derivative,
        axis=1
    )
'''

MSC_TEMPLATE = '''
def fit_msc(spectra_reference):
    """
    Fit MSC transformation using reference spectra (typically calibration set mean).

    Parameters
    ----------
    spectra_reference : np.ndarray
        Reference spectra to compute the mean reference spectrum

    Returns
    -------
    np.ndarray
        Mean reference spectrum for MSC transformation
    """
    return np.mean(spectra_reference, axis=0)


def apply_msc(spectra, reference_spectrum):
    """
    Multiplicative Scatter Correction (MSC).

    Corrects for light scattering effects by fitting each spectrum to a
    reference spectrum and removing the baseline offset and slope.

    Parameters
    ----------
    spectra : np.ndarray
        Spectral data, shape (n_samples, n_wavelengths)
    reference_spectrum : np.ndarray
        Reference spectrum, shape (n_wavelengths,)
        Typically the mean spectrum of the calibration set

    Returns
    -------
    np.ndarray
        MSC-corrected spectra, same shape as input

    Reference
    ---------
    Geladi, P., McDougall, D., & Martens, H. (1985). Linearization and
    Scatter-Correction for Near-Infrared Reflectance Spectra of Meat.
    Applied Spectroscopy, 39(3), 491-500.
    """
    spectra = np.asarray(spectra, dtype=np.float64)
    reference = np.asarray(reference_spectrum, dtype=np.float64)

    corrected = np.zeros_like(spectra)

    for i in range(spectra.shape[0]):
        # Fit: spectrum_i = a + b * reference
        fit = np.polyfit(reference, spectra[i, :], 1)

        # Avoid division by near-zero slope
        if abs(fit[0]) < 1e-10:
            corrected[i, :] = spectra[i, :]
        else:
            # Correct: spectrum_corrected = (spectrum_i - a) / b
            corrected[i, :] = (spectra[i, :] - fit[1]) / fit[0]

    return corrected
'''


def get_preprocessing_template(preprocessing_spec: str) -> tuple:
    """
    Get the appropriate preprocessing template based on specification.

    Parameters
    ----------
    preprocessing_spec : str
        Preprocessing specification like 'snv', 'deriv1_w7', 'snv_deriv1_w15'

    Returns
    -------
    tuple
        (template_code, application_code) where:
        - template_code is the function definition(s)
        - application_code is how to apply the preprocessing
    """
    templates = []
    applications = []

    if preprocessing_spec == 'raw':
        return '', 'X_processed = X.copy()'

    parts = preprocessing_spec.split('_')

    # Check for SNV
    if parts[0] == 'snv':
        templates.append(SNV_TEMPLATE)
        applications.append('X_processed = apply_snv(X)')
        parts = parts[1:]
    else:
        applications.append('X_processed = X.copy()')

    # Check for derivative
    if parts and parts[0].startswith('deriv'):
        # Parse derivative order
        deriv_order = 1 if parts[0] == 'deriv1' else 2

        # Parse window size
        window = 7  # default
        polyorder = 2 if deriv_order == 1 else 3

        if len(parts) > 1 and parts[1].startswith('w'):
            try:
                window = int(parts[1][1:])
            except ValueError:
                pass

        # Format the template with specific parameters
        deriv_template = SAVGOL_DERIVATIVE_TEMPLATE.format(
            window=window,
            polyorder=polyorder
        )
        templates.append(deriv_template)

        if len(applications) > 1 or applications[0] != 'X_processed = X.copy()':
            # Already applied SNV
            applications.append(
                f'X_processed = apply_savgol_derivative(X_processed, derivative={deriv_order}, '
                f'window_length={window})'
            )
        else:
            applications[0] = (
                f'X_processed = apply_savgol_derivative(X, derivative={deriv_order}, '
                f'window_length={window})'
            )

    template_code = '\n'.join(templates)
    application_code = '\n'.join(applications)

    return template_code, application_code
