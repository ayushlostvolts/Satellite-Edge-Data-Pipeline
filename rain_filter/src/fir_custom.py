"""Stage C: Custom Windowed-Sinc FIR Low-Pass Filter.

Taps designed from scratch in pure NumPy:
- N = 11 taps (odd)
- Cutoff fc as fraction of Nyquist (default 0.35)
- Custom window: w[k] = 0.5 * hann[k] + 0.5 * kaiser(beta=4)[k]
- Zero-order modified Bessel function I0 implemented from Taylor series
- Normalized to unit DC gain (sum = 1)
- Symmetric taps with linear phase (group delay = (N - 1) / 2)
- Applied via direct convolution with reflect padding ('same' length)
"""

from typing import Optional, Tuple
import numpy as np
from rain_filter.src.impulse import _handle_nans


def bessel_i0_custom(x: np.ndarray, num_terms: int = 25) -> np.ndarray:
    """Compute modified Bessel function of the first kind of order 0: I0(x).

    Uses Taylor series expansion:
    I0(x) = sum_{m=0}^inf ((x / 2)^m / m!)^2 = sum_{m=0}^inf (x^2 / 4)^m / (m!)^2

    Parameters
    ----------
    x : np.ndarray
        Input array.
    num_terms : int
        Number of terms in series expansion (default 25, accurate to < 1e-14).

    Returns
    -------
    val : np.ndarray
        Computed I0(x).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    res = np.ones_like(x_arr)
    term = np.ones_like(x_arr)
    x_sq_div_4 = (x_arr * x_arr) * 0.25

    for m in range(1, num_terms + 1):
        term = term * (x_sq_div_4 / (m * m))
        res = res + term

    return res


def hann_window_custom(num_taps: int) -> np.ndarray:
    """Generate Hann window of length num_taps.

    w[k] = 0.5 * (1 - cos(2 * pi * k / (num_taps - 1)))
    """
    if num_taps <= 1:
        return np.ones(num_taps, dtype=np.float64)
    k = np.arange(num_taps, dtype=np.float64)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * k / (num_taps - 1.0)))


def kaiser_window_custom(num_taps: int, beta: float = 4.0) -> np.ndarray:
    """Generate Kaiser window of length num_taps with shape parameter beta."""
    if num_taps <= 1:
        return np.ones(num_taps, dtype=np.float64)
    k = np.arange(num_taps, dtype=np.float64)
    alpha = (num_taps - 1.0) / 2.0
    # Normalized argument: sqrt(1 - ((k - alpha) / alpha)^2)
    arg_inner = 1.0 - (((k - alpha) / alpha) ** 2)
    arg_inner = np.maximum(0.0, arg_inner)
    kaiser_arg = beta * np.sqrt(arg_inner)

    i0_num = bessel_i0_custom(kaiser_arg)
    i0_denom = bessel_i0_custom(np.array([beta]))[0]

    return i0_num / i0_denom


def custom_window(num_taps: int, beta: float = 4.0) -> np.ndarray:
    """Generate hybrid window: 0.5 * Hann + 0.5 * Kaiser(beta)."""
    w_hann = hann_window_custom(num_taps)
    w_kaiser = kaiser_window_custom(num_taps, beta=beta)
    return 0.5 * w_hann + 0.5 * w_kaiser


def design_fir_taps(
    num_taps: int = 11,
    fc: float = 0.35,
    beta: float = 4.0
) -> Tuple[np.ndarray, float]:
    """Design windowed-sinc lowpass FIR filter taps.

    Parameters
    ----------
    num_taps : int
        Number of taps N (must be odd, default 11).
    fc : float
        Cutoff frequency as fraction of Nyquist (default 0.35).
    beta : float
        Kaiser window parameter beta (default 4.0).

    Returns
    -------
    taps : np.ndarray
        Symmetric FIR taps of length N with sum(taps) == 1.
    group_delay : float
        Linear phase group delay in samples: (N - 1) / 2.
    """
    if num_taps % 2 == 0 or num_taps < 3:
        raise ValueError(f"num_taps must be an odd integer >= 3, got {num_taps}")
    if not (0.0 < fc < 1.0):
        raise ValueError(f"fc must be in (0, 1), got {fc}")

    M = (num_taps - 1) // 2
    k = np.arange(num_taps, dtype=np.float64)
    n_shift = k - M

    # Ideal sinc impulse response for normalized digital cutoff omega_c = pi * fc
    omega_c = np.pi * fc
    denom = np.where(n_shift == 0.0, 1.0, np.pi * n_shift)
    sinc_vals = np.sin(omega_c * n_shift) / denom
    h_ideal = np.where(n_shift == 0.0, fc, sinc_vals)

    # Window
    win = custom_window(num_taps, beta=beta)
    h_win = h_ideal * win

    # Normalize to unit DC gain (sum = 1)
    taps = h_win / np.sum(h_win)
    group_delay = float(M)

    return taps, group_delay


def custom_fir_filter(
    x: np.ndarray,
    num_taps: int = 11,
    fc: float = 0.35,
    beta: float = 4.0
) -> np.ndarray:
    """Stage C: Apply custom FIR low-pass filter by direct convolution with reflect padding.

    Parameters
    ----------
    x : np.ndarray
        Rainfall time series of shape (T,) or (T, ...).
    num_taps : int
        Number of taps N (default 11, odd).
    fc : float
        Cutoff as fraction of Nyquist (default 0.35).
    beta : float
        Kaiser beta parameter (default 4.0).

    Returns
    -------
    y : np.ndarray
        Filtered series, non-negative, same shape as input.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    t_len = x_arr.shape[0]

    if t_len == 0:
        return np.array([])
    if t_len < num_taps:
        out = np.clip(x_arr, 0.0, None)
        out[np.isnan(x_arr) | (x_arr <= -900.0)] = np.nan
        return out

    x_clean, nan_mask, all_nan_mask = _handle_nans(x_arr)

    taps, _ = design_fir_taps(num_taps=num_taps, fc=fc, beta=beta)
    M = (num_taps - 1) // 2

    # Pad with reflect along axis 0
    pad_spec = [(M, M)] + [(0, 0)] * (x_arr.ndim - 1)
    pad_x = np.pad(x_clean, pad_spec, mode="reflect")

    # Sliding window view along axis 0
    windows = np.lib.stride_tricks.sliding_window_view(pad_x, window_shape=num_taps, axis=0)

    # Direct convolution: sum of taps * window
    y = np.sum(windows * taps, axis=-1)

    # Ensure non-negative
    y = np.clip(y, 0.0, None)

    # Restore NaNs
    y[nan_mask] = np.nan
    if x_arr.ndim > 1:
        y[:, all_nan_mask] = np.nan

    return y
