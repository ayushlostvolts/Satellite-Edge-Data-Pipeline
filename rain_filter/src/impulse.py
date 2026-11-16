"""Stage A: Isolated-Impulse Suppressor (Hampel-style, domain-aware).

Suppresses isolated sensor/bit-error spikes while strictly preserving multi-day
convective rainfall events. Implemented from scratch in NumPy with reflect padding.
"""

from typing import Optional, Tuple
import numpy as np


def _handle_nans(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify missing values (-900 or NaN), interpolate for filtering, and return mask.

    Parameters
    ----------
    x : np.ndarray
        Input array with time along axis 0.

    Returns
    -------
    x_clean : np.ndarray
        Array with NaNs temporarily interpolated along axis 0.
    nan_mask : np.ndarray
        Boolean mask where True indicates missing values in original input.
    all_nan_mask : np.ndarray
        Boolean mask where True indicates a slice along axis 0 has NO valid data.
    """
    nan_mask = np.isnan(x) | (x <= -900.0)
    x_clean = np.where(nan_mask, 0.0, x.astype(np.float64, copy=True))

    # Identify slices where entire time series is NaN
    if x.ndim == 1:
        all_nan = np.all(nan_mask)
        if not all_nan and np.any(nan_mask):
            valid_idx = np.where(~nan_mask)[0]
            nan_idx = np.where(nan_mask)[0]
            x_clean[nan_idx] = np.interp(nan_idx, valid_idx, x_clean[valid_idx])
        all_nan_mask = np.array(all_nan)
    else:
        # For multi-dimensional arrays, e.g. (T, lat, lon)
        all_nan_mask = np.all(nan_mask, axis=0)  # shape (lat, lon)
        # Interpolate along time axis for cells with partial NaNs
        # Flatten spatial dimensions for fast 1D interpolation where needed
        orig_shape = x.shape
        t_len = orig_shape[0]
        x_flat = x_clean.reshape(t_len, -1)
        mask_flat = nan_mask.reshape(t_len, -1)
        all_nan_flat = all_nan_mask.reshape(-1)

        t_idx = np.arange(t_len)
        for i in range(x_flat.shape[1]):
            if all_nan_flat[i]:
                continue
            col_mask = mask_flat[:, i]
            if np.any(col_mask):
                valid = ~col_mask
                x_flat[col_mask, i] = np.interp(t_idx[col_mask], t_idx[valid], x_flat[valid, i])

        x_clean = x_flat.reshape(orig_shape)

    return x_clean, nan_mask, all_nan_mask


def impulse_filter(
    x: np.ndarray,
    window: int = 7,
    k: float = 4.0,
    s_floor: float = 0.5,
    isolation_ratio: float = 0.35
) -> np.ndarray:
    """Stage A: Domain-aware isolated impulse suppressor.

    For each sample n with window W (default 7, odd):
    - m = median(x[n-h : n+h+1])
    - s = 1.4826 * MAD of window, h = (W - 1) // 2
    - Flag n as impulse if:
        (x[n] - m) > k * max(s, s_floor)
        AND ISOLATED: both x[n-1] and x[n+1] < m + isolation_ratio * (x[n] - m)
    - Replace flagged samples with m. Otherwise pass through unchanged.
    - Edges handled with reflect padding.
    - Vectorized across time axis (axis 0).

    Parameters
    ----------
    x : np.ndarray
        Rainfall time series of shape (T,) or (T, ...).
    window : int
        Window size W (must be odd, default 7).
    k : float
        Impulse threshold multiplier (default 4.0).
    s_floor : float
        Noise floor for scale parameter s in mm (default 0.5).
    isolation_ratio : float
        Fraction of peak height above median required for adjacent neighbors
        to be considered isolated (default 0.35, tuned from 0.50 to preserve
        asymmetric multi-day convective storm peaks).

    Returns
    -------
    y : np.ndarray
        Filtered series with impulses suppressed, non-negative, matching input shape.
    """
    if window % 2 == 0 or window < 3:
        raise ValueError(f"Window must be an odd integer >= 3, got {window}")

    x_arr = np.asarray(x, dtype=np.float64)
    orig_shape = x_arr.shape
    t_len = orig_shape[0]

    if t_len < window:
        # If series is shorter than window, return copy clipped at 0
        out = np.clip(x_arr, 0.0, None)
        out[np.isnan(x_arr) | (x_arr <= -900.0)] = np.nan
        return out

    x_clean, nan_mask, all_nan_mask = _handle_nans(x_arr)

    h = (window - 1) // 2

    # Pad with reflect along axis 0
    pad_spec = [(h, h)] + [(0, 0)] * (x_arr.ndim - 1)
    pad_x = np.pad(x_clean, pad_spec, mode="reflect")

    # Sliding window view along axis 0
    # windows shape: (t_len, ..., window)
    windows = np.lib.stride_tricks.sliding_window_view(pad_x, window_shape=window, axis=0)

    # Median and MAD along last axis (window axis)
    m = np.median(windows, axis=-1)
    abs_dev = np.abs(windows - np.expand_dims(m, axis=-1))
    mad = np.median(abs_dev, axis=-1)
    s = 1.4826 * mad

    # Extract current, previous, and next samples from pad_x
    # Index h in pad_x corresponds to n = 0
    x_curr = pad_x[h : h + t_len]
    x_prev = pad_x[h - 1 : h - 1 + t_len]
    x_next = pad_x[h + 1 : h + 1 + t_len]

    # Threshold condition
    scale_term = np.maximum(s, s_floor)
    cond_outlier = (x_curr - m) > (k * scale_term)

    # Domain-aware isolation condition:
    # Both adjacent samples must be below m + isolation_ratio * (x_curr - m)
    half_peak = m + isolation_ratio * (x_curr - m)
    cond_isolated = (x_prev < half_peak) & (x_next < half_peak)

    is_impulse = cond_outlier & cond_isolated

    y = np.where(is_impulse, m, x_curr)

    # Ensure non-negative
    y = np.clip(y, 0.0, None)

    # Restore NaNs
    y[nan_mask] = np.nan
    if x_arr.ndim > 1:
        # Where entire cell was NaN, ensure all elements remain NaN
        y[:, all_nan_mask] = np.nan

    return y
