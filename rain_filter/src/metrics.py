"""Evaluation metrics and baseline filters for comparison.

Metrics:
- SNR improvement (dB)
- MSE, MAE
- Total rainfall relative bias (%)
- Peak preservation ratio (for days where clean >= 25 mm)
- Wet-day F1 score (wet >= 2.5 mm)

Baselines (SciPy allowed):
- Moving average (5)
- Median filter (5)
- Butterworth low-pass order 3 (filtfilt)
- Savitzky-Golay (11, 3)
"""

from typing import Dict, Optional, Tuple
import numpy as np
import scipy.signal
import scipy.ndimage


def compute_metrics(
    clean: np.ndarray,
    noisy: np.ndarray,
    filtered: np.ndarray,
    peak_threshold: float = 25.0,
    wet_threshold: float = 2.5
) -> Dict[str, float]:
    """Compute all evaluation metrics between clean, noisy, and filtered series.

    Parameters
    ----------
    clean : np.ndarray
        Ground truth clean rainfall series.
    noisy : np.ndarray
        Noisy input series.
    filtered : np.ndarray
        Filtered output series.
    peak_threshold : float
        Rainfall threshold in mm to identify peak days (default 25.0 mm).
    wet_threshold : float
        Threshold in mm to identify wet days (default 2.5 mm).

    Returns
    -------
    metrics : dict
        Dictionary of computed metric values.
    """
    # Valid mask where clean and filtered are both non-NaN and finite
    valid = (
        ~np.isnan(clean)
        & ~np.isnan(filtered)
        & ~np.isnan(noisy)
        & (clean > -900.0)
        & (filtered > -900.0)
        & (noisy > -900.0)
    )

    if not np.any(valid):
        return {
            "snr_improvement_db": 0.0,
            "mse": np.nan,
            "mae": np.nan,
            "relative_bias_pct": np.nan,
            "peak_preservation_ratio": np.nan,
            "wet_f1": np.nan,
        }

    c = clean[valid]
    n = noisy[valid]
    f = filtered[valid]

    # 1. MSE and MAE
    mse_f = float(np.mean((f - c) ** 2))
    mae_f = float(np.mean(np.abs(f - c)))
    mse_n = float(np.mean((n - c) ** 2))

    # 2. SNR improvement (dB)
    if mse_f < 1e-12:
        snr_improvement = 100.0
    elif mse_n < 1e-12:
        snr_improvement = 0.0
    else:
        snr_improvement = float(10.0 * np.log10(mse_n / mse_f))

    # 3. Total rainfall relative bias (%)
    sum_c = float(np.sum(c))
    sum_f = float(np.sum(f))
    if sum_c > 1e-6:
        bias_pct = float((sum_f - sum_c) / sum_c * 100.0)
    else:
        bias_pct = 0.0

    # 4. Peak preservation ratio
    peak_mask = c >= peak_threshold
    if np.any(peak_mask):
        peak_ratio = float(np.max(f[peak_mask]) / max(np.max(c[peak_mask]), 1e-6))
    else:
        # Fallback to overall max if no days >= 25 mm
        peak_ratio = float(np.max(f) / max(np.max(c), 1e-6)) if len(c) > 0 else 1.0

    # 5. Wet-day F1 score
    true_wet = c >= wet_threshold
    pred_wet = f >= wet_threshold

    tp = np.sum(true_wet & pred_wet)
    fp = np.sum(~true_wet & pred_wet)
    fn = np.sum(true_wet & ~pred_wet)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    if precision + recall > 0:
        f1 = float(2.0 * precision * recall / (precision + recall))
    else:
        f1 = 0.0

    return {
        "snr_improvement_db": snr_improvement,
        "mse": mse_f,
        "mae": mae_f,
        "relative_bias_pct": bias_pct,
        "peak_preservation_ratio": peak_ratio,
        "wet_f1": f1,
    }


# ---------------------------------------------------------
# Baselines for comparison (scipy allowed here)
# ---------------------------------------------------------

def baseline_moving_average(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Baseline 1: Moving average (5) with reflect padding, non-negative."""
    nan_mask = np.isnan(x) | (x <= -900.0)
    x_clean = np.where(nan_mask, 0.0, x)
    if np.any(nan_mask) and not np.all(nan_mask):
        idx = np.arange(len(x))
        x_clean[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], x_clean[~nan_mask])

    res = scipy.ndimage.uniform_filter1d(x_clean.astype(np.float64), size=window, mode="reflect")
    res = np.clip(res, 0.0, None)
    res[nan_mask] = np.nan
    return res


def baseline_median(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Baseline 2: Median filter (5) with reflect padding, non-negative."""
    nan_mask = np.isnan(x) | (x <= -900.0)
    x_clean = np.where(nan_mask, 0.0, x)
    if np.any(nan_mask) and not np.all(nan_mask):
        idx = np.arange(len(x))
        x_clean[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], x_clean[~nan_mask])

    res = scipy.ndimage.median_filter(x_clean.astype(np.float64), size=window, mode="reflect")
    res = np.clip(res, 0.0, None)
    res[nan_mask] = np.nan
    return res


def baseline_butterworth(
    x: np.ndarray,
    order: int = 3,
    wn: float = 0.35
) -> np.ndarray:
    """Baseline 3: Butterworth low-pass order 3 via filtfilt, non-negative."""
    nan_mask = np.isnan(x) | (x <= -900.0)
    x_clean = np.where(nan_mask, 0.0, x)
    if np.any(nan_mask) and not np.all(nan_mask):
        idx = np.arange(len(x))
        x_clean[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], x_clean[~nan_mask])

    b, a = scipy.signal.butter(order, Wn=wn, btype="low")
    res = scipy.signal.filtfilt(b, a, x_clean.astype(np.float64))
    res = np.clip(res, 0.0, None)
    res[nan_mask] = np.nan
    return res


def baseline_savgol(
    x: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3
) -> np.ndarray:
    """Baseline 4: Savitzky-Golay (11, 3), non-negative."""
    nan_mask = np.isnan(x) | (x <= -900.0)
    x_clean = np.where(nan_mask, 0.0, x)
    if np.any(nan_mask) and not np.all(nan_mask):
        idx = np.arange(len(x))
        x_clean[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], x_clean[~nan_mask])

    res = scipy.signal.savgol_filter(
        x_clean.astype(np.float64),
        window_length=window_length,
        polyorder=polyorder,
        mode="interp"
    )
    res = np.clip(res, 0.0, None)
    res[nan_mask] = np.nan
    return res
