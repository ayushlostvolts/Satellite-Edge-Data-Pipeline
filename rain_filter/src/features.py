"""Feature Extraction Module for Satellite Edge Rainfall Processing.

Extracts time-domain and multi-resolution wavelet-domain features from denoised
rainfall series and DWT subband coefficients.

Features extracted per series/cell:
1. Wavelet Subband Energies (E_A3, E_D3, E_D2, E_D1, and percentages)
2. Convective Transient Energy Ratio: (E_D1 + E_D2) / E_total * 100%
3. Wavelet Shannon Energy Entropy: -sum(p_i * log2(p_i))
4. Detail-to-Approximation Ratio: sum(E_Details) / E_Approx
5. Peak Wavelet Transient Ratio: max(|D1|) / max(|AJ|)
6. Maximum Daily Precipitation (P_max in mm)
7. Extreme Precipitation Fraction R95p (% of total rain from days > 95th percentile)
8. Simple Daily Intensity Index (SDII: mean wet-day rain in mm/day)
9. Wet-Day Frequency (F_wet: fraction of days with rain >= 2.5 mm)
10. Consecutive Wet Days (CWD) and Consecutive Dry Days (CDD)
11. Statistical Skewness and Coefficient of Variation (CV)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from rain_filter.src.dwt import DWTResult, dwt_1d, dwt_cube


def extract_features_1d(
    series: np.ndarray,
    dwt_res: Optional[DWTResult] = None,
    wavelet: str = "haar",
    level: int = 3,
    wet_threshold: float = 2.5,
    dry_threshold: float = 1.0
) -> Dict[str, float]:
    """Extract physical and wavelet features from a 1D daily rainfall series.

    Parameters
    ----------
    series : np.ndarray
        1D daily rainfall time series (T,).
    dwt_res : DWTResult, optional
        Pre-computed DWT result. If None, computed automatically.
    wavelet : str
        Wavelet type for DWT (default 'haar').
    level : int
        Decomposition level (default 3).
    wet_threshold : float
        Rainfall threshold for wet days in mm (default 2.5 mm).
    dry_threshold : float
        Threshold below which day is considered dry in mm (default 1.0 mm).

    Returns
    -------
    features : dict
        Dictionary of named feature values.
    """
    x = np.asarray(series, dtype=np.float64)
    valid = ~np.isnan(x) & (x >= 0.0)
    x_valid = x[valid]

    t_len = len(x_valid)
    if t_len == 0 or np.sum(x_valid) <= 1e-6:
        # Dry / invalid series fallback
        return {
            "p_max_mm": 0.0,
            "total_rainfall_mm": 0.0,
            "wet_day_freq": 0.0,
            "sdii_intensity_mm": 0.0,
            "r95p_extreme_ratio_pct": 0.0,
            "max_cwd_days": 0.0,
            "max_cdd_days": float(len(x)),
            "coeff_variation": 0.0,
            "convective_energy_ratio_pct": 0.0,
            "wavelet_shannon_entropy": 0.0,
            "detail_to_approx_ratio": 0.0,
            "peak_wavelet_transient_ratio": 0.0,
            "d1_energy_pct": 0.0,
            "d2_energy_pct": 0.0,
            "d3_energy_pct": 0.0,
            "approx_energy_pct": 100.0,
        }

    # 1. Time-Domain Meteorological Features
    p_max = float(np.max(x_valid))
    p_total = float(np.sum(x_valid))

    wet_days = x_valid >= wet_threshold
    n_wet = int(np.sum(wet_days))
    wet_day_freq = float(n_wet / t_len)

    # Simple Daily Intensity Index (SDII)
    sdii = float(np.sum(x_valid[wet_days]) / n_wet) if n_wet > 0 else 0.0

    # R95p: fraction of rain from days above 95th percentile of wet days
    if n_wet >= 5:
        p95 = float(np.percentile(x_valid[wet_days], 95))
        r95_rain = float(np.sum(x_valid[x_valid >= p95]))
        r95p_ratio = float((r95_rain / p_total) * 100.0)
    else:
        r95p_ratio = 0.0

    # Consecutive Wet and Dry Days (CWD, CDD)
    max_cwd = 0
    cur_cwd = 0
    max_cdd = 0
    cur_cdd = 0

    for val in x_valid:
        if val >= wet_threshold:
            cur_cwd += 1
            max_cwd = max(max_cwd, cur_cwd)
            cur_cdd = 0
        elif val < dry_threshold:
            cur_cdd += 1
            max_cdd = max(max_cdd, cur_cdd)
            cur_cwd = 0
        else:
            cur_cwd = 0
            cur_cdd = 0

    # Coefficient of Variation (CV = std / mean)
    mean_val = float(np.mean(x_valid))
    std_val = float(np.std(x_valid))
    cv = float(std_val / mean_val) if mean_val > 1e-6 else 0.0

    # 2. Wavelet-Domain Features
    if dwt_res is None:
        dwt_res = dwt_1d(x_valid, wavelet=wavelet, level=level)

    energies = dwt_res.energies
    e_total = float(energies["total_energy"])
    e_approx = float(energies["approx_energy"])

    # Convective Transient Ratio: (D1 + D2) / E_total
    conv_ratio = float(energies["convective_transient_ratio_pct"])

    # Detail to Approximation Energy Ratio
    detail_energy_sum = e_total - e_approx
    detail_to_approx = float(detail_energy_sum / max(e_approx, 1e-12))

    # Wavelet Shannon Entropy: H = -sum(p_i * log2(p_i))
    subband_p = [float(energies["approx_energy_pct"]) / 100.0]
    for j in range(1, len(dwt_res.details) + 1):
        subband_p.append(float(energies[f"D{j}_energy_pct"]) / 100.0)

    shannon_entropy = 0.0
    for p in subband_p:
        if p > 1e-9:
            shannon_entropy -= float(p * np.log2(p))

    # Peak Wavelet Transient Ratio: max(|D1|) / max(|AJ|)
    max_d1 = float(np.max(np.abs(dwt_res.details[0]))) if len(dwt_res.details) > 0 else 0.0
    max_aj = float(np.max(np.abs(dwt_res.approx)))
    peak_transient_ratio = float(max_d1 / max(max_aj, 1e-6))

    features = {
        "p_max_mm": round(p_max, 2),
        "total_rainfall_mm": round(p_total, 2),
        "wet_day_freq": round(wet_day_freq, 4),
        "sdii_intensity_mm": round(sdii, 2),
        "r95p_extreme_ratio_pct": round(r95p_ratio, 2),
        "max_cwd_days": float(max_cwd),
        "max_cdd_days": float(max_cdd),
        "coeff_variation": round(cv, 3),
        "convective_energy_ratio_pct": round(conv_ratio, 2),
        "wavelet_shannon_entropy": round(shannon_entropy, 4),
        "detail_to_approx_ratio": round(detail_to_approx, 4),
        "peak_wavelet_transient_ratio": round(peak_transient_ratio, 3),
        "d1_energy_pct": round(float(energies.get("D1_energy_pct", 0.0)), 2),
        "d2_energy_pct": round(float(energies.get("D2_energy_pct", 0.0)), 2),
        "d3_energy_pct": round(float(energies.get("D3_energy_pct", 0.0)), 2),
        "approx_energy_pct": round(float(energies.get("approx_energy_pct", 0.0)), 2),
    }

    return features


def extract_features_cube(
    rain_cube: np.ndarray,
    wavelet: str = "haar",
    level: int = 3,
    wet_threshold: float = 2.5
) -> Dict[str, np.ndarray]:
    """Vectorized feature extraction across full 3D gridded cube (T, lat, lon).

    Returns a dictionary of 2D spatial feature maps of shape (lat, lon).
    """
    T, lat, lon = rain_cube.shape
    nan_mask = np.isnan(rain_cube)
    spatial_all_nan = np.all(nan_mask, axis=0)

    clean_cube = np.where(nan_mask, 0.0, rain_cube)

    # 1. Vectorized Time-Domain Features
    p_max_map = np.max(clean_cube, axis=0)
    p_total_map = np.sum(clean_cube, axis=0)

    wet_mask = clean_cube >= wet_threshold
    wet_days_map = np.sum(wet_mask, axis=0)
    wet_freq_map = wet_days_map / float(T)

    wet_rain_sum = np.sum(np.where(wet_mask, clean_cube, 0.0), axis=0)
    sdii_map = np.where(wet_days_map > 0, wet_rain_sum / np.maximum(wet_days_map, 1), 0.0)

    # 2. Vectorized Wavelet Features via DWT Cube
    dwt_res = dwt_cube(clean_cube, wavelet=wavelet, level=level)
    energies = dwt_res.energies

    conv_ratio_map = energies["convective_transient_ratio_pct"]
    d1_map = energies["D1_energy_pct"]
    d2_map = energies["D2_energy_pct"]
    d3_map = energies["D3_energy_pct"]
    approx_map = energies["approx_energy_pct"]

    # Shannon Entropy Map: -sum(p * log2(p))
    entropy_map = np.zeros((lat, lon), dtype=np.float64)
    all_pct_maps = [approx_map, d1_map, d2_map, d3_map]
    for p_map in all_pct_maps:
        prob = np.maximum(p_map / 100.0, 1e-12)
        valid_prob = p_map > 0.01
        entropy_map += np.where(valid_prob, -prob * np.log2(prob), 0.0)

    # Detail to Approx ratio map
    detail_energy = energies["total_energy"] - energies["approx_energy"]
    detail_approx_map = detail_energy / np.maximum(energies["approx_energy"], 1e-6)

    # Peak transient ratio: max(|D1|) / max(|AJ|)
    max_d1_map = np.max(np.abs(dwt_res.details[0]), axis=0)
    max_aj_map = np.max(np.abs(dwt_res.approx), axis=0)
    peak_trans_map = max_d1_map / np.maximum(max_aj_map, 1e-6)

    # Restore ocean NaNs
    maps = {
        "p_max_mm": p_max_map,
        "total_rainfall_mm": p_total_map,
        "wet_day_freq": wet_freq_map,
        "sdii_intensity_mm": sdii_map,
        "convective_energy_ratio_pct": conv_ratio_map,
        "wavelet_shannon_entropy": entropy_map,
        "detail_to_approx_ratio": detail_approx_map,
        "peak_wavelet_transient_ratio": peak_trans_map,
        "d1_energy_pct": d1_map,
        "d2_energy_pct": d2_map,
        "d3_energy_pct": d3_map,
        "approx_energy_pct": approx_map,
    }

    for name in maps:
        maps[name][spatial_all_nan] = np.nan

    return maps
