"""Discrete Wavelet Transform (DWT) module for satellite edge rainfall processing.

Provides multi-resolution analysis of daily rainfall time series:
- Supported wavelets: 'haar' (db1), 'db2', 'db4', and 'cdf53' (lifting scheme).
- Multi-level forward and inverse decomposition with perfect reconstruction (< 1e-12).
- Boundary handling supporting arbitrary lengths (e.g. T=365 or 366).
- Parseval energy conservation and subband energy ratio extraction.
- Edge-optimized in-place lifting scheme for CDF 5/3 (integer & float).
- Vectorized execution over 1D series and 3D gridded cubes (T, lat, lon).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


# Standard Daubechies orthogonal scaling filter coefficients
_WAVELET_FILTERS = {
    "haar": np.array([1.0, 1.0]) / np.sqrt(2.0),
    "db1": np.array([1.0, 1.0]) / np.sqrt(2.0),
    "db2": np.array([
        1.0 + np.sqrt(3.0),
        3.0 + np.sqrt(3.0),
        3.0 - np.sqrt(3.0),
        1.0 - np.sqrt(3.0)
    ]) / (4.0 * np.sqrt(2.0)),
    "db4": np.array([
        -0.010597401785069032,
         0.0328830116668852,
         0.030841381835560764,
        -0.18703481171909309,
        -0.027983769416859854,
         0.6308807679298589,
         0.7148465705529157,
         0.2303778133088965
    ])
}


@dataclass
class DWTResult:
    """Container for multi-level DWT decomposition results."""
    approx: np.ndarray
    details: List[np.ndarray]  # [D_1, D_2, ..., D_level]
    wavelet: str
    level: int
    lengths: List[int]
    original_shape: Tuple[int, ...]
    energies: Dict[str, Union[float, np.ndarray]]


def _get_wavelet_filters(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieve scaling lowpass (h) and wavelet highpass (g) filters."""
    key = name.lower()
    if key not in _WAVELET_FILTERS:
        raise ValueError(f"Unsupported filter bank wavelet '{name}'. Supported: {list(_WAVELET_FILTERS.keys())}")
    h = _WAVELET_FILTERS[key]
    L = len(h)
    k = np.arange(L)
    g = ((-1.0) ** k) * h[::-1]
    return h, g


# ---------------------------------------------------------------------------
# CDF 5/3 Lifting Scheme (In-place, Reversible, Edge-Optimized)
# ---------------------------------------------------------------------------

def lifting_cdf53_forward(
    x: np.ndarray,
    integer_mode: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """1-level CDF 5/3 forward transform via predict-update lifting scheme.

    Parameters
    ----------
    x : np.ndarray
        Input array with time along axis 0. Length along axis 0 must be even.
    integer_mode : bool
        If True, uses integer arithmetic with bit shifts and floor division.

    Returns
    -------
    a : np.ndarray
        Approximation coefficients.
    d : np.ndarray
        Detail coefficients.
    """
    xe = x[0::2].copy()
    xo = x[1::2].copy()
    t_half = xe.shape[0]

    # Predict step: d[n] = xo[n] - 0.5 * (xe[n] + xe[n+1])
    # Symmetric boundary extension on right: xe[n+1] = xe[n] at edge
    if x.ndim == 1:
        xe_ext = np.pad(xe, (0, 1), mode="edge")
        if integer_mode:
            d = xo.astype(np.int64) - ((xe.astype(np.int64) + xe_ext[1:].astype(np.int64)) >> 1)
        else:
            d = xo - 0.5 * (xe + xe_ext[1:])

        # Update step: a[n] = xe[n] + 0.25 * (d[n-1] + d[n])
        d_ext = np.pad(d, (1, 0), mode="edge")
        if integer_mode:
            a = xe.astype(np.int64) + ((d_ext[:-1].astype(np.int64) + d.astype(np.int64) + 2) >> 2)
        else:
            a = xe + 0.25 * (d_ext[:-1] + d)
    else:
        # Multi-dimensional: axis 0 is time
        pad_spec_r = [(0, 1)] + [(0, 0)] * (x.ndim - 1)
        xe_ext = np.pad(xe, pad_spec_r, mode="edge")
        d = xo - 0.5 * (xe + xe_ext[1:])

        pad_spec_l = [(1, 0)] + [(0, 0)] * (x.ndim - 1)
        d_ext = np.pad(d, pad_spec_l, mode="edge")
        a = xe + 0.25 * (d_ext[:-1] + d)

    return a, d


def lifting_cdf53_inverse(
    a: np.ndarray,
    d: np.ndarray,
    target_len: Optional[int] = None,
    integer_mode: bool = False
) -> np.ndarray:
    """1-level CDF 5/3 inverse transform via reverse lifting.

    Parameters
    ----------
    a : np.ndarray
        Approximation coefficients.
    d : np.ndarray
        Detail coefficients.
    target_len : int, optional
        Target length along axis 0.
    integer_mode : bool
        If True, uses integer arithmetic.

    Returns
    -------
    x : np.ndarray
        Reconstructed signal.
    """
    m = a.shape[0]

    # Inverse update: xe[n] = a[n] - 0.25 * (d[n-1] + d[n])
    if a.ndim == 1:
        d_ext = np.pad(d, (1, 0), mode="edge")
        if integer_mode:
            xe = a.astype(np.int64) - ((d_ext[:-1].astype(np.int64) + d.astype(np.int64) + 2) >> 2)
        else:
            xe = a - 0.25 * (d_ext[:-1] + d)

        # Inverse predict: xo[n] = d[n] + 0.5 * (xe[n] + xe[n+1])
        xe_ext = np.pad(xe, (0, 1), mode="edge")
        if integer_mode:
            xo = d.astype(np.int64) + ((xe.astype(np.int64) + xe_ext[1:].astype(np.int64)) >> 1)
        else:
            xo = d + 0.5 * (xe + xe_ext[1:])

        x = np.zeros(2 * m, dtype=a.dtype)
        x[0::2] = xe
        x[1::2] = xo
    else:
        pad_spec_l = [(1, 0)] + [(0, 0)] * (a.ndim - 1)
        d_ext = np.pad(d, pad_spec_l, mode="edge")
        xe = a - 0.25 * (d_ext[:-1] + d)

        pad_spec_r = [(0, 1)] + [(0, 0)] * (a.ndim - 1)
        xe_ext = np.pad(xe, pad_spec_r, mode="edge")
        xo = d + 0.5 * (xe + xe_ext[1:])

        out_shape = (2 * m,) + a.shape[1:]
        x = np.zeros(out_shape, dtype=a.dtype)
        x[0::2] = xe
        x[1::2] = xo

    if target_len is not None and target_len < x.shape[0]:
        x = x[:target_len]

    return x


# ---------------------------------------------------------------------------
# Orthogonal Filter Bank DWT & IDWT (Haar, db2, db4)
# ---------------------------------------------------------------------------

def _dwt_single_level_filterbank(
    x: np.ndarray,
    h: np.ndarray,
    g: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """1-level forward DWT via periodic orthogonal convolution."""
    n = x.shape[0]
    m = n // 2
    L = len(h)

    # For Haar (length 2), fast vectorization
    if L == 2:
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        a = (x[0::2] + x[1::2]) * inv_sqrt2
        d = (x[1::2] - x[0::2]) * inv_sqrt2
        return a, d

    if x.ndim == 1:
        a = np.zeros(m, dtype=np.float64)
        d = np.zeros(m, dtype=np.float64)
        for k in range(m):
            idx = (2 * k + np.arange(L)) % n
            a[k] = np.sum(h * x[idx])
            d[k] = np.sum(g * x[idx])
        return a, d
    else:
        # Multi-dimensional: axis 0 is time
        out_shape = (m,) + x.shape[1:]
        a = np.zeros(out_shape, dtype=np.float64)
        d = np.zeros(out_shape, dtype=np.float64)
        k_idx = np.arange(L)
        for k in range(m):
            idx = (2 * k + k_idx) % n
            # x[idx] has shape (L, lat, lon)
            a[k] = np.tensordot(h, x[idx], axes=(0, 0))
            d[k] = np.tensordot(g, x[idx], axes=(0, 0))
        return a, d


def _idwt_single_level_filterbank(
    a: np.ndarray,
    d: np.ndarray,
    h: np.ndarray,
    g: np.ndarray,
    target_len: Optional[int] = None
) -> np.ndarray:
    """1-level inverse DWT via periodic orthogonal synthesis."""
    m = a.shape[0]
    n = 2 * m
    L = len(h)

    # Fast Haar synthesis
    if L == 2:
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        if a.ndim == 1:
            x_rec = np.zeros(n, dtype=np.float64)
            x_rec[0::2] = (a - d) * inv_sqrt2
            x_rec[1::2] = (a + d) * inv_sqrt2
        else:
            out_shape = (n,) + a.shape[1:]
            x_rec = np.zeros(out_shape, dtype=np.float64)
            x_rec[0::2] = (a - d) * inv_sqrt2
            x_rec[1::2] = (a + d) * inv_sqrt2
        if target_len is not None and target_len < n:
            x_rec = x_rec[:target_len]
        return x_rec

    if a.ndim == 1:
        x_rec = np.zeros(n, dtype=np.float64)
        m_idx = np.arange(m)
        for i in range(n):
            tap_idx = (i - 2 * m_idx) % n
            valid = tap_idx < L
            x_rec[i] = np.sum(a[valid] * h[tap_idx[valid]]) + np.sum(d[valid] * g[tap_idx[valid]])
    else:
        out_shape = (n,) + a.shape[1:]
        x_rec = np.zeros(out_shape, dtype=np.float64)
        m_idx = np.arange(m)
        for i in range(n):
            tap_idx = (i - 2 * m_idx) % n
            valid = tap_idx < L
            v_idx = np.where(valid)[0]
            if len(v_idx) > 0:
                h_coeffs = h[tap_idx[v_idx]]
                g_coeffs = g[tap_idx[v_idx]]
                term_a = np.tensordot(h_coeffs, a[v_idx], axes=(0, 0))
                term_d = np.tensordot(g_coeffs, d[v_idx], axes=(0, 0))
                x_rec[i] = term_a + term_d

    if target_len is not None and target_len < n:
        x_rec = x_rec[:target_len]

    return x_rec


# ---------------------------------------------------------------------------
# Multi-Level DWT & IDWT APIs
# ---------------------------------------------------------------------------

def dwt_1d(
    x: np.ndarray,
    wavelet: str = "haar",
    level: int = 3
) -> DWTResult:
    """Compute multi-level 1D Discrete Wavelet Transform.

    Parameters
    ----------
    x : np.ndarray
        Input signal (shape (T,)).
    wavelet : str
        Wavelet type: 'haar' (default), 'db2', 'db4', or 'cdf53'.
    level : int
        Decomposition level (default 3).

    Returns
    -------
    result : DWTResult
        Structured object containing approximation, detail subbands, and energy metrics.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    orig_shape = x_arr.shape
    t_len = orig_shape[0]

    if level < 1:
        raise ValueError(f"Decomposition level must be >= 1, got {level}")

    wav_name = wavelet.lower()
    is_lifting = (wav_name == "cdf53")
    if not is_lifting:
        h, g = _get_wavelet_filters(wav_name)

    current = x_arr.copy()
    details = []
    lengths = [current.shape[0]]

    for lvl in range(level):
        cur_len = current.shape[0]
        # Ensure even length by padding 1 sample if odd
        if cur_len % 2 != 0:
            if current.ndim == 1:
                current = np.pad(current, (0, 1), mode="edge")
            else:
                pad_spec = [(0, 1)] + [(0, 0)] * (current.ndim - 1)
                current = np.pad(current, pad_spec, mode="edge")

        if is_lifting:
            a, d = lifting_cdf53_forward(current)
        else:
            a, d = _dwt_single_level_filterbank(current, h, g)

        details.append(d)
        current = a
        lengths.append(current.shape[0])

    # Compute energy distribution
    energies = compute_subband_energies(current, details)

    return DWTResult(
        approx=current,
        details=details,
        wavelet=wav_name,
        level=level,
        lengths=lengths,
        original_shape=orig_shape,
        energies=energies
    )


def idwt_1d(dwt_result: DWTResult) -> np.ndarray:
    """Compute multi-level Inverse Discrete Wavelet Transform.

    Parameters
    ----------
    dwt_result : DWTResult
        Output object from dwt_1d.

    Returns
    -------
    x_rec : np.ndarray
        Reconstructed time series matching original shape.
    """
    wav_name = dwt_result.wavelet.lower()
    is_lifting = (wav_name == "cdf53")
    if not is_lifting:
        h, g = _get_wavelet_filters(wav_name)

    current = dwt_result.approx
    level = len(dwt_result.details)
    lengths = dwt_result.lengths

    for lvl in range(level - 1, -1, -1):
        d = dwt_result.details[lvl]
        target_len = lengths[lvl]

        if is_lifting:
            current = lifting_cdf53_inverse(current, d, target_len=target_len)
        else:
            current = _idwt_single_level_filterbank(current, d, h, g, target_len=target_len)

    orig_t = dwt_result.original_shape[0]
    if current.shape[0] > orig_t:
        current = current[:orig_t]

    return current


# ---------------------------------------------------------------------------
# Energy & Feature Analysis
# ---------------------------------------------------------------------------

def compute_subband_energies(
    approx: np.ndarray,
    details: List[np.ndarray]
) -> Dict[str, Union[float, np.ndarray]]:
    """Compute subband energy and relative power percentages.

    Returns Parseval energy metrics across levels:
    - E_total = sum(A_J^2) + sum(sum(D_j^2))
    - percentage of energy in high-frequency convective subbands (D_1, D_2)
    - percentage of energy in synoptic seasonal subbands (A_J)
    """
    if approx.ndim == 1:
        e_approx = float(np.sum(approx ** 2))
        e_details = [float(np.sum(d ** 2)) for d in details]
        e_total = e_approx + sum(e_details)
        eps = 1e-12

        pct_approx = (e_approx / max(e_total, eps)) * 100.0
        pct_details = [(ed / max(e_total, eps)) * 100.0 for ed in e_details]

        res = {
            "total_energy": e_total,
            "approx_energy": e_approx,
            "approx_energy_pct": pct_approx,
        }
        for j, (ed, pct) in enumerate(zip(e_details, pct_details), 1):
            res[f"D{j}_energy"] = ed
            res[f"D{j}_energy_pct"] = pct

        # High-frequency transient convective energy (D1 + D2)
        transient_e = e_details[0] + (e_details[1] if len(e_details) > 1 else 0.0)
        res["convective_transient_ratio_pct"] = (transient_e / max(e_total, eps)) * 100.0
        return res
    else:
        # Multi-dimensional: energy per spatial cell along axis 0
        e_approx = np.sum(approx ** 2, axis=0)
        e_details = [np.sum(d ** 2, axis=0) for d in details]
        e_total = e_approx + sum(e_details)
        eps = 1e-12

        res = {
            "total_energy": e_total,
            "approx_energy": e_approx,
            "approx_energy_pct": (e_approx / np.maximum(e_total, eps)) * 100.0,
        }
        for j, ed in enumerate(e_details, 1):
            res[f"D{j}_energy"] = ed
            res[f"D{j}_energy_pct"] = (ed / np.maximum(e_total, eps)) * 100.0

        transient_e = e_details[0] + (e_details[1] if len(e_details) > 1 else 0.0)
        res["convective_transient_ratio_pct"] = (transient_e / np.maximum(e_total, eps)) * 100.0
        return res


# ---------------------------------------------------------------------------
# Vectorized 3D Cube Processing
# ---------------------------------------------------------------------------

def dwt_cube(
    rain_cube: np.ndarray,
    wavelet: str = "haar",
    level: int = 3
) -> DWTResult:
    """Compute DWT across full 3D gridded cube (T, lat, lon).

    Parameters
    ----------
    rain_cube : np.ndarray
        Gridded rainfall array of shape (T, lat, lon).
    wavelet : str
        Wavelet ('haar', 'db2', 'db4', 'cdf53').
    level : int
        Decomposition level (default 3).

    Returns
    -------
    result : DWTResult
        Decomposition result with 3D subband arrays and 2D energy maps.
    """
    nan_mask = np.isnan(rain_cube)
    clean_cube = np.where(nan_mask, 0.0, rain_cube)

    res = dwt_1d(clean_cube, wavelet=wavelet, level=level)

    # Restore NaN masks on 2D energy maps
    spatial_all_nan = np.all(nan_mask, axis=0)
    for k in res.energies:
        if isinstance(res.energies[k], np.ndarray) and res.energies[k].shape == spatial_all_nan.shape:
            res.energies[k][spatial_all_nan] = np.nan

    return res


def idwt_cube(dwt_result: DWTResult) -> np.ndarray:
    """Reconstruct 3D cube from DWT result."""
    return idwt_1d(dwt_result)
