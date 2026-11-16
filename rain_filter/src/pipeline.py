"""Pipeline module orchestrating the denoising filter stages.

Modes:
- "hybrid": Stage A (Hampel impulse) -> Stage B (Adaptive IIR) [default]
- "iir": Stage B only
- "fir": Stage C only
- "full": Stage A -> Stage B -> Stage C

Optional volume preservation:
- preserve_volume=True: rescales each non-overlapping 30-day block so its sum
  matches the input block sum (skipping blocks where input sum is 0).
"""

from typing import Optional, Union
import numpy as np
from rain_filter.src.adaptive_iir import adaptive_iir_filter
from rain_filter.src.fir_custom import custom_fir_filter
from rain_filter.src.impulse import impulse_filter


def _apply_volume_preservation(
    x_in: np.ndarray,
    x_out: np.ndarray,
    block_size: int = 30
) -> np.ndarray:
    """Rescale each non-overlapping 30-day block so sum(x_out) matches sum(x_in).

    Parameters
    ----------
    x_in : np.ndarray
        Original input array before filtering (axis 0 is time).
    x_out : np.ndarray
        Filtered array before volume preservation.
    block_size : int
        Size of non-overlapping blocks in days (default 30).

    Returns
    -------
    x_rescaled : np.ndarray
        Volume-preserved filtered array.
    """
    res = x_out.copy()
    t_len = x_in.shape[0]

    num_blocks = int(np.ceil(t_len / block_size))

    for b in range(num_blocks):
        start = b * block_size
        end = min((b + 1) * block_size, t_len)

        # Slice along time axis
        blk_in = x_in[start:end]
        blk_out = res[start:end]

        # Valid mask (non-NaN and finite)
        valid = ~np.isnan(blk_in) & (blk_in >= 0.0) & ~np.isnan(blk_out) & (blk_out >= 0.0)

        if x_in.ndim == 1:
            if not np.any(valid):
                continue
            sum_in = float(np.sum(blk_in[valid]))
            sum_out = float(np.sum(blk_out[valid]))
            if sum_out > 1e-6:
                scale = sum_in / sum_out
                res[start:end] = blk_out * scale
        else:
            # Multi-dimensional (e.g. T, lat, lon)
            blk_in_zeroed = np.where(valid, blk_in, 0.0)
            blk_out_zeroed = np.where(valid, blk_out, 0.0)

            sum_in = np.sum(blk_in_zeroed, axis=0, keepdims=True)
            sum_out = np.sum(blk_out_zeroed, axis=0, keepdims=True)

            can_scale = sum_out > 1e-6
            scale = np.where(can_scale, sum_in / np.maximum(sum_out, 1e-12), 1.0)
            res[start:end] = blk_out * scale

    return np.clip(res, 0.0, None)


def denoise(
    x: np.ndarray,
    mode: str = "hybrid",
    preserve_volume: bool = False,
    causal: bool = True,
    # Stage A parameters
    window_a: int = 7,
    k_a: float = 4.0,
    s_floor_a: float = 0.5,
    isolation_ratio_a: float = 0.35,
    # Stage B parameters
    a_min: float = 0.08,
    a_max: float = 0.95,
    c_b: float = 2.0,
    lam_b: float = 0.95,
    s_floor_b: float = 0.5,
    # Stage C parameters
    num_taps_c: int = 11,
    fc_c: float = 0.35,
    beta_c: float = 4.0
) -> np.ndarray:
    """Execute rainfall denoising filter pipeline.

    Parameters
    ----------
    x : np.ndarray
        Rainfall input array with shape (T,) or (T, ...).
    mode : str
        Pipeline mode:
        - "hybrid": Stage A -> Stage B (default)
        - "iir": Stage B only
        - "fir": Stage C only
        - "full": Stage A -> Stage B -> Stage C
    preserve_volume : bool
        If True, rescales each non-overlapping 30-day block to match input sum.
    causal : bool
        Whether Stage B operates causally (default True).
    window_a, k_a, s_floor_a :
        Parameters for Stage A.
    a_min, a_max, c_b, lam_b, s_floor_b :
        Parameters for Stage B.
    num_taps_c, fc_c, beta_c :
        Parameters for Stage C.

    Returns
    -------
    y : np.ndarray
        Denoised rainfall array of same shape, non-negative, with NaNs restored.
    """
    valid_modes = {"hybrid", "iir", "fir", "full"}
    mode_lower = mode.lower()
    if mode_lower not in valid_modes:
        raise ValueError(f"Unknown mode '{mode}'. Choose from {valid_modes}")

    x_arr = np.asarray(x, dtype=np.float64)
    nan_mask = np.isnan(x_arr) | (x_arr <= -900.0)

    # 1. Stage A
    if mode_lower in {"hybrid", "full"}:
        cur = impulse_filter(
            x_arr,
            window=window_a,
            k=k_a,
            s_floor=s_floor_a,
            isolation_ratio=isolation_ratio_a
        )
    else:
        cur = x_arr.copy()

    # 2. Stage B
    if mode_lower in {"hybrid", "iir", "full"}:
        cur = adaptive_iir_filter(
            cur,
            a_min=a_min,
            a_max=a_max,
            c=c_b,
            lam=lam_b,
            causal=causal,
            s_floor=s_floor_b
        )

    # 3. Stage C
    if mode_lower in {"fir", "full"}:
        cur = custom_fir_filter(
            cur,
            num_taps=num_taps_c,
            fc=fc_c,
            beta=beta_c
        )

    # 4. Optional volume preservation
    if preserve_volume:
        cur = _apply_volume_preservation(x_arr, cur, block_size=30)

    # Ensure non-negativity
    cur = np.clip(cur, 0.0, None)

    # Restore NaNs
    cur[nan_mask] = np.nan

    return cur
