"""Stage B: Adaptive-Gain IIR Filter (batch, streaming, and Q15 fixed-point).

y[n] = a[n] * x[n] + (1 - a[n]) * y[n-1],  y[0] = x[0]
Innovation: e[n] = x[n] - y[n-1]
Running variance: sigma2[n] = lam * sigma2[n-1] + (1 - lam) * (d[n]^2 / 2)
Adaptive gain: a[n] = clip(e[n]^2 / (e[n]^2 + c * sigma2[n]), a_min, a_max)
"""

from typing import Optional, Tuple, Union
import numpy as np
from rain_filter.src.impulse import _handle_nans


def compute_initial_sigma2(
    x: np.ndarray,
    init_samples: int = 14,
    s_floor: float = 0.5
) -> Union[float, np.ndarray]:
    """Compute initial noise variance from MAD of first differences.

    Parameters
    ----------
    x : np.ndarray
        Input series along axis 0.
    init_samples : int
        Number of samples to use (default 14).
    s_floor : float
        Minimum scale floor (default 0.5 mm).

    Returns
    -------
    sigma2_0 : float or np.ndarray
        Initial noise variance estimate.
    """
    n_pts = min(init_samples + 1, x.shape[0])
    if n_pts < 2:
        return np.maximum((s_floor ** 2) / 2.0, 1e-4)

    # First differences along axis 0
    diffs = np.diff(x[:n_pts], axis=0)

    if x.ndim == 1:
        med_d = np.median(diffs)
        mad_d = np.median(np.abs(diffs - med_d))
        s0 = max(1.4826 * mad_d, s_floor)
        return (s0 ** 2) / 2.0
    else:
        med_d = np.median(diffs, axis=0, keepdims=True)
        mad_d = np.median(np.abs(diffs - med_d), axis=0)
        s0 = np.maximum(1.4826 * mad_d, s_floor)
        return (s0 ** 2) / 2.0


class AdaptiveIIR:
    """Streaming Adaptive-Gain IIR filter with O(1) state per step.

    Parameters
    ----------
    a_min : float
        Minimum filter gain (default 0.08).
    a_max : float
        Maximum filter gain (default 0.95).
    c : float
        Noise scaling constant (default 2.0).
    lam : float
        Memory factor for running variance (default 0.95).
    sigma2_init : float, optional
        Initial noise variance. If None, initialized on first steps or defaults to s_floor^2 / 2.
    s_floor : float
        Noise floor in mm (default 0.5).
    """

    def __init__(
        self,
        a_min: float = 0.08,
        a_max: float = 0.95,
        c: float = 2.0,
        lam: float = 0.95,
        sigma2_init: Optional[float] = None,
        s_floor: float = 0.5
    ):
        self.a_min = float(a_min)
        self.a_max = float(a_max)
        self.c = float(c)
        self.lam = float(lam)
        self.s_floor = float(s_floor)
        self.default_sigma2_init = (s_floor ** 2) / 2.0 if sigma2_init is None else float(sigma2_init)
        self.reset(sigma2_init=sigma2_init)

    def reset(
        self,
        sigma2_init: Optional[float] = None,
        x_init: Optional[np.ndarray] = None
    ) -> None:
        """Reset internal state.

        Parameters
        ----------
        sigma2_init : float, optional
            Initial variance to set.
        x_init : np.ndarray, optional
            Initial samples array to compute initial variance from.
        """
        if x_init is not None:
            self.sigma2 = float(compute_initial_sigma2(x_init, s_floor=self.s_floor))
        elif sigma2_init is not None:
            self.sigma2 = float(sigma2_init)
        else:
            self.sigma2 = float(self.default_sigma2_init)

        self.y_prev: Optional[float] = None
        self.x_prev: Optional[float] = None
        self.step_count = 0
        self.last_a: float = self.a_max

    def step(self, x_n: float) -> float:
        """Process one incoming rainfall sample.

        Parameters
        ----------
        x_n : float
            Current sample value.

        Returns
        -------
        y_n : float
            Current filtered output sample.
        """
        x_val = float(x_n)

        if self.step_count == 0:
            self.y_prev = max(0.0, x_val)
            self.x_prev = x_val
            self.step_count += 1
            self.last_a = self.a_max
            return self.y_prev

        # 1. First difference and running variance update
        d = x_val - self.x_prev
        self.sigma2 = self.lam * self.sigma2 + (1.0 - self.lam) * (0.5 * d * d)

        # 2. Innovation
        e = x_val - self.y_prev
        e2 = e * e

        # 3. Adaptive gain
        denom = e2 + self.c * self.sigma2
        if denom > 1e-12:
            raw_a = e2 / denom
        else:
            raw_a = self.a_min

        a_n = min(self.a_max, max(self.a_min, raw_a))
        self.last_a = a_n

        # 4. State update
        y_n = a_n * x_val + (1.0 - a_n) * self.y_prev
        y_n = max(0.0, y_n)

        self.y_prev = y_n
        self.x_prev = x_val
        self.step_count += 1

        return y_n


def adaptive_iir_filter(
    x: np.ndarray,
    a_min: float = 0.08,
    a_max: float = 0.95,
    c: float = 2.0,
    lam: float = 0.95,
    causal: bool = True,
    sigma2_init: Optional[Union[float, np.ndarray]] = None,
    s_floor: float = 0.5,
    return_gains: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Stage B: Adaptive-Gain IIR filter (batch vectorized implementation).

    Parameters
    ----------
    x : np.ndarray
        Rainfall time series of shape (T,) or (T, ...).
    a_min : float
        Minimum filter gain (default 0.08).
    a_max : float
        Maximum filter gain (default 0.95).
    c : float
        Noise scaling factor (default 2.0).
    lam : float
        Variance forgetting factor (default 0.95).
    causal : bool
        If True, forward causal streaming mode.
        If False, forward-backward zero-phase mode for offline evaluation.
    sigma2_init : float or np.ndarray, optional
        Pre-computed initial noise variance. If None, computed from first ~14 samples.
    s_floor : float
        Scale floor in mm (default 0.5).
    return_gains : bool
        If True, also returns array of adaptive gains a[n].

    Returns
    -------
    y : np.ndarray
        Filtered series, non-negative, same shape as input.
    gains : np.ndarray (optional)
        Array of adaptive gains a[n] if return_gains is True.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    t_len = x_arr.shape[0]

    if t_len == 0:
        return (np.array([]), np.array([])) if return_gains else np.array([])
    if t_len == 1:
        y1 = np.clip(x_arr, 0.0, None)
        return (y1, np.full_like(y1, a_max)) if return_gains else y1

    x_clean, nan_mask, all_nan_mask = _handle_nans(x_arr)

    def _forward_pass(data: np.ndarray, init_s2: Optional[Union[float, np.ndarray]] = None):
        t_steps = data.shape[0]
        y_out = np.zeros_like(data)
        gains_out = np.zeros_like(data)

        y_out[0] = np.clip(data[0], 0.0, None)
        gains_out[0] = a_max

        if init_s2 is None:
            s2_curr = compute_initial_sigma2(data, init_samples=14, s_floor=s_floor)
        else:
            s2_curr = np.copy(init_s2)

        x_prev = data[0]
        y_prev = y_out[0]

        for n in range(1, t_steps):
            x_n = data[n]
            d = x_n - x_prev
            s2_curr = lam * s2_curr + (1.0 - lam) * (0.5 * d * d)

            e = x_n - y_prev
            e2 = e * e
            denom = e2 + c * s2_curr

            raw_a = np.where(denom > 1e-12, e2 / denom, a_min)
            a_n = np.clip(raw_a, a_min, a_max)

            y_n = a_n * x_n + (1.0 - a_n) * y_prev
            y_n = np.clip(y_n, 0.0, None)

            y_out[n] = y_n
            gains_out[n] = a_n

            x_prev = x_n
            y_prev = y_n

        return y_out, gains_out

    # Forward filtering
    y_fwd, gains_fwd = _forward_pass(x_clean, init_s2=sigma2_init)

    if causal:
        y_result = y_fwd
        gains_result = gains_fwd
    else:
        # Backward filtering for zero-phase response
        # Reverse forward filtered signal, filter again, and reverse back
        y_rev, _ = _forward_pass(y_fwd[::-1], init_s2=None)
        y_result = y_rev[::-1]
        gains_result = gains_fwd

    # Ensure non-negative
    y_result = np.clip(y_result, 0.0, None)

    # Restore NaNs
    y_result[nan_mask] = np.nan
    if x_arr.ndim > 1:
        y_result[:, all_nan_mask] = np.nan

    if return_gains:
        return y_result, gains_result
    return y_result


def adaptive_iir_q15(
    x: np.ndarray,
    a_min: float = 0.08,
    a_max: float = 0.95,
    c: float = 2.0,
    lam: float = 0.95,
    x_max_scale: float = 512.0,
    s_floor: float = 0.5
) -> np.ndarray:
    """Stage B: Fixed-point Q15 (int16) version using integer arithmetic only.

    Inputs are normalized by x_max_scale and represented in Q15 format:
    [-32768, 32767]. Intermediate operations use integer arithmetic.

    Parameters
    ----------
    x : np.ndarray
        1D rainfall series.
    a_min : float
        Minimum gain (default 0.08).
    a_max : float
        Maximum gain (default 0.95).
    c : float
        Noise scaling factor (default 2.0).
    lam : float
        Memory factor (default 0.95).
    x_max_scale : float
        Physical scale mapped to 32767 (default 512.0 mm).
    s_floor : float
        Noise floor in mm (default 0.5).

    Returns
    -------
    y_float : np.ndarray
        Filtered output series converted back to physical mm units.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    x_clean, nan_mask, _ = _handle_nans(x_arr)
    t_len = len(x_clean)

    if t_len == 0:
        return np.array([])

    SCALE_Q15 = 32767

    # Convert input to Q15 integer format
    x_q15 = np.clip(np.round((x_clean / x_max_scale) * SCALE_Q15), 0, 32767).astype(np.int32)
    y_q15 = np.zeros(t_len, dtype=np.int32)

    # Compute initial variance in Q15
    sigma2_0 = compute_initial_sigma2(x_clean, init_samples=14, s_floor=s_floor)
    s2_0_norm = sigma2_0 / (x_max_scale ** 2)
    s2_q15 = int(round(s2_0_norm * SCALE_Q15))

    # Pre-compute integer constants
    lam_q15 = int(round(lam * 32768))
    one_minus_lam_q15 = 32768 - lam_q15
    a_min_q15 = int(round(a_min * SCALE_Q15))
    a_max_q15 = int(round(a_max * SCALE_Q15))
    c_int = int(round(c))

    y_q15[0] = x_q15[0]

    for n in range(1, t_len):
        xn = int(x_q15[n])
        xn_prev = int(x_q15[n - 1])
        yn_prev = int(y_q15[n - 1])

        # Difference and running variance in Q15
        d = xn - xn_prev
        d2_half = (d * d) >> 16
        s2_q15 = (lam_q15 * s2_q15 + one_minus_lam_q15 * d2_half) >> 15

        # Innovation in Q15
        e = xn - yn_prev
        e2 = (e * e) >> 15

        denom = e2 + c_int * s2_q15
        if denom > 0:
            raw_a = (e2 << 15) // denom
        else:
            raw_a = a_min_q15

        a_cur = max(a_min_q15, min(a_max_q15, raw_a))

        # State update in Q15: y = (a * x + (1 - a) * y_prev)
        prod = a_cur * xn + (SCALE_Q15 - a_cur) * yn_prev
        y_q15[n] = max(0, prod >> 15)

    # Convert back to physical mm float
    y_float = (y_q15.astype(np.float64) / SCALE_Q15) * x_max_scale
    y_float = np.clip(y_float, 0.0, None)
    y_float[nan_mask] = np.nan

    return y_float
