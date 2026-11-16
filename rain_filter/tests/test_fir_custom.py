"""Tests for Stage C: Custom FIR filter."""

import numpy as np
import pytest
from rain_filter.src.fir_custom import (
    bessel_i0_custom,
    custom_fir_filter,
    design_fir_taps,
    hann_window_custom,
    kaiser_window_custom,
)


def test_bessel_i0_accuracy():
    """Verify Taylor series I0 matches known values (I0(0)=1, I0(2)~2.2795853)."""
    assert bessel_i0_custom(0.0) == pytest.approx(1.0, abs=1e-12)
    assert bessel_i0_custom(2.0) == pytest.approx(2.279585302336, rel=1e-8)


def test_taps_properties():
    """Taps must be symmetric, sum to 1.0, and have linear phase group delay."""
    N = 11
    taps, group_delay = design_fir_taps(num_taps=N, fc=0.35, beta=4.0)

    # 1. Sum must be 1.0 (unit DC gain)
    assert np.sum(taps) == pytest.approx(1.0, abs=1e-12)

    # 2. Symmetric taps: h[k] == h[N - 1 - k]
    assert np.allclose(taps, taps[::-1], atol=1e-14)

    # 3. Group delay must equal (N - 1) / 2
    assert group_delay == (N - 1) / 2.0


def test_matches_direct_manual_convolution():
    """Applying custom_fir_filter must match manual direct convolution loop."""
    rng = np.random.default_rng(777)
    x = rng.gamma(shape=1.0, scale=20.0, size=50)

    num_taps = 11
    M = (num_taps - 1) // 2
    taps, _ = design_fir_taps(num_taps=num_taps, fc=0.35, beta=4.0)

    # Manual convolution with reflect padding
    pad_x = np.pad(x, M, mode="reflect")
    y_manual = np.zeros_like(x)
    for n in range(len(x)):
        y_manual[n] = np.sum(pad_x[n : n + num_taps] * taps)
    y_manual = np.clip(y_manual, 0.0, None)

    # Vectorized function
    y_filt = custom_fir_filter(x, num_taps=num_taps, fc=0.35, beta=4.0)

    assert np.allclose(y_filt, y_manual, atol=1e-12)


def test_stage_c_nan_handling():
    """NaN and -999 values are masked and restored properly."""
    x = np.array([5.0, 10.0, np.nan, 20.0, -999.0, 15.0, 0.0, 12.0, 8.0, 3.0, 1.0, 0.0])
    y = custom_fir_filter(x, num_taps=5)

    assert np.isnan(y[2])
    assert np.isnan(y[4])
    valid_idx = [0, 1, 3, 5, 6, 7, 8, 9, 10, 11]
    assert np.all(~np.isnan(y[valid_idx]))
    assert np.all(y[valid_idx] >= 0.0)


def test_stage_c_multidimensional():
    """Check 3D array (T, lat, lon)."""
    T, lat, lon = 25, 2, 3
    rng = np.random.default_rng(888)
    x = rng.gamma(shape=0.9, scale=15.0, size=(T, lat, lon))
    y = custom_fir_filter(x, num_taps=11)

    assert y.shape == (T, lat, lon)
    assert np.all(y >= 0.0)
    assert not np.any(np.isnan(y))
