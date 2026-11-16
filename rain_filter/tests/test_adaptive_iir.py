"""Tests for Stage B: Adaptive-Gain IIR filter."""

import numpy as np
import pytest
from rain_filter.src.adaptive_iir import (
    AdaptiveIIR,
    adaptive_iir_filter,
    adaptive_iir_q15,
    compute_initial_sigma2,
)


def test_constant_input_yields_constant_output():
    """Constant input must produce constant output."""
    val = 15.0
    x = np.full(50, val, dtype=np.float64)
    y = adaptive_iir_filter(x, causal=True)
    assert np.allclose(y, val, atol=1e-10)

    # Check zero input
    x_zero = np.zeros(50, dtype=np.float64)
    y_zero = adaptive_iir_filter(x_zero, causal=True)
    assert np.allclose(y_zero, 0.0, atol=1e-12)


def test_step_input_converges():
    """Step input must converge to the step value."""
    x = np.zeros(60, dtype=np.float64)
    x[10:] = 40.0  # step of 40 mm at t=10

    y = adaptive_iir_filter(x, causal=True)
    # At t=59, filter should have fully converged to 40.0
    assert y[-1] == pytest.approx(40.0, rel=1e-3)
    # Output must be non-negative
    assert np.all(y >= 0.0)


def test_streaming_matches_batch_exact():
    """Streaming AdaptiveIIR output must match batch output to 1e-12."""
    rng = np.random.default_rng(123)
    x = rng.gamma(shape=1.0, scale=15.0, size=100) * (rng.random(100) < 0.4)

    # Compute batch output
    y_batch = adaptive_iir_filter(x, causal=True)

    # Run streaming with matching initial variance
    s2_0 = compute_initial_sigma2(x, init_samples=14, s_floor=0.5)
    streamer = AdaptiveIIR(sigma2_init=s2_0, s_floor=0.5)
    y_stream = np.array([streamer.step(val) for val in x])

    assert np.allclose(y_stream, y_batch, atol=1e-12)


def test_non_causal_zero_phase_mode():
    """Test causal=False runs forward-backward without error and preserves shape."""
    x = np.array([0.0, 5.0, 30.0, 100.0, 20.0, 0.0, 0.0, 50.0, 0.0])
    y_fwd = adaptive_iir_filter(x, causal=True)
    y_zp = adaptive_iir_filter(x, causal=False)

    assert y_zp.shape == x.shape
    assert np.all(y_zp >= 0.0)


def test_stage_b_nan_handling():
    """NaN and -999 values are masked and restored properly."""
    x = np.array([10.0, 12.0, np.nan, 15.0, -999.0, 20.0, 18.0])
    y = adaptive_iir_filter(x, causal=True)

    assert np.isnan(y[2])
    assert np.isnan(y[4])
    valid_idx = [0, 1, 3, 5, 6]
    assert np.all(~np.isnan(y[valid_idx]))
    assert np.all(y[valid_idx] >= 0.0)


def test_stage_b_multidimensional():
    """Vectorized 3D array (T, lat, lon)."""
    T, lat, lon = 30, 2, 3
    rng = np.random.default_rng(42)
    x = rng.gamma(shape=0.8, scale=20.0, size=(T, lat, lon))
    y = adaptive_iir_filter(x, causal=True)

    assert y.shape == (T, lat, lon)
    assert np.all(y >= 0.0)
    assert not np.any(np.isnan(y))


def test_q15_vs_float64_snr_loss():
    """Q15 fixed-point vs float64 SNR loss must be < 1 dB on test series."""
    rng = np.random.default_rng(999)
    clean = rng.gamma(shape=0.75, scale=25.0, size=365) * (rng.random(365) < 0.3)
    noisy = np.clip(clean + rng.normal(0, 1.5, size=365), 0.0, None)

    y_f64 = adaptive_iir_filter(noisy, causal=True)
    y_q15 = adaptive_iir_q15(noisy)

    # Metric: SNR loss = SNR(f64) - SNR(q15)
    mse_f64 = np.mean((y_f64 - clean) ** 2)
    mse_q15 = np.mean((y_q15 - clean) ** 2)
    snr_f64 = 10.0 * np.log10(np.mean(clean ** 2) / max(mse_f64, 1e-10))
    snr_q15 = 10.0 * np.log10(np.mean(clean ** 2) / max(mse_q15, 1e-10))

    snr_loss = snr_f64 - snr_q15
    # Must be less than 1.0 dB
    assert snr_loss < 1.0
    assert np.all(y_q15 >= 0.0)
