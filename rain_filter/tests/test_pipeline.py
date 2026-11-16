"""Tests for the unified denoising pipeline."""

import numpy as np
import pytest
from rain_filter.src.pipeline import denoise


def test_pipeline_modes_and_shapes():
    """Verify all pipeline modes produce correct shape and non-negative output."""
    rng = np.random.default_rng(42)
    x = rng.gamma(shape=0.8, scale=20.0, size=120) * (rng.random(120) < 0.3)

    for mode in ["hybrid", "iir", "fir", "full"]:
        y = denoise(x, mode=mode)
        assert y.shape == x.shape
        assert np.all(y >= 0.0)
        assert not np.any(np.isnan(y))


def test_pipeline_multidimensional_3d():
    """Verify pipeline works on 3D array (T, lat, lon)."""
    T, lat, lon = 40, 3, 4
    rng = np.random.default_rng(555)
    x = rng.gamma(shape=0.8, scale=15.0, size=(T, lat, lon))
    x[5, 1, 2] = np.nan
    x[10, 2, 3] = -999.0

    y = denoise(x, mode="hybrid", preserve_volume=True)
    assert y.shape == (T, lat, lon)
    assert np.isnan(y[5, 1, 2])
    assert np.isnan(y[10, 2, 3])
    assert np.all(y[~np.isnan(y)] >= 0.0)


def test_pipeline_volume_preservation():
    """With preserve_volume=True, total rainfall bias must be < 0.5%."""
    rng = np.random.default_rng(101)
    # 365-day series with rainy monsoon season
    clean = rng.gamma(shape=0.8, scale=30.0, size=365) * (rng.random(365) < 0.35)
    # Add noise
    noisy = np.clip(clean + rng.normal(0, 1.0, 365), 0.0, None)

    y_preserved = denoise(noisy, mode="hybrid", preserve_volume=True)
    bias_pct = abs(np.sum(y_preserved) - np.sum(noisy)) / np.sum(noisy) * 100.0

    assert bias_pct < 0.5, f"Volume bias too high: {bias_pct:.3f}%"
