"""Tests for Stage A: Isolated-Impulse Suppressor."""

import numpy as np
import pytest
from rain_filter.src.impulse import impulse_filter


def test_isolated_spike_removed():
    """An isolated 50 mm spike on zeros must be removed."""
    x = np.zeros(30, dtype=np.float64)
    x[15] = 50.0  # isolated spike
    y = impulse_filter(x, window=7, k=4.0, s_floor=0.5)

    assert y[15] == pytest.approx(0.0, abs=1e-6)
    assert np.all(y >= 0.0)


def test_three_day_event_preserved():
    """A 3-day 50 mm event must be preserved (not flagged as noise)."""
    x = np.zeros(30, dtype=np.float64)
    x[14:17] = 50.0  # 3 consecutive days of 50 mm rain
    y = impulse_filter(x, window=7, k=4.0, s_floor=0.5)

    # Multi-day convective rain should be intact
    assert np.all(y[14:17] == pytest.approx(50.0, rel=1e-5))


def test_stage_a_edge_cases_and_nonnegativity():
    """Test non-negativity, edge samples, and zero input."""
    x = np.zeros(15, dtype=np.float64)
    y = impulse_filter(x)
    assert np.all(y == 0.0)

    # Spike at first sample (edge handling with reflect padding)
    x_edge = np.zeros(20, dtype=np.float64)
    x_edge[0] = 60.0
    y_edge = impulse_filter(x_edge)
    # Reflect padding at edge: [x1, x0, x1] -> prev is x1 (0), next is x1 (0). So it gets suppressed!
    assert y_edge[0] == pytest.approx(0.0, abs=1e-6)


def test_stage_a_nan_handling():
    """Test NaN and -999 masking and restoration."""
    x = np.array([0.0, 5.0, 10.0, np.nan, 8.0, 50.0, 2.0, -999.0, 0.0])
    y = impulse_filter(x, window=5)

    # NaNs and -999 must be restored as NaN
    assert np.isnan(y[3])
    assert np.isnan(y[7])
    # Finite positions must remain finite and >= 0
    valid_idx = [0, 1, 2, 4, 5, 6, 8]
    assert np.all(~np.isnan(y[valid_idx]))
    assert np.all(y[valid_idx] >= 0.0)


def test_stage_a_multidimensional():
    """Test 3D array (T, lat, lon)."""
    T, lat, lon = 25, 3, 4
    x = np.zeros((T, lat, lon), dtype=np.float64)
    # Put isolated spike in cell (1, 2)
    x[10, 1, 2] = 75.0
    # Put 3-day event in cell (2, 1)
    x[12:15, 2, 1] = 60.0

    y = impulse_filter(x, window=7)
    assert y.shape == (T, lat, lon)
    assert y[10, 1, 2] == pytest.approx(0.0, abs=1e-6)
    assert np.all(y[12:15, 2, 1] == pytest.approx(60.0, rel=1e-5))
    assert np.all(y >= 0.0)
