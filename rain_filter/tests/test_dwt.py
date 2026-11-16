"""Unit tests for Stage 2: Discrete Wavelet Transform (DWT)."""

import numpy as np
import pytest
from rain_filter.src.dwt import (
    dwt_1d,
    dwt_cube,
    idwt_1d,
    idwt_cube,
    lifting_cdf53_forward,
    lifting_cdf53_inverse,
)


@pytest.mark.parametrize("wavelet", ["haar", "db2", "db4", "cdf53"])
@pytest.mark.parametrize("length", [64, 365, 366])
def test_perfect_reconstruction_all_wavelets(wavelet, length):
    """Verify max reconstruction error < 1e-11 across wavelets and signal lengths."""
    rng = np.random.default_rng(12345)
    # Realistic rainfall-like signal
    x = rng.gamma(shape=0.75, scale=20.0, size=length) * (rng.random(length) < 0.35)

    res = dwt_1d(x, wavelet=wavelet, level=3)
    x_rec = idwt_1d(res)

    assert x_rec.shape == x.shape
    max_err = float(np.max(np.abs(x - x_rec)))
    assert max_err < 1e-11, f"Wavelet {wavelet} length {length} reconstruction error too high: {max_err}"


@pytest.mark.parametrize("wavelet", ["haar", "db2", "db4"])
def test_parseval_energy_conservation(wavelet):
    """Orthogonal wavelets must strictly conserve total energy (Parseval's theorem)."""
    rng = np.random.default_rng(54321)
    x = rng.gamma(shape=1.0, scale=15.0, size=128)

    res = dwt_1d(x, wavelet=wavelet, level=3)
    e_in = float(np.sum(x ** 2))
    e_out = res.energies["total_energy"]

    assert e_out == pytest.approx(e_in, rel=1e-10)


def test_cdf53_integer_lifting_lossless():
    """Integer CDF 5/3 lifting scheme must achieve exact bit-level lossless reconstruction."""
    rng = np.random.default_rng(999)
    # Integer quantized rainfall (e.g. Q15 or mm * 10)
    x_int = rng.integers(0, 1000, size=100, dtype=np.int64)

    a, d = lifting_cdf53_forward(x_int, integer_mode=True)
    x_rec = lifting_cdf53_inverse(a, d, target_len=len(x_int), integer_mode=True)

    assert np.array_equal(x_rec, x_int), "Integer lifting did not achieve bit-exact lossless reconstruction"


def test_subband_energies_physical_ordering():
    """Verify subband energies sum to 100% and have correct structure."""
    rng = np.random.default_rng(777)
    x = rng.gamma(shape=0.8, scale=25.0, size=365) * (rng.random(365) < 0.3)

    res = dwt_1d(x, wavelet="haar", level=4)
    energies = res.energies

    # Sum of percentages must equal 100%
    sum_pct = energies["approx_energy_pct"] + sum(energies[f"D{j}_energy_pct"] for j in range(1, 5))
    assert sum_pct == pytest.approx(100.0, rel=1e-5)
    assert 0.0 <= energies["convective_transient_ratio_pct"] <= 100.0


def test_dwt_vectorized_3d_cube():
    """Verify DWT works on 3D cube (T, lat, lon) and correctly reconstructs."""
    T, lat, lon = 40, 3, 4
    rng = np.random.default_rng(888)
    cube = rng.gamma(shape=0.8, scale=20.0, size=(T, lat, lon))
    # Cell (1, 1) is completely NaN (like an ocean grid cell)
    cube[:, 1, 1] = np.nan
    # Cell (2, 2) has an isolated missing observation
    cube[5, 2, 2] = np.nan

    res = dwt_cube(cube, wavelet="haar", level=3)
    cube_rec = idwt_cube(res)

    assert cube_rec.shape == (T, lat, lon)
    valid = ~np.isnan(cube)
    max_err = float(np.max(np.abs(cube[valid] - cube_rec[valid])))
    assert max_err < 1e-11

    # Verify 2D energy map shape and ocean cell masking
    assert res.energies["convective_transient_ratio_pct"].shape == (lat, lon)
    assert np.isnan(res.energies["convective_transient_ratio_pct"][1, 1])
    assert not np.isnan(res.energies["convective_transient_ratio_pct"][0, 0])
