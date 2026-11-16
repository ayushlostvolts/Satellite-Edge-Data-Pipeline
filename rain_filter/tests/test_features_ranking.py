"""Unit tests for Stage 3: Feature Extraction and Ranking."""

import numpy as np
import pytest
from rain_filter.src.features import extract_features_1d, extract_features_cube
from rain_filter.src.ranking import (
    compute_correlation_score,
    compute_fisher_score,
    compute_snr_score,
    rank_features,
    select_top_k_features,
)


def test_extract_features_1d_validity():
    """Verify extracted 1D features conform to physical ranges and non-negativity."""
    rng = np.random.default_rng(101)
    series = rng.gamma(shape=0.8, scale=25.0, size=365) * (rng.random(365) < 0.35)

    feats = extract_features_1d(series)

    assert feats["p_max_mm"] > 0.0
    assert feats["total_rainfall_mm"] > 0.0
    assert 0.0 <= feats["wet_day_freq"] <= 1.0
    assert 0.0 <= feats["convective_energy_ratio_pct"] <= 100.0
    assert 0.0 <= feats["r95p_extreme_ratio_pct"] <= 100.0
    assert feats["wavelet_shannon_entropy"] >= 0.0
    assert feats["max_cwd_days"] >= 0
    assert feats["max_cdd_days"] >= 0


def test_extract_features_dry_series():
    """Verify dry/zero series handles gracefully without division by zero."""
    series = np.zeros(365, dtype=np.float64)
    feats = extract_features_1d(series)

    assert feats["p_max_mm"] == 0.0
    assert feats["total_rainfall_mm"] == 0.0
    assert feats["wet_day_freq"] == 0.0
    assert feats["max_cdd_days"] == 365.0
    assert feats["wavelet_shannon_entropy"] == 0.0


def test_extract_features_cube_shape():
    """Verify 3D cube feature extraction produces proper 2D maps and handles NaNs."""
    T, lat, lon = 50, 4, 5
    rng = np.random.default_rng(202)
    cube = rng.gamma(shape=0.8, scale=20.0, size=(T, lat, lon))
    cube[:, 0, 0] = np.nan  # ocean cell

    maps = extract_features_cube(cube, wavelet="haar", level=3)

    for name, arr in maps.items():
        assert arr.shape == (lat, lon)
        assert np.isnan(arr[0, 0]), f"Ocean cell at (0, 0) was not masked in map {name}"
        assert not np.isnan(arr[1, 1]), f"Valid cell at (1, 1) became NaN in map {name}"


def test_ranking_algorithms():
    """Verify Fisher, SNR, and composite scores correctly rank discriminative features."""
    rng = np.random.default_rng(303)
    n_samples = 100

    # Feature 0: Strong separation between classes
    f0 = np.concatenate([rng.normal(10, 2, 50), rng.normal(80, 5, 50)])
    # Feature 1: Pure random noise (no separation)
    f1 = rng.normal(50, 20, 100)
    # Feature 2: Moderate separation
    f2 = np.concatenate([rng.normal(20, 10, 50), rng.normal(45, 10, 50)])

    X = np.column_stack([f0, f1, f2])
    y = np.array([0] * 50 + [1] * 50)
    names = ["strong_feat", "noise_feat", "moderate_feat"]

    # Fisher score
    f_ranked = rank_features(X, y, names, method="fisher")
    assert f_ranked[0]["feature"] == "strong_feat"
    assert f_ranked[1]["feature"] == "moderate_feat"
    assert f_ranked[2]["feature"] == "noise_feat"

    # SNR score
    s_ranked = rank_features(X, y, names, method="snr")
    assert s_ranked[0]["feature"] == "strong_feat"

    # Composite score
    c_ranked = rank_features(X, y, names, method="composite")
    assert c_ranked[0]["feature"] == "strong_feat"

    # Top-K selection
    top1 = select_top_k_features(f_ranked, top_k=1)
    assert top1 == ["strong_feat"]
    top2 = select_top_k_features(f_ranked, top_k=2)
    assert top2 == ["strong_feat", "moderate_feat"]
