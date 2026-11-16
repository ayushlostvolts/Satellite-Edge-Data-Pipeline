"""Synthetic noise generation module for rainfall time series evaluation."""

from typing import Tuple, Union
import numpy as np


def add_gaussian_noise(
    x: np.ndarray,
    sigma: float = 1.0,
    seed: int = 42
) -> np.ndarray:
    """Add zero-mean Gaussian noise with standard deviation sigma, clipped at 0."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=x.shape)
    noisy = np.where(np.isnan(x), np.nan, np.clip(x + noise, 0.0, None))
    return noisy


def add_impulsive_noise(
    x: np.ndarray,
    fraction: float = 0.02,
    amp_min: float = 20.0,
    amp_max: float = 80.0,
    seed: int = 42
) -> np.ndarray:
    """Add positive impulsive spikes: fraction of samples Uniform(amp_min, amp_max)."""
    rng = np.random.default_rng(seed)
    noisy = x.copy()
    valid_mask = ~np.isnan(noisy) & (noisy > -900.0)
    valid_indices = np.where(valid_mask)[0] if x.ndim == 1 else np.argwhere(valid_mask)

    n_spikes = int(np.round(fraction * len(valid_indices)))
    if n_spikes > 0:
        spike_idx_choices = rng.choice(len(valid_indices), size=n_spikes, replace=False)
        amplitudes = rng.uniform(amp_min, amp_max, size=n_spikes)

        if x.ndim == 1:
            chosen = valid_indices[spike_idx_choices]
            noisy[chosen] += amplitudes
        else:
            chosen = valid_indices[spike_idx_choices]
            for i, idx in enumerate(chosen):
                noisy[tuple(idx)] += amplitudes[i]

    return np.clip(noisy, 0.0, None)


def add_mixed_noise(
    x: np.ndarray,
    sigma: float = 1.0,
    impulse_fraction: float = 0.02,
    amp_min: float = 20.0,
    amp_max: float = 80.0,
    seed: int = 42
) -> np.ndarray:
    """Mixed noise: Gaussian sigma + impulsive spikes, clipped at 0."""
    # Deterministic split seeds
    rng = np.random.default_rng(seed)
    s_gauss, s_imp = rng.integers(0, 1000000, size=2)
    step1 = add_gaussian_noise(x, sigma=sigma, seed=int(s_gauss))
    step2 = add_impulsive_noise(
        step1,
        fraction=impulse_fraction,
        amp_min=amp_min,
        amp_max=amp_max,
        seed=int(s_imp)
    )
    return np.clip(step2, 0.0, None)


def add_dropouts(
    x: np.ndarray,
    fraction: float = 0.02,
    dropout_value: float = -999.0,
    seed: int = 42
) -> np.ndarray:
    """Inject dropouts: randomly sets fraction of samples to -999.0."""
    rng = np.random.default_rng(seed)
    noisy = x.copy()
    valid_mask = ~np.isnan(noisy) & (noisy > -900.0)
    valid_indices = np.where(valid_mask)[0] if x.ndim == 1 else np.argwhere(valid_mask)

    n_dropouts = int(np.round(fraction * len(valid_indices)))
    if n_dropouts > 0:
        choices = rng.choice(len(valid_indices), size=n_dropouts, replace=False)
        if x.ndim == 1:
            chosen = valid_indices[choices]
            noisy[chosen] = dropout_value
        else:
            chosen = valid_indices[choices]
            for idx in chosen:
                noisy[tuple(idx)] = dropout_value

    return noisy
