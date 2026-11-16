"""Feature Ranking and Selection Module for Satellite Edge Telemetry.

Ranks extracted temporal and wavelet features according to their discriminative power
for onboard detection of extreme rainfall, convective cloudbursts, and flood hazards.

Implements lightweight, closed-form edge ranking algorithms in pure NumPy:
1. Fisher Discriminant Ratio (F-Score)
2. Signal-to-Noise Ratio (SNR Score)
3. Information Correlation Ratio (ICR)
4. Composite Edge Priority Score (CEPS)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


def compute_fisher_score(
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """Compute Fisher Discriminant Ratio for each feature column in X.

    F = sum_c (N_c * (mu_c - mu)^2) / sum_c (sum_{i in c} (x_i - mu_c)^2)

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (N_samples, N_features).
    y : np.ndarray
        Binary or categorical class labels of shape (N_samples,).

    Returns
    -------
    scores : np.ndarray
        Fisher scores of shape (N_features,).
    """
    classes = np.unique(y)
    n_samples, n_features = X.shape
    overall_mean = np.mean(X, axis=0)

    between_var = np.zeros(n_features, dtype=np.float64)
    within_var = np.zeros(n_features, dtype=np.float64)

    for c in classes:
        c_mask = (y == c)
        n_c = np.sum(c_mask)
        if n_c == 0:
            continue
        X_c = X[c_mask]
        mean_c = np.mean(X_c, axis=0)

        between_var += n_c * ((mean_c - overall_mean) ** 2)
        within_var += np.sum((X_c - mean_c) ** 2, axis=0)

    scores = between_var / np.maximum(within_var, 1e-12)
    return scores


def compute_snr_score(
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """Compute Signal-to-Noise Ratio (SNR) score for binary extreme event labels.

    SNR = |mu_1 - mu_0| / (sigma_1 + sigma_0)

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (N_samples, N_features).
    y : np.ndarray
        Binary labels (0: normal/arid, 1: extreme/flood hazard).

    Returns
    -------
    snr_scores : np.ndarray
        SNR scores of shape (N_features,).
    """
    mask1 = (y == 1)
    mask0 = (y == 0)

    if not np.any(mask1) or not np.any(mask0):
        return np.zeros(X.shape[1], dtype=np.float64)

    mu1 = np.mean(X[mask1], axis=0)
    mu0 = np.mean(X[mask0], axis=0)

    std1 = np.std(X[mask1], axis=0)
    std0 = np.std(X[mask0], axis=0)

    snr = np.abs(mu1 - mu0) / np.maximum(std1 + std0, 1e-12)
    return snr


def compute_correlation_score(
    X: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """Compute absolute Pearson correlation coefficient between each feature and y."""
    n_features = X.shape[1]
    corrs = np.zeros(n_features, dtype=np.float64)
    y_centered = y - np.mean(y)
    y_norm = np.linalg.norm(y_centered)
    if y_norm < 1e-12:
        return corrs

    for j in range(n_features):
        x_col = X[:, j]
        x_centered = x_col - np.mean(x_col)
        x_norm = np.linalg.norm(x_centered)
        if x_norm > 1e-12:
            corrs[j] = np.abs(np.dot(x_centered, y_centered) / (x_norm * y_norm))
        else:
            corrs[j] = 0.0

    return corrs


def rank_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    method: str = "fisher"
) -> List[Dict[str, Union[str, float, int]]]:
    """Rank features according to chosen edge ranking criterion.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (N_samples, N_features).
    y : np.ndarray
        Labels of shape (N_samples,).
    feature_names : list of str
        List of feature names matching columns of X.
    method : str
        Ranking method: 'fisher' (default), 'snr', 'correlation', or 'composite'.

    Returns
    -------
    ranked_list : list of dicts
        List of features sorted from highest to lowest rank.
    """
    method_lower = method.lower()

    if method_lower == "fisher":
        scores = compute_fisher_score(X, y)
    elif method_lower == "snr":
        scores = compute_snr_score(X, y)
    elif method_lower == "correlation":
        scores = compute_correlation_score(X, y)
    elif method_lower == "composite":
        f_scores = compute_fisher_score(X, y)
        s_scores = compute_snr_score(X, y)
        c_scores = compute_correlation_score(X, y)

        # Normalize to [0, 1] and combine
        f_norm = f_scores / max(np.max(f_scores), 1e-12)
        s_norm = s_scores / max(np.max(s_scores), 1e-12)
        c_norm = c_scores / max(np.max(c_scores), 1e-12)
        scores = (f_norm * 0.45) + (s_norm * 0.35) + (c_norm * 0.20)
    else:
        raise ValueError(f"Unknown ranking method '{method}'. Choose from 'fisher', 'snr', 'correlation', 'composite'.")

    # Sort in descending order
    sorted_indices = np.argsort(scores)[::-1]

    ranked = []
    for rank_idx, col_idx in enumerate(sorted_indices, 1):
        ranked.append({
            "rank": rank_idx,
            "feature": feature_names[col_idx],
            "score": float(scores[col_idx]),
            "method": method_lower
        })

    return ranked


def select_top_k_features(
    ranked_features: List[Dict],
    top_k: int = 5
) -> List[str]:
    """Select the top K feature names from a ranked feature list."""
    return [item["feature"] for item in ranked_features[:top_k]]
