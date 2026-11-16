"""Edge-feasibility and computational cost analysis module.

- Counts multiply, addition, division, and comparison operations per sample.
- Calculates streaming state memory footprint per cell (bytes).
- Evaluates Q15 fixed-point vs float64 performance.
- Times the vectorized pipeline implementation over a full year cube (365 x 129 x 135).
"""

from typing import Dict, Tuple
import time
import numpy as np
from rain_filter.src.adaptive_iir import adaptive_iir_filter, adaptive_iir_q15
from rain_filter.src.pipeline import denoise


def get_theoretical_ops_per_sample() -> Dict[str, Dict[str, int]]:
    """Return theoretical operation counts per sample for each filtering stage.

    Returns
    -------
    cost_dict : dict
        Breakdown of multiplies, additions/subtractions, divisions, and comparisons.
    """
    return {
        "Stage A (Hampel Impulse, W=7)": {
            "multiplies": 3,      # scale factor s, threshold k*scale, isolation ratio
            "adds_subs": 11,     # diffs in MAD, sample deviations, center offset
            "divisions": 0,
            "compares": 34,      # sorting network for window (13) + MAD (13) + outlier & 2 isolation + clip
            "description": "7-tap windowed median + MAD + isolation logic + non-negative clip"
        },
        "Stage B (Adaptive IIR, Causal)": {
            "multiplies": 7,      # d^2, lam*sigma2, (1-lam)*(d^2/2), e^2, c*sigma2, a*x, (1-a)*y
            "adds_subs": 6,      # d = x-x_prev, sigma2 update, e = x-y_prev, denom, 1-a, y update
            "divisions": 1,      # gain a = e^2 / denom
            "compares": 4,       # denom > 1e-12, clip(a_min), clip(a_max), clip(y >= 0)
            "description": "Recursive innovation tracking, running variance, and gain clipping"
        },
        "Stage C (Custom FIR, N=11 taps)": {
            "multiplies": 6,      # Exploiting tap symmetry: (N+1)//2 = 6 multiplies
            "adds_subs": 10,     # 5 pair additions + 5 accumulator additions
            "divisions": 0,
            "compares": 1,       # non-negative clip
            "description": "11-tap symmetric linear-phase FIR convolution with reflect padding"
        },
        "Hybrid Pipeline (A -> B)": {
            "multiplies": 10,
            "adds_subs": 17,
            "divisions": 1,
            "compares": 38,
            "description": "Sequential Stage A followed by Stage B"
        }
    }


def get_state_memory_footprint() -> Dict[str, Dict[str, int]]:
    """Report state memory required per grid cell and across the whole IMD grid.

    Grid dimensions: 129 (lat) x 135 (lon) = 17,415 cells.
    """
    total_cells = 129 * 135  # 17,415 cells

    return {
        "Streaming IIR Float64": {
            "bytes_per_cell": 24,  # y_prev (8B) + x_prev (8B) + sigma2 (8B)
            "total_grid_kb": int(round((total_cells * 24) / 1024)),
            "state_variables": "y_prev (float64), x_prev (float64), sigma2 (float64)"
        },
        "Streaming IIR Float32": {
            "bytes_per_cell": 12,  # y_prev (4B) + x_prev (4B) + sigma2 (4B)
            "total_grid_kb": int(round((total_cells * 12) / 1024)),
            "state_variables": "y_prev (float32), x_prev (float32), sigma2 (float32)"
        },
        "Streaming IIR Fixed-Point Q15": {
            "bytes_per_cell": 8,   # y_prev (int16: 2B) + x_prev (int16: 2B) + sigma2 (int32: 4B)
            "total_grid_kb": int(round((total_cells * 8) / 1024)),
            "state_variables": "y_prev (int16), x_prev (int16), sigma2 (int32 accumulator)"
        },
        "Streaming FIR Buffer (N=11)": {
            "bytes_per_cell": 20,  # 10 past samples in Q15 int16 (20B)
            "total_grid_kb": int(round((total_cells * 20) / 1024)),
            "state_variables": "10 past samples circular buffer (int16)"
        }
    }


def benchmark_q15_accuracy(
    x_clean: np.ndarray,
    noise_sigma: float = 1.0,
    seed: int = 42
) -> Dict[str, float]:
    """Compare fixed-point Q15 against float64 on test series.

    Returns max abs error, mean abs error, float64 SNR, Q15 SNR, and SNR loss in dB.
    """
    rng = np.random.default_rng(seed)
    noisy = np.clip(x_clean + rng.normal(0, noise_sigma, size=x_clean.shape), 0.0, None)

    y_f64 = adaptive_iir_filter(noisy, causal=True)
    y_q15 = adaptive_iir_q15(noisy)

    valid = ~np.isnan(x_clean) & ~np.isnan(y_f64) & ~np.isnan(y_q15)
    c = x_clean[valid]
    f64 = y_f64[valid]
    q15 = y_q15[valid]

    abs_err = np.abs(f64 - q15)
    max_err = float(np.max(abs_err))
    mean_err = float(np.mean(abs_err))

    mse_f64 = float(np.mean((f64 - c) ** 2))
    mse_q15 = float(np.mean((q15 - c) ** 2))

    snr_f64 = float(10.0 * np.log10(np.mean(c ** 2) / max(mse_f64, 1e-12)))
    snr_q15 = float(10.0 * np.log10(np.mean(c ** 2) / max(mse_q15, 1e-12)))
    snr_loss = snr_f64 - snr_q15

    return {
        "max_abs_error_mm": max_err,
        "mean_abs_error_mm": mean_err,
        "snr_f64_db": snr_f64,
        "snr_q15_db": snr_q15,
        "snr_loss_db": snr_loss
    }


def benchmark_cube_runtime(
    t_len: int = 365,
    n_lat: int = 129,
    n_lon: int = 135,
    num_runs: int = 3,
    seed: int = 42
) -> Dict[str, float]:
    """Time vectorized implementation over full 3D cube (T x lat x lon).

    Parameters
    ----------
    t_len : int
        Number of time steps (default 365).
    n_lat : int
        Number of latitude points (default 129).
    n_lon : int
        Number of longitude points (default 135).
    num_runs : int
        Number of runs to average (default 3).

    Returns
    -------
    timing_dict : dict
        Execution times in seconds for each stage and complete pipeline.
    """
    rng = np.random.default_rng(seed)
    # Generate representative gridded data (fraction of grid has rain)
    data = rng.gamma(0.75, 25.0, size=(t_len, n_lat, n_lon)).astype(np.float32)
    # 70% of days/cells are dry
    data *= (rng.random(size=(t_len, n_lat, n_lon)) < 0.3)

    results = {}

    for mode in ["iir", "fir", "hybrid"]:
        times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = denoise(data, mode=mode, preserve_volume=False)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        results[f"{mode}_seconds"] = float(np.mean(times))

    # Also compute throughput in cells per second
    total_samples = t_len * n_lat * n_lon
    hyb_time = results["hybrid_seconds"]
    results["throughput_samples_per_sec"] = float(total_samples / max(hyb_time, 1e-6))
    results["throughput_ms_per_day"] = float((hyb_time / t_len) * 1000.0)

    return results
