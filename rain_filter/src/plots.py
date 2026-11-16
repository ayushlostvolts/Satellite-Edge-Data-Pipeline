"""Plot generation module for IMD gridded rainfall denoising filter evaluation.

Generates 5 required figures saved to outputs/plots/:
1. plot1_monsoon_timeseries.png: Clean vs noisy vs filtered time series for one monsoon season.
2. plot2_frequency_response.png: FIR analytical response + Empirical IIR response.
3. plot3_adaptive_gain_overlay.png: Adaptive gain a[n] overlaid on rainfall time series.
4. plot4_snr_improvement_bars.png: Bar chart of SNR improvement per filter per noise type.
5. plot5_peak_scatter.png: Scatter of clean vs filtered rainfall for peak days (>= 25 mm).
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np

from rain_filter.src.adaptive_iir import adaptive_iir_filter
from rain_filter.src.fir_custom import design_fir_taps
from rain_filter.src.metrics import (
    baseline_butterworth,
    baseline_median,
    baseline_moving_average,
    baseline_savgol,
)
from rain_filter.src.pipeline import denoise


def plot1_monsoon_timeseries(
    clean: np.ndarray,
    noisy: np.ndarray,
    city_name: str = "Mumbai",
    save_path: Optional[Path] = None,
    start_day: int = 152,  # June 1
    end_day: int = 273     # Sept 30
) -> None:
    """Plot 1: Clean vs noisy vs filtered time series for one monsoon season."""
    slice_idx = slice(start_day, min(end_day, len(clean)))
    days = np.arange(start_day, min(end_day, len(clean)))

    c_sub = clean[slice_idx]
    n_sub = noisy[slice_idx]

    hyb_sub = denoise(noisy, mode="hybrid")[slice_idx]
    iir_sub = denoise(noisy, mode="iir")[slice_idx]
    fir_sub = denoise(noisy, mode="fir")[slice_idx]
    med_sub = baseline_median(noisy, window=5)[slice_idx]
    bw_sub = baseline_butterworth(noisy, order=3, wn=0.35)[slice_idx]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, sharey=True)

    # Panel 1: Clean vs Noisy
    axes[0].plot(days, c_sub, "k-", lw=1.8, label="Clean IMD Truth", alpha=0.9)
    axes[0].plot(days, n_sub, "r.", ms=4, label="Noisy (Mixed: Gauss + Impulses)", alpha=0.6)
    axes[0].set_title(f"(A) Clean Ground Truth vs Noisy Input — {city_name} Monsoon 2024", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Rainfall (mm)")
    axes[0].legend(loc="upper right", framealpha=0.9)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Custom Filters (Hybrid, IIR, FIR)
    axes[0].plot([], []) # spacing
    axes[1].plot(days, c_sub, "k--", lw=1.2, label="Clean Truth", alpha=0.5)
    axes[1].plot(days, hyb_sub, "b-", lw=1.8, label="Hybrid Filter (Ours, A->B)")
    axes[1].plot(days, iir_sub, "g-.", lw=1.4, label="Adaptive IIR (Ours, Stage B)")
    axes[1].plot(days, fir_sub, "m:", lw=1.4, label="Custom FIR (Ours, Stage C)")
    axes[1].set_title("(B) Proposed Custom Filter Pipeline Variants", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Rainfall (mm)")
    axes[1].legend(loc="upper right", framealpha=0.9)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Proposed Hybrid vs Standard Baselines
    axes[2].plot(days, c_sub, "k--", lw=1.2, label="Clean Truth", alpha=0.5)
    axes[2].plot(days, hyb_sub, "b-", lw=2.0, label="Hybrid Filter (Ours)")
    axes[2].plot(days, bw_sub, "r-", lw=1.4, label="Butterworth LPF Order 3 (filtfilt)")
    axes[2].plot(days, med_sub, "c-.", lw=1.4, label="Median Filter (5)")
    axes[2].set_title("(C) Hybrid Filter vs Classical SciPy Baselines (Peak Attenuation Comparison)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Day of Year (Monsoon Period: June 1 – September 30)")
    axes[2].set_ylabel("Rainfall (mm)")
    axes[2].legend(loc="upper right", framealpha=0.9)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot2_frequency_response(
    save_path: Optional[Path] = None,
    num_taps: int = 11,
    fc: float = 0.35,
    beta: float = 4.0
) -> None:
    """Plot 2: Custom FIR response + Empirical Adaptive IIR frequency response."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # 1. Analytical FIR Frequency Response
    taps, _ = design_fir_taps(num_taps=num_taps, fc=fc, beta=beta)
    w_freq = np.linspace(0, np.pi, 512)
    k_idx = np.arange(num_taps)
    # DTFT: H(w) = sum h[k] * exp(-j * w * k)
    H = np.zeros_like(w_freq, dtype=np.complex128)
    for k in range(num_taps):
        H += taps[k] * np.exp(-1j * w_freq * k)

    mag_db = 20.0 * np.log10(np.maximum(np.abs(H), 1e-6))
    phase_rad = np.unwrap(np.angle(H))

    norm_freq = w_freq / np.pi  # Fraction of Nyquist

    # Top-Left: FIR Magnitude
    axes[0, 0].plot(norm_freq, mag_db, "b-", lw=2.0)
    axes[0, 0].axvline(fc, color="r", linestyle="--", label=f"Cutoff fc = {fc} Nyquist")
    axes[0, 0].axhline(-3.0, color="gray", linestyle=":", label="-3 dB Cutoff Level")
    axes[0, 0].set_title("(A) Custom Windowed-Sinc FIR Magnitude Response", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Normalized Frequency (x pi rad/sample, Nyquist = 1.0)")
    axes[0, 0].set_ylabel("Magnitude (dB)")
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)
    axes[0, 0].legend(loc="lower left")

    # Bottom-Left: FIR Phase (Linear Phase)
    axes[1, 0].plot(norm_freq, phase_rad, "g-", lw=2.0, label="Unwrapped Phase")
    slope = -(num_taps - 1) / 2.0
    axes[1, 0].plot(norm_freq, slope * w_freq, "k--", lw=1.2, label=f"Ideal Linear Phase (delay = {(num_taps-1)//2})")
    axes[1, 0].set_title("(B) Custom FIR Phase Response (Exact Linear Phase)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Normalized Frequency (x pi rad/sample)")
    axes[1, 0].set_ylabel("Phase (radians)")
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)
    axes[1, 0].legend(loc="lower left")

    # 2. Empirical Adaptive IIR Response
    # Test with white noise of different amplitudes and impulse response
    rng = np.random.default_rng(42)
    sig_lens = 2048

    # White noise test at low, medium, and high signal level
    freqs = np.fft.rfftfreq(sig_lens, d=1.0)
    norm_f = freqs / 0.5  # Nyquist = 0.5 cycles/sample -> range 0 to 1

    for amp, col, lbl in [(0.5, "purple", "Small Noise (sigma=0.5 mm)"), (5.0, "orange", "Medium Rain (sigma=5 mm)"), (50.0, "red", "Heavy Convection (sigma=50 mm)")]:
        inp = rng.normal(0, amp, size=sig_lens) + 20.0  # offset so non-negative
        inp = np.clip(inp, 0.0, None)
        out = adaptive_iir_filter(inp, causal=True)
        # Power Spectral Density ratio
        psd_in = np.abs(np.fft.rfft(inp - np.mean(inp))) ** 2
        psd_out = np.abs(np.fft.rfft(out - np.mean(out))) ** 2
        smooth_tf = np.convolve(psd_out / np.maximum(psd_in, 1e-8), np.ones(15)/15, mode="same")
        mag_tf_db = 10.0 * np.log10(np.maximum(smooth_tf, 1e-4))
        axes[0, 1].plot(norm_f, mag_tf_db, color=col, lw=1.8, label=lbl)

    axes[0, 1].set_title("(C) Empirical Adaptive IIR Transfer Function (White Noise)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Normalized Frequency (Fraction of Nyquist)")
    axes[0, 1].set_ylabel("Empirical Gain (dB)")
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)
    axes[0, 1].legend(loc="lower left")

    # Impulse Response test of Adaptive IIR for large vs small spike
    time_pts = np.arange(40)
    # Large spike (80 mm)
    imp_large = np.zeros(40)
    imp_large[5] = 80.0
    out_large = adaptive_iir_filter(imp_large, causal=True)

    # Small spike (2 mm)
    imp_small = np.zeros(40)
    imp_small[5] = 2.0
    out_small = adaptive_iir_filter(imp_small, causal=True)

    axes[1, 1].plot(time_pts, out_large / 80.0, "r-o", ms=4, lw=1.8, label="Large Impulse (80 mm) Normalized")
    axes[1, 1].plot(time_pts, out_small / 2.0, "b-s", ms=4, lw=1.8, label="Small Impulse (2 mm) Normalized")
    axes[1, 1].set_title("(D) Normalized Step/Impulse Response vs Amplitude", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Sample Lag n")
    axes[1, 1].set_ylabel("Normalized Response y[n] / A")
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)
    axes[1, 1].legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot3_adaptive_gain_overlay(
    series: np.ndarray,
    city_name: str = "Cherrapunji",
    save_path: Optional[Path] = None
) -> None:
    """Plot 3: Adaptive gain a[n] over time overlaid on the rainfall series."""
    y, gains = adaptive_iir_filter(series, causal=True, return_gains=True)

    days = np.arange(len(series))

    fig, ax1 = plt.subplots(figsize=(14, 6))

    color_rain = "#1f77b4"
    ax1.set_xlabel("Day of Year (2024)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Daily Rainfall (mm)", color=color_rain, fontsize=11, fontweight="bold")
    line1 = ax1.plot(days, series, color=color_rain, lw=1.6, label="Rainfall (mm)")
    ax1.fill_between(days, series, color=color_rain, alpha=0.25)
    ax1.tick_params(axis="y", labelcolor=color_rain)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    color_gain = "#d62728"
    ax2.set_ylabel("Adaptive Gain a[n]", color=color_gain, fontsize=11, fontweight="bold")
    line2 = ax2.plot(days, gains, color=color_gain, lw=1.8, linestyle="-", label="Adaptive Gain a[n]")
    ax2.axhline(0.95, color="red", linestyle=":", alpha=0.6, label="a_max = 0.95 (Fast Tracking)")
    ax2.axhline(0.08, color="darkred", linestyle=":", alpha=0.6, label="a_min = 0.08 (Heavy Smoothing)")
    ax2.set_ylim(-0.02, 1.05)
    ax2.tick_params(axis="y", labelcolor=color_gain)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", framealpha=0.9)

    plt.title(
        f"Stage B Adaptive Gain a[n] Dynamics Overlaid on Rainfall — {city_name} (2024)\n"
        f"High Innovation (Rain Storm) -> a ~ a_max (Preserves Peaks); Low Innovation (Drying) -> a ~ a_min (Strong Filtering)",
        fontsize=12, fontweight="bold"
    )

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot4_snr_improvement_bars(
    eval_results: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None
) -> None:
    """Plot 4: Bar chart of SNR improvement per filter per noise type."""
    # eval_results: {noise_type: {filter_name: snr_imp}}
    noise_types = list(eval_results.keys())
    first_key = noise_types[0]
    filter_names = list(eval_results[first_key].keys())

    x = np.arange(len(noise_types))
    width = 0.8 / len(filter_names)

    fig, ax = plt.subplots(figsize=(14, 7))

    cmap = plt.get_cmap("tab10")
    for i, fname in enumerate(filter_names):
        vals = [eval_results[nt].get(fname, 0.0) for nt in noise_types]
        offset = (i - len(filter_names) / 2) * width + width / 2
        ax.bar(x + offset, vals, width, label=fname, color=cmap(i % 10), alpha=0.85, edgecolor="black", lw=0.5)

    ax.set_xlabel("Noise Condition", fontsize=11, fontweight="bold")
    ax.set_ylabel("SNR Improvement (dB)", fontsize=11, fontweight="bold")
    ax.set_title("SNR Improvement (dB) Across Noise Types and Filter Architectures", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(noise_types, fontsize=10, rotation=15)
    ax.axhline(0, color="k", lw=0.8, linestyle="--")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=0.9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot5_peak_scatter(
    clean: np.ndarray,
    noisy: np.ndarray,
    save_path: Optional[Path] = None,
    threshold: float = 25.0
) -> None:
    """Plot 5: Scatter of clean vs filtered for peak days (>= 25 mm)."""
    peak_mask = clean >= threshold
    if not np.any(peak_mask):
        peak_mask = clean >= 10.0

    c_peaks = clean[peak_mask]
    hyb_peaks = denoise(noisy, mode="hybrid")[peak_mask]
    bw_peaks = baseline_butterworth(noisy, order=3, wn=0.35)[peak_mask]
    med_peaks = baseline_median(noisy, window=5)[peak_mask]
    ma_peaks = baseline_moving_average(noisy, window=5)[peak_mask]

    fig, ax = plt.subplots(figsize=(9, 8))

    max_val = max(np.max(c_peaks), np.max(hyb_peaks)) * 1.1

    # 1:1 reference line
    ax.plot([0, max_val], [0, max_val], "k--", lw=1.5, label="Ideal 1:1 Preservation (100%)", alpha=0.8)

    ax.scatter(c_peaks, hyb_peaks, color="blue", marker="o", s=65, label="Proposed Hybrid (Ours)", alpha=0.85, edgecolors="darkblue")
    ax.scatter(c_peaks, bw_peaks, color="red", marker="^", s=50, label="Butterworth Order 3", alpha=0.75, edgecolors="darkred")
    ax.scatter(c_peaks, med_peaks, color="green", marker="s", s=45, label="Median Filter (5)", alpha=0.75, edgecolors="darkgreen")
    ax.scatter(c_peaks, ma_peaks, color="orange", marker="d", s=45, label="Moving Average (5)", alpha=0.75, edgecolors="darkorange")

    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect("equal")

    ax.set_xlabel("Clean Ground Truth Peak Rainfall (mm)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Filtered Peak Rainfall (mm)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Peak Rainfall Preservation Comparison (Days with Rain >= {threshold:.0f} mm)\n"
        f"Classical Filters Squeeze Extreme Events Downwards; Hybrid Preserves Peak Magnitudes",
        fontsize=12, fontweight="bold"
    )
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()
