"""Visualization module for Stage 2: Discrete Wavelet Transform (DWT).

Generates:
1. plot6_dwt_decomposition_multilevel.png: Multi-level subband decomposition (A3, D3, D2, D1)
   for representative stations across different climatic zones.
2. plot7_dwt_energy_distribution.png: Subband energy distribution comparison across stations
   and spatial map of convective transient storm energy fraction over India.
"""

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

from rain_filter.src.dwt import dwt_1d, dwt_cube
from rain_filter.src.io_utils import extract_cell, load_dataset
from rain_filter.src.pipeline import denoise


def plot6_dwt_decomposition_multilevel(
    dataset_path: str = "Dataset/RF25_ind2024_rfp25.nc",
    save_path: Optional[Path] = None,
    city_name: str = "Mumbai",
    wavelet: str = "haar",
    level: int = 3
) -> None:
    """Plot multi-level wavelet decomposition subbands (Original, A_J, D_J, ..., D_1)."""
    rain, times, lats, lons = load_dataset(dataset_path)

    cities = {
        "Kolkata": (22.5, 88.25),
        "Cherrapunji": (25.25, 91.75),
        "Mumbai": (19.0, 72.75),
        "Jaisalmer": (26.9, 70.9)
    }

    target_lat, target_lon = cities[city_name]
    raw_series, _, _ = extract_cell(rain, lats, lons, target_lat, target_lon)

    # 1. First run Stage 1 (Denoise)
    denoised_series = denoise(raw_series, mode="hybrid", preserve_volume=True)

    # 2. Run Stage 2 (DWT)
    res = dwt_1d(denoised_series, wavelet=wavelet, level=level)

    days = np.arange(len(denoised_series))
    n_subbands = level + 2  # Original, A_J, D_J, ..., D_1

    fig, axes = plt.subplots(n_subbands, 1, figsize=(14, 2.2 * n_subbands), sharex=False)

    # Subplot 0: Original Denoised Series
    axes[0].plot(days, denoised_series, color="#1f77b4", lw=1.5)
    axes[0].fill_between(days, denoised_series, color="#1f77b4", alpha=0.2)
    axes[0].set_title(f"(A) Denoised Input Rainfall x[n] — {city_name} (2024)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Rain (mm)")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Subplot 1: Approximation A_J (Seasonal Macro-Envelope)
    t_app = np.linspace(0, len(days) - 1, len(res.approx))
    axes[1].plot(t_app, res.approx, color="#2ca02c", lw=1.8, marker="o", ms=2.5)
    axes[1].set_title(
        f"(B) Approximation Level A_{level} (> {2**level * 2} days) — Seasonal Monsoon Climatological Envelope "
        f"[{res.energies['approx_energy_pct']:.1f}% Energy]",
        fontsize=11, fontweight="bold"
    )
    axes[1].set_ylabel("Coeff Value")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Subplots 2+: Details D_level down to D_1
    colors = ["#9467bd", "#ff7f0e", "#d62728", "#8c564b"]
    descriptions = {
        1: "High-Freq Cloudbursts & Convective Bursts (2-4 days)",
        2: "Synoptic Depressions & Low Pressure Systems (4-8 days)",
        3: "Intra-Seasonal Active/Break Monsoon Spells (8-16 days)",
        4: "Sub-Monthly Monsoon Oscillations (16-32 days)"
    }

    for idx, lvl in enumerate(range(level, 0, -1)):
        ax_idx = idx + 2
        d_sub = res.details[lvl - 1]
        t_det = np.linspace(0, len(days) - 1, len(d_sub))
        col = colors[(lvl - 1) % len(colors)]
        desc = descriptions.get(lvl, f"Detail Subband D_{lvl}")

        axes[ax_idx].plot(t_det, d_sub, color=col, lw=1.4)
        axes[ax_idx].axhline(0, color="gray", linestyle=":", lw=0.8)
        axes[ax_idx].set_title(
            f"(C{lvl}) Detail Level D_{lvl} — {desc} [{res.energies[f'D{lvl}_energy_pct']:.1f}% Energy]",
            fontsize=11, fontweight="bold"
        )
        axes[ax_idx].set_ylabel(f"D_{lvl} (mm)")
        axes[ax_idx].grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Day of Year (2024)", fontsize=11, fontweight="bold")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot7_dwt_energy_distribution(
    dataset_path: str = "Dataset/RF25_ind2024_rfp25.nc",
    save_path: Optional[Path] = None,
    wavelet: str = "haar",
    level: int = 3
) -> None:
    """Plot subband energy percentages across stations and 2D convective energy map."""
    rain, times, lats, lons = load_dataset(dataset_path)

    cities = {
        "Cherrapunji": (25.25, 91.75),
        "Mumbai": (19.0, 72.75),
        "Kolkata": (22.5, 88.25),
        "Jaisalmer": (26.9, 70.9)
    }

    station_energies = {}
    for cname, (lat, lon) in cities.items():
        ser, _, _ = extract_cell(rain, lats, lons, lat, lon)
        denoised = denoise(ser, mode="hybrid", preserve_volume=True)
        res = dwt_1d(denoised, wavelet=wavelet, level=level)
        station_energies[cname] = res.energies

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Stacked Bar Chart of Subband Energy Partition
    c_names = list(cities.keys())
    x_pos = np.arange(len(c_names))
    width = 0.55

    e_approx = [station_energies[c]["approx_energy_pct"] for c in c_names]
    e_d3 = [station_energies[c]["D3_energy_pct"] for c in c_names]
    e_d2 = [station_energies[c]["D2_energy_pct"] for c in c_names]
    e_d1 = [station_energies[c]["D1_energy_pct"] for c in c_names]

    axes[0].bar(x_pos, e_approx, width, label=f"Approximation A_{level} (Seasonal Envelope)", color="#2ca02c", edgecolor="black")
    axes[0].bar(x_pos, e_d3, width, bottom=e_approx, label="Detail D_3 (Active/Break 8-16d)", color="#9467bd", edgecolor="black")
    bottom_d2 = [a + b for a, b in zip(e_approx, e_d3)]
    axes[0].bar(x_pos, e_d2, width, bottom=bottom_d2, label="Detail D_2 (Depressions 4-8d)", color="#ff7f0e", edgecolor="black")
    bottom_d1 = [a + b for a, b in zip(bottom_d2, e_d2)]
    axes[0].bar(x_pos, e_d1, width, bottom=bottom_d1, label="Detail D_1 (Convective Bursts 2-4d)", color="#d62728", edgecolor="black")

    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(c_names, fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Total Subband Energy (%)", fontsize=11, fontweight="bold")
    axes[0].set_title("(A) Multi-Scale Energy Partition Across Climatic Regimes", fontsize=12, fontweight="bold")
    axes[0].set_ylim(0, 105)
    axes[0].legend(loc="upper right", framealpha=0.9)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.5)

    # Annotate high-frequency convective percentage
    for i, c in enumerate(c_names):
        pct_transient = station_energies[c]["convective_transient_ratio_pct"]
        axes[0].text(i, 102, f"Transient: {pct_transient:.1f}%", ha="center", fontsize=9, fontweight="bold", color="darkred")

    # Right: Spatial Gridded Map of Convective Transient Energy Ratio over India
    # Downsample spatially for fast visualization (every 2nd cell)
    sub_cube = rain[:, ::2, ::2]
    sub_lats = lats[::2]
    sub_lons = lons[::2]

    cube_res = dwt_cube(sub_cube, wavelet=wavelet, level=level)
    convective_map = cube_res.energies["convective_transient_ratio_pct"]

    im = axes[1].imshow(
        convective_map,
        origin="lower",
        extent=[sub_lons[0], sub_lons[-1], sub_lats[0], sub_lats[-1]],
        cmap="turbo",
        vmin=0,
        vmax=60
    )
    cbar = plt.colorbar(im, ax=axes[1], shrink=0.85)
    cbar.set_label("Convective Transient Energy Ratio (D1 + D2) %", fontsize=10, fontweight="bold")

    # Mark benchmark stations
    for cname, (lat, lon) in cities.items():
        axes[1].plot(lon, lat, "ko", ms=6)
        axes[1].text(lon + 0.6, lat - 0.2, cname, fontsize=9, fontweight="bold", color="black",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    axes[1].set_title("(B) Gridded Map: High-Frequency Convective Storm Energy (%)\nComputed Onboard via Vectorized 3D Cube DWT", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Longitude (°E)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Latitude (°N)", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle=":", alpha=0.4)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()
