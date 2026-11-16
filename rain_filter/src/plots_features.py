"""Visualization module for Stage 3: Feature Extraction and Ranking.

Generates:
1. plot8_feature_ranking_scores.png: Bar chart of feature discriminative rankings
   (Fisher Score and Composite Edge Priority Score).
2. plot9_feature_maps_spatial.png: 4-panel spatial maps of India showing the top
   ranked edge features across the subcontinent.
"""

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

from rain_filter.src.features import extract_features_1d, extract_features_cube
from rain_filter.src.io_utils import extract_cell, load_dataset
from rain_filter.src.pipeline import denoise
from rain_filter.src.ranking import rank_features


def generate_feature_ranking_plots(
    dataset_path: str = "Dataset/RF25_ind2024_rfp25.nc",
    plots_dir: str = "rain_filter/outputs/plots"
) -> None:
    """Generate Feature Ranking and Spatial Feature Maps."""
    p_dir = Path(plots_dir)
    p_dir.mkdir(parents=True, exist_ok=True)

    rain, times, lats, lons = load_dataset(dataset_path)

    # 1. Sample cells across India to build feature matrix
    # Subsample grid (every 3rd latitude and longitude) to get representative sample
    sub_lats = lats[::3]
    sub_lons = lons[::3]
    sub_cube = rain[:, ::3, ::3]

    X_list = []
    y_list = []
    names = None

    for i in range(len(sub_lats)):
        for j in range(len(sub_lons)):
            ser = sub_cube[:, i, j]
            if np.all(np.isnan(ser)) or np.sum(ser[~np.isnan(ser)]) < 1.0:
                continue

            # Run Stage 1 denoise
            denoised = denoise(ser, mode="hybrid", preserve_volume=True)
            feats = extract_features_1d(denoised, wavelet="haar", level=3)

            if names is None:
                names = list(feats.keys())

            X_list.append([feats[k] for k in names])

            # Ground truth label: 1 if extreme convective flood hazard (P_max >= 80 mm or SDII >= 25 mm)
            is_extreme = 1 if (feats["p_max_mm"] >= 80.0 or feats["sdii_intensity_mm"] >= 25.0) else 0
            y_list.append(is_extreme)

    X = np.array(X_list)
    y = np.array(y_list)

    # Compute rankings
    ranked_fisher = rank_features(X, y, names, method="fisher")
    ranked_composite = rank_features(X, y, names, method="composite")

    # -------------------------------------------------------------
    # Plot 8: Feature Ranking Bar Chart
    # -------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(14, 7))

    feat_names_plot = [r["feature"] for r in ranked_composite]
    comp_scores = [r["score"] for r in ranked_composite]
    fisher_dict = {r["feature"]: r["score"] for r in ranked_fisher}
    fisher_scores = [fisher_dict[f] for f in feat_names_plot]

    # Normalize fisher scores for visual comparison on secondary axis
    y_pos = np.arange(len(feat_names_plot))

    bars = ax1.barh(y_pos, comp_scores, align="center", color="#1f77b4", alpha=0.85, edgecolor="black")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(feat_names_plot, fontsize=10, fontweight="bold")
    ax1.invert_yaxis()  # top feature on top
    ax1.set_xlabel("Composite Edge Priority Score (CEPS)", fontsize=11, fontweight="bold", color="#1f77b4")
    ax1.set_title(
        "Onboard Satellite Feature Ranking for Extreme Precipitation / Flood Detection\n"
        "(Composite Score combining Fisher Discriminant Power, SNR, and Information Correlation)",
        fontsize=12, fontweight="bold"
    )
    ax1.grid(True, axis="x", linestyle="--", alpha=0.5)

    # Add score annotations
    for bar in bars:
        width = bar.get_width()
        ax1.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f"{width:.3f}",
                 va="center", ha="left", fontsize=9, fontweight="bold", color="#1f77b4")

    plt.tight_layout()
    plt.savefig(p_dir / "plot8_feature_ranking_scores.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # Plot 9: 4-Panel 2D Spatial Feature Maps over India
    # -------------------------------------------------------------
    maps = extract_features_cube(rain, wavelet="haar", level=3)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Panel A: Convective Transient Ratio (%)
    im0 = axes[0, 0].imshow(
        maps["convective_energy_ratio_pct"],
        origin="lower",
        extent=[lons[0], lons[-1], lats[0], lats[-1]],
        cmap="turbo",
        vmin=0,
        vmax=60
    )
    axes[0, 0].set_title("(A) Wavelet Convective Transient Energy Ratio (D1+D2) %", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Longitude (°E)")
    axes[0, 0].set_ylabel("Latitude (°N)")
    plt.colorbar(im0, ax=axes[0, 0], shrink=0.8, label="Energy Ratio (%)")
    axes[0, 0].grid(True, linestyle=":", alpha=0.4)

    # Panel B: Maximum Daily Precipitation (mm)
    im1 = axes[0, 1].imshow(
        maps["p_max_mm"],
        origin="lower",
        extent=[lons[0], lons[-1], lats[0], lats[-1]],
        cmap="YlGnBu",
        vmin=0,
        vmax=250
    )
    axes[0, 1].set_title("(B) Maximum Daily Precipitation P_max (mm)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Longitude (°E)")
    axes[0, 1].set_ylabel("Latitude (°N)")
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.8, label="Rainfall (mm/day)")
    axes[0, 1].grid(True, linestyle=":", alpha=0.4)

    # Panel C: Simple Daily Intensity Index (SDII)
    im2 = axes[1, 0].imshow(
        maps["sdii_intensity_mm"],
        origin="lower",
        extent=[lons[0], lons[-1], lats[0], lats[-1]],
        cmap="plasma",
        vmin=0,
        vmax=35
    )
    axes[1, 0].set_title("(C) Simple Daily Intensity Index (SDII in mm/wet day)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Longitude (°E)")
    axes[1, 0].set_ylabel("Latitude (°N)")
    plt.colorbar(im2, ax=axes[1, 0], shrink=0.8, label="Intensity (mm/day)")
    axes[1, 0].grid(True, linestyle=":", alpha=0.4)

    # Panel D: Wavelet Shannon Energy Entropy
    im3 = axes[1, 1].imshow(
        maps["wavelet_shannon_entropy"],
        origin="lower",
        extent=[lons[0], lons[-1], lats[0], lats[-1]],
        cmap="magma",
        vmin=0.2,
        vmax=1.8
    )
    axes[1, 1].set_title("(D) Wavelet Shannon Energy Entropy (Multi-Scale Dispersion)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Longitude (°E)")
    axes[1, 1].set_ylabel("Latitude (°N)")
    plt.colorbar(im3, ax=axes[1, 1], shrink=0.8, label="Entropy (bits)")
    axes[1, 1].grid(True, linestyle=":", alpha=0.4)

    plt.suptitle("Onboard Satellite Edge Feature Maps — India Gridded Domain (2024)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(p_dir / "plot9_feature_maps_spatial.png", dpi=300)
    plt.close()

    print("Stage 3 Feature plots 8 and 9 generated successfully!")
