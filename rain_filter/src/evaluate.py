"""Evaluation runner script for rainfall filtering experiments.

Evaluates custom filters against SciPy baselines across test cities,
noise conditions, computes metrics, writes outputs/results.csv, and produces plots.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import csv
import numpy as np

from rain_filter.src.cost import (
    benchmark_cube_runtime,
    benchmark_q15_accuracy,
    get_state_memory_footprint,
    get_theoretical_ops_per_sample,
)
from rain_filter.src.io_utils import (
    extract_cell,
    generate_synthetic_rainfall,
    load_dataset,
)
from rain_filter.src.metrics import (
    baseline_butterworth,
    baseline_median,
    baseline_moving_average,
    baseline_savgol,
    compute_metrics,
)
from rain_filter.src.noise import (
    add_dropouts,
    add_gaussian_noise,
    add_impulsive_noise,
    add_mixed_noise,
)
from rain_filter.src.pipeline import denoise
from rain_filter.src.plots import (
    plot1_monsoon_timeseries,
    plot2_frequency_response,
    plot3_adaptive_gain_overlay,
    plot4_snr_improvement_bars,
    plot5_peak_scatter,
)


def run_full_evaluation(
    dataset_path: str = "Dataset/RF25_ind2024_rfp25.nc",
    output_dir: str = "rain_filter/outputs"
) -> Tuple[List[Dict], Dict]:
    """Run full benchmark evaluation on IMD dataset.

    Parameters
    ----------
    dataset_path : str
        Path to IMD NetCDF file.
    output_dir : str
        Path to output folder for CSV and plots.

    Returns
    -------
    all_results : list of dicts
        Detailed metric rows for all experiments.
    summary : dict
        Summary statistics and acceptance checks.
    """
    out_path = Path(output_dir)
    plots_dir = out_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    dataset_file = Path(dataset_path)
    using_synthetic = False
    if not dataset_file.exists():
        print(f"[WARNING] Dataset not found at {dataset_path}. Generating synthetic series...")
        using_synthetic = True
        lats = np.array([19.0, 22.5, 25.25, 26.9])
        lons = np.array([70.9, 72.75, 88.25, 91.75])
        rain_cube = np.zeros((365, len(lats), len(lons)), dtype=np.float64)
        for i in range(len(lats)):
            for j in range(len(lons)):
                rain_cube[:, i, j] = generate_synthetic_rainfall(seed=i * 10 + j)
        times = np.arange(365)
    else:
        rain_cube, times, lats, lons = load_dataset(dataset_file)

    cities = {
        "Kolkata": (22.5, 88.25),
        "Cherrapunji": (25.25, 91.75),
        "Mumbai": (19.0, 72.75),
        "Jaisalmer": (26.9, 70.9)
    }

    # Noise definitions
    noise_scenarios = [
        ("Gaussian_0.5mm", lambda x: add_gaussian_noise(x, sigma=0.5, seed=101)),
        ("Gaussian_1.0mm", lambda x: add_gaussian_noise(x, sigma=1.0, seed=102)),
        ("Gaussian_2.0mm", lambda x: add_gaussian_noise(x, sigma=2.0, seed=103)),
        ("Gaussian_5.0mm", lambda x: add_gaussian_noise(x, sigma=5.0, seed=104)),
        ("Impulsive_1%", lambda x: add_impulsive_noise(x, fraction=0.01, seed=201)),
        ("Impulsive_2%", lambda x: add_impulsive_noise(x, fraction=0.02, seed=202)),
        ("Impulsive_5%", lambda x: add_impulsive_noise(x, fraction=0.05, seed=203)),
        ("Mixed_Gauss1_Imp2%", lambda x: add_mixed_noise(x, sigma=1.0, impulse_fraction=0.02, seed=301)),
        ("Dropouts_2%", lambda x: add_dropouts(x, fraction=0.02, seed=401))
    ]

    # Models / filters to evaluate
    filter_models = [
        ("Hybrid (Ours)", lambda n: denoise(n, mode="hybrid", preserve_volume=False)),
        ("Hybrid+VolPreserve", lambda n: denoise(n, mode="hybrid", preserve_volume=True)),
        ("IIR_Only (Ours)", lambda n: denoise(n, mode="iir")),
        ("FIR_Only (Ours)", lambda n: denoise(n, mode="fir")),
        ("Full (Ours)", lambda n: denoise(n, mode="full")),
        ("MA_5", lambda n: baseline_moving_average(n, window=5)),
        ("Median_5", lambda n: baseline_median(n, window=5)),
        ("Butterworth_3", lambda n: baseline_butterworth(n, order=3, wn=0.35)),
        ("SavitzkyGolay_11", lambda n: baseline_savgol(n, window_length=11, polyorder=3))
    ]

    all_results = []
    # Data structure for bar plot: {noise_type: {filter_name: avg_snr_imp}}
    bar_data: Dict[str, Dict[str, List[float]]] = {}

    print("\nStarting evaluation across 4 benchmark stations...")

    for city_name, (target_lat, target_lon) in cities.items():
        clean_series, lat_idx, lon_idx = extract_cell(rain_cube, lats, lons, target_lat, target_lon)
        # If sea cell or missing data, fill
        if np.all(np.isnan(clean_series)):
            clean_series = generate_synthetic_rainfall(seed=42)

        for noise_name, noise_fn in noise_scenarios:
            if noise_name not in bar_data:
                bar_data[noise_name] = {}

            noisy_series = noise_fn(clean_series)

            for filter_name, filter_fn in filter_models:
                filtered_series = filter_fn(noisy_series)
                metrics = compute_metrics(clean_series, noisy_series, filtered_series)

                row = {
                    "city": city_name,
                    "noise_type": noise_name,
                    "filter": filter_name,
                    "snr_improvement_db": round(metrics["snr_improvement_db"], 2),
                    "mse": round(metrics["mse"], 4),
                    "mae": round(metrics["mae"], 4),
                    "relative_bias_pct": round(metrics["relative_bias_pct"], 2),
                    "peak_preservation_ratio": round(metrics["peak_preservation_ratio"], 3),
                    "wet_f1": round(metrics["wet_f1"], 3)
                }
                all_results.append(row)

                if filter_name not in bar_data[noise_name]:
                    bar_data[noise_name][filter_name] = []
                bar_data[noise_name][filter_name].append(metrics["snr_improvement_db"])

    # 2. Write CSV
    csv_file = out_path / "results.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "city", "noise_type", "filter", "snr_improvement_db",
            "mse", "mae", "relative_bias_pct", "peak_preservation_ratio", "wet_f1"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Results successfully saved to {csv_file}")

    # 3. Generate All 5 Required Plots
    print("Generating plots...")

    # Plot 1: Mumbai monsoon series
    mumbai_clean, _, _ = extract_cell(rain_cube, lats, lons, 19.0, 72.75)
    mumbai_noisy = add_mixed_noise(mumbai_clean, sigma=1.0, impulse_fraction=0.02, seed=42)
    plot1_monsoon_timeseries(
        mumbai_clean,
        mumbai_noisy,
        city_name="Mumbai",
        save_path=plots_dir / "plot1_monsoon_timeseries.png"
    )

    # Plot 2: Frequency response
    plot2_frequency_response(
        save_path=plots_dir / "plot2_frequency_response.png"
    )

    # Plot 3: Cherrapunji adaptive gain overlay
    cherra_clean, _, _ = extract_cell(rain_cube, lats, lons, 25.25, 91.75)
    plot3_adaptive_gain_overlay(
        cherra_clean,
        city_name="Cherrapunji",
        save_path=plots_dir / "plot3_adaptive_gain_overlay.png"
    )

    # Plot 4: Bar chart of SNR improvement across noise conditions
    # Average SNR improvement across cities
    avg_bar_data = {}
    for nt, f_dict in bar_data.items():
        avg_bar_data[nt] = {f: float(np.mean(vals)) for f, vals in f_dict.items()}
    plot4_snr_improvement_bars(
        avg_bar_data,
        save_path=plots_dir / "plot4_snr_improvement_bars.png"
    )

    # Plot 5: Peak Scatter for all test cells combined
    all_clean_peaks = []
    all_noisy_peaks = []
    for cname, (tlat, tlon) in cities.items():
        c_ser, _, _ = extract_cell(rain_cube, lats, lons, tlat, tlon)
        n_ser = add_mixed_noise(c_ser, sigma=1.0, impulse_fraction=0.02, seed=42)
        all_clean_peaks.append(c_ser)
        all_noisy_peaks.append(n_ser)
    combined_clean = np.concatenate(all_clean_peaks)
    combined_noisy = np.concatenate(all_noisy_peaks)
    plot5_peak_scatter(
        combined_clean,
        combined_noisy,
        save_path=plots_dir / "plot5_peak_scatter.png",
        threshold=25.0
    )

    print("All 5 plots saved to outputs/plots/")

    # 4. Summary & Acceptance Check
    # Filter for Mixed_Gauss1_Imp2%
    mixed_rows = [r for r in all_results if r["noise_type"] == "Mixed_Gauss1_Imp2%"]
    hybrid_rows = [r for r in mixed_rows if r["filter"] == "Hybrid (Ours)"]
    vol_rows = [r for r in mixed_rows if r["filter"] == "Hybrid+VolPreserve"]

    summary = {
        "using_synthetic": using_synthetic,
        "hybrid_mixed_results": hybrid_rows,
        "volume_preservation_results": vol_rows
    }

    return all_results, summary


if __name__ == "__main__":
    results, summary = run_full_evaluation()
