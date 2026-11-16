"""Database and Batch Dataset Processing Engine for Satellite Edge Rainfall Pipeline.

Handles ingestion, vectorized batch computation, spatial aggregation, and extreme hotspot
detection for:
1. NetCDF Climate Gridded Datasets (.nc, .nc4) [e.g. IMD 0.25°/0.5°/1° gridded cubes]
2. Multi-station / Gridded Tabular CSV Datasets (.csv, .tsv, .txt)
3. SQLite Databases (.db, .sqlite, .sqlite3)

Vectorized over space, computing denoise, wavelet subband energies, and physical indices.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import sqlite3
import time
import numpy as np

from rain_filter.src.pipeline import denoise
from rain_filter.src.dwt import dwt_1d
from rain_filter.src.features import extract_features_1d
from rain_filter.src.ranking import rank_features
from rain_filter.src.cost import get_theoretical_ops_per_sample


def process_netcdf_database(
    filepath: Union[str, Path],
    filter_mode: str = "hybrid",
    preserve_volume: bool = True,
    max_hotspots: int = 10
) -> Dict[str, Any]:
    """Ingest and process a full NetCDF gridded rainfall dataset.

    Parameters
    ----------
    filepath : str or Path
        Path to the NetCDF file (.nc, .nc4).
    filter_mode : str
        Filtering mode ('hybrid', 'iir', 'fir', 'full').
    preserve_volume : bool
        Whether to enforce 30-day block volume preservation.
    max_hotspots : int
        Number of top extreme storm hotspots to identify.

    Returns
    -------
    Dict[str, Any]
        Comprehensive analysis dictionary containing spatial grids, hotspots, and telemetry.
    """
    import netCDF4

    t_start = time.time()
    filepath = Path(filepath)

    with netCDF4.Dataset(filepath, mode="r") as ds:
        # Detect rainfall variable
        var_name = None
        for key in ds.variables.keys():
            k_lower = key.lower()
            if any(term in k_lower for term in ["rain", "precip", "rf", "pr"]):
                var_name = key
                break
        if var_name is None:
            # Fallback to last 3D variable
            for key, val in ds.variables.items():
                if len(val.dimensions) >= 3:
                    var_name = key
                    break

        if var_name is None:
            raise ValueError(f"No suitable 3D rainfall variable found in NetCDF file {filepath.name}.")

        raw_cube = np.array(ds.variables[var_name][:], dtype=np.float32)

        # Detect coordinate dimensions
        lats = None
        lons = None
        times = None
        for k in ds.variables.keys():
            k_low = k.lower()
            if "lat" in k_low:
                lats = np.array(ds.variables[k][:], dtype=np.float64)
            elif "lon" in k_low:
                lons = np.array(ds.variables[k][:], dtype=np.float64)
            elif "time" in k_low or "day" in k_low:
                times = np.array(ds.variables[k][:])

    if lats is None:
        lats = np.linspace(6.5, 38.5, raw_cube.shape[1])
    if lons is None:
        lons = np.linspace(66.5, 100.0, raw_cube.shape[2])

    t_read = time.time() - t_start

    # Treat negative or fill values as NaN
    raw_cube[raw_cube < 0] = np.nan
    raw_cube[raw_cube > 2000] = np.nan

    T, n_lat, n_lon = raw_cube.shape
    total_samples = raw_cube.size

    # Valid land mask (cells that have at least one valid measurement)
    valid_mask = ~np.isnan(raw_cube).all(axis=0)
    land_cells_count = int(np.sum(valid_mask))

    # 1. Vectorized Denoise on Full Cube
    t_filt_start = time.time()
    filt_cube = denoise(raw_cube, mode=filter_mode, preserve_volume=preserve_volume)
    t_filt = time.time() - t_filt_start

    # 2. Spatial Summary Grids
    with np.errstate(all="ignore"):
        raw_max_grid = np.nanmax(raw_cube, axis=0)
        raw_sum_grid = np.nansum(raw_cube, axis=0)
        filt_max_grid = np.nanmax(filt_cube, axis=0)
        filt_sum_grid = np.nansum(filt_cube, axis=0)
        # Wet days count (rain >= 2.5mm)
        wet_days_grid = np.nansum(raw_cube >= 2.5, axis=0).astype(np.float32)

    # Replace all-NaN cells with 0 or None for display
    raw_max_grid[~valid_mask] = 0.0
    raw_sum_grid[~valid_mask] = 0.0
    filt_max_grid[~valid_mask] = 0.0
    filt_sum_grid[~valid_mask] = 0.0
    wet_days_grid[~valid_mask] = 0.0

    # 3. Detect Top Extreme Storm Hotspots
    flat_peaks = raw_max_grid.copy()
    flat_peaks[~valid_mask] = -1.0
    top_flat_indices = np.argsort(flat_peaks.ravel())[::-1][:max_hotspots]

    hotspots = []
    for rank_idx, idx in enumerate(top_flat_indices, 1):
        i_lat, j_lon = np.unravel_index(idx, (n_lat, n_lon))
        peak_val = float(raw_max_grid[i_lat, j_lon])
        if peak_val <= 0:
            continue

        lat_val = float(lats[i_lat])
        lon_val = float(lons[j_lon])
        cell_raw = raw_cube[:, i_lat, j_lon]
        cell_filt = filt_cube[:, i_lat, j_lon]

        # Extract DWT and features for hotspot
        dwt_res = dwt_1d(cell_filt, wavelet="haar", level=3)
        feats = extract_features_1d(cell_filt, dwt_res=dwt_res)

        # CEPS Edge Priority
        ceps = (
            0.35 * min(1.0, peak_val / 200.0) +
            0.25 * (feats.get("convective_energy_ratio_pct", 10.0) / 100.0) +
            0.25 * min(1.0, feats.get("total_rainfall_mm", 1000.0) / 4000.0) +
            0.15 * min(1.0, feats.get("sdii_intensity_mm", 15.0) / 50.0)
        )

        hotspots.append({
            "rank": rank_idx,
            "lat": round(lat_val, 2),
            "lon": round(lon_val, 2),
            "lat_idx": int(i_lat),
            "lon_idx": int(j_lon),
            "p_max_raw_mm": round(peak_val, 1),
            "p_max_filt_mm": round(float(filt_max_grid[i_lat, j_lon]), 1),
            "total_rainfall_mm": round(float(raw_sum_grid[i_lat, j_lon]), 1),
            "wet_days": int(wet_days_grid[i_lat, j_lon]),
            "convective_ratio_pct": round(feats.get("convective_energy_ratio_pct", 0.0), 1),
            "ceps_score": round(float(ceps), 3)
        })

    total_time = time.time() - t_start
    throughput = (total_samples / t_filt) if t_filt > 0 else 0

    # Downsample spatial grids for fast web canvas rendering (max 64x64 resolution)
    def _downsample_grid(grid: np.ndarray, target_rows: int = 50, target_cols: int = 50):
        r_step = max(1, grid.shape[0] // target_rows)
        c_step = max(1, grid.shape[1] // target_cols)
        sampled = grid[::r_step, ::c_step]
        # Clean NaNs to 0
        cleaned = np.nan_to_num(sampled, nan=0.0)
        return [[round(float(v), 1) for v in row] for row in cleaned]

    overall_max = float(np.nanmax(raw_max_grid))
    overall_total_vol = float(np.nansum(raw_sum_grid))

    return {
        "database_type": "NetCDF Gridded Cube (.nc)",
        "file_name": filepath.name,
        "grid_shape": [int(T), int(n_lat), int(n_lon)],
        "dimensions": {
            "time_steps": int(T),
            "latitudes": int(n_lat),
            "longitudes": int(n_lon),
            "lat_min": round(float(np.min(lats)), 2),
            "lat_max": round(float(np.max(lats)), 2),
            "lon_min": round(float(np.min(lons)), 2),
            "lon_max": round(float(np.max(lons)), 2)
        },
        "summary": {
            "total_samples": int(total_samples),
            "land_cells_count": land_cells_count,
            "overall_max_daily_mm": round(overall_max, 1),
            "overall_total_volume_mm": round(overall_total_vol, 1),
            "extreme_days_above_100mm": int(np.nansum(raw_cube >= 100.0)),
            "extreme_days_above_50mm": int(np.nansum(raw_cube >= 50.0)),
            "total_compute_time_s": round(total_time, 3),
            "filtering_time_s": round(t_filt, 3),
            "throughput_samples_per_sec": int(throughput)
        },
        "spatial_maps": {
            "max_rain": _downsample_grid(raw_max_grid),
            "filt_max_rain": _downsample_grid(filt_max_grid),
            "total_rain": _downsample_grid(raw_sum_grid),
            "wet_days": _downsample_grid(wet_days_grid)
        },
        "hotspots": hotspots
    }


def process_csv_database(
    filepath: Union[str, Path],
    filter_mode: str = "hybrid",
    preserve_volume: bool = True,
    max_hotspots: int = 10
) -> Dict[str, Any]:
    """Ingest and process a tabular CSV or text dataset.

    Supports:
    - Multi-column series (columns = stations, rows = daily timesteps)
    - Single column 1D time series
    - Table format (date, station_id, rainfall)
    """
    t_start = time.time()
    filepath = Path(filepath)

    # Try reading header
    lines = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(5000):
            line = f.readline()
            if not line:
                break
            line_str = line.strip()
            if line_str:
                lines.append(line_str)

    if not lines:
        raise ValueError("Uploaded file is empty.")

    # Determine delimiter
    sample = lines[0]
    delimiter = "," if "," in sample else ("\t" if "\t" in sample else None)

    # Check if header row exists
    header_tokens = sample.split(delimiter) if delimiter else [sample]
    has_header = False
    try:
        float(header_tokens[0])
    except ValueError:
        has_header = True

    # Parse full file with numpy
    skip_header = 1 if has_header else 0
    try:
        raw_data = np.genfromtxt(
            filepath,
            delimiter=delimiter,
            skip_header=skip_header,
            filling_values=np.nan
        )
    except Exception:
        # Fallback manual parse
        numeric_rows = []
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            if has_header:
                f.readline()
            for line in f:
                parts = line.strip().split(delimiter)
                row = []
                for p in parts:
                    try:
                        row.append(float(p))
                    except ValueError:
                        row.append(np.nan)
                if row:
                    numeric_rows.append(row)
        raw_data = np.array(numeric_rows, dtype=np.float32)

    if raw_data.ndim == 1:
        raw_data = raw_data[:, np.newaxis]

    T, n_cols = raw_data.shape
    station_names = (
        header_tokens if has_header and len(header_tokens) == n_cols
        else [f"Station_{i+1}" for i in range(n_cols)]
    )

    return _process_2d_tabular_data(
        raw_data=raw_data,
        station_names=station_names,
        file_name=filepath.name,
        db_type="Tabular CSV / Time Series Matrix",
        filter_mode=filter_mode,
        preserve_volume=preserve_volume,
        max_hotspots=max_hotspots,
        t_start=t_start
    )


def _process_2d_tabular_data(
    raw_data: np.ndarray,
    station_names: List[str],
    file_name: str,
    db_type: str,
    filter_mode: str = "hybrid",
    preserve_volume: bool = True,
    max_hotspots: int = 10,
    t_start: Optional[float] = None
) -> Dict[str, Any]:
    """Shared batch processor for 2D tabular matrices (CSV or SQLite)."""
    if t_start is None:
        t_start = time.time()

    if raw_data.ndim == 1:
        raw_data = raw_data[:, np.newaxis]

    T, n_cols = raw_data.shape
    if len(station_names) < n_cols:
        station_names = station_names + [f"Station_{i+1}" for i in range(len(station_names), n_cols)]

    # Treat negative values as NaN
    raw_data = raw_data.copy()
    raw_data[raw_data < 0] = np.nan

    # Vectorized denoise across all stations
    t_filt_start = time.time()
    filt_data = denoise(raw_data, mode=filter_mode, preserve_volume=preserve_volume)
    t_filt = time.time() - t_filt_start

    # Station summaries
    with np.errstate(all="ignore"):
        peaks = np.nanmax(raw_data, axis=0)
        totals = np.nansum(raw_data, axis=0)
        filt_peaks = np.nanmax(filt_data, axis=0)

    top_indices = np.argsort(peaks)[::-1][:max_hotspots]
    hotspots = []
    for rank_idx, col_idx in enumerate(top_indices, 1):
        st_name = station_names[col_idx]
        col_raw = raw_data[:, col_idx]
        col_filt = filt_data[:, col_idx]

        dwt_res = dwt_1d(col_filt, wavelet="haar", level=3)
        feats = extract_features_1d(col_filt, dwt_res=dwt_res)

        peak_v = float(peaks[col_idx])
        ceps = (
            0.4 * min(1.0, peak_v / 200.0) +
            0.3 * (feats.get("convective_energy_ratio_pct", 10.0) / 100.0) +
            0.3 * min(1.0, feats.get("total_rainfall_mm", 1000.0) / 4000.0)
        )

        hotspots.append({
            "rank": rank_idx,
            "station_name": st_name,
            "p_max_raw_mm": round(peak_v, 1),
            "p_max_filt_mm": round(float(filt_peaks[col_idx]), 1),
            "total_rainfall_mm": round(float(totals[col_idx]), 1),
            "convective_ratio_pct": round(feats.get("convective_energy_ratio_pct", 0.0), 1),
            "ceps_score": round(float(ceps), 3)
        })

    total_time = time.time() - t_start

    return {
        "database_type": db_type,
        "file_name": file_name,
        "grid_shape": [int(T), int(n_cols)],
        "summary": {
            "total_samples": int(raw_data.size),
            "stations_count": int(n_cols),
            "time_steps": int(T),
            "overall_max_daily_mm": round(float(np.nanmax(peaks)), 1),
            "overall_total_volume_mm": round(float(np.nansum(totals)), 1),
            "total_compute_time_s": round(total_time, 3),
            "filtering_time_s": round(t_filt, 3)
        },
        "hotspots": hotspots
    }


def process_sqlite_database(
    filepath: Union[str, Path],
    filter_mode: str = "hybrid",
    preserve_volume: bool = True,
    max_hotspots: int = 10
) -> Dict[str, Any]:
    """Ingest and process an SQLite database containing rainfall tables."""
    t_start = time.time()
    filepath = Path(filepath)
    conn = sqlite3.connect(str(filepath))
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cursor.fetchall() if not r[0].startswith("sqlite_")]

    if not tables:
        conn.close()
        raise ValueError("SQLite database contains no tables.")

    target_table = tables[0]
    cursor.execute(f"PRAGMA table_info({target_table});")
    columns = [col[1] for col in cursor.fetchall()]

    cursor.execute(f"SELECT * FROM {target_table} LIMIT 10000;")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        raise ValueError(f"Table {target_table} in SQLite database has no records.")

    # Convert to numeric array
    num_data = []
    for row in rows:
        num_row = []
        for val in row:
            try:
                num_row.append(float(val))
            except (ValueError, TypeError):
                pass
        if num_row:
            num_data.append(num_row)

    if not num_data:
        raise ValueError("Could not extract numerical time series from SQLite table.")

    arr = np.array(num_data, dtype=np.float32)
    return _process_2d_tabular_data(
        raw_data=arr,
        station_names=columns[:arr.shape[1]],
        file_name=filepath.name,
        db_type="SQLite Relational Database (.db)",
        filter_mode=filter_mode,
        preserve_volume=preserve_volume,
        max_hotspots=max_hotspots,
        t_start=t_start
    )


def export_database_results_csv(analysis: Dict[str, Any], out_path: Union[str, Path]) -> Path:
    """Generate a clean CSV summary report from database analysis results."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# SLUSH SATELLITE EDGE LAB -- DATABASE ANALYSIS REPORT\n")
        f.write(f"# File: {analysis.get('file_name', 'Unknown')}\n")
        f.write(f"# Database Type: {analysis.get('database_type', 'Unknown')}\n")
        summary = analysis.get("summary", {})
        f.write(f"# Total Samples: {summary.get('total_samples', 0)}\n")
        f.write(f"# Overall Peak (mm): {summary.get('overall_max_daily_mm', 0)}\n")
        f.write(f"# Compute Time (s): {summary.get('total_compute_time_s', 0)}\n\n")

        # Table of Hotspots
        hotspots = analysis.get("hotspots", [])
        if hotspots:
            headers = list(hotspots[0].keys())
            f.write(",".join(headers) + "\n")
            for h in hotspots:
                row = [str(h.get(k, "")) for k in headers]
                f.write(",".join(row) + "\n")

    return out_path

