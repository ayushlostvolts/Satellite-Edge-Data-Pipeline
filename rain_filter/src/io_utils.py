"""I/O and preprocessing utilities for IMD gridded daily rainfall data.

Handles loading NetCDF files, extracting specific grid cells, masking -999 and NaN
values, and generating synthetic zero-inflated Gamma rainfall series for testing.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import xarray as xr


def load_dataset(
    file_path: Union[str, Path],
    var_name: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load gridded rainfall NetCDF dataset.

    Parameters
    ----------
    file_path : str or Path
        Path to the NetCDF file (e.g., RF25_ind2024_rfp25.nc).
    var_name : str, optional
        Name of rainfall variable. If None, auto-detected (e.g. 'RAINFALL').

    Returns
    -------
    rain : np.ndarray
        Rainfall array of shape (T, lat, lon) with missing values as np.nan.
    times : np.ndarray
        Array of time coordinates.
    lats : np.ndarray
        Array of latitude coordinates.
    lons : np.ndarray
        Array of longitude coordinates.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    # Open with mask_and_scale=False first to observe raw -999, then standardize
    with xr.open_dataset(file_path) as ds:
        if var_name is None:
            # Look for common rainfall variable names
            candidates = ["RAINFALL", "rainfall", "rf", "precip", "rain"]
            for cand in candidates:
                if cand in ds.data_vars:
                    var_name = cand
                    break
            if var_name is None:
                var_name = list(ds.data_vars.keys())[0]

        da = ds[var_name]
        rain = da.values.copy()

        # Find dimension coordinate names
        time_dim = None
        lat_dim = None
        lon_dim = None

        for d in da.dims:
            d_lower = str(d).lower()
            if "time" in d_lower:
                time_dim = d
            elif "lat" in d_lower:
                lat_dim = d
            elif "lon" in d_lower:
                lon_dim = d

        times = ds[time_dim].values if time_dim is not None else np.arange(rain.shape[0])
        lats = ds[lat_dim].values if lat_dim is not None else np.arange(rain.shape[1])
        lons = ds[lon_dim].values if lon_dim is not None else np.arange(rain.shape[2])

    # Standardize missing values: any value <= -900 or NaN becomes np.nan
    mask = (rain <= -900.0) | np.isnan(rain)
    rain[mask] = np.nan

    return rain, times, lats, lons


def extract_cell(
    rain: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    target_lat: float,
    target_lon: float
) -> Tuple[np.ndarray, int, int]:
    """Extract 1-D daily rainfall series for nearest grid cell to target coordinates.

    Parameters
    ----------
    rain : np.ndarray
        Rainfall array of shape (T, lat, lon).
    lats : np.ndarray
        1D array of latitude coordinates.
    lons : np.ndarray
        1D array of longitude coordinates.
    target_lat : float
        Target latitude.
    target_lon : float
        Target longitude.

    Returns
    -------
    series : np.ndarray
        1D array of daily rainfall of shape (T,).
    lat_idx : int
        Index of nearest latitude.
    lon_idx : int
        Index of nearest longitude.
    """
    lat_idx = int(np.argmin(np.abs(lats - target_lat)))
    lon_idx = int(np.argmin(np.abs(lons - target_lon)))
    series = rain[:, lat_idx, lon_idx].copy()
    return series, lat_idx, lon_idx


def generate_synthetic_rainfall(
    length: int = 365,
    p_wet: float = 0.28,
    shape: float = 0.75,
    scale: float = 25.0,
    seed: int = 42
) -> np.ndarray:
    """Generate a realistic synthetic daily rainfall series.

    Rainfall is zero-inflated and heavy-tailed, modeled as a Bernoulli-Gamma mixture:
    P(rain > 0) = p_wet, and wet amounts follow Gamma(shape, scale).

    Parameters
    ----------
    length : int
        Length of series (default 365 days).
    p_wet : float
        Probability of a day being a wet day (> 0).
    shape : float
        Gamma distribution shape parameter k.
    scale : float
        Gamma distribution scale parameter theta.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    series : np.ndarray
        Non-negative 1D rainfall array of shape (length,).
    """
    rng = np.random.default_rng(seed)
    wet_mask = rng.random(length) < p_wet
    amounts = rng.gamma(shape=shape, scale=scale, size=length)
    series = np.zeros(length, dtype=np.float64)
    series[wet_mask] = amounts[wet_mask]
    return series
