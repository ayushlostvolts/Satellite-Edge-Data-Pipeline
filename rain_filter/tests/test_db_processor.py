"""Unit tests for the Database Ingestion and Batch Processing Engine."""

from pathlib import Path
import sqlite3
import tempfile
import numpy as np
import pytest

from rain_filter.src.db_processor import (
    process_netcdf_database,
    process_csv_database,
    process_sqlite_database,
    export_database_results_csv
)


@pytest.fixture
def sample_csv_file(tmp_path):
    """Create a temporary multi-station CSV dataset."""
    csv_file = tmp_path / "test_rainfall_stations.csv"
    rng = np.random.default_rng(42)
    # 5 stations, 100 days
    data = rng.exponential(scale=10.0, size=(100, 5))
    data[data < 2.0] = 0.0
    # Add a huge storm peak at station 2
    data[50, 2] = 185.0

    header = "Day,Station_North,Station_South,Station_Extreme,Station_Coast,Station_Inland\n"
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write(header)
        for t in range(100):
            row = [f"{t+1}"] + [f"{val:.2f}" for val in data[t]]
            f.write(",".join(row) + "\n")

    return csv_file


@pytest.fixture
def sample_sqlite_file(tmp_path):
    """Create a temporary SQLite rainfall database."""
    db_file = tmp_path / "test_rainfall.db"
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("CREATE TABLE rainfall_records (day INTEGER, rain_station1 REAL, rain_station2 REAL);")

    rng = np.random.default_rng(123)
    for day in range(1, 91):
        r1 = max(0.0, float(rng.normal(15.0, 8.0)))
        r2 = max(0.0, float(rng.normal(5.0, 4.0)))
        cur.execute("INSERT INTO rainfall_records VALUES (?, ?, ?);", (day, r1, r2))

    conn.commit()
    conn.close()
    return db_file


def test_process_csv_database(sample_csv_file):
    """Verify processing tabular CSV multi-station datasets."""
    res = process_csv_database(sample_csv_file, filter_mode="hybrid", max_hotspots=5)
    assert res["database_type"] == "Tabular CSV / Time Series Matrix"
    assert res["summary"]["total_samples"] > 0
    assert len(res["hotspots"]) > 0

    top_hotspot = res["hotspots"][0]
    assert top_hotspot["rank"] == 1
    assert top_hotspot["p_max_raw_mm"] >= 180.0
    assert "ceps_score" in top_hotspot


def test_process_sqlite_database(sample_sqlite_file):
    """Verify processing SQLite rainfall databases."""
    res = process_sqlite_database(sample_sqlite_file, filter_mode="hybrid")
    assert "summary" in res
    assert res["summary"]["total_samples"] > 0
    assert len(res["hotspots"]) > 0


def test_export_database_results_csv(sample_csv_file, tmp_path):
    """Verify exporting database analysis summary report to CSV."""
    res = process_csv_database(sample_csv_file)
    out_csv = tmp_path / "analysis_summary.csv"
    exported_path = export_database_results_csv(res, out_csv)

    assert exported_path.exists()
    content = exported_path.read_text(encoding="utf-8")
    assert "SLUSH SATELLITE EDGE LAB" in content
    assert "rank" in content.lower()


def test_process_netcdf_database_if_available():
    """Verify full NetCDF database processing on bundled IMD 2024 dataset."""
    imd_path = Path("Dataset/RF25_ind2024_rfp25.nc")
    if not imd_path.exists():
        pytest.skip("IMD NetCDF dataset not found on disk.")

    res = process_netcdf_database(imd_path, filter_mode="hybrid", max_hotspots=5)
    assert res["database_type"] == "NetCDF Gridded Cube (.nc)"
    assert res["summary"]["total_samples"] == 6373890
    assert res["summary"]["land_cells_count"] > 4000
    assert res["summary"]["overall_max_daily_mm"] > 500.0
    assert len(res["hotspots"]) == 5
    assert "spatial_maps" in res
    assert "max_rain" in res["spatial_maps"]
