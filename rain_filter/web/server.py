"""Lightweight HTTP server and API backend for the Slush Satellite Edge Frontend.

Provides REST API endpoints for:
- /api/stations : List available stations and metadata from the IMD NetCDF dataset
- /api/process  : Execute complete pipeline (Denoise -> DWT -> Features -> Ranking)
- /api/benchmark: Fetch comparative metrics against SciPy baselines from results.csv
- Static file serving for web/index.html
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
import json
import mimetypes
import os
import sys
import urllib.parse
import numpy as np

# Ensure project root is in python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "rain_filter"))

from rain_filter.src.io_utils import load_dataset, extract_cell, generate_synthetic_rainfall
from rain_filter.src.pipeline import denoise
from rain_filter.src.noise import add_gaussian_noise, add_impulsive_noise, add_mixed_noise, add_dropouts
from rain_filter.src.metrics import (
    compute_metrics,
    baseline_butterworth,
    baseline_median,
    baseline_moving_average,
    baseline_savgol
)
from rain_filter.src.dwt import dwt_1d
from rain_filter.src.features import extract_features_1d
from rain_filter.src.ranking import rank_features
from rain_filter.src.cost import (
    get_theoretical_ops_per_sample,
    get_state_memory_footprint,
    benchmark_q15_accuracy
)
from rain_filter.src.db_processor import (
    process_netcdf_database,
    process_csv_database,
    process_sqlite_database,
    export_database_results_csv
)

# Global database analysis cache
LATEST_DATABASE_ANALYSIS = None

# Global dataset cache
DATASET_FILE = project_root / "Dataset" / "RF25_ind2024_rfp25.nc"
RAIN_CUBE = None
TIMES = None
LATS = None
LONS = None

try:
    if DATASET_FILE.exists():
        print(f"[Backend] Loading IMD NetCDF dataset: {DATASET_FILE}")
        RAIN_CUBE, TIMES, LATS, LONS = load_dataset(DATASET_FILE)
        print(f"[Backend] Loaded grid: {RAIN_CUBE.shape}")
    else:
        print("[Backend] Warning: NetCDF dataset not found. Generating synthetic fallback cube.")
        RAIN_CUBE = np.zeros((366, 129, 135), dtype=np.float32)
        LATS = np.linspace(6.5, 38.5, 129)
        LONS = np.linspace(66.5, 100.0, 135)
except Exception as e:
    print(f"[Backend] Error loading dataset: {e}")

# Precompute background contrast population once at startup for sub-millisecond ranking
PRECOMPUTED_CONTRAST_SAMPLES = []
PRECOMPUTED_FEATURE_NAMES = None
try:
    print("[Backend] Precomputing contrast background population for feature ranking...")
    for i in range(15):
        syn_c = generate_synthetic_rainfall(shape=0.5 + i * 0.05, scale=10 + i * 2, seed=i)
        syn_f = denoise(syn_c, mode="hybrid")
        syn_ft = extract_features_1d(syn_f)
        if PRECOMPUTED_FEATURE_NAMES is None:
            PRECOMPUTED_FEATURE_NAMES = list(syn_ft.keys())
        PRECOMPUTED_CONTRAST_SAMPLES.append([syn_ft[k] for k in PRECOMPUTED_FEATURE_NAMES])
    print(f"[Backend] Precomputed {len(PRECOMPUTED_CONTRAST_SAMPLES)} contrast samples for {len(PRECOMPUTED_FEATURE_NAMES)} features.")
except Exception as e:
    print(f"[Backend] Notice during contrast precomputation: {e}")


PRESET_CITIES = {
    "Mumbai": {"lat": 19.0, "lon": 72.75, "tag": "Western Ghats Monsoon", "color": "#4da2ff"},
    "Cherrapunji": {"lat": 25.25, "lon": 91.75, "tag": "Extreme Meghalaya Rainfall", "color": "#fb4903"},
    "Kolkata": {"lat": 22.5, "lon": 88.25, "tag": "Gangetic Delta Convection", "color": "#55db9c"},
    "Jaisalmer": {"lat": 26.9, "lon": 70.9, "tag": "Thar Desert Arid Spells", "color": "#ffd731"}
}


class SlushAPIHandler(SimpleHTTPRequestHandler):
    """HTTP request handler serving Slush frontend and JSON API."""

    def __init__(self, *args, **kwargs):
        # Serve static files from rain_filter/web directory
        super().__init__(*args, directory=str(current_dir), **kwargs)

    def end_headers(self):
        # Prevent browser aggressive caching of HTML and API responses
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self):
        url_parsed = urllib.parse.urlparse(self.path)
        path = url_parsed.path

        if path == "/api/stations":
            self.send_json_response(PRESET_CITIES)
        elif path == "/api/telemetry":
            telemetry = {
                "ops": get_theoretical_ops_per_sample(),
                "memory": get_state_memory_footprint()
            }
            self.send_json_response(telemetry)
        elif path == "/api/download-db-results":
            report_file = project_root / "rain_filter" / "outputs" / "database_analysis_report.csv"
            if not report_file.exists():
                self.send_error(404, "No database analysis report generated yet.")
                return
            csv_bytes = report_file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.send_header("Content-Disposition", 'attachment; filename="database_analysis_report.csv"')
            self.send_header("Content-Length", str(len(csv_bytes)))
            self.end_headers()
            self.wfile.write(csv_bytes)
        elif path == "/api/download-manual":
            pdf_path = project_root / "rain_filter" / "outputs" / "RUNNING_GUIDE_MANUAL.pdf"
            if not pdf_path.exists():
                self.send_error(404, "PDF Manual not found on server.")
                return
            pdf_bytes = pdf_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Disposition", 'attachment; filename="RUNNING_GUIDE_MANUAL.pdf"')
            self.send_header("Content-Length", str(len(pdf_bytes)))
            self.end_headers()
            self.wfile.write(pdf_bytes)
        elif path == "/" or path == "/index.html":
            self.path = "/index.html"
            return super().do_GET()
        else:
            return super().do_GET()

    def do_POST(self):
        url_parsed = urllib.parse.urlparse(self.path)
        path = url_parsed.path

        if path == "/api/process":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            try:
                payload = json.loads(post_data.decode("utf-8"))
                result = self.process_pipeline_request(payload)
                self.send_json_response(result)
            except Exception as e:
                self.send_json_response({"error": str(e)}, status=500)

        elif path == "/api/load-sample-db":
            try:
                sample_file = project_root / "Dataset" / "RF25_ind2024_rfp25.nc"
                if not sample_file.exists():
                    self.send_json_response({"error": "Bundled IMD 2024 NetCDF dataset not found on disk"}, status=404)
                    return
                result = self._process_any_database(sample_file)
                self.send_json_response(result)
            except Exception as e:
                self.send_json_response({"error": f"Failed to compute sample database: {str(e)}"}, status=500)

        elif path == "/api/upload-database":
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length <= 0:
                self.send_json_response({"error": "No file data received in upload request."}, status=400)
                return

            filename = self.headers.get("X-Filename", "uploaded_dataset.nc")
            content_type = self.headers.get("Content-Type", "")

            upload_dir = project_root / "rain_filter" / "outputs" / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            saved_path = upload_dir / filename

            post_bytes = self.rfile.read(content_length)

            # Check for multipart/form-data boundary
            if "multipart/form-data" in content_type and "boundary=" in content_type:
                boundary = content_type.split("boundary=")[-1].strip().encode()
                parts = post_bytes.split(b"--" + boundary)
                saved = False
                for part in parts:
                    if b'filename="' in part:
                        header_block, file_body = part.split(b"\r\n\r\n", 1)
                        file_data = file_body.rstrip(b"\r\n--")
                        fn_lines = [line for line in header_block.split(b"\r\n") if b'filename="' in line]
                        if fn_lines:
                            fn = fn_lines[0].split(b'filename="')[1].split(b'"')[0].decode("utf-8", "ignore")
                            if fn:
                                filename = fn
                                saved_path = upload_dir / filename
                        saved_path.write_bytes(file_data)
                        saved = True
                        break
                if not saved:
                    saved_path.write_bytes(post_bytes)
            else:
                saved_path.write_bytes(post_bytes)

            try:
                result = self._process_any_database(saved_path)
                self.send_json_response(result)
            except Exception as e:
                self.send_json_response({"error": f"Failed to calculate database: {str(e)}"}, status=500)

        else:
            self.send_error(404, "Endpoint not found")

    def _process_any_database(self, filepath: Path) -> Dict[str, Any]:
        """Route to appropriate processor based on file extension and export report."""
        global LATEST_DATABASE_ANALYSIS
        fn_lower = filepath.name.lower()
        if fn_lower.endswith((".nc", ".nc4", ".grd")):
            analysis = process_netcdf_database(filepath, filter_mode="hybrid", preserve_volume=True)
        elif fn_lower.endswith((".db", ".sqlite", ".sqlite3")):
            analysis = process_sqlite_database(filepath, filter_mode="hybrid", preserve_volume=True)
        else:
            analysis = process_csv_database(filepath, filter_mode="hybrid", preserve_volume=True)

        report_file = project_root / "rain_filter" / "outputs" / "database_analysis_report.csv"
        export_database_results_csv(analysis, report_file)
        LATEST_DATABASE_ANALYSIS = analysis
        return analysis

    def send_json_response(self, data: Any, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def process_pipeline_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run the complete satellite pipeline based on user request."""
        # 1. Extract or parse input series
        lat = float(payload.get("lat", 19.0))
        lon = float(payload.get("lon", 72.75))
        custom_series = payload.get("custom_series", None)

        if custom_series is not None and len(custom_series) > 0:
            clean_series = np.array(custom_series, dtype=np.float64)
            station_name = "Custom Uploaded Series"
        elif RAIN_CUBE is not None and LATS is not None and LONS is not None:
            clean_series, lat_idx, lon_idx = extract_cell(RAIN_CUBE, LATS, LONS, lat, lon)
            # Find closest preset name if matching
            station_name = f"Grid Cell ({lat:.2f}°N, {lon:.2f}°E)"
            for name, meta in PRESET_CITIES.items():
                if abs(meta["lat"] - lat) < 0.2 and abs(meta["lon"] - lon) < 0.2:
                    station_name = name
                    break
        else:
            clean_series = generate_synthetic_rainfall(seed=42)
            station_name = "Synthetic Rainfall Series"

        # Handle NaNs in clean series
        if np.all(np.isnan(clean_series)):
            clean_series = generate_synthetic_rainfall(seed=42)

        # 2. Inject Noise (if selected)
        noise_type = payload.get("noise_type", "mixed")
        noise_param = float(payload.get("noise_param", 1.0))
        seed = int(payload.get("seed", 42))

        if noise_type == "gaussian":
            noisy_series = add_gaussian_noise(clean_series, sigma=noise_param, seed=seed)
        elif noise_type == "impulsive":
            noisy_series = add_impulsive_noise(clean_series, fraction=noise_param / 100.0, seed=seed)
        elif noise_type == "mixed":
            noisy_series = add_mixed_noise(clean_series, sigma=noise_param, impulse_fraction=0.02, seed=seed)
        elif noise_type == "dropouts":
            noisy_series = add_dropouts(clean_series, fraction=noise_param / 100.0, seed=seed)
        elif noise_type == "none":
            noisy_series = clean_series.copy()
        else:
            noisy_series = clean_series.copy()

        # 3. Stage 1: Denoise Filter
        mode = payload.get("filter_mode", "hybrid")
        preserve_vol = bool(payload.get("preserve_volume", True))
        filtered_series = denoise(noisy_series, mode=mode, preserve_volume=preserve_vol)

        # Baselines for comparison
        b_ma = baseline_moving_average(noisy_series, window=5)
        b_med = baseline_median(noisy_series, window=5)
        b_bw = baseline_butterworth(noisy_series, order=3, wn=0.35)
        b_sg = baseline_savgol(noisy_series, window_length=11, polyorder=3)

        # Metrics
        m_hybrid = compute_metrics(clean_series, noisy_series, filtered_series)
        m_ma = compute_metrics(clean_series, noisy_series, b_ma)
        m_med = compute_metrics(clean_series, noisy_series, b_med)
        m_bw = compute_metrics(clean_series, noisy_series, b_bw)
        m_sg = compute_metrics(clean_series, noisy_series, b_sg)

        # 4. Stage 2: Discrete Wavelet Transform (DWT)
        wavelet = payload.get("wavelet", "haar")
        level = int(payload.get("dwt_level", 3))
        dwt_res = dwt_1d(filtered_series, wavelet=wavelet, level=level)

        # 5. Stage 3: Feature Extraction
        feats = extract_features_1d(filtered_series, dwt_res=dwt_res)

        # 6. Stage 4: Feature Ranking
        # Generate contrast matrix for ranking contextualization
        c_names = list(feats.keys())
        contrast_samples = [[feats[k] for k in c_names]]
        if PRECOMPUTED_CONTRAST_SAMPLES and PRECOMPUTED_FEATURE_NAMES:
            for row in PRECOMPUTED_CONTRAST_SAMPLES:
                contrast_samples.append([
                    row[PRECOMPUTED_FEATURE_NAMES.index(k)] if k in PRECOMPUTED_FEATURE_NAMES else 0.0
                    for k in c_names
                ])
        else:
            # Fallback if precomputation failed
            for i in range(10):
                contrast_samples.append([feats[k] * (0.8 + 0.04 * i) for k in c_names])

        X_rank = np.array(contrast_samples)
        y_rank = np.array([1] + [0] * (len(contrast_samples) - 1))
        ranked_features = rank_features(X_rank, y_rank, c_names, method="composite")

        # 7. Edge Q15 validation
        q15_bench = benchmark_q15_accuracy(clean_series, noise_sigma=1.0)

        # Downsample or format series for JSON payload (sanitize NaN to None)
        def _clean_list(arr):
            out = []
            for v in arr:
                if np.isnan(v):
                    out.append(None)
                else:
                    out.append(round(float(v), 2))
            return out

        # Subband arrays for scalogram
        dwt_subbands = {
            f"A_{level}": _clean_list(dwt_res.approx),
        }
        for j, d in enumerate(dwt_res.details, 1):
            dwt_subbands[f"D_{j}"] = _clean_list(d)

        response = {
            "station_name": station_name,
            "lat": lat,
            "lon": lon,
            "length": len(clean_series),
            "series": {
                "clean": _clean_list(clean_series),
                "noisy": _clean_list(noisy_series),
                "filtered": _clean_list(filtered_series),
                "baseline_butterworth": _clean_list(b_bw),
                "baseline_median": _clean_list(b_med),
                "baseline_ma": _clean_list(b_ma)
            },
            "metrics": {
                "hybrid": m_hybrid,
                "butterworth": m_bw,
                "median": m_med,
                "moving_average": m_ma,
                "savitzky_golay": m_sg
            },
            "dwt": {
                "wavelet": wavelet,
                "level": level,
                "energies": {k: round(float(v), 2) for k, v in dwt_res.energies.items() if isinstance(v, (int, float))},
                "subbands": dwt_subbands
            },
            "features": feats,
            "ranked_features": ranked_features[:8],
            "edge_hardware": {
                "q15": q15_bench,
                "ops": get_theoretical_ops_per_sample()["Hybrid Pipeline (A -> B)"],
                "memory_bytes": 8
            }
        }

        return response


def start_server(port: int = 8080):
    """Start the Slush Satellite Web Application Server."""
    server_address = ("", port)
    httpd = ThreadingHTTPServer(server_address, SlushAPIHandler)
    print(f"\n=======================================================")
    print(f" SLUSH SATELLITE EDGE DASHBOARD READY!")
    print(f" URL: http://localhost:{port}")
    print(f" Web root: {current_dir}")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    start_server(port)
