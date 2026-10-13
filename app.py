from __future__ import annotations

import csv
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from analysis import (
    DataValidationError,
    WEAK_THRESHOLD_DEFAULT,
    _records_to_frame,
    analyse_rows,
    build_spatial_models,
    predict_rssi,
    validate_rows,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
HOST = "127.0.0.1"
PORT = 5000


def load_sample_rows() -> list[dict]:
    with (DATA_DIR / "wifi_data_synthetic.csv").open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class WiFiScopeHandler(BaseHTTPRequestHandler):
    server_version = "WiFiScopePython/1.0"

    def _send_json(self, payload: dict, status: int = 200):
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _send_file(self, path: Path, content_type: str):
        raw = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _read_json(self):
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(n)
            return json.loads(body.decode("utf-8") or "{}")
        except (ValueError, json.JSONDecodeError):
            raise DataValidationError("Invalid JSON request.")

    def log_message(self, fmt, *args):
        # Keep the demo terminal clean.
        return

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            return self._send_file(BASE_DIR / "index.html", "text/html; charset=utf-8")
        if path == "/api/health":
            return self._send_json({
                "status": "ok",
                "backend": "python",
                "application": "Wi-FiScope",
                "analysis_engine": "analysis.py",
            })
        if path == "/api/sample":
            return self._send_json({"rows": load_sample_rows()})
        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            rows = payload.get("rows") or load_sample_rows()
            threshold = float(payload.get("threshold", WEAK_THRESHOLD_DEFAULT))

            if path == "/api/analyze":
                return self._send_json(analyse_rows(rows, threshold))

            if path == "/api/predict":
                validate_rows(rows)
                x = float(payload["x"])
                y = float(payload["y"])
                period = str(payload.get("period", "Average"))
                df = _records_to_frame(rows)
                locations = list(dict.fromkeys(df["Location_ID"].astype(str).tolist()))
                periods = list(dict.fromkeys(df["Time_Period"].astype(str).tolist()))
                models = build_spatial_models(df, locations, periods)
                if period not in models:
                    raise DataValidationError(f"Unknown time period: {period}")
                return self._send_json({
                    "predicted_rssi": predict_rssi(models[period], x, y),
                    "period": period,
                    "x": x,
                    "y": y,
                })

            return self._send_json({"error": "Not found"}, 404)
        except (DataValidationError, ValueError, KeyError) as exc:
            return self._send_json({"error": str(exc)}, 400)
        except Exception:
            self._send_json({"error": "Unexpected server-side analysis error."}, 500)


def main():
    print("Wi-FiScope Python backend")
    print(f"Open http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    server = ThreadingHTTPServer((HOST, PORT), WiFiScopeHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Wi-FiScope...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
