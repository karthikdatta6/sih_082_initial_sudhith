"""
live_dashboard_server.py
=============================================================================
SIH26082 — Full-Stack Real-Time Comparison & Forecasting Server

Serves:
1. Frontend UI: Glassmorphism interactive dashboard (index.html)
2. Backend REST API:
   - GET /api/live-comparison : Fetches live ground truth + CAMS + computes SIH model
   - GET /api/health          : Service health status

Usage:
    python MODEL/code/live_dashboard_server.py
    Open browser at: http://localhost:8082
=============================================================================
"""
from __future__ import annotations

import http.server
import json
import os
import socketserver
import sys
import webbrowser
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.abspath(os.path.join(HERE, "..", "dashboard"))
sys.path.insert(0, HERE)

import live_aqi_comparator as comparator

PORT = 8082

class SIHLiveDashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        
        if parsed.path == "/api/live-comparison":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            
            try:
                # Run fresh live fetch and inference
                waqi, cams, met = comparator.fetch_live_data()
                forecasts, diagnostics = comparator.run_sih26082_inference(waqi, cams, met)
                
                payload = {
                    "status": "success",
                    "timestamp": comparator.datetime.now().isoformat(),
                    "ground_truth": waqi,
                    "cams_global_model": cams,
                    "meteorology": met,
                    "sih26082_model": {
                        "forecasts": forecasts,
                        "physics_diagnostics": diagnostics
                    }
                }
            except Exception as e:
                payload = {"status": "error", "message": str(e)}

            self.wfile.write(json.dumps(payload).encode("utf-8"))
            return

        elif parsed.path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy", "service": "SIH26082 Live Engine"}).encode("utf-8"))
            return

        # Default static file serving
        return super().do_GET()

def start_server():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), SIHLiveDashboardHandler) as httpd:
        print("=" * 72)
        print("  🚀 SIH26082 — LIVE AIR QUALITY & WEATHER DASHBOARD SERVER")
        print(f"  Serving on: http://localhost:{PORT}")
        print("  REST API  : http://localhost:8082/api/live-comparison")
        print("  Press Ctrl+C to stop server")
        print("=" * 72)
        
        # Populate initial cache
        try:
            comparator.run_live_comparison()
        except Exception as e:
            print(f"Initial comparison notice: {e}")

        httpd.serve_forever()

if __name__ == "__main__":
    start_server()
