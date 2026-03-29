#!/usr/bin/env python3
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent / "dist"
PORT = int(os.environ.get("PORT", "8082"))


class SPARequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        # If path targets a real file, serve it; otherwise serve index.html for SPA routes.
        requested = self.path.split("?", 1)[0].split("#", 1)[0].lstrip("/")
        file_path = ROOT / requested
        if requested and file_path.exists() and file_path.is_file():
            return super().do_GET()

        self.path = "/index.html"
        return super().do_GET()

    def do_HEAD(self):
        requested = self.path.split("?", 1)[0].split("#", 1)[0].lstrip("/")
        file_path = ROOT / requested
        if requested and file_path.exists() and file_path.is_file():
            return super().do_HEAD()

        self.path = "/index.html"
        return super().do_HEAD()


if __name__ == "__main__":
    if not ROOT.exists():
        raise SystemExit("dist folder not found. Run: npm run build")

    server = ThreadingHTTPServer(("127.0.0.1", PORT), SPARequestHandler)
    print(f"Serving SPA from {ROOT} on http://127.0.0.1:{PORT}")
    server.serve_forever()
