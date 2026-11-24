"""Render a still of every GLB in an end-to-end run folder, from one fixed angle, for before/after sheets.

    python3 e2e/capture_server.py results/e2e/<run> 8766
    # open http://127.0.0.1:8766/capture.html in a visible browser window and wait for DONE
    uv run e2e/before_after.py results/e2e/<run> --out results/e2e/<run>-before-after.jpg

GET serves the run folder, plus capture.html from this directory and glbs.txt (the GLB names in 3d/).
POST /save/<name>.png writes renders/<name>.png. The browser must be visible: a hidden page does not render.
"""
import http.server
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
(ROOT / "renders").mkdir(exist_ok=True)


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        if self.path.split("?")[0] == "/capture.html":
            return self.reply((Path(__file__).parent / "capture.html").read_bytes(), "text/html")
        if self.path == "/glbs.txt":
            return self.reply("\n".join(sorted(p.stem for p in (ROOT / "3d").glob("*.glb"))).encode(), "text/plain")
        return super().do_GET()

    def reply(self, body, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        name = Path(self.path).name
        if not self.path.startswith("/save/") or not name.endswith(".png") or "/" in name:
            self.send_error(400)
            return
        data = self.rfile.read(int(self.headers["Content-Length"]))
        (ROOT / "renders" / name).write_bytes(data)
        self.send_response(204)
        self.end_headers()

    def log_message(self, *args):
        pass


http.server.ThreadingHTTPServer(("127.0.0.1", int(sys.argv[2])), Handler).serve_forever()
