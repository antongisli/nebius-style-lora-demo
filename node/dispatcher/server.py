#!/usr/bin/env python3
"""Grid page for the eight ComfyUI instances.

Serves a page on 127.0.0.1:8080 that takes a list of prompts, spreads the jobs
over all instances (see dispatcher.py), shows the images as they finish, and
sends a chosen image to the image-to-3D workflow. Images and GLB files are
proxied from the instances' /view endpoint, so only this port needs to be
forwarded to the laptop.

Settings come from the environment (written by node/setup.sh to
~/.config/comfyui-gpu/env) and can be overridden on the command line.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dispatcher import Dispatcher, InstanceError, load_workflow  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_BODY = 1 << 20
MAX_PROMPTS = 64
MAX_VARIANTS = 32


def parse_args(argv=None) -> argparse.Namespace:
    env = os.environ.get
    default_workflows = env("WORKFLOWS_DIR") or os.path.join(os.path.dirname(os.path.dirname(HERE)), "workflows")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(env("GRID_PORT", "8080")))
    parser.add_argument("--num-gpus", type=int, default=int(env("NUM_GPUS", "8")))
    parser.add_argument("--base-port", type=int, default=int(env("BASE_PORT", "8188")))
    parser.add_argument("--instances", help="comma-separated instance URLs; overrides --num-gpus/--base-port")
    parser.add_argument("--workflows-dir", default=default_workflows)
    parser.add_argument("--max-inflight", type=int, default=2, help="jobs queued per instance at once")
    parser.add_argument("--trigger", default=env("TRIGGER_WORD", "nbsstyle"))
    parser.add_argument("--lora-name", default=env("STYLE_LORA_NAME", "style.safetensors"))
    return parser.parse_args(argv)


class App:
    def __init__(self, args: argparse.Namespace, dispatcher: Optional[Dispatcher] = None):
        self.args = args
        if args.instances:
            urls = [u.strip() for u in args.instances.split(",") if u.strip()]
        else:
            urls = [f"http://127.0.0.1:{args.base_port + i}" for i in range(args.num_gpus)]
        self.dispatcher = dispatcher or Dispatcher(urls, max_inflight=args.max_inflight, lora_name=args.lora_name)
        self.grid_template = self._load("grid_api.json")
        self.three_d_template = self._load("image_to_3d.json")
        with open(os.path.join(HERE, "index.html"), "rb") as handle:
            self.page = handle.read()

    def _load(self, name: str):
        path = os.path.join(self.args.workflows_dir, name)
        graph = load_workflow(path)
        # Use the style LoRA file name this node was set up with.
        for node in graph.values():
            if node.get("_meta", {}).get("title") == "Style LoRA":
                node["inputs"]["lora_name"] = self.args.lora_name
        return graph

    def run(self, body: Dict[str, Any]) -> Dict[str, Any]:
        prompts = body.get("prompts")
        if isinstance(prompts, str):
            prompts = prompts.splitlines()
        if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
            raise ValueError("prompts must be a list of strings")
        prompts = [p for p in prompts if p.strip()][:MAX_PROMPTS]
        if not prompts:
            raise ValueError("enter at least one prompt")
        variants = int(body.get("variants", 8))
        if not 1 <= variants <= MAX_VARIANTS:
            raise ValueError(f"variants must be between 1 and {MAX_VARIANTS}")
        strength = self._lora_strength(body)
        seed = body.get("seed")
        seed = int(seed) if seed not in (None, "") else None
        jobs = self.dispatcher.add_image_jobs(
            self.grid_template, prompts, variants, strength,
            trigger=self.args.trigger, suffix=str(body.get("suffix", "")).strip(), seed=seed)
        return {"jobs": [job.id for job in jobs]}

    def _lora_strength(self, body: Dict[str, Any]) -> Optional[float]:
        if not body.get("lora", True):
            return None
        return float(body.get("strength", 1.3))  # the selected LoRA (dataset v3) needs 1.3 with Lightning

    def warmup(self, body: Dict[str, Any]) -> Dict[str, Any]:
        jobs = self.dispatcher.add_warmup_jobs(self.grid_template, self._lora_strength(body))
        return {"jobs": [job.id for job in jobs]}

    def make_3d(self, body: Dict[str, Any]) -> Dict[str, Any]:
        job = self.dispatcher.add_3d_job(self.three_d_template, str(body.get("job", "")), int(body.get("index", 0)))
        return {"jobs": [job.id]}


class Handler(BaseHTTPRequestHandler):
    app: App = None  # set in main()
    server_version = "comfyui-grid"

    def log_message(self, fmt, *args):  # quieter than the default
        if not self.path.startswith(("/api/state", "/view")):
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self._send(200, self.app.page, "text/html; charset=utf-8")
        elif url.path == "/api/state":
            self._json(200, self.app.dispatcher.state())
        elif url.path == "/view":
            self._view({k: v[0] for k, v in parse_qs(url.query).items()})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        url = urlparse(self.path)
        # Requiring JSON makes a cross-site form post fail the CORS preflight.
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            self._json(415, {"error": "send application/json"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self._json(413, {"error": "request too large"})
            return
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("expected a JSON object")
            if url.path == "/api/run":
                result = self.app.run(body)
            elif url.path == "/api/warmup":
                result = self.app.warmup(body)
            elif url.path == "/api/3d":
                result = self.app.make_3d(body)
            elif url.path == "/api/clear":
                self.app.dispatcher.clear_finished()
                result = {}
            else:
                self._json(404, {"error": "not found"})
                return
        except (ValueError, TypeError) as error:
            self._json(400, {"error": str(error)})
            return
        self._json(200, result)

    def _view(self, params: Dict[str, str]):
        preferred = None
        if "job" in params:
            index = params.get("index", "0")
            if not index.isdigit():
                self._json(400, {"error": "index must be a number"})
                return
            output, preferred = self.app.dispatcher.output_for(params["job"], int(index))
            if output is None:
                self._json(404, {"error": "no such output"})
                return
            params = output
        if not params.get("filename"):
            self._json(400, {"error": "filename is required"})
            return
        for url in self.app.dispatcher.view_url(params, preferred):
            try:
                resp = self.app.dispatcher.http.get_raw(url, timeout=30)
            except InstanceError:
                continue
            with resp:
                self.send_response(resp.status)
                for header in ("Content-Type", "Content-Length", "Content-Disposition"):
                    if resp.headers.get(header):
                        self.send_header(header, resp.headers[header])
                self.send_header("Cache-Control", "max-age=3600" if resp.status == 200 else "no-store")
                self.end_headers()
                shutil.copyfileobj(resp, self.wfile)
            return
        self._json(502, {"error": "no ComfyUI instance is reachable"})

    def _json(self, status: int, data: Any):
        self._send(status, json.dumps(data).encode(), "application/json")

    def _send(self, status: int, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def make_server(args: argparse.Namespace, dispatcher: Optional[Dispatcher] = None) -> ThreadingHTTPServer:
    Handler.app = App(args, dispatcher)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    return server


def main(argv=None) -> int:
    args = parse_args(argv)
    server = make_server(args)
    dispatcher = Handler.app.dispatcher
    dispatcher.start()
    print(f"grid page on http://{args.host}:{server.server_address[1]} "
          f"for {len(dispatcher.instances)} instances", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        dispatcher.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
