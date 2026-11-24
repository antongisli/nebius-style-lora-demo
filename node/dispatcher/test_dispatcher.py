"""Tests for the dispatcher against mock ComfyUI instances.

Run from the repository root:  python3 -m unittest discover -s node/dispatcher -v

The mock implements the parts of the ComfyUI HTTP API the dispatcher uses:
/system_stats, /prompt, /queue, /history/<id>, /view and /models/loras. It runs
one prompt at a time, like ComfyUI, and can be stopped or restarted to
simulate a crashed or restarted instance.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import unittest
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
WORKFLOWS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "workflows")
sys.path.insert(0, HERE)

import dispatcher as dp  # noqa: E402
import server as grid_server  # noqa: E402


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class MockComfy:
    def __init__(self, index: int, delay: float = 0.05, port: int = 0):
        self.index = index
        self.delay = delay
        self.port = port or free_port()
        self.fail_execution = False
        self.received = []  # prompt graphs in arrival order
        self._reset()
        self.server = None

    def _reset(self):
        self.lock = threading.Lock()
        self.pending = []  # [(number, prompt_id, graph)]
        self.running = None
        self.history = {}
        self.number = 0
        self.stop_event = threading.Event()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self):
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _json(self, status, data):
                body = json.dumps(data).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                url = urlparse(self.path)
                if url.path == "/system_stats":
                    self._json(200, {"devices": [{"name": f"Mock GPU {mock.index}"}]})
                elif url.path == "/queue":
                    with mock.lock:
                        running = [[mock.running[0], mock.running[1], {}, {}, []]] if mock.running else []
                        pending = [[n, pid, {}, {}, []] for n, pid, _ in mock.pending]
                    self._json(200, {"queue_running": running, "queue_pending": pending})
                elif url.path.startswith("/history/"):
                    pid = url.path.split("/", 2)[2]
                    with mock.lock:
                        entry = mock.history.get(pid)
                    self._json(200, {pid: entry} if entry else {})
                elif url.path == "/models/loras":
                    self._json(200, ["Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors", "style.safetensors"])
                elif url.path == "/view":
                    q = {k: v[0] for k, v in parse_qs(url.query).items()}
                    body = f"GPU{mock.index}:{q.get('subfolder', '')}/{q.get('filename', '')}".encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self._json(404, {})

            def do_POST(self):
                if self.path != "/prompt":
                    self._json(404, {})
                    return
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                graph = body["prompt"]
                bad = [nid for nid, node in graph.items() if node["class_type"] == "Invalid"]
                if bad:
                    self._json(400, {
                        "error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation"},
                        "node_errors": {bad[0]: {"errors": [{"message": "Value not in list", "details": "x"}]}},
                    })
                    return
                pid = str(uuid.uuid4())
                with mock.lock:
                    mock.number += 1
                    mock.pending.append((mock.number, pid, graph))
                    mock.received.append(graph)
                self._json(200, {"prompt_id": pid, "number": mock.number, "node_errors": {}})

        self._reset()
        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        threading.Thread(target=self._worker, args=(self.stop_event,), daemon=True).start()
        return self

    def _worker(self, stop_event):
        while not stop_event.is_set():
            with self.lock:
                if self.running is None and self.pending:
                    self.running = self.pending.pop(0)
                item = self.running
            if item is None:
                time.sleep(0.01)
                continue
            if stop_event.wait(self.delay):
                return
            number, pid, graph = item
            prefix = next((n["inputs"].get("filename_prefix") for n in graph.values()
                           if n["class_type"] in ("SaveImage", "SaveGLB")), "ComfyUI")
            subfolder, _, name = prefix.rpartition("/")
            with self.lock:
                if self.fail_execution:
                    status = {"status_str": "error", "completed": False, "messages": [
                        ["execution_error", {"node_type": "KSampler", "exception_message": "CUDA out of memory"}]]}
                    outputs = {}
                else:
                    status = {"status_str": "success", "completed": True, "messages": []}
                    outputs = {"12": {"images": [{"filename": f"{name}_00001_.png", "subfolder": subfolder, "type": "output"}]}}
                self.history[pid] = {"prompt": [number, pid, graph], "outputs": outputs, "status": status}
                self.running = None

    def stop(self):
        """Simulates a crash: the port stops answering and the queue is lost."""
        self.stop_event.set()
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None

    def restart(self):
        """Simulates a systemd restart: same port, empty queue and history."""
        self.stop()
        self.start()


def make_dispatcher(mocks, **kwargs):
    kwargs.setdefault("max_inflight", 2)
    return dp.Dispatcher([m.url for m in mocks], **kwargs)


def run_until_done(dispatcher, timeout=10.0, on_tick=None):
    deadline = time.time() + timeout
    while time.time() < deadline:
        dispatcher.check_health()
        dispatcher.tick()
        if on_tick:
            on_tick()
        states = [job.status for job in dispatcher.jobs.values()]
        if all(s in ("done", "error") for s in states):
            return
        time.sleep(0.02)
    raise AssertionError(f"jobs did not finish: {[j.snapshot() for j in dispatcher.jobs.values()]}")


class TemplateTests(unittest.TestCase):
    def setUp(self):
        self.grid = dp.load_workflow(os.path.join(WORKFLOWS, "grid_api.json"))

    def test_fill_grid_template(self):
        graph = dp.build_image_graph(self.grid, "nbsstyle, a sword", 1234, "grid/x", 0.8)
        self.assertEqual(graph["7"]["inputs"]["text"], "nbsstyle, a sword")
        self.assertEqual(graph["10"]["inputs"]["seed"], 1234)
        self.assertEqual(graph["12"]["inputs"]["filename_prefix"], "grid/x")
        self.assertEqual(graph["5"]["inputs"]["strength_model"], 0.8)
        # The template itself is not modified.
        self.assertEqual(self.grid["10"]["inputs"]["seed"], 42)

    def test_without_lora_removes_the_style_node(self):
        graph = dp.build_image_graph(self.grid, "a sword", 1, "grid/x", None)
        self.assertNotIn("5", graph)
        self.assertEqual(graph["6"]["inputs"]["model"], ["4", 0])

    def test_compare_template_uses_one_seed(self):
        compare = dp.load_workflow(os.path.join(WORKFLOWS, "compare_lora.json"))
        graph = dp.build_image_graph(compare, "a chest", 7, "compare/x", 1.0)
        seeds = {graph[nid]["inputs"]["seed"] for nid in dp.nodes_by_class(graph, "KSampler")}
        self.assertEqual(seeds, {7})

    def test_3d_template_reads_from_output_folder(self):
        three_d = dp.load_workflow(os.path.join(WORKFLOWS, "image_to_3d.json"))
        graph = dp.build_3d_graph(three_d, {"filename": "a_00001_.png", "subfolder": "grid", "type": "output"}, "3d/p")
        self.assertEqual(graph["1"]["inputs"]["image"], "grid/a_00001_.png [output]")
        self.assertEqual(graph["35"]["inputs"]["filename_prefix"], "3d/p")

    def test_workflow_links_point_to_existing_nodes(self):
        for name in ("grid_api.json", "compare_lora.json", "image_to_3d.json"):
            graph = dp.load_workflow(os.path.join(WORKFLOWS, name))
            for nid, node in graph.items():
                for key, value in node["inputs"].items():
                    if isinstance(value, list):
                        self.assertIn(value[0], graph, f"{name} node {nid} input {key}")


class DispatcherTests(unittest.TestCase):
    def setUp(self):
        self.template = dp.load_workflow(os.path.join(WORKFLOWS, "grid_api.json"))
        self.mocks = []

    def tearDown(self):
        for mock in self.mocks:
            mock.stop()

    def start_mocks(self, count, delay=0.05):
        self.mocks = [MockComfy(i, delay=delay).start() for i in range(count)]
        return self.mocks

    def test_round_robin_spreads_jobs_evenly(self):
        mocks = self.start_mocks(8, delay=0.2)
        d = make_dispatcher(mocks, max_inflight=2)
        d.check_health()
        d.add_image_jobs(self.template, ["a sword", "a chest"], 8, 1.0, trigger="nbsstyle")
        d.schedule()
        # 16 jobs, 8 instances, 2 slots each: every instance gets two, and the
        # first eight go to GPUs 0..7 in order.
        self.assertEqual([len(m.received) for m in mocks], [2] * 8)
        first_eight = [d.jobs[jid].instance for jid in d.order[:8]]
        self.assertEqual(first_eight, list(range(8)))
        run_until_done(d)
        self.assertTrue(all(j.status == "done" for j in d.jobs.values()))
        self.assertEqual(mocks[0].received[0]["7"]["inputs"]["text"].split(",")[0], "nbsstyle")

    def test_with_suffix_continues_the_subject(self):
        d = make_dispatcher(self.start_mocks(1))
        d.add_image_jobs(self.template, ["a chest"], 1, 1.3, trigger="nbsstyle", suffix="with lime green trim, plain background")
        d.add_image_jobs(self.template, ["a chest"], 1, None, suffix="plain background")
        texts = [d.jobs[jid].graph["7"]["inputs"]["text"] for jid in d.order]
        self.assertEqual(texts, ["nbsstyle, a chest with lime green trim, plain background", "a chest, plain background"])

    def test_more_jobs_than_slots_wait_and_then_run(self):
        mocks = self.start_mocks(2)
        d = make_dispatcher(mocks, max_inflight=1)
        d.check_health()
        d.add_image_jobs(self.template, ["a sword"], 6, None)
        d.schedule()
        self.assertEqual(sum(j.status == "pending" for j in d.jobs.values()), 4)
        run_until_done(d)
        self.assertEqual(sorted(len(m.received) for m in mocks), [3, 3])

    def test_instance_down_from_the_start_is_skipped(self):
        mocks = self.start_mocks(8)
        mocks[3].stop()
        d = make_dispatcher(mocks)
        d.add_image_jobs(self.template, ["a sword"], 14, 1.0)
        run_until_done(d)
        self.assertIs(d.instances[3].up, False)
        self.assertTrue(all(j.status == "done" for j in d.jobs.values()))
        self.assertNotIn(3, {j.instance for j in d.jobs.values()})
        self.assertEqual(len(mocks[3].received), 0)

    def test_instance_crash_mid_run_requeues_its_jobs(self):
        mocks = self.start_mocks(4, delay=0.3)
        d = make_dispatcher(mocks, max_inflight=2, down_after=1)
        d.check_health()
        d.add_image_jobs(self.template, ["a chest"], 8, 1.0)
        d.schedule()
        on_two = [j.id for j in d.jobs.values() if j.instance == 2]
        self.assertEqual(len(on_two), 2)
        mocks[2].stop()
        run_until_done(d)
        for jid in on_two:
            job = d.jobs[jid]
            self.assertEqual(job.status, "done")
            self.assertNotEqual(job.instance, 2)
            self.assertEqual(job.attempts, 2)
        self.assertEqual(d.instances[2].inflight, [])

    def test_instance_restart_loses_queue_and_jobs_are_resubmitted(self):
        mocks = self.start_mocks(2, delay=0.3)
        d = make_dispatcher(mocks, max_inflight=2)
        d.check_health()
        d.add_image_jobs(self.template, ["a potion"], 4, 1.0)
        d.schedule()
        mocks[1].restart()  # answers again at once, but forgot both jobs
        run_until_done(d)
        self.assertTrue(all(j.status == "done" for j in d.jobs.values()))
        self.assertTrue(any(j.attempts == 2 for j in d.jobs.values()))

    def test_recovered_instance_gets_new_jobs(self):
        mocks = self.start_mocks(2)
        mocks[1].stop()
        d = make_dispatcher(mocks)
        d.check_health()
        self.assertIs(d.instances[1].up, False)
        mocks[1].start()
        d.check_health()
        self.assertIs(d.instances[1].up, True)
        d.add_image_jobs(self.template, ["a key"], 4, 1.0)
        run_until_done(d)
        self.assertEqual(len(mocks[1].received), 2)

    def test_all_instances_down_jobs_wait(self):
        mocks = self.start_mocks(2)
        for mock in mocks:
            mock.stop()
        d = make_dispatcher(mocks)
        d.check_health()
        d.add_image_jobs(self.template, ["a shield"], 2, 1.0)
        d.tick()
        self.assertEqual({j.status for j in d.jobs.values()}, {"pending"})
        for mock in mocks:
            mock.start()
        run_until_done(d)
        self.assertEqual({j.status for j in d.jobs.values()}, {"done"})

    def test_retries_stop_after_max_attempts(self):
        mocks = self.start_mocks(1, delay=0.2)
        d = make_dispatcher(mocks, max_attempts=2)
        d.check_health()
        d.add_image_jobs(self.template, ["a bow"], 1, 1.0)
        job = next(iter(d.jobs.values()))

        def restart_while_queued():
            if job.status in ("queued", "running"):
                mocks[0].restart()

        run_until_done(d, on_tick=restart_while_queued)
        self.assertEqual(job.status, "error")
        self.assertEqual(job.attempts, 2)
        self.assertIn("lost", job.error)

    def test_validation_error_is_not_retried(self):
        mocks = self.start_mocks(2)
        d = make_dispatcher(mocks)
        d.check_health()
        bad = {"1": {"class_type": "Invalid", "inputs": {}}}
        job = d.add_job(dp.Job(d.new_id(), "image", "bad", bad))
        run_until_done(d)
        self.assertEqual(job.status, "error")
        self.assertEqual(job.attempts, 1)
        self.assertIn("Value not in list", job.error)
        self.assertEqual([i.up for i in d.instances], [True, True])

    def test_execution_error_is_reported(self):
        mocks = self.start_mocks(1)
        mocks[0].fail_execution = True
        d = make_dispatcher(mocks)
        d.add_image_jobs(self.template, ["a lamp"], 1, 1.0)
        run_until_done(d)
        job = next(iter(d.jobs.values()))
        self.assertEqual(job.status, "error")
        self.assertIn("CUDA out of memory", job.error)

    def test_warmup_pins_one_job_per_instance(self):
        mocks = self.start_mocks(3)
        mocks[2].stop()
        d = make_dispatcher(mocks)
        d.check_health()
        d.add_warmup_jobs(self.template, 1.0)
        run_until_done(d)
        by_pin = {j.pin: j for j in d.jobs.values()}
        self.assertEqual(by_pin[0].status, "done")
        self.assertEqual(by_pin[0].instance, 0)
        self.assertEqual(by_pin[1].instance, 1)
        self.assertEqual(by_pin[2].status, "error")
        self.assertIn("down", by_pin[2].error)

    def test_3d_job_from_finished_image(self):
        mocks = self.start_mocks(2)
        d = make_dispatcher(mocks)
        d.add_image_jobs(self.template, ["a chest"], 1, 1.0)
        run_until_done(d)
        image_job = next(iter(d.jobs.values()))
        three_d = dp.load_workflow(os.path.join(WORKFLOWS, "image_to_3d.json"))
        job = d.add_3d_job(three_d, image_job.id)
        run_until_done(d)
        self.assertEqual(job.status, "done")
        sent = [g for m in mocks for g in m.received if "35" in g][0]
        self.assertTrue(sent["1"]["inputs"]["image"].endswith(" [output]"))
        with self.assertRaises(ValueError):
            d.add_3d_job(three_d, "missing")

    def test_lora_detection(self):
        mocks = self.start_mocks(1)
        d = make_dispatcher(mocks, lora_name="style.safetensors")
        d.check_health()
        self.assertIs(d.lora_available, True)
        d.lora_name = "other.safetensors"
        d.check_health()
        self.assertIs(d.lora_available, False)


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.mocks = [MockComfy(i).start() for i in range(3)]
        args = grid_server.parse_args([
            "--port", "0", "--instances", ",".join(m.url for m in self.mocks),
            "--workflows-dir", WORKFLOWS])
        dispatcher = dp.Dispatcher([m.url for m in self.mocks], health_interval=0.2, poll_interval=0.05)
        self.server = grid_server.make_server(args, dispatcher)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        dispatcher.start()
        self.dispatcher = dispatcher

    def tearDown(self):
        self.dispatcher.stop()
        self.server.shutdown()
        self.server.server_close()
        for mock in self.mocks:
            mock.stop()

    def request(self, path, body=None, content_type="application/json"):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method="POST" if data is not None else "GET")
        if data is not None:
            req.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, resp.read(), resp.headers
        except urllib.error.HTTPError as error:
            return error.code, error.read(), error.headers

    def wait_done(self, timeout=10):
        deadline = time.time() + timeout
        while time.time() < deadline:
            state = json.loads(self.request("/api/state")[1])
            if state["jobs"] and all(j["status"] in ("done", "error") for j in state["jobs"]):
                return state
            time.sleep(0.05)
        raise AssertionError("jobs did not finish")

    def test_page_run_and_view(self):
        status, body, _ = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn(b"GPU grid", body)
        status, body, _ = self.request("/api/run", {"prompts": ["a sword", "", "a chest"], "variants": 3})
        self.assertEqual(status, 200)
        self.assertEqual(len(json.loads(body)["jobs"]), 6)
        state = self.wait_done()
        self.assertEqual({j["instance"] for j in state["jobs"]}, {0, 1, 2})
        job = state["jobs"][0]
        status, body, headers = self.request(f"/view?job={job['id']}&index=0")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "image/png")
        self.assertTrue(body.startswith(f"GPU{job['instance']}:grid/".encode()))
        # Plain pass-through, as used for the GLB download.
        status, body, _ = self.request("/view?filename=prop_00001_.glb&subfolder=3d&type=output")
        self.assertEqual(status, 200)
        self.assertTrue(body.endswith(b"3d/prop_00001_.glb"))

    def test_view_falls_back_when_the_instance_is_down(self):
        self.request("/api/run", {"prompts": ["a sword"], "variants": 3})
        state = self.wait_done()
        job = next(j for j in state["jobs"] if j["instance"] == 0)
        self.mocks[0].stop()
        time.sleep(0.6)  # let the health check notice
        status, body, _ = self.request(f"/view?job={job['id']}&index=0")
        self.assertEqual(status, 200)
        self.assertFalse(body.startswith(b"GPU0:"))

    def test_rejects_bad_requests(self):
        self.assertEqual(self.request("/api/run", {"prompts": []})[0], 400)
        self.assertEqual(self.request("/api/run", {"prompts": ["x"], "variants": 0})[0], 400)
        self.assertEqual(self.request("/api/run", {"prompts": ["x"]}, content_type="text/plain")[0], 415)
        self.assertEqual(self.request("/api/3d", {"job": "nope"})[0], 400)
        self.assertEqual(self.request("/view?job=nope")[0], 404)


if __name__ == "__main__":
    unittest.main()
