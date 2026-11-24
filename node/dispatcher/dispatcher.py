"""Spreads ComfyUI jobs over several ComfyUI instances, one per GPU.

Jobs go to healthy instances in round-robin order, at most `max_inflight` per
instance, through POST /prompt. Progress comes from GET /queue and
GET /history/<prompt_id>. When an instance stops answering, or restarts and
forgets its queue, its unfinished jobs go back to the queue and run elsewhere.

Standard library only, so it runs with the system Python on the VM.
"""

from __future__ import annotations

import copy
import json
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

STYLE_LORA_TITLE = "Style LoRA"
PROMPT_TITLE = "Prompt"
IMAGE_TITLE = "Image"


class InstanceError(Exception):
    """The instance did not answer, or answered with a server error."""


class HttpClient:
    """Small JSON-over-HTTP client. Tests replace it with a fake if needed."""

    def __init__(self, timeout: float = 3.0):
        self.timeout = timeout
        # Ignore proxy settings: every instance is on 127.0.0.1.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, method: str, url: str, body: Any = None, timeout: Optional[float] = None) -> Tuple[int, Any]:
        data = None
        headers = {}
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with self._opener.open(req, timeout=timeout or self.timeout) as resp:
                return resp.status, _parse_json(resp.read())
        except urllib.error.HTTPError as error:
            if error.code >= 500:
                raise InstanceError(f"HTTP {error.code} from {url}") from error
            return error.code, _parse_json(error.read())
        except (urllib.error.URLError, OSError, ValueError) as error:
            raise InstanceError(f"{url}: {error}") from error

    def get_raw(self, url: str, timeout: Optional[float] = None):
        """Returns the open response so the caller can stream the body."""
        req = urllib.request.Request(url, method="GET")
        try:
            return self._opener.open(req, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as error:
            return error
        except (urllib.error.URLError, OSError) as error:
            raise InstanceError(f"{url}: {error}") from error


def _parse_json(raw: bytes) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode(errors="replace")


# --- Workflow templates -----------------------------------------------------

def load_workflow(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        graph = json.load(handle)
    if not isinstance(graph, dict) or not all(isinstance(n, dict) and "class_type" in n for n in graph.values()):
        raise ValueError(f"{path} is not a ComfyUI API-format workflow")
    return graph


def nodes_by_title(graph: Dict[str, Any], title: str) -> List[str]:
    return [nid for nid, node in graph.items() if node.get("_meta", {}).get("title") == title]


def nodes_by_class(graph: Dict[str, Any], class_type: str) -> List[str]:
    return [nid for nid, node in graph.items() if node["class_type"] == class_type]


def bypass_node(graph: Dict[str, Any], node_id: str, through: str = "model") -> None:
    """Removes a pass-through node (such as a LoRA loader) and reconnects its
    consumers to the node's `through` input."""
    source = graph[node_id]["inputs"][through]
    del graph[node_id]
    for node in graph.values():
        for name, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and str(value[0]) == node_id:
                node["inputs"][name] = list(source)


def build_image_graph(template: Dict[str, Any], text: str, seed: int, prefix: str,
                      lora_strength: Optional[float]) -> Dict[str, Any]:
    """Fills the grid or comparison template. lora_strength=None removes the
    style LoRA from the graph."""
    graph = copy.deepcopy(template)
    prompt_nodes = nodes_by_title(graph, PROMPT_TITLE)
    if not prompt_nodes:
        raise ValueError(f"template has no node titled '{PROMPT_TITLE}'")
    for nid in prompt_nodes:
        graph[nid]["inputs"]["text"] = text
    for nid in nodes_by_class(graph, "KSampler"):
        graph[nid]["inputs"]["seed"] = seed
    for nid in nodes_by_class(graph, "SaveImage"):
        graph[nid]["inputs"]["filename_prefix"] = prefix
    for nid in nodes_by_title(graph, STYLE_LORA_TITLE):
        if lora_strength is None:
            bypass_node(graph, nid)
        else:
            graph[nid]["inputs"]["strength_model"] = lora_strength
    return graph


def build_3d_graph(template: Dict[str, Any], image: Dict[str, str], prefix: str) -> Dict[str, Any]:
    """Points the image-to-3D template at an image in ComfyUI's output folder."""
    graph = copy.deepcopy(template)
    image_nodes = nodes_by_title(graph, IMAGE_TITLE) or nodes_by_class(graph, "LoadImage")
    if not image_nodes:
        raise ValueError("template has no LoadImage node")
    name = image["filename"]
    if image.get("subfolder"):
        name = f"{image['subfolder']}/{name}"
    # LoadImage resolves "<path> [output]" against the output folder, which
    # all instances share.
    for nid in image_nodes:
        graph[nid]["inputs"]["image"] = f"{name} [{image.get('type') or 'output'}]"
    for nid in nodes_by_class(graph, "SaveGLB"):
        graph[nid]["inputs"]["filename_prefix"] = prefix
    return graph


# --- State --------------------------------------------------------------------

class Instance:
    def __init__(self, index: int, url: str):
        self.index = index
        self.url = url.rstrip("/")
        self.up: Optional[bool] = None  # None until the first health check
        self.failures = 0
        self.inflight: List[str] = []
        self.completed = 0
        self.last_error = ""
        self.device = ""

    def snapshot(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "url": self.url,
            "up": self.up,
            "inflight": len(self.inflight),
            "completed": self.completed,
            "last_error": self.last_error,
            "device": self.device,
        }


class Job:
    def __init__(self, job_id: str, kind: str, label: str, graph: Dict[str, Any], seed: Optional[int] = None,
                 pin: Optional[int] = None, source: Optional[Dict[str, Any]] = None):
        self.id = job_id
        self.kind = kind  # "image" or "3d"
        self.label = label
        self.graph = graph
        self.seed = seed
        self.pin = pin  # instance index the job must run on (warm-up), or None
        self.source = source  # for 3d jobs: the image it was made from
        self.status = "pending"  # pending, queued, running, done, error
        self.instance: Optional[int] = None
        self.prompt_id: Optional[str] = None
        self.attempts = 0
        self.outputs: List[Dict[str, str]] = []
        self.error = ""
        self.created = time.time()
        self.submitted: Optional[float] = None
        self.started: Optional[float] = None
        self.finished: Optional[float] = None

    def snapshot(self) -> Dict[str, Any]:
        run_seconds = None
        if self.finished and (self.started or self.submitted):
            run_seconds = round(self.finished - (self.started or self.submitted), 1)
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "seed": self.seed,
            "status": self.status,
            "instance": self.instance,
            "attempts": self.attempts,
            "outputs": self.outputs,
            "error": self.error,
            "run_seconds": run_seconds,
            "source": self.source,
        }


class Dispatcher:
    def __init__(self, urls: List[str], http: Optional[HttpClient] = None, max_inflight: int = 2,
                 max_attempts: int = 3, down_after: int = 2, health_interval: float = 2.0,
                 poll_interval: float = 0.5, lora_name: str = "style.safetensors"):
        self.instances = [Instance(i, url) for i, url in enumerate(urls)]
        self.http = http or HttpClient()
        self.max_inflight = max_inflight
        self.max_attempts = max_attempts
        self.down_after = down_after
        self.health_interval = health_interval
        self.poll_interval = poll_interval
        self.lora_name = lora_name
        self.lora_available: Optional[bool] = None
        self.client_id = str(uuid.uuid4())
        self.jobs: Dict[str, Job] = {}
        self.order: List[str] = []
        self._rr = 0
        self._counter = 0
        self._run_stamp = time.strftime("%Y%m%d-%H%M%S")
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._pool = ThreadPoolExecutor(max_workers=max(4, len(urls)))

    # Job creation -------------------------------------------------------------

    def new_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"{self._run_stamp}_{self._counter:04d}"

    def add_job(self, job: Job) -> Job:
        with self._lock:
            self.jobs[job.id] = job
            self.order.append(job.id)
        return job

    def add_image_jobs(self, template: Dict[str, Any], prompts: List[str], variants: int,
                       lora_strength: Optional[float], trigger: str = "", suffix: str = "",
                       seed: Optional[int] = None) -> List[Job]:
        jobs = []
        for text in prompts:
            text = text.strip()
            if not text:
                continue
            parts = [trigger] if (trigger and lora_strength is not None) else []
            parts.append(text)
            if suffix.startswith("with "):
                # "with lime green trim and details, ..." continues the subject, as in the training captions
                parts[-1] = f"{text} {suffix}"
            elif suffix:
                parts.append(suffix)
            full = ", ".join(parts)
            for v in range(variants):
                job_seed = seed + v if seed is not None else random.randint(0, 2**32 - 1)
                job_id = self.new_id()
                graph = build_image_graph(template, full, job_seed, f"grid/{job_id}", lora_strength)
                jobs.append(self.add_job(Job(job_id, "image", text, graph, seed=job_seed)))
        return jobs

    def add_warmup_jobs(self, template: Dict[str, Any], lora_strength: Optional[float]) -> List[Job]:
        jobs = []
        for inst in self.instances:
            job_id = self.new_id()
            graph = build_image_graph(template, "a wooden crate, plain white background", 1,
                                      f"warmup/{job_id}", lora_strength)
            jobs.append(self.add_job(Job(job_id, "image", f"warm-up GPU {inst.index}", graph,
                                         seed=1, pin=inst.index)))
        return jobs

    def add_3d_job(self, template: Dict[str, Any], source_job_id: str, output_index: int = 0) -> Job:
        with self._lock:
            source = self.jobs.get(source_job_id)
            if source is None or source.status != "done" or output_index >= len(source.outputs):
                raise ValueError("pick a finished image first")
            image = dict(source.outputs[output_index])
        job_id = self.new_id()
        graph = build_3d_graph(template, image, f"3d/{job_id}")
        return self.add_job(Job(job_id, "3d", f"3D from {source.label}", graph,
                                source={"job": source_job_id, "image": image}))

    def clear_finished(self) -> None:
        with self._lock:
            keep = [jid for jid in self.order if self.jobs[jid].status not in ("done", "error")]
            for jid in self.order:
                if jid not in keep:
                    del self.jobs[jid]
            self.order = keep

    # Scheduling ---------------------------------------------------------------

    def _has_capacity(self, inst: Instance) -> bool:
        return inst.up is True and len(inst.inflight) < self.max_inflight

    def _pick_instance(self, job: Job) -> Optional[Instance]:
        if job.pin is not None:
            inst = self.instances[job.pin]
            return inst if self._has_capacity(inst) else None
        n = len(self.instances)
        for k in range(n):
            inst = self.instances[(self._rr + k) % n]
            if self._has_capacity(inst):
                self._rr = (inst.index + 1) % n
                return inst
        return None

    def schedule(self) -> None:
        """Submits pending jobs to instances with free capacity."""
        while True:
            with self._lock:
                pending = [self.jobs[jid] for jid in self.order if self.jobs[jid].status == "pending"]
                choice = None
                for job in pending:
                    if job.pin is not None and self.instances[job.pin].up is False:
                        self._fail(job, f"GPU {job.pin} is down")
                        continue
                    inst = self._pick_instance(job)
                    if inst is not None:
                        choice = (job, inst)
                        break
                if choice is None:
                    return
                job, inst = choice
                job.status = "submitting"
                job.attempts += 1
                job.instance = inst.index
                inst.inflight.append(job.id)
            self._submit(job, inst)

    def _submit(self, job: Job, inst: Instance) -> None:
        try:
            status, body = self.http.request("POST", f"{inst.url}/prompt",
                                             {"prompt": job.graph, "client_id": self.client_id})
        except InstanceError as error:
            with self._lock:
                self._release(job, inst)
                if job.status == "submitting":
                    self._retry_or_fail(job, str(error))
                self._mark_down(inst, str(error))
            return
        with self._lock:
            if job.status != "submitting" or job.instance != inst.index:
                # The health check marked the instance down meanwhile and the
                # job was already requeued; ignore this submission.
                return
            if status == 200 and isinstance(body, dict) and body.get("prompt_id"):
                job.prompt_id = body["prompt_id"]
                job.status = "queued"
                job.submitted = time.time()
            else:
                # A 400 means the workflow failed validation. That will fail on
                # every instance, so the job is not retried.
                self._release(job, inst)
                self._fail(job, describe_prompt_error(status, body))

    # Polling ------------------------------------------------------------------

    def poll(self) -> None:
        """Updates the status of submitted jobs, one request batch per instance."""
        with self._lock:
            busy = [inst for inst in self.instances if inst.inflight and inst.up]
        list(self._pool.map(self._poll_instance, busy))

    def _poll_instance(self, inst: Instance) -> None:
        with self._lock:
            # Drop entries for jobs that were requeued or cleared meanwhile.
            inst.inflight = [jid for jid in inst.inflight if self._owned_by(jid, inst)]
            jobs = [(self.jobs[jid], self.jobs[jid].prompt_id) for jid in inst.inflight
                    if self.jobs[jid].status in ("queued", "running")]
        if not jobs:
            return
        try:
            _, queue = self.http.request("GET", f"{inst.url}/queue")
            queue = queue if isinstance(queue, dict) else {}
            running = {item[1] for item in queue.get("queue_running", [])}
            waiting = {item[1] for item in queue.get("queue_pending", [])}
            for job, prompt_id in jobs:
                if prompt_id in waiting:
                    continue
                if prompt_id in running:
                    with self._lock:
                        if job.status == "queued" and job.prompt_id == prompt_id:
                            job.status = "running"
                            job.started = time.time()
                    continue
                # Not in the queue: finished, or lost because the instance restarted.
                # ComfyUI moves a prompt from the queue to the history under one
                # lock, so a prompt in neither place is gone.
                _, history = self.http.request("GET", f"{inst.url}/history/{prompt_id}")
                entry = history.get(prompt_id) if isinstance(history, dict) else None
                with self._lock:
                    if job.prompt_id != prompt_id or job.instance != inst.index:
                        continue
                    if entry is None:
                        self._release(job, inst)
                        self._retry_or_fail(job, f"GPU {inst.index} lost the job (restarted?)")
                    else:
                        self._finish(job, inst, entry)
        except InstanceError as error:
            with self._lock:
                self._note_failure(inst, str(error))

    def _owned_by(self, job_id: str, inst: Instance) -> bool:
        job = self.jobs.get(job_id)
        return (job is not None and job.instance == inst.index
                and job.status in ("submitting", "queued", "running"))

    def _finish(self, job: Job, inst: Instance, entry: Dict[str, Any]) -> None:
        self._release(job, inst)
        status = entry.get("status") or {}
        job.finished = time.time()
        if status.get("status_str") == "error":
            self._fail(job, describe_execution_error(status))
            return
        outputs = []
        for node_id, node_out in (entry.get("outputs") or {}).items():
            for key in ("images", "3d"):
                for item in node_out.get(key) or []:
                    if isinstance(item, dict) and item.get("filename"):
                        outputs.append({
                            "node": str(node_id),
                            "kind": key,
                            "filename": item["filename"],
                            "subfolder": item.get("subfolder", ""),
                            "type": item.get("type", "output"),
                        })
        job.outputs = outputs
        job.status = "done"
        inst.completed += 1

    # Health -------------------------------------------------------------------

    def check_health(self) -> None:
        results = list(self._pool.map(self._probe, self.instances))
        with self._lock:
            for inst, (ok, detail) in zip(self.instances, results):
                if ok:
                    if inst.up is not True:
                        inst.last_error = ""
                    inst.up = True
                    inst.failures = 0
                    inst.device = detail
                else:
                    self._note_failure(inst, detail)
        self._check_lora()

    def _probe(self, inst: Instance) -> Tuple[bool, str]:
        try:
            status, body = self.http.request("GET", f"{inst.url}/system_stats", timeout=2.0)
        except InstanceError as error:
            return False, str(error)
        if status != 200:
            return False, f"HTTP {status}"
        devices = body.get("devices") if isinstance(body, dict) else None
        name = devices[0].get("name", "") if devices and isinstance(devices[0], dict) else ""
        return True, str(name)

    def _check_lora(self) -> None:
        inst = next((i for i in self.instances if i.up), None)
        if inst is None:
            return
        try:
            status, body = self.http.request("GET", f"{inst.url}/models/loras", timeout=2.0)
        except InstanceError:
            return
        if status == 200 and isinstance(body, list):
            self.lora_available = self.lora_name in body

    def _note_failure(self, inst: Instance, error: str) -> None:
        inst.failures += 1
        inst.last_error = error
        if inst.failures >= self.down_after or inst.up is None:
            self._mark_down(inst, error)

    def _mark_down(self, inst: Instance, error: str) -> None:
        inst.up = False
        inst.failures = max(inst.failures, self.down_after)
        inst.last_error = error
        for jid in list(inst.inflight):
            job = self.jobs.get(jid)
            inst.inflight.remove(jid)
            if job is not None and job.status in ("queued", "running", "submitting"):
                self._retry_or_fail(job, f"GPU {inst.index} went down: {error}")

    # Helpers ------------------------------------------------------------------

    def _release(self, job: Job, inst: Instance) -> None:
        if job.id in inst.inflight:
            inst.inflight.remove(job.id)

    def _retry_or_fail(self, job: Job, reason: str) -> None:
        if job.pin is not None or job.attempts >= self.max_attempts:
            self._fail(job, reason)
            return
        job.status = "pending"
        job.prompt_id = None
        job.started = None
        job.error = reason

    def _fail(self, job: Job, reason: str) -> None:
        job.status = "error"
        job.error = reason
        job.finished = job.finished or time.time()

    # Output proxy -------------------------------------------------------------

    def view_url(self, params: Dict[str, str], preferred: Optional[int] = None) -> List[str]:
        """Candidate /view URLs, preferred instance first. All instances share
        the output folder, so any healthy one can serve a file."""
        query = urllib.parse.urlencode({k: params[k] for k in ("filename", "subfolder", "type") if k in params})
        with self._lock:
            order = sorted(self.instances, key=lambda i: (i.index != preferred, i.up is not True, i.index))
        return [f"{inst.url}/view?{query}" for inst in order if inst.up is not False]

    def output_for(self, job_id: str, index: int) -> Tuple[Optional[Dict[str, str]], Optional[int]]:
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None or index >= len(job.outputs):
                return None, None
            return dict(job.outputs[index]), job.instance

    # Background loop ----------------------------------------------------------

    def tick(self) -> None:
        self.schedule()
        self.poll()
        self.schedule()

    def start(self) -> None:
        def health_loop():
            while not self._stop.is_set():
                self.check_health()
                self._stop.wait(self.health_interval)

        def work_loop():
            while not self._stop.is_set():
                try:
                    self.tick()
                except Exception as error:  # keep the loop alive; show the error
                    print(f"dispatcher loop error: {error!r}", flush=True)
                self._stop.wait(self.poll_interval)

        self.check_health()
        for target in (health_loop, work_loop):
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=5)
        self._pool.shutdown(wait=False)

    def state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "instances": [inst.snapshot() for inst in self.instances],
                "jobs": [self.jobs[jid].snapshot() for jid in self.order],
                "lora": {"name": self.lora_name, "available": self.lora_available},
                "max_inflight": self.max_inflight,
            }


def describe_prompt_error(status: int, body: Any) -> str:
    if isinstance(body, dict):
        error = body.get("error")
        message = error.get("message", "") if isinstance(error, dict) else str(error or "")
        details = []
        for node_id, info in (body.get("node_errors") or {}).items():
            for err in (info or {}).get("errors", []):
                details.append(f"node {node_id}: {err.get('message', '')} {err.get('details', '')}".strip())
        return "; ".join([f"rejected (HTTP {status}): {message}"] + details)
    return f"rejected (HTTP {status}): {body}"


def describe_execution_error(status: Dict[str, Any]) -> str:
    for kind, data in status.get("messages") or []:
        if kind == "execution_error" and isinstance(data, dict):
            return f"{data.get('node_type', 'node')}: {data.get('exception_message', '').strip()}"
    return "execution failed"
