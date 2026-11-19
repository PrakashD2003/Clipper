"""In-process registry of asynchronous MCP render jobs with JSON snapshots.

MCP clients (e.g. ChatGPT) abandon tool calls after ~3 minutes, so rendering tools start a
background job and return immediately; the client polls ``get_render_status``.  Snapshots under
``<general.cache_dir>/render_jobs/<job_id>.json`` let status survive a server restart.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import threading
import time
from typing import Any
import uuid

from ..cache import _atomic_write_text

logger = logging.getLogger(__name__)

ACTIVE_STATUSES = ("queued", "running")
FINAL_STATUSES = ("done", "failed", "cancelled")
POLL_EVERY_SECONDS = 20
SECONDS_PER_CLIP_ESTIMATE = 90
RESTART_ERROR = "server restarted during render; call render again"

_JOBS: dict[str, dict[str, Any]] = {}
_ORDER: list[str] = []  # creation order, oldest first
_LOCK = threading.RLock()


def jobs_dir(config: Any) -> Path:
    return Path(config.get("general.cache_dir", "./.cache")) / "render_jobs"


def _snapshot(job: dict[str, Any]) -> None:
    path = job.get("snapshot_path")
    if not path:
        return
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(p, json.dumps(job, indent=2, default=str))
    except Exception:
        logger.warning("Failed to write render job snapshot %s", path, exc_info=True)


def reset_registry() -> None:
    """Forget all in-memory jobs (snapshots stay on disk). Used by tests."""
    with _LOCK:
        _JOBS.clear()
        _ORDER.clear()


def create_job(config: Any, kind: str, url: str, clips_total: int) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    job: dict[str, Any] = {
        "job_id": job_id,
        "kind": kind,
        "url": url,
        "status": "queued",
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "progress": {"clips_done": 0, "clips_total": int(clips_total), "stage": "queued"},
        "results": [],
        "summary": {},
        "error": "",
        "cancel_requested": False,
        "snapshot_path": str(jobs_dir(config) / f"{job_id}.json"),
    }
    with _LOCK:
        _JOBS[job_id] = job
        _ORDER.append(job_id)
        _snapshot(job)
    return job


def update_job(job: dict[str, Any], **fields: Any) -> None:
    with _LOCK:
        job.update(fields)
        _snapshot(job)


def set_stage(job: dict[str, Any], stage: str) -> None:
    with _LOCK:
        job["progress"]["stage"] = stage
        _snapshot(job)


def clip_done(job: dict[str, Any], result: dict[str, Any] | None = None) -> None:
    with _LOCK:
        job["progress"]["clips_done"] = int(job["progress"].get("clips_done", 0)) + 1
        if result is not None:
            job["results"].append(result)
        _snapshot(job)


def finish_job(
    job: dict[str, Any],
    status: str,
    *,
    error: str = "",
    results: list[dict[str, Any]] | None = None,
    summary: dict[str, Any] | None = None,
) -> None:
    with _LOCK:
        job["status"] = status
        job["finished_at"] = time.time()
        job["error"] = error
        if results is not None:
            job["results"] = results
        if summary is not None:
            job["summary"] = summary
        if status == "done":
            job["progress"]["stage"] = "done"
        elif status == "cancelled":
            job["progress"]["stage"] = "cancelled"
        elif status == "failed":
            job["progress"]["stage"] = "failed"
        _snapshot(job)


def request_cancel(job: dict[str, Any]) -> str:
    """Flag cancellation. Returns the resulting status."""
    with _LOCK:
        if job["status"] in FINAL_STATUSES:
            return job["status"]
        job["cancel_requested"] = True
        if job["status"] == "queued":
            finish_job(job, "cancelled")
        else:
            job["progress"]["stage"] = "cancelling (stops after the current clip)"
            _snapshot(job)
        return job["status"]


def load_snapshots(config: Any) -> None:
    """Load job snapshots not already in memory; orphaned active jobs become failed."""
    directory = jobs_dir(config)
    if not directory.is_dir():
        return
    loaded: list[dict[str, Any]] = []
    for path in directory.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or not data.get("job_id"):
            continue
        loaded.append(data)
    loaded.sort(key=lambda j: float(j.get("created_at") or 0.0))
    with _LOCK:
        for data in loaded:
            jid = str(data["job_id"])
            if jid in _JOBS:
                continue
            data.setdefault("progress", {"clips_done": 0, "clips_total": 0, "stage": ""})
            data.setdefault("results", [])
            data["snapshot_path"] = str(directory / f"{jid}.json")
            _JOBS[jid] = data
            _ORDER.append(jid)
            if data.get("status") in ACTIVE_STATUSES:
                finish_job(data, "failed", error=RESTART_ERROR)
        _ORDER.sort(key=lambda i: float(_JOBS[i].get("created_at") or 0.0))


def get_job(job_id: str) -> dict[str, Any] | None:
    with _LOCK:
        return _JOBS.get(job_id)


def queue_position(job: dict[str, Any]) -> int:
    """1 = next to start (0 when not queued)."""
    if job["status"] != "queued":
        return 0
    with _LOCK:
        queued = [i for i in _ORDER if _JOBS[i]["status"] == "queued"]
    return queued.index(job["job_id"]) + 1 if job["job_id"] in queued else 0


def public_view(job: dict[str, Any]) -> dict[str, Any]:
    view = {k: v for k, v in job.items() if k != "snapshot_path"}
    view["progress"] = dict(job["progress"])
    view["queue_position"] = queue_position(job)
    if job["status"] in ACTIVE_STATUSES:
        view["poll_every_seconds"] = POLL_EVERY_SECONDS
        view["message"] = (
            "Rendering is still in progress; a render taking several minutes is normal. "
            f"Call get_render_status(job_id) again in ~{POLL_EVERY_SECONDS} s. Do not render "
            "clips yourself or inspect files in the meantime."
        )
    elif job["status"] == "done":
        view["message"] = "Render finished; see results."
    elif job["status"] == "cancelled":
        view["message"] = "Render was cancelled; clips already finished are in results."
    else:
        view["message"] = f"Render failed: {job.get('error') or 'unknown error'}"
    return view


def list_jobs(*, url: str = "", active_only: bool = False, limit: int = 5) -> list[dict[str, Any]]:
    with _LOCK:
        jobs = [_JOBS[i] for i in reversed(_ORDER)]
    if url:
        jobs = [j for j in jobs if j.get("url") == url]
    if active_only:
        jobs = [j for j in jobs if j["status"] in ACTIVE_STATUSES]
    return jobs[:limit] if limit else jobs


def active_job_for_url(url: str) -> dict[str, Any] | None:
    for j in list_jobs(url=url, active_only=True, limit=0):
        return j
    return None


class JobProgress:
    """Progress sink handed to render coroutines (job-backed)."""

    def __init__(self, job: dict[str, Any] | None = None) -> None:
        self.job = job
        self.clip_label = ""

    @property
    def cancelled(self) -> bool:
        return bool(self.job and self.job.get("cancel_requested"))

    def start_clip(self, index: int, total: int) -> None:
        self.clip_label = f"clip {index}/{total}: " if total > 1 else ""

    def stage(self, text: str) -> None:
        if self.job is not None:
            set_stage(self.job, f"{self.clip_label}{text}")

    def clip_done(self, result: dict[str, Any] | None = None) -> None:
        if self.job is not None:
            clip_done(self.job, result)
