"""Tests for asynchronous MCP render jobs (get_render_status / cancel_render)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import threading
import time
from typing import Any

import pytest

from streamclipper.cache import StepCache
from streamclipper.config import Config
from streamclipper.mcp_server import jobs
from streamclipper.mcp_server import server as mcp_server
from streamclipper.mcp_server.server import mcp
from streamclipper.models import (
    ClipResult,
    Highlight,
    PipelineState,
    PipelineStatus,
    Platform,
    StreamInfo,
)

from tests.test_mcp_server import _make_config

URL = "https://www.youtube.com/watch?v=job12345678"
URL2 = "https://www.youtube.com/watch?v=job22222222"


def _j(result: Any) -> dict[str, Any]:
    assert getattr(result, "is_error", False) is False
    return json.loads(result.content[0].text)


async def _call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return _j(await mcp.call_tool(name, args))


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = _make_config(tmp_path)
    cache = StepCache(config)
    monkeypatch.setattr(mcp_server, "_get_config", lambda: config)
    jobs.reset_registry()
    seg = tmp_path / "temp" / "segments" / "s.mp4"
    seg.parent.mkdir(parents=True, exist_ok=True)
    seg.write_bytes(b"mp4")
    monkeypatch.setattr(mcp_server, "download_segment", lambda *a, **kw: seg)
    yield config, cache, tmp_path
    jobs.reset_registry()


def _seed(cache: StepCache, url: str, vid: str, n: int = 1) -> None:
    hs = [
        Highlight(
            start=10.0 + 100 * i, end=45.0 + 100 * i, score=9.0, title=f"H{i}", reason="r",
            final_score=9.0,
        )
        for i in range(n)
    ]
    cache.save_state(
        PipelineState(
            stream_info=StreamInfo(
                url=url, title="T", platform=Platform.YOUTUBE, streamer="S", game="GTA 6",
                duration_seconds=3600, video_id=vid,
            ),
            status=PipelineStatus.ASSEMBLING,
            highlights=hs,
            selected_highlights=hs,
        )
    )


def _slow_assemble(tmp_path: Path, gate: threading.Event, entered: threading.Event | None = None):
    def fake(self, highlight, metadata, segment_paths, clip_index, mode, *a, **kw):
        if entered is not None:
            entered.set()
        assert gate.wait(timeout=20), "gate never released"
        out = tmp_path / "output" / f"clip_{clip_index:02d}_{int(highlight.start)}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"mp4")
        return ClipResult(
            video_path=out, metadata_path=None, metadata=metadata, duration_seconds=35.0
        )

    return fake


async def _wait_status(job_id: str, wanted: set[str], timeout: float = 15.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        st = await _call("get_render_status", {"job_id": job_id})
        if st["status"] in wanted or time.monotonic() > deadline:
            return st
        await asyncio.sleep(0.05)


RENDER_ARGS = {"url": URL, "start": 10.0, "end": 45.0, "hook_title": "Hook", "mode": "raw"}


@pytest.mark.asyncio
async def test_J1_J3_render_clip_returns_started_then_status_done(env, monkeypatch) -> None:
    _, cache, tmp_path = env
    _seed(cache, URL, "job12345678")
    gate, entered = threading.Event(), threading.Event()
    monkeypatch.setattr(
        mcp_server.Pipeline, "_assemble_highlight", _slow_assemble(tmp_path, gate, entered)
    )
    t0 = time.monotonic()
    started = await _call("render_clip", dict(RENDER_ARGS))
    assert time.monotonic() - t0 < 3.0
    assert started["status"] == "started"
    assert started["poll_with"] == "get_render_status"
    assert started["poll_every_seconds"] == 20
    assert started["clips_planned"] == 1 and started["estimated_seconds"] == 90
    assert started["url"] == URL and started["job_id"]

    st = await _wait_status(started["job_id"], {"running"})
    assert st["status"] == "running"
    assert st["progress"]["clips_total"] == 1 and st["progress"]["stage"]
    assert "get_render_status" in st["message"]

    gate.set()
    st = await _wait_status(started["job_id"], {"done", "failed"})
    assert st["status"] == "done", st
    assert st["progress"]["clips_done"] == 1
    assert len(st["results"]) == 1
    res = st["results"][0]
    assert res["mode"] == "raw" and res["quarantined"] is False
    assert Path(res["video_path"]).is_file()
    # merged into cached state exactly like the synchronous path
    assert len(cache.load_state(URL).clips) == 1


@pytest.mark.asyncio
async def test_J1_validation_errors_stay_synchronous(env) -> None:
    _, cache, _ = env
    bad = await _call("render_clip", {**RENDER_ARGS, "url": "  "})
    assert "error" in bad and "Invalid URL" in bad["error"]
    bad = await _call("render_clip", {**RENDER_ARGS, "start": 50.0, "end": 10.0})
    assert "Invalid clip range" in bad["error"]
    bad = await _call("render_clip", {**RENDER_ARGS, "mode": "narrative"})
    assert "Missing narration_intro" in bad["error"]
    bad = await _call("render_all_selected", {"url": URL})
    assert "No cached pipeline state" in bad["error"]
    assert (await _call("get_render_status", {}))["count"] == 0


@pytest.mark.asyncio
async def test_J1_wait_true_keeps_old_blocking_shape(env, monkeypatch) -> None:
    _, cache, tmp_path = env
    _seed(cache, URL, "job12345678", n=2)
    gate = threading.Event()
    gate.set()
    monkeypatch.setattr(mcp_server.Pipeline, "_assemble_highlight", _slow_assemble(tmp_path, gate))
    single = await _call("render_clip", {**RENDER_ARGS, "wait": True})
    assert {"url", "mode", "bounds_adjustment", "quarantined", "quarantine_reason", "clip"} <= set(
        single
    )
    allr = await _call("render_all_selected", {"url": URL, "mode": "raw", "wait": True})
    assert "rendered_count" in allr and "skipped_count" in allr and "clips" in allr
    assert "job_id" not in allr


@pytest.mark.asyncio
async def test_J2_two_jobs_serialize_second_queued_with_position(env, monkeypatch) -> None:
    _, cache, tmp_path = env
    _seed(cache, URL, "job12345678")
    _seed(cache, URL2, "job22222222")
    gate, entered = threading.Event(), threading.Event()
    monkeypatch.setattr(
        mcp_server.Pipeline, "_assemble_highlight", _slow_assemble(tmp_path, gate, entered)
    )
    a = await _call("render_clip", dict(RENDER_ARGS))
    b = await _call("render_clip", {**RENDER_ARGS, "url": URL2})
    await _wait_status(a["job_id"], {"running"})
    sb = await _call("get_render_status", {"job_id": b["job_id"]})
    assert sb["status"] == "queued" and sb["queue_position"] == 1
    active = await _call("get_render_status", {})
    assert {j["job_id"] for j in active["active_jobs"]} == {a["job_id"], b["job_id"]}
    by_url = await _call("get_render_status", {"url": URL2})
    assert by_url["count"] == 1
    pipes = await _call("get_pipeline_status", {})
    assert any("render_job_active" in p for p in pipes["pipelines"])

    gate.set()
    assert (await _wait_status(a["job_id"], {"done", "failed"}))["status"] == "done"
    assert (await _wait_status(b["job_id"], {"done", "failed"}))["status"] == "done"


@pytest.mark.asyncio
async def test_J3_cancel_queued_job_never_starts(env, monkeypatch) -> None:
    _, cache, tmp_path = env
    _seed(cache, URL, "job12345678")
    _seed(cache, URL2, "job22222222")
    gate = threading.Event()
    calls: list[str] = []
    base = _slow_assemble(tmp_path, gate)

    def counting(self, highlight, *a, **kw):
        calls.append(highlight.title)
        return base(self, highlight, *a, **kw)

    monkeypatch.setattr(mcp_server.Pipeline, "_assemble_highlight", counting)
    a = await _call("render_clip", dict(RENDER_ARGS))
    b = await _call("render_clip", {**RENDER_ARGS, "url": URL2})
    await _wait_status(a["job_id"], {"running"})
    cancelled = await _call("cancel_render", {"job_id": b["job_id"]})
    assert cancelled["status"] == "cancelled"
    gate.set()
    await _wait_status(a["job_id"], {"done", "failed"})
    await asyncio.sleep(0.2)
    sb = await _call("get_render_status", {"job_id": b["job_id"]})
    assert sb["status"] == "cancelled" and sb["results"] == []
    assert calls == ["Hook"]
    assert "error" in await _call("cancel_render", {"job_id": "nope"})


@pytest.mark.asyncio
async def test_J3_cancel_running_batch_stops_after_current_clip(env, monkeypatch) -> None:
    _, cache, tmp_path = env
    _seed(cache, URL, "job12345678", n=3)
    gate, entered = threading.Event(), threading.Event()
    monkeypatch.setattr(
        mcp_server.Pipeline, "_assemble_highlight", _slow_assemble(tmp_path, gate, entered)
    )
    started = await _call("render_all_selected", {"url": URL, "mode": "raw"})
    assert started["status"] == "started" and started["clips_planned"] == 3
    assert started["estimated_seconds"] == 270
    assert await asyncio.to_thread(entered.wait, 10)
    await _call("cancel_render", {"job_id": started["job_id"]})
    gate.set()
    st = await _wait_status(started["job_id"], {"done", "failed", "cancelled"})
    assert st["status"] == "cancelled"
    assert len(st["results"]) == 1


@pytest.mark.asyncio
async def test_J2_snapshot_written_and_dead_running_job_reported_failed(env, monkeypatch) -> None:
    config, cache, tmp_path = env
    _seed(cache, URL, "job12345678")
    gate, entered = threading.Event(), threading.Event()
    monkeypatch.setattr(
        mcp_server.Pipeline, "_assemble_highlight", _slow_assemble(tmp_path, gate, entered)
    )
    started = await _call("render_clip", dict(RENDER_ARGS))
    await _wait_status(started["job_id"], {"running"})
    snap = tmp_path / ".cache" / "render_jobs" / f"{started['job_id']}.json"
    assert snap.is_file()
    assert json.loads(snap.read_text(encoding="utf-8"))["status"] == "running"

    # Simulate a server restart: forget memory while the snapshot still says "running".
    jobs.reset_registry()
    st = await _call("get_render_status", {"job_id": started["job_id"]})
    assert st["status"] == "failed"
    assert "server restarted during render; call render again" in st["error"]
    assert json.loads(snap.read_text(encoding="utf-8"))["status"] == "failed"
    gate.set()
    await asyncio.sleep(0.3)


def test_J5_instructions_describe_async_rendering() -> None:
    text = mcp_server.SERVER_INSTRUCTIONS
    assert "get_render_status" in text and "ASYNCHRONOUS" in text
    assert "never fall back" in text
