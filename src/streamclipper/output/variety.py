"""Persistent variety state: rotates formats / hook styles, remembers outros, caps clips per day."""
from __future__ import annotations

from datetime import date, datetime
import json
import logging
import os
from pathlib import Path
import tempfile
import threading
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_HOOK_STYLES = ["question", "hot_take", "countdown", "nobody_noticed"]
_MAX_ENTRIES = 500
_LOCK = threading.Lock()


class VarietyState:
    """JSON-backed record of recent clips (atomic writes) used to vary format, hook and outro."""

    def __init__(self, path: Path, hook_styles: list[str] | None = None) -> None:
        self.path = Path(path)
        self.hook_styles = list(hook_styles) if hook_styles else list(DEFAULT_HOOK_STYLES)

    def _load(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        entries = data.get("entries") if isinstance(data, dict) else None
        return [e for e in entries if isinstance(e, dict)] if isinstance(entries, list) else []

    def _save(self, entries: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=self.path.name + ".", suffix=".tmp", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"entries": entries[-_MAX_ENTRIES:]}, fh, indent=2)
            os.replace(tmp_name, self.path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    @staticmethod
    def _least_recent(options: list[str], entries: list[dict[str, Any]], key: str) -> str:
        """Option unused for longest (never-used first; ties resolved by option order)."""
        last_seen: dict[str, int] = {}
        for pos, entry in enumerate(entries):
            value = entry.get(key)
            if value in options:
                last_seen[value] = pos
        return min(options, key=lambda opt: (last_seen.get(opt, -1), options.index(opt)))

    def next_format(self, available: list[str]) -> str:
        opts = [str(a) for a in available if a]
        if not opts:
            raise ValueError("next_format requires at least one available format")
        return self._least_recent(opts, self._load(), "format")

    def next_hook_style(self) -> str:
        return self._least_recent(self.hook_styles, self._load(), "hook_style")

    def next_caption_preset(self, n_presets: int, streamer: str | None = None) -> int:
        """Index in [0, n_presets) unused longest for `streamer` (rotates per streamer, persisted)."""
        n = int(n_presets)
        if n <= 1:
            return 0
        entries = [
            e for e in self._load() if streamer is None or e.get("streamer") == streamer
        ]
        last_seen: dict[int, int] = {}
        for pos, entry in enumerate(entries):
            value = entry.get("caption_preset")
            if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < n:
                last_seen[value] = pos
        return min(range(n), key=lambda i: (last_seen.get(i, -1), i))

    def recent_outros(self, n: int = 20) -> list[str]:
        if n <= 0:
            return []
        outros = [str(e.get("outro") or "") for e in self._load()]
        return [o for o in outros if o][-int(n):]

    def clips_today(self, streamer: str) -> int:
        today = date.today().isoformat()
        return sum(
            1
            for e in self._load()
            if e.get("streamer") == streamer
            and e.get("date") == today
            and not e.get("caption_only")
        )

    def record_caption_preset(self, streamer: str, preset: int) -> None:
        """Persist a caption-preset pick made outside the CLI loop (MCP render path)."""
        with _LOCK:
            entries = self._load()
            entries.append(
                {
                    "streamer": streamer,
                    "caption_preset": int(preset),
                    "caption_only": True,
                    "date": date.today().isoformat(),
                    "ts": datetime.now().isoformat(timespec="seconds"),
                }
            )
            self._save(entries)

    def record(
        self,
        streamer: str,
        fmt: str,
        hook_style: str,
        outro: str,
        caption_preset: int | None = None,
    ) -> None:
        with _LOCK:
            entries = self._load()
            entries.append(
                {
                    "streamer": streamer,
                    "format": fmt,
                    "hook_style": hook_style,
                    "outro": outro,
                    "caption_preset": caption_preset,
                    "date": date.today().isoformat(),
                    "ts": datetime.now().isoformat(timespec="seconds"),
                }
            )
            self._save(entries)
