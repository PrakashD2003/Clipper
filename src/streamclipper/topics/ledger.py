"""Persistent ledger tracking produced topic videos to prevent duplicate content."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import threading
from typing import Any

from streamclipper.cache import _atomic_write_text
from streamclipper.config import Config

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()


def _tokenize(title: str, keywords: list[str] | None = None) -> set[str]:
    """Extract lowercase word tokens from title and keywords."""
    combined = (title or "") + " " + " ".join(keywords or [])
    tokens = set(re.findall(r"[a-zA-Z0-9\u0900-\u097F]+", combined.lower()))
    return tokens


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Compute token Jaccard similarity between two token sets."""
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    intersection = set_a & set_b
    return len(intersection) / len(union)


class TopicLedger:
    """JSON-backed record of produced topic videos with atomic writes and similarity checks."""

    def __init__(self, config: Config | Path | str | None = None) -> None:
        if isinstance(config, (str, Path)):
            cache_dir = Path(config)
            self.config = Config()
            self.threshold = 0.5
        elif config is not None:
            self.config = config
            cache_dir = Path(config.get("general.cache_dir", "./.cache"))
            self.threshold = float(config.get("topics.similarity_threshold", 0.5))
        else:
            self.config = Config()
            cache_dir = Path(self.config.get("general.cache_dir", "./.cache"))
            self.threshold = float(self.config.get("topics.similarity_threshold", 0.5))

        ledger_name = self.config.get("topics.ledger_name", "topics_ledger.json") or "topics_ledger.json"
        self.path = cache_dir / ledger_name

    def _load(self) -> list[dict[str, Any]]:
        """Load entries from the ledger file safely."""
        try:
            if not self.path.exists():
                return []
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        entries = data.get("entries") if isinstance(data, dict) else None
        return [e for e in entries if isinstance(e, dict)] if isinstance(entries, list) else []

    def _save(self, entries: list[dict[str, Any]]) -> None:
        """Atomically persist entries to topics_ledger.json."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"entries": entries}, indent=2)
        _atomic_write_text(self.path, payload)

    def record(
        self,
        topic_id: str,
        title: str,
        game: str,
        keywords: list[str] | None = None,
        shape: str = "numbered",
        plan_id: str = "",
    ) -> dict[str, Any]:
        """Record a produced topic video into the ledger atomically."""
        with _LOCK:
            entries = self._load()
            entry = {
                "topic_id": str(topic_id),
                "title": str(title),
                "game": str(game),
                "keywords": list(keywords or []),
                "shape": str(shape),
                "plan_id": str(plan_id or ""),
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            # If topic_id already exists, remove it so the updated record is placed at the end
            entries = [e for e in entries if e.get("topic_id") != str(topic_id)]
            entries.append(entry)

            self._save(entries)
            return entry

    def entries(self, game: str = "") -> list[dict[str, Any]]:
        """Return all ledger entries, optionally filtered by game (case-insensitive)."""
        with _LOCK:
            all_entries = self._load()
            if not game or not str(game).strip():
                return list(all_entries)
            target_game = str(game).strip().lower()
            return [
                e for e in all_entries
                if str(e.get("game", "")).strip().lower() == target_game
            ]

    def find_similar(
        self,
        title: str,
        keywords: list[str] | None = None,
        game: str = "",
    ) -> dict[str, Any] | None:
        """Find an existing ledger entry with token Jaccard similarity >= threshold.

        Returns the best matching entry or None.
        """
        game_entries = self.entries(game)
        if not game_entries:
            return None

        query_tokens = _tokenize(title, keywords)
        if not query_tokens:
            return None

        best_entry: dict[str, Any] | None = None
        best_score = 0.0

        for entry in game_entries:
            entry_tokens = _tokenize(
                str(entry.get("title", "")),
                entry.get("keywords") if isinstance(entry.get("keywords"), list) else None,
            )
            sim = _jaccard(query_tokens, entry_tokens)
            if sim >= self.threshold and sim > best_score:
                best_score = sim
                best_entry = entry

        return best_entry

    def last_shape(self, game: str = "") -> str:
        """Return the story shape used by the most recent ledger entry for this game."""
        game_entries = self.entries(game)
        if not game_entries:
            return ""
        # Guarantee returning the entry with the most recent recorded_at timestamp
        sorted_entries = sorted(game_entries, key=lambda e: str(e.get("recorded_at", "")))
        return str(sorted_entries[-1].get("shape", "") or "")
