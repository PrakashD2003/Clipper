"""SQLite-backed full-text and signal index for topic videos."""
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from streamclipper.cache import StepCache
from streamclipper.config import Config
from streamclipper.library.models import LibrarySource, SearchHit
from streamclipper.models import PipelineState

logger = logging.getLogger(__name__)


def _escape_fts5_query(query: str, joiner: str = " ") -> str:
    """Safely format and quote user terms for FTS5 queries."""
    cleaned = str(query or "").strip()
    if not cleaned:
        return ""
    raw_tokens = cleaned.split()
    escaped_tokens: list[str] = []
    for tok in raw_tokens:
        t = tok.strip("\"' ")
        if not t:
            continue
        t_escaped = t.replace('"', '""')
        escaped_tokens.append(f'"{t_escaped}"')
    return joiner.join(escaped_tokens)


class LibraryIndex:
    """Manages SQLite storage and FTS5 indexing of source videos and transcript segments."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        db_path_str = self.config.get("library.db_path", "")
        if db_path_str:
            self.db_path = Path(db_path_str)
        else:
            cache_dir = Path(self.config.get("general.cache_dir", "./.cache"))
            self.db_path = cache_dir / "library.sqlite"

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Open, yield, and reliably close SQLite connection in WAL mode."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            self._ensure_schema(conn)
            with conn:
                yield conn
        finally:
            conn.close()

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        """Create tables, FTS5 virtual tables, and triggers if not present."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );
            """
        )
        row = conn.execute(
            "SELECT version FROM schema_version WHERE version = 1;"
        ).fetchone()
        if row is not None:
            return

        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                title TEXT NOT NULL,
                channel TEXT NOT NULL,
                game TEXT NOT NULL,
                kind TEXT NOT NULL,
                language TEXT NOT NULL,
                duration_seconds INTEGER NOT NULL,
                video_id TEXT NOT NULL,
                added_at TEXT NOT NULL,
                indexed_at TEXT NOT NULL,
                segment_count INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                start REAL NOT NULL,
                end REAL NOT NULL,
                text TEXT NOT NULL,
                chat_text TEXT NOT NULL,
                heat REAL NOT NULL DEFAULT 0.0,
                loudness REAL NOT NULL DEFAULT 0.0,
                chat_rate REAL NOT NULL DEFAULT 0.0
            );

            CREATE INDEX IF NOT EXISTS idx_segments_source_id ON segments(source_id);
            CREATE INDEX IF NOT EXISTS idx_segments_start ON segments(start);

            CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
                text,
                chat_text,
                content='segments',
                content_rowid='id',
                tokenize='unicode61 remove_diacritics 2'
            );

            CREATE TRIGGER IF NOT EXISTS segments_ai AFTER INSERT ON segments BEGIN
                INSERT INTO segments_fts(rowid, text, chat_text) VALUES (new.id, new.text, new.chat_text);
            END;

            CREATE TRIGGER IF NOT EXISTS segments_ad AFTER DELETE ON segments BEGIN
                INSERT INTO segments_fts(segments_fts, rowid, text, chat_text) VALUES('delete', old.id, old.text, old.chat_text);
            END;

            CREATE TRIGGER IF NOT EXISTS segments_au AFTER UPDATE ON segments BEGIN
                INSERT INTO segments_fts(segments_fts, rowid, text, chat_text) VALUES('delete', old.id, old.text, old.chat_text);
                INSERT INTO segments_fts(rowid, text, chat_text) VALUES (new.id, new.text, new.chat_text);
            END;

            CREATE TABLE IF NOT EXISTS frame_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                t REAL NOT NULL,
                note TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_frame_notes_source_id ON frame_notes(source_id);

            CREATE VIRTUAL TABLE IF NOT EXISTS frame_notes_fts USING fts5(
                note,
                content='frame_notes',
                content_rowid='id',
                tokenize='unicode61 remove_diacritics 2'
            );

            CREATE TRIGGER IF NOT EXISTS frame_notes_ai AFTER INSERT ON frame_notes BEGIN
                INSERT INTO frame_notes_fts(rowid, note) VALUES (new.id, new.note);
            END;

            CREATE TRIGGER IF NOT EXISTS frame_notes_ad AFTER DELETE ON frame_notes BEGIN
                INSERT INTO frame_notes_fts(frame_notes_fts, rowid, note) VALUES('delete', old.id, old.note);
            END;

            CREATE TRIGGER IF NOT EXISTS frame_notes_au AFTER UPDATE ON frame_notes BEGIN
                INSERT INTO frame_notes_fts(frame_notes_fts, rowid, note) VALUES('delete', old.id, old.note);
                INSERT INTO frame_notes_fts(rowid, note) VALUES (new.id, new.note);
            END;
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (1, ?);",
            (datetime.now(timezone.utc).isoformat(),),
        )
        conn.commit()

    def get_schema_version(self) -> int:
        """Return the highest schema version recorded in the database."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT MAX(version) FROM schema_version;").fetchone()
            return int(row[0]) if row and row[0] is not None else 0

    def index_state(
        self,
        state: PipelineState,
        *,
        kind: str = "stream",
        chat_df: Any = None,
    ) -> LibrarySource:
        """Build ~30s windows across state, extract text/chat/signals, and save to SQLite."""
        cache = StepCache(self.config)
        source_id = cache.get_cache_key(state.stream_info.url)

        duration = float(state.stream_info.duration_seconds or 0.0)
        if state.transcript and state.transcript.segments:
            max_seg = max((s.end for s in state.transcript.segments), default=0.0)
            duration = max(duration, max_seg)
        if (
            chat_df is not None
            and hasattr(chat_df, "empty")
            and not chat_df.empty
            and "offset_seconds" in chat_df.columns
        ):
            max_chat = float(chat_df["offset_seconds"].max())
            duration = max(duration, max_chat)
        for spike in (state.loudness_spikes or []) + (state.heatmap_spikes or []):
            duration = max(duration, float(spike.end))

        window_seconds = max(1.0, float(self.config.get("library.window_seconds", 30)))
        window_step = max(0.1, float(self.config.get("library.window_step_seconds", 20)))
        max_chat_chars = int(self.config.get("library.max_chat_chars_per_window", 600))

        segments_to_insert: list[tuple[str, float, float, str, str, float, float, float]] = []

        if duration > 0.0:
            w_start = 0.0
            while w_start < duration:
                w_end = w_start + window_seconds

                # Midpoint transcript text
                matching_texts: list[str] = []
                if state.transcript and state.transcript.segments:
                    for seg in state.transcript.segments:
                        midpoint = (seg.start + seg.end) / 2.0
                        if w_start <= midpoint < w_end:
                            t = seg.text.strip()
                            if t:
                                matching_texts.append(t)
                window_text = " ".join(matching_texts)

                # Chat messages in window
                window_chat = ""
                chat_count = 0
                if (
                    chat_df is not None
                    and hasattr(chat_df, "empty")
                    and not chat_df.empty
                    and "offset_seconds" in chat_df.columns
                    and "message" in chat_df.columns
                ):
                    mask = (chat_df["offset_seconds"] >= w_start) & (
                        chat_df["offset_seconds"] < w_end
                    )
                    matching_chat = chat_df.loc[mask]
                    chat_count = len(matching_chat)
                    if chat_count > 0:
                        msgs = [
                            str(m).strip()
                            for m in matching_chat["message"]
                            if str(m).strip()
                        ]
                        window_chat = " ".join(msgs)
                        if len(window_chat) > max_chat_chars:
                            window_chat = window_chat[:max_chat_chars]

                chat_rate = (
                    (chat_count / (window_seconds / 60.0))
                    if window_seconds > 0
                    else 0.0
                )

                # Heatmap max overlapping heat (0–1)
                max_heat = 0.0
                for spike in state.heatmap_spikes or []:
                    if spike.start < w_end and spike.end > w_start:
                        heat_val = spike.metadata.get("heat") if spike.metadata else None
                        if heat_val is None:
                            heat_val = spike.score / 10.0 if spike.score else 0.0
                        max_heat = max(max_heat, float(heat_val))
                max_heat = min(1.0, max(0.0, max_heat))

                # Loudness max overlapping score (0–10)
                max_loudness = 0.0
                for spike in state.loudness_spikes or []:
                    if spike.start < w_end and spike.end > w_start:
                        max_loudness = max(max_loudness, float(spike.score))
                max_loudness = min(10.0, max(0.0, max_loudness))

                # Skip windows with no text, no chat, and no signal
                has_text = bool(window_text.strip())
                has_chat = bool(window_chat.strip())
                has_signal = (max_heat > 0.0) or (max_loudness > 0.0) or (chat_rate > 0.0)
                if not (has_text or has_chat or has_signal):
                    w_start += window_step
                    continue

                segments_to_insert.append(
                    (
                        source_id,
                        w_start,
                        w_end,
                        window_text,
                        window_chat,
                        max_heat,
                        max_loudness,
                        chat_rate,
                    )
                )
                w_start += window_step

        now_iso = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            existing = conn.execute(
                "SELECT added_at FROM sources WHERE source_id = ?;", (source_id,)
            ).fetchone()
            added_at = existing["added_at"] if existing else now_iso
            indexed_at = now_iso

            # Replace segments idempotently
            conn.execute("DELETE FROM segments WHERE source_id = ?;", (source_id,))
            if segments_to_insert:
                conn.executemany(
                    """
                    INSERT INTO segments
                    (source_id, start, end, text, chat_text, heat, loudness, chat_rate)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    segments_to_insert,
                )

            language = (
                state.transcript.language
                if (state.transcript and state.transcript.language)
                else "en"
            )
            source = LibrarySource(
                source_id=source_id,
                url=state.stream_info.url,
                title=state.stream_info.title,
                channel=state.stream_info.streamer,
                game=state.stream_info.game,
                kind=str(kind or "stream").strip().lower(),
                language=language,
                duration_seconds=int(duration),
                video_id=state.stream_info.video_id,
                added_at=added_at,
                indexed_at=indexed_at,
                segment_count=len(segments_to_insert),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO sources
                (source_id, url, title, channel, game, kind, language, duration_seconds, video_id, added_at, indexed_at, segment_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    source.source_id,
                    source.url,
                    source.title,
                    source.channel,
                    source.game,
                    source.kind,
                    source.language,
                    source.duration_seconds,
                    source.video_id,
                    source.added_at,
                    source.indexed_at,
                    source.segment_count,
                ),
            )
            conn.commit()

        return source

    def search(
        self,
        query: str,
        *,
        game: str | None = None,
        kinds: list[str] | str | None = None,
        channels: list[str] | str | None = None,
        limit: int = 20,
    ) -> list[SearchHit]:
        """Search segments using FTS5 MATCH, rank hits, and merge overlapping spans."""
        escaped_query = _escape_fts5_query(query)
        if not escaped_query:
            return []

        sql = """
            SELECT
                s.id,
                s.source_id,
                s.start,
                s.end,
                s.text,
                s.chat_text,
                s.heat,
                s.loudness,
                s.chat_rate,
                src.url,
                src.channel,
                src.kind,
                src.game,
                bm25(segments_fts, 1.0, 0.4) AS bm25_val,
                snippet(segments_fts, -1, '', '', '...', 15) AS snippet_text
            FROM segments_fts f
            JOIN segments s ON f.rowid = s.id
            JOIN sources src ON s.source_id = src.source_id
            WHERE segments_fts MATCH ?
        """
        params: list[Any] = [escaped_query]

        if game:
            sql += " AND LOWER(src.game) = LOWER(?)"
            params.append(game)

        if kinds:
            kind_list = [kinds] if isinstance(kinds, str) else list(kinds)
            placeholders = ", ".join("?" for _ in kind_list)
            sql += f" AND LOWER(src.kind) IN ({placeholders})"
            params.extend(k.lower() for k in kind_list)

        if channels:
            chan_list = [channels] if isinstance(channels, str) else list(channels)
            placeholders = ", ".join("?" for _ in chan_list)
            sql += f" AND LOWER(src.channel) IN ({placeholders})"
            params.extend(c.lower() for c in chan_list)

        with self._get_connection() as conn:
            try:
                rows = conn.execute(sql, params).fetchall()
                if not rows and len(query.split()) > 1:
                    # All-terms (AND) found nothing: fall back to any-term (OR);
                    # bm25 still ranks windows matching more terms higher.
                    params[0] = _escape_fts5_query(query, joiner=" OR ")
                    rows = conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError as exc:
                logger.warning("FTS5 query failed for %r: %s", query, exc)
                return []

        if not rows:
            return []

        # SQLite FTS5 bm25 returns negative numbers (more negative = better match)
        # Invert to positive values where higher = better match
        max_bm25 = max((-float(r["bm25_val"]) for r in rows), default=1.0)
        heat_boost = float(self.config.get("library.heat_boost", 0.3))

        raw_hits: list[SearchHit] = []
        for r in rows:
            pos_bm25 = -float(r["bm25_val"])
            norm_bm25 = (pos_bm25 / max_bm25) if max_bm25 > 0 else 1.0
            heat = float(r["heat"])
            loudness = float(r["loudness"])
            chat_rate = float(r["chat_rate"])

            # Small chat boost: up to 0.05
            chat_boost = 0.05 * min(chat_rate / 60.0, 1.0)
            score = norm_bm25 + (heat_boost * heat) + chat_boost

            snippet = str(r["snippet_text"] or "").strip()
            if not snippet:
                snippet = str(r["text"] or r["chat_text"] or "")[:120].strip()
            snippet = snippet.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")

            hit = SearchHit(
                source_id=str(r["source_id"]),
                url=str(r["url"]),
                channel=str(r["channel"]),
                kind=str(r["kind"]),
                start=float(r["start"]),
                end=float(r["end"]),
                snippet=snippet,
                score=round(score, 4),
                bm25=round(norm_bm25, 4),
                heat=round(heat, 4),
                loudness=round(loudness, 4),
                chat_rate=round(chat_rate, 4),
            )
            raw_hits.append(hit)

        # Merge overlapping hits within the same source
        hits_by_source: dict[str, list[SearchHit]] = {}
        for h in raw_hits:
            hits_by_source.setdefault(h.source_id, []).append(h)

        merged_hits: list[SearchHit] = []
        for _, src_hits in hits_by_source.items():
            src_hits.sort(key=lambda x: x.start)
            current = src_hits[0]
            for nxt in src_hits[1:]:
                if nxt.start <= current.end:
                    # Overlapping span
                    current.start = min(current.start, nxt.start)
                    current.end = max(current.end, nxt.end)
                    current.score = max(current.score, nxt.score)
                    current.bm25 = max(current.bm25, nxt.bm25)
                    current.heat = max(current.heat, nxt.heat)
                    current.loudness = max(current.loudness, nxt.loudness)
                    current.chat_rate = max(current.chat_rate, nxt.chat_rate)
                    if nxt.snippet and nxt.snippet not in current.snippet:
                        c_snip = current.snippet.rstrip(". ")
                        n_snip = nxt.snippet.strip(". ")
                        if c_snip and n_snip:
                            current.snippet = f"{c_snip} ... {n_snip}"
                        elif n_snip:
                            current.snippet = n_snip
                        if len(current.snippet) > 300:
                            current.snippet = current.snippet[:297].rsplit(" ", 1)[0] + "..."
                else:
                    merged_hits.append(current)
                    current = nxt
            merged_hits.append(current)

        # Final sort by score descending
        merged_hits.sort(key=lambda x: x.score, reverse=True)
        return merged_hits[:max(0, int(limit))]

    def list_sources(self, game: str | None = None) -> list[LibrarySource]:
        """List indexed sources, optionally filtered by game."""
        with self._get_connection() as conn:
            if game:
                rows = conn.execute(
                    "SELECT * FROM sources WHERE LOWER(game) = LOWER(?) ORDER BY indexed_at DESC;",
                    (game,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM sources ORDER BY indexed_at DESC;"
                ).fetchall()
            return [LibrarySource.from_dict(dict(r)) for r in rows]

    def get_source(self, source_id: str) -> LibrarySource | None:
        """Fetch a single source by its source_id."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM sources WHERE source_id = ?;", (source_id,)
            ).fetchone()
            return LibrarySource.from_dict(dict(row)) if row else None

    def remove_source(self, source_id: str) -> bool:
        """Remove a source and its segments from the library."""
        with self._get_connection() as conn:
            exists = conn.execute(
                "SELECT 1 FROM sources WHERE source_id = ?;", (source_id,)
            ).fetchone()
            if not exists:
                return False
            conn.execute("DELETE FROM segments WHERE source_id = ?;", (source_id,))
            conn.execute("DELETE FROM frame_notes WHERE source_id = ?;", (source_id,))
            conn.execute("DELETE FROM sources WHERE source_id = ?;", (source_id,))
            conn.commit()
            return True
