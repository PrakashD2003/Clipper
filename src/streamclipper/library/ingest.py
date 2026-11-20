"""Ingestion logic for registering and indexing sources into the library."""
import json
import logging
from pathlib import Path
from typing import Any

from streamclipper.cache import StepCache
from streamclipper.config import Config
from streamclipper.download.chat import ChatScraper
from streamclipper.library.index import LibraryIndex
from streamclipper.library.models import LibrarySource
from streamclipper.pipeline import Pipeline, _sanitize_path_segment

logger = logging.getLogger(__name__)


async def add_source(
    config: Config,
    url: str,
    *,
    game: str,
    kind: str = "stream",
    channel: str | None = None,
    force: bool = False,
) -> LibrarySource:
    """Prepare a stream up to analysis, load chat cache if present, and index it into the library."""
    clean_kind = str(kind or "stream").strip().lower()
    pipeline = Pipeline(config)
    state = await pipeline.prepare_stream(
        url,
        game=game,
        streamer=channel,
        force=force,
    )

    # Locate chat cache if available
    temp_dir = Path(config.get("general.temp_dir", "./temp"))
    cache_key = pipeline.cache.get_cache_key(state.stream_info.url)
    raw_video_id = str(state.stream_info.video_id or "").strip()

    chat_df = None
    scraper = ChatScraper()
    candidates: list[Path] = []
    if raw_video_id:
        safe_vid = _sanitize_path_segment(raw_video_id, fallback=cache_key)
        candidates.append(temp_dir / f"{safe_vid}_chat.csv")
    candidates.append(temp_dir / f"{cache_key}_chat.csv")

    for path in candidates:
        if path.is_file():
            chat_df = scraper.load_cache(path)
            if chat_df is not None and not chat_df.empty:
                break

    index = LibraryIndex(config)
    return index.index_state(state, kind=clean_kind, chat_df=chat_df)


def index_cached_states(
    config: Config,
    *,
    game: str | None = None,
) -> list[LibrarySource]:
    """Index every analysed PipelineState already in the cache dir (transcript present)."""
    cache = StepCache(config)
    index = LibraryIndex(config)
    scraper = ChatScraper()
    temp_dir = Path(config.get("general.temp_dir", "./temp"))

    indexed_sources: list[LibrarySource] = []
    cache_dir = Path(config.get("general.cache_dir", "./.cache"))
    if not cache_dir.is_dir():
        return indexed_sources

    for cache_file in sorted(cache_dir.glob("*.json")):
        if cache_file.stem.endswith("_transcript") or cache_file.stem.startswith("scoring_ckpt_"):
            continue
        try:
            with open(cache_file, encoding="utf-8") as f:
                data = json.load(f)
            url = data.get("stream_info", {}).get("url")
            if not url:
                continue

            state = cache.load_state(url)
            if state is None or state.transcript is None:
                continue

            if game:
                if state.stream_info.game and state.stream_info.game.lower() != game.lower():
                    continue
                if not state.stream_info.game:
                    state.stream_info.game = game

            # Check for chat cache
            cache_key = cache.get_cache_key(state.stream_info.url)
            raw_video_id = str(state.stream_info.video_id or "").strip()
            chat_df = None
            candidates: list[Path] = []
            if raw_video_id:
                safe_vid = _sanitize_path_segment(raw_video_id, fallback=cache_key)
                candidates.append(temp_dir / f"{safe_vid}_chat.csv")
            candidates.append(temp_dir / f"{cache_key}_chat.csv")

            for path in candidates:
                if path.is_file():
                    chat_df = scraper.load_cache(path)
                    if chat_df is not None and not chat_df.empty:
                        break

            source = index.index_state(state, kind="stream", chat_df=chat_df)
            indexed_sources.append(source)
        except Exception as exc:
            logger.warning("Error indexing cached state %s: %s", cache_file, exc)
            continue

    return indexed_sources
