"""YouTube live chat replay scraper and CSV cache using yt-dlp (no pytchat)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yt_dlp

from ._ytdlp_opts import apply_ytdlp_auth_opts

logger = logging.getLogger(__name__)

_DEFAULT_RETRIES = 3

CHAT_COLUMNS: list[str] = ["offset_seconds", "author", "message"]


def _empty_chat_df() -> pd.DataFrame:
    """Return an empty chat DataFrame with the canonical schema."""
    return pd.DataFrame(columns=CHAT_COLUMNS)


def _extract_message_text(runs: Any) -> str:
    """Extract readable message text from YouTube live chat message runs."""
    if not isinstance(runs, list):
        return ""

    parts: list[str] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        text = run.get("text")
        if text is not None:
            parts.append(str(text))
            continue

        emoji = run.get("emoji")
        if isinstance(emoji, dict):
            is_custom = bool(emoji.get("isCustomEmoji", False))
            emoji_id = emoji.get("emojiId")
            shortcuts = emoji.get("shortcuts")
            if not is_custom and emoji_id:
                parts.append(str(emoji_id))
            elif isinstance(shortcuts, list) and shortcuts:
                parts.append(str(shortcuts[0]))
            elif emoji_id:
                parts.append(str(emoji_id))

    return "".join(parts)


class ChatScrapeError(RuntimeError):
    """Raised when yt-dlp fails while downloading the live chat replay."""


class ChatScraper:
    """Download YouTube live_chat replay via yt-dlp and parse JSONLines records."""

    def scrape(self, url: str, temp_dir: Path | str) -> pd.DataFrame:
        """Download YouTube live_chat replay via yt-dlp and return a DataFrame.

        Returns an empty DataFrame with columns
        ``["offset_seconds", "author", "message"]`` when the stream has no chat replay.
        """
        if not isinstance(url, str) or not url.strip():
            raise ValueError("Invalid URL: URL must be a non-empty string.")

        clean_url = url.strip()
        out_dir = Path(temp_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        ydl_opts: dict[str, Any] = {
            "skip_download": True,
            "writesubtitles": True,
            "subtitleslangs": ["live_chat"],
            "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "retries": _DEFAULT_RETRIES,
            "fragment_retries": _DEFAULT_RETRIES,
            "extractor_retries": _DEFAULT_RETRIES,
        }
        apply_ytdlp_auth_opts(ydl_opts)

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(clean_url, download=True)
        except (yt_dlp.utils.DownloadError, yt_dlp.utils.ExtractorError, OSError) as exc:
            logger.warning("Failed to download live_chat for %s: %s", clean_url, exc)
            raise ChatScrapeError(f"Failed to download live_chat for {clean_url}: {exc}") from exc

        if not isinstance(info, dict):
            return _empty_chat_df()

        video_id = str(info.get("id") or "stream")
        chat_file = out_dir / f"{video_id}.live_chat.json"
        if not chat_file.is_file():
            logger.info("No live chat replay found for %s (%s)", video_id, chat_file)
            return _empty_chat_df()

        return self.parse_live_chat_file(chat_file)

    def parse_live_chat_file(self, chat_file: Path | str) -> pd.DataFrame:
        """Parse a yt-dlp ``.live_chat.json`` JSONLines file into a chat DataFrame.

        Supports both ``liveChatTextMessageRenderer`` and ``liveChatPaidMessageRenderer``,
        and skips pre-stream entries with negative ``videoOffsetTimeMsec``.
        """
        path = Path(chat_file)
        if not path.is_file():
            return _empty_chat_df()

        rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    if not isinstance(entry, dict):
                        continue

                    replay = entry.get("replayChatItemAction")
                    if not isinstance(replay, dict):
                        continue

                    raw_offset = replay.get("videoOffsetTimeMsec")
                    if raw_offset is None:
                        continue
                    offset_ms = int(raw_offset)
                    # NON-NEGATIVE OFFSET GUARD: skip pre-stream messages
                    if offset_ms < 0:
                        continue

                    actions = replay.get("actions", [])
                    if not isinstance(actions, list):
                        continue

                    for action in actions:
                        try:
                            if not isinstance(action, dict):
                                continue
                            add_action = action.get("addChatItemAction")
                            if not isinstance(add_action, dict):
                                continue
                            item_wrapper = add_action.get("item")
                            if not isinstance(item_wrapper, dict):
                                continue

                            item = (
                                item_wrapper.get("liveChatTextMessageRenderer")
                                or item_wrapper.get("liveChatPaidMessageRenderer")
                            )
                            if not isinstance(item, dict):
                                continue

                            author_obj = item.get("authorName")
                            author = (
                                str(author_obj.get("simpleText", ""))
                                if isinstance(author_obj, dict)
                                else ""
                            )

                            message_obj = item.get("message")
                            runs = (
                                message_obj.get("runs", [])
                                if isinstance(message_obj, dict)
                                else []
                            )
                            text = _extract_message_text(runs)
                            if not text.strip() and "purchaseAmountText" in item:
                                amount_obj = item.get("purchaseAmountText")
                                amount = (
                                    str(amount_obj.get("simpleText", ""))
                                    if isinstance(amount_obj, dict)
                                    else str(amount_obj or "")
                                )
                                text = f"[Super Chat {amount}]".replace("  ", " ")
                            if text.strip():
                                rows.append(
                                    {
                                        "offset_seconds": offset_ms / 1000.0,
                                        "author": author,
                                        "message": text,
                                    }
                                )
                        except (ValueError, TypeError, AttributeError):
                            continue
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue

        return pd.DataFrame(rows, columns=CHAT_COLUMNS)

    def save_cache(self, df: pd.DataFrame, path: Path | str) -> None:
        """Save chat data to CSV with canonical column ordering."""
        cache_path = Path(path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out_df = df.reindex(columns=CHAT_COLUMNS)
        out_df.to_csv(cache_path, index=False)

    def load_cache(self, path: Path | str) -> pd.DataFrame | None:
        """Load cached chat data from CSV, returning None if cache does not exist."""
        cache_path = Path(path)
        if not cache_path.is_file():
            return None

        try:
            df = pd.read_csv(
                cache_path,
                dtype={"author": str, "message": str},
                keep_default_na=False,
            )
        except pd.errors.EmptyDataError:
            return _empty_chat_df()

        if "offset_seconds" in df.columns:
            df["offset_seconds"] = pd.to_numeric(df["offset_seconds"], errors="coerce")
            df = df.dropna(subset=["offset_seconds"])

        return df.reindex(columns=CHAT_COLUMNS).reset_index(drop=True)
