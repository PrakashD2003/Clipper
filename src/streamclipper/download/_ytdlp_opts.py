"""Shared yt-dlp cookie authentication and JavaScript runtime helpers."""
from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
from typing import Any

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def find_cookie_file() -> str | None:
    """Find a Netscape cookies.txt file in the environment or project tree.

    Lookup order:
    1. Environment variable YOUTUBE_COOKIES_FILE or YT_COOKIES_FILE (if file exists).
    2. Path("cookies.txt") (current working directory).
    3. Path("cookies.txt.txt").
    4. Repo root / cookies.txt.
    5. Repo root / cookies.txt.txt.

    Never reads or logs file contents or path. Only logs 'using cookie file' at debug level.
    """
    env_val = os.environ.get("YOUTUBE_COOKIES_FILE") or os.environ.get("YT_COOKIES_FILE")
    if env_val:
        env_path = Path(env_val)
        if env_path.is_file():
            logger.debug("using cookie file")
            return str(env_path.resolve())

    for cand in (
        Path("cookies.txt"),
        Path("cookies.txt.txt"),
        _REPO_ROOT / "cookies.txt",
        _REPO_ROOT / "cookies.txt.txt",
    ):
        if cand.is_file():
            logger.debug("using cookie file")
            return str(cand.resolve())

    return None


def find_js_runtimes() -> dict[str, Any]:
    """Find local Node.js binary for solving YouTube JavaScript challenges.

    Checks shutil.which("node"), falling back to 'C:/Program Files/nodejs/node.exe'
    only if that file exists; returns {} otherwise.
    """
    node_which = shutil.which("node")
    if node_which and Path(node_which).is_file():
        return {"node": {"path": str(Path(node_which).resolve())}}

    fallback = Path("C:/Program Files/nodejs/node.exe")
    if fallback.is_file():
        return {"node": {"path": str(fallback.resolve())}}

    return {}


def apply_ytdlp_auth_opts(ydl_opts: dict[str, Any]) -> dict[str, Any]:
    """Apply cookie authentication and JS runtime options to a yt-dlp options dictionary.

    Sets 'cookiefile' when a cookie file is found, and 'js_runtimes' +
    'remote_components': ['ejs:github'] when a runtime is found.
    Never overwrites keys the caller already set. Returns the same dict.
    """
    cookie_file = find_cookie_file()
    if cookie_file and "cookiefile" not in ydl_opts:
        ydl_opts["cookiefile"] = cookie_file

    js_rts = find_js_runtimes()
    if js_rts:
        if "js_runtimes" not in ydl_opts:
            ydl_opts["js_runtimes"] = js_rts
        if "remote_components" not in ydl_opts:
            ydl_opts["remote_components"] = ["ejs:github"]

    return ydl_opts
