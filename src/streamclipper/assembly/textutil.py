"""Text helpers shared by burned-in overlays (hook banner, badges, ASS captions)."""

from __future__ import annotations

import re

# Emoji / pictographs, dingbats, variation selectors and joiners. Burned-in fonts (Arial,
# Segoe UI, libass defaults) have no colour-emoji glyphs, so these render as "tofu" boxes.
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"  # emoji, pictographs, flags, symbols & pictographs ext.
    "\U00002600-\U000027BF"  # misc symbols + dingbats
    "\U00002B00-\U00002BFF"  # arrows / stars often used as emoji
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"  # zero-width joiner
    "\U000020E3"  # combining enclosing keycap
    "]+"
)
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def strip_unrenderable(text: str | None) -> str:
    """Remove emoji the overlay fonts cannot draw and tidy the leftover whitespace."""
    if not text:
        return ""
    cleaned = _EMOJI_RE.sub(" ", str(text))
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned)
    return "\n".join(line.strip() for line in cleaned.split("\n")).strip()
