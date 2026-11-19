"""Narration language resolution (English vs Hinglish) and LLM prompt instructions."""

from __future__ import annotations

from typing import Any

ENGLISH = "english"
HINGLISH = "hinglish"

_ALIASES = {
    "english": ENGLISH,
    "en": ENGLISH,
    "hinglish": HINGLISH,
    "hi": HINGLISH,
    "hindi": HINGLISH,
}
_HINDI_TRANSCRIPT_CODES = {"hi", "ur", "hindi"}


def _normalize(value: Any) -> str | None:
    return _ALIASES.get(str(value or "").strip().lower())


def resolve_narration_language(
    config: Any, streamer: str | None, transcript_language: str | None
) -> str:
    """Return ``"english"`` or ``"hinglish"``.

    Order: per-streamer override -> ``transformation.narration_language`` (when not
    ``auto``) -> Hindi/Urdu transcript language -> english.
    """
    overrides = config.get("transformation.streamer_languages", None) or {}
    if streamer and isinstance(overrides, dict):
        key = str(streamer).strip().lower()
        for name, value in overrides.items():
            if str(name).strip().lower() == key:
                lang = _normalize(value)
                if lang:
                    return lang
    configured = str(config.get("transformation.narration_language", "auto") or "auto")
    lang = _normalize(configured)
    if lang:
        return lang
    code = str(transcript_language or "").strip().lower().replace("_", "-")
    if code and code.split("-")[0] in _HINDI_TRANSCRIPT_CODES:
        return HINGLISH
    return ENGLISH


def language_instruction(lang: str) -> str:
    """Prompt block telling the LLM which language to write spoken lines in."""
    if str(lang or "").strip().lower() != HINGLISH:
        return ""
    return (
        "LANGUAGE: Write every SPOKEN line (narration_intro, narration_outro, callout, "
        "breakdown_commentary) and the hook title in natural Hinglish: Hindi and English "
        "code-mixed, in ROMAN script only. Never use Devanagari. Use short, punchy sentences. "
        "Keep gaming terms and English slang in English. Sound like a desi gaming creator "
        "talking to friends. context_badge stays a short English phrase. "
        "The banned-phrase rules below still apply."
    )
