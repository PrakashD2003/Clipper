"""Standalone highlight scorer backed by a local Ollama LLM."""
from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import httpx
import ollama

from ..config import Config, VALID_TRANSFORMATION_MODES
from ..language import language_instruction
from ..models import ClipMetadata, Highlight, TimeRange
from .scorer import Scorer, TranscriptChunk

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

VALID_MOMENT_TYPES: frozenset[str] = frozenset(
    {
        "funny",
        "fail",
        "graphics_detail",
        "hype",
        "clutch",
        "chat_chaos",
        "lore_secret",
        "analysis",
        "reaction",
    }
)

# D4: generic phrases the narrative LLM must never produce (lower-case substring match).
BANNED_PHRASES: tuple[str, ...] = (
    "the chat went wild",
    "chat went completely wild",
    "you won't believe",
    "watch what happens",
    "subscribe for more",
    "over a million people",
    "insane moment",
    "most viral",
)

def grounded_fallback_intro(
    highlight: Highlight, streamer: str = "", game: str = ""
) -> str:
    """Short intro voiceover built only from the highlight's own title/quote/reason.

    Used when the narrative writer (Ollama pass 2) is unavailable or returns nothing, so a
    non-raw clip never opens without a voiceover. Parts containing a banned generic phrase
    are dropped.
    """
    def _clean(text: Any, limit: int) -> str:
        t = " ".join(_as_str(text).split())
        if not t:
            return ""
        if any(p in t.lower() for p in BANNED_PHRASES):
            return ""
        if len(t) > limit:
            t = t[:limit].rsplit(" ", 1)[0].rstrip(",;:- ")
        return t.rstrip(".!? ")

    title = _clean(highlight.title, 80)
    reason = _clean(_as_str(highlight.reason).split(". ")[0], 140)
    quote = _clean(highlight.best_quote, 90)
    parts: list[str] = []
    if title:
        parts.append(title)
    if quote:
        parts.append(f'{streamer or "The streamer"} says "{quote}"')
    elif reason:
        parts.append(reason)
    if not parts:
        parts.append(f"A moment from {streamer + chr(39) + 's' if streamer else 'this'} {game or 'stream'} stream")
    return ". ".join(parts) + "."


_THINK_TAG_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$",
    flags=re.DOTALL | re.IGNORECASE,
)


class OllamaConnectionError(RuntimeError):
    """Raised when Ollama is unreachable or the configured model is not available."""


class OllamaTimeoutError(OllamaConnectionError):
    """Raised when an Ollama request times out (the server is reachable but slow)."""


def _is_timeout(exc: BaseException) -> bool:
    """Return True for read/request timeouts (not connection refused or 404)."""
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        return True
    return "timed out" in str(exc).lower()


def _as_str(value: Any, default: str = "") -> str:
    """Convert value to a stripped string, mapping None cleanly to default."""
    if value is None:
        return default
    return str(value).strip()


def _parse_seconds(value: Any) -> float | None:
    """Parse a non-negative finite timestamp in seconds or MM:SS / HH:MM:SS format."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        val = float(value)
        return val if math.isfinite(val) and val >= 0.0 else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        numeric_text = text[:-1].strip() if text.endswith(("s", "S")) else text
        try:
            val = float(numeric_text)
            return val if math.isfinite(val) and val >= 0.0 else None
        except ValueError:
            pass
        if ":" in text:
            parts = text.split(":")
            try:
                nums = [float(p) for p in parts]
            except ValueError:
                return None
            if not all(math.isfinite(n) and n >= 0.0 for n in nums):
                return None
            if len(nums) == 2:
                mins, secs = nums
                if secs >= 60.0:
                    return None
                return mins * 60.0 + secs
            if len(nums) == 3:
                hrs, mins, secs = nums
                if mins >= 60.0 or secs >= 60.0:
                    return None
                return hrs * 3600.0 + mins * 60.0 + secs
    return None


def _parse_score(value: Any) -> float:
    """Parse a finite score value and clamp to [0.0, 10.0]."""
    if value is None or isinstance(value, bool):
        return 0.0
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(raw):
        return 0.0
    return max(0.0, min(10.0, raw))


def _format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS for readable LLM prompt context."""
    total = max(0, int(seconds))
    hrs, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def _clean_llm_text(raw_text: str) -> str:
    """Strip reasoning <think> blocks and optional Markdown code fences."""
    cleaned = _THINK_TAG_RE.sub("", raw_text).strip()
    fence_match = _CODE_FENCE_RE.match(cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    return cleaned


def _extract_response_text(response: Any) -> str:
    """Extract text from an ollama GenerateResponse, dict, or raw string."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        val = response.get("response")
        return "" if val is None else str(val)
    text = getattr(response, "response", None)
    if text is not None:
        return str(text)
    if hasattr(response, "get"):
        val = response.get("response")
        return "" if val is None else str(val)
    return str(response)


def _extract_model_names(list_response: Any) -> list[str]:
    """Extract installed model names from an ollama ListResponse or dict mock."""
    models = (
        list_response.get("models", [])
        if isinstance(list_response, dict)
        else getattr(list_response, "models", [])
    )
    names: list[str] = []
    for entry in models or []:
        if isinstance(entry, str):
            names.append(entry)
        elif isinstance(entry, dict):
            name = entry.get("model") or entry.get("name")
            if name:
                names.append(str(name))
        else:
            name = getattr(entry, "model", None) or getattr(entry, "name", None)
            if name:
                names.append(str(name))
    return names


def _model_matches(target: str, installed: list[str]) -> bool:
    """Check whether target model name matches any installed Ollama model."""
    if target in installed:
        return True
    if ":" not in target and f"{target}:latest" in installed:
        return True
    if target.endswith(":latest") and target[: -len(":latest")] in installed:
        return True
    return False


class OllamaScorer(Scorer):
    """Highlight scorer and metadata generator using a local Ollama server."""

    MAX_JSON_RETRIES = 3

    def __init__(self, config: Config):
        super().__init__(config)
        self.model: str = _as_str(config.get("llm.model", "qwen3:4b"), "qwen3:4b") or "qwen3:4b"
        self.base_url: str = (
            _as_str(
                config.get("llm.base_url", "http://localhost:11434"),
                "http://localhost:11434",
            )
            or "http://localhost:11434"
        )
        self.temperature: float = float(config.get("llm.temperature", 0.3))
        self.timeout: float = float(config.get("llm.timeout_seconds", 120))
        provider = _as_str(config.get("llm.provider", "ollama"), "ollama").lower()
        if provider != "ollama":
            logger.warning(
                "llm.provider is %r but OllamaScorer only supports 'ollama'; using Ollama anyway.",
                provider,
            )
        self.client = ollama.Client(host=self.base_url, timeout=self.timeout)
        self._load_prompts()

    def _load_prompts(self) -> None:
        """Load prompt templates from src/streamclipper/prompts/."""
        self.highlight_prompt = (PROMPTS_DIR / "highlight_scorer.txt").read_text(
            encoding="utf-8"
        )
        self.narrative_prompt = (PROMPTS_DIR / "narrative_writer.txt").read_text(
            encoding="utf-8"
        )
        self.hook_prompt = (PROMPTS_DIR / "hook_generator.txt").read_text(
            encoding="utf-8"
        )
        self.caption_prompt = (PROMPTS_DIR / "caption_writer.txt").read_text(
            encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Ollama client helpers
    # ------------------------------------------------------------------

    def _call_generate(self, prompt: str, *, json_format: bool = True) -> str:
        """Invoke Ollama generate API (disabling slow Qwen3 thinking tokens) and translate errors."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "options": {"temperature": self.temperature},
            "think": False,
        }
        if json_format:
            kwargs["format"] = "json"

        try:
            response = self.client.generate(**kwargs)
        except TypeError as exc:
            if "think" in str(exc).lower():
                fallback_kwargs = {k: v for k, v in kwargs.items() if k != "think"}
                try:
                    response = self.client.generate(**fallback_kwargs)
                    return _extract_response_text(response)
                except Exception as retry_exc:
                    if _is_timeout(retry_exc):
                        raise OllamaTimeoutError(
                            f"Ollama request timed out for model '{self.model}': {retry_exc}"
                        ) from retry_exc
                    raise OllamaConnectionError(
                        f"Cannot connect to Ollama at {self.base_url} for model '{self.model}': "
                        f"{retry_exc}. Ensure Ollama is running (`ollama serve`) and run "
                        f"`ollama pull {self.model}`."
                    ) from retry_exc
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self.base_url} for model '{self.model}': "
                f"{exc}. Ensure Ollama is running (`ollama serve`) and run "
                f"`ollama pull {self.model}`."
            ) from exc
        except ollama.ResponseError as exc:
            status = getattr(exc, "status_code", None)
            detail = str(exc)
            # Some older Ollama servers/models reject the `think` parameter; retry once without it.
            if status == 400 and "think" in detail.lower():
                fallback_kwargs = {k: v for k, v in kwargs.items() if k != "think"}
                try:
                    response = self.client.generate(**fallback_kwargs)
                    return _extract_response_text(response)
                except ollama.ResponseError as retry_exc:
                    exc = retry_exc
                    status = getattr(retry_exc, "status_code", None)
                    detail = str(retry_exc)
                except Exception as retry_exc:
                    if _is_timeout(retry_exc):
                        raise OllamaTimeoutError(
                            f"Ollama request timed out for model '{self.model}': {retry_exc}"
                        ) from retry_exc
                    raise OllamaConnectionError(
                        f"Cannot connect to Ollama at {self.base_url} for model '{self.model}': "
                        f"{retry_exc}. Ensure Ollama is running (`ollama serve`) and run "
                        f"`ollama pull {self.model}`."
                    ) from retry_exc

            if status == 404 or "not found" in detail.lower():
                raise OllamaConnectionError(
                    f"Ollama model '{self.model}' is not installed at {self.base_url}. "
                    f"Run `ollama pull {self.model}` to download it."
                ) from exc
            raise OllamaConnectionError(
                f"Ollama request failed for model '{self.model}' at {self.base_url}: "
                f"{detail}. Ensure Ollama is running (`ollama serve`) and the model is "
                f"pulled (`ollama pull {self.model}`)."
            ) from exc
        except Exception as exc:
            if _is_timeout(exc):
                raise OllamaTimeoutError(
                    f"Ollama request timed out after {self.timeout}s for model '{self.model}': {exc}"
                ) from exc
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self.base_url} for model '{self.model}': "
                f"{exc}. Ensure Ollama is running (`ollama serve`) and run "
                f"`ollama pull {self.model}`."
            ) from exc

        return _extract_response_text(response)

    def check_connection(self) -> bool:
        """Verify Ollama is running and the resolved model is available."""
        try:
            response = self.client.list()
        except Exception as exc:
            raise OllamaConnectionError(
                f"Cannot reach Ollama server at {self.base_url} (required model: "
                f"'{self.model}'). Start Ollama with `ollama serve` and ensure the "
                f"model is installed with `ollama pull {self.model}`. Error: {exc}"
            ) from exc

        available = _extract_model_names(response)
        if not _model_matches(self.model, available):
            installed_hint = (
                f" Installed models: {', '.join(available)}." if available else ""
            )
            raise OllamaConnectionError(
                f"Ollama model '{self.model}' is not available at {self.base_url}."
                f"{installed_hint} Pull it with: ollama pull {self.model}"
            )
        return True

    # ------------------------------------------------------------------
    # Prompt formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_loudness_data(loudness: list[TimeRange]) -> str:
        """Format overlapping loudness spikes for the highlight prompt."""
        if not loudness:
            return "None detected in this window."
        lines: list[str] = []
        for spike in loudness:
            peak = _parse_seconds(spike.metadata.get("peak_time"))
            if peak is None:
                peak = (spike.start + spike.end) / 2.0
            lines.append(
                f"- {spike.start:.1f}s to {spike.end:.1f}s "
                f"(score: {spike.score:.1f}/10, peak: {peak:.1f}s)"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_chat_data(chat: list[TimeRange]) -> str:
        """Format overlapping chat velocity spikes for the highlight prompt."""
        if not chat:
            return "None detected in this window."
        lines: list[str] = []
        for spike in chat:
            samples = spike.metadata.get("sample_messages") or []
            snippet = ""
            if samples:
                texts = [
                    _as_str(m.get("message") if m.get("message") is not None else m.get("text"))
                    for m in samples[:3]
                    if isinstance(m, dict)
                ]
                texts = [t for t in texts if t]
                if texts:
                    snippet = f" — sample chat: {', '.join(repr(t) for t in texts)}"
            lines.append(
                f"- {spike.start:.1f}s to {spike.end:.1f}s "
                f"(score: {spike.score:.1f}/10){snippet}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Scoring & response parsing
    # ------------------------------------------------------------------

    def _score_chunk(
        self,
        chunk: TranscriptChunk,
        loudness: list[TimeRange],
        chat: list[TimeRange],
    ) -> list[Highlight]:
        """Score a transcript chunk using Ollama with up to 3 JSON parse retries."""
        base_prompt = self.highlight_prompt.format(
            chunk_minutes=self.chunk_minutes,
            start_time=f"{_format_timestamp(chunk.start)} ({chunk.start:.1f}s)",
            end_time=f"{_format_timestamp(chunk.end)} ({chunk.end:.1f}s)",
            transcript_text=chunk.text,
            loudness_data=self._format_loudness_data(loudness),
            chat_data=self._format_chat_data(chat),
        )

        last_error: Exception | None = None
        for attempt in range(1, self.MAX_JSON_RETRIES + 1):
            prompt = base_prompt
            if attempt > 1:
                prompt = (
                    f"{base_prompt}\n\n"
                    "IMPORTANT: Return ONLY valid JSON (a JSON array `[...]` or "
                    '`{"highlights": [...]}`) with no extra commentary.'
                )

            raw_text = ""
            for timeout_try in (1, 2):
                try:
                    raw_text = self._call_generate(prompt, json_format=True)
                    break
                except OllamaTimeoutError as exc:
                    if timeout_try == 1:
                        logger.warning(
                            "Ollama timed out on chunk [%.1f, %.1f]; retrying once: %s",
                            chunk.start, chunk.end, exc,
                        )
                        continue
                    logger.warning(
                        "Ollama timed out twice on chunk [%.1f, %.1f]; skipping chunk: %s",
                        chunk.start, chunk.end, exc,
                    )
                    return []
            try:
                return self._parse_response(raw_text, chunk_offset=chunk.start)
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "Malformed JSON from Ollama model '%s' on chunk [%.1f, %.1f] "
                    "(attempt %d/%d): %s",
                    self.model,
                    chunk.start,
                    chunk.end,
                    attempt,
                    self.MAX_JSON_RETRIES,
                    exc,
                )

        logger.error(
            "Failed to parse Ollama response for chunk [%.1f, %.1f] after %d attempts: %s",
            chunk.start,
            chunk.end,
            self.MAX_JSON_RETRIES,
            last_error,
        )
        return []

    def _parse_response(
        self, response_text: str, chunk_offset: float = 0.0
    ) -> list[Highlight]:
        """Parse LLM JSON response (list or wrapped object) into Highlight objects."""
        cleaned = _clean_llm_text(response_text)
        data = json.loads(cleaned)

        if isinstance(data, list):
            raw_items = data
        elif isinstance(data, dict):
            if "highlights" in data and isinstance(data["highlights"], list):
                raw_items = data["highlights"]
            elif "clips" in data and isinstance(data["clips"], list):
                raw_items = data["clips"]
            elif "start" in data and "end" in data:
                raw_items = [data]
            elif not data:
                raw_items = []
            else:
                raise ValueError(
                    "JSON object response must contain a 'highlights' list or highlight fields."
                )
        else:
            raise ValueError(
                f"Expected JSON list or object from Ollama, got {type(data).__name__}."
            )

        chunk_duration = float(self.chunk_minutes) * 60.0

        highlights: list[Highlight] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue

            start = _parse_seconds(item.get("start"))
            end = _parse_seconds(item.get("end"))
            if start is None or end is None or end <= start:
                continue

            # If the whole range lies before chunk_offset it is clearly chunk-relative;
            # shift to absolute stream time. Otherwise treat it as already absolute.
            applied_offset = 0.0
            if chunk_offset > 0.0 and end < chunk_offset:
                applied_offset = chunk_offset
                start += applied_offset
                end += applied_offset

            score = _parse_score(item.get("score"))

            moment_type = _as_str(item.get("moment_type"), "hype").lower()
            if moment_type not in VALID_MOMENT_TYPES:
                moment_type = "hype"
            callout = _as_str(item.get("callout"))

            sub_clips: list[dict[str, Any]] = []
            raw_sub_clips = item.get("sub_clips")
            if isinstance(raw_sub_clips, list):
                for beat in raw_sub_clips:
                    if not isinstance(beat, dict):
                        sub_clips = []
                        break
                    b_start = _parse_seconds(beat.get("start"))
                    b_end = _parse_seconds(beat.get("end"))
                    if b_start is None or b_end is None or b_end <= b_start:
                        sub_clips = []
                        break
                    if applied_offset > 0.0 and b_end < chunk_offset:
                        b_start += applied_offset
                        b_end += applied_offset
                    sub_clips.append(
                        {
                            "start": b_start,
                            "end": b_end,
                            "label": _as_str(beat.get("label")),
                        }
                    )

            title = _as_str(item.get("title")) or "Stream Highlight"
            reason = _as_str(item.get("reason"))
            best_quote = _as_str(item.get("best_quote"))
            narration_intro = _as_str(item.get("narration_intro"))
            narration_outro = _as_str(item.get("narration_outro"))
            context_badge = _as_str(item.get("context_badge"))
            breakdown_commentary = _as_str(item.get("breakdown_commentary"))
            breakdown_callout = _as_str(item.get("breakdown_callout"))

            highlights.append(
                Highlight(
                    start=start,
                    end=end,
                    score=score,
                    title=title,
                    reason=reason,
                    best_quote=best_quote,
                    moment_type=moment_type,
                    narration_intro=narration_intro,
                    narration_outro=narration_outro,
                    context_badge=context_badge,
                    sub_clips=sub_clips,
                    breakdown_commentary=breakdown_commentary,
                    breakdown_callout=breakdown_callout,
                    callout=callout,
                )
            )

        return highlights

    # ------------------------------------------------------------------
    # Narrative & metadata generation
    # ------------------------------------------------------------------

    @staticmethod
    def _find_banned_phrase(texts: list[str], avoid_phrases: list[str] | None) -> str | None:
        """Return the first banned/avoided phrase found in *texts* (case-insensitive)."""
        haystack = " ".join(t.lower() for t in texts if t)
        for phrase in list(BANNED_PHRASES) + [
            p.strip().lower() for p in (avoid_phrases or []) if p and p.strip()
        ]:
            if phrase in haystack:
                return phrase
        return None

    def generate_narrative(
        self,
        highlight: Highlight,
        streamer: str,
        game: str,
        *,
        hook_style: str | None = None,
        avoid_phrases: list[str] | None = None,
        language: str = "english",
    ) -> tuple[str, str, str]:
        """Backfill (narration_intro, narration_outro, context_badge) via narrative_writer.txt.

        Also fills ``highlight.callout`` when the model returns one. Output containing a banned
        generic phrase (or any of *avoid_phrases*) is rejected and regenerated (max 2 retries).
        """
        chat_lines = []
        for msg in highlight.chat_burst or []:
            text = _as_str(msg.get("text") or msg.get("message")) if isinstance(msg, dict) else ""
            if text:
                chat_lines.append(f"- {text}")
        base_prompt = self.narrative_prompt.format(
            streamer=streamer or "the streamer",
            game=game or "the game",
            moment_type=_as_str(highlight.moment_type, "hype") or "hype",
            hook_title=_as_str(highlight.title),
            description=_as_str(highlight.reason),
            quote=_as_str(highlight.best_quote),
            narration_style=_as_str(
                self.config.get("transformation.narration_style", "adaptive"), "adaptive"
            )
            or "adaptive",
            persona=_as_str(
                self.config.get(
                    "transformation.persona",
                    "hyped, witty gaming commentator who reacts like a friend watching live",
                )
            ),
            transcript_excerpt=_as_str(highlight.transcript_chunk)[:1500] or "(none)",
            chat_samples="\n".join(chat_lines[:5]) or "(none)",
            avoid_phrases=", ".join(f'"{p}"' for p in (avoid_phrases or [])) or "(none)",
            hook_style=_as_str(hook_style) or "any",
            language_instruction=language_instruction(language),
        )
        existing_intro = _as_str(highlight.narration_intro)
        existing_outro = _as_str(highlight.narration_outro)
        existing_badge = _as_str(highlight.context_badge)

        banned_retries = 0
        max_banned_retries = 2
        banned_note = ""
        result: tuple[str, str, str] | None = None
        json_failures = 0
        for attempt in range(1, self.MAX_JSON_RETRIES + max_banned_retries + 1):
            prompt = base_prompt + banned_note
            if json_failures and not banned_note:
                prompt = (
                    f"{base_prompt}\n\n"
                    'IMPORTANT: Return ONLY a valid JSON object with keys '
                    '"narration_intro", "narration_outro", "callout", and "context_badge".'
                )

            raw_text = self._call_generate(prompt, json_format=True)
            try:
                data = json.loads(_clean_llm_text(raw_text))
                if not isinstance(data, dict):
                    raise ValueError("Expected JSON object for narrative response.")
                intro = _as_str(data.get("narration_intro")) or existing_intro
                outro = _as_str(data.get("narration_outro")) or existing_outro
                badge = _as_str(data.get("context_badge")) or existing_badge
                callout = _as_str(data.get("callout"))

                bad = self._find_banned_phrase(
                    [_as_str(data.get("narration_intro")), _as_str(data.get("narration_outro")), callout],
                    avoid_phrases,
                )
                if bad and banned_retries < max_banned_retries:
                    banned_retries += 1
                    logger.warning("Narrative contained banned phrase %r; regenerating.", bad)
                    banned_note = (
                        f'\n\nREJECTED: your previous answer used the banned phrase "{bad}". '
                        "Rewrite every line without it, using concrete details from the excerpt."
                    )
                    result = (intro, outro, badge)
                    continue
                if bad:
                    logger.warning("Narrative still contains banned phrase %r after retries.", bad)

                if callout and not highlight.callout:
                    highlight.callout = callout
                if not highlight.breakdown_commentary and data.get("breakdown_commentary"):
                    highlight.breakdown_commentary = _as_str(data.get("breakdown_commentary"))
                if not highlight.breakdown_callout and data.get("breakdown_callout"):
                    highlight.breakdown_callout = _as_str(data.get("breakdown_callout"))
                return intro, outro, badge
            except (json.JSONDecodeError, ValueError) as exc:
                json_failures += 1
                logger.warning(
                    "Malformed JSON from generate_narrative (attempt %d/%d): %s",
                    json_failures,
                    self.MAX_JSON_RETRIES,
                    exc,
                )
                banned_note = ""
                if json_failures >= self.MAX_JSON_RETRIES:
                    break

        if result is not None:
            return result
        return existing_intro, existing_outro, existing_badge

    def _generate_hook_title(
        self,
        highlight: Highlight,
        streamer: str,
        game: str,
        hook_style: str | None = None,
        language: str = "english",
    ) -> str:
        """Generate a short hook title using hook_generator.txt."""
        prompt = self.hook_prompt.format(
            streamer=streamer or "the streamer",
            game=game or "the game",
            description=_as_str(highlight.reason),
            quote=_as_str(highlight.best_quote),
            hook_style=_as_str(hook_style) or "any",
            language_instruction=language_instruction(language),
        )
        raw_text = self._call_generate(prompt, json_format=False)
        cleaned = _clean_llm_text(raw_text)
        for line in cleaned.splitlines():
            candidate = line.strip().strip("\"'")
            if candidate:
                return candidate
        return _as_str(highlight.title) or "Stream Highlight"

    def _finalize_hashtags(
        self, raw_tags: list[Any], streamer: str, game: str
    ) -> list[str]:
        """Normalize, deduplicate, enforce required game/streamer tags, and cap to hashtag_count."""
        max_tags = max(0, int(self.config.get("metadata.hashtag_count", 15)))
        if max_tags == 0:
            return []

        hashtags: list[str] = []
        seen_lower: set[str] = set()

        for tag in raw_tags:
            cleaned = _as_str(tag).lstrip("#").strip()
            if cleaned and cleaned.lower() not in seen_lower:
                seen_lower.add(cleaned.lower())
                hashtags.append(cleaned)

        required_tags: list[str] = []
        if self.config.get("metadata.include_game_tag", True) and game:
            game_tag = re.sub(r"\W+", "", game)
            if game_tag:
                required_tags.append(game_tag)

        if self.config.get("metadata.include_streamer_tag", True) and streamer:
            streamer_tag = re.sub(r"\W+", "", streamer)
            if streamer_tag:
                required_tags.append(streamer_tag)

        for req in required_tags:
            if req.lower() not in seen_lower:
                seen_lower.add(req.lower())
                hashtags.append(req)

        if len(hashtags) <= max_tags:
            return hashtags

        # Cap to max_tags while preserving required game/streamer tags
        required_lower = {r.lower() for r in required_tags}
        kept_required = [t for t in hashtags if t.lower() in required_lower][:max_tags]
        remaining_slots = max_tags - len(kept_required)
        kept_regular = [t for t in hashtags if t.lower() not in required_lower][:remaining_slots]
        # Preserve relative order of kept tags
        keep_set = {t.lower() for t in kept_required} | {t.lower() for t in kept_regular}
        return [t for t in hashtags if t.lower() in keep_set][:max_tags]

    def _generate_caption_and_hashtags(
        self,
        highlight: Highlight,
        streamer: str,
        game: str,
        hook_title: str,
    ) -> tuple[str, list[str]]:
        """Generate social media caption and hashtags using caption_writer.txt."""
        hashtag_count = int(self.config.get("metadata.hashtag_count", 15))
        base_prompt = self.caption_prompt.format(
            streamer=streamer or "the streamer",
            game=game or "the game",
            hook_title=hook_title,
            description=_as_str(highlight.reason),
            hashtag_count=hashtag_count,
        )

        for attempt in range(1, self.MAX_JSON_RETRIES + 1):
            prompt = base_prompt
            if attempt > 1:
                prompt = (
                    f"{base_prompt}\n\n"
                    'IMPORTANT: Return ONLY a valid JSON object with keys '
                    '"caption" and "hashtags".'
                )

            raw_text = self._call_generate(prompt, json_format=True)
            try:
                data = json.loads(_clean_llm_text(raw_text))
                if not isinstance(data, dict):
                    raise ValueError("Expected JSON object for caption response.")
                caption = _as_str(data.get("caption"))
                raw_tags = data.get("hashtags")
                tag_list = raw_tags if isinstance(raw_tags, list) else []
                hashtags = self._finalize_hashtags(tag_list, streamer, game)
                if not caption:
                    caption = _as_str(highlight.reason) or hook_title
                return caption, hashtags
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning(
                    "Malformed JSON from caption generator (attempt %d/%d): %s",
                    attempt,
                    self.MAX_JSON_RETRIES,
                    exc,
                )

        # Graceful fallback if caption LLM returns malformed JSON across all retries
        fallback_tags = self._finalize_hashtags([], streamer, game)
        return _as_str(highlight.reason) or hook_title, fallback_tags

    def generate_metadata(
        self,
        highlight: Highlight,
        streamer: str,
        game: str,
        source_url: str,
        mode: str | None = None,
        hook_style: str | None = None,
        avoid_phrases: list[str] | None = None,
        language: str = "english",
    ) -> ClipMetadata:
        """Generate hook title, narrative fields, callout, caption, and hashtags for a highlight."""
        hook_title = self._generate_hook_title(
            highlight, streamer, game, hook_style, language=language
        )

        if mode is not None and str(mode).strip():
            requested_mode = str(mode).strip().lower()
            if requested_mode not in VALID_TRANSFORMATION_MODES:
                raise ValueError(
                    f"Invalid transformation mode {mode!r}; expected one of "
                    f"{sorted(VALID_TRANSFORMATION_MODES)}"
                )
        else:
            transformation_enabled = bool(self.config.get("transformation.enabled", True))
            configured_mode = (
                _as_str(self.config.get("transformation.mode", "auto"), "auto").lower()
                or "auto"
            )
            requested_mode = configured_mode if transformation_enabled else "raw"

        if requested_mode == "auto":
            valid_beats = Scorer._validate_sub_clips(
                highlight.sub_clips,
                highlight.start,
                highlight.end,
                max_duration=self.max_clip_seconds,
            )
            transformation_mode = "story" if len(valid_beats) >= 2 else "narrative"
        else:
            transformation_mode = requested_mode

        if transformation_mode == "raw":
            narration_intro = ""
            narration_outro = ""
            context_badge = ""
            breakdown_commentary = ""
            breakdown_callout = ""
        else:
            narration_intro = _as_str(highlight.narration_intro)
            narration_outro = _as_str(highlight.narration_outro)
            context_badge = _as_str(highlight.context_badge)
            breakdown_commentary = _as_str(highlight.breakdown_commentary)
            breakdown_callout = _as_str(highlight.breakdown_callout)

            need_narrative = not (narration_intro and narration_outro and context_badge)
            if transformation_mode == "breakdown" and not breakdown_commentary:
                need_narrative = True

            if need_narrative:
                gen_intro, gen_outro, gen_badge = self.generate_narrative(
                    highlight,
                    streamer,
                    game,
                    hook_style=hook_style,
                    avoid_phrases=avoid_phrases,
                    language=language,
                )
                narration_intro = narration_intro or gen_intro
                narration_outro = narration_outro or gen_outro
                context_badge = context_badge or gen_badge
                breakdown_commentary = breakdown_commentary or _as_str(highlight.breakdown_commentary)
                breakdown_callout = breakdown_callout or _as_str(highlight.breakdown_callout)

            if transformation_mode == "breakdown" and not breakdown_commentary:
                breakdown_commentary = f"Take a closer look at that climax moment: {_as_str(highlight.reason)}"
                breakdown_callout = breakdown_callout or "Instant Replay"

        caption, hashtags = self._generate_caption_and_hashtags(
            highlight, streamer, game, hook_title
        )

        score = highlight.final_score if highlight.final_score > 0.0 else highlight.score

        return ClipMetadata(
            hook_title=hook_title,
            caption=caption,
            hashtags=hashtags,
            streamer=streamer,
            game=game,
            source_url=source_url,
            clip_start=highlight.start,
            clip_end=highlight.end,
            highlight_score=score,
            moment_type=_as_str(highlight.moment_type, "hype") or "hype",
            narration_intro=narration_intro,
            narration_outro=narration_outro,
            context_badge=context_badge,
            transformation_mode=transformation_mode,
            narration_language=str(language or "english"),
            breakdown_commentary=breakdown_commentary,
            breakdown_callout=breakdown_callout,
            music_risk=highlight.music_risk,
            music_confidence=highlight.music_confidence,
            callout=(
                _as_str(highlight.callout)
                if transformation_mode != "raw"
                and bool(self.config.get("transformation.callout_enabled", True))
                else ""
            ),
        )

    def generate_countdown_lines(
        self, highlights: list[Highlight], streamer: str, game: str
    ) -> list[str]:
        """Return one short ranking line per highlight, ordered #3 -> #1 (countdown order)."""
        n = len(highlights)
        if n == 0:
            return []
        items = "\n".join(
            f"{i + 1}. {_as_str(h.title) or 'Moment'} - {_as_str(h.reason)}"
            for i, h in enumerate(highlights)
        )
        prompt = (
            f"Write a top-{n} countdown for gaming Shorts by {streamer or 'the streamer'} "
            f"playing {game or 'the game'}. Items, listed in countdown order (#{n} first, #1 last):\n"
            f"{items}\n\n"
            "For each item write ONE short spoken ranking line (max 10 words) that names a concrete "
            'detail from the item. No generic hype. Return strict JSON: {"lines": ["...", ...]} '
            f"with exactly {n} lines in the same order."
        )
        fallback = [f"Number {n - i}: {_as_str(h.title) or 'this moment'}" for i, h in enumerate(highlights)]
        for _ in range(self.MAX_JSON_RETRIES):
            raw_text = self._call_generate(prompt, json_format=True)
            try:
                data = json.loads(_clean_llm_text(raw_text))
                lines = data.get("lines") if isinstance(data, dict) else data
                if not isinstance(lines, list):
                    raise ValueError("Expected a list of countdown lines.")
                cleaned = [_as_str(x) for x in lines]
                if len(cleaned) >= n and all(cleaned[:n]):
                    return cleaned[:n]
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning("Malformed countdown JSON: %s", exc)
        return fallback
