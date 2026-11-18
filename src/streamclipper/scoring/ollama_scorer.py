"""Standalone highlight scorer backed by a local Ollama LLM."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import ollama

from ..config import Config
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
    }
)

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$",
    flags=re.DOTALL | re.IGNORECASE,
)


class OllamaConnectionError(RuntimeError):
    """Raised when Ollama is unreachable or the configured model is not available."""


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
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return str(response.get("response", ""))
    text = getattr(response, "response", None)
    if text is not None:
        return str(text)
    if hasattr(response, "get"):
        return str(response.get("response", ""))
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
        self.model: str = str(config.get("llm.model", "qwen3:4b"))
        self.base_url: str = str(config.get("llm.base_url", "http://localhost:11434"))
        self.temperature: float = float(config.get("llm.temperature", 0.3))
        self.timeout: float = float(config.get("llm.timeout_seconds", 120))
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
        """Invoke Ollama generate API and translate connection/model errors."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "options": {"temperature": self.temperature},
        }
        if json_format:
            kwargs["format"] = "json"

        try:
            response = self.client.generate(**kwargs)
        except ollama.ResponseError as exc:
            status = getattr(exc, "status_code", None)
            detail = str(exc)
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
            peak = spike.metadata.get("peak_time", (spike.start + spike.end) / 2.0)
            lines.append(
                f"- {spike.start:.1f}s to {spike.end:.1f}s "
                f"(score: {spike.score:.1f}/10, peak: {float(peak):.1f}s)"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_chat_data(chat: list[TimeRange]) -> str:
        """Format overlapping chat velocity spikes for the highlight prompt."""
        if not chat:
            return "None detected in this window."
        lines: list[str] = []
        for spike in chat:
            samples = spike.metadata.get("sample_messages", [])
            snippet = ""
            if samples:
                texts = [
                    str(m.get("message", m.get("text", ""))).strip()
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

            raw_text = self._call_generate(prompt, json_format=True)
            try:
                highlights = self._parse_response(raw_text, chunk_offset=chunk.start)
                for h in highlights:
                    if not h.transcript_chunk:
                        h.transcript_chunk = chunk.text
                return highlights
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

        highlights: list[Highlight] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            try:
                start = float(item["start"])
                end = float(item["end"])
            except (KeyError, TypeError, ValueError):
                continue

            if end <= start:
                continue

            # If model emitted chunk-relative offsets for a non-zero chunk, shift to stream time.
            applied_offset = 0.0
            if chunk_offset > 0.0 and start >= 0.0 and end <= chunk_offset:
                applied_offset = chunk_offset
                start += applied_offset
                end += applied_offset

            try:
                raw_score = float(item.get("score", 0.0))
            except (TypeError, ValueError):
                raw_score = 0.0
            score = max(0.0, min(10.0, raw_score))

            moment_type = str(item.get("moment_type", "hype")).strip().lower()
            if moment_type not in VALID_MOMENT_TYPES:
                moment_type = "hype"

            sub_clips: list[dict[str, Any]] = []
            raw_sub_clips = item.get("sub_clips")
            if isinstance(raw_sub_clips, list):
                for beat in raw_sub_clips:
                    if not isinstance(beat, dict):
                        continue
                    try:
                        b_start = float(beat["start"])
                        b_end = float(beat["end"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if applied_offset > 0.0 and b_start >= 0.0 and b_end <= chunk_offset:
                        b_start += applied_offset
                        b_end += applied_offset
                    sub_clips.append(
                        {
                            "start": b_start,
                            "end": b_end,
                            "label": str(beat.get("label", "")).strip(),
                        }
                    )

            title = str(item.get("title", "")).strip() or "Stream Highlight"
            reason = str(item.get("reason", "")).strip()
            best_quote = str(item.get("best_quote", "")).strip()
            narration_intro = str(item.get("narration_intro", "")).strip()
            narration_outro = str(item.get("narration_outro", "")).strip()
            context_badge = str(item.get("context_badge", "")).strip()

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
                )
            )

        return highlights

    # ------------------------------------------------------------------
    # Narrative & metadata generation
    # ------------------------------------------------------------------

    def generate_narrative(
        self, highlight: Highlight, streamer: str, game: str
    ) -> tuple[str, str, str]:
        """Backfill (narration_intro, narration_outro, context_badge) via narrative_writer.txt."""
        prompt = self.narrative_prompt.format(
            streamer=streamer or "the streamer",
            game=game or "the game",
            moment_type=highlight.moment_type or "hype",
            hook_title=highlight.title,
            description=highlight.reason,
            quote=highlight.best_quote,
        )

        for attempt in range(1, self.MAX_JSON_RETRIES + 1):
            raw_text = self._call_generate(prompt, json_format=True)
            try:
                data = json.loads(_clean_llm_text(raw_text))
                if not isinstance(data, dict):
                    raise ValueError("Expected JSON object for narrative response.")
                intro = str(data.get("narration_intro", "")).strip() or highlight.narration_intro
                outro = str(data.get("narration_outro", "")).strip() or highlight.narration_outro
                badge = str(data.get("context_badge", "")).strip() or highlight.context_badge
                return intro, outro, badge
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning(
                    "Malformed JSON from generate_narrative (attempt %d/%d): %s",
                    attempt,
                    self.MAX_JSON_RETRIES,
                    exc,
                )

        return (
            highlight.narration_intro,
            highlight.narration_outro,
            highlight.context_badge,
        )

    def _generate_hook_title(
        self, highlight: Highlight, streamer: str, game: str
    ) -> str:
        """Generate a short hook title using hook_generator.txt."""
        prompt = self.hook_prompt.format(
            streamer=streamer or "the streamer",
            game=game or "the game",
            description=highlight.reason,
            quote=highlight.best_quote,
        )
        raw_text = self._call_generate(prompt, json_format=False)
        cleaned = _clean_llm_text(raw_text)
        for line in cleaned.splitlines():
            candidate = line.strip().strip("\"'")
            if candidate:
                return candidate
        return highlight.title

    def _generate_caption_and_hashtags(
        self,
        highlight: Highlight,
        streamer: str,
        game: str,
        hook_title: str,
    ) -> tuple[str, list[str]]:
        """Generate social media caption and hashtags using caption_writer.txt."""
        hashtag_count = int(self.config.get("metadata.hashtag_count", 15))
        prompt = self.caption_prompt.format(
            streamer=streamer or "the streamer",
            game=game or "the game",
            hook_title=hook_title,
            description=highlight.reason,
            hashtag_count=hashtag_count,
        )

        for attempt in range(1, self.MAX_JSON_RETRIES + 1):
            raw_text = self._call_generate(prompt, json_format=True)
            try:
                data = json.loads(_clean_llm_text(raw_text))
                if not isinstance(data, dict):
                    raise ValueError("Expected JSON object for caption response.")
                caption = str(data.get("caption", "")).strip()
                raw_tags = data.get("hashtags", [])
                hashtags: list[str] = []
                if isinstance(raw_tags, list):
                    for tag in raw_tags:
                        cleaned_tag = str(tag).lstrip("#").strip()
                        if cleaned_tag:
                            hashtags.append(cleaned_tag)
                if not caption:
                    caption = highlight.reason or hook_title
                return caption, hashtags
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning(
                    "Malformed JSON from caption generator (attempt %d/%d): %s",
                    attempt,
                    self.MAX_JSON_RETRIES,
                    exc,
                )

        # Graceful fallback if caption LLM returns malformed JSON across all retries
        fallback_tags: list[str] = []
        if self.config.get("metadata.include_game_tag", True) and game:
            fallback_tags.append(re.sub(r"\W+", "", game))
        if self.config.get("metadata.include_streamer_tag", True) and streamer:
            fallback_tags.append(re.sub(r"\W+", "", streamer))
        fallback_tags = [t for t in fallback_tags if t]
        return highlight.reason or hook_title, fallback_tags

    def generate_metadata(
        self, highlight: Highlight, streamer: str, game: str, source_url: str
    ) -> ClipMetadata:
        """Generate hook title, narrative fields, caption, and hashtags for a highlight."""
        hook_title = self._generate_hook_title(highlight, streamer, game)

        narration_intro = highlight.narration_intro.strip()
        narration_outro = highlight.narration_outro.strip()
        context_badge = highlight.context_badge.strip()

        if not (narration_intro and narration_outro and context_badge):
            gen_intro, gen_outro, gen_badge = self.generate_narrative(
                highlight, streamer, game
            )
            narration_intro = narration_intro or gen_intro
            narration_outro = narration_outro or gen_outro
            context_badge = context_badge or gen_badge

        caption, hashtags = self._generate_caption_and_hashtags(
            highlight, streamer, game, hook_title
        )

        transformation_mode = str(self.config.get("transformation.mode", "narrative"))
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
            moment_type=highlight.moment_type,
            narration_intro=narration_intro,
            narration_outro=narration_outro,
            context_badge=context_badge,
            transformation_mode=transformation_mode,
        )
