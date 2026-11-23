"""LLM-powered context refinement: analyzes a 5-minute window to find exact event boundaries."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import re
from typing import Any

from ..analysis.context_window import ContextWindow
from ..config import Config

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "refiner.txt"


@dataclass
class RefinedMoment:
    """Result of LLM context refinement."""
    exact_start: float
    exact_end: float
    peak_time: float
    moment_type: str
    title: str
    description: str
    sub_clips: list[dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.0


class ContextRefiner:
    """Refine highlight boundaries using Gemini Flash API or local Ollama."""

    def __init__(self, config: Config):
        self.config = config
        self.engine = str(config.get("refinement.engine", "auto")).lower()
        self.gemini_model = str(config.get("refinement.gemini_model", "gemini-2.5-flash"))
        api_key_env = str(config.get("refinement.gemini_api_key_env", "GEMINI_API_KEY"))
        self.gemini_api_key = os.environ.get(api_key_env, "").strip()
        self.ollama_model = str(config.get("llm.model", "auto"))
        self.ollama_base_url = str(config.get("llm.base_url", "http://localhost:11434"))

    def _resolve_engine(self) -> str:
        if self.engine != "auto":
            return self.engine
        if self.gemini_api_key:
            return "gemini"
        return "ollama"

    def _load_prompt_template(self) -> str:
        if _DEFAULT_PROMPT_PATH.is_file():
            try:
                return _DEFAULT_PROMPT_PATH.read_text(encoding="utf-8")
            except Exception as exc:
                logger.warning("Could not read prompt from %s: %s", _DEFAULT_PROMPT_PATH, exc)
        return (
            "You are a stream clip editor. Given a 5-minute context window from a livestream, "
            "identify the EXACT moment that would make the best highlight clip.\n\n"
            "STREAM CONTEXT:\n{context}\n\n"
            "PROPOSED CUT: [{proposed_start:.1f}s - {proposed_end:.1f}s]\n{hint}\n\n"
            "Respond ONLY with valid JSON."
        )

    @staticmethod
    def _fallback_moment(
        proposed_start: float, proposed_end: float, reasoning: str = ""
    ) -> RefinedMoment:
        """Construct fallback RefinedMoment matching original input bounds."""
        s = float(proposed_start)
        e = float(proposed_end)
        if e <= s:
            e = s + 30.0
        return RefinedMoment(
            exact_start=round(s, 3),
            exact_end=round(e, 3),
            peak_time=round((s + e) / 2.0, 3),
            moment_type="hype",
            title="Stream Highlight",
            description="",
            sub_clips=[],
            reasoning=reasoning,
            confidence=0.0,
        )

    async def refine(
        self,
        context: ContextWindow,
        proposed_start: float,
        proposed_end: float,
        hint: str = "",
    ) -> RefinedMoment:
        """Analyze context window and return refined boundaries."""
        template = self._load_prompt_template()
        prompt = template.format(
            context=context.to_llm_text(),
            proposed_start=proposed_start,
            proposed_end=proposed_end,
            hint=f"HINT: {hint}" if hint else "",
        )
        engine = self._resolve_engine()
        sheets = getattr(context, "contact_sheets", [])
        if engine == "gemini":
            try:
                return await self._call_gemini(
                    prompt, proposed_start, proposed_end, contact_sheets=sheets
                )
            except Exception as exc:
                logger.warning("Gemini refinement failed: %s; attempting fallback", exc)
                if self.engine == "auto":
                    try:
                        return await self._call_ollama(
                            prompt, proposed_start, proposed_end, contact_sheets=sheets
                        )
                    except Exception as ol_exc:
                        logger.warning("Ollama fallback refinement failed: %s", ol_exc)
                return self._fallback_moment(
                    proposed_start, proposed_end, f"Refinement failed: {exc}"
                )
        elif engine == "ollama":
            try:
                return await self._call_ollama(
                    prompt, proposed_start, proposed_end, contact_sheets=sheets
                )
            except Exception as exc:
                logger.warning("Ollama refinement failed: %s; returning original bounds", exc)
                return self._fallback_moment(
                    proposed_start, proposed_end, f"Ollama failed: {exc}"
                )
        elif engine == "mcp":
            raise ValueError(
                "engine='mcp' means the MCP controller reviews the context itself. "
                "Use inspect_moment_context instead of refine_highlight_bounds."
            )
        else:
            raise ValueError(f"Unknown refinement engine: {engine}")

    async def _call_gemini(
        self,
        prompt: str,
        proposed_start: float,
        proposed_end: float,
        contact_sheets: list[Path] | None = None,
    ) -> RefinedMoment:
        """Call Gemini Flash API via google-genai SDK."""
        from google import genai
        from google.genai import types

        client = (
            genai.Client(api_key=self.gemini_api_key)
            if self.gemini_api_key
            else genai.Client()
        )
        contents: Any = prompt
        if contact_sheets:
            parts: list[Any] = [prompt]
            for sheet_path in contact_sheets:
                p = Path(sheet_path)
                if p.is_file() and p.stat().st_size > 0:
                    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
                    parts.append(types.Part.from_bytes(data=p.read_bytes(), mime_type=mime))
            if len(parts) > 1:
                contents = parts

        response = await asyncio.to_thread(
            client.models.generate_content,
            model=self.gemini_model,
            contents=contents,
        )
        text = response.text if hasattr(response, "text") and response.text else ""
        return self._parse_response(text, proposed_start, proposed_end)

    async def _call_ollama(
        self,
        prompt: str,
        proposed_start: float,
        proposed_end: float,
        contact_sheets: list[Path] | None = None,
    ) -> RefinedMoment:
        """Call local Ollama model."""
        import ollama as _ollama

        client = _ollama.Client(host=self.ollama_base_url)
        model = self.ollama_model if self.ollama_model != "auto" else "qwen3:4b"

        valid_images: list[Path] = []
        if contact_sheets:
            for sheet_path in contact_sheets:
                p = Path(sheet_path)
                if p.is_file() and p.stat().st_size > 0:
                    valid_images.append(p)

        msg: dict[str, Any] = {"role": "user", "content": prompt}
        if valid_images:
            msg["images"] = valid_images
            vision_model = str(
                self.config.get("library.vision.model", "qwen2.5vl:3b") or "qwen2.5vl:3b"
            ).strip()
            if vision_model:
                model = vision_model

        response = await asyncio.to_thread(
            client.chat,
            model=model,
            messages=[msg],
        )
        content = ""
        if hasattr(response, "message") and hasattr(response.message, "content"):
            content = response.message.content or ""
        elif isinstance(response, dict):
            content = response.get("message", {}).get("content", "")
        else:
            content = str(response)
        return self._parse_response(content, proposed_start, proposed_end)

    @staticmethod
    def _parse_response(
        text: str, default_start: float = 0.0, default_end: float = 30.0
    ) -> RefinedMoment:
        """Parse LLM JSON response into RefinedMoment."""
        cleaned = text.strip()
        # Strip markdown fences ```json ... ``` or ``` ... ```
        if "```" in cleaned:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
            if match:
                cleaned = match.group(1).strip()
            else:
                cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(cleaned)
        except Exception:
            # Try to find { ... } block
            match = re.search(r"\{[\s\S]*\}", cleaned)
            if match:
                try:
                    data = json.loads(match.group(0))
                except Exception:
                    data = {}
            else:
                data = {}

        exact_start = float(data.get("exact_start", default_start))
        exact_end = float(data.get("exact_end", default_end))
        if exact_end <= exact_start:
            exact_end = exact_start + max(15.0, default_end - default_start)

        peak_time = float(data.get("peak_time", (exact_start + exact_end) / 2.0))

        raw_sub_clips = data.get("sub_clips")
        sub_clips: list[dict[str, Any]] = []
        if isinstance(raw_sub_clips, list):
            for sc in raw_sub_clips:
                if isinstance(sc, dict) and "start" in sc and "end" in sc:
                    try:
                        s = float(sc["start"])
                        e = float(sc["end"])
                        if e > s:
                            sub_clips.append(
                                {
                                    "start": s,
                                    "end": e,
                                    "label": str(sc.get("label", "Beat")),
                                }
                            )
                    except (ValueError, TypeError):
                        pass

        return RefinedMoment(
            exact_start=round(exact_start, 3),
            exact_end=round(exact_end, 3),
            peak_time=round(peak_time, 3),
            moment_type=str(data.get("moment_type", "hype")),
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            sub_clips=sub_clips,
            reasoning=str(data.get("reasoning", "")),
            confidence=float(data.get("confidence", 0.5)),
        )
