"""Local vision model integration for describing frame contact sheets via Ollama."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from streamclipper.config import Config
from streamclipper.library.frames import ContactSheet

logger = logging.getLogger(__name__)

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(
    r"```(?:json)?\s*\n?(.*?)\n?```",
    flags=re.DOTALL | re.IGNORECASE,
)


class VisionUnavailable(RuntimeError):
    """Raised when Ollama is unreachable or the vision model is missing/disabled."""


def _extract_model_names(models_response: Any) -> list[str]:
    """Extract list of model names from ollama list response."""
    names: list[str] = []
    models_list = getattr(models_response, "models", None)
    if models_list is None and isinstance(models_response, dict):
        models_list = models_response.get("models")
    if not models_list and isinstance(models_response, list):
        models_list = models_response

    for entry in models_list or []:
        if isinstance(entry, str):
            names.append(entry)
        elif isinstance(entry, dict):
            n = entry.get("model") or entry.get("name")
            if n:
                names.append(str(n))
        else:
            n = getattr(entry, "model", None) or getattr(entry, "name", None)
            if n:
                names.append(str(n))
    return names


def resolve_installed_model(target: str, installed: list[str]) -> str | None:
    """Resolve target model name to an installed model tag in Ollama.

    Handles exact match, :latest suffix variations, and hyphen/underscore normalization
    (e.g. qwen2.5vl:3b <-> qwen2.5-vl:3b).
    """
    clean_target = str(target or "").strip()
    if not clean_target:
        return None

    # 1. Exact match
    for inst in installed:
        if clean_target == inst:
            return inst

    # 2. Tag with / without :latest
    for inst in installed:
        if clean_target == inst.removesuffix(":latest"):
            return inst
        if clean_target.removesuffix(":latest") == inst:
            return inst

    # 3. Normalized comparison (ignore hyphens, underscores, case)
    def _norm(name: str) -> str:
        base = name.removesuffix(":latest").lower()
        return base.replace("-", "").replace("_", "")

    target_norm = _norm(clean_target)
    for inst in installed:
        if _norm(inst) == target_norm:
            return inst

    # 4. Target base prefix match (e.g. "qwen2.5-vl:3b" matching "qwen2.5-vl:3b-instruct" or similar)
    for inst in installed:
        inst_norm = _norm(inst)
        if target_norm and (inst_norm.startswith(target_norm) or target_norm.startswith(inst_norm)):
            return inst

    return None


def _model_matches(target: str, installed: list[str]) -> bool:
    """Check whether target model matches any installed model name."""
    return resolve_installed_model(target, installed) is not None


def describe_sheet_ollama(config: Config, sheet: ContactSheet) -> list[dict[str, Any]]:
    """Describe tiles in a contact sheet using local Ollama vision model.

    Returns a list of dicts: [{"t": seconds, "note": "..."}, ...] with at most
    15-word factual notes per tile. Raises VisionUnavailable if Ollama or the
    model is missing or disabled.
    """
    model = str(config.get("library.vision.model", "qwen2.5vl:3b") or "").strip()
    if not model:
        raise VisionUnavailable("Vision model is disabled in configuration (library.vision.model is empty).")

    base_url = str(config.get("llm.base_url", "http://localhost:11434") or "http://localhost:11434")
    timeout = float(config.get("llm.timeout_seconds", 120))

    import ollama

    client = ollama.Client(host=base_url, timeout=timeout)

    try:
        models_res = client.list()
        installed_names = _extract_model_names(models_res)
        resolved_model = resolve_installed_model(model, installed_names)
        if not resolved_model:
            raise VisionUnavailable(
                f"Vision model '{model}' not found in Ollama. Pull it with: ollama pull {model}"
            )
    except VisionUnavailable:
        raise
    except Exception as exc:
        raise VisionUnavailable(
            f"Cannot connect to Ollama at {base_url}: {exc}. Ensure Ollama is running (`ollama serve`) and run `ollama pull {model}`."
        ) from exc

    ts_str = ", ".join(f"{t:.1f}s" for t in sheet.timestamps)
    prompt = (
        f"You are analyzing a video contact sheet containing {len(sheet.timestamps)} sampled frames at timestamps: [{ts_str}].\n"
        "For each tile, write one short factual note (at most 15 words) describing location, action, and notable objects. "
        "Do not guess character or streamer names.\n"
        "Return ONLY a JSON array of objects with keys 't' (float seconds) and 'note' (string description):\n"
        '[{"t": 10.0, "note": "player driving red sports car down highway"}, ...]'
    )

    try:
        response = client.chat(
            model=resolved_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [sheet.path],
                }
            ],
            format="json",
        )
    except Exception as exc:
        raise VisionUnavailable(f"Ollama vision chat failed for model '{resolved_model}': {exc}") from exc

    content = ""
    if hasattr(response, "message") and hasattr(response.message, "content"):
        content = response.message.content or ""
    elif isinstance(response, dict):
        msg = response.get("message")
        if isinstance(msg, dict):
            content = msg.get("content") or ""
        elif isinstance(response.get("response"), str):
            content = response["response"]
    elif hasattr(response, "content"):
        content = response.content or ""

    # Remove think blocks if any
    content = _THINK_TAG_RE.sub("", content).strip()

    # Extract markdown code fence if present
    fence_match = _CODE_FENCE_RE.search(content)
    raw_json = fence_match.group(1).strip() if fence_match else content.strip()

    parsed: Any = None
    try:
        parsed = json.loads(raw_json)
    except Exception:
        # Fallback to finding outermost brackets
        start_idx = raw_json.find("[")
        end_idx = raw_json.rfind("]")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                parsed = json.loads(raw_json[start_idx : end_idx + 1])
            except Exception:
                return []
        else:
            return []

    # If wrapped in a dict e.g. {"notes": [...]} or {"tiles": [...]}
    if isinstance(parsed, dict):
        for val in parsed.values():
            if isinstance(val, list):
                parsed = val
                break
        else:
            return []

    if not isinstance(parsed, list):
        return []

    notes: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        raw_t = item.get("t")
        raw_note = item.get("note")
        if raw_t is None or raw_note is None:
            continue
        try:
            t_val = float(raw_t)
        except (ValueError, TypeError):
            continue

        note_str = str(raw_note).strip()
        if not note_str:
            continue

        # Clamp to <= 15 words
        words = note_str.split()
        if len(words) > 15:
            note_str = " ".join(words[:15])

        notes.append({"t": round(t_val, 2), "note": note_str})

    return notes
