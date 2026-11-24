"""Narration style guide + quality-check tests (energetic, interactive gamer voice)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from ollama._types import GenerateResponse

from streamclipper.cache import StepCache
from streamclipper.config import Config
from streamclipper.mcp_server import server as mcp_server
from streamclipper.mcp_server.server import mcp
from streamclipper.models import Highlight, PipelineState, PipelineStatus, Platform, StreamInfo
from streamclipper.narration_style import (
    BANNED_PHRASES,
    EXAMPLES,
    PLAY_BY_PLAY_PHRASES,
    STYLE_RULES,
    check_narration,
    check_narration_beats,
    narration_guide,
)
from streamclipper.scoring import ollama_scorer
from streamclipper.scoring.ollama_scorer import OllamaScorer
from tests.test_mcp_server import _make_config, _make_sample_transcript

_REAL_CALL_TOOL = mcp.call_tool

URL = "https://www.youtube.com/watch?v=qg123456789"
EXCERPT = "NO WAY HELICOPTER EXPLODED"

GOOD = {
    "narration_intro": "Chat, that helicopter is way too close to the roof.",
    "callout": "Wait wait wait, helicopter!",
    "narration_outro": "Did you see that helicopter explode? Be honest.",
}
BAD_INTRO = "This is the single most viral stunt from Carry's launch stream."
BAD_OUTRO = "Pure comedy right here. Subscribe for more GTA Six moments!"
BAD = {"narration_intro": BAD_INTRO, "narration_outro": BAD_OUTRO}


def _joined(problems: list[str]) -> str:
    return " | ".join(problems).lower()


# ---------------------------------------------------------------------------
# N1 guide + checks
# ---------------------------------------------------------------------------
def test_N1_guide_has_rules_examples_and_banned() -> None:
    guide = narration_guide("english", "cold open", ["Old outro line here."])
    assert STYLE_RULES in guide
    assert "STYLE EXAMPLES" in guide and "Do NOT copy" in guide
    assert "Would you have bailed out there? Be honest." in guide
    assert "cold open" in guide and "Old outro line here." in guide
    for phrase in ("pure comedy right here", "like and subscribe", "crazy moment", "kya tumne kabhi dekha", "kya tum ise khelna", "watch till end"):
        assert phrase in BANNED_PHRASES and phrase in guide
    for mt in ("funny", "fail", "graphics_detail", "hype", "clutch", "chat_chaos", "lore_secret", "analysis", "reaction"):
        assert mt in guide
    assert len(EXAMPLES["english"]) == 3 and len(EXAMPLES["hinglish"]) == 3


def test_N1_hinglish_examples_only_for_hinglish() -> None:
    en = narration_guide("english")
    hi = narration_guide("hinglish")
    assert "full send kar diya" not in en and "ROMAN script" not in en
    assert "Bhai ne police car leke bridge pe full send kar diya!" in hi
    assert "Tum hote toh brake maarte ya send karte? Comment karo." in hi
    assert "ROMAN script" in hi


def test_N1_banned_phrases_reexported_from_scorer() -> None:
    assert ollama_scorer.BANNED_PHRASES is BANNED_PHRASES


def test_N1_good_example_passes() -> None:
    assert check_narration(GOOD, transcript_excerpt=EXCERPT) == []


def test_N1_all_shipped_examples_pass_checks() -> None:
    for lang in ("english", "hinglish"):
        for ex in EXAMPLES[lang]:
            assert check_narration(ex) == [], ex


def test_N1_real_bad_lines_rejected() -> None:
    problems = check_narration(BAD, transcript_excerpt=EXCERPT)
    text = _joined(problems)
    assert "pure comedy right here" in text and "this is the single" in text
    assert "launch stream" in text and "subscribe" in text
    assert "not grounded" in text and "not interactive" in text


def test_N1_flags_each_problem_type() -> None:
    # banned phrase
    assert "in today's video" in _joined(
        check_narration({**GOOD, "narration_intro": "In today's video the helicopter blows up."})
    )
    # repeat of a recent outro (substring and token overlap)
    assert "repeats" in _joined(
        check_narration(GOOD, avoid_phrases=["Did you see that helicopter explode? Be honest."])
    )
    assert "repeats" in _joined(
        check_narration(
            GOOD, avoid_phrases=["Did you see that helicopter explode, be honest bro"]
        )
    )
    # long sentence
    long_intro = "Bro the helicopter just flew straight into the roof of the building and exploded everywhere"
    assert "words (max 14)" in _joined(check_narration({**GOOD, "narration_intro": long_intro}))
    assert "words (max 5)" in _joined(
        check_narration(GOOD, max_words_per_sentence=5)
    )
    # outro asks to subscribe / like
    assert "subscribe/follow/like" in _joined(
        check_narration({**GOOD, "narration_outro": "Hit that like button, you legend."})
    )
    # same first 3 words
    assert "same 3 words" in _joined(
        check_narration(
            {**GOOD, "narration_intro": "Did you see that coming?", "narration_outro": "Did you see that again?"}
        )
    )
    # not grounded (excerpt given) vs skipped when nothing to ground on
    assert "not grounded" in _joined(
        check_narration(GOOD, transcript_excerpt="fishing boat harbor sunset")
    )
    assert "not grounded" not in _joined(check_narration(GOOD))
    assert "ungrounded" in _joined(check_narration(GOOD, require_grounding=True))
    # chat samples ground too
    assert check_narration(GOOD, chat_samples=["helicopter!!"]) == []
    # not interactive
    flat = {**GOOD, "narration_outro": "The helicopter exploded on the roof."}
    assert "not interactive" in _joined(check_narration(flat, transcript_excerpt=EXCERPT))
    hinglish = {**GOOD, "narration_outro": "Bhai ne helicopter uda diya."}
    assert "not interactive" not in _joined(check_narration(hinglish))


# ---------------------------------------------------------------------------
# N2 Ollama retry loop
# ---------------------------------------------------------------------------
def _hl() -> Highlight:
    h = Highlight(
        start=1790.0, end=1840.0, score=9.0, title="Helicopter on the roof", reason="It explodes",
        best_quote="no way", moment_type="hype",
    )
    h.transcript_chunk = EXCERPT
    return h


def _resp(intro: str, outro: str, callout: str = "Wait for the helicopter!") -> GenerateResponse:
    return GenerateResponse(
        response=json.dumps(
            {
                "narration_intro": intro,
                "narration_outro": outro,
                "callout": callout,
                "context_badge": "🔥 Clutch",
            }
        )
    )


def test_N2_prompt_template_has_style_guide_and_renders(tmp_path: Path) -> None:
    cfg = Config(user_config_path=tmp_path / "none.yaml")
    assert "gamer friend" in str(cfg.get("transformation.persona"))
    with patch("streamclipper.scoring.ollama_scorer.ollama.Client") as cls:
        cls.return_value.generate.return_value = _resp(GOOD["narration_intro"], GOOD["narration_outro"])
        scorer = OllamaScorer(cfg)
        scorer.generate_narrative(_hl(), "Carry", "GTA 6", hook_style="cold open", language="hinglish")
        prompt = cls.return_value.generate.call_args.kwargs["prompt"]
    assert STYLE_RULES in prompt and "Bhai ne police car" in prompt and "Carry" in prompt
    assert "{style_guide}" not in prompt


def test_N2_ollama_retry_feeds_problems_back_and_keeps_best(tmp_path: Path) -> None:
    cfg = Config(user_config_path=tmp_path / "none.yaml")
    worst = _resp(BAD_INTRO, BAD_OUTRO)
    best = _resp("Bro, that helicopter is about to blow.", "It exploded on the roof.", "Wait, the helicopter!")
    with patch("streamclipper.scoring.ollama_scorer.ollama.Client") as cls:
        client = cls.return_value
        client.generate.side_effect = [worst, best, worst]
        scorer = OllamaScorer(cfg)
        h = _hl()
        intro, outro, badge = scorer.generate_narrative(h, "Carry", "GTA 6")
    assert client.generate.call_count == 3
    second_prompt = client.generate.call_args_list[1].kwargs["prompt"]
    assert "Fix these problems" in second_prompt and "banned generic phrase" in second_prompt
    assert (intro, outro) == ("Bro, that helicopter is about to blow.", "It exploded on the roof.")
    assert h.callout == "Wait, the helicopter!"


def test_N2_ollama_stops_on_first_clean_attempt(tmp_path: Path) -> None:
    cfg = Config(user_config_path=tmp_path / "none.yaml")
    with patch("streamclipper.scoring.ollama_scorer.ollama.Client") as cls:
        client = cls.return_value
        client.generate.side_effect = [_resp(GOOD["narration_intro"], GOOD["narration_outro"])]
        OllamaScorer(cfg).generate_narrative(_hl(), "Carry", "GTA 6")
    assert client.generate.call_count == 1


def test_N2_grounded_fallback_intro_is_energetic_and_clean() -> None:
    text = ollama_scorer.grounded_fallback_intro(_hl(), "Carry", "GTA 6")
    assert text.startswith("Look at this")
    assert "Helicopter on the roof" in text
    assert not any(p in text.lower() for p in BANNED_PHRASES)


# ---------------------------------------------------------------------------
# N3 MCP path
# ---------------------------------------------------------------------------
@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Config, StepCache]:
    config = _make_config(tmp_path)
    cache = StepCache(config)
    monkeypatch.setattr(mcp_server, "_get_config", lambda: config)
    cache.save_state(
        PipelineState(
            stream_info=StreamInfo(
                url=URL, title="t", platform=Platform.YOUTUBE, streamer="CarryisLive",
                game="GTA 6", duration_seconds=7200, video_id="qg123456789",
            ),
            status=PipelineStatus.SCORING,
            transcript=_make_sample_transcript(),
        )
    )
    return config, cache


def _payload(**over: Any) -> str:
    item: dict[str, Any] = {
        "start": 1790.0, "end": 1840.0, "title": "Helicopter", "moment_type": "hype",
        "score": 9.0, **GOOD,
    }
    item.update(over)
    return json.dumps([item])


def _data(res: Any) -> dict[str, Any]:
    return json.loads(res.content[0].text)


@pytest.mark.asyncio
async def test_N3_get_transcript_chunks_returns_narration_guide(env) -> None:
    data = _data(await _REAL_CALL_TOOL("get_transcript_chunks", {"url": URL, "only_with_signals": False}))
    assert "narration_guide" in data and STYLE_RULES in data["narration_guide"]
    assert data["avoid_phrases"] == []
    assert "narration_guide" in mcp_server.SERVER_INSTRUCTIONS


@pytest.mark.asyncio
async def test_N3_submit_rejects_generic_with_problems_and_accepts_good(env) -> None:
    bad = _data(
        await _REAL_CALL_TOOL(
            "submit_highlights",
            {"url": URL, "highlights": _payload(narration_intro=BAD_INTRO, narration_outro=BAD_OUTRO)},
        )
    )
    assert "error" in bad or bad.get("accepted_count") == 0
    ok_and_bad = json.loads(
        _payload()[:-1]
        + ","
        + json.dumps(
            {"start": 1795.0, "end": 1840.0, "narration_intro": BAD_INTRO, "narration_outro": BAD_OUTRO}
        )
        + "]"
    )
    mixed = _data(
        await _REAL_CALL_TOOL("submit_highlights", {"url": URL, "highlights": json.dumps(ok_and_bad)})
    )
    assert mixed["accepted_count"] == 1 and mixed["rejected_count"] == 1
    rej = mixed["rejected"][0]
    assert rej["problems"]
    assert "banned" in _joined(rej["problems"])


@pytest.mark.asyncio
async def test_N3_narration_strict_false_returns_warnings(env) -> None:
    data = _data(
        await _REAL_CALL_TOOL(
            "submit_highlights",
            {
                "url": URL,
                "highlights": _payload(narration_intro=BAD_INTRO, narration_outro=BAD_OUTRO),
                "narration_strict": False,
            },
        )
    )
    assert data["accepted_count"] == 1
    warn = [w for w in data["warnings"] if isinstance(w, dict)]
    assert warn and warn[0]["problems"]


@pytest.mark.asyncio
async def test_N3_render_clip_rejects_bad_narration_unless_lenient(env) -> None:
    args = {
        "url": URL, "start": 1790.0, "end": 1840.0, "hook_title": "T",
        "narration_intro": BAD_INTRO, "narration_outro": BAD_OUTRO,
    }
    err = _data(await _REAL_CALL_TOOL("render_clip", args))["error"]
    assert "quality check" in err and "banned generic phrase" in err
    # Lenient mode passes the narration check (it fails later, for other reasons: no audio).
    lenient = _data(await _REAL_CALL_TOOL("render_clip", {**args, "narration_strict": False}))
    assert "quality check" not in lenient.get("error", "")


# ---------------------------------------------------------------------------
# N4 Progressive narration_beats checks
# ---------------------------------------------------------------------------
def test_N4_banned_phrases_include_new_additions() -> None:
    for phrase in ("kya tumne kabhi dekha", "kya tum ise khelna", "watch till end", "you won't believe"):
        assert phrase in BANNED_PHRASES
    guide = narration_guide("english")
    for phrase in ("kya tumne kabhi dekha", "kya tum ise khelna", "watch till end", "you won't believe"):
        assert phrase in guide


def test_N4_style_rules_contain_progressive_story_arc() -> None:
    assert "PROGRESSIVE STORY ARC" in STYLE_RULES
    assert "Beat 1 (0s, Hook)" in STYLE_RULES
    assert "Beat 2 (Escalation)" in STYLE_RULES
    assert "Gap (2–4s)" in STYLE_RULES
    assert "Beat 3 (Payoff / Breakdown)" in STYLE_RULES
    assert "Beat 4 (Sharp Closer)" in STYLE_RULES


def test_N4_check_narration_beats_clean_passes() -> None:
    beats = [
        {"at_second": 0.0, "text": "He thought the helicopter was safe on that roof..."},
        {"at_second": 4.5, "text": "Engine smoking with zero clearance on both blades."},
        {"at_second": 11.0, "text": "The tail rotor clipped the billboard at full speed!"},
        {"at_second": 16.0, "text": "Never letting him pilot the escape vehicle again."},
    ]
    # Grounded against EXCERPT ("NO WAY HELICOPTER EXPLODED")
    problems = check_narration_beats(beats, transcript_excerpt=EXCERPT)
    assert problems == []


def test_N4_check_narration_beats_flags_banned_phrases() -> None:
    beats = [
        {"at_second": 0.0, "text": "Watch till end to see what happens!"},
        {"at_second": 5.0, "text": "Kya tumne kabhi dekha itna crazy helicopter?"},
        {"at_second": 10.0, "text": "Kya tum ise khelna chahoge?"},
        {"at_second": 15.0, "text": "You won't believe how this ended."},
    ]
    problems = check_narration_beats(beats)
    text = _joined(problems)
    assert "watch till end" in text
    assert "kya tumne kabhi dekha" in text
    assert "kya tum ise khelna" in text
    assert "you won't believe" in text


def test_N4_check_narration_beats_flags_sentence_length() -> None:
    beats = [
        {
            "at_second": 0.0,
            "text": "Bro the helicopter just flew directly into the side of the skyscraper and burst into giant flames",
        },
    ]
    problems = check_narration_beats(beats)
    text = _joined(problems)
    assert "words (max 14)" in text


def test_N4_check_narration_beats_flags_play_by_play_filler() -> None:
    beats = [
        {"at_second": 0.0, "text": "As you can see, the helicopter is flying."},
        {"at_second": 5.0, "text": "Now he is shooting at the cops on the roof."},
        {"at_second": 10.0, "text": "What would you do in this situation?"},
    ]
    problems = check_narration_beats(beats)
    text = _joined(problems)
    assert "play-by-play filler" in text
    assert "as you can see" in text
    assert "now he is" in text
    assert "what would you do" in text


def test_N4_check_narration_beats_flags_overlapping_timestamps() -> None:
    # 1. Out of order
    out_of_order = [
        {"at_second": 5.0, "text": "First beat."},
        {"at_second": 2.0, "text": "Second beat out of order."},
    ]
    assert "overlapping" in _joined(check_narration_beats(out_of_order))

    # 2. Same start time
    same_start = [
        {"at_second": 0.0, "text": "Beat A."},
        {"at_second": 0.0, "text": "Beat B."},
    ]
    assert "overlapping" in _joined(check_narration_beats(same_start))

    # 3. Explicit duration overlap
    explicit_overlap = [
        {"at_second": 0.0, "duration": 5.0, "text": "Beat one here."},
        {"at_second": 3.0, "duration": 3.0, "text": "Beat two overlaps."},
    ]
    problems = check_narration_beats(explicit_overlap)
    assert "overlaps" in _joined(problems)

    # 4. Spoken length overlap (without explicit duration)
    long_speech_overlap = [
        {"at_second": 0.0, "text": "The helicopter blades are spinning wildly out of control right now."},  # 11 words
        {"at_second": 1.0, "text": "Next beat starts way too early."},
    ]
    problems = check_narration_beats(long_speech_overlap)
    assert "overlaps" in _joined(problems)


def test_N4_check_narration_beats_flags_subscribe_and_repeats() -> None:
    beats = [
        {"at_second": 0.0, "text": "Check out this helicopter flight."},
        {"at_second": 6.0, "text": "Subscribe for more awesome gaming content!"},
    ]
    assert "subscribe" in _joined(check_narration_beats(beats))

    avoid = ["Never letting him pilot again."]
    repeated_beats = [
        {"at_second": 0.0, "text": "Never letting him pilot again."},
    ]
    assert "repeats" in _joined(check_narration_beats(repeated_beats, avoid_phrases=avoid))


def test_N4_check_narration_beats_grounding() -> None:
    beats = [
        {"at_second": 0.0, "text": "Looking at the beautiful sunshine on the water."},
    ]
    # EXCERPT is "NO WAY HELICOPTER EXPLODED"
    assert "not grounded" in _joined(check_narration_beats(beats, transcript_excerpt=EXCERPT))
    # Ungrounded when require_grounding is True and no excerpt
    assert "ungrounded" in _joined(check_narration_beats(beats, require_grounding=True))


def test_N4_check_narration_dict_and_integration() -> None:
    clean_beats = [
        {"at_second": 0.0, "text": "Helicopter on the edge of the building..."},
        {"at_second": 6.0, "text": "Pure panic in the cockpit right now!"},
    ]
    # Accepts dict wrapping narration_beats
    assert check_narration_beats({"narration_beats": clean_beats}, transcript_excerpt=EXCERPT) == []
    # Integration with check_narration
    assert check_narration({"narration_beats": clean_beats}, transcript_excerpt=EXCERPT) == []
    bad_beats = [{"at_second": 0.0, "text": "Watch till end guys!"}]
    assert "watch till end" in _joined(check_narration({"narration_beats": bad_beats}))


def test_N4_check_narration_beats_parses_string_and_colon_timestamps() -> None:
    # String timestamps like "00:04", "4.5s", and "1:10" should parse without ValueError
    beats = [
        {"at_second": "00:00", "text": "Helicopter on the edge of the building..."},
        {"at_second": "4.5s", "text": "Pure panic in the cockpit right now!"},
        {"at_second": "00:10", "text": "Made it across safely in the end."},
    ]
    assert check_narration_beats(beats, transcript_excerpt=EXCERPT) == []


def test_N4_check_narration_beats_flags_invalid_timestamps_without_crashing() -> None:
    # Invalid string timestamps, negative start, or end < start should report errors cleanly without throwing
    beats = [
        {"at_second": "invalid_time", "text": "Helicopter blades spinning."},
        {"at_second": -3.5, "text": "Another line."},
        {"at_second": 10.0, "end": 5.0, "text": "Reversed end time."},
    ]
    problems = check_narration_beats(beats, transcript_excerpt=EXCERPT)
    text = _joined(problems)
    assert "invalid timestamp" in text
    assert "negative timestamp" in text
    assert "before start time" in text


def test_N4_check_narration_beats_flags_contractions_and_filler() -> None:
    beats = [
        {"at_second": 0.0, "text": "Look at them panic on the roof!"},
        {"at_second": 5.0, "text": "Now he's driving straight into danger."},
        {"at_second": 10.0, "text": "Now they're calling backup."},
    ]
    problems = check_narration_beats(beats)
    text = _joined(problems)
    assert "play-by-play filler" in text
    assert "look at them" in text
    assert "now he's" in text
    assert "now they're" in text


def test_N4_check_narration_beats_malformed_inputs() -> None:
    # Non-list/dict input
    assert "must be a list" in _joined(check_narration_beats("not_a_list"))
    # List of non-dicts
    assert "must be a dictionary" in _joined(check_narration_beats([None, "string_beat"]))
    # Missing narration text
    assert "missing narration text" in _joined(check_narration_beats([{"at_second": 0.0, "text": ""}]))


def test_N4_check_narration_grounding_with_beats_and_intro() -> None:
    # When narration_intro has hook energy ("Watch this!") and beats have concrete details ("helicopter"),
    # check_narration should accept them as grounded together without spurious errors
    fields = {
        "narration_intro": "Chat, watch this immediate chaos unfold!",
        "narration_beats": [
            {"at_second": 0.0, "text": "Helicopter clipped the corner of the building..."},
            {"at_second": 6.0, "text": "Zero control left as it starts spinning!"},
        ],
    }
    assert check_narration(fields, transcript_excerpt=EXCERPT) == []

