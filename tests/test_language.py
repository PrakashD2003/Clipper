"""Tests for language-aware (English / Hinglish) narration and ElevenLabs-first voice selection."""
from __future__ import annotations

from pathlib import Path

import pytest

from streamclipper import language as lang_mod
from streamclipper.assembly import narrator as narrator_mod
from streamclipper.assembly.narrator import TransformativeNarrator
from streamclipper.config import Config
from streamclipper.language import language_instruction, resolve_narration_language
from streamclipper.models import ClipMetadata

PROMPTS = Path(lang_mod.__file__).parent / "prompts"


def _cfg(tmp_path: Path, transformation: str = "") -> Config:
    f = tmp_path / "config.yaml"
    f.write_text(("transformation:\n" + transformation) if transformation else "", encoding="utf-8")
    return Config(f)


def test_L2_resolver_default_is_english(tmp_path):
    assert resolve_narration_language(_cfg(tmp_path), "Kai", "en") == "english"
    assert resolve_narration_language(_cfg(tmp_path), None, None) == "english"


@pytest.mark.parametrize("code", ["hi", "ur", "hindi", "HI", "hi-IN"])
def test_L2_resolver_hindi_transcript_gives_hinglish(tmp_path, code):
    assert resolve_narration_language(_cfg(tmp_path), "X", code) == "hinglish"


def test_L2_resolver_config_overrides_transcript(tmp_path):
    cfg = _cfg(tmp_path, '  narration_language: "english"\n')
    assert resolve_narration_language(cfg, "X", "hi") == "english"
    cfg = _cfg(tmp_path, '  narration_language: "hinglish"\n')
    assert resolve_narration_language(cfg, "X", "en") == "hinglish"


def test_L2_resolver_streamer_override_wins_case_insensitive(tmp_path):
    cfg = _cfg(
        tmp_path,
        '  narration_language: "english"\n  streamer_languages:\n    CarryisLive: hi\n    Other: en\n',
    )
    assert resolve_narration_language(cfg, "carryislive", "en") == "hinglish"
    assert resolve_narration_language(cfg, "OTHER", "hi") == "english"
    assert resolve_narration_language(cfg, "nobody", "hi") == "english"


def test_L2_language_instruction():
    assert language_instruction("english") == ""
    text = language_instruction("hinglish")
    assert "Hinglish" in text and "ROMAN" in text
    assert "Never use Devanagari" in text


@pytest.mark.parametrize("name", ["narrative_writer.txt", "hook_generator.txt"])
def test_L3_prompts_format_with_language_instruction(name):
    tpl = (PROMPTS / name).read_text(encoding="utf-8")
    assert "{language_instruction}" in tpl
    values = {
        k: "x"
        for k in (
            "streamer game moment_type hook_title description quote narration_style persona "
            "transcript_excerpt chat_samples avoid_phrases hook_style style_guide"
        ).split()
    }
    out = tpl.format(language_instruction=language_instruction("hinglish"), **values)
    assert "ROMAN script" in out
    out_en = tpl.format(language_instruction="", **values)
    assert "{language_instruction}" not in out_en


def test_L5_clip_metadata_narration_language_roundtrip():
    md = ClipMetadata(
        hook_title="h", caption="c", hashtags=[], streamer="s", game="g",
        source_url="u", clip_start=0.0, clip_end=5.0, highlight_score=8.0,
        narration_language="hinglish",
    )
    d = md.to_dict()
    assert d["narration_language"] == "hinglish"
    assert ClipMetadata.from_dict(d).narration_language == "hinglish"
    d.pop("narration_language")
    assert ClipMetadata.from_dict(d).narration_language == ""


def _narrator(tmp_path: Path, extra: str = "") -> TransformativeNarrator:
    return TransformativeNarrator(_cfg(tmp_path, extra))


def test_L4_elevenlabs_prepended_only_when_available(tmp_path, monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    n = _narrator(tmp_path, '  elevenlabs_voice_id: "voice123"\n')
    assert "elevenlabs" not in n._engine_chain()

    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    assert n._engine_chain()[0] == "elevenlabs"

    n_no_voice = _narrator(tmp_path)
    assert "elevenlabs" not in n_no_voice._engine_chain()


def test_L4_hinglish_chain_excludes_english_only_engines(tmp_path, monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    n = _narrator(
        tmp_path,
        '  elevenlabs_voice_id: "v"\n  hinglish_tts_engines: [chatterbox, elevenlabs, kokoro, edge-tts]\n',
    )
    assert n._engine_chain("hinglish") == ["elevenlabs", "edge-tts"]
    monkeypatch.delenv("ELEVENLABS_API_KEY")
    assert n._engine_chain("hinglish") == ["edge-tts"]
    n.language = "hinglish"
    assert n._engine_chain() == ["edge-tts"]


def test_L4_hinglish_uses_hinglish_edge_voice(tmp_path, monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    voices: list[str] = []

    class FakeEdge:
        def __init__(self, voice, rate):
            voices.append(voice)

        def synthesize(self, text, out_wav):
            Path(out_wav).write_bytes(b"RIFFxxxx")
            return Path(out_wav)

    monkeypatch.setattr(narrator_mod, "EdgeTTS", FakeEdge)
    n = _narrator(tmp_path)
    res = n.synthesize_speech("Bhai ye dekho", tmp_path / "a.wav", language="hinglish")
    assert res is not None
    n.language = "hinglish"
    assert n.resolve_narration_track(None, "Ye kya tha", tmp_path / "b.wav") is not None
    assert voices == ["en-IN-PrabhatNeural", "en-IN-PrabhatNeural"]

    n.language = "english"
    n.engine_chain = ["edge-tts"]
    n.synthesize_speech("hello there", tmp_path / "c.wav")
    assert voices[-1] == "en-US-AndrewMultilingualNeural"


def test_L4_hinglish_never_runs_kokoro_or_chatterbox(tmp_path, monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    n = _narrator(tmp_path)
    ran: list[str] = []

    def fake_run(name, text, wav, language=None):
        ran.append(name)
        return None

    n._run_engine = fake_run  # type: ignore[method-assign]
    assert n.synthesize_speech("namaste", tmp_path / "x.wav", language="hinglish") is None
    assert set(ran) == {"edge-tts"}  # chain is retried (transformation.tts_retries) but never kokoro/chatterbox


def test_L4_elevenlabs_uses_per_language_voice(tmp_path, monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    used: list[str] = []

    class _FakeEleven:
        def __init__(self, voice_id: str = "", model_id: str = "", **_kw) -> None:
            self.voice_id = voice_id

        def available(self) -> bool:
            return bool(self.voice_id)

        def synthesize(self, text, out_wav):
            used.append(self.voice_id)
            return None

    monkeypatch.setattr(narrator_mod, "ElevenLabsTTS", _FakeEleven)
    n = _narrator(
        tmp_path,
        '  elevenlabs_voice_id: "en_voice"\n  elevenlabs_hinglish_voice_id: "hi_voice"\n',
    )
    n._run_engine("elevenlabs", "hello", tmp_path / "a.wav", language="english")
    n._run_engine("elevenlabs", "bhai", tmp_path / "b.wav", language="hinglish")
    assert used == ["en_voice", "hi_voice"]

    # Hinglish voice alone is enough for the Hinglish chain; English falls back to other engines.
    n2 = _narrator(tmp_path, '  elevenlabs_hinglish_voice_id: "hi_voice"\n')
    assert n2._engine_chain("hinglish")[0] == "elevenlabs"
    assert "elevenlabs" not in n2._engine_chain("english")

    # Unset Hinglish voice falls back to the main voice.
    n3 = _narrator(tmp_path, '  elevenlabs_voice_id: "en_voice"\n')
    assert n3._elevenlabs_voice_for("hinglish") == "en_voice"
