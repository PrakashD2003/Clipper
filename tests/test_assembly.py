"""Tests for Phase 4 video assembly & output modules (Packages 4.3 & 4.6)."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from streamclipper.assembly.captions import (
    AssCaptionBurner,
    _color_to_ass_style,
    _escape_ass_text,
    _escape_ffmpeg_ass_filename,
    _hex_to_ass_bgr,
    _sec_to_ass_time,
)
from streamclipper.assembly.composer import VideoComposer, remap_timeline_for_beats
from streamclipper.assembly.hooks import HookTitleOverlay
from streamclipper.config import Config, HardwareProfile, detect_hardware_profile
from streamclipper.models import ClipMetadata, Highlight, WordTimestamp
from streamclipper.output.exporter import Exporter, _close_clip_handles
from streamclipper.output.organizer import FileOrganizer
from streamclipper.pipeline import Pipeline


def _make_config(
    tmp_path: Path,
    overrides: dict | None = None,
    *,
    video_codec: str = "libx264",
) -> Config:
    """Create a deterministic Config isolated from local hardware detection."""
    profile = HardwareProfile(
        tier="cuda_mid_tensor" if video_codec == "h264_nvenc" else "cpu_only",
        gpu_name="Test GPU" if video_codec == "h264_nvenc" else "CPU",
        vram_gb=6.0 if video_codec == "h264_nvenc" else 0.0,
        compute_capability="8.9" if video_codec == "h264_nvenc" else "N/A",
        has_tensor_cores=video_codec == "h264_nvenc",
        has_nvenc=video_codec == "h264_nvenc",
        device="cuda" if video_codec == "h264_nvenc" else "cpu",
        model_size="base",
        compute_type="int8",
        batch_size=4,
        llm_model="qwen3:4b",
        video_codec=video_codec,
    )
    cfg_path = tmp_path / "user_config.yaml"
    if overrides:
        cfg_path.write_text(yaml.safe_dump(overrides), encoding="utf-8")
        return Config(user_config_path=cfg_path, hardware_profile=profile)
    return Config(user_config_path=tmp_path / "missing.yaml", hardware_profile=profile)


# ---------------------------------------------------------------------------
# Package 4.3 — Helper function unit tests
# ---------------------------------------------------------------------------


class TestSecToAssTime:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0.0, "0:00:00.00"),
            (1.25, "0:00:01.25"),
            (9.99, "0:00:09.99"),
            (59.99, "0:00:59.99"),
            (59.995, "0:01:00.00"),
            (59.999, "0:01:00.00"),
            (60.0, "0:01:00.00"),
            (65.5, "0:01:05.50"),
            (3599.999, "1:00:00.00"),
            (3661.07, "1:01:01.07"),
            (-5.0, "0:00:00.00"),
            (-0.001, "0:00:00.00"),
        ],
    )
    def test_formats_and_avoids_sixty_rollover(self, seconds: float, expected: str) -> None:
        assert _sec_to_ass_time(seconds) == expected

    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_values_clamp_to_zero(self, bad_value: float) -> None:
        assert _sec_to_ass_time(bad_value) == "0:00:00.00"


class TestColorConversion:
    @pytest.mark.parametrize(
        ("hex_color", "expected"),
        [
            ("#FFD700", "&H00D7FF&"),
            ("#ff0000", "&H0000FF&"),
            ("#00FF00", "&H00FF00&"),
            ("#0000FF", "&HFF0000&"),
            ("1a2b3c", "&H3C2B1A&"),
            ("  #ABCDEF  ", "&HEFCDAB&"),
        ],
    )
    def test_hex_to_ass_bgr_valid(self, hex_color: str, expected: str) -> None:
        assert _hex_to_ass_bgr(hex_color) == expected

    @pytest.mark.parametrize(
        "invalid_hex",
        [
            "",
            "   ",
            "#FFF",
            "#12345",
            "#1234567",
            "##FFD700",
            "#GGGGGG",
            "-12345",
            "+12345",
            "12_345",
            "gold",
        ],
    )
    def test_hex_to_ass_bgr_invalid_returns_default(self, invalid_hex: str) -> None:
        assert _hex_to_ass_bgr(invalid_hex) == "&H00D7FF&"
        assert _hex_to_ass_bgr(invalid_hex, default="&HFFFFFF&") == "&HFFFFFF&"

    @pytest.mark.parametrize(
        ("color", "expected"),
        [
            ("white", "&H00FFFFFF"),
            ("  WHITE ", "&H00FFFFFF"),
            ("black", "&H00000000"),
            ("yellow", "&H0000FFFF"),
            ("gold", "&H0000D7FF"),
            ("red", "&H000000FF"),
            ("cyan", "&H00FFFF00"),
            ("#FFD700", "&H0000D7FF"),
            ("#112233", "&H00332211"),
        ],
    )
    def test_color_to_ass_style_named_and_hex(self, color: str, expected: str) -> None:
        assert _color_to_ass_style(color) == expected

    def test_color_to_ass_style_unknown_returns_default(self) -> None:
        assert _color_to_ass_style("chartreuse", default="&H00123456") == "&H00123456"
        assert _color_to_ass_style("", default="&H00000000") == "&H00000000"


class TestEscapeAssText:
    def test_replaces_all_newline_variants_with_spaces(self) -> None:
        assert _escape_ass_text("line1\r\nline2\rline3\nline4") == "line1 line2 line3 line4"

    def test_escapes_braces(self) -> None:
        assert _escape_ass_text(r"{\c&H0000FF&}HACK") == r"\{\c&H0000FF&\}HACK"

    def test_neutralizes_ass_control_codes_without_doubling_literal_backslashes(self) -> None:
        escaped = _escape_ass_text(r"a\Nb\nc\hd C:\test\file")
        assert escaped == "a\\\u200bNb\\\u200bnc\\\u200bhd C:\\test\\file"
        assert r"\\" not in escaped

    def test_empty_string_returns_empty(self) -> None:
        assert _escape_ass_text("") == ""


# ---------------------------------------------------------------------------
# Package 4.3 — AssCaptionBurner.generate_ass tests
# ---------------------------------------------------------------------------


class TestAssGeneration:
    def test_generate_ass_hormozi_style_and_chat_ticker(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        burner = AssCaptionBurner(cfg)
        words = [
            WordTimestamp("oh", 10.0, 10.3),
            WordTimestamp("my", 10.3, 10.6),
            WordTimestamp("god", 10.6, 11.0),
            WordTimestamp("bro", 11.0, 11.4),
            WordTimestamp("no", 11.5, 11.8),
            WordTimestamp("way", 11.8, 12.2),
        ]
        chat_burst = [
            {"author": "Gamer1", "message": "LMAO WHAT", "offset_seconds": 10.5},
            {"author": "Viewer2", "message": "NO WAY {gg} \\N", "offset_seconds": 11.0},
        ]
        ass_path = tmp_path / "sub" / "clip_01.ass"
        result = burner.generate_ass(words, clip_start_offset=10.0, ass_path=ass_path, chat_burst=chat_burst)

        assert result == ass_path
        assert ass_path.is_file()
        content = ass_path.read_text(encoding="utf-8")

        # Header & styles verification
        assert "PlayResX: 1080" in content
        assert "PlayResY: 1920" in content
        assert "WrapStyle: 0" in content
        assert "Style: Hormozi,Arial,72,&H00FFFFFF,&H0000D7FF,&H00000000,&H80000000" in content
        # MarginV = round(1920 * (1.0 - 0.75)) = 480
        assert ",4,0,2,60,60,480,1" in content
        # ChatTicker uses BorderStyle=3 with semi-transparent OutlineColour=&H99000000
        assert (
            "Style: ChatTicker,Arial,34,&H00FFFFFF,&H00FFFFFF,&H99000000,&H99000000,"
            "-1,0,0,0,100,100,0,0,3,2,0,9,40,40,260,1"
        ) in content

        hormozi_lines = [ln for ln in content.splitlines() if ",Hormozi,," in ln]
        chat_lines = [ln for ln in content.splitlines() if ",ChatTicker,," in ln]

        assert len(hormozi_lines) == 6
        # Group 1 (4 words), active word highlighted in gold BGR &H00D7FF& and reset to white &HFFFFFF&
        assert (
            hormozi_lines[0]
            == r"Dialogue: 0,0:00:00.00,0:00:00.30,Hormozi,,0,0,0,,{\c&H00D7FF&}OH{\c&HFFFFFF&} MY GOD BRO"
        )
        assert (
            hormozi_lines[2]
            == r"Dialogue: 0,0:00:00.60,0:00:01.00,Hormozi,,0,0,0,,OH MY {\c&H00D7FF&}GOD{\c&HFFFFFF&} BRO"
        )
        # Group 2 (remaining 2 words)
        assert (
            hormozi_lines[4]
            == r"Dialogue: 0,0:00:01.50,0:00:01.80,Hormozi,,0,0,0,,{\c&H00D7FF&}NO{\c&HFFFFFF&} WAY"
        )
        assert (
            hormozi_lines[5]
            == r"Dialogue: 0,0:00:01.80,0:00:02.20,Hormozi,,0,0,0,,NO {\c&H00D7FF&}WAY{\c&HFFFFFF&}"
        )

        # ChatTicker dialogue events on Layer 1 with 2.5s duration & escaped characters
        assert len(chat_lines) == 2
        assert (
            chat_lines[0]
            == "Dialogue: 1,0:00:00.50,0:00:03.00,ChatTicker,,0,0,0,,Gamer1: LMAO WHAT"
        )
        assert (
            chat_lines[1]
            == "Dialogue: 1,0:00:01.00,0:00:03.50,ChatTicker,,0,0,0,,Viewer2: NO WAY \\{gg\\} \\\u200bN"
        )

    def test_generate_ass_suppresses_streamer_words_during_narration_and_emits_narration_cards(
        self, tmp_path: Path
    ) -> None:
        """Streamer words falling inside ducked narration windows are suppressed; narration cards are emitted."""
        cfg = _make_config(tmp_path)
        burner = AssCaptionBurner(cfg)
        words = [
            # Words in intro window [0..2.0s] relative to offset 100.0:
            WordTimestamp("intro", 100.0, 100.8),
            WordTimestamp("ducked", 100.8, 101.5),
            # Climax streamer speech [3.0..7.0s]:
            WordTimestamp("climax", 103.0, 103.5),
            WordTimestamp("reaction", 103.5, 104.0),
            # Words in outro window [8.0..10.0s]:
            WordTimestamp("outro", 108.5, 109.5),
        ]
        ass_path = tmp_path / "sub" / "narrated.ass"
        burner.generate_ass(
            words=words,
            clip_start_offset=100.0,
            ass_path=ass_path,
            narration_intro="Look at this insane move",
            narration_outro="He actually pulled it off",
            intro_dur=2.0,
            outro_dur=2.0,
            clip_duration=10.0,
        )

        content = ass_path.read_text(encoding="utf-8")
        hormozi_lines = [ln for ln in content.splitlines() if ",Hormozi,," in ln]

        # Intro narration card emitted at 0.0 -> 2.0s
        assert any(
            "0:00:00.00,0:00:02.00,Hormozi" in ln and "LOOK AT THIS INSANE MOVE" in ln
            for ln in hormozi_lines
        )
        # Outro narration card emitted at 8.0 -> 10.0s (10.0 - 2.0 = 8.0s)
        assert any(
            "0:00:08.00,0:00:10.00,Hormozi" in ln and "HE ACTUALLY PULLED IT OFF" in ln
            for ln in hormozi_lines
        )

        # Climax streamer words ("CLIMAX", "REACTION") are present
        assert any("CLIMAX" in ln for ln in hormozi_lines)
        assert any("REACTION" in ln for ln in hormozi_lines)

        # Ducked words ("intro ducked", "outro") are suppressed from streamer captions
        assert not any("DUCKED" in ln for ln in hormozi_lines)

    def test_generate_ass_accepts_raw_dict_words(self, tmp_path: Path) -> None:
        """generate_ass gracefully accepts dictionary word mappings from WhisperX."""
        cfg = _make_config(tmp_path)
        burner = AssCaptionBurner(cfg)
        dict_words = [
            {"word": "whisper", "start": 10.0, "end": 10.5, "score": 0.9},
            {"text": "dictionary", "start": 10.5, "end": 11.0, "confidence": 0.95},
        ]
        ass_path = tmp_path / "dict_words.ass"
        burner.generate_ass(
            words=dict_words,
            clip_start_offset=10.0,
            ass_path=ass_path,
        )
        content = ass_path.read_text(encoding="utf-8")
        assert "WHISPER" in content
        assert "DICTIONARY" in content

    def test_custom_colors_font_and_group_size(self, tmp_path: Path) -> None:
        cfg = _make_config(
            tmp_path,
            overrides={
                "captions": {
                    "style": "hormozi",
                    "words_per_group": 3,
                    "font": "Impact",
                    "font_size": 80,
                    "color": "#00FF00",
                    "highlight_color": "#FF00AA",
                    "stroke_color": "red",
                    "stroke_width": 6,
                    "position_y_ratio": 0.80,
                }
            },
        )
        burner = AssCaptionBurner(cfg)
        words = [
            WordTimestamp("a", 0.0, 0.5),
            WordTimestamp("b", 0.5, 1.0),
            WordTimestamp("c", 1.0, 1.5),
            WordTimestamp("d", 1.5, 2.0),
        ]
        ass_path = tmp_path / "custom.ass"
        burner.generate_ass(words, clip_start_offset=0.0, ass_path=ass_path)
        content = ass_path.read_text(encoding="utf-8")

        # #00FF00 -> &H0000FF00, #FF00AA -> &HAA00FF& (style &H00AA00FF), red -> &H000000FF
        # margin_v = round(1920 * 0.20) = 384
        assert (
            "Style: Hormozi,Impact,80,&H0000FF00,&H00AA00FF,&H000000FF,&H80000000,"
            "-1,0,0,0,100,100,0,0,1,6,0,2,60,60,384,1"
        ) in content

        hormozi_lines = [ln for ln in content.splitlines() if ",Hormozi,," in ln]
        assert len(hormozi_lines) == 4
        assert (
            hormozi_lines[0]
            == r"Dialogue: 0,0:00:00.00,0:00:00.50,Hormozi,,0,0,0,,{\c&HAA00FF&}A{\c&H00FF00&} B C"
        )
        assert (
            hormozi_lines[3]
            == r"Dialogue: 0,0:00:01.50,0:00:02.00,Hormozi,,0,0,0,,{\c&HAA00FF&}D{\c&H00FF00&}"
        )

    def test_single_word_style(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, overrides={"captions": {"style": "single_word", "words_per_group": 5}})
        burner = AssCaptionBurner(cfg)
        words = [
            WordTimestamp("boom", 5.0, 5.4),
            WordTimestamp("headshot", 5.4, 6.0),
        ]
        ass_path = tmp_path / "single.ass"
        burner.generate_ass(words, clip_start_offset=5.0, ass_path=ass_path)
        hormozi_lines = [
            ln for ln in ass_path.read_text(encoding="utf-8").splitlines() if ",Hormozi,," in ln
        ]
        assert hormozi_lines == [
            r"Dialogue: 0,0:00:00.00,0:00:00.40,Hormozi,,0,0,0,,{\c&H00D7FF&}BOOM{\c&HFFFFFF&}",
            r"Dialogue: 0,0:00:00.40,0:00:01.00,Hormozi,,0,0,0,,{\c&H00D7FF&}HEADSHOT{\c&HFFFFFF&}",
        ]

    def test_sentence_style(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, overrides={"captions": {"style": "sentence", "words_per_group": 3}})
        burner = AssCaptionBurner(cfg)
        words = [
            WordTimestamp("what", 20.0, 20.3),
            WordTimestamp("just", 20.3, 20.6),
            WordTimestamp("happened", 20.6, 21.2),
            WordTimestamp("here", 21.5, 22.0),
        ]
        ass_path = tmp_path / "sentence.ass"
        burner.generate_ass(words, clip_start_offset=20.0, ass_path=ass_path)
        hormozi_lines = [
            ln for ln in ass_path.read_text(encoding="utf-8").splitlines() if ",Hormozi,," in ln
        ]
        assert hormozi_lines == [
            "Dialogue: 0,0:00:00.00,0:00:01.20,Hormozi,,0,0,0,,WHAT JUST HAPPENED",
            "Dialogue: 0,0:00:01.50,0:00:02.00,Hormozi,,0,0,0,,HERE",
        ]

    def test_chat_ticker_caps_at_five_and_truncates_and_skips_blanks(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        burner = AssCaptionBurner(cfg)
        chat_burst = [
            {"author": "Empty", "message": "   ", "offset_seconds": 10.0},
            {"author": "A" * 30, "message": "M" * 80, "offset_seconds": 10.0},
            {"author": "", "message": "msg2", "offset_seconds": None},
            {"author": "u3", "message": "msg3", "offset_seconds": 5.0},  # before clip_start_offset -> 0.0
            {"author": "u4", "message": "msg4", "offset_seconds": "invalid"},
            {"author": "u5", "message": "msg5", "offset_seconds": 12.0},
            {"author": "u6", "message": "msg6 should be excluded", "offset_seconds": 13.0},
            {"author": "u7", "message": "msg7 should be excluded", "offset_seconds": 14.0},
        ]
        ass_path = tmp_path / "chat_cap.ass"
        burner.generate_ass([], clip_start_offset=10.0, ass_path=ass_path, chat_burst=chat_burst)
        chat_lines = [
            ln for ln in ass_path.read_text(encoding="utf-8").splitlines() if ",ChatTicker,," in ln
        ]

        assert len(chat_lines) == 5
        assert f"{'A' * 20}: {'M' * 60}" in chat_lines[0]
        assert "viewer: msg2" in chat_lines[1]
        assert "Dialogue: 1,0:00:00.00,0:00:02.50,ChatTicker,,0,0,0,,u3: msg3" == chat_lines[2]
        assert "Dialogue: 1,0:00:00.00,0:00:02.50,ChatTicker,,0,0,0,,u4: msg4" == chat_lines[3]
        assert "Dialogue: 1,0:00:02.00,0:00:04.50,ChatTicker,,0,0,0,,u5: msg5" == chat_lines[4]

    def test_chat_ticker_disabled_by_config(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, overrides={"transformation": {"show_chat_ticker": False}})
        burner = AssCaptionBurner(cfg)
        chat_burst = [{"author": "User", "message": "POGGERS", "offset_seconds": 1.0}]
        ass_path = tmp_path / "no_chat.ass"
        burner.generate_ass(
            [WordTimestamp("hello", 0.0, 0.5)],
            clip_start_offset=0.0,
            ass_path=ass_path,
            chat_burst=chat_burst,
        )
        content = ass_path.read_text(encoding="utf-8")
        assert ",ChatTicker,," not in content
        assert ",Hormozi,," in content

    def test_word_escaping_and_edge_cases(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        burner = AssCaptionBurner(cfg)
        words = [
            WordTimestamp("   ", 0.0, 0.1),  # whitespace-only skipped
            WordTimestamp("{wow}\n", 5.0, 4.5),  # inverted end clamped to start
            WordTimestamp("back\\", 5.5, 6.0),  # trailing backslash must not escape closing tag
            WordTimestamp(r"a\nb", 6.0, 6.5),  # uppercased -> A\NB -> neutralized
        ]
        ass_path = tmp_path / "edge.ass"
        burner.generate_ass(words, clip_start_offset=10.0, ass_path=ass_path)
        hormozi_lines = [
            ln for ln in ass_path.read_text(encoding="utf-8").splitlines() if ",Hormozi,," in ln
        ]
        assert len(hormozi_lines) == 3
        # All timestamps clamped safely to 0:00:00.00
        assert hormozi_lines[0].startswith("Dialogue: 0,0:00:00.00,0:00:00.00,Hormozi,,")
        # Escaped braces
        assert r"\{WOW\}" in hormozi_lines[0]
        # Trailing backslash separated from closing {\c&HFFFFFF&} by ZWSP
        assert "BACK\\\u200b{\\c&HFFFFFF&}" in hormozi_lines[1]
        # \N neutralized without doubling backslash
        assert "A\\\u200bNB" in hormozi_lines[2]

    def test_chat_ticker_accepts_text_key_and_timestamp_variations(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        burner = AssCaptionBurner(cfg)
        chat_burst = [
            {"author": "ChatterA", "text": "USING TEXT KEY", "timestamp": 11.0},
            {"author": "ChatterB", "message": "", "text": "EMPTY MESSAGE FALLBACK", "time": 12.0},
            {"author": "ChatterC", "message": "OFFSET KEY", "offset": 13.5},
        ]
        ass_path = tmp_path / "text_key.ass"
        burner.generate_ass([], clip_start_offset=10.0, ass_path=ass_path, chat_burst=chat_burst)
        chat_lines = [
            ln for ln in ass_path.read_text(encoding="utf-8").splitlines() if ",ChatTicker,," in ln
        ]
        assert chat_lines == [
            "Dialogue: 1,0:00:01.00,0:00:03.50,ChatTicker,,0,0,0,,ChatterA: USING TEXT KEY",
            "Dialogue: 1,0:00:02.00,0:00:04.50,ChatTicker,,0,0,0,,ChatterB: EMPTY MESSAGE FALLBACK",
            "Dialogue: 1,0:00:03.50,0:00:06.00,ChatTicker,,0,0,0,,ChatterC: OFFSET KEY",
        ]

    def test_defensive_init_handles_null_and_invalid_config_values(self, tmp_path: Path) -> None:
        cfg = _make_config(
            tmp_path,
            overrides={
                "captions": {
                    "style": "unsupported_style",
                    "words_per_group": None,
                    "font_size": None,
                    "stroke_width": None,
                    "position_y_ratio": None,
                },
            },
        )
        burner = AssCaptionBurner(cfg)
        assert burner.style == "hormozi"
        assert burner.words_per_group == 4
        assert burner.font_size == 72
        assert burner.stroke_width == 4
        assert (burner.width, burner.height) == (1080, 1920)
        assert burner.margin_v == 480

        ass_path = tmp_path / "safe_defaults.ass"
        burner.generate_ass([WordTimestamp("ok", 0.0, 0.5)], 0.0, ass_path)
        content = ass_path.read_text(encoding="utf-8")
        assert "None" not in content
        assert (
            "Style: Hormozi,Arial,72,&H00FFFFFF,&H0000D7FF,&H00000000,&H80000000,"
            "-1,0,0,0,100,100,0,0,1,4,0,2,60,60,480,1"
        ) in content

    def test_integrates_with_composer_remap_timeline_for_beats(self, tmp_path: Path) -> None:
        """Multi-beat remapped words and chat_burst render onto stitched 0.0-based ASS timeline."""
        cfg = _make_config(tmp_path)
        burner = AssCaptionBurner(cfg)
        beats = [
            {"start": 100.0, "end": 105.0},
            {"start": 120.0, "end": 125.0},
        ]
        words = [
            WordTimestamp("setup", 101.0, 102.0),
            WordTimestamp("dead_air", 110.0, 111.0),
            WordTimestamp("payoff", 121.0, 122.0),
        ]
        chat_burst = [
            {"author": "V1", "message": "wait for it", "offset_seconds": 102.0},
            {"author": "V2", "message": "cut air", "offset_seconds": 112.0},
            {"author": "V3", "text": "WHOA!", "timestamp": 122.0},
        ]
        remapped_words, remapped_chat, peak_offset = remap_timeline_for_beats(
            words, chat_burst, peak_time=122.5, beats=beats
        )
        assert peak_offset == pytest.approx(7.5)

        ass_path = tmp_path / "stitched.ass"
        burner.generate_ass(remapped_words, clip_start_offset=0.0, ass_path=ass_path, chat_burst=remapped_chat)
        content = ass_path.read_text(encoding="utf-8")
        hormozi_lines = [ln for ln in content.splitlines() if ",Hormozi,," in ln]
        chat_lines = [ln for ln in content.splitlines() if ",ChatTicker,," in ln]

        assert len(hormozi_lines) == 2
        assert "DEAD_AIR" not in content
        assert hormozi_lines[0].startswith("Dialogue: 0,0:00:01.00,0:00:02.00,Hormozi,,")
        assert hormozi_lines[1].startswith("Dialogue: 0,0:00:06.00,0:00:07.00,Hormozi,,")
        assert chat_lines == [
            "Dialogue: 1,0:00:02.00,0:00:04.50,ChatTicker,,0,0,0,,V1: wait for it",
            "Dialogue: 1,0:00:07.00,0:00:09.50,ChatTicker,,0,0,0,,V3: WHOA!",
        ]


# ---------------------------------------------------------------------------
# Package 4.3 — FFmpeg command building & Windows-safe burn_captions tests
# ---------------------------------------------------------------------------


class TestFfmpegBurnCaptions:
    def test_build_ffmpeg_cmd_libx264(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, video_codec="libx264")
        burner = AssCaptionBurner(cfg)
        in_vid = tmp_path / "in.mp4"
        out_vid = tmp_path / "out.mp4"

        cmd = burner._build_ffmpeg_cmd(in_vid, "clip_01.ass", out_vid, "libx264", 23)
        assert cmd == [
            "ffmpeg",
            "-y",
            "-i",
            str(in_vid.resolve()),
            "-vf",
            "ass=clip_01.ass",
            "-c:v",
            "libx264",
            "-crf",
            "23",
            "-c:a",
            "copy",
            str(out_vid.resolve()),
        ]

    def test_build_ffmpeg_cmd_nvenc(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, video_codec="h264_nvenc")
        burner = AssCaptionBurner(cfg)
        in_vid = tmp_path / "in.mp4"
        out_vid = tmp_path / "out.mp4"

        cmd = burner._build_ffmpeg_cmd(in_vid, "clip_01.ass", out_vid, "h264_nvenc", 20)
        assert cmd == [
            "ffmpeg",
            "-y",
            "-i",
            str(in_vid.resolve()),
            "-vf",
            "ass=clip_01.ass",
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-cq",
            "20",
            "-c:a",
            "copy",
            str(out_vid.resolve()),
        ]

    def test_burn_captions_uses_windows_safe_cwd_and_relative_ass_name(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, video_codec="libx264")
        burner = AssCaptionBurner(cfg)

        in_vid = tmp_path / "videos" / "raw.mp4"
        ass_dir = tmp_path / "subs dir"
        ass_path = ass_dir / "captions.ass"
        out_vid = tmp_path / "final" / "clip_01.mp4"

        with patch("streamclipper.assembly.captions.subprocess.run") as mock_run:
            result = burner.burn_captions(in_vid, ass_path, out_vid, crf=21)

        assert result == out_vid
        assert out_vid.parent.is_dir()
        mock_run.assert_called_once()
        cmd = mock_run.call_args.args[0]
        kwargs = mock_run.call_args.kwargs

        assert kwargs["cwd"] == str(ass_path.resolve().parent)
        assert kwargs["check"] is True
        assert kwargs["capture_output"] is True
        vf_index = cmd.index("-vf")
        # Relative filename only — zero Windows drive-letter colon inside ass=...
        assert cmd[vf_index + 1] == "ass=captions.ass"
        assert ":" not in cmd[vf_index + 1]
        assert cmd[cmd.index("-i") + 1] == str(in_vid.resolve())
        assert cmd[-1] == str(out_vid.resolve())
        assert "-c:a" in cmd and cmd[cmd.index("-c:a") + 1] == "copy"
        assert "-crf" in cmd and cmd[cmd.index("-crf") + 1] == "21"

    def test_burn_captions_nvenc_falls_back_to_libx264_on_failure(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, video_codec="h264_nvenc")
        burner = AssCaptionBurner(cfg)

        in_vid = tmp_path / "raw.mp4"
        ass_path = tmp_path / "clip.ass"
        out_vid = tmp_path / "out.mp4"

        side_effects = [
            subprocess.CalledProcessError(returncode=1, cmd=["ffmpeg"], stderr=b"NVENC init failed"),
            subprocess.CompletedProcess(args=["ffmpeg"], returncode=0),
        ]
        with patch("streamclipper.assembly.captions.subprocess.run", side_effect=side_effects) as mock_run:
            result = burner.burn_captions(in_vid, ass_path, out_vid, crf=22)

        assert result == out_vid
        assert mock_run.call_count == 2

        first_cmd = mock_run.call_args_list[0].args[0]
        second_cmd = mock_run.call_args_list[1].args[0]
        assert "h264_nvenc" in first_cmd
        assert "-cq" in first_cmd and first_cmd[first_cmd.index("-cq") + 1] == "22"
        assert "libx264" in second_cmd
        assert "-crf" in second_cmd and second_cmd[second_cmd.index("-crf") + 1] == "22"
        assert mock_run.call_args_list[0].kwargs["cwd"] == str(ass_path.resolve().parent)
        assert mock_run.call_args_list[1].kwargs["cwd"] == str(ass_path.resolve().parent)

    def test_burn_captions_libx264_error_reraises_without_retry(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, video_codec="libx264")
        burner = AssCaptionBurner(cfg)

        err = subprocess.CalledProcessError(returncode=1, cmd=["ffmpeg"], stderr=b"bad input")
        with patch("streamclipper.assembly.captions.subprocess.run", side_effect=err) as mock_run:
            with pytest.raises(subprocess.CalledProcessError):
                burner.burn_captions(tmp_path / "in.mp4", tmp_path / "a.ass", tmp_path / "out.mp4")

        assert mock_run.call_count == 1

    def test_burn_captions_codec_override_argument(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, video_codec="libx264")
        burner = AssCaptionBurner(cfg)

        with patch("streamclipper.assembly.captions.subprocess.run") as mock_run:
            burner.burn_captions(
                tmp_path / "in.mp4",
                tmp_path / "a.ass",
                tmp_path / "out.mp4",
                crf=19,
                codec="h264_nvenc",
            )

        cmd = mock_run.call_args.args[0]
        assert "h264_nvenc" in cmd
        assert "-preset" in cmd and cmd[cmd.index("-preset") + 1] == "p4"
        assert "-cq" in cmd and cmd[cmd.index("-cq") + 1] == "19"

    def test_build_ffmpeg_cmd_normalizes_codec_case_and_escapes_special_ass_name(
        self, tmp_path: Path
    ) -> None:
        cfg = _make_config(tmp_path, video_codec="libx264")
        burner = AssCaptionBurner(cfg)
        assert _escape_ffmpeg_ass_filename("clip_01.ass") == "clip_01.ass"
        assert _escape_ffmpeg_ass_filename("clip,01.ass") == r"clip\,01.ass"
        assert _escape_ffmpeg_ass_filename("clip[1];=a.ass") == r"clip\[1\]\;\\=a.ass"

        cmd = burner._build_ffmpeg_cmd(
            tmp_path / "in.mp4", "clip,01.ass", tmp_path / "out.mp4", "H264_NVENC", 21
        )
        assert "h264_nvenc" in cmd
        assert "-cq" in cmd and cmd[cmd.index("-cq") + 1] == "21"
        assert cmd[cmd.index("-vf") + 1] == r"ass=clip\,01.ass"

    def test_burn_captions_rejects_same_input_and_output_path(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, video_codec="libx264")
        burner = AssCaptionBurner(cfg)
        same_path = tmp_path / "video.mp4"
        with pytest.raises(ValueError, match="different paths"):
            burner.burn_captions(same_path, tmp_path / "clip.ass", same_path)


# ---------------------------------------------------------------------------
# Package 4.3 — Real FFmpeg libass integration test (runs when ffmpeg is installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available on PATH")
class TestRealFfmpegLibassIntegration:
    def test_end_to_end_real_ffmpeg_ass_burn_in(self, tmp_path: Path) -> None:
        """Generate a real tiny MP4 + ASS file (Hormozi + ChatTicker) and burn via real FFmpeg."""
        in_video = tmp_path / "synthetic_input.mp4"
        out_video = tmp_path / "output" / "burned_clip.mp4"
        ass_path = tmp_path / "subtitles" / "clip_01.ass"

        # Synthesize a 1.5s 180x320 MP4 with AAC audio
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=navy:s=180x320:d=1.5:r=15",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=1.5",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                str(in_video),
            ],
            check=True,
            capture_output=True,
        )

        cfg = _make_config(
            tmp_path,
            overrides={"video": {"resolution": [180, 320]}},
            video_codec="libx264",
        )
        burner = AssCaptionBurner(cfg)
        words = [
            WordTimestamp("watch", 10.0, 10.4),
            WordTimestamp("{this}", 10.4, 10.8),
            WordTimestamp(r"C:\new", 10.8, 11.4),
        ]
        chat_burst = [
            {"author": "Chatter", "message": "INSANE {clip} \\N", "offset_seconds": 10.2},
        ]
        burner.generate_ass(words, clip_start_offset=10.0, ass_path=ass_path, chat_burst=chat_burst)

        result = burner.burn_captions(in_video, ass_path, out_video, crf=28, codec="libx264")
        assert result == out_video
        assert out_video.is_file()
        assert out_video.stat().st_size > 1000

        if shutil.which("ffprobe"):
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "stream=codec_type,codec_name",
                    "-of",
                    "csv=p=0",
                    str(out_video),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            streams = {line.strip() for line in probe.stdout.splitlines() if line.strip()}
            assert "h264,video" in streams or "video,h264" in streams
            assert "aac,audio" in streams or "audio,aac" in streams

    def test_real_ffmpeg_handles_special_filtergraph_chars_in_ass_filename(
        self, tmp_path: Path
    ) -> None:
        """ASS filenames with commas, quotes, brackets, semicolons, =, and leading spaces burn cleanly."""
        in_video = tmp_path / "input.mp4"
        out_video = tmp_path / "output.mp4"
        ass_path = tmp_path / " clip, '[1];=a.ass"

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=180x320:d=0.6:r=10",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(in_video),
            ],
            check=True,
            capture_output=True,
        )

        cfg = _make_config(
            tmp_path,
            overrides={"video": {"resolution": [180, 320]}},
            video_codec="libx264",
        )
        burner = AssCaptionBurner(cfg)
        burner.generate_ass([WordTimestamp("hello", 0.0, 0.5)], 0.0, ass_path)

        result = burner.burn_captions(in_video, ass_path, out_video, crf=28, codec="libx264")
        assert result == out_video
        assert out_video.is_file()
        assert out_video.stat().st_size > 500

    def test_real_ffmpeg_nvenc_burn_in_when_gpu_available(self, tmp_path: Path) -> None:
        """Verify live h264_nvenc (-preset p4 -cq) burn-in directly on NVENC-capable hosts."""
        hw = detect_hardware_profile()
        if not hw.has_nvenc:
            pytest.skip("No NVENC-capable GPU detected on this host")

        in_video = tmp_path / "nvenc_in.mp4"
        out_video = tmp_path / "nvenc_out.mp4"
        ass_path = tmp_path / "nvenc.ass"

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=navy:s=180x320:d=1.0:r=15",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=1.0",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                str(in_video),
            ],
            check=True,
            capture_output=True,
        )

        cfg = _make_config(
            tmp_path,
            overrides={"video": {"resolution": [180, 320]}},
            video_codec="h264_nvenc",
        )
        burner = AssCaptionBurner(cfg)
        burner.generate_ass(
            [WordTimestamp("nvenc", 0.0, 0.5), WordTimestamp("works", 0.5, 1.0)],
            clip_start_offset=0.0,
            ass_path=ass_path,
            chat_burst=[{"author": "GPU", "text": "FAST ENCODE", "offset_seconds": 0.1}],
        )

        # Verify direct NVENC command (not silent fallback) succeeds on real hardware
        cmd = burner._build_ffmpeg_cmd(in_video, ass_path.name, out_video, "h264_nvenc", 23)
        subprocess.run(cmd, cwd=str(ass_path.resolve().parent), check=True, capture_output=True)
        assert out_video.is_file()
        assert out_video.stat().st_size > 1000


# ---------------------------------------------------------------------------
# Package 4.6 — FileOrganizer unit tests
# ---------------------------------------------------------------------------


def _sample_clip_metadata(**overrides: Any) -> ClipMetadata:
    base: dict[str, Any] = {
        "hook_title": "INSANE ROOFTOP ESCAPE!",
        "caption": "TimTheTatman barely survives a 5-star chase in GTA 6!",
        "hashtags": ["GTA6", "TimTheTatman", "Gaming"],
        "streamer": "TimTheTatman",
        "game": "GTA 6",
        "source_url": "https://youtube.com/watch?v=gta6_test",
        "clip_start": 100.0,
        "clip_end": 142.5,
        "highlight_score": 9.3,
        "moment_type": "clutch",
        "narration_intro": "Five stars on his tail and zero exits left.",
        "narration_outro": "Somehow he clears six cop cars without a scratch.",
        "context_badge": "Clutch Escape",
        "transformation_mode": "story",
    }
    base.update(overrides)
    return ClipMetadata(**base)


class TestFileOrganizer:
    @pytest.mark.parametrize(
        ("raw", "fallback", "expected"),
        [
            ("GTA 6", "unknown", "GTA 6"),
            ('GTA: 6 / "Vice" *City?<>|', "unknown", "GTA 6 Vice City"),
            ("  ..TimTheTatman..  ", "unknown", "TimTheTatman"),
            ("foo\\bar:baz", "unknown", "foobarbaz"),
            ("   ...   ", "UnknownGame", "UnknownGame"),
            ('<>:"/\\|?*', "fallback_name", "fallback_name"),
            ("", "unknown", "unknown"),
        ],
    )
    def test_sanitize_name_strips_forbidden_chars_and_dots(
        self, raw: str, fallback: str, expected: str
    ) -> None:
        assert FileOrganizer._sanitize_name(raw, fallback=fallback) == expected

    def test_get_clip_path_default_structure_and_sanitization(self, tmp_path: Path) -> None:
        cfg = _make_config(
            tmp_path,
            overrides={"general": {"output_dir": str(tmp_path / "out")}},
        )
        organizer = FileOrganizer(cfg)

        clip_path = organizer.get_clip_path(
            game='GTA: 6 / "Vice"?',
            streamer="  ..Tim*The|Tatman.. ",
            index=3,
            date_str="2026-11-18",
        )
        expected = tmp_path / "out" / "GTA 6 Vice" / "TimTheTatman" / "2026-11-18" / "clip_03.mp4"
        assert clip_path == expected
        assert clip_path.parent.is_dir()

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("CON", "CON_"),
            ("nul", "nul_"),
            ("AUX", "AUX_"),
            ("prn", "prn_"),
            ("COM1", "COM1_"),
            ("lpt9", "lpt9_"),
            ("con.txt", "con.txt_"),
        ],
    )
    def test_sanitize_name_and_get_clip_path_guard_windows_reserved_names(
        self, tmp_path: Path, raw: str, expected: str
    ) -> None:
        assert FileOrganizer._sanitize_name(raw) == expected
        cfg = _make_config(
            tmp_path,
            overrides={"general": {"output_dir": str(tmp_path / "out")}},
        )
        organizer = FileOrganizer(cfg)
        clip_path = organizer.get_clip_path(
            game=raw,
            streamer="NUL",
            index=1,
            date_str="2026-11-18",
        )
        assert clip_path.parent.is_dir()
        clip_path.write_bytes(b"ok")
        assert clip_path.is_file()

    def test_get_clip_path_custom_templates_and_traversal_protection(self, tmp_path: Path) -> None:
        cfg = _make_config(
            tmp_path,
            overrides={
                "general": {"output_dir": str(tmp_path / "out")},
                "output": {
                    "structure": "{streamer}/{game}",
                    "filename_template": "short_{index:03d}",
                },
            },
        )
        organizer = FileOrganizer(cfg)
        clip_path = organizer.get_clip_path("GTA 6", "KaiCenat", 7, date_str="2026-11-18")
        assert clip_path == tmp_path / "out" / "KaiCenat" / "GTA 6" / "short_007.mp4"

        # Path traversal & malformed templates must stay strictly inside output_dir
        bad_cfg = _make_config(
            tmp_path,
            overrides={
                "general": {"output_dir": str(tmp_path / "safe_out")},
                "output": {
                    "structure": "../../{bad_key}/../{game}",
                    "filename_template": "../../evil_{missing}",
                },
            },
        )
        safe_org = FileOrganizer(bad_cfg)
        safe_path = safe_org.get_clip_path("GTA 6", "Streamer", 1, date_str="2026-11-18")
        assert (tmp_path / "safe_out").resolve() in safe_path.resolve().parents
        assert safe_path.name == "clip_01.mp4"

    def test_save_clip_writes_metadata_json_with_ypp_fields_and_transcript(
        self, tmp_path: Path
    ) -> None:
        cfg = _make_config(
            tmp_path,
            overrides={"general": {"output_dir": str(tmp_path / "out")}},
        )
        organizer = FileOrganizer(cfg)
        video_path = organizer.get_clip_path("GTA 6", "TimTheTatman", 1, date_str="2026-11-18")
        video_path.write_bytes(b"fake-mp4-bytes")

        meta = _sample_clip_metadata()
        result = organizer.save_clip(
            video_path=video_path,
            metadata=meta,
            transcript_text="WATCH THIS JUMP BRO!",
        )

        assert result.video_path == video_path
        assert result.metadata_path == video_path.with_suffix(".json")
        assert result.metadata_path is not None and result.metadata_path.is_file()
        assert result.duration_seconds == pytest.approx(42.5)

        saved_json = json.loads(result.metadata_path.read_text(encoding="utf-8"))
        assert saved_json["hook_title"] == "INSANE ROOFTOP ESCAPE!"
        assert saved_json["moment_type"] == "clutch"
        assert saved_json["narration_intro"] == "Five stars on his tail and zero exits left."
        assert saved_json["narration_outro"] == "Somehow he clears six cop cars without a scratch."
        assert saved_json["context_badge"] == "Clutch Escape"
        assert saved_json["transformation_mode"] == "story"
        assert saved_json["hashtags"] == ["GTA6", "TimTheTatman", "Gaming"]

        transcript_path = video_path.with_name("clip_01_transcript.txt")
        assert transcript_path.is_file()
        assert transcript_path.read_text(encoding="utf-8") == "WATCH THIS JUMP BRO!"

    def test_save_clip_respects_disabled_json_and_transcript_flags(self, tmp_path: Path) -> None:
        cfg = _make_config(
            tmp_path,
            overrides={
                "general": {"output_dir": str(tmp_path / "out")},
                "output": {
                    "save_metadata_json": False,
                    "save_transcript_txt": False,
                },
            },
        )
        organizer = FileOrganizer(cfg)
        video_path = organizer.get_clip_path("GTA 6", "Tim", 2, date_str="2026-11-18")
        video_path.write_bytes(b"mp4")

        result = organizer.save_clip(video_path, _sample_clip_metadata(), transcript_text="hello")
        assert result.metadata_path is None
        assert not video_path.with_suffix(".json").exists()
        assert not video_path.with_name("clip_02_transcript.txt").exists()


# ---------------------------------------------------------------------------
# Package 4.6 — Exporter unit tests (encode params, ducking, ASS, handle cleanup)
# ---------------------------------------------------------------------------


class _TrackedHandle:
    """Fake MoviePy VideoFileClip handle recording close order."""

    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events
        self.closed = False

    def close(self) -> None:
        self.closed = True
        self.events.append(f"close:{self.name}")


class _FakeCompositeClip:
    """Fake MoviePy CompositeVideoClip capturing write_videofile kwargs and handle cleanup order."""

    def __init__(
        self,
        events: list[str],
        *,
        duration: float = 12.0,
        has_audio: bool = True,
        source_clips: list[_TrackedHandle] | None = None,
        write_error: Exception | None = None,
    ) -> None:
        self.events = events
        self.duration = duration
        self.audio = object() if has_audio else None
        self.closed = False
        self.write_calls: list[tuple[str, dict[str, Any]]] = []
        self._write_error = write_error
        self._source_clips = source_clips or []
        self._source_clip = self._source_clips[0] if self._source_clips else None

    def write_videofile(self, filename: str, **kwargs: Any) -> None:
        self.events.append("write_videofile")
        self.write_calls.append((filename, kwargs))
        if self._write_error is not None:
            raise self._write_error
        Path(filename).write_bytes(b"intermediate-mp4")

    def close(self) -> None:
        self.closed = True
        self.events.append("close:composite")


class TestExporter:
    def test_export_ultrafast_crf18_narrator_ducking_ass_burn_and_handle_cleanup_order(
        self, tmp_path: Path
    ) -> None:
        cfg = _make_config(
            tmp_path,
            overrides={
                "general": {"temp_dir": str(tmp_path / "temp")},
                "video": {"fps": 30, "crf": 21, "audio_codec": "aac"},
            },
            video_codec="h264_nvenc",
        )
        exporter = Exporter(cfg)

        events: list[str] = []
        src1 = _TrackedHandle("src1", events)
        src2 = _TrackedHandle("src2", events)
        fake_clip = _FakeCompositeClip(events, duration=15.0, source_clips=[src1, src2])

        user_intro = tmp_path / "human_intro.wav"
        user_intro.write_bytes(b"RIFF....WAVE")
        output_path = tmp_path / "out" / "clip_01.mp4"

        def fake_prepare(
            *,
            narration_intro: str,
            narration_outro: str,
            voice_intro_path: Path | None,
            voice_outro_path: Path | None,
            temp_dir: Path | None,
        ) -> tuple[Path | None, Path | None]:
            events.append("narrator:prepare")
            assert fake_clip.closed and src1.closed and src2.closed
            assert temp_dir is not None and temp_dir.is_dir()
            outro_synth = temp_dir / "narrator_outro.wav"
            outro_synth.write_bytes(b"synth-outro")
            return voice_intro_path, outro_synth

        def fake_mix(
            *,
            input_video: Path,
            output_video: Path,
            intro_wav: Path | None,
            outro_wav: Path | None,
            clip_duration: float,
        ) -> Path:
            events.append("narrator:mix")
            assert input_video.is_file()
            assert intro_wav == user_intro
            assert outro_wav is not None and outro_wav.is_file()
            assert clip_duration == pytest.approx(15.0)
            output_video.write_bytes(b"mixed-mp4")
            return output_video

        burn_calls: list[dict[str, Any]] = []
        gen_ass_calls: list[dict[str, Any]] = []

        def fake_gen_ass(
            *,
            words: list[WordTimestamp],
            clip_start_offset: float,
            ass_path: Path,
            chat_burst: list[dict[str, Any]] | None,
        ) -> Path:
            events.append("captions:generate_ass")
            gen_ass_calls.append(
                {
                    "words": words,
                    "clip_start_offset": clip_start_offset,
                    "ass_path": ass_path,
                    "chat_burst": chat_burst,
                }
            )
            ass_path.write_text("[Script Info]\n", encoding="utf-8")
            return ass_path

        def fake_burn(
            *,
            input_video: Path,
            ass_path: Path,
            output_video: Path,
            crf: int,
            codec: str,
        ) -> Path:
            events.append("captions:burn")
            burn_calls.append(
                {
                    "input_video": input_video,
                    "ass_path": ass_path,
                    "output_video": output_video,
                    "crf": crf,
                    "codec": codec,
                }
            )
            output_video.write_bytes(b"final-burned-mp4")
            return output_video

        progress_steps: list[float] = []
        words = [WordTimestamp("WHOA", 10.0, 10.5)]
        chat_burst = [{"author": "Fan", "message": "INSANE!", "offset_seconds": 10.2}]

        with (
            patch.object(exporter.narrator, "prepare_narration_tracks", side_effect=fake_prepare),
            patch.object(exporter.narrator, "mix_narration_with_ducking", side_effect=fake_mix),
            patch.object(exporter.caption_burner, "generate_ass", side_effect=fake_gen_ass),
            patch.object(exporter.caption_burner, "burn_captions", side_effect=fake_burn),
        ):
            result = exporter.export(
                clip=fake_clip,  # type: ignore[arg-type]
                words=words,
                clip_start_offset=10.0,
                output_path=output_path,
                progress_callback=progress_steps.append,
                chat_burst=chat_burst,
                narration_intro="Watch this!",
                narration_outro="Unbelievable ending!",
                voice_intro_path=user_intro,
                mode="narrative",
            )

        assert result == output_path
        assert output_path.read_bytes() == b"final-burned-mp4"

        # Verify ultrafast CRF 18 intermediate render kwargs
        assert len(fake_clip.write_calls) == 1
        _, w_kwargs = fake_clip.write_calls[0]
        assert w_kwargs["codec"] == "libx264"
        assert w_kwargs["preset"] == "ultrafast"
        assert w_kwargs["ffmpeg_params"] == ["-crf", "18"]
        assert w_kwargs["fps"] == 30
        assert w_kwargs["audio"] is True
        assert w_kwargs["audio_codec"] == "aac"

        # Verify ALL MoviePy handles were closed BEFORE narrator mix & ASS burn-in
        close_indices = [events.index(e) for e in ("close:composite", "close:src1", "close:src2")]
        assert max(close_indices) < events.index("narrator:prepare")
        assert events.index("narrator:prepare") < events.index("narrator:mix")
        assert events.index("narrator:mix") < events.index("captions:generate_ass")
        assert events.index("captions:generate_ass") < events.index("captions:burn")

        # Verify ASS burn-in received mixed video, chat_burst, configured crf & resolved codec
        assert gen_ass_calls[0]["chat_burst"] == chat_burst
        assert burn_calls[0]["input_video"].name == "intermediate_mixed.mp4"
        assert burn_calls[0]["crf"] == 21
        assert burn_calls[0]["codec"] == "h264_nvenc"

        # Verify intermediate work_dir was completely cleaned up while user voice file was preserved
        assert list((tmp_path / "temp").glob("streamclipper_export_*")) == []
        assert user_intro.is_file()
        assert progress_steps == [0.0, 0.5, 0.75, 1.0]

    def test_export_forwards_effective_clamped_durations_to_generate_ass(
        self, tmp_path: Path
    ) -> None:
        """Exporter must compute and forward effective clamped durations to generate_ass."""
        cfg = _make_config(
            tmp_path,
            overrides={"general": {"temp_dir": str(tmp_path / "temp")}},
        )
        exporter = Exporter(cfg)
        captured_kwargs: dict[str, Any] = {}

        def fake_gen_ass(**kwargs: Any) -> Path:
            captured_kwargs.update(kwargs)
            kwargs["ass_path"].write_text("[Script Info]\n", encoding="utf-8")
            return kwargs["ass_path"]

        fake_clip = _FakeCompositeClip(
            events=[],
            duration=8.0,
            has_audio=True,
        )
        fake_intro = tmp_path / "intro.wav"
        fake_intro.write_bytes(b"RIFF....WAVE")
        fake_outro = tmp_path / "outro.wav"
        fake_outro.write_bytes(b"RIFF....WAVE")

        with (
            patch.object(
                exporter.narrator,
                "prepare_narration_tracks",
                return_value=(fake_intro, fake_outro),
            ),
            patch.object(
                exporter.narrator,
                "_probe_audio_duration",
                side_effect=[4.0, 4.0],
            ),
            patch.object(
                exporter.narrator,
                "mix_narration_with_ducking",
                return_value=tmp_path / "mixed.mp4",
            ),
            patch.object(
                exporter.caption_burner, "generate_ass", side_effect=fake_gen_ass
            ),
            patch.object(
                exporter.caption_burner,
                "burn_captions",
                return_value=tmp_path / "final.mp4",
            ),
        ):
            exporter.export(
                clip=fake_clip,  # type: ignore[arg-type]
                words=[WordTimestamp("test", 0.0, 1.0)],
                clip_start_offset=0.0,
                output_path=tmp_path / "out.mp4",
                narration_intro="Intro",
                narration_outro="Outro",
                mode="narrative",
            )

        # On an 8.0s clip, intro=4.0 and outro=4.0 must be clamped to 3.0s each (75% budget)
        assert captured_kwargs["intro_dur"] == pytest.approx(3.0)
        assert captured_kwargs["outro_dur"] == pytest.approx(3.0)
        assert captured_kwargs["clip_duration"] == pytest.approx(8.0)

    def test_export_closes_all_handles_and_cleans_temp_on_write_error(
        self, tmp_path: Path
    ) -> None:
        cfg = _make_config(
            tmp_path,
            overrides={"general": {"temp_dir": str(tmp_path / "temp")}},
        )
        exporter = Exporter(cfg)
        events: list[str] = []
        src1 = _TrackedHandle("src1", events)
        src2 = _TrackedHandle("src2", events)
        failing_clip = _FakeCompositeClip(
            events,
            source_clips=[src1, src2],
            write_error=RuntimeError("simulated render crash"),
        )

        with pytest.raises(RuntimeError, match="simulated render crash"):
            exporter.export(
                clip=failing_clip,  # type: ignore[arg-type]
                words=[],
                clip_start_offset=0.0,
                output_path=tmp_path / "out" / "failed.mp4",
            )

        assert failing_clip.closed
        assert src1.closed
        assert src2.closed
        assert list((tmp_path / "temp").glob("streamclipper_export_*")) == []

    def test_export_raw_mode_skips_narrator_and_chat_ticker(self, tmp_path: Path) -> None:
        cfg = _make_config(
            tmp_path,
            overrides={"general": {"temp_dir": str(tmp_path / "temp")}},
        )
        exporter = Exporter(cfg)
        events: list[str] = []
        fake_clip = _FakeCompositeClip(events)

        with (
            patch.object(exporter.narrator, "prepare_narration_tracks") as mock_prep,
            patch.object(exporter.narrator, "mix_narration_with_ducking") as mock_mix,
            patch.object(exporter.caption_burner, "generate_ass") as mock_ass,
            patch.object(
                exporter.caption_burner,
                "burn_captions",
                side_effect=lambda **kw: kw["output_video"],
            ) as mock_burn,
        ):
            exporter.export(
                clip=fake_clip,  # type: ignore[arg-type]
                words=[WordTimestamp("RAW", 0.0, 0.5)],
                clip_start_offset=0.0,
                output_path=tmp_path / "raw_clip.mp4",
                chat_burst=[{"author": "U", "message": "SHOULD NOT BURN", "offset_seconds": 0.1}],
                narration_intro="Should be skipped",
                narration_outro="Also skipped",
                mode="raw",
            )

        mock_prep.assert_not_called()
        mock_mix.assert_not_called()
        mock_ass.assert_called_once()
        assert mock_ass.call_args.kwargs["chat_burst"] is None
        mock_burn.assert_called_once()

    def test_export_explicit_mode_overrides_disabled_config_and_restores_narrator(
        self, tmp_path: Path
    ) -> None:
        cfg = _make_config(
            tmp_path,
            overrides={
                "general": {"temp_dir": str(tmp_path / "temp")},
                "transformation": {"enabled": False, "mode": "raw"},
            },
        )
        exporter = Exporter(cfg)
        assert exporter.narrator.enabled is False
        assert exporter.narrator.mode == "raw"

        observed_state: list[tuple[bool, str]] = []

        def fake_prep(**kw: Any) -> tuple[Path | None, Path | None]:
            observed_state.append((exporter.narrator.enabled, exporter.narrator.mode))
            wav = Path(kw["temp_dir"]) / "intro.wav"
            wav.write_bytes(b"wav")
            return wav, None

        def fake_mix(**kw: Any) -> Path:
            Path(kw["output_video"]).write_bytes(b"mixed")
            return Path(kw["output_video"])

        events: list[str] = []
        fake_clip = _FakeCompositeClip(events)

        with (
            patch.object(exporter.narrator, "prepare_narration_tracks", side_effect=fake_prep),
            patch.object(exporter.narrator, "mix_narration_with_ducking", side_effect=fake_mix),
            patch.object(
                exporter,
                "_finalize_without_captions",
                side_effect=lambda inp, out: out,
            ),
        ):
            exporter.export(
                clip=fake_clip,  # type: ignore[arg-type]
                words=[],
                clip_start_offset=0.0,
                output_path=tmp_path / "override.mp4",
                narration_intro="Explicit override intro",
                mode="narrative",
            )

        assert observed_state == [(True, "narrative")]
        # Persistent narrator settings restored cleanly in finally
        assert exporter.narrator.enabled is False
        assert exporter.narrator.mode == "raw"

    def test_export_chat_burst_only_still_burns_ass_when_enabled(self, tmp_path: Path) -> None:
        cfg = _make_config(
            tmp_path,
            overrides={"general": {"temp_dir": str(tmp_path / "temp")}},
        )
        exporter = Exporter(cfg)
        fake_clip = _FakeCompositeClip([])
        chat_burst = [{"author": "Viewer", "message": "W CLIP", "offset_seconds": 2.0}]

        with (
            patch.object(exporter.caption_burner, "generate_ass") as mock_ass,
            patch.object(
                exporter.caption_burner,
                "burn_captions",
                side_effect=lambda **kw: kw["output_video"],
            ) as mock_burn,
            patch.object(exporter, "_finalize_without_captions") as mock_fin,
        ):
            exporter.export(
                clip=fake_clip,  # type: ignore[arg-type]
                words=[],
                clip_start_offset=0.0,
                output_path=tmp_path / "chat_only.mp4",
                chat_burst=chat_burst,
                mode="narrative",
            )

        mock_ass.assert_called_once()
        assert mock_ass.call_args.kwargs["chat_burst"] == chat_burst
        mock_burn.assert_called_once()
        mock_fin.assert_not_called()

    def test_export_without_captions_finalizes_nvenc_with_libx264_fallback(
        self, tmp_path: Path
    ) -> None:
        cfg = _make_config(
            tmp_path,
            overrides={
                "general": {"temp_dir": str(tmp_path / "temp")},
                "video": {"crf": 22},
            },
            video_codec="h264_nvenc",
        )
        exporter = Exporter(cfg)
        fake_clip = _FakeCompositeClip([])
        out_path = tmp_path / "out" / "plain.mp4"

        side_effects = [
            subprocess.CalledProcessError(returncode=1, cmd=["ffmpeg"], stderr=b"nvenc fail"),
            subprocess.CompletedProcess(args=["ffmpeg"], returncode=0),
        ]
        with patch("streamclipper.output.exporter.subprocess.run", side_effect=side_effects) as mock_run:
            result = exporter.export(
                clip=fake_clip,  # type: ignore[arg-type]
                words=[],
                clip_start_offset=0.0,
                output_path=out_path,
                mode="raw",
            )

        assert result == out_path
        assert mock_run.call_count == 2
        first_cmd = mock_run.call_args_list[0].args[0]
        second_cmd = mock_run.call_args_list[1].args[0]
        assert "h264_nvenc" in first_cmd and "-cq" in first_cmd
        assert "libx264" in second_cmd and "-crf" in second_cmd
        assert "-c:a" in second_cmd and second_cmd[second_cmd.index("-c:a") + 1] == "copy"

    def test_close_clip_handles_deduplicates_nested_sources(self) -> None:
        events: list[str] = []
        shared_src = _TrackedHandle("shared", events)
        mid = _FakeCompositeClip(events, source_clips=[shared_src])
        outer = MagicMock()
        outer._source_clip = shared_src
        outer._source_clips = [shared_src, mid]
        outer.close.side_effect = lambda: events.append("close:outer")

        _close_clip_handles(outer)
        assert events.count("close:shared") == 1
        assert events.count("close:composite") == 1
        assert events.count("close:outer") == 1


# ---------------------------------------------------------------------------
# Package 4.6 — Real FFmpeg + MoviePy End-to-End Windows Integration Tests
# ---------------------------------------------------------------------------


def _synth_test_video(path: Path, *, duration: float = 2.0, freq: int = 440, color: str = "navy") -> None:
    """Create a tiny real 16:9 H.264 + AAC MP4 for end-to-end integration tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s=320x180:d={duration}:r=15",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={freq}:duration={duration}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def _synth_test_wav(path: Path, *, duration: float = 0.6, freq: int = 660) -> None:
    """Create a tiny real 24kHz mono WAV (matching Kokoro output format) for ducking tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={freq}:sample_rate=24000:duration={duration}",
            "-ac",
            "1",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available on PATH")
class TestPackage46RealEndToEndIntegration:
    def test_real_e2e_compose_hook_narrator_export_and_organize(self, tmp_path: Path) -> None:
        """Full Phase 4 chain on real media: Composer -> Hook -> Exporter (ducking + ASS) -> FileOrganizer."""
        seg_mp4 = tmp_path / "segments" / "clip_seg.mp4"
        intro_wav = tmp_path / "human_intro.wav"
        _synth_test_video(seg_mp4, duration=2.2, freq=440, color="navy")
        _synth_test_wav(intro_wav, duration=0.6, freq=700)

        cfg = _make_config(
            tmp_path,
            overrides={
                "general": {
                    "output_dir": str(tmp_path / "output"),
                    "temp_dir": str(tmp_path / "temp"),
                },
                "video": {"resolution": [180, 320], "fps": 15, "crf": 28},
                "hook_title": {"duration_seconds": 1.2, "font_size": 16},
                "captions": {"font_size": 20},
                "transformation": {"enabled": True, "mode": "narrative", "tts_engine": "none"},
            },
            video_codec="libx264",
        )

        composed = VideoComposer(cfg).compose(seg_mp4, facecam=None, peak_offset=1.0)
        composed = HookTitleOverlay(cfg).add_hook(
            composed,
            hook_title="WILD MOMENT",
            context_badge="Clutch Escape",
        )

        organizer = FileOrganizer(cfg)
        out_mp4 = organizer.get_clip_path("GTA 6", "TimTheTatman", 1, date_str="2026-11-18")

        final_mp4 = Exporter(cfg).export(
            clip=composed,
            words=[WordTimestamp("INSANE", 10.2, 10.9), WordTimestamp("CLUTCH", 11.0, 11.8)],
            clip_start_offset=10.0,
            output_path=out_mp4,
            chat_burst=[{"author": "Viewer1", "message": "NO WAY!!", "offset_seconds": 10.5}],
            voice_intro_path=intro_wav,
            mode="narrative",
        )

        clip_result = organizer.save_clip(
            video_path=final_mp4,
            metadata=_sample_clip_metadata(transformation_mode="narrative"),
            transcript_text="INSANE CLUTCH",
        )

        assert final_mp4.is_file() and final_mp4.stat().st_size > 1000
        assert clip_result.metadata_path is not None and clip_result.metadata_path.is_file()
        assert final_mp4.with_name("clip_01_transcript.txt").is_file()
        assert clip_result.duration_seconds > 1.5
        # Zero Windows file-lock leftovers & original human mic WAV preserved
        assert list((tmp_path / "temp").glob("streamclipper_export_*")) == []
        assert intro_wav.is_file()
        # Source segment can be unlinked on Windows (proving _source_clip handle was closed!)
        seg_mp4.unlink()
        assert not seg_mp4.exists()

    def test_real_e2e_pipeline_assemble_highlight_single_and_multi_beat(
        self, tmp_path: Path
    ) -> None:
        """Verify Pipeline._assemble_highlight wires all Phase 4 modules end-to-end with zero mocks."""
        seg1 = tmp_path / "segments" / "beat1.mp4"
        seg2 = tmp_path / "segments" / "beat2.mp4"
        intro_wav = tmp_path / "mic_intro.wav"
        _synth_test_video(seg1, duration=2.0, freq=440, color="darkgreen")
        _synth_test_video(seg2, duration=2.0, freq=520, color="darkred")
        _synth_test_wav(intro_wav, duration=0.5, freq=680)

        cfg = _make_config(
            tmp_path,
            overrides={
                "general": {
                    "output_dir": str(tmp_path / "output"),
                    "temp_dir": str(tmp_path / "temp"),
                },
                "video": {"resolution": [180, 320], "fps": 15, "crf": 28},
                "hook_title": {"duration_seconds": 1.0, "font_size": 16},
                "captions": {"font_size": 20},
                "transformation": {"enabled": True, "mode": "auto", "tts_engine": "none"},
            },
            video_codec="libx264",
        )
        pipeline = Pipeline(cfg)

        # 1. Single-cut narrative mode
        h_single = Highlight(
            start=10.0,
            end=12.0,
            score=9.0,
            title="SINGLE CUT TEST",
            reason="Great moment",
            transcript_chunk="WATCH THIS",
            word_timestamps=[WordTimestamp("WATCH", 10.2, 10.8), WordTimestamp("THIS", 10.9, 11.5)],
            moment_type="funny",
            context_badge="Instant Karma",
            peak_time=11.0,
            chat_burst=[{"author": "Chat1", "message": "LMAO", "offset_seconds": 10.5}],
        )
        res_single = pipeline._assemble_highlight(
            highlight=h_single,
            metadata=_sample_clip_metadata(
                hook_title="SINGLE CUT TEST",
                clip_start=10.0,
                clip_end=12.0,
                transformation_mode="narrative",
            ),
            segment_paths=[seg1],
            clip_index=1,
            mode="narrative",
            voice_intro_path=intro_wav,
            voice_outro_path=None,
            buffer_seconds=0.0,
        )
        assert res_single.video_path.is_file() and res_single.video_path.stat().st_size > 1000
        assert res_single.metadata_path is not None and res_single.metadata_path.is_file()

        # 2. Multi-beat story mode (2 segments stitched + remapped words/chat/peak)
        beats = [
            {"start": 100.0, "end": 101.5, "label": "Setup"},
            {"start": 120.0, "end": 121.5, "label": "Payoff"},
        ]
        h_story = Highlight(
            start=100.0,
            end=121.5,
            score=9.4,
            title="STORY MODE TEST",
            reason="Setup and payoff",
            transcript_chunk="WAIT FOR IT BOOM",
            word_timestamps=[
                WordTimestamp("WAIT", 100.2, 100.8),
                WordTimestamp("DEAD_AIR", 110.0, 111.0),
                WordTimestamp("BOOM", 120.2, 121.0),
            ],
            moment_type="clutch",
            context_badge="Clutch Escape",
            peak_time=120.6,
            chat_burst=[{"author": "Chat2", "message": "POGGERS", "offset_seconds": 120.5}],
            sub_clips=beats,
        )
        res_story = pipeline._assemble_highlight(
            highlight=h_story,
            metadata=_sample_clip_metadata(
                hook_title="STORY MODE TEST",
                clip_start=100.0,
                clip_end=121.5,
                transformation_mode="story",
            ),
            segment_paths=[seg1, seg2],
            clip_index=2,
            mode="story",
            voice_intro_path=intro_wav,
            voice_outro_path=None,
            valid_beats=beats,
            buffer_seconds=0.25,
        )
        assert res_story.video_path.is_file() and res_story.video_path.stat().st_size > 1000
        assert res_story.metadata_path is not None and res_story.metadata_path.is_file()
        assert res_story.duration_seconds == pytest.approx(3.0, abs=0.4)
        # Verify Windows handles for BOTH beat segments were cleanly closed
        seg1.unlink()
        seg2.unlink()
        assert not seg1.exists() and not seg2.exists()
