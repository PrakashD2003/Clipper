"""Tests for Phase 4 video assembly & output modules (Packages 4.3 & 4.6)."""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from streamclipper.assembly.captions import (
    AssCaptionBurner,
    _chunk_words,
    _color_to_ass_style,
    _color_to_ass_style_with_alpha,
    _escape_ass_text,
    _escape_ffmpeg_ass_filename,
    _hex_to_ass_bgr,
    _is_font_installed,
    _resolve_highlight_bgr,
    _sec_to_ass_time,
    resolve_font,
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
        assert "ScaledBorderAndShadow: yes" in content
        assert "Style: Hormozi,Impact,82,&H00FFFFFF,&H0000E6FF,&H00000000,&H90000000" in content
        # MarginV = round(1920 * (1.0 - 0.75)) = 480
        assert ",1,8,6,2,60,60,480,1" in content
        # ChatTicker uses BorderStyle=3 with semi-transparent OutlineColour=&H99000000
        assert (
            "Style: ChatTicker,Impact,34,&H00FFFFFF,&H00FFFFFF,&H99000000,&H99000000,"
            "-1,0,0,0,100,100,0,0,3,2,0,9,40,40,260,1"
        ) in content

        hormozi_lines = [ln for ln in content.splitlines() if ",Hormozi,," in ln]
        chat_lines = [ln for ln in content.splitlines() if ",ChatTicker,," in ln]

        assert len(hormozi_lines) == 6
        # Group 1 (3 words), active word highlighted in electric gold BGR &H00E6FF& with bounce pop animation
        assert (
            hormozi_lines[0]
            == r"Dialogue: 0,0:00:00.00,0:00:00.30,Hormozi,,0,0,0,,{\c&H00E6FF&\fscx100\fscy100\t(0,70,\fscx130\fscy130)\t(70,140,\fscx100\fscy100)}OH{\c&HFFFFFF&\fscx100\fscy100} MY GOD"
        )
        assert (
            hormozi_lines[2]
            == r"Dialogue: 0,0:00:00.60,0:00:01.00,Hormozi,,0,0,0,,OH MY {\c&H00E6FF&\fscx100\fscy100\t(0,70,\fscx130\fscy130)\t(70,140,\fscx100\fscy100)}GOD{\c&HFFFFFF&\fscx100\fscy100}"
        )
        # Group 2 (remaining 3 words)
        assert (
            hormozi_lines[4]
            == r"Dialogue: 0,0:00:01.50,0:00:01.80,Hormozi,,0,0,0,,BRO {\c&H00E6FF&\fscx100\fscy100\t(0,70,\fscx130\fscy130)\t(70,140,\fscx100\fscy100)}NO{\c&HFFFFFF&\fscx100\fscy100} WAY"
        )
        assert (
            hormozi_lines[5]
            == r"Dialogue: 0,0:00:01.80,0:00:02.20,Hormozi,,0,0,0,,BRO NO {\c&H00E6FF&\fscx100\fscy100\t(0,70,\fscx130\fscy130)\t(70,140,\fscx100\fscy100)}WAY{\c&HFFFFFF&\fscx100\fscy100}"
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

        # Intro narration chunk events emitted across 0.0 -> 2.0s
        intro_events = [ln for ln in hormozi_lines if "LOOK" in ln or "INSANE" in ln]
        assert len(intro_events) == 5
        clean_intro = [re.sub(r"\{.*?\}", "", ln) for ln in intro_events]
        assert any("LOOK AT THIS" in ln for ln in clean_intro)
        assert any("INSANE MOVE" in ln for ln in clean_intro)
        assert any("0:00:00.00," in ln for ln in intro_events)
        assert any(",0:00:02.00,Hormozi" in ln for ln in intro_events)

        # Outro narration chunk events emitted across 8.0 -> 10.0s
        outro_events = [ln for ln in hormozi_lines if "ACTUALLY" in ln or "PULLED" in ln or "OFF" in ln]
        assert len(outro_events) == 5
        clean_outro = [re.sub(r"\{.*?\}", "", ln) for ln in outro_events]
        assert any("HE ACTUALLY PULLED" in ln for ln in clean_outro)
        assert any("IT OFF" in ln for ln in clean_outro)
        assert any("0:00:08.00," in ln for ln in outro_events)
        assert any(",0:00:10.00,Hormozi" in ln for ln in outro_events)

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
                    "animation": "none",
                    "shadow_offset": 0,
                    "shadow_color": "&H80000000",
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
            r"Dialogue: 0,0:00:00.00,0:00:00.40,Hormozi,,0,0,0,,{\c&H00E6FF&}BOOM{\c&HFFFFFF&}",
            r"Dialogue: 0,0:00:00.40,0:00:01.00,Hormozi,,0,0,0,,{\c&H00E6FF&}HEADSHOT{\c&HFFFFFF&}",
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
        # Trailing backslash separated from closing {\c&HFFFFFF&...} by ZWSP
        assert "BACK\\\u200b{\\c&HFFFFFF&" in hormozi_lines[1]
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
        assert burner.words_per_group == 3
        assert burner.font_size == 82
        assert burner.stroke_width == 8
        assert (burner.width, burner.height) == (1080, 1920)
        assert burner.margin_v == 480

        ass_path = tmp_path / "safe_defaults.ass"
        burner.generate_ass([WordTimestamp("ok", 0.0, 0.5)], 0.0, ass_path)
        content = ass_path.read_text(encoding="utf-8")
        assert "None" not in content
        assert (
            "Style: Hormozi,Impact,82,&H00FFFFFF,&H0000E6FF,&H00000000,&H90000000,"
            "-1,0,0,0,100,100,0,0,1,8,6,2,60,60,480,1"
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
            clip_duration: float | None = None,
        ) -> tuple[Path | None, Path | None]:
            events.append("narrator:prepare")
            assert clip_duration == pytest.approx(15.0)  # G7: exporter passes the budget source
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
            **_narration_kwargs: Any,
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
                side_effect=lambda p: 4.0 if p is not None else None,
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
        with (
            patch.object(exporter.narrator, "normalize_audio", return_value=None),
            patch("streamclipper.output.exporter.subprocess.run", side_effect=side_effects) as mock_run,
        ):
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
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify Pipeline._assemble_highlight wires all Phase 4 modules end-to-end with zero mocks."""
        from streamclipper.assembly import facecam as _facecam_mod

        _orig_detect = _facecam_mod.FacecamDetector.detect

        def _detect_compat(self, paths, *args, streamer=None, **kw):  # C5 call shape
            first = paths[0] if isinstance(paths, (list, tuple)) else paths
            return _orig_detect(self, first, *args, **kw)

        monkeypatch.setattr(_facecam_mod.FacecamDetector, "detect", _detect_compat)
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



# ---------------------------------------------------------------------------
# Review fixes: M11 / M13 / L9 exporter regressions
# ---------------------------------------------------------------------------


def _narration_patches(exporter: Exporter, tmp_path: Path, mix_kwargs: dict[str, Any]):
    fake_intro = tmp_path / "intro.wav"
    fake_intro.write_bytes(b"RIFF....WAVE")
    return (
        patch.object(
            exporter.narrator,
            "prepare_narration_tracks",
            return_value=(fake_intro, None),
        ),
        patch.object(
            exporter.narrator,
            "_probe_audio_duration",
            side_effect=lambda p: 2.0 if p is not None else None,
        ),
        patch.object(exporter.narrator, "mix_narration_with_ducking", **mix_kwargs),
    )


def test_m11_mix_failure_keeps_unnarrated_video_and_clears_windows(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path, overrides={"general": {"temp_dir": str(tmp_path / "temp")}})
    exporter = Exporter(cfg)
    captured: dict[str, Any] = {}

    def fake_gen_ass(**kwargs: Any) -> Path:
        captured.update(kwargs)
        kwargs["ass_path"].write_text("[Script Info]\n", encoding="utf-8")
        return kwargs["ass_path"]

    p1, p2, p3 = _narration_patches(
        exporter, tmp_path, {"side_effect": RuntimeError("mix boom")}
    )
    with (
        p1,
        p2,
        p3,
        patch.object(exporter.narrator, "normalize_audio", return_value=None),
        patch.object(exporter.caption_burner, "generate_ass", side_effect=fake_gen_ass),
        patch.object(
            exporter.caption_burner, "burn_captions", return_value=tmp_path / "final.mp4"
        ) as burn,
    ):
        exporter.export(
            clip=_FakeCompositeClip([], duration=8.0),  # type: ignore[arg-type]
            words=[WordTimestamp("hi", 0.0, 1.0)],
            clip_start_offset=0.0,
            output_path=tmp_path / "out.mp4",
            narration_intro="Intro",
            mode="narrative",
        )
    assert captured["intro_dur"] is None
    assert captured["outro_dur"] is None
    assert captured["breakdown_window"] is None
    assert Path(burn.call_args.kwargs["input_video"]).name == "intermediate_raw.mp4"


def test_m13_narration_card_alone_generates_and_burns_ass(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path, overrides={"general": {"temp_dir": str(tmp_path / "temp")}})
    exporter = Exporter(cfg)

    def fake_gen_ass(**kwargs: Any) -> Path:
        kwargs["ass_path"].write_text("[Script Info]\n", encoding="utf-8")
        return kwargs["ass_path"]

    p1, p2, p3 = _narration_patches(
        exporter, tmp_path, {"return_value": tmp_path / "mixed.mp4"}
    )
    with (
        p1,
        p2,
        p3,
        patch.object(
            exporter.caption_burner, "generate_ass", side_effect=fake_gen_ass
        ) as gen,
        patch.object(
            exporter.caption_burner, "burn_captions", return_value=tmp_path / "final.mp4"
        ) as burn,
    ):
        exporter.export(
            clip=_FakeCompositeClip([], duration=8.0),  # type: ignore[arg-type]
            words=[],
            clip_start_offset=0.0,
            output_path=tmp_path / "out.mp4",
            narration_intro="Intro",
            mode="narrative",
        )
    assert gen.call_count == 1
    assert gen.call_args.kwargs["intro_dur"] == pytest.approx(2.0)
    assert burn.call_count == 1


def test_l9_ffmpeg_stderr_tail_logged_on_failure(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    cfg = _make_config(
        tmp_path,
        overrides={"general": {"temp_dir": str(tmp_path / "temp")}},
        video_codec="h264_nvenc",
    )
    exporter = Exporter(cfg)
    side_effects = [
        subprocess.CalledProcessError(1, ["ffmpeg"], stderr=b"x" * 600 + b"NVENC-BOOM"),
        subprocess.CompletedProcess(args=["ffmpeg"], returncode=0),
    ]
    with caplog.at_level(logging.WARNING, logger="streamclipper.output.exporter"):
        with (
            patch.object(exporter.narrator, "normalize_audio", return_value=None),
            patch("streamclipper.output.exporter.subprocess.run", side_effect=side_effects),
        ):
            exporter.export(
                clip=_FakeCompositeClip([]),  # type: ignore[arg-type]
                words=[],
                clip_start_offset=0.0,
                output_path=tmp_path / "o" / "p.mp4",
                mode="raw",
            )
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "NVENC-BOOM" in joined
    assert "x" * 490 in joined and "x" * 520 not in joined


# ---------------------------------------------------------------------------
# Output Quality v3 - Batch E exporter wiring (E4 / E6 / E7)
# ---------------------------------------------------------------------------


def _v3_exporter(tmp_path: Path):
    from unittest.mock import MagicMock

    from streamclipper.output.exporter import Exporter

    cfg = _make_config(
        tmp_path,
        overrides={"general": {"temp_dir": str(tmp_path / "temp")}},
        video_codec="libx264",
    )
    exporter = Exporter(cfg)
    narrator = MagicMock()
    narrator.enabled = True
    narrator.mode = "narrative"
    narrator.prepare_narration_tracks.return_value = (None, None)
    narrator._probe_audio_duration.return_value = 2.0
    exporter.narrator = narrator
    exporter.caption_burner = MagicMock()
    return exporter, narrator


def _v3_clip():
    class _Clip:  # plain object: MagicMock would recurse forever in _close_clip_handles
        duration = 20.0
        audio = object()

        def write_videofile(self, path, **kw):
            Path(path).write_bytes(b"v")

        def close(self):
            pass

    return _Clip()


def test_E7_normalize_audio_called_when_no_mix_ran(tmp_path: Path) -> None:
    exporter, narrator = _v3_exporter(tmp_path)

    def _norm(src, dst):
        Path(dst).write_bytes(b"n")
        return Path(dst)

    narrator.normalize_audio.side_effect = _norm
    seen = {}

    def _finalize(src, dst):
        seen["src"] = Path(src).name
        Path(dst).write_bytes(b"f")
        return Path(dst)

    exporter._finalize_without_captions = _finalize  # type: ignore[method-assign]
    exporter.export(
        _v3_clip(), [], 0.0, tmp_path / "o.mp4", mode="narrative"
    )
    assert narrator.normalize_audio.called
    assert seen["src"] == "intermediate_norm.mp4"
    assert not narrator.mix_narration_with_ducking.called


def test_E4_E6_callout_and_duck_full_reach_the_mix(tmp_path: Path) -> None:
    exporter, narrator = _v3_exporter(tmp_path)
    narrator.prepare_callout_track.return_value = tmp_path / "callout.wav"

    def _mix(**kw):
        Path(kw["output_video"]).write_bytes(b"m")
        return Path(kw["output_video"])

    narrator.mix_narration_with_ducking.side_effect = _mix
    exporter._finalize_without_captions = lambda s, d: Path(d).write_bytes(b"f") or Path(d)  # type: ignore[method-assign]
    exporter.export(
        _v3_clip(),
        [],
        0.0,
        tmp_path / "o.mp4",
        mode="narrative",
        callout="Watch the tire",
        callout_start=8.0,
        duck_full=True,
        layout="LAYOUT",
    )
    kw = narrator.mix_narration_with_ducking.call_args[1]
    assert kw["callout_wav"] == tmp_path / "callout.wav"
    assert kw["callout_start"] == 8.0
    assert kw["duck_full"] is True
    ass_kw = exporter.caption_burner.generate_ass.call_args[1]
    assert ass_kw["layout"] == "LAYOUT"
    assert ass_kw["callout"] == "Watch the tire"
    assert ass_kw["callout_window"][0] == 8.0



# ---------------------------------------------------------------------------
# Output Quality v3 - Batch E exporter wiring (E4 / E6 / E7)
# ---------------------------------------------------------------------------


def _v3_exporter(tmp_path: Path):
    from unittest.mock import MagicMock

    from streamclipper.output.exporter import Exporter

    cfg = _make_config(
        tmp_path,
        overrides={"general": {"temp_dir": str(tmp_path / "temp")}},
        video_codec="libx264",
    )
    exporter = Exporter(cfg)
    narrator = MagicMock()
    narrator.enabled = True
    narrator.mode = "narrative"
    narrator.prepare_narration_tracks.return_value = (None, None)
    narrator._probe_audio_duration.return_value = 2.0
    exporter.narrator = narrator
    exporter.caption_burner = MagicMock()
    return exporter, narrator


def _v3_clip():
    class _Clip:  # plain object: MagicMock would recurse forever in _close_clip_handles
        duration = 20.0
        audio = object()

        def write_videofile(self, path, **kw):
            Path(path).write_bytes(b"v")

        def close(self):
            pass

    return _Clip()


def test_E7_normalize_audio_called_when_no_mix_ran(tmp_path: Path) -> None:
    exporter, narrator = _v3_exporter(tmp_path)

    def _norm(src, dst):
        Path(dst).write_bytes(b"n")
        return Path(dst)

    narrator.normalize_audio.side_effect = _norm
    seen = {}

    def _finalize(src, dst):
        seen["src"] = Path(src).name
        Path(dst).write_bytes(b"f")
        return Path(dst)

    exporter._finalize_without_captions = _finalize  # type: ignore[method-assign]
    exporter.export(
        _v3_clip(), [], 0.0, tmp_path / "o.mp4", mode="narrative"
    )
    assert narrator.normalize_audio.called
    assert seen["src"] == "intermediate_norm.mp4"
    assert not narrator.mix_narration_with_ducking.called


def test_E4_E6_callout_and_duck_full_reach_the_mix(tmp_path: Path) -> None:
    exporter, narrator = _v3_exporter(tmp_path)
    narrator.prepare_callout_track.return_value = tmp_path / "callout.wav"

    def _mix(**kw):
        Path(kw["output_video"]).write_bytes(b"m")
        return Path(kw["output_video"])

    narrator.mix_narration_with_ducking.side_effect = _mix
    exporter._finalize_without_captions = lambda s, d: Path(d).write_bytes(b"f") or Path(d)  # type: ignore[method-assign]
    exporter.export(
        _v3_clip(),
        [],
        0.0,
        tmp_path / "o.mp4",
        mode="narrative",
        callout="Watch the tire",
        callout_start=8.0,
        duck_full=True,
        layout="LAYOUT",
    )
    kw = narrator.mix_narration_with_ducking.call_args[1]
    assert kw["callout_wav"] == tmp_path / "callout.wav"
    assert kw["callout_start"] == 8.0
    assert kw["duck_full"] is True
    ass_kw = exporter.caption_burner.generate_ass.call_args[1]
    assert ass_kw["layout"] == "LAYOUT"
    assert ass_kw["callout"] == "Watch the tire"
    assert ass_kw["callout_window"][0] == 8.0



# ---------------------------------------------------------------------------
# Output Quality v3 - Batch E exporter wiring (E4 / E6 / E7)
# ---------------------------------------------------------------------------


def _v3_exporter(tmp_path: Path):
    from unittest.mock import MagicMock

    from streamclipper.output.exporter import Exporter

    cfg = _make_config(
        tmp_path,
        overrides={"general": {"temp_dir": str(tmp_path / "temp")}},
        video_codec="libx264",
    )
    exporter = Exporter(cfg)
    narrator = MagicMock()
    narrator.enabled = True
    narrator.mode = "narrative"
    narrator.prepare_narration_tracks.return_value = (None, None)
    narrator._probe_audio_duration.return_value = 2.0
    exporter.narrator = narrator
    exporter.caption_burner = MagicMock()
    return exporter, narrator


def _v3_clip():
    class _Clip:  # plain object: MagicMock would recurse forever in _close_clip_handles
        duration = 20.0
        audio = object()

        def write_videofile(self, path, **kw):
            Path(path).write_bytes(b"v")

        def close(self):
            pass

    return _Clip()


def test_E7_normalize_audio_called_when_no_mix_ran(tmp_path: Path) -> None:
    exporter, narrator = _v3_exporter(tmp_path)

    def _norm(src, dst):
        Path(dst).write_bytes(b"n")
        return Path(dst)

    narrator.normalize_audio.side_effect = _norm
    seen = {}

    def _finalize(src, dst):
        seen["src"] = Path(src).name
        Path(dst).write_bytes(b"f")
        return Path(dst)

    exporter._finalize_without_captions = _finalize  # type: ignore[method-assign]
    exporter.export(
        _v3_clip(), [], 0.0, tmp_path / "o.mp4", mode="narrative"
    )
    assert narrator.normalize_audio.called
    assert seen["src"] == "intermediate_norm.mp4"
    assert not narrator.mix_narration_with_ducking.called


def test_E4_E6_callout_and_duck_full_reach_the_mix(tmp_path: Path) -> None:
    exporter, narrator = _v3_exporter(tmp_path)
    narrator.prepare_callout_track.return_value = tmp_path / "callout.wav"

    def _mix(**kw):
        Path(kw["output_video"]).write_bytes(b"m")
        return Path(kw["output_video"])

    narrator.mix_narration_with_ducking.side_effect = _mix
    exporter._finalize_without_captions = lambda s, d: Path(d).write_bytes(b"f") or Path(d)  # type: ignore[method-assign]
    exporter.export(
        _v3_clip(),
        [],
        0.0,
        tmp_path / "o.mp4",
        mode="narrative",
        callout="Watch the tire",
        callout_start=8.0,
        duck_full=True,
        layout="LAYOUT",
    )
    kw = narrator.mix_narration_with_ducking.call_args[1]
    assert kw["callout_wav"] == tmp_path / "callout.wav"
    assert kw["callout_start"] == 8.0
    assert kw["duck_full"] is True
    ass_kw = exporter.caption_burner.generate_ass.call_args[1]
    assert ass_kw["layout"] == "LAYOUT"
    assert ass_kw["callout"] == "Watch the tire"
    assert ass_kw["callout_window"][0] == 8.0



def test_v3_strip_unrenderable_removes_emoji_for_overlays() -> None:
    from streamclipper.assembly.captions import _escape_ass_text
    from streamclipper.assembly.textutil import strip_unrenderable

    assert strip_unrenderable("WILDEST STUNT & CRASH! \U0001f631") == "WILDEST STUNT & CRASH!"
    assert strip_unrenderable("GG ❤️ wp \U0001f468‍\U0001f4bb") == "GG wp"
    assert strip_unrenderable("") == ""
    assert _escape_ass_text("hype \U0001f525 {x}") == r"hype \{x\}"


class TestModernCaptionStyles:
    """Tests for modern short-form subtitle styles, pop/bounce animations, color sweeps, box & shadow."""

    def test_bounce_style_and_animation(self, tmp_path: Path) -> None:
        cfg = _make_config(
            tmp_path,
            overrides={
                "captions": {
                    "style": "bounce",
                    "words_per_group": 3,
                    "animation_scale": 1.20,
                    "animation_duration_ms": 140,
                }
            },
        )
        burner = AssCaptionBurner(cfg)
        assert burner.style == "bounce"
        assert burner.animation == "bounce"
        assert burner.animation_scale == 1.20
        assert burner.animation_duration_ms == 140

        words = [
            WordTimestamp("look", 0.0, 0.4),
            WordTimestamp("at", 0.4, 0.7),
            WordTimestamp("this", 0.7, 1.2),
        ]
        ass_path = tmp_path / "bounce.ass"
        burner.generate_ass(words, clip_start_offset=0.0, ass_path=ass_path)
        content = ass_path.read_text(encoding="utf-8")
        hormozi_lines = [ln for ln in content.splitlines() if ",Hormozi,," in ln]

        assert len(hormozi_lines) == 3
        # First word 'LOOK' has bounce animation: 100% -> 120% at t1, then 120% -> 100% at t2
        assert r"\fscx100\fscy100" in hormozi_lines[0]
        assert r"\t(0,70,\fscx120\fscy120)" in hormozi_lines[0]
        assert r"\t(70,140,\fscx100\fscy100)" in hormozi_lines[0]
        assert "LOOK" in hormozi_lines[0]
        # Active word closes with reset to primary color and 100% scale
        assert r"{\c&HFFFFFF&\fscx100\fscy100} AT THIS" in hormozi_lines[0]

        # Second word 'AT' active
        assert r"LOOK {\c&H00E6FF&\fscx100\fscy100" in hormozi_lines[1]
        assert "AT" in hormozi_lines[1]

    def test_pop_style_and_animation(self, tmp_path: Path) -> None:
        cfg = _make_config(
            tmp_path,
            overrides={
                "captions": {
                    "style": "pop",
                    "animation_scale": 1.25,
                    "animation_duration_ms": 100,
                }
            },
        )
        burner = AssCaptionBurner(cfg)
        assert burner.style == "pop"
        assert burner.animation == "pop"

        words = [WordTimestamp("insane", 1.0, 1.5)]
        ass_path = tmp_path / "pop.ass"
        burner.generate_ass(words, clip_start_offset=1.0, ass_path=ass_path)
        content = ass_path.read_text(encoding="utf-8")
        hormozi_lines = [ln for ln in content.splitlines() if ",Hormozi,," in ln]

        assert len(hormozi_lines) == 1
        # Pop: starts at 125% and scales down to 100% over 100ms
        assert r"\fscx125\fscy125\t(0,100,\fscx100\fscy100)" in hormozi_lines[0]
        assert "INSANE" in hormozi_lines[0]
        assert r"{\c&HFFFFFF&\fscx100\fscy100}" in hormozi_lines[0]

    def test_color_sweep_animation(self, tmp_path: Path) -> None:
        cfg = _make_config(
            tmp_path,
            overrides={
                "captions": {
                    "style": "hormozi",
                    "animation": "sweep",
                    "animation_duration_ms": 150,
                    "color": "white",
                    "highlight_color": "#FFD700",
                }
            },
        )
        burner = AssCaptionBurner(cfg)
        assert burner.animation == "sweep"

        words = [WordTimestamp("speed", 2.0, 2.5), WordTimestamp("run", 2.5, 3.0)]
        ass_path = tmp_path / "sweep.ass"
        burner.generate_ass(words, clip_start_offset=2.0, ass_path=ass_path)
        content = ass_path.read_text(encoding="utf-8")
        hormozi_lines = [ln for ln in content.splitlines() if ",Hormozi,," in ln]

        assert len(hormozi_lines) == 2
        # Color sweep: active word transitions smoothly from primary color to highlight color
        assert r"{\c&HFFFFFF&\t(0,150,\c&H00D7FF&)}SPEED{\c&HFFFFFF&} RUN" in hormozi_lines[0]

    def test_karaoke_style_progressive_fill(self, tmp_path: Path) -> None:
        cfg = _make_config(
            tmp_path,
            overrides={
                "captions": {
                    "style": "karaoke",
                    "words_per_group": 3,
                }
            },
        )
        burner = AssCaptionBurner(cfg)
        assert burner.style == "karaoke"

        words = [
            WordTimestamp("epic", 0.0, 0.35),
            WordTimestamp("clutch", 0.35, 0.80),
            WordTimestamp("win", 0.80, 1.10),
        ]
        ass_path = tmp_path / "karaoke.ass"
        burner.generate_ass(words, clip_start_offset=0.0, ass_path=ass_path)
        content = ass_path.read_text(encoding="utf-8")
        hormozi_lines = [ln for ln in content.splitlines() if ",Hormozi,," in ln]

        # Karaoke emits single dialogue event per group with \kf tags
        assert len(hormozi_lines) == 1
        line = hormozi_lines[0]
        assert "0:00:00.00,0:00:01.10,Hormozi" in line
        # 0.35s -> 35cs, 0.45s -> 45cs, 0.30s -> 30cs
        assert r"{\kf35}EPIC {\kf45}CLUTCH {\kf30}WIN" in line

        # Verify karaoke style definition line:
        # In ASS karaoke (\kf), SecondaryColour is the unsung initial color (white)
        # and PrimaryColour is the sung target fill color (gold &H0000E6FF)
        style_line = [ln for ln in content.splitlines() if ln.startswith("Style: Hormozi,")][0]
        assert "Style: Hormozi,Impact,82,&H0000E6FF,&H00FFFFFF," in style_line

    def test_background_box_styling(self, tmp_path: Path) -> None:
        cfg = _make_config(
            tmp_path,
            overrides={
                "captions": {
                    "style": "box",
                    "box_padding": 8,
                    "box_color": "black",
                    "box_opacity": 0.60,
                }
            },
        )
        burner = AssCaptionBurner(cfg)
        assert burner.box_style is True
        assert burner.box_padding == 8

        words = [WordTimestamp("boxed", 0.0, 0.5)]
        ass_path = tmp_path / "box.ass"
        burner.generate_ass(words, clip_start_offset=0.0, ass_path=ass_path)
        content = ass_path.read_text(encoding="utf-8")

        # In ASS V4+ Styles for BorderStyle=3 (box):
        # Format: ..., BorderStyle, Outline, Shadow, Alignment, ...
        # BorderStyle = 3, Outline = 8 (padding), Shadow = 0
        style_line = [ln for ln in content.splitlines() if ln.startswith("Style: Hormozi,")][0]
        # Box opacity 0.60 -> alpha = round(0.40 * 255) = 102 = 0x66 -> &H66000000
        assert "&H66000000" in style_line
        assert ",3,8,0,2," in style_line

    def test_drop_shadow_styling(self, tmp_path: Path) -> None:
        cfg = _make_config(
            tmp_path,
            overrides={
                "captions": {
                    "style": "hormozi",
                    "stroke_width": 4,
                    "shadow_offset": 3.5,
                    "shadow_color": "#FF0000",  # red shadow
                }
            },
        )
        burner = AssCaptionBurner(cfg)
        assert burner.shadow_offset == 3.5
        assert burner.shadow_color == "&H000000FF"

        words = [WordTimestamp("shadow", 0.0, 0.5)]
        ass_path = tmp_path / "shadow.ass"
        burner.generate_ass(words, clip_start_offset=0.0, ass_path=ass_path)
        content = ass_path.read_text(encoding="utf-8")

        style_line = [ln for ln in content.splitlines() if ln.startswith("Style: Hormozi,")][0]
        # BorderStyle = 1, Outline = 4, Shadow = 3.5
        assert ",&H000000FF," in style_line
        assert ",1,4,3.5,2," in style_line

    def test_style_preset_overrides_animation_and_box_and_shadow(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        burner = AssCaptionBurner(cfg)
        assert burner.animation == "bounce"
        assert burner.box_style is False
        assert burner.shadow_offset == 6

        preset = {
            "font": "Impact",
            "primary_color": "#00FF00",
            "highlight_color": "#FF00FF",
            "outline_color": "#000000",
            "animation": "pop",
            "animation_scale": 1.30,
            "animation_duration_ms": 110,
            "box_style": True,
            "box_padding": 10,
            "box_color": "black",
            "box_opacity": 0.50,
            "shadow_offset": 2,
        }

        words = [WordTimestamp("override", 0.0, 0.5)]
        ass_path = tmp_path / "preset_override.ass"
        burner.generate_ass(words, clip_start_offset=0.0, ass_path=ass_path, style_preset=preset)
        content = ass_path.read_text(encoding="utf-8")

        # Verify preset settings applied to ASS
        assert "Style: Hormozi,Impact," in content
        # Box style was set in preset -> BorderStyle 3, Outline 10
        assert ",3,10,0,2," in content
        # Pop animation was set in preset -> fscx130
        assert r"\fscx130\fscy130\t(0,110,\fscx100\fscy100)" in content

        # Verify burner state was completely restored
        assert burner.animation == "bounce"
        assert burner.box_style is False
        assert burner.shadow_offset == 6
        assert burner.font == "Impact"

    def test_defensive_init_new_caption_keys(self, tmp_path: Path) -> None:
        cfg = _make_config(
            tmp_path,
            overrides={
                "captions": {
                    "animation": "invalid_anim",
                    "animation_scale": "not_a_number",
                    "animation_duration_ms": None,
                    "shadow_offset": None,
                    "shadow_color": None,
                    "box_style": "invalid_bool_fallback",
                    "box_padding": None,
                    "box_opacity": None,
                    "box_color": None,
                }
            },
        )
        burner = AssCaptionBurner(cfg)
        assert burner.animation == "bounce"
        assert burner.animation_scale == 1.30
        assert burner.animation_duration_ms == 140
        assert burner.shadow_offset == 6
        assert burner.shadow_color == "&H90000000"
        assert burner.box_padding == 6
        assert burner.box_opacity == 0.60

    def test_real_ffmpeg_burns_new_styles_without_syntax_errors(self, tmp_path: Path) -> None:
        """Verify FFmpeg libass burns modern styles (bounce, pop, karaoke, box, shadow) without errors."""
        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg not available on PATH")

        video_path = tmp_path / "input.mp4"
        # Generate a tiny 1-second 320x240 synthetic test video
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=1:r=15",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(video_path),
            ],
            check=True,
            capture_output=True,
        )

        test_configs = [
            ("bounce", {"captions": {"style": "bounce", "animation_scale": 1.2}}),
            ("pop", {"captions": {"style": "pop", "animation_scale": 1.15}}),
            ("sweep", {"captions": {"style": "hormozi", "animation": "sweep"}}),
            ("karaoke", {"captions": {"style": "karaoke"}}),
            ("box", {"captions": {"style": "box", "box_padding": 4, "box_opacity": 0.5}}),
            ("shadow", {"captions": {"style": "hormozi", "shadow_offset": 2}}),
        ]

        words = [
            WordTimestamp("hello", 0.0, 0.4),
            WordTimestamp("world", 0.4, 0.8),
        ]

        for name, overrides in test_configs:
            cfg = _make_config(
                tmp_path,
                overrides={
                    **overrides,
                    "video": {"resolution": [320, 240]},
                },
            )
            burner = AssCaptionBurner(cfg)
            ass_path = tmp_path / f"{name}.ass"
            burner.generate_ass(words, clip_start_offset=0.0, ass_path=ass_path)
            assert ass_path.is_file()

            out_path = tmp_path / f"out_{name}.mp4"
            result = burner.burn_captions(
                input_video=video_path,
                ass_path=ass_path,
                output_video=out_path,
                codec="libx264",
            )
            assert result.is_file()
            assert result.stat().st_size > 1000

    def test_rapid_speech_short_word_animation_bounds_clamped(self, tmp_path: Path) -> None:
        """Verify that rapid speech or short Whisper alignments (< 50ms) never cause animation overshoots."""
        cfg = _make_config(
            tmp_path,
            overrides={
                "captions": {
                    "style": "bounce",
                    "animation_duration_ms": 140,
                    "animation_scale": 1.20,
                }
            },
        )
        burner = AssCaptionBurner(cfg)
        # 30ms word and 50ms word
        words = [
            WordTimestamp("quick", 0.00, 0.03),
            WordTimestamp("shot", 0.03, 0.08),
        ]
        ass_path = tmp_path / "rapid.ass"
        burner.generate_ass(words, clip_start_offset=0.0, ass_path=ass_path)
        content = ass_path.read_text(encoding="utf-8")
        hormozi_lines = [ln for ln in content.splitlines() if ",Hormozi,," in ln]

        assert len(hormozi_lines) == 2
        # Word 1: dur_ms is 30ms.
        # Bounce keyframes t1, t2 must satisfy 0 < t1 < t2 <= 30
        match1 = re.search(r"\\t\(0,(\d+),\\fscx\d+\\fscy\d+\)\\t\((\d+),(\d+),\\fscx100\\fscy100\)", hormozi_lines[0])
        assert match1 is not None
        t0_1, t1_1, t2_1 = int(match1.group(1)), int(match1.group(2)), int(match1.group(3))
        assert t0_1 == t1_1
        assert 0 < t1_1 < t2_1 <= 30

        # Word 2: dur_ms is 50ms.
        match2 = re.search(r"\\t\(0,(\d+),\\fscx\d+\\fscy\d+\)\\t\((\d+),(\d+),\\fscx100\\fscy100\)", hormozi_lines[1])
        assert match2 is not None
        _, t1_2, t2_2 = int(match2.group(1)), int(match2.group(2)), int(match2.group(3))
        assert 0 < t1_2 < t2_2 <= 50

    def test_sentence_style_bounce_animation(self, tmp_path: Path) -> None:
        """Verify sentence style produces valid two-phase bounce animation when configured."""
        cfg = _make_config(
            tmp_path,
            overrides={
                "captions": {
                    "style": "sentence",
                    "animation": "bounce",
                    "animation_scale": 1.15,
                    "animation_duration_ms": 120,
                }
            },
        )
        burner = AssCaptionBurner(cfg)
        words = [WordTimestamp("full", 0.0, 0.5), WordTimestamp("sentence", 0.5, 1.0)]
        ass_path = tmp_path / "sentence_bounce.ass"
        burner.generate_ass(words, clip_start_offset=0.0, ass_path=ass_path)
        content = ass_path.read_text(encoding="utf-8")
        hormozi_lines = [ln for ln in content.splitlines() if ",Hormozi,," in ln]

        assert len(hormozi_lines) == 1
        line = hormozi_lines[0]
        assert r"\fscx100\fscy100\t(0,60,\fscx115\fscy115)\t(60,120,\fscx100\fscy100)" in line
        assert "FULL SENTENCE" in line

    def test_shadow_color_rgba_and_alpha(self, tmp_path: Path) -> None:
        """Verify 8-digit RGBA hex and ASS &HAABBGGRR parse cleanly for drop shadow."""
        cfg = _make_config(
            tmp_path,
            overrides={
                "captions": {
                    "style": "hormozi",
                    "shadow_offset": 3,
                    "shadow_color": "#00000080",  # CSS RGBA: semi-transparent black
                }
            },
        )
        burner = AssCaptionBurner(cfg)
        # CSS alpha 0x80 (128) -> ASS inverted alpha 255 - 128 = 127 = 0x7F -> &H7F000000
        assert burner.shadow_color == "&H7F000000"
        words = [WordTimestamp("test", 0.0, 0.5)]
        ass_path = tmp_path / "shadow_rgba.ass"
        burner.generate_ass(words, clip_start_offset=0.0, ass_path=ass_path)
        content = ass_path.read_text(encoding="utf-8")
        assert ",&H7F000000," in content

    def test_color_to_ass_style_with_alpha_boundaries(self) -> None:
        """Verify boundary and opacity conversions for CSS RGBA hex, named colors, and ASS formats."""
        # CSS 8-digit hex #RRGGBBAA: #11223380 -> CSS alpha 128 -> ASS alpha 127 = 0x7F -> &H7F332211
        assert _color_to_ass_style_with_alpha("#11223380") == "&H7F332211"
        assert _color_to_ass_style_with_alpha("#11223300") == "&HFF332211"
        assert _color_to_ass_style_with_alpha("#112233FF") == "&H00332211"

        # 6-digit hex with opacity
        assert _color_to_ass_style_with_alpha("#112233", opacity=0.0) == "&HFF332211"
        assert _color_to_ass_style_with_alpha("#112233", opacity=1.0) == "&H00332211"
        assert _color_to_ass_style_with_alpha("#112233", opacity=0.5) == "&H80332211"

        # Named colors and ASS style colors
        assert _color_to_ass_style_with_alpha("&H80332211") == "&H80332211"
        assert _color_to_ass_style_with_alpha("white", opacity=0.5) == "&H80FFFFFF"
        assert _color_to_ass_style_with_alpha("black", opacity=0.6) == "&H66000000"
        assert _color_to_ass_style_with_alpha("", default="&H99000000") == "&H99000000"
        assert _color_to_ass_style_with_alpha("invalid_color", default="&H99000000") == "&H99000000"

    def test_zero_duration_and_inverted_words(self, tmp_path: Path) -> None:
        """Verify zero-duration and inverted timestamp anomalies clamp gracefully without error."""
        cfg = _make_config(
            tmp_path,
            overrides={
                "captions": {
                    "style": "bounce",
                    "animation_duration_ms": 120,
                    "animation_scale": 1.25,
                }
            },
        )
        burner = AssCaptionBurner(cfg)
        words = [
            WordTimestamp("ZERO", 1.0, 1.0),
            WordTimestamp("INVERTED", 2.5, 2.0),
            WordTimestamp("NORMAL", 3.0, 3.5),
        ]
        ass_path = tmp_path / "zero_inv.ass"
        burner.generate_ass(words, clip_start_offset=0.0, ass_path=ass_path)
        content = ass_path.read_text(encoding="utf-8")
        lines = [ln for ln in content.splitlines() if ",Hormozi,," in ln]
        assert len(lines) == 3

        m_zero = re.search(r"\\t\(0,(\d+).*?\\t\((\d+),(\d+)", lines[0])
        assert m_zero is not None
        t1, t1_b, t2 = int(m_zero.group(1)), int(m_zero.group(2)), int(m_zero.group(3))
        assert t1 == t1_b
        assert 0 < t1 < t2

    def test_special_chars_and_formatting_injection(self, tmp_path: Path) -> None:
        """Verify braces, slashes, and control characters do not break ASS syntax."""
        cfg = _make_config(tmp_path, overrides={"captions": {"style": "pop"}})
        burner = AssCaptionBurner(cfg)
        words = [
            WordTimestamp("hello {\\b1} bold", 0.0, 0.5),
            WordTimestamp("newline \\N next", 0.5, 1.0),
            WordTimestamp("slash \\ end\\", 1.0, 1.5),
            WordTimestamp("emoji 🎮🔥 test", 1.5, 2.0),
        ]
        ass_path = tmp_path / "special.ass"
        burner.generate_ass(words, clip_start_offset=0.0, ass_path=ass_path)
        content = ass_path.read_text(encoding="utf-8")
        assert r"\{" in content
        assert r"\}" in content
        lines = [ln for ln in content.splitlines() if ",Hormozi,," in ln]
        assert len(lines) == 4

    def test_multiline_box_custom_resolutions(self, tmp_path: Path) -> None:
        """Verify long multi-line sentences with box styling on non-standard vertical resolutions."""
        resolutions = [[720, 1280], [1080, 1920], [1440, 2560], [1080, 1080]]
        long_sentence = [
            WordTimestamp(f"w{i}", i * 0.2, (i + 1) * 0.2) for i in range(25)
        ]

        for res in resolutions:
            cfg = _make_config(
                tmp_path,
                overrides={
                    "captions": {
                        "style": "sentence",
                        "box_style": True,
                        "box_padding": 12,
                        "box_color": "#112233",
                        "box_opacity": 0.8,
                    },
                    "video": {"resolution": res},
                },
            )
            burner = AssCaptionBurner(cfg)
            assert burner.width == res[0]
            assert burner.height == res[1]

            ass_path = tmp_path / f"box_{res[0]}x{res[1]}.ass"
            burner.generate_ass(long_sentence, clip_start_offset=0.0, ass_path=ass_path)
            content = ass_path.read_text(encoding="utf-8")

            assert f"PlayResX: {res[0]}" in content
            assert f"PlayResY: {res[1]}" in content
            assert ",3,12,0,2," in content
            assert "&H33332211" in content

    def test_full_pipeline_elements_concurrent_with_animated_captions(self, tmp_path: Path) -> None:
        """Verify bounce captions render concurrently with narration intro/outro cards and chat ticker."""
        cfg = _make_config(
            tmp_path,
            overrides={
                "captions": {
                    "style": "bounce",
                    "animation_scale": 1.25,
                    "shadow_offset": 3,
                    "shadow_color": "#000000AA",
                },
                "transformation": {
                    "show_chat_ticker": True,
                },
            },
        )
        burner = AssCaptionBurner(cfg)
        words = [
            WordTimestamp("intro_overlap", 0.5, 1.5),
            WordTimestamp("mid_word_1", 2.5, 3.0),
            WordTimestamp("mid_word_2", 3.0, 3.5),
            WordTimestamp("outro_overlap", 6.5, 7.5),
        ]
        chat_burst = [
            {"author": "Chatter1", "message": "POGGERS!", "offset_seconds": 2.8},
            {"author": "Chatter2", "message": "LMAO", "offset_seconds": 3.2},
        ]
        ass_path = tmp_path / "concurrent.ass"
        burner.generate_ass(
            words=words,
            clip_start_offset=0.0,
            ass_path=ass_path,
            chat_burst=chat_burst,
            narration_intro="Look at this insane play!",
            narration_outro="Did you see that ending?!",
            intro_dur=2.0,
            outro_dur=2.0,
            clip_duration=8.0,
        )
        content = ass_path.read_text(encoding="utf-8")

        clean_lines = [re.sub(r"\{.*?\}", "", ln) for ln in content.splitlines()]
        assert any("LOOK AT THIS" in ln for ln in clean_lines)
        assert any("INSANE PLAY" in ln for ln in clean_lines)
        assert any("DID YOU SEE" in ln for ln in clean_lines)
        assert any("THAT ENDING" in ln for ln in clean_lines)
        assert r"\fscx" in content
        assert "INTRO_OVERLAP" not in content
        assert "OUTRO_OVERLAP" not in content
        assert "MID_WORD_1" in content
        assert "MID_WORD_2" in content
        assert "Chatter1: POGGERS!" in content
        assert "Chatter2: LMAO" in content

    def test_whisper_sub_10ms_word_durations(self, tmp_path: Path) -> None:
        """Test extreme 1ms - 8ms Whisper timestamp anomalies do not cause arithmetic errors."""
        cfg = _make_config(
            tmp_path,
            overrides={
                "captions": {
                    "style": "bounce",
                    "animation_duration_ms": 120,
                    "animation_scale": 1.15,
                }
            },
        )
        burner = AssCaptionBurner(cfg)
        words = [
            WordTimestamp("sub_1ms", 1.000, 1.001),
            WordTimestamp("sub_5ms", 1.001, 1.006),
            WordTimestamp("sub_8ms", 1.006, 1.014),
        ]
        ass_path = tmp_path / "extreme_short.ass"
        burner.generate_ass(words, clip_start_offset=0.0, ass_path=ass_path)
        content = ass_path.read_text(encoding="utf-8")

        lines = [ln for ln in content.splitlines() if ",Hormozi,," in ln]
        assert len(lines) == 3
        for ln in lines:
            m = re.search(r"\\t\(0,(\d+).*?\\t\((\d+),(\d+)", ln)
            assert m is not None
            t1, t1_b, t2 = int(m.group(1)), int(m.group(2)), int(m.group(3))
            assert t1 == t1_b
            assert 0 < t1 < t2

    def test_karaoke_dead_air_pauses_wipe(self, tmp_path: Path) -> None:
        """Verify that in karaoke mode, dead air between words pauses the wipe rather than skipping silence."""
        cfg = _make_config(
            tmp_path,
            overrides={"captions": {"style": "karaoke", "words_per_group": 4}},
        )
        burner = AssCaptionBurner(cfg)
        words = [
            WordTimestamp("wait", 1.0, 1.5),
            WordTimestamp("for", 3.0, 3.4),
            WordTimestamp("it", 3.4, 3.8),
        ]
        ass_path = tmp_path / "karaoke_gap.ass"
        burner.generate_ass(words, clip_start_offset=0.0, ass_path=ass_path)
        content = ass_path.read_text(encoding="utf-8")
        lines = [ln for ln in content.splitlines() if ",Hormozi,," in ln]
        assert len(lines) == 1
        line = lines[0]
        # First word: 0.5s = 50cs wipe
        # Gap between word 0 and word 1: 1.5s = 150cs pause
        # Second word: 0.4s = 40cs wipe
        # Third word contiguous: 0.4s = 40cs wipe
        assert r"{\kf50}WAIT{\k150} {\kf40}FOR {\kf40}IT" in line

    def test_color_conversion_ass_formats_and_inline_bgr(self) -> None:
        """Verify ASS style colors with trailing ampersands and inline BGR hex formats parse correctly."""
        from streamclipper.assembly.captions import _color_to_ass_style
        # ASS 10-char with trailing ampersand: &H8000D7FF& -> &H8000D7FF
        assert _color_to_ass_style("&H8000D7FF&") == "&H8000D7FF"
        assert _color_to_ass_style_with_alpha("&H8000D7FF&") == "&H8000D7FF"
        assert _color_to_ass_style_with_alpha("&H8000D7FF&", opacity=1.0) == "&H0000D7FF"

        # ASS inline BGR: &H00D7FF (or &H00D7FF&) -> &H0000D7FF
        assert _color_to_ass_style("&H00D7FF") == "&H0000D7FF"
        assert _color_to_ass_style("&H00D7FF&") == "&H0000D7FF"
        assert _color_to_ass_style_with_alpha("&H00D7FF") == "&H0000D7FF"
        assert _color_to_ass_style_with_alpha("&H00D7FF&", opacity=0.5) == "&H8000D7FF"

    def test_style_preset_box_opacity_alone_recomputes_box_color(self, tmp_path: Path) -> None:
        """Verify that style_preset with box_opacity alone updates box_color alpha without requiring box_color."""
        cfg = _make_config(
            tmp_path,
            overrides={"captions": {"box_style": True, "box_color": "black", "box_opacity": 0.60}},
        )
        burner = AssCaptionBurner(cfg)
        assert burner.box_color == "&H66000000"

        # Override box_opacity to 0.10 (almost transparent: 0.9 * 255 = 230 = 0xE6)
        words = [WordTimestamp("test", 0.0, 0.5)]
        ass_path = tmp_path / "preset_box_op.ass"
        burner.generate_ass(words, clip_start_offset=0.0, ass_path=ass_path, style_preset={"box_opacity": 0.10})
        content = ass_path.read_text(encoding="utf-8")
        hormozi_line = [ln for ln in content.splitlines() if ln.startswith("Style: Hormozi,")][0]
        assert ",&HE6000000,&HE6000000," in hormozi_line
        assert burner.box_color == "&H66000000"  # State fully restored

    def test_style_preset_style_and_stroke_width_override(self, tmp_path: Path) -> None:
        """Verify style_preset can override style (e.g. karaoke) and stroke_width and restore state."""
        cfg = _make_config(tmp_path, overrides={"captions": {"style": "hormozi", "stroke_width": 4}})
        burner = AssCaptionBurner(cfg)
        assert burner.style == "hormozi"
        assert burner.stroke_width == 4

        words = [WordTimestamp("sing", 1.0, 1.5), WordTimestamp("song", 1.5, 2.0)]
        ass_path = tmp_path / "preset_karaoke.ass"
        burner.generate_ass(
            words,
            clip_start_offset=0.0,
            ass_path=ass_path,
            style_preset={"style": "karaoke", "stroke_width": 6},
        )
        content = ass_path.read_text(encoding="utf-8")
        hormozi_lines = [ln for ln in content.splitlines() if ",Hormozi,," in ln]
        assert len(hormozi_lines) == 1
        assert r"{\kf50}SING {\kf50}SONG" in hormozi_lines[0]
        style_line = [ln for ln in content.splitlines() if ln.startswith("Style: Hormozi,")][0]
        assert ",1,6,6,2," in style_line

        # State fully restored
        assert burner.style == "hormozi"
        assert burner.stroke_width == 4

    def test_karaoke_overlapping_timestamps_and_trailing_backslash(self, tmp_path: Path) -> None:
        """Verify karaoke mode handles overlapping timestamps by extending dialogue duration and sanitizes backslashes."""
        cfg = _make_config(
            tmp_path,
            overrides={"captions": {"style": "karaoke", "words_per_group": 2}},
        )
        burner = AssCaptionBurner(cfg)
        # Word 0: 1.0 -> 2.5 (150cs), Word 1: 2.0 -> 3.0 (100cs, overlaps with word 0)
        # Total wipe = 150cs + 100cs = 250cs (2.5s). Starts at 1.0s -> ends at least at 3.5s
        words = [
            WordTimestamp("escape\\", 1.0, 2.5),
            WordTimestamp("next", 2.0, 3.0),
        ]
        ass_path = tmp_path / "karaoke_overlap.ass"
        burner.generate_ass(words, clip_start_offset=0.0, ass_path=ass_path)
        content = ass_path.read_text(encoding="utf-8")
        hormozi_lines = [ln for ln in content.splitlines() if ",Hormozi,," in ln]
        assert len(hormozi_lines) == 1
        line = hormozi_lines[0]

        # Trailing backslash escaped with zero-width space so {\kf does not get corrupted
        assert "ESCAPE\\\u200b" in line
        assert r"{\kf100}NEXT" in line
        # Dialogue line should extend to at least 0:00:03.50 (not cut off at 0:00:03.00)
        m = re.search(r"Dialogue: \d+,([^,]+),([^,]+),Hormozi", line)
        assert m is not None
        start_ts, end_ts = m.group(1), m.group(2)
        assert start_ts == "0:00:01.00"
        assert end_ts == "0:00:03.50"

    def test_resolve_caption_style_moment_type_mapping_and_overrides(self) -> None:
        """Verify resolve_caption_style maps moment_types and respects explicit overrides."""
        from streamclipper.assembly.captions import resolve_caption_style

        # 1. Automatic mapping based on moment_type
        for m in ("funny", "chat_chaos", "fail"):
            assert resolve_caption_style("auto", moment_type=m) == "bounce"
            assert resolve_caption_style(None, moment_type=m) == "bounce"
            assert resolve_caption_style("", moment_type=m) == "bounce"

        for m in ("hype", "clutch"):
            assert resolve_caption_style("auto", moment_type=m) == "pop"
            assert resolve_caption_style(None, moment_type=m) == "pop"

        for m in ("graphics_detail", "analysis", "lore_secret"):
            assert resolve_caption_style("auto", moment_type=m) == "box"
            assert resolve_caption_style(None, moment_type=m) == "box"

        # Fallback / unmapped moment_types use default_style
        assert resolve_caption_style("auto", moment_type="reaction", default_style="hormozi") == "hormozi"
        assert resolve_caption_style("auto", moment_type="unknown", default_style="sentence") == "sentence"
        assert resolve_caption_style("auto", moment_type=None, default_style="hormozi") == "hormozi"

        # 2. Explicit overrides override moment_type
        assert resolve_caption_style("karaoke", moment_type="funny") == "karaoke"
        assert resolve_caption_style("pop", moment_type="fail") == "pop"
        assert resolve_caption_style("box", moment_type="hype") == "box"
        assert resolve_caption_style("bounce", moment_type="graphics_detail") == "bounce"
        assert resolve_caption_style("single_word", moment_type="clutch") == "single_word"

    def test_generate_ass_auto_resolves_from_moment_type(self, tmp_path: Path) -> None:
        """Verify generate_ass applies moment_type subtitle styling when caption_style='auto'."""
        cfg = _make_config(tmp_path, overrides={"captions": {"style": "hormozi"}})
        burner = AssCaptionBurner(cfg)
        words = [WordTimestamp("look", 0.0, 0.5), WordTimestamp("here", 0.5, 1.0)]

        # funny -> bounce
        p_funny = tmp_path / "funny.ass"
        burner.generate_ass(words, clip_start_offset=0.0, ass_path=p_funny, moment_type="funny")
        c_funny = p_funny.read_text(encoding="utf-8")
        assert r"\fscx100\fscy100" in c_funny
        assert r"\t(0," in c_funny

        # hype -> pop
        p_hype = tmp_path / "hype.ass"
        burner.generate_ass(words, clip_start_offset=0.0, ass_path=p_hype, moment_type="hype")
        c_hype = p_hype.read_text(encoding="utf-8")
        assert r"\fscx" in c_hype
        assert r"\t(0," in c_hype

        # graphics_detail -> box (BorderStyle=3)
        p_box = tmp_path / "box.ass"
        burner.generate_ass(words, clip_start_offset=0.0, ass_path=p_box, moment_type="graphics_detail")
        c_box = p_box.read_text(encoding="utf-8")
        # Style: Hormozi with BorderStyle=3
        hormozi_style = [ln for ln in c_box.splitlines() if ln.startswith("Style: Hormozi,")][0]
        parts = hormozi_style.split(",")
        # Format field 15 is BorderStyle (0-indexed: index 15 is BorderStyle)
        assert parts[15] == "3"

    def test_generate_ass_explicit_caption_style_override(self, tmp_path: Path) -> None:
        """Verify explicit caption_style takes precedence over moment_type in generate_ass."""
        cfg = _make_config(tmp_path, overrides={"captions": {"style": "hormozi"}})
        burner = AssCaptionBurner(cfg)
        words = [WordTimestamp("karaoke", 0.0, 0.5), WordTimestamp("test", 0.5, 1.0)]

        # Explicit karaoke on funny moment -> emits \kf karaoke tags, NOT bounce
        p_kar = tmp_path / "explicit_karaoke.ass"
        burner.generate_ass(
            words,
            clip_start_offset=0.0,
            ass_path=p_kar,
            caption_style="karaoke",
            moment_type="funny",
        )
        c_kar = p_kar.read_text(encoding="utf-8")
        assert r"\kf" in c_kar

        # Burner attributes restored cleanly after invocation
        assert burner.style == "hormozi"

    def test_generate_ass_style_preset_blends_with_moment_type_and_caption_style(
        self, tmp_path: Path
    ) -> None:
        """Verify style_preset (font/colors) blends with moment_type / caption_style animation."""
        cfg = _make_config(tmp_path, overrides={"captions": {"style": "hormozi"}})
        burner = AssCaptionBurner(cfg)
        words = [WordTimestamp("speed", 0.0, 0.5), WordTimestamp("test", 0.5, 1.0)]

        # 1. Preset 0 has animation: "none", but funny moment_type MUST still bounce!
        p_funny = tmp_path / "preset_funny.ass"
        burner.generate_ass(
            words,
            clip_start_offset=0.0,
            ass_path=p_funny,
            style_preset={"font": "CustomFont", "primary_color": "#FFEB3B", "animation": "none"},
            moment_type="funny",
        )
        c_funny = p_funny.read_text(encoding="utf-8")
        assert "CustomFont" in c_funny
        assert r"\fscx100\fscy100" in c_funny
        assert r"\t(0," in c_funny  # Bounce animation tag generated!

        # 2. Preset 2 has animation: "bounce", but graphics_detail moment_type MUST be static boxed text
        p_box = tmp_path / "preset_box.ass"
        burner.generate_ass(
            words,
            clip_start_offset=0.0,
            ass_path=p_box,
            style_preset={"font": "DetailFont", "animation": "bounce"},
            moment_type="graphics_detail",
        )
        c_box = p_box.read_text(encoding="utf-8")
        assert "DetailFont" in c_box
        hormozi_style = [ln for ln in c_box.splitlines() if ln.startswith("Style: Hormozi,")][0]
        assert hormozi_style.split(",")[15] == "3"  # BorderStyle=3
        # Should NOT contain bounce animation
        events = [ln for ln in c_box.splitlines() if ln.startswith("Dialogue: ") and ",Hormozi,," in ln]
        for ev in events:
            assert r"\t(0," not in ev

        # 3. Preset with animation: "none", but explicit caption_style="pop" MUST pop!
        p_pop = tmp_path / "preset_pop.ass"
        burner.generate_ass(
            words,
            clip_start_offset=0.0,
            ass_path=p_pop,
            style_preset={"font": "HypeFont", "animation": "none"},
            caption_style="pop",
        )
        c_pop = p_pop.read_text(encoding="utf-8")
        assert "HypeFont" in c_pop
        assert r"\fscx" in c_pop
        assert r"\t(0," in c_pop

        # 4. Unmapped moment_type="reaction" preserves preset's animation
        p_react = tmp_path / "preset_react.ass"
        burner.generate_ass(
            words,
            clip_start_offset=0.0,
            ass_path=p_react,
            style_preset={"font": "ReactFont", "animation": "pop"},
            moment_type="reaction",
        )
        c_react = p_react.read_text(encoding="utf-8")
        assert "ReactFont" in c_react
        assert r"\fscx" in c_react

    def test_font_fallback_to_arial_black_when_impact_unavailable(self) -> None:
        """When Impact is not installed, resolve_font falls back automatically to Arial Black."""
        def fake_installed(font_family: str) -> bool:
            return font_family.lower() == "arial black"

        with patch("streamclipper.assembly.captions._is_font_installed", side_effect=fake_installed):
            resolved = resolve_font("Impact")
            assert resolved == "Arial Black"

        # Explicit custom font preserved even if not found in system
        with patch("streamclipper.assembly.captions._is_font_installed", return_value=False):
            assert resolve_font("CustomStreamerFont") == "CustomStreamerFont"

    def test_chunk_words_balancing(self) -> None:
        """Word chunking balances into natural 2-3 word segments without 1-word trailing orphans."""
        assert _chunk_words(["HELLO", "WORLD"], max_chunk_size=3) == [["HELLO", "WORLD"]]
        assert _chunk_words(["ONE", "TWO", "THREE"], max_chunk_size=3) == [["ONE", "TWO", "THREE"]]
        # 4 words -> 2 and 2 (borrowing avoids trailing orphan)
        assert _chunk_words(["ONE", "TWO", "THREE", "FOUR"], max_chunk_size=3) == [
            ["ONE", "TWO"],
            ["THREE", "FOUR"],
        ]
        # 5 words -> 3 and 2
        assert _chunk_words(["A", "B", "C", "D", "E"], max_chunk_size=3) == [
            ["A", "B", "C"],
            ["D", "E"],
        ]

    def test_modernized_creator_style_presets(self, tmp_path: Path) -> None:
        """Verify modern creator palettes (Neon Green, Electric Gold, Cyber Cyan, Fire Crimson) with 3D depth."""
        cfg = _make_config(tmp_path)
        presets = cfg.get("captions.style_presets")
        assert len(presets) == 4

        # Preset 1: Neon Green Pop
        p1 = presets[0]
        assert p1["font"] == "Impact"
        assert p1["primary_color"] == "#FFFFFF"
        assert p1["highlight_color"] == "#00FF66"
        assert p1["outline_color"] == "#000000"
        assert p1["stroke_width"] == 8
        assert p1["shadow_offset"] == 6
        assert p1["shadow_color"] == "&H90000000"
        assert p1["animation"] == "bounce"

        # Preset 2: Electric Gold Pop
        p2 = presets[1]
        assert p2["highlight_color"] == "#FFE600"
        assert p2["animation"] == "pop"

        # Preset 3: Cyber Cyan Pop
        p3 = presets[2]
        assert p3["highlight_color"] == "#00F0FF"
        assert p3["animation"] == "bounce"

        # Preset 4: Fire Crimson Pop
        p4 = presets[3]
        assert p4["highlight_color"] == "#FF2A2A"
        assert p4["animation"] == "pop"

        # Test ASS rendering with Preset 1 (Neon Green)
        burner = AssCaptionBurner(cfg)
        words = [WordTimestamp("viral", 0.0, 0.4), WordTimestamp("shorts", 0.4, 0.8)]
        ass_path = tmp_path / "preset_neon.ass"
        burner.generate_ass(words, clip_start_offset=0.0, ass_path=ass_path, style_preset=p1)
        content = ass_path.read_text(encoding="utf-8")

        assert "ScaledBorderAndShadow: yes" in content
        # #00FF66 in BGR is &H66FF00& -> ASS style &H0066FF00
        assert "Style: Hormozi,Impact,82,&H00FFFFFF,&H0066FF00,&H00000000,&H90000000" in content
        assert ",1,8,6,2," in content
        assert r"\fscx130\fscy130" in content

    def test_chunked_narration_intro_outro_dynamic_flow(self, tmp_path: Path) -> None:
        """Narration intro/outro splits 15+ word commentary into 2-3 word chunks with active highlighting and pop animation."""
        cfg = _make_config(tmp_path)
        burner = AssCaptionBurner(cfg)
        long_intro = "Look closely at the water physics and insane car handling in this brand new GTA 6 clip"
        long_outro = "He survived the entire five star chase without even scratching the paint"

        ass_path = tmp_path / "chunked_flow.ass"
        burner.generate_ass(
            words=[],
            clip_start_offset=0.0,
            ass_path=ass_path,
            narration_intro=long_intro,
            narration_outro=long_outro,
            intro_dur=4.5,
            outro_dur=3.5,
            clip_duration=20.0,
        )
        content = ass_path.read_text(encoding="utf-8")
        hormozi_lines = [ln for ln in content.splitlines() if ",Hormozi,," in ln]

        # Intro has 17 words -> 17 dialogue lines (one per active word highlight within its 2-3 word chunk)
        # Outro has 12 words -> 12 dialogue lines
        assert len(hormozi_lines) == 29

        # Verify no single dialogue line contains the entire sentence (no 5-line static wall of text)
        for ln in hormozi_lines:
            # Strip ASS override tags to count words in the displayed subtitle line
            clean_text = re.sub(r"\{.*?\}", "", ln.split(",,")[-1]).strip()
            word_count = len(clean_text.split())
            assert word_count <= 3, f"Line has too many words ({word_count}): {clean_text}"

        # Verify active word pop animation tags are present on every dialogue line
        for ln in hormozi_lines:
            assert r"\fscx130\fscy130" in ln, f"Missing pop animation in: {ln}"
            assert r"\c&H" in ln, f"Missing active highlight in: {ln}"

        # Verify time boundaries
        # First intro chunk starts at 0:00:00.00
        assert hormozi_lines[0].startswith("Dialogue: 0,0:00:00.00,")
        # Last intro line ends at 4.5s -> 0:00:04.50
        assert "0:00:04.50,Hormozi" in hormozi_lines[16]

        # First outro chunk starts at (20.0 - 3.5 = 16.5s) -> 0:00:16.50
        assert hormozi_lines[17].startswith("Dialogue: 0,0:00:16.50,")
        # Last outro line ends at 20.0s -> 0:00:20.00
        assert "0:00:20.00,Hormozi" in hormozi_lines[-1]

    def test_resolve_highlight_bgr_formats(self) -> None:
        """Verify hex, named, and ASS formats parse cleanly to inline ASS BGR tag."""
        assert _resolve_highlight_bgr("#FFE600") == "&H00E6FF&"
        assert _resolve_highlight_bgr("yellow") == "&H00FFFF&"
        assert _resolve_highlight_bgr("&H00FF66&") == "&H00FF66&"
        assert _resolve_highlight_bgr("&H0066FF00") == "&H66FF00&"
        assert _resolve_highlight_bgr("", default="&H00E6FF&") == "&H00E6FF&"
        assert _resolve_highlight_bgr("invalid_color", default="&H00E6FF&") == "&H00E6FF&"

    def test_style_preset_aliased_keys(self, tmp_path: Path) -> None:
        """Verify style_preset accepts color/stroke_color/outline_width aliases."""
        cfg = _make_config(tmp_path)
        burner = AssCaptionBurner(cfg)
        preset = {
            "color": "#112233",
            "stroke_color": "#445566",
            "outline_width": 10,
            "highlight_color": "&H00FF88&",
        }
        words = [WordTimestamp("alias", 0.0, 0.5)]
        ass_path = tmp_path / "aliased.ass"
        burner.generate_ass(words, clip_start_offset=0.0, ass_path=ass_path, style_preset=preset)
        content = ass_path.read_text(encoding="utf-8")
        # PrimaryColour from #112233 -> &H00332211
        # OutlineColour from #445566 -> &H00665544
        # Outline width = 10
        assert "&H00332211" in content
        assert "&H00665544" in content
        assert ",1,10,6,2," in content
        assert r"\c&H00FF88&" in content

    def test_speech_caption_words_per_group_three(self, tmp_path: Path) -> None:
        """Verify speech words are grouped cleanly by words_per_group=3."""
        cfg = _make_config(tmp_path)
        burner = AssCaptionBurner(cfg)
        words = [
            WordTimestamp("one", 0.0, 0.3),
            WordTimestamp("two", 0.3, 0.6),
            WordTimestamp("three", 0.6, 0.9),
            WordTimestamp("four", 0.9, 1.2),
        ]
        ass_path = tmp_path / "grouped_speech.ass"
        burner.generate_ass(words, clip_start_offset=0.0, ass_path=ass_path)
        content = ass_path.read_text(encoding="utf-8")
        hormozi_lines = [ln for ln in content.splitlines() if ",Hormozi,," in ln]

        assert len(hormozi_lines) == 4
        clean_lines = [re.sub(r"\{.*?\}", "", ln.split(",,")[-1]).strip() for ln in hormozi_lines]
        assert clean_lines[0] == "ONE TWO THREE"
        assert clean_lines[1] == "ONE TWO THREE"
        assert clean_lines[2] == "ONE TWO THREE"
        assert clean_lines[3] == "FOUR"

    def test_callout_style_inherits_3d_shadow(self, tmp_path: Path) -> None:
        """Verify Style: Callout inherits shadow_offset and shadow_color for 3D depth."""
        cfg = _make_config(
            tmp_path,
            overrides={
                "captions": {
                    "shadow_offset": 6,
                    "shadow_color": "&H90000000",
                    "stroke_width": 8,
                }
            },
        )
        burner = AssCaptionBurner(cfg)
        from streamclipper.assembly.layout import LayoutGeometry, Box
        dummy_layout = LayoutGeometry(
            kind="blur_fill",
            width=1080,
            height=1920,
            hook_band=Box(0, 0, 1080, 100),
            facecam_box=None,
            game_box=Box(0, 480, 1080, 960),
            caption_band=Box(0, 1440, 1080, 480),
        )
        words = [WordTimestamp("test", 0.0, 0.5)]
        ass_path = tmp_path / "callout_3d.ass"
        burner.generate_ass(
            words,
            clip_start_offset=0.0,
            ass_path=ass_path,
            layout=dummy_layout,
            callout="NICE PLAY",
            callout_window=(0.0, 1.0),
        )
        content = ass_path.read_text(encoding="utf-8")
        callout_style_line = [ln for ln in content.splitlines() if ln.startswith("Style: Callout")][0]
        # Outline 8, Shadow 6, BackColour &H90000000
        assert "&H90000000" in callout_style_line
        assert ",1,8,6,8," in callout_style_line

    def test_sentence_style_chunked_narration_animation(self, tmp_path: Path) -> None:
        """Narration chunks in sentence style receive pop/bounce animation tags."""
        cfg = _make_config(
            tmp_path,
            overrides={
                "captions": {
                    "style": "sentence",
                    "animation": "pop",
                    "animation_scale": 1.30,
                    "animation_duration_ms": 140,
                }
            },
        )
        burner = AssCaptionBurner(cfg)
        events = burner._generate_narration_chunk_events(
            "Look at this crazy car handling",
            start_time=0.0,
            end_time=3.0,
            primary_inline_bgr="&HFFFFFF&",
        )
        assert len(events) >= 2
        for ev in events:
            assert r"\fscx130\fscy130" in ev



