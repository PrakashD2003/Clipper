"""Tests for Phase 4 video assembly modules (Package 4.3: ASS captions & chat ticker)."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from unittest.mock import patch

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
from streamclipper.assembly.composer import remap_timeline_for_beats
from streamclipper.config import Config, HardwareProfile, detect_hardware_profile
from streamclipper.models import WordTimestamp


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
