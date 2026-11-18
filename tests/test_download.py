"""Unit tests for the audio and video segment downloaders (Packages 2.1 & 2.2)."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yt_dlp

from streamclipper.download.audio import (
    AudioDownloadError,
    download_audio,
    extract_stream_info,
)
from streamclipper.download.video import (
    VIDEO_FORMAT_1080P,
    VideoDownloadError,
    download_multiple_segments,
    download_segment,
)
from streamclipper.models import Platform, StreamInfo

SAMPLE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
SAMPLE_INFO: dict[str, Any] = {
    "id": "dQw4w9WgXcQ",
    "title": "GTA 6 Launch Day 4-Hour Stream VOD",
    "uploader": "ViceCityStreamer",
    "channel": "ViceCityStreamer Channel",
    "duration": 14400,
    "view_count": 250_000,
    "extractor_key": "Youtube",
    "webpage_url": SAMPLE_URL,
}


def _patch_yt_dlp(
    monkeypatch: pytest.MonkeyPatch,
    *,
    extract_return: Any = None,
    extract_side_effect: Exception | None = None,
    download_side_effect: Any = None,
) -> tuple[list[dict[str, Any]], list[MagicMock]]:
    """Patch yt_dlp.YoutubeDL and capture every instantiation's options and mock instance."""
    captured_opts: list[dict[str, Any]] = []
    instances: list[MagicMock] = []

    def _factory(opts: dict[str, Any]) -> MagicMock:
        captured_opts.append(opts)
        inst = MagicMock()
        inst.__enter__.return_value = inst
        inst.__exit__.return_value = False

        if extract_side_effect is not None:
            inst.extract_info.side_effect = extract_side_effect
        else:
            inst.extract_info.return_value = (
                SAMPLE_INFO if extract_return is None else extract_return
            )

        if download_side_effect is not None:
            inst.download.side_effect = download_side_effect

        instances.append(inst)
        return inst

    monkeypatch.setattr(yt_dlp, "YoutubeDL", _factory)
    return captured_opts, instances


# ---------------------------------------------------------------------------
# Metadata Extraction Tests
# ---------------------------------------------------------------------------


class TestExtractStreamInfo:
    def test_extract_stream_info_maps_fields_accurately(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured_opts, instances = _patch_yt_dlp(monkeypatch)

        info = extract_stream_info(SAMPLE_URL)

        assert isinstance(info, StreamInfo)
        assert info.url == SAMPLE_URL
        assert info.title == "GTA 6 Launch Day 4-Hour Stream VOD"
        assert info.platform == Platform.YOUTUBE
        assert info.streamer == "ViceCityStreamer"
        assert info.game == ""
        assert info.duration_seconds == 14400
        assert info.view_count == 250_000
        assert info.video_id == "dQw4w9WgXcQ"

        assert len(captured_opts) == 1
        assert captured_opts[0]["skip_download"] is True
        assert captured_opts[0]["noplaylist"] is True
        instances[0].extract_info.assert_called_once_with(SAMPLE_URL, download=False)

    def test_extract_stream_info_preserves_caller_short_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        short_url = "https://youtu.be/dQw4w9WgXcQ"
        _patch_yt_dlp(monkeypatch)

        info = extract_stream_info(short_url)

        assert info.url == short_url
        assert info.video_id == "dQw4w9WgXcQ"

    @pytest.mark.parametrize(
        "metadata,expected_streamer",
        [
            ({"id": "v1", "title": "T", "channel": "ChannelName"}, "ChannelName"),
            ({"id": "v2", "title": "T", "creator": "CreatorName"}, "CreatorName"),
            ({"id": "v3", "title": "T"}, "Unknown"),
        ],
    )
    def test_extract_stream_info_streamer_fallbacks_and_missing_optional_fields(
        self,
        monkeypatch: pytest.MonkeyPatch,
        metadata: dict[str, Any],
        expected_streamer: str,
    ) -> None:
        _patch_yt_dlp(monkeypatch, extract_return=metadata)

        info = extract_stream_info(SAMPLE_URL)

        assert info.streamer == expected_streamer
        assert info.duration_seconds == 0
        assert info.view_count == 0

    @pytest.mark.parametrize(
        "extractor_key,url,expected_platform",
        [
            ("Youtube", "https://www.youtube.com/watch?v=abc", Platform.YOUTUBE),
            ("TwitchVod", "https://www.twitch.tv/videos/12345", Platform.TWITCH),
            ("Kick", "https://kick.com/streamer/videos/12345", Platform.KICK),
        ],
    )
    def test_extract_stream_info_detects_platform(
        self,
        monkeypatch: pytest.MonkeyPatch,
        extractor_key: str,
        url: str,
        expected_platform: Platform,
    ) -> None:
        _patch_yt_dlp(
            monkeypatch,
            extract_return={
                "id": "plat123",
                "title": "Stream",
                "uploader": "Host",
                "extractor_key": extractor_key,
                "duration": 3600,
                "view_count": 500,
            },
        )

        info = extract_stream_info(url)
        assert info.platform == expected_platform

    def test_extract_stream_info_raises_when_info_is_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured_opts: list[dict[str, Any]] = []

        def _factory(opts: dict[str, Any]) -> MagicMock:
            captured_opts.append(opts)
            inst = MagicMock()
            inst.__enter__.return_value = inst
            inst.__exit__.return_value = False
            inst.extract_info.return_value = None
            return inst

        monkeypatch.setattr(yt_dlp, "YoutubeDL", _factory)

        with pytest.raises(AudioDownloadError, match="Could not extract stream metadata"):
            extract_stream_info(SAMPLE_URL)


# ---------------------------------------------------------------------------
# Audio Download Tests
# ---------------------------------------------------------------------------


class TestDownloadAudio:
    def test_download_audio_configures_16khz_mono_wav_and_returns_result(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "nested_temp"
        captured_opts, instances = _patch_yt_dlp(monkeypatch)

        audio_path, stream_info = download_audio(SAMPLE_URL, output_dir)

        assert output_dir.is_dir()
        assert audio_path == output_dir / "dQw4w9WgXcQ.wav"
        assert stream_info.video_id == "dQw4w9WgXcQ"
        assert stream_info.title == SAMPLE_INFO["title"]
        assert stream_info.streamer == SAMPLE_INFO["uploader"]
        assert stream_info.duration_seconds == 14400
        assert stream_info.view_count == 250_000

        # Call 0: metadata extraction; Call 1: audio download
        assert len(captured_opts) == 2
        dl_opts = captured_opts[1]
        assert dl_opts["format"] == "bestaudio/best"
        assert dl_opts["noplaylist"] is True
        assert dl_opts["outtmpl"] == str(output_dir / "%(id)s.%(ext)s")
        assert dl_opts["postprocessors"] == [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ]
        assert dl_opts["postprocessor_args"] == {
            "extractaudio": ["-ar", "16000", "-ac", "1"],
        }
        assert "progress_hooks" not in dl_opts
        instances[1].download.assert_called_once_with([SAMPLE_URL])

    def test_download_audio_skips_download_when_cached_wav_exists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cached_wav = tmp_path / "dQw4w9WgXcQ.wav"
        cached_wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

        captured_opts, instances = _patch_yt_dlp(monkeypatch)

        audio_path, stream_info = download_audio(SAMPLE_URL, tmp_path)

        assert audio_path == cached_wav
        assert stream_info.video_id == "dQw4w9WgXcQ"
        assert stream_info.title == SAMPLE_INFO["title"]
        # Only metadata extraction ran; no download YoutubeDL instance created
        assert len(captured_opts) == 1
        instances[0].download.assert_not_called()

    def test_download_audio_forwards_progress_callback(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        events: list[dict[str, Any]] = []

        def _on_progress(payload: dict[str, Any]) -> None:
            events.append(payload)

        captured_opts: list[dict[str, Any]] = []

        def _factory(opts: dict[str, Any]) -> MagicMock:
            captured_opts.append(opts)
            inst = MagicMock()
            inst.__enter__.return_value = inst
            inst.__exit__.return_value = False
            inst.extract_info.return_value = SAMPLE_INFO

            def _simulate_download(_urls: list[str]) -> None:
                for hook in opts.get("progress_hooks", []):
                    hook({"status": "downloading", "downloaded_bytes": 1024, "total_bytes": 2048})
                    hook({"status": "finished", "filename": str(tmp_path / "dQw4w9WgXcQ.wav")})

            inst.download.side_effect = _simulate_download
            return inst

        monkeypatch.setattr(yt_dlp, "YoutubeDL", _factory)

        audio_path, _ = download_audio(SAMPLE_URL, tmp_path, progress_callback=_on_progress)

        assert audio_path == tmp_path / "dQw4w9WgXcQ.wav"
        assert captured_opts[1]["progress_hooks"] == [_on_progress]
        assert len(events) == 2
        assert events[0]["status"] == "downloading"
        assert events[1]["status"] == "finished"

    def test_download_audio_raises_when_video_id_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_yt_dlp(
            monkeypatch,
            extract_return={"title": "No ID Stream", "uploader": "Someone", "duration": 60},
        )

        with pytest.raises(AudioDownloadError, match="Could not determine video ID"):
            download_audio(SAMPLE_URL, tmp_path)


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------


class TestDownloadErrorHandling:
    @pytest.mark.parametrize(
        "yt_dlp_message,expected_substring",
        [
            ("ERROR: [youtube] abc: Private video. Sign in if you've been granted access", "private"),
            ("ERROR: [youtube] abc: Video unavailable. This video has been removed", "deleted or unavailable"),
            (
                "ERROR: [youtube] abc: Sign in to confirm your age. This video may be inappropriate",
                "age-restricted",
            ),
            (
                "ERROR: [youtube] abc: The uploader has not made this video available in your country",
                "region-locked",
            ),
            ("ERROR: 'not-a-url' is not a valid URL.", "Invalid or unsupported stream URL"),
            ("ERROR: Unsupported URL: https://example.invalid/stream", "Invalid or unsupported stream URL"),
            ("ERROR: Unexpected network timeout", "Failed to download audio"),
        ],
    )
    def test_extract_stream_info_translates_download_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
        yt_dlp_message: str,
        expected_substring: str,
    ) -> None:
        _patch_yt_dlp(
            monkeypatch,
            extract_side_effect=yt_dlp.utils.DownloadError(yt_dlp_message),
        )

        with pytest.raises(AudioDownloadError) as exc_info:
            extract_stream_info("https://www.youtube.com/watch?v=bad123")

        assert expected_substring.lower() in str(exc_info.value).lower()
        assert "https://www.youtube.com/watch?v=bad123" in str(exc_info.value)

    def test_download_audio_translates_download_error_during_download_phase(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_yt_dlp(
            monkeypatch,
            download_side_effect=yt_dlp.utils.DownloadError(
                "ERROR: Sign in to confirm your age"
            ),
        )

        with pytest.raises(AudioDownloadError, match="age-restricted"):
            download_audio(SAMPLE_URL, tmp_path)

    def test_download_audio_translates_postprocessing_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_yt_dlp(
            monkeypatch,
            download_side_effect=yt_dlp.utils.PostProcessingError(
                "ffprobe and ffmpeg not found"
            ),
        )

        with pytest.raises(AudioDownloadError, match="ffprobe and ffmpeg not found"):
            download_audio(SAMPLE_URL, tmp_path)

    @pytest.mark.parametrize("bad_url", ["", "   "])
    def test_empty_url_raises_descriptive_error(self, bad_url: str, tmp_path: Path) -> None:
        with pytest.raises(AudioDownloadError, match="non-empty string"):
            extract_stream_info(bad_url)

        with pytest.raises(AudioDownloadError, match="non-empty string"):
            download_audio(bad_url, tmp_path)


# ---------------------------------------------------------------------------
# Video Segment Downloader Tests (Package 2.2)
# ---------------------------------------------------------------------------


class TestVideoSegmentDownloader:
    def test_download_segment_options_and_default_buffer_padding(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "segments"
        captured_opts, instances = _patch_yt_dlp(monkeypatch)

        range_calls: list[tuple[Any, Any]] = []
        orig_range_func = yt_dlp.utils.download_range_func

        def _spy_range_func(chapters: Any, ranges: Any, *args: Any, **kwargs: Any) -> Any:
            range_calls.append((chapters, ranges))
            return orig_range_func(chapters, ranges, *args, **kwargs)

        monkeypatch.setattr(yt_dlp.utils, "download_range_func", _spy_range_func)

        result = download_segment(SAMPLE_URL, 100.0, 145.0, output_dir)

        expected_path = output_dir / "dQw4w9WgXcQ_100_145.mp4"
        assert output_dir.is_dir()
        assert result == expected_path

        # Call 0: video_id probe; Call 1: ranged video download
        assert len(captured_opts) == 2
        assert captured_opts[0]["skip_download"] is True
        assert captured_opts[0]["noplaylist"] is True
        instances[0].extract_info.assert_called_once_with(SAMPLE_URL, download=False)

        dl_opts = captured_opts[1]
        assert dl_opts["format"] == VIDEO_FORMAT_1080P
        assert dl_opts["format"] == "bestvideo[height<=1080]+bestaudio/best[height<=1080]"
        assert dl_opts["merge_output_format"] == "mp4"
        assert dl_opts["force_keyframes_at_cuts"] is True
        assert dl_opts["noplaylist"] is True
        assert dl_opts["outtmpl"] == str(output_dir / "dQw4w9WgXcQ_100_145.%(ext)s")

        assert range_calls == [(None, [(97.0, 148.0)])]
        resolved_ranges = list(dl_opts["download_ranges"]({}, None))
        assert resolved_ranges == [{"start_time": 97.0, "end_time": 148.0}]
        instances[1].download.assert_called_once_with([SAMPLE_URL])

    def test_download_segment_clamps_actual_start_at_zero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured_opts, _ = _patch_yt_dlp(monkeypatch)

        result = download_segment(
            SAMPLE_URL,
            start_seconds=1.5,
            end_seconds=12.0,
            output_dir=tmp_path,
            buffer_seconds=3.0,
        )

        # Filename uses requested (unbuffered) int bounds
        assert result == tmp_path / "dQw4w9WgXcQ_1_12.mp4"
        dl_opts = captured_opts[1]
        resolved_ranges = list(dl_opts["download_ranges"]({}, None))
        assert resolved_ranges == [{"start_time": 0.0, "end_time": 15.0}]

    @pytest.mark.parametrize(
        "buffer_seconds,expected_start,expected_end",
        [
            (0.0, 20.0, 50.0),
            (5.0, 15.0, 55.0),
            (25.0, 0.0, 75.0),
        ],
    )
    def test_download_segment_custom_buffer_seconds(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        buffer_seconds: float,
        expected_start: float,
        expected_end: float,
    ) -> None:
        captured_opts, _ = _patch_yt_dlp(monkeypatch)

        result = download_segment(
            SAMPLE_URL,
            start_seconds=20.0,
            end_seconds=50.0,
            output_dir=tmp_path,
            buffer_seconds=buffer_seconds,
        )

        assert result == tmp_path / "dQw4w9WgXcQ_20_50.mp4"
        resolved_ranges = list(captured_opts[1]["download_ranges"]({}, None))
        assert resolved_ranges == [{"start_time": expected_start, "end_time": expected_end}]

    def test_download_segment_skips_when_cached_mp4_exists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cached_mp4 = tmp_path / "dQw4w9WgXcQ_100_145.mp4"
        cached_mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42")

        captured_opts, instances = _patch_yt_dlp(monkeypatch)

        result = download_segment(SAMPLE_URL, 100.0, 145.0, tmp_path)

        assert result == cached_mp4
        # Only metadata probe ran; no download YoutubeDL created
        assert len(captured_opts) == 1
        instances[0].download.assert_not_called()

    def test_download_multiple_segments_reuses_video_id_and_skips_cached(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Pre-cache beat #2 so only beat #1 and beat #3 should trigger downloads
        cached_beat_2 = tmp_path / "dQw4w9WgXcQ_60_75.mp4"
        cached_beat_2.write_bytes(b"\x00\x00\x00\x18ftypmp42")

        captured_opts, instances = _patch_yt_dlp(monkeypatch)

        segments = [(2.0, 10.0), (60.0, 75.0), (120.5, 138.9)]
        paths = download_multiple_segments(
            SAMPLE_URL,
            segments=segments,
            output_dir=tmp_path,
            buffer_seconds=4.0,
        )

        assert paths == [
            tmp_path / "dQw4w9WgXcQ_2_10.mp4",
            cached_beat_2,
            tmp_path / "dQw4w9WgXcQ_120_138.mp4",
        ]

        # 1 metadata probe + 2 actual segment downloads (beat #2 skipped)
        assert len(captured_opts) == 3
        instances[0].extract_info.assert_called_once_with(SAMPLE_URL, download=False)

        beat_1_ranges = list(captured_opts[1]["download_ranges"]({}, None))
        assert beat_1_ranges == [{"start_time": 0.0, "end_time": 14.0}]
        assert captured_opts[1]["force_keyframes_at_cuts"] is True
        assert captured_opts[1]["merge_output_format"] == "mp4"
        instances[1].download.assert_called_once_with([SAMPLE_URL])

        beat_3_ranges = list(captured_opts[2]["download_ranges"]({}, None))
        assert beat_3_ranges == [{"start_time": 116.5, "end_time": 142.9}]
        assert captured_opts[2]["force_keyframes_at_cuts"] is True
        assert captured_opts[2]["merge_output_format"] == "mp4"
        instances[2].download.assert_called_once_with([SAMPLE_URL])

    def test_download_multiple_segments_empty_list_skips_yt_dlp(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured_opts, _ = _patch_yt_dlp(monkeypatch)

        result = download_multiple_segments(SAMPLE_URL, [], tmp_path)

        assert result == []
        assert captured_opts == []

    @pytest.mark.parametrize(
        "url,start,end,buf",
        [
            ("", 10.0, 20.0, 3.0),
            ("   ", 10.0, 20.0, 3.0),
            (SAMPLE_URL, -1.0, 20.0, 3.0),
            (SAMPLE_URL, 20.0, 20.0, 3.0),
            (SAMPLE_URL, 25.0, 20.0, 3.0),
            (SAMPLE_URL, 10.0, 20.0, -0.5),
        ],
    )
    def test_download_segment_validation_rejects_invalid_arguments(
        self,
        tmp_path: Path,
        url: str,
        start: float,
        end: float,
        buf: float,
    ) -> None:
        with pytest.raises(ValueError):
            download_segment(url, start, end, tmp_path, buffer_seconds=buf)

    def test_download_multiple_segments_validates_all_segments_before_network(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured_opts, _ = _patch_yt_dlp(monkeypatch)

        with pytest.raises(ValueError):
            download_multiple_segments(
                SAMPLE_URL,
                segments=[(10.0, 20.0), (50.0, 40.0)],
                output_dir=tmp_path,
            )

        assert captured_opts == []

    def test_download_segment_wraps_yt_dlp_errors(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_yt_dlp(
            monkeypatch,
            download_side_effect=yt_dlp.utils.DownloadError(
                "ERROR: [youtube] abc: Private video"
            ),
        )

        with pytest.raises(VideoDownloadError, match="private"):
            download_segment(SAMPLE_URL, 10.0, 20.0, tmp_path)

    def test_download_segment_raises_when_video_id_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_yt_dlp(monkeypatch, extract_return={"title": "Missing ID"})

        with pytest.raises(VideoDownloadError, match="Could not determine video ID"):
            download_segment(SAMPLE_URL, 10.0, 20.0, tmp_path)

    def test_download_audio_handles_na_metadata_and_offline_sidecar_cache(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_yt_dlp(
            monkeypatch,
            extract_return={
                "id": "dQw4w9WgXcQ",
                "title": "NA Metadata Stream",
                "uploader": "Streamer",
                "duration": "NA",
                "view_count": float("nan"),
            },
        )
        info = extract_stream_info(SAMPLE_URL)
        assert info.duration_seconds == 0
        assert info.view_count == 0

        # Pre-seed WAV + info sidecar -> offline fast-path must make zero yt-dlp calls
        (tmp_path / "dQw4w9WgXcQ.wav").write_bytes(b"RIFF....WAVE")
        (tmp_path / "dQw4w9WgXcQ.info.json").write_text(info.to_json(), encoding="utf-8")
        captured_opts, _ = _patch_yt_dlp(monkeypatch)

        audio_path, cached_info = download_audio(SAMPLE_URL, tmp_path)
        assert audio_path == tmp_path / "dQw4w9WgXcQ.wav"
        assert cached_info.title == "NA Metadata Stream"
        assert captured_opts == []

    @pytest.mark.parametrize(
        "start,end,buf",
        [
            (float("nan"), 20.0, 3.0),
            (10.0, float("inf"), 3.0),
            (10.0, 20.0, float("nan")),
            (True, 20.0, 3.0),
        ],
    )
    def test_download_segment_rejects_non_finite_and_bool_ranges(
        self, tmp_path: Path, start: Any, end: Any, buf: Any
    ) -> None:
        with pytest.raises(ValueError):
            download_segment(SAMPLE_URL, start, end, tmp_path, buffer_seconds=buf)

    def test_download_segment_invalidates_cache_on_buffer_change_and_detects_alt_container(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        mp4 = tmp_path / "dQw4w9WgXcQ_10_40.mp4"
        mp4.write_bytes(b"old-buf-3")
        (tmp_path / "dQw4w9WgXcQ_10_40.meta.json").write_text(
            '{"start_seconds": 10.0, "end_seconds": 40.0, "buffer_seconds": 3.0}',
            encoding="utf-8",
        )
        captured_opts, _ = _patch_yt_dlp(monkeypatch)
        # Same bounds with buffer_seconds=5.0 must re-download instead of returning stale buffer=3.0
        download_segment(SAMPLE_URL, 10.0, 40.0, tmp_path, buffer_seconds=5.0)
        assert len(captured_opts) == 2

        # Alternate container fallback (.webm) when yt-dlp doesn't produce .mp4
        alt_dir = tmp_path / "alt"
        alt_dir.mkdir()

        def _write_webm(_urls: list[str]) -> None:
            (alt_dir / "dQw4w9WgXcQ_50_80.webm").write_bytes(b"webm-data")

        _, alt_instances = _patch_yt_dlp(monkeypatch, download_side_effect=_write_webm)
        resolved = download_segment(SAMPLE_URL, 50.0, 80.0, alt_dir)
        assert resolved == alt_dir / "dQw4w9WgXcQ_50_80.webm"

        # Subsequent call with identical bounds/buffer reuses cached .webm without re-downloading
        cached_alt = download_segment(SAMPLE_URL, 50.0, 80.0, alt_dir)
        assert cached_alt == alt_dir / "dQw4w9WgXcQ_50_80.webm"
        assert sum(inst.download.call_count for inst in alt_instances) == 1

        # Changing buffer_seconds unlinks stale .mp4 so a newly written .webm is NOT masked by stale .mp4
        stale_dir = tmp_path / "stale"
        stale_dir.mkdir()
        stale_mp4 = stale_dir / "dQw4w9WgXcQ_10_50.mp4"
        stale_mp4.write_bytes(b"old-buffer-3")
        (stale_dir / "dQw4w9WgXcQ_10_50.meta.json").write_text(
            '{"start_seconds": 10.0, "end_seconds": 50.0, "buffer_seconds": 3.0}',
            encoding="utf-8",
        )

        def _write_new_webm(_urls: list[str]) -> None:
            (stale_dir / "dQw4w9WgXcQ_10_50.webm").write_bytes(b"new-buffer-5")

        _patch_yt_dlp(monkeypatch, download_side_effect=_write_new_webm)
        refreshed = download_segment(SAMPLE_URL, 10.0, 50.0, stale_dir, buffer_seconds=5.0)
        assert refreshed == stale_dir / "dQw4w9WgXcQ_10_50.webm"
        assert refreshed.read_bytes() == b"new-buffer-5"
        assert not stale_mp4.exists()
