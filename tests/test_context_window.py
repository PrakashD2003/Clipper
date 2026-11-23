"""Unit tests for 5-minute context window extraction with interleaved transcript and chat."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from streamclipper.analysis.context_window import (
    ChatBucket,
    ContextWindow,
    TimelineSegment,
    extract_context_window,
)
from streamclipper.models import (
    TranscriptSegment,
    TranscriptionResult,
    WordTimestamp,
)


@pytest.fixture
def sample_transcript() -> TranscriptionResult:
    segments = [
        TranscriptSegment(
            text="Look at that incredible stunt!",
            start=100.0,
            end=105.0,
            words=[
                WordTimestamp("Look", 100.0, 101.0, 0.99),
                WordTimestamp("at", 101.0, 101.5, 0.99),
                WordTimestamp("that", 101.5, 102.0, 0.99),
                WordTimestamp("incredible", 102.0, 103.5, 0.99),
                WordTimestamp("stunt!", 103.5, 105.0, 0.99),
            ],
        ),
        TranscriptSegment(
            text="I can't believe we survived that drop.",
            start=115.0,
            end=120.0,
            words=[
                WordTimestamp("I", 115.0, 115.5, 0.99),
                WordTimestamp("can't", 115.5, 116.0, 0.99),
                WordTimestamp("believe", 116.0, 117.0, 0.99),
                WordTimestamp("we", 117.0, 117.5, 0.99),
                WordTimestamp("survived", 117.5, 118.5, 0.99),
                WordTimestamp("that", 118.5, 119.0, 0.99),
                WordTimestamp("drop.", 119.0, 120.0, 0.99),
            ],
        ),
    ]
    return TranscriptionResult(
        segments=segments,
        full_text="Look at that incredible stunt! I can't believe we survived that drop.",
        language="en",
    )


def test_extract_context_window_with_transcript_and_chat(
    tmp_path: Path, sample_transcript: TranscriptionResult
) -> None:
    # Create chat CSV:
    # Normal chat: 1 msg every 10s from 0 to 200s (avg ~0.1 msgs/s)
    # Burst at 105-110s: 10 msgs in 5s (2.0 msgs/s > 3x avg)
    rows = []
    for t in range(0, 200, 10):
        rows.append({"offset_seconds": float(t), "author": f"user{t}", "message": f"msg {t}"})
    for i in range(10):
        rows.append({"offset_seconds": 106.0 + i * 0.3, "author": f"fan{i}", "message": "OMG WHAT A JUMP!!!"})

    chat_csv = tmp_path / "stream_chat.csv"
    pd.DataFrame(rows).to_csv(chat_csv, index=False)

    ctx = extract_context_window(
        sample_transcript,
        chat_csv,
        center_time=110.0,
        window_before=60.0,  # 50.0 to 170.0
        window_after=60.0,
        bucket_seconds=5.0,
    )

    assert ctx.center_time == 110.0
    assert ctx.window_start == 50.0
    assert ctx.window_end == 170.0
    assert ctx.total_chat_messages > 0
    assert ctx.avg_chat_rate > 0.0

    # Burst should be detected around 105.0s
    assert len(ctx.burst_timestamps) >= 1
    assert any(105.0 <= b <= 110.0 for b in ctx.burst_timestamps)

    # Timeline text generation
    llm_text = ctx.to_llm_text()
    assert "=== STREAM CONTEXT WINDOW" in llm_text
    assert "Look at that incredible stunt!" in llm_text
    assert "[BURST]" in llm_text
    assert "OMG WHAT A JUMP!!!" in llm_text


def test_extract_context_window_without_chat(sample_transcript: TranscriptionResult) -> None:
    ctx = extract_context_window(
        sample_transcript,
        None,
        center_time=100.0,
        window_before=30.0,
        window_after=30.0,
    )
    assert ctx.total_chat_messages == 0
    assert ctx.avg_chat_rate == 0.0
    assert ctx.burst_timestamps == []

    text = ctx.to_llm_text()
    assert "Chat: 0 msgs" in text
    assert "Bursts at: none" in text
    assert "Live Chat: (no messages)" in text


def test_extract_context_window_clamps_start_at_zero(
    sample_transcript: TranscriptionResult,
) -> None:
    ctx = extract_context_window(
        sample_transcript,
        None,
        center_time=30.0,
        window_before=180.0,
        window_after=60.0,
    )
    assert ctx.window_start == 0.0
    assert ctx.window_end == 90.0


def test_extract_context_window_chat_at_exact_win_end(
    tmp_path: Path, sample_transcript: TranscriptionResult
) -> None:
    """Chat message at the exact window_end boundary must be captured and not dropped."""
    chat_csv = tmp_path / "end_chat.csv"
    pd.DataFrame([
        {"offset_seconds": 150.0, "author": "user1", "message": "exact end msg"},
    ]).to_csv(chat_csv, index=False)

    ctx = extract_context_window(
        sample_transcript,
        chat_csv,
        center_time=100.0,
        window_before=50.0,  # [50.0, 150.0]
        window_after=50.0,
        bucket_seconds=5.0,
    )
    assert ctx.total_chat_messages == 1
    # Verify that the message is present in the final segment
    assert any(
        seg.chat_bucket and "exact end msg" in seg.chat_bucket.top_messages
        for seg in ctx.segments
    )


def test_extract_context_window_no_false_bursts_on_sparse_chat(
    tmp_path: Path, sample_transcript: TranscriptionResult
) -> None:
    """Sparse chat (e.g. only 2 messages in 5 minutes) must not be flagged as a burst."""
    chat_csv = tmp_path / "sparse_chat.csv"
    pd.DataFrame([
        {"offset_seconds": 100.0, "author": "user1", "message": "hello"},
        {"offset_seconds": 101.0, "author": "user2", "message": "world"},
    ]).to_csv(chat_csv, index=False)

    ctx = extract_context_window(
        sample_transcript,
        chat_csv,
        center_time=100.0,
        window_before=150.0,
        window_after=150.0,
        bucket_seconds=5.0,
    )
    assert ctx.burst_timestamps == []


def test_extract_context_window_with_compulsory_vision(
    tmp_path: Path, sample_transcript: TranscriptionResult
) -> None:
    """When url is provided, contact sheets are extracted and attached to ContextWindow."""
    from unittest.mock import patch
    from streamclipper.library.frames import ContactSheet

    s1_path = tmp_path / "sheet1.png"
    s2_path = tmp_path / "sheet2.png"
    s1_path.write_bytes(b"png1")
    s2_path.write_bytes(b"png2")

    mock_sheets = [
        ContactSheet(path=s1_path, start=30.0, end=120.0, timestamps=[30.0, 40.0]),
        ContactSheet(path=s2_path, start=120.0, end=210.0, timestamps=[120.0, 130.0]),
    ]

    with patch(
        "streamclipper.library.frames.extract_moment_contact_sheets",
        return_value=mock_sheets,
    ) as mock_extract:
        ctx = extract_context_window(
            sample_transcript,
            None,
            center_time=150.0,
            url="https://www.youtube.com/watch?v=compulsory_test",
            temp_dir=tmp_path,
        )

        assert mock_extract.called
        assert len(ctx.contact_sheets) == 2
        assert ctx.contact_sheets[0] == s1_path
        assert ctx.contact_sheets[1] == s2_path
        assert len(ctx.sheet_timestamps) == 2
        assert ctx.sheet_timestamps[0] == [30.0, 40.0]

        llm_text = ctx.to_llm_text()
        assert "Visual Contact Sheets:" in llm_text
        assert "Sheet 1 (sheet1.png)" in llm_text


def test_extract_context_window_vision_failure_falls_back_gracefully(
    tmp_path: Path, sample_transcript: TranscriptionResult
) -> None:
    """When proxy video extraction fails, fall back to transcript+chat without raising."""
    from unittest.mock import patch

    with patch(
        "streamclipper.library.frames.extract_moment_contact_sheets",
        side_effect=RuntimeError("Proxy download timed out"),
    ):
        ctx = extract_context_window(
            sample_transcript,
            None,
            center_time=150.0,
            url="https://www.youtube.com/watch?v=fail_test",
            temp_dir=tmp_path,
        )

        assert ctx.contact_sheets == []
        assert ctx.sheet_timestamps == []
        # Transcript segments should still be present
        assert len(ctx.segments) > 0


def test_context_window_to_llm_text_string_paths_and_missing_timestamps(
    tmp_path: Path, sample_transcript: TranscriptionResult
) -> None:
    """to_llm_text safely handles string sheet paths and mismatched sheet_timestamps."""
    sheet_p = tmp_path / "sheet_test.png"
    sheet_p.write_bytes(b"data")

    ctx = extract_context_window(
        sample_transcript,
        None,
        center_time=100.0,
        contact_sheets=[str(sheet_p)],  # Passed as str, not Path
        sheet_timestamps=[],  # Empty timestamps
    )
    assert len(ctx.contact_sheets) == 1
    assert isinstance(ctx.contact_sheets[0], Path)

    text = ctx.to_llm_text()
    assert "Visual Contact Sheets:" in text
    assert "sheet_test.png" in text



