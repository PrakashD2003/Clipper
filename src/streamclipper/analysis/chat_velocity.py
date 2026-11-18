"""Chat velocity spike analyzer with YPP live-chat burst sample metadata."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from ..config import Config
from ..models import SignalType, TimeRange


class ChatVelocityAnalyzer:
    """Detect chat velocity spikes from scraped live chat messages."""

    MAX_SAMPLE_MESSAGES = 5

    def __init__(self, config: Config):
        self.window_seconds = float(config.get("analysis.chat.window_seconds", 10))
        self.prominence_factor = float(config.get("analysis.chat.prominence_factor", 3.0))
        self.min_messages = int(config.get("analysis.chat.min_messages", 50))
        self.merge_gap = float(
            config.get("analysis.chat.merge_gap_seconds", self.window_seconds)
        )

        if self.window_seconds <= 0:
            raise ValueError(
                f"analysis.chat.window_seconds must be > 0, got {self.window_seconds}"
            )
        if self.prominence_factor < 0:
            raise ValueError(
                f"analysis.chat.prominence_factor must be >= 0, got {self.prominence_factor}"
            )
        if self.min_messages < 0:
            raise ValueError(
                f"analysis.chat.min_messages must be >= 0, got {self.min_messages}"
            )
        if self.merge_gap < 0:
            raise ValueError(
                f"analysis.chat.merge_gap_seconds must be >= 0, got {self.merge_gap}"
            )

    def analyze(self, chat_df: pd.DataFrame | None) -> list[TimeRange]:
        """Detect chat velocity spikes and attach up to 5 representative sample messages."""
        valid_df = self._sanitize_dataframe(chat_df)
        if len(valid_df) < self.min_messages:
            return []

        offsets = valid_df["offset_seconds"].to_numpy(dtype=float)
        if offsets.size == 0:
            return []

        max_time = float(np.max(offsets))
        num_bins = int(np.floor(max_time / self.window_seconds)) + 1
        if num_bins < 2:
            return []

        bin_indices = np.floor(offsets / self.window_seconds).astype(int)
        counts = np.bincount(bin_indices, minlength=num_bins).astype(float)

        mean = float(np.mean(counts))
        std = float(np.std(counts))
        if std < 1e-6:
            return []

        threshold = mean + self.prominence_factor * std

        # Pad ends strictly below minimum so spikes in bin 0 or the final bin are detected
        pad_value = float(np.min(counts)) - 1.0
        padded_counts = np.pad(counts, (1, 1), mode="constant", constant_values=pad_value)
        peaks, _ = find_peaks(padded_counts, height=threshold)
        if len(peaks) == 0:
            return []

        peak_indices = peaks - 1
        max_excess = max(float(np.max(counts[peak_indices]) - mean), 1e-9)

        raw_spikes: list[TimeRange] = []
        for peak_idx in peak_indices:
            idx = int(peak_idx)
            peak_count = float(counts[idx])
            score = round(
                min(10.0, max(0.0, ((peak_count - mean) / max_excess) * 10.0)),
                2,
            )
            start = round(idx * self.window_seconds, 3)
            end = round((idx + 1) * self.window_seconds, 3)
            raw_spikes.append(
                TimeRange(
                    start=start,
                    end=end,
                    score=score,
                    signal_type=SignalType.CHAT_VELOCITY,
                    metadata={"peak_bin_count": int(counts[idx])},
                )
            )

        merged_spikes = self._merge_nearby(raw_spikes)
        for spike in merged_spikes:
            samples, msg_count = self._extract_sample_messages(
                valid_df, spike.start, spike.end, max_samples=self.MAX_SAMPLE_MESSAGES
            )
            spike.metadata["sample_messages"] = samples
            spike.metadata["message_count"] = msg_count

        return merged_spikes

    def _sanitize_dataframe(self, chat_df: pd.DataFrame | None) -> pd.DataFrame:
        """Drop invalid/NaN/negative offset_seconds rows defensively."""
        if chat_df is None or chat_df.empty or "offset_seconds" not in chat_df.columns:
            return pd.DataFrame(columns=["offset_seconds", "author", "message"])

        valid = chat_df.copy()
        valid["offset_seconds"] = pd.to_numeric(valid["offset_seconds"], errors="coerce")
        valid = valid.dropna(subset=["offset_seconds"])
        valid = valid[valid["offset_seconds"] >= 0.0]

        if "author" not in valid.columns:
            valid["author"] = ""
        if "message" not in valid.columns:
            valid["message"] = ""

        return valid

    def _merge_nearby(self, spikes: list[TimeRange]) -> list[TimeRange]:
        """Merge spikes within merge_gap seconds, keeping max score and peak_bin_count."""
        if not spikes:
            return []

        ordered = sorted(spikes, key=lambda s: (s.start, s.end))
        first = ordered[0]
        merged: list[TimeRange] = [
            TimeRange(
                start=first.start,
                end=first.end,
                score=first.score,
                signal_type=SignalType.CHAT_VELOCITY,
                metadata=dict(first.metadata),
            )
        ]

        for spike in ordered[1:]:
            current = merged[-1]
            if spike.start - current.end <= self.merge_gap:
                new_end = max(current.end, spike.end)
                best_score = max(current.score, spike.score)
                best_peak_count = max(
                    int(current.metadata.get("peak_bin_count", 0)),
                    int(spike.metadata.get("peak_bin_count", 0)),
                )
                merged[-1] = TimeRange(
                    start=current.start,
                    end=new_end,
                    score=best_score,
                    signal_type=SignalType.CHAT_VELOCITY,
                    metadata={"peak_bin_count": best_peak_count},
                )
            else:
                merged.append(
                    TimeRange(
                        start=spike.start,
                        end=spike.end,
                        score=spike.score,
                        signal_type=SignalType.CHAT_VELOCITY,
                        metadata=dict(spike.metadata),
                    )
                )

        return merged

    def _extract_sample_messages(
        self,
        chat_df: pd.DataFrame,
        start: float,
        end: float,
        max_samples: int = 5,
    ) -> tuple[list[dict[str, Any]], int]:
        """Extract up to max_samples representative viewer messages within [start, end]."""
        mask = (chat_df["offset_seconds"] >= start) & (chat_df["offset_seconds"] <= end)
        window_df = chat_df.loc[mask].sort_values("offset_seconds", kind="mergesort")
        total_count = int(len(window_df))
        if total_count == 0:
            return [], 0

        all_messages: list[dict[str, Any]] = []
        unique_messages: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str]] = set()

        for row in window_df.itertuples(index=False):
            author_val = getattr(row, "author", "")
            msg_val = getattr(row, "message", "")
            offset_val = float(getattr(row, "offset_seconds"))

            author = "" if pd.isna(author_val) else str(author_val)
            message = "" if pd.isna(msg_val) else str(msg_val)
            if not message.strip():
                continue

            entry = {
                "author": author,
                "message": message,
                "offset_seconds": round(offset_val, 3),
            }
            all_messages.append(entry)

            key = (author, message)
            if key not in seen_keys:
                seen_keys.add(key)
                unique_messages.append(entry)

        pool = unique_messages if len(unique_messages) >= max_samples else all_messages
        if len(pool) <= max_samples:
            return pool, total_count

        sample_indices = np.linspace(0, len(pool) - 1, max_samples, dtype=int)
        return [pool[int(i)] for i in sample_indices], total_count
