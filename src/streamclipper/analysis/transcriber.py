"""Hardware-aware WhisperX transcriber with two-stage VRAM cleanup and faster-whisper fallback."""
from __future__ import annotations

import gc
import inspect
import json
import logging
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - CPU-only environment without PyTorch
    torch = None  # type: ignore[assignment]

from ..config import Config
from ..models import TranscriptionResult, TranscriptSegment, WordTimestamp

logger = logging.getLogger(__name__)


def _clear_cuda_cache() -> None:
    """Run Python GC and release unused PyTorch CUDA VRAM when available."""
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


class Transcriber:
    """Transcribes stream audio with word-level timestamps using hardware-resolved settings."""

    def __init__(self, config: Config):
        """Initialize with resolved hardware-aware settings from Config."""
        self.config = config
        self.engine: str = str(config.get("transcription.engine", "whisperx"))
        self.device: str = str(config.get("transcription.device", "cpu"))
        self.model_size: str = str(config.get("transcription.model_size", "base"))
        self.compute_type: str = str(config.get("transcription.compute_type", "int8"))
        self.batch_size: int = int(config.get("transcription.batch_size", 4))
        self.language: str = str(config.get("transcription.language", "auto"))

        # Safety guard: float16 / int8_float16 are unsupported on CPU backends
        if self.device == "cpu" and self.compute_type in ("float16", "int8_float16"):
            logger.warning(
                "compute_type=%s is unsupported on CPU; downgrading to 'int8'.",
                self.compute_type,
            )
            self.compute_type = "int8"

        hw = getattr(config, "hardware_profile", None)
        logger.info(
            "Transcriber configured: tier=%s gpu=%s engine=%s device=%s model=%s compute_type=%s batch_size=%d",
            hw.tier if hw else "custom",
            hw.gpu_name if hw else "unknown",
            self.engine,
            self.device,
            self.model_size,
            self.compute_type,
            self.batch_size,
        )

    def transcribe(self, audio_path: Path | str) -> TranscriptionResult:
        """Transcribe audio file and return word-level results."""
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        if self.engine == "faster-whisper":
            return self._transcribe_faster_whisper(path)

        if self.engine == "whisperx":
            try:
                return self._transcribe_whisperx(path)
            except ImportError:
                logger.warning(
                    "whisperx is not installed; falling back to faster-whisper."
                )
                return self._transcribe_faster_whisper(path)

        raise ValueError(f"Unsupported transcription engine: {self.engine!r}")

    def _transcribe_whisperx(self, audio_path: Path) -> TranscriptionResult:
        """WhisperX path with alignment and explicit two-stage VRAM cleanup."""
        import whisperx

        model: Any = None
        audio: Any = None
        model_a: Any = None
        asr_cleaned = False

        try:
            model = whisperx.load_model(
                self.model_size,
                self.device,
                compute_type=self.compute_type,
            )
            audio = whisperx.load_audio(str(audio_path))
            transcribe_kwargs: dict[str, Any] = {"batch_size": self.batch_size}
            if self.language and self.language.lower() != "auto":
                try:
                    sig = inspect.signature(model.transcribe)
                    if "language" in sig.parameters or any(
                        p.kind == inspect.Parameter.VAR_KEYWORD
                        for p in sig.parameters.values()
                    ):
                        transcribe_kwargs["language"] = self.language
                except (TypeError, ValueError):
                    transcribe_kwargs["language"] = self.language
            result = model.transcribe(audio, **transcribe_kwargs)
            raw_detected = result.get("language")
            if raw_detected:
                detected_lang = str(raw_detected)
            elif self.language and self.language.lower() != "auto":
                detected_lang = self.language
            else:
                detected_lang = "en"

            # Stage 1: Free ASR model before loading wav2vec2 alignment model (critical on 6 GB GPUs)
            del model
            model = None
            _clear_cuda_cache()
            asr_cleaned = True

            # Stage 2: Align ASR segments for word-level timestamps
            try:
                model_a, metadata = whisperx.load_align_model(
                    language_code=detected_lang,
                    device=self.device,
                )
                aligned = whisperx.align(
                    result.get("segments", []),
                    model_a,
                    metadata,
                    audio,
                    self.device,
                    return_char_alignments=False,
                )
            except Exception as exc:
                logger.warning(
                    "WhisperX alignment failed for language %r (%s); keeping ASR output.",
                    detected_lang,
                    exc,
                )
                model_a = None
                audio = None
                _clear_cuda_cache()
                try:
                    import faster_whisper  # noqa: F401
                except ImportError:
                    return self._convert_aligned_segments(
                        result.get("segments", []),
                        detected_lang,
                    )
                return self._transcribe_faster_whisper(audio_path)

            # Stage 3: Free alignment model and audio buffer so Ollama has full VRAM in Phase 3
            del model_a, audio
            model_a = None
            audio = None
            _clear_cuda_cache()

            return self._convert_aligned_segments(
                aligned.get("segments", []),
                detected_lang,
            )
        except Exception:
            # Release any remaining GPU allocations on unexpected failure
            if not asr_cleaned and model is not None:
                del model
            if model_a is not None:
                del model_a
            if audio is not None:
                del audio
            _clear_cuda_cache()
            raise

    def _convert_aligned_segments(
        self,
        segments: list[dict[str, Any]],
        language: str,
    ) -> TranscriptionResult:
        """Convert WhisperX aligned segment dictionaries into TranscriptionResult."""
        transcript_segments: list[TranscriptSegment] = []

        for seg in segments:
            seg_text = str(seg.get("text", "")).strip()
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", seg_start))
            if seg_end < seg_start:
                seg_end = seg_start

            words: list[WordTimestamp] = []
            prev_end = seg_start
            for raw_word in seg.get("words") or []:
                word_text = str(raw_word.get("word", "")).strip()
                if not word_text:
                    continue

                raw_start = raw_word.get("start")
                raw_end = raw_word.get("end")
                raw_score = raw_word.get("score")

                w_start = float(raw_start) if raw_start is not None else prev_end
                w_end = float(raw_end) if raw_end is not None else max(w_start, prev_end)
                if w_end < w_start:
                    w_end = w_start
                confidence = float(raw_score) if raw_score is not None else 0.0

                words.append(
                    WordTimestamp(
                        text=word_text,
                        start=w_start,
                        end=w_end,
                        confidence=confidence,
                    )
                )
                prev_end = w_end

            transcript_segments.append(
                TranscriptSegment(
                    text=seg_text,
                    start=seg_start,
                    end=seg_end,
                    words=words,
                )
            )

        full_text = " ".join(s.text for s in transcript_segments if s.text)
        return TranscriptionResult(
            segments=transcript_segments,
            full_text=full_text,
            language=language,
        )

    def _transcribe_faster_whisper(self, audio_path: Path) -> TranscriptionResult:
        """Fallback using faster-whisper with word_timestamps=True and VRAM cleanup."""
        from faster_whisper import WhisperModel

        model: Any = None
        try:
            model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            fw_language = (
                None
                if not self.language or self.language.lower() == "auto"
                else self.language
            )
            segments_iter, info = model.transcribe(
                str(audio_path),
                language=fw_language,
                word_timestamps=True,
                vad_filter=True,
            )
            info_lang = getattr(info, "language", None)
            if info_lang:
                detected_lang = str(info_lang)
            elif self.language and self.language.lower() != "auto":
                detected_lang = self.language
            else:
                detected_lang = "en"

            transcript_segments: list[TranscriptSegment] = []
            for seg in segments_iter:
                seg_text = str(getattr(seg, "text", "")).strip()
                seg_start = float(getattr(seg, "start", 0.0))
                seg_end = float(getattr(seg, "end", seg_start))
                if seg_end < seg_start:
                    seg_end = seg_start

                words: list[WordTimestamp] = []
                prev_end = seg_start
                for raw_word in getattr(seg, "words", None) or []:
                    word_text = str(getattr(raw_word, "word", "")).strip()
                    if not word_text:
                        continue

                    raw_start = getattr(raw_word, "start", None)
                    raw_end = getattr(raw_word, "end", None)
                    raw_prob = getattr(raw_word, "probability", None)

                    w_start = float(raw_start) if raw_start is not None else prev_end
                    w_end = float(raw_end) if raw_end is not None else max(w_start, prev_end)
                    if w_end < w_start:
                        w_end = w_start
                    confidence = float(raw_prob) if raw_prob is not None else 0.0

                    words.append(
                        WordTimestamp(
                            text=word_text,
                            start=w_start,
                            end=w_end,
                            confidence=confidence,
                        )
                    )
                    prev_end = w_end

                transcript_segments.append(
                    TranscriptSegment(
                        text=seg_text,
                        start=seg_start,
                        end=seg_end,
                        words=words,
                    )
                )

            full_text = " ".join(s.text for s in transcript_segments if s.text)
            return TranscriptionResult(
                segments=transcript_segments,
                full_text=full_text,
                language=detected_lang,
            )
        finally:
            if model is not None:
                del model
            _clear_cuda_cache()

    def save_transcript(self, result: TranscriptionResult, output_path: Path | str) -> None:
        """Persist TranscriptionResult to JSON."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result.to_json(), encoding="utf-8")

    def load_transcript(self, path: Path | str) -> TranscriptionResult | None:
        """Load cached TranscriptionResult from JSON."""
        in_path = Path(path)
        if not in_path.exists():
            return None
        try:
            data = json.loads(in_path.read_text(encoding="utf-8"))
            return TranscriptionResult.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Failed to load transcript cache %s: %s", in_path, exc)
            return None
