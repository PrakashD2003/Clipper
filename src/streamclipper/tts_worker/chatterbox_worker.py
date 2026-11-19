"""Standalone Chatterbox TTS worker (runs in the separate TTS venv).

Imports only ``chatterbox``, torch/torchaudio and the stdlib so the main environment's
dependencies are never required. Usage:
    python -m streamclipper.tts_worker.chatterbox_worker --text "..." --out out.wav [--ref ref.wav]
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chatterbox TTS worker")
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ref", default="")
    args = parser.parse_args(argv)

    import torch
    import torchaudio as ta
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxTurboTTS.from_pretrained(device=device)
    wav = model.generate(args.text, audio_prompt_path=args.ref or None)
    ta.save(args.out, wav, model.sr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
