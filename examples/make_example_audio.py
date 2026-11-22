#!/usr/bin/env python
"""Generate SYNTHETIC example recordings for exercising the pipeline plumbing.

These are pure tones, not speech: they let you try `prepare-corpus` / `validate-corpus`
without any private data. A forced aligner run on them yields meaningless alignments.
Replace them with real recordings (and a manifest you control) for actual work.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

OUT = Path(__file__).resolve().parent / "audio"


def tone(path: Path, seconds: float, sr: int, channels: int, freq: float) -> None:
    t = np.arange(int(seconds * sr)) / sr
    mono = 0.3 * np.sin(2 * np.pi * freq * t)
    data = mono if channels == 1 else np.stack([mono, 0.5 * mono], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr, subtype="PCM_16")


if __name__ == "__main__":
    tone(OUT / "ex_001.wav", 2.5, 16000, 1, 220.0)   # already 16 kHz mono
    tone(OUT / "ex_002.wav", 2.0, 44100, 2, 330.0)   # stereo 44.1 kHz -> exercises down-mix + resampling
    tone(OUT / "ex_003.wav", 2.0, 22050, 1, 440.0)   # no speaker id in the manifest -> unique pseudo speaker
    print(f"wrote 3 synthetic tone files to {OUT}")
