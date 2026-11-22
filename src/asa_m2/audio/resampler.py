"""Down-mixing, resampling and PCM WAV export.

* Resampling uses ``scipy.signal.resample_poly`` (polyphase FIR with a Kaiser window,
  anti-aliasing included). Sample-rate *metadata* is never changed without resampling.
* No amplitude normalisation is applied; samples that exceed full scale after
  resampling are clipped *and counted* in the transformation record.
* Every step is returned as a transformation record so the corpus manifest can document
  exactly what happened to each recording.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from asa_m2.audio.loader import AudioData
from asa_m2.config import AudioConfig
from asa_m2.exceptions import AudioError
from asa_m2.utils.serialization import check_writable

RESAMPLE_METHOD = "scipy.signal.resample_poly (polyphase FIR, kaiser window beta=5.0)"


def to_mono(samples: np.ndarray, policy: str) -> np.ndarray:
    """Collapse ``(frames, channels)`` to ``(frames,)`` according to ``policy``."""
    if samples.ndim != 2:
        raise AudioError(f"expected a (frames, channels) array, got shape {samples.shape}")
    channels = samples.shape[1]
    if channels == 1:
        return samples[:, 0].copy()
    if policy == "mean":
        return samples.mean(axis=1)
    if policy == "left":
        return samples[:, 0].copy()
    if policy == "right":
        return samples[:, 1].copy()
    if policy == "reject":
        raise AudioError(f"audio has {channels} channels and stereo_to_mono policy is 'reject'")
    raise AudioError(f"unknown stereo_to_mono policy {policy!r}")


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample a 1-D signal from ``src_rate`` to ``dst_rate`` (returns a copy if equal)."""
    if samples.ndim != 1:
        raise AudioError("resample() expects a 1-D signal; down-mix first")
    if src_rate <= 0 or dst_rate <= 0:
        raise AudioError(f"invalid sample rates: {src_rate} -> {dst_rate}")
    if samples.size == 0:
        raise AudioError("cannot resample an empty signal")
    if src_rate == dst_rate:
        return samples.copy()
    g = math.gcd(src_rate, dst_rate)
    return resample_poly(samples, dst_rate // g, src_rate // g)


@dataclass
class PreparedAudio:
    samples: np.ndarray  # 1-D float64
    sample_rate: int
    transformations: list[dict[str, Any]] = field(default_factory=list)


def prepare_audio(data: AudioData, config: AudioConfig) -> PreparedAudio:
    """Apply the documented M2 audio policy: down-mix -> resample -> (clip guard)."""
    ops: list[dict[str, Any]] = []
    channels = data.info.channels
    mono = to_mono(data.samples, config.stereo_to_mono)
    ops.append({"op": "to_mono", "policy": config.stereo_to_mono if channels > 1 else "already_mono",
                "channels_in": channels})
    src, dst = data.info.sample_rate, config.target_sample_rate
    if src != dst:
        out = resample(mono, src, dst)
        ops.append({"op": "resample", "method": RESAMPLE_METHOD, "from_hz": src, "to_hz": dst,
                    "samples_in": int(mono.shape[0]), "samples_out": int(out.shape[0])})
    else:
        out = mono
        ops.append({"op": "resample", "method": "skipped_same_rate", "from_hz": src, "to_hz": dst,
                    "samples_in": int(mono.shape[0]), "samples_out": int(mono.shape[0])})
    over = int(np.sum(np.abs(out) > 1.0))
    if over:
        out = np.clip(out, -1.0, 1.0)
        ops.append({"op": "clip", "reason": "samples exceeded full scale after processing", "count": over})
    return PreparedAudio(out, dst, ops)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int, subtype: str, *, overwrite: bool = False) -> None:
    """Write a mono PCM WAV. Refuses to overwrite unless explicitly allowed."""
    check_writable(path, overwrite=overwrite)
    if samples.ndim != 1:
        raise AudioError("write_wav expects a 1-D (mono) signal")
    if float(np.max(np.abs(samples))) > 1.0:
        raise AudioError("samples exceed full scale; clip explicitly before writing")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sample_rate, subtype=subtype, format="WAV")
