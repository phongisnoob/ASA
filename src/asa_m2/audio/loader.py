"""Audio reading (via libsndfile / ``soundfile``) with explicit, typed errors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from asa_m2.exceptions import AudioNotFoundError, AudioReadError


@dataclass(frozen=True)
class AudioInfo:
    path: Path
    sample_rate: int
    channels: int
    frames: int
    format: str
    subtype: str

    @property
    def duration_s(self) -> float:
        return self.frames / self.sample_rate if self.sample_rate else 0.0


@dataclass(frozen=True)
class AudioData:
    info: AudioInfo
    samples: np.ndarray  # shape (frames, channels), float64 in [-1, 1]


def _check_file(path: Path) -> None:
    if not path.exists():
        raise AudioNotFoundError(f"audio file not found: {path}", details={"path": str(path)})
    if not path.is_file():
        raise AudioReadError(f"audio path is not a file: {path}", details={"path": str(path)})


_UNKNOWN_SIZES = (0, 0xFFFFFFFF)  # streamed WAVs may legitimately declare an unknown size


def _check_wav_integrity(path: Path) -> None:
    """Detect truncated RIFF/WAVE files.

    libsndfile silently accepts a WAV whose data chunk is shorter than its header
    declares and simply reports fewer frames, so a half-written recording would look
    valid. We compare the declared RIFF and ``data`` chunk sizes with the real file size.
    """
    size = path.stat().st_size
    with open(path, "rb") as handle:
        head = handle.read(12)
        if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            return  # not a RIFF/WAVE container (RF64, FLAC, ...): nothing to check here
        declared = int.from_bytes(head[4:8], "little")
        if declared not in _UNKNOWN_SIZES and declared + 8 > size:
            raise AudioReadError(
                f"{path.name} is truncated: the RIFF header declares {declared + 8} bytes but the file has {size}",
                details={"path": str(path), "declared": declared + 8, "actual": size},
            )
        while True:
            chunk = handle.read(8)
            if len(chunk) < 8:
                return
            cid, csize = chunk[:4], int.from_bytes(chunk[4:], "little")
            if cid == b"data":
                if csize not in _UNKNOWN_SIZES and handle.tell() + csize > size:
                    raise AudioReadError(
                        f"{path.name} is truncated: its data chunk declares {csize} bytes but only "
                        f"{size - handle.tell()} are present",
                        details={"path": str(path)},
                    )
                return
            handle.seek(csize + (csize & 1), 1)


def probe_audio(path: str | Path) -> AudioInfo:
    """Read header information only. Raises on missing, truncated or unreadable files."""
    p = Path(path)
    _check_file(p)
    _check_wav_integrity(p)
    try:
        info = sf.info(str(p))
    except (sf.LibsndfileError, RuntimeError, OSError) as exc:
        raise AudioReadError(f"cannot read audio header of {p.name}: {exc}", details={"path": str(p)}) from exc
    return AudioInfo(p, int(info.samplerate), int(info.channels), int(info.frames), info.format, info.subtype)


def load_audio(path: str | Path) -> AudioData:
    """Decode the whole file as float64 ``(frames, channels)``."""
    info = probe_audio(path)
    try:
        samples, sr = sf.read(str(info.path), dtype="float64", always_2d=True)
    except (sf.LibsndfileError, RuntimeError, OSError) as exc:
        raise AudioReadError(f"cannot decode audio {info.path.name}: {exc}", details={"path": str(info.path)}) from exc
    if int(sr) != info.sample_rate:  # pragma: no cover - defensive
        raise AudioReadError(f"sample rate changed while reading {info.path.name}")
    if samples.shape[0] != info.frames:
        raise AudioReadError(
            f"{info.path.name}: header promises {info.frames} frames but {samples.shape[0]} were decoded "
            "(truncated or corrupt file)", details={"path": str(info.path)},
        )
    return AudioData(info, samples)
