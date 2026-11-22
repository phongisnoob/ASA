"""Audio validation. Reports *all* problems; never alters the audio."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from asa_m2.audio.loader import AudioInfo, load_audio, probe_audio
from asa_m2.config import AudioConfig
from asa_m2.exceptions import (
    AudioNotFoundError,
    AudioReadError,
    AudioValidationError,
)


@dataclass(frozen=True)
class AudioIssue:
    code: str
    message: str
    severity: str  # "error" | "warning"


@dataclass
class AudioReport:
    path: Path
    info: AudioInfo | None = None
    issues: list[AudioIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[AudioIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[AudioIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise AudioValidationError(
                f"{self.path.name}: " + "; ".join(f"[{i.code}] {i.message}" for i in self.errors),
                details={"codes": [i.code for i in self.errors], "path": str(self.path)},
            )


def validate_audio(path: str | Path, config: AudioConfig, *, check_samples: bool = True) -> AudioReport:
    """Validate existence, readability, format, duration, sample rate and signal sanity."""
    p = Path(path)
    report = AudioReport(path=p)

    def add(code: str, message: str, severity: str = "error") -> None:
        report.issues.append(AudioIssue(code, message, severity))

    if p.suffix.lower() not in config.supported_extensions:
        add("unsupported_extension", f"extension {p.suffix!r} not in {config.supported_extensions}")
    try:
        info = probe_audio(p)
    except AudioNotFoundError as exc:
        add("missing_file", exc.message)
        return report
    except AudioReadError as exc:
        add("unreadable", exc.message)
        return report
    report.info = info
    if info.frames == 0:
        add("empty_audio", "file contains zero audio frames")
        return report
    if info.sample_rate < config.min_source_sample_rate:
        add("sample_rate_too_low", f"{info.sample_rate} Hz < minimum {config.min_source_sample_rate} Hz")
    if info.duration_s < config.min_duration_s:
        add("duration_too_short", f"{info.duration_s:.3f}s < minimum {config.min_duration_s}s")
    if info.duration_s > config.max_duration_s:
        add("duration_too_long", f"{info.duration_s:.1f}s > maximum {config.max_duration_s}s")
    if info.channels > 1 and config.stereo_to_mono == "reject":
        add("not_mono", f"{info.channels} channels and audio.stereo_to_mono is 'reject'")
    if info.sample_rate != config.target_sample_rate:
        add("sample_rate_differs", f"{info.sample_rate} Hz will be resampled to {config.target_sample_rate} Hz", "warning")
    if check_samples and report.ok:
        try:
            data = load_audio(p)
        except AudioReadError as exc:
            add("unreadable", exc.message)
            return report
        x = data.samples
        if not np.all(np.isfinite(x)):
            add("non_finite_samples", "audio contains NaN or Inf samples")
        elif not np.any(x):
            add("digital_silence", "all samples are exactly zero; nothing can be aligned")
        else:
            clipped = float(np.mean(np.abs(x) >= 0.999))
            if clipped > config.max_clipping_fraction:
                add("clipping", f"{clipped:.2%} of samples at full scale (limit {config.max_clipping_fraction:.2%})", "warning")
    return report
