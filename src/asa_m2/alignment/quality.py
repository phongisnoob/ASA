"""Alignment quality control (M2.7).

Everything here produces *warnings and counts*; no interval is ever deleted because it
looks suspicious. Duration thresholds come from configuration and are **alignment
sanity checks, not pronunciation-error thresholds** - a short phone interval says
something about the aligner, not about the learner.

Gold evaluation (:func:`evaluate_boundaries`) is provided for when manual reference
alignments exist. Nothing in this package claims that alignment quality on HANU or
L2-ARCTIC has been validated; that requires running this on their manual annotations.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Literal

from asa_m2.alignment.textgrid_parser import ParsedInterval, ParsedTier
from asa_m2.config import QCConfig
from asa_m2.schemas import PhoneAlignment, QualityFlag, WordAlignment

Relation = Literal["equal", "substituted", "missing", "extra"]


# ---------------------------------------------------------------------------------
# sequence matching
# ---------------------------------------------------------------------------------
def match_sequences(expected: Sequence[str], aligned: Sequence[str]) -> list[tuple[int | None, int | None, Relation]]:
    """Order-preserving match of expected vs aligned labels.

    Returns ``(expected_index, aligned_index, relation)`` triples. Using a diff (rather
    than positional pairing) means one missing word does not shift every later word.
    """
    matcher = SequenceMatcher(a=list(expected), b=list(aligned), autojunk=False)
    out: list[tuple[int | None, int | None, Relation]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            out += [(i1 + k, j1 + k, "equal") for k in range(i2 - i1)]
        elif tag == "replace":
            n = min(i2 - i1, j2 - j1)
            out += [(i1 + k, j1 + k, "substituted") for k in range(n)]
            out += [(i1 + k, None, "missing") for k in range(n, i2 - i1)]
            out += [(None, j1 + k, "extra") for k in range(n, j2 - j1)]
        elif tag == "delete":
            out += [(i1 + k, None, "missing") for k in range(i2 - i1)]
        else:  # insert
            out += [(None, j1 + k, "extra") for k in range(j2 - j1)]
    return out


@dataclass(frozen=True)
class PhoneDiff:
    missing: int
    extra: int
    substituted: int

    @property
    def identical(self) -> bool:
        return not (self.missing or self.extra or self.substituted)


def diff_phones(expected: Sequence[str], aligned: Sequence[str]) -> PhoneDiff:
    """Count missing / extra / substituted phone labels between two base-phone sequences."""
    missing = extra = substituted = 0
    for tag, i1, i2, j1, j2 in SequenceMatcher(a=list(expected), b=list(aligned), autojunk=False).get_opcodes():
        if tag == "delete":
            missing += i2 - i1
        elif tag == "insert":
            extra += j2 - j1
        elif tag == "replace":
            n = min(i2 - i1, j2 - j1)
            substituted += n
            missing += (i2 - i1) - n
            extra += (j2 - j1) - n
    return PhoneDiff(missing, extra, substituted)


# ---------------------------------------------------------------------------------
# duration checks
# ---------------------------------------------------------------------------------
def check_durations(
    words: Iterable[WordAlignment], phones: Iterable[PhoneAlignment], qc: QCConfig
) -> list[QualityFlag]:
    """Warn about unusually short/long intervals (configurable; never pronunciation verdicts)."""
    flags: list[QualityFlag] = []
    note = "alignment sanity warning, not a pronunciation-error threshold"
    for p in phones:
        if p.label_role != "phone":
            continue
        d = p.end_time - p.start_time
        if d < qc.min_phone_duration_s:
            flags.append(QualityFlag(
                code="phone_too_short", severity="warning", scope="phone", phone_index=p.phone_index,
                word_index=p.word_index, start_time=p.start_time, end_time=p.end_time,
                message=f"phone {p.aligned_label!r} lasts {d * 1000:.1f} ms (< {qc.min_phone_duration_s * 1000:.0f} ms); {note}",
                details={"duration_s": d, "threshold_s": qc.min_phone_duration_s}))
        elif d > qc.max_phone_duration_s:
            flags.append(QualityFlag(
                code="phone_too_long", severity="warning", scope="phone", phone_index=p.phone_index,
                word_index=p.word_index, start_time=p.start_time, end_time=p.end_time,
                message=f"phone {p.aligned_label!r} lasts {d:.3f} s (> {qc.max_phone_duration_s} s); {note}",
                details={"duration_s": d, "threshold_s": qc.max_phone_duration_s}))
    for w in words:
        if w.label_role not in {"word", "oov_marker"}:
            continue
        d = w.end_time - w.start_time
        if d < qc.min_word_duration_s:
            flags.append(QualityFlag(
                code="word_too_short", severity="warning", scope="word", word_index=w.word_index,
                start_time=w.start_time, end_time=w.end_time,
                message=f"word {w.label!r} lasts {d * 1000:.1f} ms (< {qc.min_word_duration_s * 1000:.0f} ms); {note}",
                details={"duration_s": d, "threshold_s": qc.min_word_duration_s}))
        elif d > qc.max_word_duration_s:
            flags.append(QualityFlag(
                code="word_too_long", severity="warning", scope="word", word_index=w.word_index,
                start_time=w.start_time, end_time=w.end_time,
                message=f"word {w.label!r} lasts {d:.3f} s (> {qc.max_word_duration_s} s); {note}",
                details={"duration_s": d, "threshold_s": qc.max_word_duration_s}))
    return flags


# ---------------------------------------------------------------------------------
# gold boundary evaluation
# ---------------------------------------------------------------------------------
_STRESS_RE = re.compile(r"[0-9]+$")


@dataclass
class FileBoundaryResult:
    name: str
    failed: bool
    reason: str | None = None
    errors_s: list[float] = field(default_factory=list)  # absolute boundary errors


@dataclass(frozen=True)
class BoundaryEvaluation:
    n_files: int
    n_failed: int
    failure_rate: float | None
    n_boundaries: int
    mae_s: float | None
    median_abs_error_s: float | None
    pct_within_tolerance: float | None
    tolerance_s: float


def _speech(tier: ParsedTier, silence: set[str], strip_stress: bool) -> list[tuple[ParsedInterval, str]]:
    out = []
    for iv in tier.intervals:
        label = iv.label.strip()
        if label == "" or label in silence:
            continue
        key = _STRESS_RE.sub("", label) if strip_stress else label
        out.append((iv, key.lower()))
    return out


def compare_tiers(
    name: str,
    predicted: ParsedTier,
    gold: ParsedTier,
    *,
    silence_labels: Iterable[str] = ("sil", "sp", "<eps>"),
    strip_stress: bool = True,
) -> FileBoundaryResult:
    """Absolute boundary errors between a predicted and a gold tier.

    Non-silence labels must agree as sequences (after optional stress stripping and
    lower-casing); otherwise the file counts as an alignment failure and no boundary
    errors are recorded - boundaries of different units cannot be compared meaningfully.
    """
    sil = set(silence_labels)
    p, g = _speech(predicted, sil, strip_stress), _speech(gold, sil, strip_stress)
    if [k for _, k in p] != [k for _, k in g]:
        return FileBoundaryResult(name, True, f"label sequences differ ({len(p)} predicted vs {len(g)} gold intervals)")
    boundaries: dict[float, float] = {}
    for (piv, _), (giv, _) in zip(p, g, strict=True):  # label sequences were verified equal
        boundaries.setdefault(round(giv.start, 6), abs(piv.start - giv.start))
        boundaries.setdefault(round(giv.end, 6), abs(piv.end - giv.end))
    return FileBoundaryResult(name, False, None, list(boundaries.values()))


def evaluate_boundaries(results: Sequence[FileBoundaryResult], tolerance_s: float = 0.02) -> BoundaryEvaluation:
    """Pool per-file results: boundary MAE, median absolute error, % within tolerance, failure rate."""
    errors = [e for r in results if not r.failed for e in r.errors_s]
    failed = sum(1 for r in results if r.failed)
    return BoundaryEvaluation(
        n_files=len(results),
        n_failed=failed,
        failure_rate=(failed / len(results)) if results else None,
        n_boundaries=len(errors),
        mae_s=statistics.fmean(errors) if errors else None,
        median_abs_error_s=statistics.median(errors) if errors else None,
        pct_within_tolerance=(100.0 * sum(e <= tolerance_s for e in errors) / len(errors)) if errors else None,
        tolerance_s=tolerance_s,
    )
