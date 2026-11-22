"""Semantic validation of parsed TextGrids (intervals, tiers, time consistency).

Checks never modify or drop intervals; they only *report*. The thresholds that exist
(gap tolerance, duration mismatch tolerance) come from :class:`~asa_m2.config.QCConfig`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from asa_m2.alignment.textgrid_parser import ParsedTextGrid, ParsedTier, TextGridIssue
from asa_m2.config import QCConfig


def interval_role(label: str, silence_labels: Iterable[str], oov_labels: Iterable[str], kind: str) -> str:
    """Classify an interval label as ``silence``, ``oov_marker`` or ``kind`` ('word'/'phone').

    Empty or whitespace-only labels are silence *explicitly* (they are never skipped
    unseen): they appear in the output with ``label_role='silence'``.
    """
    stripped = label.strip()
    if stripped == "" or stripped in set(silence_labels):
        return "silence"
    if stripped in set(oov_labels):
        return "oov_marker"
    return kind


def validate_tier(tier: ParsedTier, qc: QCConfig) -> list[TextGridIssue]:
    issues: list[TextGridIssue] = []

    def add(code: str, severity: str, message: str, idx: int | None = None, **details: object) -> None:
        issues.append(TextGridIssue(code, severity, message, tier=tier.name, interval_index=idx, details=dict(details)))

    if not tier.is_interval_tier:
        return issues
    if not all(math.isfinite(v) for v in (tier.xmin, tier.xmax)) or not tier.xmax > tier.xmin:
        add("invalid_tier_range", "error", f"tier range [{tier.xmin}, {tier.xmax}] is invalid")
    if not tier.intervals:
        add("empty_tier", "warning", "tier contains no intervals")
        return issues
    prev = None
    for iv in tier.intervals:
        if not (math.isfinite(iv.start) and math.isfinite(iv.end)):
            add("non_finite_time", "error", "interval has a non-finite boundary", iv.index, start=iv.start, end=iv.end)
            prev = None
            continue
        if iv.start < 0:
            add("negative_start", "error", f"interval starts before 0 ({iv.start})", iv.index, start=iv.start)
        if iv.end < iv.start:
            add("negative_duration", "error", f"end ({iv.end}) is before start ({iv.start})", iv.index,
                start=iv.start, end=iv.end, label=iv.label)
        elif iv.end == iv.start:
            add("zero_duration", "error", f"zero-length interval at {iv.start}", iv.index, start=iv.start, label=iv.label)
        if prev is not None:
            if iv.start < prev.start - qc.overlap_tolerance_s:
                add("unsorted_intervals", "error", "intervals are not in chronological order", iv.index,
                    previous_start=prev.start, start=iv.start)
            if iv.start < prev.end - qc.overlap_tolerance_s:
                add("overlap", "error",
                    f"interval overlaps the previous one by {prev.end - iv.start:.6f}s", iv.index,
                    previous_index=prev.index, previous_end=prev.end, start=iv.start)
            elif iv.start - prev.end > qc.gap_tolerance_s:
                add("gap", "warning", f"{iv.start - prev.end:.6f}s gap before this interval", iv.index,
                    previous_end=prev.end, start=iv.start)
        prev = iv
    first, last = tier.intervals[0], tier.intervals[-1]
    if math.isfinite(first.start) and first.start - tier.xmin > qc.gap_tolerance_s:
        add("starts_after_tier_xmin", "warning", f"first interval starts {first.start - tier.xmin:.6f}s after tier xmin", 0)
    if math.isfinite(last.end):
        if last.end > tier.xmax + qc.gap_tolerance_s:
            add("exceeds_tier_xmax", "error", f"last interval ends at {last.end}, beyond tier xmax {tier.xmax}", last.index)
        elif tier.xmax - last.end > qc.gap_tolerance_s:
            add("ends_before_tier_xmax", "warning", f"last interval ends {tier.xmax - last.end:.6f}s before tier xmax", last.index)
    return issues


def validate_textgrid(
    parsed: ParsedTextGrid, qc: QCConfig, *, audio_duration_s: float | None = None
) -> list[TextGridIssue]:
    """Validate every interval tier plus global time consistency."""
    issues: list[TextGridIssue] = list(parsed.issues)
    for tier in parsed.tiers:
        issues.extend(validate_tier(tier, qc))
        if tier.is_interval_tier and abs(tier.xmax - parsed.xmax) > qc.gap_tolerance_s:
            issues.append(TextGridIssue("tier_range_mismatch", "warning",
                                        f"tier xmax {tier.xmax} differs from file xmax {parsed.xmax}", tier=tier.name))
    if audio_duration_s is not None and audio_duration_s > 0:
        diff = abs(parsed.xmax - audio_duration_s)
        if diff > qc.duration_mismatch_tolerance_s:
            hint = ""
            ratio = parsed.xmax / audio_duration_s
            if 900 < ratio < 1100:
                hint = " (ratio ~1000: times may be in milliseconds rather than seconds)"
            elif 0.0009 < ratio < 0.0011:
                hint = " (ratio ~0.001: times may be in kiloseconds/unit error)"
            issues.append(TextGridIssue(
                "duration_mismatch", "warning",
                f"TextGrid duration {parsed.xmax:.3f}s differs from audio duration {audio_duration_s:.3f}s{hint}",
                details={"textgrid_xmax": parsed.xmax, "audio_duration_s": audio_duration_s},
            ))
    return issues


def has_errors(issues: Iterable[TextGridIssue]) -> bool:
    return any(i.severity == "error" for i in issues)
