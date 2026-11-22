"""Assemble the canonical :class:`~asa_m2.schemas.SampleRecord` from G2P + alignment.

Rules enforced here
-------------------
* Timestamps come only from the aligner output; invalid intervals are *excluded and
  reported*, never repaired or invented.
* Aligned phone labels are **targets placed by the aligner**; ``realized_phone`` stays
  ``None``. Resolving a word's variant from the aligner's phones is recorded as
  ``selection_method="alignment_match"`` - the aligner's decision, not evidence of what
  the learner said.
* Words are matched to expected tokens with a sequence diff, so a missing word is
  reported as missing instead of shifting every later word.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from asa_m2.alignment.quality import check_durations, diff_phones, match_sequences
from asa_m2.alignment.textgrid_parser import ParsedTextGrid, TextGridIssue, select_tier
from asa_m2.alignment.validator import interval_role, validate_textgrid
from asa_m2.config import AppConfig
from asa_m2.exceptions import AlignmentError
from asa_m2.g2p.variants import build_canonical_phones
from asa_m2.phonemes.inventory import PhoneInventory
from asa_m2.phonemes.mapping import AlignerLabelMapper, LabelRole
from asa_m2.provenance import build_provenance
from asa_m2.schemas import (
    AlignmentMetadata,
    G2PResult,
    PhoneAlignment,
    QualityFlag,
    SampleRecord,
    TokenRecord,
    WordAlignment,
)


@dataclass(frozen=True)
class IntervalIn:
    """An aligner-produced interval handed to :func:`assemble_sample_record`."""

    index: int
    start: float
    end: float
    label: str


def _is_valid(iv: IntervalIn) -> bool:
    return math.isfinite(iv.start) and math.isfinite(iv.end) and iv.start >= 0 and iv.end > iv.start


def _stress_agrees(variant_stress: Sequence[int | None], aligned_stress: Sequence[int | None]) -> bool:
    """Compare stress on *vowels only* (consonants never carry stress).

    If the aligner's vowel labels carry no stress digits (stress-less acoustic model) the
    comparison is skipped: the labels simply hold no stress information.
    """
    pos = [k for k, st in enumerate(variant_stress) if st is not None]
    aligned_vowels = [aligned_stress[k] for k in pos]
    if not aligned_vowels or any(a is None for a in aligned_vowels):
        return True
    return [variant_stress[k] for k in pos] == aligned_vowels


def _flag(code: str, severity: str, scope: str, message: str, **kw: object) -> QualityFlag:
    return QualityFlag(code=code, severity=severity, scope=scope, message=message, **kw)


def _issue_to_flag(issue: TextGridIssue, word_tier: str, phone_tier: str | None) -> QualityFlag:
    scope = "word" if issue.tier == word_tier else "phone" if issue.tier == phone_tier else "sample"
    return QualityFlag(
        code=f"textgrid_{issue.code}", severity=issue.severity, scope=scope,
        message=issue.message,
        details={"tier": issue.tier, "interval_index": issue.interval_index, **issue.details},
    )


def assemble_sample_record(
    *,
    g2p: G2PResult,
    word_intervals: Sequence[IntervalIn],
    phone_intervals: Sequence[IntervalIn] | None,
    word_tier_name: str,
    phone_tier_name: str | None,
    config: AppConfig,
    inventory: PhoneInventory,
    sample_id: str,
    speaker_id: str | None,
    reference_text: str | None,
    human_verbatim: str | None,
    alignment_metadata: AlignmentMetadata,
    mapper: AlignerLabelMapper | None = None,
    source_file: str | None = None,
    extra_flags: Sequence[QualityFlag] = (),
    inputs: dict[str, object] | None = None,
) -> SampleRecord:
    """Core assembly shared by the MFA/TextGrid path and word-only aligners."""
    qc = config.alignment.qc
    tg_cfg = config.alignment.textgrid
    silence = set(mapper.silence_labels if mapper else ()) | set(tg_cfg.extra_silence_labels)
    oov = set(mapper.oov_labels if mapper else ()) | set(tg_cfg.extra_oov_labels)
    flags: list[QualityFlag] = list(g2p.quality_flags) + list(extra_flags)
    tokens: list[TokenRecord] = [t.model_copy(deep=True) for t in g2p.tokens]

    # ---- words ------------------------------------------------------------------
    valid_words = [iv for iv in word_intervals if _is_valid(iv)]
    for iv in word_intervals:
        if not _is_valid(iv):
            flags.append(_flag("invalid_interval_excluded", "error", "word",
                               f"word interval #{iv.index} ({iv.label!r}: {iv.start}..{iv.end}) is invalid and was excluded; "
                               "no timestamps were repaired or invented",
                               details={"interval_index": iv.index, "start": iv.start, "end": iv.end}))
    roles = [interval_role(iv.label, silence, oov, "word") for iv in valid_words]
    speech_pos = [k for k, r in enumerate(roles) if r != "silence"]
    aligned_labels = [valid_words[k].label.strip().lower() for k in speech_pos]
    expected_tokens = [t for t in tokens if t.approved_for_alignment]
    expected_labels = [t.normalized_token for t in expected_tokens]

    word_to_token: dict[int, int] = {}  # valid_words position -> token.word_index
    matched_tokens: set[int] = set()
    missing_words = extra_words = 0
    missing_phone_lower_bound = 0
    for e_i, a_i, relation in match_sequences(expected_labels, aligned_labels):
        if relation in {"equal", "substituted"}:
            assert e_i is not None and a_i is not None
            tok, pos = expected_tokens[e_i], speech_pos[a_i]
            word_to_token[pos] = tok.word_index
            matched_tokens.add(tok.word_index)
            if relation == "substituted":
                if roles[pos] == "oov_marker":
                    flags.append(_flag("aligner_oov_marker", "warning", "word",
                                       f"aligner labelled the expected word {tok.normalized_token!r} as {valid_words[pos].label!r}",
                                       word_index=tok.word_index, start_time=valid_words[pos].start, end_time=valid_words[pos].end))
                else:
                    flags.append(_flag("word_label_mismatch", "warning", "word",
                                       f"expected {tok.normalized_token!r} but aligner label is {valid_words[pos].label!r}",
                                       word_index=tok.word_index, start_time=valid_words[pos].start, end_time=valid_words[pos].end))
        elif relation == "missing":
            assert e_i is not None
            tok = expected_tokens[e_i]
            missing_words += 1
            missing_phone_lower_bound += min((len(v.base_phones) for v in tok.variants), default=0)
            flags.append(_flag("missing_expected_word", "error", "word",
                               f"expected word {tok.normalized_token!r} has no aligned interval", word_index=tok.word_index))
        else:
            assert a_i is not None
            pos = speech_pos[a_i]
            extra_words += 1
            flags.append(_flag("unexpected_word_label", "warning", "word",
                               f"aligned word label {valid_words[pos].label!r} matches no expected token",
                               start_time=valid_words[pos].start, end_time=valid_words[pos].end,
                               details={"interval_index": valid_words[pos].index}))
    word_alignments = [
        WordAlignment(
            word_index=word_to_token.get(k), label=iv.label, label_role=roles[k],
            start_time=iv.start, end_time=iv.end, source_tier=word_tier_name,
            interval_index=iv.index, source_file=source_file,
        )
        for k, iv in enumerate(valid_words)
    ]
    oov_word_intervals = sum(1 for r in roles if r == "oov_marker")

    # ---- phones -----------------------------------------------------------------
    phone_alignments: list[PhoneAlignment] = []
    missing_phones = extra_phones = mismatched_phones = unknown_labels = 0
    oov_phone_intervals = 0
    if phone_intervals is not None and mapper is not None:
        valid_phones = [iv for iv in phone_intervals if _is_valid(iv)]
        for iv in phone_intervals:
            if not _is_valid(iv):
                flags.append(_flag("invalid_interval_excluded", "error", "phone",
                                   f"phone interval #{iv.index} ({iv.label!r}: {iv.start}..{iv.end}) is invalid and was excluded",
                                   details={"interval_index": iv.index, "start": iv.start, "end": iv.end}))
        order = sorted(range(len(valid_words)), key=lambda k: valid_words[k].start)
        by_token: dict[int, list[int]] = {}  # token index -> indices into phone_alignments
        counter = 0
        for iv in valid_phones:
            info = mapper.classify(iv.label)
            role = {LabelRole.PHONE: "phone", LabelRole.SILENCE: "silence",
                    LabelRole.OOV: "oov_marker", LabelRole.UNKNOWN: "unknown"}[info.role]
            mid = (iv.start + iv.end) / 2
            w_pos = next((k for k in order if valid_words[k].start <= mid < valid_words[k].end), None)
            w_index = word_to_token.get(w_pos) if (w_pos is not None and role != "silence") else None
            phone_idx = None
            if role in {"phone", "oov_marker", "unknown"}:
                phone_idx = counter
                counter += 1
            if role == "unknown":
                unknown_labels += 1
                flags.append(_flag("unexpected_phone_label", "warning", "phone",
                                   f"phone label {iv.label!r} is not in the inventory ({info.reason})",
                                   phone_index=phone_idx, word_index=w_index, start_time=iv.start, end_time=iv.end))
            if role == "oov_marker":
                oov_phone_intervals += 1
            pa = PhoneAlignment(
                phone_index=phone_idx, interval_index=iv.index, word_index=w_index, aligned_label=iv.label,
                label_role=role,
                aligned_base_phone=info.base_phone, aligned_stress=info.stress,
                aligned_phone_id=inventory.phone_id(info.base_phone) if info.base_phone else None,
                start_time=iv.start, end_time=iv.end, source_tier=phone_tier_name or "phones",
                source_file=source_file,
            )
            phone_alignments.append(pa)
            if role == "phone" and w_index is not None:
                by_token.setdefault(w_index, []).append(len(phone_alignments) - 1)

        # variant resolution + per-word phone comparison
        for ti, tok in enumerate(tokens):
            if not tok.approved_for_alignment or tok.word_index not in matched_tokens:
                continue
            idxs = by_token.get(tok.word_index, [])
            aligned_bases = [phone_alignments[i].aligned_base_phone or "" for i in idxs]
            aligned_stress = [phone_alignments[i].aligned_stress for i in idxs]
            if not idxs:
                flags.append(_flag("word_without_phones", "warning", "word",
                                   f"aligned word {tok.normalized_token!r} contains no phone intervals",
                                   word_index=tok.word_index))
                missing_phones += min((len(v.base_phones) for v in tok.variants), default=0)
                continue
            exact = [v for v in tok.variants if v.base_phones == aligned_bases and _stress_agrees(v.stress, aligned_stress)]
            if tok.selected_variant_id is None:
                if len(exact) == 1:
                    tokens[ti] = tok.model_copy(update={
                        "selected_variant_id": exact[0].variant_id, "selection_method": "alignment_match",
                        "flags": [f for f in tok.flags if f != "variant_unselected"] + ["variant_resolved_by_alignment"],
                    })
                    flags.append(_flag("variant_resolved_by_alignment", "info", "word",
                                       f"aligner's phones match only variant {exact[0].variant_id}; this records the aligner's "
                                       "choice, not what the learner said", word_index=tok.word_index))
                    continue
                if len(exact) > 1:
                    flags.append(_flag("variant_ambiguous_after_alignment", "info", "word",
                                       f"{len(exact)} variants match the aligned phones; none selected",
                                       word_index=tok.word_index))
                    continue
                reference = max(tok.variants, key=lambda v: SequenceMatcher(None, v.base_phones, aligned_bases).ratio())
            else:
                reference = tok.selected_variant()  # type: ignore[assignment]
                if exact and reference is not None and reference.variant_id not in {v.variant_id for v in exact}:
                    flags.append(_flag("aligned_phones_match_other_variant", "info", "word",
                                       f"aligned phones match {exact[0].variant_id}, not the selected {reference.variant_id}",
                                       word_index=tok.word_index))
                    continue
                if exact:
                    continue
            assert reference is not None
            d = diff_phones(reference.base_phones, aligned_bases)
            missing_phones += d.missing
            extra_phones += d.extra
            mismatched_phones += d.substituted
            flags.append(_flag("phones_do_not_match_any_variant", "warning", "word",
                               f"aligned phones {aligned_bases} do not match any candidate for {tok.normalized_token!r} "
                               f"(closest: {reference.variant_id} = {reference.base_phones})",
                               word_index=tok.word_index,
                               details={"aligned": aligned_bases, "closest_variant": reference.variant_id,
                                        "missing": d.missing, "extra": d.extra, "substituted": d.substituted}))

        # canonical target per aligned phone (only when the aligned sequence equals the selected variant)
        canonical = build_canonical_phones(tokens)
        start_of: dict[int, int] = {}
        for c in canonical:
            start_of.setdefault(c.word_index, c.index)
        for tok in tokens:
            variant = tok.selected_variant()
            idxs = by_token.get(tok.word_index, [])
            if variant is None or len(idxs) != len(variant.base_phones):
                continue
            if [phone_alignments[i].aligned_base_phone for i in idxs] != variant.base_phones:
                continue
            for pos, i in enumerate(idxs):
                phone_alignments[i] = phone_alignments[i].model_copy(update={
                    "canonical_phone_index": start_of[tok.word_index] + pos,
                    "canonical_target_phone": variant.base_phones[pos],
                    "canonical_target_stress": variant.stress[pos],
                })
    canonical_phones = build_canonical_phones(tokens)

    # ---- QC ---------------------------------------------------------------------
    flags += check_durations(word_alignments, phone_alignments, qc)
    unresolved = [t for t in tokens if t.resolution_status == "unresolved"]
    marked_mapped = {w.word_index for w in word_alignments if w.label_role == "oov_marker" and w.word_index is not None}
    marked_unmapped = sum(1 for w in word_alignments if w.label_role == "oov_marker" and w.word_index is None)
    # distinct OOV words: unresolved in G2P or labelled as OOV by the aligner (+ OOV intervals traceable to no token)
    oov_count = len({t.word_index for t in unresolved} | marked_mapped) + marked_unmapped
    errors = [f for f in flags if f.severity == "error"]
    usable_words = any(w.label_role in {"word", "oov_marker"} for w in word_alignments)
    status = "failed" if not usable_words else "partial" if errors else "success"
    meta = alignment_metadata.model_copy(update={
        "status": status,
        # phone-level output exists whenever a phone tier produced entries (even if they are all OOV/silence markers)
        "granularity": "none" if not usable_words else "phone" if phone_alignments else "word",
        "qc_summary": {
            "oov_count": oov_count,
            "unresolved_tokens": len(unresolved),
            "oov_word_intervals": oov_word_intervals,
            "oov_phone_intervals": oov_phone_intervals,
            "missing_words": missing_words,
            "extra_aligned_words": extra_words,
            "missing_phone_count": missing_phones + missing_phone_lower_bound,
            "extra_phone_count": extra_phones,
            "phone_label_mismatch_count": mismatched_phones,
            "unexpected_label_count": unknown_labels,
            "short_phone_count": sum(f.code == "phone_too_short" for f in flags),
            "long_phone_count": sum(f.code == "phone_too_long" for f in flags),
            "short_word_count": sum(f.code == "word_too_short" for f in flags),
            "long_word_count": sum(f.code == "word_too_long" for f in flags),
            "invalid_interval_count": sum(f.code == "invalid_interval_excluded" for f in flags),
            "error_flag_count": len(errors),
            "warning_flag_count": sum(f.severity == "warning" for f in flags),
            "missing_phone_count_note": "missing words are counted with the shortest candidate variant (lower bound)",
        },
    })
    provenance = build_provenance(config, inventory, g2p_sources=(g2p.provenance.g2p_sources if g2p.provenance else {}),
                                  inputs=inputs or {})
    return SampleRecord(
        sample_id=sample_id, speaker_id=speaker_id, reference_text=reference_text, human_verbatim=human_verbatim,
        g2p_input_field=g2p.text_field, normalization=g2p.normalization, tokens=tokens,
        canonical_phones=canonical_phones, word_alignments=word_alignments, phone_alignments=phone_alignments,
        alignment_metadata=meta,
        quality_flags=flags if config.output.include_quality_flags else [],
        provenance=provenance,
    )


def record_from_textgrid(
    *,
    g2p: G2PResult,
    textgrid: ParsedTextGrid,
    config: AppConfig,
    inventory: PhoneInventory,
    mapper: AlignerLabelMapper,
    sample_id: str,
    speaker_id: str | None,
    reference_text: str | None,
    human_verbatim: str | None,
    alignment_metadata: AlignmentMetadata,
    word_tier: str | None = None,
    phone_tier: str | None = None,
    audio_duration_s: float | None = None,
    inputs: dict[str, object] | None = None,
) -> SampleRecord:
    """Build a :class:`SampleRecord` from a parsed TextGrid (tier names are configurable)."""
    tg_cfg = config.alignment.textgrid
    w_tier = select_tier(textgrid, tg_cfg.word_tier_names, explicit_name=word_tier,
                         allow_speaker_prefix=tg_cfg.allow_speaker_prefix, kind="word")
    p_tier = None
    missing_phone_tier: QualityFlag | None = None
    try:
        p_tier = select_tier(textgrid, tg_cfg.phone_tier_names, explicit_name=phone_tier,
                             allow_speaker_prefix=tg_cfg.allow_speaker_prefix, kind="phone")
    except Exception as exc:  # no phone tier: word-level output only, reported explicitly
        if phone_tier is not None:
            raise
        missing_phone_tier = _flag("missing_phone_tier", "warning", "sample",
                                   f"no phone tier found; output is word-level only ({exc})")
    issues = validate_textgrid(textgrid, config.alignment.qc, audio_duration_s=audio_duration_s)
    extra = [_issue_to_flag(i, w_tier.name, p_tier.name if p_tier else None) for i in issues]
    if missing_phone_tier:
        extra.append(missing_phone_tier)
    merged_inputs = {"textgrid": {"name": textgrid.path.name, "sha256": textgrid.sha256}, **(inputs or {})}
    return assemble_sample_record(
        g2p=g2p,
        word_intervals=[IntervalIn(i.index, i.start, i.end, i.label) for i in w_tier.intervals],
        phone_intervals=[IntervalIn(i.index, i.start, i.end, i.label) for i in p_tier.intervals] if p_tier else None,
        word_tier_name=w_tier.name, phone_tier_name=p_tier.name if p_tier else None,
        config=config, inventory=inventory, sample_id=sample_id, speaker_id=speaker_id,
        reference_text=reference_text, human_verbatim=human_verbatim,
        alignment_metadata=alignment_metadata, mapper=mapper, source_file=textgrid.path.name,
        extra_flags=extra, inputs=merged_inputs,
    )


def write_word_textgrid(path: Path, words: Sequence[tuple[float, float, str]], duration: float, *, overwrite: bool = False) -> Path:
    """Write a word-tier TextGrid with ``praatio`` (used for word-level aligners such as Qwen)."""
    from praatio import textgrid as pt
    from praatio.utilities.constants import Interval  # signature verified against praatio 6.2.2

    from asa_m2.utils.serialization import check_writable

    check_writable(path, overwrite=overwrite)
    if not words:
        raise AlignmentError("cannot write a TextGrid without intervals")
    tier = pt.IntervalTier("words", [Interval(s, e, lab) for s, e, lab in words], 0.0, duration)
    tg = pt.Textgrid()
    tg.addTier(tier)
    path.parent.mkdir(parents=True, exist_ok=True)
    tg.save(str(path), format="long_textgrid", includeBlankSpaces=True)
    return path
