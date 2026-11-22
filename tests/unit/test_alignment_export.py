"""M2.6/M2.7 export + quality control: the M2 -> M3/M4 contract in action."""

import pytest
from pydantic import ValidationError

from asa_m2.alignment import (
    IntervalIn,
    assemble_sample_record,
    compare_tiers,
    evaluate_boundaries,
    parse_textgrid,
    record_from_textgrid,
)
from asa_m2.alignment.quality import diff_phones, match_sequences
from asa_m2.exceptions import TextGridError
from asa_m2.schemas import AlignmentMetadata, PhoneAlignment, SampleRecord

from ..conftest import aligned_tiers  # noqa: TID252

META = AlignmentMetadata(aligner="MFA", aligner_version="3.4.3", granularity="phone", status="success")


@pytest.fixture()
def export(engine, config, inventory, mapper, write_tg):
    def _export(text, tiers, xmax=None, *, cfg=None, name="s.TextGrid", audio_duration_s=None, g2p=None, **kw):
        g2p = g2p or engine.convert(text, sample_id="S1")
        tg = parse_textgrid(write_tg(name, tiers, xmax))
        return record_from_textgrid(
            g2p=g2p, textgrid=tg, config=cfg or config, inventory=inventory, mapper=mapper, sample_id="S1",
            speaker_id="SPK_1", reference_text=text, human_verbatim=None, alignment_metadata=META,
            audio_duration_s=audio_duration_s, **kw)
    return _export


def codes(rec, severity=None):
    return sorted({f.code for f in rec.quality_flags if severity is None or f.severity == severity})


THINK = [("i", ["AY1"]), ("think", ["TH", "IH1", "NG", "K"])]


def test_clean_alignment_succeeds_with_traceable_intervals(export):
    tiers, dur = aligned_tiers(THINK)
    rec = export("I think", tiers, dur)
    assert rec.alignment_metadata.status == "success" and rec.alignment_metadata.granularity == "phone"
    assert [(w.label, w.word_index, w.label_role) for w in rec.word_alignments] == [
        ("", None, "silence"), ("i", 0, "word"), ("think", 1, "word"), ("", None, "silence")]
    assert all(w.source_tier == "words" and w.source_file == "s.TextGrid" for w in rec.word_alignments)
    assert [(p.aligned_label, p.word_index) for p in rec.phone_alignments if p.label_role == "phone"] == [
        ("AY1", 0), ("TH", 1), ("IH1", 1), ("NG", 1), ("K", 1)]
    assert codes(rec, "error") == []


def test_timestamps_come_only_from_the_aligner(export):
    tiers, dur = aligned_tiers(THINK)
    rec = export("I think", tiers, dur)
    got = {(w.label, w.start_time, w.end_time) for w in rec.word_alignments if w.label}
    assert got == {(lab, a, b) for a, b, lab in tiers["words"] if lab}
    got_p = [(p.aligned_label, p.start_time, p.end_time) for p in rec.phone_alignments if p.label_role == "phone"]
    assert got_p == [(lab, a, b) for a, b, lab in tiers["phones"] if lab]


def test_canonical_target_aligned_label_and_realized_phone_are_distinct(export):
    tiers, dur = aligned_tiers(THINK)
    rec = export("I think", tiers, dur)
    p = next(p for p in rec.phone_alignments if p.aligned_label == "IH1")
    assert (p.canonical_target_phone, p.canonical_target_stress, p.aligned_base_phone, p.aligned_stress) == ("IH", 1, "IH", 1)
    assert p.realized_phone is None and p.realized_phone_status == "not_determined_by_forced_alignment"
    assert all(q.realized_phone is None for q in rec.phone_alignments)  # never a verified realised phone
    assert rec.canonical_phones[p.canonical_phone_index].base_phone == "IH"  # links back into canonical_phones


def test_realized_phone_cannot_be_set_on_the_schema():
    base = dict(interval_index=0, aligned_label="AY1", label_role="phone", start_time=0.0, end_time=0.1, source_tier="phones")
    PhoneAlignment(**base)
    with pytest.raises(ValidationError, match="cannot determine realised phones"):
        PhoneAlignment(**base, realized_phone="AY")


def test_alignment_resolves_variant_and_labels_it_as_aligner_choice(export):
    tiers, dur = aligned_tiers([("this", ["DH", "IH0", "S"])])
    rec = export("this", tiers, dur)
    tok = rec.tokens[0]
    assert tok.selected_variant_id == "cmudict:this:2" and tok.selection_method == "alignment_match"
    assert "variant_resolved_by_alignment" in codes(rec, "info")
    assert [c.variant_id for c in rec.canonical_phones] == ["cmudict:this:2"] * 3
    flag = next(f for f in rec.quality_flags if f.code == "variant_resolved_by_alignment")
    assert "not what the learner said" in flag.message


def test_variant_stays_unselected_when_aligned_labels_carry_no_stress(export):
    tiers, dur = aligned_tiers([("this", ["DH", "IH", "S"])])  # both variants match once stress is ignored
    rec = export("this", tiers, dur)
    assert rec.tokens[0].selected_variant_id is None and rec.canonical_phones == []
    assert "variant_ambiguous_after_alignment" in codes(rec, "info")


def test_policy_selected_variant_is_not_overridden_by_alignment(export, config, inventory):
    from asa_m2.g2p import G2PEngine
    cfg = config.model_copy(deep=True)
    cfg.g2p.variant_policy = "first_listed"
    g2p = G2PEngine.from_config(cfg, inventory).convert("this", sample_id="S1")
    assert g2p.tokens[0].selection_method == "policy_first_listed"
    tiers, dur = aligned_tiers([("this", ["DH", "IH0", "S"])])  # aligner chose the 2nd variant
    rec = export("this", tiers, dur, g2p=g2p)
    tok = rec.tokens[0]
    assert tok.selected_variant_id == "cmudict:this:1" and tok.selection_method == "policy_first_listed"
    assert "aligned_phones_match_other_variant" in codes(rec, "info")
    # target and aligned label stay distinct and both visible: target IH1 vs aligner's IH0
    ih = next(p for p in rec.phone_alignments if p.aligned_base_phone == "IH")
    assert (ih.canonical_target_phone, ih.canonical_target_stress, ih.aligned_stress) == ("IH", 1, 0)


def test_missing_word_is_reported_and_later_words_still_map_correctly(export):
    tiers, dur = aligned_tiers([("i", ["AY1"]), ("cat", ["K", "AE1", "T"])])  # 'think' never aligned
    rec = export("I think cat", tiers, dur)
    assert "missing_expected_word" in codes(rec, "error")
    miss = next(f for f in rec.quality_flags if f.code == "missing_expected_word")
    assert miss.word_index == 1 and "think" in miss.message
    cat = next(w for w in rec.word_alignments if w.label == "cat")
    assert cat.word_index == 2  # sequence diff: no positional shift
    assert rec.alignment_metadata.status == "partial"
    qc = rec.alignment_metadata.qc_summary
    assert qc["missing_words"] == 1 and qc["missing_phone_count"] == 4  # 'think' = TH IH NG K (lower bound)


def test_extra_word_label_is_reported_without_a_token(export):
    tiers, dur = aligned_tiers([("i", ["AY1"]), ("um", ["AH1", "M"]), ("think", ["TH", "IH1", "NG", "K"])])
    rec = export("I think", tiers, dur)
    assert "unexpected_word_label" in codes(rec, "warning")
    um = next(w for w in rec.word_alignments if w.label == "um")
    assert um.word_index is None and rec.alignment_metadata.qc_summary["extra_aligned_words"] == 1


def test_aligner_oov_markers_are_counted(export):
    tiers, dur = aligned_tiers([("i", ["AY1"]), ("<unk>", ["spn"])])
    rec = export("I think", tiers, dur)
    assert "aligner_oov_marker" in codes(rec, "warning")
    w = next(w for w in rec.word_alignments if w.label == "<unk>")
    assert w.label_role == "oov_marker" and w.word_index == 1
    p = next(p for p in rec.phone_alignments if p.aligned_label == "spn")
    assert p.label_role == "oov_marker" and p.aligned_base_phone is None
    qc = rec.alignment_metadata.qc_summary
    assert qc["oov_count"] == 1 and qc["oov_word_intervals"] == 1 and qc["oov_phone_intervals"] == 1


def test_unresolved_g2p_words_count_as_oov_and_make_status_partial(engine, config, inventory, mapper, write_tg):
    g2p = engine.convert("the blorptastic cat", sample_id="S1")
    tiers, dur = aligned_tiers([("the", ["DH", "AH0"]), ("cat", ["K", "AE1", "T"])])
    rec = record_from_textgrid(g2p=g2p, textgrid=parse_textgrid(write_tg("u.TextGrid", tiers, dur)), config=config,
                               inventory=inventory, mapper=mapper, sample_id="S1", speaker_id=None,
                               reference_text="the blorptastic cat", human_verbatim=None, alignment_metadata=META)
    assert rec.alignment_metadata.qc_summary["oov_count"] == 1 and rec.alignment_metadata.qc_summary["unresolved_tokens"] == 1
    assert "unresolved_word" in codes(rec, "error") and rec.alignment_metadata.status == "partial"


def test_unknown_phone_label_flagged_never_remapped(export):
    tiers, dur = aligned_tiers([("i", ["AY1"]), ("think", ["TH", "ʔ", "NG", "K"])])
    rec = export("I think", tiers, dur)
    assert "unexpected_phone_label" in codes(rec, "warning")
    bad = next(p for p in rec.phone_alignments if p.aligned_label == "ʔ")
    assert bad.label_role == "unknown" and bad.aligned_base_phone is None and bad.canonical_target_phone is None
    assert rec.alignment_metadata.qc_summary["unexpected_label_count"] == 1


def test_phones_not_matching_any_variant_report_counts_and_closest(export):
    tiers, dur = aligned_tiers([("think", ["TH", "IH1", "K"])])  # NG missing
    rec = export("think", tiers, dur)
    f = next(f for f in rec.quality_flags if f.code == "phones_do_not_match_any_variant")
    assert f.details["missing"] == 1 and f.details["closest_variant"] == "cmudict:think:1"
    qc = rec.alignment_metadata.qc_summary
    assert qc["missing_phone_count"] == 1 and qc["extra_phone_count"] == 0
    assert all(p.canonical_target_phone is None for p in rec.phone_alignments if p.label_role == "phone")  # no false link


def test_word_without_phones(export):
    tiers, dur = aligned_tiers(THINK)
    tiers["phones"] = [iv for iv in tiers["phones"] if iv[2] not in {"TH", "IH1", "NG", "K"}]
    tiers["phones"].sort()
    rec = export("I think", tiers, dur)
    assert "word_without_phones" in codes(rec, "warning")


def test_silence_intervals_are_kept_explicitly(export):
    tiers, dur = aligned_tiers(THINK)
    tiers["phones"][0] = (0.0, 0.1, "sil")
    rec = export("I think", tiers, dur)
    sil = [p for p in rec.phone_alignments if p.label_role == "silence"]
    assert len(sil) == 2 and all(p.phone_index is None and p.aligned_phone_id is None for p in sil)
    assert {p.aligned_label for p in sil} == {"sil", ""}


# ---- invalid intervals: excluded + reported, never repaired ----------------------------------
def test_invalid_intervals_are_excluded_and_reported_without_inventing_times(export):
    tiers, dur = aligned_tiers(THINK)
    tiers["words"][2] = (tiers["words"][2][1], tiers["words"][2][0], "think")  # end < start
    rec = export("I think", tiers, dur)
    assert "invalid_interval_excluded" in codes(rec, "error") and "textgrid_negative_duration" in codes(rec, "error")
    assert "think" not in [w.label for w in rec.word_alignments]
    assert all(w.end_time > w.start_time for w in rec.word_alignments)
    assert rec.alignment_metadata.qc_summary["invalid_interval_count"] == 1 and rec.alignment_metadata.status == "partial"


def test_overlapping_intervals_are_flagged_but_both_kept(export):
    tiers, dur = aligned_tiers(THINK)
    a, b = tiers["words"][1], tiers["words"][2]
    tiers["words"][2] = (a[1] - 0.05, b[1], b[2])
    rec = export("I think", tiers, dur)
    assert "textgrid_overlap" in codes(rec, "error")
    assert {"i", "think"} <= {w.label for w in rec.word_alignments}


def test_failed_when_there_are_no_usable_word_intervals(export):
    rec = export("I think", {"words": [(0.0, 1.0, "")], "phones": [(0.0, 1.0, "")]}, 1.0)
    assert rec.alignment_metadata.status == "failed" and rec.alignment_metadata.granularity == "none"
    assert "missing_expected_word" in codes(rec, "error")


# ---- tiers -------------------------------------------------------------------------------------------
def test_phone_tier_containing_only_oov_markers_does_not_crash(export):
    """Regression: all-`spn` phone tier used to yield granularity 'word' with phone entries -> schema error."""
    tiers, dur = aligned_tiers([("i", ["spn"]), ("think", ["spn"])])
    rec = export("I think", tiers, dur)
    assert rec.alignment_metadata.granularity == "phone" and rec.alignment_metadata.qc_summary["oov_phone_intervals"] == 2
    assert [p.label_role for p in rec.phone_alignments if p.label_role != "silence"] == ["oov_marker", "oov_marker"]
    assert rec.alignment_metadata.status in {"success", "partial"}


def test_word_only_textgrid_gives_word_granularity(export):
    tiers, dur = aligned_tiers(THINK)
    rec = export("I think", {"words": tiers["words"]}, dur)
    assert rec.alignment_metadata.granularity == "word" and rec.phone_alignments == []
    assert "missing_phone_tier" in codes(rec, "warning")


def test_missing_word_tier_raises(export):
    tiers, dur = aligned_tiers(THINK)
    with pytest.raises(TextGridError, match="no word tier"):
        export("I think", {"phones": tiers["phones"]}, dur)


def test_custom_tier_names_and_explicit_selection(export):
    tiers, dur = aligned_tiers(THINK)
    renamed = {"Wort": tiers["words"], "Laut": tiers["phones"]}
    rec = export("I think", renamed, dur, word_tier="Wort", phone_tier="Laut")
    assert rec.alignment_metadata.status == "success" and rec.word_alignments[1].source_tier == "Wort"


def test_speaker_prefixed_tiers(export):
    tiers, dur = aligned_tiers(THINK)
    rec = export("I think", {"spk1 - words": tiers["words"], "spk1 - phones": tiers["phones"]}, dur)
    assert rec.alignment_metadata.status == "success"


def test_audio_duration_mismatch_is_flagged(export):
    tiers, dur = aligned_tiers(THINK)
    rec = export("I think", tiers, dur, audio_duration_s=dur + 2.0)
    assert "textgrid_duration_mismatch" in codes(rec, "warning")


# ---- thresholds are configurable warnings, never deletions ---------------------------------------
def test_duration_thresholds_warn_but_never_delete(export, config):
    tiers, dur = aligned_tiers(THINK, ph_dur=0.015)  # 15 ms phones
    rec = export("I think", tiers, dur)
    short = [f for f in rec.quality_flags if f.code == "phone_too_short"]
    assert len(short) == 5 and all(f.severity == "warning" for f in short)
    assert "not a pronunciation-error threshold" in short[0].message
    assert len([p for p in rec.phone_alignments if p.label_role == "phone"]) == 5  # nothing removed
    assert rec.alignment_metadata.status == "success"
    relaxed = config.model_copy(deep=True)
    relaxed.alignment.qc.min_phone_duration_s = 0.01
    assert "phone_too_short" not in codes(export("I think", tiers, dur, cfg=relaxed, name="r.TextGrid"))


def test_long_intervals_flagged(export, config):
    tiers, dur = aligned_tiers([("think", ["TH", "IH1", "NG", "K"])], ph_dur=1.2)  # phones 1.2 s, word 4.8 s
    c = codes(export("think", tiers, dur))
    assert "phone_too_long" in c and "word_too_long" in c


def test_qc_summary_has_all_required_indicators(export):
    tiers, dur = aligned_tiers(THINK)
    qc = export("I think", tiers, dur).alignment_metadata.qc_summary
    for key in ("oov_count", "missing_phone_count", "missing_words", "short_phone_count", "long_phone_count",
                "invalid_interval_count", "unexpected_label_count", "error_flag_count", "warning_flag_count"):
        assert key in qc


# ---- schema-level guarantees ----------------------------------------------------------------------
def test_record_roundtrips_json_and_validates(export):
    tiers, dur = aligned_tiers(THINK)
    rec = export("I think", tiers, dur)
    again = SampleRecord.model_validate_json(rec.model_dump_json())
    assert again == rec and again.schema_version == "1.0"
    assert rec.provenance.inputs["textgrid"]["name"] == "s.TextGrid" and len(rec.provenance.inputs["textgrid"]["sha256"]) == 64
    assert "/" not in rec.provenance.inputs["textgrid"]["name"]  # no absolute paths leak into outputs


def test_reference_text_and_verbatim_are_separate_fields(engine, config, inventory, mapper, write_tg):
    tiers, dur = aligned_tiers(THINK)
    g2p = engine.convert("I think", sample_id="S1")
    rec = record_from_textgrid(g2p=g2p, textgrid=parse_textgrid(write_tg("v.TextGrid", tiers, dur)), config=config,
                               inventory=inventory, mapper=mapper, sample_id="S1", speaker_id=None,
                               reference_text="I think", human_verbatim="I thinks", alignment_metadata=META)
    assert (rec.reference_text, rec.human_verbatim, rec.g2p_input_field) == ("I think", "I thinks", "reference_text")
    with pytest.raises(ValidationError, match="at least one"):
        SampleRecord(sample_id="x")
    with pytest.raises(ValidationError, match="missing field"):
        SampleRecord(sample_id="x", reference_text="a", g2p_input_field="human_verbatim")


def test_word_granularity_forbids_phone_alignments():
    with pytest.raises(ValidationError, match="must be empty"):
        SampleRecord(sample_id="x", reference_text="a",
                     alignment_metadata=AlignmentMetadata(aligner="Q", granularity="word", status="success"),
                     phone_alignments=[PhoneAlignment(interval_index=0, aligned_label="A", label_role="phone",
                                                      start_time=0, end_time=1, source_tier="p")])
    with pytest.raises(ValidationError, match="not_run"):
        AlignmentMetadata(status="not_run", granularity="word")


def test_include_flags_and_provenance_switches(export, config):
    cfg = config.model_copy(deep=True)
    cfg.output.include_quality_flags = False
    cfg.output.include_provenance = False
    tiers, dur = aligned_tiers(THINK, ph_dur=0.015)
    rec = export("I think", tiers, dur, cfg=cfg)
    assert rec.quality_flags == [] and rec.provenance is None
    assert rec.alignment_metadata.qc_summary["short_phone_count"] == 5  # summary still computed


def test_word_only_assembly_for_word_level_aligners(engine, config, inventory):
    g2p = engine.convert("I think", sample_id="S1")
    rec = assemble_sample_record(
        g2p=g2p, word_intervals=[IntervalIn(0, 0.1, 0.4, "i"), IntervalIn(1, 0.4, 0.9, "think")], phone_intervals=None,
        word_tier_name="qwen_words", phone_tier_name=None, config=config, inventory=inventory, sample_id="S1",
        speaker_id=None, reference_text="I think", human_verbatim=None,
        alignment_metadata=AlignmentMetadata(aligner="Qwen3-ForcedAligner", granularity="word", status="success"))
    assert rec.alignment_metadata.granularity == "word" and rec.phone_alignments == []
    assert [w.word_index for w in rec.word_alignments] == [0, 1]
    assert rec.tokens[1].selection_method == "sole_variant"  # word aligner cannot resolve variants


# ---- sequence helpers ------------------------------------------------------------------------------
def test_match_sequences_handles_insertions_deletions_substitutions():
    assert match_sequences(["a", "b", "c"], ["a", "b", "c"]) == [(0, 0, "equal"), (1, 1, "equal"), (2, 2, "equal")]
    rel = {(e, a): r for e, a, r in match_sequences(["a", "b", "c"], ["a", "c"])}
    assert rel[(1, None)] == "missing" and rel[(2, 1)] == "equal"
    rel2 = {(e, a): r for e, a, r in match_sequences(["a", "b"], ["a", "x", "b"])}
    assert rel2[(None, 1)] == "extra"
    assert match_sequences(["a"], ["z"]) == [(0, 0, "substituted")]


def test_diff_phones():
    assert diff_phones(["A", "B"], ["A", "B"]).identical
    d = diff_phones(["A", "B", "C"], ["A", "X"])
    assert (d.missing, d.extra, d.substituted) == (1, 0, 1)
    assert diff_phones(["A"], ["A", "B", "C"]).extra == 2


# ---- gold evaluation -----------------------------------------------------------------------------------
def tier_of(write_tg, name, ivs):
    return parse_textgrid(write_tg(name, {"phones": ivs}, max(i[1] for i in ivs))).get_tier("phones")


GOLD = [(0, 0.1, ""), (0.1, 0.2, "AY1"), (0.2, 0.4, "TH"), (0.4, 0.5, "")]


def test_perfect_prediction_has_zero_error(write_tg):
    r = compare_tiers("a", tier_of(write_tg, "p.TextGrid", GOLD), tier_of(write_tg, "g.TextGrid", GOLD))
    ev = evaluate_boundaries([r], 0.02)
    assert not r.failed and ev.mae_s == 0 and ev.median_abs_error_s == 0 and ev.pct_within_tolerance == 100.0
    # silence is excluded; the speech boundaries are 0.1, 0.2 and 0.4 (the shared 0.2 is counted once)
    assert ev.n_boundaries == 3 and ev.failure_rate == 0


def test_boundary_errors_are_computed_exactly(write_tg):
    pred = [(0, 0.1, ""), (0.1, 0.23, "AY1"), (0.23, 0.45, "TH"), (0.45, 0.5, "")]
    r = compare_tiers("a", tier_of(write_tg, "p.TextGrid", pred), tier_of(write_tg, "g.TextGrid", GOLD))
    assert sorted(round(e, 4) for e in r.errors_s) == [0.0, 0.03, 0.05]  # boundaries 0.1 (exact), 0.2->0.23, 0.4->0.45
    ev = evaluate_boundaries([r], 0.02)
    assert ev.mae_s == pytest.approx(0.08 / 3) and ev.median_abs_error_s == pytest.approx(0.03)
    assert ev.pct_within_tolerance == pytest.approx(100 / 3) and ev.tolerance_s == 0.02


def test_stress_is_ignored_by_default_and_failures_are_counted(write_tg):
    pred = [(0, 0.1, ""), (0.1, 0.2, "AY0"), (0.2, 0.4, "TH"), (0.4, 0.5, "")]
    ok = compare_tiers("a", tier_of(write_tg, "p.TextGrid", pred), tier_of(write_tg, "g.TextGrid", GOLD))
    strict = compare_tiers("a", tier_of(write_tg, "p2.TextGrid", pred), tier_of(write_tg, "g2.TextGrid", GOLD), strip_stress=False)
    assert not ok.failed and strict.failed and "label sequences differ" in strict.reason
    ev = evaluate_boundaries([ok, strict, strict], 0.02)
    assert ev.n_files == 3 and ev.n_failed == 2 and ev.failure_rate == pytest.approx(2 / 3)


def test_evaluation_of_nothing_is_none_not_zero():
    ev = evaluate_boundaries([], 0.02)
    assert ev.failure_rate is None and ev.mae_s is None and ev.pct_within_tolerance is None
