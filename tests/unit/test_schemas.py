"""Schema/contract tests: JSON Schema stays in sync; invariants of the data contract."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from asa_m2.schemas import (
    CanonicalPhone,
    G2PResult,
    PronunciationVariant,
    SampleRecord,
    Span,
    TokenRecord,
    WordAlignment,
    sample_record_json_schema,
)

ROOT = Path(__file__).resolve().parents[2]


def test_checked_in_json_schema_is_current():
    expected = json.dumps(sample_record_json_schema(), indent=2, ensure_ascii=False) + "\n"
    for p in (ROOT / "docs/schema/sample_record.schema.json", ROOT / "examples/expected_output_schema.json"):
        assert p.read_text(encoding="utf-8") == expected, f"{p} is stale: run python scripts/export_schema.py"


def test_schema_documents_required_contract_fields():
    props = sample_record_json_schema()["properties"]
    for key in ("schema_version", "sample_id", "speaker_id", "reference_text", "human_verbatim", "tokens", "canonical_phones",
                "word_alignments", "phone_alignments", "alignment_metadata", "quality_flags", "provenance"):
        assert key in props
    defs = sample_record_json_schema()["$defs"]
    assert "realized_phone" in defs["PhoneAlignment"]["properties"]
    assert defs["AlignmentMetadata"]["properties"]["time_unit"]["const"] == "seconds"


def test_unknown_fields_are_rejected_by_the_contract():
    with pytest.raises(ValidationError):
        SampleRecord(sample_id="x", reference_text="a", gop_score=0.9)  # M4 data must not sneak into M2 output
    with pytest.raises(ValidationError):
        WordAlignment(label="a", label_role="word", start_time=0, end_time=1, source_tier="w", interval_index=0, pronunciation_score=1)


def test_no_m3_m4_concepts_exist_in_the_schema():
    text = json.dumps(sample_record_json_schema()).lower()
    for forbidden in ("gop", "mispronunciation", "posterior", "pronunciation_score", "mdd_confidence", "fluency"):
        assert forbidden not in text


def variant(**kw):
    base = dict(variant_id="v:1", source="s", raw_arpabet=["AY1"], base_phones=["AY"], stress=[1], phone_ids=[6], ipa_display=["aɪ"])
    return PronunciationVariant(**{**base, **kw})


def test_variant_invariants():
    variant()
    with pytest.raises(ValidationError, match="equal length"):
        variant(stress=[1, 0])
    with pytest.raises(ValidationError, match="at least one phone"):
        variant(raw_arpabet=[], base_phones=[], stress=[], phone_ids=[], ipa_display=[])


def token(**kw):
    base = dict(word_index=0, original_token="a", normalized_token="a", original_span=Span(start=0, end=1), token_kind="word")
    return TokenRecord(**{**base, **kw})


def test_token_invariants():
    token()  # unresolved default
    with pytest.raises(ValidationError, match="no variants"):
        token(resolution_status="unresolved", variants=[variant()])
    with pytest.raises(ValidationError, match="at least one variant"):
        token(resolution_status="resolved")
    with pytest.raises(ValidationError, match="does not refer"):
        token(resolution_status="resolved", variants=[variant()], selected_variant_id="nope")
    with pytest.raises(ValidationError, match="requires selected_variant_id"):
        token(resolution_status="resolved", variants=[variant()], selection_method="sole_variant")
    ok = token(resolution_status="resolved", variants=[variant()], selected_variant_id="v:1", selection_method="sole_variant",
               approved_for_alignment=True)
    assert ok.selected_variant().variant_id == "v:1"


def test_span_and_time_validation():
    with pytest.raises(ValidationError):
        Span(start=5, end=2)
    with pytest.raises(ValidationError):
        Span(start=-1, end=2)
    with pytest.raises(ValidationError, match="end_time must be"):
        WordAlignment(label="a", label_role="word", start_time=1, end_time=1, source_tier="w", interval_index=0)
    with pytest.raises(ValidationError):
        WordAlignment(label="a", label_role="word", start_time=-0.1, end_time=1, source_tier="w", interval_index=0)


def test_g2p_result_alignable_property(engine):
    assert engine.convert("the cat").alignable and not engine.convert("the blorptastic cat").alignable
    assert not engine.convert("").alignable


def test_g2p_result_roundtrips_json(engine):
    res = engine.convert("Dr. Smith has 21 books.", sample_id="x")
    assert G2PResult.model_validate_json(res.model_dump_json()) == res


def test_checked_in_example_g2p_output_validates():
    data = json.loads((ROOT / "examples/example_g2p_output.json").read_text(encoding="utf-8"))
    assert data["kind"] == "asa_m2_g2p_batch" and data["count"] == len(data["records"]) == 3
    for rec in data["records"]:
        G2PResult.model_validate(rec)


def test_canonical_phone_fields_carry_ids_and_stress():
    cp = CanonicalPhone(index=0, word_index=0, token="i", variant_id="cmudict:i:1", base_phone="AY", phone_id=6, stress=1,
                        ipa_display="aɪ", selection_method="sole_variant")
    assert (cp.base_phone, cp.phone_id, cp.stress) == ("AY", 6, 1)
