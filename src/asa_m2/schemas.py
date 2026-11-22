"""Validated data contract between M2 and M3/M4 (schema version 1.0).

Hard rules encoded in the models
--------------------------------
* ``PhoneAlignment.realized_phone`` is *always* ``None``: forced alignment aligns a
  supplied target sequence, it never identifies what the learner actually produced.
* ``reference_text`` (what the learner was expected to say) and ``human_verbatim``
  (what the learner actually said) are separate fields and never interchangeable;
  ``g2p_input_field`` records which one the pronunciations were derived from.
* Variants are never collapsed silently: ``selected_variant_id`` is ``None`` unless a
  documented ``selection_method`` produced a selection.
* Times are always seconds (``AlignmentMetadata.time_unit``).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from asa_m2 import SCHEMA_VERSION

Severity = Literal["info", "warning", "error"]
Scope = Literal["sample", "word", "phone", "audio", "corpus"]
LabelRoleName = Literal["word", "phone", "silence", "oov_marker", "unknown"]
SelectionMethod = Literal[
    "sole_variant",  # the pronunciation source offered exactly one valid variant
    "policy_first_listed",  # explicit config policy (g2p.variant_policy: first_listed)
    "manual_override",  # custom lexicon / user override named the variant
    "alignment_match",  # the aligner's chosen phones match exactly one candidate (NOT evidence of realisation)
    "unselected",  # several candidates remain; no selection made
    "none",  # no usable pronunciation
]
ResolutionStatus = Literal["resolved", "resolved_fallback", "unresolved"]
Granularity = Literal["phone", "word", "none"]
AlignStatus = Literal["success", "partial", "failed", "not_run"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Span(_Model):
    """Character offsets into the ORIGINAL text (end exclusive)."""

    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _order(self) -> Span:
        if self.end < self.start:
            raise ValueError("span end < start")
        return self


class QualityFlag(_Model):
    code: str
    severity: Severity
    scope: Scope
    message: str
    word_index: int | None = None
    phone_index: int | None = None
    start_time: float | None = None
    end_time: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class PronunciationVariant(_Model):
    variant_id: str
    source: str
    source_detail: str | None = None
    raw_arpabet: list[str]
    base_phones: list[str]
    stress: list[int | None]
    phone_ids: list[int]
    ipa_display: list[str]
    approximate_mapping: bool = False

    @model_validator(mode="after")
    def _lengths(self) -> PronunciationVariant:
        n = len(self.base_phones)
        if not (len(self.stress) == len(self.phone_ids) == len(self.ipa_display) == n):
            raise ValueError("base_phones, stress, phone_ids and ipa_display must have equal length")
        if n == 0:
            raise ValueError("a pronunciation variant needs at least one phone")
        return self


class TokenRecord(_Model):
    word_index: int = Field(ge=0)
    original_token: str
    normalized_token: str
    original_span: Span
    token_kind: str
    normalization_rules: list[str] = Field(default_factory=list)
    supported: bool = True
    variants: list[PronunciationVariant] = Field(default_factory=list)
    selected_variant_id: str | None = None
    selection_method: SelectionMethod = "none"
    pronunciation_source: str | None = None
    resolution_status: ResolutionStatus = "unresolved"
    resolution_reason: str | None = None
    approved_for_alignment: bool = False
    flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistency(self) -> TokenRecord:
        ids = {v.variant_id for v in self.variants}
        if self.selected_variant_id is not None and self.selected_variant_id not in ids:
            raise ValueError("selected_variant_id does not refer to a listed variant")
        if self.resolution_status == "unresolved" and (self.variants or self.approved_for_alignment):
            raise ValueError("unresolved tokens must have no variants and cannot be approved for alignment")
        if self.resolution_status != "unresolved" and not self.variants:
            raise ValueError("resolved tokens must list at least one variant")
        if self.selected_variant_id is None and self.selection_method in {
            "sole_variant", "policy_first_listed", "manual_override", "alignment_match",
        }:
            raise ValueError(f"selection_method {self.selection_method!r} requires selected_variant_id")
        return self

    def selected_variant(self) -> PronunciationVariant | None:
        for v in self.variants:
            if v.variant_id == self.selected_variant_id:
                return v
        return None


class CanonicalPhone(_Model):
    """One phone of the *resolved* canonical sequence (words with a selected variant only)."""

    index: int = Field(ge=0)
    word_index: int = Field(ge=0)
    token: str
    variant_id: str
    base_phone: str
    phone_id: int
    stress: int | None = None
    ipa_display: str
    selection_method: SelectionMethod


class WordAlignment(_Model):
    word_index: int | None = None  # index into tokens[]; None when not traceable to a token
    label: str
    label_role: LabelRoleName
    start_time: float = Field(ge=0)
    end_time: float
    source_tier: str
    interval_index: int = Field(ge=0)
    source_file: str | None = None

    @model_validator(mode="after")
    def _times(self) -> WordAlignment:
        if not self.end_time > self.start_time:
            raise ValueError("end_time must be > start_time")
        return self


class PhoneAlignment(_Model):
    """Aligned *target* phone interval.

    Three distinct notions are kept apart:

    * ``canonical_target_*``  - the phone the canonical pronunciation expects here;
    * ``aligned_*``           - the label the aligner placed on the interval;
    * ``realized_phone``      - what the learner actually produced. M2 cannot know this,
      so it is always ``None``.
    """

    phone_index: int | None = None
    interval_index: int = Field(ge=0)
    word_index: int | None = None
    aligned_label: str
    label_role: LabelRoleName
    aligned_base_phone: str | None = None
    aligned_stress: int | None = None
    aligned_phone_id: int | None = None
    canonical_phone_index: int | None = None
    canonical_target_phone: str | None = None
    canonical_target_stress: int | None = None
    start_time: float = Field(ge=0)
    end_time: float
    source_tier: str
    source_file: str | None = None
    realized_phone: None = None
    realized_phone_status: Literal["not_determined_by_forced_alignment"] = "not_determined_by_forced_alignment"

    @field_validator("realized_phone", mode="before")
    @classmethod
    def _never_realised(cls, value: object) -> None:
        if value is not None:
            raise ValueError(
                "M2 cannot determine realised phones: forced alignment only aligns target phones"
            )
        return None

    @model_validator(mode="after")
    def _times(self) -> PhoneAlignment:
        if not self.end_time > self.start_time:
            raise ValueError("end_time must be > start_time")
        return self


class AlignmentMetadata(_Model):
    aligner: str = "none"
    aligner_version: str | None = None
    granularity: Granularity = "none"
    status: AlignStatus = "not_run"
    time_unit: Literal["seconds"] = "seconds"
    model: dict[str, Any] = Field(default_factory=dict)
    command: list[str] | None = None
    notes: list[str] = Field(default_factory=list)
    qc_summary: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _coherent(self) -> AlignmentMetadata:
        if self.status == "not_run" and self.granularity != "none":
            raise ValueError("status 'not_run' requires granularity 'none'")
        return self


class NormalizationSummary(_Model):
    normalizer_version: str
    original_text: str
    normalized_text: str
    dropped_spans: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class Provenance(_Model):
    asa_m2_version: str
    schema_version: str = SCHEMA_VERSION
    created_at: str | None = None
    config_fingerprint: str | None = None
    inventory: dict[str, str] = Field(default_factory=dict)
    g2p_sources: dict[str, str | None] = Field(default_factory=dict)
    python_version: str | None = None
    platform: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)


class G2PResult(_Model):
    """Output of the G2P stage for one text (no audio, no alignment)."""

    schema_version: str = SCHEMA_VERSION
    sample_id: str | None = None
    text_field: Literal["reference_text", "human_verbatim"] = "reference_text"
    input_text: str
    normalization: NormalizationSummary
    tokens: list[TokenRecord]
    canonical_phones: list[CanonicalPhone]
    quality_flags: list[QualityFlag] = Field(default_factory=list)
    provenance: Provenance | None = None

    @property
    def unresolved_tokens(self) -> list[TokenRecord]:
        return [t for t in self.tokens if t.resolution_status == "unresolved"]

    @property
    def alignable(self) -> bool:
        """True when every token is approved for forced alignment."""
        return bool(self.tokens) and all(t.approved_for_alignment for t in self.tokens)


class SampleRecord(_Model):
    """Canonical per-recording output consumed by M3/M4."""

    schema_version: str = SCHEMA_VERSION
    sample_id: str
    speaker_id: str | None = None
    reference_text: str | None = None
    human_verbatim: str | None = None
    g2p_input_field: Literal["reference_text", "human_verbatim"] = "reference_text"
    normalization: NormalizationSummary | None = None
    tokens: list[TokenRecord] = Field(default_factory=list)
    canonical_phones: list[CanonicalPhone] = Field(default_factory=list)
    word_alignments: list[WordAlignment] = Field(default_factory=list)
    phone_alignments: list[PhoneAlignment] = Field(default_factory=list)
    alignment_metadata: AlignmentMetadata = Field(default_factory=AlignmentMetadata)
    quality_flags: list[QualityFlag] = Field(default_factory=list)
    provenance: Provenance | None = None

    @model_validator(mode="after")
    def _texts(self) -> SampleRecord:
        if self.reference_text is None and self.human_verbatim is None:
            raise ValueError("at least one of reference_text / human_verbatim is required")
        if getattr(self, self.g2p_input_field) is None:
            raise ValueError(f"g2p_input_field={self.g2p_input_field!r} refers to a missing field")
        if self.alignment_metadata.granularity == "word" and self.phone_alignments:
            raise ValueError("phone_alignments must be empty when granularity is 'word'")
        return self


def sample_record_json_schema() -> dict[str, Any]:
    """JSON Schema of :class:`SampleRecord` (checked in at docs/schema/)."""
    schema = SampleRecord.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "ASA M2 SampleRecord"
    return schema
