"""Forced-alignment adapters, TextGrid parsing/validation, QC and export."""

from asa_m2.alignment.exporter import (
    IntervalIn,
    assemble_sample_record,
    record_from_textgrid,
    write_word_textgrid,
)
from asa_m2.alignment.mfa_adapter import MFAAdapter, MFARunResult
from asa_m2.alignment.quality import (
    BoundaryEvaluation,
    FileBoundaryResult,
    compare_tiers,
    evaluate_boundaries,
)
from asa_m2.alignment.textgrid_parser import (
    ParsedInterval,
    ParsedTextGrid,
    ParsedTier,
    TextGridIssue,
    parse_textgrid,
    select_tier,
)
from asa_m2.alignment.validator import has_errors, interval_role, validate_textgrid

__all__ = [
    "BoundaryEvaluation", "FileBoundaryResult", "IntervalIn", "MFAAdapter", "MFARunResult",
    "ParsedInterval", "ParsedTextGrid", "ParsedTier", "TextGridIssue", "assemble_sample_record",
    "compare_tiers", "evaluate_boundaries", "has_errors", "interval_role", "parse_textgrid",
    "record_from_textgrid", "select_tier", "validate_textgrid", "write_word_textgrid",
]
