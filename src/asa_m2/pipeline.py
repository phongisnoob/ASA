"""End-to-end orchestration: validated corpus -> MFA -> TextGrids -> canonical JSON records."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from asa_m2.alignment.exporter import record_from_textgrid
from asa_m2.alignment.mfa_adapter import MFAAdapter, MFARunResult
from asa_m2.alignment.textgrid_parser import parse_textgrid
from asa_m2.audio.corpus import MANIFEST_NAME, load_corpus_manifest, validate_corpus
from asa_m2.config import AppConfig
from asa_m2.exceptions import CorpusError, OutputExistsError, TextGridError
from asa_m2.phonemes.inventory import PhoneInventory
from asa_m2.phonemes.mapping import AlignerLabelMapper
from asa_m2.provenance import build_provenance
from asa_m2.schemas import AlignmentMetadata, G2PResult, QualityFlag, SampleRecord
from asa_m2.utils.serialization import dumps_json, write_text_atomic

log = logging.getLogger(__name__)


@dataclass
class SampleOutcome:
    sample_id: str
    status: str
    json_path: Path | None
    textgrid: Path | None
    error_flags: int = 0
    warning_flags: int = 0
    qc_summary: dict[str, object] = field(default_factory=dict)


@dataclass
class PipelineSummary:
    output_dir: Path
    mfa: MFARunResult | None
    outcomes: list[SampleOutcome]

    @property
    def n_success(self) -> int:
        return sum(o.status == "success" for o in self.outcomes)

    @property
    def has_failures(self) -> bool:
        return any(o.status != "success" for o in self.outcomes)


def align_corpus_with_mfa(
    corpus_dir: str | Path,
    output_dir: str | Path,
    config: AppConfig,
    inventory: PhoneInventory,
    mapper: AlignerLabelMapper,
    adapter: MFAAdapter,
    *,
    overwrite: bool = False,
    run_mfa_validate: bool = False,
) -> PipelineSummary:
    """Validate the corpus, run MFA, and export one verified JSON record per sample.

    A sample only reports ``status == "success"`` when a TextGrid exists, parses, and
    produced no error-level quality flag. Failed or partial samples are written too
    (with explicit flags) so nothing is silently dropped.
    """
    corpus, out = Path(corpus_dir).resolve(), Path(output_dir).resolve()  # absolute paths for MFA and for logs
    manifest = load_corpus_manifest(corpus)
    report = validate_corpus(corpus, config, inventory)
    if not report.ok:
        raise CorpusError("corpus failed validation; refusing to align:\n  - "
                          + "\n  - ".join(f"[{i.code}] {i.message}" for i in report.errors[:20]),
                          details={"codes": [i.code for i in report.errors]})
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise OutputExistsError(f"alignment output {out} is not empty; pass --overwrite to reuse it", details={"path": str(out)})
    tg_dir, json_dir, log_dir = out / "textgrids", out / "json", out / "logs"
    # nothing is created before MFA's preconditions (executable, version) pass: a failed precondition must
    # leave no half-made output directory behind (json/ is created lazily when the first record is written)

    mcfg = config.alignment.mfa
    dictionary = str(corpus / str(manifest.dictionary["path"])) if mcfg.dictionary == "generated" else mcfg.dictionary
    if run_mfa_validate or mcfg.run_mfa_validate:
        adapter.validate_corpus(corpus / config.corpus.mfa_corpus_dirname, dictionary=dictionary, log_dir=log_dir)

    ids = [s.sample_id for s in manifest.samples]
    result = adapter.align(
        corpus / config.corpus.mfa_corpus_dirname, tg_dir, expected_samples=ids, dictionary=dictionary,
        overwrite=overwrite, log_dir=log_dir, raise_on_missing=False,
    )
    public_cmd = public_command(result.command, {
        str(corpus / config.corpus.mfa_corpus_dirname): "<corpus>", dictionary: "<dictionary>", str(tg_dir): "<output>"})
    model_info = {
        "mode": mcfg.mode, "dictionary": mcfg.dictionary if mcfg.dictionary != "generated" else "generated:" + str(manifest.dictionary["path"]),
        "dictionary_sha256": manifest.dictionary.get("sha256") if mcfg.dictionary == "generated" else None,
        "acoustic_model": Path(mcfg.acoustic_model).name if mcfg.acoustic_model and os.path.isabs(mcfg.acoustic_model) else mcfg.acoustic_model,
        "hf_model_id": mcfg.hf_model_id, "output_format": mcfg.output_format,
    }
    outcomes: list[SampleOutcome] = []
    for s in manifest.samples:
        g2p = G2PResult.model_validate_json((corpus / s.g2p_file).read_text(encoding="utf-8"))
        meta = AlignmentMetadata(aligner="MFA", aligner_version=result.version, granularity="phone", status="success",
                                 model=model_info, command=public_cmd, notes=list(result.warnings))
        common = dict(
            sample_id=s.sample_id, speaker_id=s.speaker_id,
            reference_text=g2p.input_text if s.g2p_input_field == "reference_text" else None,
            human_verbatim=g2p.input_text if s.g2p_input_field == "human_verbatim" else None,
        )
        inputs = {"audio": {"name": Path(s.audio.output_wav).name, "sha256": s.audio.output_sha256,
                            "sample_rate": s.audio.output_sample_rate}}
        tg_path = result.textgrids.get(s.sample_id)
        record: SampleRecord
        if tg_path is None:
            record = _failure_record(g2p, config, inventory, meta, common, inputs,
                                     "no_textgrid_produced", f"MFA produced no TextGrid for {s.sample_id}")
        else:
            try:
                record = record_from_textgrid(
                    g2p=g2p, textgrid=parse_textgrid(tg_path), config=config, inventory=inventory, mapper=mapper,
                    alignment_metadata=meta, audio_duration_s=s.audio.output_duration_s, inputs=inputs, **common,  # type: ignore[arg-type]
                )
            except TextGridError as exc:
                record = _failure_record(g2p, config, inventory, meta, common, inputs, "textgrid_unusable", exc.message)
        json_path = json_dir / f"{s.sample_id}.json"
        write_text_atomic(json_path, dumps_json(record.model_dump(mode="json"), indent=config.output.json_indent), overwrite=overwrite)
        outcomes.append(SampleOutcome(
            s.sample_id, record.alignment_metadata.status, json_path, tg_path,
            error_flags=sum(f.severity == "error" for f in record.quality_flags),
            warning_flags=sum(f.severity == "warning" for f in record.quality_flags),
            qc_summary=record.alignment_metadata.qc_summary,
        ))
    summary = PipelineSummary(out, result, outcomes)
    write_text_atomic(out / "alignment_summary.json", dumps_json({
        "schema_version": "1.0", "kind": "asa_m2_alignment_summary",
        "mfa_version": result.version, "command": public_cmd, "mfa_status": result.status,
        "n_samples": len(outcomes), "n_success": summary.n_success,
        "samples": [{"sample_id": o.sample_id, "status": o.status, "json": o.json_path.name if o.json_path else None,
                     "error_flags": o.error_flags, "warning_flags": o.warning_flags, "qc_summary": o.qc_summary}
                    for o in outcomes],
        "note": "status 'success' means: TextGrid present and parsed, no error-level quality flag. "
                "It does NOT mean the learner pronounced the target phones correctly.",
    }, indent=config.output.json_indent), overwrite=overwrite)
    return summary


def public_command(command: Sequence[str], replacements: Mapping[str, str]) -> list[str]:
    """Command line for *shared* output records: no absolute (possibly user-identifying) paths.

    Known directories become placeholders (``<corpus>``, ``<dictionary>``, ``<output>``); any other absolute
    path is reduced to ``<abs>/<name>``. The exact command is preserved separately in ``logs/*.command.json``.
    """
    out: list[str] = []
    for i, arg in enumerate(command):
        if arg in replacements:
            out.append(replacements[arg])
        elif i == 0:
            out.append(Path(arg).name)  # executable
        elif os.path.isabs(arg):
            out.append(f"<abs>/{Path(arg).name}")
        else:
            out.append(arg)
    return out


def _failure_record(
    g2p: G2PResult, config: AppConfig, inventory: PhoneInventory, meta: AlignmentMetadata,
    common: Mapping[str, object], inputs: Mapping[str, object], code: str, message: str,
) -> SampleRecord:
    return SampleRecord(
        normalization=g2p.normalization, tokens=g2p.tokens, canonical_phones=g2p.canonical_phones,
        g2p_input_field=g2p.text_field,
        alignment_metadata=meta.model_copy(update={"status": "failed", "granularity": "none"}),
        quality_flags=list(g2p.quality_flags) + [QualityFlag(code=code, severity="error", scope="sample", message=message)],
        provenance=build_provenance(config, inventory, g2p_sources=g2p.provenance.g2p_sources if g2p.provenance else {}, inputs=inputs),
        **common,
    )


__all__ = ["MANIFEST_NAME", "PipelineSummary", "SampleOutcome", "align_corpus_with_mfa"]
