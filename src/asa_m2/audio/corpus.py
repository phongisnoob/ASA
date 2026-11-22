"""Corpus preparation for forced alignment (M2.4).

Input: a CSV manifest (see ``examples/input_manifest.csv``). Output layout::

    <out>/
      mfa_corpus/<speaker_folder>/<sample_id>.wav     normalised mono PCM audio (originals untouched)
      mfa_corpus/<speaker_folder>/<sample_id>.lab     transcript the aligner will read
      dictionary/asa_m2_corpus.dict                   pronunciation dictionary for exactly these words
      g2p/<sample_id>.g2p.json                        canonical G2P result per sample (feeds export)
      corpus_manifest.json                            reproducible record of everything done

Safety properties
-----------------
* originals are only ever *read*; output paths are checked against them;
* all file/folder names derive from validated identifiers (no path separators, no ``..``);
* unknown speakers get a unique pseudo-ID per recording (never a shared bucket) unless the
  user explicitly opts into ``speaker.missing_id_policy: shared``;
* PII-style manifest columns (names, student IDs ...) are rejected, not copied;
* unresolved (OOV) words block alignment unless explicitly handled.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from asa_m2 import SCHEMA_VERSION
from asa_m2.audio.loader import load_audio
from asa_m2.audio.resampler import prepare_audio, write_wav
from asa_m2.audio.validation import validate_audio
from asa_m2.config import AppConfig
from asa_m2.exceptions import CorpusError, OutputExistsError, UnsafePathError
from asa_m2.g2p.engine import G2PEngine
from asa_m2.phonemes.inventory import PhoneInventory
from asa_m2.schemas import G2PResult
from asa_m2.utils.paths import (
    ensure_not_source,
    ensure_within,
    is_within,
    sha256_file,
    validate_identifier,
)
from asa_m2.utils.serialization import dumps_json, write_text_atomic

log = logging.getLogger(__name__)

MANIFEST_NAME = "corpus_manifest.json"
REQUIRED_COLUMNS = ("sample_id", "audio_path")
TEXT_COLUMNS = ("reference_text", "reference_path", "human_verbatim")
OPTIONAL_COLUMNS = ("speaker_id",)
TextFieldChoice = Literal["auto", "reference_text", "human_verbatim"]


# ---------------------------------------------------------------------------------
# manifest models
# ---------------------------------------------------------------------------------
class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CorpusAudioRecord(_M):
    source_name: str
    source_relpath: str | None = None
    source_sha256: str
    source_sample_rate: int
    source_channels: int
    source_duration_s: float
    output_wav: str
    output_sha256: str
    output_sample_rate: int
    output_samples: int
    output_duration_s: float
    transformations: list[dict[str, Any]]


class CorpusSample(_M):
    sample_id: str
    speaker_id: str
    speaker_folder: str
    speaker_id_source: Literal["manifest", "pseudo_unique_per_recording", "shared_policy"]
    g2p_input_field: Literal["reference_text", "human_verbatim"]
    normalized_text: str
    num_tokens: int
    transcript_file: str
    g2p_file: str
    audio: CorpusAudioRecord
    warnings: list[str] = Field(default_factory=list)


class ExcludedSample(_M):
    sample_id: str | None
    line: int
    codes: list[str]
    messages: list[str]


class CorpusManifest(_M):
    schema_version: str = SCHEMA_VERSION
    kind: Literal["asa_m2_corpus_manifest"] = "asa_m2_corpus_manifest"
    created_at: str | None = None
    config_fingerprint: str
    inventory: dict[str, str]
    target_sample_rate: int
    transcript_extension: str
    speaker_policy: dict[str, Any]
    dictionary: dict[str, Any]
    samples: list[CorpusSample]
    excluded: list[ExcludedSample] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@dataclass
class ManifestRow:
    line: int
    sample_id: str
    audio_path: Path
    reference_text: str | None
    reference_path: Path | None
    human_verbatim: str | None
    speaker_id: str | None


@dataclass(frozen=True)
class RowIssue:
    line: int
    sample_id: str | None
    code: str
    message: str


# ---------------------------------------------------------------------------------
# reading the input manifest
# ---------------------------------------------------------------------------------
def pseudo_speaker_id(sample_id: str, prefix: str = "SPK_UNK") -> str:
    """Deterministic, anonymous, unique-per-recording speaker ID."""
    digest = hashlib.sha256(f"asa_m2:{sample_id}".encode()).hexdigest()[:10]
    return f"{prefix}_{digest}"


def _cell_getter(rec: dict[str, str | None]):
    """Return ``get(column) -> stripped value or None`` bound to one CSV record."""

    def get(key: str) -> str | None:
        return (rec.get(key) or "").strip() or None

    return get


def read_manifest(path: Path, config: AppConfig) -> tuple[list[ManifestRow], list[RowIssue], list[str]]:
    """Parse the CSV manifest. Returns ``(rows, row_issues, warnings)``.

    Raises :class:`CorpusError` for file-level problems (missing file, missing required
    columns, PII-style columns).
    """
    if not path.is_file():
        raise CorpusError(f"manifest not found: {path}")
    base = path.parent
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = [h.strip() for h in (reader.fieldnames or [])]
        if not header:
            raise CorpusError(f"manifest {path.name} is empty or has no header row")
        reader.fieldnames = header
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            raise CorpusError(f"manifest is missing required column(s): {missing}; found {header}")
        if not any(c in header for c in TEXT_COLUMNS):
            raise CorpusError(f"manifest needs at least one text column among {list(TEXT_COLUMNS)}")
        blocked = sorted({h for h in header if h.lower() in {b.lower() for b in config.corpus.blocked_manifest_columns}})
        if blocked:
            raise CorpusError(
                f"manifest contains personally-identifying column(s) {blocked}. M2 neither needs nor accepts "
                "names, student IDs, addresses or similar; remove them (use an anonymised speaker_id instead)."
            )
        warnings = []
        extra = [h for h in header if h not in REQUIRED_COLUMNS + TEXT_COLUMNS + OPTIONAL_COLUMNS]
        if extra:
            warnings.append(f"ignored manifest column(s) not used by M2: {extra}")
        rows: list[ManifestRow] = []
        issues: list[RowIssue] = []
        seen: dict[str, int] = {}
        for rec in reader:
            line = reader.line_num
            get = _cell_getter(rec)
            sid = get("sample_id")
            if sid is None:
                issues.append(RowIssue(line, None, "missing_sample_id", "sample_id is empty"))
                continue
            try:
                validate_identifier(sid, kind="sample_id")
            except UnsafePathError as exc:
                issues.append(RowIssue(line, sid, "unsafe_sample_id", exc.message))
                continue
            if sid in seen:
                issues.append(RowIssue(line, sid, "duplicate_sample_id",
                                       f"sample_id {sid!r} already used on line {seen[sid]}; both rows are ambiguous"))
                continue
            seen[sid] = line
            audio = get("audio_path")
            if audio is None:
                issues.append(RowIssue(line, sid, "missing_audio_path", "audio_path is empty"))
                continue
            ap = Path(audio)
            ap = ap if ap.is_absolute() else base / ap
            ref_p = get("reference_path")
            rows.append(ManifestRow(
                line=line, sample_id=sid, audio_path=ap, reference_text=get("reference_text"),
                reference_path=(Path(ref_p) if ref_p and Path(ref_p).is_absolute() else base / ref_p) if ref_p else None,
                human_verbatim=get("human_verbatim"), speaker_id=get("speaker_id"),
            ))
        dup_ids = {i.sample_id for i in issues if i.code == "duplicate_sample_id"}
        if dup_ids:  # the first occurrence is just as ambiguous as the later ones
            for r in [r for r in rows if r.sample_id in dup_ids]:
                issues.append(RowIssue(r.line, r.sample_id, "duplicate_sample_id", f"sample_id {r.sample_id!r} is duplicated"))
            rows = [r for r in rows if r.sample_id not in dup_ids]
    if not rows and not issues:
        raise CorpusError(f"manifest {path.name} contains no data rows")
    return rows, issues, warnings


# ---------------------------------------------------------------------------------
# preparation
# ---------------------------------------------------------------------------------
def _choose_text(row: ManifestRow, choice: TextFieldChoice) -> tuple[str | None, str | None, str | None]:
    """Return ``(field_used, text, problem)``. ``reference_text`` wins under ``auto``."""
    reference = row.reference_text
    if reference is None and row.reference_path is not None:
        try:
            reference = row.reference_path.read_text(encoding="utf-8").strip() or None
        except OSError as exc:
            return None, None, f"cannot read reference_path {row.reference_path.name}: {exc}"
    if choice == "human_verbatim" or (choice == "auto" and reference is None):
        return ("human_verbatim", row.human_verbatim, None) if row.human_verbatim else (None, None, "no human_verbatim text")
    if reference is None:
        return None, None, "no reference_text for text-dependent alignment"
    return "reference_text", reference, None


def _dictionary_lines(results: list[G2PResult], config: AppConfig) -> list[str]:
    """One ``word<TAB>phones`` line per exported variant, deterministic order."""
    entries: dict[str, list[list[str]]] = {}
    for res in results:
        for tok in res.tokens:
            if not tok.approved_for_alignment:
                continue
            chosen = ([tok.selected_variant()] if not config.g2p.allow_multiple_variants else tok.variants)
            for v in chosen:
                if v is None:
                    raise CorpusError(f"word {tok.normalized_token!r} has no selected variant but variants are restricted to one")
                phones = list(v.raw_arpabet) if config.phonemes.preserve_stress else list(v.base_phones)
                bucket = entries.setdefault(tok.normalized_token, [])
                if phones not in bucket:
                    bucket.append(phones)
    return [f"{w}\t{' '.join(p)}" for w in sorted(entries) for p in entries[w]]


def _clean_owned(out: Path, config: AppConfig, sources: list[Path]) -> None:
    c = config.corpus
    owned = [out / c.mfa_corpus_dirname, out / c.dictionary_dirname, out / "g2p", out / MANIFEST_NAME]
    for path in owned:
        for src in sources:
            if is_within(path, src):
                raise UnsafePathError(f"source audio {src} lives inside {path}, which would be deleted; choose another output directory")
    for path in owned:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def prepare_corpus(
    manifest_path: str | Path,
    output_dir: str | Path,
    config: AppConfig,
    engine: G2PEngine,
    *,
    overwrite: bool = False,
    text_field: TextFieldChoice = "auto",
) -> CorpusManifest:
    """Prepare an MFA-compatible corpus (see module docstring). Returns the written manifest."""
    # resolve once: path-safety checks compare resolved paths, and relative --output values are common
    manifest_path, out = Path(manifest_path).resolve(), Path(output_dir).resolve()
    rows, issues, warnings = read_manifest(manifest_path, config)
    inventory = engine.inventory
    excluded: list[ExcludedSample] = []
    soft_excluded: list[ExcludedSample] = []  # unresolved-OOV samples (only when not fatal); always recorded
    ready: list[tuple[ManifestRow, str, str, G2PResult, list[str]]] = []
    row_problems: dict[int, list[tuple[str, str]]] = {}
    for i in issues:
        row_problems.setdefault(i.line, []).append((i.code, i.message))
    sample_for_line: dict[int, str | None] = {i.line: i.sample_id for i in issues}

    for row in rows:
        problems: list[tuple[str, str]] = []
        report = validate_audio(row.audio_path, config.audio)
        for iss in report.errors:
            problems.append((f"audio_{iss.code}", iss.message))
        used, text, problem = _choose_text(row, text_field)
        if problem:
            problems.append(("missing_text", problem))
        res: G2PResult | None = None
        if text is not None and used is not None:
            res = engine.convert(text, sample_id=row.sample_id, text_field=used)
            if not res.tokens:
                problems.append(("empty_text", "text produced no tokens"))
            elif res.unresolved_tokens:
                words = [f"{t.original_token!r} ({t.resolution_reason})" for t in res.unresolved_tokens]
                if config.g2p.fail_on_unresolved_oov:
                    raise CorpusError(
                        f"sample {row.sample_id!r}: {len(words)} unresolved word(s) cannot enter forced alignment: "
                        + ", ".join(words)
                        + ". Add them to a custom lexicon (g2p.custom_lexicon_path), enable the eSpeak fallback "
                        "(g2p.fallback_backend: espeak), or set g2p.fail_on_unresolved_oov: false to exclude such samples.",
                        details={"sample_id": row.sample_id, "words": [t.original_token for t in res.unresolved_tokens]},
                    )
                soft_excluded.append(ExcludedSample(sample_id=row.sample_id, line=row.line, codes=["unresolved_words"],
                                                    messages=["; ".join(words)]))
                continue  # excluded from alignment explicitly; other problems of this row are moot
        if row.speaker_id is not None:
            try:
                validate_identifier(row.speaker_id, kind="speaker_id")
            except UnsafePathError as exc:
                problems.append(("unsafe_speaker_id", exc.message))
        elif config.speaker.missing_id_policy == "error":
            problems.append(("missing_speaker_id", "speaker_id is empty and speaker.missing_id_policy is 'error'"))
        if problems:
            row_problems.setdefault(row.line, []).extend(problems)
            sample_for_line[row.line] = row.sample_id
        elif res is not None and used is not None:
            ready.append((row, used, text or "", res, [w.message for w in report.warnings]))

    if row_problems:
        if config.corpus.on_error == "fail":
            lines = [f"line {ln} ({sample_for_line.get(ln)}): [{code}] {msg}"
                     for ln, ps in sorted(row_problems.items()) for code, msg in ps]
            raise CorpusError(
                f"{len(lines)} problem(s) in the manifest; nothing was written (set corpus.on_error: skip to exclude "
                "bad rows instead):\n  - " + "\n  - ".join(lines[:50]) + ("\n  - ..." if len(lines) > 50 else ""),
                details={"problems": lines},
            )
        for ln, ps in sorted(row_problems.items()):
            excluded.append(ExcludedSample(sample_id=sample_for_line.get(ln), line=ln,
                                           codes=[c for c, _ in ps], messages=[m for _, m in ps]))
    excluded.extend(soft_excluded)
    if not ready:
        raise CorpusError("no usable samples remain after validation; nothing to prepare")

    # ---- output directory --------------------------------------------------------
    c = config.corpus
    sources = [r.audio_path for r, *_ in ready]
    owned_exists = any((out / n).exists() for n in (c.mfa_corpus_dirname, c.dictionary_dirname, "g2p", MANIFEST_NAME))
    if owned_exists:
        if not overwrite:
            raise OutputExistsError(f"{out} already contains a prepared corpus; pass --overwrite to replace it "
                                    "(original recordings are never touched)", details={"path": str(out)})
        _clean_owned(out, config, sources)
    corpus_root = out / c.mfa_corpus_dirname
    corpus_root.mkdir(parents=True, exist_ok=True)

    samples: list[CorpusSample] = []
    results: list[G2PResult] = []
    for row, used, _text, res, audio_warnings in ready:
        if row.speaker_id is not None:
            speaker, source = row.speaker_id, "manifest"
        elif config.speaker.missing_id_policy == "shared":
            speaker, source = str(config.speaker.shared_speaker_id), "shared_policy"
        else:
            speaker, source = pseudo_speaker_id(row.sample_id, config.speaker.pseudo_id_prefix), "pseudo_unique_per_recording"
        folder = row.sample_id if config.speaker.grouping == "per_recording" else speaker
        validate_identifier(folder, kind="speaker folder")
        wav_path = ensure_within(corpus_root, corpus_root / folder / f"{row.sample_id}.wav")
        lab_path = ensure_within(corpus_root, corpus_root / folder / f"{row.sample_id}{c.transcript_extension}")
        ensure_not_source(wav_path, sources)
        ensure_not_source(lab_path, sources)

        data = load_audio(row.audio_path)
        prepared = prepare_audio(data, config.audio)
        write_wav(wav_path, prepared.samples, prepared.sample_rate, config.audio.subtype, overwrite=False)
        normalized = res.normalization.normalized_text
        write_text_atomic(lab_path, normalized + "\n", overwrite=False)
        g2p_path = out / "g2p" / f"{row.sample_id}.g2p.json"
        write_text_atomic(g2p_path, dumps_json(res.model_dump(mode="json"), indent=config.output.json_indent), overwrite=False)
        try:
            relpath = str(row.audio_path.resolve().relative_to(manifest_path.parent.resolve()))
        except ValueError:
            relpath = None
        samples.append(CorpusSample(
            sample_id=row.sample_id, speaker_id=speaker, speaker_folder=folder, speaker_id_source=source,
            g2p_input_field=used,
            normalized_text=normalized, num_tokens=len(res.tokens),
            transcript_file=str(lab_path.relative_to(out).as_posix()), g2p_file=str(g2p_path.relative_to(out).as_posix()),
            audio=CorpusAudioRecord(
                source_name=row.audio_path.name, source_relpath=relpath, source_sha256=sha256_file(row.audio_path),
                source_sample_rate=data.info.sample_rate, source_channels=data.info.channels,
                source_duration_s=round(data.info.duration_s, 6), output_wav=str(wav_path.relative_to(out).as_posix()),
                output_sha256=sha256_file(wav_path), output_sample_rate=prepared.sample_rate,
                output_samples=int(prepared.samples.shape[0]),
                output_duration_s=round(prepared.samples.shape[0] / prepared.sample_rate, 6),
                transformations=prepared.transformations,
            ),
            warnings=audio_warnings,
        ))
        results.append(res)

    lines = _dictionary_lines(results, config)
    dict_dir = out / c.dictionary_dirname
    dict_path = dict_dir / c.dictionary_filename
    write_text_atomic(dict_path, "\n".join(lines) + "\n", overwrite=False)

    # speaker-grouping risk notes
    folders: dict[str, set[str]] = {}
    for s in samples:
        folders.setdefault(s.speaker_folder, set()).add(s.speaker_id)
    if config.speaker.missing_id_policy == "shared" and any(s.speaker_id_source == "shared_policy" for s in samples):
        warnings.append("speaker.missing_id_policy is 'shared': unrelated recordings share one speaker folder, which "
                        "makes MFA speaker adaptation treat them as one voice (see docs/research_limitations.md)")
    manifest = CorpusManifest(
        created_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat() if config.output.include_timestamps else None,
        config_fingerprint=config.fingerprint(),
        inventory={"id": inventory.inventory_id, "version": inventory.inventory_version},
        target_sample_rate=config.audio.target_sample_rate,
        transcript_extension=c.transcript_extension,
        speaker_policy={"missing_id_policy": config.speaker.missing_id_policy, "grouping": config.speaker.grouping,
                        "speakers": len({s.speaker_id for s in samples}), "pseudo_speakers": sum(
                            s.speaker_id_source == "pseudo_unique_per_recording" for s in samples)},
        dictionary={"path": dict_path.relative_to(out).as_posix(), "entries": len(lines),
                    "words": len({ln.split("\t", 1)[0] for ln in lines}),
                    "phone_style": "arpabet_with_stress" if config.phonemes.preserve_stress else "arpabet_base_phones",
                    "variants": "all" if config.g2p.allow_multiple_variants else "selected_only",
                    "sha256": sha256_file(dict_path)},
        samples=samples, excluded=excluded, warnings=warnings,
    )
    write_text_atomic(out / MANIFEST_NAME, dumps_json(manifest.model_dump(mode="json"), indent=config.output.json_indent), overwrite=False)
    log.info("corpus prepared", extra={"samples": len(samples), "excluded": len(excluded), "output": str(out)})
    return manifest


def load_corpus_manifest(corpus_dir: str | Path) -> CorpusManifest:
    path = Path(corpus_dir) / MANIFEST_NAME
    if not path.is_file():
        raise CorpusError(f"{MANIFEST_NAME} not found in {corpus_dir}; run `asa-m2 prepare-corpus` first")
    try:
        return CorpusManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise CorpusError(f"{path} is not a valid corpus manifest: {exc}") from exc


# ---------------------------------------------------------------------------------
# validation of a prepared corpus
# ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class CorpusIssue:
    code: str
    severity: str  # "error" | "warning" | "info"
    message: str
    sample_id: str | None = None


@dataclass
class CorpusValidationReport:
    corpus_dir: Path
    issues: list[CorpusIssue] = field(default_factory=list)
    n_samples: int = 0

    @property
    def errors(self) -> list[CorpusIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[CorpusIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_corpus(
    corpus_dir: str | Path,
    config: AppConfig,
    inventory: PhoneInventory,
    *,
    acoustic_phones: list[str] | None = None,
) -> CorpusValidationReport:
    """Validate a prepared corpus *before* alignment (files, integrity, dictionary, speakers)."""
    root = Path(corpus_dir)
    report = CorpusValidationReport(root)

    def add(code: str, severity: str, message: str, sample_id: str | None = None) -> None:
        report.issues.append(CorpusIssue(code, severity, message, sample_id))

    try:
        manifest = load_corpus_manifest(root)
    except CorpusError as exc:
        add("manifest_invalid", "error", exc.message)
        return report
    report.n_samples = len(manifest.samples)
    if manifest.schema_version != SCHEMA_VERSION:
        add("schema_version_mismatch", "error", f"manifest schema {manifest.schema_version} != supported {SCHEMA_VERSION}")
    if manifest.inventory.get("id") != inventory.inventory_id or manifest.inventory.get("version") != inventory.inventory_version:
        add("inventory_mismatch", "error",
            f"corpus was prepared with inventory {manifest.inventory}, but {inventory.inventory_id} "
            f"{inventory.inventory_version} is loaded")

    from asa_m2.audio.loader import probe_audio
    from asa_m2.exceptions import AudioError

    ids: set[str] = set()
    folders: dict[str, set[str]] = {}
    lab_words: set[str] = set()
    known_files: set[Path] = set()
    for s in manifest.samples:
        if s.sample_id in ids:
            add("duplicate_sample_id", "error", "sample_id appears twice in the corpus manifest", s.sample_id)
        ids.add(s.sample_id)
        folders.setdefault(s.speaker_folder, set()).add(s.speaker_id)
        wav, lab, g2p = root / s.audio.output_wav, root / s.transcript_file, root / s.g2p_file
        known_files.update({wav.resolve(), lab.resolve()})
        for path, what in ((wav, "audio"), (lab, "transcript"), (g2p, "G2P result")):
            if not path.is_file():
                add(f"missing_{what.split()[0].lower()}", "error", f"{what} file {path.relative_to(root)} is missing", s.sample_id)
        if wav.is_file():
            try:
                info = probe_audio(wav)
            except AudioError as exc:
                add("audio_unreadable", "error", exc.message, s.sample_id)
            else:
                if info.sample_rate != manifest.target_sample_rate:
                    add("sample_rate_mismatch", "error", f"{info.sample_rate} Hz != corpus target {manifest.target_sample_rate} Hz", s.sample_id)
                if info.channels != 1:
                    add("not_mono", "error", f"{info.channels} channels; aligner input must be mono", s.sample_id)
                if info.frames != s.audio.output_samples:
                    add("length_mismatch", "error", f"{info.frames} frames on disk vs {s.audio.output_samples} recorded", s.sample_id)
            if sha256_file(wav) != s.audio.output_sha256:
                add("audio_modified", "error", "audio checksum differs from the manifest (file changed after preparation)", s.sample_id)
        if lab.is_file():
            text = lab.read_text(encoding="utf-8").strip()
            if text != s.normalized_text:
                add("transcript_mismatch", "error", "transcript file differs from the manifest's normalized_text", s.sample_id)
            if not text:
                add("empty_transcript", "error", "transcript is empty", s.sample_id)
            lab_words.update(text.split())
        if config.speaker.grouping == "per_recording" and s.speaker_folder != s.sample_id:
            add("speaker_folder_mismatch", "error", "per_recording grouping but the folder is not the sample_id", s.sample_id)
    for folder, spk in folders.items():
        if len(spk) > 1:
            add("speaker_mixing", "error", f"folder {folder!r} mixes speaker IDs {sorted(spk)}; MFA treats a folder as one speaker")
    corpus_root = root / config.corpus.mfa_corpus_dirname
    if corpus_root.is_dir():
        for p in corpus_root.rglob("*"):
            if p.is_file() and p.resolve() not in known_files:
                add("orphan_file", "warning", f"file {p.relative_to(root)} is not listed in the corpus manifest")
    else:
        add("missing_corpus_dir", "error", f"{config.corpus.mfa_corpus_dirname}/ not found")

    # dictionary
    dict_path = root / str(manifest.dictionary.get("path", ""))
    if not dict_path.is_file():
        add("missing_dictionary", "error", "pronunciation dictionary is missing")
    else:
        if sha256_file(dict_path) != manifest.dictionary.get("sha256"):
            add("dictionary_modified", "warning", "dictionary checksum differs from the manifest (edited after preparation)")
        words: set[str] = set()
        phones: set[str] = set()
        for n, line in enumerate(dict_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 2 or not parts[1].strip():
                add("dictionary_malformed", "error", f"dictionary line {n} is not 'word<TAB>phones': {line[:60]!r}")
                continue
            words.add(parts[0])
            phones.update(parts[1].split())
        missing = sorted(lab_words - words)
        if missing:
            add("word_missing_from_dictionary", "error",
                f"{len(missing)} transcript word(s) have no dictionary entry (the aligner would treat them as OOV): {missing[:10]}")
        from asa_m2.exceptions import InventoryError
        for ph in sorted(phones):
            try:
                inventory.parse_symbol(ph)
            except InventoryError as exc:
                add("dictionary_phone_not_in_inventory", "error", f"dictionary phone {ph!r}: {exc.message}")
        if acoustic_phones is not None:
            absent = sorted(phones - set(acoustic_phones))
            if absent:
                add("dictionary_phone_not_in_acoustic_model", "error",
                    f"{len(absent)} dictionary phone(s) are not in the supplied acoustic phone set: {absent[:15]}. "
                    "The dictionary phone style (stress digits) must match the acoustic model.")
            rep = inventory.check_compatibility(acoustic_phones, other_name="acoustic model")
            if rep.extra_in_other:
                add("acoustic_model_extra_phones", "info",
                    f"acoustic model has phone(s) outside the canonical inventory (e.g. special symbols): {list(rep.extra_in_other)[:10]}")
            if rep.missing_in_other:
                add("acoustic_model_missing_phones", "warning",
                    f"canonical phones absent from the acoustic model: {list(rep.missing_in_other)}")
    if manifest.speaker_policy.get("pseudo_speakers"):
        add("pseudo_speakers", "info",
            f"{manifest.speaker_policy['pseudo_speakers']} recording(s) have no speaker ID and were given unique pseudo IDs; "
            "speaker adaptation cannot pool them (see docs/research_limitations.md)")
    return report
