"""``asa-m2`` command line interface.

Results go to stdout; logs go to stderr. Exit codes are documented in
``asa_m2.exceptions`` (0 ok, 1 internal, 2 usage, 3 config, 4 data, 5 external tool,
6 alignment, 7 optional dependency).
"""

from __future__ import annotations

import argparse
import csv
import logging
import platform
import sys
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

from asa_m2 import SCHEMA_VERSION, __version__
from asa_m2.config import AppConfig, load_config
from asa_m2.exceptions import (
    EXIT_CONFIG,
    EXIT_DATA,
    EXIT_EXTERNAL,
    EXIT_INTERNAL,
    EXIT_OK,
    ConfigError,
    M2Error,
    OutputExistsError,
)
from asa_m2.logging_config import setup_logging
from asa_m2.phonemes.inventory import PhoneInventory, phone_set_from_lines
from asa_m2.phonemes.mapping import AlignerLabelMapper
from asa_m2.utils.serialization import dumps_json, write_text_atomic

log = logging.getLogger("asa_m2.cli")


# ---------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------
def _common_parent() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", action="append", metavar="YAML",
                   help="YAML config (repeatable; later files override). Default: $ASA_M2_CONFIG or packaged default.yaml")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="log verbosity (default INFO)")
    p.add_argument("--log-file", metavar="PATH", help="also write JSON-lines logs to this file")
    p.add_argument("--json-logs", action="store_true", help="emit logs on stderr as JSON lines")
    return p


def _load_all(args: argparse.Namespace) -> tuple[AppConfig, PhoneInventory, AlignerLabelMapper]:
    cfg = load_config(args.config)
    inventory = PhoneInventory.from_files(cfg.phonemes.inventory_path)
    mapper = AlignerLabelMapper.from_mapping_file(cfg.phonemes.mapping_path, inventory)
    return cfg, inventory, mapper


def _emit(args: argparse.Namespace, payload: Any, *, default_name: str | None = None) -> None:
    """Write JSON to ``--output`` (never clobbering without --overwrite) or to stdout."""
    out = getattr(args, "output", None)
    cfg_indent = 2
    text = dumps_json(payload, indent=cfg_indent)
    if out:
        write_text_atomic(Path(out), text, overwrite=getattr(args, "overwrite", False))
        log.info("wrote output", extra={"path": str(out)})
    else:
        sys.stdout.write(text)


def _require_file(path: str | None, what: str) -> Path:
    if not path or not Path(path).is_file():
        raise ConfigError(f"{what} not found: {path}")
    return Path(path)


# ---------------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------------
def _pkg_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def cmd_check(args: argparse.Namespace) -> int:
    items: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str, *, required: bool = True) -> None:
        items.append({"name": name, "status": "ok" if ok else ("FAIL" if required else "missing (optional)"),
                      "required": required, "detail": detail})

    add("python", sys.version_info >= (3, 10), f"{platform.python_version()} ({platform.platform()})")
    add("asa-m2", True, f"{__version__} (schema {SCHEMA_VERSION})")
    for dist in ("cmudict", "praatio", "soundfile", "numpy", "scipy", "PyYAML", "pydantic"):
        v = _pkg_version(dist)
        add(f"dependency:{dist}", v is not None, v or "not installed")
    try:
        import soundfile as sf
        add("libsndfile", True, sf.__libsndfile_version__)
    except Exception as exc:  # pragma: no cover
        add("libsndfile", False, str(exc))

    cfg: AppConfig | None = None
    inventory: PhoneInventory | None = None
    try:
        cfg, inventory, _mapper = _load_all(args)
        add("config", True, f"valid (fingerprint {cfg.fingerprint()[:12]})")
        add("phone inventory", True, f"{inventory.inventory_id} v{inventory.inventory_version}, {len(inventory)} phones, "
                                      f"CTC blank id {inventory.ctc_blank_id} (not a phoneme)")
    except M2Error as exc:
        add("config / inventory", False, exc.message)
    if cfg is not None and inventory is not None:
        try:
            from asa_m2.g2p.cmudict_backend import CMUdictBackend
            backend = CMUdictBackend()
            add("cmudict", len(backend) > 0, f"{len(backend)} entries ({backend.source_detail})")
        except M2Error as exc:
            add("cmudict", False, exc.message)
        if cfg.g2p.custom_lexicon_path:
            try:
                from asa_m2.g2p.custom_lexicon import CustomLexicon
                lex = CustomLexicon.from_file(cfg.g2p.custom_lexicon_path, inventory, cfg.g2p.custom_lexicon_mode)
                add("custom lexicon", True, f"{len(lex)} words, mode {lex.mode}")
            except M2Error as exc:
                add("custom lexicon", False, exc.message)
        from asa_m2.g2p.espeak_backend import EspeakBackend
        from asa_m2.phonemes.mapping import PhoneMapping
        try:
            mapping = PhoneMapping.from_file(cfg.phonemes.mapping_path, cfg.g2p.espeak.mapping_name, inventory)
            es = EspeakBackend(inventory, mapping, executable=cfg.g2p.espeak.executable, voice=cfg.g2p.espeak.voice)
            configured = cfg.g2p.fallback_backend == "espeak"
            add("espeak-ng", es.is_available(),
                (f"version {es.version()}" if es.is_available() else f"{cfg.g2p.espeak.executable!r} not on PATH")
                + ("; configured as G2P fallback" if configured else "; fallback not enabled"),
                required=configured)
        except M2Error as exc:
            add("espeak mapping", False, exc.message)
        from asa_m2.alignment.mfa_adapter import MFAAdapter
        adapter = MFAAdapter(cfg.alignment.mfa)
        try:
            exe = adapter.find_executable()
            version = adapter.get_version()
            warns = adapter.check_version(version)
            add("mfa", True, f"{exe} version {version}" + (f" [{'; '.join(warns)}]" if warns else " (verified)"), required=False)
        except M2Error as exc:
            add("mfa", False, exc.message, required=False)
        from asa_m2.alignment import qwen_adapter
        add("qwen3-forcedaligner (optional)", qwen_adapter.is_available(),
            "qwen_asr and torch importable (live integration unverified)" if qwen_adapter.is_available()
            else "qwen_asr/torch not installed (optional; word-level aligner only)", required=False)

    required_extra = set(args.require or [])
    failures = [i for i in items if i["required"] and i["status"] != "ok"]
    for i in items:
        if i["name"].split(" ")[0].split(":")[0] in required_extra and i["status"] != "ok":
            failures.append(i)
    if args.json:
        sys.stdout.write(dumps_json({"ok": not failures, "checks": items}))
    else:
        for i in items:
            sys.stdout.write(f"[{i['status']:>17}] {i['name']}: {i['detail']}\n")
        sys.stdout.write("\nresult: " + ("OK" if not failures else f"{len(failures)} required check(s) failed") + "\n")
    if failures:
        if any(f["name"] == "config / inventory" for f in failures):
            return EXIT_CONFIG
        return EXIT_EXTERNAL if any(f["name"] in {"mfa", "espeak-ng"} for f in failures) else EXIT_DATA
    return EXIT_OK


# ---------------------------------------------------------------------------------
# g2p
# ---------------------------------------------------------------------------------
def _read_g2p_records(path: Path) -> list[tuple[str | None, str]]:
    suffix = path.suffix.lower()
    records: list[tuple[str | None, str]] = []
    if suffix == ".csv":
        with open(path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            cols = reader.fieldnames or []
            col = next((c for c in ("reference_text", "text", "human_verbatim") if c in cols), None)
            if col is None:
                raise ConfigError(f"{path.name}: CSV needs a 'reference_text', 'text' or 'human_verbatim' column")
            for i, row in enumerate(reader, start=1):
                text = (row.get(col) or "").strip()
                if text:
                    records.append(((row.get("sample_id") or f"line_{i:04d}").strip(), text))
        return records
    for i, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if "\t" in line:
            sid, text = line.split("\t", 1)
            records.append((sid.strip() or f"line_{i:04d}", text.strip()))
        else:
            records.append((f"line_{i:04d}", line.strip()))
    return records


def cmd_g2p(args: argparse.Namespace) -> int:
    from asa_m2.g2p.engine import G2PEngine

    if bool(args.text) == bool(args.input):
        raise ConfigError("provide exactly one of --text or --input")
    cfg, inventory, _ = _load_all(args)
    if args.fallback:
        cfg.g2p.fallback_backend = "espeak"
    if args.lexicon:
        cfg.g2p.custom_lexicon_path = str(_require_file(args.lexicon, "custom lexicon"))
        cfg.g2p.custom_lexicon_mode = args.lexicon_mode
    if args.variant_policy:
        cfg.g2p.variant_policy = args.variant_policy
    engine = G2PEngine.from_config(cfg, inventory)
    records = [(None, args.text)] if args.text else _read_g2p_records(_require_file(args.input, "input file"))
    if not records:
        raise ConfigError("no text records to process")
    results = [engine.convert(text, sample_id=sid, text_field=args.text_field).model_dump(mode="json") for sid, text in records]
    payload = {"schema_version": SCHEMA_VERSION, "kind": "asa_m2_g2p_batch", "count": len(results), "records": results}
    unresolved = [(r["sample_id"], t["original_token"], t["resolution_reason"])
                  for r in results for t in r["tokens"] if t["resolution_status"] == "unresolved"]
    if args.output or args.json:
        _emit(args, payload)
    if not args.json:
        if args.output:
            sys.stdout.write(f"wrote {len(results)} record(s) to {args.output}\n")
        else:
            for r in results:
                _print_g2p_table(r)
    for sid, word, reason in unresolved:
        log.error("unresolved word", extra={"sample": sid, "word": word, "reason": reason})
    if unresolved and cfg.g2p.fail_on_unresolved_oov:
        sys.stderr.write(f"error: {len(unresolved)} unresolved word(s) (blocked from forced alignment); "
                         "see output. Add custom lexicon entries, enable --fallback espeak, or set "
                         "g2p.fail_on_unresolved_oov: false\n")
        return EXIT_DATA
    return EXIT_OK


def _print_g2p_table(rec: dict[str, Any]) -> None:
    if rec.get("sample_id"):
        sys.stdout.write(f"# {rec['sample_id']}\n")
    sys.stdout.write(f"text:       {rec['input_text']}\nnormalized: {rec['normalization']['normalized_text']}\n")
    for t in rec["tokens"]:
        if t["resolution_status"] == "unresolved":
            sys.stdout.write(f"  [{t['word_index']}] {t['original_token']!r:<14} UNRESOLVED: {t['resolution_reason']}\n")
            continue
        prons = " | ".join(" ".join(v["raw_arpabet"]) for v in t["variants"])
        sel = t["selected_variant_id"] or "none selected"
        sys.stdout.write(f"  [{t['word_index']}] {t['normalized_token']:<14} {prons:<40} "
                         f"({t['pronunciation_source']}; {t['selection_method']}: {sel})\n")
    sys.stdout.write("\n")


# ---------------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------------
def cmd_prepare_corpus(args: argparse.Namespace) -> int:
    from asa_m2.audio.corpus import prepare_corpus
    from asa_m2.g2p.engine import G2PEngine

    cfg, inventory, _ = _load_all(args)
    if args.fallback:
        cfg.g2p.fallback_backend = "espeak"
    if args.lexicon:
        cfg.g2p.custom_lexicon_path = str(_require_file(args.lexicon, "custom lexicon"))
    if args.target_sample_rate:
        cfg.audio.target_sample_rate = args.target_sample_rate
    engine = G2PEngine.from_config(cfg, inventory)
    manifest = prepare_corpus(_require_file(args.manifest, "manifest"), args.output, cfg, engine,
                              overwrite=args.overwrite, text_field=args.text_field)
    sys.stdout.write(
        f"prepared {len(manifest.samples)} sample(s) ({len(manifest.excluded)} excluded) in {args.output}\n"
        f"  speakers: {manifest.speaker_policy['speakers']} (pseudo: {manifest.speaker_policy['pseudo_speakers']})\n"
        f"  dictionary: {manifest.dictionary['path']} ({manifest.dictionary['entries']} entries)\n"
        f"  target sample rate: {manifest.target_sample_rate} Hz\n"
    )
    for ex in manifest.excluded:
        sys.stdout.write(f"  excluded line {ex.line} ({ex.sample_id}): {'; '.join(ex.messages)}\n")
    for w in manifest.warnings:
        log.warning(w)
    return EXIT_OK


def cmd_validate_corpus(args: argparse.Namespace) -> int:
    from asa_m2.audio.corpus import validate_corpus

    cfg, inventory, _ = _load_all(args)
    phones = None
    if args.acoustic_phones:
        phones = phone_set_from_lines(_require_file(args.acoustic_phones, "acoustic phone list").read_text(encoding="utf-8").splitlines())
    report = validate_corpus(args.corpus, cfg, inventory, acoustic_phones=phones)
    if args.json:
        sys.stdout.write(dumps_json({"ok": report.ok, "n_samples": report.n_samples,
                                     "issues": [i.__dict__ for i in report.issues]}))
    else:
        for i in report.issues:
            sys.stdout.write(f"[{i.severity.upper():>7}] {i.code}{f' ({i.sample_id})' if i.sample_id else ''}: {i.message}\n")
        sys.stdout.write(f"\n{report.n_samples} sample(s): {len(report.errors)} error(s), {len(report.warnings)} warning(s) -> "
                         f"{'VALID' if report.ok else 'INVALID'}\n")
    return EXIT_OK if report.ok else EXIT_DATA


# ---------------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------------
def cmd_align_mfa(args: argparse.Namespace) -> int:
    from asa_m2.alignment.mfa_adapter import MFAAdapter
    from asa_m2.pipeline import align_corpus_with_mfa

    cfg, inventory, mapper = _load_all(args)
    m = cfg.alignment.mfa
    if args.mode:
        m.mode = args.mode
    if args.acoustic_model:
        m.acoustic_model = args.acoustic_model
    if args.hf_model_id:
        m.hf_model_id = args.hf_model_id
    if args.dictionary:
        m.dictionary = args.dictionary
    if args.num_jobs:
        m.num_jobs = args.num_jobs
    if args.allow_unverified_mfa:
        m.allow_unverified_versions = True
    summary = align_corpus_with_mfa(args.corpus, args.output, cfg, inventory, mapper, MFAAdapter(m),
                                    overwrite=args.overwrite, run_mfa_validate=args.mfa_validate)
    sys.stdout.write(f"MFA {summary.mfa.version if summary.mfa else '?'}: {summary.n_success}/{len(summary.outcomes)} sample(s) aligned successfully\n")
    for o in summary.outcomes:
        sys.stdout.write(f"  {o.sample_id}: {o.status} (errors={o.error_flags}, warnings={o.warning_flags})\n")
    sys.stdout.write(f"outputs: {summary.output_dir}/{{textgrids,json,logs}}, alignment_summary.json\n"
                     "note: 'success' means aligned and structurally sound - not that the learner pronounced the targets correctly.\n")
    if summary.has_failures:
        sys.stderr.write("error: one or more samples did not align successfully; see alignment_summary.json and the JSON flags\n")
        return 6
    return EXIT_OK


def cmd_align_qwen(args: argparse.Namespace) -> int:
    from asa_m2.alignment.exporter import assemble_sample_record, write_word_textgrid
    from asa_m2.alignment.qwen_adapter import QwenForcedAlignerAdapter
    from asa_m2.audio.loader import probe_audio
    from asa_m2.g2p.engine import G2PEngine
    from asa_m2.schemas import AlignmentMetadata

    cfg, inventory, _ = _load_all(args)
    audio = _require_file(args.audio, "audio file")
    if args.device:
        cfg.alignment.qwen.device = args.device
    engine = G2PEngine.from_config(cfg, inventory)
    g2p = engine.convert(args.text, sample_id=args.sample_id, text_field=args.text_field)
    words = [t.normalized_token for t in g2p.tokens if t.supported]
    adapter = QwenForcedAlignerAdapter(cfg.alignment.qwen)
    result = adapter.align_words(audio, words)
    info = probe_audio(audio)
    meta = AlignmentMetadata(
        aligner="Qwen3-ForcedAligner", aligner_version=result.model_info.get("qwen_asr_version"), granularity="word",
        status="success", model=result.model_info,
        notes=["word-level timestamps only; no phone alignment is available from this aligner",
               "live integration of this adapter was not verified in the M2 build environment"],
    )
    record = assemble_sample_record(
        g2p=g2p, word_intervals=result.intervals, phone_intervals=None, word_tier_name="qwen_words",
        phone_tier_name=None, config=cfg, inventory=inventory, sample_id=args.sample_id or audio.stem,
        speaker_id=args.speaker_id,
        reference_text=args.text if args.text_field == "reference_text" else None,
        human_verbatim=args.text if args.text_field == "human_verbatim" else None,
        alignment_metadata=meta, source_file=audio.name,
    )
    if args.textgrid_output:
        write_word_textgrid(Path(args.textgrid_output), [(i.start, i.end, i.label) for i in result.intervals],
                            info.duration_s, overwrite=args.overwrite)
    _emit(args, record.model_dump(mode="json"))
    return EXIT_OK


# ---------------------------------------------------------------------------------
# textgrid inspection / export / evaluation
# ---------------------------------------------------------------------------------
def cmd_inspect_textgrid(args: argparse.Namespace) -> int:
    from asa_m2.alignment.textgrid_parser import parse_textgrid
    from asa_m2.alignment.validator import validate_textgrid

    cfg = load_config(args.config)
    parsed = parse_textgrid(_require_file(args.input, "TextGrid"))
    issues = validate_textgrid(parsed, cfg.alignment.qc)
    if args.json:
        data = parsed.to_dict()
        data["issues"] = [i.__dict__ for i in issues]
        sys.stdout.write(dumps_json(data))
        return EXIT_OK if not any(i.severity == "error" for i in issues) else 6
    sys.stdout.write(f"file: {parsed.path.name}  format: {parsed.text_format}  encoding: {parsed.encoding}  "
                     f"range: {parsed.xmin}..{parsed.xmax} s\nsha256: {parsed.sha256}\ntiers ({len(parsed.tiers)}):\n")
    for t in parsed.tiers:
        n = len(t.intervals) if t.is_interval_tier else len(t.points)
        empty = sum(1 for i in t.intervals if not i.label.strip())
        sys.stdout.write(f"  - {t.name!r} [{t.tier_type}] {n} entries ({empty} empty), range {t.xmin}..{t.xmax}\n")
        for iv in t.intervals[: args.max_intervals]:
            sys.stdout.write(f"      {iv.index:>4}  {iv.start:9.4f} {iv.end:9.4f}  {iv.label!r}\n")
        if len(t.intervals) > args.max_intervals:
            sys.stdout.write(f"      ... {len(t.intervals) - args.max_intervals} more (use --max-intervals or --json)\n")
    errors = [i for i in issues if i.severity == "error"]
    sys.stdout.write(f"\nvalidation: {len(errors)} error(s), {sum(i.severity == 'warning' for i in issues)} warning(s)\n")
    for i in issues:
        loc = f" tier={i.tier!r}" if i.tier else ""
        loc += f" interval={i.interval_index}" if i.interval_index is not None else ""
        sys.stdout.write(f"  [{i.severity.upper()}] {i.code}{loc}: {i.message}\n")
    return EXIT_OK if not errors else 6


def cmd_export(args: argparse.Namespace) -> int:
    from asa_m2.alignment.exporter import record_from_textgrid
    from asa_m2.alignment.textgrid_parser import parse_textgrid
    from asa_m2.alignment.validator import validate_textgrid
    from asa_m2.audio.loader import probe_audio
    from asa_m2.g2p.engine import G2PEngine
    from asa_m2.schemas import AlignmentMetadata, G2PResult

    cfg, inventory, mapper = _load_all(args)
    parsed = parse_textgrid(_require_file(args.textgrid, "TextGrid"))
    sample_id = args.sample_id or parsed.path.stem
    duration = probe_audio(_require_file(args.audio, "audio file")).duration_s if args.audio else None
    if not (args.g2p_json or args.text):
        issues = validate_textgrid(parsed, cfg.alignment.qc, audio_duration_s=duration)
        data = parsed.to_dict()
        data.update({"kind": "asa_m2_textgrid_export", "schema_version": SCHEMA_VERSION, "sample_id": sample_id,
                     "issues": [i.__dict__ for i in issues],
                     "note": "alignment-only export: no canonical text was supplied, so no token/phone linkage or QC against "
                             "expected words was possible (use --text or --g2p-json)"})
        _emit(args, data)
        return EXIT_OK
    if args.g2p_json:
        g2p = G2PResult.model_validate_json(_require_file(args.g2p_json, "G2P JSON").read_text(encoding="utf-8"))
    else:
        g2p = G2PEngine.from_config(cfg, inventory).convert(args.text, sample_id=sample_id, text_field=args.text_field)
    meta = AlignmentMetadata(aligner=args.aligner_name, aligner_version=args.aligner_version, granularity="phone", status="success",
                             notes=["exported from an existing TextGrid; aligner configuration not known to asa-m2"])
    record = record_from_textgrid(
        g2p=g2p, textgrid=parsed, config=cfg, inventory=inventory, mapper=mapper, sample_id=sample_id,
        speaker_id=args.speaker_id,
        reference_text=g2p.input_text if g2p.text_field == "reference_text" else None,
        human_verbatim=g2p.input_text if g2p.text_field == "human_verbatim" else None,
        alignment_metadata=meta, word_tier=args.word_tier, phone_tier=args.phone_tier, audio_duration_s=duration,
    )
    _emit(args, record.model_dump(mode="json"))
    return EXIT_OK if record.alignment_metadata.status == "success" else 6


def cmd_evaluate_boundaries(args: argparse.Namespace) -> int:
    from asa_m2.alignment.quality import compare_tiers, evaluate_boundaries
    from asa_m2.alignment.textgrid_parser import parse_textgrid, select_tier

    cfg = load_config(args.config)
    tg = cfg.alignment.textgrid
    names = tg.phone_tier_names if args.tier == "phone" else tg.word_tier_names
    pred, gold = Path(args.pred), Path(args.gold)
    if pred.is_dir() != gold.is_dir():
        raise ConfigError("--pred and --gold must both be files or both be directories")
    pairs = ([(p, gold / p.name) for p in sorted(pred.glob("*.TextGrid"))] if pred.is_dir() else [(pred, gold)])
    if not pairs:
        raise ConfigError(f"no .TextGrid files found in {pred}")
    results = []
    for p, g in pairs:
        if not g.is_file():
            from asa_m2.alignment.quality import FileBoundaryResult
            results.append(FileBoundaryResult(p.name, True, f"no gold TextGrid {g.name}"))
            continue
        pt, gt = parse_textgrid(p), parse_textgrid(g)
        results.append(compare_tiers(
            p.name, select_tier(pt, names, allow_speaker_prefix=tg.allow_speaker_prefix, kind=args.tier),
            select_tier(gt, names, allow_speaker_prefix=tg.allow_speaker_prefix, kind=args.tier),
            silence_labels={"sil", "sp", "<eps>", *tg.extra_silence_labels}, strip_stress=not args.keep_stress))
    ev = evaluate_boundaries(results, args.tolerance if args.tolerance else cfg.alignment.qc.boundary_tolerance_s)
    payload = {"kind": "asa_m2_boundary_evaluation", "tier": args.tier, **ev.__dict__,
               "failed_files": [{"file": r.name, "reason": r.reason} for r in results if r.failed],
               "note": "numbers describe agreement with the SUPPLIED gold annotations only"}
    _emit(args, payload)
    return EXIT_OK


# ---------------------------------------------------------------------------------
# parser / main
# ---------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parent = _common_parent()
    parser = argparse.ArgumentParser(prog="asa-m2", description="ASA Module M2 - Phonetic Front-end")
    parser.add_argument("--version", action="version", version=f"asa-m2 {__version__} (schema {SCHEMA_VERSION})")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def add(name: str, help_: str, func: Any) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, parents=[parent], help=help_, description=help_)
        sp.set_defaults(func=func)
        return sp

    p = add("check", "check dependencies, configuration and optional components", cmd_check)
    p.add_argument("--json", action="store_true")
    p.add_argument("--require", action="append", choices=["mfa", "espeak-ng", "qwen3-forcedaligner"],
                   help="fail (non-zero exit) if this optional component is unavailable (repeatable)")

    p = add("g2p", "canonical pronunciations for text (CMUdict; optional eSpeak-ng fallback)", cmd_g2p)
    p.add_argument("--text", help="a single text to process")
    p.add_argument("--input", help="file with one record per line (.txt: 'id<TAB>text' or text) or a .csv")
    p.add_argument("--output", help="write JSON here (refuses to overwrite without --overwrite)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--json", action="store_true", help="print JSON to stdout instead of a table")
    p.add_argument("--text-field", choices=["reference_text", "human_verbatim"], default="reference_text",
                   help="which kind of text this is (default reference_text; never interchangeable)")
    p.add_argument("--fallback", choices=["espeak"], help="enable the eSpeak-ng fallback for OOV words")
    p.add_argument("--lexicon", help="custom pronunciation lexicon file")
    p.add_argument("--lexicon-mode", choices=["override", "augment"], default="override")
    p.add_argument("--variant-policy", choices=["keep_all", "first_listed"], help="override g2p.variant_policy")

    p = add("prepare-corpus", "normalise audio and build an MFA-compatible corpus + dictionary", cmd_prepare_corpus)
    p.add_argument("--manifest", required=True, help="CSV manifest (sample_id, audio_path, reference_text|reference_path|human_verbatim, [speaker_id])")
    p.add_argument("--output", required=True, help="output corpus directory")
    p.add_argument("--overwrite", action="store_true", help="replace a previously prepared corpus in --output")
    p.add_argument("--text-field", choices=["auto", "reference_text", "human_verbatim"], default="auto")
    p.add_argument("--target-sample-rate", type=int, help="override audio.target_sample_rate")
    p.add_argument("--fallback", choices=["espeak"], help="enable eSpeak-ng fallback for OOV words")
    p.add_argument("--lexicon", help="custom pronunciation lexicon file")

    p = add("validate-corpus", "validate a prepared corpus before alignment", cmd_validate_corpus)
    p.add_argument("--corpus", required=True)
    p.add_argument("--acoustic-phones", help="text file with the acoustic model's phone set (one per line) for a compatibility check")
    p.add_argument("--json", action="store_true")

    p = add("align-mfa", "run Montreal Forced Aligner (version-gated) and export JSON records", cmd_align_mfa)
    p.add_argument("--corpus", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--mode", choices=["legacy", "hf"], help="legacy => `mfa align`, hf => `mfa align_hf`")
    p.add_argument("--acoustic-model", help="acoustic model name or path (legacy mode)")
    p.add_argument("--hf-model-id", help="Hugging Face model id (hf mode)")
    p.add_argument("--dictionary", help="MFA dictionary name/path (default: the corpus-generated dictionary)")
    p.add_argument("--num-jobs", type=int)
    p.add_argument("--mfa-validate", action="store_true", help="also run `mfa validate` first (legacy mode)")
    p.add_argument("--allow-unverified-mfa", action="store_true", help="try MFA versions outside the verified range")

    p = add("align-qwen", "OPTIONAL word-level alignment with Qwen3-ForcedAligner (live use unverified)", cmd_align_qwen)
    p.add_argument("--audio", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--output", help="write the JSON record here (default: stdout)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--textgrid-output", help="also write a word-tier TextGrid")
    p.add_argument("--sample-id")
    p.add_argument("--speaker-id")
    p.add_argument("--device", help="override alignment.qwen.device (e.g. cuda:0, cpu)")
    p.add_argument("--text-field", choices=["reference_text", "human_verbatim"], default="reference_text")

    p = add("inspect-textgrid", "list tiers, intervals and validation issues of a TextGrid", cmd_inspect_textgrid)
    p.add_argument("--input", required=True)
    p.add_argument("--max-intervals", type=int, default=8)
    p.add_argument("--json", action="store_true")

    p = add("export", "export a TextGrid (optionally linked to canonical text) as structured JSON", cmd_export)
    p.add_argument("--textgrid", required=True)
    p.add_argument("--output", help="JSON output path (default: stdout)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--text", help="canonical text (runs G2P) to link tokens/phones and run QC")
    p.add_argument("--g2p-json", help="G2P result JSON (e.g. corpus/g2p/<id>.g2p.json) instead of --text")
    p.add_argument("--text-field", choices=["reference_text", "human_verbatim"], default="reference_text")
    p.add_argument("--audio", help="audio file; its duration is checked against the TextGrid")
    p.add_argument("--sample-id")
    p.add_argument("--speaker-id")
    p.add_argument("--word-tier")
    p.add_argument("--phone-tier")
    p.add_argument("--aligner-name", default="unknown", help="recorded in alignment_metadata.aligner")
    p.add_argument("--aligner-version")

    p = add("evaluate-boundaries", "boundary error vs gold TextGrids (file or directory of .TextGrid)", cmd_evaluate_boundaries)
    p.add_argument("--pred", required=True)
    p.add_argument("--gold", required=True)
    p.add_argument("--tier", choices=["phone", "word"], default="phone")
    p.add_argument("--tolerance", type=float, help="seconds (default qc.boundary_tolerance_s)")
    p.add_argument("--keep-stress", action="store_true", help="compare phone labels including stress digits")
    p.add_argument("--output")
    p.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse already printed usage; surface its code (0 for --help/--version, 2 for errors)
        return int(exc.code) if isinstance(exc.code, int) else EXIT_INTERNAL
    try:
        setup_logging(args.log_level, log_file=args.log_file, json_format=args.json_logs)
    except ValueError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    log.info("running command", extra={"cmd": args.command, "asa_m2": __version__})
    try:
        code = int(args.func(args))
        log.info("command finished", extra={"cmd": args.command, "exit_code": code})
        return code
    except OutputExistsError as exc:
        sys.stderr.write(f"error: {exc.message}\n")
        return exc.exit_code
    except M2Error as exc:
        log.error(exc.message)
        sys.stderr.write(f"error: {exc.message}\n")
        return exc.exit_code
    except KeyboardInterrupt:
        sys.stderr.write("interrupted\n")
        return 130
    except Exception as exc:
        log.exception("unexpected error")
        sys.stderr.write(f"internal error: {type(exc).__name__}: {exc} (this is a bug; please report with --log-level DEBUG)\n")
        return EXIT_INTERNAL


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

__all__ = ["build_parser", "main"]
