# M2 architecture

## Responsibilities

M2 sits between the *inputs* (text, audio) and the research modules M3/M4. It answers: **"what phone sequence is the learner expected to produce, and where does a forced
aligner place those target phones in the recording?"** It does not answer *what the learner actually produced* (M3/M4).

```
src/asa_m2/
├── normalization/text_normalizer.py   M2.1  auditable tokenisation (spans, rules, dropped list, verify())
├── g2p/
│   ├── cmudict_backend.py             primary source; all variants kept in file order
│   ├── custom_lexicon.py              user lexicon (override|augment, !select)
│   ├── espeak_backend.py              opt-in fallback; strict documented mapping
│   ├── variants.py                    ARPABET → PronunciationVariant; canonical-phone flattening
│   └── engine.py                      lookup order, variant policy, OOV gate (ensure_alignable)
├── phonemes/
│   ├── inventory.py                   M2.3 versioned inventory + validation + compatibility
│   └── mapping.py                     PhoneMapping (eSpeak→ARPABET), AlignerLabelMapper (MFA labels)
├── audio/
│   ├── loader.py validation.py        probing, truncated-WAV detection, sanity checks
│   ├── resampler.py                   to_mono, resample_poly, PCM WAV export, transformation records
│   └── corpus.py                      manifest → MFA corpus + dictionary; validate_corpus
├── alignment/
│   ├── mfa_adapter.py                 M2.5 version-gated subprocess wrapper
│   ├── textgrid_parser.py validator.py  M2.6 raw reader, issues, tier selection
│   ├── quality.py                     M2.7 sequence matching, duration warnings, gold evaluation
│   ├── exporter.py                    SampleRecord assembly (shared by MFA and word-only aligners)
│   └── qwen_adapter.py                M2.8 optional, lazy
├── schemas.py                         the data contract (pydantic)
├── pipeline.py                        align-mfa orchestration
├── config.py cli.py logging_config.py provenance.py exceptions.py utils/
```

## Data flow

1. `reference_text` → **TextNormalizer** → `NormToken[]` (+ `dropped` spans, warnings). `verify()` proves every non-space character is either in a token span or recorded as dropped.
2. **G2PEngine** attaches `PronunciationVariant[]` to each token (lookup order: custom lexicon override → CMUdict [+ custom augment] → eSpeak fallback if enabled → *unresolved*).
   Unresolved tokens have `approved_for_alignment=false`; `ensure_alignable()` and `prepare_corpus()` refuse to pass them downstream.
3. **prepare_corpus** validates audio, down-mixes, resamples, writes `mfa_corpus/<speaker>/<id>.{wav,lab}`, a dictionary containing exactly the needed words (all variants, stress kept by default), per-sample `g2p/*.json` and `corpus_manifest.json`.
4. **MFAAdapter** checks the executable and version, runs `mfa align` (or `align_hf`) as a list-argv subprocess (no shell), saves stdout/stderr/command, and verifies one TextGrid per expected sample.
5. **parse_textgrid → validate_textgrid** report structural/semantic problems; **assemble_sample_record** matches aligned words to expected tokens with a sequence diff, assigns phones to words by interval midpoint, resolves variants from the aligner's phones (`alignment_match`),
   computes QC and writes the `SampleRecord`.

## Design decisions (and why)

| Decision | Reason |
|---|---|
| MFA is invoked as an external command | MFA's Python internals and CLI change between releases; only commands verified against the 3.4.3 source are emitted; unsupported versions fail early. |
| Own TextGrid reader instead of `praatio` for parsing | `praatio.openTextgrid` refuses overlapping / negative / zero-length intervals (raising a bare `IndexError` on malformed input) — M2 must *report* those. `praatio` is used for writing and as a differential-test oracle (25 random files × 2 formats agree). |
| `argparse`, no Typer | Zero extra dependencies; deterministic exit codes; easy to test via `main(argv)`. |
| No `pandas` | The manifest is a small CSV; the `csv` module avoids dtype coercion (`007` → `7`). |
| pydantic models for config and schema | Startup validation (`extra="forbid"` catches typos) and a machine-readable JSON Schema from the same source. |
| `configs/` + packaged `resources/` | Editable files for users, importable defaults for installed wheels; a test fails if they drift. |
| Fake `mfa` in tests | Exercises the *wrapper* (version probe, argv, exit codes, logs, missing outputs) deterministically. It proves nothing about real MFA and is never reported as such. |
| Module-level cache of the CMUdict | ~126k read-only entries; parsing per instance made the suite 10× slower. `lookup` returns copies. |

## Configuration

`configs/default.yaml` composes `g2p.yaml`, `alignment.yaml`, `audio.yaml` through `include:`; later keys override earlier ones. Several `--config` files may be given (later wins).
Relative `inventory_path`/`mapping_path`/`custom_lexicon_path` resolve against the declaring file; if a config does not mention the inventory, the packaged copy is used (never the current directory).
All sections and defaults: `src/asa_m2/config.py`.

## Extending

* **New aligner:** produce `IntervalIn` lists and call `assemble_sample_record(...)` (this is how the Qwen adapter works); declare the true `granularity`.
* **New phone inventory version:** only *append* IDs; bump `inventory_version`; regenerate with `scripts/generate_inventory.py`; mappings refuse a mismatched version.
* **New eSpeak unit:** add it to `ESPEAK_ENTRIES` (with `confidence`) or to `ESPEAK_UNMAPPED_BY_DESIGN` in `scripts/generate_inventory.py`; regenerate and sync.

## Qwen3-ForcedAligner adapter (optional)

`alignment/qwen_adapter.py`; **live integration unverified** (no GPU / model download in the build environment).

Facts taken from the official README and `qwen_asr/inference/qwen3_forced_aligner.py` of <https://github.com/QwenLM/Qwen3-ASR>:
`from qwen_asr import Qwen3ForcedAligner`; `Qwen3ForcedAligner.from_pretrained(name, dtype=..., device_map=...)`; `model.align(audio=, text=, language=)` returns one result per input,
iterable over items with `.text/.start_time/.end_time` in **seconds** (rounded to ms); audio is converted to mono 16 kHz; English text is split on whitespace and each token is stripped to letters, digits and `'`;
timestamps are for words (characters for CJK) — **never phones**; roughly 5 minutes of audio per call.

What the adapter therefore does: loads lazily (importing the module loads nothing heavy; `asa-m2 check` only calls `importlib.util.find_spec`); passes `(np.ndarray, sr)` rather than URLs (URL/base64 inputs are refused, so the adapter itself makes no network calls — the Hugging Face download
happens inside `from_pretrained`); pre-checks that every token survives Qwen's cleaning unchanged and that the returned unit count/text match 1:1 (otherwise it raises instead of guessing); reports `granularity="word"` and an empty `phone_alignments`;
translates OOM/download/missing-dependency problems into `QwenResourceError`/`OptionalDependencyError` (exit 7). The newer Transformers-native `*-hf` checkpoints mentioned upstream were **not** inspected and are **not** supported.
