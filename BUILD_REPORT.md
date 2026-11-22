# ASA M2 — Build report

Build date: 2026-11-11 · Package: `asa-m2 1.0.0` · Schema: `1.0` · Inventory: `asa-arpabet-cmu39 v1.0.0`

## 1. Summary

A modular, tested Python package implementing M2.1–M2.8 and the M2→M3/M4 data contract. Everything that could be executed in the build environment **was** executed.
Two external components could **not** be run live — Montreal Forced Aligner and Qwen3-ForcedAligner — and are reported as such below. **No evaluation on L2-ARCTIC or HANU was performed.**

## 2. Files created (102 files, including this report)

| Area | Files | Content |
|---|---|---|
| `src/asa_m2/` | 42 (34 `.py` modules, 6,057 lines + 8 packaged resources) | normalization · g2p (CMUdict, custom lexicon, eSpeak adapter, engine) · phonemes (inventory, mapping) · audio (loader, validation, resampler, corpus) · alignment (MFA adapter, Qwen adapter, TextGrid parser, validator, QC, exporter) · schemas · config · cli · pipeline · provenance · logging · utils |
| `configs/` | 8 | `default.yaml`, `g2p.yaml`, `alignment.yaml`, `audio.yaml`, `phoneme_inventory.json`, `phone_to_id.json`, `id_to_phone.json`, `phoneme_mapping.json` (byte-identical copies are packaged in `src/asa_m2/resources/`; a test enforces it) |
| `tests/` | 24 (23 `.py`, 4,287 lines) | `unit/` (15 modules) · `integration/` (3 modules) · `fixtures/fake_mfa.py` (**fake** `mfa`, not MFA) · `conftest.py` |
| `docs/` | 5 | `architecture.md`, `data_contract.md`, `troubleshooting.md`, `research_limitations.md`, `schema/sample_record.schema.json` |
| `scripts/` | 6 | `validate_inventory.py`, `inspect_textgrid.py`, `prepare_corpus.py` (+ `generate_inventory.py`, `export_schema.py`, `sync_resources.py`) |
| `examples/` | 6 | `input_manifest.csv`, `sample_reference.txt`, `expected_output_schema.json`, `example_g2p_output.json` (real output), `custom_lexicon.dict`, `make_example_audio.py` (synthetic tones; WAVs not shipped) |
| root | 7 + `data/` | `README.md`, `LICENSE` (MIT placeholder), `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`, `.gitignore`, `.env.example`, `BUILD_REPORT.md`; `data/{raw,processed,outputs}/.gitkeep` |

## 3. Environment and dependencies tested

Python **3.12.3** (Ubuntu 24.04, 1 CPU, no GPU). Declared range ≥ 3.10; **3.10/3.11 and Windows were not tested**.
cmudict 1.1.3 · praatio 6.2.2 · soundfile 0.14.0 (libsndfile 1.2.2) · numpy 2.5.4 · scipy 1.18.1 · PyYAML 6.0.3 · pydantic 2.14.3 · pytest 9.2.0 · pytest-cov 7.1.0 · ruff 0.17.4 · mypy 2.4.1 · eSpeak NG 1.51.
`pandas`/`typer` (listed as options in the brief) are deliberately **not** used (stdlib `csv`/`argparse` suffice; see `docs/architecture.md`).

## 3a. Test results

| Suite | Result |
|---|---|
| Unit tests (`tests/unit`, no external software) | **543 passed** |
| Integration tests (`tests/integration`) | **9 passed** (live eSpeak-ng) · **2 skipped**: live MFA (`mfa` not usable here) and live Qwen (opt-in, not installed). Skipped ≠ passed |
| Total | **552 passed, 2 skipped, 0 failed** |
| Line coverage | 95 % (3,730 statements); `pipeline.py` 100 %, `corpus.py` 96 %, `cli.py` 93 %, `qwen_adapter.py` 77 % (the uncovered lines are the real model-loading path) |
| `ruff check src tests scripts examples` | clean |
| `mypy src` | no issues (34 files) |
| Generated-file guards | `generate_inventory.py --check`, `export_schema.py --check`, `configs/` ⇄ `resources/` equality: all pass |
| Documentation checks | every `asa-m2 …` command in README/docs parses against the real CLI; all relative links/anchors resolve; flag-code catalogue and exit-code table match the code |
| Clean-environment install | wheel built; installed with pinned `requirements.txt` into a fresh venv; run from an unrelated directory; README workflow executed (prepare → validate → align with a **fake** `mfa` → inspect → export → evaluate; failure paths exit 5/7 as documented) |

Notable test designs: a differential test of the TextGrid parser against `praatio` (25 random files × long/short formats, identical results); a fresh-sample test of the eSpeak mapping against the real binary (1,500 words not used to derive the table: 92.2 % resolved, 7.8 % unresolved by design, **0** unexpected units); resampling tests checking sample counts, preserved tone frequency and anti-aliasing; invariant tests for the normaliser ("no character silently lost"); tests scanning every output file for leaked absolute paths.

## 4. External component status

| Component | Status | Evidence / limits |
|---|---|---|
| **MFA** | ❌ **not run** | `pip install montreal-forced-aligner` installs **3.4.3**, but `mfa version` fails with `ModuleNotFoundError: No module named '_kalpy'` (Kaldi bindings are conda-forge only; conda was unavailable). `asa-m2 check` reported this real failure correctly. Commands/options/tier names/labels were **read from the 3.4.3 source**; the wrapper is exercised against a **fake** `mfa`. Supported: 3.4.3 (verified), other 3.4.x (warning), everything else refused. Live test is written and skips with a reason. |
| **eSpeak-ng** | ✅ **installed (1.51) and exercised** | live integration tests pass; mapping validated on real output |
| **Qwen3-ForcedAligner adapter** | ⚠️ **implemented, live use unverified** | written from the official README and `qwen_asr/inference/qwen3_forced_aligner.py`; unit-tested with an injected fake model; no GPU/model here. Word-level only; `*-hf` checkpoints not inspected/supported. |
| L2-ARCTIC / HANU | ❌ **no evaluation** | no data used; `evaluate-boundaries` ready for manual annotations |

## 5. Defects found *by running things* (and fixed)

Verification — especially the clean-environment run of the README — exposed defects that passing unit tests did not reveal at first:

1. **Relative `--output` crashed** `prepare-corpus` (the README's own example): path-safety compared resolved vs unresolved paths. All early tests used absolute `tmp_path`s. *Fixed; regression tests use relative paths.*
2. **Partial `--config` files lost the inventory** (default path resolved against the CWD). *Fixed: defaults point at the packaged resources.*
3. **Config fingerprint embedded machine-specific absolute paths** (not reproducible across installs). *Fixed: path-independent; test added.*
4. **Truncated WAV files were silently accepted** (libsndfile reports fewer frames). *Fixed: RIFF chunk-size integrity check.*
5. **Variant resolution ignored the aligner's stress** (required stress on *every* label, but consonants carry none). *Fixed: vowels only.*
6. **Crash when a phone tier held only OOV/silence labels** (granularity/schema mismatch). *Fixed; regression test.*
7. **Unresolved-OOV samples aborted the whole corpus even when configured to be excluded.** *Fixed.*
8. **Failed precondition (unsupported MFA version) left a half-created output directory**, blocking the retry. *Fixed.*
9. **Shared JSON records embedded the absolute MFA command line** (could leak user names). *Fixed: sanitized form in records; exact command only in the local log; tests scan all outputs.*
10. Misused `praatio` API (guessed `Interval` location) caught by tests and fixed against the real signature; abbreviation `Dr.` double-counted its period (caught by the normaliser's own `verify()`); several uninformative error messages improved (e.g. broken MFA now shows the real exception line).
11. Documentation facts re-verified against the source after writing (e.g. 8 — not 12 — `approximate` eSpeak mappings).

## 6. Known limitations (full list: `docs/research_limitations.md`)

G2P ≠ realised pronunciation · forced alignment ≠ verified pronunciation (`realized_phone` is always `null`) · L2 alignment quality unmeasured · MFA/Qwen not run live · English only ·
number/abbreviation expansion is one reading · eSpeak fallback is lower-confidence and deliberately unmapped for flap/glottal stop · default thresholds are starting points · acoustic-model phone sets must be supplied as a file for compatibility checks
(automatic extraction from MFA model archives was not implemented because the archive format could not be verified) · licences of third-party data/tools were not reviewed; the MIT `LICENSE` is a placeholder.

## 7. Exact commands

**Install (Ubuntu 24.04, Python 3.12):**
```bash
sudo apt-get update && sudo apt-get install -y python3-venv espeak-ng      # espeak-ng optional (fallback G2P)
unzip ASA_M2_Phonetic_Frontend.zip && cd ASA_M2_Phonetic_Frontend
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e . --no-deps
asa-m2 check
```
**Run the tests:** `pytest` (all) · `pytest tests/unit -q` (no external software) · `pytest tests/integration -rs` (shows skip reasons).
**Core workflow:**
```bash
asa-m2 g2p --text "I think this is important."
python examples/make_example_audio.py        # synthetic tones, only to exercise the plumbing
asa-m2 prepare-corpus --manifest examples/input_manifest.csv --output data/processed/corpus
asa-m2 validate-corpus --corpus data/processed/corpus
# needs MFA installed via conda (README §8) and an installed acoustic model:
asa-m2 align-mfa --corpus data/processed/corpus --output data/outputs/alignment --acoustic-model english_us_arpa
```
