# Research limitations, assumptions and unverified items

Written to be read *before* using M2 output in an experiment.

## What M2 outputs are — and are not

1. **G2P gives canonical pronunciation candidates, not the pronunciation that occurred.** CMUdict is a General-American-oriented lexicon; 8,447 of its 126,052 entries (6.7 %, cmudict 1.1.3) have more than one variant.
   Variant order carries no frequency or correctness information. M2 keeps all variants and selects none unless a documented method does.
2. **Forced alignment places a *supplied* target sequence on the audio.** It does not show that the learner produced those phones. Aligned target phones are never labelled as realised phones; `realized_phone` is always `null`. Realised-phone evidence belongs to M3/M4.
3. **`alignment_match` is the aligner's decision among candidates** (e.g. MFA chooses a dictionary variant by likelihood). It must not be used as a label of what the learner said.
4. **L2 speech can align worse than native speech.** Substitutions, deletions and disfluencies violate the aligner's assumption that the text matches the audio; boundaries may then be systematically wrong *without any flag*. M2's QC (missing words, odd durations, OOV markers) detects only some of this.
5. **Duration thresholds are not pronunciation thresholds.** `phone_too_short/long` etc. are sanity warnings tunable per corpus.

## Not evaluated

* **No experiment has been run on L2-ARCTIC or HANU.** No claim is made about alignment accuracy on either corpus. `asa-m2 evaluate-boundaries` computes boundary MAE, median absolute error, % within tolerance and failure rate *when you supply manual reference alignments*; those numbers would be the first real evidence.
* The build used only synthetic data (tones, generated TextGrids). Test coverage proves the code's behaviour on those inputs, not suitability for real learner audio.

## Not run in the build environment

| Component | What was verified instead |
|---|---|
| Montreal Forced Aligner | CLI/option surface read from the **MFA 3.4.3 source** (commands, option names, tier naming, silence/OOV labels, `.lab` precedence, `MFA_ROOT_DIR`); `english_us_arpa` dictionary v2.0.0 format inspected. A pip-only install of 3.4.3 cannot start without the conda-only `_kalpy` module. The wrapper is tested against a **fake** `mfa`. Whether the generated dictionary format (`word<TAB>phones`) is accepted by every MFA 3.4.x was not run; it is MFA's basic documented dictionary format. The acoustic model archive could not be downloaded, so automatic phone-set extraction from model archives is **not** implemented (supply a phone list file). |
| Qwen3-ForcedAligner | Interface taken from the official README and source file; not loaded, not run (no GPU, no model). Time unit (seconds), tokenisation and output structure are from the source. The `*-hf` checkpoints were not inspected. |
| Windows, Python 3.10/3.11 | untested (syntax kept 3.10-compatible) |

## Assumptions and approximations

* **eSpeak→ARPABET mapping** is a documented table derived from the units eSpeak actually emitted on 12,000 CMUdict words, then checked on a *fresh* 1,500-word sample: 92.2 % resolved, 7.8 % unresolved by design, 0 unexpected units. 8 entries are marked `approximate` (`ᵻ`→IH, syllabic `əl`/`n̩`→AH L/N, `ɐ`, `o`, `iə`, `aɪɚ`, `aɪə`) and are flagged on every use. The flap `ɾ` (T/D), glottal stop `ʔ`, `x`, `ɬ`, trilled `r` are deliberately unmapped.
  eSpeak is a synthesiser front-end, not a lexicon: fallback pronunciations are lower-confidence than CMUdict.
* **Number/abbreviation expansion is one reading** (`1999` → cardinal, not a year). Ambiguous cases warn. Currency, URLs, emails, alphanumerics (`3D`), dotted acronyms and emoji are *unsupported* (unresolved), never guessed. English only.
* **Articulatory metadata and IPA labels** are conventional General-American descriptions, offered as convenience. Dialects differ.
* **Phone inventory**: 39 CMUdict phones. If M3's model uses a different set (e.g. with a silence class, other stress treatment, or IPA), a mapping is needed; `check_compatibility` only reports differences.
* **CTC blank**: reserved ID 0 by convention only; not a phoneme, not silence. M3 may move it in its output layer and must document the offset.
* **Default thresholds** (e.g. 20 ms minimum phone, 0.8 s maximum phone, 20 ms boundary tolerance, 16 kHz target rate) are starting points, not findings.
* `speaker_id` plumbing assumes IDs are anonymised by the data owner; M2 rejects PII-style manifest columns but cannot tell whether an ID itself encodes a person.

## Speaker grouping

MFA treats each corpus folder as one speaker and may adapt acoustic features per speaker. The 3.4.3 help for `--single_speaker` says the mode *"creates multiprocessing splits based on utterances rather than speakers"* and *"also disables speaker adaptation"* (exposed here as `alignment.mfa.single_speaker`).

| Mistake | Consequence |
|---|---|
| Different speakers in one folder | adaptation is estimated on a mixture; alignments can degrade, sometimes silently |
| One speaker split across folders | adaptation has less data per folder; usually mild but noisy for short recordings |
| Unknown speakers pooled (`missing_id_policy: shared`) | the first mistake, at scale; M2 warns and records it in the manifest |
| Unknown speakers kept apart (default) | no pooling benefit, but no contamination; `alignment.mfa.single_speaker: true` is an alternative |

Which choice gives the best boundaries on a given corpus is an empirical question M2 does not answer.

## Data handling

* Originals are read-only; outputs go to a separate directory; hashes of originals and processed audio are in `corpus_manifest.json`.
* Absolute paths are not written into output records (base names + SHA-256 only), but the corpus manifest stores `source_relpath` (relative to the manifest) — avoid identifying folder names.
* Pseudo speaker IDs are deterministic hashes of `sample_id`; if `sample_id` itself identifies a person, so can the pseudo ID's provenance. Use anonymised sample IDs.
* The MIT licence in `LICENSE` is a placeholder chosen by the build. CMUdict, eSpeak NG, MFA and the models/datasets you use each have their own licences, which were **not reviewed** here — check them before redistributing derived resources.
