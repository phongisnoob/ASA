# Troubleshooting

## Exit codes

| Code | Meaning | Typical cause |
|---|---|---|
| 0 | success | — |
| 1 | internal error | a bug: re-run with `--log-level DEBUG --log-file run.log` and report |
| 2 | usage error | unknown command/option, missing required argument, invalid `ASA_M2_LOG_LEVEL` |
| 3 | configuration error | bad/missing YAML, unknown key (typos are rejected), missing input file/option combination |
| 4 | input/data error | unresolved OOV words, invalid manifest/corpus/audio, existing output without `--overwrite` |
| 5 | external tool error | MFA missing / broken / unsupported version / non-zero exit / timeout; required eSpeak missing |
| 6 | alignment error | TextGrid unreadable or with errors; one or more samples not `success` |
| 7 | optional dependency | eSpeak fallback requested but unavailable; Qwen adapter dependencies missing or out of memory |

Logs go to **stderr**, results to **stdout**. `--log-file` writes JSON lines; `--json-logs` makes stderr JSON.

## `asa-m2 check` says something is missing

| Line | Meaning / action |
|---|---|
| `dependency:… FAIL` | `pip install -r requirements.txt` |
| `config / inventory FAIL` (exit 3) | the message names the YAML/JSON problem; unknown config keys are errors |
| `espeak-ng missing (optional)` | only needed with `--fallback espeak`; `sudo apt-get install espeak-ng` |
| `mfa missing (optional)` | install MFA with conda (README §8); `asa-m2 check --require mfa` exits 5 if absent |
| `mfa: mfa version failed … No module named '_kalpy'` | MFA was installed with pip only. Kaldi/kalpy come from conda-forge: `conda install -c conda-forge montreal-forced-aligner` in a clean conda environment |
| `mfa: unsupported MFA version X` | install a 3.4.x release, or try `--allow-unverified-mfa` / `alignment.mfa.allow_unverified_versions: true` (unsupported; commands may differ) |

## MFA

| Symptom | Cause / fix |
|---|---|
| `alignment.mfa.acoustic_model is required` (exit 3) | pass `--acoustic-model <name|path>` or set it in YAML. asa-m2 never guesses a model. Download with `mfa model download acoustic <name>` (MFA names are MFA's) |
| `exited with code N` (exit 5) | read `<out>/logs/mfa_align*.stderr.log` and `*.command.json` (exact command + MFA config). Re-run the command by hand to reproduce |
| `exited with code 0 but K expected TextGrid(s) are missing` | exit 0 is *not* proof of success. Check the stdout log for skipped files (unsupported audio, empty transcript, OOV handling). Samples without TextGrid get a `failed` JSON record (`no_textgrid_produced`) |
| aligner OOV markers (`spn`, `<unk>`) | the dictionary lacks the word or the dictionary and acoustic model disagree. `validate-corpus` flags `word_missing_from_dictionary`; compare phone styles (stress digits) with `--acoustic-phones` |
| `dictionary_phone_not_in_acoustic_model` | the model has no stress digits (or different symbols). Set `phonemes.preserve_stress: false` or use a matching model; M2 does not remap phones |
| timeouts | raise `alignment.mfa.timeout_s`; use `--num-jobs`. Partial logs are saved |
| `alignment output … is not empty` | pass `--overwrite` (MFA's own `--overwrite` is passed accordingly; without it `--no_overwrite` is passed) |
| very short phones / many `phone_too_short` | alignment sanity warnings, not pronunciation verdicts; tune `alignment.qc.*` or inspect the audio (noise, wrong sample rate, wrong language model) |
| `align_hf` | Hugging Face models: `--mode hf --hf-model-id <id>`. `--ignore_oovs` is not an `align_hf` option in the verified CLI |

## G2P and OOV

* *Unresolved word* → choose one: add it to a custom lexicon (`--lexicon`), opt in to the eSpeak fallback (`--fallback espeak`), correct the text, or set `g2p.fail_on_unresolved_oov: false` to **exclude** the affected samples (recorded in `corpus_manifest.json → excluded`).
* `unsupported_token:mixed_letters_and_digits` / `url_not_supported` / `unsupported_symbol` / `dotted_acronym_reading_ambiguous` → rewrite the reference text in words (`3D` → "three D"), or add a lexicon entry for the normalised token.
* eSpeak fallback `espeak_unmapped_phone` → the word contains a unit that cannot be mapped without guessing (flap, glottal stop, …). Add a lexicon entry.
* Words with a possessive/clitic missing from CMUdict (`john's`) are OOV by design; add them to the lexicon.
* Several variants are *expected* (6.7 % of CMUdict entries have more than one). They are not an error. `variant_unselected` is informational.

## TextGrid

* `no word tier found … tiers present: [...]` → tier names are not assumed. Use `asa-m2 inspect-textgrid`, then `--word-tier/--phone-tier` or `alignment.textgrid.word_tier_names`.
* `several speaker-prefixed word tiers found` → the file holds several speakers; choose one tier explicitly.
* `overlap`, `negative_duration`, `zero_duration` → the file is malformed (often a hand-edited annotation). The intervals concerned are excluded from the JSON and flagged; the original file is untouched.
* `duration_mismatch (… milliseconds …)` → times in the wrong unit, or the TextGrid belongs to different audio.
* `declared_size_mismatch` → header and content disagree; usually a truncated or hand-edited file.

## Audio and corpus

* `audio_unreadable … truncated` → the WAV header promises more data than the file contains (interrupted copy/recording). libsndfile alone would silently accept it; M2 refuses it.
* `audio_digital_silence`, `audio_duration_too_short/long`, `audio_sample_rate_too_low` → fix or exclude the recording (`corpus.on_error: skip`).
* A sample-rate *warning* (`sample_rate_differs`) is informational: the audio is genuinely resampled to `audio.target_sample_rate`. Set that value from your acoustic model's requirements (16 kHz is a default, not an assumption).
* **Speaker grouping.** MFA treats each folder as one speaker and adapts to it. Missing IDs get unique pseudo-IDs (no pooling); `speaker.grouping: per_recording` isolates every recording; `missing_id_policy: shared` pools unrelated recordings (warned). See `docs/research_limitations.md`.
* `unsafe_sample_id` / `unsafe_speaker_id` → identifiers may only contain `A-Z a-z 0-9 _ . -`, start with a letter/digit, no `..`.
* `manifest contains personally-identifying column(s)` → remove `name`, `student_id`, `address`, … ; use an anonymised `speaker_id`.
* `already contains a prepared corpus` / `already exists` → pass `--overwrite`. Only M2-owned paths (`mfa_corpus/`, `dictionary/`, `g2p/`, `corpus_manifest.json`) are replaced, and only if no source recording lives inside them.

## Optional Qwen adapter

* `needs pip install qwen-asr` (exit 7): the optional dependency is absent. Core M2 does not need it.
* `would be altered by Qwen's tokenisation` → a token contains characters Qwen strips (e.g. a hyphen). The normaliser splits hyphens by default; check `normalization.hyphen_policy`.
* `returned N units for M words` → the aligner's tokenisation disagreed with ours; M2 refuses to guess a mapping.
* `not enough GPU memory` / CUDA OOM → use `--device cpu`, `alignment.qwen.dtype: float32`, or shorter audio (limit ≈ 5 min).
* Live integration of this adapter was **not verified** by the build.

## Reproducing a problem

```bash
asa-m2 check --json            # redirect to check.json to attach it
asa-m2 g2p --text "I think this is important." --log-level DEBUG --log-file run.log   # any command accepts these flags
```
Attach `check.json`, `run.log`, the `logs/*.command.json` of an MFA run and (if shareable) the TextGrid.
