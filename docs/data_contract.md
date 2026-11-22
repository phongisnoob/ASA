# Data contract: M2 → M3 / M4 (schema 1.0)

Authoritative definition: `src/asa_m2/schemas.py`; JSON Schema: [`schema/sample_record.schema.json`](schema/sample_record.schema.json) (kept in sync by a test).
Times are **seconds** (`alignment_metadata.time_unit`). Unknown fields are rejected (`extra="forbid"`).

## Top level (`SampleRecord`)

| Field | Meaning |
|---|---|
| `schema_version` | `"1.0"` |
| `sample_id`, `speaker_id` | anonymised IDs; unknown speakers carry a unique pseudo-ID `SPK_UNK_<hash>` |
| `reference_text` | what the learner was **expected** to say (text-dependent tasks) |
| `human_verbatim` | what the learner **actually said** (human transcript). **Never interchangeable** with `reference_text` |
| `g2p_input_field` | which of the two the pronunciations were derived from |
| `normalization` | normaliser version, original/normalised text, dropped spans, warnings |
| `tokens[]` | one `TokenRecord` per normalised token (see below) |
| `canonical_phones[]` | flattened phones of words with a **selected** variant |
| `word_alignments[]`, `phone_alignments[]` | aligner-produced target intervals; `phone_alignments` is empty when `granularity` is `word` |
| `alignment_metadata` | `aligner`, `aligner_version`, `granularity` (`phone|word|none`), `status` (`success|partial|failed|not_run`), `model`, `command`, `notes`, `qc_summary` |
| `quality_flags[]` | `{code, severity (info|warning|error), scope, message, word_index, phone_index, start_time, end_time, details}` |
| `provenance` | M2 version, config fingerprint (path-independent), inventory id/version, G2P source versions, input **base names** + SHA-256, Python/platform, optional timestamp. No absolute paths; `alignment_metadata.command` holds a sanitized command (`<corpus>`, `<dictionary>`, `<output>`, `<abs>/name`) — the exact command is in `logs/*.command.json` |

### `TokenRecord`
`word_index`, `original_token`, `normalized_token`, `original_span{start,end}` (character offsets into the original text), `token_kind`, `normalization_rules[]`, `supported`,
`variants[]`, `selected_variant_id`, `selection_method`, `pronunciation_source`, `resolution_status` (`resolved|resolved_fallback|unresolved`), `resolution_reason`, `approved_for_alignment`, `flags[]`.

### `PronunciationVariant`
`variant_id` (`cmudict:<word>:<k>`, `custom:<word>:<k>`, `espeak-ng:<word>:1`; `k` = position in the source's listing, which carries no claim of correctness), `source`, `source_detail`,
`raw_arpabet` (exactly as given, e.g. `["DH","IH1","S"]`), `base_phones` (no digits), `stress` (`0|1|2`, `null` for consonants), `phone_ids` (inventory IDs), `ipa_display`, `approximate_mapping`.

### `selection_method`
| Value | Meaning |
|---|---|
| `sole_variant` | the source offered exactly one valid variant |
| `policy_first_listed` | explicit config `g2p.variant_policy: first_listed` (flagged; not evidence of correctness) |
| `manual_override` | custom-lexicon `!select` |
| `alignment_match` | the **aligner's phones** match exactly one candidate. Records the aligner's decision among candidates; it is **not** evidence of what the learner said |
| `unselected` | several candidates, none selected (default) — consumers must use `variants[]` |
| `none` | unresolved |

### `CanonicalPhone`
`index`, `word_index`, `token`, `variant_id`, `base_phone`, `phone_id`, `stress`, `ipa_display`, `selection_method`. Words with an unselected variant contribute **no** entries (their candidates stay in `tokens[].variants`).

### `WordAlignment` / `PhoneAlignment`
Common: `start_time`, `end_time` (`end > start ≥ 0`, enforced), `source_tier`, `interval_index` (position in the source TextGrid), `source_file` (base name), `label_role` ∈ `word|phone|silence|oov_marker|unknown`.
`PhoneAlignment` keeps three notions apart:

| Field(s) | Notion |
|---|---|
| `canonical_phone_index`, `canonical_target_phone`, `canonical_target_stress` | the phone the **canonical pronunciation expects** (only filled when the aligned phones equal the selected variant's base phones) |
| `aligned_label`, `aligned_base_phone`, `aligned_stress`, `aligned_phone_id` | the label the **aligner placed** on the interval |
| `realized_phone` | what the learner **actually produced** — **always `null`** (validator-enforced), with `realized_phone_status: "not_determined_by_forced_alignment"` |

Silence intervals (empty label, `sil`, `sp`, `<eps>`) and OOV markers (`spn`, `<unk>`) are **kept** with `label_role` `silence`/`oov_marker`; they are not inventory phones and get no `phone_id`. `phone_index` counts non-silence phone-tier entries.
Invalid intervals (negative/zero duration, negative start, non-finite) are excluded and reported (`invalid_interval_excluded`); overlapping but individually valid intervals are kept and flagged.

### `alignment_metadata.status`
`success`: usable word intervals and no error-level flag · `partial`: usable data but ≥ 1 error-level flag (e.g. missing expected word, unresolved G2P word, invalid interval) · `failed`: no usable word interval / no TextGrid · `not_run`. **`success` never means "pronounced correctly".**

### `qc_summary`
`oov_count`, `unresolved_tokens`, `oov_word_intervals`, `oov_phone_intervals`, `missing_words`, `extra_aligned_words`, `missing_phone_count` (words missing entirely count with their *shortest* candidate: a lower bound), `extra_phone_count`,
`phone_label_mismatch_count`, `unexpected_label_count`, `short/long_phone_count`, `short/long_word_count`, `invalid_interval_count`, `error_flag_count`, `warning_flag_count`.

## Quality-flag catalogue

| Code | Severity | Raised when |
|---|---|---|
| `no_tokens` | error | text produced no tokens |
| `normalization_<code>` | warning | normaliser warning: `empty_input`, `no_pronounceable_tokens`, `digit_by_digit`, `ambiguous_number` |
| `unresolved_word` | error | no pronunciation from any source (or unsupported token); blocked from alignment |
| `oov_cmudict` | warning | word not in CMUdict (resolved by lexicon/fallback) |
| `fallback_pronunciation` | warning | pronunciation from the eSpeak-ng fallback |
| `approximate_phone_mapping` | warning | an `approximate` eSpeak→ARPABET mapping was used |
| `variant_unselected` | info | several variants kept, none selected |
| `variant_chosen_by_policy` | info | `first_listed` policy applied |
| `rejected_variants` | warning | some source variants contained unknown phones and were rejected |
| `invalid_interval_excluded` | error | interval with invalid times excluded from the output |
| `missing_expected_word` | error | expected token has no aligned interval |
| `unexpected_word_label` | warning | aligned word matches no expected token |
| `word_label_mismatch` | warning | aligned label differs from the expected token |
| `aligner_oov_marker` | warning | aligner labelled an expected word `<unk>`/`spn` |
| `unexpected_phone_label` | warning | phone label not in the inventory (never remapped) |
| `word_without_phones` | warning | aligned word contains no phone intervals |
| `phones_do_not_match_any_variant` | warning | aligned phones equal no candidate; details give closest variant and counts |
| `variant_resolved_by_alignment` | info | variant selected from the aligner's phones |
| `variant_ambiguous_after_alignment` | info | several variants match the aligned phones |
| `aligned_phones_match_other_variant` | info | aligned phones match a variant other than the selected one |
| `missing_phone_tier` | warning | no phone tier: word-level output only |
| `phone_too_short/long`, `word_too_short/long` | warning | duration outside configurable limits (alignment sanity warning, **not** a pronunciation threshold) |
| `no_textgrid_produced`, `textgrid_unusable` | error | aligner produced no / an unreadable TextGrid |
| `textgrid_<issue>` | per issue | `declared_size_mismatch`, `declared_tier_count_mismatch`, `duration_mismatch`, `empty_tier`, `ends_before_tier_xmax`, `exceeds_tier_xmax`, `gap`, `invalid_tier_range`, `negative_duration`, `negative_start`, `non_finite_time`, `overlap`, `starts_after_tier_xmin`, `tier_range_mismatch`, `trailing_content`, `unsorted_intervals`, `zero_duration` |

## Phone vocabulary files (for M3)

* `phoneme_inventory.json` — full inventory, stress rules, special symbols, silence policy.
* `phone_to_id.json` / `id_to_phone.json` — `{schema_version, inventory_id, inventory_version, phone_to_id|id_to_phone: {...}}`. IDs 1–39; **0 is reserved for the CTC blank by convention only** and is absent from both maps.
* `phoneme_mapping.json` — documented mappings (eSpeak→ARPABET with `exact|approximate` confidence and a deliberately-unmapped list; MFA ARPA label policy with silence/OOV label sets).
* All four carry `inventory_id` + `inventory_version`; loaders refuse mismatches. A downstream model whose phone set differs must declare the difference (`check_compatibility`); M2 never assumes equality.
