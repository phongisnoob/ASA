"""G2P engine: normalisation -> pronunciation lookup -> auditable token records.

Lookup order per token
----------------------
1. custom lexicon, mode ``override``  (replaces CMUdict for that word)
2. CMUdict                              (+ custom lexicon variants when mode is ``augment``)
3. eSpeak-ng fallback                   (only when ``g2p.fallback_backend: espeak``)
4. otherwise the token is **unresolved** and blocked from forced alignment.

Variant policy
--------------
``keep_all`` (default): every valid variant is kept and *no* variant is selected when
several exist (``selection_method="unselected"``). ``first_listed`` is an explicit,
flagged opt-in. A custom-lexicon ``!select`` is a documented manual override. After
alignment the aligner's own phone choice can resolve the variant (``alignment_match``);
that records the aligner's decision, not what the learner said.
"""

from __future__ import annotations

import logging

from asa_m2.config import AppConfig
from asa_m2.exceptions import (
    G2PError,
    InventoryError,
    OptionalDependencyError,
    UnresolvedPronunciationError,
)
from asa_m2.g2p.cmudict_backend import CMUdictBackend
from asa_m2.g2p.custom_lexicon import CustomLexicon
from asa_m2.g2p.espeak_backend import EspeakBackend, EspeakFallbackError
from asa_m2.g2p.variants import build_canonical_phones, build_variant
from asa_m2.normalization.text_normalizer import (
    NORMALIZER_VERSION,
    NormalizationResult,
    NormToken,
    TextNormalizer,
)
from asa_m2.phonemes.inventory import PhoneInventory
from asa_m2.phonemes.mapping import PhoneMapping
from asa_m2.provenance import build_provenance
from asa_m2.schemas import (
    G2PResult,
    NormalizationSummary,
    PronunciationVariant,
    QualityFlag,
    SelectionMethod,
    Span,
    TokenRecord,
)

log = logging.getLogger(__name__)


class G2PEngine:
    def __init__(
        self,
        config: AppConfig,
        inventory: PhoneInventory,
        *,
        cmudict: CMUdictBackend | None = None,
        espeak: EspeakBackend | None = None,
        lexicon: CustomLexicon | None = None,
        normalizer: TextNormalizer | None = None,
    ) -> None:
        self.config = config
        self.inventory = inventory
        self.cmudict = cmudict or CMUdictBackend()
        self.lexicon = lexicon
        self.espeak = espeak
        self.normalizer = normalizer or TextNormalizer(config.normalization)
        if config.g2p.fallback_backend == "espeak":
            if self.espeak is None:
                raise OptionalDependencyError(
                    "g2p.fallback_backend is 'espeak' but no eSpeak backend was supplied; "
                    "use G2PEngine.from_config() or install espeak-ng"
                )
            if not self.espeak.is_available():
                raise OptionalDependencyError(
                    f"g2p.fallback_backend is 'espeak' but {self.espeak.executable!r} is not on PATH "
                    "(install espeak-ng, or set g2p.fallback_backend: null)"
                )
        elif self.espeak is not None:
            log.debug("eSpeak backend supplied but fallback is not configured; it will not be used")
            self.espeak = None

    # ---- construction ------------------------------------------------------------
    @classmethod
    def from_config(cls, config: AppConfig, inventory: PhoneInventory | None = None) -> G2PEngine:
        inventory = inventory or PhoneInventory.from_files(config.phonemes.inventory_path)
        lexicon = None
        if config.g2p.custom_lexicon_path:
            lexicon = CustomLexicon.from_file(
                config.g2p.custom_lexicon_path, inventory, config.g2p.custom_lexicon_mode
            )
        espeak = None
        if config.g2p.fallback_backend == "espeak":
            mapping = PhoneMapping.from_file(
                config.phonemes.mapping_path, config.g2p.espeak.mapping_name, inventory
            )
            espeak = EspeakBackend(
                inventory, mapping, executable=config.g2p.espeak.executable,
                voice=config.g2p.espeak.voice, timeout_s=config.g2p.espeak.timeout_s,
            )
        return cls(config, inventory, lexicon=lexicon, espeak=espeak)

    # ---- public API --------------------------------------------------------------
    def convert(
        self,
        text: str,
        *,
        sample_id: str | None = None,
        text_field: str = "reference_text",
    ) -> G2PResult:
        """Normalise ``text`` and attach pronunciation candidates to every token."""
        if text_field not in {"reference_text", "human_verbatim"}:
            raise G2PError(f"text_field must be 'reference_text' or 'human_verbatim', got {text_field!r}")
        norm = self.normalizer.normalize(text)
        tokens = [self._token_record(t) for t in norm.tokens]
        canonical = build_canonical_phones(tokens)
        flags = self._quality_flags(norm, tokens)
        provenance = build_provenance(
            self.config, self.inventory, g2p_sources=self.source_versions(),
        )
        return G2PResult(
            sample_id=sample_id,
            text_field=text_field,
            input_text=text,
            normalization=NormalizationSummary(
                normalizer_version=NORMALIZER_VERSION,
                original_text=norm.original_text,
                normalized_text=norm.normalized_text,
                dropped_spans=[d.__dict__ for d in norm.dropped],
                warnings=list(norm.warnings),
            ),
            tokens=tokens,
            canonical_phones=canonical,
            quality_flags=flags if self.config.output.include_quality_flags else [],
            provenance=provenance,
        )

    def source_versions(self) -> dict[str, str | None]:
        sources: dict[str, str | None] = {"cmudict": self.cmudict.source_detail}
        if self.lexicon is not None:
            sources["custom_lexicon"] = (
                f"{self.lexicon.path} sha256={self.lexicon.sha256} mode={self.lexicon.mode}"
            )
        if self.espeak is not None:
            sources["espeak-ng"] = self.espeak.source_detail
        return sources

    # ---- internals ---------------------------------------------------------------
    def _token_record(self, tok: NormToken) -> TokenRecord:
        base = dict(
            word_index=tok.index,
            original_token=tok.original_text,
            normalized_token=tok.normalized,
            original_span=Span(start=tok.span_start, end=tok.span_end),
            token_kind=tok.kind,
            normalization_rules=list(tok.rules),
            supported=tok.supported,
        )
        if not tok.supported:
            return TokenRecord(
                **base, resolution_status="unresolved",
                resolution_reason=f"unsupported_token:{tok.unsupported_reason}",
                flags=["unsupported_token"],
            )
        word = tok.normalized
        flags: list[str] = []
        variants: list[PronunciationVariant] = []
        rejected: list[str] = []

        lex_entries = self.lexicon.lookup(word) if self.lexicon else []
        in_cmudict = word in self.cmudict
        if not in_cmudict:
            flags.append("oov_cmudict")

        if lex_entries and self.lexicon is not None and self.lexicon.mode == "override":
            variants += self._variants("custom", "custom_lexicon", word, lex_entries,
                                       f"custom_lexicon:{self.lexicon.path}", rejected)
            flags.append("custom_lexicon_override")
        else:
            variants += self._variants("cmudict", "cmudict", word, self.cmudict.lookup(word),
                                       self.cmudict.source_detail, rejected)
            if lex_entries and self.lexicon is not None:
                variants += self._variants("custom", "custom_lexicon", word, lex_entries,
                                           f"custom_lexicon:{self.lexicon.path}", rejected,
                                           start=len(variants) + 1)
                flags.append("custom_lexicon_augment")
        if rejected:
            flags.append("rejected_variants")

        fallback_reason: str | None = None
        if not variants and self.espeak is not None:
            try:
                variants.append(self.espeak.pronounce(word))
                flags.append("fallback_pronunciation")
                if variants[0].approximate_mapping:
                    flags.append("approximate_phone_mapping")
            except EspeakFallbackError as exc:
                fallback_reason = f"{exc.reason}:{exc.message}"
                flags.append("fallback_failed")

        if not variants:
            reason = fallback_reason or (
                "oov_no_pronunciation" if not rejected else "all_variants_rejected:" + "; ".join(rejected)
            )
            return TokenRecord(**base, resolution_status="unresolved", resolution_reason=reason, flags=flags)

        sources = sorted({v.source for v in variants})
        status = "resolved_fallback" if sources == ["espeak-ng"] else "resolved"
        selected, method = self._select(word, variants, flags)
        return TokenRecord(
            **base, variants=variants, selected_variant_id=selected, selection_method=method,
            pronunciation_source="+".join(sources), resolution_status=status,
            resolution_reason=None, approved_for_alignment=True, flags=flags,
        )

    def _variants(
        self, id_prefix: str, source: str, word: str, entries: list[list[str]], detail: str,
        rejected: list[str], start: int = 1,
    ) -> list[PronunciationVariant]:
        out: list[PronunciationVariant] = []
        for offset, symbols in enumerate(entries):
            vid = f"{id_prefix}:{word}:{start + offset}"
            try:
                out.append(build_variant(vid, source, symbols, self.inventory, source_detail=detail))
            except InventoryError as exc:
                rejected.append(f"{vid}: {exc.message}")
        return out

    def _select(
        self, word: str, variants: list[PronunciationVariant], flags: list[str]
    ) -> tuple[str | None, SelectionMethod]:
        manual = self.lexicon.selection(word) if self.lexicon else None
        if manual is not None:
            if manual > len(variants):
                raise G2PError(
                    f"custom lexicon '!select {word} {manual}' but only {len(variants)} variant(s) exist"
                )
            flags.append("variant_manual_override")
            return variants[manual - 1].variant_id, "manual_override"
        if len(variants) == 1:
            return variants[0].variant_id, "sole_variant"
        if self.config.g2p.variant_policy == "first_listed":
            flags.append("variant_chosen_by_policy")
            return variants[0].variant_id, "policy_first_listed"
        flags.append("variant_unselected")
        return None, "unselected"

    def _quality_flags(self, norm: NormalizationResult, tokens: list[TokenRecord]) -> list[QualityFlag]:
        flags: list[QualityFlag] = []
        if not tokens:
            flags.append(QualityFlag(code="no_tokens", severity="error", scope="sample",
                                     message="text produced no tokens"))
        for warning in norm.warnings:
            code = warning.split(":", 1)[0]
            flags.append(QualityFlag(code=f"normalization_{code}", severity="warning", scope="sample",
                                     message=warning))
        for tok in tokens:
            kw = {"scope": "word", "word_index": tok.word_index}
            if tok.resolution_status == "unresolved":
                flags.append(QualityFlag(
                    code="unresolved_word", severity="error",
                    message=f"{tok.original_token!r} is unresolved ({tok.resolution_reason}) and is blocked "
                            "from forced alignment", details={"reason": tok.resolution_reason}, **kw))
                continue
            if "oov_cmudict" in tok.flags:
                flags.append(QualityFlag(code="oov_cmudict", severity="warning",
                                         message=f"{tok.normalized_token!r} is not in CMUdict", **kw))
            if "fallback_pronunciation" in tok.flags:
                flags.append(QualityFlag(code="fallback_pronunciation", severity="warning",
                                         message="pronunciation comes from the eSpeak-ng fallback (lower confidence than CMUdict)", **kw))
            if "approximate_phone_mapping" in tok.flags:
                flags.append(QualityFlag(code="approximate_phone_mapping", severity="warning",
                                         message="an approximate eSpeak->ARPABET mapping was used", **kw))
            if "variant_unselected" in tok.flags:
                flags.append(QualityFlag(code="variant_unselected", severity="info",
                                         message=f"{len(tok.variants)} variants kept; none selected", **kw))
            if "variant_chosen_by_policy" in tok.flags:
                flags.append(QualityFlag(code="variant_chosen_by_policy", severity="info",
                                         message="first listed variant selected by explicit policy; not evidence of correctness", **kw))
            if "rejected_variants" in tok.flags:
                flags.append(QualityFlag(code="rejected_variants", severity="warning",
                                         message="some source variants were rejected (unknown phones)", **kw))
        return flags


def ensure_alignable(result: G2PResult, *, allow_unresolved: bool = False) -> None:
    """Gate before forced alignment.

    Raises :class:`UnresolvedPronunciationError` listing every unresolved word unless the
    caller explicitly passes ``allow_unresolved=True`` (an explicit override).
    """
    if not result.tokens:
        raise UnresolvedPronunciationError("text has no tokens; nothing to align")
    bad = result.unresolved_tokens
    if bad and not allow_unresolved:
        words = [f"{t.original_token!r} ({t.resolution_reason})" for t in bad]
        raise UnresolvedPronunciationError(
            f"{len(bad)} unresolved word(s) cannot enter forced alignment without an override or "
            "approved fallback: " + ", ".join(words),
            details={"words": [t.original_token for t in bad]},
        )
