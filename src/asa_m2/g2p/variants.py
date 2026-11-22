"""Shared construction of :class:`PronunciationVariant` objects from ARPABET symbols."""

from __future__ import annotations

from collections.abc import Sequence

from asa_m2.phonemes.inventory import PhoneInventory
from asa_m2.schemas import CanonicalPhone, PronunciationVariant, TokenRecord


def build_variant(
    variant_id: str,
    source: str,
    raw_symbols: Sequence[str],
    inventory: PhoneInventory,
    *,
    source_detail: str | None = None,
    approximate_mapping: bool = False,
) -> PronunciationVariant:
    """Parse ARPABET symbols against the inventory.

    The raw sequence is preserved exactly; base phones and stress are derived from it.
    Raises :class:`~asa_m2.exceptions.InventoryError` on any unknown/illegal symbol.
    """
    parsed = [inventory.parse_symbol(sym) for sym in raw_symbols]
    return PronunciationVariant(
        variant_id=variant_id,
        source=source,
        source_detail=source_detail,
        raw_arpabet=list(raw_symbols),
        base_phones=[p.base for p in parsed],
        stress=[p.stress for p in parsed],
        phone_ids=[inventory.phone_id(p.base) for p in parsed],
        ipa_display=[inventory.ipa(p.base) for p in parsed],
        approximate_mapping=approximate_mapping,
    )


def build_canonical_phones(tokens: Sequence[TokenRecord]) -> list[CanonicalPhone]:
    """Flatten the *selected* variants of ``tokens`` into the canonical phone sequence.

    Tokens without a selected variant (unresolved, or several candidates and no selection)
    contribute nothing; their candidates stay available in ``tokens[].variants``.
    """
    out: list[CanonicalPhone] = []
    for tok in tokens:
        variant = tok.selected_variant()
        if variant is None:
            continue
        for pos, base in enumerate(variant.base_phones):
            out.append(
                CanonicalPhone(
                    index=len(out), word_index=tok.word_index, token=tok.normalized_token,
                    variant_id=variant.variant_id, base_phone=base, phone_id=variant.phone_ids[pos],
                    stress=variant.stress[pos], ipa_display=variant.ipa_display[pos],
                    selection_method=tok.selection_method,
                )
            )
    return out
