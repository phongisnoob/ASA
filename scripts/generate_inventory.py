#!/usr/bin/env python
"""Generate the four versioned phone-inventory JSON files.

This script is the single place where the inventory *data* is authored. It writes the
files to ``configs/`` and mirrors them into ``src/asa_m2/resources/`` (the packaged
copy used when asa-m2 is installed without the repository checkout).

Usage::

    python scripts/generate_inventory.py            # write files
    python scripts/generate_inventory.py --check    # exit 1 if files on disk differ

ID POLICY (stable IDs)
    IDs 1..39 are assigned alphabetically over the 39 CMUdict base phones. They are
    *frozen*: a future inventory version may only APPEND new IDs, never renumber.
    ID 0 is reserved for the CTC blank *by convention only*; the blank is not a
    phoneme, is not silence, and is therefore absent from phone_to_id/id_to_phone.

ARTICULATORY METADATA
    Compiled from standard phonetics descriptions of General American English. It is
    a convenience for feature-based analysis, not an authoritative linguistic claim;
    individual dialects differ.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVENTORY_ID = "asa-arpabet-cmu39"
INVENTORY_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0"

# (arpabet, ipa, ipa_variants, height, backness, rounding, diphthong, rhotic)
VOWELS = [
    ("AA", "ɑ", [], "open", "back", "unrounded", False, False),
    ("AE", "æ", [], "near-open", "front", "unrounded", False, False),
    ("AH", "ʌ", ["ə"], "open-mid", "back", "unrounded", False, False),
    ("AO", "ɔ", [], "open-mid", "back", "rounded", False, False),
    ("AW", "aʊ", [], "open", "front-to-back", "unrounded-to-rounded", True, False),
    ("AY", "aɪ", [], "open", "front-to-front", "unrounded", True, False),
    ("EH", "ɛ", [], "open-mid", "front", "unrounded", False, False),
    ("ER", "ɝ", ["ɚ"], "mid", "central", "unrounded", False, True),
    ("EY", "eɪ", [], "close-mid", "front-to-front", "unrounded", True, False),
    ("IH", "ɪ", [], "near-close", "near-front", "unrounded", False, False),
    ("IY", "i", [], "close", "front", "unrounded", False, False),
    ("OW", "oʊ", [], "close-mid", "back", "rounded", True, False),
    ("OY", "ɔɪ", [], "open-mid", "back-to-front", "rounded-to-unrounded", True, False),
    ("UH", "ʊ", [], "near-close", "near-back", "rounded", False, False),
    ("UW", "u", [], "close", "back", "rounded", False, False),
]

# (arpabet, ipa, place, manner, voiced)
CONSONANTS = [
    ("B", "b", "bilabial", "plosive", True),
    ("CH", "tʃ", "postalveolar", "affricate", False),
    ("D", "d", "alveolar", "plosive", True),
    ("DH", "ð", "dental", "fricative", True),
    ("F", "f", "labiodental", "fricative", False),
    ("G", "ɡ", "velar", "plosive", True),
    ("HH", "h", "glottal", "fricative", False),
    ("JH", "dʒ", "postalveolar", "affricate", True),
    ("K", "k", "velar", "plosive", False),
    ("L", "l", "alveolar", "lateral-approximant", True),
    ("M", "m", "bilabial", "nasal", True),
    ("N", "n", "alveolar", "nasal", True),
    ("NG", "ŋ", "velar", "nasal", True),
    ("P", "p", "bilabial", "plosive", False),
    ("R", "ɹ", "alveolar", "approximant", True),
    ("S", "s", "alveolar", "fricative", False),
    ("SH", "ʃ", "postalveolar", "fricative", False),
    ("T", "t", "alveolar", "plosive", False),
    ("TH", "θ", "dental", "fricative", False),
    ("V", "v", "labiodental", "fricative", True),
    ("W", "w", "labial-velar", "approximant", True),
    ("Y", "j", "palatal", "approximant", True),
    ("Z", "z", "alveolar", "fricative", True),
    ("ZH", "ʒ", "postalveolar", "fricative", True),
]

# eSpeak-ng (`espeak-ng -q -v en-us --ipa --sep=_`) phoneme units -> ARPABET base phones.
# Derived from the units actually emitted over a 12,000-word CMUdict sample (see
# docs/research_limitations.md). confidence: "exact" = standard correspondence;
# "approximate" = documented simplification the pipeline flags on every use.
ESPEAK_ENTRIES: dict[str, tuple[list[str], str, str]] = {
    # consonants
    "b": (["B"], "exact", ""), "d": (["D"], "exact", ""), "f": (["F"], "exact", ""),
    "ɡ": (["G"], "exact", "U+0261 LATIN SMALL LETTER SCRIPT G"),
    "h": (["HH"], "exact", ""), "j": (["Y"], "exact", ""), "k": (["K"], "exact", ""),
    "l": (["L"], "exact", ""), "m": (["M"], "exact", ""), "n": (["N"], "exact", ""),
    "ŋ": (["NG"], "exact", ""), "p": (["P"], "exact", ""), "ɹ": (["R"], "exact", ""),
    "s": (["S"], "exact", ""), "ʃ": (["SH"], "exact", ""), "t": (["T"], "exact", ""),
    "θ": (["TH"], "exact", ""), "ð": (["DH"], "exact", ""), "v": (["V"], "exact", ""),
    "w": (["W"], "exact", ""), "z": (["Z"], "exact", ""), "ʒ": (["ZH"], "exact", ""),
    "tʃ": (["CH"], "exact", "affricate emitted as one unit"),
    "dʒ": (["JH"], "exact", "affricate emitted as one unit"),
    # syllabic consonants -> reduced vowel + consonant (CMUdict convention AH0 + C)
    "əl": (["AH", "L"], "approximate", "syllabic /l/ expanded as AH + L"),
    "n̩": (["AH", "N"], "approximate", "syllabic /n/ (n + U+0329) expanded as AH + N"),
    # vowels
    "æ": (["AE"], "exact", ""), "ɛ": (["EH"], "exact", ""), "ɪ": (["IH"], "exact", ""),
    "i": (["IY"], "exact", "happy-vowel / word-final i"), "iː": (["IY"], "exact", ""),
    "ɑː": (["AA"], "exact", ""), "ɑ": (["AA"], "exact", ""),
    "ɔ": (["AO"], "exact", ""), "ɔː": (["AO"], "exact", ""), "oː": (["AO"], "exact",
        "followed by a separate ɹ unit in rhotic contexts (e.g. 'boring' b_ˈoː_ɹ_ɪ_ŋ)"),
    "o": (["AO"], "approximate", "bare 'o' occurs rarely (e.g. 'tolkien'); mapped to AO"),
    "ʊ": (["UH"], "exact", ""), "uː": (["UW"], "exact", ""),
    "ʌ": (["AH"], "exact", ""), "ɐ": (["AH"], "approximate", "near-open central; mapped to AH"),
    "ə": (["AH"], "exact", "schwa; stress 0 unless a stress mark precedes it"),
    "ᵻ": (["IH"], "approximate",
          "eSpeak reduced vowel between ɪ and ə; CMUdict uses IH0 or AH0 depending on word"),
    "ɜː": (["ER"], "exact", ""), "ɚ": (["ER"], "exact", ""),
    "eɪ": (["EY"], "exact", ""), "aɪ": (["AY"], "exact", ""), "ɔɪ": (["OY"], "exact", ""),
    "aʊ": (["AW"], "exact", ""), "oʊ": (["OW"], "exact", ""),
    # r-coloured vowel units (vowel + ɹ emitted as one unit)
    "ɑːɹ": (["AA", "R"], "exact", ""), "ɑɹ": (["AA", "R"], "exact", ""),
    "ɔːɹ": (["AO", "R"], "exact", ""), "oːɹ": (["AO", "R"], "exact", ""),
    "ɛɹ": (["EH", "R"], "exact", ""), "ɪɹ": (["IH", "R"], "exact", ""),
    "ʊɹ": (["UH", "R"], "exact", ""),
    # multi-vowel units
    "iə": (["IY", "AH"], "approximate", "as in 'lithium', 'gregarious'"),
    "aɪɚ": (["AY", "ER"], "approximate", "as in 'fire'; CMUdict also lists AY R variants"),
    "aɪə": (["AY", "AH"], "approximate", "as in 'diet'"),
}

ESPEAK_UNMAPPED_BY_DESIGN: dict[str, str] = {
    "ɾ": "alveolar flap: allophone of /t/ or /d/ ('platter' vs 'ladder'); mapping to T or D "
         "would be a guess, so words containing it are reported as unresolved",
    "ʔ": "glottal stop: no CMUdict phone (CMUdict writes T, as in 'shorten'); not guessed",
    "x": "voiceless velar fricative: not part of the English ARPABET inventory",
    "ɬ": "voiceless lateral fricative: not part of the English ARPABET inventory",
    "r": "alveolar trill: not equivalent to ARPABET R (ɹ); not guessed",
}

MFA_SILENCE_LABELS = ["", "sil", "sp", "<eps>"]
MFA_OOV_LABELS = ["spn", "<unk>"]


def build() -> dict[str, dict]:
    phones: list[dict] = []
    all_base = sorted([v[0] for v in VOWELS] + [c[0] for c in CONSONANTS])
    ids = {sym: i for i, sym in enumerate(all_base, start=1)}
    for arp, ipa, variants, height, back, rounding, diph, rhotic in VOWELS:
        phones.append({
            "id": ids[arp], "arpabet": arp, "ipa": ipa, "ipa_variants": variants,
            "category": "vowel", "stress_bearing": True,
            "articulatory": {"height": height, "backness": back, "rounding": rounding,
                             "diphthong": diph, "rhotic": rhotic},
        })
    for arp, ipa, place, manner, voiced in CONSONANTS:
        phones.append({
            "id": ids[arp], "arpabet": arp, "ipa": ipa, "ipa_variants": [],
            "category": "consonant", "stress_bearing": False,
            "articulatory": {"place": place, "manner": manner, "voiced": voiced},
        })
    phones.sort(key=lambda p: p["id"])

    meta = {"schema_version": SCHEMA_VERSION, "inventory_id": INVENTORY_ID,
            "inventory_version": INVENTORY_VERSION}
    inventory = {
        **meta,
        "description": "Canonical English phone inventory for ASA M2/M3/M4 (CMUdict ARPABET, "
                       "39 base phones). Stress is stored separately from phone identity.",
        "canonical_representation": "ARPABET",
        "display_representation": "IPA (General American; display only)",
        "id_policy": {
            "assignment": "alphabetical over ARPABET base phones, frozen at 1.0.0",
            "evolution": "append-only: new phones receive new IDs; existing IDs never change",
            "reserved_ids": {"0": "CTC blank by convention (not a phoneme, not silence)"},
        },
        "stress": {
            "values": [0, 1, 2],
            "labels": {"0": "unstressed", "1": "primary", "2": "secondary"},
            "applies_to": "vowel",
            "separate_from_phone_identity": True,
            "symbol_pattern": "^([A-Z]+)([012])?$",
            "note": "ARPABET stress digits are parsed off the symbol (AY1 -> AY + 1). "
                    "Phone IDs identify the base phone only.",
        },
        "special_symbols": [
            {"symbol": "<blank>", "id": 0, "role": "ctc_blank", "is_phoneme": False,
             "is_silence": False,
             "description": "CTC blank index by convention. Not a phoneme and not automatically "
                            "silence. M3 may relocate it in its output layer; if so it must "
                            "document the offset."},
        ],
        "silence_policy": {
            "has_silence_class": False,
            "note": "No silence class is defined because the downstream acoustic model's "
                    "support is unknown. Aligner silence labels (sil, sp, <eps>, empty) are "
                    "kept as label roles in alignment output, never as inventory phones.",
        },
        "phones": phones,
    }
    phone_to_id = {**meta, "phone_to_id": {p["arpabet"]: p["id"] for p in phones}}
    id_to_phone = {**meta, "id_to_phone": {str(p["id"]): p["arpabet"] for p in phones}}

    mapping = {
        **meta,
        "mappings": {
            "espeak_en_us_ipa": {
                "description": "eSpeak-ng IPA phoneme units -> canonical ARPABET base phones.",
                "source_command": "espeak-ng -q -v en-us --ipa --sep=_ -- <word>",
                "verified_with": "espeak-ng 1.51 (Ubuntu 24.04 package), en-us voice",
                "stress_marks": {"primary": "ˈ", "secondary": "ˌ"},
                "stress_assignment": (
                    "The stress mark is stripped from the unit it prefixes. Primary -> 1, "
                    "secondary -> 2, applied to the FIRST vowel in the unit's ARPABET list; any "
                    "further vowels in the same unit and all unmarked vowels get stress 0."
                ),
                "entries": {
                    unit: {"arpabet": arp, "confidence": conf, **({"note": note} if note else {})}
                    for unit, (arp, conf, note) in ESPEAK_ENTRIES.items()
                },
                "unmapped_by_design": ESPEAK_UNMAPPED_BY_DESIGN,
            },
            "mfa_english_us_arpa": {
                "description": "Labels of MFA ARPABET dictionaries/models -> canonical phones.",
                "label_policy": "identity_after_stress_parse",
                "note": "Verified against the published english_us_arpa dictionary v2.0.0: phone "
                        "labels are ARPABET with stress digits on vowels (e.g. IH0, AO1) plus "
                        "'spn' for OOV. Labels not in the inventory are reported, never remapped.",
                "silence_labels": MFA_SILENCE_LABELS,
                "oov_labels": MFA_OOV_LABELS,
            },
        },
    }
    return {
        "phoneme_inventory.json": inventory,
        "phone_to_id.json": phone_to_id,
        "id_to_phone.json": id_to_phone,
        "phoneme_mapping.json": mapping,
    }


def render(obj: dict) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if files differ from generator")
    args = parser.parse_args(argv)
    targets = [ROOT / "configs", ROOT / "src" / "asa_m2" / "resources"]
    stale = []
    for name, obj in build().items():
        text = render(obj)
        for directory in targets:
            path = directory / name
            if args.check:
                if not path.exists() or path.read_text(encoding="utf-8") != text:
                    stale.append(str(path))
            else:
                directory.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
    if args.check:
        if stale:
            print("STALE:\n  " + "\n  ".join(stale), file=sys.stderr)
            return 1
        print("inventory files are up to date")
        return 0
    print(f"wrote {len(build())} files to {', '.join(str(t) for t in targets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
