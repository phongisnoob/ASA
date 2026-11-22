#!/usr/bin/env python
"""Validate the phone-inventory JSON files (and optionally an acoustic model's phone list).

Usage:
    python scripts/validate_inventory.py [--dir configs] [--acoustic-phones phones.txt]

Exit code 0 = valid, 4 = invalid. The checks are the same ones asa-m2 runs at start-up
(duplicate IDs/symbols, stress separation, CTC blank handling, cross-file consistency,
mapping targets). ``--acoustic-phones`` compares the canonical base-phone set with a
plain-text phone list (one phone per line); obtain one from your acoustic model, e.g. with
`mfa model inspect acoustic <model>` (command verified to exist in MFA 3.4.3).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from asa_m2.exceptions import M2Error
from asa_m2.phonemes.inventory import PhoneInventory, phone_set_from_lines
from asa_m2.phonemes.mapping import AlignerLabelMapper, PhoneMapping


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parents[1] / "configs"),
                    help="directory with phoneme_inventory.json, phone_to_id.json, id_to_phone.json, phoneme_mapping.json")
    ap.add_argument("--acoustic-phones", help="text file with the acoustic model's phone set")
    args = ap.parse_args(argv)
    d = Path(args.dir)
    try:
        inv = PhoneInventory.from_files(d / "phoneme_inventory.json", d / "phone_to_id.json", d / "id_to_phone.json")
        print(f"OK inventory {inv.inventory_id} v{inv.inventory_version}: {len(inv)} phones, CTC blank id {inv.ctc_blank_id} (not a phoneme)")
        mapping = PhoneMapping.from_file(d / "phoneme_mapping.json", "espeak_en_us_ipa", inv)
        print(f"OK mapping espeak_en_us_ipa: {len(mapping)} units, {len(mapping.unmapped_by_design)} deliberately unmapped")
        mapper = AlignerLabelMapper.from_mapping_file(d / "phoneme_mapping.json", inv)
        print(f"OK aligner labels: silence={sorted(mapper.silence_labels)} oov={sorted(mapper.oov_labels)}")
        if args.acoustic_phones:
            phones = phone_set_from_lines(Path(args.acoustic_phones).read_text(encoding="utf-8").splitlines())
            rep = inv.check_compatibility(phones, other_name="acoustic model")
            print(f"{'OK' if rep.compatible else 'DIFF'} compatibility with {args.acoustic_phones}: "
                  f"missing in model={list(rep.missing_in_other)} extra in model={list(rep.extra_in_other)}")
    except M2Error as exc:
        print(f"INVALID: {exc.message}", file=sys.stderr)
        return exc.exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
