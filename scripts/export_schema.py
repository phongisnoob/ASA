#!/usr/bin/env python
"""Write the JSON Schema of the M2 output record.

    python scripts/export_schema.py          # (re)write docs/schema + examples copy
    python scripts/export_schema.py --check  # exit 1 if the checked-in files are stale
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from asa_m2.schemas import sample_record_json_schema

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [ROOT / "docs" / "schema" / "sample_record.schema.json", ROOT / "examples" / "expected_output_schema.json"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    text = json.dumps(sample_record_json_schema(), indent=2, ensure_ascii=False) + "\n"
    stale = []
    for path in TARGETS:
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                stale.append(str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    if args.check:
        if stale:
            print("STALE: " + ", ".join(stale), file=sys.stderr)
            return 1
        print("schema files are up to date")
    else:
        print("wrote " + ", ".join(str(t) for t in TARGETS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
