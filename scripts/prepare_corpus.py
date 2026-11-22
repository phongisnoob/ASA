#!/usr/bin/env python
"""Thin wrapper: `python scripts/prepare_corpus.py --manifest M.csv --output DIR`.

Equivalent to `asa-m2 prepare-corpus ...` (all options are passed through).
"""
from __future__ import annotations

import sys

from asa_m2.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["prepare-corpus", *sys.argv[1:]]))
