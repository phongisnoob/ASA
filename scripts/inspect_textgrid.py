#!/usr/bin/env python
"""Thin wrapper: `python scripts/inspect_textgrid.py FILE.TextGrid [--json]`.

Equivalent to `asa-m2 inspect-textgrid --input FILE.TextGrid`.
"""
from __future__ import annotations

import sys

from asa_m2.cli import main

if __name__ == "__main__":
    args = sys.argv[1:]
    if args and not args[0].startswith("-"):
        args = ["--input", args[0], *args[1:]]
    raise SystemExit(main(["inspect-textgrid", *args]))
