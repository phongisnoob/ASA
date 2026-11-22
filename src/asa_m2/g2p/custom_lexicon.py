"""User-defined pronunciation lexicon with explicit override semantics.

File format (UTF-8, one entry per line)::

    # comment (';;;' also accepted)
    hanoi    HH AE0 N OY1          # several lines for one word = several variants
    hanoi    HH AA0 N OY1
    !select hanoi 2                # optional: manually select variant #2 of the FINAL list

* Phones are ARPABET with optional stress digits and are validated against the
  inventory; errors carry file and line numbers.
* ``mode: override`` replaces CMUdict variants for the listed words; ``augment`` appends
  the lexicon variants after them.
* ``!select`` makes a *manual override* of the variant choice (index is 1-based over the
  final candidate list for that word).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from asa_m2.exceptions import InventoryError, LexiconError
from asa_m2.phonemes.inventory import PhoneInventory

_WORD_RE = re.compile(r"^[a-z0-9][a-z0-9'\-]*$")
_VARIANT_SUFFIX_RE = re.compile(r"\(\d+\)$")


class CustomLexicon:
    def __init__(self, path: str | None, mode: str = "override") -> None:
        if mode not in {"override", "augment"}:
            raise LexiconError(f"lexicon mode must be 'override' or 'augment', got {mode!r}")
        self.path = path
        self.mode = mode
        self.entries: dict[str, list[list[str]]] = {}
        self.selections: dict[str, int] = {}
        self.sha256: str | None = None

    @classmethod
    def from_file(cls, path: str | Path, inventory: PhoneInventory, mode: str = "override") -> CustomLexicon:
        p = Path(path)
        try:
            raw = p.read_bytes()
        except FileNotFoundError:
            raise LexiconError(f"custom lexicon not found: {p}") from None
        lex = cls(p.name, mode)
        lex.sha256 = hashlib.sha256(raw).hexdigest()
        lex._parse(raw.decode("utf-8"), inventory)
        return lex

    @classmethod
    def from_text(cls, text: str, inventory: PhoneInventory, mode: str = "override") -> CustomLexicon:
        lex = cls("<inline>", mode)
        lex._parse(text, inventory)
        return lex

    def _parse(self, text: str, inventory: PhoneInventory) -> None:
        errors: list[str] = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            content = line.split("#", 1)[0].strip()
            if not content or content.startswith(";;;"):
                continue
            parts = content.split()
            if parts[0] == "!select":
                if len(parts) != 3 or not parts[2].isdigit() or int(parts[2]) < 1:
                    errors.append(f"{self.path}:{lineno}: '!select' needs '!select <word> <n>' with n >= 1")
                else:
                    self.selections[parts[1].lower()] = int(parts[2])
                continue
            word = _VARIANT_SUFFIX_RE.sub("", parts[0].lower())
            if not _WORD_RE.match(word):
                errors.append(f"{self.path}:{lineno}: invalid word {parts[0]!r}")
                continue
            symbols = parts[1:]
            if not symbols:
                errors.append(f"{self.path}:{lineno}: word {word!r} has no phones")
                continue
            try:
                for sym in symbols:
                    inventory.parse_symbol(sym)
            except InventoryError as exc:
                errors.append(f"{self.path}:{lineno}: {exc.message}")
                continue
            self.entries.setdefault(word, []).append(symbols)
        if errors:
            raise LexiconError("invalid custom lexicon:\n  - " + "\n  - ".join(errors), details={"errors": errors})

    def __contains__(self, word: object) -> bool:
        return isinstance(word, str) and word.lower() in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    def lookup(self, word: str) -> list[list[str]]:
        return [list(v) for v in self.entries.get(word.lower(), [])]

    def selection(self, word: str) -> int | None:
        return self.selections.get(word.lower())
