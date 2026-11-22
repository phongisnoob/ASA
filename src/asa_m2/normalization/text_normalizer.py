"""Auditable text normaliser.

Contract
--------
* The original text is never modified; it is stored on the result.
* Every normalised token records the *character span* it came from, the rules applied
  and, for one-to-many expansions (``"21"`` -> ``twenty one``), its position within the
  expansion. A token is therefore always traceable to the original text.
* Nothing is silently deleted. Punctuation that is not pronounced is listed in
  ``dropped``; things the normaliser cannot handle (URLs, ``3D``, ``$``, emoji ...) stay
  in the token stream flagged ``supported=False`` so that G2P reports them as unresolved.
* :meth:`NormalizationResult.verify` proves the "nothing lost" invariant.

Numbers are expanded with ONE reading (cardinal, digit-by-digit for leading zeros). A
number can have several valid readings (``1999`` as a year, ``101`` as a course), so
ambiguous cases raise a warning; prefer spelled-out numbers in read-aloud reference text.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field

from asa_m2.config import NormalizationConfig
from asa_m2.exceptions import NormalizationError

NORMALIZER_VERSION = "1.0.0"

_APOSTROPHES = "'\u2019\u02bc"
_DROPPABLE_PUNCT = set(".,!?;:\"()[]{}\u2026\u2014\u2013-\u201c\u201d\u2018\u00ab\u00bb\u00bf\u00a1'\u2019\u02bc")

_TOKEN_RE = re.compile(
    r"""
    (?P<url>(?:https?://|www\.)\S+)
  | (?P<email>[^\s@]+@[^\s@]+\.[^\s@]+)
  | (?P<dotted>(?<![^\W\d_])(?:[^\W\d_]\.){2,})
  | (?P<ordinal>(?<![^\W_])\d+(?:st|nd|rd|th)(?![^\W_]))
  | (?P<number>(?<![^\W_])\d+(?:,\d{3})*(?:\.\d+)?(?![^\W_]))
  | (?P<run>[^\W_]+(?:['\u2019\u02bc\-][^\W_]+)*)
  | (?P<other>\S)
    """,
    re.VERBOSE | re.IGNORECASE,
)

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
         "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (10**3, "thousand")]
_MAX_NUMBER = 10**15
_ORDINAL_IRREGULAR = {"one": "first", "two": "second", "three": "third", "five": "fifth",
                      "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}


def number_to_words(n: int) -> list[str]:
    """Cardinal reading of a non-negative integer below 10**15 (US style, no 'and')."""
    if n < 0 or n >= _MAX_NUMBER:
        raise NormalizationError(f"number {n} outside supported range 0..{_MAX_NUMBER - 1}")
    if n < 20:
        return [_ONES[n]]
    if n < 100:
        tens, ones = divmod(n, 10)
        return [_TENS[tens]] + ([_ONES[ones]] if ones else [])
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return [_ONES[hundreds], "hundred"] + (number_to_words(rest) if rest else [])
    for scale, name in _SCALES:
        if n >= scale:
            head, rest = divmod(n, scale)
            return number_to_words(head) + [name] + (number_to_words(rest) if rest else [])
    raise AssertionError("unreachable")  # pragma: no cover


def ordinal_to_words(n: int) -> list[str]:
    """Ordinal reading (``21`` -> ``twenty first``)."""
    words = number_to_words(n)
    last = words[-1]
    if last in _ORDINAL_IRREGULAR:
        words[-1] = _ORDINAL_IRREGULAR[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return words


@dataclass(frozen=True)
class NormToken:
    """One normalised token with full provenance."""

    index: int
    original_text: str
    span_start: int
    span_end: int
    normalized: str
    kind: str
    supported: bool = True
    rules: tuple[str, ...] = ()
    unsupported_reason: str | None = None
    source_index: int = 0
    expansion_index: int = 0
    expansion_size: int = 1
    sub_start: int = 0
    sub_end: int = 0


@dataclass(frozen=True)
class DroppedSpan:
    """Text that was removed from the pronunciation stream (never silently)."""

    text: str
    start: int
    end: int
    kind: str
    reason: str


@dataclass
class NormalizationResult:
    original_text: str
    tokens: list[NormToken] = field(default_factory=list)
    dropped: list[DroppedSpan] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def normalized_text(self) -> str:
        """Space-joined *supported* normalised tokens (unsupported ones are listed separately)."""
        return " ".join(t.normalized for t in self.tokens if t.supported)

    @property
    def unsupported_tokens(self) -> list[NormToken]:
        return [t for t in self.tokens if not t.supported]

    def audit(self) -> list[dict[str, object]]:
        """Flat audit table: one row per normalised token."""
        return [
            {
                "index": t.index, "original": t.original_text, "normalized": t.normalized,
                "span": [t.span_start, t.span_end], "kind": t.kind, "rules": list(t.rules),
                "supported": t.supported, "reason": t.unsupported_reason,
                "expansion": f"{t.expansion_index + 1}/{t.expansion_size}",
            }
            for t in self.tokens
        ]

    def verify(self) -> None:
        """Assert traceability and the 'nothing silently lost' invariant."""
        text = self.original_text
        covered = [False] * len(text)
        for i, t in enumerate(self.tokens):
            if t.index != i:
                raise NormalizationError(f"token indices must be consecutive (got {t.index} at {i})")
            if text[t.span_start:t.span_end] != t.original_text:
                raise NormalizationError(f"token {i} span does not reproduce its original text")
            if not (t.span_start <= t.sub_start <= t.sub_end <= t.span_end):
                raise NormalizationError(f"token {i} sub-span lies outside its span")
            for k in range(t.span_start, t.span_end):
                covered[k] = True
        for d in self.dropped:
            if text[d.start:d.end] != d.text:
                raise NormalizationError(f"dropped span {d.start}:{d.end} does not match the original")
            for k in range(d.start, d.end):
                if covered[k] and not text[k].isspace():
                    raise NormalizationError(f"dropped span overlaps token at char {k}")
                covered[k] = True
        for k, ch in enumerate(text):
            if not ch.isspace() and not covered[k]:
                raise NormalizationError(f"character {ch!r} at {k} is neither tokenised nor recorded as dropped")


class TextNormalizer:
    """Configurable, auditable normaliser (see module docstring)."""

    def __init__(self, config: NormalizationConfig | None = None) -> None:
        self.config = config or NormalizationConfig()
        self._abbr = {k.lower(): v for k, v in self.config.abbreviations.items()}

    # ---- public ------------------------------------------------------------------
    def normalize(self, text: str) -> NormalizationResult:
        if not isinstance(text, str):
            raise NormalizationError(f"text must be a string, got {type(text).__name__}")
        result = NormalizationResult(original_text=text)
        if not text.strip():
            result.warnings.append("empty_input: no text to normalise")
            return result
        source_index = 0
        pos = 0
        while True:
            match = _TOKEN_RE.search(text, pos)
            if match is None:
                break
            kind = match.lastgroup or "other"
            start, end = match.span()
            if kind == "run" and text[end:end + 1] == "." and match.group().lower() in self._abbr:
                kind, end = "abbreviation", end + 1  # the period belongs to the abbreviation
            for piece in self._handle(text, kind, start, end, source_index, result):
                result.tokens.append(NormToken(**{**piece.__dict__, "index": len(result.tokens)}))
            source_index += 1
            pos = end
        if not result.tokens:
            result.warnings.append("no_pronounceable_tokens: input contained only punctuation or unsupported text")
        result.verify()
        return result

    # ---- internals ---------------------------------------------------------------
    def _handle(
        self, text: str, kind: str, start: int, end: int, src: int, result: NormalizationResult
    ) -> Iterator[NormToken]:
        raw = text[start:end]
        if kind in {"url", "email"}:
            yield self._unsupported(raw, start, end, kind, f"{kind}_not_supported", src)
        elif kind == "dotted":
            yield self._unsupported(raw, start, end, "acronym", "dotted_acronym_reading_ambiguous", src)
        elif kind == "ordinal":
            yield from self._ordinal(raw, start, end, src, result)
        elif kind == "number":
            yield from self._number(raw, start, end, src, result)
        elif kind == "abbreviation":
            key = raw[:-1].lower()
            yield from self._expand(raw, start, end, self._abbr[key].split(), "abbreviation",
                                    (f"abbreviation_expanded:{key}",), src)
        elif kind == "run":
            yield from self._run(raw, start, end, src, result)
        else:
            yield from self._other(raw, start, end, src, result)

    def _unsupported(self, raw: str, start: int, end: int, kind: str, reason: str, src: int) -> NormToken:
        return NormToken(
            index=0, original_text=raw, span_start=start, span_end=end,
            normalized=raw.lower() if self.config.lowercase else raw, kind=kind, supported=False,
            rules=(), unsupported_reason=reason, source_index=src, sub_start=start, sub_end=end,
        )

    def _expand(
        self, raw: str, start: int, end: int, words: list[str], kind: str, rules: tuple[str, ...], src: int
    ) -> Iterator[NormToken]:
        for i, word in enumerate(words):
            yield NormToken(
                index=0, original_text=raw, span_start=start, span_end=end, normalized=word,
                kind=kind, supported=True, rules=rules, source_index=src, expansion_index=i,
                expansion_size=len(words), sub_start=start, sub_end=end,
            )

    def _number(self, raw: str, start: int, end: int, src: int, result: NormalizationResult) -> Iterator[NormToken]:
        if self.config.numbers == "flag":
            yield self._unsupported(raw, start, end, "number", "number_expansion_disabled", src)
            return
        clean = raw.replace(",", "")
        whole, _, frac = clean.partition(".")
        if len(whole) > 1 and whole.startswith("0") and not frac:
            words = [w for d in whole for w in number_to_words(int(d))]
            rules: tuple[str, ...] = ("number_expanded:digit_by_digit",)
            result.warnings.append(f"digit_by_digit: {raw!r} has leading zeros and was read digit by digit")
        else:
            if int(whole) >= _MAX_NUMBER:
                yield self._unsupported(raw, start, end, "number", "number_too_large", src)
                return
            words = number_to_words(int(whole))
            rules = ("number_expanded:cardinal",)
            if frac:
                words += ["point"] + [w for d in frac for w in number_to_words(int(d))]
                rules = ("number_expanded:decimal",)
            if "," not in raw and not frac and 1100 <= int(whole) <= 2099:
                result.warnings.append(
                    f"ambiguous_number: {raw!r} was read as a cardinal number; a year reading would differ"
                )
        yield from self._expand(raw, start, end, words, "number", rules, src)

    def _ordinal(self, raw: str, start: int, end: int, src: int, result: NormalizationResult) -> Iterator[NormToken]:
        if self.config.ordinals == "flag":
            yield self._unsupported(raw, start, end, "ordinal", "ordinal_expansion_disabled", src)
            return
        digits = re.match(r"\d+", raw)
        assert digits is not None
        value = int(digits.group())
        if value >= _MAX_NUMBER:
            yield self._unsupported(raw, start, end, "ordinal", "number_too_large", src)
            return
        yield from self._expand(raw, start, end, ordinal_to_words(value), "ordinal",
                                ("ordinal_expanded",), src)

    def _run(
        self, raw: str, start: int, end: int, src: int, result: NormalizationResult
    ) -> Iterator[NormToken]:
        has_digit = any(c.isdigit() for c in raw)
        has_letter = any(c.isalpha() for c in raw)
        if has_digit and has_letter:
            yield self._unsupported(raw, start, end, "alphanumeric", "mixed_letters_and_digits", src)
            return
        if has_digit:  # digits joined by '-' or apostrophes, e.g. "12-34" (numbers alone are handled earlier)
            yield self._unsupported(raw, start, end, "alphanumeric", "digit_sequence_with_joiners", src)
            return
        rules: list[str] = []
        pieces: list[tuple[str, int, int]] = []
        if self.config.hyphen_policy == "split" and "-" in raw:
            offset = start
            for part in raw.split("-"):
                pieces.append((part, offset, offset + len(part)))
                offset += len(part) + 1
            rules.append("hyphen_split")
        else:
            pieces.append((raw, start, end))
        for i, (part, p_start, p_end) in enumerate(pieces):
            norm = part
            part_rules = list(rules)
            if self.config.normalize_unicode_apostrophes and any(c in norm for c in "\u2019\u02bc"):
                norm = norm.replace("\u2019", "'").replace("\u02bc", "'")
                part_rules.append("unicode_apostrophe_normalized")
            if self.config.lowercase and norm != norm.lower():
                norm = norm.lower()
                part_rules.append("lowercase")
            kind = "contraction" if any(c in part for c in _APOSTROPHES) else "word"
            yield NormToken(
                index=0, original_text=raw, span_start=start, span_end=end, normalized=norm,
                kind=kind, supported=True, rules=tuple(part_rules), source_index=src,
                expansion_index=i, expansion_size=len(pieces), sub_start=p_start, sub_end=p_end,
            )

    def _other(self, raw: str, start: int, end: int, src: int, result: NormalizationResult) -> Iterator[NormToken]:
        ch = raw
        if ch in self.config.symbols:
            words = self.config.symbols[ch].split()
            yield from self._expand(raw, start, end, words, "symbol", (f"symbol_mapped:{ch}",), src)
            return
        category = unicodedata.category(ch)
        if category == "Cf":  # zero-width / format characters
            result.dropped.append(DroppedSpan(raw, start, end, "format_char", "invisible format character"))
            return
        if ch in _DROPPABLE_PUNCT:
            result.dropped.append(DroppedSpan(raw, start, end, "punctuation", "punctuation is not pronounced"))
            return
        reason = "unsupported_symbol" if category.startswith(("S", "P")) else f"unsupported_character_{category}"
        yield self._unsupported(raw, start, end, "symbol", reason, src)
