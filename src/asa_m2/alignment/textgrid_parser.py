"""TextGrid reader (Praat long and short text formats) with diagnostic output.

Why not just ``praatio``?  ``praatio.openTextgrid`` refuses files containing overlapping,
negative-length or zero-length intervals (and raises a bare ``IndexError`` on malformed
input). M2 must *detect and report* those cases rather than crash, so this module reads
the raw structure itself. ``praatio`` is still used for **writing** TextGrids and as an
independent oracle in the test-suite.

Nothing is repaired: intervals are returned exactly as written, in file order, together
with structural issues found while reading. Semantic checks live in ``validator.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from asa_m2.exceptions import TextGridError
from asa_m2.utils.paths import sha256_file


@dataclass(frozen=True)
class TextGridIssue:
    """A structural or semantic problem found in a TextGrid."""

    code: str
    severity: str  # "error" | "warning" | "info"
    message: str
    tier: str | None = None
    interval_index: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedInterval:
    index: int  # 0-based position in the tier as written
    start: float
    end: float
    label: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class ParsedPoint:
    index: int
    time: float
    mark: str


@dataclass
class ParsedTier:
    name: str
    tier_type: str  # "IntervalTier" | "TextTier"
    xmin: float
    xmax: float
    intervals: list[ParsedInterval] = field(default_factory=list)
    points: list[ParsedPoint] = field(default_factory=list)

    @property
    def is_interval_tier(self) -> bool:
        return self.tier_type == "IntervalTier"


@dataclass
class ParsedTextGrid:
    path: Path
    sha256: str
    text_format: str  # "long" | "short"
    encoding: str
    xmin: float
    xmax: float
    tiers: list[ParsedTier]
    issues: list[TextGridIssue] = field(default_factory=list)

    @property
    def tier_names(self) -> list[str]:
        return [t.name for t in self.tiers]

    def get_tier(self, name: str) -> ParsedTier:
        for t in self.tiers:
            if t.name == name:
                return t
        raise TextGridError(f"tier {name!r} not found; available tiers: {self.tier_names}")

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly structure (original labels and times, nothing dropped)."""
        return {
            "source_file": self.path.name,
            "sha256": self.sha256,
            "text_format": self.text_format,
            "encoding": self.encoding,
            "xmin": self.xmin,
            "xmax": self.xmax,
            "time_unit": "seconds",
            "tiers": [
                {
                    "name": t.name,
                    "type": t.tier_type,
                    "xmin": t.xmin,
                    "xmax": t.xmax,
                    "intervals": [
                        {"index": i.index, "start": i.start, "end": i.end, "label": i.label}
                        for i in t.intervals
                    ],
                    "points": [{"index": p.index, "time": p.time, "mark": p.mark} for p in t.points],
                }
                for t in self.tiers
            ],
            "issues": [issue.__dict__ for issue in self.issues],
        }


# ---------------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------------
def _decode(raw: bytes, path: Path) -> tuple[str, str]:
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig"), "utf-8-sig"
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return raw.decode("utf-16"), "utf-16"
        except UnicodeDecodeError as exc:
            raise TextGridError(f"{path.name}: invalid UTF-16 TextGrid: {exc}") from exc
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError as exc:
        raise TextGridError(
            f"{path.name}: cannot decode as UTF-8 and has no UTF-16 byte-order mark: {exc}"
        ) from exc


def _unquote(token: str) -> str:
    return token[1:-1].replace('""', '"')


def _to_float(token: str, what: str, path: Path) -> float:
    try:
        return float(token)
    except ValueError:
        raise TextGridError(f"{path.name}: {what} {token!r} is not a number") from None


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


# ---------------------------------------------------------------------------------
# long format
# ---------------------------------------------------------------------------------
_LONG_RE = re.compile(
    r"""
    (?P<hdr>(?P<hkey>item|intervals|points)\s*\[\s*(?P<hidx>\d*)\s*\]\s*:)
  | (?P<kv>(?P<key>(?:(?:intervals|points)\s*:\s*)?size|xmin|xmax|class|name|text|number|time|mark)
        \s*=\s*(?P<val>"(?:[^"]|"")*"|\S+))
    """,
    re.VERBOSE,
)


def _parse_long(text: str, path: Path) -> tuple[float, float, list[ParsedTier], list[TextGridIssue]]:
    issues: list[TextGridIssue] = []
    gx: dict[str, float] = {}
    tiers: list[dict[str, Any]] = []
    cur_tier: dict[str, Any] | None = None
    cur_entry: dict[str, Any] | None = None
    declared_tiers: int | None = None

    def need_str(val: str, what: str, pos: int) -> str:
        if not (len(val) >= 2 and val[0] == '"' and val[-1] == '"'):
            raise TextGridError(f"{path.name} line {_line_of(text, pos)}: {what} must be a quoted string, got {val!r}")
        return _unquote(val)

    for m in _LONG_RE.finditer(text):
        if m.group("hdr"):
            kind, idx = m.group("hkey"), m.group("hidx")
            if kind == "item":
                if idx:  # "item [k]:" starts a tier ("item []:" is the list header)
                    cur_tier = {"name": None, "class": None, "xmin": None, "xmax": None, "entries": [], "declared": None}
                    tiers.append(cur_tier)
                    cur_entry = None
            else:
                if cur_tier is None:
                    raise TextGridError(f"{path.name} line {_line_of(text, m.start())}: '{kind} [..]' outside of a tier")
                cur_entry = {"kind": kind, "xmin": None, "xmax": None, "text": None, "number": None, "mark": None,
                             "pos": m.start()}
                cur_tier["entries"].append(cur_entry)
            continue
        key = re.sub(r"\s+", "", m.group("key"))
        val = m.group("val")
        pos = m.start()
        if key == "size":
            n = int(_to_float(val, "size", path))
            if cur_tier is None:
                declared_tiers = n
        elif key in {"intervals:size", "points:size"}:
            if cur_tier is None:
                raise TextGridError(f"{path.name} line {_line_of(text, pos)}: '{key}' outside of a tier")
            cur_tier["declared"] = int(_to_float(val, key, path))
        elif key == "class":
            if cur_tier is None:
                continue  # 'Object class = "TextGrid"' header line
            cur_tier["class"] = need_str(val, "class", pos)
        elif key == "name":
            if cur_tier is None:
                raise TextGridError(f"{path.name} line {_line_of(text, pos)}: 'name' outside of a tier")
            cur_tier["name"] = need_str(val, "tier name", pos)
        elif key in {"xmin", "xmax"}:
            number = _to_float(val, key, path)
            if cur_entry is not None:
                cur_entry[key] = number
            elif cur_tier is not None:
                cur_tier[key] = number
            else:
                gx[key] = number
        elif key == "text":
            if cur_entry is None:
                raise TextGridError(f"{path.name} line {_line_of(text, pos)}: 'text' outside of an interval")
            cur_entry["text"] = need_str(val, "interval text", pos)
        elif key == "mark":
            if cur_entry is None:
                raise TextGridError(f"{path.name} line {_line_of(text, pos)}: 'mark' outside of a point")
            cur_entry["mark"] = need_str(val, "point mark", pos)
        elif key in {"number", "time"}:
            if cur_entry is None:
                raise TextGridError(f"{path.name} line {_line_of(text, pos)}: '{key}' outside of a point")
            cur_entry["number"] = _to_float(val, key, path)

    if "xmin" not in gx or "xmax" not in gx:
        raise TextGridError(f"{path.name}: missing global xmin/xmax; not a valid TextGrid")
    if not tiers:
        raise TextGridError(f"{path.name}: TextGrid contains no tiers")
    if declared_tiers is not None and declared_tiers != len(tiers):
        issues.append(TextGridIssue("declared_tier_count_mismatch", "warning",
                                    f"header declares {declared_tiers} tier(s) but {len(tiers)} were read"))

    parsed: list[ParsedTier] = []
    for t in tiers:
        label = t["name"] if t["name"] is not None else "<unnamed>"
        if t["name"] is None or t["class"] is None or t["xmin"] is None or t["xmax"] is None:
            raise TextGridError(f"{path.name}: tier {label!r} is missing class/name/xmin/xmax")
        tier = ParsedTier(t["name"], t["class"], t["xmin"], t["xmax"])
        for i, e in enumerate(t["entries"]):
            where = f"{path.name} line {_line_of(text, e['pos'])}"
            if e["kind"] == "intervals":
                if e["xmin"] is None or e["xmax"] is None or e["text"] is None:
                    raise TextGridError(f"{where}: incomplete interval {i + 1} in tier {label!r}")
                tier.intervals.append(ParsedInterval(i, e["xmin"], e["xmax"], e["text"]))
            else:
                if e["number"] is None or e["mark"] is None:
                    raise TextGridError(f"{where}: incomplete point {i + 1} in tier {label!r}")
                tier.points.append(ParsedPoint(i, e["number"], e["mark"]))
        found = len(tier.intervals) if tier.tier_type == "IntervalTier" else len(tier.points)
        if t["declared"] is not None and t["declared"] != found:
            issues.append(TextGridIssue("declared_size_mismatch", "warning",
                                        f"tier {label!r} declares {t['declared']} entries but {found} were read",
                                        tier=label))
        parsed.append(tier)
    return gx["xmin"], gx["xmax"], parsed, issues


# ---------------------------------------------------------------------------------
# short format
# ---------------------------------------------------------------------------------
_SHORT_TOKEN_RE = re.compile(r'"(?:[^"]|"")*"|<exists>|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|\S+')


def _parse_short(text: str, path: Path) -> tuple[float, float, list[ParsedTier], list[TextGridIssue]]:
    # drop the two header lines ('File type = ...', 'Object class = ...')
    body = re.sub(r'^\s*File type\s*=\s*"[^"]*"\s*\n\s*Object class\s*=\s*"[^"]*"\s*', "", text, count=1)
    tokens = _SHORT_TOKEN_RE.findall(body)
    pos = 0

    def take(what: str) -> str:
        nonlocal pos
        if pos >= len(tokens):
            raise TextGridError(f"{path.name}: unexpected end of file while reading {what}")
        tok = tokens[pos]
        pos += 1
        return tok

    def take_num(what: str) -> float:
        return _to_float(take(what), what, path)

    def take_str(what: str) -> str:
        tok = take(what)
        if not (len(tok) >= 2 and tok[0] == '"' and tok[-1] == '"'):
            raise TextGridError(f"{path.name}: expected a quoted string for {what}, got {tok!r}")
        return _unquote(tok)

    xmin, xmax = take_num("xmin"), take_num("xmax")
    exists = take("'<exists>' marker")
    if exists.strip('"') != "<exists>" and exists != "<exists>":
        raise TextGridError(f"{path.name}: expected '<exists>' after xmax, got {exists!r}")
    n_tiers = int(take_num("tier count"))
    if n_tiers <= 0:
        raise TextGridError(f"{path.name}: TextGrid declares {n_tiers} tiers")
    tiers: list[ParsedTier] = []
    for _ in range(n_tiers):
        ttype = take_str("tier class")
        name = take_str("tier name")
        t_min, t_max = take_num("tier xmin"), take_num("tier xmax")
        count = int(take_num("entry count"))
        tier = ParsedTier(name, ttype, t_min, t_max)
        for i in range(count):
            if ttype == "IntervalTier":
                a, b = take_num("interval xmin"), take_num("interval xmax")
                tier.intervals.append(ParsedInterval(i, a, b, take_str("interval text")))
            elif ttype == "TextTier":
                tier.points.append(ParsedPoint(i, take_num("point time"), take_str("point mark")))
            else:
                raise TextGridError(f"{path.name}: unknown tier class {ttype!r}")
        tiers.append(tier)
    issues: list[TextGridIssue] = []
    if pos != len(tokens):
        issues.append(TextGridIssue("trailing_content", "warning",
                                    f"{len(tokens) - pos} unexpected token(s) after the last tier"))
    return xmin, xmax, tiers, issues


# ---------------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------------
def parse_textgrid(path: str | Path) -> ParsedTextGrid:
    """Parse a TextGrid file. Raises :class:`TextGridError` for unreadable/malformed files.

    Overlaps, negative durations etc. do **not** raise here; they are surfaced by
    :func:`asa_m2.alignment.validator.validate_textgrid`.
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise TextGridError(f"TextGrid file not found: {p}")
    raw = p.read_bytes()
    if not raw.strip():
        raise TextGridError(f"{p.name}: TextGrid file is empty")
    text, encoding = _decode(raw, p)
    if "ooTextFile" not in text[:200] or "TextGrid" not in text[:300]:
        raise TextGridError(f"{p.name}: not a Praat TextGrid (missing 'ooTextFile' / 'TextGrid' header)")
    is_long = re.search(r"\bxmin\s*=", text) is not None
    try:
        xmin, xmax, tiers, issues = (_parse_long if is_long else _parse_short)(text, p)
    except TextGridError:
        raise
    except Exception as exc:  # defensive: turn any parser bug on odd input into a clear error
        raise TextGridError(f"{p.name}: malformed TextGrid ({type(exc).__name__}: {exc})") from exc
    return ParsedTextGrid(
        path=p, sha256=sha256_file(p), text_format="long" if is_long else "short",
        encoding=encoding, xmin=xmin, xmax=xmax, tiers=tiers, issues=issues,
    )


def select_tier(
    parsed: ParsedTextGrid,
    candidates: list[str],
    *,
    explicit_name: str | None = None,
    allow_speaker_prefix: bool = True,
    kind: str = "tier",
) -> ParsedTier:
    """Pick the tier for ``kind`` ('word'/'phone') without assuming fixed names.

    Order: explicit name > exact candidate name > case-insensitive candidate name >
    ``"<speaker> - <candidate>"`` (only when unambiguous). Ambiguity or absence raises a
    :class:`TextGridError` that lists the tiers actually present.
    """
    interval_tiers = [t for t in parsed.tiers if t.is_interval_tier]
    if explicit_name is not None:
        for t in interval_tiers:
            if t.name == explicit_name:
                return t
        raise TextGridError(
            f"requested {kind} tier {explicit_name!r} not found among interval tiers {[t.name for t in interval_tiers]}"
        )
    lowered = [c.lower() for c in candidates]
    for t in interval_tiers:
        if t.name in candidates:
            return t
    for t in interval_tiers:
        if t.name.lower() in lowered:
            return t
    if allow_speaker_prefix:
        matches = [t for t in interval_tiers
                   if any(re.fullmatch(rf".+ - {re.escape(c)}", t.name, re.IGNORECASE) for c in candidates)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise TextGridError(
                f"several speaker-prefixed {kind} tiers found ({[t.name for t in matches]}); "
                f"choose one explicitly (e.g. --{kind}-tier)"
            )
    raise TextGridError(
        f"no {kind} tier found (looked for {candidates}); tiers present: {parsed.tier_names}. "
        f"Use `asa-m2 inspect-textgrid` and pass --{kind}-tier explicitly."
    )
