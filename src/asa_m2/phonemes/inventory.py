"""Versioned canonical phone inventory (ARPABET base phones + IPA display).

Design rules enforced here (see docs/data_contract.md):

* stress is stored *separately* from phone identity - base phones never contain digits;
* the CTC blank is a reserved special symbol: not a phoneme, not silence;
* no silence class is invented;
* unknown symbols/IDs raise instead of being remapped.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from asa_m2.exceptions import InventoryError

_BASE_RE = re.compile(r"^[A-Z]+$")
_SYMBOL_RE = re.compile(r"^([A-Z]+)([0-9])?$")
_CATEGORIES = {"vowel", "consonant"}


@dataclass(frozen=True)
class Phone:
    """One base phone of the canonical inventory."""

    id: int
    arpabet: str
    ipa: str
    category: str
    stress_bearing: bool
    ipa_variants: tuple[str, ...] = ()
    articulatory: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_vowel(self) -> bool:
        return self.category == "vowel"


@dataclass(frozen=True)
class ParsedPhone:
    """An ARPABET symbol split into base phone and stress (``None`` for consonants)."""

    raw: str
    base: str
    stress: int | None


@dataclass(frozen=True)
class CompatibilityReport:
    """Result of comparing the canonical inventory with another phone set."""

    other_name: str
    missing_in_other: tuple[str, ...]  # canonical phones the other model lacks
    extra_in_other: tuple[str, ...]  # phones the other model has that are not canonical

    @property
    def compatible(self) -> bool:
        return not self.missing_in_other and not self.extra_in_other


def validate_inventory_data(
    inventory: Mapping[str, Any],
    phone_to_id: Mapping[str, Any] | None = None,
    id_to_phone: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return a list of human-readable problems (empty list == valid)."""
    problems: list[str] = []

    for key in ("schema_version", "inventory_id", "inventory_version", "phones"):
        if key not in inventory:
            problems.append(f"inventory is missing required key {key!r}")
    phones = inventory.get("phones")
    if not isinstance(phones, list) or not phones:
        problems.append("inventory 'phones' must be a non-empty list")
        return problems

    seen_ids: dict[int, str] = {}
    seen_symbols: dict[str, int] = {}
    seen_ipa: dict[str, str] = {}
    for idx, phone in enumerate(phones):
        where = f"phones[{idx}]"
        if not isinstance(phone, dict):
            problems.append(f"{where} is not an object")
            continue
        arp, pid, ipa = phone.get("arpabet"), phone.get("id"), phone.get("ipa")
        cat = phone.get("category")
        if not isinstance(arp, str) or not _BASE_RE.match(arp):
            problems.append(
                f"{where}: arpabet label {arp!r} must be uppercase letters only "
                "(stress digits belong in the stress field, not the phone identity)"
            )
            continue
        if not isinstance(pid, int) or isinstance(pid, bool) or pid < 0:
            problems.append(f"{where} ({arp}): id {pid!r} must be a non-negative integer")
        elif pid in seen_ids:
            problems.append(f"duplicate phone id {pid}: {seen_ids[pid]!r} and {arp!r}")
        else:
            seen_ids[pid] = arp
        if arp in seen_symbols:
            problems.append(f"duplicate arpabet symbol {arp!r} (ids {seen_symbols[arp]} and {pid})")
        else:
            seen_symbols[arp] = pid if isinstance(pid, int) else -1
        if not isinstance(ipa, str) or not ipa:
            problems.append(f"{where} ({arp}): missing ipa display label")
        elif ipa in seen_ipa:
            problems.append(f"duplicate ipa label {ipa!r}: {seen_ipa[ipa]!r} and {arp!r}")
        else:
            seen_ipa[ipa] = arp
        if cat not in _CATEGORIES:
            problems.append(f"{where} ({arp}): category {cat!r} must be one of {sorted(_CATEGORIES)}")
        if cat == "consonant" and phone.get("stress_bearing"):
            problems.append(f"{where} ({arp}): consonants cannot be stress_bearing")
        if cat == "vowel" and not phone.get("stress_bearing"):
            problems.append(f"{where} ({arp}): vowels must be stress_bearing")

    # special symbols / CTC blank / silence
    specials = inventory.get("special_symbols", [])
    blank_entries = [s for s in specials if isinstance(s, dict) and s.get("role") == "ctc_blank"]
    if len(blank_entries) != 1:
        problems.append(f"exactly one special symbol with role 'ctc_blank' is required, found {len(blank_entries)}")
    for sp in specials:
        if not isinstance(sp, dict):
            problems.append("special_symbols entries must be objects")
            continue
        sid, sym = sp.get("id"), sp.get("symbol")
        if sid in seen_ids:
            problems.append(f"special symbol {sym!r} id {sid} collides with phone {seen_ids[sid]!r}")
        if sym in seen_symbols:
            problems.append(f"special symbol {sym!r} collides with a phone label")
        if sp.get("is_phoneme") is not False:
            problems.append(f"special symbol {sym!r}: is_phoneme must be false")
        if sp.get("role") == "ctc_blank" and sp.get("is_silence") is not False:
            problems.append("the CTC blank must not be declared as silence (is_silence must be false)")
    has_silence = inventory.get("silence_policy", {}).get("has_silence_class", False)
    silence_specials = [s for s in specials if isinstance(s, dict) and s.get("is_silence")]
    if has_silence and not silence_specials:
        problems.append("silence_policy.has_silence_class is true but no silence special symbol is defined")
    if not has_silence and silence_specials:
        problems.append("a silence special symbol exists but silence_policy.has_silence_class is false")

    stress = inventory.get("stress", {})
    values = stress.get("values")
    if not isinstance(values, list) or not all(isinstance(v, int) for v in values):
        problems.append("stress.values must be a list of integers")
    if stress.get("separate_from_phone_identity") is not True:
        problems.append("stress.separate_from_phone_identity must be true")

    # companion files
    meta_keys = ("schema_version", "inventory_id", "inventory_version")
    expected_p2i = {p["arpabet"]: p["id"] for p in phones if isinstance(p, dict) and "arpabet" in p and "id" in p}
    if phone_to_id is not None:
        for key in meta_keys:
            if phone_to_id.get(key) != inventory.get(key):
                problems.append(f"phone_to_id.{key} {phone_to_id.get(key)!r} != inventory {inventory.get(key)!r}")
        table = phone_to_id.get("phone_to_id", {})
        if table != expected_p2i:
            diff = sorted(set(table) ^ set(expected_p2i)) or sorted(
                k for k in table if table[k] != expected_p2i.get(k)
            )
            problems.append(f"phone_to_id does not match inventory phones (differences: {diff})")
        for blank in blank_entries:
            if blank.get("symbol") in table:
                problems.append("the CTC blank must not appear in phone_to_id")
    if id_to_phone is not None:
        for key in meta_keys:
            if id_to_phone.get(key) != inventory.get(key):
                problems.append(f"id_to_phone.{key} {id_to_phone.get(key)!r} != inventory {inventory.get(key)!r}")
        table2 = id_to_phone.get("id_to_phone", {})
        try:
            as_int = {int(k): v for k, v in table2.items()}
        except (TypeError, ValueError):
            problems.append("id_to_phone keys must be integer strings")
            as_int = {}
        if as_int != {v: k for k, v in expected_p2i.items()}:
            problems.append("id_to_phone does not match inventory phones")
        if len(as_int) != len(table2):
            problems.append("id_to_phone contains duplicate integer keys")
    return problems


class PhoneInventory:
    """Loaded, validated canonical phone inventory."""

    def __init__(
        self,
        data: Mapping[str, Any],
        phone_to_id: Mapping[str, Any] | None = None,
        id_to_phone: Mapping[str, Any] | None = None,
    ) -> None:
        problems = validate_inventory_data(data, phone_to_id, id_to_phone)
        if problems:
            raise InventoryError(
                "invalid phone inventory:\n  - " + "\n  - ".join(problems),
                details={"problems": problems},
            )
        self._data = dict(data)
        self._phones: dict[str, Phone] = {}
        for p in data["phones"]:
            self._phones[p["arpabet"]] = Phone(
                id=p["id"],
                arpabet=p["arpabet"],
                ipa=p["ipa"],
                category=p["category"],
                stress_bearing=bool(p["stress_bearing"]),
                ipa_variants=tuple(p.get("ipa_variants", [])),
                articulatory=dict(p.get("articulatory", {})),
            )
        self._by_id = {p.id: p for p in self._phones.values()}
        self._stress_values = frozenset(data["stress"]["values"])
        blank = next(s for s in data.get("special_symbols", []) if s.get("role") == "ctc_blank")
        self.ctc_blank_id: int = blank["id"]
        self.ctc_blank_symbol: str = blank["symbol"]

    # ---- construction ------------------------------------------------------------
    @classmethod
    def from_files(
        cls,
        inventory_path: str | Path,
        phone_to_id_path: str | Path | None = None,
        id_to_phone_path: str | Path | None = None,
        *,
        companions: bool = True,
    ) -> PhoneInventory:
        """Load an inventory; sibling ``phone_to_id.json``/``id_to_phone.json`` are cross-checked."""
        inv_path = Path(inventory_path)
        data = _read_json(inv_path)
        p2i = i2p = None
        if companions:
            p2i_path = Path(phone_to_id_path) if phone_to_id_path else inv_path.with_name("phone_to_id.json")
            i2p_path = Path(id_to_phone_path) if id_to_phone_path else inv_path.with_name("id_to_phone.json")
            if p2i_path.exists():
                p2i = _read_json(p2i_path)
            elif phone_to_id_path:
                raise InventoryError(f"phone_to_id file not found: {p2i_path}")
            if i2p_path.exists():
                i2p = _read_json(i2p_path)
            elif id_to_phone_path:
                raise InventoryError(f"id_to_phone file not found: {i2p_path}")
        return cls(data, p2i, i2p)

    @classmethod
    def load_default(cls) -> PhoneInventory:
        """Load the inventory packaged inside asa_m2 (``asa_m2/resources``)."""
        from asa_m2.config import resources_dir

        return cls.from_files(resources_dir() / "phoneme_inventory.json")

    # ---- metadata ----------------------------------------------------------------
    @property
    def inventory_id(self) -> str:
        return str(self._data["inventory_id"])

    @property
    def inventory_version(self) -> str:
        return str(self._data["inventory_version"])

    @property
    def schema_version(self) -> str:
        return str(self._data["schema_version"])

    @property
    def phones(self) -> tuple[Phone, ...]:
        return tuple(sorted(self._phones.values(), key=lambda p: p.id))

    @property
    def base_symbols(self) -> tuple[str, ...]:
        return tuple(p.arpabet for p in self.phones)

    @property
    def phone_to_id(self) -> dict[str, int]:
        return {p.arpabet: p.id for p in self.phones}

    @property
    def id_to_phone(self) -> dict[int, str]:
        return {p.id: p.arpabet for p in self.phones}

    def __contains__(self, base: object) -> bool:
        return base in self._phones

    def __len__(self) -> int:
        return len(self._phones)

    # ---- lookups -----------------------------------------------------------------
    def get(self, base: str) -> Phone:
        try:
            return self._phones[base]
        except KeyError:
            raise InventoryError(f"unknown phone {base!r}: not in inventory {self.inventory_id}") from None

    def phone_id(self, base: str) -> int:
        return self.get(base).id

    def phone_from_id(self, pid: int) -> Phone:
        try:
            return self._by_id[pid]
        except KeyError:
            raise InventoryError(f"unknown phone id {pid!r}") from None

    def is_vowel(self, base: str) -> bool:
        return self.get(base).is_vowel

    def ipa(self, base: str) -> str:
        return self.get(base).ipa

    # ---- symbols & stress --------------------------------------------------------
    def parse_symbol(self, symbol: str) -> ParsedPhone:
        """Split e.g. ``AY1`` into base ``AY`` and stress ``1``.

        Raises :class:`InventoryError` for unknown phones, for stress on consonants and
        for stress values outside the inventory's allowed set. Vowels without a digit
        are accepted with ``stress=None`` (stress unspecified), never defaulted to 0.
        """
        m = _SYMBOL_RE.match(symbol) if isinstance(symbol, str) else None
        if not m:
            raise InventoryError(f"malformed phone symbol {symbol!r}")
        base, digit = m.group(1), m.group(2)
        phone = self.get(base)
        if digit is None:
            return ParsedPhone(symbol, base, None)
        stress = int(digit)
        if not phone.stress_bearing:
            raise InventoryError(f"symbol {symbol!r}: consonant {base!r} cannot carry stress")
        if stress not in self._stress_values:
            raise InventoryError(f"symbol {symbol!r}: stress {stress} not in {sorted(self._stress_values)}")
        return ParsedPhone(symbol, base, stress)

    def encode(self, bases: Iterable[str]) -> list[int]:
        """Base phones -> IDs. Raises on any unknown phone (never skips)."""
        return [self.phone_id(b) for b in bases]

    def decode(self, ids: Iterable[int]) -> list[str]:
        return [self.phone_from_id(i).arpabet for i in ids]

    # ---- compatibility -----------------------------------------------------------
    def check_compatibility(self, other_phones: Iterable[str], *, other_name: str = "other") -> CompatibilityReport:
        """Compare the canonical base-phone set with another model's phone set.

        ``other_phones`` may carry stress digits; they are stripped for comparison. The
        report is informational: the caller decides whether a mismatch is acceptable.
        """
        other = {re.sub(r"[0-9]+$", "", str(p)) for p in other_phones}
        ours = set(self._phones)
        return CompatibilityReport(
            other_name=other_name,
            missing_in_other=tuple(sorted(ours - other)),
            extra_in_other=tuple(sorted(other - ours)),
        )

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._data))


def _read_json(path: Path) -> Any:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise InventoryError(f"inventory file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise InventoryError(f"invalid JSON in {path}: {exc}") from exc


def phone_set_from_lines(lines: Sequence[str]) -> list[str]:
    """Parse a plain-text phone list (one phone per line, ``#`` comments allowed)."""
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
