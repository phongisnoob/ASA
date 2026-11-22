"""Documented phone mappings (eSpeak-ng -> ARPABET) and aligner-label classification.

Nothing is ever dropped or remapped silently: an unknown source unit raises
:class:`UnmappedPhoneError` listing *all* offending units.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from asa_m2.exceptions import InventoryError, MappingError, UnmappedPhoneError
from asa_m2.phonemes.inventory import PhoneInventory

_CONFIDENCE = {"exact", "approximate"}


@dataclass(frozen=True)
class MappedUnit:
    """Result of mapping one source unit."""

    source: str
    arpabet: tuple[str, ...]
    confidence: str  # "exact" | "approximate"


class PhoneMapping:
    """A table mapping source-model units to canonical ARPABET base phones."""

    def __init__(self, name: str, spec: Mapping[str, Any], inventory: PhoneInventory) -> None:
        self.name = name
        self.inventory = inventory
        entries = spec.get("entries")
        if not isinstance(entries, dict) or not entries:
            raise MappingError(f"mapping {name!r}: 'entries' must be a non-empty object")
        problems: list[str] = []
        table: dict[str, MappedUnit] = {}
        for unit, rule in entries.items():
            if not unit:
                problems.append("empty source unit")
                continue
            arp = rule.get("arpabet") if isinstance(rule, dict) else None
            conf = rule.get("confidence") if isinstance(rule, dict) else None
            if not isinstance(arp, list) or not arp:
                problems.append(f"{unit!r}: 'arpabet' must be a non-empty list")
                continue
            bad = [a for a in arp if a not in inventory]
            if bad:
                problems.append(f"{unit!r}: target phones {bad} are not in inventory {inventory.inventory_id}")
                continue
            if conf not in _CONFIDENCE:
                problems.append(f"{unit!r}: confidence {conf!r} must be one of {sorted(_CONFIDENCE)}")
                continue
            table[unit] = MappedUnit(unit, tuple(arp), conf)
        self.unmapped_by_design: dict[str, str] = dict(spec.get("unmapped_by_design", {}))
        overlap = sorted(set(table) & set(self.unmapped_by_design))
        if overlap:
            problems.append(f"units both mapped and 'unmapped_by_design': {overlap}")
        if problems:
            raise MappingError(
                f"invalid mapping {name!r}:\n  - " + "\n  - ".join(problems), details={"problems": problems}
            )
        self._table = table
        self.stress_marks: dict[str, str] = dict(spec.get("stress_marks", {}))

    @classmethod
    def from_file(cls, path: str | Path, name: str, inventory: PhoneInventory) -> PhoneMapping:
        data = _load_mapping_file(Path(path), inventory)
        mappings = data.get("mappings", {})
        if name not in mappings:
            raise MappingError(f"mapping {name!r} not found in {path} (available: {sorted(mappings)})")
        return cls(name, mappings[name], inventory)

    def __contains__(self, unit: object) -> bool:
        return unit in self._table

    def __len__(self) -> int:
        return len(self._table)

    @property
    def units(self) -> tuple[str, ...]:
        return tuple(self._table)

    def map_unit(self, unit: str) -> MappedUnit:
        try:
            return self._table[unit]
        except KeyError:
            reason = self.unmapped_by_design.get(unit)
            suffix = f" (deliberately unmapped: {reason})" if reason else ""
            raise UnmappedPhoneError(
                f"unit {unit!r} has no documented mapping in {self.name!r}{suffix}",
                details={"units": [unit]},
            ) from None

    def map_units(self, units: Sequence[str]) -> list[MappedUnit]:
        """Strictly map every unit; raise once listing *all* unmapped units."""
        mapped: list[MappedUnit] = []
        missing: list[str] = []
        for unit in units:
            if unit in self._table:
                mapped.append(self._table[unit])
            else:
                missing.append(unit)
        if missing:
            reasons = {u: self.unmapped_by_design[u] for u in missing if u in self.unmapped_by_design}
            raise UnmappedPhoneError(
                f"{len(missing)} unit(s) have no documented mapping in {self.name!r}: {missing}",
                details={"units": missing, "reasons": reasons},
            )
        return mapped


def _load_mapping_file(path: Path, inventory: PhoneInventory) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        raise MappingError(f"mapping file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise MappingError(f"invalid JSON in {path}: {exc}") from exc
    for key in ("inventory_id", "inventory_version"):
        expected = inventory.inventory_id if key == "inventory_id" else inventory.inventory_version
        if data.get(key) != expected:
            raise MappingError(
                f"mapping file {path} has {key}={data.get(key)!r} but the loaded inventory has {expected!r}; "
                "refusing to use mismatched versions"
            )
    return data


class LabelRole(str, Enum):
    """Role of a label found in aligner output."""

    PHONE = "phone"
    SILENCE = "silence"
    OOV = "oov_marker"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ClassifiedLabel:
    label: str
    role: LabelRole
    base_phone: str | None = None
    stress: int | None = None
    reason: str | None = None


class AlignerLabelMapper:
    """Classify labels from an ARPABET-based aligner (e.g. MFA ``english_us_arpa``).

    The label policy is ``identity_after_stress_parse``: a label is valid iff it is an
    inventory phone, optionally followed by a stress digit. Everything else is reported
    as ``UNKNOWN`` with a reason - there is no fuzzy matching and no case folding.
    """

    def __init__(
        self,
        inventory: PhoneInventory,
        silence_labels: Iterable[str],
        oov_labels: Iterable[str],
    ) -> None:
        self.inventory = inventory
        self.silence_labels = frozenset(silence_labels)
        self.oov_labels = frozenset(oov_labels)
        overlap = self.silence_labels & self.oov_labels
        if overlap:
            raise MappingError(f"labels cannot be both silence and OOV: {sorted(overlap)}")

    @classmethod
    def from_mapping_file(cls, path: str | Path, inventory: PhoneInventory, name: str = "mfa_english_us_arpa") -> AlignerLabelMapper:
        data = _load_mapping_file(Path(path), inventory)
        spec = data.get("mappings", {}).get(name)
        if not spec or spec.get("label_policy") != "identity_after_stress_parse":
            raise MappingError(f"mapping {name!r} missing or has unsupported label_policy in {path}")
        return cls(inventory, spec.get("silence_labels", []), spec.get("oov_labels", []))

    def classify(self, label: str) -> ClassifiedLabel:
        stripped = label.strip()
        if stripped in self.silence_labels or (stripped == "" and "" in self.silence_labels):
            return ClassifiedLabel(label, LabelRole.SILENCE)
        if stripped in self.oov_labels:
            return ClassifiedLabel(label, LabelRole.OOV)
        try:
            parsed = self.inventory.parse_symbol(stripped)
        except InventoryError as exc:
            return ClassifiedLabel(label, LabelRole.UNKNOWN, reason=exc.message)
        return ClassifiedLabel(label, LabelRole.PHONE, parsed.base, parsed.stress)
