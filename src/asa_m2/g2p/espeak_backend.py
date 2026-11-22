"""eSpeak-ng fallback adapter (opt-in via ``g2p.fallback_backend: espeak``).

* Invoked as ``espeak-ng -q -v <voice> --ipa --sep=_ -- <word>`` without a shell.
* Output units are mapped to canonical ARPABET *only* through the documented
  ``espeak_en_us_ipa`` table in ``phoneme_mapping.json``.
* Anything that cannot be mapped (flap, glottal stop, language switches, multi-word
  output, ...) makes the word *unresolved* with an explicit reason. Phones are never
  dropped and never guessed.
* eSpeak is a rule-based synthesiser front-end, not a pronunciation lexicon: fallback
  pronunciations are lower-confidence than CMUdict and are flagged as such.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

from asa_m2.exceptions import G2PError, UnmappedPhoneError
from asa_m2.g2p.variants import build_variant
from asa_m2.phonemes.inventory import PhoneInventory
from asa_m2.phonemes.mapping import PhoneMapping
from asa_m2.schemas import PronunciationVariant

_PRIMARY, _SECONDARY = "\u02c8", "\u02cc"
_VERSION_RE = re.compile(r"(\d+\.\d+(?:\.\d+)?)")
Runner = Callable[..., "subprocess.CompletedProcess[str]"]


class EspeakFallbackError(G2PError):
    """eSpeak could not provide a trustworthy pronunciation (``reason`` is a stable code)."""

    def __init__(self, reason: str, message: str, *, units: list[str] | None = None) -> None:
        super().__init__(message, details={"reason": reason, "units": units or []})
        self.reason = reason


@dataclass(frozen=True)
class EspeakInfo:
    executable: str
    version: str | None


class EspeakBackend:
    name = "espeak-ng"

    def __init__(
        self,
        inventory: PhoneInventory,
        mapping: PhoneMapping,
        *,
        executable: str = "espeak-ng",
        voice: str = "en-us",
        timeout_s: float = 10.0,
        runner: Runner | None = None,
    ) -> None:
        self.inventory = inventory
        self.mapping = mapping
        self.executable = executable
        self.voice = voice
        self.timeout_s = timeout_s
        self._run: Runner = runner or subprocess.run
        self._version: str | None = None
        self._version_checked = False

    # ---- environment -------------------------------------------------------------
    def is_available(self) -> bool:
        return shutil.which(self.executable) is not None

    def version(self) -> str | None:
        """Installed eSpeak-ng version string, or ``None`` if it cannot be determined."""
        if not self._version_checked:
            self._version_checked = True
            try:
                proc = self._run(
                    [self.executable, "--version"], capture_output=True, text=True,
                    encoding="utf-8", timeout=self.timeout_s, check=False,
                )
                match = _VERSION_RE.search(proc.stdout or "")
                self._version = match.group(1) if match else None
            except (OSError, subprocess.SubprocessError):
                self._version = None
        return self._version

    @property
    def source_detail(self) -> str:
        return f"espeak-ng {self.version() or 'unknown'}; voice={self.voice}"

    # ---- conversion --------------------------------------------------------------
    def raw_ipa_units(self, word: str) -> list[str]:
        """Run eSpeak and return its ``_``-separated phoneme units (stress marks kept)."""
        if not word or any(ch.isspace() for ch in word):
            raise EspeakFallbackError("espeak_invalid_input", f"eSpeak fallback needs a single word, got {word!r}")
        try:
            proc = self._run(
                [self.executable, "-q", "-v", self.voice, "--ipa", "--sep=_", "--", word],
                capture_output=True, text=True, encoding="utf-8", timeout=self.timeout_s, check=False,
            )
        except FileNotFoundError:
            raise G2PError(f"eSpeak-ng executable {self.executable!r} not found on PATH") from None
        except subprocess.TimeoutExpired:
            raise EspeakFallbackError("espeak_timeout", f"eSpeak-ng timed out after {self.timeout_s}s for {word!r}") from None
        if proc.returncode != 0:
            raise EspeakFallbackError(
                "espeak_failed", f"eSpeak-ng exited with {proc.returncode} for {word!r}: {(proc.stderr or '').strip()[:200]}"
            )
        out = (proc.stdout or "").strip()
        if not out:
            raise EspeakFallbackError("espeak_empty_output", f"eSpeak-ng produced no phonemes for {word!r}")
        if "(" in out or ")" in out:
            raise EspeakFallbackError("espeak_language_switch", f"eSpeak-ng switched language for {word!r}: {out!r}")
        if any(ch.isspace() for ch in out):
            raise EspeakFallbackError("espeak_multiword_output", f"eSpeak-ng returned several words for {word!r}: {out!r}")
        return [u for u in out.split("_") if u]

    def pronounce(self, word: str, *, variant_id: str | None = None) -> PronunciationVariant:
        """Return one mapped variant, or raise :class:`EspeakFallbackError`."""
        raw_units = self.raw_ipa_units(word)
        stripped: list[tuple[str, str | None]] = []
        for unit in raw_units:
            mark = unit[0] if unit and unit[0] in (_PRIMARY, _SECONDARY) else None
            stripped.append((unit.lstrip(_PRIMARY + _SECONDARY), mark))
        try:
            mapped = self.mapping.map_units([u for u, _ in stripped])
        except UnmappedPhoneError as exc:
            units = list(exc.details.get("units", []))  # type: ignore[call-overload]
            raise EspeakFallbackError(
                "espeak_unmapped_phone",
                f"{word!r}: eSpeak units {units} have no documented ARPABET mapping",
                units=units,
            ) from exc
        symbols: list[str] = []
        approximate = False
        for (_, mark), mapped_unit in zip(stripped, mapped, strict=True):  # one mapped unit per stripped unit
            approximate = approximate or mapped_unit.confidence == "approximate"
            stress_given = False
            for phone in mapped_unit.arpabet:
                if self.inventory.is_vowel(phone):
                    if mark is not None and not stress_given:
                        stress = 1 if mark == _PRIMARY else 2
                        stress_given = True
                    else:
                        stress = 0
                    symbols.append(f"{phone}{stress}")
                else:
                    symbols.append(phone)
        return build_variant(
            variant_id or f"espeak-ng:{word}:1", "espeak-ng", symbols, self.inventory,
            source_detail=self.source_detail, approximate_mapping=approximate,
        )
