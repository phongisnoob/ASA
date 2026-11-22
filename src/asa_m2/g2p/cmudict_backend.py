"""CMU Pronouncing Dictionary backend (primary pronunciation source).

All variants are returned in the order they appear in the dictionary
(``word``, ``word(2)``, ``word(3)`` ...). Order carries no claim about which variant is
"correct" or most frequent; variant IDs are ``cmudict:<word>:<k>`` with ``k`` 1-based.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from importlib import metadata

from asa_m2.exceptions import G2PError


@lru_cache(maxsize=1)
def _load_packaged_cmudict() -> Mapping[str, Sequence[Sequence[str]]]:
    """Parse the packaged CMUdict once per process.

    Module-level cache is justified: the data is large (~126k entries), read-only, and never
    mutated or handed out by reference (``lookup`` returns copies).
    """
    import cmudict  # local data shipped with the package; no network access

    return cmudict.dict()


class CMUdictBackend:
    name = "cmudict"

    def __init__(self, entries: Mapping[str, Sequence[Sequence[str]]] | None = None) -> None:
        if entries is None:
            try:
                entries = _load_packaged_cmudict()
            except ImportError as exc:  # pragma: no cover - dependency is mandatory
                raise G2PError("the 'cmudict' package is required for G2P (pip install cmudict)") from exc
            self.version: str | None = _package_version("cmudict")
        else:
            self.version = None
        self._entries = entries

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, word: object) -> bool:
        return isinstance(word, str) and word.lower() in self._entries

    def lookup(self, word: str) -> list[list[str]]:
        """Return every dictionary variant of ``word`` (empty list = out of vocabulary)."""
        return [list(v) for v in self._entries.get(word.lower(), [])]

    @property
    def source_detail(self) -> str:
        return f"cmudict=={self.version}" if self.version else "cmudict"


def _package_version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None
