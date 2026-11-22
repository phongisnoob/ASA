"""Path-safety helpers.

M2 must never overwrite original recordings and must never write outside the output
directory it was given. All identifiers that become file or folder names pass through
:func:`validate_identifier`.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path

from asa_m2.exceptions import UnsafePathError

# Conservative whitelist: letters, digits, underscore, dot, hyphen; no leading dot/hyphen.
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def validate_identifier(value: str, *, kind: str = "identifier") -> str:
    """Return ``value`` if it is safe to use as a file/folder name, else raise.

    Rejects path separators, ``..``, whitespace, empty strings and leading dots.
    """
    if not isinstance(value, str) or not IDENTIFIER_RE.match(value) or ".." in value:
        raise UnsafePathError(
            f"unsafe or invalid {kind} {value!r}: allowed characters are A-Z a-z 0-9 _ . - "
            "(max 128, must start with a letter or digit, no '..')",
            details={"kind": kind, "value": value},
        )
    return value


def is_within(base: Path, candidate: Path) -> bool:
    """True when ``candidate`` resolves to a location inside (or equal to) ``base``."""
    try:
        candidate.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def ensure_within(base: Path, candidate: Path) -> Path:
    """Return the resolved ``candidate`` or raise :class:`UnsafePathError` if it escapes ``base``."""
    if not is_within(base, candidate):
        raise UnsafePathError(
            f"path {str(candidate)!r} escapes the allowed directory {str(base)!r}",
            details={"base": str(base), "path": str(candidate)},
        )
    return candidate.resolve()


def ensure_not_source(output_path: Path, protected: Iterable[Path]) -> None:
    """Raise if ``output_path`` is one of the protected source files (or inside one)."""
    out = output_path.resolve()
    for src in protected:
        if out == src.resolve():
            raise UnsafePathError(
                f"refusing to write to {str(out)!r}: it is a protected source file",
                details={"path": str(out)},
            )


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
