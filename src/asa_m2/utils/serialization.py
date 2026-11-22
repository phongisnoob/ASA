"""JSON / text writing helpers with explicit overwrite control."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from asa_m2.exceptions import OutputExistsError


def dumps_json(obj: Any, *, indent: int | None = 2) -> str:
    """Deterministic JSON text (UTF-8 friendly, insertion order preserved)."""
    return json.dumps(obj, indent=indent, ensure_ascii=False, default=_default) + "\n"


def _default(value: object) -> object:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def check_writable(path: Path, *, overwrite: bool) -> None:
    """Raise :class:`OutputExistsError` if ``path`` exists and overwrite was not requested."""
    if path.exists() and not overwrite:
        raise OutputExistsError(
            f"output {str(path)!r} already exists; pass --overwrite to replace it",
            details={"path": str(path)},
        )


def write_text_atomic(path: Path, text: str, *, overwrite: bool = False) -> Path:
    """Write ``text`` atomically (temp file + rename). Refuses to clobber unless ``overwrite``."""
    check_writable(path, overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return path


def write_json(path: Path, obj: Any, *, overwrite: bool = False, indent: int | None = 2) -> Path:
    return write_text_atomic(path, dumps_json(obj, indent=indent), overwrite=overwrite)


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
