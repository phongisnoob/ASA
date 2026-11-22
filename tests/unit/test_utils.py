"""Path-safety, serialization and logging helpers."""

import json
import logging
from pathlib import Path

import pytest

from asa_m2.exceptions import OutputExistsError, UnsafePathError
from asa_m2.logging_config import JsonFormatter, resolve_level, setup_logging
from asa_m2.schemas import Span
from asa_m2.utils.paths import (
    ensure_not_source,
    ensure_within,
    is_within,
    sha256_file,
    sha256_text,
    validate_identifier,
)
from asa_m2.utils.serialization import (
    check_writable,
    dumps_json,
    read_json,
    write_json,
    write_text_atomic,
)


@pytest.mark.parametrize("good", ["S001", "a", "SPK_001", "spk-1.v2", "A" * 128])
def test_valid_identifiers(good):
    assert validate_identifier(good) == good


@pytest.mark.parametrize("bad", ["", "../x", "a/b", "a\\b", " a", "a b", ".hidden", "-x", "a..b", "A" * 129, "é", "a\nb", "a\x00b"])
def test_invalid_identifiers(bad):
    with pytest.raises(UnsafePathError):
        validate_identifier(bad, kind="sample_id")


def test_validate_identifier_rejects_non_strings():
    with pytest.raises(UnsafePathError):
        validate_identifier(None)  # type: ignore[arg-type]


def test_within_and_traversal(tmp_path):
    base = tmp_path / "out"
    base.mkdir()
    assert is_within(base, base / "a" / "b.wav") and is_within(base, base)
    assert not is_within(base, base / ".." / "evil.wav") and not is_within(base, tmp_path / "other")
    assert ensure_within(base, base / "x.wav") == (base / "x.wav").resolve()
    with pytest.raises(UnsafePathError, match="escapes"):
        ensure_within(base, base / ".." / "evil.wav")


def test_symlink_escape_is_detected(tmp_path):
    base, outside = tmp_path / "out", tmp_path / "outside"
    base.mkdir(), outside.mkdir()
    (base / "link").symlink_to(outside, target_is_directory=True)
    assert not is_within(base, base / "link" / "file.wav")


def test_ensure_not_source(tmp_path):
    src = tmp_path / "a.wav"
    src.write_bytes(b"x")
    with pytest.raises(UnsafePathError, match="protected source"):
        ensure_not_source(tmp_path / "." / "a.wav", [src])
    ensure_not_source(tmp_path / "b.wav", [src])


def test_hashes(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"abc")
    assert sha256_file(p) == sha256_text("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_write_text_atomic_and_overwrite_rules(tmp_path):
    p = tmp_path / "d" / "f.txt"
    write_text_atomic(p, "one\n")
    with pytest.raises(OutputExistsError, match="--overwrite"):
        write_text_atomic(p, "two\n")
    assert p.read_text(encoding="utf-8") == "one\n"  # untouched after the refusal
    write_text_atomic(p, "two\n", overwrite=True)
    assert p.read_text(encoding="utf-8") == "two\n"
    assert [x.name for x in p.parent.iterdir()] == ["f.txt"]  # no temp files left behind


def test_write_failure_leaves_no_temp_file_and_keeps_original(tmp_path, monkeypatch):
    p = tmp_path / "f.txt"
    write_text_atomic(p, "keep\n")
    monkeypatch.setattr("os.replace", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        write_text_atomic(p, "new\n", overwrite=True)
    monkeypatch.undo()
    assert p.read_text(encoding="utf-8") == "keep\n" and [x.name for x in tmp_path.iterdir()] == ["f.txt"]


def test_json_helpers(tmp_path):
    p = write_json(tmp_path / "x.json", {"span": Span(start=1, end=2), "path": Path("a/b"), "set": {2, 1}, "u": "é"})
    assert read_json(p) == {"span": {"start": 1, "end": 2}, "path": "a/b", "set": [1, 2], "u": "é"}
    assert "é" in p.read_text(encoding="utf-8") and p.read_text(encoding="utf-8").endswith("}\n")  # readable, not \u-escaped
    with pytest.raises(TypeError):
        dumps_json({"x": object()})
    check_writable(tmp_path / "new.json", overwrite=False)
    with pytest.raises(OutputExistsError):
        check_writable(p, overwrite=False)


def test_resolve_level(monkeypatch):
    assert resolve_level("debug") == logging.DEBUG
    monkeypatch.setenv("ASA_M2_LOG_LEVEL", "WARNING")
    assert resolve_level(None) == logging.WARNING and resolve_level("ERROR") == logging.ERROR  # flag beats env
    with pytest.raises(ValueError):
        resolve_level("LOUD")


def test_setup_logging_is_idempotent_and_writes_json_file(tmp_path):
    log_file = tmp_path / "logs" / "run.log"
    logger = setup_logging("INFO", log_file=log_file)
    logger = setup_logging("INFO", log_file=log_file)
    assert len(logger.handlers) == 2  # replaced, not stacked
    logging.getLogger("asa_m2.test").info("hello", extra={"sample": "S1"})
    for h in logger.handlers:
        h.flush()
    rec = json.loads(log_file.read_text(encoding="utf-8").splitlines()[-1])
    assert rec["msg"] == "hello" and rec["sample"] == "S1" and rec["level"] == "INFO"
    setup_logging("INFO")  # restore: no file handler


def test_json_formatter_includes_exceptions():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        record = logging.LogRecord("n", logging.ERROR, "f", 1, "failed", (), sys.exc_info())
    assert "ValueError: boom" in json.loads(JsonFormatter().format(record))["exc"]
