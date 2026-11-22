"""Structured logging helpers.

All asa-m2 loggers live under the ``asa_m2`` namespace. ``setup_logging`` configures
them once per process; library code only ever calls ``logging.getLogger(__name__)``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

LOGGER_NAME = "asa_m2"
_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """One JSON object per line, including any ``extra=`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class KeyValueFormatter(logging.Formatter):
    """Human-readable single-line format with ``key=value`` extras."""

    def format(self, record: logging.LogRecord) -> str:
        base = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<7} {record.name}: {record.getMessage()}"
        extras = " ".join(
            f"{k}={v!r}" for k, v in record.__dict__.items() if k not in _RESERVED
        )
        text = f"{base} [{extras}]" if extras else base
        if record.exc_info:
            text += "\n" + self.formatException(record.exc_info)
        return text


def resolve_level(level: str | None) -> int:
    """Resolve a level name (CLI flag > ``ASA_M2_LOG_LEVEL`` > INFO)."""
    name = (level or os.environ.get("ASA_M2_LOG_LEVEL") or "INFO").upper()
    if name not in _VALID_LEVELS:
        raise ValueError(f"invalid log level {name!r}; choose from {sorted(_VALID_LEVELS)}")
    return int(getattr(logging, name))


def setup_logging(
    level: str | None = None,
    *,
    log_file: str | Path | None = None,
    json_format: bool = False,
) -> logging.Logger:
    """Configure the ``asa_m2`` logger (idempotent: replaces previous handlers)."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(resolve_level(level))
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter: logging.Formatter = JsonFormatter() if json_format else KeyValueFormatter()
    stream = logging.StreamHandler(sys.stderr)  # stdout is reserved for command results
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)
    return logger
