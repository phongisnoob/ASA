"""Explicit exception hierarchy for ASA M2.

Every error raised deliberately by this package derives from :class:`M2Error` and
carries an ``exit_code`` that the CLI maps to the process exit status.

Exit-code convention
--------------------
==== ===========================================================
Code Meaning
==== ===========================================================
0    success
1    unexpected internal error (a bug - please report)
2    command-line usage error (argparse default)
3    configuration error
4    input / data validation error (bad text, audio, corpus, ...)
5    external tool error (MFA missing, unsupported or failed, ...)
6    alignment / TextGrid error
7    optional dependency missing or unusable
==== ===========================================================
"""

from __future__ import annotations

EXIT_OK = 0
EXIT_INTERNAL = 1
EXIT_USAGE = 2
EXIT_CONFIG = 3
EXIT_DATA = 4
EXIT_EXTERNAL = 5
EXIT_ALIGNMENT = 6
EXIT_OPTIONAL_DEP = 7


class M2Error(Exception):
    """Base class for all deliberate ASA M2 errors."""

    exit_code: int = EXIT_INTERNAL

    def __init__(self, message: str, *, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, object] = details or {}


# --- configuration -----------------------------------------------------------------
class ConfigError(M2Error):
    exit_code = EXIT_CONFIG


# --- text / G2P / inventory --------------------------------------------------------
class NormalizationError(M2Error):
    exit_code = EXIT_DATA


class G2PError(M2Error):
    exit_code = EXIT_DATA


class UnresolvedPronunciationError(G2PError):
    """Raised when unresolved words would enter forced alignment without an override."""


class LexiconError(G2PError):
    """Malformed custom lexicon file."""


class InventoryError(M2Error):
    exit_code = EXIT_DATA


class MappingError(InventoryError):
    """A phone mapping is invalid."""


class UnmappedPhoneError(MappingError):
    """A source phone has no documented mapping (never silently dropped or remapped)."""


# --- audio / corpus ----------------------------------------------------------------
class AudioError(M2Error):
    exit_code = EXIT_DATA


class AudioNotFoundError(AudioError):
    pass


class AudioReadError(AudioError):
    pass


class AudioValidationError(AudioError):
    pass


class CorpusError(M2Error):
    exit_code = EXIT_DATA


class UnsafePathError(M2Error):
    """A path would escape its allowed directory or overwrite protected source data."""

    exit_code = EXIT_DATA


class OutputExistsError(M2Error):
    """Refusing to overwrite existing output without explicit permission."""

    exit_code = EXIT_DATA


# --- external tools ----------------------------------------------------------------
class MFAError(M2Error):
    exit_code = EXIT_EXTERNAL


class MFANotInstalledError(MFAError):
    pass


class MFAVersionError(MFAError):
    pass


class MFACommandError(MFAError):
    pass


class MFAOutputError(MFAError):
    pass


# --- alignment ---------------------------------------------------------------------
class TextGridError(M2Error):
    exit_code = EXIT_ALIGNMENT


class AlignmentError(M2Error):
    exit_code = EXIT_ALIGNMENT


class OptionalDependencyError(M2Error):
    exit_code = EXIT_OPTIONAL_DEP


class QwenResourceError(OptionalDependencyError):
    """Qwen3-ForcedAligner could not run (no GPU / out of memory / model unavailable)."""
