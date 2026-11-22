"""Version-aware wrapper around the external ``mfa`` command.

MFA is invoked as a subprocess (never imported) because its internals and CLI change
between releases.

Verified CLI surface (MFA 3.4.3 source, ``montreal_forced_aligner/command_line``):

* ``mfa version``
* ``mfa align CORPUS DICTIONARY ACOUSTIC_MODEL OUTPUT [options]``       (legacy models)
* ``mfa align_hf CORPUS MODEL_ID OUTPUT [options]``                     (Hugging Face models)
* ``mfa validate CORPUS DICTIONARY [--acoustic_model_path M] [options]``
* ``align`` options used here: ``--output_format {long_textgrid,short_textgrid,json,csv}``,
  ``--ignore_oovs``, ``--fine_tune``; ``align_hf``: ``--dialect``, ``--use_g2p``.
* common options used here: ``--num_jobs``, ``--clean/--no_clean``, ``--overwrite/--no_overwrite``,
  ``--temporary_directory``, ``--single_speaker``, ``--textgrid_cleanup/--no_textgrid_cleanup``.
* ``--beam`` / ``--retry_beam`` are accepted as pass-through aligner parameters.
* ``MFA_ROOT_DIR`` selects MFA's root directory.

Only versions listed in ``alignment.mfa.supported_version_prefixes`` are accepted (override
with ``allow_unverified_versions: true`` at your own risk). Source of truth for the options:
https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner and
https://montreal-forced-aligner.readthedocs.io/
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from asa_m2.config import MFAConfig
from asa_m2.exceptions import (
    ConfigError,
    MFACommandError,
    MFANotInstalledError,
    MFAOutputError,
    MFAVersionError,
    OutputExistsError,
)

log = logging.getLogger(__name__)
_VERSION_RE = re.compile(r"(?<![\d.])(\d+)\.(\d+)\.(\d+)(?![\d.])")
Runner = Callable[..., "subprocess.CompletedProcess[str]"]


@dataclass
class MFARunResult:
    command: list[str]
    version: str | None
    return_code: int | None
    duration_s: float
    stdout_log: Path | None
    stderr_log: Path | None
    command_file: Path | None
    expected_samples: list[str]
    textgrids: dict[str, Path]
    missing_samples: list[str]
    status: str  # "success" | "partial" | "failed"
    warnings: list[str] = field(default_factory=list)


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}.{i}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise OutputExistsError(f"cannot find a free log file name near {path}")  # pragma: no cover


class MFAAdapter:
    """Controlled MFA invocation with version gating, logging and output verification."""

    def __init__(
        self,
        config: MFAConfig,
        *,
        runner: Runner | None = None,
        which: Callable[[str], str | None] | None = None,
    ) -> None:
        self.config = config
        self._run: Runner = runner or subprocess.run
        self._which = which or shutil.which
        self._executable: str | None = None
        self._version: str | None = None

    # ---- environment -------------------------------------------------------------
    def find_executable(self) -> str:
        if self._executable is None:
            found = self._which(self.config.executable)
            if not found:
                raise MFANotInstalledError(
                    f"Montreal Forced Aligner executable {self.config.executable!r} was not found on PATH. "
                    "Install MFA in its own conda environment and activate it (see README, 'MFA installation').",
                    details={"executable": self.config.executable},
                )
            self._executable = found
        return self._executable

    def get_version(self) -> str:
        """Return ``major.minor.patch`` reported by ``mfa version``."""
        if self._version is None:
            exe = self.find_executable()
            try:
                proc = self._run([exe, "version"], capture_output=True, text=True, encoding="utf-8",
                                 timeout=120, check=False, env=self._env())
            except FileNotFoundError:
                raise MFANotInstalledError(f"cannot execute {exe!r}") from None
            except subprocess.TimeoutExpired:
                raise MFACommandError("`mfa version` did not finish within 120 s") from None
            if proc.returncode != 0:
                raise MFAVersionError(
                    f"`mfa version` failed (exit {proc.returncode}): {_tail(proc.stderr or proc.stdout or '', 3)}. "
                    "The MFA installation looks broken: MFA needs its Kaldi/kalpy binaries, which `pip install` alone does not "
                    "provide - install MFA with conda (see README, 'MFA installation').",
                    details={"stderr": proc.stderr},
                )
            match = _VERSION_RE.search((proc.stdout or "") + "\n" + (proc.stderr or ""))
            if not match:
                raise MFAVersionError(
                    f"could not parse an MFA version from `mfa version` output: {(proc.stdout or '').strip()[:200]!r}"
                )
            self._version = ".".join(match.groups())
        return self._version

    def check_version(self, version: str | None = None) -> list[str]:
        """Raise :class:`MFAVersionError` for unsupported versions; return advisory warnings."""
        version = version or self.get_version()
        cfg = self.config
        if version in cfg.verified_versions:
            return []
        if any(version.startswith(p) for p in cfg.supported_version_prefixes):
            return [f"MFA {version}: series accepted but this exact patch version was not inspected "
                    f"(verified: {cfg.verified_versions})"]
        if cfg.allow_unverified_versions:
            return [f"MFA {version} is outside the verified range {cfg.verified_versions}; "
                    "proceeding because alignment.mfa.allow_unverified_versions is true. Commands may fail."]
        raise MFAVersionError(
            f"unsupported MFA version {version}. Verified: {cfg.verified_versions}; accepted series: "
            f"{cfg.supported_version_prefixes}. Install a supported release, or set "
            "alignment.mfa.allow_unverified_versions: true to try anyway (unsupported).",
            details={"version": version},
        )

    # ---- command construction ----------------------------------------------------
    def _common_options(self, *, overwrite: bool) -> list[str]:
        c = self.config
        opts = ["--output_format", c.output_format, "--clean", "--overwrite" if overwrite else "--no_overwrite"]
        if c.num_jobs:
            opts += ["--num_jobs", str(c.num_jobs)]
        if c.temporary_directory:
            opts += ["--temporary_directory", c.temporary_directory]
        if c.single_speaker:
            opts.append("--single_speaker")
        if c.textgrid_cleanup is not None:
            opts.append("--textgrid_cleanup" if c.textgrid_cleanup else "--no_textgrid_cleanup")
        if c.beam:
            opts += ["--beam", str(c.beam)]
        if c.retry_beam:
            opts += ["--retry_beam", str(c.retry_beam)]
        return opts

    def build_align_command(
        self, corpus_dir: Path, output_dir: Path, *, dictionary: str | None = None, overwrite: bool = False
    ) -> list[str]:
        """Return the ``mfa align``/``mfa align_hf`` argv list (no shell)."""
        c = self.config
        exe = self.find_executable()
        if c.mode == "legacy":
            if not c.acoustic_model:
                raise ConfigError("alignment.mfa.acoustic_model is required for mode 'legacy' "
                                  "(e.g. an installed model name or a path to a model zip)")
            dict_arg = dictionary if (c.dictionary == "generated" and dictionary) else c.dictionary
            if dict_arg == "generated":
                raise ConfigError("alignment.mfa.dictionary is 'generated' but no generated dictionary path was supplied; "
                                  "run `asa-m2 prepare-corpus` first or configure a dictionary name/path")
            cmd = [exe, "align", str(corpus_dir), dict_arg, c.acoustic_model, str(output_dir)]
            cmd += self._common_options(overwrite=overwrite)
            if c.ignore_oovs:
                cmd.append("--ignore_oovs")
            if c.fine_tune:
                cmd.append("--fine_tune")
        else:
            if not c.hf_model_id:
                raise ConfigError("alignment.mfa.hf_model_id is required for mode 'hf'")
            if c.ignore_oovs:
                raise ConfigError("alignment.mfa.ignore_oovs is not an option of `mfa align_hf` in the verified CLI")
            cmd = [exe, "align_hf", str(corpus_dir), c.hf_model_id, str(output_dir)]
            cmd += self._common_options(overwrite=overwrite)
            if c.hf_dialect:
                cmd += ["--dialect", c.hf_dialect]
            if c.hf_use_g2p:
                cmd.append("--use_g2p")
            if c.fine_tune:
                cmd.append("--fine_tune")
        cmd += list(c.extra_args)
        return cmd

    def build_validate_command(self, corpus_dir: Path, dictionary: str) -> list[str]:
        if self.config.mode != "legacy":
            raise ConfigError("`mfa validate` wrapper is only implemented for mode 'legacy'")
        cmd = [self.find_executable(), "validate", str(corpus_dir), dictionary, "--clean"]
        if self.config.acoustic_model:
            cmd += ["--acoustic_model_path", self.config.acoustic_model]
        if self.config.num_jobs:
            cmd += ["--num_jobs", str(self.config.num_jobs)]
        return cmd

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.config.mfa_root_dir:
            env["MFA_ROOT_DIR"] = self.config.mfa_root_dir
        return env

    # ---- execution ---------------------------------------------------------------
    def _execute(self, cmd: list[str], log_dir: Path | None, tag: str, version: str | None) -> tuple[subprocess.CompletedProcess[str] | None, float, Path | None, Path | None, Path | None, str | None]:
        stdout_log = stderr_log = command_file = None
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            command_file = _unique_path(log_dir / f"{tag}.command.json")
            command_file.write_text(json.dumps({
                "command": cmd, "mfa_version": version, "started_utc": datetime.now(timezone.utc).isoformat(),
                "env_additions": {"MFA_ROOT_DIR": self.config.mfa_root_dir} if self.config.mfa_root_dir else {},
                "mfa_config": self.config.model_dump(mode="json"),
            }, indent=2) + "\n", encoding="utf-8")
        log.info("running MFA", extra={"command": cmd})
        started = time.monotonic()
        proc = None
        timeout_note: str | None = None
        partial_out = partial_err = ""
        try:
            proc = self._run(cmd, capture_output=True, text=True, encoding="utf-8",
                             timeout=self.config.timeout_s, check=False, env=self._env())
        except FileNotFoundError:
            raise MFANotInstalledError(f"cannot execute {cmd[0]!r}") from None
        except subprocess.TimeoutExpired as exc:
            timeout_note = f"MFA timed out after {self.config.timeout_s} s"
            partial_out = _as_text(exc.stdout)
            partial_err = _as_text(exc.stderr)
        duration = time.monotonic() - started
        if log_dir is not None:
            stdout_log = _unique_path(log_dir / f"{tag}.stdout.log")
            stderr_log = _unique_path(log_dir / f"{tag}.stderr.log")
            stdout_log.write_text(proc.stdout if proc else partial_out, encoding="utf-8")
            stderr_log.write_text(proc.stderr if proc else partial_err, encoding="utf-8")
        return proc, duration, stdout_log, stderr_log, command_file, timeout_note

    def validate_corpus(self, corpus_dir: Path, *, dictionary: str, log_dir: Path | None = None) -> MFARunResult:
        """Run ``mfa validate`` (legacy mode); raises :class:`MFACommandError` on failure."""
        version = self.get_version()
        warnings = self.check_version(version)
        cmd = self.build_validate_command(corpus_dir, dictionary)
        proc, duration, out_log, err_log, cmd_file, timeout_note = self._execute(cmd, log_dir, "mfa_validate", version)
        if timeout_note or proc is None or proc.returncode != 0:
            raise MFACommandError(
                f"`mfa validate` failed: {timeout_note or _tail(proc.stderr if proc else '')}",
                details={"stdout_log": str(out_log), "stderr_log": str(err_log), "command": cmd},
            )
        return MFARunResult(cmd, version, proc.returncode, duration, out_log, err_log, cmd_file, [], {}, [], "success", warnings)

    def align(
        self,
        corpus_dir: Path,
        output_dir: Path,
        *,
        expected_samples: Sequence[str],
        dictionary: str | None = None,
        overwrite: bool = False,
        log_dir: Path | None = None,
        raise_on_missing: bool | None = None,
    ) -> MFARunResult:
        """Run alignment and verify that every expected TextGrid was produced.

        Raises :class:`MFAVersionError`, :class:`MFACommandError` (non-zero exit / timeout)
        or :class:`MFAOutputError` (missing TextGrids and ``fail_on_missing_output``).
        Logs and the exact command are always written when ``log_dir`` is given.
        """
        version = self.get_version()
        warnings = self.check_version(version)
        for w in warnings:
            log.warning(w)
        if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
            raise OutputExistsError(f"alignment output directory {output_dir} is not empty; pass --overwrite to reuse it",
                                    details={"path": str(output_dir)})
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = self.build_align_command(corpus_dir, output_dir, dictionary=dictionary, overwrite=overwrite)
        proc, duration, out_log, err_log, cmd_file, timeout_note = self._execute(cmd, log_dir, "mfa_align", version)
        details: dict[str, Any] = {"stdout_log": str(out_log), "stderr_log": str(err_log), "command": cmd}
        if timeout_note:
            raise MFACommandError(f"{timeout_note}; partial logs saved", details=details)
        assert proc is not None
        if proc.returncode != 0:
            raise MFACommandError(
                f"MFA exited with code {proc.returncode}: {_tail(proc.stderr or proc.stdout)}", details=details | {"return_code": proc.returncode}
            )
        found = {p.stem: p for p in sorted(output_dir.rglob("*.TextGrid"))}
        expected = list(expected_samples)
        missing = [s for s in expected if s not in found]
        unexpected = sorted(set(found) - set(expected))
        if unexpected:
            warnings.append(f"{len(unexpected)} unexpected TextGrid(s) in output: {unexpected[:5]}")
        status = "success" if not missing else "failed" if not (set(expected) & set(found)) else "partial"
        result = MFARunResult(cmd, version, proc.returncode, duration, out_log, err_log, cmd_file, expected,
                              {s: found[s] for s in expected if s in found}, missing, status, warnings)
        must_raise = self.config.fail_on_missing_output if raise_on_missing is None else raise_on_missing
        if missing and must_raise:
            raise MFAOutputError(
                f"MFA exited with code 0 but {len(missing)} of {len(expected)} expected TextGrid(s) are missing: "
                f"{missing[:10]}. A zero exit status is not proof of success; see the logs and "
                "`asa-m2` troubleshooting guide.", details=details | {"missing": missing},
            )
        return result


def _as_text(value: bytes | str | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else value


def _tail(text: str, lines: int = 8) -> str:
    return " | ".join(ln for ln in (text or "").strip().splitlines()[-lines:])
