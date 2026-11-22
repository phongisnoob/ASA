"""Runtime configuration (YAML -> validated pydantic models).

* Unknown keys are rejected (a typo must fail loudly, not be ignored).
* ``include:`` lets one YAML compose others; later/own keys override included ones.
* Relative filesystem paths (inventory, mapping, custom lexicon) are resolved against
  the directory of the YAML file that declares them.
* Nothing here reads the network or the environment except ``ASA_M2_CONFIG`` which
  selects the default config file.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from asa_m2.exceptions import ConfigError

CONFIG_ENV_VAR = "ASA_M2_CONFIG"
_PATH_KEYS: tuple[tuple[str, ...], ...] = (
    ("phonemes", "inventory_path"),
    ("phonemes", "mapping_path"),
    ("g2p", "custom_lexicon_path"),
)


def resources_dir() -> Path:
    """Directory holding the packaged default configs and inventory JSON files."""
    return Path(__file__).resolve().parent / "resources"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ProjectConfig(_Strict):
    name: str = "ASA_M2"
    schema_version: str = "1.0"


class AudioConfig(_Strict):
    target_sample_rate: int = Field(16000, gt=0)
    channels: Literal[1] = 1
    output_format: Literal["wav"] = "wav"
    subtype: Literal["PCM_16", "PCM_24", "PCM_32"] = "PCM_16"
    preserve_original: bool = True
    stereo_to_mono: Literal["mean", "left", "right", "reject"] = "mean"
    supported_extensions: list[str] = [".wav", ".flac", ".ogg", ".mp3"]
    min_duration_s: float = Field(0.1, gt=0)
    max_duration_s: float = Field(300.0, gt=0)
    min_source_sample_rate: int = Field(8000, gt=0)
    max_clipping_fraction: float = Field(0.01, ge=0, le=1)

    @field_validator("preserve_original")
    @classmethod
    def _must_preserve(cls, value: bool) -> bool:
        if not value:
            raise ValueError("audio.preserve_original must be true: M2 never modifies original recordings")
        return value

    @field_validator("supported_extensions")
    @classmethod
    def _extensions(cls, value: list[str]) -> list[str]:
        out = []
        for ext in value:
            if not ext.startswith(".") or len(ext) < 2:
                raise ValueError(f"extension {ext!r} must look like '.wav'")
            out.append(ext.lower())
        return out

    @model_validator(mode="after")
    def _durations(self) -> AudioConfig:
        if self.min_duration_s >= self.max_duration_s:
            raise ValueError("audio.min_duration_s must be < audio.max_duration_s")
        return self


class CorpusConfig(_Strict):
    transcript_extension: Literal[".lab", ".txt"] = ".lab"
    on_error: Literal["fail", "skip"] = "fail"
    blocked_manifest_columns: list[str] = [
        "name", "student_name", "student_id", "address", "email", "phone", "birthdate", "dob",
    ]
    mfa_corpus_dirname: str = "mfa_corpus"
    dictionary_dirname: str = "dictionary"
    dictionary_filename: str = "asa_m2_corpus.dict"


class SpeakerConfig(_Strict):
    missing_id_policy: Literal["unique_per_recording", "error", "shared"] = "unique_per_recording"
    shared_speaker_id: str | None = None
    grouping: Literal["by_speaker_id", "per_recording"] = "by_speaker_id"
    pseudo_id_prefix: str = "SPK_UNK"

    @model_validator(mode="after")
    def _shared(self) -> SpeakerConfig:
        if self.missing_id_policy == "shared" and not self.shared_speaker_id:
            raise ValueError("speaker.missing_id_policy 'shared' requires speaker.shared_speaker_id")
        return self


class NormalizationConfig(_Strict):
    lowercase: bool = True
    normalize_unicode_apostrophes: bool = True
    hyphen_policy: Literal["split", "keep"] = "split"
    numbers: Literal["expand", "flag"] = "expand"
    ordinals: Literal["expand", "flag"] = "expand"
    abbreviations: dict[str, str] = {
        "mr": "mister", "mrs": "missus", "dr": "doctor", "prof": "professor",
        "jr": "junior", "sr": "senior", "vs": "versus", "etc": "et cetera",
    }
    symbols: dict[str, str] = {"&": "and", "%": "percent", "+": "plus", "=": "equals", "@": "at"}


class EspeakConfig(_Strict):
    executable: str = "espeak-ng"
    voice: str = "en-us"
    timeout_s: float = Field(10.0, gt=0)
    mapping_name: str = "espeak_en_us_ipa"


class G2PConfig(_Strict):
    primary_backend: Literal["cmudict"] = "cmudict"
    fallback_backend: Literal["espeak"] | None = None
    allow_multiple_variants: bool = True
    variant_policy: Literal["keep_all", "first_listed"] = "keep_all"
    fail_on_unresolved_oov: bool = True
    custom_lexicon_path: str | None = None
    custom_lexicon_mode: Literal["override", "augment"] = "override"
    espeak: EspeakConfig = EspeakConfig()

    @model_validator(mode="after")
    def _variants(self) -> G2PConfig:
        if not self.allow_multiple_variants and self.variant_policy != "first_listed":
            raise ValueError(
                "g2p.allow_multiple_variants=false exports a single variant to the aligner dictionary; "
                "that is a pronunciation choice and must be explicit: set g2p.variant_policy: first_listed"
            )
        return self


def _packaged(name: str) -> str:
    return str(resources_dir() / name)


class PhonemesConfig(_Strict):
    # Defaults point at the copies packaged with asa_m2, so a partial user config that does not
    # mention these keys still finds the inventory (never resolved against the current directory).
    inventory_path: str = Field(default_factory=lambda: _packaged("phoneme_inventory.json"))
    mapping_path: str = Field(default_factory=lambda: _packaged("phoneme_mapping.json"))
    preserve_stress: bool = True


class MFAConfig(_Strict):
    mode: Literal["legacy", "hf"] = "legacy"
    executable: str = "mfa"
    dictionary: str = "generated"
    acoustic_model: str | None = None
    hf_model_id: str | None = None
    hf_dialect: str | None = None
    hf_use_g2p: bool = False
    output_format: Literal["long_textgrid", "short_textgrid"] = "long_textgrid"
    num_jobs: int | None = Field(None, gt=0)
    beam: int | None = Field(None, gt=0)
    retry_beam: int | None = Field(None, gt=0)
    single_speaker: bool = False
    textgrid_cleanup: bool | None = None
    temporary_directory: str | None = None
    mfa_root_dir: str | None = None
    ignore_oovs: bool = False
    fine_tune: bool = False
    extra_args: list[str] = []
    timeout_s: float = Field(7200.0, gt=0)
    # Versions whose CLI source was inspected. Prefix match on "major.minor.".
    verified_versions: list[str] = ["3.4.3"]
    supported_version_prefixes: list[str] = ["3.4."]
    allow_unverified_versions: bool = False
    fail_on_missing_output: bool = True
    run_mfa_validate: bool = False


class QwenConfig(_Strict):
    model_name: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    language: str = "English"
    device: str = "auto"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    max_audio_seconds: float = Field(300.0, gt=0)


class TextGridConfig(_Strict):
    word_tier_names: list[str] = ["words", "word"]
    phone_tier_names: list[str] = ["phones", "phone"]
    allow_speaker_prefix: bool = True
    extra_silence_labels: list[str] = []
    extra_oov_labels: list[str] = []


class QCConfig(_Strict):
    """Alignment sanity thresholds. NOT pronunciation-error thresholds."""

    min_phone_duration_s: float = Field(0.02, ge=0)
    max_phone_duration_s: float = Field(0.8, gt=0)
    min_word_duration_s: float = Field(0.03, ge=0)
    max_word_duration_s: float = Field(3.0, gt=0)
    gap_tolerance_s: float = Field(0.001, ge=0)
    overlap_tolerance_s: float = Field(1e-6, ge=0)
    duration_mismatch_tolerance_s: float = Field(0.05, ge=0)
    boundary_tolerance_s: float = Field(0.02, gt=0)

    @model_validator(mode="after")
    def _order(self) -> QCConfig:
        if self.min_phone_duration_s >= self.max_phone_duration_s:
            raise ValueError("qc.min_phone_duration_s must be < qc.max_phone_duration_s")
        if self.min_word_duration_s >= self.max_word_duration_s:
            raise ValueError("qc.min_word_duration_s must be < qc.max_word_duration_s")
        return self


class AlignmentConfig(_Strict):
    primary: Literal["mfa"] = "mfa"
    optional_word_aligner: Literal["qwen"] | None = "qwen"
    export_textgrid: bool = True
    export_json: bool = True
    mfa: MFAConfig = MFAConfig()
    qwen: QwenConfig = QwenConfig()
    textgrid: TextGridConfig = TextGridConfig()
    qc: QCConfig = QCConfig()


class OutputConfig(_Strict):
    include_provenance: bool = True
    include_quality_flags: bool = True
    include_timestamps: bool = True
    json_indent: int | None = 2


class AppConfig(_Strict):
    """Complete validated configuration."""

    project: ProjectConfig = ProjectConfig()
    audio: AudioConfig = AudioConfig()
    corpus: CorpusConfig = CorpusConfig()
    speaker: SpeakerConfig = SpeakerConfig()
    normalization: NormalizationConfig = NormalizationConfig()
    g2p: G2PConfig = G2PConfig()
    phonemes: PhonemesConfig = PhonemesConfig()
    alignment: AlignmentConfig = AlignmentConfig()
    output: OutputConfig = OutputConfig()

    def fingerprint(self) -> str:
        """SHA-256 over the canonical JSON form (recorded in provenance).

        Filesystem paths are reduced to their file names so the fingerprint does not depend on
        where the project is installed; the inventory id/version and lexicon hash are recorded
        separately in provenance.
        """
        data = self.model_dump(mode="json")
        for section, key in _PATH_KEYS:
            value = data.get(section, {}).get(key)
            if isinstance(value, str) and value:
                data[section][key] = Path(value).name
        blob = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_yaml_tree(path: Path, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in _stack:
        chain = " -> ".join(str(p) for p in (*_stack, path))
        raise ConfigError(f"circular 'include' in configuration: {chain}")
    try:
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except FileNotFoundError:
        raise ConfigError(f"configuration file not found: {path}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"configuration {path} must contain a mapping at top level")
    includes = raw.pop("include", [])
    if isinstance(includes, str):
        includes = [includes]
    if not isinstance(includes, list):
        raise ConfigError(f"'include' in {path} must be a list of file names")
    merged: dict[str, Any] = {}
    for inc in includes:
        merged = _deep_merge(merged, _load_yaml_tree(path.parent / str(inc), (*_stack, path)))
    merged = _deep_merge(merged, raw)
    # resolve declared filesystem paths relative to *this* file, but only keys it defines itself
    for keys in _PATH_KEYS:
        node: Any = raw
        for k in keys:
            node = node.get(k) if isinstance(node, dict) else None
        if isinstance(node, str) and node:
            target = merged
            for k in keys[:-1]:
                target = target[k]
            candidate = Path(node)
            if not candidate.is_absolute():
                target[keys[-1]] = str((path.parent / candidate).resolve())
    return merged


def load_config(paths: Sequence[str | Path] | str | Path | None = None) -> AppConfig:
    """Load and validate configuration.

    ``paths`` may be one or several YAML files merged in order (later wins). With
    ``None`` the file named by ``$ASA_M2_CONFIG`` is used, falling back to the packaged
    ``resources/default.yaml``.
    """
    if paths is None:
        env = os.environ.get(CONFIG_ENV_VAR)
        paths = [env] if env else [resources_dir() / "default.yaml"]
    elif isinstance(paths, (str, Path)):
        paths = [paths]
    merged: dict[str, Any] = {}
    for p in paths:
        merged = _deep_merge(merged, _load_yaml_tree(Path(p)))
    try:
        return AppConfig.model_validate(merged)
    except ValidationError as exc:
        lines = [f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in exc.errors()]
        raise ConfigError(
            "invalid configuration:\n  - " + "\n  - ".join(lines), details={"errors": lines}
        ) from exc
