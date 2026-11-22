"""Configuration loading and validation."""

from pathlib import Path

import pytest

from asa_m2.config import AppConfig, load_config, resources_dir
from asa_m2.exceptions import ConfigError


def test_default_config_loads_composed_from_includes(config):
    assert config.project.schema_version == "1.0"
    assert config.audio.target_sample_rate == 16000 and config.audio.preserve_original
    assert config.g2p.primary_backend == "cmudict" and config.g2p.fallback_backend is None
    assert config.g2p.fail_on_unresolved_oov and config.g2p.allow_multiple_variants
    assert config.speaker.missing_id_policy == "unique_per_recording"
    assert config.alignment.primary == "mfa" and config.alignment.optional_word_aligner == "qwen"
    assert Path(config.phonemes.inventory_path).is_file() and Path(config.phonemes.inventory_path).is_absolute()


def test_fingerprint_is_stable_and_changes_with_settings(config):
    a = config.fingerprint()
    assert a == load_config().fingerprint() and len(a) == 64
    config.audio.target_sample_rate = 22050
    assert config.fingerprint() != a


def test_unknown_keys_are_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("audio:\n  target_sample_rte: 8000\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="target_sample_rte"):
        load_config(p)


@pytest.mark.parametrize("yaml_text,fragment", [
    ("audio:\n  target_sample_rate: -5\n", "target_sample_rate"),
    ("audio:\n  channels: 2\n", "channels"),
    ("audio:\n  preserve_original: false\n", "never modifies"),
    ("audio:\n  stereo_to_mono: sum\n", "stereo_to_mono"),
    ("g2p:\n  primary_backend: festival\n", "primary_backend"),
    ("g2p:\n  allow_multiple_variants: false\n", "first_listed"),
    ("speaker:\n  missing_id_policy: shared\n", "shared_speaker_id"),
    ("alignment:\n  qc:\n    min_phone_duration_s: 5\n    max_phone_duration_s: 1\n", "min_phone_duration_s"),
    ("alignment:\n  mfa:\n    num_jobs: 0\n", "num_jobs"),
    ("output: 5\n", "output"),
])
def test_invalid_values_fail_at_startup(tmp_path, yaml_text, fragment):
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ConfigError, match=fragment):
        load_config(p)


def test_explicit_single_variant_export_is_valid_with_policy(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("g2p:\n  allow_multiple_variants: false\n  variant_policy: first_listed\n", encoding="utf-8")
    assert load_config(p).g2p.variant_policy == "first_listed"


def test_missing_and_malformed_yaml(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(bad)
    lst = tmp_path / "list.yaml"
    lst.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(lst)


def test_empty_yaml_gives_defaults(tmp_path):
    p = tmp_path / "e.yaml"
    p.write_text("", encoding="utf-8")
    assert load_config(p).audio.target_sample_rate == 16000


def test_circular_include_detected(tmp_path):
    (tmp_path / "a.yaml").write_text("include: [b.yaml]\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("include: [a.yaml]\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="circular"):
        load_config(tmp_path / "a.yaml")


def test_multiple_files_merge_later_wins_and_relative_paths_resolve_to_file(tmp_path):
    (tmp_path / "base.yaml").write_text("audio:\n  target_sample_rate: 8000\n  min_duration_s: 0.2\n", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "over.yaml").write_text("audio:\n  target_sample_rate: 22050\ng2p:\n  custom_lexicon_path: lex.dict\n", encoding="utf-8")
    cfg = load_config([tmp_path / "base.yaml", sub / "over.yaml"])
    assert cfg.audio.target_sample_rate == 22050 and cfg.audio.min_duration_s == 0.2
    assert cfg.g2p.custom_lexicon_path == str((sub / "lex.dict").resolve())


def test_env_var_selects_default_config(tmp_path, monkeypatch):
    p = tmp_path / "env.yaml"
    p.write_text("audio:\n  target_sample_rate: 12345\n", encoding="utf-8")
    monkeypatch.setenv("ASA_M2_CONFIG", str(p))
    assert load_config().audio.target_sample_rate == 12345


def test_repo_configs_directory_matches_packaged_resources():
    """configs/ (user-editable) and asa_m2/resources/ (packaged) must never drift apart."""
    root = Path(__file__).resolve().parents[2]
    names = ["default.yaml", "g2p.yaml", "alignment.yaml", "audio.yaml", "phoneme_inventory.json",
             "phone_to_id.json", "id_to_phone.json", "phoneme_mapping.json"]
    for n in names:
        assert (root / "configs" / n).read_bytes() == (resources_dir() / n).read_bytes(), n


def test_repo_inventory_files_match_the_generator():
    import importlib.util
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("gen_inv", root / "scripts" / "generate_inventory.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name, obj in mod.build().items():
        assert (root / "configs" / name).read_text(encoding="utf-8") == mod.render(obj), name


def test_appconfig_defaults_are_valid_without_files():
    assert AppConfig().audio.target_sample_rate == 16000


def test_partial_config_still_finds_packaged_inventory_from_any_directory(tmp_path, monkeypatch):
    """Regression: a config that never mentions phonemes.* used to resolve 'phoneme_inventory.json' against the CWD."""
    monkeypatch.chdir(tmp_path)  # a directory that certainly has no inventory
    p = tmp_path / "partial.yaml"
    p.write_text("g2p:\n  fail_on_unresolved_oov: false\n", encoding="utf-8")
    cfg = load_config(p)
    assert Path(cfg.phonemes.inventory_path) == resources_dir() / "phoneme_inventory.json"
    from asa_m2.phonemes import PhoneInventory
    assert len(PhoneInventory.from_files(cfg.phonemes.inventory_path)) == 39


def test_fingerprint_does_not_depend_on_install_location(tmp_path):
    a = load_config()
    b = load_config()
    b.phonemes.inventory_path = str(tmp_path / "elsewhere" / "phoneme_inventory.json")
    assert a.fingerprint() == b.fingerprint()  # same file name => same fingerprint, whatever the directory
    b.audio.target_sample_rate = 8000
    assert a.fingerprint() != b.fingerprint()
