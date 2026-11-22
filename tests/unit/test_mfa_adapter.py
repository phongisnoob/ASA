"""M2.5 MFA wrapper. Uses a FAKE `mfa` executable (tests/fixtures/fake_mfa.py): these tests prove
asa-m2's *wrapper* behaviour; they do NOT prove anything about real Montreal Forced Aligner."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from asa_m2.alignment.mfa_adapter import MFAAdapter
from asa_m2.config import MFAConfig
from asa_m2.exceptions import (
    ConfigError,
    MFACommandError,
    MFANotInstalledError,
    MFAOutputError,
    MFAVersionError,
    OutputExistsError,
)

from ..conftest import read_argv_log  # noqa: TID252


def cfg(**kw) -> MFAConfig:
    base = dict(acoustic_model="english_us_arpa", dictionary="generated", fail_on_missing_output=True)
    return MFAConfig(**{**base, **kw})


@pytest.fixture()
def corpus(tmp_path, wav_factory):
    root = tmp_path / "corpus"
    for spk, sid, text in (("A", "S1", "this is the cat"), ("B", "S2", "hello world")):
        wav_factory(root / spk / f"{sid}.wav", seconds=3.0)
        (root / spk / f"{sid}.lab").write_text(text + "\n", encoding="utf-8")
    d = tmp_path / "dict.dict"
    d.write_text("this\tDH IH1 S\nthis\tDH IH0 S\nis\tIH1 Z\nthe\tDH AH0\ncat\tK AE1 T\nhello\tHH AH0 L OW1\nworld\tW ER1 L D\n", encoding="utf-8")
    return root, d


# ---- installation / version ---------------------------------------------------------
def test_mfa_not_installed(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))  # nothing on PATH
    with pytest.raises(MFANotInstalledError, match="not found on PATH") as exc:
        MFAAdapter(cfg()).find_executable()
    assert "conda" in exc.value.message and exc.value.exit_code == 5


def test_version_detected_and_verified(fake_mfa):
    a = MFAAdapter(cfg())
    assert a.get_version() == "3.4.3" and a.check_version() == []


def test_exact_patch_not_inspected_gives_warning(fake_mfa, monkeypatch):
    monkeypatch.setenv("FAKE_MFA_VERSION", "3.4.9")
    a = MFAAdapter(cfg())
    warns = a.check_version()
    assert len(warns) == 1 and "not inspected" in warns[0]


@pytest.mark.parametrize("version", ["2.2.17", "3.0.0", "3.5.0", "4.0.1"])
def test_unsupported_versions_fail_with_useful_message(fake_mfa, monkeypatch, version):
    monkeypatch.setenv("FAKE_MFA_VERSION", version)
    with pytest.raises(MFAVersionError, match="unsupported MFA version") as exc:
        MFAAdapter(cfg()).check_version()
    assert version in exc.value.message and "allow_unverified_versions" in exc.value.message and "3.4." in exc.value.message


def test_unverified_version_can_be_explicitly_allowed_with_warning(fake_mfa, monkeypatch):
    monkeypatch.setenv("FAKE_MFA_VERSION", "3.1.2")
    warns = MFAAdapter(cfg(allow_unverified_versions=True)).check_version()
    assert warns and "outside the verified range" in warns[0]


def test_garbled_version_output(fake_mfa, monkeypatch):
    monkeypatch.setenv("FAKE_MFA_MODE", "bad_version")
    with pytest.raises(MFAVersionError, match="could not parse"):
        MFAAdapter(cfg()).get_version()


def test_broken_installation_reported(fake_mfa, monkeypatch):
    monkeypatch.setenv("FAKE_MFA_MODE", "broken_install")
    with pytest.raises(MFAVersionError, match="installation looks broken") as exc:
        MFAAdapter(cfg()).get_version()
    assert "_kalpy" in exc.value.details["stderr"]


# ---- command construction ----------------------------------------------------------------
def test_legacy_align_command(fake_mfa, tmp_path):
    a = MFAAdapter(cfg(num_jobs=4, beam=100, retry_beam=400, single_speaker=True, textgrid_cleanup=False,
                       temporary_directory="/tmp/mfa_tmp", ignore_oovs=True, fine_tune=True, extra_args=["--quiet"]))
    cmd = a.build_align_command(Path("/c"), Path("/o"), dictionary="/d.dict")
    assert cmd[1:6] == ["align", "/c", "/d.dict", "english_us_arpa", "/o"]  # positional order verified in 3.4.3 source
    for flag, value in (("--output_format", "long_textgrid"), ("--num_jobs", "4"), ("--beam", "100"),
                        ("--retry_beam", "400"), ("--temporary_directory", "/tmp/mfa_tmp")):
        assert cmd[cmd.index(flag) + 1] == value
    for flag in ("--clean", "--no_overwrite", "--single_speaker", "--no_textgrid_cleanup", "--ignore_oovs", "--fine_tune"):
        assert flag in cmd
    assert cmd[-1] == "--quiet"  # extra args are appended verbatim (and logged)


def test_overwrite_flag_and_defaults_do_not_invent_options(fake_mfa):
    cmd = MFAAdapter(cfg()).build_align_command(Path("/c"), Path("/o"), dictionary="/d", overwrite=True)
    assert "--overwrite" in cmd and "--no_overwrite" not in cmd
    for absent in ("--num_jobs", "--beam", "--single_speaker", "--textgrid_cleanup", "--no_textgrid_cleanup", "--ignore_oovs"):
        assert absent not in cmd


def test_textgrid_cleanup_true_flag(fake_mfa):
    assert "--textgrid_cleanup" in MFAAdapter(cfg(textgrid_cleanup=True)).build_align_command(Path("/c"), Path("/o"), dictionary="/d")


def test_explicit_dictionary_name_overrides_generated(fake_mfa):
    cmd = MFAAdapter(cfg(dictionary="english_us_arpa")).build_align_command(Path("/c"), Path("/o"), dictionary="/ignored")
    assert cmd[3] == "english_us_arpa" and "/ignored" not in cmd


def test_hf_mode_uses_align_hf_and_its_own_options(fake_mfa):
    cmd = MFAAdapter(cfg(mode="hf", hf_model_id="org/model", hf_dialect="us", hf_use_g2p=True, acoustic_model=None)
                     ).build_align_command(Path("/c"), Path("/o"))
    assert cmd[1:5] == ["align_hf", "/c", "org/model", "/o"] and "--dialect" in cmd and "--use_g2p" in cmd
    assert "english_us_arpa" not in cmd


def test_hf_mode_rejects_options_not_in_the_verified_cli(fake_mfa):
    with pytest.raises(ConfigError, match="ignore_oovs"):
        MFAAdapter(cfg(mode="hf", hf_model_id="x", ignore_oovs=True)).build_align_command(Path("/c"), Path("/o"))


@pytest.mark.parametrize("kw,msg", [
    (dict(acoustic_model=None), "acoustic_model is required"),
    (dict(mode="hf", hf_model_id=None), "hf_model_id is required"),
])
def test_missing_model_configuration(fake_mfa, kw, msg):
    with pytest.raises(ConfigError, match=msg):
        MFAAdapter(cfg(**kw)).build_align_command(Path("/c"), Path("/o"), dictionary="/d")


def test_generated_dictionary_without_path_is_an_error(fake_mfa):
    with pytest.raises(ConfigError, match="prepare-corpus"):
        MFAAdapter(cfg()).build_align_command(Path("/c"), Path("/o"))


def test_validate_command(fake_mfa):
    cmd = MFAAdapter(cfg(num_jobs=2)).build_validate_command(Path("/c"), "/d.dict")
    assert cmd[1:5] == ["validate", "/c", "/d.dict", "--clean"] and cmd[cmd.index("--acoustic_model_path") + 1] == "english_us_arpa"
    with pytest.raises(ConfigError):
        MFAAdapter(cfg(mode="hf", hf_model_id="x")).build_validate_command(Path("/c"), "d")


# ---- execution -------------------------------------------------------------------------------------
def test_successful_alignment_runs_and_verifies_outputs(fake_mfa, corpus, tmp_path):
    root, d = corpus
    res = MFAAdapter(cfg()).align(root, tmp_path / "out", expected_samples=["S1", "S2"], dictionary=str(d), log_dir=tmp_path / "logs")
    assert res.status == "success" and res.version == "3.4.3" and res.return_code == 0 and res.missing_samples == []
    assert sorted(res.textgrids) == ["S1", "S2"] and all(p.is_file() for p in res.textgrids.values())
    argv = read_argv_log(fake_mfa)
    align_call = next(c["argv"] for c in argv if c["argv"][0] == "align")
    assert align_call[:5] == ["align", str(root), str(d), "english_us_arpa", str(tmp_path / "out")]
    assert "--clean" in align_call and "--no_overwrite" in align_call


def test_logs_and_command_metadata_are_preserved(fake_mfa, corpus, tmp_path):
    root, d = corpus
    res = MFAAdapter(cfg(mfa_root_dir="/some/root")).align(root, tmp_path / "out", expected_samples=["S1", "S2"],
                                                           dictionary=str(d), log_dir=tmp_path / "logs")
    assert "fake MFA aligned 2" in res.stdout_log.read_text(encoding="utf-8")
    meta = json.loads(res.command_file.read_text(encoding="utf-8"))
    assert meta["command"] == res.command and meta["mfa_version"] == "3.4.3" and meta["env_additions"] == {"MFA_ROOT_DIR": "/some/root"}
    assert meta["mfa_config"]["acoustic_model"] == "english_us_arpa"
    assert {c["MFA_ROOT_DIR"] for c in read_argv_log(fake_mfa)} == {"/some/root"}  # really passed to the subprocess


def test_logs_are_never_overwritten(fake_mfa, corpus, tmp_path):
    root, d = corpus
    a = MFAAdapter(cfg())
    r1 = a.align(root, tmp_path / "o1", expected_samples=["S1", "S2"], dictionary=str(d), log_dir=tmp_path / "logs")
    r2 = a.align(root, tmp_path / "o2", expected_samples=["S1", "S2"], dictionary=str(d), log_dir=tmp_path / "logs")
    assert r1.stdout_log != r2.stdout_log and r1.stdout_log.exists() and r2.stdout_log.exists()


def test_nonzero_exit_is_a_failure_with_stderr_and_logs(fake_mfa, corpus, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MFA_MODE", "exit1")
    root, d = corpus
    with pytest.raises(MFACommandError, match="exited with code 1") as exc:
        MFAAdapter(cfg()).align(root, tmp_path / "out", expected_samples=["S1", "S2"], dictionary=str(d), log_dir=tmp_path / "logs")
    assert "simulated failure in feature generation" in exc.value.message
    assert exc.value.details["return_code"] == 1
    assert "simulated failure" in Path(exc.value.details["stderr_log"]).read_text(encoding="utf-8")  # error logging
    assert Path(exc.value.details["stdout_log"]).exists()


def test_zero_exit_with_missing_output_is_not_success(fake_mfa, corpus, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MFA_MODE", "missing_last")
    root, d = corpus
    with pytest.raises(MFAOutputError, match="zero exit status is not proof") as exc:
        MFAAdapter(cfg()).align(root, tmp_path / "out", expected_samples=["S1", "S2"], dictionary=str(d), log_dir=tmp_path / "logs")
    assert exc.value.details["missing"] == ["S2"]


def test_missing_output_can_be_reported_without_raising(fake_mfa, corpus, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MFA_MODE", "missing_last")
    root, d = corpus
    res = MFAAdapter(cfg()).align(root, tmp_path / "out", expected_samples=["S1", "S2"], dictionary=str(d), raise_on_missing=False)
    assert res.status == "partial" and res.missing_samples == ["S2"] and sorted(res.textgrids) == ["S1"]
    # fail_on_missing_output: false => reported (status partial) instead of raised; FAKE mode still skips S2, S3 never existed
    res2 = MFAAdapter(cfg(fail_on_missing_output=False)).align(root, tmp_path / "out2", expected_samples=["S1", "S2", "S3"], dictionary=str(d))
    assert res2.status == "partial" and res2.missing_samples == ["S2", "S3"]


def test_all_outputs_missing_is_failed(fake_mfa, corpus, tmp_path):
    root, d = corpus
    res = MFAAdapter(cfg()).align(root, tmp_path / "o", expected_samples=["X1", "X2"], dictionary=str(d), raise_on_missing=False)
    assert res.status == "failed" and res.textgrids == {}
    assert any("unexpected TextGrid" in w for w in res.warnings)


def test_timeout_kills_and_reports(fake_mfa, corpus, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MFA_MODE", "hang")
    root, d = corpus
    with pytest.raises(MFACommandError, match="timed out"):
        MFAAdapter(cfg(timeout_s=1)).align(root, tmp_path / "out", expected_samples=["S1"], dictionary=str(d), log_dir=tmp_path / "logs")
    assert list((tmp_path / "logs").glob("mfa_align*.stderr.log"))  # logs saved even on timeout


def test_unsupported_version_blocks_alignment_before_running(fake_mfa, corpus, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MFA_VERSION", "2.0.0")
    root, d = corpus
    with pytest.raises(MFAVersionError):
        MFAAdapter(cfg()).align(root, tmp_path / "out", expected_samples=["S1"], dictionary=str(d))
    assert all(c["argv"][0] != "align" for c in read_argv_log(fake_mfa))  # `align` never invoked


def test_non_empty_output_dir_requires_overwrite(fake_mfa, corpus, tmp_path):
    root, d = corpus
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(OutputExistsError):
        MFAAdapter(cfg()).align(root, out, expected_samples=["S1", "S2"], dictionary=str(d))
    res = MFAAdapter(cfg()).align(root, out, expected_samples=["S1", "S2"], dictionary=str(d), overwrite=True)
    assert res.status == "success" and "--overwrite" in res.command


def test_validate_corpus_wrapper(fake_mfa, corpus, tmp_path, monkeypatch):
    root, d = corpus
    res = MFAAdapter(cfg()).validate_corpus(root, dictionary=str(d), log_dir=tmp_path / "logs")
    assert res.status == "success" and res.command[1] == "validate"
    monkeypatch.setenv("FAKE_MFA_MODE", "validate_fail")
    with pytest.raises(MFACommandError, match="validate"):
        MFAAdapter(cfg()).validate_corpus(root, dictionary=str(d))


def test_runner_is_a_list_argv_never_a_shell(corpus, tmp_path):
    seen = []

    def runner(cmd, **kw):
        seen.append((cmd, kw))
        return SimpleNamespace(stdout="3.4.3\n", stderr="", returncode=0)

    a = MFAAdapter(cfg(), runner=runner, which=lambda n: f"/opt/bin/{n}")
    assert a.get_version() == "3.4.3"
    cmd, kw = seen[0]
    assert cmd == ["/opt/bin/mfa", "version"] and isinstance(cmd, list) and not kw.get("shell")


def test_executable_vanishing_between_check_and_run(corpus, tmp_path):
    root, d = corpus

    def runner(cmd, **kw):
        if cmd[1] == "version":
            return SimpleNamespace(stdout="3.4.3", stderr="", returncode=0)
        raise FileNotFoundError(cmd[0])

    a = MFAAdapter(cfg(), runner=runner, which=lambda n: "/x/mfa")
    with pytest.raises(MFANotInstalledError):
        a.align(root, tmp_path / "o", expected_samples=["S1"], dictionary=str(d))


def test_version_probe_timeout(corpus):
    def runner(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 120)

    with pytest.raises(MFACommandError, match="did not finish"):
        MFAAdapter(cfg(), runner=runner, which=lambda n: "/x/mfa").get_version()
