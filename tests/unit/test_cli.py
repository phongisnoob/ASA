"""CLI behaviour: arguments, outputs, overwrite safety and exit codes (see asa_m2.exceptions)."""

import json

import pytest

from asa_m2.cli import build_parser, main

from ..conftest import aligned_tiers  # noqa: TID252


def run(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


@pytest.fixture()
def ws(tmp_path, wav_factory):
    wav_factory(tmp_path / "raw/a.wav", seconds=3.0, sr=44100, channels=2)
    wav_factory(tmp_path / "raw/b.wav", seconds=3.0, sr=16000)
    (tmp_path / "m.csv").write_text("sample_id,audio_path,reference_text,speaker_id\nS1,raw/a.wav,I think this is important.,SPK_1\n"
                                    "S2,raw/b.wav,The cat sat,SPK_2\n", encoding="utf-8")
    return tmp_path


@pytest.fixture()
def mfa_config(ws):
    p = ws / "mfa.yaml"
    p.write_text("alignment:\n  mfa:\n    acoustic_model: english_us_arpa\n", encoding="utf-8")
    return str(p)


# ---- usage ---------------------------------------------------------------------------------
def test_no_command_and_bad_arguments_are_usage_errors(capsys):
    assert run(capsys)[0] == 2
    assert run(capsys, "nonsense")[0] == 2
    code, _, err = run(capsys, "g2p", "--bogus")
    assert code == 2 and "unrecognized arguments" in err
    assert run(capsys, "prepare-corpus", "--manifest", "x")[0] == 2  # --output missing


def test_help_and_version_exit_zero(capsys):
    assert main(["--help"]) == 0 and main(["--version"]) == 0
    out = capsys.readouterr().out
    assert "asa-m2 1.0.0" in out and "align-mfa" in out


def test_every_documented_command_exists():
    sub = next(a for a in build_parser()._actions if a.dest == "command")
    assert set(sub.choices) == {"check", "g2p", "prepare-corpus", "validate-corpus", "align-mfa", "align-qwen",
                                "inspect-textgrid", "export", "evaluate-boundaries"}


def test_invalid_log_level_env_is_reported(capsys, monkeypatch):
    monkeypatch.setenv("ASA_M2_LOG_LEVEL", "LOUD")
    code, _, err = run(capsys, "check")
    assert code == 2 and "invalid log level" in err


def test_bad_config_path_is_a_config_error(capsys):
    code, out, _ = run(capsys, "check", "--config", "/no/such.yaml")
    assert code == 3 and "configuration file not found" in out  # check reports the problem instead of crashing
    code2, out2, _ = run(capsys, "check", "--config", "/no/such.yaml", "--json")
    assert code2 == 3 and json.loads(out2)["ok"] is False
    assert run(capsys, "g2p", "--text", "hi", "--config", "/no/such.yaml")[0] == 3


# ---- check ------------------------------------------------------------------------------------
def test_check_json_reports_components(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))  # no mfa, no espeak
    code, out, _ = run(capsys, "check", "--json")
    data = json.loads(out)
    assert code == 0 and data["ok"]
    by = {c["name"]: c for c in data["checks"]}
    assert by["mfa"]["status"] == "missing (optional)" and by["cmudict"]["status"] == "ok"
    assert "CTC blank id 0" in by["phone inventory"]["detail"]


def test_check_require_flag_fails_when_optional_component_absent(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    code, out, _ = run(capsys, "check", "--require", "mfa")
    assert code == 5 and "required check(s) failed" in out
    assert run(capsys, "check", "--require", "qwen3-forcedaligner")[0] == 4


def test_check_with_fake_mfa_reports_version(capsys, fake_mfa):
    code, out, _ = run(capsys, "check", "--require", "mfa")
    assert code == 0 and "version 3.4.3 (verified)" in out


def test_check_unsupported_mfa_version(capsys, fake_mfa, monkeypatch):
    monkeypatch.setenv("FAKE_MFA_VERSION", "2.0.0")
    code, out, _ = run(capsys, "check", "--require", "mfa")
    assert code == 5 and "unsupported MFA version" in out


# ---- g2p ---------------------------------------------------------------------------------------
def test_g2p_text_table_output(capsys):
    code, out, _ = run(capsys, "g2p", "--text", "I think this is important.")
    assert code == 0 and "TH IH1 NG K" in out and "DH IH1 S | DH IH0 S" in out and "none selected" in out


def test_g2p_json_stdout(capsys):
    code, out, _ = run(capsys, "g2p", "--text", "I think", "--json")
    data = json.loads(out)
    assert code == 0 and data["kind"] == "asa_m2_g2p_batch" and data["count"] == 1
    assert data["records"][0]["tokens"][1]["variants"][0]["raw_arpabet"] == ["TH", "IH1", "NG", "K"]


def test_g2p_requires_exactly_one_input(capsys):
    assert run(capsys, "g2p")[0] == 3
    assert run(capsys, "g2p", "--text", "a", "--input", "b")[0] == 3
    assert run(capsys, "g2p", "--input", "/no/file.txt")[0] == 3


def test_g2p_input_file_and_output_overwrite_protection(capsys, tmp_path):
    inp = tmp_path / "in.txt"
    inp.write_text("# comment\nS1\tI think this is important.\n\nThe cat sat\n", encoding="utf-8")
    out = tmp_path / "phones.json"
    code, stdout, _ = run(capsys, "g2p", "--input", str(inp), "--output", str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0 and data["count"] == 2 and [r["sample_id"] for r in data["records"]] == ["S1", "line_0004"]
    assert "wrote 2 record(s)" in stdout
    code2, _, err = run(capsys, "g2p", "--input", str(inp), "--output", str(out))
    assert code2 == 4 and "already exists" in err and "--overwrite" in err
    assert run(capsys, "g2p", "--input", str(inp), "--output", str(out), "--overwrite")[0] == 0


def test_g2p_csv_input(capsys, tmp_path):
    inp = tmp_path / "in.csv"
    inp.write_text("sample_id,reference_text\nA,hello world\nB,the cat\n", encoding="utf-8")
    code, out, _ = run(capsys, "g2p", "--input", str(inp), "--json")
    assert code == 0 and [r["sample_id"] for r in json.loads(out)["records"]] == ["A", "B"]
    bad = tmp_path / "bad.csv"
    bad.write_text("id,foo\n1,x\n", encoding="utf-8")
    assert run(capsys, "g2p", "--input", str(bad))[0] == 3


def test_g2p_oov_exits_4_and_still_writes_the_result(capsys, tmp_path):
    out = tmp_path / "o.json"
    code, _, err = run(capsys, "g2p", "--text", "the blorptastic cat", "--output", str(out))
    rec = json.loads(out.read_text(encoding="utf-8"))["records"][0]
    assert code == 4 and "unresolved word" in err and "blocked from forced alignment" in err
    assert [t["resolution_status"] for t in rec["tokens"]] == ["resolved", "unresolved", "resolved"]


def test_g2p_oov_not_fatal_when_configured(capsys, tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("g2p:\n  fail_on_unresolved_oov: false\n", encoding="utf-8")
    assert run(capsys, "g2p", "--text", "blorptastic", "--config", str(cfg))[0] == 0


def test_g2p_custom_lexicon_resolves_oov(capsys, tmp_path):
    lex = tmp_path / "lex.dict"
    lex.write_text("blorptastic B L AO1 R P T AE1 S T IH0 K\n", encoding="utf-8")
    code, out, _ = run(capsys, "g2p", "--text", "blorptastic", "--lexicon", str(lex))
    assert code == 0 and "custom_lexicon" in out
    bad = tmp_path / "bad.dict"
    bad.write_text("word XX\n", encoding="utf-8")
    assert run(capsys, "g2p", "--text", "word", "--lexicon", str(bad))[0] == 4


def test_g2p_variant_policy_flag(capsys):
    code, out, _ = run(capsys, "g2p", "--text", "this", "--variant-policy", "first_listed")
    assert code == 0 and "policy_first_listed" in out


def test_g2p_human_verbatim_field_is_recorded(capsys):
    _, out, _ = run(capsys, "g2p", "--text", "uh well", "--text-field", "human_verbatim", "--json")
    assert json.loads(out)["records"][0]["text_field"] == "human_verbatim"


def test_g2p_fallback_requested_but_espeak_missing_is_exit_7(capsys, tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("g2p:\n  espeak:\n    executable: definitely-not-installed-espeak\n", encoding="utf-8")
    code, _, err = run(capsys, "g2p", "--text", "hi", "--fallback", "espeak", "--config", str(cfg))
    assert code == 7 and "not on PATH" in err


# ---- corpus --------------------------------------------------------------------------------------
def test_prepare_and_validate_corpus(capsys, ws):
    out = ws / "corpus"
    code, stdout, _ = run(capsys, "prepare-corpus", "--manifest", str(ws / "m.csv"), "--output", str(out))
    assert code == 0 and "prepared 2 sample(s)" in stdout and "16000 Hz" in stdout
    code, stdout, _ = run(capsys, "validate-corpus", "--corpus", str(out))
    assert code == 0 and "VALID" in stdout
    code, js, _ = run(capsys, "validate-corpus", "--corpus", str(out), "--json")
    assert code == 0 and json.loads(js)["ok"]


def test_prepare_corpus_protects_existing_output_and_originals(capsys, ws):
    out = ws / "corpus"
    run(capsys, "prepare-corpus", "--manifest", str(ws / "m.csv"), "--output", str(out))
    code, _, err = run(capsys, "prepare-corpus", "--manifest", str(ws / "m.csv"), "--output", str(out))
    assert code == 4 and "--overwrite" in err
    assert run(capsys, "prepare-corpus", "--manifest", str(ws / "m.csv"), "--output", str(out), "--overwrite")[0] == 0
    assert (ws / "raw/a.wav").is_file() and (ws / "raw/b.wav").is_file()


def test_prepare_corpus_errors(capsys, ws):
    assert run(capsys, "prepare-corpus", "--manifest", "/no/m.csv", "--output", str(ws / "o"))[0] == 3
    (ws / "bad.csv").write_text("sample_id,audio_path,reference_text\nS1,raw/missing.wav,hi\n", encoding="utf-8")
    code, _, err = run(capsys, "prepare-corpus", "--manifest", str(ws / "bad.csv"), "--output", str(ws / "o"))
    assert code == 4 and "audio_missing_file" in err and not (ws / "o").exists()


def test_prepare_corpus_target_sample_rate_override(capsys, ws):
    run(capsys, "prepare-corpus", "--manifest", str(ws / "m.csv"), "--output", str(ws / "c8"), "--target-sample-rate", "8000")
    import soundfile as sf
    assert sf.info(str(ws / "c8/mfa_corpus/SPK_2/S2.wav")).samplerate == 8000


def test_validate_corpus_invalid_exit_4_and_acoustic_phone_check(capsys, ws):
    out = ws / "corpus"
    run(capsys, "prepare-corpus", "--manifest", str(ws / "m.csv"), "--output", str(out))
    phones = ws / "phones.txt"
    phones.write_text("# no stress digits => incompatible with a stress-marked dictionary\n" + "\n".join(
        ["AA", "AE", "AH", "AO", "B", "D", "DH", "EH", "IH", "NG", "K", "S", "T", "Z", "TH", "M", "P", "R", "AY", "IY"]) + "\n", encoding="utf-8")
    code, stdout, _ = run(capsys, "validate-corpus", "--corpus", str(out), "--acoustic-phones", str(phones))
    assert code == 4 and "dictionary_phone_not_in_acoustic_model" in stdout and "INVALID" in stdout
    (out / "mfa_corpus/SPK_1/S1.lab").write_text("tampered\n", encoding="utf-8")
    assert run(capsys, "validate-corpus", "--corpus", str(out))[0] == 4


# ---- align-mfa (FAKE mfa) ----------------------------------------------------------------------------
def prepared(capsys, ws):
    run(capsys, "prepare-corpus", "--manifest", str(ws / "m.csv"), "--output", str(ws / "corpus"))


def test_align_mfa_end_to_end_with_fake(capsys, ws, fake_mfa, mfa_config):
    prepared(capsys, ws)
    code, out, _ = run(capsys, "align-mfa", "--corpus", str(ws / "corpus"), "--output", str(ws / "al"), "--config", mfa_config)
    assert code == 0 and "2/2 sample(s) aligned successfully" in out and "not that the learner pronounced" in out
    rec = json.loads((ws / "al/json/S1.json").read_text(encoding="utf-8"))
    assert rec["alignment_metadata"]["status"] == "success" and rec["phone_alignments"][1]["realized_phone"] is None


def test_align_mfa_flags_via_cli_arguments(capsys, ws, fake_mfa):
    from ..conftest import read_argv_log  # noqa: TID252
    prepared(capsys, ws)
    code, _, _ = run(capsys, "align-mfa", "--corpus", str(ws / "corpus"), "--output", str(ws / "al"),
                     "--acoustic-model", "my_model", "--num-jobs", "3", "--mfa-validate")
    assert code == 0
    align = next(c["argv"] for c in read_argv_log(fake_mfa) if c["argv"][0] == "align")
    assert align[3] == "my_model" and align[align.index("--num_jobs") + 1] == "3"


def test_align_mfa_failure_exit_codes(capsys, ws, fake_mfa, mfa_config, monkeypatch):
    prepared(capsys, ws)
    args = ("align-mfa", "--corpus", str(ws / "corpus"), "--config", mfa_config)
    monkeypatch.setenv("FAKE_MFA_MODE", "missing_last")
    code, out, err = run(capsys, *args, "--output", str(ws / "al1"))
    assert code == 6 and "did not align successfully" in err and "failed" in out
    monkeypatch.setenv("FAKE_MFA_MODE", "exit1")
    code, _, err = run(capsys, *args, "--output", str(ws / "al2"))
    assert code == 5 and "exited with code 1" in err
    monkeypatch.setenv("FAKE_MFA_MODE", "ok")
    monkeypatch.setenv("FAKE_MFA_VERSION", "2.2.1")
    code, _, err = run(capsys, *args, "--output", str(ws / "al3"))
    assert code == 5 and "unsupported MFA version" in err
    assert run(capsys, *args, "--output", str(ws / "al3"), "--allow-unverified-mfa")[0] == 0


def test_align_mfa_without_mfa_installed(capsys, ws, mfa_config, monkeypatch, tmp_path):
    prepared(capsys, ws)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    code, _, err = run(capsys, "align-mfa", "--corpus", str(ws / "corpus"), "--output", str(ws / "al"), "--config", mfa_config)
    assert code == 5 and "not found on PATH" in err


def test_align_mfa_requires_model_configuration(capsys, ws, fake_mfa):
    prepared(capsys, ws)
    code, _, err = run(capsys, "align-mfa", "--corpus", str(ws / "corpus"), "--output", str(ws / "al"))
    assert code == 3 and "acoustic_model is required" in err


def test_align_mfa_overwrite_protection(capsys, ws, fake_mfa, mfa_config):
    prepared(capsys, ws)
    args = ("align-mfa", "--corpus", str(ws / "corpus"), "--output", str(ws / "al"), "--config", mfa_config)
    assert run(capsys, *args)[0] == 0
    assert run(capsys, *args)[0] == 4
    assert run(capsys, *args, "--overwrite")[0] == 0


def test_align_mfa_on_unprepared_corpus(capsys, ws, fake_mfa, mfa_config):
    code, _, err = run(capsys, "align-mfa", "--corpus", str(ws / "nothing"), "--output", str(ws / "al"), "--config", mfa_config)
    assert code == 4 and "corpus_manifest.json not found" in err


# ---- TextGrid commands -----------------------------------------------------------------------------------
@pytest.fixture()
def tg(write_tg):
    tiers, dur = aligned_tiers([("i", ["AY1"]), ("think", ["TH", "IH1", "NG", "K"])])
    return write_tg("s.TextGrid", tiers, dur), tiers, dur


def test_inspect_textgrid_text_and_json(capsys, tg):
    path = tg[0]
    code, out, _ = run(capsys, "inspect-textgrid", "--input", str(path), "--max-intervals", "3")
    assert code == 0 and "'words' [IntervalTier]" in out and "'phones'" in out and "0 error(s)" in out and "more" in out
    code, js, _ = run(capsys, "inspect-textgrid", "--input", str(path), "--json")
    assert code == 0 and json.loads(js)["tiers"][1]["name"] == "phones"


def test_inspect_textgrid_invalid_and_malformed(capsys, write_tg, tmp_path):
    bad = write_tg("bad.TextGrid", {"w": [(0, 1.0, "a"), (0.5, 1.5, "b")]}, 1.5)
    code, out, _ = run(capsys, "inspect-textgrid", "--input", str(bad))
    assert code == 6 and "[ERROR] overlap" in out
    junk = tmp_path / "junk.TextGrid"
    junk.write_text("nope", encoding="utf-8")
    code, _, err = run(capsys, "inspect-textgrid", "--input", str(junk))
    assert code == 6 and "not a Praat TextGrid" in err
    assert run(capsys, "inspect-textgrid", "--input", "/missing.TextGrid")[0] == 3


def test_export_alignment_only_without_text(capsys, tg, tmp_path):
    out = tmp_path / "o.json"
    code, _, _ = run(capsys, "export", "--textgrid", str(tg[0]), "--output", str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0 and data["kind"] == "asa_m2_textgrid_export" and "alignment-only" in data["note"]
    assert data["tiers"][0]["intervals"][1]["label"] == "i"
    assert run(capsys, "export", "--textgrid", str(tg[0]), "--output", str(out))[0] == 4  # no silent overwrite
    assert run(capsys, "export", "--textgrid", str(tg[0]), "--output", str(out), "--overwrite")[0] == 0


def test_export_with_text_builds_the_full_record(capsys, tg, tmp_path):
    out = tmp_path / "rec.json"
    code, _, _ = run(capsys, "export", "--textgrid", str(tg[0]), "--text", "I think", "--sample-id", "S9", "--speaker-id", "SPK_9",
                     "--aligner-name", "MFA", "--aligner-version", "3.4.3", "--output", str(out))
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0 and rec["sample_id"] == "S9" and rec["speaker_id"] == "SPK_9" and rec["reference_text"] == "I think"
    assert rec["alignment_metadata"]["aligner"] == "MFA" and rec["alignment_metadata"]["status"] == "success"
    assert rec["canonical_phones"][0]["base_phone"] == "AY"


def test_export_with_g2p_json_and_exit_code_for_partial(capsys, tg, tmp_path, write_tg):
    out = tmp_path / "g.json"
    run(capsys, "g2p", "--text", "I think cat", "--json", "--output", str(out))
    record = json.loads(out.read_text(encoding="utf-8"))["records"][0]
    single = tmp_path / "single_g2p.json"
    single.write_text(json.dumps(record), encoding="utf-8")
    code, stdout, _ = run(capsys, "export", "--textgrid", str(tg[0]), "--g2p-json", str(single))  # 'cat' is missing in the TextGrid
    assert code == 6 and json.loads(stdout)["alignment_metadata"]["status"] == "partial"


def test_export_tier_options_and_audio_duration(capsys, write_tg, tmp_path, wav_factory):
    tiers, dur = aligned_tiers([("i", ["AY1"])])
    p = write_tg("c.TextGrid", {"Wort": tiers["words"], "Laut": tiers["phones"]}, dur)
    wav = wav_factory(tmp_path / "a.wav", seconds=dur + 3)
    code, stdout, _ = run(capsys, "export", "--textgrid", str(p), "--text", "I", "--word-tier", "Wort", "--phone-tier", "Laut", "--audio", str(wav))
    rec = json.loads(stdout)
    assert code == 0 and any(f["code"] == "textgrid_duration_mismatch" for f in rec["quality_flags"])
    assert run(capsys, "export", "--textgrid", str(p), "--text", "I")[0] == 6  # default tier names do not exist


def test_evaluate_boundaries_cli(capsys, write_tg, tmp_path):
    gold, dur = aligned_tiers([("i", ["AY1"]), ("think", ["TH", "IH1", "NG", "K"])])
    pred = {k: [(a + 0.01, b + 0.01, lab) if lab else (a, b, lab) for a, b, lab in v] for k, v in gold.items()}
    gdir, pdir = tmp_path / "gold", tmp_path / "pred"
    gdir.mkdir(), pdir.mkdir()
    (gdir / "s.TextGrid").write_text(write_tg("g.TextGrid", gold, dur).read_text(encoding="utf-8"), encoding="utf-8")
    (pdir / "s.TextGrid").write_text(write_tg("p.TextGrid", pred, dur).read_text(encoding="utf-8"), encoding="utf-8")
    (pdir / "extra.TextGrid").write_text((pdir / "s.TextGrid").read_text(encoding="utf-8"), encoding="utf-8")  # no gold counterpart
    code, out, _ = run(capsys, "evaluate-boundaries", "--pred", str(pdir), "--gold", str(gdir), "--tolerance", "0.02")
    data = json.loads(out)
    assert code == 0 and data["n_files"] == 2 and data["n_failed"] == 1 and data["failure_rate"] == 0.5
    assert data["failed_files"][0]["file"] == "extra.TextGrid" and "no gold" in data["failed_files"][0]["reason"]
    assert data["mae_s"] == pytest.approx(0.01, abs=1e-6) and data["pct_within_tolerance"] == 100.0
    assert "SUPPLIED gold" in data["note"]
    assert run(capsys, "evaluate-boundaries", "--pred", str(pdir), "--gold", str(gdir / "s.TextGrid"))[0] == 3


# ---- align-qwen (optional) ---------------------------------------------------------------------------------
def test_align_qwen_without_optional_dependencies_exits_7(capsys, tmp_path, wav_factory, monkeypatch):
    from asa_m2.alignment import qwen_adapter
    monkeypatch.setattr(qwen_adapter, "is_available", lambda: False)
    wav = wav_factory(tmp_path / "a.wav", seconds=1.0)
    code, _, err = run(capsys, "align-qwen", "--audio", str(wav), "--text", "I think")
    assert code == 7 and "pip install qwen-asr" in err


def test_align_qwen_with_injected_model_writes_record_and_textgrid(capsys, tmp_path, wav_factory, monkeypatch):
    from types import SimpleNamespace

    from asa_m2.alignment import qwen_adapter

    class Fake:
        def align(self, audio, text, language):
            return [[SimpleNamespace(text=w, start_time=0.1 * i, end_time=0.1 * i + 0.08) for i, w in enumerate(text.split())]]

    orig = qwen_adapter.QwenForcedAlignerAdapter.__init__

    def init(self, config, *, model_factory=None):
        orig(self, config, model_factory=lambda name: Fake())

    monkeypatch.setattr(qwen_adapter.QwenForcedAlignerAdapter, "__init__", init)
    wav = wav_factory(tmp_path / "a.wav", seconds=1.0)
    out, tgp = tmp_path / "q.json", tmp_path / "q.TextGrid"
    code, _, _ = run(capsys, "align-qwen", "--audio", str(wav), "--text", "I think", "--output", str(out), "--textgrid-output", str(tgp))
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0 and rec["alignment_metadata"]["granularity"] == "word" and rec["phone_alignments"] == []
    assert rec["alignment_metadata"]["aligner"] == "Qwen3-ForcedAligner" and any("not verified" in n for n in rec["alignment_metadata"]["notes"])
    assert tgp.is_file() and "words" in tgp.read_text(encoding="utf-8")


def test_log_file_and_json_logs(capsys, tmp_path):
    log = tmp_path / "run.log"
    code, _, err = run(capsys, "g2p", "--text", "hi", "--log-level", "DEBUG", "--log-file", str(log), "--json-logs")
    assert code == 0
    lines = [json.loads(ln) for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines and all("level" in ln for ln in lines)


def test_full_cli_workflow_with_relative_paths(capsys, ws, fake_mfa, mfa_config, monkeypatch):
    """The exact shape of the README workflow: relative --manifest/--corpus/--output arguments."""
    monkeypatch.chdir(ws)
    assert run(capsys, "prepare-corpus", "--manifest", "m.csv", "--output", "data/processed/corpus")[0] == 0
    assert run(capsys, "validate-corpus", "--corpus", "data/processed/corpus")[0] == 0
    code, out, _ = run(capsys, "align-mfa", "--corpus", "data/processed/corpus", "--output", "data/outputs/alignment", "--config", mfa_config)
    assert code == 0 and "2/2 sample(s) aligned successfully" in out
    assert (ws / "data/outputs/alignment/json/S1.json").is_file()
    assert run(capsys, "g2p", "--input", "nothing_here.txt")[0] == 3


@pytest.mark.parametrize("argv", [["check", "--json"], ["g2p", "--text", "hi", "--json"]])
def test_every_command_logs_its_operation_to_stderr_not_stdout(capsys, argv):
    code, out, err = run(capsys, *argv)
    assert code == 0 and "running command" in err and "command finished" in err and "exit_code=0" in err
    json.loads(out)  # stdout stays machine-readable: logs never pollute it
