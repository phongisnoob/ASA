"""End-to-end orchestration with the FAKE mfa: corpus -> alignment -> canonical JSON records."""

import json

import pytest

from asa_m2.alignment.mfa_adapter import MFAAdapter
from asa_m2.audio.corpus import prepare_corpus
from asa_m2.config import load_config
from asa_m2.exceptions import CorpusError, MFAVersionError, OutputExistsError
from asa_m2.g2p import G2PEngine
from asa_m2.phonemes import AlignerLabelMapper, PhoneInventory
from asa_m2.pipeline import align_corpus_with_mfa
from asa_m2.schemas import SampleRecord


@pytest.fixture()
def setup(tmp_path, wav_factory, fake_mfa):
    cfg = load_config()
    cfg.alignment.mfa.acoustic_model = "english_us_arpa"
    cfg.alignment.mfa.dictionary = "generated"
    inv = PhoneInventory.from_files(cfg.phonemes.inventory_path)
    mapper = AlignerLabelMapper.from_mapping_file(cfg.phonemes.mapping_path, inv)
    wav_factory(tmp_path / "raw/a.wav", seconds=3.0, sr=44100, channels=2)
    wav_factory(tmp_path / "raw/b.wav", seconds=3.0, sr=16000)
    wav_factory(tmp_path / "raw/c.wav", seconds=3.0, sr=16000)
    m = tmp_path / "m.csv"
    m.write_text("sample_id,audio_path,reference_text,speaker_id\nS1,raw/a.wav,I think this is important.,SPK_1\n"
                 "S2,raw/b.wav,The cat sat,SPK_2\nS3,raw/c.wav,Hello world,\n", encoding="utf-8")
    prepare_corpus(m, tmp_path / "corpus", cfg, G2PEngine.from_config(cfg, inv))
    return cfg, inv, mapper, tmp_path


def run(setup, **kw):
    cfg, inv, mapper, tmp = setup
    return align_corpus_with_mfa(tmp / "corpus", tmp / "align", cfg, inv, mapper, MFAAdapter(cfg.alignment.mfa), **kw)


def load(tmp, sid):
    return SampleRecord.model_validate_json((tmp / "align/json" / f"{sid}.json").read_text(encoding="utf-8"))


def test_full_run_produces_valid_records_for_every_sample(setup):
    summary = run(setup)
    tmp = setup[3]
    assert summary.n_success == 3 and not summary.has_failures
    assert sorted(p.name for p in (tmp / "align/json").glob("*.json")) == ["S1.json", "S2.json", "S3.json"]
    assert (tmp / "align/alignment_summary.json").is_file() and list((tmp / "align/logs").glob("mfa_align*.command.json"))
    assert all(p.is_file() for p in (tmp / "align/textgrids").rglob("*.TextGrid"))


def test_record_content_follows_the_data_contract(setup):
    run(setup)
    rec = load(setup[3], "S1")
    assert rec.schema_version == "1.0" and (rec.sample_id, rec.speaker_id) == ("S1", "SPK_1")
    assert rec.reference_text == "I think this is important." and rec.human_verbatim is None
    assert rec.alignment_metadata.aligner == "MFA" and rec.alignment_metadata.aligner_version == "3.4.3"
    assert rec.alignment_metadata.granularity == "phone" and rec.alignment_metadata.status == "success"
    assert rec.alignment_metadata.model["acoustic_model"] == "english_us_arpa" and rec.alignment_metadata.command[1] == "align"
    assert [t.normalized_token for t in rec.tokens] == ["i", "think", "this", "is", "important"]
    assert all(p.realized_phone is None for p in rec.phone_alignments)
    assert rec.provenance.inputs["audio"]["sample_rate"] == 16000 and rec.provenance.config_fingerprint
    # the fake aligner (like MFA) picks the FIRST dictionary variant; the exporter records that as the aligner's choice
    this = next(t for t in rec.tokens if t.normalized_token == "this")
    assert this.selected_variant_id == "cmudict:this:1" and this.selection_method == "alignment_match"


def test_unknown_speaker_record_carries_pseudo_id(setup):
    run(setup)
    assert load(setup[3], "S3").speaker_id.startswith("SPK_UNK_")


def test_summary_states_what_success_means(setup):
    run(setup)
    summary = json.loads((setup[3] / "align/alignment_summary.json").read_text(encoding="utf-8"))
    assert summary["n_success"] == 3 and "NOT mean the learner pronounced" in summary["note"]
    assert summary["samples"][0]["qc_summary"]["missing_words"] == 0


def test_missing_textgrid_becomes_an_explicit_failed_record(setup, monkeypatch):
    monkeypatch.setenv("FAKE_MFA_MODE", "missing_last")
    summary = run(setup)
    assert summary.has_failures and summary.n_success == 2
    failed = next(o for o in summary.outcomes if o.status == "failed")
    rec = load(setup[3], failed.sample_id)
    assert rec.alignment_metadata.status == "failed" and rec.alignment_metadata.granularity == "none"
    assert rec.word_alignments == [] and rec.phone_alignments == []  # no fabricated timestamps
    assert any(f.code == "no_textgrid_produced" and f.severity == "error" for f in rec.quality_flags)
    assert rec.tokens  # canonical G2P still present


def test_dropped_word_is_reported_as_partial(setup, monkeypatch):
    monkeypatch.setenv("FAKE_MFA_MODE", "drop_word")
    summary = run(setup)
    assert summary.has_failures
    rec = load(setup[3], "S2")
    assert rec.alignment_metadata.status == "partial"
    assert any(f.code == "missing_expected_word" and f.word_index == 1 for f in rec.quality_flags)


def test_oov_marker_output_is_counted(setup, monkeypatch):
    monkeypatch.setenv("FAKE_MFA_MODE", "oov_last")
    run(setup)
    rec = load(setup[3], "S2")
    assert rec.alignment_metadata.qc_summary["oov_count"] == 1 and rec.alignment_metadata.status == "success"


def test_corrupted_textgrid_becomes_failed_record_not_crash(setup, monkeypatch):
    monkeypatch.setenv("FAKE_MFA_MODE", "corrupt_last")
    summary = run(setup)
    assert summary.has_failures and summary.n_success == 2
    bad = next(o for o in summary.outcomes if o.status == "failed")
    rec = load(setup[3], bad.sample_id)
    assert rec.word_alignments == [] and rec.phone_alignments == []  # nothing fabricated from an unreadable file
    flag = next(f for f in rec.quality_flags if f.code == "textgrid_unusable")
    assert flag.severity == "error" and "not a Praat TextGrid" in flag.message
    assert next((setup[3] / "align/textgrids").rglob(f"{bad.sample_id}.TextGrid")).read_text(encoding="utf-8") == "this is not a TextGrid"  # original preserved


def test_unsupported_mfa_version_stops_the_pipeline(setup, monkeypatch):
    monkeypatch.setenv("FAKE_MFA_VERSION", "2.2.1")
    with pytest.raises(MFAVersionError):
        run(setup)
    assert not (setup[3] / "align").exists()  # a failed precondition leaves nothing behind
    monkeypatch.setenv("FAKE_MFA_VERSION", "3.4.3")
    assert run(setup).n_success == 3  # ...so the retry needs no --overwrite


def test_invalid_corpus_is_refused_before_mfa_runs(setup, fake_mfa):
    cfg, inv, mapper, tmp = setup
    (tmp / "corpus/mfa_corpus/SPK_1/S1.lab").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(CorpusError, match="transcript_mismatch"):
        run(setup)
    from ..conftest import read_argv_log  # noqa: TID252
    assert not fake_mfa.exists() or all(c["argv"][0] != "align" for c in read_argv_log(fake_mfa))


def test_existing_output_requires_overwrite(setup):
    run(setup)
    with pytest.raises(OutputExistsError):
        run(setup)
    assert run(setup, overwrite=True).n_success == 3


def test_optional_mfa_validate_step_runs(setup, fake_mfa):
    from ..conftest import read_argv_log  # noqa: TID252
    run(setup, run_mfa_validate=True)
    verbs = [c["argv"][0] for c in read_argv_log(fake_mfa)]
    assert "validate" in verbs and verbs.index("validate") < verbs.index("align")  # validated BEFORE aligning


def test_explicit_mfa_dictionary_name_is_used_verbatim(setup, fake_mfa):
    from ..conftest import read_argv_log  # noqa: TID252
    cfg, inv, mapper, tmp = setup
    cfg.alignment.mfa.dictionary = "english_us_arpa"
    summary = run(setup)
    align = next(c["argv"] for c in read_argv_log(fake_mfa) if c["argv"][0] == "align")
    assert align[2] == "english_us_arpa" and len(summary.outcomes) == 3  # named dictionary passed through untouched


def test_shared_outputs_contain_no_absolute_paths_but_the_local_audit_log_keeps_the_exact_command(setup):
    cfg, inv, mapper, tmp = setup
    cfg.alignment.mfa.temporary_directory = str(tmp / "mfa_tmp")
    run(setup)
    for f in list((tmp / "align/json").glob("*.json")) + [tmp / "align/alignment_summary.json"]:
        assert str(tmp) not in f.read_text(encoding="utf-8"), f"absolute path leaked into {f.name}"
    rec = load(tmp, "S1")
    cmd = rec.alignment_metadata.command
    assert cmd[0] == "mfa" and cmd[1:3] == ["align", "<corpus>"] and cmd[3] == "<dictionary>" and cmd[5] == "<output>"
    assert cmd[cmd.index("--temporary_directory") + 1] == "<abs>/mfa_tmp"
    audit = json.loads(next((tmp / "align/logs").glob("mfa_align*.command.json")).read_text(encoding="utf-8"))
    assert str(tmp) in " ".join(audit["command"])  # the exact command is preserved where it belongs


def test_public_command_helper():
    from asa_m2.pipeline import public_command
    out = public_command(["/opt/conda/envs/x/bin/mfa", "align", "/home/u/c", "/home/u/d.dict", "model", "relative", "--n", "3", "/home/u/extra"],
                         {"/home/u/c": "<corpus>", "/home/u/d.dict": "<dictionary>"})
    assert out == ["mfa", "align", "<corpus>", "<dictionary>", "model", "relative", "--n", "3", "<abs>/extra"]
