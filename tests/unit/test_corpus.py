"""M2.4 corpus preparation + validation: integrity, speaker isolation, safe paths."""

import hashlib
import json
from pathlib import Path

import pytest
import soundfile as sf
from pydantic import ValidationError

from asa_m2.audio.corpus import (
    MANIFEST_NAME,
    load_corpus_manifest,
    prepare_corpus,
    pseudo_speaker_id,
    read_manifest,
    validate_corpus,
)
from asa_m2.config import load_config
from asa_m2.exceptions import CorpusError, OutputExistsError, UnsafePathError
from asa_m2.g2p import G2PEngine


@pytest.fixture()
def raw(tmp_path, wav_factory):
    d = tmp_path / "raw"
    wav_factory(d / "a.wav", seconds=1.0, sr=44100, channels=2, freq=300)
    wav_factory(d / "b.wav", seconds=1.0, sr=16000, freq=500)
    wav_factory(d / "c.wav", seconds=1.0, sr=22050, freq=700)
    return d


def write_manifest(tmp_path: Path, rows: list[str], header: str = "sample_id,audio_path,reference_text,speaker_id") -> Path:
    p = tmp_path / "manifest.csv"
    p.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return p


def run(tmp_path, manifest, config=None, engine=None, **kw):
    config = config or load_config()
    engine = engine or G2PEngine.from_config(config)
    return prepare_corpus(manifest, tmp_path / "out", config, engine, **kw)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---- the happy path ----------------------------------------------------------------
def test_matching_wav_and_transcript_are_produced(tmp_path, raw):
    m = write_manifest(tmp_path, ["S001,raw/a.wav,I think this is important.,SPK_001", "S002,raw/b.wav,Hello world,SPK_002"])
    man = run(tmp_path, m)
    out = tmp_path / "out"
    assert [s.sample_id for s in man.samples] == ["S001", "S002"] and man.excluded == []
    wav, lab = out / "mfa_corpus/SPK_001/S001.wav", out / "mfa_corpus/SPK_001/S001.lab"
    assert wav.is_file() and lab.read_text(encoding="utf-8") == "i think this is important\n"
    info = sf.info(str(wav))
    assert (info.samplerate, info.channels, info.subtype) == (16000, 1, "PCM_16")
    assert (out / "g2p/S001.g2p.json").is_file() and (out / "dictionary/asa_m2_corpus.dict").is_file()


def test_manifest_records_all_transformations_and_hashes(tmp_path, raw):
    man = run(tmp_path, write_manifest(tmp_path, ["S001,raw/a.wav,hello,SPK_001"]))
    a = man.samples[0].audio
    assert (a.source_sample_rate, a.source_channels, a.output_sample_rate, a.output_samples) == (44100, 2, 16000, 16000)
    assert a.source_sha256 == sha(raw / "a.wav") and a.output_sha256 == sha(tmp_path / "out" / a.output_wav)
    assert [t["op"] for t in a.transformations] == ["to_mono", "resample"]
    assert a.source_relpath == "raw/a.wav" and man.inventory["id"] == "asa-arpabet-cmu39"
    assert load_corpus_manifest(tmp_path / "out").samples[0].sample_id == "S001"  # round-trips


def test_original_recordings_are_never_modified(tmp_path, raw):
    before = {p.name: sha(p) for p in raw.iterdir()}
    mtimes = {p.name: p.stat().st_mtime_ns for p in raw.iterdir()}
    run(tmp_path, write_manifest(tmp_path, ["S1,raw/a.wav,hello,A", "S2,raw/b.wav,world,B"]))
    assert {p.name: sha(p) for p in raw.iterdir()} == before
    assert {p.name: p.stat().st_mtime_ns for p in raw.iterdir()} == mtimes
    assert {p.name for p in raw.iterdir()} == {"a.wav", "b.wav", "c.wav"}  # no extra files next to originals


def test_deterministic_when_timestamps_disabled(tmp_path, raw):
    cfg = load_config()
    cfg.output.include_timestamps = False
    m = write_manifest(tmp_path, ["S1,raw/a.wav,hello world,A"])
    r1 = prepare_corpus(m, tmp_path / "o1", cfg, G2PEngine.from_config(cfg))
    r2 = prepare_corpus(m, tmp_path / "o2", cfg, G2PEngine.from_config(cfg))
    assert r1.model_dump() == r2.model_dump()
    assert (tmp_path / "o1/mfa_corpus/A/S1.wav").read_bytes() == (tmp_path / "o2/mfa_corpus/A/S1.wav").read_bytes()


def test_dictionary_contains_every_variant_and_only_needed_words(tmp_path, raw):
    run(tmp_path, write_manifest(tmp_path, ["S1,raw/a.wav,this is the cat,A"]))
    lines = (tmp_path / "out/dictionary/asa_m2_corpus.dict").read_text(encoding="utf-8").splitlines()
    assert "this\tDH IH1 S" in lines and "this\tDH IH0 S" in lines  # all variants exported
    assert sum(ln.startswith("the\t") for ln in lines) == 3
    assert not any(ln.startswith("dog\t") for ln in lines)
    assert lines == sorted(lines, key=lambda s: s.split("\t")[0])  # deterministic order


def test_dictionary_without_stress_when_configured(tmp_path, raw):
    cfg = load_config()
    cfg.phonemes.preserve_stress = False
    prepare_corpus(write_manifest(tmp_path, ["S1,raw/a.wav,this,A"]), tmp_path / "o", cfg, G2PEngine.from_config(cfg))
    lines = (tmp_path / "o/dictionary/asa_m2_corpus.dict").read_text(encoding="utf-8").splitlines()
    assert lines == ["this\tDH IH S"]  # the two variants collapse; never duplicated


def test_single_variant_dictionary_requires_policy_and_exports_only_selected(tmp_path, raw):
    cfg = load_config()
    cfg.g2p.variant_policy = "first_listed"
    cfg.g2p.allow_multiple_variants = False
    prepare_corpus(write_manifest(tmp_path, ["S1,raw/a.wav,this,A"]), tmp_path / "o", cfg, G2PEngine.from_config(cfg))
    assert (tmp_path / "o/dictionary/asa_m2_corpus.dict").read_text(encoding="utf-8") == "this\tDH IH1 S\n"


# ---- missing / invalid inputs ----------------------------------------------------------
def test_missing_wav_is_reported_and_nothing_is_written(tmp_path, raw):
    m = write_manifest(tmp_path, ["S1,raw/a.wav,hello,A", "S2,raw/gone.wav,world,B"])
    with pytest.raises(CorpusError, match=r"audio_missing_file") as exc:
        run(tmp_path, m)
    assert "line 3" in str(exc.value) and not (tmp_path / "out").exists()


def test_missing_transcript_is_reported(tmp_path, raw):
    m = write_manifest(tmp_path, ["S1,raw/a.wav,,A"])
    with pytest.raises(CorpusError, match="missing_text"):
        run(tmp_path, m)


def test_all_problems_are_listed_at_once(tmp_path, raw):
    m = write_manifest(tmp_path, ["S1,raw/gone.wav,hello,A", "S2,raw/b.wav,,B", "S3,raw/c.wav,fine,bad/spk"])
    with pytest.raises(CorpusError) as exc:
        run(tmp_path, m)
    assert len(exc.value.details["problems"]) == 3


def test_on_error_skip_excludes_bad_rows_and_records_them(tmp_path, raw):
    cfg = load_config()
    cfg.corpus.on_error = "skip"
    m = write_manifest(tmp_path, ["S1,raw/a.wav,hello,A", "S2,raw/gone.wav,world,B", "S3,raw/c.wav,,C"])
    man = prepare_corpus(m, tmp_path / "o", cfg, G2PEngine.from_config(cfg))
    assert [s.sample_id for s in man.samples] == ["S1"]
    assert {(e.sample_id, e.codes[0]) for e in man.excluded} == {("S2", "audio_missing_file"), ("S3", "missing_text")}


def test_no_usable_samples_is_an_error(tmp_path, raw):
    cfg = load_config()
    cfg.corpus.on_error = "skip"
    with pytest.raises(CorpusError, match="no usable samples"):
        prepare_corpus(write_manifest(tmp_path, ["S1,raw/gone.wav,hi,A"]), tmp_path / "o", cfg, G2PEngine.from_config(cfg))


def test_duplicate_sample_id_makes_all_occurrences_ambiguous(tmp_path, raw):
    m = write_manifest(tmp_path, ["S1,raw/a.wav,hello,A", "S1,raw/b.wav,world,B", "S2,raw/c.wav,ok,C"])
    rows, issues, _ = read_manifest(m, load_config())
    assert [r.sample_id for r in rows] == ["S2"]
    dups = [i for i in issues if i.code == "duplicate_sample_id"]
    assert sorted(i.line for i in dups) == [2, 3]  # BOTH occurrences are flagged: neither can be trusted
    with pytest.raises(CorpusError, match="duplicate_sample_id"):
        run(tmp_path, m)


@pytest.mark.parametrize("bad_id", ["../evil", "a/b", "a\\b", ".hidden", "with space", "x..y", "-lead", "é", "a" * 200])
def test_unsafe_sample_ids_rejected(tmp_path, raw, bad_id):
    m = write_manifest(tmp_path, [f'"{bad_id}",raw/a.wav,hello,A'])
    with pytest.raises(CorpusError, match="unsafe_sample_id"):
        run(tmp_path, m)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("bad_spk", ["../x", "a/b", "sp ace", ".x"])
def test_unsafe_speaker_ids_rejected(tmp_path, raw, bad_spk):
    m = write_manifest(tmp_path, [f'S1,raw/a.wav,hello,"{bad_spk}"'])
    with pytest.raises(CorpusError, match="unsafe_speaker_id"):
        run(tmp_path, m)


def test_pii_columns_are_rejected_not_copied(tmp_path, raw):
    m = write_manifest(tmp_path, ["S1,raw/a.wav,hello,A,Nguyen Van A"], header="sample_id,audio_path,reference_text,speaker_id,student_name")
    with pytest.raises(CorpusError, match="personally-identifying"):
        run(tmp_path, m)
    assert not (tmp_path / "out").exists()


def test_manifest_file_level_errors(tmp_path):
    cfg = load_config()
    with pytest.raises(CorpusError, match="not found"):
        read_manifest(tmp_path / "none.csv", cfg)
    p = tmp_path / "e.csv"
    p.write_text("", encoding="utf-8")
    with pytest.raises(CorpusError, match="empty"):
        read_manifest(p, cfg)
    p.write_text("sample_id,reference_text\nS1,hi\n", encoding="utf-8")
    with pytest.raises(CorpusError, match="required column"):
        read_manifest(p, cfg)
    p.write_text("sample_id,audio_path\nS1,a.wav\n", encoding="utf-8")
    with pytest.raises(CorpusError, match="text column"):
        read_manifest(p, cfg)
    p.write_text("sample_id,audio_path,reference_text\n", encoding="utf-8")
    with pytest.raises(CorpusError, match="no data rows"):
        read_manifest(p, cfg)


def test_extra_columns_warn_and_are_ignored(tmp_path, raw):
    m = write_manifest(tmp_path, ["S1,raw/a.wav,hello,A,whatever"], header="sample_id,audio_path,reference_text,speaker_id,comment")
    man = run(tmp_path, m)
    assert any("comment" in w for w in man.warnings)
    assert "whatever" not in (tmp_path / "out" / MANIFEST_NAME).read_text(encoding="utf-8")


def test_bom_and_whitespace_in_manifest_tolerated(tmp_path, raw):
    p = tmp_path / "m.csv"
    p.write_text("\ufeffsample_id , audio_path,reference_text\n S1 ,raw/a.wav, hello world \n", encoding="utf-8")
    man = run(tmp_path, p)
    assert man.samples[0].normalized_text == "hello world"


# ---- speakers ----------------------------------------------------------------------
def test_missing_speaker_id_gets_unique_pseudo_id_per_recording(tmp_path, raw):
    m = write_manifest(tmp_path, ["S1,raw/a.wav,hello,", "S2,raw/b.wav,world,", "S3,raw/c.wav,again,REAL"])
    man = run(tmp_path, m)
    s1, s2, s3 = man.samples
    assert s1.speaker_id != s2.speaker_id  # unrelated recordings are never grouped together
    assert s1.speaker_id.startswith("SPK_UNK_") and s1.speaker_id_source == "pseudo_unique_per_recording"
    assert s1.speaker_folder != s2.speaker_folder and s3.speaker_id_source == "manifest"
    assert s1.speaker_id == pseudo_speaker_id("S1")  # deterministic, so reruns are reproducible
    assert man.speaker_policy["pseudo_speakers"] == 2


def test_pseudo_ids_contain_no_sample_id_text(tmp_path):
    pid = pseudo_speaker_id("student_nguyen_001")
    assert "nguyen" not in pid and pid.startswith("SPK_UNK_") and len(pid) == len("SPK_UNK_") + 10


def test_speaker_isolation_recordings_of_one_speaker_share_a_folder_others_do_not(tmp_path, raw):
    m = write_manifest(tmp_path, ["S1,raw/a.wav,one,SPK_A", "S2,raw/b.wav,two,SPK_A", "S3,raw/c.wav,three,SPK_B"])
    man = run(tmp_path, m)
    folders = {}
    for s in man.samples:
        folders.setdefault(s.speaker_folder, set()).add(s.speaker_id)
    assert folders == {"SPK_A": {"SPK_A"}, "SPK_B": {"SPK_B"}}
    assert sorted(p.name for p in (tmp_path / "out/mfa_corpus/SPK_A").glob("*.wav")) == ["S1.wav", "S2.wav"]


def test_per_recording_grouping_puts_each_recording_in_its_own_folder(tmp_path, raw):
    cfg = load_config()
    cfg.speaker.grouping = "per_recording"
    m = write_manifest(tmp_path, ["S1,raw/a.wav,one,SPK_A", "S2,raw/b.wav,two,SPK_A"])
    man = prepare_corpus(m, tmp_path / "o", cfg, G2PEngine.from_config(cfg))
    assert [s.speaker_folder for s in man.samples] == ["S1", "S2"] and {s.speaker_id for s in man.samples} == {"SPK_A"}
    assert validate_corpus(tmp_path / "o", cfg, G2PEngine.from_config(cfg).inventory).ok


def test_missing_speaker_policy_error(tmp_path, raw):
    cfg = load_config()
    cfg.speaker.missing_id_policy = "error"
    with pytest.raises(CorpusError, match="missing_speaker_id"):
        prepare_corpus(write_manifest(tmp_path, ["S1,raw/a.wav,hello,"]), tmp_path / "o", cfg, G2PEngine.from_config(cfg))


def test_shared_speaker_policy_is_explicit_and_warns(tmp_path, raw):
    cfg = load_config()
    with pytest.raises(ValidationError, match="shared_speaker_id"):  # the policy cannot be enabled without an explicit ID
        cfg.speaker.missing_id_policy = "shared"
    cfg.speaker.shared_speaker_id = "SPK_ALL"
    cfg.speaker.missing_id_policy = "shared"
    man = prepare_corpus(write_manifest(tmp_path, ["S1,raw/a.wav,one,", "S2,raw/b.wav,two,"]), tmp_path / "o", cfg, G2PEngine.from_config(cfg))
    assert {s.speaker_id for s in man.samples} == {"SPK_ALL"} and {s.speaker_id_source for s in man.samples} == {"shared_policy"}
    assert any("share one speaker folder" in w for w in man.warnings)


# ---- text selection / OOV --------------------------------------------------------------
def test_reference_text_is_preferred_and_verbatim_is_kept_separate(tmp_path, raw):
    m = write_manifest(tmp_path, ["S1,raw/a.wav,the cat sat,A,the cat sit"], header="sample_id,audio_path,reference_text,speaker_id,human_verbatim")
    man = run(tmp_path, m)
    assert man.samples[0].g2p_input_field == "reference_text" and man.samples[0].normalized_text == "the cat sat"
    man2 = run(tmp_path, m, text_field="human_verbatim", overwrite=True)
    assert man2.samples[0].g2p_input_field == "human_verbatim" and man2.samples[0].normalized_text == "the cat sit"


def test_spontaneous_speech_uses_verbatim_only_when_no_reference(tmp_path, raw):
    m = write_manifest(tmp_path, ["S1,raw/a.wav,,A,uh well I went"], header="sample_id,audio_path,reference_text,speaker_id,human_verbatim")
    man = run(tmp_path, m)
    assert man.samples[0].g2p_input_field == "human_verbatim"
    assert json.loads((tmp_path / "out/g2p/S1.g2p.json").read_text(encoding="utf-8"))["text_field"] == "human_verbatim"


def test_reference_path_reads_text_file(tmp_path, raw):
    (tmp_path / "ref.txt").write_text("I think this is important.\n", encoding="utf-8")
    m = write_manifest(tmp_path, ["S1,raw/a.wav,ref.txt,A"], header="sample_id,audio_path,reference_path,speaker_id")
    assert run(tmp_path, m).samples[0].normalized_text == "i think this is important"
    m2 = write_manifest(tmp_path, ["S1,raw/a.wav,missing.txt,A"], header="sample_id,audio_path,reference_path,speaker_id")
    with pytest.raises(CorpusError, match="cannot read reference_path"):
        run(tmp_path, m2, overwrite=True)


def test_unresolved_oov_blocks_corpus_by_default(tmp_path, raw):
    m = write_manifest(tmp_path, ["S1,raw/a.wav,the blorptastic cat,A"])
    with pytest.raises(CorpusError, match="blorptastic") as exc:
        run(tmp_path, m)
    assert exc.value.details["words"] == ["blorptastic"] and not (tmp_path / "out").exists()


def test_unresolved_oov_excluded_when_not_fatal(tmp_path, raw):
    cfg = load_config()
    cfg.g2p.fail_on_unresolved_oov = False
    m = write_manifest(tmp_path, ["S1,raw/a.wav,the blorptastic cat,A", "S2,raw/b.wav,the cat,B"])
    man = prepare_corpus(m, tmp_path / "o", cfg, G2PEngine.from_config(cfg))
    assert [s.sample_id for s in man.samples] == ["S2"]
    assert [e.sample_id for e in man.excluded] == ["S1"] and man.excluded[0].codes == ["unresolved_words"]
    assert "blorptastic" in man.excluded[0].messages[0]
    assert not (tmp_path / "o/mfa_corpus/A").exists()  # excluded sample never enters the aligner corpus


def test_custom_lexicon_lets_an_oov_sample_through(tmp_path, raw, inventory):
    from asa_m2.g2p import CustomLexicon
    cfg = load_config()
    lex = CustomLexicon.from_text("blorptastic B L AO1 R P T AE1 S T IH0 K\n", inventory)
    eng = G2PEngine(cfg, inventory, lexicon=lex)
    man = prepare_corpus(write_manifest(tmp_path, ["S1,raw/a.wav,a blorptastic cat,A"]), tmp_path / "o", cfg, eng)
    assert "blorptastic\tB L AO1 R P T AE1 S T IH0 K" in (tmp_path / "o/dictionary/asa_m2_corpus.dict").read_text(encoding="utf-8")
    assert man.samples[0].num_tokens == 3


# ---- overwrite / path safety -----------------------------------------------------------
def test_overwrite_protection_and_explicit_overwrite(tmp_path, raw):
    m = write_manifest(tmp_path, ["S1,raw/a.wav,hello,A"])
    run(tmp_path, m)
    with pytest.raises(OutputExistsError):
        run(tmp_path, m)
    (tmp_path / "out" / "unrelated.txt").write_text("keep me", encoding="utf-8")
    run(tmp_path, m, overwrite=True)
    assert (tmp_path / "out" / "unrelated.txt").read_text(encoding="utf-8") == "keep me"  # only M2-owned paths are replaced


def test_output_inside_source_tree_cannot_delete_originals(tmp_path, wav_factory):
    src = tmp_path / "out" / "mfa_corpus" / "A"
    wav_factory(src / "orig.wav")
    m = write_manifest(tmp_path, ["S1,out/mfa_corpus/A/orig.wav,hello,A"])
    (tmp_path / "out" / "g2p").mkdir()  # makes the output look like an existing corpus
    with pytest.raises(UnsafePathError, match="would be deleted"):
        run(tmp_path, m, overwrite=True)
    assert (src / "orig.wav").is_file()


def test_output_never_overwrites_a_source_file(tmp_path, wav_factory):
    """Even with colliding names, outputs live under mfa_corpus/<speaker>/ and never replace the original."""
    orig = wav_factory(tmp_path / "S1.wav", seconds=1.0, sr=44100)
    before = sha(orig)
    m = write_manifest(tmp_path, ["S1,S1.wav,hello,A"])
    prepare_corpus(m, tmp_path, load_config(), G2PEngine.from_config(load_config()))  # output dir == source dir!
    assert sha(orig) == before and (tmp_path / "mfa_corpus/A/S1.wav").is_file()


def test_invalid_audio_row_is_reported_with_code(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    with pytest.raises(CorpusError, match="audio_unreadable"):
        run(tmp_path, write_manifest(tmp_path, ["S1,bad.wav,hello,A"]))


# ---- validate_corpus ---------------------------------------------------------------
@pytest.fixture()
def prepared(tmp_path, raw):
    cfg = load_config()
    eng = G2PEngine.from_config(cfg)
    prepare_corpus(write_manifest(tmp_path, ["S1,raw/a.wav,this is the cat,A", "S2,raw/b.wav,hello world,B"]), tmp_path / "c", cfg, eng)
    return tmp_path / "c", cfg, eng.inventory


def codes(report, severity=None):
    return sorted(i.code for i in report.issues if severity is None or i.severity == severity)


def test_validate_clean_corpus(prepared):
    root, cfg, inv = prepared
    rep = validate_corpus(root, cfg, inv)
    assert rep.ok and rep.n_samples == 2 and codes(rep, "error") == []


def test_validate_detects_missing_files(prepared):
    root, cfg, inv = prepared
    (root / "mfa_corpus/A/S1.wav").unlink()
    (root / "mfa_corpus/B/S2.lab").unlink()
    c = codes(validate_corpus(root, cfg, inv), "error")
    assert "missing_audio" in c and "missing_transcript" in c


def test_validate_detects_modified_audio_and_transcript(prepared):
    root, cfg, inv = prepared
    import numpy as np
    sf.write(str(root / "mfa_corpus/A/S1.wav"), np.ones(16000) * 0.1, 16000)
    (root / "mfa_corpus/B/S2.lab").write_text("something else\n", encoding="utf-8")
    c = codes(validate_corpus(root, cfg, inv), "error")
    assert "audio_modified" in c and "transcript_mismatch" in c


def test_validate_detects_wrong_sample_rate(prepared):
    root, cfg, inv = prepared
    import numpy as np
    sf.write(str(root / "mfa_corpus/B/S2.wav"), np.ones(16000) * 0.1, 8000)
    assert "sample_rate_mismatch" in codes(validate_corpus(root, cfg, inv), "error")


def test_validate_detects_stereo(prepared):
    root, cfg, inv = prepared
    import numpy as np
    sf.write(str(root / "mfa_corpus/B/S2.wav"), np.ones((16000, 2)) * 0.1, 16000)
    assert "not_mono" in codes(validate_corpus(root, cfg, inv), "error")


def test_validate_detects_orphans_and_speaker_mixing(prepared):
    root, cfg, inv = prepared
    (root / "mfa_corpus/A/stray.wav").write_bytes(b"x")
    man = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    man["samples"][1]["speaker_folder"] = "A"
    (root / MANIFEST_NAME).write_text(json.dumps(man), encoding="utf-8")
    c = codes(validate_corpus(root, cfg, inv))
    assert "orphan_file" in c and "speaker_mixing" in c


def test_validate_dictionary_problems(prepared):
    root, cfg, inv = prepared
    d = root / "dictionary/asa_m2_corpus.dict"
    d.write_text("cat\tK AE1 T\nbroken line without tab\nbad\tXX Y\n", encoding="utf-8")
    rep = validate_corpus(root, cfg, inv)
    c = codes(rep, "error")
    assert {"dictionary_malformed", "dictionary_phone_not_in_inventory", "word_missing_from_dictionary"} <= set(c)
    assert "dictionary_modified" in codes(rep, "warning")
    d.unlink()
    assert "missing_dictionary" in codes(validate_corpus(root, cfg, inv), "error")


def test_validate_acoustic_phone_set_compatibility(prepared):
    root, cfg, inv = prepared
    full = [f"{p}{d}" if inv.is_vowel(p) else p for p in inv.base_symbols for d in ("0", "1", "2")] + ["spn", "sil"]
    rep = validate_corpus(root, cfg, inv, acoustic_phones=full)
    assert rep.ok and "acoustic_model_extra_phones" in codes(rep, "info")
    no_stress = list(inv.base_symbols)  # a model without stress digits cannot use a stress-marked dictionary
    bad = validate_corpus(root, cfg, inv, acoustic_phones=no_stress)
    assert "dictionary_phone_not_in_acoustic_model" in codes(bad, "error")


def test_validate_inventory_and_schema_mismatch_and_bad_manifest(prepared, tmp_path):
    root, cfg, inv = prepared
    man = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    man["inventory"]["version"] = "9.9.9"
    man["schema_version"] = "0.1"
    (root / MANIFEST_NAME).write_text(json.dumps(man), encoding="utf-8")
    assert {"inventory_mismatch", "schema_version_mismatch"} <= set(codes(validate_corpus(root, cfg, inv), "error"))
    (root / MANIFEST_NAME).write_text("{broken", encoding="utf-8")
    assert codes(validate_corpus(root, cfg, inv), "error") == ["manifest_invalid"]
    assert codes(validate_corpus(tmp_path / "nowhere", cfg, inv), "error") == ["manifest_invalid"]


def test_relative_paths_work_as_in_the_readme(tmp_path, raw, monkeypatch):
    """Regression: a relative --output used to crash with an internal ValueError (relative_to on resolved paths)."""
    monkeypatch.chdir(tmp_path)
    write_manifest(tmp_path, ["S1,raw/a.wav,hello world,A"])
    cfg = load_config()
    eng = G2PEngine.from_config(cfg)
    man = prepare_corpus("manifest.csv", "data/processed/corpus", cfg, eng)
    assert man.samples[0].audio.output_wav == "mfa_corpus/A/S1.wav" and man.samples[0].audio.source_relpath == "raw/a.wav"
    assert (tmp_path / "data/processed/corpus/mfa_corpus/A/S1.wav").is_file()
    rep = validate_corpus("data/processed/corpus", cfg, eng.inventory)
    assert rep.ok and rep.n_samples == 1
    with pytest.raises(OutputExistsError):
        prepare_corpus("manifest.csv", "data/processed/corpus", cfg, eng)
    prepare_corpus("manifest.csv", "./data/processed/../processed/corpus", cfg, eng, overwrite=True)  # non-normalised path too


def test_corpus_artifacts_contain_no_absolute_paths(tmp_path, raw):
    run(tmp_path, write_manifest(tmp_path, ["S1,raw/a.wav,hello world,A", "S2,raw/b.wav,the cat,"]))
    for f in (tmp_path / "out").rglob("*.json"):
        assert str(tmp_path) not in f.read_text(encoding="utf-8"), f"absolute path leaked into {f.name}"
    assert str(tmp_path) not in (tmp_path / "out/dictionary/asa_m2_corpus.dict").read_text(encoding="utf-8")
