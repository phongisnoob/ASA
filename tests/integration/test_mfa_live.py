"""Integration: REAL Montreal Forced Aligner. Skipped (not passed) unless MFA is functional AND a model is named.

Requirements to run:
  * `mfa version` works in the active environment (conda install; see README),
  * the acoustic model is already installed:  export ASA_M2_MFA_ACOUSTIC_MODEL=english_us_arpa
  * espeak-ng (used only to synthesise a test utterance; no private audio is needed).
This test has NOT been run in the build environment (MFA could not be installed there).
"""

import os
import shutil
import subprocess

import pytest

from asa_m2.alignment.mfa_adapter import MFAAdapter
from asa_m2.audio.corpus import prepare_corpus
from asa_m2.config import load_config
from asa_m2.exceptions import M2Error
from asa_m2.g2p import G2PEngine
from asa_m2.phonemes import AlignerLabelMapper, PhoneInventory
from asa_m2.pipeline import align_corpus_with_mfa

pytestmark = pytest.mark.integration


def _skip_reason() -> str | None:
    if shutil.which("mfa") is None:
        return "mfa is not on PATH (install via conda; see README)"
    if not os.environ.get("ASA_M2_MFA_ACOUSTIC_MODEL"):
        return "set ASA_M2_MFA_ACOUSTIC_MODEL to an installed acoustic model (e.g. english_us_arpa)"
    if shutil.which("espeak-ng") is None:
        return "espeak-ng needed to synthesise the test utterance"
    try:
        MFAAdapter(load_config().alignment.mfa).check_version()
    except M2Error as exc:
        return f"MFA unusable or unsupported: {exc.message[:160]}"
    return None


REASON = _skip_reason()
if REASON:
    pytest.skip(f"live MFA test NOT run: {REASON}", allow_module_level=True)


def test_real_mfa_alignment_of_synthesised_speech(tmp_path):
    cfg = load_config()
    cfg.alignment.mfa.acoustic_model = os.environ["ASA_M2_MFA_ACOUSTIC_MODEL"]
    cfg.alignment.mfa.num_jobs = 1
    wav = tmp_path / "raw" / "utt.wav"
    wav.parent.mkdir()
    text = "I think this is important"
    subprocess.run(["espeak-ng", "-v", "en-us", "-w", str(wav), text], check=True, capture_output=True)
    manifest = tmp_path / "m.csv"
    manifest.write_text(f"sample_id,audio_path,reference_text,speaker_id\nU1,raw/utt.wav,{text},SPK_TTS\n", encoding="utf-8")
    inv = PhoneInventory.from_files(cfg.phonemes.inventory_path)
    prepare_corpus(manifest, tmp_path / "corpus", cfg, G2PEngine.from_config(cfg, inv))
    mapper = AlignerLabelMapper.from_mapping_file(cfg.phonemes.mapping_path, inv)
    summary = align_corpus_with_mfa(tmp_path / "corpus", tmp_path / "align", cfg, inv, mapper, MFAAdapter(cfg.alignment.mfa))
    outcome = summary.outcomes[0]
    assert outcome.textgrid is not None and outcome.status in {"success", "partial"}
    from asa_m2.schemas import SampleRecord
    rec = SampleRecord.model_validate_json(outcome.json_path.read_text(encoding="utf-8"))
    assert [w.label for w in rec.word_alignments if w.label_role == "word"] == text.lower().split()
    assert rec.alignment_metadata.aligner_version == summary.mfa.version
    assert all(p.realized_phone is None for p in rec.phone_alignments)
