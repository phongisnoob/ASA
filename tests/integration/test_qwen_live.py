"""Integration: REAL Qwen3-ForcedAligner. Opt-in only (large download, GPU recommended).

To run:  pip install qwen-asr ; export ASA_M2_RUN_QWEN=1 ; pytest tests/integration/test_qwen_live.py
NOT run in the build environment (no GPU, model not downloaded): live integration is UNVERIFIED.
"""

import os
import shutil
import subprocess

import pytest

from asa_m2.alignment import qwen_adapter
from asa_m2.alignment.qwen_adapter import QwenForcedAlignerAdapter
from asa_m2.config import QwenConfig

pytestmark = pytest.mark.integration

if os.environ.get("ASA_M2_RUN_QWEN") != "1":
    pytest.skip("live Qwen test NOT run: set ASA_M2_RUN_QWEN=1 to opt in (downloads a model)", allow_module_level=True)
if not qwen_adapter.is_available():
    pytest.skip("live Qwen test NOT run: qwen_asr / torch are not installed (pip install qwen-asr)", allow_module_level=True)
if shutil.which("espeak-ng") is None:
    pytest.skip("live Qwen test NOT run: espeak-ng needed to synthesise the test utterance", allow_module_level=True)


def test_real_qwen_word_alignment(tmp_path):
    wav = tmp_path / "u.wav"
    subprocess.run(["espeak-ng", "-v", "en-us", "-w", str(wav), "I think this is important"], check=True, capture_output=True)
    res = QwenForcedAlignerAdapter(QwenConfig()).align_words(wav, ["i", "think", "this", "is", "important"])
    assert [i.label for i in res.intervals] == ["i", "think", "this", "is", "important"]
    assert all(i.end > i.start >= 0 for i in res.intervals)
    assert [i.start for i in res.intervals] == sorted(i.start for i in res.intervals)
