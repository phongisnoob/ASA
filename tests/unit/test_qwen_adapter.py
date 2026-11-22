"""M2.8 optional Qwen3-ForcedAligner adapter. Live integration is UNVERIFIED in this build:
these tests use an injected fake model and check the adapter's own logic only."""

import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from asa_m2.alignment import qwen_adapter
from asa_m2.alignment.qwen_adapter import QwenForcedAlignerAdapter, qwen_would_keep
from asa_m2.config import QwenConfig
from asa_m2.exceptions import AlignmentError, OptionalDependencyError, QwenResourceError


class FakeModel:
    def __init__(self, items=None, raises=None):
        self.items, self.raises, self.calls = items, raises, []

    def align(self, audio, text, language):
        self.calls.append((audio, text, language))
        if self.raises:
            raise self.raises
        items = self.items if self.items is not None else [
            SimpleNamespace(text=w, start_time=round(0.2 * i, 3), end_time=round(0.2 * i + 0.15, 3)) for i, w in enumerate(text.split())]
        return [items]  # one result per input audio, as in the official README


def adapter(model=None, **cfg):
    model = model or FakeModel()
    return QwenForcedAlignerAdapter(QwenConfig(**cfg), model_factory=lambda name: model), model


def test_importing_the_adapter_loads_nothing_heavy():
    assert "torch" not in sys.modules or "qwen_asr" not in sys.modules
    assert "qwen_asr" not in sys.modules
    importlib.reload(qwen_adapter)
    assert "qwen_asr" not in sys.modules and QwenForcedAlignerAdapter.granularity == "word"


def test_model_loading_is_lazy():
    calls = []
    a = QwenForcedAlignerAdapter(QwenConfig(), model_factory=lambda name: calls.append(name) or FakeModel())
    assert calls == [] and a._model is None  # constructing does not load
    a.load()
    a.load()
    assert calls == ["Qwen/Qwen3-ForcedAligner-0.6B"]  # loaded once


def test_word_level_alignment_preserves_times_and_units(tmp_path, wav_factory):
    a, model = adapter()
    res = a.align_words(wav_factory(tmp_path / "x.wav", seconds=2.0, sr=22050), ["i", "think"])
    assert [(i.label, i.start, i.end) for i in res.intervals] == [("i", 0.0, 0.15), ("think", 0.2, 0.35)]
    audio, text, language = model.calls[0]
    assert isinstance(audio, tuple) and audio[0].dtype == np.float32 and audio[0].ndim == 1 and audio[1] == 22050
    assert (text, language) == ("i think", "English")
    assert res.model_info["model_name"] == "Qwen/Qwen3-ForcedAligner-0.6B"


def test_stereo_audio_is_downmixed(tmp_path, wav_factory):
    a, model = adapter()
    a.align_words(wav_factory(tmp_path / "s.wav", seconds=1.0, channels=2), ["hi"])
    assert model.calls[0][0][0].ndim == 1


def test_unit_count_mismatch_is_refused_not_guessed(tmp_path, wav_factory):
    a, _ = adapter(FakeModel(items=[SimpleNamespace(text="i", start_time=0.0, end_time=0.1)]))
    with pytest.raises(AlignmentError, match="returned 1 units for 2 words"):
        a.align_words(wav_factory(tmp_path / "x.wav"), ["i", "think"])


def test_unit_text_mismatch_is_refused(tmp_path, wav_factory):
    items = [SimpleNamespace(text="i", start_time=0, end_time=0.1), SimpleNamespace(text="thing", start_time=0.1, end_time=0.3)]
    a, _ = adapter(FakeModel(items=items))
    with pytest.raises(AlignmentError, match="expected 'think'"):
        a.align_words(wav_factory(tmp_path / "x.wav"), ["i", "think"])


@pytest.mark.parametrize("token,kept", [("think", "think"), ("don't", "don't"), ("well-known", "wellknown"), ("a.b", "ab"), ("caf\u00e9", "caf\u00e9")])
def test_qwen_token_cleaning_replica(token, kept):
    assert qwen_would_keep(token) == kept


def test_tokens_that_qwen_would_alter_are_rejected_up_front(tmp_path, wav_factory):
    a, model = adapter()
    with pytest.raises(AlignmentError, match="would be altered"):
        a.align_words(wav_factory(tmp_path / "x.wav"), ["well-known"])
    assert model.calls == []  # nothing was run


def test_url_and_base64_audio_refused_no_hidden_network(tmp_path):
    a, model = adapter()
    for bad in ("https://example.com/a.wav", "data:audio/wav;base64,AAAA", "s3://bucket/a.wav"):
        with pytest.raises(AlignmentError, match="URL"):
            a.align_words(bad, ["hi"])
    assert model.calls == []


def test_missing_audio_file_and_empty_words(tmp_path):
    a, _ = adapter()
    with pytest.raises(AlignmentError, match="not found"):
        a.align_words(tmp_path / "none.wav", ["hi"])
    with pytest.raises(AlignmentError, match="no words"):
        a.align_words(tmp_path / "none.wav", [])


def test_audio_longer_than_supported_limit_is_refused(tmp_path, wav_factory):
    a, _ = adapter(max_audio_seconds=1.0)
    with pytest.raises(AlignmentError, match="at most"):
        a.align_words(wav_factory(tmp_path / "long.wav", seconds=2.0), ["hi"])


def test_gpu_out_of_memory_is_translated(tmp_path, wav_factory):
    a, _ = adapter(FakeModel(raises=RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")))
    with pytest.raises(QwenResourceError, match="out of memory"):
        a.align_words(wav_factory(tmp_path / "x.wav"), ["hi"])


def test_other_model_failures_become_alignment_errors(tmp_path, wav_factory):
    a, _ = adapter(FakeModel(raises=ValueError("bad shape")))
    with pytest.raises(AlignmentError, match="ValueError: bad shape"):
        a.align_words(wav_factory(tmp_path / "x.wav"), ["hi"])


def test_missing_optional_dependency_is_reported_clearly(monkeypatch):
    monkeypatch.setattr(qwen_adapter, "is_available", lambda: False)
    with pytest.raises(OptionalDependencyError, match="pip install qwen-asr"):
        QwenForcedAlignerAdapter(QwenConfig()).load()


def test_adapter_reports_unverified_live_integration_and_official_urls():
    assert qwen_adapter.LIVE_INTEGRATION_VERIFIED is False
    assert "https://github.com/QwenLM/Qwen3-ASR" in qwen_adapter.OFFICIAL_URLS


def test_adapter_never_emits_phone_level_data(tmp_path, wav_factory, engine, config, inventory):
    from asa_m2.alignment import assemble_sample_record
    from asa_m2.schemas import AlignmentMetadata
    a, _ = adapter()
    res = a.align_words(wav_factory(tmp_path / "x.wav", seconds=2.0), ["i", "think"])
    rec = assemble_sample_record(
        g2p=engine.convert("I think"), word_intervals=res.intervals, phone_intervals=None, word_tier_name="qwen_words",
        phone_tier_name=None, config=config, inventory=inventory, sample_id="S", speaker_id=None, reference_text="I think",
        human_verbatim=None, alignment_metadata=AlignmentMetadata(aligner="Qwen3-ForcedAligner", granularity="word", status="success"))
    assert rec.alignment_metadata.granularity == "word" and rec.phone_alignments == []
