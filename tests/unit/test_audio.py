"""M2.4 audio: loading, validation, down-mix, *real* resampling, safe WAV export."""

import numpy as np
import pytest
import soundfile as sf
from pydantic import ValidationError

from asa_m2.audio import (
    load_audio,
    prepare_audio,
    probe_audio,
    resample,
    to_mono,
    validate_audio,
    write_wav,
)
from asa_m2.config import AudioConfig
from asa_m2.exceptions import (
    AudioError,
    AudioNotFoundError,
    AudioReadError,
    AudioValidationError,
    OutputExistsError,
)


def dominant_freq(x: np.ndarray, sr: int) -> float:
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.fft.rfftfreq(len(x), 1 / sr)[spec.argmax()])


@pytest.fixture()
def cfg() -> AudioConfig:
    return AudioConfig()


# ---- loading -----------------------------------------------------------------------
def test_mono_wav_probe_and_load(tmp_path, wav_factory):
    p = wav_factory(tmp_path / "m.wav", seconds=0.5, sr=16000)
    info = probe_audio(p)
    assert (info.sample_rate, info.channels, info.frames) == (16000, 1, 8000) and info.duration_s == pytest.approx(0.5)
    data = load_audio(p)
    assert data.samples.shape == (8000, 1) and data.samples.dtype == np.float64


def test_stereo_wav_loaded_with_two_channels(tmp_path, wav_factory):
    data = load_audio(wav_factory(tmp_path / "s.wav", channels=2))
    assert data.samples.shape[1] == 2


def test_missing_file(tmp_path):
    with pytest.raises(AudioNotFoundError):
        probe_audio(tmp_path / "nope.wav")
    with pytest.raises(AudioNotFoundError):
        load_audio(tmp_path / "nope.wav")


def test_directory_is_not_audio(tmp_path):
    with pytest.raises(AudioReadError, match="not a file"):
        probe_audio(tmp_path)


def test_corrupt_audio_is_reported_not_crashed(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"this is definitely not a RIFF file" * 10)
    with pytest.raises(AudioReadError):
        probe_audio(bad)


def test_truncated_wav_detected(tmp_path, wav_factory):
    p = wav_factory(tmp_path / "t.wav", seconds=1.0)
    raw = p.read_bytes()
    p.write_bytes(raw[: len(raw) // 2])  # header promises more frames than exist
    with pytest.raises(AudioReadError):
        load_audio(p)


# ---- validation --------------------------------------------------------------------
def test_valid_audio_passes(tmp_path, wav_factory, cfg):
    rep = validate_audio(wav_factory(tmp_path / "ok.wav", seconds=1.0, sr=16000), cfg)
    assert rep.ok and rep.issues == []


def test_incorrect_sample_rate_is_warned_and_low_rate_is_rejected(tmp_path, wav_factory, cfg):
    rep = validate_audio(wav_factory(tmp_path / "a.wav", sr=44100), cfg)
    assert rep.ok and [i.code for i in rep.warnings] == ["sample_rate_differs"]
    low = validate_audio(wav_factory(tmp_path / "b.wav", sr=4000), cfg)
    assert not low.ok and "sample_rate_too_low" in [i.code for i in low.errors]


def test_missing_audio_validation(tmp_path, cfg):
    rep = validate_audio(tmp_path / "absent.wav", cfg)
    assert [i.code for i in rep.errors] == ["missing_file"]
    with pytest.raises(AudioValidationError, match="missing_file"):
        rep.raise_if_invalid()


def test_empty_audio_rejected(tmp_path, cfg):
    p = tmp_path / "empty.wav"
    sf.write(str(p), np.zeros(0), 16000)
    assert [i.code for i in validate_audio(p, cfg).errors] == ["empty_audio"]


def test_corrupt_audio_validation(tmp_path, cfg):
    p = tmp_path / "bad.wav"
    p.write_bytes(b"garbage")
    assert [i.code for i in validate_audio(p, cfg).errors] == ["unreadable"]


def test_duration_limits(tmp_path, wav_factory):
    cfg = AudioConfig(min_duration_s=0.5, max_duration_s=2.0)
    assert [i.code for i in validate_audio(wav_factory(tmp_path / "s.wav", seconds=0.2), cfg).errors] == ["duration_too_short"]
    assert [i.code for i in validate_audio(wav_factory(tmp_path / "l.wav", seconds=3.0), cfg).errors] == ["duration_too_long"]
    assert validate_audio(wav_factory(tmp_path / "k.wav", seconds=1.0), cfg).ok


def test_digital_silence_and_clipping(tmp_path, cfg):
    z = tmp_path / "z.wav"
    sf.write(str(z), np.zeros(16000), 16000)
    assert [i.code for i in validate_audio(z, cfg).errors] == ["digital_silence"]
    c = tmp_path / "c.wav"
    sf.write(str(c), np.ones(16000) * 0.9999, 16000, subtype="PCM_16")
    assert "clipping" in [i.code for i in validate_audio(c, cfg).warnings]


def test_unsupported_extension(tmp_path, wav_factory, cfg):
    p = wav_factory(tmp_path / "x.wav")
    q = p.rename(tmp_path / "x.xyz")
    assert "unsupported_extension" in [i.code for i in validate_audio(q, cfg).errors]


def test_reject_policy_for_stereo(tmp_path, wav_factory):
    cfg = AudioConfig(stereo_to_mono="reject")
    assert "not_mono" in [i.code for i in validate_audio(wav_factory(tmp_path / "s.wav", channels=2), cfg).errors]


def test_validation_does_not_modify_the_file(tmp_path, wav_factory, cfg):
    p = wav_factory(tmp_path / "m.wav", sr=44100)
    before = p.read_bytes()
    validate_audio(p, cfg)
    assert p.read_bytes() == before


def test_audio_config_rejects_modifying_originals():
    with pytest.raises(ValidationError, match="never modifies original"):
        AudioConfig(preserve_original=False)
    with pytest.raises(ValidationError, match="min_duration_s"):
        AudioConfig(min_duration_s=5, max_duration_s=1)


# ---- down-mix ----------------------------------------------------------------------
def test_to_mono_policies():
    st = np.stack([np.ones(4), np.zeros(4)], axis=1)
    assert np.allclose(to_mono(st, "mean"), 0.5)
    assert np.allclose(to_mono(st, "left"), 1.0) and np.allclose(to_mono(st, "right"), 0.0)
    with pytest.raises(AudioError, match="reject"):
        to_mono(st, "reject")
    with pytest.raises(AudioError, match="unknown"):
        to_mono(st, "weird")
    assert to_mono(np.ones((3, 1)), "reject").shape == (3,)  # already mono is always fine
    with pytest.raises(AudioError):
        to_mono(np.ones(3), "mean")  # wrong rank


# ---- resampling (must actually change samples) ------------------------------------------
@pytest.mark.parametrize("src,dst", [(44100, 16000), (48000, 16000), (22050, 16000), (8000, 16000), (16000, 8000), (11025, 16000)])
def test_resampling_changes_sample_count_correctly_and_preserves_frequency(src, dst):
    t = np.arange(src) / src  # exactly 1 second
    x = np.sin(2 * np.pi * 440 * t)
    y = resample(x, src, dst)
    assert len(y) == dst  # 1 s stays 1 s
    assert len(y) != len(x) or src == dst
    assert dominant_freq(y, dst) == pytest.approx(440, abs=2)


def test_resampling_is_antialiased():
    """A 7 kHz tone is above the new 4 kHz Nyquist: it must be attenuated, not aliased."""
    src, dst = 16000, 8000
    t = np.arange(src) / src
    y = resample(np.sin(2 * np.pi * 7000 * t), src, dst)
    assert np.sqrt(np.mean(y[100:-100] ** 2)) < 0.01  # RMS of the input was ~0.707


def test_resampling_same_rate_returns_copy_not_alias():
    x = np.random.default_rng(0).standard_normal(100)
    y = resample(x, 16000, 16000)
    assert np.array_equal(x, y) and y is not x


def test_resample_input_validation():
    with pytest.raises(AudioError):
        resample(np.zeros(0), 16000, 8000)
    with pytest.raises(AudioError):
        resample(np.ones((4, 2)), 16000, 8000)
    with pytest.raises(AudioError):
        resample(np.ones(4), 0, 8000)


def test_prepare_audio_records_every_transformation(tmp_path, wav_factory, cfg):
    data = load_audio(wav_factory(tmp_path / "s.wav", seconds=1.0, sr=44100, channels=2, freq=300))
    prep = prepare_audio(data, cfg)
    assert prep.sample_rate == 16000 and prep.samples.shape == (16000,)
    ops = {o["op"]: o for o in prep.transformations}
    assert ops["to_mono"]["policy"] == "mean" and ops["to_mono"]["channels_in"] == 2
    assert (ops["resample"]["from_hz"], ops["resample"]["to_hz"], ops["resample"]["samples_in"], ops["resample"]["samples_out"]) == (44100, 16000, 44100, 16000)
    assert "polyphase" in ops["resample"]["method"]
    assert dominant_freq(prep.samples, 16000) == pytest.approx(300, abs=2)


def test_prepare_audio_same_rate_is_recorded_as_skipped(tmp_path, wav_factory, cfg):
    prep = prepare_audio(load_audio(wav_factory(tmp_path / "m.wav", sr=16000)), cfg)
    assert [o["method"] for o in prep.transformations if o["op"] == "resample"] == ["skipped_same_rate"]


def test_configurable_target_rate(tmp_path, wav_factory):
    prep = prepare_audio(load_audio(wav_factory(tmp_path / "m.wav", seconds=1, sr=16000)), AudioConfig(target_sample_rate=22050))
    assert prep.sample_rate == 22050 and len(prep.samples) == 22050


def test_overshoot_is_clipped_and_counted(tmp_path, cfg):
    p = tmp_path / "loud.wav"
    sq = np.where(np.arange(8000) % 40 < 20, 0.99, -0.99)  # square wave: resampling ringing overshoots
    sf.write(str(p), sq, 8000)
    prep = prepare_audio(load_audio(p), cfg)
    assert np.max(np.abs(prep.samples)) <= 1.0
    clip = [o for o in prep.transformations if o["op"] == "clip"]
    assert clip and clip[0]["count"] > 0  # recorded, not silent


# ---- writing -----------------------------------------------------------------------
def test_write_wav_roundtrip_and_pcm_subtype(tmp_path):
    x = 0.5 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)
    out = tmp_path / "o" / "x.wav"
    write_wav(out, x, 16000, "PCM_16")
    info = sf.info(str(out))
    assert (info.samplerate, info.channels, info.subtype, info.format) == (16000, 1, "PCM_16", "WAV")
    back, _ = sf.read(str(out))
    assert np.max(np.abs(back - x)) < 1e-3


def test_write_wav_refuses_overwrite_and_bad_input(tmp_path):
    out = tmp_path / "x.wav"
    write_wav(out, np.zeros(10) + 0.1, 16000, "PCM_16")
    with pytest.raises(OutputExistsError):
        write_wav(out, np.zeros(10) + 0.1, 16000, "PCM_16")
    write_wav(out, np.zeros(10) + 0.2, 16000, "PCM_16", overwrite=True)
    with pytest.raises(AudioError, match="full scale"):
        write_wav(tmp_path / "y.wav", np.array([1.5, 0.0]), 16000, "PCM_16")
    with pytest.raises(AudioError, match="mono"):
        write_wav(tmp_path / "z.wav", np.zeros((10, 2)), 16000, "PCM_16")
