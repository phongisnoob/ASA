"""Integration: REAL eSpeak-ng. Skipped (not passed) when the binary is absent."""

import random
import shutil
import subprocess

import pytest

from asa_m2.config import load_config
from asa_m2.g2p import CMUdictBackend, G2PEngine
from asa_m2.g2p.espeak_backend import EspeakBackend, EspeakFallbackError
from asa_m2.phonemes import PhoneInventory, PhoneMapping

pytestmark = pytest.mark.integration

if shutil.which("espeak-ng") is None:
    pytest.skip("espeak-ng is not installed (apt install espeak-ng); live eSpeak tests NOT run", allow_module_level=True)


@pytest.fixture(scope="module")
def backend():
    cfg = load_config()
    inv = PhoneInventory.from_files(cfg.phonemes.inventory_path)
    return EspeakBackend(inv, PhoneMapping.from_file(cfg.phonemes.mapping_path, "espeak_en_us_ipa", inv)), inv


def test_version_is_detected(backend):
    assert backend[0].version() is not None and backend[0].is_available()


@pytest.mark.parametrize("word,expected", [
    ("think", ["TH", "IH1", "NG", "K"]),
    ("sheesh", ["SH", "IY1", "SH"]),
    ("church", ["CH", "ER1", "CH"]),
])
def test_known_words_map_as_documented(backend, word, expected):
    assert backend[0].pronounce(word).raw_arpabet == expected


def test_flap_word_is_unresolved_not_guessed(backend):
    with pytest.raises(EspeakFallbackError) as exc:
        backend[0].pronounce("platter")
    assert exc.value.reason == "espeak_unmapped_phone" and "ɾ" in exc.value.details["units"]


def test_mapping_is_complete_on_a_fresh_vocabulary_sample(backend):
    """Every eSpeak unit seen on 1,500 random CMUdict words (a different seed from the one the mapping
    table was derived from) is either mapped or *deliberately* unmapped - no unknown unit slips through."""
    es, inv = backend
    words = sorted(w for w in CMUdictBackend()._entries if w.isalpha())
    sample = random.Random(20260610).sample(words, 1500)
    resolved = by_design = unexpected = 0
    unexpected_units: set[str] = set()
    for w in sample:
        try:
            v = es.pronounce(w)
        except EspeakFallbackError as exc:
            if exc.reason == "espeak_unmapped_phone" and set(exc.details["units"]) <= set(es.mapping.unmapped_by_design):
                by_design += 1
            else:
                unexpected += 1
                unexpected_units |= set(exc.details["units"]) or {exc.reason}
            continue
        assert all(p in inv for p in v.base_phones)  # every output phone is in the canonical inventory
        resolved += 1
    print(f"\neSpeak fallback coverage on 1,500 CMUdict words: resolved={resolved} by_design_unmapped={by_design} unexpected={unexpected}")
    assert unexpected == 0, f"unexpected eSpeak output that the mapping does not cover: {sorted(unexpected_units)}"
    assert resolved / len(sample) > 0.85


def test_fallback_engine_end_to_end_with_real_espeak():
    cfg = load_config()
    cfg.g2p.fallback_backend = "espeak"
    res = G2PEngine.from_config(cfg).convert("the zorblax cat")
    t = next(t for t in res.tokens if t.normalized_token == "zorblax")
    assert t.resolution_status in {"resolved_fallback", "unresolved"}  # either way it is explicit, never silent
    if t.resolution_status == "resolved_fallback":
        assert t.pronunciation_source == "espeak-ng" and "fallback_pronunciation" in t.flags
    assert next(t for t in res.tokens if t.normalized_token == "cat").pronunciation_source == "cmudict"


def test_shell_metacharacters_in_a_word_never_reach_a_shell(backend, tmp_path):
    marker = tmp_path / "pwned"
    with pytest.raises(EspeakFallbackError, match="single word"):  # whitespace => rejected before any process starts
        backend[0].pronounce(f"x;touch {marker}")
    try:  # no whitespace: eSpeak receives it as ONE literal argument after `--`; no shell is involved
        backend[0].pronounce(f"a;touch{marker}".replace("/", "_"))
    except EspeakFallbackError:
        pass  # unmappable output is an acceptable, explicit outcome
    assert not marker.exists()


def test_espeak_binary_responds():
    assert subprocess.run(["espeak-ng", "--version"], capture_output=True, text=True).returncode == 0
