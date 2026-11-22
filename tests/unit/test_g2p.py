"""M2.2 G2P: CMUdict variants, stress, OOV, lexicon, eSpeak fallback, alignment gating."""

import subprocess
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from asa_m2.config import load_config
from asa_m2.exceptions import (
    G2PError,
    InventoryError,
    LexiconError,
    OptionalDependencyError,
    UnresolvedPronunciationError,
)
from asa_m2.g2p import CMUdictBackend, CustomLexicon, EspeakBackend, G2PEngine, ensure_alignable
from asa_m2.g2p.espeak_backend import EspeakFallbackError
from asa_m2.g2p.variants import build_variant
from asa_m2.phonemes import PhoneMapping
from asa_m2.schemas import G2PResult, TokenRecord


def tok(res: G2PResult, word: str) -> TokenRecord:
    return next(t for t in res.tokens if t.normalized_token == word)


# ---- CMUdict -----------------------------------------------------------------------
def test_example_sentence_follows_cmudict(engine):
    res = engine.convert("I think this is important.")
    assert [t.normalized_token for t in res.tokens] == ["i", "think", "this", "is", "important"]
    assert tok(res, "i").variants[0].raw_arpabet == ["AY1"]
    assert tok(res, "think").variants[0].raw_arpabet == ["TH", "IH1", "NG", "K"]
    this = tok(res, "this")
    assert [v.raw_arpabet for v in this.variants] == [["DH", "IH1", "S"], ["DH", "IH0", "S"]]  # both CMUdict variants kept


def test_all_cmudict_variants_are_kept_and_none_silently_selected(engine):
    the = tok(engine.convert("the"), "the")
    assert len(the.variants) == 3
    assert the.selected_variant_id is None and the.selection_method == "unselected"
    assert "variant_unselected" in the.flags
    assert [v.variant_id for v in the.variants] == ["cmudict:the:1", "cmudict:the:2", "cmudict:the:3"]


def test_sole_variant_is_marked_as_such(engine):
    t = tok(engine.convert("cat"), "cat")
    assert t.selection_method == "sole_variant" and t.selected_variant_id == "cmudict:cat:1"


def test_variant_ids_pronunciation_source_and_status(engine):
    t = tok(engine.convert("think"), "think")
    assert t.pronunciation_source == "cmudict" and t.resolution_status == "resolved"
    assert t.approved_for_alignment and t.variants[0].source_detail.startswith("cmudict")


def test_stress_is_preserved_and_separated(engine):
    v = tok(engine.convert("important"), "important").variants[0]
    assert v.raw_arpabet == ["IH2", "M", "P", "AO1", "R", "T", "AH0", "N", "T"]  # raw ARPABET untouched
    assert v.base_phones == ["IH", "M", "P", "AO", "R", "T", "AH", "N", "T"]  # no digits in identity
    assert v.stress == [2, None, None, 1, None, None, 0, None, None]
    assert all(not any(c.isdigit() for c in b) for b in v.base_phones)
    assert v.phone_ids == [17, 22, 27, 4, 28, 31, 3, 23, 31]


def test_ipa_is_display_only(engine):
    v = tok(engine.convert("think"), "think").variants[0]
    assert v.ipa_display == ["θ", "ɪ", "ŋ", "k"] and v.base_phones[0] == "TH"


def test_case_insensitive_lookup_preserves_original_token(engine):
    t = engine.convert("THINK").tokens[0]
    assert t.original_token == "THINK" and t.normalized_token == "think" and t.resolution_status == "resolved"


def test_canonical_phones_only_for_selected_variants(engine):
    res = engine.convert("I think")
    assert [(p.token, p.base_phone, p.stress) for p in res.canonical_phones] == [
        ("i", "AY", 1), ("think", "TH", None), ("think", "IH", 1), ("think", "NG", None), ("think", "K", None)]
    assert [p.index for p in res.canonical_phones] == list(range(5))
    res2 = engine.convert("this cat")  # 'this' is unselected -> contributes no canonical phones
    assert {p.token for p in res2.canonical_phones} == {"cat"}


def test_word_indices_and_spans(engine):
    text = "Hello, world!"
    res = engine.convert(text)
    assert [t.word_index for t in res.tokens] == [0, 1]
    assert [text[t.original_span.start:t.original_span.end] for t in res.tokens] == ["Hello", "world"]


def test_text_field_is_recorded_and_validated(engine):
    assert engine.convert("hi").text_field == "reference_text"
    assert engine.convert("hi", text_field="human_verbatim").text_field == "human_verbatim"
    with pytest.raises(G2PError):
        engine.convert("hi", text_field="answer")


# ---- OOV ---------------------------------------------------------------------------
def test_oov_is_detected_and_unresolved(engine):
    res = engine.convert("the blorptastic cat")
    bad = tok(res, "blorptastic")
    assert bad.resolution_status == "unresolved" and bad.resolution_reason == "oov_no_pronunciation"
    assert not bad.approved_for_alignment and bad.variants == [] and "oov_cmudict" in bad.flags
    assert [t.original_token for t in res.unresolved_tokens] == ["blorptastic"]
    assert any(f.code == "unresolved_word" and f.severity == "error" and f.word_index == 1 for f in res.quality_flags)


def test_unsupported_normalised_token_is_unresolved_not_dropped(engine):
    res = engine.convert("see 3D now")
    t = tok(res, "3d")
    assert t.resolution_status == "unresolved" and t.resolution_reason == "unsupported_token:mixed_letters_and_digits"
    assert len(res.tokens) == 3


def test_unresolved_words_are_blocked_from_alignment(engine):
    res = engine.convert("the blorptastic cat")
    assert not res.alignable
    with pytest.raises(UnresolvedPronunciationError, match="blorptastic"):
        ensure_alignable(res)
    ensure_alignable(res, allow_unresolved=True)  # explicit override only
    ensure_alignable(engine.convert("the cat"))


def test_ensure_alignable_rejects_empty(engine):
    with pytest.raises(UnresolvedPronunciationError, match="no tokens"):
        ensure_alignable(engine.convert(""))
    assert any(f.code == "no_tokens" for f in engine.convert("").quality_flags)


def test_empty_text_gives_error_flag_not_exception(engine):
    res = engine.convert("   ")
    assert res.tokens == [] and not res.alignable


# ---- variant policy ----------------------------------------------------------------
def test_first_listed_policy_is_explicit_and_flagged(config, inventory):
    config.g2p.variant_policy = "first_listed"
    t = tok(G2PEngine.from_config(config, inventory).convert("this"), "this")
    assert t.selected_variant_id == "cmudict:this:1" and t.selection_method == "policy_first_listed"
    assert "variant_chosen_by_policy" in t.flags
    assert len(t.variants) == 2  # alternatives are still retained


def test_single_variant_export_requires_explicit_policy():
    with pytest.raises(ValidationError, match="first_listed"):
        cfg = load_config()
        cfg.g2p.allow_multiple_variants = False


# ---- custom lexicon ----------------------------------------------------------------
LEX = """# test lexicon
;;; another comment style
hanoi   HH AA0 N OY1
hanoi   HH AE0 N OY1     # second variant
blorptastic B L AO1 R P T AE1 S T IH0 K
"""


@pytest.fixture()
def lex_engine(config, inventory):
    def build(text, mode="override"):
        lexicon = CustomLexicon.from_text(text, inventory, mode)
        return G2PEngine(config, inventory, lexicon=lexicon)
    return build


def test_custom_lexicon_resolves_oov_with_source_recorded(lex_engine):
    t = tok(lex_engine(LEX).convert("a blorptastic idea"), "blorptastic")
    assert t.resolution_status == "resolved" and t.pronunciation_source == "custom_lexicon"
    assert t.variants[0].variant_id == "custom:blorptastic:1" and "oov_cmudict" in t.flags
    assert "custom_lexicon_override" in t.flags


def test_custom_lexicon_override_replaces_cmudict(lex_engine):
    t = tok(lex_engine("think  TH IH2 NG K\n").convert("think"), "think")
    assert [v.raw_arpabet for v in t.variants] == [["TH", "IH2", "NG", "K"]]  # CMUdict's IH1 variant not mixed in
    assert t.pronunciation_source == "custom_lexicon"


def test_custom_lexicon_augment_appends_after_cmudict(lex_engine):
    t = tok(lex_engine("think  TH IH2 NG K\n", "augment").convert("think"), "think")
    assert [v.variant_id for v in t.variants] == ["cmudict:think:1", "custom:think:2"]
    assert t.pronunciation_source == "cmudict+custom_lexicon" and t.selection_method == "unselected"


def test_custom_lexicon_multiple_variants_and_manual_select(lex_engine):
    eng = lex_engine(LEX + "!select hanoi 2\n")
    t = tok(eng.convert("hanoi"), "hanoi")
    assert [v.raw_arpabet[1] for v in t.variants] == ["AA0", "AE0"]
    assert t.selected_variant_id == "custom:hanoi:2" and t.selection_method == "manual_override"
    assert "variant_manual_override" in t.flags


def test_manual_select_out_of_range_is_an_error(lex_engine):
    with pytest.raises(G2PError, match="only 2 variant"):
        lex_engine(LEX + "!select hanoi 5\n").convert("hanoi")


@pytest.mark.parametrize("line,fragment", [
    ("word", "no phones"), ("word  XX Y", "unknown phone"), ("word  AA9", "stress"),
    ("bad$word  AA", "invalid word"), ("!select word", "!select"), ("!select word 0", "!select"),
])
def test_custom_lexicon_errors_carry_line_numbers(inventory, line, fragment):
    with pytest.raises(LexiconError, match=fragment) as exc:
        CustomLexicon.from_text("ok  AA1\n" + line + "\n", inventory)
    assert "<inline>:2" in str(exc.value)


def test_custom_lexicon_from_file_records_hash_and_missing_file(tmp_path, inventory):
    p = tmp_path / "lex.dict"
    p.write_text(LEX, encoding="utf-8")
    lex = CustomLexicon.from_file(p, inventory)
    assert len(lex) == 2 and len(lex.sha256 or "") == 64 and "hanoi" in lex
    with pytest.raises(LexiconError, match="not found"):
        CustomLexicon.from_file(tmp_path / "absent.dict", inventory)
    with pytest.raises(LexiconError, match="mode"):
        CustomLexicon.from_text("a AA", inventory, "merge")


def test_lexicon_in_provenance(lex_engine):
    prov = lex_engine(LEX).convert("hanoi").provenance
    assert "custom_lexicon" in prov.g2p_sources and "mode=override" in prov.g2p_sources["custom_lexicon"]


def test_variants_rejected_with_unknown_phones_are_reported(config, inventory):
    backend = CMUdictBackend({"odd": [["AA1", "ZZ"], ["B", "AO1"]], "bad": [["QQ"]]})
    eng = G2PEngine(config, inventory, cmudict=backend)
    odd = tok(eng.convert("odd"), "odd")
    assert [v.variant_id for v in odd.variants] == ["cmudict:odd:2"] and "rejected_variants" in odd.flags
    bad = tok(eng.convert("bad"), "bad")
    assert bad.resolution_status == "unresolved" and bad.resolution_reason.startswith("all_variants_rejected")


def test_build_variant_preserves_raw_and_validates(inventory):
    v = build_variant("x:1", "test", ["AY1", "T"], inventory)
    assert v.raw_arpabet == ["AY1", "T"] and v.base_phones == ["AY", "T"] and v.stress == [1, None]
    with pytest.raises(InventoryError, match="unknown phone"):
        build_variant("x:2", "test", ["AY1", "QQ"], inventory)


# ---- eSpeak (fake runner => deterministic) --------------------------------------------
def fake_runner(stdout="", returncode=0, stderr="", raises=None):
    def run(cmd, **kw):
        if raises:
            raise raises
        if "--version" in cmd:
            return SimpleNamespace(stdout="eSpeak NG text-to-speech: 1.51  Data at: /x", stderr="", returncode=0)
        run.calls.append(cmd)
        return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)
    run.calls = []
    return run


@pytest.fixture()
def espeak_mapping(inventory, config):
    return PhoneMapping.from_file(config.phonemes.mapping_path, "espeak_en_us_ipa", inventory)


def make_espeak(inventory, mapping, **kw):
    return EspeakBackend(inventory, mapping, runner=fake_runner(**kw))


def test_espeak_maps_units_and_assigns_stress(inventory, espeak_mapping):
    es = make_espeak(inventory, espeak_mapping, stdout="θ_ˈɪ_ŋ_k\n")
    v = es.pronounce("think")
    assert v.raw_arpabet == ["TH", "IH1", "NG", "K"] and v.source == "espeak-ng"
    assert v.variant_id == "espeak-ng:think:1" and "1.51" in v.source_detail and not v.approximate_mapping


def test_espeak_secondary_stress_and_unmarked_vowels(inventory, espeak_mapping):
    v = make_espeak(inventory, espeak_mapping, stdout="f_ˌoʊ_t_ə\n").pronounce("foto")
    assert v.raw_arpabet == ["F", "OW2", "T", "AH0"]


def test_espeak_multi_vowel_unit_stress_goes_to_first_vowel(inventory, espeak_mapping):
    v = make_espeak(inventory, espeak_mapping, stdout="f_ˈaɪɚ\n").pronounce("fire")
    assert v.raw_arpabet == ["F", "AY1", "ER0"] and v.approximate_mapping  # flagged approximate


def test_espeak_r_coloured_unit_expands(inventory, espeak_mapping):
    v = make_espeak(inventory, espeak_mapping, stdout="b_ˈoːɹ\n").pronounce("bor")
    assert v.raw_arpabet == ["B", "AO1", "R"]


def test_espeak_unmapped_phone_makes_word_unresolved_with_units_listed(inventory, espeak_mapping):
    es = make_espeak(inventory, espeak_mapping, stdout="p_l_ˈæ_ɾ_ɚ\n")
    with pytest.raises(EspeakFallbackError) as exc:
        es.pronounce("platter")
    assert exc.value.reason == "espeak_unmapped_phone" and exc.value.details["units"] == ["ɾ"]


@pytest.mark.parametrize("stdout,code", [
    ("", "espeak_empty_output"), ("(fr)b_ɔ̃(en)\n", "espeak_language_switch"), ("a_b c_d\n", "espeak_multiword_output"),
])
def test_espeak_untrustworthy_output_rejected(inventory, espeak_mapping, stdout, code):
    with pytest.raises(EspeakFallbackError) as exc:
        make_espeak(inventory, espeak_mapping, stdout=stdout).pronounce("x")
    assert exc.value.reason == code


def test_espeak_nonzero_exit_timeout_and_missing_binary(inventory, espeak_mapping):
    with pytest.raises(EspeakFallbackError) as e1:
        make_espeak(inventory, espeak_mapping, returncode=3, stderr="boom").pronounce("x")
    assert e1.value.reason == "espeak_failed"
    with pytest.raises(EspeakFallbackError) as e2:
        make_espeak(inventory, espeak_mapping, raises=subprocess.TimeoutExpired("espeak-ng", 1)).pronounce("x")
    assert e2.value.reason == "espeak_timeout"
    with pytest.raises(G2PError, match="not found"):
        make_espeak(inventory, espeak_mapping, raises=FileNotFoundError()).pronounce("x")


def test_espeak_command_is_a_list_without_shell_and_guards_option_injection(inventory, espeak_mapping):
    run = fake_runner(stdout="b\n")
    EspeakBackend(inventory, espeak_mapping, runner=run).pronounce("-rf")
    cmd = run.calls[0]
    assert isinstance(cmd, list) and cmd[-2:] == ["--", "-rf"] and cmd[:2] == ["espeak-ng", "-q"]
    with pytest.raises(EspeakFallbackError, match="single word"):
        EspeakBackend(inventory, espeak_mapping, runner=run).pronounce("two words")


def fallback_engine(config, inventory, mapping, **kw):
    config.g2p.fallback_backend = "espeak"
    es = make_espeak(inventory, mapping, **kw)
    es.is_available = lambda: True  # type: ignore[method-assign]
    return G2PEngine(config, inventory, espeak=es)


def test_fallback_only_used_for_oov_and_flagged(config, inventory, espeak_mapping):
    eng = fallback_engine(config, inventory, espeak_mapping, stdout="b_l_ˈɔː_ɹ_p\n")
    res = eng.convert("the blorp")
    assert tok(res, "the").pronunciation_source == "cmudict"  # in-vocabulary words never touch eSpeak
    t = tok(res, "blorp")
    assert t.resolution_status == "resolved_fallback" and t.approved_for_alignment
    assert t.pronunciation_source == "espeak-ng" and {"oov_cmudict", "fallback_pronunciation"} <= set(t.flags)
    assert any(f.code == "fallback_pronunciation" and f.severity == "warning" for f in res.quality_flags)
    assert res.provenance.g2p_sources["espeak-ng"].startswith("espeak-ng 1.51")


def test_fallback_not_used_unless_explicitly_configured(config, inventory, espeak_mapping):
    es = make_espeak(inventory, espeak_mapping, stdout="b\n")
    eng = G2PEngine(config, inventory, espeak=es)  # fallback_backend is null
    assert eng.espeak is None
    assert tok(eng.convert("blorp"), "blorp").resolution_status == "unresolved"
    assert es._run.calls == []  # type: ignore[attr-defined]


def test_fallback_failure_leaves_word_unresolved_with_reason(config, inventory, espeak_mapping):
    eng = fallback_engine(config, inventory, espeak_mapping, stdout="p_l_ˈæ_ɾ_ɚ\n")
    t = tok(eng.convert("platterx"), "platterx")
    assert t.resolution_status == "unresolved" and t.resolution_reason.startswith("espeak_unmapped_phone")
    assert "fallback_failed" in t.flags and not t.approved_for_alignment


def test_configured_espeak_but_unavailable_is_a_hard_error(config, inventory, espeak_mapping):
    config.g2p.fallback_backend = "espeak"
    es = EspeakBackend(inventory, espeak_mapping, executable="definitely-not-installed-espeak")
    with pytest.raises(OptionalDependencyError, match="not on PATH"):
        G2PEngine(config, inventory, espeak=es)
    with pytest.raises(OptionalDependencyError):
        G2PEngine(config, inventory)  # configured but not supplied


def test_cmudict_backend_basics():
    be = CMUdictBackend()
    assert len(be) > 100_000 and "think" in be and "THINK" in be and "xyzzyq" not in be
    assert be.lookup("xyzzyq") == [] and be.version
