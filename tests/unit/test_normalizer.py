"""M2.1 text normalisation: auditability, traceability, nothing silently lost."""

import pytest

from asa_m2.config import NormalizationConfig
from asa_m2.exceptions import NormalizationError
from asa_m2.normalization import TextNormalizer, number_to_words, ordinal_to_words


@pytest.fixture()
def norm() -> TextNormalizer:
    return TextNormalizer(NormalizationConfig())


def words(res):
    return [t.normalized for t in res.tokens if t.supported]


def test_empty_and_whitespace_input_gives_warning_not_crash(norm):
    for text in ("", "   ", "\n\t"):
        res = norm.normalize(text)
        assert res.tokens == [] and res.normalized_text == ""
        assert any(w.startswith("empty_input") for w in res.warnings)


def test_only_punctuation_is_recorded_as_dropped_with_warning(norm):
    res = norm.normalize("?!...")
    assert res.tokens == []
    assert [d.text for d in res.dropped] == ["?", "!", ".", ".", "."]
    assert any("no_pronounceable_tokens" in w for w in res.warnings)


def test_basic_sentence_preserves_order_and_original(norm):
    text = "I think this is important."
    res = norm.normalize(text)
    assert res.original_text == text  # original is never modified
    assert words(res) == ["i", "think", "this", "is", "important"]
    assert [t.index for t in res.tokens] == list(range(5))
    assert [d.text for d in res.dropped] == ["."]  # punctuation is recorded, not silently deleted


def test_punctuation_variants_dropped_and_audited(norm):
    res = norm.normalize('"Hello," she said; (really!)')
    assert words(res) == ["hello", "she", "said", "really"]
    kinds = {d.kind for d in res.dropped}
    assert kinds == {"punctuation"}
    res.verify()


def test_contractions_stay_single_tokens(norm):
    res = norm.normalize("Don't worry, it's what we couldn't do.")
    assert words(res) == ["don't", "worry", "it's", "what", "we", "couldn't", "do"]
    assert [t.kind for t in res.tokens if "'" in t.normalized] == ["contraction"] * 3


def test_unicode_apostrophe_normalised_with_rule_recorded(norm):
    res = norm.normalize("It\u2019s fine")
    tok = res.tokens[0]
    assert tok.normalized == "it's" and tok.original_text == "It\u2019s"
    assert "unicode_apostrophe_normalized" in tok.rules and "lowercase" in tok.rules


def test_quote_marks_around_words_are_punctuation_not_part_of_word(norm):
    res = norm.normalize("'quoted' words")
    assert words(res) == ["quoted", "words"]
    assert [d.text for d in res.dropped] == ["'", "'"]


def test_hyphenated_words_split_with_traceable_subspans(norm):
    text = "a well-known fact"
    res = norm.normalize(text)
    assert words(res) == ["a", "well", "known", "fact"]
    well, known = res.tokens[1], res.tokens[2]
    assert well.original_text == known.original_text == "well-known"
    assert text[well.sub_start:well.sub_end] == "well" and text[known.sub_start:known.sub_end] == "known"
    assert (well.expansion_index, known.expansion_index, well.expansion_size) == (0, 1, 2)


def test_hyphen_policy_keep(norm):
    res = TextNormalizer(NormalizationConfig(hyphen_policy="keep")).normalize("well-known")
    assert words(res) == ["well-known"]


@pytest.mark.parametrize("n,expected", [
    (0, "zero"), (7, "seven"), (13, "thirteen"), (21, "twenty one"), (100, "one hundred"),
    (101, "one hundred one"), (999, "nine hundred ninety nine"), (1000, "one thousand"),
    (12345, "twelve thousand three hundred forty five"), (2_000_001, "two million one"),
])
def test_number_to_words(n, expected):
    assert " ".join(number_to_words(n)) == expected


def test_number_out_of_range_raises():
    with pytest.raises(NormalizationError):
        number_to_words(10**15)
    with pytest.raises(NormalizationError):
        number_to_words(-1)


@pytest.mark.parametrize("n,expected", [(1, "first"), (2, "second"), (3, "third"), (5, "fifth"), (12, "twelfth"),
                                         (20, "twentieth"), (21, "twenty first"), (100, "one hundredth")])
def test_ordinals(n, expected):
    assert " ".join(ordinal_to_words(n)) == expected


def test_numbers_expand_with_expansion_trace(norm):
    res = norm.normalize("I have 21 books and 3.5 cakes, 1,000 pens, the 3rd one.")
    assert words(res) == ["i", "have", "twenty", "one", "books", "and", "three", "point", "five", "cakes",
                          "one", "thousand", "pens", "the", "third", "one"]
    twenty, one = res.tokens[2], res.tokens[3]
    assert twenty.original_text == one.original_text == "21"
    assert twenty.rules == ("number_expanded:cardinal",)
    assert (twenty.expansion_index, one.expansion_index, twenty.expansion_size) == (0, 1, 2)


def test_ambiguous_year_like_number_warns(norm):
    res = norm.normalize("in 1999")
    assert any("ambiguous_number" in w for w in res.warnings)


def test_leading_zero_numbers_read_digit_by_digit_with_warning(norm):
    res = norm.normalize("007")
    assert words(res) == ["zero", "zero", "seven"]
    assert any("digit_by_digit" in w for w in res.warnings)


def test_number_expansion_can_be_disabled_and_is_then_flagged(norm):
    res = TextNormalizer(NormalizationConfig(numbers="flag", ordinals="flag")).normalize("5 and 3rd")
    assert [t.supported for t in res.tokens] == [False, True, False]
    assert res.tokens[0].unsupported_reason == "number_expansion_disabled"


def test_abbreviation_with_period_consumes_period_without_double_counting(norm):
    res = norm.normalize("Dr. Smith")
    assert words(res) == ["doctor", "smith"]
    assert res.tokens[0].original_text == "Dr." and res.dropped == []
    res.verify()


def test_abbreviation_expanding_to_two_words(norm):
    res = norm.normalize("books, pens etc. and more")
    assert words(res)[2:5] == ["et", "cetera", "and"]


def test_symbols_mapped_or_flagged(norm):
    res = norm.normalize("salt & pepper 50%")
    assert words(res) == ["salt", "and", "pepper", "fifty", "percent"]
    res2 = norm.normalize("cost $5")
    assert [(t.original_text, t.supported) for t in res2.tokens] == [("cost", True), ("$", False), ("5", True)]
    assert res2.tokens[1].unsupported_reason == "unsupported_symbol"


@pytest.mark.parametrize("text,reason", [
    ("www.example.com", "url_not_supported"),
    ("https://x.org/a?b=1", "url_not_supported"),
    ("me@site.org", "email_not_supported"),
    ("3D", "mixed_letters_and_digits"),
    ("COVID19", "mixed_letters_and_digits"),
    ("U.S.A.", "dotted_acronym_reading_ambiguous"),
])
def test_unsupported_tokens_stay_in_stream_flagged(norm, text, reason):
    res = norm.normalize(text)
    assert len(res.tokens) == 1
    tok = res.tokens[0]
    assert (tok.supported, tok.unsupported_reason, tok.original_text) == (False, reason, text)
    assert res.normalized_text == ""  # excluded from the normalised text...
    assert res.unsupported_tokens == [tok]  # ...but never silently deleted


def test_emoji_is_unsupported_not_dropped(norm):
    res = norm.normalize("hi \U0001F600")
    assert [t.supported for t in res.tokens] == [True, False]


def test_zero_width_characters_are_recorded_as_dropped(norm):
    res = norm.normalize("a\u200bb c")
    assert [d.kind for d in res.dropped] == ["format_char"]
    assert words(res) == ["a", "b", "c"]


def test_non_string_input_rejected(norm):
    with pytest.raises(NormalizationError):
        norm.normalize(None)  # type: ignore[arg-type]


@pytest.mark.parametrize("text", [
    "I think this is important.", "Dr. Smith paid 21 dollars for the 3rd item (it\u2019s well-known).",
    "www.x.com & 3D $5 U.S.A. 1999 007 1,000 3.14", "'tis the season... isn't it?!", "\u00e9l caf\u00e9 na\u00efve",
    "a-b-c d--e", "tab\tseparated\nlines", "50%+ off @ noon = great", "\u201csmart quotes\u201d \u2018single\u2019",
])
def test_traceability_invariant_on_varied_inputs(norm, text):
    """Every token maps back to its original span and every non-space char is accounted for."""
    res = norm.normalize(text)
    res.verify()
    for t in res.tokens:
        assert text[t.span_start:t.span_end] == t.original_text


def test_verify_detects_a_lost_character(norm):
    res = norm.normalize("hello world")
    res.tokens.pop()  # simulate silent deletion
    with pytest.raises(NormalizationError, match="neither tokenised nor recorded"):
        res.verify()


def test_config_is_applied_lowercase_off():
    res = TextNormalizer(NormalizationConfig(lowercase=False)).normalize("Hello")
    assert res.tokens[0].normalized == "Hello"


def test_custom_abbreviation_table():
    res = TextNormalizer(NormalizationConfig(abbreviations={"st": "street"})).normalize("Main St. here")
    assert [t.normalized for t in res.tokens] == ["main", "street", "here"]


def test_audit_table_has_one_row_per_token(norm):
    res = norm.normalize("21 cats")
    rows = res.audit()
    assert len(rows) == len(res.tokens) == 3
    assert rows[0]["expansion"] == "1/2" and rows[0]["original"] == "21"
