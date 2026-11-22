"""M2.6 TextGrid parsing and validation."""

import json
import random

import pytest

from asa_m2.alignment import (
    has_errors,
    interval_role,
    parse_textgrid,
    select_tier,
    validate_textgrid,
    write_word_textgrid,
)
from asa_m2.config import QCConfig
from asa_m2.exceptions import AlignmentError, OutputExistsError, TextGridError

from ..conftest import aligned_tiers  # noqa: TID252

QC = QCConfig()


def codes(issues, severity=None):
    return sorted(i.code for i in issues if severity is None or i.severity == severity)


@pytest.fixture(params=["long", "short"])
def fmt(request):
    return request.param


# ---- valid files ---------------------------------------------------------------------
def test_valid_word_tier(write_tg, fmt):
    p = write_tg("w.TextGrid", {"words": [(0, 0.5, ""), (0.5, 1.2, "hello"), (1.2, 2.0, "")]}, short=fmt == "short")
    tg = parse_textgrid(p)
    t = tg.get_tier("words")
    assert tg.text_format == fmt and (tg.xmin, tg.xmax) == (0, 2.0) and t.tier_type == "IntervalTier"
    assert [(i.start, i.end, i.label) for i in t.intervals] == [(0, 0.5, ""), (0.5, 1.2, "hello"), (1.2, 2.0, "")]
    assert not has_errors(validate_textgrid(tg, QC))


def test_valid_phone_tier_with_word_tier(write_tg, fmt):
    tiers, dur = aligned_tiers([("this", ["DH", "IH1", "S"]), ("is", ["IH1", "Z"])])
    tg = parse_textgrid(write_tg("a.TextGrid", tiers, dur, short=fmt == "short"))
    assert tg.tier_names == ["words", "phones"]
    assert [i.label for i in tg.get_tier("phones").intervals if i.label] == ["DH", "IH1", "S", "IH1", "Z"]
    assert codes(validate_textgrid(tg, QC), "error") == []


def test_labels_preserved_exactly_including_quotes_unicode_and_whitespace(write_tg, fmt):
    labels = ['say "hi"', "caf\u00e9 \u00fc\u00f1\u00ed", " padded ", "a b", "AY1"]
    ivs = [(i, i + 1, lab) for i, lab in enumerate(labels)]
    tg = parse_textgrid(write_tg("l.TextGrid", {"t": ivs}, short=fmt == "short"))
    assert [i.label for i in tg.get_tier("t").intervals] == labels


def test_multiline_label_in_long_format(tmp_path):
    p = tmp_path / "m.TextGrid"
    p.write_text('File type = "ooTextFile"\nObject class = "TextGrid"\n\nxmin = 0\nxmax = 1\ntiers? <exists>\nsize = 1\nitem []:\n'
                 '    item [1]:\n        class = "IntervalTier"\n        name = "t"\n        xmin = 0\n        xmax = 1\n'
                 '        intervals: size = 1\n        intervals [1]:\n            xmin = 0\n            xmax = 1\n'
                 '            text = "line one\nline two"\n', encoding="utf-8")
    assert parse_textgrid(p).get_tier("t").intervals[0].label == "line one\nline two"


def test_label_containing_keywords_is_not_misparsed(write_tg):
    p = write_tg("k.TextGrid", {"t": [(0, 1, 'xmin = 5 text = "x"'), (1, 2, "intervals [3]:")]})
    ivs = parse_textgrid(p).get_tier("t").intervals
    assert [i.label for i in ivs] == ['xmin = 5 text = "x"', "intervals [3]:"] and (ivs[1].start, ivs[1].end) == (1, 2)


def test_utf16_and_utf8_bom_files(write_tg, tmp_path):
    p = write_tg("u.TextGrid", {"t": [(0, 1, "x\u00e9")]})
    text = p.read_text(encoding="utf-8")
    p16 = tmp_path / "u16.TextGrid"
    p16.write_bytes(text.encode("utf-16"))
    pbom = tmp_path / "ubom.TextGrid"
    pbom.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
    for q, enc in ((p16, "utf-16"), (pbom, "utf-8-sig")):
        tg = parse_textgrid(q)
        assert tg.encoding == enc and tg.get_tier("t").intervals[0].label == "x\u00e9"


def test_original_file_is_never_modified(write_tg):
    p = write_tg("o.TextGrid", {"t": [(0, 1, "a")]})
    before, mtime = p.read_bytes(), p.stat().st_mtime_ns
    tg = parse_textgrid(p)
    validate_textgrid(tg, QC)
    tg.to_dict()
    assert p.read_bytes() == before and p.stat().st_mtime_ns == mtime
    assert len(tg.sha256) == 64


def test_to_dict_is_json_exportable_and_lossless(write_tg):
    p = write_tg("j.TextGrid", {"words": [(0, 0.5, ""), (0.5, 1, "hi")]})
    d = json.loads(json.dumps(parse_textgrid(p).to_dict()))
    assert d["time_unit"] == "seconds" and d["source_file"] == "j.TextGrid"
    assert d["tiers"][0]["intervals"][1] == {"index": 1, "start": 0.5, "end": 1, "label": "hi"}


def test_point_tier_is_parsed_and_ignored_for_selection(tmp_path):
    p = tmp_path / "pt.TextGrid"
    p.write_text('File type = "ooTextFile"\nObject class = "TextGrid"\n\n0\n2\n<exists>\n2\n"TextTier"\n"events"\n0\n2\n1\n0.5\n"beep"\n'
                 '"IntervalTier"\n"words"\n0\n2\n1\n0\n2\n"hi"\n', encoding="utf-8")
    tg = parse_textgrid(p)
    assert tg.get_tier("events").points[0].mark == "beep" and not tg.get_tier("events").is_interval_tier
    assert select_tier(tg, ["words"]).name == "words"
    assert codes(validate_textgrid(tg, QC), "error") == []


# ---- empty intervals & silence labels ------------------------------------------------------
def test_empty_intervals_are_classified_explicitly_as_silence():
    sil, oov = {"", "sil", "sp", "<eps>"}, {"spn", "<unk>"}
    assert interval_role("", sil, oov, "word") == "silence"
    assert interval_role("   ", sil, oov, "word") == "silence"
    assert interval_role("sil", sil, oov, "phone") == "silence"
    assert interval_role("<eps>", sil, oov, "word") == "silence"
    assert interval_role("spn", sil, oov, "phone") == "oov_marker"
    assert interval_role("<unk>", sil, oov, "word") == "oov_marker"
    assert interval_role("hello", sil, oov, "word") == "word" and interval_role("AY1", sil, oov, "phone") == "phone"


def test_empty_tier_is_warned(write_tg):
    p = write_tg("e.TextGrid", {"words": []}, 1.0)
    assert codes(validate_textgrid(parse_textgrid(p), QC)) == ["empty_tier"]


# ---- invalid intervals ------------------------------------------------------------------------
def test_negative_duration_detected(write_tg):
    p = write_tg("n.TextGrid", {"w": [(0, 0.5, ""), (1.0, 0.5, "bad"), (1.0, 2.0, "")]}, 2.0)
    issues = validate_textgrid(parse_textgrid(p), QC)
    bad = [i for i in issues if i.code == "negative_duration"]
    assert len(bad) == 1 and bad[0].interval_index == 1 and bad[0].details["label"] == "bad"


def test_zero_duration_detected(write_tg):
    p = write_tg("z.TextGrid", {"w": [(0, 0.5, "a"), (0.5, 0.5, "z"), (0.5, 2.0, "b")]}, 2.0)
    assert "zero_duration" in codes(validate_textgrid(parse_textgrid(p), QC), "error")


def test_negative_start_detected(write_tg):
    p = write_tg("s.TextGrid", {"w": [(-0.5, 1.0, "a")]}, 1.0)
    assert "negative_start" in codes(validate_textgrid(parse_textgrid(p), QC), "error")


def test_overlapping_intervals_detected(write_tg):
    p = write_tg("ov.TextGrid", {"w": [(0, 1.0, "a"), (0.8, 1.5, "b"), (1.5, 2.0, "")]}, 2.0)
    ov = [i for i in validate_textgrid(parse_textgrid(p), QC) if i.code == "overlap"]
    assert len(ov) == 1 and ov[0].interval_index == 1 and ov[0].details["previous_index"] == 0
    assert "0.2" in ov[0].message


def test_unsorted_intervals_detected(write_tg):
    p = write_tg("us.TextGrid", {"w": [(1.0, 2.0, "b"), (0, 1.0, "a")]}, 2.0)
    assert "unsorted_intervals" in codes(validate_textgrid(parse_textgrid(p), QC), "error")


def test_gap_is_a_warning_not_an_error(write_tg):
    p = write_tg("g.TextGrid", {"w": [(0, 0.5, "a"), (0.7, 2.0, "b")]}, 2.0)
    issues = validate_textgrid(parse_textgrid(p), QC)
    assert "gap" in codes(issues, "warning") and not has_errors(issues)


def test_tier_range_problems(write_tg):
    p = write_tg("r.TextGrid", {"w": [(0.2, 1.5, "a")]}, 1.0)
    c = codes(validate_textgrid(parse_textgrid(p), QC))
    assert "starts_after_tier_xmin" in c and "exceeds_tier_xmax" in c
    p2 = write_tg("r2.TextGrid", {"w": [(0, 0.5, "a")]}, 1.0)
    assert "ends_before_tier_xmax" in codes(validate_textgrid(parse_textgrid(p2), QC), "warning")


def test_duration_mismatch_with_audio_and_unit_hint(write_tg):
    p = write_tg("d.TextGrid", {"w": [(0, 2.0, "a")]}, 2.0)
    tg = parse_textgrid(p)
    assert codes(validate_textgrid(tg, QC, audio_duration_s=2.01)) == []  # inside tolerance
    mism = [i for i in validate_textgrid(tg, QC, audio_duration_s=3.5) if i.code == "duration_mismatch"]
    assert len(mism) == 1 and "milliseconds" not in mism[0].message
    ms = write_tg("ms.TextGrid", {"w": [(0, 2000.0, "a")]}, 2000.0)
    hint = [i for i in validate_textgrid(parse_textgrid(ms), QC, audio_duration_s=2.0) if i.code == "duration_mismatch"]
    assert hint and "milliseconds" in hint[0].message  # consistent-unit check


def test_validation_thresholds_come_from_config(write_tg):
    p = write_tg("t.TextGrid", {"w": [(0, 0.5, "a"), (0.52, 1.0, "b")]}, 1.0)
    tg = parse_textgrid(p)
    assert "gap" in codes(validate_textgrid(tg, QCConfig(gap_tolerance_s=0.001)))
    assert "gap" not in codes(validate_textgrid(tg, QCConfig(gap_tolerance_s=0.05)))


def test_declared_size_mismatch_is_reported(tmp_path):
    p = tmp_path / "sz.TextGrid"
    p.write_text('File type = "ooTextFile"\nObject class = "TextGrid"\n\nxmin = 0\nxmax = 1\ntiers? <exists>\nsize = 1\nitem []:\n'
                 '    item [1]:\n        class = "IntervalTier"\n        name = "t"\n        xmin = 0\n        xmax = 1\n'
                 '        intervals: size = 3\n        intervals [1]:\n            xmin = 0\n            xmax = 1\n            text = "a"\n',
                 encoding="utf-8")
    assert "declared_size_mismatch" in codes(parse_textgrid(p).issues)


# ---- malformed / missing ---------------------------------------------------------------------
@pytest.mark.parametrize("content,fragment", [
    (b"", "empty"), (b"   \n\n", "empty"), (b"garbage not a textgrid", "not a Praat TextGrid"),
    (b"\xff\xfe\x00", "UTF-16"),
    (b'File type = "ooTextFile"\nObject class = "TextGrid"\n\nxmin = 0\nxmax = 1\n', "no tiers"),
    (b'File type = "ooTextFile"\nObject class = "TextGrid"\n\n0\n1\n<exists>\n1\n"IntervalTier"\n"t"\n0\n1\n2\n0\n0.5\n"a"\n', "unexpected end"),
    (b'File type = "ooTextFile"\nObject class = "TextGrid"\n\n0\nabc\n<exists>\n1\n', "not a number"),
    (b'File type = "ooTextFile"\nObject class = "TextGrid"\n\n0\n1\n<exists>\n1\n"WeirdTier"\n"t"\n0\n1\n1\n0\n1\n"a"\n', "unknown tier class"),
    (b'File type = "ooTextFile"\nObject class = "TextGrid"\n\n0\n1\n<exists>\n0\n', "0 tiers"),
])
def test_malformed_files_raise_clear_errors(tmp_path, content, fragment):
    p = tmp_path / "bad.TextGrid"
    p.write_bytes(content)
    with pytest.raises(TextGridError, match=fragment):
        parse_textgrid(p)


def test_incomplete_interval_in_long_format(tmp_path):
    p = tmp_path / "inc.TextGrid"
    p.write_text('File type = "ooTextFile"\nObject class = "TextGrid"\n\nxmin = 0\nxmax = 1\ntiers? <exists>\nsize = 1\nitem []:\n'
                 '    item [1]:\n        class = "IntervalTier"\n        name = "t"\n        xmin = 0\n        xmax = 1\n'
                 '        intervals: size = 1\n        intervals [1]:\n            xmin = 0\n            xmax = 1\n', encoding="utf-8")
    with pytest.raises(TextGridError, match="incomplete interval 1"):
        parse_textgrid(p)


def test_missing_file_and_directory(tmp_path):
    with pytest.raises(TextGridError, match="not found"):
        parse_textgrid(tmp_path / "nope.TextGrid")
    with pytest.raises(TextGridError, match="not found"):
        parse_textgrid(tmp_path)


def test_undecodable_bytes(tmp_path):
    p = tmp_path / "latin.TextGrid"
    p.write_bytes(b'File type = "ooTextFile"\nObject class = "TextGrid"\n\xe9\xe9\n')
    with pytest.raises(TextGridError, match="decode"):
        parse_textgrid(p)


# ---- tier selection ------------------------------------------------------------------------------
def make(write_tg, names, short=False):
    return parse_textgrid(write_tg("x.TextGrid", {n: [(0, 1, "a")] for n in names}, 1.0, short=short))


def test_select_tier_default_candidate_names(write_tg):
    tg = make(write_tg, ["words", "phones"])
    assert select_tier(tg, ["words"], kind="word").name == "words"
    assert select_tier(tg, ["phones", "phone"], kind="phone").name == "phones"


def test_select_tier_is_configurable_not_hardcoded(write_tg):
    tg = make(write_tg, ["Wort", "Laute"])
    assert select_tier(tg, ["Wort"], kind="word").name == "Wort"
    with pytest.raises(TextGridError, match="no word tier found"):
        select_tier(tg, ["words"], kind="word")


def test_select_tier_case_insensitive_fallback(write_tg):
    assert select_tier(make(write_tg, ["Words"]), ["words"]).name == "Words"


def test_select_tier_explicit_name_wins(write_tg):
    tg = make(write_tg, ["words", "custom"])
    assert select_tier(tg, ["words"], explicit_name="custom").name == "custom"
    with pytest.raises(TextGridError, match="requested"):
        select_tier(tg, ["words"], explicit_name="absent")


def test_missing_tier_error_lists_available_tiers_and_hints_inspect(write_tg):
    with pytest.raises(TextGridError) as exc:
        select_tier(make(write_tg, ["alpha", "beta"]), ["words"], kind="word")
    msg = str(exc.value)
    assert "alpha" in msg and "beta" in msg and "inspect-textgrid" in msg


def test_single_speaker_prefixed_tier_is_accepted(write_tg):
    tg = make(write_tg, ["speaker1 - words", "speaker1 - phones"])
    assert select_tier(tg, ["words"]).name == "speaker1 - words"
    assert select_tier(tg, ["phones"]).name == "speaker1 - phones"
    with pytest.raises(TextGridError):
        select_tier(tg, ["words"], allow_speaker_prefix=False)


def test_several_speakers_require_explicit_choice(write_tg):
    tg = make(write_tg, ["a - words", "b - words"])
    with pytest.raises(TextGridError, match="several speaker-prefixed"):
        select_tier(tg, ["words"], kind="word")
    assert select_tier(tg, ["words"], explicit_name="b - words").name == "b - words"


def test_point_tier_never_selected(tmp_path):
    p = tmp_path / "pt.TextGrid"
    p.write_text('File type = "ooTextFile"\nObject class = "TextGrid"\n\n0\n2\n<exists>\n1\n"TextTier"\n"words"\n0\n2\n0\n', encoding="utf-8")
    with pytest.raises(TextGridError, match="no word tier"):
        select_tier(parse_textgrid(p), ["words"], kind="word")


# ---- differential test against praatio (independent oracle) ---------------------------------------
def test_parser_agrees_with_praatio_on_random_valid_textgrids(tmp_path):
    from praatio import textgrid as pt
    from praatio.utilities.constants import Interval

    rng = random.Random(1234)
    for n in range(25):
        t, ivs = 0.0, []
        for _ in range(rng.randint(1, 12)):
            d = round(rng.uniform(0.01, 0.8), 4)
            ivs.append((round(t, 4), round(t + d, 4), rng.choice(["", "AY1", "th", 'q"uote', "w x", "caf\u00e9", "<unk>"])))
            t += d
        xmax = round(t + 0.1, 4)
        ivs.append((round(t, 4), xmax, ""))
        tier = pt.IntervalTier("words", [Interval(*i) for i in ivs], 0.0, xmax)
        tg = pt.Textgrid()
        tg.addTier(tier)
        for kind in ("long_textgrid", "short_textgrid"):
            path = tmp_path / f"r{n}_{kind}.TextGrid"
            tg.save(str(path), format=kind, includeBlankSpaces=True)
            mine = parse_textgrid(path)
            theirs = pt.openTextgrid(str(path), includeEmptyIntervals=True).getTier("words").entries
            assert [(i.start, i.end, i.label) for i in mine.get_tier("words").intervals] == \
                   [(e.start, e.end, e.label) for e in theirs], (n, kind)
            assert (mine.xmin, mine.xmax) == (0.0, xmax)


def test_write_word_textgrid_roundtrips_through_parser(tmp_path):
    p = write_word_textgrid(tmp_path / "q.TextGrid", [(0.1, 0.5, "hello"), (0.5, 0.9, "world")], 1.2)
    tg = parse_textgrid(p)
    assert [(i.start, i.end, i.label) for i in tg.get_tier("words").intervals] == \
           [(0.0, 0.1, ""), (0.1, 0.5, "hello"), (0.5, 0.9, "world"), (0.9, 1.2, "")]
    with pytest.raises(OutputExistsError):
        write_word_textgrid(p, [(0, 1, "x")], 1.0)  # no silent overwrite
    with pytest.raises(AlignmentError, match="without intervals"):
        write_word_textgrid(tmp_path / "e.TextGrid", [], 1.0)
