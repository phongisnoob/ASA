"""M2.3 phone inventory and mapping validation."""

import copy
import json

import pytest

from asa_m2.config import resources_dir
from asa_m2.exceptions import InventoryError, MappingError, UnmappedPhoneError
from asa_m2.phonemes import AlignerLabelMapper, LabelRole, PhoneInventory, PhoneMapping
from asa_m2.phonemes.inventory import validate_inventory_data

RES = resources_dir()


def load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


@pytest.fixture()
def data():
    return load("phoneme_inventory.json")


def problems(data, **kw):
    return validate_inventory_data(data, kw.get("p2i"), kw.get("i2p"))


def test_shipped_inventory_is_valid_and_complete(inventory):
    assert len(inventory) == 39
    assert sum(p.is_vowel for p in inventory.phones) == 15
    assert inventory.inventory_id == "asa-arpabet-cmu39" and inventory.inventory_version == "1.0.0"
    assert [p.id for p in inventory.phones] == list(range(1, 40))  # stable alphabetical IDs
    assert inventory.phone_id("AA") == 1 and inventory.phone_id("ZH") == 39


def test_all_four_files_are_consistent():
    inv, p2i, i2p, mp = (load(n) for n in ("phoneme_inventory.json", "phone_to_id.json", "id_to_phone.json", "phoneme_mapping.json"))
    assert validate_inventory_data(inv, p2i, i2p) == []
    for f in (p2i, i2p, mp):
        assert (f["inventory_id"], f["inventory_version"]) == (inv["inventory_id"], inv["inventory_version"])


def test_duplicate_ids_detected(data):
    data["phones"][1]["id"] = data["phones"][0]["id"]
    assert any("duplicate phone id" in p for p in problems(data))


def test_duplicate_symbols_detected(data):
    data["phones"][1]["arpabet"] = data["phones"][0]["arpabet"]
    assert any("duplicate arpabet symbol" in p for p in problems(data))


def test_duplicate_ipa_detected(data):
    data["phones"][1]["ipa"] = data["phones"][0]["ipa"]
    assert any("duplicate ipa label" in p for p in problems(data))


def test_stress_digit_in_base_label_rejected(data):
    data["phones"][0]["arpabet"] = "AA1"
    assert any("stress digits belong in the stress field" in p for p in problems(data))


def test_stress_must_be_separate_from_identity(data):
    data["stress"]["separate_from_phone_identity"] = False
    assert any("separate_from_phone_identity" in p for p in problems(data))


def test_consonant_cannot_be_stress_bearing(data):
    cons = next(p for p in data["phones"] if p["category"] == "consonant")
    cons["stress_bearing"] = True
    assert any("cannot be stress_bearing" in p for p in problems(data))


def test_invalid_category_and_negative_id(data):
    data["phones"][0]["category"] = "glide"
    data["phones"][1]["id"] = -4
    out = problems(data)
    assert any("category" in p for p in out) and any("non-negative" in p for p in out)


def test_ctc_blank_is_not_a_phoneme_and_not_silence(inventory, data):
    assert inventory.ctc_blank_id == 0 and inventory.ctc_blank_symbol == "<blank>"
    assert "<blank>" not in inventory and 0 not in inventory.id_to_phone
    assert all(p.arpabet not in {"sil", "SIL", "sp"} for p in inventory.phones)  # no invented silence class
    assert data["silence_policy"]["has_silence_class"] is False
    data["special_symbols"][0]["is_phoneme"] = True
    assert any("is_phoneme must be false" in p for p in problems(data))


def test_ctc_blank_must_not_be_declared_silence(data):
    data["special_symbols"][0]["is_silence"] = True
    out = problems(data)
    assert any("must not be declared as silence" in p for p in out)


def test_blank_id_cannot_collide_with_phone(data):
    data["special_symbols"][0]["id"] = 5
    assert any("collides with phone" in p for p in problems(data))


def test_exactly_one_blank_required(data):
    data["special_symbols"] = []
    assert any("ctc_blank" in p for p in problems(data))


def test_silence_policy_consistency(data):
    data["silence_policy"]["has_silence_class"] = True
    assert any("no silence special symbol" in p for p in problems(data))


def test_companion_files_must_match():
    inv, p2i, i2p = (load(n) for n in ("phoneme_inventory.json", "phone_to_id.json", "id_to_phone.json"))
    bad = copy.deepcopy(p2i)
    bad["phone_to_id"]["AA"] = 99
    assert any("phone_to_id does not match" in p for p in validate_inventory_data(inv, bad, i2p))
    bad2 = copy.deepcopy(i2p)
    bad2["id_to_phone"]["1"] = "ZZ"
    assert any("id_to_phone does not match" in p for p in validate_inventory_data(inv, p2i, bad2))
    bad3 = copy.deepcopy(p2i)
    bad3["inventory_version"] = "9.9.9"
    assert any("inventory_version" in p for p in validate_inventory_data(inv, bad3, i2p))
    bad4 = copy.deepcopy(p2i)
    bad4["phone_to_id"]["<blank>"] = 0
    out = validate_inventory_data(inv, bad4, i2p)
    assert any("does not match" in p or "blank" in p for p in out)


def test_invalid_inventory_raises_with_all_problems(data):
    data["phones"][1]["id"] = data["phones"][0]["id"]
    data["phones"][2]["arpabet"] = data["phones"][3]["arpabet"]
    with pytest.raises(InventoryError) as exc:
        PhoneInventory(data)
    assert len(exc.value.details["problems"]) >= 2


@pytest.mark.parametrize("sym,base,stress", [("AY1", "AY", 1), ("IH0", "IH", 0), ("ER2", "ER", 2), ("TH", "TH", None), ("AH", "AH", None)])
def test_parse_symbol_separates_stress(inventory, sym, base, stress):
    parsed = inventory.parse_symbol(sym)
    assert (parsed.base, parsed.stress, parsed.raw) == (base, stress, sym)


@pytest.mark.parametrize("bad", ["XX", "AY3", "T1", "ay1", "", "AY1X", "1AY", "Æ"])
def test_parse_symbol_rejects_unknown_or_illegal(inventory, bad):
    with pytest.raises(InventoryError):
        inventory.parse_symbol(bad)


def test_encode_decode_roundtrip_and_unknown_rejected(inventory):
    ids = inventory.encode(["DH", "IH", "S"])
    assert inventory.decode(ids) == ["DH", "IH", "S"]
    with pytest.raises(InventoryError, match="unknown phone"):
        inventory.encode(["DH", "XYZ"])  # never skipped, never remapped
    with pytest.raises(InventoryError):
        inventory.phone_from_id(0)  # blank is not a phone
    with pytest.raises(InventoryError):
        inventory.phone_from_id(40)


def test_ipa_is_display_only_and_not_confused_with_arpabet(inventory):
    assert inventory.ipa("TH") == "θ" and inventory.ipa("AY") == "aɪ"
    with pytest.raises(InventoryError):
        inventory.parse_symbol("θ")  # IPA is never accepted where ARPABET is expected
    with pytest.raises(InventoryError):
        inventory.get("aɪ")


def test_compatibility_report(inventory):
    other = [f"{p}{1 if inventory.is_vowel(p) else ''}" for p in inventory.base_symbols if p != "ZH"] + ["spn", "sil"]
    rep = inventory.check_compatibility(other, other_name="model-x")
    assert rep.missing_in_other == ("ZH",) and set(rep.extra_in_other) == {"spn", "sil"}
    assert not rep.compatible
    assert inventory.check_compatibility(inventory.base_symbols).compatible


def test_from_files_missing_and_bad_json(tmp_path):
    with pytest.raises(InventoryError, match="not found"):
        PhoneInventory.from_files(tmp_path / "nope.json")
    bad = tmp_path / "phoneme_inventory.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(InventoryError, match="invalid JSON"):
        PhoneInventory.from_files(bad)


# ---- mappings ----------------------------------------------------------------------
@pytest.fixture()
def espeak_map(inventory):
    return PhoneMapping.from_file(RES / "phoneme_mapping.json", "espeak_en_us_ipa", inventory)


def test_mapping_targets_all_exist_in_inventory(espeak_map, inventory):
    for unit in espeak_map.units:
        assert all(p in inventory for p in espeak_map.map_unit(unit).arpabet)


def test_mapping_exact_and_approximate(espeak_map):
    assert espeak_map.map_unit("θ").arpabet == ("TH",) and espeak_map.map_unit("θ").confidence == "exact"
    assert espeak_map.map_unit("ᵻ").confidence == "approximate"
    assert espeak_map.map_unit("oːɹ").arpabet == ("AO", "R")


def test_unmapped_unit_raises_never_dropped(espeak_map):
    with pytest.raises(UnmappedPhoneError) as exc:
        espeak_map.map_units(["θ", "ɾ", "ʔ", "qq"])
    assert exc.value.details["units"] == ["ɾ", "ʔ", "qq"]  # all offending units listed
    assert "ɾ" in exc.value.details["reasons"] and "flap" in exc.value.details["reasons"]["ɾ"]


def test_deliberately_unmapped_units_are_not_in_table(espeak_map):
    for unit in ("ɾ", "ʔ", "x", "ɬ", "r"):
        assert unit not in espeak_map and unit in espeak_map.unmapped_by_design


def _spec(entries, **extra):
    return {"entries": entries, **extra}


def test_mapping_rejects_unknown_target_phone(inventory):
    with pytest.raises(MappingError, match="not in inventory"):
        PhoneMapping("m", _spec({"a": {"arpabet": ["QQ"], "confidence": "exact"}}), inventory)


def test_mapping_rejects_bad_confidence_empty_and_overlap(inventory):
    with pytest.raises(MappingError, match="confidence"):
        PhoneMapping("m", _spec({"a": {"arpabet": ["AA"], "confidence": "maybe"}}), inventory)
    with pytest.raises(MappingError, match="non-empty list"):
        PhoneMapping("m", _spec({"a": {"arpabet": [], "confidence": "exact"}}), inventory)
    with pytest.raises(MappingError, match="both mapped and"):
        PhoneMapping("m", _spec({"a": {"arpabet": ["AA"], "confidence": "exact"}}, unmapped_by_design={"a": "x"}), inventory)
    with pytest.raises(MappingError, match="non-empty"):
        PhoneMapping("m", {"entries": {}}, inventory)


def test_mapping_file_version_mismatch_refused(tmp_path, inventory):
    data = load("phoneme_mapping.json")
    data["inventory_version"] = "0.0.1"
    p = tmp_path / "m.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(MappingError, match="mismatched versions"):
        PhoneMapping.from_file(p, "espeak_en_us_ipa", inventory)
    with pytest.raises(MappingError, match="not found in"):
        PhoneMapping.from_file(RES / "phoneme_mapping.json", "nope", inventory)
    with pytest.raises(MappingError, match="mapping file not found"):
        PhoneMapping.from_file(tmp_path / "absent.json", "x", inventory)


# ---- aligner label classification ---------------------------------------------------
@pytest.mark.parametrize("label,role,base,stress", [
    ("AY1", LabelRole.PHONE, "AY", 1), ("T", LabelRole.PHONE, "T", None), ("", LabelRole.SILENCE, None, None),
    ("sil", LabelRole.SILENCE, None, None), ("sp", LabelRole.SILENCE, None, None), ("<eps>", LabelRole.SILENCE, None, None),
    ("spn", LabelRole.OOV, None, None), ("<unk>", LabelRole.OOV, None, None), ("  ", LabelRole.SILENCE, None, None),
])
def test_label_classification(mapper, label, role, base, stress):
    c = mapper.classify(label)
    assert (c.role, c.base_phone, c.stress) == (role, base, stress)


@pytest.mark.parametrize("label", ["ay1", "AY9", "XX", "ʃ", "T2"])
def test_unknown_labels_reported_not_remapped(mapper, label):
    c = mapper.classify(label)
    assert c.role is LabelRole.UNKNOWN and c.base_phone is None and c.reason


def test_ctc_blank_is_not_classified_as_silence_or_phone(mapper):
    c = mapper.classify("<blank>")
    assert c.role is LabelRole.UNKNOWN


def test_label_mapper_rejects_overlapping_roles(inventory):
    with pytest.raises(MappingError):
        AlignerLabelMapper(inventory, ["sil"], ["sil"])
