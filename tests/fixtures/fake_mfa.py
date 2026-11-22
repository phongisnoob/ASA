#!/usr/bin/env python3
"""A FAKE `mfa` used only to exercise asa-m2's subprocess wrapper in tests.

This is NOT Montreal Forced Aligner. It mimics just enough of the verified CLI surface
(`mfa version`, `mfa align CORPUS DICT MODEL OUTPUT ...`, `mfa align_hf CORPUS MODEL OUTPUT ...`,
`mfa validate ...`) to produce synthetic, evenly-spaced alignments. Behaviour is switched
with environment variables:

  FAKE_MFA_MODE      ok | exit1 | missing_last | broken_install | bad_version | oov_last |
                     drop_word | corrupt_last | validate_fail | hang
  FAKE_MFA_VERSION   version string printed by `mfa version` (default 3.4.3)
  FAKE_MFA_ARGV_LOG  if set, every invocation's argv is appended there as a JSON line
"""
import json
import os
import sys
import time
from pathlib import Path

import soundfile as sf

mode = os.environ.get("FAKE_MFA_MODE", "ok")
version = os.environ.get("FAKE_MFA_VERSION", "3.4.3")
argv = sys.argv[1:]
if os.environ.get("FAKE_MFA_ARGV_LOG"):
    with open(os.environ["FAKE_MFA_ARGV_LOG"], "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"argv": argv, "MFA_ROOT_DIR": os.environ.get("MFA_ROOT_DIR")}) + "\n")


def tg_long(duration, tiers):
    out = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "", "xmin = 0", f"xmax = {duration}",
           "tiers? <exists>", f"size = {len(tiers)}", "item []:"]
    for i, (name, ivs) in enumerate(tiers, 1):
        out += [f"    item [{i}]:", '        class = "IntervalTier"', f'        name = "{name}"', "        xmin = 0",
                f"        xmax = {duration}", f"        intervals: size = {len(ivs)}"]
        for j, (a, b, lab) in enumerate(ivs, 1):
            out += [f"        intervals [{j}]:", f"            xmin = {a}", f"            xmax = {b}", f'            text = "{lab}"']
    return "\n".join(out) + "\n"


def read_dict(path):
    d = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            w, p = line.split("\t", 1)
            d.setdefault(w, p.split())  # first listed variant
    return d


if argv[:1] == ["version"]:
    if mode == "broken_install":
        print("ModuleNotFoundError: No module named '_kalpy'", file=sys.stderr)
        sys.exit(1)
    print("???" if mode == "bad_version" else version)
    sys.exit(0)

if argv[:1] == ["validate"]:
    sys.exit(1 if mode == "validate_fail" else 0)

if argv[:1] in (["align"], ["align_hf"]):
    if mode == "hang":
        time.sleep(60)
    if mode == "exit1":
        print("fake MFA: simulated failure in feature generation", file=sys.stderr)
        sys.exit(1)
    if argv[0] == "align":
        corpus, dictionary, model, output = argv[1:5]
        pron = read_dict(dictionary) if Path(dictionary).is_file() else {}  # a NAME (model store) is not a file
    else:
        corpus, model, output = argv[1:4]
        pron = {}
    wavs = sorted(Path(corpus).rglob("*.wav"))
    if mode == "missing_last" and wavs:
        wavs = wavs[:-1]
    for wav in wavs:
        words = (wav.with_suffix(".lab")).read_text(encoding="utf-8").split()
        dur = round(sf.info(str(wav)).duration, 4)
        if mode == "drop_word" and len(words) > 1:
            kept = words[:1] + words[2:]
        else:
            kept = words
        t, lead = 0.0, 0.1
        t = lead
        wiv, piv = [(0.0, lead, "")], [(0.0, lead, "")]
        for k, w in enumerate(kept):
            phones = pron.get(w, ["spn"])
            if mode == "oov_last" and k == len(kept) - 1:
                w, phones = "<unk>", ["spn"]
            start = t
            for ph in phones:
                piv.append((round(t, 4), round(t + 0.06, 4), ph))
                t += 0.06
            wiv.append((round(start, 4), round(t, 4), w))
        if t < dur:
            wiv.append((round(t, 4), dur, ""))
            piv.append((round(t, 4), dur, ""))
        else:
            dur = round(t, 4)
        out = Path(output) / wav.parent.name / (wav.stem + ".TextGrid")
        out.parent.mkdir(parents=True, exist_ok=True)
        if mode == "corrupt_last" and wav == wavs[-1]:
            out.write_text("this is not a TextGrid", encoding="utf-8")
            continue
        out.write_text(tg_long(dur, [("words", wiv), ("phones", piv)]), encoding="utf-8")
    print(f"fake MFA aligned {len(wavs)} file(s)")
    sys.exit(0)

print(f"fake mfa: unknown command {argv}", file=sys.stderr)
sys.exit(2)
