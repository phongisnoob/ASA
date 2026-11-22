"""Shared fixtures. Everything is synthetic: no private audio or datasets are needed."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asa_m2.config import AppConfig, load_config
from asa_m2.g2p.engine import G2PEngine
from asa_m2.phonemes.inventory import PhoneInventory
from asa_m2.phonemes.mapping import AlignerLabelMapper

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def config() -> AppConfig:
    return load_config()


@pytest.fixture()
def inventory(config: AppConfig) -> PhoneInventory:
    return PhoneInventory.from_files(config.phonemes.inventory_path)


@pytest.fixture()
def mapper(config: AppConfig, inventory: PhoneInventory) -> AlignerLabelMapper:
    return AlignerLabelMapper.from_mapping_file(config.phonemes.mapping_path, inventory)


@pytest.fixture()
def engine(config: AppConfig, inventory: PhoneInventory) -> G2PEngine:
    return G2PEngine.from_config(config, inventory)


# ---- synthetic audio ---------------------------------------------------------------
def make_wav(path: Path, *, seconds: float = 1.0, sr: int = 16000, channels: int = 1,
             freq: float = 440.0, amp: float = 0.4, subtype: str = "PCM_16") -> Path:
    t = np.arange(int(seconds * sr)) / sr
    mono = amp * np.sin(2 * np.pi * freq * t)
    data = mono if channels == 1 else np.stack([mono * (1.0 if c == 0 else 0.5) for c in range(channels)], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr, subtype=subtype)
    return path


@pytest.fixture()
def wav_factory():
    return make_wav


# ---- synthetic TextGrids -----------------------------------------------------------
def long_textgrid(tiers: dict[str, list[tuple[float, float, str]]], xmax: float | None = None) -> str:
    xmax = xmax if xmax is not None else max((iv[1] for ivs in tiers.values() for iv in ivs), default=1.0)
    out = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "", "xmin = 0", f"xmax = {xmax}",
           "tiers? <exists>", f"size = {len(tiers)}", "item []:"]
    for i, (name, ivs) in enumerate(tiers.items(), 1):
        out += [f"    item [{i}]:", '        class = "IntervalTier"', f'        name = "{name}"', "        xmin = 0",
                f"        xmax = {xmax}", f"        intervals: size = {len(ivs)}"]
        for j, (a, b, lab) in enumerate(ivs, 1):
            lab = lab.replace('"', '""')
            out += [f"        intervals [{j}]:", f"            xmin = {a}", f"            xmax = {b}", f'            text = "{lab}"']
    return "\n".join(out) + "\n"


def short_textgrid(tiers: dict[str, list[tuple[float, float, str]]], xmax: float) -> str:
    out = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "", "0", f"{xmax}", "<exists>", f"{len(tiers)}"]
    for name, ivs in tiers.items():
        out += ['"IntervalTier"', f'"{name}"', "0", f"{xmax}", f"{len(ivs)}"]
        for a, b, lab in ivs:
            out += [f"{a}", f"{b}", '"' + lab.replace('"', '""') + '"']
    return "\n".join(out) + "\n"


@pytest.fixture()
def write_tg(tmp_path: Path):
    def _write(name: str, tiers, xmax=None, *, short: bool = False) -> Path:
        p = tmp_path / name
        text = short_textgrid(tiers, xmax if xmax is not None else max(iv[1] for ivs in tiers.values() for iv in ivs)) if short \
            else long_textgrid(tiers, xmax)
        p.write_text(text, encoding="utf-8")
        return p
    return _write


def aligned_tiers(words: list[tuple[str, list[str]]], lead: float = 0.1, ph_dur: float = 0.06, tail: float = 0.1):
    """Build consistent word+phone tiers for ``[(word, [phones...]), ...]``."""
    wiv, piv, t = [(0.0, lead, "")], [(0.0, lead, "")], lead
    for w, phones in words:
        start = t
        for ph in phones:
            piv.append((round(t, 4), round(t + ph_dur, 4), ph))
            t += ph_dur
        wiv.append((round(start, 4), round(t, 4), w))
    wiv.append((round(t, 4), round(t + tail, 4), ""))
    piv.append((round(t, 4), round(t + tail, 4), ""))
    return {"words": wiv, "phones": piv}, round(t + tail, 4)


# ---- fake MFA executable -----------------------------------------------------------
@pytest.fixture()
def fake_mfa(tmp_path: Path, monkeypatch):
    """Put a FAKE `mfa` on PATH (see tests/fixtures/fake_mfa.py). Returns the argv-log path."""
    bin_dir = tmp_path / "fake_bin"
    bin_dir.mkdir()
    exe = bin_dir / "mfa"
    exe.write_text(f"#!{sys.executable}\n" + (FIXTURES / "fake_mfa.py").read_text(encoding="utf-8").split("\n", 1)[1],
                   encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    log = tmp_path / "fake_mfa_argv.jsonl"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_MFA_ARGV_LOG", str(log))
    monkeypatch.delenv("FAKE_MFA_MODE", raising=False)
    monkeypatch.delenv("FAKE_MFA_VERSION", raising=False)
    return log


def read_argv_log(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
