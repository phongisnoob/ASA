#!/usr/bin/env python
"""Copy the user-editable configs/ files into the packaged resources directory.

`configs/` is the canonical, editable location; `src/asa_m2/resources/` is the copy that
ships inside the wheel. tests/unit/test_config.py fails if they drift apart.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["default.yaml", "g2p.yaml", "alignment.yaml", "audio.yaml", "phoneme_inventory.json",
         "phone_to_id.json", "id_to_phone.json", "phoneme_mapping.json"]

if __name__ == "__main__":
    for name in NAMES:
        shutil.copyfile(ROOT / "configs" / name, ROOT / "src" / "asa_m2" / "resources" / name)
    print(f"synced {len(NAMES)} files")
