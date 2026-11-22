"""Provenance record construction (what produced an output, from which inputs)."""

from __future__ import annotations

import platform
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from asa_m2 import SCHEMA_VERSION, __version__
from asa_m2.config import AppConfig
from asa_m2.phonemes.inventory import PhoneInventory
from asa_m2.schemas import Provenance


def build_provenance(
    config: AppConfig,
    inventory: PhoneInventory,
    *,
    g2p_sources: Mapping[str, str | None] | None = None,
    inputs: Mapping[str, Any] | None = None,
) -> Provenance | None:
    """Return a provenance object, or ``None`` when ``output.include_provenance`` is false.

    Absolute paths are deliberately *not* recorded (they can leak user names); inputs are
    identified by base name + SHA-256 supplied by the caller.
    """
    if not config.output.include_provenance:
        return None
    created = (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        if config.output.include_timestamps
        else None
    )
    return Provenance(
        asa_m2_version=__version__,
        schema_version=SCHEMA_VERSION,
        created_at=created,
        config_fingerprint=config.fingerprint(),
        inventory={"id": inventory.inventory_id, "version": inventory.inventory_version},
        g2p_sources=dict(g2p_sources or {}),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        inputs=dict(inputs or {}),
    )
