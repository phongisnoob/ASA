"""ASA Module M2 - Phonetic Front-end.

M2 produces canonical phonetic representations, pronunciation dictionaries and forced
alignment *target* intervals for downstream modules (M3 acoustic modelling, M4 MDD).
It deliberately does **not** recognise realised phones, compute GOP, detect
mispronunciations or score speech.
"""

__version__ = "1.0.0"
SCHEMA_VERSION = "1.0"

__all__ = ["SCHEMA_VERSION", "__version__"]
