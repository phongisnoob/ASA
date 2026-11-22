"""Grapheme-to-phoneme: CMUdict primary, optional eSpeak-ng fallback, custom lexicon."""

from asa_m2.g2p.cmudict_backend import CMUdictBackend
from asa_m2.g2p.custom_lexicon import CustomLexicon
from asa_m2.g2p.engine import G2PEngine, ensure_alignable
from asa_m2.g2p.espeak_backend import EspeakBackend

__all__ = ["CMUdictBackend", "CustomLexicon", "EspeakBackend", "G2PEngine", "ensure_alignable"]
