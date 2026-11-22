"""OPTIONAL adapter for Qwen3-ForcedAligner (word-level timestamps).

Status: **live integration unverified in the build environment** (no GPU, model not
downloaded). The interface below was written against the official sources:

* README: https://github.com/QwenLM/Qwen3-ASR  ("ForcedAligner Usage")
* Source: ``qwen_asr/inference/qwen3_forced_aligner.py`` (``Qwen3ForcedAligner``)

Facts taken from those sources (not invented):

* ``from qwen_asr import Qwen3ForcedAligner``
* ``Qwen3ForcedAligner.from_pretrained(name, dtype=..., device_map=...)``
* ``model.align(audio=..., text=..., language=...)`` returns one result per input; each result
  is iterable over items with ``.text``, ``.start_time``, ``.end_time`` (seconds, rounded to ms)
* audio may be a path / URL / base64 / ``(np.ndarray, sr)``; it is converted to mono 16 kHz
* English text is tokenised by whitespace; characters other than letters, digits and ``'``
  are stripped from each token (so we pre-check that our tokens survive unchanged)
* timestamps are for words (or characters for CJK) - **never phones**; supports up to ~5 min

This adapter therefore reports ``granularity="word"`` only and never produces phone
alignments. Model loading is lazy; importing this module loads nothing heavy. Network
URLs are refused as audio input (no hidden network calls from the adapter itself; the
Hugging Face download happens inside ``from_pretrained`` when you first load the model).
The newer ``*-hf`` Transformers-native checkpoints mentioned in the upstream README are
NOT supported here because their API was not inspected.
"""

from __future__ import annotations

import importlib.util
import logging
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from asa_m2.alignment.exporter import IntervalIn
from asa_m2.config import QwenConfig
from asa_m2.exceptions import AlignmentError, OptionalDependencyError, QwenResourceError

log = logging.getLogger(__name__)

OFFICIAL_URLS = (
    "https://github.com/QwenLM/Qwen3-ASR",
    "https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B",
)
LIVE_INTEGRATION_VERIFIED = False


def qwen_would_keep(token: str) -> str:
    """Replicates the English token cleaning of ``Qwen3ForceAlignProcessor.clean_token``."""
    return "".join(ch for ch in token if ch == "'" or unicodedata.category(ch)[0] in {"L", "N"})


@dataclass(frozen=True)
class QwenAlignment:
    intervals: list[IntervalIn]
    model_info: dict[str, Any]


def is_available() -> bool:
    """True when both ``qwen_asr`` and ``torch`` can be imported (nothing is imported here)."""
    return all(importlib.util.find_spec(m) is not None for m in ("qwen_asr", "torch"))


class QwenForcedAlignerAdapter:
    granularity = "word"
    aligner_name = "Qwen3-ForcedAligner"

    def __init__(self, config: QwenConfig, *, model_factory: Callable[..., Any] | None = None) -> None:
        self.config = config
        self._factory = model_factory  # injectable for tests; None => real qwen_asr
        self._model: Any = None
        self._info: dict[str, Any] = {}

    # ---- lazy loading ------------------------------------------------------------
    def load(self) -> None:
        if self._model is not None:
            return
        if self._factory is not None:
            self._model = self._factory(self.config.model_name)
            self._info = {"model_name": self.config.model_name, "device": "injected", "loader": "model_factory"}
            return
        if not is_available():
            raise OptionalDependencyError(
                "the optional Qwen adapter needs `pip install qwen-asr` (which brings torch). "
                "Core M2 does not require it. See README, 'Optional: Qwen3-ForcedAligner'."
            )
        import torch  # noqa: PLC0415 - deliberate lazy import
        from qwen_asr import Qwen3ForcedAligner  # noqa: PLC0415

        device = self.config.device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                log.warning("no CUDA device found; Qwen3-ForcedAligner will run on CPU (slow)")
        dtype = getattr(torch, self.config.dtype)
        if device == "cpu" and self.config.dtype != "float32":
            log.warning("using %s on CPU; if this fails set alignment.qwen.dtype: float32", self.config.dtype)
        try:
            self._model = Qwen3ForcedAligner.from_pretrained(self.config.model_name, dtype=dtype, device_map=device)
        except Exception as exc:  # broad on purpose: map GPU/memory/network problems to a clear error
            raise _translate_load_error(exc, device) from exc
        self._info = {
            "model_name": self.config.model_name, "device": device, "dtype": self.config.dtype,
            "qwen_asr_version": _version("qwen-asr"), "torch_version": _version("torch"),
        }

    # ---- alignment ---------------------------------------------------------------
    def align_words(self, audio_path: str | Path, words: Sequence[str]) -> QwenAlignment:
        """Align ``words`` (already normalised tokens) to the audio; word-level only."""
        if not words:
            raise AlignmentError("no words to align")
        for w in words:
            if qwen_would_keep(w) != w or not w or " " in w:
                raise AlignmentError(
                    f"token {w!r} would be altered by Qwen's tokenisation ({qwen_would_keep(w)!r}); "
                    "it cannot be mapped one-to-one, so alignment was not attempted"
                )
        text = str(audio_path)
        if "://" in text or text.startswith("data:"):
            raise AlignmentError("URL / base64 audio inputs are refused; pass a local file path")
        path = Path(audio_path)
        if not path.is_file():
            raise AlignmentError(f"audio file not found: {path}")
        samples, sr = sf.read(str(path), dtype="float32", always_2d=True)
        duration = samples.shape[0] / sr
        if duration > self.config.max_audio_seconds:
            raise AlignmentError(
                f"audio is {duration:.1f}s but the aligner supports at most ~{self.config.max_audio_seconds:.0f}s per call"
            )
        mono = np.ascontiguousarray(samples.mean(axis=1), dtype=np.float32)
        self.load()
        try:
            results = self._model.align(audio=(mono, int(sr)), text=" ".join(words), language=self.config.language)
        except Exception as exc:
            raise _translate_runtime_error(exc) from exc
        items = list(results[0])
        if len(items) != len(words):
            raise AlignmentError(
                f"aligner returned {len(items)} units for {len(words)} words; refusing to guess a mapping"
            )
        intervals: list[IntervalIn] = []
        for i, (item, word) in enumerate(zip(items, words, strict=True)):  # lengths verified above
            if str(item.text).lower() != word.lower():
                raise AlignmentError(f"aligner unit #{i} is {item.text!r} but expected {word!r}")
            intervals.append(IntervalIn(i, float(item.start_time), float(item.end_time), word))
        return QwenAlignment(intervals, dict(self._info))


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"


def _translate_load_error(exc: Exception, device: str) -> Exception:
    if _is_oom(exc):
        return QwenResourceError(
            f"not enough GPU memory to load the Qwen3-ForcedAligner on {device}: try dtype float16/bfloat16, "
            "free other GPU processes, or run on CPU (alignment.qwen.device: cpu)."
        )
    if isinstance(exc, (OSError, ConnectionError)):
        return QwenResourceError(
            f"could not download/load model {exc!s}. Pre-download it (huggingface-cli download "
            "Qwen/Qwen3-ForcedAligner-0.6B) or check connectivity."
        )
    return QwenResourceError(f"failed to load Qwen3-ForcedAligner: {type(exc).__name__}: {exc}")


def _translate_runtime_error(exc: Exception) -> Exception:
    if _is_oom(exc):
        return QwenResourceError("GPU ran out of memory during alignment; shorten the audio or use a smaller dtype/CPU.")
    return AlignmentError(f"Qwen3-ForcedAligner failed: {type(exc).__name__}: {exc}")
