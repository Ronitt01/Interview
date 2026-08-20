"""E0 and E0b — the detectors every later number is measured against.

A model with no baseline beside it is an unfalsifiable claim. These two exist so
that "F1 0.87" can be read as "F1 0.87 against 0.71 for a silence timer", which
is the only form in which the number carries information.

Both baselines score a clip the same way, and the choice is deliberate: **the
score is the duration of trailing silence, in milliseconds.** They differ only in
how they decide which frames are speech — a fixed energy gate for E0, a trained
neural VAD for E0b.

Why trailing silence rather than a binary rule with a hardcoded 500 ms cutoff:
a single cutoff produces one point, and one point cannot be compared against a
model that has a whole ROC curve. Emitting a continuous score lets the same
threshold sweep and the same cost-based operating-point selection run on the
baseline as on the model, so the comparison is like-for-like. The chosen
threshold is then directly interpretable — "this baseline fires after 320 ms of
silence" — which is more than can usually be said of a learned score.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import SAMPLE_RATE
from .audio import frame, rms_dbfs


@dataclass
class EnergyBaselineConfig:
    """E0: frame the clip, gate on energy, measure the silence at the end."""

    frame_ms: float = 20.0
    hop_ms: float = 10.0
    # Speech is "this many dB above the clip's own noise floor". Relative rather
    # than absolute, because absolute dBFS gates break the moment recording gain
    # changes — which across 12 source corpora it certainly does.
    speech_margin_db: float = 12.0
    # The floor is the 10th percentile of frame energy, not the minimum: a single
    # near-zero frame (a dropout, a codec artefact) would drag a min-based floor
    # down and make everything look like speech.
    floor_percentile: float = 10.0
    # The gate is also held this far below the clip's loud frames (95th
    # percentile). Without this cap, a clip that is speech from end to end has
    # floor ~= peak, so `floor + margin` lands *above* every frame and the
    # detector reports the whole clip as silence — the exact opposite of the
    # truth. Capping against the loud end means the gate always separates
    # something, and a clip with no real silence correctly yields no silence.
    peak_percentile: float = 95.0
    peak_drop_db: float = 20.0
    # Absolute presence check, applied to the *un-normalised* waveform.
    #
    # A purely relative gate has no way to tell a quiet room from loud speech —
    # it only ever sees ratios, so normalised silence looks exactly like
    # normalised speech. This threshold is the one absolute reference the
    # detector gets: if even the loudest frames of a clip sit below it, the clip
    # contains no speech and the whole thing is silence. -50 dBFS sits roughly
    # midway between recorded room tone (-60 to -70) and conversational speech
    # (-30 to -15), and it is the reason this baseline is *not* peak-normalised
    # first.
    min_speech_dbfs: float = -50.0
    # Bridge gaps shorter than this so that stop consonants and glottal closures
    # inside a word do not read as end-of-turn.
    min_gap_ms: float = 60.0


def _speech_mask(wave: np.ndarray, cfg: EnergyBaselineConfig) -> np.ndarray:
    # Deliberately *not* normalised — see EnergyBaselineConfig.min_speech_dbfs.
    wave = np.asarray(wave, dtype=np.float32)

    frame_len = max(1, int(cfg.frame_ms * SAMPLE_RATE / 1000))
    hop = max(1, int(cfg.hop_ms * SAMPLE_RATE / 1000))
    frames = frame(wave, frame_len, hop)
    energies = np.asarray([rms_dbfs(f) for f in frames], dtype=np.float64)

    finite = energies[np.isfinite(energies)]
    if finite.size == 0:
        return np.zeros(len(energies), dtype=bool)

    loud = float(np.percentile(finite, cfg.peak_percentile))
    if loud < cfg.min_speech_dbfs:
        # No frame is loud enough to be speech: the clip is silence throughout.
        return np.zeros(len(energies), dtype=bool)

    floor = float(np.percentile(finite, cfg.floor_percentile))
    # See EnergyBaselineConfig.peak_percentile for why this is a min, not a max.
    gate = min(floor + cfg.speech_margin_db, loud - cfg.peak_drop_db)
    mask = energies >= gate

    # Bridge short gaps.
    bridge = max(1, int(cfg.min_gap_ms / cfg.hop_ms))
    if bridge > 1 and mask.any():
        idx = np.flatnonzero(mask)
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= bridge:
                mask[a:b] = True
    return mask


def trailing_silence_ms(wave: np.ndarray, cfg: EnergyBaselineConfig | None = None) -> float:
    """Milliseconds of non-speech at the end of the clip.

    Returns the full clip duration when no frame passes the gate — a clip with no
    detectable speech has, by this detector's reckoning, been silent throughout.
    """
    cfg = cfg or EnergyBaselineConfig()
    mask = _speech_mask(wave, cfg)
    total_ms = 1000.0 * wave.size / SAMPLE_RATE
    if not mask.any():
        return total_ms
    last = int(np.flatnonzero(mask)[-1])
    tail_frames = len(mask) - 1 - last
    return float(tail_frames * cfg.hop_ms)


class EnergyBaseline:
    """E0. No training, no weights, 0 MB on disk."""

    name = "E0"
    description = "energy-gated trailing-silence timer"

    def __init__(self, cfg: EnergyBaselineConfig | None = None) -> None:
        self.cfg = cfg or EnergyBaselineConfig()

    def score(self, wave: np.ndarray) -> float:
        return trailing_silence_ms(wave, self.cfg)

    def score_many(self, waves) -> np.ndarray:
        return np.asarray([self.score(w) for w in waves], dtype=np.float64)

    @property
    def size_mb(self) -> float:
        return 0.0

    @property
    def params(self) -> int:
        return 0


class SileroBaseline:
    """E0b. A trained VAD, still no turn-detection training.

    This is the interesting baseline. Silero knows what speech is far better than
    an energy gate does, so if E1 barely beats E0b, the model has learned
    "detect silence" and not "detect a finished thought" — and the report should
    say that rather than celebrate the F1.

    Loaded from the standalone ``silero-vad`` package rather than through an
    agent framework, so the baseline has no dependency on anything but torch.
    """

    name = "E0b"
    description = "Silero VAD trailing-silence timer"

    def __init__(self, threshold: float = 0.5, min_silence_ms: int = 0) -> None:
        self.threshold = threshold
        self.min_silence_ms = min_silence_ms
        self._model = None
        self._ts_fn = None

    def _load(self):
        if self._model is None:
            from silero_vad import get_speech_timestamps, load_silero_vad

            self._model = load_silero_vad()
            self._ts_fn = get_speech_timestamps
        return self._model, self._ts_fn

    def score(self, wave: np.ndarray) -> float:
        import torch

        model, get_ts = self._load()
        total_ms = 1000.0 * wave.size / SAMPLE_RATE
        # Silero needs at least ~32 ms to produce a window.
        if wave.size < 512:
            return total_ms
        spans = get_ts(
            torch.from_numpy(np.asarray(wave, dtype=np.float32)),
            model,
            sampling_rate=SAMPLE_RATE,
            threshold=self.threshold,
            min_silence_duration_ms=self.min_silence_ms,
            return_seconds=False,
        )
        if not spans:
            return total_ms
        last_end = int(spans[-1]["end"])
        return max(0.0, 1000.0 * (wave.size - last_end) / SAMPLE_RATE)

    def score_many(self, waves) -> np.ndarray:
        return np.asarray([self.score(w) for w in waves], dtype=np.float64)

    @property
    def size_mb(self) -> float:
        """Silero's JIT model on disk, for the size column."""
        try:
            from pathlib import Path

            import silero_vad

            files = list(Path(silero_vad.__file__).parent.rglob("*.jit"))
            files += list(Path(silero_vad.__file__).parent.rglob("*.onnx"))
            return round(sum(f.stat().st_size for f in files) / 1e6, 2) if files else 1.8
        except Exception:
            return 1.8


def evaluate_baseline(
    baseline,
    cache,
    indices,
    name: str | None = None,
    max_false_interruption: float | None = None,
    progress_every: int = 2000,
):
    """Score a baseline over cached clips and return an :class:`Evaluation`.

    Note the scores are *milliseconds*, not probabilities, so the reported
    threshold reads as a silence duration. :func:`src.metrics.sweep` works on any
    monotone score, which is exactly why it was written that way.
    """
    from .metrics import DEFAULT_FI_BUDGET, evaluate

    # None means "use the table's shared ceiling", not "no ceiling" — an
    # unconstrained baseline would report its best-F1 point and rank above a
    # good model that was held to a budget.
    if max_false_interruption is None:
        max_false_interruption = DEFAULT_FI_BUDGET

    scores, labels = [], []
    for k, i in enumerate(np.asarray(indices, dtype=np.int64)):
        wave = cache.wave(int(i))
        scores.append(baseline.score(wave))
        labels.append(cache.label(int(i)))
        if progress_every and (k + 1) % progress_every == 0:
            print(f"  {baseline.name}: {k + 1:,d}/{len(indices):,d}", flush=True)

    ev = evaluate(
        name or baseline.name,
        np.asarray(labels),
        np.asarray(scores),
        max_false_interruption=max_false_interruption,
        score_unit="ms_trailing_silence",
        detector=baseline.description,
    )
    ev.size_mb = getattr(baseline, "size_mb", None)
    ev.params_m = (getattr(baseline, "params", 0) or 0) / 1e6
    return ev
