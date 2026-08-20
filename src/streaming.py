"""Rolling-window streaming detector: the part that makes this a component.

Offline classification of pre-cut clips is the common submission, and it answers
a question nobody asks in production. In a real call the audio arrives in 10-20 ms
chunks, there is no "clip", and the detector has to decide *while* the person is
still talking. This module is that detector.

Three mechanisms sit between the model's per-window probability and an emitted
``USER_TURN_ENDED``, and each exists for a specific failure it prevents:

**Smoothing (EMA).** A single window crossing the threshold is noise — a breath,
a codec glitch, a moment of low energy mid-word. A sustained rise is a decision.
An exponential moving average is used rather than a boxcar because it needs one
float of state instead of a history buffer, and because it weights the most
recent window most heavily, which is the right prior when the thing you are
detecting just happened.

**Minimum silence.** Natural speech is full of sub-200 ms gaps: stop closures,
in-breaths, the pause before a word someone is searching for. Requiring the
smoothed probability to *stay* above the entry threshold for ``min_silence_ms``
before firing is what separates "they stopped" from "they paused".

**Hysteresis.** One threshold sitting at 0.5 chatters: a signal hovering near the
boundary crosses it repeatedly and the detector starts and stops. Entering on
0.70 and exiting on 0.45 means that once the detector has committed to "turn
ending", it takes real evidence to pull it back. This is the same reason a
thermostat has a dead band.

The latency that gets reported is measured **end to end** — from the arrival of
the audio chunk that completed the decision to the emission of the event —
because that is what a user experiences. Model forward time alone is a component
of it, not a substitute for it.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterator

import numpy as np

from . import SAMPLE_RATE
from .audio import WindowSpec, fit_window, normalise


class State(str, Enum):
    SPEAKING = "speaking"
    PENDING_END = "pending_end"
    ENDED = "ended"


@dataclass
class StreamingConfig:
    """Everything the state machine needs. Every default is defended above."""

    window_seconds: float = 8.0
    hop_ms: float = 160.0
    enter_threshold: float = 0.70
    exit_threshold: float = 0.45
    ema_alpha: float = 0.4
    min_silence_ms: float = 200.0
    # Refuse to emit until this much audio has been seen, so a stream that opens
    # on silence does not immediately fire.
    warmup_ms: float = 300.0
    normalise_mode: str = "peak"
    # After emitting, ignore further audio until reset() — a turn ends once.
    latch: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1], got {self.ema_alpha}")
        if self.exit_threshold > self.enter_threshold:
            raise ValueError(
                f"exit_threshold ({self.exit_threshold}) must be <= "
                f"enter_threshold ({self.enter_threshold}); otherwise the "
                "hysteresis band is inverted and the detector chatters by design"
            )
        if self.hop_ms <= 0:
            raise ValueError("hop_ms must be positive")

    @property
    def hop_samples(self) -> int:
        return max(1, int(round(self.hop_ms * SAMPLE_RATE / 1000.0)))

    @property
    def window_spec(self) -> WindowSpec:
        return WindowSpec(self.window_seconds)

    def to_dict(self) -> dict:
        return {
            "window_seconds": self.window_seconds,
            "hop_ms": self.hop_ms,
            "enter_threshold": self.enter_threshold,
            "exit_threshold": self.exit_threshold,
            "ema_alpha": self.ema_alpha,
            "min_silence_ms": self.min_silence_ms,
            "warmup_ms": self.warmup_ms,
            "latch": self.latch,
        }


@dataclass
class Decision:
    """One emitted event, with the numbers needed to score it."""

    event: str
    audio_time_ms: float
    """Position in the stream, in ms of audio, when the decision was emitted."""
    wall_latency_ms: float
    """Wall-clock ms from chunk arrival to emission. Includes the model."""
    probability: float
    smoothed: float
    held_ms: float
    """How long the smoothed probability stayed above the entry threshold."""

    def to_dict(self) -> dict:
        return {
            "event": self.event,
            "audio_time_ms": round(self.audio_time_ms, 1),
            "wall_latency_ms": round(self.wall_latency_ms, 2),
            "probability": round(self.probability, 4),
            "smoothed": round(self.smoothed, 4),
            "held_ms": round(self.held_ms, 1),
        }


@dataclass
class Trace:
    """Per-hop record, for plotting the timeline in the demo and notebooks."""

    audio_time_ms: list[float] = field(default_factory=list)
    probability: list[float] = field(default_factory=list)
    smoothed: list[float] = field(default_factory=list)
    state: list[str] = field(default_factory=list)
    infer_ms: list[float] = field(default_factory=list)

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "audio_time_ms": np.asarray(self.audio_time_ms),
            "probability": np.asarray(self.probability),
            "smoothed": np.asarray(self.smoothed),
            "infer_ms": np.asarray(self.infer_ms),
        }


ScoreFn = Callable[[np.ndarray], float]
"""Takes a fixed-length window, returns P(turn ended). Any detector fits."""


class StreamingTurnDetector:
    """Feed it audio chunks; it emits at most one ``USER_TURN_ENDED`` per turn.

    Deliberately agnostic about what produces the probability: pass a
    :class:`src.inference.TurnPredictor`, a baseline, or a lambda. That is what
    lets the streaming layer be benchmarked against E0 on identical machinery.
    """

    EVENT = "USER_TURN_ENDED"

    def __init__(self, score_fn: ScoreFn, cfg: StreamingConfig | None = None) -> None:
        self.score_fn = score_fn
        self.cfg = cfg or StreamingConfig()
        self._buf = deque(maxlen=self.cfg.window_spec.samples)
        self.reset()

    # -- lifecycle ---------------------------------------------------------- #
    def reset(self) -> None:
        self._buf.clear()
        self._pending = np.zeros(0, dtype=np.float32)
        self.state = State.SPEAKING
        self.smoothed: float | None = None
        self.samples_seen = 0
        self._above_since_ms: float | None = None
        self._emitted = False
        self.trace = Trace()

    @property
    def audio_time_ms(self) -> float:
        return 1000.0 * self.samples_seen / SAMPLE_RATE

    # -- the hot path ------------------------------------------------------- #
    def push(self, chunk: np.ndarray) -> Decision | None:
        """Accept an audio chunk of any length. Returns an event or ``None``.

        Chunk length is decoupled from hop length on purpose: a WebRTC track
        delivers 10 or 20 ms frames, but running the encoder every 10 ms is
        wasteful when the window is 8 s. Chunks accumulate and the model runs once
        per ``hop_ms``. This is the same accumulation lesson the voice-agent work
        learned the hard way — handling audio per-frame instead of per-hop burns
        CPU for no accuracy.
        """
        arrival = time.perf_counter()
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return None

        if self._emitted and self.cfg.latch:
            self.samples_seen += chunk.size
            return None

        self._pending = (
            np.concatenate([self._pending, chunk]) if self._pending.size else chunk
        )
        self.samples_seen += chunk.size
        self._buf.extend(chunk.tolist())

        hop = self.cfg.hop_samples
        decision: Decision | None = None
        while self._pending.size >= hop:
            self._pending = self._pending[hop:]
            decision = self._step(arrival)
            if decision is not None:
                break
        return decision

    def _step(self, arrival: float) -> Decision | None:
        t0 = time.perf_counter()
        window = fit_window(
            normalise(np.asarray(self._buf, dtype=np.float32), self.cfg.normalise_mode),
            self.cfg.window_spec,
        )
        prob = float(self.score_fn(window))
        infer_ms = (time.perf_counter() - t0) * 1000.0

        a = self.cfg.ema_alpha
        self.smoothed = prob if self.smoothed is None else a * prob + (1 - a) * self.smoothed

        now_ms = self.audio_time_ms
        self.trace.audio_time_ms.append(now_ms)
        self.trace.probability.append(prob)
        self.trace.smoothed.append(self.smoothed)
        self.trace.infer_ms.append(infer_ms)

        decision = self._advance(now_ms, prob, arrival)
        self.trace.state.append(self.state.value)
        return decision

    def _advance(self, now_ms: float, prob: float, arrival: float) -> Decision | None:
        """The state machine. Hysteresis and min-silence live here."""
        assert self.smoothed is not None
        s = self.smoothed

        if now_ms < self.cfg.warmup_ms:
            return None

        if self.state in (State.SPEAKING, State.PENDING_END):
            if s >= self.cfg.enter_threshold:
                if self._above_since_ms is None:
                    self._above_since_ms = now_ms
                    self.state = State.PENDING_END
                held = now_ms - self._above_since_ms
                if held >= self.cfg.min_silence_ms:
                    self.state = State.ENDED
                    self._emitted = True
                    return Decision(
                        event=self.EVENT,
                        audio_time_ms=now_ms,
                        wall_latency_ms=(time.perf_counter() - arrival) * 1000.0,
                        probability=prob,
                        smoothed=s,
                        held_ms=held,
                    )
            elif s < self.cfg.exit_threshold:
                # Dropped out of the hysteresis band: the pause was a pause.
                self._above_since_ms = None
                self.state = State.SPEAKING
            # Between exit and enter thresholds: hold state, hold the timer.
        return None

    # -- offline driving ---------------------------------------------------- #
    def run(self, wave: np.ndarray, chunk_ms: float = 20.0) -> Decision | None:
        """Replay a complete waveform as if it had streamed in.

        ``chunk_ms=20`` matches a typical WebRTC frame, so the trace this
        produces has the same granularity the live path would.
        """
        self.reset()
        step = max(1, int(chunk_ms * SAMPLE_RATE / 1000.0))
        wave = np.asarray(wave, dtype=np.float32).reshape(-1)
        for start in range(0, wave.size, step):
            d = self.push(wave[start : start + step])
            if d is not None:
                return d
        return None


# --------------------------------------------------------------------------- #
# scoring the streaming detector
# --------------------------------------------------------------------------- #
@dataclass
class StreamResult:
    fired: bool
    audio_time_ms: float | None
    wall_latency_ms: float | None
    clip_ms: float
    label: int
    """Ground truth: 1 if the clip ends at a real turn boundary."""

    @property
    def decision_lag_ms(self) -> float | None:
        """Audio consumed *past* the clip's own end-point, in ms.

        Negative means the detector fired before the clip ended — which for a
        positive clip is early but plausible (the endpoint evidence is present
        before the last sample), and for a negative clip is a false interruption.
        """
        if not self.fired or self.audio_time_ms is None:
            return None
        return self.audio_time_ms - self.clip_ms


def run_stream_eval(
    detector: StreamingTurnDetector,
    cache,
    indices,
    chunk_ms: float = 20.0,
    progress_every: int = 200,
) -> list[StreamResult]:
    """Replay clips through the streaming detector.

    Two honest limitations, stated here because they belong next to the code.

    **Pre-cut clips.** Each cached clip is a complete utterance, so replaying it
    measures how promptly the detector reacts to a boundary already at the end of
    the audio. It does not measure behaviour across a continuous multi-turn
    conversation, because the corpus contains none.

    **The train/stream distribution gap — read this before trusting a streaming
    number.** A model trained on these clips only ever sees windows whose *right
    edge coincides with an annotated boundary*: positives end at a real turn end,
    negatives end at a real mid-utterance point. In streaming, the right edge is
    wherever the hop happens to land — usually neither. That is a genuine
    distribution shift the training data does not cover, and it shows up
    dramatically: a model measured at a 3.3% false-interruption rate on clips was
    measured at 46% streamed, because it fires on ordinary intra-utterance pauses
    that no training example ever labelled.

    Clip-level scores therefore **do not transfer to streaming**, and quoting one
    as evidence for the other would be wrong. Three ways to close the gap, in
    increasing order of effort:

    1. tighten the streaming parameters — a higher entry threshold and a longer
       minimum silence trade recall for interruptions (see ``--sweep``);
    2. train on **randomly-offset crops** labelled by whether the crop's right
       edge is genuinely a boundary, so the model sees arbitrary window
       alignments during training;
    3. calibrate the operating threshold *on streamed replay* rather than on clip
       classification, which is the smallest change that makes the reported
       number describe the deployed system.
    """
    out: list[StreamResult] = []
    idx = np.asarray(indices, dtype=np.int64)
    for k, i in enumerate(idx):
        wave = cache.wave(int(i))
        d = detector.run(wave, chunk_ms=chunk_ms)
        out.append(
            StreamResult(
                fired=d is not None,
                audio_time_ms=d.audio_time_ms if d else None,
                wall_latency_ms=d.wall_latency_ms if d else None,
                clip_ms=1000.0 * wave.size / SAMPLE_RATE,
                label=cache.label(int(i)),
            )
        )
        if progress_every and (k + 1) % progress_every == 0:
            print(f"  stream: {k + 1:,d}/{idx.size:,d}", flush=True)
    return out


EARLY_FIRE_TOLERANCE_MS = 1000.0
"""How far before a clip's end a fire still counts as detecting *that* endpoint.

This constant fixes a scoring trap that silently flatters any streaming detector
evaluated on pre-cut clips. The corpus's positive clips have a median duration
around 7 s and contain internal pauses. A naive detector fires at the first
sustained pause — often seconds before the clip ends — and a scorer that only
asks "did it fire on a positive clip?" records a true positive. But firing
mid-utterance *is* an interruption; the clip happening to end later does not
redeem it.

So a fire more than this far before the clip's end is counted as a false
interruption regardless of the clip's label. One second is generous — it allows
the detector to anticipate a boundary from falling pitch and trailing energy,
which is the behaviour we actually want — while still catching a detector that
fired during a mid-sentence pause.
"""


def summarise_stream(
    results: list[StreamResult],
    early_tolerance_ms: float = EARLY_FIRE_TOLERANCE_MS,
) -> dict:
    """Turn stream results into the numbers the report quotes.

    ``time_to_decide`` is reported over *correctly fired positives* only. Mixing
    in the false and early fires would let a detector improve its latency by
    interrupting more, which is exactly backwards.

    See :data:`EARLY_FIRE_TOLERANCE_MS` for why an early fire on a positive clip
    is scored as a false interruption rather than a hit.
    """
    from .metrics import Confusion, percentiles

    tp = fp = tn = fn = 0
    early_fires = 0
    lags: list[float] = []

    for r in results:
        lag = r.decision_lag_ms
        too_early = (
            r.fired and lag is not None and lag < -abs(early_tolerance_ms)
        )
        if too_early:
            early_fires += 1

        if r.label == 1:
            if not r.fired:
                fn += 1
            elif too_early:
                # Fired during the utterance, not at its end: an interruption.
                fp += 1
            else:
                tp += 1
                if lag is not None:
                    lags.append(lag)
        else:
            if r.fired:
                fp += 1
            else:
                tn += 1

    conf = Confusion(tp=tp, fp=fp, tn=tn, fn=fn)
    walls = [r.wall_latency_ms for r in results if r.wall_latency_ms is not None]
    return {
        "confusion": conf.to_dict(),
        "time_to_decide_ms": percentiles(lags),
        "wall_latency_ms": percentiles(walls),
        "fired_rate": sum(1 for r in results if r.fired) / max(len(results), 1),
        "early_fires": early_fires,
        "early_fire_rate": early_fires / max(len(results), 1),
        "early_tolerance_ms": float(early_tolerance_ms),
    }


def iter_chunks(wave: np.ndarray, chunk_ms: float = 20.0) -> Iterator[np.ndarray]:
    step = max(1, int(chunk_ms * SAMPLE_RATE / 1000.0))
    wave = np.asarray(wave, dtype=np.float32).reshape(-1)
    for start in range(0, wave.size, step):
        yield wave[start : start + step]
