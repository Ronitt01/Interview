"""The streaming state machine: smoothing, minimum silence, hysteresis.

Each of the three mechanisms gets a test that fails if the mechanism is removed.
That is the point — a smoothing parameter nothing tests is a smoothing parameter
nobody knows is working.
"""
from __future__ import annotations

import numpy as np
import pytest

from src import SAMPLE_RATE
from src.streaming import (
    State,
    StreamingConfig,
    StreamingTurnDetector,
    StreamResult,
    summarise_stream,
)


class ScriptedScore:
    """Returns ``high`` once the detector has consumed ``flip_ms`` of audio."""

    def __init__(self, detector, flip_ms, low=0.02, high=0.98):
        self.d, self.flip_ms, self.low, self.high = detector, flip_ms, low, high

    def __call__(self, window):
        return self.high if self.d.audio_time_ms >= self.flip_ms else self.low


class Flapping:
    """Alternates high/low every hop — the chattering a single threshold causes."""

    def __init__(self):
        self.n = 0

    def __call__(self, window):
        self.n += 1
        return 0.95 if self.n % 2 == 0 else 0.10


def silence(seconds):
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


# --------------------------------------------------------------------------- #
# config validation
# --------------------------------------------------------------------------- #
def test_inverted_hysteresis_band_is_rejected():
    """exit > enter inverts the dead band and makes the detector chatter by
    design; that is a config bug, not a tuning choice."""
    with pytest.raises(ValueError, match="hysteresis band is inverted"):
        StreamingConfig(enter_threshold=0.4, exit_threshold=0.8)


def test_bad_ema_alpha_is_rejected():
    with pytest.raises(ValueError, match="ema_alpha"):
        StreamingConfig(ema_alpha=0.0)
    with pytest.raises(ValueError, match="ema_alpha"):
        StreamingConfig(ema_alpha=1.5)
    StreamingConfig(ema_alpha=1.0)  # 1.0 means "no smoothing" and is legal


def test_bad_hop_is_rejected():
    with pytest.raises(ValueError, match="hop_ms"):
        StreamingConfig(hop_ms=0)


# --------------------------------------------------------------------------- #
# minimum silence
# --------------------------------------------------------------------------- #
def test_min_silence_delays_the_decision_by_the_configured_hold():
    """Without this hold, every sub-200 ms gap in natural speech fires."""
    cfg = StreamingConfig(
        window_seconds=1.0, hop_ms=100, min_silence_ms=300,
        ema_alpha=1.0, enter_threshold=0.5, exit_threshold=0.5, warmup_ms=0,
    )
    det = StreamingTurnDetector(lambda w: 0.0, cfg)
    det.score_fn = ScriptedScore(det, flip_ms=500)
    d = det.run(silence(3.0), chunk_ms=20)
    assert d is not None
    assert d.held_ms == pytest.approx(300, abs=1e-6)
    # Score goes high at 500 ms; +300 ms hold = fires at 800 ms.
    assert d.audio_time_ms == pytest.approx(800, abs=100)


def test_a_short_pause_shorter_than_min_silence_never_fires():
    """The core false-interruption guard: a brief gap is a pause, not a turn end."""
    cfg = StreamingConfig(
        window_seconds=1.0, hop_ms=50, min_silence_ms=400,
        ema_alpha=1.0, enter_threshold=0.5, exit_threshold=0.5, warmup_ms=0,
    )

    class Blip:
        """High for only 150 ms, then low again."""

        def __init__(self, d):
            self.d = d

        def __call__(self, w):
            return 0.9 if 500 <= self.d.audio_time_ms < 650 else 0.05

    det = StreamingTurnDetector(lambda w: 0.0, cfg)
    det.score_fn = Blip(det)
    assert det.run(silence(2.0), chunk_ms=20) is None


def test_zero_min_silence_fires_on_the_first_crossing():
    cfg = StreamingConfig(
        window_seconds=1.0, hop_ms=50, min_silence_ms=0,
        ema_alpha=1.0, enter_threshold=0.5, exit_threshold=0.5, warmup_ms=0,
    )
    det = StreamingTurnDetector(lambda w: 0.0, cfg)
    det.score_fn = ScriptedScore(det, flip_ms=300)
    d = det.run(silence(1.5), chunk_ms=20)
    assert d is not None and d.held_ms == 0.0


# --------------------------------------------------------------------------- #
# hysteresis
# --------------------------------------------------------------------------- #
def test_hysteresis_suppresses_a_flapping_score():
    """A signal alternating across a single boundary must not restart the hold
    timer every hop. With a dead band, state persists between the thresholds."""
    cfg = StreamingConfig(
        window_seconds=1.0, hop_ms=50, min_silence_ms=200,
        ema_alpha=1.0, enter_threshold=0.90, exit_threshold=0.05, warmup_ms=0,
    )
    det = StreamingTurnDetector(Flapping(), cfg)
    d = det.run(silence(2.0), chunk_ms=20)
    # 0.10 sits inside the band (above exit 0.05, below enter 0.90), so the
    # timer is held rather than reset, and the hold completes.
    assert d is not None, "hysteresis failed to hold state across the dead band"


def test_dropping_below_exit_threshold_resets_the_hold():
    """The other half of hysteresis: real evidence must be able to cancel."""
    cfg = StreamingConfig(
        window_seconds=1.0, hop_ms=50, min_silence_ms=300,
        ema_alpha=1.0, enter_threshold=0.7, exit_threshold=0.3, warmup_ms=0,
    )

    class HighThenLow:
        def __init__(self, d):
            self.d = d

        def __call__(self, w):
            t = self.d.audio_time_ms
            if 200 <= t < 400:
                return 0.9  # starts the hold
            if 400 <= t < 700:
                return 0.05  # clearly below exit -> cancel
            return 0.1

    det = StreamingTurnDetector(lambda w: 0.0, cfg)
    det.score_fn = HighThenLow(det)
    assert det.run(silence(1.2), chunk_ms=20) is None
    assert det.state is State.SPEAKING


# --------------------------------------------------------------------------- #
# smoothing
# --------------------------------------------------------------------------- #
def test_ema_smoothing_rejects_a_single_spike():
    """One window crossing the threshold is noise; a sustained rise is a
    decision. With alpha=0.3 a lone spike cannot reach 0.7 in one hop."""
    cfg = StreamingConfig(
        window_seconds=1.0, hop_ms=50, min_silence_ms=0,
        ema_alpha=0.3, enter_threshold=0.7, exit_threshold=0.2, warmup_ms=0,
    )

    class OneSpike:
        def __init__(self, d):
            self.d, self.fired = d, False

        def __call__(self, w):
            if not self.fired and self.d.audio_time_ms >= 300:
                self.fired = True
                return 1.0
            return 0.0

    det = StreamingTurnDetector(lambda w: 0.0, cfg)
    det.score_fn = OneSpike(det)
    assert det.run(silence(1.0), chunk_ms=20) is None


def test_alpha_one_disables_smoothing():
    cfg = StreamingConfig(
        window_seconds=1.0, hop_ms=50, min_silence_ms=0,
        ema_alpha=1.0, enter_threshold=0.7, exit_threshold=0.2, warmup_ms=0,
    )
    det = StreamingTurnDetector(lambda w: 0.0, cfg)
    det.score_fn = ScriptedScore(det, flip_ms=200, high=0.99)
    d = det.run(silence(1.0), chunk_ms=20)
    assert d is not None and d.smoothed == pytest.approx(0.99)


# --------------------------------------------------------------------------- #
# warmup, latching, chunk decoupling
# --------------------------------------------------------------------------- #
def test_warmup_prevents_an_immediate_fire_on_an_opening_silence():
    cfg = StreamingConfig(
        window_seconds=1.0, hop_ms=50, min_silence_ms=0,
        ema_alpha=1.0, enter_threshold=0.5, exit_threshold=0.5, warmup_ms=600,
    )
    det = StreamingTurnDetector(lambda w: 0.99, cfg)
    d = det.run(silence(2.0), chunk_ms=20)
    assert d is not None
    assert d.audio_time_ms >= 600, "fired inside the warmup window"


def test_latch_means_a_turn_ends_only_once():
    cfg = StreamingConfig(
        window_seconds=1.0, hop_ms=50, min_silence_ms=0,
        ema_alpha=1.0, enter_threshold=0.5, exit_threshold=0.5, warmup_ms=0, latch=True,
    )
    det = StreamingTurnDetector(lambda w: 0.99, cfg)
    first = det.push(silence(0.5))
    assert first is not None
    for _ in range(10):
        assert det.push(silence(0.1)) is None


def test_chunk_size_is_decoupled_from_hop_size():
    """A WebRTC track delivers 10-20 ms frames but running an 8 s encoder every
    10 ms is waste. Chunks accumulate; the model runs once per hop."""
    cfg = StreamingConfig(
        window_seconds=1.0, hop_ms=200, min_silence_ms=0,
        ema_alpha=1.0, enter_threshold=2.0, exit_threshold=2.0, warmup_ms=0,
    )
    calls = {"n": 0}

    def counting(w):
        calls["n"] += 1
        return 0.0

    det = StreamingTurnDetector(counting, cfg)
    det.run(silence(1.0), chunk_ms=10)  # 100 chunks of 10 ms
    assert calls["n"] == pytest.approx(5, abs=1), (
        f"model ran {calls['n']} times for 1 s at a 200 ms hop; expected ~5"
    )


def test_reset_clears_all_state():
    cfg = StreamingConfig(window_seconds=1.0, hop_ms=50, warmup_ms=0)
    det = StreamingTurnDetector(lambda w: 0.99, cfg)
    det.run(silence(1.0), chunk_ms=20)
    det.reset()
    assert det.state is State.SPEAKING
    assert det.smoothed is None
    assert det.samples_seen == 0
    assert det.trace.probability == []


def test_trace_records_one_entry_per_hop():
    cfg = StreamingConfig(
        window_seconds=1.0, hop_ms=100, enter_threshold=2.0, exit_threshold=2.0, warmup_ms=0,
    )
    det = StreamingTurnDetector(lambda w: 0.1, cfg)
    det.run(silence(1.0), chunk_ms=20)
    assert len(det.trace.probability) == 10
    assert len(det.trace.smoothed) == len(det.trace.state) == 10


def test_empty_chunk_is_ignored():
    det = StreamingTurnDetector(lambda w: 0.0, StreamingConfig())
    assert det.push(np.zeros(0, dtype=np.float32)) is None


# --------------------------------------------------------------------------- #
# stream scoring
# --------------------------------------------------------------------------- #
def test_decision_lag_is_relative_to_the_clip_end():
    r = StreamResult(fired=True, audio_time_ms=1200, wall_latency_ms=5,
                     clip_ms=1000, label=1)
    assert r.decision_lag_ms == pytest.approx(200)
    early = StreamResult(fired=True, audio_time_ms=900, wall_latency_ms=5,
                         clip_ms=1000, label=1)
    assert early.decision_lag_ms == pytest.approx(-100)
    never = StreamResult(fired=False, audio_time_ms=None, wall_latency_ms=None,
                         clip_ms=1000, label=1)
    assert never.decision_lag_ms is None


def test_summarise_uses_only_correct_positives_for_time_to_decide():
    """Mixing false fires into the latency would let a detector improve its
    latency by interrupting more — exactly backwards."""
    results = [
        StreamResult(True, 1100, 4.0, 1000, 1),   # tp, lag +100
        StreamResult(True, 1300, 4.0, 1000, 1),   # tp, lag +300
        StreamResult(False, None, None, 1000, 1),  # fn
        StreamResult(True, 900, 4.0, 1000, 0),    # fp — must not affect ttd
        StreamResult(False, None, None, 1000, 0),  # tn
    ]
    s = summarise_stream(results)
    assert s["confusion"]["tp"] == 2
    assert s["confusion"]["fn"] == 1
    assert s["confusion"]["fp"] == 1
    assert s["confusion"]["tn"] == 1
    assert s["time_to_decide_ms"]["n"] == 2
    assert s["time_to_decide_ms"]["p50"] == pytest.approx(200)


def test_firing_far_before_the_clip_end_counts_as_an_interruption():
    """The scoring trap this guards.

    A positive clip is a completed utterance that may contain internal pauses. A
    detector that fires seconds early has interrupted the speaker; the clip
    ending later does not make that a hit. Without this rule, a naive detector
    that fires at every pause scores a near-perfect recall on positives.
    """
    results = [
        # 8 s clip, fired at 1 s -> 7 s early. Interruption, not a detection.
        StreamResult(True, 1000, 4.0, 8000, 1),
        # fired 200 ms before the end: legitimate anticipation
        StreamResult(True, 7800, 4.0, 8000, 1),
    ]
    s = summarise_stream(results, early_tolerance_ms=1000.0)
    assert s["confusion"]["fp"] == 1, "early fire was not counted as an interruption"
    assert s["confusion"]["tp"] == 1
    assert s["early_fires"] == 1
    # And the 7 s early fire must not enter the latency distribution.
    assert s["time_to_decide_ms"]["n"] == 1
    assert s["time_to_decide_ms"]["p50"] == pytest.approx(-200)


def test_early_tolerance_is_configurable():
    results = [StreamResult(True, 5000, 4.0, 8000, 1)]  # 3 s early
    strict = summarise_stream(results, early_tolerance_ms=1000.0)
    loose = summarise_stream(results, early_tolerance_ms=5000.0)
    assert strict["confusion"]["fp"] == 1
    assert loose["confusion"]["tp"] == 1
