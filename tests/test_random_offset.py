"""Random-offset cropping: the label must follow the crop, not the clip.

These tests exist because the failure mode is silent. If the crop happens and the
label does not change, training looks like it is learning about arbitrary window
alignments while actually being taught that mid-utterance moments are endpoints —
which is worse than not doing it at all.
"""
import numpy as np
import pytest

from src import SAMPLE_RATE
from src.dataset import RandomOffsetConfig, apply_random_offset


def wave(seconds: float) -> np.ndarray:
    return np.ones(int(seconds * SAMPLE_RATE), dtype=np.float32)


def test_disabled_is_a_no_op():
    cfg = RandomOffsetConfig(enabled=False)
    w = wave(5.0)
    out, label, shift = apply_random_offset(w, 1, cfg, np.random.default_rng(0))
    assert out.size == w.size
    assert label == 1
    assert shift == 0.0


def test_prob_zero_never_shifts():
    cfg = RandomOffsetConfig(enabled=True, prob=0.0)
    for seed in range(20):
        out, label, shift = apply_random_offset(
            wave(5.0), 1, cfg, np.random.default_rng(seed)
        )
        assert shift == 0.0 and label == 1 and out.size == 5 * SAMPLE_RATE


def test_a_large_shift_relabels_a_positive_as_negative():
    # tolerance 150 ms, shift forced large by making min == max reachable only
    # far from the boundary.
    cfg = RandomOffsetConfig(
        enabled=True, prob=1.0, max_shift_ms=2000.0, tolerance_ms=150.0
    )
    relabelled = 0
    for seed in range(200):
        _out, label, shift = apply_random_offset(
            wave(6.0), 1, cfg, np.random.default_rng(seed)
        )
        if shift > cfg.tolerance_ms:
            assert label == 0, f"shift {shift:.1f} ms should have relabelled"
            relabelled += 1
        else:
            assert label == 1
    # With a 2 s range and a 150 ms tolerance the vast majority must relabel;
    # if this ever drops the sampling has broken.
    assert relabelled > 150


def test_a_shift_inside_the_tolerance_keeps_the_label():
    cfg = RandomOffsetConfig(
        enabled=True, prob=1.0, max_shift_ms=100.0, tolerance_ms=99.0
    )
    for seed in range(50):
        _out, label, shift = apply_random_offset(
            wave(6.0), 1, cfg, np.random.default_rng(seed)
        )
        if shift <= cfg.tolerance_ms:
            assert label == 1


def test_negatives_stay_negative_however_far_they_are_cropped():
    cfg = RandomOffsetConfig(enabled=True, prob=1.0, max_shift_ms=3000.0)
    for seed in range(100):
        _out, label, _shift = apply_random_offset(
            wave(6.0), 0, cfg, np.random.default_rng(seed)
        )
        assert label == 0


def test_the_crop_only_ever_removes_from_the_end():
    cfg = RandomOffsetConfig(enabled=True, prob=1.0, max_shift_ms=2000.0)
    w = np.arange(6 * SAMPLE_RATE, dtype=np.float32)
    out, _label, shift = apply_random_offset(w, 1, cfg, np.random.default_rng(3))
    assert shift > 0.0, "expected this seed to shift"
    assert out.size < w.size
    # A prefix, so the head is untouched and only the tail is discarded.
    assert np.array_equal(out, w[: out.size])


def test_min_keep_is_respected():
    cfg = RandomOffsetConfig(
        enabled=True, prob=1.0, max_shift_ms=100000.0, tolerance_ms=150.0,
        min_keep_ms=1000.0,
    )
    floor = int(1000.0 * SAMPLE_RATE / 1000.0)
    for seed in range(100):
        out, _label, _shift = apply_random_offset(
            wave(3.0), 1, cfg, np.random.default_rng(seed)
        )
        assert out.size >= floor


def test_a_clip_shorter_than_min_keep_is_left_alone():
    cfg = RandomOffsetConfig(enabled=True, prob=1.0, min_keep_ms=2000.0)
    w = wave(1.0)
    out, label, shift = apply_random_offset(w, 1, cfg, np.random.default_rng(0))
    assert out.size == w.size and label == 1 and shift == 0.0


def test_same_seed_gives_the_same_crop():
    cfg = RandomOffsetConfig(enabled=True, prob=1.0)
    a = apply_random_offset(wave(6.0), 1, cfg, np.random.default_rng(11))
    b = apply_random_offset(wave(6.0), 1, cfg, np.random.default_rng(11))
    assert a[0].size == b[0].size and a[1] == b[1] and a[2] == b[2]


def test_tolerance_swallowing_the_whole_range_is_rejected():
    # Otherwise the config silently becomes a no-op that looks enabled.
    with pytest.raises(ValueError, match="would ever be relabelled"):
        RandomOffsetConfig(enabled=True, max_shift_ms=100.0, tolerance_ms=100.0)


def test_bad_probability_is_rejected():
    with pytest.raises(ValueError, match="prob must be"):
        RandomOffsetConfig(enabled=True, prob=1.5)


def test_the_realised_positive_rate_drops_and_is_predictable():
    """The point of the mechanism, stated as a number.

    With prob=0.5 and a tolerance covering a small slice of the shift range,
    roughly half of the positives survive. This pins the magnitude so a future
    change to the sampling cannot quietly alter the class balance.
    """
    cfg = RandomOffsetConfig(
        enabled=True, prob=0.5, max_shift_ms=4000.0, tolerance_ms=150.0
    )
    kept = 0
    n = 2000
    for seed in range(n):
        _out, label, _shift = apply_random_offset(
            wave(8.0), 1, cfg, np.random.default_rng(seed)
        )
        kept += label
    rate = kept / n
    assert 0.45 < rate < 0.60, rate
