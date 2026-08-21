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


# --------------------------------------------------------------------------- #
# TurnDataset integration, and the guarantee E1/E2 depend on
# --------------------------------------------------------------------------- #
class StubCache:
    """Minimal stand-in for WaveCache.

    Only the four members TurnDataset touches: wave(), label(), lengths and
    meta. Keeps these tests fast and independent of an on-disk cache, so they
    run in CI where no corpus exists.
    """

    def __init__(self, n=12, seconds=6.0, seed=0):
        rng = np.random.default_rng(seed)
        self.n = n
        self._waves = [
            (rng.standard_normal(int(seconds * SAMPLE_RATE)) * 0.1).astype(np.float32)
            for _ in range(n)
        ]
        self._labels = [i % 2 for i in range(n)]
        self.lengths = np.asarray([w.size for w in self._waves], dtype=np.int64)
        self.meta = [{"dataset": "stub", "id": str(i)} for i in range(n)]

    def __len__(self):
        return self.n

    def wave(self, i):
        return self._waves[int(i)]

    def label(self, i):
        return self._labels[int(i)]


def make_ds(offset_cfg=None, n=12, **kw):
    from src.audio import WindowSpec
    from src.dataset import TurnDataset

    cache = StubCache(n=n)
    idx = np.arange(len(cache))
    return TurnDataset(cache, idx, WindowSpec(2.0), "peak", None,
                       offset_cfg=offset_cfg, **kw)


def test_omitting_offset_cfg_and_passing_a_disabled_one_are_the_same():
    """The guarantee E1 and the whole E2 sweep rest on.

    Both configs must feed the model identical tensors, or the refactor changed
    training for every run that predates it.
    """
    from src.dataset import RandomOffsetConfig

    a = make_ds(None)
    b = make_ds(RandomOffsetConfig(enabled=False))
    for i in range(len(a)):
        mel_a, lab_a = a[i]
        mel_b, lab_b = b[i]
        assert np.array_equal(mel_a, mel_b), f"row {i} mel differs"
        assert lab_a == lab_b


def test_enabling_offsets_actually_changes_the_tensors():
    """Guards the test above from passing because the feature is dead."""
    from src.dataset import RandomOffsetConfig

    off = make_ds(None)
    on = make_ds(RandomOffsetConfig(enabled=True, prob=1.0, max_shift_ms=2000.0))
    differing = sum(
        0 if np.array_equal(off[i][0], on[i][0]) else 1 for i in range(len(off))
    )
    assert differing > 0, "offsets are enabled but nothing changed"


def test_labels_with_no_arguments_is_unchanged_even_when_offsets_are_on():
    """Every pre-existing caller uses labels() with no arguments and must keep
    seeing the cache's own labels."""
    from src.dataset import RandomOffsetConfig

    cfg = RandomOffsetConfig(enabled=True, prob=1.0, max_shift_ms=3000.0)
    ds = make_ds(cfg)
    raw = np.asarray([ds.cache.label(int(i)) for i in ds.indices], dtype=np.int64)
    assert np.array_equal(ds.labels(), raw)


def test_effective_labels_are_a_no_op_when_offsets_are_disabled():
    ds = make_ds(None)
    assert np.array_equal(ds.labels(), ds.labels(effective=True))


def test_effective_labels_match_what_getitem_actually_returns():
    """pos_weight is computed from effective labels without decoding audio, so
    the replay has to agree with the real path. If it drifts, the loss is
    weighted for a class balance the model is not being shown."""
    from src.dataset import RandomOffsetConfig

    cfg = RandomOffsetConfig(enabled=True, prob=0.6, max_shift_ms=2500.0)
    ds = make_ds(cfg)
    replayed = ds.labels(effective=True)
    actual = np.asarray([ds[i][1] for i in range(len(ds))], dtype=np.int64)
    assert np.array_equal(replayed, actual)


def test_pos_weight_tracks_the_effective_balance():
    """Relabelling turns positives into negatives, so neg/pos must rise.

    prob=0.5 deliberately, not 1.0: at prob=1.0 the only positives that survive
    are those whose sampled shift landed inside the tolerance, which on a small
    set can be none at all -- and pos_weight then returns its 1.0 guard rather
    than a ratio, so the assertion would compare against a fallback. See
    test_a_wide_shift_at_prob_one_can_collapse_the_positive_class.
    """
    from src.dataset import RandomOffsetConfig

    plain = make_ds(None, n=40)
    shifted = make_ds(RandomOffsetConfig(enabled=True, prob=0.5,
                                         max_shift_ms=4000.0), n=40)
    assert plain.pos_weight() == pytest.approx(1.0)
    assert shifted.pos_weight() > plain.pos_weight()


def test_a_wide_shift_at_prob_one_can_collapse_the_positive_class():
    """Documents the degenerate corner, and why training warns about it.

    At prob=1.0 with a 4 s range against a 150 ms tolerance, almost every
    positive is cropped off its boundary and relabelled. The run would still
    train -- on a nearly single-class set -- and would produce a confident model
    that always says "not finished". training/train.py prints a warning below 5%
    for exactly this reason.
    """
    from src.dataset import RandomOffsetConfig

    ds = make_ds(RandomOffsetConfig(enabled=True, prob=1.0,
                                    max_shift_ms=4000.0), n=40)
    assert ds.labels().mean() == pytest.approx(0.5)
    assert ds.labels(effective=True).mean() < 0.05


def test_set_epoch_changes_the_crops_but_stays_reproducible():
    from src.dataset import RandomOffsetConfig

    ds = make_ds(RandomOffsetConfig(enabled=True, prob=1.0, max_shift_ms=3000.0))

    def snapshot():
        # Compare the tensors, not the labels: at a wide shift nearly every
        # label is 0 in every epoch, so labels would match while the crops --
        # the thing set_epoch is supposed to vary -- differ.
        return [ds[i][0].copy() for i in range(len(ds))]

    ds.set_epoch(0)
    e0 = snapshot()
    ds.set_epoch(1)
    e1 = snapshot()
    ds.set_epoch(0)
    e0_again = snapshot()

    assert all(np.array_equal(a, b) for a, b in zip(e0, e0_again)),         "same epoch must reproduce exactly"
    assert any(not np.array_equal(a, b) for a, b in zip(e0, e1)),         "different epochs should sample different crops"


# --------------------------------------------------------------------------- #
# the shipped E1 / E2 configs must stay inert
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", [
    "e1_frozen_linear", "e2_window_0p5", "e2_window_1p0",
    "e2_window_1p5", "e2_window_2p0", "e2_window_4p0",
])
def test_e1_and_e2_configs_have_both_new_features_off(name):
    """If either default flips on, E1 stops being reproducible and every E2 row
    stops being controlled against it."""
    from pathlib import Path

    from training.train import TrainConfig

    cfg = TrainConfig.load(
        Path(__file__).resolve().parents[1] / "configs" / f"{name}.yaml"
    )
    assert cfg.random_offset == {}
    assert cfg.random_offset_config().enabled is False
    assert cfg.hard_negative_file is None


def test_the_dataclass_defaults_are_off():
    from training.train import TrainConfig

    cfg = TrainConfig()
    assert cfg.random_offset_config().enabled is False
    assert cfg.hard_negative_file is None
