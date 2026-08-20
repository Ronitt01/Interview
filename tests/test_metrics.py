"""Metric definitions, and the degeneracy guard on operating-point selection."""
from __future__ import annotations

import numpy as np
import pytest

from src.metrics import (
    DEFAULT_FI_BUDGET,
    DEFAULT_FP_COST,
    MIN_USEFUL_RECALL,
    Confusion,
    confusion_at,
    evaluate,
    operating_points,
    percentiles,
    pick_threshold,
    pr_auc,
    roc_auc,
    sweep,
)


# --------------------------------------------------------------------------- #
# confusion arithmetic
# --------------------------------------------------------------------------- #
def test_headline_rates_are_the_ones_the_brief_names():
    """false_interruption_rate is FPR; missed_endpoint_rate is FNR.

    Getting these two the wrong way round would invert the entire report, so
    they are pinned against hand-computed values.
    """
    c = Confusion(tp=80, fp=10, tn=90, fn=20)
    assert c.false_interruption_rate == pytest.approx(10 / 100)  # fp / (fp + tn)
    assert c.missed_endpoint_rate == pytest.approx(20 / 100)  # fn / (fn + tp)
    assert c.precision == pytest.approx(80 / 90)
    assert c.recall == pytest.approx(80 / 100)
    assert c.accuracy == pytest.approx(170 / 200)
    assert c.specificity == pytest.approx(90 / 100)


def test_cost_weights_false_interruptions_more_heavily():
    """One false interruption must cost more than one missed endpoint."""
    interrupt = Confusion(tp=50, fp=10, tn=90, fn=50)
    miss = Confusion(tp=40, fp=0, tn=100, fn=60)
    assert interrupt.false_interruption_rate > 0
    assert miss.false_interruption_rate == 0
    # Equal error counts, but the interrupting one must be penalised harder.
    a = Confusion(tp=90, fp=10, tn=90, fn=10)
    b = Confusion(tp=80, fp=0, tn=100, fn=20)
    assert a.cost(DEFAULT_FP_COST) > b.cost(DEFAULT_FP_COST)


def test_empty_denominators_do_not_raise():
    c = Confusion(tp=0, fp=0, tn=0, fn=0)
    assert c.f1 == 0.0 and c.precision == 0.0 and c.recall == 0.0
    assert c.false_interruption_rate == 0.0


def test_threshold_is_inclusive_so_zero_predicts_all_positive():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.0, 0.4, 0.6, 1.0])
    c = confusion_at(y, p, 0.0)
    assert c.tp + c.fp == 4, "threshold 0.0 must predict everything positive"


def test_confusion_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shape mismatch"):
        confusion_at(np.zeros(5), np.zeros(3))


# --------------------------------------------------------------------------- #
# AUC
# --------------------------------------------------------------------------- #
def test_auc_of_a_perfect_ranker_is_one():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert roc_auc(y, p) == pytest.approx(1.0)
    assert pr_auc(y, p) == pytest.approx(1.0)


def test_auc_on_a_single_class_slice_is_nan_not_zero():
    """A per-language slice can easily be single-class. Reporting 0.0 there would
    read as "the model failed on this language" instead of "undefined"."""
    y = np.zeros(20, dtype=int)
    p = np.random.default_rng(0).random(20)
    assert np.isnan(roc_auc(y, p))
    assert np.isnan(pr_auc(y, p))


# --------------------------------------------------------------------------- #
# sweep and selection
# --------------------------------------------------------------------------- #
def test_sweep_uses_observed_scores_not_a_fixed_grid():
    """A fixed 0..1 grid misses the useful threshold when scores cluster — and a
    well-trained binary classifier's scores always cluster."""
    y = np.array([0, 0, 1, 1])
    p = np.array([0.9001, 0.9002, 0.9003, 0.9004])
    points = sweep(y, p)
    best = max(points, key=lambda tc: tc[1].f1)
    assert best[1].f1 == pytest.approx(1.0), "sweep failed to separate clustered scores"


def test_sweep_is_capped_for_large_inputs():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 30_000)
    p = rng.random(30_000)
    assert len(sweep(y, p)) <= 2000


def test_degenerate_never_fire_point_is_excluded_by_default():
    """The guard that matters.

    For a weak detector, `4*FPR + FNR` is minimised by never firing: that scores
    1.0, and a poor ranker cannot buy a full point of FNR for under a quarter of
    FPR. Correct arithmetic, useless operating point — so pick_threshold must
    not return it.
    """
    rng = np.random.default_rng(3)
    n = 3000
    y = rng.integers(0, 2, n)
    # Deliberately weak: heavy overlap between the classes.
    p = np.clip(0.5 + 0.06 * (2 * y - 1) + rng.normal(0, 0.25, n), 0, 1)

    thr, conf = pick_threshold(y, p)
    assert conf.recall >= MIN_USEFUL_RECALL, (
        f"selection returned a non-firing detector (recall {conf.recall:.4f})"
    )

    ops = operating_points(y, p)
    # The degenerate corner still gets *reported*, just not selected.
    assert "degenerate" in ops["cost_optimal"]


def test_guard_can_be_switched_off_to_expose_the_degenerate_corner():
    rng = np.random.default_rng(3)
    n = 2000
    y = rng.integers(0, 2, n)
    p = np.clip(0.5 + 0.04 * (2 * y - 1) + rng.normal(0, 0.3, n), 0, 1)
    _, unguarded = pick_threshold(y, p, min_recall=0.0)
    _, guarded = pick_threshold(y, p, min_recall=MIN_USEFUL_RECALL)
    assert guarded.recall >= unguarded.recall


def test_fi_budget_mode_respects_the_ceiling_and_maximises_recall():
    rng = np.random.default_rng(1)
    n = 4000
    y = rng.integers(0, 2, n)
    p = np.clip(0.5 + 0.3 * (2 * y - 1) + rng.normal(0, 0.2, n), 0, 1)

    thr, conf = pick_threshold(y, p, max_false_interruption=0.05)
    assert conf.false_interruption_rate <= 0.05 + 1e-9

    # Nothing feasible should beat it on recall.
    for _, other in sweep(y, p):
        if other.false_interruption_rate <= 0.05 and other.recall >= MIN_USEFUL_RECALL:
            assert other.recall <= conf.recall + 1e-9


def test_operating_points_reports_all_three():
    rng = np.random.default_rng(2)
    n = 1500
    y = rng.integers(0, 2, n)
    p = np.clip(0.5 + 0.25 * (2 * y - 1) + rng.normal(0, 0.2, n), 0, 1)
    ops = operating_points(y, p)
    assert set(ops) == {"best_f1", "fi_budget", "cost_optimal"}
    # best_f1 must be at least as good on F1 as anything else, by construction.
    assert ops["best_f1"]["f1"] >= ops["fi_budget"]["f1"] - 1e-9
    assert ops["fi_budget"]["budget"] == DEFAULT_FI_BUDGET


# --------------------------------------------------------------------------- #
# evaluate()
# --------------------------------------------------------------------------- #
def test_evaluate_holds_every_detector_to_the_same_ceiling_by_default():
    rng = np.random.default_rng(4)
    n = 3000
    y = rng.integers(0, 2, n)
    p = np.clip(0.5 + 0.28 * (2 * y - 1) + rng.normal(0, 0.2, n), 0, 1)
    ev = evaluate("test", y, p)
    assert ev.confusion["false_interruption_rate"] <= DEFAULT_FI_BUDGET + 1e-9
    assert "operating_points" in ev.extra


def test_explicit_threshold_is_applied_not_reselected():
    """The correct protocol: threshold chosen on val, applied to test."""
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    ev = evaluate("test", y, p, threshold=0.5)
    assert ev.threshold == 0.5
    assert ev.confusion["tp"] == 2 and ev.confusion["fp"] == 0


def test_row_renders_unmeasured_fields_as_none_not_zero():
    """"Not measured yet" and "measured as zero" are materially different."""
    y = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.8, 0.3, 0.7])
    row = evaluate("E1", y, p).row()
    assert row["ttd_p50_ms"] is None
    assert row["cpu_p50_ms"] is None
    assert row["f1"] is not None


# --------------------------------------------------------------------------- #
# latency reporting
# --------------------------------------------------------------------------- #
def test_percentiles_never_reports_a_mean():
    """The habit carried over from the voice-agent work: the tail is what a user
    hears, and a mean hides it."""
    out = percentiles([1, 1, 1, 1, 1, 1, 1, 1, 1, 500])
    assert "mean" not in out
    assert set(out) == {"p50", "p95", "p99", "max", "n"}
    assert out["p50"] == 1.0
    assert out["max"] == 500.0


def test_percentiles_of_empty_is_zeroed_not_an_error():
    out = percentiles([])
    assert out["n"] == 0 and out["p50"] == 0.0
