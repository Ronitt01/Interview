"""Leakage tests. The Day-2 requirement that leakage be asserted in code.

These are the tests that would have caught the mistake if the split logic were
wrong, which is the only reason to write them. Each one states the mistake it
catches.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.splits import (
    LeakageError,
    SplitReport,
    assert_no_leakage,
    assign_groups,
    build_report,
    group_key,
)


def rows(spec):
    """``spec`` is a list of ``(dataset, n_rows, n_positive)``."""
    out, k = [], 0
    for name, n, pos in spec:
        for i in range(n):
            out.append(
                {
                    "id": f"clip-{k}",
                    "dataset": name,
                    "language": "eng",
                    "endpoint_bool": i < pos,
                }
            )
            k += 1
    return out


CORPUS = [
    ("chirp3_1", 300, 150),
    ("chirp3_2", 200, 100),
    ("liva_1", 100, 50),
    ("midcentury_1", 60, 30),
    ("rime_2", 40, 20),
    ("human_5", 20, 10),
]


def test_group_key_is_stable_and_explicit():
    r = {"dataset": "liva_1", "language": "hin"}
    assert group_key(r, ("dataset",)) == "liva_1"
    assert group_key(r, ("dataset", "language")) == "liva_1|hin"
    # A missing group column must raise rather than silently produce "None",
    # which would put every unkeyed row into one giant group.
    with pytest.raises(KeyError):
        group_key(r, ("speaker_id",))


def test_every_group_lands_in_exactly_one_split():
    """The mistake: a group's rows spread across train and val."""
    all_rows = rows(CORPUS)
    sizes = {}
    for r in all_rows:
        g = r["dataset"]
        n, p = sizes.get(g, (0, 0))
        sizes[g] = (n + 1, p + int(r["endpoint_bool"]))

    assignment = assign_groups(sizes, {"train": 0.8, "val": 0.2}, seed=0)
    assert set(assignment) == set(sizes)
    assert set(assignment.values()) <= {"train", "val"}

    by_split = {"train": [], "val": []}
    for r in all_rows:
        by_split[assignment[r["dataset"]]].append(r)
    assert_no_leakage(by_split, ("dataset",))  # must not raise


def test_leakage_is_detected_when_a_group_spans_splits():
    """The test that proves the assertion actually asserts."""
    a = [{"id": "1", "dataset": "liva_1", "endpoint_bool": True}]
    b = [{"id": "2", "dataset": "liva_1", "endpoint_bool": False}]
    with pytest.raises(LeakageError, match="group leakage"):
        assert_no_leakage({"train": a, "val": b}, ("dataset",))


def test_duplicate_clip_id_is_detected_even_when_groups_differ():
    """The mistake grouping alone cannot catch: the same clip present twice.

    If a corpus lists one recording under two source names, grouping by source
    puts it in both splits and the group check passes. The id check is what
    catches it.
    """
    a = [{"id": "dupe", "dataset": "corpus_a", "endpoint_bool": True}]
    b = [{"id": "dupe", "dataset": "corpus_b", "endpoint_bool": True}]
    with pytest.raises(LeakageError, match="clip-id leakage"):
        assert_no_leakage({"train": a, "val": b}, ("dataset",))


def test_split_respects_requested_proportions_approximately():
    """Whole-group assignment cannot hit a fraction exactly; it must get close.

    The tolerance is wide on purpose — with six groups of very unequal size,
    demanding tighter would be demanding something the data cannot give. The
    report is what states the achieved number.
    """
    all_rows = rows(CORPUS)
    sizes = {}
    for r in all_rows:
        g = r["dataset"]
        n, p = sizes.get(g, (0, 0))
        sizes[g] = (n + 1, p + int(r["endpoint_bool"]))

    assignment = assign_groups(sizes, {"train": 0.8, "val": 0.2}, seed=0)
    rep = build_report(all_rows, assignment, ("dataset",))
    total = sum(rep.counts.values())
    assert total == len(all_rows)
    val_frac = rep.counts.get("val", 0) / total
    assert 0.05 < val_frac < 0.45, f"val fraction {val_frac:.3f} is nowhere near 0.2"


def test_split_keeps_class_balance_close():
    """A split that is size-correct but class-skewed is still a bad split."""
    all_rows = rows(CORPUS)
    sizes = {}
    for r in all_rows:
        g = r["dataset"]
        n, p = sizes.get(g, (0, 0))
        sizes[g] = (n + 1, p + int(r["endpoint_bool"]))
    assignment = assign_groups(sizes, {"train": 0.8, "val": 0.2}, seed=0)
    rep = build_report(all_rows, assignment, ("dataset",))
    rates = list(rep.positive_rate.values())
    assert max(rates) - min(rates) < 0.15, f"positive rates diverge: {rep.positive_rate}"


def test_assignment_is_deterministic_for_a_seed():
    sizes = {"a": (100, 50), "b": (80, 40), "c": (60, 30), "d": (40, 20)}
    first = assign_groups(sizes, {"train": 0.75, "val": 0.25}, seed=7)
    second = assign_groups(sizes, {"train": 0.75, "val": 0.25}, seed=7)
    assert first == second


def test_report_states_the_speaker_limitation_rather_than_hiding_it():
    rep = SplitReport(
        group_keys=("dataset",),
        counts={"train": 100, "val": 25},
        positive_rate={"train": 0.5, "val": 0.5},
        n_groups={"train": 3, "val": 2},
        speaker_ids_available=False,
    )
    text = rep.limitations()
    assert "no speaker identifier" in text
    assert "known limitation" in text
    # It must not claim something it cannot support.
    assert "speaker-aware" in text and "not possible" in text


def test_rejects_bad_fractions():
    sizes = {"a": (10, 5)}
    with pytest.raises(ValueError):
        assign_groups(sizes, {"train": -0.5, "val": 0.5})
    with pytest.raises(ValueError):
        assign_groups(sizes, {"train": 0.0, "val": 0.0})
    with pytest.raises(ValueError):
        assign_groups({}, {"train": 1.0})
