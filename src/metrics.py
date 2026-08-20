"""The scoring vocabulary, with the two headline metrics defined precisely.

Accuracy is close to useless on this task and F1 alone is not much better,
because the two error directions have wildly different costs in a live call:

* **False interruption** — the detector says the turn ended while the user is
  still mid-thought, so the agent talks over them. The user stops, gets
  confused, repeats themselves. This is the failure people actually notice and
  the one that makes a voice agent feel broken.
* **Missed endpoint** — the detector waits when the user has finished. The agent
  feels a beat slow. Annoying, far more forgivable.

So the reported operating point is chosen by *cost*, not by max F1. The default
weighting treats one false interruption as worth roughly four missed endpoints;
:data:`DEFAULT_FP_COST` says so in one place and the report quotes it.

Naming. The brief asks for "false interruption rate" and "time-to-decide". In
this module those are:

    false_interruption_rate = FP / (FP + TN)     # among turns that had NOT ended
    missed_endpoint_rate    = FN / (FN + TP)     # among turns that HAD ended
    time_to_decide          = wall-clock ms from turn end to the decision

``time_to_decide`` cannot be computed from clip-level classification alone — it
is a property of the streaming detector, so it is measured in
:mod:`src.streaming` and carried here for reporting rather than derived.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

DEFAULT_FP_COST = 4.0
"""How many missed endpoints one false interruption is worth.

4.0 is a judgement call, not a measurement, and it is stated here so a reviewer
can disagree with a number rather than with a vibe. The reasoning: a missed
endpoint costs the user a fraction of a second of dead air; a false interruption
costs a derailed turn plus the user's re-statement, which in the logs of the
voice-agent work ran 3-6x the duration of the dead air it replaced.
"""


# --------------------------------------------------------------------------- #
# confusion matrix and the rates derived from it
# --------------------------------------------------------------------------- #
@dataclass
class Confusion:
    """Counts at one threshold. Positive class = "the turn has ended"."""

    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def accuracy(self) -> float:
        return _div(self.tp + self.tn, self.n)

    @property
    def precision(self) -> float:
        return _div(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        """Also the true-positive rate: of turns that ended, how many we caught."""
        return _div(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return _div(2 * p * r, p + r)

    @property
    def specificity(self) -> float:
        return _div(self.tn, self.tn + self.fp)

    @property
    def false_interruption_rate(self) -> float:
        """FPR. Of turns that had *not* ended, the share we cut off.

        This is the headline number. A detector that never fires scores 0 here,
        which is why it is always read next to :attr:`recall`.
        """
        return _div(self.fp, self.fp + self.tn)

    @property
    def missed_endpoint_rate(self) -> float:
        """FNR. Of turns that *had* ended, the share we sat through."""
        return _div(self.fn, self.fn + self.tp)

    @property
    def balanced_accuracy(self) -> float:
        return 0.5 * (self.recall + self.specificity)

    def cost(self, fp_cost: float = DEFAULT_FP_COST) -> float:
        """Weighted error rate used to pick the operating point.

        Rates rather than raw counts, so the value does not move when the test
        set's class balance changes.
        """
        return fp_cost * self.false_interruption_rate + self.missed_endpoint_rate

    def to_dict(self) -> dict:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "n": self.n,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "specificity": self.specificity,
            "false_interruption_rate": self.false_interruption_rate,
            "missed_endpoint_rate": self.missed_endpoint_rate,
            "balanced_accuracy": self.balanced_accuracy,
            "cost": self.cost(),
        }

    def __str__(self) -> str:
        return (
            f"n={self.n:,d}  acc={self.accuracy:.4f}  f1={self.f1:.4f}  "
            f"false_interrupt={self.false_interruption_rate:.4f}  "
            f"missed={self.missed_endpoint_rate:.4f}"
        )

    def matrix_str(self) -> str:
        """The 2x2, laid out so it cannot be misread."""
        return (
            "                 pred: not-ended    pred: ended\n"
            f"  true not-ended   {self.tn:>10,d}     {self.fp:>10,d}  <- false interruptions\n"
            f"  true ended       {self.fn:>10,d}     {self.tp:>10,d}\n"
            "                          ^ missed endpoints"
        )


def _div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def confusion_at(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> Confusion:
    """Confusion counts for ``y_prob >= threshold``.

    ``>=`` rather than ``>`` so that a threshold of 0.0 predicts everything
    positive, which is the degenerate corner a sweep should be able to reach.
    """
    y_true = np.asarray(y_true).astype(bool)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_prob.shape}")
    pred = y_prob >= threshold
    return Confusion(
        tp=int(np.sum(pred & y_true)),
        fp=int(np.sum(pred & ~y_true)),
        tn=int(np.sum(~pred & ~y_true)),
        fn=int(np.sum(~pred & y_true)),
    )


# --------------------------------------------------------------------------- #
# threshold-independent summaries
# --------------------------------------------------------------------------- #
def roc_auc(y_true, y_prob) -> float:
    """Area under the ROC curve, threshold-free.

    Reported alongside the operating point because AUC answers "does the model
    rank clips correctly" while the operating point answers "is it usable at one
    setting". A model can win on AUC and still be unusable if the good region of
    its curve sits at an unacceptable false-interruption rate.
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")  # undefined on a single-class slice; say so, don't fake it
    return float(roc_auc_score(y_true, np.asarray(y_prob, dtype=np.float64)))


def pr_auc(y_true, y_prob) -> float:
    """Average precision. More informative than ROC-AUC under class imbalance."""
    from sklearn.metrics import average_precision_score

    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, np.asarray(y_prob, dtype=np.float64)))


def sweep(
    y_true,
    y_prob,
    thresholds: np.ndarray | None = None,
) -> list[tuple[float, Confusion]]:
    """Confusion at each threshold.

    Defaults to the midpoints between observed unique scores plus the two
    degenerate ends. Sweeping a fixed 0..1 grid instead would miss the threshold
    that actually matters whenever a model's scores cluster — and a well-trained
    binary classifier's scores always cluster.
    """
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if thresholds is None:
        uniq = np.unique(y_prob)
        if uniq.size > 1:
            mids = (uniq[:-1] + uniq[1:]) / 2.0
        else:
            mids = uniq
        thresholds = np.concatenate([[0.0], mids, [np.nextafter(1.0, 2.0)]])
        if thresholds.size > 2000:  # keep the sweep cheap on 30k-row test sets
            idx = np.linspace(0, thresholds.size - 1, 2000).astype(int)
            thresholds = thresholds[idx]
    return [(float(t), confusion_at(y_true, y_prob, float(t))) for t in thresholds]


DEFAULT_FI_BUDGET = 0.10
"""The shared false-interruption ceiling every reported row is held to.

A single ceiling across all detectors is what makes the experiment table
comparable. If each row picked its own favourite tradeoff, E0 would quote its
best-F1 point (78% of turns interrupted) and a good model would quote a
conservative one, and the table would rank them backwards. 10% is a judgement
call — one interruption per ten held turns — stated here so it can be argued
with as a number.
"""

MIN_USEFUL_RECALL = 0.05
"""Below this the detector has stopped being a detector.

Why this guard has to exist. ``cost = fp_cost * FPR + FNR`` is minimised by
*never firing* whenever a detector is weak: never firing scores ``4*0 + 1 = 1.0``,
and a detector whose ROC-AUC is only ~0.65 cannot buy a full point of FNR for
less than a quarter-point of FPR. So the arithmetic correctly concludes "the
safest thing to do with a bad endpoint detector is never interrupt anyone" — a
true statement that yields F1 = 0 and a table row that cannot be compared with
anything.

That degenerate corner is a *finding*, not a number to report as the operating
point. :func:`pick_threshold` therefore excludes it by default and
:func:`operating_points` reports it explicitly when it occurs.
"""


def pick_threshold(
    y_true,
    y_prob,
    fp_cost: float = DEFAULT_FP_COST,
    max_false_interruption: float | None = None,
    min_recall: float = MIN_USEFUL_RECALL,
) -> tuple[float, Confusion]:
    """Choose the operating point.

    Two modes, and the second is the one to prefer when a product constraint
    exists:

    * ``max_false_interruption=None`` — minimise ``fp_cost * FPR + FNR`` over
      thresholds that still fire (see :data:`MIN_USEFUL_RECALL`).
    * ``max_false_interruption=0.05`` — among thresholds whose false-interruption
      rate is at or below 5%, take the one with the best recall. This is how a
      real deployment picks: the interruption budget is a hard product
      requirement and accuracy is optimised inside it.

    ``min_recall=0.0`` disables the degeneracy guard, which is what
    :func:`operating_points` uses when it deliberately wants to *show* the
    degenerate corner.
    """
    points = sweep(y_true, y_prob)
    if not points:
        raise ValueError("empty sweep")

    usable = [tc for tc in points if tc[1].recall >= min_recall] or points

    if max_false_interruption is not None:
        feasible = [
            tc for tc in usable
            if tc[1].false_interruption_rate <= max_false_interruption
        ]
        if feasible:
            return max(feasible, key=lambda tc: (tc[1].recall, -tc[0]))

    return min(usable, key=lambda tc: (tc[1].cost(fp_cost), tc[0]))


def operating_points(
    y_true,
    y_prob,
    fp_cost: float = DEFAULT_FP_COST,
    fi_budget: float = DEFAULT_FI_BUDGET,
) -> dict:
    """Report every operating point that matters, including the degenerate one.

    Three points, because no single one tells the whole story for a detector
    whose ROC curve is mediocre:

    * ``best_f1`` — the point a conventional classifier report would quote.
    * ``fi_budget`` — best recall subject to a hard false-interruption ceiling.
      This is the one that is comparable across detectors and the one the
      experiment table uses, because every detector is held to the same ceiling.
    * ``cost_optimal`` — minimiser of the weighted cost, with the guard *off*.
      ``degenerate`` is True when this collapses to "never fire", which is the
      honest signal that the detector is not yet good enough to be worth
      interrupting anyone with.
    """
    points = sweep(y_true, y_prob)
    best_f1 = max(points, key=lambda tc: (tc[1].f1, -tc[0]))
    raw_cost = min(points, key=lambda tc: (tc[1].cost(fp_cost), tc[0]))
    budgeted = pick_threshold(y_true, y_prob, fp_cost, fi_budget)
    return {
        "best_f1": {"threshold": best_f1[0], **best_f1[1].to_dict()},
        "fi_budget": {
            "threshold": budgeted[0],
            "budget": fi_budget,
            "satisfied": budgeted[1].false_interruption_rate <= fi_budget,
            **budgeted[1].to_dict(),
        },
        "cost_optimal": {
            "threshold": raw_cost[0],
            "degenerate": raw_cost[1].recall < MIN_USEFUL_RECALL,
            **raw_cost[1].to_dict(),
        },
    }


# --------------------------------------------------------------------------- #
# the object that becomes an experiment-table row
# --------------------------------------------------------------------------- #
@dataclass
class Evaluation:
    """A complete scored result for one model on one slice."""

    name: str
    n: int
    threshold: float
    confusion: dict
    roc_auc: float
    pr_auc: float
    positive_rate: float
    # Filled by src.streaming / src.optimize; None means "not measured yet",
    # which is materially different from zero and is rendered as "-".
    time_to_decide_p50_ms: float | None = None
    time_to_decide_p95_ms: float | None = None
    cpu_latency_p50_ms: float | None = None
    cpu_latency_p95_ms: float | None = None
    size_mb: float | None = None
    params_m: float | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def row(self) -> dict:
        """One flat dict, the shape the experiment table wants."""
        c = self.confusion
        return {
            "id": self.name,
            "n": self.n,
            "threshold": round(self.threshold, 4),
            "acc": round(c["accuracy"], 4),
            "f1": round(c["f1"], 4),
            "precision": round(c["precision"], 4),
            "recall": round(c["recall"], 4),
            "false_interrupt": round(c["false_interruption_rate"], 4),
            "missed": round(c["missed_endpoint_rate"], 4),
            "roc_auc": None if _isnan(self.roc_auc) else round(self.roc_auc, 4),
            "pr_auc": None if _isnan(self.pr_auc) else round(self.pr_auc, 4),
            "ttd_p50_ms": _r(self.time_to_decide_p50_ms),
            "cpu_p50_ms": _r(self.cpu_latency_p50_ms),
            "cpu_p95_ms": _r(self.cpu_latency_p95_ms),
            "size_mb": _r(self.size_mb, 2),
            "params_m": _r(self.params_m, 2),
        }


def _isnan(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def _r(x, nd: int = 1):
    return None if x is None else round(float(x), nd)


def evaluate(
    name: str,
    y_true,
    y_prob,
    threshold: float | None = None,
    fp_cost: float = DEFAULT_FP_COST,
    max_false_interruption: float | None = DEFAULT_FI_BUDGET,
    **extra,
) -> Evaluation:
    """Score predictions and package them as an :class:`Evaluation`.

    ``threshold=None`` selects the operating point from the data via
    :func:`pick_threshold`. Pass an explicit threshold when scoring a *held-out*
    set with a threshold chosen on validation — which is the correct protocol,
    and the reason this argument exists.

    ``max_false_interruption`` defaults to :data:`DEFAULT_FI_BUDGET` rather than
    ``None`` so every detector in the table is held to the same ceiling.
    A shared ceiling is what makes the rows comparable; letting each detector
    pick its own favourite tradeoff does not.

    The full set of operating points is always attached under
    ``extra["operating_points"]``, so the degenerate corner stays visible even
    though it is not what gets reported.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    ops = operating_points(y_true, y_prob, fp_cost, max_false_interruption or DEFAULT_FI_BUDGET)

    if threshold is None:
        threshold, conf = pick_threshold(
            y_true, y_prob, fp_cost, max_false_interruption
        )
    else:
        conf = confusion_at(y_true, y_prob, threshold)

    return Evaluation(
        name=name,
        n=int(y_true.size),
        threshold=float(threshold),
        confusion=conf.to_dict(),
        roc_auc=roc_auc(y_true, y_prob),
        pr_auc=pr_auc(y_true, y_prob),
        positive_rate=float(y_true.mean()) if y_true.size else 0.0,
        extra={**extra, "operating_points": ops},
    )


def percentiles(values: Sequence[float]) -> dict[str, float]:
    """p50/p95/p99/max. Never a mean.

    A mean latency hides the tail, and the tail is what a user hears. This is the
    one reporting habit carried over wholesale from the voice-agent work.
    """
    a = np.asarray(list(values), dtype=np.float64)
    if a.size == 0:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "n": 0}
    return {
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
        "n": int(a.size),
    }
