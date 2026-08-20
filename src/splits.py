"""Group-aware partitioning, and the assertion that proves it held.

The honest situation, established on Day 1 by reading the dataset card rather
than by assuming:

* ``smart-turn-data-v3.2-train`` (270,946 rows) and ``smart-turn-data-v3.1-test``
  (31,473 rows) are published as separate datasets. So the **test set is given**
  and we do not invent one — inventing a split when the authors published one is
  how a submission ends up with numbers nobody can compare.
* Neither dataset carries a **speaker ID**. The columns are ``id`` (a per-clip
  UUID), ``language``, ``endpoint_bool``, ``midfiller``, ``endfiller``,
  ``synthetic``, ``dataset``, ``audioduration``, ``spoken_text``. A
  speaker-aware split is therefore *impossible*, and this module does not
  pretend otherwise.
* The coarsest honest grouping available is ``dataset`` — the source corpus
  (~12 values: Pipecat team, Liva AI, Midcentury, MundoAI, and others). Clips
  from one corpus share speakers, recording chain, and TTS voices, so holding a
  corpus out is the strongest generalisation claim the metadata supports.

What that buys and what it costs is stated in
:meth:`SplitReport.limitations`, and the report reproduces it verbatim. The cost
is real: holding out whole corpora means the validation set is drawn from a
different distribution than training, so validation scores read lower than a
random split would give. That is the point — a random split here would leak
speakers and flatter the model.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

DEFAULT_GROUP_KEYS: tuple[str, ...] = ("dataset",)
"""Columns whose combination defines a group that must not span splits."""


class LeakageError(AssertionError):
    """Raised when a group appears in more than one split.

    An exception type of its own so the test suite can assert on it and a
    training run cannot swallow it as a generic assertion.
    """


@dataclass
class SplitReport:
    """Everything a reviewer needs to judge whether the split is defensible."""

    group_keys: tuple[str, ...]
    counts: dict[str, int]
    positive_rate: dict[str, float]
    n_groups: dict[str, int]
    group_assignment: dict[str, str] = field(default_factory=dict)
    speaker_ids_available: bool = False

    def limitations(self) -> str:
        """The paragraph that goes in the report. Stated, not buried."""
        if self.speaker_ids_available:
            return "Splits are speaker-aware: no speaker appears in two splits."
        return (
            "The Smart Turn corpus carries no speaker identifier, so a "
            "speaker-aware split is not possible. Partitioning is grouped by "
            f"{'+'.join(self.group_keys)} instead — the source corpus — which "
            "prevents a corpus (and therefore its speakers, recording chain and "
            "TTS voices) from spanning two splits. Residual risk: if the same "
            "human speaker contributed to two different source corpora, that "
            "speaker can still appear on both sides. Nothing in the published "
            "metadata lets us detect or rule that out, so this is reported as a "
            "known limitation rather than claimed as clean."
        )

    def to_dict(self) -> dict:
        return {
            "group_keys": list(self.group_keys),
            "counts": self.counts,
            "positive_rate": self.positive_rate,
            "n_groups": self.n_groups,
            "speaker_ids_available": self.speaker_ids_available,
            "limitations": self.limitations(),
            "group_assignment": self.group_assignment,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def __str__(self) -> str:
        lines = [f"split grouped by {'+'.join(self.group_keys)}"]
        for name in self.counts:
            lines.append(
                f"  {name:<6s} n={self.counts[name]:>7,d}  "
                f"pos={self.positive_rate[name]:.3f}  "
                f"groups={self.n_groups[name]}"
            )
        return "\n".join(lines)


def group_key(row: Mapping, keys: Sequence[str] = DEFAULT_GROUP_KEYS) -> str:
    """Stable string key for the group a row belongs to."""
    missing = [k for k in keys if k not in row]
    if missing:
        raise KeyError(
            f"group key column(s) {missing} not in row; available: {sorted(row)}"
        )
    return "|".join(str(row[k]) for k in keys)


def assign_groups(
    groups: Mapping[str, tuple[int, int]],
    fractions: Mapping[str, float],
    seed: int = 0,
) -> dict[str, str]:
    """Assign whole groups to splits, balancing size *and* class ratio.

    ``groups`` maps group key → ``(n_rows, n_positive)``. ``fractions`` maps
    split name → target share of rows.

    Greedy: process groups largest-first and give each to whichever split is
    furthest below its row quota, breaking ties toward the split whose positive
    rate would move closest to the corpus rate. Largest-first matters because a
    handful of corpora dominate the row count — assigning those last would leave
    no room to balance around them.

    With ~12 groups this cannot hit the requested fractions exactly. It gets
    close and :class:`SplitReport` reports what it actually achieved, which is
    the number that belongs in the write-up.
    """
    if not groups:
        raise ValueError("no groups to assign")
    bad = {k: v for k, v in fractions.items() if v < 0}
    if bad:
        raise ValueError(f"negative split fractions: {bad}")
    total_frac = sum(fractions.values())
    if total_frac <= 0:
        raise ValueError("split fractions sum to zero")

    total_rows = sum(n for n, _ in groups.values())
    total_pos = sum(p for _, p in groups.values())
    corpus_rate = total_pos / total_rows if total_rows else 0.0
    quota = {name: total_rows * (f / total_frac) for name, f in fractions.items()}

    rng = np.random.default_rng(seed)
    # Deterministic order: size desc, then key asc so ties never depend on dict
    # iteration order. The rng only breaks exact numeric ties below.
    ordered = sorted(groups.items(), key=lambda kv: (-kv[1][0], kv[0]))

    placed_rows: dict[str, int] = {name: 0 for name in fractions}
    placed_pos: dict[str, int] = {name: 0 for name in fractions}
    assignment: dict[str, str] = {}

    for gkey, (n_rows, n_pos) in ordered:
        best_name, best_score = None, None
        for name in fractions:
            deficit = quota[name] - placed_rows[name]
            # Primary: fill the emptiest split. Secondary: keep positive rate
            # near the corpus rate, scaled so it only ever breaks near-ties.
            after_rows = placed_rows[name] + n_rows
            after_rate = (placed_pos[name] + n_pos) / after_rows if after_rows else 0.0
            score = deficit - abs(after_rate - corpus_rate) * n_rows * 0.25
            score += rng.random() * 1e-6
            if best_score is None or score > best_score:
                best_name, best_score = name, score
        assert best_name is not None
        assignment[gkey] = best_name
        placed_rows[best_name] += n_rows
        placed_pos[best_name] += n_pos

    return assignment


def build_report(
    rows: Iterable[Mapping],
    assignment: Mapping[str, str],
    group_keys: Sequence[str] = DEFAULT_GROUP_KEYS,
    label_column: str = "endpoint_bool",
) -> SplitReport:
    """Tally what the assignment actually produced."""
    counts: Counter[str] = Counter()
    positives: Counter[str] = Counter()
    groups_per_split: defaultdict[str, set[str]] = defaultdict(set)

    for row in rows:
        gkey = group_key(row, group_keys)
        split = assignment.get(gkey)
        if split is None:
            raise KeyError(f"group {gkey!r} has no split assignment")
        counts[split] += 1
        positives[split] += int(bool(row[label_column]))
        groups_per_split[split].add(gkey)

    return SplitReport(
        group_keys=tuple(group_keys),
        counts=dict(counts),
        positive_rate={
            s: (positives[s] / counts[s] if counts[s] else 0.0) for s in counts
        },
        n_groups={s: len(g) for s, g in groups_per_split.items()},
        group_assignment=dict(assignment),
        speaker_ids_available=False,
    )


def assert_no_leakage(
    split_rows: Mapping[str, Iterable[Mapping]],
    group_keys: Sequence[str] = DEFAULT_GROUP_KEYS,
    id_column: str = "id",
) -> None:
    """Fail loudly if any group — or any single clip — spans two splits.

    This is the Day-2 requirement that leakage be asserted *in code, not in
    prose*. Two independent checks, because they catch different mistakes:

    1. **Group overlap** — the split logic itself being wrong.
    2. **Clip-ID overlap** — the same clip present twice, which grouping cannot
       catch if the corpus has duplicate rows under different group values.
    """
    seen_group: dict[str, str] = {}
    collisions: list[str] = []
    for split, rows in split_rows.items():
        for row in rows:
            gkey = group_key(row, group_keys)
            prior = seen_group.setdefault(gkey, split)
            if prior != split:
                collisions.append(f"group {gkey!r} in both {prior!r} and {split!r}")
    if collisions:
        raise LeakageError(
            "group leakage across splits:\n  " + "\n  ".join(sorted(set(collisions))[:20])
        )

    seen_id: dict[str, str] = {}
    dupes: list[str] = []
    for split, rows in split_rows.items():
        for row in rows:
            if id_column not in row:
                continue
            cid = str(row[id_column])
            prior = seen_id.setdefault(cid, split)
            if prior != split:
                dupes.append(f"clip {cid!r} in both {prior!r} and {split!r}")
    if dupes:
        raise LeakageError(
            "clip-id leakage across splits:\n  " + "\n  ".join(sorted(set(dupes))[:20])
        )
