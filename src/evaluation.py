"""Scoring runs, slicing results, and appending to the experiment table.

Manual bookkeeping is where experiment tables quietly become fiction: a run gets
re-run with a tweaked config, the row does not get updated, and by day nine the
table describes a model that no longer exists. So nothing here asks a human to
copy a number. :class:`ExperimentTable` is append-only, keyed by run id, and
every row records the config hash that produced it.

The slicing here is doing real work rather than decoration. The corpus ships
``midfiller``, ``endfiller``, ``synthetic`` and ``language`` flags, which means
the error modes the plan expected to hand-categorise on Day 9 are already
labelled. Breaking the score out along them turns "the model struggles with
hesitation" from a hunch into a number.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .metrics import Evaluation, evaluate

ROW_ORDER = (
    "id",
    "model",
    "window",
    "n",
    "threshold",
    "acc",
    "f1",
    "precision",
    "recall",
    "false_interrupt",
    "missed",
    "roc_auc",
    "pr_auc",
    "ttd_p50_ms",
    "cpu_p50_ms",
    "cpu_p95_ms",
    "size_mb",
    "params_m",
    "notes",
)


def config_hash(cfg: dict) -> str:
    """Short stable hash of a config, so a row can be traced to its run."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:10]


# --------------------------------------------------------------------------- #
# the table
# --------------------------------------------------------------------------- #
class ExperimentTable:
    """Append-only CSV plus a rendered Markdown view.

    Two files rather than one: the CSV is what code reads and appends to, the
    Markdown is what goes in the report. Generating the Markdown from the CSV
    means the report can never drift from the data.
    """

    def __init__(self, path: str | Path = "artifacts/experiments.csv") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))

    def append(self, row: dict) -> None:
        """Add a row. Re-running the same id replaces it rather than duplicating."""
        rows = [r for r in self.rows() if r.get("id") != str(row.get("id"))]
        clean = {k: ("" if row.get(k) is None else row.get(k)) for k in ROW_ORDER}
        for k, v in row.items():  # keep any extra columns a caller added
            if k not in clean:
                clean[k] = "" if v is None else v
        rows.append(clean)

        fields = list(ROW_ORDER) + [
            k for k in clean if k not in ROW_ORDER
        ]
        with self.path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in sorted(rows, key=_row_sort_key):
                w.writerow(r)

    def add_evaluation(
        self,
        ev: Evaluation,
        model: str,
        window: str,
        notes: str = "",
        **extra,
    ) -> dict:
        row = ev.row()
        row.update({"model": model, "window": window, "notes": notes})
        row.update(extra)
        self.append(row)
        return row

    def to_markdown(self, columns: Sequence[str] | None = None) -> str:
        rows = self.rows()
        if not rows:
            return "_No runs recorded yet._"
        cols = list(columns or ROW_ORDER)
        cols = [c for c in cols if any(r.get(c) not in (None, "") for r in rows)]
        head = "| " + " | ".join(cols) + " |"
        rule = "|" + "|".join("---" for _ in cols) + "|"
        body = [
            "| " + " | ".join(str(r.get(c, "") or "—") for c in cols) + " |"
            for r in sorted(rows, key=_row_sort_key)
        ]
        return "\n".join([head, rule, *body])

    def default_markdown_path(self) -> Path:
        """Where the markdown goes when the caller does not say.

        Derived from the CSV this table was opened on, not hardcoded. Every call
        site passes no argument, so a hardcoded default meant a run pointed at a
        scratch ``table_path`` would still overwrite ``report/experiments.md``
        with its throwaway rows -- a silent corruption of the real table by a
        sandboxed run. Deriving it keeps a sandboxed table's output sandboxed.
        """
        canonical = Path("artifacts/experiments.csv")
        try:
            is_canonical = self.path.resolve() == (
                Path(__file__).resolve().parents[1] / canonical).resolve()
        except OSError:
            is_canonical = False
        if is_canonical:
            return Path(__file__).resolve().parents[1] / "report" / "experiments.md"
        return self.path.with_suffix(".md")

    def save_markdown(self, path: str | Path | None = None) -> Path:
        p = Path(path) if path is not None else self.default_markdown_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            "# Experiment matrix\n\n"
            f"Generated from `{self.path.as_posix()}` — do not edit by hand.\n\n"
            + self.to_markdown()
            + "\n",
            encoding="utf-8",
        )
        return p

    def print_row(self, row: dict) -> None:
        """The single-line summary a training run prints when it finishes.

        Values are stringified first: a row read back from CSV holds strings, but
        a row straight out of :meth:`add_evaluation` holds floats, and a width
        format spec applied to a float raises.
        """

        def cell(key: str, width: int) -> str:
            v = row.get(key)
            if v is None or v == "":
                v = "—"
            elif isinstance(v, float):
                v = f"{v:.4f}"
            return f"{str(v):<{width}s}"

        print(
            f"\n  {cell('id', 6)} {str(row.get('model', ''))[:34]:<34s} "
            f"win={cell('window', 5)} "
            f"f1={cell('f1', 7)} "
            f"false_interrupt={cell('false_interrupt', 7)} "
            f"missed={cell('missed', 7)}",
            flush=True,
        )


def _row_sort_key(r: dict):
    """Sort E0, E0b, E1, E2 ... naturally rather than lexically."""
    rid = str(r.get("id", ""))
    digits = "".join(c for c in rid if c.isdigit())
    return (int(digits) if digits else 999, rid)


# --------------------------------------------------------------------------- #
# scoring a predictor
# --------------------------------------------------------------------------- #
def predict_over(predictor, cache, indices, batch_size: int = 32) -> np.ndarray:
    return predictor.predict_cached(cache, indices, batch_size=batch_size)


def evaluate_predictor(
    predictor,
    cache,
    indices,
    name: str,
    threshold: float | None = None,
    max_false_interruption: float | None = None,
    batch_size: int = 32,
) -> tuple[Evaluation, np.ndarray]:
    """Score a predictor over cached rows.

    Returns ``(evaluation, probabilities)`` — the probabilities come back so that
    slice reports and error analysis do not have to re-run the model.
    """
    idx = np.asarray(indices, dtype=np.int64)
    probs = predict_over(predictor, cache, idx, batch_size)
    y = np.asarray([cache.label(int(i)) for i in idx])
    ev = evaluate(
        name,
        y,
        probs,
        threshold=threshold,
        max_false_interruption=max_false_interruption,
    )
    info = predictor.info()
    ev.size_mb = info.size_mb
    ev.params_m = info.params_m
    return ev, probs


# --------------------------------------------------------------------------- #
# slices
# --------------------------------------------------------------------------- #
@dataclass
class Slice:
    name: str
    indices: np.ndarray
    note: str = ""


def build_slices(cache, indices) -> list[Slice]:
    """The breakdowns the report shows as separate tables.

    Kept separate and never averaged into the headline number. Averaging a
    synthetic-audio score into a human-audio score produces a figure that
    describes neither.
    """
    idx = np.asarray(indices, dtype=np.int64)
    meta = [cache.meta[int(i)] for i in idx]
    out: list[Slice] = []

    def sub(mask, name, note=""):
        m = np.asarray(mask, dtype=bool)
        if m.sum() >= 25:  # below this a rate is noise, not a measurement
            out.append(Slice(name, idx[m], note))

    langs: dict[str, int] = {}
    for m in meta:
        langs[m.get("language")] = langs.get(m.get("language"), 0) + 1
    for lang, _ in sorted(langs.items(), key=lambda kv: -kv[1])[:8]:
        sub([m.get("language") == lang for m in meta], f"lang={lang}")

    for lang in ("hin", "ben", "mar"):
        if lang in langs:
            sub(
                [m.get("language") == lang for m in meta],
                f"indic={lang}",
                "Indic slice — reported separately, never averaged in",
            )

    # midfiller/endfiller are tri-state in the corpus: True, False, or null for
    # rows that were never annotated. Folding null into False would silently mix
    # "annotated as having no filler" with "unknown", and the filler slices are
    # the ones the error analysis leans on hardest — so null gets its own slice.
    sub([m.get("midfiller") is True for m in meta], "midfiller=True",
        "mid-utterance hesitation: the false-interruption trap")
    sub([m.get("endfiller") is True for m in meta], "endfiller=True",
        "trailing filler ('umm...', 'matlab...') — labelled not-ended in the corpus")
    sub([m.get("midfiller") is False and m.get("endfiller") is False for m in meta],
        "no_filler(annotated)", "annotated as carrying neither filler")
    sub([m.get("midfiller") is None and m.get("endfiller") is None for m in meta],
        "filler=unannotated", "no filler annotation — excluded from filler claims")

    sub([m.get("synthetic") is True for m in meta], "synthetic=True",
        "TTS audio — clean and evenly paced, treat as an upper bound")
    sub([m.get("synthetic") is False for m in meta], "synthetic=False",
        "human-recorded audio — the honest headline slice")

    dur = np.asarray([float(m.get("audioduration") or 0.0) for m in meta])
    sub(dur < 1.5, "duration<1.5s", "short utterances")
    sub(dur >= 6.0, "duration>=6s", "long utterances")

    for src in sorted({str(m.get("dataset")) for m in meta}):
        sub([str(m.get("dataset")) == src for m in meta], f"source={src}")

    return out


def slice_report(
    cache,
    indices,
    probs: np.ndarray,
    threshold: float,
    min_n: int = 25,
) -> list[dict]:
    """Score every slice at a *fixed* threshold.

    Fixed, not re-optimised per slice: re-picking the threshold on each slice
    would report the best case for every subgroup and describe a system that
    cannot exist, since one deployed detector has one threshold.
    """
    from .metrics import confusion_at

    idx = np.asarray(indices, dtype=np.int64)
    pos = {int(v): k for k, v in enumerate(idx)}
    rows: list[dict] = []

    for sl in build_slices(cache, idx):
        take = np.asarray([pos[int(i)] for i in sl.indices], dtype=np.int64)
        if take.size < min_n:
            continue
        y = np.asarray([cache.label(int(i)) for i in sl.indices])
        p = probs[take]
        c = confusion_at(y, p, threshold)
        rows.append(
            {
                "slice": sl.name,
                "n": int(take.size),
                "positive_rate": round(float(y.mean()), 3),
                "f1": round(c.f1, 4),
                "recall": round(c.recall, 4),
                "false_interrupt": round(c.false_interruption_rate, 4),
                "missed": round(c.missed_endpoint_rate, 4),
                "note": sl.note,
            }
        )
    return sorted(rows, key=lambda r: -r["n"])


def slices_to_markdown(rows: Iterable[dict], title: str = "Per-slice results") -> str:
    rows = list(rows)
    if not rows:
        return f"### {title}\n\n_No slice met the minimum sample count._\n"
    cols = ["slice", "n", "positive_rate", "f1", "recall", "false_interrupt", "missed", "note"]
    lines = [
        f"### {title}",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def save_json(obj, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return p
