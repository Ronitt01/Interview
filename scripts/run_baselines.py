"""Gate 3 — put E0 and E0b on the board before any neural work.

    python scripts/run_baselines.py --cache data/cache/test

Prints a filled experiment-table row for each baseline and appends it to
``artifacts/experiments.csv``. Everything after this is measured against these
two numbers; a model reported without them beside it is an unfalsifiable claim.

Both baselines are also shown at all three operating points, because for a weak
detector the cost-optimal point collapses to "never fire" and that fact is worth
seeing rather than hiding behind a single row.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines import (  # noqa: E402
    EnergyBaseline,
    EnergyBaselineConfig,
    SileroBaseline,
    evaluate_baseline,
)
from src.dataset import WaveCache  # noqa: E402
from src.evaluation import ExperimentTable, slice_report, slices_to_markdown  # noqa: E402
from src.metrics import confusion_at  # noqa: E402


def show_operating_points(name: str, ev) -> None:
    ops = ev.extra["operating_points"]
    print(f"\n  {name}  roc_auc={ev.roc_auc:.4f}  pr_auc={ev.pr_auc:.4f}  n={ev.n:,d}")
    for key, label in (
        ("best_f1", "best F1"),
        ("fi_budget", "FI budget (reported)"),
        ("cost_optimal", "cost-optimal"),
    ):
        o = ops[key]
        note = ""
        if key == "cost_optimal" and o.get("degenerate"):
            note = "  <-- degenerate: never fires"
        if key == "fi_budget" and not o.get("satisfied"):
            note = "  <-- ceiling not satisfiable"
        print(
            f"    {label:<22s} thr={o['threshold']:>8.1f} ms  "
            f"f1={o['f1']:.4f}  recall={o['recall']:.4f}  "
            f"false_interrupt={o['false_interruption_rate']:.4f}{note}"
        )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="data/cache/test")
    ap.add_argument("--table", default="artifacts/experiments.csv")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--skip-silero", action="store_true",
                    help="energy baseline only (Silero is ~50x slower per clip)")
    ap.add_argument("--slices", action="store_true",
                    help="also write the per-slice breakdown for each baseline")
    ap.add_argument("--out-dir", default="artifacts/runs/baselines")
    args = ap.parse_args(argv)

    cache = WaveCache(args.cache)
    idx = np.arange(len(cache))
    if args.max_rows:
        idx = idx[: args.max_rows]

    print(f"\n  cache: {args.cache}")
    print(f"  {cache.summary()}")
    print(f"  scoring {idx.size:,d} clips")

    table = ExperimentTable(args.table)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detectors = [("E0", EnergyBaseline(EnergyBaselineConfig()))]
    if not args.skip_silero:
        detectors.append(("E0b", SileroBaseline()))

    for name, det in detectors:
        print(f"\n  running {name} ({det.description}) ...")
        ev = evaluate_baseline(det, cache, idx, name)
        show_operating_points(name, ev)

        conf = confusion_at(
            np.asarray([cache.label(int(i)) for i in idx]),
            np.asarray([det.score(cache.wave(int(i))) for i in idx]),
            ev.threshold,
        )
        print("\n" + conf.matrix_str())

        ev.save(out_dir / f"{name}.json")
        row = table.add_evaluation(
            ev,
            model=det.description,
            window="whole clip",
            notes=f"threshold in ms of trailing silence; scored on {Path(args.cache).name}",
        )
        table.print_row(row)

        if args.slices:
            probs = np.asarray([det.score(cache.wave(int(i))) for i in idx])
            rows = slice_report(cache, idx, probs, ev.threshold)
            (out_dir / f"{name}-slices.md").write_text(
                slices_to_markdown(rows, f"{name} — per-slice"), encoding="utf-8"
            )
            print(f"    slices -> {out_dir / f'{name}-slices.md'}")

    table.save_markdown()
    print(f"\n  table -> {args.table}")
    print(f"  markdown -> report/experiments.md")
    print("\n" + table.to_markdown(
        ["id", "model", "n", "threshold", "f1", "recall", "false_interrupt", "missed", "roc_auc", "size_mb"]
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
