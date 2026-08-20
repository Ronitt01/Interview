"""Assemble a sweep comparison table from run artefacts. Read-only.

    python scripts/summarise_sweep.py --runs E2_w0p5 E2_w1p0 E2_w1p5 E2_w2p0 E2_w4p0 E1
    python scripts/summarise_sweep.py --prefix E2 --include E1

Reads `artifacts/runs/<run_id>/{evaluation.json,history.json,config.yaml.json}`
and prints one row per run. It writes nothing except the optional `--out`
markdown file, trains nothing, and loads no model — so it cannot change a result
it is reporting on.

**Which split the numbers come from.** A run configured with
`test_cache_dir: null` evaluates validation only, so its `evaluation.json` holds
validation metrics and the `split` column reads `val`. A run that also scored a
test set reports test there instead. The column is printed rather than assumed,
because comparing a val row against a test row would be meaningless and the
table should make that visible rather than hide it.

`cost` is the selection criterion — `fp_cost * false_interruption_rate +
missed_endpoint_rate`, lower is better. It is what training itself minimised for
early stopping, so it is the consistent basis for picking a winner. F1 is shown
alongside because a reviewer will look for it, not because it drives the choice.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_run(run_dir: Path) -> dict | None:
    """Pull every field the comparison needs out of one run directory."""
    ev_path = run_dir / "evaluation.json"
    if not ev_path.exists():
        return None
    ev = json.loads(ev_path.read_text(encoding="utf-8"))
    conf = ev.get("confusion", {})

    cfg = {}
    cfg_path = run_dir / "config.yaml.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    # history.json carries per-epoch val_cost and wall-clock seconds. A run
    # scored via --eval-only has no history, which is why this is optional.
    hist = []
    hist_path = run_dir / "history.json"
    if hist_path.exists():
        hist = json.loads(hist_path.read_text(encoding="utf-8"))

    best_cost = min((h.get("val_cost") for h in hist if h.get("val_cost") is not None),
                    default=None)
    train_seconds = sum(h.get("seconds", 0.0) for h in hist) or None
    best_epoch = None
    if hist and best_cost is not None:
        best_epoch = min(
            (h for h in hist if h.get("val_cost") == best_cost),
            key=lambda h: h["epoch"],
        )["epoch"]

    # Which split this evaluation.json describes, taken from how the pipeline
    # names it rather than guessed: training scores test as `evaluate(run_id, ...)`
    # and validation as `evaluate(f"{run_id}/val", ...)`, so a trailing "/val" is
    # an exact signal. Falling back to the config only if the name is absent.
    name = str(ev.get("name") or run_dir.name)
    if name.endswith("/val"):
        split = "val"
    elif cfg.get("test_cache_dir") in (None, "", "null"):
        split = "val"
    else:
        split = "test"

    return {
        "id": name[:-4] if name.endswith("/val") else name,
        "window": f"{cfg.get('window_seconds')}s" if cfg.get("window_seconds") else "?",
        "window_num": cfg.get("window_seconds"),
        "split": split,
        "n": ev.get("n"),
        "threshold": ev.get("threshold"),
        "accuracy": conf.get("accuracy"),
        "precision": conf.get("precision"),
        "recall": conf.get("recall"),
        "f1": conf.get("f1"),
        "false_interrupt": conf.get("false_interruption_rate"),
        "missed": conf.get("missed_endpoint_rate"),
        "cost": conf.get("cost"),
        "best_val_cost": best_cost,
        "roc_auc": ev.get("roc_auc"),
        "pr_auc": ev.get("pr_auc"),
        "epochs_run": len(hist) or None,
        "best_epoch": best_epoch,
        "train_seconds": train_seconds,
        "params_m": ev.get("params_m"),
        "size_mb": ev.get("size_mb"),
    }


def fmt(v, nd=4, dash="-"):
    if v is None:
        return dash
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


COLUMNS = [
    ("id", "run", 0),
    ("window", "window", 0),
    ("split", "split", 0),
    ("n", "n", 0),
    ("threshold", "threshold", 4),
    ("f1", "F1", 4),
    ("precision", "precision", 4),
    ("recall", "recall", 4),
    ("false_interrupt", "false_int", 4),
    ("missed", "missed", 4),
    ("cost", "cost", 4),
    ("roc_auc", "roc_auc", 4),
    ("train_seconds", "train_s", 0),
    ("best_epoch", "best_ep", 0),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", default="artifacts/runs")
    ap.add_argument("--runs", nargs="*", default=None, help="explicit run ids, in order")
    ap.add_argument("--prefix", default=None, help="include every run id starting with this")
    ap.add_argument("--include", nargs="*", default=[], help="extra run ids to append")
    ap.add_argument("--sort-by", default="window_num",
                    choices=("window_num", "cost", "f1", "id"))
    ap.add_argument("--out", default=None, help="also write a markdown table here")
    args = ap.parse_args(argv)

    root = Path(args.artifacts)
    if not root.exists():
        raise SystemExit(f"no artefacts directory at {root}")

    if args.runs:
        names = list(args.runs)
    elif args.prefix:
        names = sorted(d.name for d in root.iterdir()
                       if d.is_dir() and d.name.startswith(args.prefix))
    else:
        names = sorted(d.name for d in root.iterdir() if d.is_dir())
    names += [n for n in args.include if n not in names]

    rows, missing = [], []
    for n in names:
        r = load_run(root / n)
        (rows.append(r) if r else missing.append(n))

    if not rows:
        raise SystemExit(
            f"no runs with an evaluation.json found under {root} "
            f"(looked for: {', '.join(names) or 'everything'})"
        )
    if missing:
        print(f"  no evaluation.json for: {', '.join(missing)}  (not yet run?)\n")

    key = args.sort_by
    rows.sort(key=lambda r: (r.get(key) is None, r.get(key)))

    widths = {c: max(len(h), *(len(fmt(r[c], nd)) for r in rows)) for c, h, nd in COLUMNS}
    print("  " + "  ".join(h.rjust(widths[c]) for c, h, _ in COLUMNS))
    print("  " + "  ".join("-" * widths[c] for c, _, _ in COLUMNS))
    for r in rows:
        print("  " + "  ".join(fmt(r[c], nd).rjust(widths[c]) for c, _, nd in COLUMNS))

    val_rows = [r for r in rows if r["split"] == "val" and r["cost"] is not None]
    if val_rows:
        best = min(val_rows, key=lambda r: r["cost"])
        print(f"\n  lowest validation cost: {best['id']} "
              f"(window {best['window']}, cost {best['cost']:.4f}, F1 {fmt(best['f1'])})")
        best_f1 = max(val_rows, key=lambda r: (r["f1"] or 0))
        if best_f1["id"] != best["id"]:
            print(f"  highest validation F1  : {best_f1['id']} "
                  f"(window {best_f1['window']}, F1 {fmt(best_f1['f1'])}, "
                  f"cost {best_f1['cost']:.4f})")
            print("  -> cost and F1 disagree; cost is the training objective, so it "
                  "decides unless you state otherwise")

    mixed = {r["split"] for r in rows}
    if len(mixed) > 1:
        print(f"\n  NOTE: rows mix splits {sorted(mixed)}. Only compare rows with the "
              "same split — a val row against a test row is not a comparison.")

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        head = "| " + " | ".join(h for _, h, _ in COLUMNS) + " |"
        rule = "|" + "|".join("---" for _ in COLUMNS) + "|"
        body = ["| " + " | ".join(fmt(r[c], nd) for c, _, nd in COLUMNS) + " |" for r in rows]
        p.write_text(
            "# Window sweep (E2)\n\nGenerated by `scripts/summarise_sweep.py` from "
            "run artefacts — do not edit by hand.\n\n" + "\n".join([head, rule, *body]) + "\n",
            encoding="utf-8",
        )
        print(f"\n  markdown -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
