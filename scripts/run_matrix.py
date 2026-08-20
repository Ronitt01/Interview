"""Days 6-7 — run the experiment matrix, one variable at a time.

    python scripts/run_matrix.py                  # every config in configs/
    python scripts/run_matrix.py --only E1 E2      # just those run ids
    python scripts/run_matrix.py --dry-run         # prove the wiring first

Each config is a single-variable change from E1. Runs are sequential rather than
parallel because a free T4 has one GPU and two runs sharing it are slower than
two runs in sequence, plus the memory pressure makes the failure mode a
mid-training OOM rather than a queue.

A failed run does not abort the sweep. It is recorded and the next config starts,
because losing eight good runs to one bad config is the worst outcome here. The
summary at the end lists what failed and why.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def config_run_id(path: Path) -> str:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return str(raw.get("run_id", path.stem))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--only", nargs="*", default=None, help="run ids to include")
    ap.add_argument("--skip", nargs="*", default=None, help="run ids to exclude")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--table", default="artifacts/experiments.csv")
    ap.add_argument("--continue-on-error", action="store_true", default=True)
    args = ap.parse_args(argv)

    paths = sorted(Path(args.configs).glob("*.yaml"))
    if not paths:
        print(f"  no configs found in {args.configs}")
        return 1

    jobs = []
    for p in paths:
        rid = config_run_id(p)
        if args.only and rid not in args.only:
            continue
        if args.skip and rid in args.skip:
            continue
        jobs.append((rid, p))

    if not jobs:
        print("  nothing selected")
        return 1

    print(f"\n  {len(jobs)} run(s) queued:")
    for rid, p in jobs:
        print(f"    {rid:<10s} {p.name}")
    print()

    results = []
    for i, (rid, p) in enumerate(jobs, 1):
        cmd = [sys.executable, "-m", "training.train", "--config", str(p)]
        if args.dry_run:
            cmd.append("--dry-run")
        if args.device:
            cmd += ["--device", args.device]

        print(f"\n{'=' * 78}\n  [{i}/{len(jobs)}] {rid}\n{'=' * 78}", flush=True)
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=REPO_ROOT)
        secs = time.perf_counter() - t0
        ok = proc.returncode == 0
        results.append({"run_id": rid, "config": p.name, "ok": ok,
                        "returncode": proc.returncode, "seconds": secs})
        print(f"\n  [{i}/{len(jobs)}] {rid}: {'ok' if ok else f'FAILED (rc={proc.returncode})'} "
              f"in {secs / 60:.1f} min", flush=True)
        if not ok and not args.continue_on_error:
            break

    # ---- summary --------------------------------------------------------- #
    print(f"\n{'=' * 78}\n  matrix summary\n{'=' * 78}")
    for r in results:
        print(f"    {r['run_id']:<10s} {'ok    ' if r['ok'] else 'FAILED'} "
              f"{r['seconds'] / 60:>6.1f} min  ({r['config']})")
    failed = [r for r in results if not r["ok"]]
    print(f"\n  {len(results) - len(failed)}/{len(results)} succeeded")

    from src.evaluation import ExperimentTable

    table = ExperimentTable(args.table)
    table.save_markdown()
    print("\n" + table.to_markdown(
        ["id", "model", "window", "n", "f1", "recall", "false_interrupt", "missed",
         "roc_auc", "size_mb", "params_m"]
    ))
    print(f"\n  table -> {args.table}\n  markdown -> report/experiments.md")

    if failed:
        print(f"\n  failed runs: {', '.join(r['run_id'] for r in failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
