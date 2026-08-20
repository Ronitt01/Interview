"""Day 9 — pull the failures, categorise them, and mine hard negatives.

    python scripts/error_analysis.py --checkpoint weights/E1-best.pt
    python scripts/error_analysis.py --checkpoint weights/E1-best.pt --mine-hard-negatives

Weighted toward false interruptions, because those are the failures that make a
voice agent feel broken, and because a set of 30 failures sampled uniformly from
a detector operating at a 10% interruption ceiling would be almost all missed
endpoints and would teach nothing.

The categorisation uses the corpus's own ``midfiller`` / ``endfiller`` /
``synthetic`` / ``language`` flags rather than a hand-built taxonomy. That was a
Day-1 finding worth the change: the error modes the plan expected to categorise
by ear are already labelled, so "the model struggles with hesitation" becomes a
number instead of an impression.

``--mine-hard-negatives`` writes the indices of the worst failures so the next
training run can oversample them. The retrain outcome gets reported **even when
it makes things worse** — a documented failed fix reads as rigour; a silently
dropped one reads as cherry-picking.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import SAMPLE_RATE  # noqa: E402
from src.dataset import WaveCache  # noqa: E402
from src.evaluation import evaluate_predictor, slice_report, slices_to_markdown  # noqa: E402


def failure_mode(row: dict, kind: str) -> str:
    """Name the failure using the corpus's own annotations.

    Ordered most-specific-first: a clip that is both mid- and end-filler is
    described by its end filler, because that is the cue at the boundary the
    detector actually had to judge.
    """
    if kind == "false_interruption":
        if row.get("endfiller") is True:
            return "cut off a trailing filler ('umm', 'matlab')"
        if row.get("midfiller") is True:
            return "cut off a mid-utterance hesitation"
        dur = float(row.get("audioduration") or 0)
        if dur < 1.5:
            return "fired on a short backchannel"
        return "fired mid-utterance, no filler annotation"
    if row.get("midfiller") is True:
        return "missed an endpoint after an internal hesitation"
    dur = float(row.get("audioduration") or 0)
    if dur < 1.5:
        return "missed a very short complete utterance"
    if dur >= 8.0:
        return "missed an endpoint on a long utterance"
    return "missed a plain endpoint"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--cache", default="data/cache/test")
    ap.add_argument("--n", type=int, default=30, help="failures to inspect")
    ap.add_argument("--fi-share", type=float, default=0.6,
                    help="share of the sample that should be false interruptions")
    ap.add_argument("--mine-hard-negatives", action="store_true")
    ap.add_argument("--hard-negative-count", type=int, default=500)
    ap.add_argument("--out", default="artifacts/runs/error_analysis")
    args = ap.parse_args(argv)

    from src.inference import TurnPredictor

    if args.onnx:
        pred = TurnPredictor(backend="onnx", onnx_path=args.onnx)
        label = Path(args.onnx).stem
    elif args.checkpoint:
        pred = TurnPredictor(args.checkpoint, backend="torch")
        label = Path(args.checkpoint).stem
    else:
        ap.error("pass --checkpoint or --onnx")

    cache = WaveCache(args.cache)
    idx = np.arange(len(cache))
    print(f"\n  predictor: {pred.info()}")
    print(f"  cache: {args.cache} ({idx.size:,d} clips)")

    ev, probs = evaluate_predictor(pred, cache, idx, label, threshold=pred.threshold)
    thr = ev.threshold
    y = np.asarray([cache.label(int(i)) for i in idx])
    pred_pos = probs >= thr

    fi_mask = pred_pos & (y == 0)   # false interruptions
    me_mask = (~pred_pos) & (y == 1)  # missed endpoints
    print(f"\n  at threshold {thr:.4f}: "
          f"{int(fi_mask.sum())} false interruptions, "
          f"{int(me_mask.sum())} missed endpoints")

    # -- sample, weighted toward the failures that matter ------------------ #
    def worst(mask, want, key):
        """Most-confident failures first — those are the informative ones."""
        rows = np.flatnonzero(mask)
        if rows.size == 0:
            return rows
        order = rows[np.argsort(key(probs[rows]))]
        return order[:want]

    n_fi = min(int(round(args.n * args.fi_share)), int(fi_mask.sum()))
    n_me = min(args.n - n_fi, int(me_mask.sum()))
    fi_sel = worst(fi_mask, n_fi, lambda p: -p)   # highest prob = most confident FP
    me_sel = worst(me_mask, n_me, lambda p: p)    # lowest prob = most confident FN

    print(f"  inspecting {fi_sel.size} false interruptions "
          f"+ {me_sel.size} missed endpoints (most-confident first)")

    records = []
    for sel, kind in ((fi_sel, "false_interruption"), (me_sel, "missed_endpoint")):
        for j in sel:
            row = cache.meta[int(idx[j])]
            records.append({
                "cache_index": int(idx[j]),
                "clip_id": row.get("id"),
                "kind": kind,
                "probability": round(float(probs[j]), 4),
                "threshold": round(float(thr), 4),
                "label": int(y[j]),
                "language": row.get("language"),
                "source": row.get("dataset"),
                "duration_s": round(float(row.get("audioduration") or 0), 2),
                "midfiller": row.get("midfiller"),
                "endfiller": row.get("endfiller"),
                "synthetic": row.get("synthetic"),
                "mode": failure_mode(row, kind),
            })

    # -- the categorised table --------------------------------------------- #
    print("\n  failure modes")
    for kind in ("false_interruption", "missed_endpoint"):
        subset = [r for r in records if r["kind"] == kind]
        if not subset:
            continue
        print(f"\n    {kind} ({len(subset)})")
        for mode, n in Counter(r["mode"] for r in subset).most_common():
            print(f"      {n:>3d}  {mode}")

    print("\n  worst individual failures")
    for r in sorted(records, key=lambda r: -abs(r["probability"] - r["threshold"]))[:12]:
        print(f"    p={r['probability']:.3f} thr={r['threshold']:.3f} "
              f"[{r['kind'][:2]}] {r['language']}/{r['source']} "
              f"{r['duration_s']:>5.2f}s  {r['mode']}")

    # -- where the errors concentrate -------------------------------------- #
    print("\n  false-interruption rate by annotation")
    for name, mask in (
        ("endfiller=True", np.asarray([cache.meta[int(i)].get("endfiller") is True for i in idx])),
        ("midfiller=True", np.asarray([cache.meta[int(i)].get("midfiller") is True for i in idx])),
        ("synthetic=True", np.asarray([cache.meta[int(i)].get("synthetic") is True for i in idx])),
        ("synthetic=False", np.asarray([cache.meta[int(i)].get("synthetic") is False for i in idx])),
    ):
        neg = mask & (y == 0)
        if neg.sum() >= 10:
            rate = float((pred_pos & neg).sum() / neg.sum())
            print(f"    {name:<18s} n_neg={int(neg.sum()):>5d}  FI={rate:.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{label}-failures.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

    rows = slice_report(cache, idx, probs, thr)
    (out / f"{label}-slices.md").write_text(
        slices_to_markdown(rows, f"{label} — per-slice (test)"), encoding="utf-8"
    )
    np.save(out / f"{label}-probs.npy", probs)

    # -- hard-negative mining ---------------------------------------------- #
    if args.mine_hard_negatives:
        # The clips the model is most confidently wrong about, both directions.
        margin = np.where(y == 1, thr - probs, probs - thr)
        hard = np.flatnonzero(margin > 0)
        hard = hard[np.argsort(-margin[hard])][: args.hard_negative_count]
        hard_idx = idx[hard]
        np.save(out / "hard_negative_indices.npy", hard_idx)
        counts = Counter(cache.meta[int(i)].get("dataset") for i in hard_idx)
        print(f"\n  mined {hard_idx.size} hard negatives -> "
              f"{out / 'hard_negative_indices.npy'}")
        print(f"    by source: {dict(counts.most_common(6))}")
        print(
            "    Feed these into the next run by oversampling the indices, then\n"
            "    report the outcome either way. A retrain that made things worse\n"
            "    is a result; a retrain quietly dropped is cherry-picking."
        )

    print(f"\n  written -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
