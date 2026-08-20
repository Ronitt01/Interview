"""Day 11 — score the streaming detector and tune its three knobs.

    python scripts/stream_eval.py --checkpoint weights/E1-best.pt \
        --cache data/cache/test --sweep

Two modes:

* default — run one streaming configuration and report its confusion matrix,
  time-to-decide percentiles, and end-to-end wall latency;
* ``--sweep`` — grid over ``min_silence_ms``, ``ema_alpha`` and the hysteresis
  band, and print the frontier. This is where the streaming parameters actually
  get chosen, and choosing them on a grid beats choosing them by feel.

The limitation, stated because it belongs next to the numbers: cached clips are
*pre-cut* utterances, so replaying them measures how promptly the detector reacts
to a boundary already at the end of the audio. It does not measure behaviour
across a continuous multi-turn conversation — the corpus has none. The Hinglish
stress clips are continuous single utterances, which covers part of the gap.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset import WaveCache  # noqa: E402
from src.inference import TurnPredictor  # noqa: E402
from src.metrics import Confusion  # noqa: E402
from src.streaming import (  # noqa: E402
    StreamingConfig,
    StreamingTurnDetector,
    run_stream_eval,
    summarise_stream,
)


def describe(summary: dict, label: str) -> str:
    c = summary["confusion"]
    ttd = summary["time_to_decide_ms"]
    wall = summary["wall_latency_ms"]
    return (
        f"  {label:<44s} f1={c['f1']:.4f}  recall={c['recall']:.4f}  "
        f"FI={c['false_interruption_rate']:.4f}  "
        f"ttd_p50={ttd['p50']:7.1f}ms  ttd_p95={ttd['p95']:7.1f}ms  "
        f"wall_p95={wall['p95']:6.1f}ms"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--onnx", default=None, help="use an ONNX graph instead (faster)")
    ap.add_argument("--cache", default="data/cache/test")
    ap.add_argument("--max-rows", type=int, default=300,
                    help="streaming replay is ~40x the cost of clip scoring, so this "
                         "defaults to a subset; raise it for the final number")
    ap.add_argument("--chunk-ms", type=float, default=20.0, help="WebRTC frame size")
    ap.add_argument("--hop-ms", type=float, default=160.0)
    ap.add_argument("--enter", type=float, default=0.70)
    ap.add_argument("--exit", dest="exit_thr", type=float, default=0.45)
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--min-silence", type=float, default=200.0)
    ap.add_argument("--baseline", action="store_true",
                    help="stream the E0 energy baseline instead of a model, on the "
                         "same machinery — the like-for-like streaming comparison")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", default="artifacts/runs/streaming")
    args = ap.parse_args(argv)

    cache = WaveCache(args.cache)
    idx = np.arange(len(cache))
    if args.max_rows:
        idx = idx[: args.max_rows]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # -- the score function ------------------------------------------------ #
    if args.baseline:
        from src.baselines import EnergyBaseline

        eb = EnergyBaseline()
        # Map trailing-silence ms onto [0,1] so the same thresholds apply. 800 ms
        # is the saturation point: beyond it the baseline is certain either way.
        def score_fn(window):
            return float(min(eb.score(window) / 800.0, 1.0))

        window_seconds = 2.0
        label = "E0 energy (streamed)"
    else:
        if args.onnx:
            pred = TurnPredictor(backend="onnx", onnx_path=args.onnx, threads=1)
        elif args.checkpoint:
            pred = TurnPredictor(args.checkpoint, backend="torch")
        else:
            ap.error("pass --checkpoint or --onnx, or use --baseline")
        score_fn = pred.as_score_fn()
        window_seconds = pred.cfg.window_seconds
        label = f"{Path(args.onnx or args.checkpoint).stem} (streamed)"
        print(f"  predictor: {pred.info()}")

    print(f"  cache: {args.cache} — replaying {idx.size:,d} clips at {args.chunk_ms:g} ms chunks")
    print(f"  window: {window_seconds:g}s  hop: {args.hop_ms:g} ms")

    def run(cfg: StreamingConfig):
        det = StreamingTurnDetector(score_fn, cfg)
        res = run_stream_eval(det, cache, idx, chunk_ms=args.chunk_ms, progress_every=0)
        return summarise_stream(res)

    if not args.sweep:
        cfg = StreamingConfig(
            window_seconds=window_seconds, hop_ms=args.hop_ms,
            enter_threshold=args.enter, exit_threshold=args.exit_thr,
            ema_alpha=args.alpha, min_silence_ms=args.min_silence,
        )
        print(f"\n  config: {json.dumps(cfg.to_dict())}\n")
        s = run(cfg)
        print(describe(s, label))
        c = Confusion(**{k: s["confusion"][k] for k in ("tp", "fp", "tn", "fn")})
        print("\n" + c.matrix_str())
        print(f"\n  time-to-decide (audio ms past the clip's own end, correct positives only):")
        for k, v in s["time_to_decide_ms"].items():
            print(f"    {k:>4s}  {v:.1f}")
        print(f"\n  wall latency (chunk arrival -> event emission, includes the model):")
        for k, v in s["wall_latency_ms"].items():
            print(f"    {k:>4s}  {v:.2f}")
        (out / "single.json").write_text(
            json.dumps({"config": cfg.to_dict(), "summary": s}, indent=2), encoding="utf-8"
        )
        return 0

    # -- sweep -------------------------------------------------------------- #
    grid = list(
        itertools.product(
            (0.0, 100.0, 200.0, 400.0),      # min_silence_ms
            (1.0, 0.6, 0.4, 0.2),            # ema_alpha (1.0 = no smoothing)
            ((0.5, 0.5), (0.7, 0.45), (0.8, 0.35)),  # (enter, exit)
        )
    )
    print(f"\n  sweeping {len(grid)} configurations\n")
    rows = []
    for min_sil, alpha, (enter, exit_thr) in grid:
        cfg = StreamingConfig(
            window_seconds=window_seconds, hop_ms=args.hop_ms,
            enter_threshold=enter, exit_threshold=exit_thr,
            ema_alpha=alpha, min_silence_ms=min_sil,
        )
        s = run(cfg)
        tag = f"sil={min_sil:>5.0f} a={alpha:.1f} band={enter:.2f}/{exit_thr:.2f}"
        print(describe(s, tag))
        rows.append({"config": cfg.to_dict(), "summary": s, "tag": tag})

    (out / "sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\n  best by F1:")
    for r in sorted(rows, key=lambda r: -r["summary"]["confusion"]["f1"])[:3]:
        print(describe(r["summary"], r["tag"]))

    print("\n  best recall at false-interruption rate <= 0.10:")
    feasible = [r for r in rows if r["summary"]["confusion"]["false_interruption_rate"] <= 0.10]
    if not feasible:
        print("    none — no configuration met the ceiling on this subset")
    for r in sorted(feasible, key=lambda r: -r["summary"]["confusion"]["recall"])[:3]:
        print(describe(r["summary"], r["tag"]))

    print(f"\n  sweep -> {out / 'sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
