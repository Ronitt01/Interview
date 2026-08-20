"""Day 10 — quantise, export, benchmark, and state the accuracy cost.

    python scripts/optimize_model.py --checkpoint weights/E1-best.pt \
        --cache data/cache/test

Produces, and reports the delta between, four artefacts:

    float32 torch       the trained checkpoint
    int8 torch          dynamic quantisation of Linear/GRU
    float32 ONNX        graph export, Python removed from the loop
    int8 ONNX           the deployment artefact

The sentence this script exists to let the report write, with real numbers:

    "Model B improves F1 by X% while staying under Y MB and Z ms CPU latency."

Reference to sanity-check against: the published Smart Turn v3 is 8 MB int8 ONNX
/ 32 MB unquantised at ~8M parameters, ~10-12 ms on some CPUs. Our architecture
is the same, so a wildly different number means a bug here, not a breakthrough.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset import WaveCache  # noqa: E402
from src.evaluation import ExperimentTable, evaluate_predictor  # noqa: E402
from src.inference import TurnPredictor, load_checkpoint  # noqa: E402
from src.model import build_model  # noqa: E402
from src.optimize import (  # noqa: E402
    accuracy_delta,
    benchmark_predictor,
    export_onnx,
    file_size_mb,
    onnx_total_size_mb,
    front_end_cost,
    model_size_mb,
    quantize_dynamic,
    quantize_onnx,
    save_results,
)


def main(argv=None) -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache", default="data/cache/test")
    ap.add_argument("--out-dir", default="weights")
    ap.add_argument("--artifacts", default="artifacts/runs/optimize")
    ap.add_argument("--table", default="artifacts/experiments.csv")
    ap.add_argument("--threads", type=int, nargs="*", default=[1, 4])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--max-rows", type=int, default=None,
                    help="cap the accuracy re-scoring (latency is unaffected)")
    ap.add_argument("--skip-onnx", action="store_true")
    args = ap.parse_args(argv)

    out_dir, art = Path(args.out_dir), Path(args.artifacts)
    out_dir.mkdir(parents=True, exist_ok=True)
    art.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(args.checkpoint)
    stem = Path(args.checkpoint).stem.replace("-best", "")
    threshold = float(ckpt.get("threshold", 0.5))

    cache = WaveCache(args.cache)
    idx = np.arange(len(cache))
    if args.max_rows:
        idx = idx[: args.max_rows]

    print(f"\n  checkpoint : {args.checkpoint}")
    print(f"  threshold  : {threshold:.4f} (from the checkpoint — chosen on val)")
    print(f"  cache      : {args.cache} ({idx.size:,d} clips)")

    results, deltas, rows = [], [], []

    # ---- float32 torch --------------------------------------------------- #
    print("\n  [1/4] float32 torch")
    fp32 = TurnPredictor(args.checkpoint, backend="torch")
    ev_fp32, _ = evaluate_predictor(fp32, cache, idx, f"{stem}-fp32", threshold=threshold)
    print(f"    f1={ev_fp32.confusion['f1']:.4f}  size={ev_fp32.size_mb:.2f} MB")
    results += benchmark_predictor(fp32, "float32 torch", tuple(args.threads), n_iters=args.iters)

    # ---- how much of the latency is the front end, not the model? -------- #
    fe = front_end_cost(fp32.window, n_iters=args.iters)
    print(f"    mel front-end alone: p50 {fe.p50_ms:.2f} ms "
          f"({100 * fe.p50_ms / max(results[0].p50_ms, 1e-9):.0f}% of total)")
    results.append(fe)

    # ---- int8 torch ------------------------------------------------------ #
    print("\n  [2/4] int8 torch (dynamic)")
    int8 = TurnPredictor(args.checkpoint, backend="torch-int8")
    ev_int8, _ = evaluate_predictor(int8, cache, idx, f"{stem}-int8", threshold=threshold)
    model = build_model(ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    q = quantize_dynamic(model)
    ev_int8.size_mb = model_size_mb(q)
    print(f"    f1={ev_int8.confusion['f1']:.4f}  size={ev_int8.size_mb:.2f} MB")
    results += benchmark_predictor(int8, "int8 torch", tuple(args.threads), n_iters=args.iters)
    deltas.append(accuracy_delta(ev_fp32, ev_int8, "torch dynamic int8"))

    # ---- ONNX ------------------------------------------------------------ #
    if not args.skip_onnx:
        print("\n  [3/4] float32 ONNX")
        fresh = build_model(ckpt["model_config"])
        fresh.load_state_dict(ckpt["state_dict"])
        onnx_path = export_onnx(fresh, out_dir / f"{stem}.onnx", threshold=threshold)
        print(f"    exported {onnx_path.name} ({onnx_total_size_mb(onnx_path):.2f} MB)")
        p_onnx = TurnPredictor(backend="onnx", onnx_path=onnx_path, threads=1)
        ev_onnx, _ = evaluate_predictor(p_onnx, cache, idx, f"{stem}-onnx", threshold=threshold)
        ev_onnx.size_mb = onnx_total_size_mb(onnx_path)
        print(f"    f1={ev_onnx.confusion['f1']:.4f}")
        results += benchmark_predictor(p_onnx, "float32 ONNX", tuple(args.threads), n_iters=args.iters)
        deltas.append(accuracy_delta(ev_fp32, ev_onnx, "onnx export (fp32)"))

        print("\n  [4/4] int8 ONNX")
        q_path = quantize_onnx(onnx_path)
        print(f"    quantised {q_path.name} ({onnx_total_size_mb(q_path):.2f} MB)")
        p_q = TurnPredictor(backend="onnx", onnx_path=q_path, threads=1)
        ev_q, _ = evaluate_predictor(p_q, cache, idx, f"{stem}-onnx-int8", threshold=threshold)
        ev_q.size_mb = onnx_total_size_mb(q_path)
        print(f"    f1={ev_q.confusion['f1']:.4f}")
        results += benchmark_predictor(p_q, "int8 ONNX", tuple(args.threads), n_iters=args.iters)
        deltas.append(accuracy_delta(ev_fp32, ev_q, "onnx dynamic int8"))

        table = ExperimentTable(args.table)
        best = min(
            (r for r in results if r.batch_size == 1 and r.label.startswith("int8 ONNX")),
            key=lambda r: r.p50_ms,
            default=None,
        )
        ev_q.cpu_latency_p50_ms = best.p50_ms if best else None
        ev_q.cpu_latency_p95_ms = best.p95_ms if best else None
        ev_q.size_mb = onnx_total_size_mb(q_path)
        rows.append(
            table.add_evaluation(
                ev_q, model=f"{stem} int8 ONNX", window=str(fp32.window),
                notes="quantised deployment artefact",
            )
        )
        table.save_markdown()

    # ---- report ---------------------------------------------------------- #
    print("\n  latency (batch 1, synthetic input, p50/p95 — never a mean)")
    for r in results:
        if r.batch_size == 1:
            print(f"    {r}")

    print("\n  accuracy cost of each step")
    for d in deltas:
        print(
            f"    {d['step']:<22s} f1 {d['f1_before']:.4f} -> {d['f1_after']:.4f} "
            f"({d['f1_delta']:+.4f})  size {d['size_mb_before']:.2f} -> "
            f"{d['size_mb_after']:.2f} MB  [{d['verdict']}]"
        )

    save_results(results, art / "latency.json")
    (art / "accuracy_deltas.json").write_text(json.dumps(deltas, indent=2), encoding="utf-8")

    # The sentence, assembled from the numbers actually measured.
    fastest = min((r for r in results if r.batch_size == 1 and "ONNX" in r.label),
                  key=lambda r: r.p50_ms, default=results[0])
    final_ev = ev_q if not args.skip_onnx else ev_int8
    print(
        f"\n  >>> {stem}: F1 {final_ev.confusion['f1']:.4f} at "
        f"{final_ev.confusion['false_interruption_rate']:.4f} false-interruption "
        f"rate, {final_ev.size_mb:.2f} MB, {fastest.p50_ms:.1f} ms CPU p50 "
        f"({fastest.p95_ms:.1f} ms p95, {fastest.threads} thread)"
    )
    print(f"\n  artefacts -> {art}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
