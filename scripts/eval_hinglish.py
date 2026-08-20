"""Day 8 — score a detector on the Hinglish stress set, as a separate table.

    python scripts/eval_hinglish.py --checkpoint weights/E1-best.pt
    python scripts/eval_hinglish.py --baseline          # E0 for comparison

Reported **separately and never averaged into the main test set**, for two
reasons that are different from each other:

1. It is synthetic, so its numbers do not transfer directly to real speech.
2. It is deliberately adversarial — the negatives were written to sound finished
   — so its class difficulty is nothing like the corpus's.

Averaging either property into a headline figure produces a number that
describes neither set.

**Which direction the synthetic bias runs depends on the detector**, and this is
worth stating precisely because "TTS is an upper bound" is the lazy version of
the claim and it is not always true:

* For a **prosody/spectral model**, TTS is *easier* than real speech — it is
  clean, evenly paced, and has no disfluent timing. Its score here is an
  optimistic estimate.
* For a **silence-timer baseline**, TTS is *harder* — Bulbul emits almost no
  trailing silence after a completed utterance, so the one cue the baseline
  depends on is largely absent. E0 measures below chance on this set (ROC-AUC
  ~0.48) for exactly that reason, and that is a property of the audio rather
  than a fresh insight about the baseline.

So the honest reading is: this set measures whether a detector uses *linguistic
completion* cues rather than *duration of silence*. A detector that scores well
here is reading the former.

The breakdown by ``category`` is the point of the whole exercise. A detector can
have a respectable overall F1 here and still fire on every ``end_filler`` clip,
which in a live call is the failure a user would describe as "it keeps cutting me
off". The per-category table is what makes that visible.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import SAMPLE_RATE  # noqa: E402
from src.metrics import confusion_at, evaluate  # noqa: E402


def load_manifest(clips_dir: Path) -> list[dict]:
    path = clips_dir / "manifest.jsonl"
    if not path.exists():
        raise SystemExit(
            f"no manifest at {path}\nRun: python scripts/synthesize_hinglish.py"
        )
    return [
        json.loads(l)
        for l in path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


def main(argv=None) -> int:
    import soundfile as sf

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default="data/hinglish/clips")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--baseline", action="store_true", help="score E0 instead of a model")
    ap.add_argument("--threshold", type=float, default=None,
                    help="explicit threshold; defaults to the checkpoint's (chosen on val)")
    ap.add_argument("--out", default="artifacts/runs/hinglish")
    args = ap.parse_args(argv)

    clips_dir = Path(args.clips)
    rows = load_manifest(clips_dir)
    print(f"\n  clips: {clips_dir} ({len(rows)} in manifest)")

    # -- scorer ------------------------------------------------------------ #
    if args.baseline:
        from src.baselines import EnergyBaseline

        eb = EnergyBaseline()
        # Scores are ms of trailing silence; the sweep handles any monotone score.
        def score(wave, sr):
            from src.audio import resample

            return eb.score(resample(wave, sr, SAMPLE_RATE))

        label, threshold = "E0 energy baseline", args.threshold
        unit = "ms trailing silence"
    else:
        from src.inference import TurnPredictor

        if args.onnx:
            pred = TurnPredictor(backend="onnx", onnx_path=args.onnx)
        elif args.checkpoint:
            pred = TurnPredictor(args.checkpoint, backend="torch")
        else:
            ap.error("pass --checkpoint, --onnx, or --baseline")
        print(f"  predictor: {pred.info()}")

        def score(wave, sr):
            return pred.predict(wave, sr)

        label = Path(args.onnx or args.checkpoint).stem
        threshold = args.threshold if args.threshold is not None else pred.threshold
        unit = "probability"

    # -- score ------------------------------------------------------------- #
    scores, labels, kept = [], [], []
    for r in rows:
        path = clips_dir / r["file"]
        if not path.exists():
            print(f"    missing: {r['file']}")
            continue
        wave, sr = sf.read(path, dtype="float32", always_2d=False)
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        scores.append(score(wave, int(sr)))
        labels.append(int(bool(r["endpoint_bool"])))
        kept.append(r)

    scores_a = np.asarray(scores, dtype=np.float64)
    labels_a = np.asarray(labels, dtype=int)
    print(f"  scored {labels_a.size} clips "
          f"({int(labels_a.sum())} endpoint / {int((labels_a == 0).sum())} not-endpoint)")

    ev = evaluate(f"hinglish/{label}", labels_a, scores_a, threshold=threshold)
    conf = confusion_at(labels_a, scores_a, ev.threshold)

    print(f"\n  === Hinglish stress set (synthetic — upper bound) ===")
    print(f"  threshold {ev.threshold:.4f} ({unit})"
          + ("  [from checkpoint, chosen on val]" if args.threshold is None and not args.baseline else ""))
    print(f"  {conf}")
    print(f"  roc_auc {ev.roc_auc:.4f}  pr_auc {ev.pr_auc:.4f}")
    print("\n" + conf.matrix_str())

    # -- per-category ------------------------------------------------------ #
    by_cat: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(kept):
        by_cat[r["category"]].append(i)

    print(f"\n  per category (threshold held fixed at {ev.threshold:.4f})")
    print(f"    {'category':<20s} {'n':>4s} {'pos':>4s} {'fired':>6s} "
          f"{'f1':>7s} {'recall':>7s} {'false_int':>10s}")
    cat_rows = []
    for cat in sorted(by_cat):
        take = np.asarray(by_cat[cat])
        y, p = labels_a[take], scores_a[take]
        c = confusion_at(y, p, ev.threshold)
        fired = int((p >= ev.threshold).sum())
        print(f"    {cat:<20s} {take.size:>4d} {int(y.sum()):>4d} {fired:>6d} "
              f"{c.f1:>7.4f} {c.recall:>7.4f} {c.false_interruption_rate:>10.4f}")
        cat_rows.append({
            "category": cat, "n": int(take.size), "positives": int(y.sum()),
            "fired": fired, "f1": round(c.f1, 4), "recall": round(c.recall, 4),
            "false_interrupt": round(c.false_interruption_rate, 4),
        })

    # The single most diagnostic line in this whole script.
    open_cats = [c for c in cat_rows if c["positives"] == 0]
    if open_cats:
        total_open = sum(c["n"] for c in open_cats)
        total_fired = sum(c["fired"] for c in open_cats)
        print(
            f"\n  >>> on deliberately-incomplete utterances "
            f"({', '.join(c['category'] for c in open_cats)}): "
            f"fired on {total_fired}/{total_open} "
            f"({100 * total_fired / max(total_open, 1):.1f}%) — "
            "each one of these is a user being cut off mid-sentence"
        )

    # -- per-speaker ------------------------------------------------------- #
    by_spk: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(kept):
        by_spk[r["speaker"]].append(i)
    if len(by_spk) > 1:
        print(f"\n  per speaker (a large spread means the set measures voices, not the phenomenon)")
        for spk in sorted(by_spk):
            take = np.asarray(by_spk[spk])
            c = confusion_at(labels_a[take], scores_a[take], ev.threshold)
            print(f"    {spk:<12s} n={take.size:>4d}  f1={c.f1:.4f}  "
                  f"false_int={c.false_interruption_rate:.4f}")

    # -- pause variants ---------------------------------------------------- #
    pause_rows = [i for i, r in enumerate(kept) if r.get("pause_ms")]
    if pause_rows:
        print(f"\n  mid-utterance pause variants (all complete utterances, label=endpoint)")
        by_pause: dict[int, list[int]] = defaultdict(list)
        for i in pause_rows:
            by_pause[int(kept[i]["pause_ms"])].append(i)
        base = [i for i, r in enumerate(kept)
                if not r.get("pause_ms") and r["endpoint_bool"]]
        if base:
            b = np.asarray(base)
            print(f"    {'no pause':<12s} n={b.size:>4d}  "
                  f"recall={confusion_at(labels_a[b], scores_a[b], ev.threshold).recall:.4f}")
        for ms in sorted(by_pause):
            take = np.asarray(by_pause[ms])
            c = confusion_at(labels_a[take], scores_a[take], ev.threshold)
            print(f"    {f'pause {ms}ms':<12s} n={take.size:>4d}  recall={c.recall:.4f}")
        print("    (a pause inserted mid-utterance must not stop the detector from\n"
              "     recognising that the utterance still completes)")

    # -- write ------------------------------------------------------------- #
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Slugify: labels like "E0 energy baseline" would otherwise produce filenames
    # with spaces, which are awkward to reference from a report or a shell.
    slug = "".join(c if c.isalnum() or c in "-_." else "-" for c in label).strip("-")
    ev.save(out / f"{slug}.json")
    (out / f"{slug}-categories.json").write_text(json.dumps(cat_rows, indent=2), encoding="utf-8")

    md = [
        f"### Hinglish stress set — {label}",
        "",
        "Synthetic (Bulbul TTS), deliberately adversarial negatives. "
        "**Reported separately; never averaged into the main test set.** "
        "Treat as an upper bound on real-world performance.",
        "",
        f"Overall at threshold {ev.threshold:.4f}: F1 {conf.f1:.4f}, "
        f"recall {conf.recall:.4f}, false-interruption rate "
        f"{conf.false_interruption_rate:.4f} (n={conf.n}).",
        "",
        "| category | n | positives | fired | f1 | recall | false_interrupt |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in cat_rows:
        md.append(
            f"| {c['category']} | {c['n']} | {c['positives']} | {c['fired']} | "
            f"{c['f1']} | {c['recall']} | {c['false_interrupt']} |"
        )
    (out / f"{slug}-table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n  written -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
