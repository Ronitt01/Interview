"""Day 12 — the demo. Built last, touched first.

    python demo/app.py                                  # auto-picks a checkpoint
    python demo/app.py --checkpoint weights/E5-best.pt
    python demo/app.py --onnx weights/E5-int8.onnx --share

What it has to do to "feel real" rather than feel like a notebook with a text
box:

* record from the mic **or** upload, and take a pre-loaded example;
* show the waveform with the endpoint marked, not just a number;
* run the **streaming** detector over the clip and plot the probability timeline,
  so you can watch the decision form rather than being handed a verdict;
* expose the three streaming knobs live — threshold, minimum silence, smoothing —
  because the whole argument of §6 is that those knobs matter, and a reviewer
  should be able to move them and see it;
* report measured latency, not a claim about latency.

The pre-loaded examples include a deliberate mid-utterance pause, so a reviewer
can hear the detector *hold* rather than fire. That is the single most convincing
thing this demo can show.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import SAMPLE_RATE  # noqa: E402
from src.audio import resample, to_mono  # noqa: E402
from src.streaming import StreamingConfig, StreamingTurnDetector  # noqa: E402

# Matplotlib must be headless before pyplot is imported, or a Space with no
# display raises on the first plot instead of on startup.
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------- #
# model loading
# --------------------------------------------------------------------------- #
def find_checkpoint(explicit: str | None) -> tuple[str | None, str | None]:
    """Return ``(checkpoint, onnx)``, preferring an int8 ONNX graph if present.

    Auto-discovery exists so that ``python demo/app.py`` works on a fresh clone
    without the reviewer having to know which run id won. It prefers the
    quantised graph because that is the artefact that would actually deploy.
    """
    if explicit:
        return (explicit, None)
    weights = REPO_ROOT / "weights"
    if not weights.exists():
        return (None, None)
    for pattern in ("*-int8.onnx", "*.onnx"):
        found = sorted(weights.glob(pattern))
        if found:
            return (None, str(found[-1]))
    found = sorted(weights.glob("*-best.pt")) or sorted(weights.glob("*.pt"))
    return (str(found[-1]), None) if found else (None, None)


class Backend:
    """Wraps a predictor, or falls back to the energy baseline if none exists.

    The fallback matters: a reviewer who clones the repo and runs the demo before
    training anything should get a working UI that says plainly what it is
    running, rather than a stack trace.
    """

    def __init__(self, checkpoint: str | None, onnx: str | None) -> None:
        self.is_model = False
        self.label = "E0 energy baseline (no trained checkpoint found)"
        self.threshold = 0.5
        self.window_seconds = 2.0
        self.info = "no checkpoint — falling back to the energy baseline"

        if checkpoint or onnx:
            from src.inference import TurnPredictor

            self.pred = (
                TurnPredictor(backend="onnx", onnx_path=onnx, threads=1)
                if onnx
                else TurnPredictor(checkpoint, backend="torch")
            )
            self.is_model = True
            self.threshold = self.pred.threshold
            self.window_seconds = self.pred.cfg.window_seconds
            self.label = Path(onnx or checkpoint).name
            self.info = str(self.pred.info())
            self._score = self.pred.as_score_fn()
        else:
            from src.baselines import EnergyBaseline

            eb = EnergyBaseline()

            def score(window):
                # Map ms of trailing silence into [0,1] so the same UI
                # thresholds apply. 800 ms is saturation.
                return float(min(eb.score(window) / 800.0, 1.0))

            self._score = score

    def score_fn(self):
        return self._score


BACKEND: Backend | None = None


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def make_figure(wave, trace, decision, enter_thr, exit_thr):
    """Waveform on top, probability timeline below, endpoint marked on both."""
    dur = wave.size / SAMPLE_RATE
    fig, (ax_w, ax_p) = plt.subplots(
        2, 1, figsize=(11, 5), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.4], "hspace": 0.12},
    )

    t = np.arange(wave.size) / SAMPLE_RATE
    ax_w.plot(t, wave, lw=0.5, color="#4A45C9")
    ax_w.set_ylabel("amplitude")
    ax_w.set_ylim(-1.05, 1.05)
    ax_w.grid(alpha=0.2)
    ax_w.set_title("waveform")

    ms = np.asarray(trace.audio_time_ms) / 1000.0
    if ms.size:
        ax_p.plot(ms, trace.probability, lw=1.0, alpha=0.45,
                  color="#5B6280", label="raw P(turn ended)")
        ax_p.plot(ms, trace.smoothed, lw=2.0, color="#4A45C9", label="smoothed (EMA)")
    ax_p.axhline(enter_thr, ls="--", lw=1.0, color="#A93A28", label=f"enter {enter_thr:.2f}")
    ax_p.axhline(exit_thr, ls=":", lw=1.0, color="#9C6410", label=f"exit {exit_thr:.2f}")
    ax_p.fill_between([0, dur], exit_thr, enter_thr, color="#9C6410", alpha=0.07)
    ax_p.set_ylim(-0.03, 1.03)
    ax_p.set_ylabel("P(turn ended)")
    ax_p.set_xlabel("time (s)")
    ax_p.grid(alpha=0.2)

    if decision is not None:
        x = decision.audio_time_ms / 1000.0
        for ax in (ax_w, ax_p):
            ax.axvline(x, color="#2E7358", lw=2.0)
        ax_p.annotate(
            "USER_TURN_ENDED", xy=(x, 0.5), xytext=(6, 0),
            textcoords="offset points", color="#2E7358",
            fontsize=9, fontweight="bold", va="center",
        )
    ax_p.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax_p.set_xlim(0, max(dur, 0.1))
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# the callback
# --------------------------------------------------------------------------- #
def analyse(audio, enter_thr, exit_thr, min_silence_ms, alpha, hop_ms, chunk_ms):
    assert BACKEND is not None
    if audio is None:
        return "Record or upload a clip first.", None, {}

    sr, wave = audio
    wave = to_mono(np.asarray(wave, dtype=np.float32))
    # Gradio hands back int16 for recorded audio and float for some uploads.
    if np.issubdtype(wave.dtype, np.integer) or np.max(np.abs(wave)) > 1.5:
        wave = wave / 32768.0
    wave = resample(wave, int(sr), SAMPLE_RATE)
    if wave.size < int(0.05 * SAMPLE_RATE):
        return "Clip is too short to analyse (under 50 ms).", None, {}

    if exit_thr > enter_thr:  # the UI allows it; the config would raise
        exit_thr = enter_thr

    cfg = StreamingConfig(
        window_seconds=BACKEND.window_seconds,
        hop_ms=float(hop_ms),
        enter_threshold=float(enter_thr),
        exit_threshold=float(exit_thr),
        ema_alpha=float(alpha),
        min_silence_ms=float(min_silence_ms),
    )
    det = StreamingTurnDetector(BACKEND.score_fn(), cfg)

    t0 = time.perf_counter()
    decision = det.run(wave, chunk_ms=float(chunk_ms))
    total_ms = (time.perf_counter() - t0) * 1000.0

    dur_ms = 1000.0 * wave.size / SAMPLE_RATE
    infer = np.asarray(det.trace.infer_ms)
    final_p = det.trace.smoothed[-1] if det.trace.smoothed else 0.0

    if decision is not None:
        verdict = (
            f"### `USER_TURN_ENDED`\n\n"
            f"Fired at **{decision.audio_time_ms:.0f} ms** of "
            f"{dur_ms:.0f} ms audio "
            f"({decision.audio_time_ms - dur_ms:+.0f} ms relative to the clip end).\n\n"
            f"- smoothed probability at decision: **{decision.smoothed:.3f}**\n"
            f"- held above the entry threshold for **{decision.held_ms:.0f} ms**\n"
            f"- end-to-end wall latency: **{decision.wall_latency_ms:.2f} ms**"
        )
    else:
        verdict = (
            f"### Still speaking — held\n\n"
            f"No endpoint emitted across {dur_ms:.0f} ms of audio.\n\n"
            f"- final smoothed probability: **{final_p:.3f}** "
            f"(entry threshold {enter_thr:.2f})\n"
            f"- this is the correct output for a trailing filler or a "
            f"mid-sentence pause"
        )

    metrics = {
        "detector": BACKEND.label,
        "window_seconds": BACKEND.window_seconds,
        "clip_ms": round(dur_ms, 1),
        "hops_evaluated": int(infer.size),
        "model_ms_per_hop_p50": round(float(np.percentile(infer, 50)), 3) if infer.size else None,
        "model_ms_per_hop_p95": round(float(np.percentile(infer, 95)), 3) if infer.size else None,
        "replay_total_ms": round(total_ms, 1),
        "real_time_factor": round(total_ms / max(dur_ms, 1e-9), 4),
        "fired": decision is not None,
        "audio_time_ms": round(decision.audio_time_ms, 1) if decision else None,
        "final_smoothed": round(float(final_p), 4),
    }
    fig = make_figure(wave, det.trace, decision, float(enter_thr), float(exit_thr))
    return verdict, fig, metrics


# --------------------------------------------------------------------------- #
# examples
# --------------------------------------------------------------------------- #
def gather_examples() -> list[list]:
    """Pre-loaded clips, with the hard cases first.

    Deliberately ordered so the first thing a reviewer clicks is a clip that
    should *not* fire. Leading with an easy positive teaches nothing.
    """
    clips = REPO_ROOT / "data" / "hinglish" / "clips"
    if not clips.exists():
        return []
    import json

    manifest = clips / "manifest.jsonl"
    if not manifest.exists():
        return [[str(p)] for p in sorted(clips.glob("*.wav"))[:8]]

    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    priority = {
        "end_filler": 0, "mid_hesitation": 1, "code_switch": 2,
        "end_filler_dev": 3, "complete": 4, "complete_en": 5, "complete_dev": 6,
    }
    rows.sort(key=lambda r: (0 if r.get("pause_ms") else 1,
                             priority.get(r["category"], 9), r["file"]))
    out = []
    seen = set()
    for r in rows:
        key = (r["category"], bool(r.get("pause_ms")))
        if key in seen:
            continue
        seen.add(key)
        p = clips / r["file"]
        if p.exists():
            out.append([str(p)])
        if len(out) >= 10:
            break
    return out


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def build_ui():
    import gradio as gr

    assert BACKEND is not None
    warn = (
        ""
        if BACKEND.is_model
        else "\n> **No trained checkpoint found.** Running the E0 energy baseline "
        "so the UI works on a fresh clone. Train a model and restart to see the "
        "real detector.\n"
    )

    with gr.Blocks(title="Smart Turn Detection", theme=gr.themes.Soft()) as ui:
        gr.Markdown(
            f"""# Smart turn detection — audio-only endpointing

Decides whether a speaker has **finished their turn**, from the waveform alone —
no transcript. The detector runs as a rolling window over the audio, exactly as
it would in a live call, and the timeline below shows the decision forming.

**Detector:** `{BACKEND.info}`
{warn}
Try a `end_filler` example first — a clip ending in *"...matlab"* or *"...umm"*.
It should **hold**, not fire. That is the failure mode that makes a voice agent
feel broken."""
        )

        with gr.Row():
            with gr.Column(scale=3):
                audio = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="Record or upload (16 kHz mono is ideal; anything is resampled)",
                )
                run = gr.Button("Detect turn end", variant="primary")
            with gr.Column(scale=2):
                gr.Markdown("#### Streaming parameters")
                enter = gr.Slider(0.05, 0.99, value=BACKEND.threshold, step=0.01,
                                  label="enter threshold — fire above this")
                exit_ = gr.Slider(0.01, 0.99, value=max(0.05, BACKEND.threshold - 0.25),
                                  step=0.01,
                                  label="exit threshold — hysteresis; cancel below this")
                min_sil = gr.Slider(0, 1000, value=200, step=25,
                                    label="minimum silence (ms) — must hold this long")
                alpha = gr.Slider(0.05, 1.0, value=0.4, step=0.05,
                                  label="EMA alpha — 1.0 disables smoothing")
                with gr.Accordion("Advanced", open=False):
                    hop = gr.Slider(20, 500, value=160, step=20,
                                    label="hop (ms) — how often the model runs")
                    chunk = gr.Slider(10, 100, value=20, step=10,
                                      label="chunk (ms) — simulated WebRTC frame size")

        verdict = gr.Markdown()
        plot = gr.Plot(label="waveform and probability timeline")
        metrics = gr.JSON(label="measured latency and trace summary")

        inputs = [audio, enter, exit_, min_sil, alpha, hop, chunk]
        outputs = [verdict, plot, metrics]
        run.click(analyse, inputs=inputs, outputs=outputs)
        audio.change(analyse, inputs=inputs, outputs=outputs)

        ex = gather_examples()
        if ex:
            gr.Examples(examples=ex, inputs=audio, label="Pre-loaded Hinglish examples")
        else:
            gr.Markdown(
                "_No example clips found. Generate them with_ "
                "`python scripts/synthesize_hinglish.py`."
            )

        gr.Markdown(
            """---
#### What the three parameters do

- **Minimum silence** — natural speech has sub-200 ms gaps: stop closures,
  in-breaths, the pause before a word someone is searching for. Requiring the
  probability to *stay* high separates "they stopped" from "they paused".
- **EMA smoothing** — one window crossing the threshold is noise. A sustained
  rise is a decision.
- **Hysteresis** — a single threshold at 0.5 chatters when the signal hovers near
  it. Entering at 0.70 and exiting at 0.45 means it takes real evidence to
  cancel a decision in progress. Same reason a thermostat has a dead band.

Latency shown is measured end-to-end, from chunk arrival to event emission — not
model forward time alone."""
        )
    return ui


def main(argv=None) -> int:
    global BACKEND
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)

    ckpt, onnx = (args.checkpoint, args.onnx)
    if not ckpt and not onnx:
        ckpt, onnx = find_checkpoint(None)
    BACKEND = Backend(ckpt, onnx)
    print(f"  detector: {BACKEND.info}")

    # Keyword args here are deliberately minimal. `show_api=False` was removed in
    # Gradio 6 and passing it raises TypeError at launch — a crash that only
    # appears when the server actually starts, so it survives any test that just
    # builds the Blocks object. Anything added here must be verified against the
    # pinned Gradio version in requirements.lock.txt.
    build_ui().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
