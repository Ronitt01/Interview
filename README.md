---
title: Smart Turn Detection
emoji: 🎙️
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.25.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Audio-only turn-end detection for voice agents (Whisper Tiny)
---

# Smart turn detection

Audio-only endpointing for voice agents: given a waveform, decide whether the
speaker has **finished their turn**. No transcript, no LLM — the decision is made
from prosody and spectral cues alone, as a rolling window, in real time.

Whisper Tiny encoder + a shallow classification head (7.67 M parameters), wrapped
in a streaming detector with smoothing, minimum-silence and hysteresis, plus a
Gradio demo you can talk into.

---

## Why this is not just a silence timer

The naive rule — wait for N ms of silence — cannot work, because pause duration
does not separate the two cases. Someone searching for a word pauses 300–600 ms.
Someone who has finished pauses 300–600 ms.

Measured here, on 400 clips of the published test set:

| detector | F1 | false interruption | ROC-AUC |
|---|---|---|---|
| E0 — energy trailing-silence timer | 0.2833 | 0.0977 | 0.6602 |
| E0b — Silero VAD trailing-silence timer | 0.2468 | 0.0977 | 0.6433 |
| **E0 run as a real streaming detector** | **0.0000** | 0.1951 | — |

All three held to a shared 10% false-interruption ceiling. The energy gate
*beating* Silero is the informative part: knowing where speech **is** does not
tell you whether a thought is **finished**. And streamed, the silence timer
collapses entirely — every apparent hit was a fire seconds before the utterance
ended, on an internal pause.

Full reasoning and every number: **[report/technical_report.md](report/technical_report.md)**.

### The finding worth reading first

A clip-level score **does not predict streaming behaviour**. Same model, same
threshold, measured two ways:

| evaluation | F1 | false interruption |
|---|---|---|
| clip-level classification | 0.3009 | **0.033** |
| streamed, 20 ms chunks | 0.0270 | **0.460** |

A 14× worse interruption rate. Not a bug in the streaming code — a distribution
shift the training data does not cover. In training, every window's right edge
lands on an annotated boundary; in streaming it lands wherever the hop falls, so
the model fires on ordinary intra-utterance pauses no training example ever
labelled.

And tightening the streaming parameters **does not** fix it — measured across a
threshold/min-silence grid, the detector jumps straight from interrupting 46% of
turns to never firing at all, with no usable point in between. The fix has to be
in training (randomly-offset crops), not in tuning.

This is why `src/streaming.py` is an evaluated component rather than a wrapper
assumed to inherit the model's accuracy. A submission that only classified clips
would ship a detector interrupting 46% of held turns while reporting 3%.
Details in [§6.1](report/technical_report.md).

---

## Quickstart

```bash
# 1. environment (Python 3.13; CPU wheels for local work)
python -m venv .venv
.venv/Scripts/python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -r requirements.txt

# 2. data — a small slice first, to prove the path end to end in minutes
.venv/Scripts/python scripts/prepare_data.py --split test --all-languages --max-rows 400
.venv/Scripts/python scripts/prepare_data.py --split train --all-languages --max-rows 1200

# 3. Gate 3 — baselines before any neural work
.venv/Scripts/python scripts/run_baselines.py --cache data/cache/test --slices

# 4. prove the training wiring, then train
.venv/Scripts/python -m training.train --config configs/e1_frozen_linear.yaml --dry-run
.venv/Scripts/python -m training.train --config configs/e1_frozen_linear.yaml

# 5. the demo (works before training — falls back to E0 and says so)
.venv/Scripts/python demo/app.py
```

On Linux/macOS/Colab use `python` and `pip` in place of `.venv/Scripts/python`.
`scripts/setup.sh` does steps 1–2 in one go.

Scaling up to the real corpus is one flag — see [Data](#data).

---

## Layout

| path | what it is |
|---|---|
| [src/audio.py](src/audio.py) | waveform front end: mono, resample, normalise, **left-pad**. Every decision has its reason in the docstring beside it |
| [src/features.py](src/features.py) | log-mel matched to the Whisper encoder, at the window length we actually use rather than Whisper's 30 s default |
| [src/splits.py](src/splits.py) | group-aware partitioning and the leakage assertion that runs on every training run |
| [src/dataset.py](src/dataset.py) | streaming download → int16 memmap cache → torch Dataset |
| [src/augment.py](src/augment.py) | augmentation, and which augmentations are *illegal* on this task |
| [src/model.py](src/model.py) | Whisper Tiny encoder + `linear` / `mlp` / `gru` / `attn` heads |
| [src/metrics.py](src/metrics.py) | false-interruption rate, missed-endpoint rate, operating-point selection |
| [src/evaluation.py](src/evaluation.py) | slice reports and the auto-appending experiment table |
| [src/baselines.py](src/baselines.py) | E0 energy gate, E0b Silero VAD |
| [src/streaming.py](src/streaming.py) | the rolling-window detector — smoothing, min-silence, hysteresis |
| [src/inference.py](src/inference.py) | standalone predictor. Imports nothing from `training/` |
| [src/optimize.py](src/optimize.py) | quantisation, ONNX export, CPU benchmarking |
| [training/train.py](training/train.py) | one config in, one table row out. Checkpoints every epoch, resumes automatically |
| [configs/](configs/) | 12 experiment configs, one variable changed each |
| [scripts/](scripts/) | the twelve commands in [§14 of the report](report/technical_report.md) |
| [data/hinglish/](data/hinglish/) | 44 hand-labelled Hinglish phrases + the Bulbul renderer |
| [demo/app.py](demo/app.py) | Gradio demo with a live probability timeline (CLI) |
| [app.py](app.py) | Hugging Face Space entry point — thin adapter over `demo/app.py` |
| [notebooks/](notebooks/) | EDA (executed, outputs committed) and error analysis |
| [tests/](tests/) | 81 tests. Each pins a decision, not just a code path |

---

## Data

The published corpus is large:

| | rows | size |
|---|---|---|
| `pipecat-ai/smart-turn-data-v3.2-train` | 270,946 | 41.4 GB |
| `pipecat-ai/smart-turn-data-v3.1-test` | 31,473 | 4.3 GB |

Both are streamed, filtered, and cached locally as 16 kHz int16, so a session
that dies mid-download only loses the rows not yet written.

```bash
# the working subset: English + the three Indic languages present
python scripts/prepare_data.py --split train --languages eng hin ben mar --max-rows 40000
python scripts/prepare_data.py --split test  --languages eng hin ben mar

# only human-recorded audio — the honest headline slice (82% of the corpus is TTS)
python scripts/prepare_data.py --split train --human-only

# everything
python scripts/prepare_data.py --split train --all-languages
```

`python scripts/prepare_data.py --info` prints the sizes and the disk estimate
before committing to a download.

---

## Answers to the review questions

**What problem does this solve?** A voice agent needs to know when to start
talking. Endpointing on silence duration alone cannot work (see above), so this
predicts turn completion from the audio directly.

**What is the model?** `openai/whisper-tiny` encoder (frozen by default) →
pooling → a shallow classifier. 7.67 M parameters measured. Deliberately the same
architecture as the published `smart-turn-v3` so our numbers sit next to a public
reference (8 MB int8 ONNX, ~10–12 ms CPU) rather than floating free.

**Why Whisper Tiny?** Its encoder is pretrained on a very large amount of
multilingual speech, so its features already encode prosody and phonetics; it is
small enough to fine-tune on a free T4 and to quantise to single-digit MB; and it
is the reference implementation's choice, which makes the comparison meaningful.

**Why is the encoder frozen?** A frozen encoder trains in minutes and gives a
clean read on whether the head can separate the classes at all. If it cannot, a
bigger fine-tune is unlikely to rescue it and the problem is upstream in the
data. Unfreezing is E7 — an experiment, not the default.

**How is the data split?** By **source corpus** (`dataset`), not randomly. The
corpus carries **no speaker ID** — verified, 1,200 unique clip UUIDs across 1,200
rows — so a speaker-aware split is impossible and is not claimed. The published
train/test split is used as given; validation is carved out of train, grouped.
Two leakage assertions run on every training run and abort it on failure.

**What about leakage?** Asserted in code (`src/splits.py:assert_no_leakage`),
twice: group overlap and clip-ID overlap. Ten tests cover it, including one that
deliberately leaks and asserts the exception fires.

**What metrics, and why not accuracy?** **False-interruption rate**
(FP/(FP+TN)) and **missed-endpoint rate** (FN/(FN+TP)), reported with F1,
precision, recall, ROC-AUC, PR-AUC and the confusion matrix. Accuracy hides the
asymmetry: interrupting a user is far worse than making them wait. One false
interruption is weighted as four missed endpoints, and every row is held to a
shared 10% interruption ceiling so the rows are comparable.

**How is the operating threshold chosen?** On **validation**, then applied
unchanged to test. Re-picking on test would report the best case for a threshold
nobody could have known in advance. Note the cost function's degenerate corner:
for any weak detector, `4·FPR + FNR` is minimised by *never firing*. That is
excluded from selection and reported explicitly as a finding.

**Does it stream?** Yes — that is the point.
`src/streaming.py:StreamingTurnDetector` takes arbitrary audio chunks, runs the
model once per hop, and emits `USER_TURN_ENDED` after EMA smoothing, a
minimum-silence hold and a hysteresis band. Latency is measured end to end, from
chunk arrival to event emission.

**How fast and how small?** `scripts/optimize_model.py` produces float32 torch,
int8 torch, float32 ONNX and int8 ONNX, and reports each step's size, latency
(p50/p95/p99, never a mean) and accuracy delta separately — including when a step
makes accuracy worse.

**What about Hindi and Hinglish?** The corpus does contain `hin`, `ben` and `mar`,
reported as separate slices. What it lacks is code-switched Hinglish, Indian
English accent and domain vocabulary, so `data/hinglish/` adds 44 hand-labelled
phrases across 7 categories rendered by Bulbul TTS — 202 clips, including
complete utterances with real silence spliced mid-utterance. Reported as its own
table, never averaged in.

**What are the limitations?** Listed plainly in
[§13 of the report](report/technical_report.md). The two that matter most:
**clip-level scores do not transfer to streaming** (§6.1, and parameter tuning
does not close it), and the verification model scored **F1 0.349 on TTS audio
against 0.054 on human audio** — a 6.5× gap indicating it substantially learned
TTS artefacts, which is why `--human-only` exists and why the `synthetic=False`
slice is the honest headline. Also: no speaker-aware split is possible, the
Hinglish set is TTS, noise/reverb augmentation is synthetic, and the currently
measured numbers are on subsets sized to prove the pipeline.

---

## Tests

```bash
.venv/Scripts/python -m pytest        # 81 tests
```

They pin decisions rather than code paths. A sample:

- resampling attenuates an out-of-band tone far more than index striding, so
  aliasing cannot become a feature the model learns;
- `prepare()` normalises before padding, so loudness cannot correlate with clip
  duration;
- masking augmentation never touches the tail, checked over 40 seeds, because a
  masked tail silently relabels the clip;
- the hysteresis band holds state across a flapping score, and an inverted band
  is rejected at construction;
- the operating-point selector refuses the degenerate never-fire solution;
- a fire more than 1 s before the clip end is scored as an interruption, not a
  hit.

---

## Deploying the demo

Two entry points, deliberately:

| file | contract | use |
|---|---|---|
| [demo/app.py](demo/app.py) | argparse CLI — `--checkpoint`, `--onnx`, `--port`, `--share` | local development |
| [app.py](app.py) | launches on import at `0.0.0.0:7860` | what a Hugging Face Space runs |

`app.py` is a thin adapter; all UI and inference logic lives in `demo/app.py`, so
there is no second copy to keep in sync.

```bash
python app.py                                    # exactly what the Space runs
python demo/app.py --onnx weights/*-int8.onnx    # local, explicit weights
docker build -t smart-turn . && docker run -p 7860:7860 smart-turn
```

**Hugging Face Space.** The YAML front matter at the top of this README is the
Space configuration (`sdk: gradio`, `app_file: app.py`), so pushing this repo to
a Space works without edits. Both entry points resolve weights through
`find_checkpoint`, which prefers an int8 ONNX graph and falls back to the E0
energy baseline with a visible in-UI notice if nothing is trained yet — a Space
that boots and explains itself beats one that shows a stack trace.

### Which weights are in git

| tracked | not tracked |
|---|---|
| `weights/*.onnx` + `*.json` sidecars — the ~8 MB int8 deployment artefact, so a fresh clone can run the demo immediately | `weights/*.pt` — ~31 MB torch checkpoints, needed only to retrain or re-export |

Regenerate the `.pt` files with `python -m training.train --config ...`, then
`python scripts/optimize_model.py --checkpoint ...` to re-export the ONNX.

---

## Environment

Copy `.env.example` to `.env`. Only one key is needed, and only for regenerating
the Hinglish clips:

```
SARVAM_API_KEY=...      # https://dashboard.sarvam.ai — Bulbul TTS
```

Everything else — training, evaluation, the demo — runs without any credentials.
