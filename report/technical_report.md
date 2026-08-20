# Audio-only turn detection — technical report

Given a waveform, decide whether the speaker has finished their turn. No
transcript, no LLM.

**How to read the numbers in this document.** Every figure is labelled with the
run that produced it. Three labels are used and they mean different things:

| label | meaning |
|---|---|
| *measured* | produced by a command in this repo, on the data named beside it |
| `TBD` | the experiment is running now; the cell will be filled from its artefacts |
| *not run* | deliberately not executed, with the reason given |

No figure in this report is estimated, extrapolated, or carried over from a
published result. Where a number is absent, the cell says so.

**Current state in one paragraph.** The pipeline is built and tested end to end.
Two baselines and one full-scale model (E1) have been trained and scored on a
held-out test set. E1 has strong ranking ability (test ROC-AUC 0.8900) but its
reported operating point is extremely conservative — it misses 85.09% of
endpoints — and §9.6 identifies the threshold-selection rule, not the model, as
the first thing to fix. A window sweep (E2) is running. The streaming detector is
built, measured, and **measured to be inadequate as currently trained** (§11.2);
the cause is identified and the fix is scoped but not implemented.

---

## 1. Problem

A voice agent has to decide when to start talking. That decision is made from
audio, in real time, before the user's sentence has been transcribed — because
waiting for an ASR result and then an LLM judgement adds latency the user hears
as a lag.

Two errors, with very different costs:

| error | what the user experiences | cost |
|---|---|---|
| **False interruption** — fires while the user is mid-thought | the agent talks over them; they stop, get confused, repeat themselves | high — this is the failure that makes an agent feel broken |
| **Missed endpoint** — waits after the user finished | the agent feels a beat slow | low — annoying, forgivable |

This report is organised around that asymmetry. `src/metrics.py` weights one
false interruption as four missed endpoints (`DEFAULT_FP_COST = 4.0`), and
`DEFAULT_FI_BUDGET = 0.10` defines a 10% false-interruption ceiling. Both are
judgement calls, written as named constants so they can be argued with rather
than buried. §9.6 shows that which of these two rules is used to pick the
threshold changes the headline F1 by more than any modelling decision made so
far.

### Reference point

The published `pipecat-ai/smart-turn-v3` is a Whisper Tiny encoder plus a
shallow linear classifier: ~8 M parameters, 8 MB int8 ONNX / 32 MB unquantised,
~10–12 ms CPU. **This project uses the same architecture deliberately**, so the
measurements here sit next to a public reference instead of floating free. A
wildly different size or latency figure would mean a measurement bug, not a
breakthrough. This report does not claim to beat that reference; it claims to
measure a comparable system honestly, including where it fails.

---

## 2. Why turn detection is difficult

**Pause duration does not separate the two cases.** A speaker searching for a
word pauses 300–600 ms. A speaker who has finished pauses 300–600 ms. There is no
silence threshold that splits them. This is not a claim inherited from a paper —
§6 measures it: a tuned energy-gated silence timer reaches ROC-AUC 0.6602, and
when run as a real streaming detector its F1 falls to **0.0000**, because every
apparent success was a fire seconds before the clip actually ended.

Four properties make the task harder than it first looks.

**The evidence is localised but the context is not.** Whether a turn has ended is
mostly decided in the last few hundred milliseconds — falling pitch, decaying
energy, a filler that trails off. But *how* to read those cues depends on what
came before: the same flat intonation ends a short confirmation and sits
mid-sentence in a long explanation. So a short window may miss the context and a
long window may dilute the evidence. Which effect dominates is an empirical
question, and it is what E2 (§10) exists to answer.

**The two error directions have different costs, so accuracy is the wrong
metric.** A detector that never fires has a 0% false-interruption rate and is
useless. A detector optimised for F1 can interrupt most held turns — E0's
best-F1 threshold interrupts **78.6%** of them (§6.2). Any single-number summary
hides this, which is why every row here carries both error rates.

**The cost function has a degenerate corner.** `4·FPR + FNR` is minimised by
never firing, for any detector whose ranking is not strong enough to buy a full
point of FNR for a quarter-point of FPR. This is not hypothetical: it is what the
baselines do (§6.2) and, as §9.6 shows, it is what pushed E1's operating point
down to 7.4% validation recall.

**Offline clip classification does not predict streaming behaviour.** This is the
most consequential finding in this report and it has its own section (§11.2). In
training, every window's right edge coincides with an annotated boundary. In
streaming, the right edge is wherever the hop lands. For the same model at the
same threshold, the false-interruption rate went from 0.033 to 0.460 — a 14×
degradation — when evaluated as a stream rather than as clips.

---

## 3. Dataset

| | rows | size |
|---|---|---|
| `pipecat-ai/smart-turn-data-v3.2-train` | 270,946 | 41.4 GB |
| `pipecat-ai/smart-turn-data-v3.1-test` | 31,473 | 4.3 GB |

Columns: `audio`, `audioduration`, `id`, `language`, `endpoint_bool` (the label),
`midfiller`, `endfiller`, `synthetic`, `dataset`, `spoken_text`.

**A published train/test split already exists**, so it is used rather than
reinvented. Validation is carved out of train.

### 3.1 What was actually cached and trained on

*Measured*, from the cache manifests:

| | train | held-out test |
|---|---:|---:|
| clips | 40,000 | 10,791 |
| audio | 88.51 h | 24.43 h |
| cache size | 9.6 GB | 2.81 GB |
| positive rate | 0.4956 | 0.4960 |
| `eng` | 28,548 | 7,722 |
| `hin` | 5,268 | 1,295 |
| `ben` | 3,413 | 1,000 |
| `mar` | 2,771 | 774 |

Both caches are restricted to `eng hin ben mar`. Validation is 20% of the 40,000
train clips, grouped by source corpus.

### 3.2 The four questions answered before any training

*Measured* on a 1,200-clip sample of v3.2-train (`notebooks/eda.ipynb`, executed
with outputs committed). These were checked before writing a model, because each
answer changes a design decision.

**Are there speaker IDs? No.** 1,200 unique `id` values across 1,200 rows — `id`
is a per-clip UUID. There is no speaker, voice, or session column.

> **A speaker-aware split is therefore impossible and is not claimed.** Splits
> are grouped by `dataset` — the source corpus (11 values in the sample:
> `chirp3_1`, `chirp3_2`, `liva_1`, `midcentury_1`, `rime_2`, `mundo_1`,
> `human_5`, `orpheus_grammar_1`, `orpheus_endfiller_1`, `orpheus_midfiller_1`,
> `human_convcollector_1`). Grouping by source prevents a corpus — and therefore
> its speakers, recording chain, and TTS voices — from spanning two splits.
> **Residual risk:** if one human speaker contributed to two source corpora they
> can still appear on both sides. No published metadata lets us detect that, so
> it is stated as a known limitation rather than ruled out.

**Class balance: 620 / 580, ratio 1.07×.** Essentially balanced, and the
full-scale caches confirm it (positive rate 0.4956 train, 0.4960 test). The plan
anticipated severe imbalance; it did not materialise. `pos_weight` is applied
only when the ratio exceeds 1.2.

**Is there Indic audio? Yes — and this changed the plan.** 23 languages, with
`hin` 62, `ben` 34, `mar` 33 in the sample (~10.7% Indic). The plan assumed none.
So a real held-out Indic slice exists and is reported separately. What is still
absent — and what `data/hinglish/` supplies — is *code-switched* Hinglish,
Indian-English accent, and logistics-domain vocabulary.

**Duration: median 7.16 s, p95 13.8 s, max 22.2 s.** This sets the window:

| window | share of clips fully contained |
|---|---|
| 0.5 s | 0.0% |
| 1.0 s | 0.1% |
| 2.0 s | 0.7% |
| 4.0 s | 13.2% |
| 8.0 s | 59.2% |

An 8 s window — the reference's ceiling — still truncates 41% of clips. That is
acceptable *because the trim is from the left*: the endpoint evidence is in the
tail. It is also why E2 sweeps downward from 8 s rather than upward.

**82% of the corpus is synthetic** (986 / 214 in the sample). Every headline is
therefore also reported on the `synthetic=False` slice — and §14.5 shows that
distinction is not academic.

### 3.3 The most useful thing found in the data

`endfiller=True` has a **positive rate of 0.000 across 302 clips**. A trailing
filler *never* marks a turn end. The corpus has pre-labelled the exact
false-interruption trap:

| annotation | n | positive rate |
|---|---|---|
| `endfiller=True` | 302 | **0.000** |
| `midfiller=True` | 468 | 0.515 |
| neither (annotated) | 276 | 0.884 |
| unannotated (`null`) | 263 | 0.513 |

Two consequences. First, the error modes that would otherwise need
hand-categorising are already labelled, so "the model struggles with hesitation"
becomes a number instead of an impression (§14). Second, these flags are
**tri-state** — `True`, `False`, or `null` — and folding `null` into `False`
would mix "annotated as no filler" with "unknown". `src/evaluation.py` gives
`null` its own slice.

---

## 4. Data preparation

Order: **mono → resample → normalise → window**. Implemented in
`src/audio.py:prepare`. Each step earns its place:

| decision | reason |
|---|---|
| Average channels rather than take channel 0 | some corpora put the speaker on one channel and near-silence on the other; picking blind would silently reduce half the data to noise |
| Resample with a windowed-sinc filter, not index striding | striding aliases — a 15 kHz component in a 44.1 kHz clip folds down into the speech band and the model learns our artefact. Tested in `tests/test_preprocessing.py` |
| Peak-normalise to −1 dBFS | loudness varies with recording conditions and carries no information about turn completion. Un-normalised, a model can score well by learning *which corpus* a clip came from, since gain correlates with source. *Measured* spread: RMS p5 −23.2 dBFS, p95 −15.9 dBFS |
| Normalise **before** windowing | a short clip padded to 8 s and then normalised would be scaled against its own padding; under RMS normalisation loudness would then correlate with duration |
| Headroom of 1 dB, never full scale | leaves room so a later resample or int16 round-trip cannot clip |
| **Left-pad; left-trim** | the turn boundary is always the *end* of the clip. Keeping audio flush to the right edge puts the decision-relevant moment at a fixed input position for every sample. Right-padding would slide it around with clip length and force the model to learn a position-invariance it never needs. This matches the reference implementation |

### 4.1 Cache design

One streaming pass writes a flat int16 memmap plus an offsets index. What is
cached is the **resampled waveform at its natural length** — not mel, not a fixed
window — and that choice is what keeps the experiment matrix open. Caching mel
would freeze the window length and make E2 impossible without a re-cache;
caching a fixed 8 s window would waste ~2× the disk and turn the short-window
sweep into a crop of padding rather than of audio.

`datasets` ≥ 5 routes audio decoding through `torchcodec`, which needs a matching
FFmpeg and is awkward on Windows and some Colab images. We set `decode=False` and
decode with `soundfile` — one less native dependency, identical result, and the
same decode path the demo uses for uploads.

---

## 5. Leakage and split methodology

`src/splits.py:assert_no_leakage` runs two independent checks on every training
run, and the run aborts if either fails:

1. **Group overlap** — catches the split logic being wrong.
2. **Clip-ID overlap** — catches the same clip appearing under two source names,
   which grouping alone cannot detect.

Nine tests in `tests/test_splits.py` cover both, including one that deliberately
leaks and asserts the exception is raised — because an assertion nothing tests is
an assertion nobody knows works.

*Measured* on the 1,200-clip development cache: train 961 / val 239 (80/20
requested), positive rate 0.517 / 0.515, 4 groups / 7 groups, leakage assertion
passed. The same assertion runs on the 40,000-clip cache at the start of every
full run, including E1 and every E2 window.

### 5.1 Test-set discipline

Stated as rules, because the value of a held-out number is entirely in whether
these were followed:

- The threshold is selected on **validation** and applied unchanged to test.
  `training/train.py` prints `re-selected on test? : NO` and names the function
  that is not called.
- No threshold has been tuned on test, for any run, at any point.
- Test is read once per reported model, for final scoring only.
- The five E2 windows carry `test_cache_dir: null`, so the sweep cannot touch
  test even accidentally. See §10.3.
- Train and test come from two separately-published datasets, so they share no
  source rows by construction. Validation is carved from train, grouped by
  source corpus.

One caveat that cuts the other way, and belongs here rather than in a footnote:
**the baselines do not meet this standard.** `scripts/run_baselines.py` sweeps
the baseline threshold on the same 400 clips it then reports
(`src/baselines.py:evaluate_baseline` calls `evaluate` on the cache it was
handed). That is an in-sample threshold — the best case for a threshold nobody
could have chosen in advance. So E0 and E0b are optimistic bounds while E1 is
not, and §9.5 explains why that makes the obvious E0-vs-E1 comparison invalid as
it currently stands.

The one thing that is *not* claimed anywhere: a speaker-disjoint split. It is
impossible with this metadata (§3.2) and its absence is a real limitation (§17).

---
## 6. Baselines

A model with no baseline beside it is an unfalsifiable claim. Two were built
before any neural work.

Both score a clip identically: **the score is the duration of trailing silence in
milliseconds.** They differ only in how they decide which frames are speech — a
fixed energy gate for E0, Silero VAD for E0b. Emitting a continuous score rather
than a hardcoded 500 ms rule means the same threshold sweep and the same
operating-point selection run on the baseline as on the model, and the chosen
threshold is directly interpretable: "this baseline fires after 485 ms of
silence."

### 6.1 Measured — clip-level, on 400 clips of v3.1-test

```
python scripts/run_baselines.py --cache data/cache/test --slices
```

| id | detector | thr | F1 | recall | false interrupt | missed | ROC-AUC | size |
|---|---|---|---|---|---|---|---|---|
| E0 | energy-gated trailing silence | 485 ms | 0.2833 | 0.1838 | 0.0977 | 0.8162 | 0.6602 | 0 MB |
| E0b | Silero VAD trailing silence | 706 ms | 0.2468 | 0.1568 | 0.0977 | 0.8432 | 0.6433 | 10.0 MB |

Threshold selected under the shared 10% false-interruption ceiling, **in-sample**
(§5.1).

**The energy gate slightly beats Silero.** That is worth dwelling on: Silero knows
what speech is far better than an energy threshold does, and it still loses.
Knowing where speech *is* does not tell you whether a thought is *finished*. This
is direct evidence that the task is not silence detection — and it is the reason
E0b exists rather than E0 alone.

**These numbers are on a 400-clip subset, not the 10,791-clip test cache used for
E1.** Re-running them on the full cache is CPU-only work and is listed as an open
item (§18.2, item 6). Until it is done, E0 and E1 are not directly
comparable.

### 6.2 The degenerate operating point — a finding, not a number to report

At the cost-optimal threshold both baselines **stop firing entirely**:

| detector | operating point | thr | F1 | recall | false interrupt |
|---|---|---|---|---|---|
| E0 | best F1 | 35 ms | 0.6565 | 0.9351 | **0.7860** |
| E0 | FI budget ≤ 10% *(reported)* | 485 ms | 0.2833 | 0.1838 | 0.0977 |
| E0 | cost-optimal | 2610 ms | 0.0000 | 0.0000 | 0.0047 |

The cost function `4·FPR + FNR` is minimised by never firing: never firing scores
`4·0 + 1 = 1.0`, and a detector with ROC-AUC 0.66 cannot buy a full point of FNR
for less than a quarter-point of FPR. The arithmetic correctly concludes *"the
safest thing to do with a bad endpoint detector is never interrupt anyone."*

That is a true statement and a useless operating point. So `pick_threshold`
excludes it by default (`MIN_USEFUL_RECALL = 0.05`), `operating_points` reports it
explicitly flagged `degenerate`, and the row that goes in the table is the one at
the shared 10% ceiling. Note also E0's best-F1 point interrupts **78.6%** of held
turns — which is why one shared ceiling across all detectors is what makes the
table meaningful at all.

**This finding comes back.** §9.6 shows the same degenerate pull acting on E1 —
which has ROC-AUC 0.8900, not 0.66 — and that it was not caught there because the
trained configs never set the ceiling.

### 6.3 Streamed E0 — F1 collapses to zero

```
python scripts/stream_eval.py --baseline --cache data/cache/test --max-rows 60
```

| detector | F1 | recall | false interrupt | early fires |
|---|---|---|---|---|
| E0, streamed | **0.0000** | 0.0000 | 0.1951 | 4/60 |

Run as a real rolling-window detector, the energy baseline is worthless. Every
apparent success was a fire *seconds before* the clip ended — it triggered on an
internal pause, mid-utterance.

This required fixing the scoring, and the fix matters. A scorer that only asks
"did it fire on a positive clip?" records those as true positives, because the
clip does eventually end. But firing mid-utterance **is** an interruption; the
clip ending later does not redeem it. `src/streaming.py:EARLY_FIRE_TOLERANCE_MS`
counts a fire more than 1 s before the clip end as a false interruption
regardless of label. Before that fix, streamed E0 reported `ttd_p50 = −7780 ms` —
a "time to decide" 7.8 seconds *before* the boundary — which is the tell that the
metric was measuring the wrong thing.

**This is the strongest single argument for the whole project:** duration of
silence cannot do streaming endpointing, because natural speech contains pauses
indistinguishable from turn ends by duration alone.

---

## 7. Model architecture

Whisper Tiny encoder + a swappable classification head. **7.67 M parameters
measured** for the linear variant, matching the reference's "~8M".

Two implementation points that matter:

**Positional-embedding truncation.** `openai/whisper-tiny` ships embeddings for
1,500 encoder positions — 30 s of audio, because that is what Whisper's ASR
decoder expects. Our window is at most 8 s. Feeding 30 s of mostly-zeros costs
roughly 4× the encoder FLOPs for no information — the difference between a 12 ms
and a 50 ms inference. `src/model.py:build_backbone` truncates `embed_positions`
*and* rewrites `config.max_source_positions` to agree; leaving them inconsistent
trips an internal shape assertion.

**Freezing is the default; unfreezing is an experiment.** A frozen encoder trains
in minutes on a free T4 and gives a clean read on whether the head can separate
the classes at all. If it cannot, a larger fine-tune is unlikely to rescue it and
the problem is upstream in the data. E1–E5 run frozen (1.2 k trainable
parameters); E7 unfreezes the top 2 blocks (3.55 M trainable, *measured*).

### 7.1 Heads

All four are implemented and verified to forward correctly. Only `linear` has
been trained at full scale.

| head | params | hypothesis under test | trained? |
|---|---|---|---|
| `linear` | 1,153 | the reference design; can frozen features separate the classes at all? | **yes — E1** |
| `mlp` | 25,473 | is the *head* the bottleneck, or the frozen features? | no — E3 config exists |
| `gru` | 173,697 | endpointing is about *trajectory* — falling pitch, decaying energy, a filler that trails. Pooling throws ordering away; a recurrent head keeps it | no — E4 config exists |
| `attn` | 1,538 | mean pooling dilutes evidence living in the last few hundred ms against seconds of mid-utterance. Isolates *pooling* from head capacity | no — E4b config exists |

A bidirectional GRU is defensible despite this being a streaming problem: the
detector runs on a completed window, so the backward pass only ever sees audio
that has already arrived.

---

## 8. Training methodology

### 8.1 The loop

Per-epoch: BCE-with-logits (`pos_weight` applied only when the class ratio
exceeds 1.2), AdamW, OneCycle LR when the run is long enough to schedule
(`MIN_SCHEDULED_STEPS = 20`, flat LR below that), AMP on CUDA, early stopping on
validation cost with patience 3. Every epoch checkpoints unconditionally and
resumes automatically, because a free Colab session disconnects without warning
and an uncheckpointed run is a lost day. Verified working — a dry run resumed
from epoch 1 on re-invocation.

Every run is driven by a YAML config, and an unknown key **raises** rather than
being ignored: a silently typo'd key means the experiment you think you ran is
not the one that ran.

### 8.2 How results are scored

```
false_interruption_rate = FP / (FP + TN)    # among turns that had NOT ended
missed_endpoint_rate    = FN / (FN + TP)    # among turns that HAD ended
time_to_decide          = ms from turn end to the emitted decision
```

Accuracy is close to useless here and F1 alone is not much better, because the
two error directions have different costs. Reported alongside: precision, recall,
specificity, balanced accuracy, ROC-AUC, PR-AUC, and the full confusion matrix.

Three deliberate choices:

- **Threshold chosen on validation, applied unchanged to test.** Re-picking on
  test reports the best case for a threshold nobody could have known in advance.
- **Latency as p50/p95/p99/max, never a mean.** A mean hides the tail and the
  tail is what a user hears. `percentiles()` does not expose a mean at all —
  enforced by a test.
- **Single-class slices return `NaN` for AUC rather than 0.0**, because a
  per-language slice is often single-class and "0.0" would read as "the model
  failed on this language" instead of "undefined".

**And one choice that turned out to be a defect.** `pick_threshold` has two
modes: minimise `fp_cost·FPR + FNR`, or maximise recall subject to a
false-interruption ceiling. Which one runs depends on `max_false_interruption`,
which defaults to `None`. **No config in `configs/` sets it**, so every trained
run selects by cost minimisation, while the baselines — whose helper defaults to
`DEFAULT_FI_BUDGET` — select under the 10% ceiling. Two different rules across
rows of the same table. §9.6 works through what that did to E1.

### 8.3 The pipeline-verification run

To exercise the whole chain on real audio before committing GPU hours, a
deliberately small run was trained: 1.0 s window, 3 epochs, 600 train rows,
frozen encoder (`configs/smoke_e1.yaml`, run id `SMOKE`). **This is not a
reportable result** — it is far too small — but it is a genuinely trained model,
and because most of the downstream tooling (streaming, optimisation, error
analysis) was measured against it, its numbers appear in this report and are
labelled `SMOKE` wherever they do.

| detector | F1 | false interrupt | missed | threshold |
|---|---|---|---|---|
| E0 | 0.2833 | 0.0977 | 0.8162 | 485 ms |
| E0b | 0.2468 | 0.0977 | 0.8432 | 706 ms |
| **SMOKE** (3 epochs, 600 rows) | **0.3009** | **0.0326** | 0.8162 | 0.681 |

Training loss fell 0.684 → 0.547 → 0.511. ROC-AUC **0.766 against 0.660 (E0) and
0.643 (E0b)**. An encouraging sign for E1 rather than a result to quote.

### 8.4 Augmentation — and which augmentations are illegal here

Most augmentation libraries offer a grab-bag. On this task some entries are
actively harmful, and knowing which is part of understanding the problem. Nothing
below has been used in a reported run; E5 is configured and not yet trained.

**Safe** — changes the channel, not the boundary: gain, additive noise,
band-limiting, reverb. A clip that ended still ended after you add noise.

**Dangerous:**

- *Time-stretch / speed* alters pause duration. A 300 ms mid-sentence pause
  stretched 1.3× becomes 390 ms — squarely turn-final. **The label stops
  describing the audio.** Kept but bounded to ±10% and **off by default**.
- *Cropping from the end* removes the endpoint evidence and relabels a positive
  as a negative. Never applied. (Note the contrast with §12: random-offset
  cropping is safe *only* because it re-derives the label from the crop.)
- *Time-masking near the tail* is the same problem in weaker form. Masking is
  confined to the first 70% of the clip — enforced by a test that runs 40 seeds
  and asserts the tail is untouched.

Augmentation is seeded per-sample from the clip index: unseeded augmentation
makes an experiment table unreproducible in a way that is very hard to notice.
Ten tests in `tests/test_augment.py`, including one asserting every non-speed
augmentation is length-preserving.

---
## 9. E1 — baseline results

E1 is the go/no-go run: the reference architecture, trained at full scale, scored
once on a held-out test set. It is a **baseline, not the final system**, and the
sections below say why in specific terms.

### 9.1 Setup

| | |
|---|---|
| config | `configs/e1_frozen_linear.yaml` |
| architecture | Whisper Tiny encoder (frozen) → mean pool → linear head |
| trainable parameters | 1,153 of 7.67 M |
| window | 8.0 s, left-trimmed / left-padded |
| features | 80-bin log-mel, hop 160, encoder stride 2 |
| loss | BCE-with-logits, `pos_weight` conditional on class ratio > 1.2 |
| optimiser | AdamW, lr 1e-3, weight decay 0.01, OneCycle |
| epochs | 8, early stopping on validation cost, patience 3 |
| batch size | 32, AMP enabled |
| seed | 1234 |
| hardware | Colab T4, Python 3.12.13, PyTorch 2.11.0+cu128, CUDA 12.8 |

### 9.2 Why a frozen Whisper Tiny encoder is the right first experiment

Three reasons, in order of importance.

**It answers a question that has to be answered first.** If a linear probe on
frozen features cannot separate turn-end from mid-turn, then the features do not
carry the distinction and no amount of head capacity or fine-tuning is the fix —
the problem is upstream, in the labels or the preprocessing. A frozen probe is
the cheapest instrument that can tell you that, and running it before anything
expensive is the point.

**It is the published reference architecture**, not our invention (§1). Matching
it means our parameter count, artefact size, and latency can be checked against
public numbers — which is how §15 caught a measurement error that would otherwise
have gone unnoticed.

**It fits the deployment constraint by construction.** 1,153 trainable
parameters train in minutes on a free T4, and the resulting artefact is the same
8 MB int8 ONNX graph the reference ships. Starting from something that already
meets the size and latency budget means every later experiment is a question
about accuracy alone, not a negotiation with the budget.

The cost of this choice is stated in §9.7: a frozen encoder was never trained on
this task, so the features are Whisper's ASR representation, and mean pooling
discards the temporal ordering that endpointing arguably depends on most.

### 9.3 Data

Trained on the 40,000-clip cache (88.51 h, positive rate 0.4956), validated on a
20% group-held-out slice of it, tested once on the 10,791-clip cache built from
the separately published `v3.1-test` (24.43 h, positive rate 0.4960). Language
composition for both is in §3.1. Leakage assertions passed at the start of the
run.

### 9.4 Validation and test methodology

The threshold was selected on validation by `src/metrics.py:pick_threshold` and
applied unchanged to test. **0.909298.** No threshold was tuned on test. The test
cache was read once, for final scoring.

### 9.5 Results

*Measured.* Threshold 0.909298 for both columns.

| metric | validation | held-out test |
|---|---:|---:|
| n | — | 10,791 |
| F1 | 0.1363 | 0.2585 |
| precision | 0.8395 | 0.9696 |
| recall | 0.0742 | 0.1491 |
| false-interruption rate | 0.0142 | 0.0046 |
| missed-endpoint rate | 0.9258 | 0.8509 |
| ROC-AUC | TBD | 0.8900 |
| PR-AUC | TBD | 0.8757 |

Validation ROC-AUC and PR-AUC are `TBD` because they were not transcribed with
the rest; they are in `artifacts/runs/E1/evaluation.json`, which has not yet been
synced out of the Colab session.

**On comparing this to E0.** E0's test F1 is 0.2833 against E1's 0.2585, and the
tempting one-line reading — "the neural model does not beat the energy baseline" —
is not supported, for three separate reasons: the two were scored on different
test sets (400 clips vs 10,791), with thresholds chosen under different rules
(10% FI ceiling vs cost minimisation), one in-sample and one out-of-sample (§5.1).
Three confounds in one comparison is not a result. What *is* comparable is the
ranking metric: **ROC-AUC 0.8900 for E1 against 0.6602 for E0 and 0.6433 for
E0b**, and that gap is large enough to survive the differences in set size. The
like-for-like comparison requires re-running the baselines on the full test cache
(§18.2, item 6).

### 9.6 Interpretation — a strong ranker at a badly-chosen operating point

The two headline numbers point in opposite directions and reconciling them is the
most useful thing in this section.

**ROC-AUC 0.8900 says the model ranks well.** Given a random turn-ended clip and
a random turn-held clip, it assigns the higher score to the correct one 89% of
the time. On a balanced test set that is a real, substantial signal — and it is
25 points above the best baseline.

**F1 0.2585 says the deployed decision is poor.** Precision 0.9696 at recall
0.1491: when it fires it is almost always right, but it fires on only 15% of
genuine endpoints. In product terms the agent almost never talks over you and
very often leaves you waiting. **85.09% of endpoints are missed on test.**

These are not in conflict, because they measure different things. ROC-AUC is
threshold-free and summarises the whole ranking; F1 is computed at one specific
threshold. A model can rank well and still score poorly at a badly placed
operating point, and that is what happened here.

**Why the operating point is where it is.** `pick_threshold` was called with
`max_false_interruption=None`, because no config sets it (§8.2). That selects the
cost-minimising branch: minimise `4·FPR + FNR` subject only to
`recall >= MIN_USEFUL_RECALL`, where `MIN_USEFUL_RECALL = 0.05`. E1's validation
recall is **0.0742** — just above that 0.05 floor. That is the signature of the
guard binding: the cost objective was pulling the threshold toward the degenerate
"never fire" corner described in §6.2, and the recall floor is what stopped it.
The corner that was found and guarded against on a ROC-AUC-0.66 baseline turns out
to bite a ROC-AUC-0.89 model too.

**How much headroom this implies.** PR-AUC is 0.8757 against a positive rate of
0.4960, so it is well above the 0.496 a random ranker would score. PR-AUC is the
recall-weighted mean of precision across the curve, so a value that high means
precision stays high across much of the recall range — it cannot be concentrated
in the first 15% of recall. **Operating points with a substantially better F1 than
0.2585 therefore exist on this same trained model, at a lower threshold.** That
is a claim about the shape of a curve we have measured, not a prediction about a
model we have not trained.

What is *not* claimed: a specific better F1. Naming one requires reading the
validation threshold sweep and re-scoring, which has not been done. Nor is it
claimed that re-selecting the threshold would make the streaming detector usable
— §11.2 measured that separately and the answer there is no.

**The consequence for what to do next.** The cheapest available improvement to
the reported result is not a new architecture, a bigger head, or more data. It is
setting `max_false_interruption: 0.10` so the threshold is chosen by maximising
recall inside the false-interruption budget that the report already claims
governs every row — then re-selecting on validation and re-scoring test once.
E1's test false-interruption rate is 0.0046, roughly **21× below** the 10%
ceiling, so there is a large amount of unspent interruption budget being left on
the table. This changes no weights and retrains nothing. It is listed first in
§18.2.

### 9.7 Limitations of this result

1. **It is a baseline.** A frozen ASR encoder with a 1,153-parameter linear probe
   and mean pooling is the *floor* of this architecture family, not a tuned
   system. E3/E4/E4b/E7 exist to test head capacity, temporal structure, and
   fine-tuning, and none has been run.
2. **The operating point is chosen by a rule with a known degenerate corner**
   (§9.6). The reported F1 is a property of that rule as much as of the model.
3. **Mean pooling discards ordering**, which is plausibly the main cue for
   endpointing. This is a hypothesis, tested by E4/E4b, not a diagnosis.
4. **Single seed, no confidence interval.** One run at seed 1234. Nothing here
   quantifies run-to-run variance, so small differences between E2 windows should
   not be over-read.
5. **Validation and test are not exchangeable.** F1 roughly doubles from val to
   test at the same threshold. They come from two different published datasets
   with different source-corpus mixes; which specific difference drives the gap
   has not been isolated, and no causal claim is made about it.
6. **Clip-level only.** E1 has not been evaluated as a stream, and §11.2 shows
   clip-level numbers do not predict streaming behaviour. E1's streaming
   behaviour is unmeasured and must not be inferred from the table above.
7. **No slice breakdown transcribed yet.** The per-language, per-`synthetic`, and
   per-filler slices for E1 are in the run artefacts and are not yet in this
   report. Given the 6.5× TTS-vs-human gap found on the verification run
   (§14.5), the aggregate figures above may be optimistic for human audio, and
   that should be checked before E1 is quoted anywhere.

---

## 10. E2 — window sweep

**Status: running in Colab. No results in this section. Nothing below is a
prediction.**

### 10.1 Hypothesis

> Temporal window length materially affects endpoint detection, because the
> evidence for an endpoint is localised in the final few hundred milliseconds
> while the conversational context needed to interpret that evidence varies in
> length.

Two mechanisms pull in opposite directions, which is why this is worth measuring
rather than reasoning about:

- **Too short** and the window cannot see the structure of the utterance it is
  judging. There is direct evidence for this concern: on the 1.0 s verification
  run, **14 of 23 missed endpoints were on utterances of 8 s or longer** (§14.1)
  — a model seeing the last 8% of a 13 s clip has no view of its shape.
- **Too long** and the ~300 ms that actually carries the decision is mean-pooled
  against seconds of mid-utterance audio. With a linear head over a mean-pooled
  encoder, a longer window mechanically dilutes the tail's contribution.

E1 sits at the long end (8.0 s) with recall 0.1491. Whether dilution is part of
why is what the sweep tests.

### 10.2 Controlled variables

E2 differs from E1 in **`window_seconds` only**. Verified mechanically by
constructing `TrainConfig` from each YAML and diffing every field against E1's:
the sole differences are `window_seconds` and `test_cache_dir` (§10.3).

Held constant across all five windows and E1: architecture (`head: linear`,
`pool: mean`), `freeze_encoder: true`, loss and `use_pos_weight`,
`val_fraction: 0.2`, `group_keys: [dataset]`, `normalise_mode: peak`,
`seed: 1234`, `epochs: 8`, `batch_size: 32`, `lr: 1.0e-3`, `fp_cost: 4.0`,
`early_stop_patience: 3`, and the same 40,000-clip train cache.

Windows swept: **0.5, 1.0, 1.5, 2.0, 4.0 s.** E1 supplies the **8.0 s** point, so
the sweep covers six window lengths without retraining E1.

### 10.3 Why no E2 row touches the test set

All five configs carry `test_cache_dir: null`. This is deliberate. The winner is
selected on validation; scoring five candidates on the held-out set before making
that decision would spend the test set's independence five times over and turn it
into a second validation set. One test evaluation runs **after** the winner is
chosen, on the winner alone:

```bash
python -m training.train --eval-only --checkpoint weights/<WINNER>-best.pt
```

`--eval-only` rebuilds the model from the checkpoint, reloads the threshold the
checkpoint already stores, reconstructs the validation split from the config
saved inside the checkpoint, re-runs the leakage assertion, and scores. It never
calls `pick_threshold`, so it cannot re-tune on test.

### 10.4 Selection rule, fixed in advance

Stated before the results exist, so it cannot be chosen to flatter them:

1. **Validation only.** Test plays no part in selecting the winner.
2. **Lowest validation cost** (`4·FI + missed`) wins — that is the objective
   training itself minimised for early stopping, so it is the consistent basis.
3. If **cost and F1 disagree**, both are reported and the disagreement is stated
   rather than resolved silently. `scripts/summarise_sweep.py` flags this case.
4. Because §9.6 established that this rule pins every trained run near the
   degeneracy floor, each window's threshold and recall are reported next to its
   cost. A sweep in which all six windows sit at recall ≈ 0.05–0.08 would be
   evidence that the *rule*, not the window, is the binding constraint — which is
   itself a reportable outcome.
5. Single seed per window, so a small cost difference is not treated as a real
   difference (§9.7 item 4).

### 10.5 Results

**Not yet available.** Filled from run artefacts by:

```bash
python scripts/run_matrix.py --only E2_w0p5 E2_w1p0 E2_w1p5 E2_w2p0 E2_w4p0
python scripts/summarise_sweep.py --prefix E2 --include E1 --out report/e2_window_sweep.md
```

| window | Val F1 | Val precision | Val recall | Val FI | Val missed | Val cost | threshold | train s | latency p50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 s | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 1.0 s | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 1.5 s | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 2.0 s | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 4.0 s | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 8.0 s *(= E1)* | 0.1363 | 0.8395 | 0.0742 | 0.0142 | 0.9258 | TBD | 0.909298 | TBD | TBD |

Winner: **TBD**. Winner's held-out test result: **not run** — it runs once, after
the winner is known.

E1's validation cost is `TBD` rather than computed here on purpose: `4·0.0142 +
0.9258` is arithmetic anyone can do, but the reported cost should come from the
same code path as every other row so a definitional difference cannot hide in it.

### 10.6 What each possible outcome would mean

Written in advance, because a hypothesis that cannot be falsified is not worth
testing.

| outcome | reading | what follows |
|---|---|---|
| A shorter window clearly wins on validation cost | tail dilution under mean pooling is real and material | adopt it; revisit `attn` pooling (E4b), which attacks the same mechanism directly |
| 8 s (E1) remains best | the long-utterance context matters more than dilution; §14.1's long-utterance misses were a 1 s artefact | window is not the bottleneck — move to §12 |
| All windows cluster within noise | window length is not the binding constraint at this operating point | stop sweeping windows; the constraint is elsewhere (§9.6 or §12) |
| All windows sit at recall ≈ the 0.05 floor | the *selection rule* is the binding constraint, not the window | fix the rule first (§18.1); the sweep is uninformative until then |

---
## 11. Streaming evaluation

Offline classification of pre-cut clips answers a question nobody asks in
production. In a live call, audio arrives in 10–20 ms chunks, there is no "clip",
and the detector must decide *while* the person is still talking.

```
in → rolling buffer → resample/normalise/mel → encoder+head → EMA → threshold
   + min-silence + hysteresis → USER_TURN_ENDED
```

### 11.1 What sits between a probability and an emitted event

Three mechanisms, each preventing a specific failure, each with a test that fails
if it is removed:

- **EMA smoothing** — a single window crossing the threshold is noise: a breath, a
  codec glitch, a low-energy moment mid-word. A sustained rise is a decision. EMA
  over a boxcar because it needs one float of state and weights the most recent
  window most heavily, which is the right prior for something that just happened.
- **Minimum silence (200 ms default)** — natural speech is full of sub-200 ms
  gaps: stop closures, in-breaths, the pause before a searched-for word.
  Requiring the probability to *stay* high separates "they stopped" from "they
  paused".
- **Hysteresis (enter 0.70 / exit 0.45)** — one threshold at 0.5 chatters when the
  signal hovers near it. A dead band means it takes real evidence to cancel a
  decision in progress. Same reason a thermostat has one. An inverted band is
  rejected at construction, not tuned around.

**Chunk size is decoupled from hop size.** A WebRTC track delivers 20 ms frames,
but running an 8 s encoder every 20 ms is waste. Chunks accumulate; the model runs
once per hop (160 ms default). *Measured:* 5 model calls per second of audio at a
200 ms hop, verified by test.

### 11.2 The most important finding in this report: clip scores do not transfer to streaming

**Why clip-level accuracy is not evidence about streaming behaviour.** A
clip-level positive label says *this clip ends at a turn boundary*. It does not
say that firing anywhere inside the clip is correct. A detector that fires during
an internal pause has interrupted the user; the fact that the clip eventually
ends does not redeem that fire. Clip-level scoring cannot see the difference,
because it only ever asks the model one question per clip, at the one moment the
annotator chose.

Measured, same `SMOKE` model, same threshold, two evaluation protocols:

| evaluation | F1 | false interruption |
|---|---|---|
| clip-level classification | 0.3009 | **0.0326** |
| streamed, 20 ms chunks, 160 ms hop | 0.0270 | **0.4598** |

A 14× worse interruption rate. This is not a bug in the streaming code — it is a
**distribution shift the training data does not cover**.

In training, every window's **right edge coincides with an annotated boundary**:
positive clips end at a real turn end, negative clips end at a real mid-utterance
point. Both are curated moments. In streaming, the right edge is wherever the hop
lands — which is almost never either of those. The model is asked about window
alignments it never saw, and it answers by firing on ordinary intra-utterance
pauses that no training example ever labelled.

**So quoting a clip-level F1 as evidence of streaming quality would be the single
most misleading thing this report could do.** A submission that only classified
clips would ship a detector with a 46% interruption rate while reporting 3%.

### 11.3 Three metrics that only exist because of this

The failure above is invisible to naive scoring. These make it visible:

**Early-fire scoring.** `src/streaming.py:EARLY_FIRE_TOLERANCE_MS = 1000` counts a
fire more than 1 s before the clip end as a false interruption **regardless of the
clip's label**. Without it, a detector that fires 7.8 s early is recorded as a
true positive because the clip eventually ends. That exact case occurred: streamed
E0 reported `ttd_p50 = −7780 ms`, a "time to decide" 7.8 seconds *before* the
boundary, which is what exposed the scoring bug.

**False-interruption rate as a first-class number**, not a side effect of a
confusion matrix. It is the error the user actually notices (§1), it is the error
clip-level scoring under-counts by 14×, and it is the one metric that must be
reported for a streamed detector even when F1 is flattering.

**Time to decide**, signed, in milliseconds from the true boundary to the emitted
event. A positive value is lag the user hears; a negative value is an
interruption. Reporting it signed is what makes an early fire legible as a
different failure from a slow one, rather than both appearing as "error".

### 11.4 Parameter tuning cannot fix it — measured

The obvious response is to tighten the streaming parameters: raise the entry
threshold, lengthen the minimum silence, smooth harder. Tested directly, on 120
clips:

| enter | min silence | EMA α | F1 | recall | false interrupt | early fires |
|---|---|---|---|---|---|---|
| 0.680 | 200 ms | 0.40 | 0.0270 | 0.0303 | **0.4598** | 36 |
| 0.900 | 400 ms | 0.30 | 0.0000 | **0.0000** | 0.0000 | 0 |
| 0.970 | 800 ms | 0.20 | 0.0000 | **0.0000** | 0.0000 | 0 |
| 0.995 | 1200 ms | 0.15 | 0.0000 | **0.0000** | 0.0000 | 0 |

**There is no usable operating point.** The detector either interrupts 46% of held
turns or never fires at all — it moves from one degenerate corner to the other
with nothing in between. That is not a tuning problem; a tunable detector would
trade recall for interruptions gradually across that range.

What the discontinuity implies: the smoothed probability spends almost all its
time either near 1 on intra-utterance pauses or near 0 elsewhere, so any threshold
above the pause-triggered peaks also excludes the genuine endpoints. The model is
confidently wrong rather than uncertain, and no amount of smoothing recovers a
signal that is not there.

So of three candidate fixes, the measurement **rules out the cheapest**:

1. ~~Tighten the streaming parameters~~ — **tested and ruled out** by the table
   above.
2. **Train on randomly-offset crops** so arbitrary window alignments appear in
   training. This is the actual fix, and it is §12.
3. **Calibrate the threshold on streamed replay** rather than on clip
   classification. Necessary regardless, but on this evidence not sufficient
   alone.

This finding is also why `src/streaming.py` exists as its own evaluated component
rather than a thin wrapper assumed to inherit the model's accuracy. Building it
and measuring it is what surfaced the gap.

### 11.5 What has and has not been streamed

| model | streamed? |
|---|---|
| E0 energy baseline | **yes** — §6.3 |
| SMOKE verification run | **yes** — §11.2, §11.4 |
| **E1** | **no — not yet run** |
| E2 winner | no — pending the sweep |

E1's streaming behaviour is unmeasured. Given §11.2 it must be measured directly
and must not be inferred from E1's clip-level table (§9.5).

---

## 12. Random-offset training

**Status: designed and scoped. Not implemented, not run.** This section exists to
state the intended fix precisely, because §11.2 identified the cause and a report
that diagnoses without naming the remedy is incomplete.

### 12.1 The change

Training currently takes each clip's final `window_seconds` of audio, so the
window's right edge always lands on the annotated boundary. The change is to
sample a random offset per example and **re-derive the label from where the crop
actually ends**:

- If the crop's right edge lands at (or within a small tolerance of) the annotated
  turn end, the example is positive.
- If it lands anywhere earlier — mid-utterance, mid-pause, mid-word — it is
  negative, *even if the source clip's label is positive*.

That second rule is the whole point. It is what teaches the model that an internal
pause is not an endpoint, which is exactly the confusion §11.2 measured. It is
also the reason this is safe while "cropping from the end" is forbidden in §8.4:
the illegal version crops and keeps the old label, this version crops and
recomputes it.

It is a change to `src/dataset.py` — offset sampling and label derivation — not to
the model, the loss, or the streaming code.

### 12.2 What it would need to demonstrate

Fixed in advance so the result is falsifiable:

1. **Streamed false-interruption rate falls substantially below 0.4598** at a
   threshold with non-zero recall. This is the primary criterion; clip-level
   metrics are secondary here.
2. **A usable middle exists** — the §11.4 grid, re-run, must show a gradual trade
   between recall and interruptions instead of two degenerate corners. A
   detector that is tunable is the actual deliverable.
3. **Clip-level performance does not collapse.** Random offsets make the training
   task harder and shift the positive rate, so some clip-level regression is
   expected and acceptable. A large one would mean the offset distribution or the
   tolerance is wrong.

If (1) and (2) fail, that is reported as a failed fix rather than dropped. A
documented negative result is evidence of method; a quietly abandoned experiment
is not.

### 12.3 Why it is not started yet

Sequencing, not doubt. E2 is running and shares the same train cache and GPU; and
if E2 shows window length materially changes behaviour, random-offset training
should be built on the winning window rather than on 8 s. Starting both at once
would confound the two changes.

---

## 13. Hinglish and Indian-speech evaluation

### 13.1 What was built

`data/hinglish/phrases.py` — 44 hand-authored, hand-labelled phrases across 7
categories, rendered by Sarvam Bulbul v3 across multiple speakers. **202 clips
generated and verified** (16 kHz, RMS −12 to −13 dBFS, conversational level).

| category | what it tests |
|---|---|
| `end_filler` | trails on "matlab", "umm", "actually", "ki" — must **not** fire |
| `mid_hesitation` | filler mid-sentence, self-corrected numbers, digit strings |
| `code_switch` | switches language and stops mid-clause |
| `complete` / `complete_en` / `complete_dev` | genuinely finished, Hinglish / Indian English / Devanagari |
| `*_pause{250,400,700}` | complete utterances with real silence spliced mid-utterance |

Pause lengths of 250/400/700 ms bracket the 300–350 ms region where a thinking
pause and a turn-final pause become genuinely hard to separate. The spliced gap
uses room tone at a realistic level, not digital zero — a perfectly silent gap is
a giveaway no real recording has.

Multiple speakers per phrase is not decoration: with one voice a model could score
well by memorising that voice. *Measured* per-speaker spread for E0 was modest
(F1 0.247 vs 0.209), so the set measures the phenomenon more than the voice.

### 13.2 Measured — E0 on the Hinglish set

```
python scripts/eval_hinglish.py --baseline
```

Overall: **ROC-AUC 0.4757** — *worse than chance*. Per category:

| category | n | positives | fired | false interrupt |
|---|---|---|---|---|
| `end_filler` | 20 | 0 | 2 | 0.1000 |
| `mid_hesitation` | 16 | 0 | 1 | 0.0625 |
| `code_switch` | 10 | 0 | 0 | 0.0000 |
| `complete` | 96 | 96 | 16 | — |
| `complete_dev` | 24 | 24 | 0 | — |

### 13.3 A correction to the usual TTS caveat

"TTS is an upper bound" is the lazy version of the caveat, and it is **not always
true**. The direction of the synthetic bias depends on the detector:

- For a **prosody/spectral model**, TTS is *easier* — clean, evenly paced, no
  disfluent timing. Its score is optimistic.
- For a **silence-timer baseline**, TTS is *harder* — Bulbul emits almost no
  trailing silence after a completed utterance, so the one cue the baseline
  depends on is largely absent. That is why E0 lands below chance here, and it is
  a property of the audio rather than a fresh insight about the baseline.

The honest reading: **this set measures whether a detector uses linguistic
completion cues rather than duration of silence.** A detector that scores well
here is reading the former.

Every manifest row carries `synthetic: true`, so no downstream script can average
these into a headline figure.

### 13.4 Measured — the model on the Hinglish set

```
python scripts/eval_hinglish.py --checkpoint weights/SMOKE-best.pt
```

| detector | ROC-AUC | fired on deliberately-incomplete utterances |
|---|---|---|
| E0 energy baseline | 0.4757 (below chance) | **3 / 50** |
| SMOKE model | 0.6000 | **0 / 50** |

**Zero false interruptions across all 50 incomplete utterances** — every
`end_filler`, `mid_hesitation`, `code_switch` and Devanagari filler clip was
correctly held. That is the behaviour the stress set was built to check.

The trade is visible too: recall on complete utterances is only 0.06–0.25, so the
model is *conservative* — it rarely interrupts and often waits. Given the two
error costs that is the right corner to fail into, but it is a failure to fix with
more training rather than a result to celebrate. Note also that this is the same
conservatism §9.6 diagnoses at full scale, so "zero false interruptions" and "85%
of endpoints missed" are two views of one operating point, not two independent
findings.

Per-speaker F1 differed (shubh 0.292, ritu 0.124), which is a caution about the
set: with only two voices, some of the variance is voice rather than phenomenon.

### 13.5 Held-out Indic audio in the main corpus

Separate from the hand-built set, the main caches contain real `hin` / `ben` /
`mar` audio (§3.1: 1,295 / 1,000 / 774 test clips). Per-language slices are
produced by `src/evaluation.py:slice_report` for every run. **E1's per-language
breakdown has not been transcribed into this report yet** (§9.7 item 7); it is in
the run artefacts.

### 13.6 What is still missing

Real recorded Hinglish. TTS covers vocabulary, code-switching, and pause
placement; it does not reproduce genuine disfluent timing, overlapping speech, or
real room acoustics. Also missing: E1 evaluated on this set — only `SMOKE` has
been. This is a stated gap, not a solved problem.

---
## 14. Error analysis

**Every number in this section is from the `SMOKE` verification run** (1.0 s
window, 3 epochs, 600 train rows, 400 test clips, threshold 0.681). It is too
small to be a reportable result, and it is included because it is what has been
analysed. **E1's error analysis has not been run** — the counts below will be
regenerated against E1 and the E2 winner, and the categories are expected to
survive even where the counts do not.

### How failures are sampled

`scripts/error_analysis.py` samples failures **weighted 60/40 toward false
interruptions** — at a 10% ceiling a uniform sample would be almost all missed
endpoints and would teach nothing — and takes the *most confident* failures
first, because a confidently wrong prediction is the model's fault while a
borderline one is the threshold's.

Modes are named from the corpus's own annotations rather than a hand-built
taxonomy, which is §3.3 paying off:

| observed | named mode |
|---|---|
| FP with `endfiller=True` | cut off a trailing filler |
| FP with `midfiller=True` | cut off a mid-utterance hesitation |
| FP, duration < 1.5 s | fired on a short backchannel |
| FN with `midfiller=True` | missed an endpoint after an internal hesitation |
| FN, duration ≥ 8 s | missed an endpoint on a long utterance |

```
python scripts/error_analysis.py --checkpoint weights/SMOKE-best.pt --mine-hard-negatives
```

7 false interruptions, 23 missed endpoints at threshold 0.681.

### 14.1 Long utterances

| missed endpoints | n |
|---|---|
| on a **long utterance** (≥ 8 s) | **14** |
| plain endpoint | 5 |
| after an internal hesitation | 4 |

**14 of 23 misses are long utterances.** The `SMOKE` model has a 1.0 s window and
the corpus median is 7.16 s, so on a 13 s clip it sees the last 8% of the audio
and has no view of the utterance's structure. That is a direct mechanical
explanation and it is what E2 (§10) tests. It also argues the reference's 8 s
default is a considered choice rather than a conservative one.

**Open question this raises for E1.** E1 *has* the 8 s window and still misses
85.09% of endpoints on test. So either the long-utterance mode is not the binding
constraint at 8 s, or it is and something else dominates. §9.6 argues the
operating point dominates. E1's own error analysis will settle which, and until
it is run this is explicitly an open question rather than a conclusion.

### 14.2 Internal pauses

The failure mode with the largest measured consequence, and the one clip-level
scoring is nearly blind to.

Three of seven false interruptions were mid-utterance with **no filler
annotation** — a plain internal pause, not a hesitation the corpus had flagged.
That is a small count on a small run, but it is the same mechanism that produces
the streaming collapse in §11.2 at full force: streamed, the same model's
false-interruption rate is **0.4598**, and §11.4 shows the fires are concentrated
on intra-utterance pauses.

The asymmetry is worth stating plainly: **clip-level evaluation sees 3 of these;
streaming evaluation sees the same failure at 14× the rate.** Internal pauses are
where the clip-level and streaming pictures diverge most, and they are the target
of §12.

### 14.3 Fillers

| slice | n (negatives) | false interrupt |
|---|---|---|
| `endfiller=True` | 117 | **0.0085** |
| `midfiller=True` | 98 | 0.0408 |

The model almost never cuts off a *trailing* filler (0.85%) — the most
user-visible failure — but is **5× worse on mid-utterance hesitation**. That
ordering makes sense against §3.3: `endfiller=True` has a positive rate of exactly
0.000, so the pattern is perfectly consistent in training, whereas `midfiller=True`
is 51.5% positive and therefore genuinely ambiguous. The model has learned the
clean rule and not the ambiguous one.

Corroborated independently on the hand-built Hinglish set: **0 fires across 50
deliberately-incomplete utterances**, including every `end_filler` clip (§13.4).

### 14.4 Hinglish and code-switching

Measured in §13.4. Summary: on the hand-built set the model held every one of 50
incomplete utterances, including code-switched and Devanagari fillers, while
recall on complete utterances was 0.06–0.25 — conservative in both directions.
Per-category counts are in §13.2.

Two caveats belong in the error analysis rather than only in §13: the set is TTS,
so it does not exercise genuine disfluent timing; and only `SMOKE` has been
scored on it. The real held-out Indic audio in the main test cache (1,295 `hin` /
1,000 `ben` / 774 `mar` clips) is sliced automatically per run, and **E1's
per-language numbers are not yet transcribed**.

### 14.5 Synthetic vs human audio

The largest measured gap in the project, and the one most likely to mislead a
reader of the aggregate.

| slice | n | F1 | recall | false interrupt |
|---|---|---|---|---|
| `synthetic=True` (TTS) | 334 | **0.3492** | 0.2185 | 0.0273 |
| `synthetic=False` (human) | 66 | **0.0541** | 0.0294 | 0.0625 |

**A 6.5× F1 gap and a 2.3× worse interruption rate on human audio.** On 66 human
clips the model is barely better than chance.

The straightforward reading is that a 3-epoch model trained on a corpus that is
82% synthetic has substantially learned **TTS artefacts** rather than the prosody
of finished speech. Two things restrain that reading: 66 clips is a small sample,
and a 3-epoch run on 600 rows is not a converged model. So the gap is real and
its *cause* is a hypothesis.

Either way the reporting consequence is the same, and it applies to E1: **the
aggregate is not the number to trust.** "Beats both baselines" was true of the
full 400-clip set and *not* true of the human subset. E1's `synthetic=False`
slice must be read before E1's aggregate is quoted anywhere (§9.7 item 7).

### 14.6 Speaker variation

There is no speaker ID in the corpus (§3.2), so speaker variation cannot be
measured on the main data at all. What exists is indirect, from the hand-built
set where the speaker *is* known:

| detector | per-speaker spread |
|---|---|
| E0 baseline | F1 0.247 vs 0.209 |
| SMOKE model | F1 0.292 (shubh) vs 0.124 (ritu) |

The model's spread is wider than the baseline's — 2.4× versus 1.2×. With two
voices that is a caution, not a finding: it could be voice sensitivity or it could
be which phrases each voice happened to render. More speakers would separate
those. **This is the weakest-evidenced category here** and is listed because its
absence is itself a limitation, not because the numbers support a conclusion.

### 14.7 Streaming and window alignment

The dominant failure mode, measured in §11.2 and §11.4: false-interruption rate
0.0326 clip-level against **0.4598** streamed, for the same model at the same
threshold, with no usable operating point in between. Cause identified as
training-window alignment; fix scoped in §12.

Listed here as an error category rather than only as a streaming result, because
it is the failure mode with the largest gap between measured and *apparent*
severity — and therefore the one most likely to be missed by a submission that
evaluates clips only.

### 14.8 Hard negatives — mined, not yet used

`--mine-hard-negatives` writes the indices of the most confidently-wrong clips to
`artifacts/runs/error_analysis/hard_negative_indices.npy`. 158 were mined,
concentrated in `chirp3_1` (68) and `chirp3_2` (49) — telling you which source the
next run is short of.

> **Not implemented: the retrain.** Mining produces the indices; nothing consumes
> them. There is no config that oversamples them and no code path in
> `src/dataset.py` or `training/train.py` that reads the file. The loop *symptom →
> cause → fix → re-measure* stops at "cause" for this category, and it is listed
> as future work rather than claimed as done. When it is implemented, the outcome
> gets reported either way.

`notebooks/error_analysis.ipynb` plays the failures inline, and degrades cleanly
with instructions when no checkpoint exists.

---

## 15. Model optimisation

`scripts/optimize_model.py` produces four artefacts and reports the delta between
each **separately** — bundling them would make it impossible to say which paid
off:

1. float32 torch (the checkpoint)
2. int8 torch (dynamic quantisation of Linear/GRU)
3. float32 ONNX (Python out of the loop, graph fusion)
4. int8 ONNX (the deployment artefact)

Dynamic rather than static quantisation: no calibration set needed, and it
quantises exactly the weights that dominate an 8 M-parameter encoder. Attention
matmuls stay float, which is why the speedup is real but not 4×.

The script also measures the **mel front-end in isolation**, because at an 8 s
window it is a non-trivial share of a 12 ms budget and no amount of quantisation
touches it. Optimising the wrong half is a common waste.

Benchmarking is batch size 1 (a live call has one stream), single and multi
threaded, 15 warmup iterations discarded — the first calls pay for lazy kernel
selection and thread-pool spin-up, and including them puts a 200 ms outlier in a
12 ms distribution.

### 15.1 Measured — all four artefacts, on the `SMOKE` run

```
python scripts/optimize_model.py --checkpoint weights/SMOKE-best.pt --cache data/cache/test
```

| artefact | size | F1 | Δ F1 | CPU p50 (1 thread) | p95 |
|---|---|---|---|---|---|
| float32 torch | 30.64 MB | 0.3009 | — | 17.8 ms | 19.6 ms |
| int8 torch (dynamic) | 9.42 MB | 0.1015 | **−0.1994** | 11.8 ms | 12.9 ms |
| float32 ONNX | 30.94 MB | 0.3009 | +0.0000 | 13.4 ms | 13.8 ms |
| **int8 ONNX** | **8.02 MB** | 0.2648 | −0.0360 | 18.5 ms | 21.5 ms |

mel front-end alone: **0.48 ms p50 — 3% of total.** So the encoder dominates and
optimising the front end would be optimising the wrong half.

**Size validates against the reference.** Published `smart-turn-v3` is 8 MB int8
ONNX / 32 MB unquantised. We measured **8.02 MB** and **30.94 MB** at the same
parameter count. That agreement is the main reason to trust the rest of the
measurement — and it is what caught an earlier bug where the exporter had left
the weights in a sibling `.onnx.data` file and the size read as 0.36 MB.

Four results worth stating plainly rather than smoothing over:

1. **ONNX fp32 export is accuracy-exact** (Δ = 0.0000). Correct — an export that
   moved the numbers would mean the graph did not match the module.
2. **torch dynamic int8 is far worse than ONNX dynamic int8** (−0.199 vs −0.036)
   at a *larger* size (9.42 vs 8.02 MB). onnxruntime quantises per-channel where
   torch's dynamic path is per-tensor, and on a transformer encoder that
   difference is large. Had only torch quantisation been tried, the conclusion
   "int8 is unusable here" would have been confidently wrong.
3. **int8 ONNX is *slower* than fp32 ONNX** on this machine — 18.5 ms vs 13.4 ms.
   Dynamic quantisation inserts quantise/dequantise nodes around each matmul, and
   without VNNI-class int8 instructions the arithmetic saving does not pay for
   them. On this CPU int8 buys a **3.9× size reduction at a latency cost**, which
   is the opposite of the usual assumption and is why the benchmark exists rather
   than being taken on faith.
4. **The −0.036 F1 cost of int8 should be re-measured on a converged model.** A
   model at F1 0.30 has its scores bunched near the threshold, so small
   perturbations flip many predictions. Quoting this as the final quantisation
   cost would overstate it.

Peak RSS is 0.7–1.2 GB across backends, dominated by the runtime rather than the
8 MB of weights. On a memory-constrained deployment that, not the checkpoint
size, is the binding constraint.

### 15.2 E1 and the E2 winner

| run | size | F1 delta under int8 | CPU p50 | CPU p95 |
|---|---|---|---|---|
| E1 (8 s window) | TBD | TBD | TBD | TBD |
| E2 winner | TBD | TBD | TBD | TBD |

**The E1 benchmark is running in Colab; these cells are not estimates and are not
copied from the `SMOKE` row.** Two of them will differ materially: the `SMOKE`
numbers are for a 1 s window, and E1's window is 8× longer, so encoder cost and
therefore latency will not carry over. Size and parameter count should be
essentially unchanged, since window length changes the number of encoder
positions, not the weights — but that is a prediction, and the table stays `TBD`
until it is measured.

---

## 16. Real-time deployment

### 16.1 What exists

| piece | state |
|---|---|
| Streaming detector (`src/streaming.py`) | built, 20 tests, measured end to end |
| int8 ONNX artefact | built and committed (`weights/SMOKE-int8.onnx`, 8.02 MB) |
| Gradio demo (`demo/app.py`) | built; **verified to launch and serve HTTP 200** on Gradio 6.25.0 |
| HF Space entry point (`app.py`) | built; front matter in `README.md`; verified to launch |
| Docker (`Dockerfile`) | built, not yet run in CI |

The demo resolves weights in priority order — int8 ONNX, then a torch checkpoint,
then the E0 energy baseline — and states in the UI which backend it is running.
A fresh clone with no trained weights therefore boots and explains itself rather
than showing a stack trace.

Both entry points are verified by actually starting the server and fetching the
page, not by constructing the Blocks object. That distinction is not pedantic:
an earlier `show_api=False` argument, removed in Gradio 6, raised `TypeError`
only at `launch()` — so every check that stopped at "the UI object builds" passed
while the demo was broken.

### 16.2 Latency budget

*Measured* on the demo path, end to end — chunk arrival to event emission, not
model forward time:

| path | p50 | p95 |
|---|---|---|
| energy backend, per hop | 1.2 ms | 2.1 ms |
| int8 ONNX (`SMOKE`, 1 s window), per emitted decision | 19.1 ms | 20.6 ms |

Real-time factor 0.018 for the energy path — 56× faster than real time. At a
160 ms hop the model runs ~6 times per second of audio, so a ~19 ms decision
consumes roughly 12% of one core. **The equivalent figure for E1's 8 s window is
`TBD`** (§15.2) and will be larger.

The architectural point that makes this work is in §11.1: chunk size is decoupled
from hop size, so a 20 ms WebRTC frame does not trigger a 20 ms encoder pass.

### 16.3 What would still be needed for production

Stated because "deployable" is a claim that should come with its own caveats:

- **A streaming-calibrated threshold.** The current threshold is selected on clip
  classification, and §11.2 shows that does not transfer.
- **§12 completed.** As trained, the streaming detector interrupts 46% of held
  turns (§11.4). It is a working system that is not yet an accurate one.
- **Barge-in handling and agent-state awareness** — the detector answers "has the
  user stopped", not "should the agent speak now", which also depends on whether
  the agent is mid-sentence.
- **A load and soak test.** Peak RSS of 0.7–1.2 GB per process (§15.1) sets the
  per-instance concurrency ceiling, and it has not been measured under
  concurrency.

---
## 17. Limitations

Stated plainly, because a declared limitation is worth more than a suspiciously
clean result.

1. **No speaker-aware split is possible.** Grouped by source corpus instead. One
   speaker contributing to two corpora would still span splits and nothing in the
   metadata can detect it (§3.2).
2. **The reported operating point is chosen by a rule with a known degenerate
   corner**, and no trained config sets the false-interruption ceiling the report
   otherwise describes as shared (§8.2, §9.6). E1's headline F1 is a property of
   that rule as much as of the model. This is the most consequential defect
   currently known.
3. **Clip-level scores do not predict streaming behaviour** — 0.033 → 0.460
   false-interruption rate for the same model and threshold (§11.2). Until §12 is
   done, any streaming number must be measured directly and never inferred. The
   1 s early-fire tolerance bounds, but does not remove, this weakness.
4. **E1 has been evaluated clip-level only.** Not streamed, not on the Hinglish
   set, and its slice breakdown is not yet transcribed. Most of the downstream
   analysis in §11, §13, §14 and §15 is on the much smaller `SMOKE` run and is
   labelled as such throughout.
5. **The baselines are scored on a different, smaller test set with an in-sample
   threshold** (§5.1, §6.1). E0-vs-E1 is not currently a like-for-like comparison.
6. **82% of the corpus is synthetic**, and the measured TTS-vs-human F1 gap is
   6.5× on the verification run (§14.5). The aggregate figure is optimistic for
   human audio.
7. **The Hinglish set is TTS.** It covers vocabulary, code-switching and pause
   placement, not genuine disfluent timing, overlapping speech, or real room
   acoustics.
8. **Speaker robustness is essentially unmeasured** — no speaker IDs in the
   corpus, and only two voices in the hand-built set (§14.6).
9. **Single seed per experiment, no confidence intervals.** Run-to-run variance is
   unquantified, so small differences between E2 windows must not be over-read.
10. **Noise and reverb augmentation are synthetic** — white noise and an
    exponential-decay impulse, not a licensed noise corpus or measured IRs. And
    no reported run used augmentation at all.
11. **`fp_cost = 4.0` and the 10% ceiling are judgement calls**, stated as named
    constants so they can be argued with rather than assumed.
12. **Test coverage is deliberately uneven.** 81 tests, concentrated where a
    silent error would corrupt a result: preprocessing 24, streaming 20, metrics
    18, augmentation 10, splits 9. Six modules have **no** direct unit tests —
    `dataset`, `model`, `evaluation`, `baselines`, `inference`, `optimize` — and
    are exercised only through the scripts that call them. That is a real gap.

---

## 18. Final recommendation

### 18.1 The single highest-value next action

**Re-select E1's operating point under the false-interruption budget, on
validation, and re-score test once.** Set `max_false_interruption: 0.10` in the
config, re-run threshold selection on validation, apply the new threshold to test.

Why this first, ahead of any modelling change:

- It retrains nothing and changes no weights. It is minutes of CPU-to-GPU-free
  work against an existing checkpoint.
- The evidence that it will help is already measured, not assumed: test ROC-AUC
  0.8900 and PR-AUC 0.8757 against a 0.4960 positive rate say the ranking supports
  much better operating points (§9.6), and the current false-interruption rate of
  0.0046 is ~21× below the ceiling the report claims to enforce.
- It removes a genuine inconsistency: baselines are currently selected under the
  ceiling and trained models under cost minimisation, so the experiment table
  compares rows picked by two different rules (§8.2).
- Until it is done, every subsequent experiment — including E2 — is being compared
  at an operating point pinned near the degeneracy floor, which compresses the
  differences the experiments are meant to reveal.

The honest framing when reporting it: this improves *how the model is used*, not
how good the model is. ROC-AUC will not move. That is the point — it is why the
change is cheap and why the current F1 understates the system.

### 18.2 Then, in order

| # | action | depends on | why |
|---|---|---|---|
| 1 | Re-select E1's threshold under the FI budget (§18.1) | nothing | largest measured gain per unit of compute |
| 2 | Finish E2, pick the winner on validation, one test eval | sweep running | answers whether window is the constraint (§10) |
| 3 | Implement random-offset training (§12) on the winning window | 2 | the only fix with measured evidence behind it for the streaming collapse |
| 4 | Re-run the §11.4 streaming grid on the result | 3 | verifies (or falsifies) the fix against the number it targets |
| 5 | Run E1's error analysis and slice report | nothing | §14 is currently all `SMOKE`; the categories need E1-scale counts |
| 6 | Re-run baselines on the full 10,791-clip test cache | nothing | makes E0-vs-E1 a real comparison (§9.5) |
| 7 | Benchmark E1/winner for size and latency | benchmark running | the last empty column in the deployment story (§15.2) |
| 8 | E3/E4/E4b (head capacity, temporal structure, pooling) | 2 | only worth running once the operating point and the window are settled |
| 9 | Hard-negative retrain (§14.8) | 5 | closes the one analysis loop that currently stops at diagnosis |

Items 1, 5 and 6 need no GPU and no new code. Items 3 and 9 are the two places
where this project's analysis has identified a fix and not yet applied it, and
both are scoped rather than open-ended.

### 18.3 What this project currently supports as a claim

> "I built an audio-only turn detector end to end, established baselines,
> trained it at scale, measured it honestly — and the measurement found that the
> clip-level result does not describe the streaming behaviour, which is the
> failure most submissions in this space would not have caught."

Against that, item by item:

| claim | status |
|---|---|
| understood the problem | **done** — data questions answered before modelling; three planning assumptions corrected (Appendix C) |
| designed the data pipeline | **done** — grouped splits, leakage asserted in code and tested, 24 preprocessing tests |
| established baselines | **done** — E0 and E0b, clip-level *and* streamed. Caveat: 400-clip set, in-sample threshold (§5.1) |
| trained at scale | **done** — E1 on 40,000 clips, one held-out test evaluation, threshold from validation |
| ran controlled experiments | **partly** — 12 configs; E1 complete, E2 running, E3–E7 not started |
| analysed failures | **partly** — seven categories, counts from the verification run; E1's own error analysis not yet run |
| optimised inference | **done for `SMOKE`** — four artefacts, isolated deltas, 8.02 MB int8 ONNX matching the published reference. E1 benchmark pending |
| deployable real-time system | **built and verified to run** — streaming detector, demo, Space entry point. **And measured to be inadequate as trained** (§11.2), with the cause identified and the fix scoped |

The last row is the one that matters most, and overstating it would be the easiest
mistake to make. The accurate version is not "I built a working real-time
detector". It is: **I built the real-time detector, measured it properly, and
found that clip-level training does not produce a usable streaming detector — and
that finding is only available to someone who built the streaming path and scored
it separately.**

The remaining work is mostly compute and sequencing rather than unsolved problems.
The two genuine technical gaps — the threshold-selection rule (§9.6) and
random-offset training (§12) — are both diagnosed, both scoped to specific files,
and neither is an open research question.

---

## Appendix A — reproducibility

- Every run is driven by a YAML config; an unknown key raises rather than being
  ignored, because a silently typo'd key means the experiment you think you ran is
  not the one that ran.
- Seeded across `random`, `numpy`, and `torch`.
  `torch.use_deterministic_algorithms` is deliberately **not** set: on a free T4
  the cudnn fallback costs more time than the run-to-run variance it removes.
  Seeded-but-not-bitwise-deterministic is the honest description.
- **Checkpoint every epoch, unconditionally**, and resume automatically. A free
  Colab session disconnects without warning. Verified — a dry run resumed from
  epoch 1 on re-invocation.
- Each run writes `config`, `history`, `split_report`, `evaluation`, per-slice
  markdown, and raw probability arrays to `artifacts/runs/<run_id>/`.
- `requirements.lock.txt` — 186 pinned packages, frozen after install.
- 81 tests pass (`pytest`). Distribution and the coverage gap are in §17 item 12.
- `report/experiments.md` is generated from `artifacts/experiments.csv` and never
  hand-edited. `report/final_experiment_table.md` is the one hand-maintained
  table, because it holds results whose artefacts are still in Colab; every cell
  there is either transcribed from an executed run or marked `TBD`.

## Appendix B — commands

```bash
# data
python scripts/prepare_data.py --info
python scripts/prepare_data.py --split train --languages eng hin ben mar --max-rows 40000
python scripts/prepare_data.py --split test  --languages eng hin ben mar
jupyter notebook notebooks/eda.ipynb

# baselines, before any neural work
python scripts/run_baselines.py --cache data/cache/test --slices

# train
python -m training.train --config configs/e1_frozen_linear.yaml --dry-run
python -m training.train --config configs/e1_frozen_linear.yaml

# the window sweep (validation only), then one test eval on the winner
python scripts/run_matrix.py --only E2_w0p5 E2_w1p0 E2_w1p5 E2_w2p0 E2_w4p0
python scripts/summarise_sweep.py --prefix E2 --include E1 --out report/e2_window_sweep.md
python -m training.train --eval-only --checkpoint weights/<WINNER>-best.pt

# analysis
python scripts/eval_hinglish.py --checkpoint weights/E1-best.pt
python scripts/error_analysis.py --checkpoint weights/E1-best.pt --mine-hard-negatives
python scripts/optimize_model.py --checkpoint weights/E1-best.pt
python scripts/stream_eval.py --checkpoint weights/E1-best.pt --sweep
python scripts/stream_eval.py --baseline    # the like-for-like comparison

# demo
python demo/app.py     # CLI
python app.py          # exactly what the HF Space runs
```

## Appendix C — three planning assumptions that were wrong

Recorded because checking beat assuming, and each one changed the work.

| assumption | reality | what changed |
|---|---|---|
| "Assume no Indian-language audio" | `hin`, `ben`, `mar` all present | a real held-out Indic slice exists; the hand-built set narrowed to code-switching and domain vocabulary, which the corpus genuinely lacks |
| "Severe class imbalance is a risk" | 1.07× — essentially balanced | class weighting became conditional rather than assumed |
| "Whisper Tiny + linear head is our design choice" | that *is* the published v3 architecture | gave a public reference point (8 MB int8, ~10–12 ms CPU) to check our measurements against — which is what validated the size benchmark in §15.1 |

And five things nothing anticipated, each found by building and measuring:

- **`endfiller` / `midfiller` annotations exist.** The error taxonomy was already
  labelled in the corpus, and `endfiller=True` has a positive rate of exactly
  0.000 across 302 clips.
- **The cost function has a degenerate corner.** `4·FPR + FNR` is minimised by
  never firing for any weak detector. Found on the baselines — and §9.6 shows it
  reached E1 anyway, through a default that no config overrides.
- **Clip-level scores do not transfer to streaming** — 0.033 → 0.460
  false-interruption rate for the same model at the same threshold (§11.2). The
  most consequential finding here, and one that only appears if you build the
  streaming path and measure it separately.
- **Streaming scoring needs an early-fire rule.** Without it, a detector firing
  7.8 s before a boundary is recorded as a true positive because the clip
  eventually ends. A `time_to_decide` of **−7780 ms** was the tell.
- **torch and onnxruntime int8 are not equivalent.** −0.199 F1 versus −0.036 at a
  *larger* size, because one quantises per-tensor and the other per-channel. And
  int8 was *slower* than fp32 on this CPU. Testing only one path would have
  produced a confidently wrong conclusion in either direction.
