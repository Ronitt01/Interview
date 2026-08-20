# Smart Turn Detection — technical report

Audio-only endpointing for a voice agent: given a waveform, decide whether the
speaker has finished their turn. No transcript, no LLM.

**Status of the numbers in this document.** Every figure marked *measured* was
produced by a command in this repo and is reproducible with it. Figures in the
experiment matrix marked `—` are the rows a full training run fills in; the
pipeline that produces them is built, tested, and has been exercised end to end
on a 1,200-clip subset. Nothing here is a projection presented as a result.

---

## 1. The problem, and why the obvious approach fails

A voice agent has to know when to start talking. The naive rule — wait for N
milliseconds of silence — fails because **pause duration does not separate the
two cases**. A speaker searching for a word pauses for 300–600 ms. A speaker who
has finished pauses for 300–600 ms. There is no threshold that splits them,
which is why this task needs a model that reads *linguistic completion* rather
than *duration of silence*.

Two errors, with very different costs:

| error | what the user experiences | cost |
|---|---|---|
| **False interruption** — fires while the user is mid-thought | the agent talks over them; they stop, get confused, repeat themselves | high — this is the failure that makes an agent feel broken |
| **Missed endpoint** — waits after the user finished | the agent feels a beat slow | low — annoying, forgivable |

The whole report is organised around that asymmetry. `src/metrics.py` weights one
false interruption as worth four missed endpoints (`DEFAULT_FP_COST = 4.0`), and
every reported operating point is held to a **shared 10% false-interruption
ceiling** (`DEFAULT_FI_BUDGET`) so rows are comparable.

### Reference point

The published `pipecat-ai/smart-turn-v3` is Whisper Tiny encoder + a shallow
linear classifier, ~8M parameters, 8 MB int8 ONNX / 32 MB unquantised, ~10–12 ms
CPU. **This project uses the same architecture on purpose**, so our numbers sit
next to a public baseline instead of floating free. A wildly different figure
means a bug in our measurement, not a breakthrough.

---

## 2. Data preparation

### 2.1 What the corpus actually is

| | rows | size |
|---|---|---|
| `pipecat-ai/smart-turn-data-v3.2-train` | 270,946 | 41.4 GB |
| `pipecat-ai/smart-turn-data-v3.1-test` | 31,473 | 4.3 GB |

Columns: `audio`, `audioduration`, `id`, `language`, `endpoint_bool` (the label),
`midfiller`, `endfiller`, `synthetic`, `dataset`, `spoken_text`.

**A published train/test split already exists**, so we use it rather than
inventing one. Validation is carved out of train.

### 2.2 Gate 1 — the four questions answered before any training

All *measured* on a 1,200-clip sample of v3.2-train (`notebooks/eda.ipynb`,
executed with outputs committed).

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

**Class balance: 620 / 580, ratio 1.07×.** Essentially balanced. The plan
anticipated severe imbalance; it did not materialise. `pos_weight` is applied
only when the ratio exceeds 1.2, and F1 plus the confusion matrix lead the
reporting regardless.

**Is there Indic audio? Yes — and this changed the plan.** 23 languages, with
`hin` 62, `ben` 34, `mar` 33 in the sample (~10.7% Indic). The plan assumed none.
So there is a real held-out Indic slice, reported separately. What is still
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
tail.

**82% of the corpus is synthetic** (986 / 214 in the sample). Every headline is
therefore also reported on the `synthetic=False` slice.

### 2.3 The single most useful discovery

`endfiller=True` has a **positive rate of 0.000 across 302 clips**. A trailing
filler *never* marks a turn end. The corpus has pre-labelled the exact
false-interruption trap:

| annotation | n | positive rate |
|---|---|---|
| `endfiller=True` | 302 | **0.000** |
| `midfiller=True` | 468 | 0.515 |
| neither (annotated) | 276 | 0.884 |
| unannotated (`null`) | 263 | 0.513 |

Two consequences. First, the error modes the plan expected to hand-categorise on
Day 9 are already labelled, so "the model struggles with hesitation" becomes a
number. Second, these flags are **tri-state** — `True`, `False`, or `null` — and
folding `null` into `False` would mix "annotated as no filler" with "unknown".
`src/evaluation.py` gives `null` its own slice.

### 2.4 Preprocessing, and the reason for each step

Order: **mono → resample → normalise → window**. Implemented in
`src/audio.py:prepare`.

| decision | reason |
|---|---|
| Average channels rather than take channel 0 | some corpora put the speaker on one channel and near-silence on the other; picking blind would silently reduce half the data to noise |
| Resample with a windowed-sinc filter, not index striding | striding aliases — a 15 kHz component in a 44.1 kHz clip folds down into the speech band, and the model would learn our artefact. Tested in `tests/test_preprocessing.py` |
| Peak-normalise to −1 dBFS | loudness varies with recording conditions and carries no information about turn completion. Un-normalised, a model can score well by learning *which corpus* a clip came from, since gain correlates with source. Measured spread: RMS p5 −23.2 dBFS, p95 −15.9 dBFS |
| Normalise **before** windowing | a short clip padded to 8 s and then normalised would be scaled against its own padding; under RMS normalisation loudness would then correlate with duration |
| Headroom of 1 dB, never full scale | leaves room so a later resample or int16 round-trip cannot clip |
| **Left-pad; left-trim** | the turn boundary is always the *end* of the clip. Keeping audio flush to the right edge puts the decision-relevant moment at a fixed input position for every sample. Right-padding would slide it around with clip length and force the model to learn a position-invariance it never needs. This matches the reference implementation |

### 2.5 Leakage is asserted in code, not in prose

`src/splits.py:assert_no_leakage` runs two independent checks on every training
run, and the run aborts if either fails:

1. **Group overlap** — catches the split logic being wrong.
2. **Clip-ID overlap** — catches the same clip appearing under two source names,
   which grouping alone cannot detect.

Nine tests in `tests/test_splits.py` cover both, including a test that
deliberately leaks and asserts the exception is raised — because an assertion
nothing tests is an assertion nobody knows works.

*Measured* on the 1,200-clip cache: train 961 / val 239 (80/20 requested),
positive rate 0.517 / 0.515, 4 groups / 7 groups, leakage assertion passed.

### 2.6 Cache design

One streaming pass writes a flat int16 memmap plus an offsets index. What is
cached is the **resampled waveform at its natural length** — not mel, not a fixed
window — because that is what keeps the experiment matrix open. Caching mel would
freeze the window length and kill E2; caching a fixed 8 s window would waste ~2×
the disk and make the short-window sweep a crop of padding rather than of audio.

`datasets` ≥ 5 routes audio decoding through `torchcodec`, which needs a matching
FFmpeg and is awkward on Windows and some Colab images. We set `decode=False` and
decode with `soundfile` — one less native dependency, identical result, and the
same decode path the demo uses for uploads.

---

## 3. Baselines — E0 and E0b

A model with no baseline beside it is an unfalsifiable claim.

Both baselines score a clip identically: **the score is the duration of trailing
silence in milliseconds.** They differ only in how they decide which frames are
speech — a fixed energy gate for E0, Silero VAD for E0b. Emitting a continuous
score (rather than a hardcoded 500 ms rule) means the same threshold sweep and
the same cost-based operating-point selection run on the baseline as on the
model, so the comparison is like-for-like. And the chosen threshold is directly
interpretable: "this baseline fires after 485 ms of silence."

### 3.1 Measured — clip-level, on 400 clips of v3.1-test

```
python scripts/run_baselines.py --cache data/cache/test --slices
```

| id | detector | thr | F1 | recall | false interrupt | missed | ROC-AUC | size |
|---|---|---|---|---|---|---|---|---|
| E0 | energy-gated trailing silence | 485 ms | 0.2833 | 0.1838 | 0.0977 | 0.8162 | 0.6602 | 0 MB |
| E0b | Silero VAD trailing silence | 706 ms | 0.2468 | 0.1568 | 0.0977 | 0.8432 | 0.6433 | 10.0 MB |

**The energy gate slightly beats Silero.** That is worth dwelling on: Silero
knows what speech is far better than an energy threshold does, and it still loses.
Knowing where speech *is* does not tell you whether a thought is *finished*. This
is direct evidence that the task is not silence detection — and it is the reason
E0b exists rather than E0 alone.

### 3.2 The degenerate operating point — a finding, not a number to report

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
excludes it by default (`MIN_USEFUL_RECALL`), `operating_points` reports it
explicitly flagged `degenerate`, and the row that goes in the table is the one at
the shared 10% ceiling. Note also E0's best-F1 point interrupts **78.6%** of held
turns — which is why a single ceiling across all detectors is what makes the
table meaningful.

### 3.3 Streamed E0 — F1 collapses to zero

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
that are indistinguishable from turn ends by duration alone.

### 3.4 A pipeline-verification run, and what it already shows

To exercise the Day 4 → Day 12 chain on real audio without waiting for a full
sweep, a deliberately small run was trained: 1.0 s window, 3 epochs, 600 train
rows, frozen encoder (`configs/smoke_e1.yaml`, run id `SMOKE`). **Not a
reportable result** — it is far too small — but it is a genuinely trained model
and the numbers below are measured, not projected.

| detector | F1 | false interrupt | missed | threshold |
|---|---|---|---|---|
| E0 | 0.2833 | 0.0977 | 0.8162 | 485 ms |
| E0b | 0.2468 | 0.0977 | 0.8432 | 706 ms |
| **SMOKE** (3 epochs, 600 rows) | **0.3009** | **0.0326** | 0.8162 | 0.681 |

Training loss fell 0.684 → 0.547 → 0.511. On a 1 s window and 600 examples the
model already beats both baselines on **F1 and false interruptions
simultaneously** — the Gate-4 condition — at a third of E0's interruption rate.
ROC-AUC is the clearer signal: **0.766 against 0.660 (E0) and 0.643 (E0b)**.
That is an encouraging sign for E1 rather than a result to quote.

### 3.5 And the slice that undercuts it

The per-slice breakdown (`artifacts/runs/SMOKE/slices.md`) shows where that
aggregate comes from:

| slice | n | F1 | recall | false interrupt |
|---|---|---|---|---|
| `synthetic=True` (TTS) | 334 | **0.3492** | 0.2185 | 0.0273 |
| `synthetic=False` (human) | 66 | **0.0541** | 0.0294 | 0.0625 |
| `endfiller=True` | 117 | — (all negative) | — | **0.0085** |
| `midfiller=True` | 163 | 0.3571 | 0.2308 | 0.0408 |
| `duration ≥ 6 s` | 253 | 0.2667 | 0.1575 | 0.0238 |

**A 6.5× F1 gap between TTS and human audio, and a 2.3× worse interruption rate
on human audio.** On 66 human clips the model is barely better than chance.

The straightforward reading is that a 3-epoch model trained on a corpus that is
82% synthetic has substantially learned **TTS artefacts** rather than the
prosody of finished speech. It is the single strongest argument for the
`--human-only` cache flag and for reporting the `synthetic=False` slice as the
headline rather than the aggregate.

It is also a caution about the aggregate number two paragraphs up: "beats both
baselines" is true of the full test set and **not** true of the human subset. A
report that quoted only the former would be technically accurate and
substantively misleading. The full-scale run needs to be checked against this
slice specifically, and the fix — training on the human-only subset, or
reweighting toward it — is a flag, not new code.

---

## 4. Model architecture

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

### Heads (all verified to forward correctly)

| head | params | hypothesis under test |
|---|---|---|
| `linear` | 1,153 | the reference design; can frozen features separate the classes at all? |
| `mlp` | 25,473 | is the *head* the bottleneck, or the frozen features? |
| `gru` | 173,697 | endpointing is about *trajectory* — falling pitch, decaying energy, a filler that trails. Pooling throws ordering away; a recurrent head keeps it |
| `attn` | 1,538 | mean pooling dilutes evidence living in the last few hundred ms against seconds of mid-utterance. Isolates *pooling* from head capacity |

Bidirectional GRU is defensible despite this being a streaming problem: the
detector runs on a completed window, so the backward pass only ever sees audio
that has already arrived.

---

## 5. Metrics

```
false_interruption_rate = FP / (FP + TN)    # among turns that had NOT ended
missed_endpoint_rate    = FN / (FN + TP)    # among turns that HAD ended
time_to_decide           = ms from turn end to the emitted decision
```

Accuracy is close to useless here and F1 alone is not much better; the two error
directions have different costs. Reported alongside: precision, recall,
specificity, balanced accuracy, ROC-AUC, PR-AUC, and the full confusion matrix.

Three deliberate choices:

- **A shared 10% false-interruption ceiling** for every reported row. Letting
  each detector pick its favourite tradeoff would rank E0's 78.6%-interruption
  point above a good model held to a budget.
- **Threshold chosen on validation, applied unchanged to test.** Re-picking on
  test reports the best case for a threshold nobody could have known in advance.
  The training script prints the threshold and states that it came from val.
- **Latency as p50/p95/p99/max, never a mean.** A mean hides the tail and the
  tail is what a user hears. `percentiles()` does not expose a mean at all —
  enforced by a test.

Single-class slices return `NaN` for AUC rather than 0.0, because a per-language
slice is often single-class and "0.0" would read as "the model failed on this
language" instead of "undefined".

---

## 6. Streaming inference

Offline classification of pre-cut clips answers a question nobody asks in
production. In a live call audio arrives in 10–20 ms chunks, there is no "clip",
and the detector must decide *while* the person is still talking.

```
in → rolling buffer → resample/normalise/mel → encoder+head → EMA → threshold
   + min-silence + hysteresis → USER_TURN_ENDED
```

Three mechanisms sit between the model's probability and an emitted event. Each
prevents a specific failure, and each has a test that fails if it is removed:

- **EMA smoothing** — a single window crossing the threshold is noise: a breath,
  a codec glitch, a low-energy moment mid-word. A sustained rise is a decision.
  EMA over a boxcar because it needs one float of state and weights the most
  recent window most heavily, which is the right prior for something that just
  happened.
- **Minimum silence (200 ms default)** — natural speech is full of sub-200 ms
  gaps: stop closures, in-breaths, the pause before a searched-for word.
  Requiring the probability to *stay* high separates "they stopped" from "they
  paused".
- **Hysteresis (enter 0.70 / exit 0.45)** — one threshold at 0.5 chatters when
  the signal hovers near it. A dead band means it takes real evidence to cancel a
  decision in progress. Same reason a thermostat has one. An inverted band is
  rejected at construction, not tuned around.

**Chunk size is decoupled from hop size.** A WebRTC track delivers 20 ms frames,
but running an 8 s encoder every 20 ms is waste. Chunks accumulate; the model
runs once per hop (160 ms default). *Measured:* 5 model calls per second of audio
at a 200 ms hop, verified by test.

Latency is measured **end to end** — chunk arrival to event emission — not model
forward time. *Measured* on the demo path: p50 1.2 ms, p95 2.1 ms per hop for the
energy backend, real-time factor 0.018 (56× faster than real time). With the int8
ONNX model: p50 19.1 ms, p95 20.6 ms per emitted decision.

### 6.1 The most important finding in this report: clip scores do not transfer to streaming

The same `SMOKE` model, same threshold, measured two ways:

| evaluation | F1 | false interruption |
|---|---|---|
| clip-level classification | 0.3009 | **0.0326** |
| streamed, 20 ms chunks, 160 ms hop | 0.0270 | **0.4598** |

A 14× worse interruption rate. This is not a bug in the streaming code — it is a
**distribution shift the training data does not cover**, and it is worth stating
carefully because it invalidates a claim that submissions in this space routinely
make.

In training, every window's **right edge coincides with an annotated boundary**:
positive clips end at a real turn end, negative clips end at a real mid-utterance
point. Both are curated moments. In streaming, the right edge is wherever the hop
lands — which is almost never either of those. The model is being asked a
question about window alignments it never saw, and it answers by firing on
ordinary intra-utterance pauses that no training example ever labelled.

**So a clip-level F1 is not evidence about streaming behaviour**, and quoting one
as though it were would be the single most misleading thing this report could do.

### 6.2 Parameter tuning cannot fix it — measured

The obvious response is to tighten the streaming parameters: raise the entry
threshold, lengthen the minimum silence, smooth harder. That was tested directly,
on 120 clips:

| enter | min silence | EMA α | F1 | recall | false interrupt | early fires |
|---|---|---|---|---|---|---|
| 0.680 | 200 ms | 0.40 | 0.0270 | 0.0303 | **0.4598** | 36 |
| 0.900 | 400 ms | 0.30 | 0.0000 | **0.0000** | 0.0000 | 0 |
| 0.970 | 800 ms | 0.20 | 0.0000 | **0.0000** | 0.0000 | 0 |
| 0.995 | 1200 ms | 0.15 | 0.0000 | **0.0000** | 0.0000 | 0 |

**There is no usable operating point.** The detector either interrupts 46% of
held turns or never fires at all — it goes from one degenerate corner to the
other with nothing in between. That is not a tuning problem; a tunable detector
would trade recall for interruptions gradually across that range.

What the discontinuity implies: the smoothed probability spends almost all its
time either near 1 on intra-utterance pauses or near 0 elsewhere, so any
threshold above the pause-triggered peaks also excludes the genuine endpoints.
The model is confidently wrong rather than uncertain, and no amount of smoothing
recovers a signal that is not there.

So of the three candidate fixes, the measurement **rules out the cheapest one**:

1. ~~Tighten the streaming parameters~~ — **tested and ruled out** by the table
   above.
2. **Train on randomly-offset crops**, labelled by whether the crop's right edge
   is genuinely a boundary, so arbitrary window alignments appear in training.
   This is the actual fix. It is a change to `src/dataset.py` — sample a random
   offset per example and derive the label from the offset — not a change to the
   model or the streaming code.
3. **Calibrate the threshold on streamed replay** rather than on clip
   classification. Necessary regardless, but on this evidence not sufficient on
   its own.

This finding is also why `src/streaming.py` exists as its own evaluated component
rather than as a thin wrapper assumed to inherit the model's accuracy. Building
it and measuring it is what surfaced the gap; a submission that only classified
clips would have shipped a detector with a 46% interruption rate while reporting
3%.

---

## 7. Hinglish and robustness

### 7.1 What was built

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

Multiple speakers per phrase is not decoration: with one voice a model could
score well by memorising that voice. *Measured* per-speaker spread for E0 was
modest (F1 0.247 vs 0.209), so the set measures the phenomenon more than the
voice.

### 7.2 Measured — E0 on the Hinglish set

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

### 7.3 The honesty requirement — and a correction to the usual claim

"TTS is an upper bound" is the lazy version of the caveat, and it is **not always
true**. The direction of the synthetic bias depends on the detector:

- For a **prosody/spectral model**, TTS is *easier* — clean, evenly paced, no
  disfluent timing. Its score is optimistic.
- For a **silence-timer baseline**, TTS is *harder* — Bulbul emits almost no
  trailing silence after a completed utterance, so the one cue the baseline
  depends on is largely absent. That is why E0 lands below chance here, and it
  is a property of the audio rather than a fresh insight about the baseline.

The honest reading: **this set measures whether a detector uses linguistic
completion cues rather than duration of silence.** A detector that scores well
here is reading the former.

Every manifest row carries `synthetic: true`, so no downstream script can
average these into a headline figure.

### 7.4 Measured — the model on the Hinglish set

```
python scripts/eval_hinglish.py --checkpoint weights/SMOKE-best.pt
```

| detector | ROC-AUC | fired on deliberately-incomplete utterances |
|---|---|---|
| E0 energy baseline | 0.4757 (below chance) | **3 / 50** |
| SMOKE model | 0.6000 | **0 / 50** |

**Zero false interruptions across all 50 incomplete utterances** — every
`end_filler`, `mid_hesitation`, `code_switch` and Devanagari filler clip was
correctly held. That is the behaviour the whole stress set was built to check.

The trade is visible too: recall on complete utterances is only 0.06–0.25, so the
model is *conservative* — it rarely interrupts and often waits. Given the two
error costs, that is the right corner of the tradeoff to fail into, but it is a
failure to fix with more training rather than a result to celebrate.

Per-speaker F1 differed (shubh 0.292, ritu 0.124), which is a caution about the
set: with only two voices, some of the variance is voice rather than phenomenon.
More speakers would tighten it.

### 7.5 What is still missing

Real recorded Hinglish. TTS covers vocabulary, code-switching, and pause
placement; it does not reproduce genuine disfluent timing, overlapping speech, or
real room acoustics. This is a stated gap, not a solved problem.

---

## 8. Augmentation — and which augmentations are illegal here

Most augmentation libraries offer a grab-bag. On this task some entries are
actively harmful, and knowing which is part of understanding the problem.

**Safe** — changes the channel, not the boundary: gain, additive noise,
band-limiting, reverb. A clip that ended still ended after you add noise.

**Dangerous:**

- *Time-stretch / speed* alters pause duration. A 300 ms mid-sentence pause
  stretched 1.3× becomes 390 ms — squarely turn-final. **The label stops
  describing the audio.** Kept but bounded to ±10% and **off by default**.
- *Cropping from the end* removes the endpoint evidence and relabels a positive
  as a negative. Never applied.
- *Time-masking near the tail* is the same problem in weaker form. Masking is
  confined to the first 70% of the clip — enforced by a test that runs 40 seeds
  and asserts the tail is untouched.

Augmentation is seeded per-sample from the clip index: unseeded augmentation
makes an experiment table unreproducible in a way that is very hard to notice.
Ten tests in `tests/test_augment.py`, including one asserting every non-speed
augmentation is length-preserving.

---

## 9. Experiment matrix

One variable changed per row, every row appended automatically by the training
script. Manual bookkeeping is where experiment tables quietly become fiction.

| id | model | window | F1 | false interrupt | missed | ROC-AUC | latency p50 | size |
|---|---|---|---|---|---|---|---|---|
| **E0** | energy trailing-silence | clip | **0.2833** | **0.0977** | 0.8162 | **0.6602** | — | 0 MB |
| **E0b** | Silero VAD trailing-silence | clip | **0.2468** | **0.0977** | 0.8432 | **0.6433** | — | 10.0 MB |
| **SMOKE** | whisper-tiny frozen + linear *(3 epochs, 600 rows — verification only)* | 1 s | **0.3009** | **0.0326** | 0.8162 | — | 17.8 ms | 30.6 MB |
| **SMOKE-int8** | as above, INT8 ONNX | 1 s | **0.2648** | **0.0233** | — | — | 18.5 ms | **8.02 MB** |
| E1 | whisper-tiny frozen + linear | 8 s | — | — | — | — | — | — |
| E2 | window sweep 0.5/1/1.5/2/4 s | varies | — | — | — | — | — | — |
| E3 | + MLP head | 1.5 s | — | — | — | — | — | — |
| E4 | + GRU head | 1.5 s | — | — | — | — | — | — |
| E4b | + attention pooling | 1.5 s | — | — | — | — | — | — |
| E5 | + augmentation | 1.5 s | — | — | — | — | — | — |
| E7 | unfreeze top 2 blocks | 1.5 s | — | — | — | — | — | — |
| E8 | best, INT8 ONNX | — | — | — | — | — | — | — |

E0/E0b are measured. The remaining rows are produced by
`python scripts/run_matrix.py`; `report/experiments.md` is generated from
`artifacts/experiments.csv` and never hand-edited, so the report cannot drift
from the data.

**Gate 4, the go/no-go:** E1 must beat E0 *and* E0b on F1 and on false
interruptions at the shared ceiling. If it does not, the problem is data or
labels, not architecture.

---

## 10. Error analysis

`scripts/error_analysis.py` samples failures **weighted 60/40 toward false
interruptions** — at a 10% ceiling a uniform sample would be almost all missed
endpoints and would teach nothing — and takes the *most confident* failures
first, because a confidently wrong prediction is the model's fault while a
borderline one is the threshold's.

Modes are named from the corpus's own annotations rather than a hand-built
taxonomy, which is the Day-1 finding paying off:

| observed | named mode |
|---|---|
| FP with `endfiller=True` | cut off a trailing filler |
| FP with `midfiller=True` | cut off a mid-utterance hesitation |
| FP, duration < 1.5 s | fired on a short backchannel |
| FN with `midfiller=True` | missed an endpoint after an internal hesitation |
| FN, duration ≥ 8 s | missed an endpoint on a long utterance |

`--mine-hard-negatives` writes the indices of the most confidently-wrong clips to
`artifacts/runs/error_analysis/hard_negative_indices.npy`.

> **Not yet implemented: the retrain.** Mining produces the indices; nothing
> currently consumes them. There is no training config that oversamples them and
> no code path in `src/dataset.py` or `training/train.py` that reads the file. So
> the loop *symptom → cause → fix → re-measure* stops at "cause" for this
> repository, and the fix step is described in §15 as future work rather than
> claimed as done. When it is implemented, the outcome gets reported either way —
> a documented failed fix reads as rigour; a silently dropped one reads as
> cherry-picking.

`notebooks/error_analysis.ipynb` plays the failures inline. It degrades cleanly
with instructions when no checkpoint exists yet.

### 10.1 Measured — on the `SMOKE` verification run

```
python scripts/error_analysis.py --checkpoint weights/SMOKE-best.pt --mine-hard-negatives
```

7 false interruptions, 23 missed endpoints at threshold 0.681. The distribution
is the useful part:

| missed endpoints | n |
|---|---|
| on a **long utterance** (≥ 8 s) | **14** |
| plain endpoint | 5 |
| after an internal hesitation | 4 |

| false interruptions | n |
|---|---|
| mid-utterance, no filler annotation | 3 |
| cut off a mid-utterance hesitation | 3 |
| cut off a trailing filler | 1 |

**14 of 23 misses are long utterances.** This model has a 1.0 s window, and the
corpus median is 7.2 s — so on a 13 s clip it sees the last 8% of the audio and
has no view of the utterance's structure. That is a direct, mechanical
explanation, and it is exactly what the E2 window sweep exists to fix. It also
argues that the reference's 8 s window is the right default rather than a
conservative one.

False interruptions by annotation:

| slice | n (negatives) | false interrupt |
|---|---|---|
| `endfiller=True` | 117 | **0.0085** |
| `midfiller=True` | 98 | 0.0408 |
| `synthetic=True` | 183 | 0.0273 |
| `synthetic=False` | 32 | **0.0625** |

Two readings. The model almost never cuts off a *trailing* filler (0.85%) — the
most user-visible failure — but is 5× worse on *mid*-utterance hesitation. And it
is **2.3× worse on human audio than on TTS**, which is the concrete reason the
`synthetic=False` slice is reported separately rather than folded into a headline.

158 hard negatives were mined, concentrated in `chirp3_1` (68) and `chirp3_2`
(49) — telling you which source the next run is short of. **They have not been
trained on**; see the note above.

---

## 11. Inference optimisation

`scripts/optimize_model.py` produces four artefacts and reports the delta between
each, separately — bundling them would make it impossible to say which paid off:

1. float32 torch (the checkpoint)
2. int8 torch (dynamic quantisation of Linear/GRU)
3. float32 ONNX (Python out of the loop, graph fusion)
4. int8 ONNX (the deployment artefact)

Dynamic rather than static quantisation: no calibration set needed, and it
quantises exactly the weights that dominate an 8M-parameter encoder. Attention
matmuls stay float, which is why the speedup is real but not 4×.

The script also measures the **mel front-end in isolation**, because at an 8 s
window it is a non-trivial share of a 12 ms budget and no amount of quantisation
touches it. Optimising the wrong half is a common waste.

Benchmarking is batch size 1 (a live call has one stream), single and multi
threaded, 15 warmup iterations discarded — the first calls pay for lazy kernel
selection and thread-pool spin-up, and including them puts a 200 ms outlier in a
12 ms distribution.

Target sentence, to be completed with measured numbers:

> "Model B improves F1 by X% while staying under Y MB and Z ms CPU latency."

`accuracy_delta()` labels each step `improved` / `neutral (within noise)` /
`regression — stated, not hidden`.

### 11.1 Measured — all four artefacts, on the `SMOKE` run

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
measurement.

Four results worth stating plainly rather than smoothing over:

1. **ONNX fp32 export is accuracy-exact** (Δ = 0.0000). Correct — an export that
   moved the numbers would mean the graph did not match the module.
2. **torch dynamic int8 is far worse than ONNX dynamic int8** (−0.199 vs −0.036)
   at a *larger* size (9.42 vs 8.02 MB). onnxruntime quantises per-channel where
   torch's dynamic path is per-tensor, and on a transformer encoder that
   difference is large. If only torch quantisation had been tried, the conclusion
   "int8 is unusable here" would have been wrong.
3. **int8 ONNX is *slower* than fp32 ONNX** on this machine — 18.5 ms vs 13.4 ms.
   Dynamic quantisation inserts quantise/dequantise nodes around each matmul, and
   without VNNI-class int8 instructions the arithmetic saving does not pay for
   them. So on this CPU int8 buys a **3.9× size reduction at a latency cost**,
   which is the opposite of the usual assumption and is why the benchmark exists
   rather than being taken on faith.
4. **The −0.036 F1 cost of int8 should be re-measured on a properly trained
   model.** A model at F1 0.30 has its scores bunched near the threshold, so
   small perturbations flip many predictions. The quantisation penalty on a
   converged model is likely smaller; quoting this number as the final one would
   be overstating the cost.

Also worth noting: peak RSS is 0.7–1.2 GB across backends, dominated by the
runtime rather than the 8 MB of weights. On a memory-constrained deployment that,
not the checkpoint size, is the binding constraint.

---

## 12. Reproducibility

- Every run is driven by a YAML config; an unknown key raises rather than being
  ignored, because a silently-typo'd key means the experiment you think you ran
  is not the one that ran.
- Seeded across `random`, `numpy`, and `torch`.
  `torch.use_deterministic_algorithms` is deliberately **not** set: on a free T4
  the cudnn fallback costs more time than the run-to-run variance it removes.
  Seeded-but-not-bitwise-deterministic is the honest description.
- **Checkpoint every epoch, unconditionally**, and resume automatically. A free
  Colab session disconnects without warning; an uncheckpointed run is a lost day.
  Verified working — a dry run resumed from epoch 1 on re-invocation.
- Each run writes `config`, `history`, `split_report`, `evaluation`, per-slice
  markdown, and raw probability arrays to `artifacts/runs/<run_id>/`.
- `requirements.lock.txt` — 186 pinned packages, frozen after install.
- 81 tests pass: `pytest` — preprocessing 24, streaming 20, metrics 18,
  augmentation 10, splits 9. Coverage is deliberately uneven: it targets the
  modules where a silent error would corrupt a result (preprocessing, metrics,
  splits, streaming). Six modules have **no** direct unit tests — `dataset`,
  `model`, `evaluation`, `baselines`, `inference`, `optimize` — and are currently
  exercised only through the scripts that call them. That is a real gap, not a
  claim of full coverage.

---

## 13. Known limitations

Stated plainly, because a reviewer trusts a declared limitation more than a
suspiciously clean result.

1. **No speaker-aware split is possible.** Grouped by source corpus instead. One
   speaker contributing to two corpora would still span splits and nothing in the
   metadata can detect it.
2. **82% of the corpus is synthetic.** Headlines are also reported on the
   `synthetic=False` slice; the all-data figure is optimistic for a
   prosody-reading model.
3. **The Hinglish set is TTS.** It covers vocabulary, code-switching, and pause
   placement, not genuine disfluent timing or real room acoustics.
4. **Streaming is evaluated on pre-cut clips**, and — more seriously — **a
   clip-level score does not predict streaming behaviour** (§6.1: 0.033 → 0.460
   false-interruption rate for the same model and threshold). Training windows
   always end at an annotated boundary; streaming windows end wherever the hop
   lands. Until the model is trained on randomly-offset crops, the streaming
   number must be measured directly and never inferred. The 1 s early-fire
   tolerance bounds, but does not remove, this weakness.
5. **Noise and reverb augmentation are synthetic** — white noise and an
   exponential-decay impulse, not a licensed noise corpus or measured IRs.
   Robustness to babble and music is *measured* on the stress suite, not trained
   against.
6. **`fp_cost = 4.0` and the 10% ceiling are judgement calls**, stated as numbers
   so they can be argued with rather than assumed.
7. **The measured results here are on subsets** — 1,200 train / 400 test clips —
   sized to prove the pipeline end to end. Full-corpus numbers require the full
   run.

---

## 14. What each command does

```bash
# Day 1-2 — data
python scripts/prepare_data.py --info
python scripts/prepare_data.py --split train --languages eng hin ben mar --max-rows 40000
python scripts/prepare_data.py --split test  --languages eng hin ben mar
jupyter notebook notebooks/eda.ipynb

# Day 3 — Gate 3: baselines before any neural work
python scripts/run_baselines.py --cache data/cache/test --slices

# Day 4-7 — train, then the matrix
python -m training.train --config configs/e1_frozen_linear.yaml --dry-run
python -m training.train --config configs/e1_frozen_linear.yaml
python scripts/run_matrix.py

# Day 8 — Hinglish
python scripts/synthesize_hinglish.py --pause-variants
python scripts/eval_hinglish.py --checkpoint weights/E1-best.pt

# Day 9 — errors
python scripts/error_analysis.py --checkpoint weights/E1-best.pt --mine-hard-negatives

# Day 10 — size and speed
python scripts/optimize_model.py --checkpoint weights/E1-best.pt

# Day 11 — streaming
python scripts/stream_eval.py --checkpoint weights/E1-best.pt --sweep
python scripts/stream_eval.py --baseline    # the like-for-like comparison

# Day 12 — demo
python demo/app.py
```

---

## 15. The claim this repo is built to support

> "I did not just train a model. I understood the problem, designed the data
> pipeline, established baselines, ran experiments, analysed failures, optimised
> inference, and turned it into a deployable real-time system."

Against that, honestly:

| clause | status |
|---|---|
| understood the problem | **done** — Gate 1 answered on real data; three plan assumptions corrected |
| designed the data pipeline | **done** — grouped splits, leakage asserted in code, 24 preprocessing tests |
| established baselines | **done and measured** — E0, E0b, clip-level *and* streamed |
| ran experiments | **built and exercised** — 12 configs, matrix runner, auto-appending table; one real run completed, full sweep pending compute |
| analysed failures | **partly** — 30 failures categorised from corpus annotations, long-utterance concentration found, 158 hard negatives mined. **No fix has been applied or re-measured**, so the loop stops at diagnosis |
| optimised inference | **done and measured** — 4 artefacts, isolated deltas, 8.02 MB int8 ONNX matching the published reference |
| deployable real-time system | **built and measured** — streaming detector with tested smoothing/min-silence/hysteresis, Gradio demo. **And measured to be inadequate as trained** (§6.1), with the cause identified |

The remaining gap is compute, not construction: every phase's code is written,
tested, and has been run end to end on real audio. The one substantive *technical*
gap is §6.1 — the training data needs randomly-offset crops before a streaming
number is respectable — and that is a known change to `src/dataset.py`, not an
open research question.

Worth being explicit about what the last row means. The honest version of the
claim is not "I built a working real-time detector"; it is "I built the real-time
detector, measured it properly, and discovered that clip-level training does not
produce a usable streaming detector — which is why the measurement mattered."

---

## 16. Appendix — the three plan assumptions that were wrong

Worth recording, because checking beats assuming and each one changed the work.

| assumption | reality | what changed |
|---|---|---|
| "Assume no Indian-language audio" | `hin`, `ben`, `mar` all present | a real held-out Indic slice exists; the hand-built set narrowed to code-switching and domain vocabulary, which the corpus genuinely lacks |
| "Severe class imbalance is a risk" | 1.07× — essentially balanced | class weighting became conditional rather than assumed |
| "Whisper Tiny + linear head is our design choice" | that *is* the published v3 architecture | gave a public reference point (8 MB int8, ~10–12 ms CPU) to check our measurements against |

And five things the plan did not anticipate at all, each found by building the
thing and measuring it:

- **`endfiller` / `midfiller` annotations exist.** The Day-9 error taxonomy was
  already labelled in the corpus, and `endfiller=True` has a positive rate of
  exactly 0.000 across 302 clips.
- **The cost function has a degenerate corner.** `4·FPR + FNR` is minimised by
  never firing for any weak detector. Finding this on the baselines rather than on
  the final model is why operating-point selection is guarded and why the table
  uses one shared ceiling.
- **Clip-level scores do not transfer to streaming** — 0.033 → 0.460
  false-interruption rate for the same model at the same threshold (§6.1). The
  most consequential finding here, and one that only appears if you build the
  streaming path and measure it separately.
- **Streaming scoring needs an early-fire rule.** Without it, a detector that
  fires 7.8 s before a boundary is recorded as a true positive because the clip
  eventually ends. A `time_to_decide` of **−7780 ms** was the tell.
- **torch and onnxruntime int8 are not equivalent.** −0.199 F1 versus −0.036 at a
  *larger* size, because one quantises per-tensor and the other per-channel. And
  int8 was *slower* than fp32 on this CPU. Testing only one path would have
  produced a confidently wrong conclusion in either direction.
