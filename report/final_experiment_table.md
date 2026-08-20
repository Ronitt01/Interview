# Final experiment table

**This file is hand-maintained.** It is the one table in the repo that is not
machine-generated, because it has to hold results from runs whose artefacts live
in a Colab session that has not been synced back yet. Every cell is either a
number transcribed from an executed run — with its provenance named below — or
the literal string `TBD`. There are no estimates, no projections, and no
interpolations.

`report/experiments.md` remains the machine-generated table, written from
`artifacts/experiments.csv` by `src/evaluation.py:ExperimentTable`. Where the two
disagree, the generated one is authoritative for the rows it contains; this file
is broader because it also records runs whose artefacts are still remote.

---

## Read this before reading the table

Three things make naive row-to-row comparison wrong. They are stated here rather
than in a footnote because a reviewer who misses them will draw a wrong
conclusion from a correct table.

**1. Two different test sets.** E0/E0b were scored on a 400-clip subset built
during pipeline development. E1 was scored on the full 10,791-clip held-out test
cache. A 400-clip F1 has a wide confidence interval and the two `n` values are
27× apart, so *"E0 scores 0.2833 and E1 scores 0.2585, so the baseline wins"* is
not a supported reading. **The baselines have not yet been re-run on the full
test cache.** Until they are, E0 vs E1 is not a like-for-like comparison. This is
CPU-only work — no GPU, no training — and it is listed as an open item.

**2. Two different threshold provenances.** `scripts/run_baselines.py` sweeps the
threshold on the same clips it then reports (`src/baselines.py:evaluate_baseline`
calls `evaluate` on the cache it was handed). That is an **in-sample** threshold:
the best case for a threshold nobody could have chosen in advance. E1's threshold
was selected on validation and applied unchanged to test — **out-of-sample**. The
asymmetry favours the baselines, so a baseline row is an optimistic bound and the
E1 row is not.

**3. `val` and `test` columns are not interchangeable.** The E1 val and test
splits differ in positive rate and in source-corpus mix (they come from two
different published datasets, v3.2-train and v3.1-test). E1's test F1 is roughly
double its val F1 at the same threshold; that is a property of the two sets, not
evidence of improvement.

---

## The table

| Experiment | Question | Window | Architecture | Val F1 | Val Precision | Val Recall | Val FI | Val Missed | Test F1 | Test FI | Test Missed | Latency p50 | Winner |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **E0** | can trailing-silence duration alone do this? | whole clip | energy gate + silence timer | n/a | n/a | n/a | n/a | n/a | 0.2833 | 0.0977 | 0.8162 | — | no — reference floor |
| **E0b** | does a real VAD beat an energy gate? | whole clip | Silero VAD + silence timer | n/a | n/a | n/a | n/a | n/a | 0.2468 | 0.0977 | 0.8432 | — | no — loses to E0 |
| **E1** | can a frozen encoder + linear head separate the classes at all? | 8.0 s | whisper-tiny frozen, mean pool, linear | 0.1363 | 0.8395 | 0.0742 | 0.0142 | 0.9258 | 0.2585 | 0.0046 | 0.8509 | TBD | reference point for E2 |
| **E2 / 0.5 s** | is the window the binding constraint? | 0.5 s | as E1 | TBD | TBD | TBD | TBD | TBD | not run | not run | not run | TBD | TBD |
| **E2 / 1.0 s** | " | 1.0 s | as E1 | TBD | TBD | TBD | TBD | TBD | not run | not run | not run | TBD | TBD |
| **E2 / 1.5 s** | " | 1.5 s | as E1 | TBD | TBD | TBD | TBD | TBD | not run | not run | not run | TBD | TBD |
| **E2 / 2.0 s** | " | 2.0 s | as E1 | TBD | TBD | TBD | TBD | TBD | not run | not run | not run | TBD | TBD |
| **E2 / 4.0 s** | " | 4.0 s | as E1 | TBD | TBD | TBD | TBD | TBD | not run | not run | not run | TBD | TBD |
| **E2 / 8.0 s** | " | 8.0 s | as E1 | 0.1363 | 0.8395 | 0.0742 | 0.0142 | 0.9258 | 0.2585 | 0.0046 | 0.8509 | TBD | = E1, supplies the 8 s point |
| **E8** | what does the deployment artefact cost in accuracy? | winner's | winner, INT8 ONNX | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

`not run` in the E2 test columns is deliberate and is not the same as `TBD`. The
sweep configs carry `test_cache_dir: null`, so no E2 window scores the held-out
set. Scoring all five would spend the test set's independence five times over
before a decision was made. One test evaluation runs afterwards, on the
validation-selected winner, via `--eval-only`. See §10 of the technical report.

Rows E3, E4, E4b, E5, E7 have configs in `configs/` and are not yet run; they are
omitted here rather than listed as all-TBD, because a table of empty rows implies
a schedule this project has not committed to.

---

## E1 in full

The table above carries the columns the comparison needs. These are the rest of
the E1 numbers, so nothing measured is lost to column budget.

Run: `configs/e1_frozen_linear.yaml`, Colab T4, Python 3.12.13, PyTorch
2.11.0+cu128. Threshold **0.909298**, selected on validation, applied unchanged
to test.

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

Val ROC-AUC/PR-AUC are `TBD` here only because they were not transcribed with
the rest; they are in `artifacts/runs/E1/evaluation.json` in the Colab session and
should be filled in when those artefacts are synced.

## Data the results are computed on

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

Validation is carved out of the 40,000 train clips at `val_fraction: 0.2`,
grouped by `dataset`. The test cache is built from the separately-published
`smart-turn-data-v3.1-test`, so train and test share no source rows by
construction.

## Test-set integrity

- Threshold selected on validation, applied unchanged to test.
- No threshold was tuned on test, at any point, for any run.
- The test set was read once per reported model, for final scoring only.
- Splits are grouped by `dataset` (source corpus), with group-overlap and
  clip-ID-overlap assertions that abort the run on failure
  (`src/splits.py:assert_no_leakage`).
- Speaker IDs do not exist in the source metadata, so a speaker-aware split is
  impossible. Documented as a limitation, not claimed as done.

## What is still outstanding, and why

| item | blocked on |
|---|---|
| E2 val results (5 windows) | Colab sweep in progress |
| E2 winner's held-out test result | the sweep finishing, then one `--eval-only` |
| Latency p50 for every trained row | Colab `optimize_model.py` benchmark in progress |
| E1 artefacts in `artifacts/runs/E1/` | not yet synced from Colab |
| E0/E0b on the full 10,791-clip test cache | nothing — CPU-only, not yet run |
| E8 (winner quantised) | the winner being known |
