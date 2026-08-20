# Submission checklist

Status is marked **only where it has been verified**, and the verification is
named. Anything not verified says so rather than being assumed complete.

| status | meaning |
|---|---|
| ✅ | done and checked, by the command or file named |
| 🟡 | partly done — what is missing is stated |
| ⬜ | not done |
| ⏳ | running now (Colab); blocked on compute, not on work |

Last verified: 2026-08-21.

---

## Repository and reproducibility

| item | status | verified by / gap |
|---|---|---|
| Working GitHub repo | ✅ | `origin https://github.com/Ronitt01/Interview.git`, `main` in sync with remote, 3 commits |
| Reproducible setup | ✅ | `requirements.txt` + `requirements.lock.txt` (186 pins), `scripts/setup.sh`, `Dockerfile` |
| Training configs | ✅ | 12 YAML configs tracked in `configs/`; unknown keys raise rather than being ignored |
| Tests | ✅ | 81 pass — `pytest -q` run 2026-08-21 |
| Test coverage honest about gaps | ✅ | six untested modules named in the report, §17 item 12 |
| Seeded, resumable training | ✅ | seed 1234; per-epoch checkpoint + auto-resume verified by dry run |
| Licence | ✅ | Apache-2.0, `LICENSE` |
| Docker image built and run in CI | ⬜ | `Dockerfile` exists, never built in CI |

## Model artefacts

| item | status | verified by / gap |
|---|---|---|
| Deployment artefact committed | ✅ | `weights/SMOKE-int8.onnx` (8.02 MB) + sidecar, tracked; `.gitignore` policy documented |
| Weights policy is coherent | ✅ | int8 ONNX tracked, `.pt`/fp32 ONNX excluded as derivable — stated in `.gitignore` |
| **E1 weights committed** | ⬜ | `weights/E1-best.pt` is in the Colab session only; not synced |
| **E1 int8 ONNX exported** | ⏳ | optimisation benchmark running in Colab |
| E2 winner exported | ⬜ | winner not yet known |

## Results

| item | status | verified by / gap |
|---|---|---|
| Baselines (E0, E0b) | 🟡 | measured — but on a 400-clip subset with an **in-sample** threshold; not comparable to E1 until re-run on the full test cache |
| **Held-out test result** | ✅ | E1 on 10,791 clips: F1 0.2585, precision 0.9696, recall 0.1491, FI 0.0046, missed 0.8509, ROC-AUC 0.8900, PR-AUC 0.8757 |
| Threshold from validation, unchanged on test | ✅ | 0.909298; `--eval-only` never calls `pick_threshold` |
| No test-set tuning, anywhere | ✅ | E2 configs carry `test_cache_dir: null` so the sweep cannot touch test |
| Leakage assertions | ✅ | group + clip-ID overlap, abort on failure, 9 tests including a deliberate-leak test |
| Final experiment table | 🟡 | `report/final_experiment_table.md` — structure complete, E1 populated, E2 rows `TBD` |
| **E2 window sweep** | ⏳ | 5 windows running in Colab; E1 supplies the 8 s point |
| E2 winner's held-out test eval | ⬜ | runs once, after the winner is chosen on validation |
| E1 per-slice / per-language breakdown | ⬜ | produced per run; not yet transcribed out of Colab |
| Confidence intervals / multi-seed | ⬜ | single seed per experiment; variance unquantified |

## Analysis

| item | status | verified by / gap |
|---|---|---|
| **Streaming evaluation** | 🟡 | built and measured — E0 and `SMOKE` only. Found the 0.033 → 0.460 clip-vs-stream gap. **E1 not yet streamed** |
| Early-fire scoring | ✅ | `EARLY_FIRE_TOLERANCE_MS = 1000`; caught a `ttd_p50` of −7780 ms |
| Streaming tuning grid | ✅ | 4-point grid, ruled out parameter tuning as a fix |
| **Error analysis** | 🟡 | seven categories, all counts from the `SMOKE` run; **E1's error analysis not run** |
| Hard negatives mined | 🟡 | 158 mined; **nothing consumes them** — the retrain is not implemented |
| **Hinglish evaluation** | 🟡 | 202 TTS clips, 7 categories, E0 and `SMOKE` scored; **E1 not scored on it** |
| Held-out Indic audio in main corpus | 🟡 | 3,069 test clips across `hin`/`ben`/`mar`; sliced per run, not yet transcribed |
| **Latency benchmark** | 🟡 | four artefacts measured on `SMOKE` (1 s window); **E1 8 s benchmark running** |
| Size validated against public reference | ✅ | 8.02 MB int8 / 30.94 MB fp32 vs published 8 MB / 32 MB |

## Deliverables

| item | status | verified by / gap |
|---|---|---|
| **Gradio demo** | ✅ | launches and serves **HTTP 200** — verified 2026-08-21 by starting the server and fetching the page, not by building the Blocks object |
| **HF Space entry point** | ✅ | `app.py` + README front matter (`sdk: gradio`, `sdk_version: 6.25.0`, `app_file: app.py`); verified to launch |
| Space actually deployed | ⬜ | not pushed to Hugging Face yet |
| Demo degrades without weights | ✅ | falls back to the E0 baseline and says so in the UI |
| **README** | ✅ | HF front matter, install, commands, weights policy, results pointer |
| **Technical report** | ✅ | `report/technical_report.md`, 18 sections |
| Generated experiment table | ✅ | `report/experiments.md`, regenerated from `artifacts/experiments.csv` |

---

## The four things a reviewer is most likely to ask about

1. **"Your F1 is 0.2585 but ROC-AUC is 0.8900 — which is it?"** Both, and §9.6 of
   the report explains why: the model ranks well and the operating point is badly
   placed, because no config sets `max_false_interruption` so threshold selection
   falls to a cost function with a known degenerate corner. Fixing that is the
   first item in the roadmap and requires no retraining.

2. **"Does the baseline beat your model?"** Not established either way — E0's
   0.2833 and E1's 0.2585 come from different test sets with thresholds chosen
   under different rules, one in-sample. The comparable number is ROC-AUC: 0.8900
   vs 0.6602. Re-running the baselines on the full test cache is an open item.

3. **"Does it work in real time?"** It runs in real time — measured, 56× faster
   than real time on the energy path, ~19 ms per decision for the int8 model at a
   1 s window. It is not yet *accurate* in real time: streamed, the verification
   model interrupts 46% of held turns (§11.2), and parameter tuning was measured
   and ruled out as a fix.

4. **"What did you find that you did not expect?"** That clip-level accuracy does
   not describe streaming behaviour at all — a 14× worse false-interruption rate
   for the same model at the same threshold. Everything about how this project
   reports results follows from that.
