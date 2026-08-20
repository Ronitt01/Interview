# Experiment matrix

Generated from `artifacts/experiments.csv` — do not edit by hand.

| id | model | window | n | threshold | acc | f1 | precision | recall | false_interrupt | missed | roc_auc | pr_auc | size_mb | params_m | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| REH_w0p5/val | whisper-tiny linear | 0.5s | 239 | 0.6289 | 0.4854 | 0.1022 | 0.5 | 0.0569 | 0.0603 | 0.9431 | 0.5032 | 0.5137 | 30.6 | 7.64 | window sweep: 0.5s (E1 config otherwise unchanged) |
| REH_w1p0/val | whisper-tiny linear | 1s | 239 | 0.6484 | 0.5063 | 0.1061 | 0.7778 | 0.0569 | 0.0172 | 0.9431 | 0.5878 | 0.5751 | 30.64 | 7.65 | window sweep: 1.0s (E1 config otherwise unchanged) |
