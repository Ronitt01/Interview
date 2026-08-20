"""The orchestrator's safety logic.

These are the parts of scripts/run_autonomous.py whose failure would not raise —
it would just quietly produce a number that means something other than what the
report says it means. Selecting on test, scoring two candidates on the held-out
set, or letting a template's test_cache_dir survive into a generated config all
fail silently, so each gets a test.

No training happens here: every fixture is a hand-written artefact tree.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def load_module(tmp_root: Path):
    """Import run_autonomous with REPO_ROOT pointed at a throwaway tree."""
    spec = importlib.util.spec_from_file_location(
        "run_autonomous_under_test", REPO / "scripts" / "run_autonomous.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_autonomous_under_test"] = mod
    spec.loader.exec_module(mod)
    mod.REPO_ROOT = tmp_root
    mod.STATE_PATH = tmp_root / "artifacts" / "autonomous_state.json"
    mod.LOG_DIR = tmp_root / "artifacts" / "autonomous_logs"
    mod.GEN_CONFIG_DIR = tmp_root / "configs" / "generated"
    return mod


def write_run(root: Path, run_id, *, cost, f1, fi=0.02, missed=0.9, recall=0.08,
              window=8.0, test_cache=None, val_file=False, with_weights=True,
              name=None):
    """Fabricate one run directory. Fixture only -- never a reported number."""
    d = root / "artifacts" / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml.json").write_text(json.dumps({
        "run_id": run_id, "window_seconds": window, "test_cache_dir": test_cache,
    }), encoding="utf-8")
    ev = {
        "name": name or (f"{run_id}/val" if not test_cache else run_id),
        "n": 100, "threshold": 0.5, "roc_auc": 0.8,
        "confusion": {"f1": f1, "precision": 0.8, "recall": recall,
                      "false_interruption_rate": fi,
                      "missed_endpoint_rate": missed, "cost": cost},
    }
    (d / "evaluation.json").write_text(json.dumps(ev), encoding="utf-8")
    if val_file:
        (d / "evaluation_val.json").write_text(json.dumps(ev), encoding="utf-8")
    (d / "history.json").write_text(json.dumps(
        [{"epoch": 1, "val_cost": cost, "seconds": 10.0}]), encoding="utf-8")
    if with_weights:
        w = root / "weights"
        w.mkdir(parents=True, exist_ok=True)
        (w / f"{run_id}-best.pt").write_bytes(b"not a real checkpoint")
    return d


# --------------------------------------------------------------------------- #
# completeness
# --------------------------------------------------------------------------- #
def test_a_run_without_weights_is_not_complete(tmp_path):
    """The session died between scoring and saving. Later stages need the file."""
    m = load_module(tmp_path)
    write_run(tmp_path, "E2_w1p0", cost=0.9, f1=0.2, with_weights=False)
    assert m.run_complete("E2_w1p0") is False


def test_a_run_with_both_is_complete(tmp_path):
    m = load_module(tmp_path)
    write_run(tmp_path, "E2_w1p0", cost=0.9, f1=0.2)
    assert m.run_complete("E2_w1p0") is True


def test_a_truncated_evaluation_json_is_not_complete(tmp_path):
    m = load_module(tmp_path)
    d = tmp_path / "artifacts" / "runs" / "E2_w1p0"
    d.mkdir(parents=True)
    (d / "evaluation.json").write_text('{"confusion":', encoding="utf-8")
    assert m.run_complete("E2_w1p0") is False


# --------------------------------------------------------------------------- #
# selection is validation-only
# --------------------------------------------------------------------------- #
def test_selection_picks_the_lowest_validation_cost(tmp_path):
    m = load_module(tmp_path)
    write_run(tmp_path, "E2_w0p5", cost=0.95, f1=0.10, window=0.5)
    write_run(tmp_path, "E2_w1p5", cost=0.71, f1=0.25, window=1.5)
    write_run(tmp_path, "E2_w4p0", cost=0.88, f1=0.19, window=4.0)
    best, rows = m.select_by_val_cost(["E2_w0p5", "E2_w1p5", "E2_w4p0"])
    assert best["run_id"] == "E2_w1p5"
    assert len(rows) == 3


def test_a_test_scored_run_is_refused_as_a_selection_input(tmp_path):
    """The failure this prevents.

    A run configured with a test cache has *test* numbers in evaluation.json. If
    those were read as validation the winner would be chosen on test, and nothing
    would raise -- the numbers are the right shape.
    """
    m = load_module(tmp_path)
    write_run(tmp_path, "E_TEST", cost=0.10, f1=0.90,
              test_cache="data/cache/test", name="E_TEST")
    assert m.val_metrics("E_TEST") is None

    write_run(tmp_path, "E_VAL", cost=0.80, f1=0.20)
    best, rows = m.select_by_val_cost(["E_TEST", "E_VAL"])
    # E_TEST looks far better and is still not eligible.
    assert best["run_id"] == "E_VAL"
    assert [r["run_id"] for r in rows] == ["E_VAL"]


def test_a_test_scored_run_is_usable_via_its_validation_sidecar(tmp_path):
    """E1 trained with a test cache configured, so its val numbers live in
    evaluation_val.json. That file is unambiguous and is allowed."""
    m = load_module(tmp_path)
    write_run(tmp_path, "E1", cost=0.98, f1=0.13,
              test_cache="data/cache/test", val_file=True, name="E1")
    got = m.val_metrics("E1")
    assert got is not None
    assert got["source"] == "evaluation_val.json"
    assert got["cost"] == pytest.approx(0.98)


def test_selection_returns_nothing_when_no_candidate_has_val_metrics(tmp_path):
    m = load_module(tmp_path)
    best, rows = m.select_by_val_cost(["nope", "also_nope"])
    assert best is None and rows == []


# --------------------------------------------------------------------------- #
# the test-read budget
# --------------------------------------------------------------------------- #
def test_the_same_claim_twice_is_idempotent_so_resume_works(tmp_path):
    m = load_module(tmp_path)
    st = m.State(tmp_path / "state.json")
    assert m.claim_test_read(st, "E2_w1p5", "winner_clip_eval") is True
    assert m.claim_test_read(st, "E2_w1p5", "winner_clip_eval") is True
    assert len(st.data["test_reads"]) == 1


def test_a_second_model_cannot_take_a_spent_purpose(tmp_path):
    """The failure this prevents: scoring two candidates on the held-out set,
    which is selection on test however it is described afterwards."""
    m = load_module(tmp_path)
    st = m.State(tmp_path / "state.json")
    assert m.claim_test_read(st, "E2_w1p5", "winner_clip_eval") is True
    assert m.claim_test_read(st, "E2_w4p0", "winner_clip_eval") is False
    assert len(st.data["test_reads"]) == 1
    assert st.data["test_reads"][0]["model"] == "E2_w1p5"


def test_distinct_purposes_each_get_one_read(tmp_path):
    m = load_module(tmp_path)
    st = m.State(tmp_path / "state.json")
    assert m.claim_test_read(st, "E2_w1p5", "winner_clip_eval") is True
    assert m.claim_test_read(st, "E9", "final_streaming_benchmark") is True
    assert len(st.data["test_reads"]) == 2


def test_the_budget_survives_a_restart(tmp_path):
    m = load_module(tmp_path)
    p = tmp_path / "state.json"
    st = m.State(p)
    m.claim_test_read(st, "E2_w1p5", "winner_clip_eval")
    # process dies, comes back
    st2 = m.State(p)
    assert m.claim_test_read(st2, "E2_w4p0", "winner_clip_eval") is False


def test_a_corrupt_state_file_is_kept_not_silently_overwritten(tmp_path):
    m = load_module(tmp_path)
    p = tmp_path / "state.json"
    p.write_text("{ not json", encoding="utf-8")
    st = m.State(p)
    assert st.data["stages"] == {}
    assert p.with_suffix(".corrupt.json").exists()


# --------------------------------------------------------------------------- #
# generated configs cannot reach the test set
# --------------------------------------------------------------------------- #
def test_derive_config_forces_test_cache_to_null(tmp_path):
    import yaml

    m = load_module(tmp_path)
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({
        "run_id": "E3", "test_cache_dir": "data/cache/test",
        "window_seconds": 1.5, "head": "mlp",
    }), encoding="utf-8")

    out = m.derive_config(base, "E3", window_seconds=4.0, reason="winner window")
    got = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert got["test_cache_dir"] is None      # the template asked for test
    assert got["window_seconds"] == 4.0       # retargeted onto the winner
    assert got["run_id"] == "E3"
    assert got["head"] == "mlp"               # everything else preserved
    assert "generated" in out.parts


def test_a_derived_config_still_loads_as_a_TrainConfig(tmp_path):
    """Guards against the generated YAML drifting from the dataclass, which
    would only surface as a crash minutes into a GPU run."""
    import yaml

    from training.train import TrainConfig

    m = load_module(tmp_path)
    base = REPO / "configs" / "e2_window_1p5.yaml"
    out = m.derive_config(base, "E2_w1p5_probe", window_seconds=2.0)
    cfg = TrainConfig.load(out)
    assert cfg.test_cache_dir is None
    assert cfg.window_seconds == 2.0
    assert cfg.run_id == "E2_w1p5_probe"


def test_every_shipped_template_survives_derivation(tmp_path):
    """Every config the orchestrator derives from must round-trip, or the run
    dies partway through the plan."""
    from training.train import TrainConfig

    m = load_module(tmp_path)
    templates = [
        "e2_window_0p5.yaml", "e2_window_1p0.yaml", "e2_window_1p5.yaml",
        "e2_window_2p0.yaml", "e2_window_4p0.yaml",
        "e3_mlp_head.yaml", "e4_gru_head.yaml", "e4b_attn_pool.yaml",
        "e5_augmentation.yaml", "e9_random_offset.yaml",
        "e10_hard_negatives.yaml",
    ]
    for t in templates:
        out = m.derive_config(REPO / "configs" / t, f"PROBE_{t[:6]}",
                              window_seconds=1.5)
        cfg = TrainConfig.load(out)
        assert cfg.test_cache_dir is None, t


# --------------------------------------------------------------------------- #
# auto-fixes
# --------------------------------------------------------------------------- #
def test_oom_halves_batch_size_in_the_config_not_on_the_command_line(tmp_path):
    """train.py has no --batch-size flag.

    Appending one would fail with "unrecognized arguments" at the exact moment a
    GPU ran out of memory, which is the situation the fix exists for. So the fix
    rewrites the generated config, which also leaves an auditable record of the
    batch size that actually ran.
    """
    import yaml

    m = load_module(tmp_path)
    cfg = tmp_path / "E3.yaml"
    cfg.write_text(yaml.safe_dump({"run_id": "E3", "batch_size": 32,
                                   "notes": "x"}), encoding="utf-8")
    cmd = [sys.executable, "-m", "training.train", "--config", str(cfg)]

    got = m.find_fix("RuntimeError: CUDA out of memory. Tried to allocate", cmd)
    assert got is not None
    fix, new = got
    assert fix.name == "cuda-oom"
    assert fix.changes_semantics is True      # must not look comparable
    assert new == cmd, "the command must not gain a flag train.py cannot parse"
    after = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert after["batch_size"] == 16
    assert "NOT strictly controlled" in after["notes"]


def test_the_oom_fix_gives_up_rather_than_halving_forever(tmp_path):
    import yaml

    m = load_module(tmp_path)
    cfg = tmp_path / "E3.yaml"
    cfg.write_text(yaml.safe_dump({"run_id": "E3", "batch_size": 4}),
                   encoding="utf-8")
    cmd = [sys.executable, "-m", "training.train", "--config", str(cfg)]
    assert m.find_fix("CUDA out of memory", cmd) is None


def test_the_oom_fix_declines_when_there_is_no_config_to_edit(tmp_path):
    m = load_module(tmp_path)
    cmd = [sys.executable, "scripts/stream_eval.py", "--checkpoint", "w.pt"]
    assert m.find_fix("CUDA out of memory", cmd) is None


def test_onnx_failure_falls_back_to_the_torch_arm(tmp_path):
    m = load_module(tmp_path)
    cmd = [sys.executable, "scripts/optimize_model.py", "--checkpoint", "w.pt"]
    got = m.find_fix("onnxruntime.capi.InferenceError: shape mismatch", cmd)
    assert got is not None
    fix, new = got
    assert "--skip-onnx" in new
    assert fix.changes_semantics is False    # measures less, means the same


def test_an_unrecognised_failure_gets_no_fix(tmp_path):
    """Better to leave a stage failed than to guess at a repair."""
    m = load_module(tmp_path)
    cmd = [sys.executable, "-m", "training.train", "--config", "x.yaml"]
    assert m.find_fix("AssertionError: labels are not what I expected", cmd) is None


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def test_the_report_says_not_run_rather_than_inventing_a_number(tmp_path):
    m = load_module(tmp_path)
    (tmp_path / "report").mkdir(parents=True)
    (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)
    write_run(tmp_path, "E1", cost=0.98, f1=0.13, val_file=True)

    args = m.argparse.Namespace(
        train_cache="data/cache/train", test_cache="data/cache/test",
        gap_fi=0.05, stream_rows=300, stream_timeout=10, optimize_rows=100,
        optimize_timeout=10,
    )
    ctx = m.Ctx(state=m.State(tmp_path / "state.json"), args=args)
    out = m.write_final_report(ctx)
    text = out.read_text(encoding="utf-8")

    assert "not run" in text
    assert "Held-out test reads: **0**" in text
    # No streaming or latency artefacts exist, so neither section may claim one.
    assert "## Streaming metrics" in text
    assert "## Latency and model size" in text


def test_readiness_is_a_filesystem_measurement(tmp_path):
    m = load_module(tmp_path)
    args = m.argparse.Namespace(
        train_cache="t", test_cache="s", gap_fi=0.05, stream_rows=1,
        stream_timeout=1, optimize_rows=1, optimize_timeout=1,
    )
    ctx = m.Ctx(state=m.State(tmp_path / "state.json"), args=args)

    before, checks = m.readiness(ctx)
    assert before == 0

    write_run(tmp_path, "E1", cost=0.98, f1=0.13, val_file=True)
    after, _ = m.readiness(ctx)
    assert after == before + 1, "E1 completing should move exactly one check"


# --------------------------------------------------------------------------- #
# recovering validation numbers for a run that also scored test
# --------------------------------------------------------------------------- #
def test_val_metrics_recomputes_from_arrays_when_only_test_json_exists(tmp_path):
    """E1's exact situation, and the most misleading way this could fail.

    A training run configured with a test cache writes TEST numbers to
    evaluation.json and keeps validation only as arrays. If that run were simply
    excluded from selection, the sweep would lose its 8 s point while the table
    still looked complete. So the confusion is recomputed from the run's own
    val_probs/val_labels at the threshold it recorded.
    """
    m = load_module(tmp_path)
    d = write_run(tmp_path, "E1", cost=0.85, f1=0.99,      # f1 here is the TEST
                  test_cache="data/cache/test", name="E1")  # number, unusable
    # Arrays the run saved, plus the threshold it selected on validation.
    rng = np.random.default_rng(0)
    y = (rng.random(200) > 0.5).astype(np.int64)
    probs = np.where(y == 1, rng.uniform(0.4, 1.0, 200),
                     rng.uniform(0.0, 0.6, 200))
    np.save(d / "val_labels.npy", y)
    np.save(d / "val_probs.npy", probs)
    (d / "history.json").write_text(json.dumps(
        [{"epoch": 1, "val_cost": 0.85, "threshold": 0.7, "seconds": 1.0}]),
        encoding="utf-8")

    got = m.val_metrics("E1")
    assert got is not None, "E1 must not be dropped from the comparison"
    assert got["source"] == "recomputed from val_probs.npy"
    assert got["threshold"] == pytest.approx(0.7)
    assert got["n"] == 200
    # The recomputed F1 is from validation, so it must NOT be the test value.
    assert got["f1"] != pytest.approx(0.99)


def test_recomputation_reproduces_the_runs_own_recorded_val_numbers(tmp_path):
    """Cross-check: recomputing at the recorded threshold must land on the
    val_f1 the training loop itself printed. If these ever diverge, one of the
    two is measuring something else."""
    m = load_module(tmp_path)
    d = write_run(tmp_path, "R", cost=0.5, f1=0.0, test_cache="data/cache/test",
                  name="R")
    y = np.array([1, 1, 1, 0, 0, 0, 1, 0], dtype=np.int64)
    probs = np.array([0.9, 0.8, 0.2, 0.1, 0.75, 0.3, 0.95, 0.05])
    np.save(d / "val_labels.npy", y)
    np.save(d / "val_probs.npy", probs)

    from src.metrics import confusion_at
    expected = confusion_at(y, probs, 0.7)
    (d / "history.json").write_text(json.dumps(
        [{"epoch": 1, "val_cost": expected.cost(4.0), "threshold": 0.7,
          "val_f1": expected.f1, "seconds": 1.0}]), encoding="utf-8")

    got = m.val_metrics("R")
    assert got["f1"] == pytest.approx(expected.f1)
    assert got["recall"] == pytest.approx(expected.recall)
    assert got["false_interruption_rate"] == pytest.approx(
        expected.false_interruption_rate)


def test_a_run_with_no_arrays_and_no_val_json_is_still_excluded(tmp_path):
    """The fallback must not become a way to admit a test-only run."""
    m = load_module(tmp_path)
    write_run(tmp_path, "T", cost=0.1, f1=0.95, test_cache="data/cache/test",
              name="T")
    # no val_probs.npy / val_labels.npy written
    assert m.val_metrics("T") is None


# --------------------------------------------------------------------------- #
# a sandboxed experiment table must not overwrite the real one
# --------------------------------------------------------------------------- #
def test_a_sandboxed_table_writes_its_markdown_beside_itself(tmp_path):
    """Regression. Every call site calls save_markdown() with no argument, so a
    hardcoded default meant a rehearsal pointed at a scratch table_path silently
    overwrote report/experiments.md with throwaway rows. It happened twice.
    """
    from src.evaluation import ExperimentTable

    scratch = tmp_path / "scratch" / "experiments.csv"
    t = ExperimentTable(str(scratch))
    t.append({"id": "REHEARSAL", "model": "throwaway", "f1": 0.01})
    out = t.save_markdown()

    assert out == scratch.with_suffix(".md")
    assert "report" not in out.parts
    assert "REHEARSAL" in out.read_text(encoding="utf-8")


def test_the_canonical_table_still_writes_report_experiments_md():
    from src.evaluation import ExperimentTable

    t = ExperimentTable(str(REPO / "artifacts" / "experiments.csv"))
    assert t.default_markdown_path() == REPO / "report" / "experiments.md"
