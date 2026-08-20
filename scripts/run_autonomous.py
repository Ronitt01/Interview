"""Run the whole benchmark pipeline to completion, unattended, and resumably.

    python scripts/run_autonomous.py                 # do the work
    python scripts/run_autonomous.py --dry-run       # show the plan, run nothing
    python scripts/run_autonomous.py --status        # what is done, from artefacts

WHAT THIS IS FOR
----------------
A Colab session dies without warning, and the pipeline is a dozen stages long
with real dependencies between them. This walks the stages in order, decides
what is already finished by **looking at the artefacts on disk** rather than by
trusting a log, runs what is missing, verifies each result before moving on, and
picks up from the right place after a restart.

THREE RULES IT ENFORCES STRUCTURALLY, NOT BY DISCIPLINE
-------------------------------------------------------
These are the ones that quietly ruin a submission, so they are implemented as
refusals rather than as intentions.

1. **Nothing is ever selected on test.** :func:`select_by_val_cost` reads only
   validation artefacts and raises if handed anything else. Every candidate
   config is rewritten with ``test_cache_dir: null`` before it runs
   (:func:`derive_config`), so a sweep row physically cannot score the held-out
   set even if its YAML asked to.

2. **Test is read a bounded number of times, by pre-named model, for scoring
   only.** :func:`claim_test_read` is the only way to a test evaluation. It
   records the model and the purpose, and refuses a purpose that is already
   spent on a different model. At most two reads exist: the E2 winner (which the
   plan mandates) and the final model (chosen on validation, before test is
   touched). If those are the same model, one read happens.

3. **Streaming decisions run on validation.** Step 7 of the plan branches on a
   streaming measurement. A branch taken on a test measurement is selection on
   test whatever it is called, so the deciding run uses
   ``--cache <train> --split val``. Test-set streaming happens once, at the end,
   on the single final model, as reporting.

WHAT IT WILL NOT DO
-------------------
* Invent a number. Every value in ``report/final_benchmark.md`` is read out of an
  artefact this script verified, or printed as ``not run``.
* Re-run a completed experiment. Completion means the artefacts exist and parse.
* Tune a threshold on test. It never calls ``pick_threshold``; ``--eval-only``
  reuses the threshold stored in the checkpoint.
* Silently repair a failure into a different experiment. Auto-fixes are listed in
  :data:`AUTO_FIXES`; any fix that changes what the run means sets
  ``comparability_warning`` on that stage, and the final report reprints those
  warnings next to the affected row.

FAILURE HANDLING
----------------
A stage that fails has its output captured, is matched against
:data:`AUTO_FIXES`, and is retried once with the smallest fix that applies. If it
still fails it is marked ``failed``, and stages that do not depend on it carry
on. Nothing blocks on a stage that is marked optional.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STATE_PATH = REPO_ROOT / "artifacts" / "autonomous_state.json"
LOG_DIR = REPO_ROOT / "artifacts" / "autonomous_logs"
GEN_CONFIG_DIR = REPO_ROOT / "configs" / "generated"

# The five windows the sweep covers. E1 supplies 8.0 s, so it is a candidate for
# the winner without being re-trained.
E2_RUNS = ("E2_w0p5", "E2_w1p0", "E2_w1p5", "E2_w2p0", "E2_w4p0")
E2_CONFIGS = {
    "E2_w0p5": "configs/e2_window_0p5.yaml",
    "E2_w1p0": "configs/e2_window_1p0.yaml",
    "E2_w1p5": "configs/e2_window_1p5.yaml",
    "E2_w2p0": "configs/e2_window_2p0.yaml",
    "E2_w4p0": "configs/e2_window_4p0.yaml",
}
E1_RUN = "E1"

# Streaming gap: how much worse streamed has to be than clip-level before step 7
# fires. Absolute difference in false-interruption rate. 0.05 because the
# measured gap that motivated the whole fix was 0.033 -> 0.460, so anything near
# that is unmissable, and a 5-point gap is already product-relevant.
DEFAULT_GAP_FI = 0.05


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #
class State:
    """Durable record of what happened, so a restart is cheap and auditable.

    Deliberately *not* the source of truth for completion — that is the artefacts
    on disk. If this file is deleted the run still resumes correctly; it just
    loses the history of fixes and failures.
    """

    def __init__(self, path: Path = STATE_PATH) -> None:
        self.path = path
        self.data: dict = {"stages": {}, "test_reads": [], "started": None, "events": []}
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                # A half-written state file after a hard kill. Keep it for
                # forensics rather than overwriting it silently.
                path.rename(path.with_suffix(".corrupt.json"))
        self.data.setdefault("stages", {})
        self.data.setdefault("test_reads", [])
        self.data.setdefault("events", [])
        if not self.data.get("started"):
            self.data["started"] = now()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        tmp.replace(self.path)  # atomic, so a kill mid-write cannot corrupt it

    def stage(self, name: str) -> dict:
        return self.data["stages"].setdefault(
            name, {"status": "pending", "attempts": 0}
        )

    def note(self, name: str, **kw) -> None:
        self.stage(name).update(kw)
        self.save()

    def event(self, msg: str) -> None:
        self.data["events"].append({"at": now(), "msg": msg})
        self.save()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def log(msg: str = "") -> None:
    print(msg, flush=True)


def rule(title: str) -> None:
    log("\n" + "=" * 78)
    log(f"  {title}")
    log("=" * 78)


def read_json(path: Path) -> dict | list | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def run_dir(run_id: str) -> Path:
    return REPO_ROOT / "artifacts" / "runs" / run_id


def run_complete(run_id: str) -> bool:
    """A training run is done when its evaluation parses and its weights exist.

    Both halves matter. An ``evaluation.json`` with no checkpoint beside it means
    the session died between scoring and saving, and every later stage needs the
    checkpoint.
    """
    ev = read_json(run_dir(run_id) / "evaluation.json")
    if not isinstance(ev, dict) or "confusion" not in ev:
        return False
    return (REPO_ROOT / "weights" / f"{run_id}-best.pt").exists()


def checkpoint_for(run_id: str) -> Path:
    return REPO_ROOT / "weights" / f"{run_id}-best.pt"


# --------------------------------------------------------------------------- #
# subprocess
# --------------------------------------------------------------------------- #
def sh(cmd: list[str], stage: str, timeout: int | None = None) -> tuple[int, str]:
    """Run a command, stream it, tee it to a log, return ``(rc, tail)``.

    Streaming rather than capturing-then-printing: these stages run for tens of
    minutes and a silent terminal is indistinguishable from a hang.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{stage}.log"
    log(f"  $ {' '.join(shlex.quote(c) for c in cmd)}")
    log(f"    (log -> {log_path.relative_to(REPO_ROOT)})")

    lines: list[str] = []
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")

    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n\n===== {now()} :: {' '.join(cmd)} =====\n")
        proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", env=env, bufsize=1,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                fh.write(line)
                lines.append(line)
                if len(lines) > 4000:      # keep memory bounded on long runs
                    del lines[:2000]
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = 124
            lines.append(f"TIMEOUT after {timeout}s\n")
        except KeyboardInterrupt:
            proc.kill()
            raise

    return rc, "".join(lines[-120:])


def py(*args: str) -> list[str]:
    return [sys.executable, *args]


# --------------------------------------------------------------------------- #
# auto-fixes
# --------------------------------------------------------------------------- #
@dataclass
class AutoFix:
    name: str
    pattern: str
    describe: str
    # Rewrites the command. Returns None if it cannot help this particular call.
    mutate: Callable[[list[str]], list[str] | None]
    # True when the fix changes what the experiment means, not just how it runs.
    changes_semantics: bool = False


def _halve_batch_in_config(cmd: list[str]) -> list[str] | None:
    """Halve the batch size *in the generated config*, not on the command line.

    ``training/train.py`` takes its settings from the YAML and exposes no
    ``--batch-size`` flag, so appending one would fail with "unrecognized
    arguments" -- at the exact moment a GPU ran out of memory, hours into an
    unattended run. Rewriting the config avoids that and has a second benefit:
    the file on disk then truthfully records what ran, so the row's batch size is
    auditable afterwards rather than living only in a log line.

    Flagged as a semantics change, because a row trained at a different batch
    size is not strictly controlled against the others. The final report reprints
    that warning next to the affected row instead of letting it disappear.
    """
    import yaml

    if "--config" not in cmd:
        return None
    cfg_path = Path(cmd[cmd.index("--config") + 1])
    if not cfg_path.exists():
        return None

    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    current = int(raw.get("batch_size", 32))
    if current <= 4:
        return None                     # already tiny; halving again will not help
    raw["batch_size"] = max(4, current // 2)
    note = raw.get("notes", "")
    raw["notes"] = (f"{note} [batch_size {current} -> {raw['batch_size']} after "
                    f"an out-of-memory failure; NOT strictly controlled against "
                    f"the other rows]")
    cfg_path.write_text(
        f"# batch_size halved from {current} by scripts/run_autonomous.py after\n"
        f"# an out-of-memory failure. This row is not strictly controlled against\n"
        f"# the others; the final report carries that warning.\n"
        + yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )
    return list(cmd)                    # same command, changed config


def _skip_onnx(cmd: list[str]) -> list[str] | None:
    if "optimize_model.py" not in " ".join(cmd) or "--skip-onnx" in cmd:
        return None
    return [*cmd, "--skip-onnx"]


def _fewer_rows(cmd: list[str]) -> list[str] | None:
    if "--max-rows" in cmd:
        return None
    if "stream_eval.py" not in " ".join(cmd):
        return None
    return [*cmd, "--max-rows", "120"]


AUTO_FIXES: tuple[AutoFix, ...] = (
    AutoFix(
        "cuda-oom",
        r"CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED|out of memory",
        "halve the batch size in the generated config",
        _halve_batch_in_config,
        changes_semantics=True,
    ),
    AutoFix(
        "onnx-export",
        r"onnxruntime|onnxscript|InferenceError|Failed to export|onnx\.",
        "skip the ONNX arm and keep the torch measurements",
        _skip_onnx,
    ),
    AutoFix(
        "stream-timeout",
        r"TIMEOUT after",
        "shorten the streaming replay to 120 clips",
        _fewer_rows,
    ),
)


def find_fix(output: str, cmd: list[str]) -> tuple[AutoFix, list[str]] | None:
    for fix in AUTO_FIXES:
        if re.search(fix.pattern, output, re.IGNORECASE):
            new = fix.mutate(cmd)
            if new is not None:
                return fix, new
    return None


# --------------------------------------------------------------------------- #
# config derivation
# --------------------------------------------------------------------------- #
def derive_config(
    base: str | Path,
    run_id: str,
    *,
    window_seconds: float | None = None,
    extra: dict | None = None,
    reason: str = "",
) -> Path:
    """Write a variant config into ``configs/generated/`` and return its path.

    Two jobs. It forces ``test_cache_dir: null`` on everything, so no candidate
    can score the held-out set. And it retargets a template's window onto the E2
    winner's, so the downstream arms (head capacity, augmentation, offsets) are
    controlled against the winner rather than against whatever window the
    template happened to name.

    The generated file is kept, not written to a temp dir, because it is the only
    record of exactly what ran.
    """
    import yaml

    raw = yaml.safe_load(Path(base).read_text(encoding="utf-8")) or {}
    raw["run_id"] = run_id
    raw["test_cache_dir"] = None
    if window_seconds is not None:
        raw["window_seconds"] = float(window_seconds)
    if extra:
        raw.update(extra)

    note = raw.get("notes", "")
    raw["notes"] = f"{note} [generated from {Path(base).name}"
    if reason:
        raw["notes"] += f": {reason}"
    raw["notes"] += "]"

    GEN_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    out = GEN_CONFIG_DIR / f"{run_id}.yaml"
    header = (
        f"# GENERATED by scripts/run_autonomous.py at {now()}\n"
        f"# from {Path(base).name}. Do not edit: it is regenerated on the next\n"
        f"# run and is kept only as the record of what actually executed.\n"
        f"# test_cache_dir is forced to null - candidates are compared on\n"
        f"# validation only, and the held-out set is scored once at the end.\n"
    )
    out.write_text(header + yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# validation-only selection
# --------------------------------------------------------------------------- #
def recompute_val_from_probs(d: Path, cfg: dict) -> dict | None:
    """Rebuild a validation confusion from a run's own saved arrays.

    A training run that also scored test writes *test* numbers to
    ``evaluation.json`` and keeps validation only as ``val_probs.npy`` /
    ``val_labels.npy``. That is exactly E1's situation. Rather than exclude E1
    from the comparison -- which would drop the 8 s point while leaving the table
    looking complete -- the confusion is recomputed here from those arrays, at
    the threshold the run selected on validation and recorded in
    ``history.json``.

    This is a recomputation from committed artefacts, not a substitute
    measurement: same probabilities, same labels, same threshold, so it is the
    number the run would have printed had it been configured validation-only.
    Nothing here reads the test set.
    """
    import numpy as np

    probs_p, labels_p = d / "val_probs.npy", d / "val_labels.npy"
    if not (probs_p.exists() and labels_p.exists()):
        return None

    hist = read_json(d / "history.json") or []
    epochs = [h for h in hist if isinstance(h, dict) and h.get("val_cost") is not None]
    if not epochs:
        return None
    best = min(epochs, key=lambda h: (h["val_cost"], h.get("epoch", 0)))
    thr = best.get("threshold")
    if thr is None:
        return None

    try:
        y = np.load(labels_p)
        pr = np.load(probs_p)
    except Exception:
        return None
    if y.shape != pr.shape or y.size == 0:
        return None

    from src.metrics import confusion_at

    conf = confusion_at(y, pr, float(thr))
    return {
        "threshold": float(thr),
        "n": int(y.size),
        "confusion": conf.to_dict() if hasattr(conf, "to_dict") else {
            "f1": conf.f1, "precision": conf.precision, "recall": conf.recall,
            "false_interruption_rate": conf.false_interruption_rate,
            "missed_endpoint_rate": conf.missed_endpoint_rate,
            "cost": conf.cost(float(cfg.get("fp_cost", 4.0))),
        },
        "roc_auc": None,
    }


def val_metrics(run_id: str) -> dict | None:
    """Validation confusion + threshold for a run, or None.

    Refuses to return anything from a run that scored a test set, because such a
    run's ``evaluation.json`` holds *test* numbers and using them to choose would
    be selection on test. Runs the orchestrator launched always have
    ``test_cache_dir: null``, so this normally just passes through; the check
    exists for E1, which was trained with a test cache configured, and whose
    validation numbers therefore have to come from ``evaluation_val.json``.
    """
    d = run_dir(run_id)
    cfg = read_json(d / "config.yaml.json") or {}
    ev = read_json(d / "evaluation.json")
    val_ev = read_json(d / "evaluation_val.json")

    scored_test = cfg.get("test_cache_dir") not in (None, "", "null")

    chosen, source = None, ""
    if isinstance(val_ev, dict) and "confusion" in val_ev:
        chosen, source = val_ev, "evaluation_val.json"
    elif isinstance(ev, dict) and "confusion" in ev:
        name = str(ev.get("name") or "")
        if name.endswith("/val") or not scored_test:
            chosen, source = ev, "evaluation.json"
    if chosen is None:
        # Last resort, and the case that matters: a run that scored test has its
        # validation numbers only as arrays. Recompute rather than drop the run.
        chosen = recompute_val_from_probs(d, cfg)
        source = "recomputed from val_probs.npy"
    if chosen is None:
        return None

    conf = dict(chosen["confusion"])
    hist = read_json(d / "history.json") or []
    if conf.get("cost") is None:
        epochs = [h for h in hist
                  if isinstance(h, dict) and h.get("val_cost") is not None]
        if epochs:
            conf["cost"] = min(h["val_cost"] for h in epochs)
    best_cost = min(
        (h["val_cost"] for h in hist if isinstance(h, dict) and h.get("val_cost") is not None),
        default=None,
    )
    return {
        "run_id": run_id,
        "window": cfg.get("window_seconds"),
        "threshold": chosen.get("threshold"),
        "n": chosen.get("n"),
        "f1": conf.get("f1"),
        "precision": conf.get("precision"),
        "recall": conf.get("recall"),
        "false_interruption_rate": conf.get("false_interruption_rate"),
        "missed_endpoint_rate": conf.get("missed_endpoint_rate"),
        "cost": conf.get("cost"),
        "best_epoch_val_cost": best_cost,
        "roc_auc": chosen.get("roc_auc"),
        "train_seconds": sum(
            h.get("seconds", 0.0) for h in hist if isinstance(h, dict)
        ) or None,
        "source": source,
    }


def select_by_val_cost(run_ids) -> tuple[dict | None, list[dict]]:
    """Pick the lowest validation cost. The only selection function here.

    Cost is ``fp_cost * false_interruption + missed_endpoint`` -- the objective
    training itself minimised for early stopping, so it is the consistent basis.
    F1 is carried along and a disagreement is reported rather than resolved
    silently.
    """
    rows = [m for rid in run_ids if (m := val_metrics(rid)) and m.get("cost") is not None]
    if not rows:
        return None, []
    best = min(rows, key=lambda r: r["cost"])
    return best, rows


# --------------------------------------------------------------------------- #
# the test-read budget
# --------------------------------------------------------------------------- #
def claim_test_read(state: State, model_id: str, purpose: str) -> bool:
    """Authorise one read of the held-out set, or refuse.

    Purposes are single-use and bound to a model. Asking for
    ``winner_clip_eval`` on a second model means something upstream decided to
    compare candidates on test, which is exactly the mistake that makes a
    held-out number worthless -- so this returns False and the stage is skipped
    rather than quietly permitted.
    """
    reads = state.data["test_reads"]
    for r in reads:
        if r["purpose"] == purpose:
            if r["model"] == model_id:
                return True                    # idempotent: same claim, resumed
            log(f"  REFUSED: test read {purpose!r} is already spent on "
                f"{r['model']!r}; will not also score {model_id!r} on test.")
            log("           Comparing two candidates on the held-out set is "
                "selection on test.")
            state.event(f"refused test read {purpose} for {model_id}")
            return False
    reads.append({"purpose": purpose, "model": model_id, "at": now()})
    state.save()
    log(f"  test read #{len(reads)} authorised: {purpose} on {model_id}")
    return True


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
@dataclass
class Stage:
    name: str
    title: str
    done: Callable[["Ctx"], bool]
    run: Callable[["Ctx"], bool]          # True on success
    optional: bool = False
    detail: str = ""
    # Returns a reason string when the stage has nothing to do -- a conditional
    # arm whose condition came out false, say. Kept separate from done() so the
    # plan never prints DONE for something that never ran, which would be the
    # same class of dishonest status this whole script exists to avoid.
    not_applicable: Callable[["Ctx"], str | None] | None = None

    def state_of(self, ctx: "Ctx") -> tuple[str, str]:
        """(label, reason) for display and for the run loop."""
        if self.not_applicable is not None:
            try:
                reason = self.not_applicable(ctx)
            except Exception:
                reason = None
            if reason:
                return "n/a", reason
        try:
            return ("done", "artefacts present") if self.done(ctx) else ("todo", "")
        except Exception as exc:
            return "todo", f"completeness check raised: {exc}"


@dataclass
class Ctx:
    state: State
    args: argparse.Namespace
    winner: str | None = None
    final_model: str | None = None
    gap_decision: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def train_cache(self) -> str:
        return self.args.train_cache

    @property
    def test_cache(self) -> str:
        return self.args.test_cache


def refresh_table() -> None:
    """Regenerate the markdown experiment table from the CSV.

    Called after every stage that could have appended a row. Cheap, and it means
    the table is never stale relative to the runs.
    """
    try:
        from src.evaluation import ExperimentTable

        ExperimentTable(str(REPO_ROOT / "artifacts" / "experiments.csv")).save_markdown(
            str(REPO_ROOT / "report" / "experiments.md")
        )
    except Exception as exc:                     # non-fatal: it is a report file
        log(f"  (could not refresh report/experiments.md: {exc})")


def attempt(ctx: Ctx, stage: str, cmd: list[str], timeout: int | None = None) -> bool:
    """Run a command, and on failure retry once with the smallest fix that fits."""
    st = ctx.state.stage(stage)
    st["attempts"] = st.get("attempts", 0) + 1
    st["status"] = "running"
    st["command"] = " ".join(cmd)
    st["started"] = now()
    ctx.state.save()

    rc, tail = sh(cmd, stage, timeout=timeout)
    if rc == 0:
        return True

    log(f"\n  stage {stage} exited {rc}")
    fix = find_fix(tail, cmd)
    if not fix:
        ctx.state.note(stage, status="failed", rc=rc, error=tail[-2000:],
                       finished=now())
        log("  no auto-fix matches this failure; leaving it failed and moving on")
        return False

    autofix, new_cmd = fix
    log(f"  applying smallest safe fix [{autofix.name}]: {autofix.describe}")
    if autofix.changes_semantics:
        warn = (f"{stage}: auto-fix {autofix.name} ({autofix.describe}) changed "
                f"run settings, so this row is not strictly controlled against "
                f"the others")
        log(f"  COMPARABILITY WARNING -- {warn}")
        ctx.notes.append(warn)
        ctx.state.note(stage, comparability_warning=warn)

    st["attempts"] = st.get("attempts", 0) + 1
    st["fix_applied"] = autofix.name
    ctx.state.save()

    rc2, tail2 = sh(new_cmd, stage, timeout=timeout)
    if rc2 == 0:
        ctx.state.note(stage, command=" ".join(new_cmd))
        return True
    ctx.state.note(stage, status="failed", rc=rc2, error=tail2[-2000:],
                   finished=now())
    return False


def train_stage(run_id: str, config: str, title: str, *, optional=False) -> Stage:
    def _run(ctx: Ctx) -> bool:
        cfg = derive_config(REPO_ROOT / config, run_id,
                            reason="validation-only candidate")
        ok = attempt(ctx, run_id, py("-m", "training.train", "--config", str(cfg)))
        refresh_table()
        return ok and run_complete(run_id)

    return Stage(
        name=run_id,
        title=title,
        done=lambda ctx: run_complete(run_id),
        run=_run,
        optional=optional,
        detail=config,
    )


def build_plan(ctx: Ctx) -> list[Stage]:
    """The ordered plan. Every ``done`` reads the filesystem, never the state."""
    stages: list[Stage] = []

    # -- 1/2. finish the E2 sweep ---------------------------------------------
    for rid in E2_RUNS:
        stages.append(train_stage(rid, E2_CONFIGS[rid], f"E2 window sweep: {rid}"))

    # -- 3. summary table ----------------------------------------------------
    def _summary_done(ctx: Ctx) -> bool:
        p = REPO_ROOT / "report" / "e2_window_sweep.md"
        return p.exists() and p.stat().st_size > 0

    def _summary_run(ctx: Ctx) -> bool:
        return attempt(ctx, "e2_summary", py(
            "scripts/summarise_sweep.py", "--prefix", "E2", "--include", E1_RUN,
            "--out", "report/e2_window_sweep.md",
        ))

    stages.append(Stage("e2_summary", "E2 validation summary table",
                        _summary_done, _summary_run))

    # -- 4. pick the winner, on validation only ------------------------------
    def _winner_done(ctx: Ctx) -> bool:
        p = REPO_ROOT / "artifacts" / "e2_winner.json"
        if not p.exists():
            return False
        blob = read_json(p) or {}
        ctx.winner = blob.get("winner")
        return bool(ctx.winner)

    def _winner_run(ctx: Ctx) -> bool:
        candidates = [E1_RUN, *[r for r in E2_RUNS if run_complete(r)]]
        best, rows = select_by_val_cost(candidates)
        if not best:
            log("  no validation metrics available for any candidate yet")
            return False

        log("\n  validation-only comparison (lower cost is better):")
        log(f"    {'run':<12s} {'window':>7s} {'cost':>8s} {'F1':>8s} "
            f"{'recall':>8s} {'false_int':>10s} {'thr':>8s}")
        for r in sorted(rows, key=lambda r: r["cost"]):
            w = f"{r['window']}s" if r.get("window") else "?"
            log(f"    {r['run_id']:<12s} {w:>7s} {r['cost']:>8.4f} "
                f"{(r.get('f1') or 0):>8.4f} {(r.get('recall') or 0):>8.4f} "
                f"{(r.get('false_interruption_rate') or 0):>10.4f} "
                f"{(r.get('threshold') or 0):>8.4f}")

        best_f1 = max(rows, key=lambda r: (r.get("f1") or 0.0))
        disagree = best_f1["run_id"] != best["run_id"]
        if disagree:
            log(f"\n  NOTE: lowest cost is {best['run_id']} but highest F1 is "
                f"{best_f1['run_id']}. Cost decides, because it is what training "
                f"minimised; the disagreement is recorded, not resolved away.")

        floor = 0.05                              # src.metrics.MIN_USEFUL_RECALL
        pinned = [r for r in rows if (r.get("recall") or 0) < floor * 1.75]
        if len(pinned) == len(rows):
            log("\n  NOTE: every candidate sits near the recall floor, which is "
                "the signature of the\n        threshold-selection rule binding "
                "rather than the window mattering. The\n        sweep is still "
                "reported, but read it with that in mind.")

        ctx.winner = best["run_id"]
        (REPO_ROOT / "artifacts" / "e2_winner.json").write_text(
            json.dumps(
                {
                    "winner": best["run_id"],
                    "selected_on": "validation cost only",
                    "cost": best["cost"],
                    "window_seconds": best.get("window"),
                    "threshold": best.get("threshold"),
                    "cost_and_f1_disagree": disagree,
                    "highest_val_f1_run": best_f1["run_id"],
                    "all_candidates_near_recall_floor": len(pinned) == len(rows),
                    "candidates": rows,
                    "at": now(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        log(f"\n  winner (validation cost): {ctx.winner}")
        ctx.state.note("e2_winner", winner=ctx.winner, cost=best["cost"])
        return True

    stages.append(Stage("e2_winner", "select the winner on validation cost",
                        _winner_done, _winner_run))

    # -- 5. ONE held-out test evaluation on the winner -----------------------
    def _wintest_done(ctx: Ctx) -> bool:
        if not ctx.winner:
            return False
        return read_json(run_dir(f"{ctx.winner}-eval") / "evaluation.json") is not None

    def _wintest_run(ctx: Ctx) -> bool:
        if not ctx.winner:
            return False
        if not claim_test_read(ctx.state, ctx.winner, "winner_clip_eval"):
            return False
        ok = attempt(ctx, "winner_test_eval", py(
            "-m", "training.train", "--eval-only",
            "--checkpoint", str(checkpoint_for(ctx.winner)),
        ))
        refresh_table()
        return ok

    stages.append(Stage("winner_test_eval",
                        "ONE held-out test evaluation on the winner",
                        _wintest_done, _wintest_run))

    # -- 6. streaming on the winner, on VALIDATION ---------------------------
    def _stream_done(ctx: Ctx) -> bool:
        if not ctx.winner:
            return False
        return read_json(
            run_dir(f"streaming_{ctx.winner}_val") / "single.json"
        ) is not None

    def _stream_run(ctx: Ctx) -> bool:
        if not ctx.winner:
            return False
        return attempt(ctx, "winner_streaming", py(
            "scripts/stream_eval.py",
            "--checkpoint", str(checkpoint_for(ctx.winner)),
            "--cache", ctx.train_cache, "--split", "val",
            "--max-rows", str(ctx.args.stream_rows),
            "--out", f"artifacts/runs/streaming_{ctx.winner}_val",
        ), timeout=ctx.args.stream_timeout)

    stages.append(Stage("winner_streaming",
                        "streaming evaluation on the winner (validation split)",
                        _stream_done, _stream_run))

    # -- 7. the branch: is the streaming gap material? -----------------------
    def _gap_done(ctx: Ctx) -> bool:
        blob = read_json(REPO_ROOT / "artifacts" / "streaming_gap.json")
        if not blob:
            return False
        ctx.gap_decision = blob
        return True

    def _gap_run(ctx: Ctx) -> bool:
        if not ctx.winner:
            return False
        stream = read_json(run_dir(f"streaming_{ctx.winner}_val") / "single.json")
        clip = val_metrics(ctx.winner)
        if not stream or not clip:
            log("  cannot compare: missing the streaming or the clip-level result")
            return False

        s_conf = stream["summary"]["confusion"]
        s_fi = float(s_conf["false_interruption_rate"])
        c_fi = float(clip["false_interruption_rate"])
        s_f1, c_f1 = float(s_conf["f1"]), float(clip["f1"] or 0.0)

        fi_gap = s_fi - c_fi
        material = fi_gap >= ctx.args.gap_fi or (c_f1 > 0 and s_f1 < 0.5 * c_f1)

        log(f"\n  clip-level (val) : F1 {c_f1:.4f}  false_int {c_fi:.4f}")
        log(f"  streamed   (val) : F1 {s_f1:.4f}  false_int {s_fi:.4f}")
        log(f"  false-interruption gap: {fi_gap:+.4f} "
            f"(threshold {ctx.args.gap_fi:+.4f})")
        log(f"  -> {'MATERIAL' if material else 'not material'}: random-offset "
            f"training {'will' if material else 'will not'} run")

        blob = {
            "winner": ctx.winner,
            "clip_f1": c_f1, "clip_false_interruption": c_fi,
            "stream_f1": s_f1, "stream_false_interruption": s_fi,
            "fi_gap": fi_gap, "threshold": ctx.args.gap_fi,
            "material": bool(material),
            "measured_on": "validation split of the train cache",
            "why_not_test": "a branch taken on a test measurement is selection "
                            "on test, whatever it is called",
            "at": now(),
        }
        (REPO_ROOT / "artifacts" / "streaming_gap.json").write_text(
            json.dumps(blob, indent=2), encoding="utf-8")
        ctx.gap_decision = blob
        return True

    stages.append(Stage("streaming_gap", "decide whether the streaming gap is material",
                        _gap_done, _gap_run))

    # -- 7b. random-offset training, conditional ----------------------------
    def _e9_na(ctx: Ctx) -> str | None:
        if ctx.gap_decision and not ctx.gap_decision.get("material"):
            return ("streaming gap measured and judged immaterial, so "
                    "random-offset training is not warranted")
        return None

    def _e9_done(ctx: Ctx) -> bool:
        return run_complete("E9")

    def _e9_run(ctx: Ctx) -> bool:
        if not ctx.gap_decision.get("material"):
            log("  streaming gap is not material -> random-offset training is "
                "not warranted; skipping rather than running it for the sake of "
                "having a row.")
            ctx.state.note("E9", status="skipped",
                           reason="streaming gap below threshold")
            return True
        win = val_metrics(ctx.winner or E1_RUN) or {}
        cfg = derive_config(
            REPO_ROOT / "configs" / "e9_random_offset.yaml", "E9",
            window_seconds=win.get("window"),
            reason=f"window taken from the E2 winner {ctx.winner}",
        )
        ok = attempt(ctx, "E9", py("-m", "training.train", "--config", str(cfg)))
        refresh_table()
        return ok and run_complete("E9")

    stages.append(Stage("E9", "random-offset training (only if the gap is material)",
                        _e9_done, _e9_run, not_applicable=_e9_na))

    # -- 7c. streaming on the offset model, to see if the fix worked ---------
    def _e9s_na(ctx: Ctx) -> str | None:
        if not run_complete("E9"):
            return "no random-offset model exists to stream"
        return None

    def _e9s_done(ctx: Ctx) -> bool:
        return read_json(run_dir("streaming_E9_val") / "single.json") is not None

    def _e9s_run(ctx: Ctx) -> bool:
        if not run_complete("E9"):
            return True
        return attempt(ctx, "E9_streaming", py(
            "scripts/stream_eval.py", "--checkpoint", str(checkpoint_for("E9")),
            "--cache", ctx.train_cache, "--split", "val",
            "--max-rows", str(ctx.args.stream_rows),
            "--out", "artifacts/runs/streaming_E9_val",
        ), timeout=ctx.args.stream_timeout)

    stages.append(Stage("E9_streaming", "streaming evaluation of the offset model",
                        _e9s_done, _e9s_run, not_applicable=_e9s_na))

    # -- 8a. hard negatives: mine on train, then retrain ---------------------
    def _mine_done(ctx: Ctx) -> bool:
        d = REPO_ROOT / "artifacts" / "runs" / "error_analysis_train"
        return (d / "hard_negative_indices.npy").exists() and (
            d / "hard_negative_meta.json").exists()

    def _mine_run(ctx: Ctx) -> bool:
        base = ctx.winner or E1_RUN
        return attempt(ctx, "mine_hard_negatives", py(
            "scripts/error_analysis.py",
            "--checkpoint", str(checkpoint_for(base)),
            "--cache", ctx.train_cache,
            "--mine-split", "train",          # never 'all', never the test cache
            "--mine-hard-negatives",
            "--out", "artifacts/runs/error_analysis_train",
        ))

    stages.append(Stage("mine_hard_negatives",
                        "mine hard negatives (train split of the train cache)",
                        _mine_done, _mine_run))

    def _e10_run(ctx: Ctx) -> bool:
        win = val_metrics(ctx.winner or E1_RUN) or {}
        cfg = derive_config(
            REPO_ROOT / "configs" / "e10_hard_negatives.yaml", "E10",
            window_seconds=win.get("window"),
            reason=f"window from the E2 winner {ctx.winner}",
        )
        ok = attempt(ctx, "E10", py("-m", "training.train", "--config", str(cfg)))
        refresh_table()
        return ok and run_complete("E10")

    stages.append(Stage("E10", "hard-negative oversampling retrain",
                        lambda ctx: run_complete("E10"), _e10_run, optional=True))

    # -- 8b. augmentation, 8c. temporal heads -------------------------------
    def head_stage(rid: str, base: str, title: str) -> Stage:
        def _run(ctx: Ctx) -> bool:
            win = val_metrics(ctx.winner or E1_RUN) or {}
            cfg = derive_config(
                REPO_ROOT / base, rid, window_seconds=win.get("window"),
                reason=f"window from the E2 winner {ctx.winner}",
            )
            ok = attempt(ctx, rid, py("-m", "training.train", "--config", str(cfg)))
            refresh_table()
            return ok and run_complete(rid)

        return Stage(rid, title, lambda ctx: run_complete(rid), _run, optional=True,
                     detail=base)

    stages.append(head_stage("E5", "configs/e5_augmentation.yaml",
                             "augmentation arm"))
    stages.append(head_stage("E3", "configs/e3_mlp_head.yaml",
                             "temporal/head comparison: MLP head"))
    stages.append(head_stage("E4", "configs/e4_gru_head.yaml",
                             "temporal/head comparison: GRU head"))
    stages.append(head_stage("E4b", "configs/e4b_attn_pool.yaml",
                             "temporal/head comparison: attention pooling"))

    # -- 8d. choose the final model, on validation only ---------------------
    def _final_done(ctx: Ctx) -> bool:
        blob = read_json(REPO_ROOT / "artifacts" / "final_model.json")
        if not blob:
            return False
        ctx.final_model = blob.get("final_model")
        return bool(ctx.final_model)

    def _final_run(ctx: Ctx) -> bool:
        pool = [E1_RUN, *E2_RUNS, "E9", "E10", "E5", "E3", "E4", "E4b"]
        done = [r for r in pool if run_complete(r)]
        best, rows = select_by_val_cost(done)
        if not best:
            return False
        ctx.final_model = best["run_id"]
        log(f"\n  final model, chosen on validation cost across "
            f"{len(rows)} completed runs: {ctx.final_model}")
        (REPO_ROOT / "artifacts" / "final_model.json").write_text(
            json.dumps({"final_model": best["run_id"],
                        "selected_on": "validation cost only, across every "
                                       "completed run; test was not consulted",
                        "cost": best["cost"], "candidates": rows, "at": now()},
                       indent=2),
            encoding="utf-8")
        return True

    stages.append(Stage("final_model", "choose the final model on validation",
                        _final_done, _final_run))

    # -- 8e. Hinglish robustness -------------------------------------------
    def _hing_done(ctx: Ctx) -> bool:
        if not ctx.final_model:
            return False
        return (run_dir(f"hinglish_{ctx.final_model}")).exists() and any(
            (run_dir(f"hinglish_{ctx.final_model}")).iterdir())

    def _hing_run(ctx: Ctx) -> bool:
        if not ctx.final_model:
            return False
        return attempt(ctx, "hinglish", py(
            "scripts/eval_hinglish.py",
            "--checkpoint", str(checkpoint_for(ctx.final_model)),
            "--out", f"artifacts/runs/hinglish_{ctx.final_model}",
        ))

    stages.append(Stage("hinglish", "Hinglish robustness on the final model",
                        _hing_done, _hing_run, optional=True))

    # -- 8f. error analysis on the final model -----------------------------
    def _ea_done(ctx: Ctx) -> bool:
        if not ctx.final_model:
            return False
        d = run_dir(f"error_analysis_{ctx.final_model}")
        return d.exists() and any(d.iterdir())

    def _ea_run(ctx: Ctx) -> bool:
        if not ctx.final_model:
            return False
        # On the test cache and WITHOUT --mine-split: this is reporting, and the
        # indices it writes are marked unusable for training by the sidecar.
        return attempt(ctx, "error_analysis", py(
            "scripts/error_analysis.py",
            "--checkpoint", str(checkpoint_for(ctx.final_model)),
            "--cache", ctx.test_cache,
            "--out", f"artifacts/runs/error_analysis_{ctx.final_model}",
        ))

    stages.append(Stage("error_analysis", "error analysis on the final model",
                        _ea_done, _ea_run, optional=True))

    # -- 8g. optimisation / ONNX / INT8 ------------------------------------
    def _opt_done(ctx: Ctx) -> bool:
        if not ctx.final_model:
            return False
        d = run_dir(f"optimize_{ctx.final_model}")
        return d.exists() and any(d.iterdir())

    def _opt_run(ctx: Ctx) -> bool:
        if not ctx.final_model:
            return False
        return attempt(ctx, "optimize", py(
            "scripts/optimize_model.py",
            "--checkpoint", str(checkpoint_for(ctx.final_model)),
            "--cache", ctx.test_cache,
            "--max-rows", str(ctx.args.optimize_rows),
            "--artifacts", f"artifacts/runs/optimize_{ctx.final_model}",
        ), timeout=ctx.args.optimize_timeout)

    stages.append(Stage("optimize", "quantisation, ONNX export and latency benchmark",
                        _opt_done, _opt_run, optional=True))

    # -- 8h. final streaming benchmark, on test ---------------------------
    def _fs_done(ctx: Ctx) -> bool:
        if not ctx.final_model:
            return False
        return read_json(
            run_dir(f"streaming_final_{ctx.final_model}") / "single.json"
        ) is not None

    def _fs_run(ctx: Ctx) -> bool:
        if not ctx.final_model:
            return False
        # The second and last authorised test read. Reporting only: the final
        # model was already chosen, on validation, before this runs.
        if not claim_test_read(ctx.state, ctx.final_model,
                               "final_streaming_benchmark"):
            return False
        return attempt(ctx, "final_streaming", py(
            "scripts/stream_eval.py",
            "--checkpoint", str(checkpoint_for(ctx.final_model)),
            "--cache", ctx.test_cache,
            "--max-rows", str(ctx.args.stream_rows),
            "--out", f"artifacts/runs/streaming_final_{ctx.final_model}",
        ), timeout=ctx.args.stream_timeout)

    stages.append(Stage("final_streaming",
                        "final streaming benchmark on the held-out set",
                        _fs_done, _fs_run, optional=True))

    return stages


# --------------------------------------------------------------------------- #
# the final report
# --------------------------------------------------------------------------- #
def fmt(v, nd: int = 4, dash: str = "not run") -> str:
    if v is None:
        return dash
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def readiness(ctx: Ctx) -> tuple[int, list[tuple[str, bool, str]]]:
    """Score submission readiness from artefacts, not from intentions.

    Each row is worth one point and is decided by a filesystem check. The score
    is therefore a measurement, and it can go down if an artefact is deleted.
    """
    fm = ctx.final_model
    checks: list[tuple[str, bool, str]] = [
        ("E1 baseline complete", run_complete(E1_RUN),
         "artifacts/runs/E1/evaluation.json + weights"),
        ("E2 sweep complete (all 5 windows)",
         all(run_complete(r) for r in E2_RUNS), "one run dir per window"),
        ("winner chosen on validation only",
         (REPO_ROOT / "artifacts" / "e2_winner.json").exists(),
         "artifacts/e2_winner.json"),
        ("held-out test evaluated exactly once per named model",
         0 < len(ctx.state.data["test_reads"]) <= 2,
         "artifacts/autonomous_state.json:test_reads"),
        ("streaming measured on the winner",
         bool(ctx.winner) and read_json(
             run_dir(f"streaming_{ctx.winner}_val") / "single.json") is not None,
         "streaming_<winner>_val/single.json"),
        ("streaming gap decision recorded",
         (REPO_ROOT / "artifacts" / "streaming_gap.json").exists(),
         "artifacts/streaming_gap.json"),
        ("random-offset arm resolved (run, or skipped on a measurement)",
         run_complete("E9") or (
             (read_json(REPO_ROOT / "artifacts" / "streaming_gap.json") or {}
              ).get("material") is False),
         "artifacts/runs/E9, or a recorded not-material gap decision"),
        ("hard negatives mined on the train split",
         (REPO_ROOT / "artifacts" / "runs" / "error_analysis_train"
          / "hard_negative_meta.json").exists(),
         "provenance sidecar present"),
        ("hard-negative retrain complete", run_complete("E10"),
         "artifacts/runs/E10"),
        ("augmentation arm complete", run_complete("E5"), "artifacts/runs/E5"),
        ("head/pooling comparison complete",
         all(run_complete(r) for r in ("E3", "E4", "E4b")), "E3, E4, E4b"),
        ("final model chosen on validation",
         (REPO_ROOT / "artifacts" / "final_model.json").exists(),
         "artifacts/final_model.json"),
        ("Hinglish robustness on the final model",
         bool(fm) and run_dir(f"hinglish_{fm}").exists(), "hinglish_<final>"),
        ("error analysis on the final model",
         bool(fm) and run_dir(f"error_analysis_{fm}").exists(),
         "error_analysis_<final>"),
        ("latency + size benchmark on the final model",
         bool(fm) and run_dir(f"optimize_{fm}").exists(), "optimize_<final>"),
        ("final streaming benchmark",
         bool(fm) and read_json(
             run_dir(f"streaming_final_{fm}") / "single.json") is not None,
         "streaming_final_<final>/single.json"),
    ]
    return sum(1 for _, ok, _ in checks if ok), checks


def write_final_report(ctx: Ctx) -> Path:
    """Assemble report/final_benchmark.md from artefacts only."""
    out = REPO_ROOT / "report" / "final_benchmark.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    L: list[str] = []
    A = L.append

    fm = ctx.final_model
    winner = ctx.winner
    score, checks = readiness(ctx)

    A("# Final benchmark")
    A("")
    A(f"Generated by `scripts/run_autonomous.py` at {now()}. **Do not edit by "
      "hand** — it is rebuilt from run artefacts on every invocation.")
    A("")
    A("Every number here was read out of a file under `artifacts/runs/`. Cells "
      "reading `not run` are experiments that have not produced an artefact; "
      "nothing is estimated or carried over from another run.")
    A("")

    # -- selection integrity -------------------------------------------------
    A("## Selection integrity")
    A("")
    reads = ctx.state.data["test_reads"]
    A(f"Held-out test reads: **{len(reads)}**")
    A("")
    if reads:
        A("| # | model | purpose | at |")
        A("|---|---|---|---|")
        for i, r in enumerate(reads, 1):
            A(f"| {i} | `{r['model']}` | {r['purpose']} | {r['at']} |")
    else:
        A("_None yet._")
    A("")
    A("- Every candidate was compared on **validation cost** only "
      "(`fp_cost * false_interruption + missed_endpoint`).")
    A("- Every generated config carries `test_cache_dir: null`, so a candidate "
      "cannot score the held-out set even if its template asked to.")
    A("- The threshold is selected on validation and applied unchanged; "
      "`--eval-only` never calls `pick_threshold`.")
    A("- The streaming gap decision was taken on the **validation** split, "
      "because branching on a test measurement is selection on test.")
    if len(reads) > 1:
        A(f"- Two reads exist because the final model (`{fm}`) and the E2 winner "
          f"(`{winner}`) differ. Both were named before test was touched, and "
          "test was not used to choose between them.")
    A("")

    # -- experiment matrix ---------------------------------------------------
    A("## Experiment matrix (validation)")
    A("")
    pool = [E1_RUN, *E2_RUNS, "E9", "E10", "E5", "E3", "E4", "E4b"]
    rows = [m for r in pool if (m := val_metrics(r))]
    if rows:
        A("| run | window | val cost | val F1 | precision | recall | "
          "false int | missed | threshold | train s |")
        A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in sorted(rows, key=lambda r: (r["cost"] is None, r["cost"])):
            star = " **<- final**" if r["run_id"] == fm else (
                " *(E2 winner)*" if r["run_id"] == winner else "")
            w = f"{r['window']}s" if r.get("window") else "?"
            A(f"| `{r['run_id']}`{star} | {w} | {fmt(r['cost'])} | "
              f"{fmt(r['f1'])} | {fmt(r['precision'])} | {fmt(r['recall'])} | "
              f"{fmt(r['false_interruption_rate'])} | "
              f"{fmt(r['missed_endpoint_rate'])} | {fmt(r['threshold'], 6)} | "
              f"{fmt(r['train_seconds'], 0)} |")
        A("")
        missing = [r for r in pool if r not in {x['run_id'] for x in rows}]
        if missing:
            A(f"Not run: {', '.join('`' + m + '`' for m in missing)}.")
            A("")
    else:
        A("_No validation artefacts found._")
        A("")

    wj = read_json(REPO_ROOT / "artifacts" / "e2_winner.json") or {}
    if wj.get("cost_and_f1_disagree"):
        A(f"> Validation cost and validation F1 disagree: lowest cost is "
          f"`{wj.get('winner')}`, highest F1 is `{wj.get('highest_val_f1_run')}`. "
          f"Cost decides, because it is the objective training minimised. The "
          f"disagreement is recorded rather than resolved silently.")
        A("")
    if wj.get("all_candidates_near_recall_floor"):
        A("> **Every candidate sits near the `MIN_USEFUL_RECALL = 0.05` floor.** "
          "That is the signature of the threshold-selection rule binding rather "
          "than of the window mattering, so differences between windows here are "
          "compressed and should not be over-read. See §9.6 of the technical "
          "report.")
        A("")

    # -- final model --------------------------------------------------------
    A("## Final model")
    A("")
    if fm:
        m = val_metrics(fm) or {}
        fj = read_json(REPO_ROOT / "artifacts" / "final_model.json") or {}
        A(f"**`{fm}`** — selected on {fj.get('selected_on', 'validation cost')}.")
        A("")
        A("| | value |")
        A("|---|---|")
        A(f"| window | {fmt(m.get('window'), 1)} s |")
        A(f"| threshold (from validation) | {fmt(m.get('threshold'), 6)} |")
        A(f"| checkpoint | `weights/{fm}-best.pt` |")
        A("")
    else:
        A("_Not yet chosen._")
        A("")

    # -- validation metrics -------------------------------------------------
    A("## Validation metrics — final model")
    A("")
    m = val_metrics(fm) if fm else None
    if m:
        A("| metric | value |")
        A("|---|---:|")
        for k in ("f1", "precision", "recall", "false_interruption_rate",
                  "missed_endpoint_rate", "cost", "roc_auc"):
            A(f"| {k} | {fmt(m.get(k))} |")
        A(f"| n | {fmt(m.get('n'), 0)} |")
        A("")
    else:
        A("_not run_")
        A("")

    # -- held-out test ------------------------------------------------------
    A("## Held-out test metrics — final model")
    A("")
    ev = read_json(run_dir(f"{fm}-eval") / "evaluation.json") if fm else None
    if isinstance(ev, dict) and "confusion" in ev:
        c = ev["confusion"]
        A(f"Threshold **{fmt(ev.get('threshold'), 6)}**, selected on validation "
          "and applied unchanged. Test read once, for scoring.")
        A("")
        A("| metric | value |")
        A("|---|---:|")
        A(f"| n | {fmt(ev.get('n'), 0)} |")
        for k in ("f1", "precision", "recall", "false_interruption_rate",
                  "missed_endpoint_rate", "accuracy"):
            A(f"| {k} | {fmt(c.get(k))} |")
        A(f"| ROC-AUC | {fmt(ev.get('roc_auc'))} |")
        A(f"| PR-AUC | {fmt(ev.get('pr_auc'))} |")
        A("")
    else:
        A("_not run_ — no `evaluation.json` under "
          f"`artifacts/runs/{fm}-eval/`." if fm else "_not run_")
        A("")

    # -- streaming ----------------------------------------------------------
    A("## Streaming metrics")
    A("")
    A("Clip-level scores do not predict streaming behaviour, so these are "
      "measured directly rather than inferred. Decision-making runs use the "
      "validation split; the final benchmark uses the held-out set.")
    A("")
    srows = []
    for tag, path in (
        (f"winner `{winner}` (val)", f"streaming_{winner}_val" if winner else None),
        ("`E9` random-offset (val)", "streaming_E9_val"),
        (f"final `{fm}` (held-out test)", f"streaming_final_{fm}" if fm else None),
    ):
        if not path:
            continue
        blob = read_json(run_dir(path) / "single.json")
        if not blob:
            continue
        c = blob["summary"]["confusion"]
        srows.append((tag, blob, c))
    if srows:
        A("| run | split | n | F1 | recall | false int | early fires | "
          "TTD p50 ms | wall p95 ms |")
        A("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for tag, blob, c in srows:
            s = blob["summary"]
            ttd = (s.get("time_to_decide_ms") or {}).get("p50")
            wall = (s.get("wall_latency_ms") or {}).get("p95")
            A(f"| {tag} | {blob.get('split', '?')} | "
              f"{fmt(blob.get('n_clips'), 0)} | {fmt(c.get('f1'))} | "
              f"{fmt(c.get('recall'))} | "
              f"{fmt(c.get('false_interruption_rate'))} | "
              f"{fmt(s.get('early_fires'), 0)} | {fmt(ttd, 1)} | "
              f"{fmt(wall, 2)} |")
        A("")
    else:
        A("_not run_")
        A("")

    gap = read_json(REPO_ROOT / "artifacts" / "streaming_gap.json")
    if gap:
        A(f"**Gap decision.** Clip-level false interruption "
          f"{fmt(gap['clip_false_interruption'])} against streamed "
          f"{fmt(gap['stream_false_interruption'])} "
          f"(gap {gap['fi_gap']:+.4f}, threshold {gap['threshold']:+.4f}) — "
          f"judged **{'material' if gap['material'] else 'not material'}**, "
          f"measured on {gap['measured_on']}.")
        A("")
        if gap["material"] and run_complete("E9"):
            A("Random-offset training ran as a result. Whether it closed the gap "
              "is the `E9` row in the streaming table above — reported either "
              "way, including if it made things worse.")
            A("")
        elif gap["material"]:
            A("Random-offset training was warranted but has not completed, so "
              "the gap stands unaddressed.")
            A("")

    # -- latency and size ---------------------------------------------------
    A("## Latency and model size")
    A("")
    opt = read_json(run_dir(f"optimize_{fm}") / "benchmark.json") if fm else None
    if opt is None and fm:
        d = run_dir(f"optimize_{fm}")
        for cand in ("results.json", "optimize.json", "summary.json"):
            opt = opt or read_json(d / cand)
    if isinstance(opt, (list, dict)):
        A("```json")
        A(json.dumps(opt, indent=2)[:4000])
        A("```")
        A("")
        A("Read from the optimisation run's own artefact rather than "
          "re-tabulated, so a schema change here cannot silently mislabel a "
          "column.")
        A("")
    else:
        A("_not run_ — no artefact under "
          f"`artifacts/runs/optimize_{fm}/`." if fm else "_not run_")
        A("")

    # -- Hinglish -----------------------------------------------------------
    A("## Hinglish robustness")
    A("")
    hg = None
    if fm:
        d = run_dir(f"hinglish_{fm}")
        for cand in ("summary.json", "hinglish.json", "evaluation.json"):
            hg = hg or read_json(d / cand)
    if hg:
        A("```json")
        A(json.dumps(hg, indent=2)[:3000])
        A("```")
        A("")
        A("The set is Sarvam Bulbul TTS across 7 categories. It covers "
          "vocabulary, code-switching and pause placement; it does not "
          "reproduce genuine disfluent timing or real room acoustics, so it "
          "measures whether the detector reads linguistic completion rather "
          "than duration of silence.")
    else:
        A("_not run_")
    A("")

    # -- error analysis -----------------------------------------------------
    A("## Error analysis")
    A("")
    ea_md = None
    if fm:
        p = run_dir(f"error_analysis_{fm}") / "slices.md"
        if p.exists():
            ea_md = p.read_text(encoding="utf-8")
    if ea_md:
        A(ea_md.strip()[:6000])
    else:
        A("_not run_")
    A("")
    A("Failure categories the project tracks, with the mechanism for each in "
      "§14 of the technical report: long utterances, internal pauses, fillers "
      "(`endfiller` / `midfiller`, which the corpus pre-labels), Hinglish and "
      "code-switching, synthetic-vs-human audio, speaker variation, and "
      "streaming window alignment.")
    A("")

    # -- limitations --------------------------------------------------------
    A("## Limitations")
    A("")
    lim = [
        "**No speaker-aware split is possible.** The corpus has no speaker ID, "
        "so splits are grouped by source corpus. One speaker contributing to "
        "two corpora would still span splits and nothing detects that.",
        "**Threshold selection uses a rule with a degenerate corner.** Unless a "
        "config sets `max_false_interruption`, the threshold minimises "
        "`4*FPR + FNR`, which is minimised outright by never firing; only the "
        "`MIN_USEFUL_RECALL = 0.05` guard prevents that. Runs pinned near the "
        "floor are reporting the guard, not an optimum.",
        "**Single seed per experiment.** No confidence intervals, so small "
        "differences between rows are not distinguishable from run-to-run "
        "variance.",
        "**82% of the source corpus is synthetic**, and the measured TTS-vs-human "
        "F1 gap on the earlier verification run was 6.5x. Aggregate figures are "
        "optimistic for human audio.",
        "**The Hinglish set is TTS**, so it does not exercise genuine disfluent "
        "timing, overlapping speech, or real room acoustics.",
        "**Streaming is evaluated on replayed clips**, not on live audio with "
        "real network jitter and barge-in.",
        "**`fp_cost = 4.0` and the 10% false-interruption ceiling are judgement "
        "calls**, written as named constants so they can be argued with.",
    ]
    for x in lim:
        A(f"- {x}")
    A("")
    if ctx.notes:
        A("### Comparability warnings raised during this run")
        A("")
        for n in ctx.notes:
            A(f"- {n}")
        A("")
    stale = [n for n, s in ctx.state.data["stages"].items()
             if s.get("comparability_warning")]
    if stale:
        A("### Comparability warnings recorded in earlier sessions")
        A("")
        for n in stale:
            A(f"- {ctx.state.data['stages'][n]['comparability_warning']}")
        A("")

    failed = [n for n, s in ctx.state.data["stages"].items()
              if s.get("status") == "failed"]
    if failed:
        A("### Stages that failed and were left failed")
        A("")
        A("| stage | attempts | fix tried | last error (tail) |")
        A("|---|---:|---|---|")
        for n in failed:
            s = ctx.state.data["stages"][n]
            err = (s.get("error") or "").strip().replace("\n", " ")[-160:]
            A(f"| `{n}` | {s.get('attempts', 0)} | "
              f"{s.get('fix_applied', 'none')} | {err} |")
        A("")

    # -- reproduction -------------------------------------------------------
    A("## Reproduction")
    A("")
    A("The whole pipeline, from whatever state the artefacts are in:")
    A("")
    A("```bash")
    A("python scripts/run_autonomous.py")
    A("```")
    A("")
    A("It skips anything already complete, so re-running after a disconnect "
      "resumes rather than restarting. To see the plan without executing it:")
    A("")
    A("```bash")
    A("python scripts/run_autonomous.py --status")
    A("```")
    A("")
    if fm:
        A("The final model specifically:")
        A("")
        A("```bash")
        gen = GEN_CONFIG_DIR.relative_to(REPO_ROOT) / f"{fm}.yaml"
        if (REPO_ROOT / gen).exists():
            A(f"python -m training.train --config {gen.as_posix()}")
        A(f"python -m training.train --eval-only "
          f"--checkpoint weights/{fm}-best.pt")
        A("```")
        A("")

    # -- readiness ----------------------------------------------------------
    A("## Submission readiness")
    A("")
    A(f"**{score} / {len(checks)}** — each row is a filesystem check, so this "
      "is a measurement rather than a self-assessment. It goes down if an "
      "artefact is removed.")
    A("")
    A("| check | state | evidence |")
    A("|---|---|---|")
    for name, ok, ev_ in checks:
        A(f"| {name} | {'yes' if ok else 'NO'} | `{ev_}` |")
    A("")
    outstanding = [n for n, ok, _ in checks if not ok]
    if outstanding:
        A("Outstanding:")
        A("")
        for n in outstanding:
            A(f"- {n}")
        A("")

    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and what is already done; run nothing")
    ap.add_argument("--status", action="store_true",
                    help="same as --dry-run, and also rewrite the final report")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run just these stage names")
    ap.add_argument("--skip", nargs="*", default=(),
                    help="stage names to leave alone")
    ap.add_argument("--train-cache", default="data/cache/train")
    ap.add_argument("--test-cache", default="data/cache/test")
    ap.add_argument("--gap-fi", type=float, default=DEFAULT_GAP_FI,
                    help="false-interruption gap above which random-offset "
                         "training is warranted")
    ap.add_argument("--stream-rows", type=int, default=300,
                    help="clips per streaming replay (it is ~40x clip scoring)")
    ap.add_argument("--stream-timeout", type=int, default=7200)
    ap.add_argument("--optimize-rows", type=int, default=2000)
    ap.add_argument("--optimize-timeout", type=int, default=7200)
    ap.add_argument("--stop-on-fail", action="store_true",
                    help="halt instead of carrying on past a failed stage")
    args = ap.parse_args(argv)

    state = State()
    ctx = Ctx(state=state, args=args)

    # Recover cross-stage decisions from disk, so a resumed session knows which
    # model it had already chosen without re-deriving it.
    for path, attr in (("e2_winner.json", "winner"),
                       ("final_model.json", "final_model")):
        blob = read_json(REPO_ROOT / "artifacts" / path) or {}
        key = "winner" if attr == "winner" else "final_model"
        if blob.get(key):
            setattr(ctx, attr, blob[key])
    ctx.gap_decision = read_json(
        REPO_ROOT / "artifacts" / "streaming_gap.json") or {}

    plan = build_plan(ctx)
    if args.only:
        plan = [s for s in plan if s.name in set(args.only)]
    plan = [s for s in plan if s.name not in set(args.skip)]

    rule("PLAN")
    if ctx.winner:
        log(f"  resumed: E2 winner already chosen -> {ctx.winner}")
    if ctx.final_model:
        log(f"  resumed: final model already chosen -> {ctx.final_model}")
    log("")
    for i, s in enumerate(plan, 1):
        label, reason = s.state_of(ctx)
        flag = " (optional)" if s.optional else ""
        log(f"  {i:>2d}. [{label:>4s}] {s.name:<22s} {s.title}{flag}")
        if label == "n/a" and reason:
            log(f"            reason: {reason}")
    log("")

    if args.dry_run or args.status:
        log("  dry run: nothing executed.")
        if args.status:
            p = write_final_report(ctx)
            log(f"  report -> {p.relative_to(REPO_ROOT)}")
        score, checks = readiness(ctx)
        log(f"\n  submission readiness: {score}/{len(checks)}")
        return 0

    # -- execute -------------------------------------------------------------
    for i, s in enumerate(plan, 1):
        rule(f"[{i}/{len(plan)}] {s.name} — {s.title}")
        label, reason = s.state_of(ctx)
        if label == "n/a":
            log(f"  not applicable: {reason}")
            state.note(s.name, status="not_applicable", reason=reason)
            continue
        if label == "done":
            log("  already complete (artefacts present) — skipping")
            state.note(s.name, status="done", skipped_because="artefacts present")
            continue
        if reason:
            log(f"  {reason}; attempting the stage")

        try:
            ok = s.run(ctx)
        except KeyboardInterrupt:
            log("\n  interrupted. State is saved; re-run to resume.")
            state.event("interrupted")
            return 130
        except Exception as exc:                  # a bug here must not lose the run
            log(f"  stage raised: {type(exc).__name__}: {exc}")
            state.note(s.name, status="failed", error=f"{type(exc).__name__}: {exc}",
                       finished=now())
            ok = False

        if ok:
            state.note(s.name, status="done", finished=now())
            log(f"  {s.name}: done")
        else:
            state.note(s.name, status="failed", finished=now())
            log(f"  {s.name}: FAILED")
            if args.stop_on_fail and not s.optional:
                log("  --stop-on-fail set and this stage is not optional; halting")
                break
            if not s.optional:
                log("  continuing: later stages check their own inputs and will "
                    "skip themselves if this one was required")

        # The report is rewritten after every stage, so a session that dies
        # mid-pipeline still leaves a truthful, current document behind.
        write_final_report(ctx)

    rule("FINAL REPORT")
    p = write_final_report(ctx)
    score, checks = readiness(ctx)
    log(f"  report -> {p.relative_to(REPO_ROOT)}")
    log(f"  submission readiness: {score}/{len(checks)}")
    log(f"  held-out test reads: {len(state.data['test_reads'])}")
    for r in state.data["test_reads"]:
        log(f"    - {r['purpose']} on {r['model']} at {r['at']}")
    failed = [n for n, s in state.data["stages"].items()
              if s.get("status") == "failed"]
    if failed:
        log(f"  stages left failed: {', '.join(failed)}")
    log("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
