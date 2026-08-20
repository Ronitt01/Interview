"""Train one configuration. Seeded, checkpointed, and it fills its own table row.

Run it as:

    python -m training.train --config configs/e1_frozen_linear.yaml

Design points that exist because of the free-T4 constraint:

* **Checkpoint every epoch, unconditionally.** A free Colab session disconnects
  without warning and an uncheckpointed run is a lost day. The checkpoint goes
  to ``--checkpoint-dir``, which on Colab should be a Drive path.
* **Resume is automatic.** Re-running the same command after a disconnect picks
  up from ``last.pt`` rather than starting over. That behaviour is on by default
  because the failure it protects against is the common case here, not the rare one.
* **Validation picks the threshold, test never does.** The operating point is
  selected on validation and then *applied* to the test set. Selecting it on test
  would report the best case for a threshold nobody could have known in advance,
  which is the most common way a submission like this quietly overstates itself.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.audio import WindowSpec  # noqa: E402
from src.augment import AugmentConfig  # noqa: E402
from src.dataset import TurnDataset, WaveCache, make_loader, split_indices  # noqa: E402
from src.evaluation import ExperimentTable, slice_report, slices_to_markdown  # noqa: E402
from src.inference import save_checkpoint  # noqa: E402
from src.metrics import DEFAULT_FP_COST, confusion_at, evaluate, pick_threshold  # noqa: E402
from src.model import ModelConfig, build_model  # noqa: E402


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    run_id: str = "E1"
    notes: str = ""
    cache_dir: str = "data/cache/train"
    test_cache_dir: str | None = "data/cache/test"

    # model
    window_seconds: float = 8.0
    head: str = "linear"
    pool: str = "mean"
    hidden_size: int = 256
    dropout: float = 0.1
    freeze_encoder: bool = True
    unfreeze_top_blocks: int = 0
    backbone: str = "openai/whisper-tiny"

    # data
    val_fraction: float = 0.2
    group_keys: tuple[str, ...] = ("dataset",)
    normalise_mode: str = "peak"
    balanced_sampler: bool = False
    use_pos_weight: bool = True
    augment: dict = field(default_factory=dict)

    # optimisation
    epochs: int = 6
    batch_size: int = 32
    lr: float = 1e-3
    encoder_lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    grad_clip: float = 1.0
    num_workers: int = 0
    seed: int = 1234
    amp: bool = True
    max_train_rows: int | None = None
    max_val_rows: int | None = None

    # selection
    fp_cost: float = DEFAULT_FP_COST
    max_false_interruption: float | None = None
    early_stop_patience: int = 3

    # output
    checkpoint_dir: str = "weights"
    artifacts_dir: str = "artifacts/runs"
    table_path: str = "artifacts/experiments.csv"

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        unknown = set(raw) - set(known)
        if unknown:
            # Loud, because a silently-ignored typo'd key means the experiment
            # you think you ran is not the experiment that ran.
            raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
        if "group_keys" in known:
            known["group_keys"] = tuple(known["group_keys"])
        return cls(**known)

    def model_config(self) -> ModelConfig:
        return ModelConfig(
            backbone=self.backbone,
            window_seconds=self.window_seconds,
            head=self.head,
            pool=self.pool,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            freeze_encoder=self.freeze_encoder,
            unfreeze_top_blocks=self.unfreeze_top_blocks,
        )

    def augment_config(self) -> AugmentConfig:
        return AugmentConfig(**self.augment) if self.augment else AugmentConfig()

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["group_keys"] = list(self.group_keys)
        return d


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed every generator that can affect a result.

    ``torch.use_deterministic_algorithms`` is deliberately *not* set: it makes
    some cudnn kernels fall back to much slower paths, and on a free T4 the time
    cost is larger than the run-to-run variance it removes. Seeded-but-not-bitwise
    -deterministic is the honest description, and it is what the report says.
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# one epoch
# --------------------------------------------------------------------------- #
def train_epoch(model, loader, optimiser, scheduler, scaler, criterion, device, grad_clip):
    import torch

    model.train()
    total, n = 0.0, 0
    t0 = time.perf_counter()
    for step, (mel, label) in enumerate(loader):
        mel, label = mel.to(device, non_blocking=True), label.to(device, non_blocking=True)
        optimiser.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = criterion(model(mel), label)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)
            scaler.step(optimiser)
            scaler.update()
        else:
            loss = criterion(model(mel), label)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)
            optimiser.step()

        if scheduler is not None:
            scheduler.step()

        total += float(loss.detach()) * label.size(0)
        n += int(label.size(0))
        if step % 50 == 0:
            print(
                f"    step {step:>5d}/{len(loader)}  loss {total / max(n, 1):.4f}",
                flush=True,
            )
    return total / max(n, 1), time.perf_counter() - t0


def predict_split(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for mel, label in loader:
            logits = model(mel.to(device))
            probs.append(torch.sigmoid(logits).float().cpu().numpy())
            ys.append(label.numpy())
    return (
        np.concatenate(probs) if probs else np.zeros(0),
        np.concatenate(ys) if ys else np.zeros(0),
    )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="one tiny epoch on a few hundred rows, to prove the wiring")
    args = ap.parse_args(argv)

    cfg = TrainConfig.load(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    if args.dry_run:
        cfg.epochs = 1
        cfg.max_train_rows = cfg.max_train_rows or 256
        cfg.max_val_rows = cfg.max_val_rows or 128
        cfg.run_id = f"{cfg.run_id}-dry"

    print(f"\n=== {cfg.run_id} ===")
    print(f"  device={device}  seed={cfg.seed}")

    # -- data ------------------------------------------------------------- #
    cache = WaveCache(cfg.cache_dir)
    print(f"  train cache: {cache.summary()}")

    idx, split_rep = split_indices(
        cache,
        {"train": 1.0 - cfg.val_fraction, "val": cfg.val_fraction},
        group_keys=cfg.group_keys,
        seed=cfg.seed,
    )
    print(f"  {split_rep}")
    print(f"  NOTE: {split_rep.limitations()}")

    from src.splits import assert_no_leakage

    assert_no_leakage(
        {k: [cache.meta[int(i)] for i in v] for k, v in idx.items()},
        group_keys=cfg.group_keys,
    )
    print("  leakage assertion: passed")

    train_idx, val_idx = idx["train"], idx["val"]
    if cfg.max_train_rows:
        train_idx = train_idx[: cfg.max_train_rows]
    if cfg.max_val_rows:
        val_idx = val_idx[: cfg.max_val_rows]

    window = WindowSpec(cfg.window_seconds)
    aug = cfg.augment_config()
    train_ds = TurnDataset(cache, train_idx, window, cfg.normalise_mode, aug)
    val_ds = TurnDataset(cache, val_idx, window, cfg.normalise_mode, None)

    train_loader = make_loader(
        train_ds, cfg.batch_size, shuffle=not cfg.balanced_sampler,
        balanced=cfg.balanced_sampler, num_workers=cfg.num_workers, seed=cfg.seed,
    )
    val_loader = make_loader(val_ds, cfg.batch_size, num_workers=cfg.num_workers)

    # -- model ------------------------------------------------------------ #
    model = build_model(cfg.model_config()).to(device)
    print(f"  {model.describe()}")

    pos_weight = None
    if cfg.use_pos_weight and not cfg.balanced_sampler:
        pw = train_ds.pos_weight()
        if abs(pw - 1.0) > 0.05:
            pos_weight = torch.tensor([pw], device=device)
            print(f"  class imbalance: pos_weight={pw:.3f}")
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Two parameter groups: the head learns fast, an unfrozen encoder learns
    # slowly. One shared LR either cooks the pretrained features or starves the
    # head, and that shows up as "unfreezing made it worse" when the real cause
    # is the learning rate.
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    groups = [{"params": head_params, "lr": cfg.lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": cfg.encoder_lr})
        print(f"  encoder: {sum(p.numel() for p in enc_params):,d} trainable @ lr {cfg.encoder_lr}")
    optimiser = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)

    total_steps = max(1, len(train_loader) * cfg.epochs)
    # OneCycleLR needs enough steps to have distinct warmup and anneal phases;
    # below ~20 the phase boundaries collapse onto each other and it divides by
    # zero. A one-cycle schedule over a handful of steps would be meaningless
    # anyway, so a short run (a --dry-run, a tiny subset) gets a flat LR and says so.
    MIN_SCHEDULED_STEPS = 20
    scheduler = None
    if total_steps >= MIN_SCHEDULED_STEPS:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimiser,
            max_lr=[g["lr"] for g in groups],
            total_steps=total_steps,
            pct_start=min(max(cfg.warmup_frac, 0.02), 0.5),
            anneal_strategy="cos",
        )
    else:
        print(
            f"  scheduler: flat LR ({total_steps} steps < {MIN_SCHEDULED_STEPS}; "
            "one-cycle needs more to be meaningful)"
        )
    use_amp = cfg.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path = ckpt_dir / f"{cfg.run_id}-last.pt"
    best_path = ckpt_dir / f"{cfg.run_id}-best.pt"
    state_path = ckpt_dir / f"{cfg.run_id}-trainstate.json"

    start_epoch, best_cost, bad_epochs = 0, float("inf"), 0
    if last_path.exists() and not args.no_resume and state_path.exists():
        st = json.loads(state_path.read_text(encoding="utf-8"))
        blob = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(blob["state_dict"])
        start_epoch = int(st.get("epoch", 0))
        best_cost = float(st.get("best_cost", float("inf")))
        print(f"  resumed from epoch {start_epoch} (best cost {best_cost:.4f})")

    # -- loop ------------------------------------------------------------- #
    history = []
    for epoch in range(start_epoch, cfg.epochs):
        train_ds.set_epoch(epoch)
        print(f"\n  epoch {epoch + 1}/{cfg.epochs}")
        loss, secs = train_epoch(
            model, train_loader, optimiser, scheduler, scaler, criterion, device, cfg.grad_clip
        )
        probs, ys = predict_split(model, val_loader, device)
        thr, conf = pick_threshold(ys, probs, cfg.fp_cost, cfg.max_false_interruption)
        cost = conf.cost(cfg.fp_cost)
        print(
            f"    train loss {loss:.4f} ({secs:.0f}s)  |  val f1 {conf.f1:.4f}  "
            f"false_interrupt {conf.false_interruption_rate:.4f}  "
            f"missed {conf.missed_endpoint_rate:.4f}  cost {cost:.4f}  thr {thr:.3f}"
        )
        history.append(
            {"epoch": epoch + 1, "train_loss": loss, "val_f1": conf.f1,
             "val_cost": cost, "threshold": thr, "seconds": secs}
        )

        save_checkpoint(last_path, model, thr, {"epoch": epoch + 1, "config": cfg.to_dict()})
        state_path.write_text(
            json.dumps({"epoch": epoch + 1, "best_cost": min(best_cost, cost)}, indent=2),
            encoding="utf-8",
        )

        if cost < best_cost - 1e-5:
            best_cost, bad_epochs = cost, 0
            save_checkpoint(
                best_path, model, thr,
                {"epoch": epoch + 1, "config": cfg.to_dict(), "val_cost": cost},
            )
            print(f"    new best (cost {cost:.4f}) -> {best_path.name}")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stop_patience:
                print(f"    early stop: {bad_epochs} epochs without improvement")
                break

    if not best_path.exists():
        raise RuntimeError("training finished without writing a best checkpoint")

    # -- final evaluation ------------------------------------------------- #
    print("\n  final evaluation")
    blob = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(blob["state_dict"])
    val_threshold = float(blob["threshold"])

    probs, ys = predict_split(model, val_loader, device)
    val_ev = evaluate(f"{cfg.run_id}/val", ys, probs, threshold=val_threshold)
    print(f"    val : {confusion_at(ys, probs, val_threshold)}")

    report_ev = val_ev
    report_split = "val"
    test_probs = None
    if cfg.test_cache_dir and Path(cfg.test_cache_dir).exists():
        test_cache = WaveCache(cfg.test_cache_dir)
        test_ds = TurnDataset(test_cache, np.arange(len(test_cache)), window, cfg.normalise_mode, None)
        test_loader = make_loader(test_ds, cfg.batch_size, num_workers=cfg.num_workers)
        test_probs, test_ys = predict_split(model, test_loader, device)
        # Threshold from validation, applied to test. Never re-picked here.
        report_ev = evaluate(cfg.run_id, test_ys, test_probs, threshold=val_threshold)
        report_split = "test"
        print(f"    test: {confusion_at(test_ys, test_probs, val_threshold)}")
        print(f"      (threshold {val_threshold:.3f} chosen on val, applied unchanged)")
    else:
        print(f"    test cache {cfg.test_cache_dir} absent — reporting val only")

    counts = model.parameter_counts()
    report_ev.params_m = counts["total"] / 1e6
    from src.optimize import model_size_mb

    report_ev.size_mb = model_size_mb(model)

    # -- artefacts -------------------------------------------------------- #
    run_dir = Path(cfg.artifacts_dir) / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    split_rep.save(run_dir / "split_report.json")
    report_ev.save(run_dir / "evaluation.json")
    np.save(run_dir / "val_probs.npy", probs)
    np.save(run_dir / "val_labels.npy", ys)
    np.save(run_dir / "val_indices.npy", val_idx)
    if test_probs is not None:
        np.save(run_dir / "test_probs.npy", test_probs)

    if report_split == "test" and test_probs is not None:
        rows = slice_report(WaveCache(cfg.test_cache_dir), np.arange(len(test_probs)),
                            test_probs, val_threshold)
        (run_dir / "slices.md").write_text(
            slices_to_markdown(rows, f"{cfg.run_id} — per-slice (test)"), encoding="utf-8"
        )

    table = ExperimentTable(cfg.table_path)
    row = table.add_evaluation(
        report_ev,
        model=f"{cfg.backbone.split('/')[-1]} {cfg.head}"
        + (f" +unfreeze{cfg.unfreeze_top_blocks}" if cfg.unfreeze_top_blocks else "")
        + (" +aug" if cfg.augment.get("enabled") else ""),
        window=str(WindowSpec(cfg.window_seconds)),
        notes=cfg.notes or f"scored on {report_split}",
    )
    table.print_row(row)
    table.save_markdown()

    print(f"\n  artefacts -> {run_dir}")
    print(f"  best weights -> {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
