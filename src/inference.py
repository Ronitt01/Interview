"""Standalone prediction. Imports nothing from ``training/``.

That separation is a hard rule, not a style preference. The demo, the latency
benchmark, the ONNX exporter and the streaming detector all load weights through
this module, and none of them should drag in an optimiser, a scheduler, a
dataloader or a Hub download path. If this file ever needs something from
``training/``, the thing it needs is in the wrong place.

A checkpoint carries everything required to reproduce a prediction: the model
config (so the encoder is rebuilt at the right window length), the weights, the
operating threshold chosen on validation, and provenance for the report. Loading
a checkpoint therefore never needs a config file passed alongside it — a
checkpoint that needs external context to interpret is a checkpoint that will be
misinterpreted.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from . import SAMPLE_RATE
from .audio import WindowSpec, fit_window, normalise, prepare
from .features import MelFrontEnd
from .model import ModelConfig, TurnDetector, build_model

Backend = Literal["torch", "torch-int8", "onnx"]

CHECKPOINT_VERSION = 1


# --------------------------------------------------------------------------- #
# checkpoint IO
# --------------------------------------------------------------------------- #
def save_checkpoint(
    path: str | Path,
    model: TurnDetector,
    threshold: float = 0.5,
    metadata: dict | None = None,
) -> Path:
    """Write weights + config + threshold + provenance in one file."""
    import torch

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "model_config": model.cfg.to_dict(),
            "state_dict": model.state_dict(),
            "threshold": float(threshold),
            "metadata": dict(metadata or {}),
        },
        p,
    )
    return p


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    import torch

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no checkpoint at {p}")
    # weights_only=False: our checkpoints contain plain dicts alongside tensors.
    # Only ever load checkpoints this repo produced.
    ckpt = torch.load(p, map_location=map_location, weights_only=False)
    if "model_config" not in ckpt or "state_dict" not in ckpt:
        raise ValueError(
            f"{p} is not a turn-detector checkpoint (keys: {sorted(ckpt)[:8]})"
        )
    return ckpt


# --------------------------------------------------------------------------- #
# the predictor
# --------------------------------------------------------------------------- #
@dataclass
class PredictorInfo:
    backend: str
    window_seconds: float
    threshold: float
    params_m: float
    size_mb: float
    head: str

    def __str__(self) -> str:
        return (
            f"{self.backend} | window {self.window_seconds:g}s | "
            f"head {self.head} | thr {self.threshold:.3f} | "
            f"{self.params_m:.2f}M params | {self.size_mb:.2f} MB"
        )


class TurnPredictor:
    """Waveform in, P(turn ended) out. One class, three backends.

    Parameters
    ----------
    backend:
        ``"torch"`` is float32 eager. ``"torch-int8"`` applies dynamic
        quantisation at load time — see :mod:`src.optimize` for what that costs
        in accuracy. ``"onnx"`` loads a graph exported by
        :func:`src.optimize.export_onnx` and is the fastest CPU path.
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        backend: Backend = "torch",
        onnx_path: str | Path | None = None,
        device: str = "cpu",
        threads: int | None = None,
    ) -> None:
        self.backend = backend
        self.device = device
        self._session = None
        self._model = None

        if backend == "onnx":
            if onnx_path is None:
                raise ValueError("backend='onnx' requires onnx_path")
            meta_path = Path(onnx_path).with_suffix(".json")
            if not meta_path.exists():
                raise FileNotFoundError(
                    f"{meta_path} not found. export_onnx writes it alongside the "
                    "graph; it carries the window length and threshold that the "
                    "graph alone cannot express."
                )
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.cfg = ModelConfig.from_dict(meta["model_config"])
            self.threshold = float(meta.get("threshold", 0.5))
            self._size_mb = Path(onnx_path).stat().st_size / 1e6
            self._session = self._make_session(onnx_path, threads)
            self._params_m = float(meta.get("params_m", 0.0))
        else:
            if checkpoint is None:
                raise ValueError(f"backend={backend!r} requires a checkpoint")
            ckpt = load_checkpoint(checkpoint, map_location=device)
            self.cfg = ModelConfig.from_dict(ckpt["model_config"])
            self.threshold = float(ckpt.get("threshold", 0.5))
            model = build_model(self.cfg)
            model.load_state_dict(ckpt["state_dict"])
            model.eval().to(device)
            self._params_m = model.parameter_counts()["total"] / 1e6

            if backend == "torch-int8":
                from .optimize import quantize_dynamic

                model = quantize_dynamic(model)
            self._model = model
            self._size_mb = Path(checkpoint).stat().st_size / 1e6
            if threads is not None:
                import torch

                torch.set_num_threads(int(threads))

        self.window = WindowSpec(self.cfg.window_seconds)
        self.front_end = MelFrontEnd(self.window, n_mels=self.cfg.n_mels)

    @staticmethod
    def _make_session(onnx_path, threads: int | None):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads is not None:
            # Both matter: intra bounds the threads inside one op, inter bounds
            # parallel ops. Leaving inter unset lets ORT oversubscribe a small
            # box and makes the p95 worse than the p50 suggests.
            opts.intra_op_num_threads = int(threads)
            opts.inter_op_num_threads = 1
        return ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

    # -- prediction --------------------------------------------------------- #
    def _forward(self, mel: np.ndarray) -> np.ndarray:
        """``(B, n_mels, n_frames)`` → probabilities ``(B,)``."""
        if self._session is not None:
            out = self._session.run(None, {"mel": mel.astype(np.float32)})[0]
            logits = np.asarray(out, dtype=np.float64).reshape(-1)
        else:
            import torch

            with torch.no_grad():
                t = torch.from_numpy(mel.astype(np.float32)).to(self.device)
                logits = self._model(t).cpu().numpy().astype(np.float64).reshape(-1)
        return 1.0 / (1.0 + np.exp(-logits))

    def predict_window(self, wave: np.ndarray) -> float:
        """Score an already-windowed, already-normalised waveform.

        The hot path for :class:`src.streaming.StreamingTurnDetector`, which has
        done the windowing itself and must not pay for it twice.
        """
        mel = self.front_end(np.asarray(wave, dtype=np.float32))
        return float(self._forward(mel)[0])

    def predict(self, wave: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
        """Score a raw clip of any length and sample rate."""
        prepared = prepare(wave, sample_rate, self.window, self.cfg.resolved.get("normalise", "peak"))
        return self.predict_window(prepared)

    def predict_batch(
        self,
        waves: list[np.ndarray],
        sample_rate: int = SAMPLE_RATE,
        batch_size: int = 32,
    ) -> np.ndarray:
        out: list[np.ndarray] = []
        for i in range(0, len(waves), batch_size):
            chunk = waves[i : i + batch_size]
            prepared = np.stack(
                [prepare(w, sample_rate, self.window) for w in chunk]
            )
            out.append(self._forward(self.front_end(prepared)))
        return np.concatenate(out) if out else np.zeros(0)

    def predict_cached(self, cache, indices, batch_size: int = 32) -> np.ndarray:
        """Score rows of a :class:`src.dataset.WaveCache`. Already 16 kHz mono."""
        idx = np.asarray(indices, dtype=np.int64)
        out: list[np.ndarray] = []
        for i in range(0, idx.size, batch_size):
            waves = np.stack(
                [
                    fit_window(normalise(cache.wave(int(j)), "peak"), self.window)
                    for j in idx[i : i + batch_size]
                ]
            )
            out.append(self._forward(self.front_end(waves)))
        return np.concatenate(out) if out else np.zeros(0)

    def decide(self, wave: np.ndarray, sample_rate: int = SAMPLE_RATE) -> tuple[bool, float]:
        """``(turn_ended, probability)`` at the checkpoint's own threshold."""
        p = self.predict(wave, sample_rate)
        return p >= self.threshold, p

    # -- for the streaming detector ---------------------------------------- #
    def as_score_fn(self):
        """A :data:`src.streaming.ScoreFn` bound to this predictor."""
        return self.predict_window

    # -- reporting ---------------------------------------------------------- #
    def info(self) -> PredictorInfo:
        return PredictorInfo(
            backend=self.backend,
            window_seconds=self.cfg.window_seconds,
            threshold=self.threshold,
            params_m=self._params_m,
            size_mb=self._size_mb,
            head=self.cfg.head,
        )

    def __repr__(self) -> str:
        return f"TurnPredictor({self.info()})"
