"""Make it small and fast, and measure honestly what that cost.

The sentence this module exists to let the report write, with real numbers in it:

    "Model B improves F1 by X% while staying under Y MB and Z ms CPU latency."

Three things are done here, in the order that makes each one's effect visible
separately — bundling them would make it impossible to say which one paid off:

1. **Dynamic INT8 quantisation** of the linear layers. Dynamic rather than
   static because it needs no calibration set and quantises exactly the weights
   that dominate an 8M-parameter transformer encoder. The attention matmuls stay
   float, which is why the speedup is real but not 4x.
2. **ONNX export**, which removes Python from the inference loop and lets
   onnxruntime fuse the graph. This is usually the larger win on CPU and it is
   measured separately from quantisation for that reason.
3. **Benchmarking** at batch size 1, single and multi threaded, reporting p50 and
   p95 — never a mean. A mean latency hides the tail and the tail is what a user
   hears.

Reference point to beat or match: the published Smart Turn v3 checkpoint is
8 MB int8 ONNX / 32 MB unquantised, and its authors report ~10-12 ms on some
CPUs and ~65 ms on a standard cloud instance. Our architecture is the same, so a
wildly different number means a bug in this module, not a breakthrough.
"""
from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features import MelFrontEnd
from .audio import WindowSpec
from .model import TurnDetector


# --------------------------------------------------------------------------- #
# size
# --------------------------------------------------------------------------- #
def model_size_mb(model) -> float:
    """On-disk MB, measured by serialising rather than by summing ``numel``.

    Summing parameter counts times 4 bytes gets the wrong answer for a quantised
    model, whose int8 weights carry float scales and zero-points alongside them.
    Serialising to a buffer measures what actually ships.
    """
    import io

    import torch

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


def file_size_mb(path: str | Path) -> float:
    return Path(path).stat().st_size / 1e6


def onnx_total_size_mb(path: str | Path) -> float:
    """Size of an ONNX artefact **including any external data files**.

    torch's exporter may place initializers in a sibling ``<name>.onnx.data``
    rather than inline. Measuring only the ``.onnx`` then reports a 31 MB model
    as 0.36 MB — technically the size of that file, and completely wrong as the
    size of the thing you would deploy. Both files have to ship together, so both
    are counted.
    """
    p = Path(path)
    total = p.stat().st_size
    for extra in (p.with_suffix(p.suffix + ".data"), p.parent / f"{p.stem}.data"):
        if extra.exists() and extra != p:
            total += extra.stat().st_size
    return total / 1e6


# --------------------------------------------------------------------------- #
# quantisation
# --------------------------------------------------------------------------- #
def quantize_dynamic(model: TurnDetector):
    """Dynamic INT8 over ``nn.Linear`` and ``nn.GRU``.

    Returns a *new* module; the original is untouched so the caller can score
    both and report the delta. Runs on CPU only — quantised kernels have no CUDA
    path, which is fine because this is a CPU-deployment optimisation.
    """
    import torch
    import torch.nn as nn

    model = model.eval().to("cpu")
    return torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear, nn.GRU}, dtype=torch.qint8
    )


# --------------------------------------------------------------------------- #
# ONNX
# --------------------------------------------------------------------------- #
def export_onnx(
    model: TurnDetector,
    path: str | Path,
    threshold: float = 0.5,
    opset: int = 18,
    dynamic_batch: bool = True,
) -> Path:
    """Export to ONNX, and write the metadata sidecar the graph cannot hold.

    The sidecar (``<name>.json``) carries the window length, mel count, head kind
    and operating threshold. Without it an ONNX file is an anonymous tensor
    function: you can run it but you cannot know what audio to feed it or where
    to cut its output. :class:`src.inference.TurnPredictor` refuses to load a
    graph whose sidecar is missing, on purpose.
    """
    import torch

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().to("cpu")

    dummy = torch.zeros(1, model.cfg.n_mels, model.cfg.n_mel_frames, dtype=torch.float32)
    dyn = {"mel": {0: "batch"}, "logit": {0: "batch"}} if dynamic_batch else None

    # verbose=False is not cosmetic on Windows: torch's exporter prints progress
    # lines containing emoji, and a cp1252 stdout raises UnicodeEncodeError while
    # encoding them — killing the script *after* the export has already
    # succeeded. Silencing the progress output removes the failure entirely.
    torch.onnx.export(
        model,
        (dummy,),
        str(p),
        input_names=["mel"],
        output_names=["logit"],
        dynamic_axes=dyn,
        opset_version=opset,
        do_constant_folding=True,
        verbose=False,
    )

    # Consolidate immediately: the exporter may have written initializers to a
    # sibling .onnx.data, and every consumer downstream (the size measurement,
    # the quantiser, the demo) is simpler with one self-contained file.
    strip_initializer_value_info(p)

    counts = model.parameter_counts()
    p.with_suffix(".json").write_text(
        json.dumps(
            {
                "model_config": model.cfg.to_dict(),
                "threshold": float(threshold),
                "params_m": counts["total"] / 1e6,
                "opset": opset,
                "input": {"name": "mel", "shape": [1, model.cfg.n_mels, model.cfg.n_mel_frames]},
                "output": {"name": "logit", "activation": "apply sigmoid to get P(turn ended)"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return p


def strip_initializer_value_info(path: str | Path) -> int:
    """Remove ``value_info`` annotations, returning how many were dropped.

    Why this is needed. torch 2.13's exporter emits ``value_info`` entries for
    graph *initializers* — the weights and biases — alongside the intermediate
    tensors it is meant to describe. onnxruntime's quantiser then runs its own
    shape inference, disagrees with those entries, and aborts with

        InferenceError: Inferred shape and existing shape differ in
        dimension 0: (384) vs (1)

    where 384 is ``d_model``. The graph itself is correct: inputs and outputs are
    properly shaped and it runs fine in onnxruntime. Only the optional
    intermediate annotation is inconsistent.

    ``value_info`` is exactly that — optional. Clearing it lets the quantiser
    infer shapes from scratch, which is what it was going to do anyway. Inputs,
    outputs and initializers are untouched, so the graph's contract is unchanged.

    Side effect, and a deliberate one: loading with external data and saving
    without it **consolidates** a ``<name>.onnx`` + ``<name>.onnx.data`` pair
    into one self-contained file. That is what you want to deploy — a single
    artefact rather than two files that must travel together — so it is done
    explicitly rather than left to the default.
    """
    import onnx

    p = Path(path)
    model = onnx.load(str(p))  # pulls in external data if present
    n = len(model.graph.value_info)
    del model.graph.value_info[:]
    onnx.save(model, str(p), save_as_external_data=False)

    # The .data file is now redundant; leaving it would double the artefact size
    # on disk and invite shipping a stale copy.
    for extra in (p.with_suffix(p.suffix + ".data"), p.parent / f"{p.stem}.data"):
        if extra.exists() and extra != p:
            extra.unlink()
    return n


def quantize_onnx(src: str | Path, dst: str | Path | None = None) -> Path:
    """Dynamic INT8 quantisation of an exported graph.

    This is the artefact that should land near the reference 8 MB. The sidecar is
    copied across because the quantised graph needs exactly the same input
    contract.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic as ort_quantize

    src = Path(src)
    dst = Path(dst) if dst else src.with_name(src.stem + "-int8.onnx")

    dropped = strip_initializer_value_info(src)
    if dropped:
        print(f"    cleared {dropped} value_info annotations before quantising")

    ort_quantize(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
    )
    side = src.with_suffix(".json")
    if side.exists():
        meta = json.loads(side.read_text(encoding="utf-8"))
        meta["quantisation"] = "onnxruntime dynamic QInt8"
        dst.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return dst


# --------------------------------------------------------------------------- #
# benchmarking
# --------------------------------------------------------------------------- #
@dataclass
class LatencyResult:
    label: str
    threads: int
    batch_size: int
    window_seconds: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    mean_ms: float
    n: int
    real_time_factor: float
    peak_rss_mb: float | None = None
    size_mb: float | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "threads": self.threads,
            "batch_size": self.batch_size,
            "window_seconds": self.window_seconds,
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "mean_ms": round(self.mean_ms, 2),
            "n": self.n,
            "real_time_factor": round(self.real_time_factor, 5),
            "peak_rss_mb": None if self.peak_rss_mb is None else round(self.peak_rss_mb, 1),
            "size_mb": None if self.size_mb is None else round(self.size_mb, 2),
            **self.extra,
        }

    def __str__(self) -> str:
        rss = f", rss {self.peak_rss_mb:.0f} MB" if self.peak_rss_mb else ""
        return (
            f"{self.label:<26s} threads={self.threads} bs={self.batch_size}  "
            f"p50 {self.p50_ms:6.2f} ms  p95 {self.p95_ms:6.2f} ms  "
            f"RTF {self.real_time_factor:.4f}{rss}"
        )


def _peak_rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return None


def benchmark(
    predict_fn,
    window: WindowSpec,
    label: str,
    threads: int = 1,
    batch_size: int = 1,
    n_warmup: int = 15,
    n_iters: int = 200,
    seed: int = 0,
    size_mb: float | None = None,
) -> LatencyResult:
    """Time ``predict_fn`` on synthetic noise of the right shape.

    Synthetic input rather than real clips: this measures the *compute*, and real
    audio would add cache-miss variance from the corpus reader without changing
    the FLOPs. Accuracy is measured elsewhere on real audio; this number is
    latency only.

    Warmup matters more than it looks. The first few calls pay for lazy kernel
    selection, ONNX graph optimisation and thread-pool spin-up; including them
    would put a 200 ms outlier in a 12 ms distribution and wreck the p99.
    """
    rng = np.random.default_rng(seed)
    batch = rng.normal(0, 0.05, size=(batch_size, window.samples)).astype(np.float32)
    single = batch[0]

    payload = single if batch_size == 1 else batch
    for _ in range(n_warmup):
        predict_fn(payload)

    gc.collect()
    times: list[float] = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        predict_fn(payload)
        times.append((time.perf_counter() - t0) * 1000.0)

    a = np.asarray(times)
    audio_ms = window.seconds * 1000.0 * batch_size
    return LatencyResult(
        label=label,
        threads=threads,
        batch_size=batch_size,
        window_seconds=window.seconds,
        p50_ms=float(np.percentile(a, 50)),
        p95_ms=float(np.percentile(a, 95)),
        p99_ms=float(np.percentile(a, 99)),
        max_ms=float(a.max()),
        mean_ms=float(a.mean()),
        n=int(a.size),
        real_time_factor=float(np.percentile(a, 50)) / audio_ms,
        peak_rss_mb=_peak_rss_mb(),
        size_mb=size_mb,
    )


def benchmark_predictor(
    predictor,
    label: str,
    threads_list: tuple[int, ...] = (1, 4),
    batch_sizes: tuple[int, ...] = (1,),
    n_iters: int = 200,
) -> list[LatencyResult]:
    """Sweep threads and batch size for one predictor.

    Batch size 1 is the case that matters — a live call has exactly one stream —
    but a larger batch is included when asked because it separates "the model is
    slow" from "we are latency-bound on per-call overhead".
    """
    import torch

    out: list[LatencyResult] = []
    info = predictor.info()
    for threads in threads_list:
        torch.set_num_threads(int(threads))
        for bs in batch_sizes:
            fn = (
                predictor.predict_window
                if bs == 1
                else (lambda b: predictor._forward(predictor.front_end(b)))
            )
            out.append(
                benchmark(
                    fn,
                    predictor.window,
                    label=f"{label} ({predictor.backend})",
                    threads=threads,
                    batch_size=bs,
                    n_iters=n_iters,
                    size_mb=info.size_mb,
                )
            )
    return out


def front_end_cost(window: WindowSpec, n_iters: int = 200) -> LatencyResult:
    """How much of the latency is mel extraction rather than the model.

    Worth knowing before optimising the wrong half: at an 8 s window the mel
    front-end is a non-trivial share of a 12 ms budget, and no amount of
    quantisation touches it.
    """
    front = MelFrontEnd(window)
    rng = np.random.default_rng(0)
    wave = rng.normal(0, 0.05, size=window.samples).astype(np.float32)
    return benchmark(front, window, label="mel front-end only", n_iters=n_iters)


def accuracy_delta(before, after, name: str = "quantisation") -> dict:
    """The regression table entry for an optimisation step.

    Reports the delta even when it is negative — a quantisation that costs 0.4 F1
    for a 4x size reduction is a *result*, and hiding it is how a submission
    starts reading as cherry-picked.
    """
    b, a = before.confusion, after.confusion
    return {
        "step": name,
        "f1_before": round(b["f1"], 4),
        "f1_after": round(a["f1"], 4),
        "f1_delta": round(a["f1"] - b["f1"], 4),
        "false_interrupt_before": round(b["false_interruption_rate"], 4),
        "false_interrupt_after": round(a["false_interruption_rate"], 4),
        "false_interrupt_delta": round(
            a["false_interruption_rate"] - b["false_interruption_rate"], 4
        ),
        "size_mb_before": before.size_mb,
        "size_mb_after": after.size_mb,
        "verdict": _verdict(a["f1"] - b["f1"]),
    }


def _verdict(delta: float) -> str:
    if delta >= 0.002:
        return "improved"
    if delta > -0.005:
        return "neutral (within noise)"
    return "regression — stated, not hidden"


def save_results(results: list[LatencyResult], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([r.to_dict() for r in results], indent=2), encoding="utf-8")
    return p
