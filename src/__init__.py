"""Smart turn detection — audio-only endpointing for voice agents.

Package layout, in the order the pipeline uses it:

    config       typed run configuration loaded from configs/*.yaml
    audio        waveform IO, resampling, normalisation, left-padding
    features     log-mel extraction matched to the Whisper encoder
    splits       group-aware train/val partitioning + leakage assertions
    dataset      torch Dataset over the Smart Turn parquet corpus
    model        Whisper Tiny encoder + interchangeable classification heads
    metrics      the scoring vocabulary, incl. false-interruption rate
    evaluation   threshold sweeps, operating-point selection, report rows
    baselines    E0 energy/pause detector and E0b Silero VAD reference
    inference    standalone predictor — loads weights, no training imports
    streaming    rolling-window detector with smoothing and hysteresis
    optimize     quantisation, ONNX export, CPU latency benchmarking
"""

__version__ = "0.1.0"


def _make_stdout_unicode_safe() -> None:
    """Stop a Windows cp1252 console from killing a script over one character.

    Several dependencies print non-ASCII progress output (torch's ONNX exporter
    uses emoji). On a default Windows console that is cp1252, and encoding those
    raises UnicodeEncodeError — which surfaces as a crash *after* the real work
    succeeded, in a traceback that points at the print rather than the cause.
    Reconfiguring to UTF-8 with replacement makes the output slightly uglier in
    the worst case and removes a whole class of spurious failure.
    """
    import sys

    for stream in (sys.stdout, sys.stderr):
        enc = (getattr(stream, "encoding", "") or "").lower()
        if enc.replace("-", "") in ("utf8", "utf8mb4"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass  # not a reconfigurable stream (piped, captured, closed)


_make_stdout_unicode_safe()

SAMPLE_RATE = 16_000
"""Every stage assumes 16 kHz mono. The Whisper encoder is trained at this rate
and the reference Smart Turn implementation accepts nothing else."""

LABEL_COLUMN = "endpoint_bool"
"""Positive class means the speaker has finished their turn."""

POSITIVE = 1
NEGATIVE = 0
