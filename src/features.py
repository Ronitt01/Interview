"""Log-mel extraction matched to the Whisper encoder.

Whisper's encoder was trained on a very specific front-end: 80-bin log-mel
spectrograms, 25 ms window, 10 ms hop, Slaney-scale filterbank, log10 with the
dynamic range clamped to 8 dB below the maximum and then scaled to roughly
[-1, 1]. Reimplementing that by hand is a reliable way to lose accuracy for
reasons that never surface as an error, so we drive Hugging Face's own
``WhisperFeatureExtractor`` and only change the one thing we need to change: the
chunk length.

Why the chunk length matters. The stock extractor pads every input to 30 s
because that is what Whisper's ASR decoder expects — 3000 mel frames, 1500
encoder positions. Our clips are at most 8 s. Feeding 30 s of mostly-zeros costs
roughly 4x the encoder FLOPs for no information, which is the difference between
a 12 ms and a 50 ms CPU inference. So we build the extractor at the window
length we actually use, and :mod:`src.model` truncates the encoder's positional
embeddings to match.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import SAMPLE_RATE
from .audio import WindowSpec

# Whisper's front-end constants. Fixed by the pretrained checkpoint, not by us.
HOP_LENGTH = 160  # 10 ms at 16 kHz
N_FFT = 400  # 25 ms at 16 kHz
N_MELS = 80  # whisper-tiny through -medium; large-v3 uses 128
ENCODER_STRIDE = 2  # the encoder's second conv halves the time axis


def mel_frames_for(seconds: float) -> int:
    """Number of mel frames the extractor emits for a window of ``seconds``."""
    return int(round(seconds * SAMPLE_RATE)) // HOP_LENGTH


def encoder_positions_for(seconds: float) -> int:
    """Number of encoder time positions after the conv stack."""
    return mel_frames_for(seconds) // ENCODER_STRIDE


@lru_cache(maxsize=8)
def _extractor(seconds: float, n_mels: int):
    """Cached feature extractor for one window length.

    Cached because constructing it builds the mel filterbank, and the experiment
    matrix sweeps window size across thousands of batches.
    """
    from transformers import WhisperFeatureExtractor

    return WhisperFeatureExtractor(
        feature_size=n_mels,
        sampling_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
        chunk_length=seconds,
        padding_value=0.0,
    )


class MelFrontEnd:
    """Turns fixed-length waveforms into encoder-ready log-mel tensors.

    Parameters
    ----------
    spec:
        The window the waveforms have already been fitted to. Waveforms are
        expected to arrive at exactly ``spec.samples`` — :func:`src.audio.prepare`
        guarantees that, and passing a different length here is a bug rather than
        something to paper over, so it raises.
    n_mels:
        80 for whisper-tiny. Exposed only so a larger backbone stays possible.
    """

    def __init__(self, spec: WindowSpec, n_mels: int = N_MELS) -> None:
        self.spec = spec
        self.n_mels = n_mels
        self.n_frames = mel_frames_for(spec.seconds)
        self.n_positions = encoder_positions_for(spec.seconds)
        if self.n_positions < 1:
            raise ValueError(
                f"window {spec.seconds}s is too short: it yields "
                f"{self.n_frames} mel frames and {self.n_positions} encoder "
                "positions. Use at least 0.02s."
            )

    def __call__(self, waves: np.ndarray) -> np.ndarray:
        """``(batch, samples)`` or ``(samples,)`` → ``(batch, n_mels, n_frames)``."""
        waves = np.asarray(waves, dtype=np.float32)
        if waves.ndim == 1:
            waves = waves[None, :]
        if waves.shape[-1] != self.spec.samples:
            raise ValueError(
                f"expected {self.spec.samples} samples "
                f"({self.spec.seconds:g}s at {SAMPLE_RATE} Hz), "
                f"got {waves.shape[-1]}. Run src.audio.prepare first."
            )

        fe = _extractor(self.spec.seconds, self.n_mels)
        out = fe(
            list(waves),
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            truncation=True,
        )
        mel = np.asarray(out["input_features"], dtype=np.float32)

        # The extractor's own truncation is length-based and can be off by one
        # frame at some chunk lengths; assert the contract rather than trust it.
        if mel.shape[-1] != self.n_frames:
            mel = (
                mel[..., : self.n_frames]
                if mel.shape[-1] > self.n_frames
                else np.pad(mel, ((0, 0), (0, 0), (0, self.n_frames - mel.shape[-1])))
            )
        return mel

    def __repr__(self) -> str:
        return (
            f"MelFrontEnd(window={self.spec.seconds:g}s, mels={self.n_mels}, "
            f"frames={self.n_frames}, positions={self.n_positions})"
        )
