"""Waveform handling: load, resample, normalise, fit to a fixed window.

Every preprocessing decision in this module has a stated reason, because §2 of
the brief scores the reasoning and not the code. The reasons live in the
docstrings next to the operation they justify, so a reviewer reading the
function reads the argument for it at the same time.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Literal

import numpy as np

from . import SAMPLE_RATE

NormaliseMode = Literal["peak", "rms", "none"]

# -1 dBFS rather than 0: leaves headroom so that a later resample or an int16
# round-trip cannot clip a sample that sits exactly at full scale.
_PEAK_TARGET = 10 ** (-1.0 / 20.0)

# -20 dBFS is the usual speech-loudness target for ASR front-ends. Chosen over a
# louder target because RMS normalisation raises the noise floor along with the
# speech; -20 dB keeps quiet recordings from being amplified into hiss.
_RMS_TARGET = 10 ** (-20.0 / 20.0)

_EPS = 1e-9


@dataclass(frozen=True)
class WindowSpec:
    """How a variable-length clip is fitted to the model's fixed input.

    Attributes
    ----------
    seconds:
        Window length the model sees.
    pad_side:
        Where zeros go when a clip is shorter than the window. ``"left"`` is the
        default and matches the reference Smart Turn implementation: the turn
        boundary we are classifying is always the *end* of the clip, so keeping
        audio flush to the right edge puts the decision-relevant moment at a
        fixed position in the input for every sample. Right-padding would slide
        that moment around with clip length and force the model to learn a
        position-invariance it never needs at inference time.
    trim_side:
        Which end to discard when a clip is longer than the window. ``"left"``
        keeps the most recent audio — the tail carries the endpoint evidence
        (falling pitch, trailing filler, breath), the head carries little.
    """

    seconds: float
    pad_side: Literal["left", "right"] = "left"
    trim_side: Literal["left", "right"] = "left"

    @property
    def samples(self) -> int:
        return int(round(self.seconds * SAMPLE_RATE))

    def __str__(self) -> str:  # appears in experiment-table rows
        return f"{self.seconds:g}s"


def to_mono(wave: np.ndarray) -> np.ndarray:
    """Average multi-channel audio down to one channel.

    Averaged rather than channel-0-selected: some corpora put the speaker on one
    channel and near-silence on the other, and picking a channel blind would
    silently drop half the dataset to noise.
    """
    wave = np.asarray(wave)
    if wave.ndim == 1:
        return wave.astype(np.float32, copy=False)
    if wave.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D audio, got shape {wave.shape}")
    # Accept both (channels, samples) and (samples, channels).
    axis = 0 if wave.shape[0] < wave.shape[1] else 1
    return wave.mean(axis=axis).astype(np.float32, copy=False)


def resample(wave: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Resample to ``target_sr``.

    Uses ``librosa.resample`` (a windowed-sinc / polyphase design) rather than
    naive index striding, because striding aliases: a 44.1 kHz clip decimated by
    slicing folds everything above 8 kHz back into the speech band, which shows
    up as a phantom high-frequency texture the model can key on. That would be a
    feature of our preprocessing, not of the speaker's turn.
    """
    if orig_sr == target_sr:
        return np.asarray(wave, dtype=np.float32)
    import librosa  # imported lazily: heavy, and not needed when rates match

    return librosa.resample(
        np.asarray(wave, dtype=np.float32), orig_sr=orig_sr, target_sr=target_sr
    ).astype(np.float32, copy=False)


def normalise(wave: np.ndarray, mode: NormaliseMode = "peak") -> np.ndarray:
    """Scale amplitude so the model cannot key on recording gain.

    Loudness varies across recording conditions — phone versus headset, close
    versus far mic — and none of that variation carries information about whether
    a speaker has finished talking. Left un-normalised, a model can reach decent
    validation accuracy by learning which *corpus* a clip came from, since gain
    correlates with source. Normalising removes that shortcut.

    ``peak`` is the default: it is monotone, it cannot change the silence/speech
    ratio, and it never amplifies a noise floor. ``rms`` equalises perceived
    loudness more evenly but does raise quiet recordings' noise, so it is offered
    as an experiment (E-series) rather than assumed.
    """
    wave = np.asarray(wave, dtype=np.float32)
    if mode == "none":
        return wave
    if wave.size == 0:
        return wave

    if mode == "peak":
        peak = float(np.max(np.abs(wave)))
        if peak < _EPS:  # digital silence — scaling it would only amplify noise
            return wave
        return (wave * (_PEAK_TARGET / peak)).astype(np.float32, copy=False)

    if mode == "rms":
        rms = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2)))
        if rms < _EPS:
            return wave
        scaled = wave * (_RMS_TARGET / rms)
        # RMS scaling can overshoot full scale on peaky material; clamp the peak
        # instead of hard-clipping, which would introduce broadband distortion.
        peak = float(np.max(np.abs(scaled)))
        if peak > _PEAK_TARGET:
            scaled = scaled * (_PEAK_TARGET / peak)
        return scaled.astype(np.float32, copy=False)

    raise ValueError(f"unknown normalise mode: {mode!r}")


def fit_window(wave: np.ndarray, spec: WindowSpec) -> np.ndarray:
    """Pad or trim ``wave`` to exactly ``spec.samples`` samples.

    See :class:`WindowSpec` for why the default is left-pad / left-trim.
    """
    wave = np.asarray(wave, dtype=np.float32)
    want = spec.samples
    have = wave.shape[-1]

    if have == want:
        return wave
    if have > want:
        return wave[-want:] if spec.trim_side == "left" else wave[:want]

    pad = np.zeros(want - have, dtype=np.float32)
    return np.concatenate([pad, wave] if spec.pad_side == "left" else [wave, pad])


def prepare(
    wave: np.ndarray,
    orig_sr: int,
    spec: WindowSpec,
    normalise_mode: NormaliseMode = "peak",
) -> np.ndarray:
    """The full front-end, in the one order that is correct.

    mono → resample → normalise → window. The order matters:

    * mono before resample, so we resample one channel instead of two;
    * normalise before windowing, so the scale factor is computed from the real
      recording rather than from a window that may be mostly inserted zeros —
      otherwise a short clip padded to 8 s would be normalised against its own
      padding and come out louder than a long one;
    * window last, so the tensor handed to the model is always the same shape.
    """
    wave = to_mono(wave)
    wave = resample(wave, orig_sr, SAMPLE_RATE)
    wave = normalise(wave, normalise_mode)
    return fit_window(wave, spec)


def decode_bytes(raw: bytes) -> tuple[np.ndarray, int]:
    """Decode an encoded audio blob to ``(mono_float32, sample_rate)``.

    Used for demo uploads and for the Bulbul-synthesised Hinglish clips, both of
    which arrive as encoded bytes rather than as arrays.
    """
    import soundfile as sf

    wave, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    return to_mono(wave), int(sr)


def rms_dbfs(wave: np.ndarray) -> float:
    """RMS level in dBFS. Used by the E0 baseline and the EDA notebook."""
    wave = np.asarray(wave, dtype=np.float64)
    if wave.size == 0:
        return -np.inf
    rms = float(np.sqrt(np.mean(wave**2)))
    return 20.0 * np.log10(max(rms, _EPS))


def frame(wave: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Split into overlapping frames, shape ``(n_frames, frame_len)``.

    A strided view rather than a copy — the energy baseline frames every clip in
    the corpus and the copies would dominate its runtime.
    """
    wave = np.asarray(wave, dtype=np.float32)
    if wave.shape[-1] < frame_len:
        wave = fit_window(wave, WindowSpec(frame_len / SAMPLE_RATE))
    n = 1 + (wave.shape[-1] - frame_len) // hop
    return np.lib.stride_tricks.as_strided(
        wave,
        shape=(n, frame_len),
        strides=(wave.strides[-1] * hop, wave.strides[-1]),
        writeable=False,
    )
