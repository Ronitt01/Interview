"""Waveform augmentation, and a note on which augmentations are *legal* here.

Most audio augmentation libraries offer a grab-bag: noise, gain, pitch, speed,
time-stretch, reverb, time-masking. On this task some of those are actively
harmful, and knowing which is part of understanding the problem:

**Safe** — transformations that change the channel but not the turn boundary:
gain, additive noise, band-limiting, reverb/room impulse. A clip that ended is
still a clip that ended after you add noise to it.

**Dangerous** — transformations that move or destroy the boundary:

* *Time-stretch / speed change* alters pause duration. A 300 ms mid-sentence
  pause stretched 1.3x becomes a 390 ms pause, which is squarely inside
  turn-final territory. The label no longer describes the audio. We keep speed
  perturbation but bound it tightly (±10%) and expose it as its own experiment
  so its effect is visible rather than bundled.
* *Random cropping from the end* removes the endpoint evidence entirely and
  relabels a positive as a negative. Never applied.
* *Time-masking near the tail* has the same problem in weaker form; masking is
  restricted to the first 70% of the clip.

Every augmentation here is seeded per-sample from the clip index so that two
runs of the same config see the same augmented audio. Unseeded augmentation
makes an experiment table unreproducible in a way that is very hard to notice.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import SAMPLE_RATE


@dataclass
class AugmentConfig:
    """Probabilities and ranges. All off by default — augmentation is E5."""

    enabled: bool = False
    gain_prob: float = 0.5
    gain_db: tuple[float, float] = (-8.0, 6.0)
    noise_prob: float = 0.5
    snr_db: tuple[float, float] = (8.0, 30.0)
    speed_prob: float = 0.0  # off by default: see module docstring
    speed_range: tuple[float, float] = (0.9, 1.1)
    reverb_prob: float = 0.2
    reverb_decay: tuple[float, float] = (0.15, 0.5)
    mask_prob: float = 0.2
    mask_max_ms: float = 120.0
    # Masking never touches the last (1 - mask_head_fraction) of the clip.
    mask_head_fraction: float = 0.7

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "gain_prob": self.gain_prob,
            "gain_db": list(self.gain_db),
            "noise_prob": self.noise_prob,
            "snr_db": list(self.snr_db),
            "speed_prob": self.speed_prob,
            "speed_range": list(self.speed_range),
            "reverb_prob": self.reverb_prob,
            "mask_prob": self.mask_prob,
            "mask_max_ms": self.mask_max_ms,
        }


def _gain(wave: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    db = rng.uniform(*cfg.gain_db)
    return wave * float(10.0 ** (db / 20.0))


def _noise(wave: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    """Additive white noise at a sampled SNR.

    White rather than sampled from a noise corpus: we have no licensed noise set
    in this repo, and white noise is the honest stand-in. The report says so —
    real-world robustness to *babble* and *music* is measured on the Day-8
    stress suite instead of trained against here, which is the weaker claim but
    the true one.
    """
    speech_power = float(np.mean(wave.astype(np.float64) ** 2))
    if speech_power <= 0:
        return wave
    snr = rng.uniform(*cfg.snr_db)
    noise_power = speech_power / (10.0 ** (snr / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=wave.shape).astype(np.float32)
    return wave + noise


def _speed(wave: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    """Resample-based speed change. Alters duration *and* pitch together."""
    factor = float(rng.uniform(*cfg.speed_range))
    if abs(factor - 1.0) < 1e-3:
        return wave
    n = max(1, int(round(wave.shape[-1] / factor)))
    idx = np.linspace(0, wave.shape[-1] - 1, n)
    return np.interp(idx, np.arange(wave.shape[-1]), wave).astype(np.float32)


def _reverb(wave: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    """Cheap exponential-decay room impulse.

    A synthetic IR rather than a measured one, for the same licensing reason as
    the noise. It is enough to blur onsets and smear the decay after speech
    stops, which is the property that matters for endpointing: reverb makes a
    genuine silence look less silent.
    """
    decay = float(rng.uniform(*cfg.reverb_decay))
    ir_len = int(0.25 * SAMPLE_RATE)
    t = np.arange(ir_len, dtype=np.float32) / SAMPLE_RATE
    ir = np.exp(-t / max(decay, 1e-3)).astype(np.float32)
    ir *= rng.normal(0.0, 1.0, size=ir_len).astype(np.float32)
    ir[0] = 1.0  # keep the direct path dominant
    out = np.convolve(wave, ir, mode="full")[: wave.shape[-1]]
    peak = float(np.max(np.abs(out)))
    if peak > 0:
        out = out * (float(np.max(np.abs(wave))) / peak)
    return out.astype(np.float32)


def _mask(wave: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig) -> np.ndarray:
    """Zero a short span, restricted to the head of the clip."""
    n = wave.shape[-1]
    limit = int(n * cfg.mask_head_fraction)
    span = int(rng.uniform(0.0, cfg.mask_max_ms) * SAMPLE_RATE / 1000.0)
    if span <= 0 or limit <= span:
        return wave
    start = int(rng.integers(0, limit - span))
    out = wave.copy()
    out[start : start + span] = 0.0
    return out


def augment(
    wave: np.ndarray,
    cfg: AugmentConfig,
    seed: int,
) -> np.ndarray:
    """Apply the configured augmentations. Deterministic given ``seed``.

    Applied in physical order — speed, then room, then noise, then gain — because
    that is the order the corresponding real-world effects compose in. Adding
    noise before reverb would put the microphone's noise floor inside the room,
    which is backwards and audibly wrong.
    """
    if not cfg.enabled:
        return wave
    rng = np.random.default_rng(seed)
    out = np.asarray(wave, dtype=np.float32)

    if cfg.speed_prob and rng.random() < cfg.speed_prob:
        out = _speed(out, rng, cfg)
    if cfg.mask_prob and rng.random() < cfg.mask_prob:
        out = _mask(out, rng, cfg)
    if cfg.reverb_prob and rng.random() < cfg.reverb_prob:
        out = _reverb(out, rng, cfg)
    if cfg.noise_prob and rng.random() < cfg.noise_prob:
        out = _noise(out, rng, cfg)
    if cfg.gain_prob and rng.random() < cfg.gain_prob:
        out = _gain(out, rng, cfg)

    # Guard against a gain/noise combination pushing past full scale. Clipping
    # here would be a distortion the model could learn to associate with the
    # augmented (and therefore training-only) distribution.
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32, copy=False)
