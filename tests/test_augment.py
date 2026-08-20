"""Augmentation safety. The claim under test: augmentation must not move the
turn boundary, because if it does the label stops describing the audio.
"""
from __future__ import annotations

import numpy as np
import pytest

from src import SAMPLE_RATE
from src.augment import AugmentConfig, augment


def wave(seconds=1.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * 180 * t) + 0.01 * rng.normal(0, 1, t.size)).astype(
        np.float32
    )


def test_disabled_by_default_is_the_identity():
    """Augmentation is E5, not a default. A silent default would mean E1 through
    E4 were never actually run on clean audio."""
    w = wave()
    assert AugmentConfig().enabled is False
    assert np.array_equal(augment(w, AugmentConfig(), seed=1), w)


def test_same_seed_gives_identical_audio():
    """Unseeded augmentation makes an experiment table unreproducible in a way
    that is very hard to notice."""
    cfg = AugmentConfig(enabled=True)
    a = augment(wave(), cfg, seed=42)
    b = augment(wave(), cfg, seed=42)
    assert np.array_equal(a, b)


def test_different_seeds_give_different_audio():
    cfg = AugmentConfig(enabled=True)
    a = augment(wave(), cfg, seed=1)
    b = augment(wave(), cfg, seed=2)
    assert not np.array_equal(a, b)


def test_speed_perturbation_is_off_by_default():
    """The dangerous one. Stretching a 300 ms mid-sentence pause by 1.3x makes it
    390 ms, which is turn-final territory — the label no longer describes the
    audio. It stays opt-in and tightly bounded."""
    cfg = AugmentConfig(enabled=True)
    assert cfg.speed_prob == 0.0
    assert cfg.speed_range == (0.9, 1.1), "speed range must stay within +/-10%"


def test_length_is_preserved_when_speed_is_off():
    """Every other augmentation must be length-preserving, or the trailing
    silence the label depends on would change duration."""
    cfg = AugmentConfig(
        enabled=True, gain_prob=1.0, noise_prob=1.0, reverb_prob=1.0,
        mask_prob=1.0, speed_prob=0.0,
    )
    w = wave()
    for seed in range(12):
        out = augment(w, cfg, seed=seed)
        assert out.size == w.size, f"seed {seed} changed clip length"


def test_masking_never_touches_the_tail():
    """A mask over the last moments would erase the endpoint evidence and
    effectively relabel the clip. Masking is confined to the head."""
    cfg = AugmentConfig(
        enabled=True, gain_prob=0.0, noise_prob=0.0, reverb_prob=0.0,
        mask_prob=1.0, mask_max_ms=200.0, mask_head_fraction=0.7, speed_prob=0.0,
    )
    w = np.ones(SAMPLE_RATE, dtype=np.float32)
    tail_start = int(SAMPLE_RATE * 0.7)
    for seed in range(40):
        out = augment(w, cfg, seed=seed)
        assert np.all(out[tail_start:] != 0.0), (
            f"seed {seed} masked into the tail — that erases the boundary"
        )


def test_output_never_exceeds_full_scale():
    """Clipping is a distortion the model could learn to associate with the
    training-only augmented distribution."""
    cfg = AugmentConfig(
        enabled=True, gain_prob=1.0, gain_db=(10.0, 14.0), noise_prob=1.0,
        snr_db=(3.0, 5.0), reverb_prob=1.0, mask_prob=0.0, speed_prob=0.0,
    )
    for seed in range(20):
        out = augment(wave(), cfg, seed=seed)
        assert float(np.max(np.abs(out))) <= 1.0 + 1e-6


def test_noise_respects_the_requested_snr_roughly():
    cfg = AugmentConfig(
        enabled=True, gain_prob=0.0, noise_prob=1.0, snr_db=(20.0, 20.0),
        reverb_prob=0.0, mask_prob=0.0, speed_prob=0.0,
    )
    w = wave()
    out = augment(w, cfg, seed=5)
    residual = out - w
    speech_power = float(np.mean(w.astype(np.float64) ** 2))
    noise_power = float(np.mean(residual.astype(np.float64) ** 2))
    snr = 10 * np.log10(speech_power / noise_power)
    assert 15.0 < snr < 25.0, f"requested 20 dB SNR, measured {snr:.1f} dB"


def test_speed_change_scales_duration_when_enabled():
    cfg = AugmentConfig(
        enabled=True, gain_prob=0.0, noise_prob=0.0, reverb_prob=0.0,
        mask_prob=0.0, speed_prob=1.0, speed_range=(1.1, 1.1),
    )
    w = wave(1.0)
    out = augment(w, cfg, seed=0)
    # factor 1.1 => shorter by ~10%
    assert out.size == pytest.approx(w.size / 1.1, rel=0.02)


def test_output_is_float32():
    cfg = AugmentConfig(enabled=True)
    assert augment(wave(), cfg, seed=0).dtype == np.float32
