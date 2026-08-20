"""Preprocessing invariants. Each test pins a decision made in src/audio.py.

If one of these fails, a preprocessing choice changed — which is allowed, but it
should be a deliberate change with the reasoning updated, not a silent drift.
"""
from __future__ import annotations

import numpy as np
import pytest

from src import SAMPLE_RATE
from src.audio import (
    WindowSpec,
    fit_window,
    frame,
    normalise,
    prepare,
    resample,
    rms_dbfs,
    to_mono,
)
from src.features import MelFrontEnd, encoder_positions_for, mel_frames_for


# --------------------------------------------------------------------------- #
# mono
# --------------------------------------------------------------------------- #
def test_stereo_is_averaged_not_channel_picked():
    """Picking channel 0 blind would silently zero any corpus that puts the
    speaker on channel 1."""
    left = np.zeros(1000, dtype=np.float32)
    right = np.ones(1000, dtype=np.float32) * 0.5
    mono = to_mono(np.stack([left, right]))
    assert mono.shape == (1000,)
    assert np.allclose(mono, 0.25)


def test_mono_handles_both_channel_orders():
    a = to_mono(np.zeros((2, 500), dtype=np.float32))
    b = to_mono(np.zeros((500, 2), dtype=np.float32))
    assert a.shape == b.shape == (500,)


# --------------------------------------------------------------------------- #
# windowing — the left-padding contract
# --------------------------------------------------------------------------- #
def test_short_clip_is_left_padded_so_speech_ends_at_the_right_edge():
    """The contract the model depends on: the turn boundary is always at the end.

    Right-padding would slide the decision-relevant moment around with clip
    length, which is the whole reason this is left-padded.
    """
    spec = WindowSpec(2.0)
    wave = np.ones(int(0.5 * SAMPLE_RATE), dtype=np.float32)
    out = fit_window(wave, spec)
    assert out.shape == (spec.samples,)
    n_pad = spec.samples - wave.size
    assert np.all(out[:n_pad] == 0.0), "padding is not at the start"
    assert np.all(out[n_pad:] == 1.0), "audio is not flush to the right edge"


def test_long_clip_is_trimmed_from_the_left_keeping_the_tail():
    """The tail carries the endpoint evidence; the head carries little."""
    spec = WindowSpec(1.0)
    wave = np.concatenate(
        [
            np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32),
            np.ones(int(1.0 * SAMPLE_RATE), dtype=np.float32),
        ]
    )
    out = fit_window(wave, spec)
    assert out.shape == (spec.samples,)
    assert np.all(out == 1.0), "trimming discarded the tail instead of the head"


def test_exact_length_is_untouched():
    spec = WindowSpec(1.0)
    wave = np.linspace(-1, 1, spec.samples).astype(np.float32)
    assert np.array_equal(fit_window(wave, spec), wave)


def test_right_padding_is_available_but_not_the_default():
    spec = WindowSpec(1.0, pad_side="right")
    wave = np.ones(100, dtype=np.float32)
    out = fit_window(wave, spec)
    assert np.all(out[:100] == 1.0) and np.all(out[100:] == 0.0)
    assert WindowSpec(1.0).pad_side == "left"


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
def test_peak_normalise_hits_the_target_and_leaves_headroom():
    wave = (np.random.default_rng(0).normal(0, 0.01, 4000)).astype(np.float32)
    out = normalise(wave, "peak")
    peak = float(np.max(np.abs(out)))
    assert peak < 1.0, "no headroom left — an int16 round-trip could clip"
    assert 0.85 < peak < 0.95, f"peak {peak:.3f} is not the -1 dBFS target"


def test_normalise_preserves_shape_of_the_signal():
    """Normalisation must be a scalar multiply, not a nonlinearity: it cannot be
    allowed to change the silence/speech ratio the detector reads."""
    wave = np.array([0.1, -0.2, 0.05, 0.4, 0.0], dtype=np.float32)
    out = normalise(wave, "peak")
    ratio = out[wave != 0] / wave[wave != 0]
    assert np.allclose(ratio, ratio[0]), "normalisation was not a uniform scale"


def test_digital_silence_is_not_amplified():
    """Scaling silence up would only amplify a noise floor into hiss."""
    wave = np.zeros(1000, dtype=np.float32)
    assert np.array_equal(normalise(wave, "peak"), wave)
    assert np.array_equal(normalise(wave, "rms"), wave)


def test_rms_normalise_does_not_clip():
    """RMS scaling overshoots on peaky material; it must clamp, not clip."""
    wave = np.zeros(4000, dtype=np.float32)
    wave[::400] = 0.9  # sparse spikes: low RMS, high peak
    out = normalise(wave, "rms")
    assert float(np.max(np.abs(out))) <= 1.0


def test_normalise_none_is_a_no_op():
    wave = np.array([0.3, -0.7], dtype=np.float32)
    assert np.array_equal(normalise(wave, "none"), wave)


def test_unknown_normalise_mode_raises():
    with pytest.raises(ValueError):
        normalise(np.zeros(10, dtype=np.float32), "loudness")


# --------------------------------------------------------------------------- #
# resampling
# --------------------------------------------------------------------------- #
def test_resample_changes_length_proportionally():
    wave = np.random.default_rng(0).normal(0, 0.1, 44100).astype(np.float32)
    out = resample(wave, 44100, 16000)
    assert abs(out.size - 16000) < 100


def test_resample_is_a_no_op_at_matching_rate():
    wave = np.random.default_rng(0).normal(0, 0.1, 1000).astype(np.float32)
    assert np.array_equal(resample(wave, 16000, 16000), wave)


def test_resample_does_not_alias_a_tone_into_the_speech_band():
    """The reason librosa is used instead of index striding.

    A 15 kHz tone sampled at 44.1 kHz and decimated by slicing folds down to
    roughly 1 kHz — squarely inside the speech band — and the model would learn
    that artefact. A proper anti-aliased resample removes the tone instead.
    """
    sr = 44100
    t = np.arange(sr) / sr
    tone = np.sin(2 * np.pi * 15000 * t).astype(np.float32)
    proper = resample(tone, sr, 16000)
    strided = tone[:: sr // 16000][: proper.size]
    assert rms_dbfs(proper) < rms_dbfs(strided) - 10, (
        "anti-aliased resample should attenuate an out-of-band tone far more "
        "than naive striding does"
    )


# --------------------------------------------------------------------------- #
# order of operations in prepare()
# --------------------------------------------------------------------------- #
def test_prepare_normalises_before_padding_not_after():
    """The bug this prevents: a short clip normalised against its own padding.

    If windowing came first, a 1 s clip padded to 8 s would have its peak
    computed over 7 s of inserted zeros — which does not change a *peak*, but
    does change an RMS. Under RMS normalisation the short clip would come out
    much louder than a long one, and loudness would then correlate with duration.
    """
    spec = WindowSpec(4.0)
    short = np.full(int(0.5 * SAMPLE_RATE), 0.2, dtype=np.float32)
    long = np.full(int(4.0 * SAMPLE_RATE), 0.2, dtype=np.float32)

    s = prepare(short, SAMPLE_RATE, spec, "rms")
    lg = prepare(long, SAMPLE_RATE, spec, "rms")

    s_level = rms_dbfs(s[s != 0])
    l_level = rms_dbfs(lg)
    assert abs(s_level - l_level) < 1.0, (
        f"short clip normalised to {s_level:.1f} dBFS but long clip to "
        f"{l_level:.1f} — normalisation saw the padding"
    )


def test_prepare_output_is_always_the_window_length():
    spec = WindowSpec(1.5)
    for dur in (0.05, 0.5, 1.5, 3.0, 12.0):
        wave = np.random.default_rng(0).normal(0, 0.1, int(dur * SAMPLE_RATE)).astype(np.float32)
        assert prepare(wave, SAMPLE_RATE, spec).shape == (spec.samples,)


# --------------------------------------------------------------------------- #
# framing
# --------------------------------------------------------------------------- #
def test_frame_shapes_and_hop():
    wave = np.arange(1000, dtype=np.float32)
    f = frame(wave, frame_len=100, hop=50)
    assert f.shape == (19, 100)
    assert f[0][0] == 0 and f[1][0] == 50


def test_frame_pads_a_clip_shorter_than_one_frame():
    f = frame(np.ones(10, dtype=np.float32), frame_len=100, hop=50)
    assert f.shape[1] == 100


# --------------------------------------------------------------------------- #
# mel front end
# --------------------------------------------------------------------------- #
def test_mel_frame_and_position_arithmetic():
    """These two numbers must agree with the encoder's expectations exactly."""
    assert mel_frames_for(1.0) == 100  # 10 ms hop
    assert encoder_positions_for(1.0) == 50  # conv stride 2
    assert mel_frames_for(8.0) == 800
    assert encoder_positions_for(8.0) == 400


def test_mel_shape_matches_the_declared_contract():
    spec = WindowSpec(1.0)
    fe = MelFrontEnd(spec)
    wave = np.random.default_rng(0).normal(0, 0.05, spec.samples).astype(np.float32)
    mel = fe(wave)
    assert mel.shape == (1, 80, fe.n_frames)
    assert mel.dtype == np.float32


def test_mel_batches():
    spec = WindowSpec(1.0)
    fe = MelFrontEnd(spec)
    waves = np.random.default_rng(0).normal(0, 0.05, (4, spec.samples)).astype(np.float32)
    assert fe(waves).shape == (4, 80, fe.n_frames)


def test_mel_rejects_wrong_length_rather_than_padding_silently():
    """A silent pad here would hide a missing prepare() call upstream."""
    spec = WindowSpec(1.0)
    fe = MelFrontEnd(spec)
    with pytest.raises(ValueError, match="expected"):
        fe(np.zeros(spec.samples // 2, dtype=np.float32))


def test_mel_rejects_a_window_too_short_to_produce_a_position():
    with pytest.raises(ValueError, match="too short"):
        MelFrontEnd(WindowSpec(0.005))
