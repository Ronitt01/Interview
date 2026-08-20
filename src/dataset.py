"""The Smart Turn corpus: streaming download, on-disk cache, torch Dataset.

The size problem, and why the cache is shaped the way it is.

``smart-turn-data-v3.2-train`` is 41.4 GB across 270,946 rows;
``smart-turn-data-v3.1-test`` is 4.3 GB across 31,473. A free Colab instance has
roughly 70-100 GB of disk but a session that dies mid-download has to start over,
and the audio arrives as compressed blobs that have to be decoded on every epoch
if we keep them in that form. Both problems are solved by one pass:

    stream from the Hub -> decode -> mono -> resample to 16 kHz -> int16
      -> append to one flat memmap, record (offset, length) + metadata

What gets cached is the **resampled waveform, at its natural length** — not the
mel spectrogram and not a fixed-length window. That choice is what keeps the
experiment matrix open:

* caching mel would freeze the window length and the mel config, killing E2 and
  the sample-rate arm;
* caching a fixed 8 s window would waste 2x the disk on a corpus whose median
  clip is a few seconds, and would make the 0.5-2.0 s window sweep a crop of a
  pad rather than of real audio;
* caching frozen-encoder embeddings would be smallest of all (~1.5 KB/clip
  pooled) and is genuinely tempting for E1, but it forecloses augmentation and
  every window experiment. :func:`cache_embeddings` exists for the case where
  only the head is being swept, and is documented as the shortcut it is.

int16 rather than float32 halves the cache for no measurable cost: the source
audio was almost certainly 16-bit to begin with, and 16-bit covers 96 dB of
range against speech that occupies maybe 60.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from . import LABEL_COLUMN, SAMPLE_RATE
from .audio import WindowSpec, normalise, resample, to_mono
from .augment import AugmentConfig, augment
from .features import MelFrontEnd

TRAIN_DATASET = "pipecat-ai/smart-turn-data-v3.2-train"
TEST_DATASET = "pipecat-ai/smart-turn-data-v3.1-test"

META_COLUMNS = (
    "id",
    "language",
    "endpoint_bool",
    "midfiller",
    "endfiller",
    "synthetic",
    "dataset",
    "audioduration",
)

INT16_SCALE = 32767.0


# --------------------------------------------------------------------------- #
# cache writing
# --------------------------------------------------------------------------- #
@dataclass
class CacheStats:
    rows: int = 0
    skipped: int = 0
    samples: int = 0
    seconds: float = 0.0
    reasons: dict = field(default_factory=dict)

    def note_skip(self, reason: str) -> None:
        self.skipped += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def __str__(self) -> str:
        gb = self.samples * 2 / 1e9
        return (
            f"{self.rows:,d} clips, {self.seconds / 3600:.2f} h audio, "
            f"{gb:.2f} GB cache, {self.skipped:,d} skipped {self.reasons or ''}"
        )


def build_cache(
    out_dir: str | Path,
    hf_dataset: str,
    split: str = "train",
    languages: Sequence[str] | None = None,
    max_rows: int | None = None,
    max_seconds: float = 30.0,
    min_seconds: float = 0.15,
    streaming: bool = True,
    include_synthetic: bool | None = None,
    progress_every: int = 2000,
) -> CacheStats:
    """One streaming pass from the Hub into a local cache.

    Parameters
    ----------
    languages:
        Keep only these language codes (e.g. ``["eng", "hin"]``). ``None`` keeps
        all 23. Filtering here rather than at training time is what makes the
        corpus fit: an English+Hindi subset is a fraction of the full 41 GB.
    max_rows:
        Stop after this many *kept* rows. The single most useful knob for getting
        a full pipeline running end-to-end before committing hours to a download.
    max_seconds:
        Clips longer than this are truncated from the *left* (keeping the tail,
        where the endpoint evidence is), not skipped — the corpus goes up to
        32.6 s and discarding those rows would bias the duration distribution.
    include_synthetic:
        ``None`` keeps both. ``False`` keeps only human-recorded audio, which is
        the honest slice to report a headline number on given how much of this
        corpus is TTS.
    streaming:
        Stream rather than download-then-read. Slower per row but never needs the
        full 41 GB on disk, and survives a session restart with only the rows
        already written.
    """
    from datasets import load_dataset

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wave_path = out / "waves.i16"
    meta_rows: list[dict] = []
    offsets: list[int] = []
    lengths: list[int] = []
    stats = CacheStats()

    lang_filter = set(languages) if languages else None
    max_samples = int(max_seconds * SAMPLE_RATE)
    min_samples = int(min_seconds * SAMPLE_RATE)

    ds = load_dataset(hf_dataset, split=split, streaming=streaming)

    # Decode the audio ourselves rather than letting `datasets` do it.
    #
    # datasets >= 5 routes Audio decoding through torchcodec, which needs a
    # matching FFmpeg install and is awkward to get onto Windows and onto some
    # Colab images. Turning decoding off hands us the raw encoded bytes, which
    # soundfile reads directly — one less native dependency, identical result,
    # and it keeps the decode path the same one the demo uses for uploads.
    try:
        from datasets import Audio

        ds = ds.cast_column("audio", Audio(decode=False))
        raw_audio = True
    except Exception:
        raw_audio = False

    cursor = 0
    with wave_path.open("wb") as sink:
        for row in ds:
            if max_rows is not None and stats.rows >= max_rows:
                break

            lang = row.get("language")
            if lang_filter is not None and lang not in lang_filter:
                stats.note_skip("language")
                continue
            if include_synthetic is not None:
                if bool(row.get("synthetic", False)) != include_synthetic:
                    stats.note_skip("synthetic")
                    continue

            audio = row.get("audio")
            if audio is None:
                stats.note_skip("no_audio")
                continue
            try:
                wave, sr = _decode_audio_field(audio, raw_audio)
                wave = resample(wave, sr, SAMPLE_RATE)
            except Exception as exc:  # a handful of corrupt blobs is normal
                stats.note_skip(f"decode:{type(exc).__name__}")
                continue

            if wave.size < min_samples:
                stats.note_skip("too_short")
                continue
            if wave.size > max_samples:
                wave = wave[-max_samples:]

            pcm = np.clip(wave, -1.0, 1.0)
            sink.write((pcm * INT16_SCALE).astype("<i2").tobytes())

            offsets.append(cursor)
            lengths.append(int(pcm.size))
            cursor += int(pcm.size)

            meta_rows.append(
                {
                    k: (row.get(k) if k != "audioduration" else float(pcm.size) / SAMPLE_RATE)
                    for k in META_COLUMNS
                }
            )
            stats.rows += 1
            stats.samples += int(pcm.size)
            stats.seconds += pcm.size / SAMPLE_RATE

            if progress_every and stats.rows % progress_every == 0:
                print(f"  cached {stats.rows:,d} clips  ({stats.seconds / 3600:.2f} h)", flush=True)

    if stats.rows == 0:
        raise RuntimeError(
            f"cached zero rows from {hf_dataset}. Filters were "
            f"languages={languages}, include_synthetic={include_synthetic}. "
            f"Skips: {stats.reasons}"
        )

    np.save(out / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    np.save(out / "lengths.npy", np.asarray(lengths, dtype=np.int64))
    _write_meta(out / "meta.jsonl", meta_rows)
    (out / "cache_info.json").write_text(
        json.dumps(
            {
                "hf_dataset": hf_dataset,
                "split": split,
                "sample_rate": SAMPLE_RATE,
                "dtype": "int16",
                "rows": stats.rows,
                "samples": stats.samples,
                "hours": stats.seconds / 3600.0,
                "languages": list(languages) if languages else "all",
                "include_synthetic": include_synthetic,
                "max_seconds": max_seconds,
                "min_seconds": min_seconds,
                "skipped": stats.skipped,
                "skip_reasons": stats.reasons,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return stats


def _decode_audio_field(audio, raw: bool) -> tuple[np.ndarray, int]:
    """Get ``(mono_float32, sample_rate)`` from a datasets audio cell.

    Handles both shapes so the cache builder works whether or not the Audio
    column could be switched to ``decode=False``:

    * ``decode=False`` → ``{"bytes": ..., "path": ...}``, decoded by soundfile;
    * ``decode=True``  → ``{"array": ..., "sampling_rate": ...}``.
    """
    import io

    if raw or "array" not in audio:
        blob = audio.get("bytes")
        if blob:
            import soundfile as sf

            wave, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=False)
            return to_mono(wave), int(sr)
        path = audio.get("path")
        if not path:
            raise ValueError("audio cell has neither bytes nor path")
        import soundfile as sf

        wave, sr = sf.read(path, dtype="float32", always_2d=False)
        return to_mono(wave), int(sr)

    return to_mono(np.asarray(audio["array"], dtype=np.float32)), int(audio["sampling_rate"])


def _write_meta(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(_jsonable(r)) + "\n")


def _jsonable(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, (np.bool_,)):
            v = bool(v)
        elif isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v)
        out[k] = v
    return out


# --------------------------------------------------------------------------- #
# cache reading
# --------------------------------------------------------------------------- #
class WaveCache:
    """Memmapped reader over a cache written by :func:`build_cache`."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.dir = Path(cache_dir)
        missing = [
            f
            for f in ("waves.i16", "offsets.npy", "lengths.npy", "meta.jsonl")
            if not (self.dir / f).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"cache at {self.dir} is missing {missing}. "
                "Run scripts/prepare_data.py first."
            )
        self.offsets = np.load(self.dir / "offsets.npy")
        self.lengths = np.load(self.dir / "lengths.npy")
        self.meta = [
            json.loads(line)
            for line in (self.dir / "meta.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not (len(self.offsets) == len(self.lengths) == len(self.meta)):
            raise ValueError(
                f"cache is inconsistent: {len(self.offsets)} offsets, "
                f"{len(self.lengths)} lengths, {len(self.meta)} meta rows"
            )
        # mode="r" so a DataLoader worker cannot corrupt the cache, and so the
        # OS page cache is shared between workers instead of duplicated.
        self._waves = np.memmap(self.dir / "waves.i16", dtype="<i2", mode="r")
        self.info = json.loads((self.dir / "cache_info.json").read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.offsets)

    def wave(self, i: int) -> np.ndarray:
        """Float32 waveform for row ``i``, at its natural length."""
        o, n = int(self.offsets[i]), int(self.lengths[i])
        return self._waves[o : o + n].astype(np.float32) / INT16_SCALE

    def label(self, i: int) -> int:
        return int(bool(self.meta[i][LABEL_COLUMN]))

    def labels(self) -> np.ndarray:
        return np.asarray([self.label(i) for i in range(len(self))], dtype=np.int64)

    def column(self, name: str) -> list:
        return [m.get(name) for m in self.meta]

    def indices_where(self, **equals) -> np.ndarray:
        """Row indices matching every ``column=value`` pair.

        Used for the per-language and per-filler slices the report breaks out —
        ``cache.indices_where(language="hin")``, ``midfiller=True``, and so on.
        """
        keep = []
        for i, m in enumerate(self.meta):
            if all(m.get(k) == v for k, v in equals.items()):
                keep.append(i)
        return np.asarray(keep, dtype=np.int64)

    def summary(self) -> str:
        y = self.labels()
        langs = {}
        for m in self.meta:
            langs[m.get("language")] = langs.get(m.get("language"), 0) + 1
        top = sorted(langs.items(), key=lambda kv: -kv[1])[:6]
        hours = float(self.lengths.sum()) / SAMPLE_RATE / 3600.0
        return (
            f"{len(self):,d} clips, {hours:.2f} h, "
            f"positive rate {y.mean():.3f}\n"
            f"  languages: {', '.join(f'{k}={v:,d}' for k, v in top)}"
            f"{' ...' if len(langs) > 6 else ''}"
        )


# --------------------------------------------------------------------------- #
# torch Dataset
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# random-offset cropping
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RandomOffsetConfig:
    """Move a training window's right edge off the annotated boundary.

    **The problem this exists to fix.** Every cached clip ends at a moment an
    annotator chose: a positive clip ends at a real turn end, a negative clip
    ends at a real mid-utterance point. So in training the window's right edge is
    always a curated instant. In streaming it is wherever the hop happens to
    land, which is almost never either of those. That mismatch was measured and
    it is large: the same model at the same threshold went from a 0.033
    false-interruption rate on clips to 0.460 streamed.

    **The fix, and why it is not the forbidden crop.** :mod:`src.augment` bans
    cropping from the end, because removing the tail of a positive clip destroys
    the endpoint evidence while the label still says "ended" -- the label stops
    describing the audio. This crops from the end *and re-derives the label from
    where the crop actually lands*, which is the opposite operation: it
    manufactures correctly-labelled negatives at exactly the window alignments
    streaming produces and training never had.

    Attributes
    ----------
    enabled:
        Off by default. E1 and the E2 sweep must stay reproducible unchanged.
    prob:
        Fraction of training examples given a shifted right edge. The rest keep
        their original alignment, so the model still sees curated boundaries.
    max_shift_ms:
        Upper bound on how far back the right edge moves, sampled uniformly from
        [0, max_shift_ms].
    tolerance_ms:
        A shift this small leaves the edge effectively still at the boundary, so
        the original label is kept. Related to
        :data:`src.streaming.EARLY_FIRE_TOLERANCE_MS` but deliberately tighter:
        that one bounds how early a *fire* may be and still be forgiven, this one
        bounds how early a *training window* may end and still be an endpoint.
    min_keep_ms:
        Never crop a clip shorter than this. A 40 ms fragment carries no prosody
        and would just be noise with a confident label.

    Any shift beyond ``tolerance_ms`` produces a **negative**, whatever the
    clip's own label was -- that is the entire teaching signal. It also lowers
    the realised positive rate, which :mod:`training.train` prints so the change
    is visible rather than silent; ``use_pos_weight`` then compensates in the
    loss.
    """

    enabled: bool = False
    prob: float = 0.5
    max_shift_ms: float = 4000.0
    tolerance_ms: float = 150.0
    min_keep_ms: float = 500.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.prob <= 1.0:
            raise ValueError(f"prob must be in [0, 1], got {self.prob}")
        if self.max_shift_ms < 0 or self.tolerance_ms < 0 or self.min_keep_ms <= 0:
            raise ValueError("random-offset durations must be non-negative")
        if self.enabled and self.tolerance_ms >= self.max_shift_ms:
            # Every shift would land inside the tolerance, so every example would
            # keep its label and the mechanism would be a silent no-op that looks
            # like it is working.
            raise ValueError(
                f"tolerance_ms ({self.tolerance_ms}) >= max_shift_ms "
                f"({self.max_shift_ms}): no example would ever be relabelled"
            )


def apply_random_offset(
    wave: np.ndarray,
    label: int,
    cfg: RandomOffsetConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, float]:
    """Crop to an earlier right edge. Returns ``(wave, label, shift_ms)``.

    Pure, with the generator injected, so it is testable without a cache.
    ``shift_ms == 0.0`` means the example was left alone.
    """
    if not cfg.enabled or rng.random() >= cfg.prob:
        return wave, int(label), 0.0

    n = int(wave.size)
    min_keep = int(round(cfg.min_keep_ms * SAMPLE_RATE / 1000.0))
    if n <= min_keep:
        return wave, int(label), 0.0

    max_shift = min(int(round(cfg.max_shift_ms * SAMPLE_RATE / 1000.0)), n - min_keep)
    if max_shift <= 0:
        return wave, int(label), 0.0

    shift = int(rng.integers(0, max_shift + 1))
    if shift == 0:
        return wave, int(label), 0.0

    shift_ms = shift * 1000.0 / SAMPLE_RATE
    # Beyond the tolerance the crop no longer ends at the boundary, so whatever
    # the clip was annotated as, this particular window is mid-utterance.
    new_label = int(label) if shift_ms <= cfg.tolerance_ms else 0
    return wave[: n - shift], new_label, shift_ms


# --------------------------------------------------------------------------- #
# hard-negative index files
# --------------------------------------------------------------------------- #
HARD_NEGATIVE_META = "hard_negative_meta.json"


def load_hard_negatives(
    path: str | Path,
    cache_dir: str | Path,
    n_clips: int,
    allowed: Sequence[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Load mined hard-negative indices, refusing to load the wrong ones.

    **Why this is defensive rather than a one-line np.load.** Indices are
    positions into one specific cache. ``scripts/error_analysis.py`` defaults to
    mining on ``data/cache/test``, so a file mined with the defaults indexes the
    *held-out test set* -- and oversampling those into training would be training
    on test, silently invalidating every number in the project. The failure mode
    is invisible: the indices are just integers and they load fine.

    So a sidecar written at mining time records which cache the indices came
    from, and this refuses to proceed unless it matches. ``allowed`` further
    restricts to a split, so mined indices cannot pull validation rows into
    training either.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no hard-negative index file at {p}")

    idx = np.asarray(np.load(p), dtype=np.int64).ravel()

    meta_path = p.parent / HARD_NEGATIVE_META
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{p} has no {HARD_NEGATIVE_META} beside it, so which cache these "
            "indices belong to is unknown. Re-mine with a current "
            "scripts/error_analysis.py, which writes the sidecar. Refusing to "
            "guess: if these were mined on the test cache, using them would be "
            "training on held-out data."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    mined_on = str(meta.get("cache_dir", ""))
    want = str(cache_dir)
    if Path(mined_on).name != Path(want).name:
        raise ValueError(
            f"hard negatives were mined on {mined_on!r} but training reads "
            f"{want!r}. Indices are cache-relative and are not transferable "
            "between caches."
        )
    if int(meta.get("n_clips", -1)) != int(n_clips):
        raise ValueError(
            f"hard negatives were mined on a cache of {meta.get('n_clips')} "
            f"clips; this cache has {n_clips}. The cache has been rebuilt, so "
            "the indices no longer point at the clips they were chosen for."
        )

    if idx.size and (idx.min() < 0 or idx.max() >= n_clips):
        raise ValueError(
            f"hard-negative indices out of range for a {n_clips}-clip cache "
            f"(min {idx.min()}, max {idx.max()})"
        )

    if allowed is not None:
        idx = idx[np.isin(idx, np.asarray(allowed, dtype=np.int64))]

    return idx


class TurnDataset:
    """Log-mel + label, for one split.

    Not a subclass of ``torch.utils.data.Dataset`` by inheritance — it satisfies
    the protocol (``__len__``/``__getitem__``) without importing torch at module
    scope, so the EDA notebook and the baseline scripts can use it on a machine
    where torch is not installed.
    """

    def __init__(
        self,
        cache: WaveCache,
        indices: Sequence[int] | np.ndarray,
        window: WindowSpec,
        normalise_mode: str = "peak",
        augment_cfg: AugmentConfig | None = None,
        n_mels: int = 80,
        epoch: int = 0,
        return_meta: bool = False,
        offset_cfg: RandomOffsetConfig | None = None,
    ) -> None:
        self.cache = cache
        self.indices = np.asarray(indices, dtype=np.int64)
        self.window = window
        self.normalise_mode = normalise_mode
        self.augment_cfg = augment_cfg or AugmentConfig()
        self.offset_cfg = offset_cfg or RandomOffsetConfig()
        self.front_end = MelFrontEnd(window, n_mels=n_mels)
        self.epoch = epoch
        self.return_meta = return_meta

    def __len__(self) -> int:
        return int(self.indices.size)

    def set_epoch(self, epoch: int) -> None:
        """Advance the augmentation seed so epochs differ but stay reproducible."""
        self.epoch = int(epoch)

    def __getitem__(self, i: int):
        row = int(self.indices[i])
        wave = self.cache.wave(row)
        label = self.cache.label(row)

        # Augment on the natural-length waveform, before windowing: a masked span
        # or a speed change applied after padding would land partly on zeros.
        if self.augment_cfg.enabled:
            wave = augment(wave, self.augment_cfg, seed=row * 8191 + self.epoch)

        # Random offset comes after augmentation and before normalise/window,
        # for the same reason: it has to act on real audio, not on padding. The
        # seed multiplier differs from the augmentation one so the two streams
        # are not correlated across rows.
        if self.offset_cfg.enabled:
            rng = np.random.default_rng(row * 7919 + self.epoch * 104729)
            wave, label, _shift_ms = apply_random_offset(
                wave, label, self.offset_cfg, rng
            )

        wave = normalise(wave, self.normalise_mode)
        from .audio import fit_window

        wave = fit_window(wave, self.window)
        mel = self.front_end(wave)[0]  # (n_mels, n_frames)

        if self.return_meta:
            return mel, label, self.cache.meta[row]
        return mel, label

    def labels(self, effective: bool = False) -> np.ndarray:
        """Labels for this split.

        ``effective=True`` replays the random-offset relabelling for the current
        epoch, without decoding any audio, and is what class balance must be
        computed from once offsets are on. Reading the raw cache labels there
        would report a balanced ~0.50 positive rate while the model was actually
        being shown ~0.26 -- and ``pos_weight`` would be wrong in the direction
        that matters. Returns raw cache labels when offsets are disabled, so
        every existing caller is unaffected.
        """
        raw = np.asarray(
            [self.cache.label(int(i)) for i in self.indices], dtype=np.int64
        )
        if not effective or not self.offset_cfg.enabled:
            return raw

        out = raw.copy()
        for k, row in enumerate(self.indices):
            row = int(row)
            rng = np.random.default_rng(row * 7919 + self.epoch * 104729)
            n = int(self.cache.lengths[row])
            _w, lab, _s = apply_random_offset(
                np.empty(n, dtype=np.float32), int(raw[k]), self.offset_cfg, rng
            )
            out[k] = lab
        return out

    def class_weights(self) -> tuple[float, float]:
        """``(weight_neg, weight_pos)`` inversely proportional to frequency.

        Returned rather than applied, so the caller decides between weighting the
        loss and balancing the sampler — they are not equivalent, and which one
        is right depends on how severe the imbalance turns out to be.
        """
        y = self.labels(effective=True)
        n = y.size
        pos = int(y.sum())
        neg = n - pos
        if pos == 0 or neg == 0:
            return 1.0, 1.0
        return n / (2.0 * neg), n / (2.0 * pos)

    def pos_weight(self) -> float:
        """``neg/pos``, the value ``BCEWithLogitsLoss(pos_weight=...)`` wants."""
        y = self.labels(effective=True)
        pos = int(y.sum())
        return float((y.size - pos) / pos) if pos else 1.0


def collate(batch):
    """Stack into ``(mel, label)`` tensors."""
    import torch

    mels = np.stack([b[0] for b in batch]).astype(np.float32)
    labels = np.asarray([b[1] for b in batch], dtype=np.float32)
    return torch.from_numpy(mels), torch.from_numpy(labels)


def make_loader(
    ds: TurnDataset,
    batch_size: int = 32,
    shuffle: bool = False,
    balanced: bool = False,
    num_workers: int = 0,
    seed: int = 0,
):
    """DataLoader with an optional class-balanced sampler.

    ``num_workers`` defaults to 0 because the cache is a memmap: the OS page
    cache already does the prefetching that workers would, and on Windows each
    worker re-imports the module and re-opens the memmap, which for this access
    pattern costs more than it saves. Raise it on Linux/Colab if profiling says so.
    """
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    sampler = None
    if balanced:
        y = ds.labels()
        w_neg, w_pos = ds.class_weights()
        weights = np.where(y == 1, w_pos, w_neg).astype(np.float64)
        g = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights),
            num_samples=len(ds),
            replacement=True,
            generator=g,
        )
        shuffle = False  # mutually exclusive with a sampler

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def split_indices(
    cache: WaveCache,
    fractions: dict[str, float],
    group_keys: Sequence[str] = ("dataset",),
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], "object"]:
    """Group-aware split over a cache. Returns ``(indices_by_split, report)``."""
    from .splits import assign_groups, build_report, group_key

    groups: dict[str, list[int]] = {}
    for i, m in enumerate(cache.meta):
        groups.setdefault(group_key(m, group_keys), []).append(i)

    sizes = {
        g: (len(rows), sum(int(bool(cache.meta[i][LABEL_COLUMN])) for i in rows))
        for g, rows in groups.items()
    }
    assignment = assign_groups(sizes, fractions, seed=seed)

    out: dict[str, list[int]] = {name: [] for name in fractions}
    for g, rows in groups.items():
        out[assignment[g]].extend(rows)

    report = build_report(cache.meta, assignment, group_keys, LABEL_COLUMN)
    return {k: np.asarray(sorted(v), dtype=np.int64) for k, v in out.items()}, report


def iter_batched(seq: Sequence, size: int) -> Iterator[list]:
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


def cache_embeddings(
    model,
    cache: WaveCache,
    indices: Sequence[int],
    window: WindowSpec,
    out_path: str | Path,
    batch_size: int = 64,
    device: str = "cpu",
) -> Path:
    """Precompute pooled encoder embeddings for a *frozen* encoder.

    The shortcut described in the module docstring: ~1.5 KB per clip instead of
    ~128 KB, which turns a head sweep from minutes-per-epoch into
    seconds-per-epoch. Valid only while the encoder is frozen and the window and
    augmentation are fixed — so it is used for the head arm of the matrix (E3,
    E4) and explicitly *not* for E2, E5, or E7.
    """
    import torch

    model = model.to(device).eval()
    front = MelFrontEnd(window)
    from .audio import fit_window

    embs: list[np.ndarray] = []
    with torch.no_grad():
        for chunk in iter_batched(list(indices), batch_size):
            waves = np.stack(
                [fit_window(normalise(cache.wave(int(i)), "peak"), window) for i in chunk]
            )
            mel = torch.from_numpy(front(waves)).to(device)
            hidden = model.encoder(mel).last_hidden_state
            embs.append(hidden.mean(dim=1).cpu().numpy().astype(np.float32))

    arr = np.concatenate(embs, axis=0)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, arr)
    return out


def estimate_cache_gb(rows: int, mean_seconds: float = 4.0) -> float:
    """Disk a cache will need. Called by the prepare script before it commits."""
    return rows * mean_seconds * SAMPLE_RATE * 2 / 1e9


def describe_corpus_sizes() -> str:
    """The published corpus sizes, so the plan's disk math is in the repo."""
    return (
        f"{TRAIN_DATASET}: 270,946 rows / 41.4 GB\n"
        f"{TEST_DATASET}:  31,473 rows /  4.3 GB\n"
        "A full-corpus cache at 16 kHz int16 is roughly "
        f"{estimate_cache_gb(270946):.0f} GB, which is why the default "
        "configuration filters by language and caps rows."
    )


def sanity_check(cache: WaveCache, n: int = 5) -> str:
    """Human-readable spot check. Printed by the prepare script when it finishes."""
    lines = [cache.summary(), ""]
    step = max(1, len(cache) // max(n, 1))
    for i in range(0, min(len(cache), n * step), step):
        m = cache.meta[i]
        w = cache.wave(i)
        lines.append(
            f"  [{i:>6d}] {m.get('language','?'):>4s} "
            f"label={int(bool(m[LABEL_COLUMN]))} "
            f"dur={w.size / SAMPLE_RATE:5.2f}s "
            f"peak={np.max(np.abs(w)):.3f} "
            f"src={str(m.get('dataset'))[:24]}"
        )
    return "\n".join(lines)


def n_batches(n_rows: int, batch_size: int) -> int:
    return int(math.ceil(n_rows / batch_size))
