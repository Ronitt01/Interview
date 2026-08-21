"""The hard-negative provenance guard.

Why this file is as paranoid as it is. `scripts/error_analysis.py` defaults to
`--cache data/cache/test`, so an index file mined with the defaults points into
the **held-out test set**. Oversampling those indices in training would be
training on test — and the failure is completely silent, because indices are
just integers and they load, index, and train perfectly well. Nothing crashes;
the held-out number simply stops meaning anything, and every result in the
project becomes unpublishable.

So loading is a refusal-by-default operation, and each refusal gets a test:
mismatched cache, resized cache, absent sidecar, out-of-range indices, and
indices that belong to a different split of the right cache.
"""
import json

import numpy as np
import pytest

from src.dataset import HARD_NEGATIVE_META, load_hard_negatives


def write_indices(d, idx, *, cache_dir="data/cache/train", n_clips=1000,
                 sidecar=True, mine_split="train"):
    d.mkdir(parents=True, exist_ok=True)
    p = d / "hard_negative_indices.npy"
    np.save(p, np.asarray(idx, dtype=np.int64))
    if sidecar:
        (d / HARD_NEGATIVE_META).write_text(
            json.dumps({
                "cache_dir": cache_dir, "n_clips": n_clips,
                "mine_split": mine_split, "count": len(idx),
                "usable_for_training": mine_split == "train",
            }),
            encoding="utf-8",
        )
    return p


# --------------------------------------------------------------------------- #
# the refusals
# --------------------------------------------------------------------------- #
def test_indices_mined_on_the_test_cache_are_refused(tmp_path):
    """The one that matters. This is what error_analysis.py writes by default."""
    p = write_indices(tmp_path, [1, 2, 3], cache_dir="data/cache/test",
                      mine_split="all")
    with pytest.raises(ValueError, match="mined on"):
        load_hard_negatives(p, cache_dir="data/cache/train", n_clips=1000)


def test_an_index_file_with_no_sidecar_is_refused(tmp_path):
    """No sidecar means the cache is unknown, and the default cache is the test
    set. Guessing is the one thing that must not happen."""
    p = write_indices(tmp_path, [1, 2, 3], sidecar=False)
    with pytest.raises(FileNotFoundError, match=HARD_NEGATIVE_META):
        load_hard_negatives(p, cache_dir="data/cache/train", n_clips=1000)


def test_the_refusal_message_says_why_it_matters(tmp_path):
    """A guard whose message does not explain itself gets worked around."""
    p = write_indices(tmp_path, [1], sidecar=False)
    with pytest.raises(FileNotFoundError) as exc:
        load_hard_negatives(p, cache_dir="data/cache/train", n_clips=1000)
    assert "held-out" in str(exc.value)


def test_a_rebuilt_cache_of_a_different_size_is_refused(tmp_path):
    """Indices are positions. Re-cache with different filters and position 8,412
    is a different clip, so the mined set silently becomes arbitrary."""
    p = write_indices(tmp_path, [1, 2, 3], n_clips=40000)
    with pytest.raises(ValueError, match="cache of 40000 clips"):
        load_hard_negatives(p, cache_dir="data/cache/train", n_clips=39000)


def test_out_of_range_indices_are_refused(tmp_path):
    p = write_indices(tmp_path, [1, 2, 99999], n_clips=1000)
    with pytest.raises(ValueError, match="out of range"):
        load_hard_negatives(p, cache_dir="data/cache/train", n_clips=1000)


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="no hard-negative index file"):
        load_hard_negatives(tmp_path / "nope.npy",
                            cache_dir="data/cache/train", n_clips=1000)


# --------------------------------------------------------------------------- #
# what it allows
# --------------------------------------------------------------------------- #
def test_correct_provenance_loads(tmp_path):
    p = write_indices(tmp_path, [4, 9, 16], n_clips=1000)
    got = load_hard_negatives(p, cache_dir="data/cache/train", n_clips=1000)
    assert got.tolist() == [4, 9, 16]
    assert got.dtype == np.int64


def test_the_cache_is_matched_by_name_not_by_absolute_path(tmp_path):
    """Colab mounts the repo at /content and a laptop does not, so an absolute
    path comparison would refuse every correctly-mined file after a move."""
    p = write_indices(tmp_path, [1, 2], cache_dir="/content/proj/data/cache/train",
                      n_clips=1000)
    got = load_hard_negatives(
        p, cache_dir="C:/Users/x/proj/data/cache/train", n_clips=1000
    )
    assert got.tolist() == [1, 2]


def test_allowed_restricts_to_a_split(tmp_path):
    """The second guard. Even a correctly-mined file must not pull a validation
    row into training -- that would leak val into train and inflate val scores.
    """
    p = write_indices(tmp_path, [1, 2, 3, 4, 5], n_clips=1000)
    train_idx = np.array([1, 3, 5, 7], dtype=np.int64)
    got = load_hard_negatives(p, cache_dir="data/cache/train", n_clips=1000,
                              allowed=train_idx)
    assert got.tolist() == [1, 3, 5]        # 2 and 4 are validation rows


def test_an_empty_result_is_returned_rather_than_raised(tmp_path):
    """If nothing survives the split filter, training should say so and carry on
    unchanged -- not abort. There is nothing wrong with the file."""
    p = write_indices(tmp_path, [2, 4], n_clips=1000)
    got = load_hard_negatives(p, cache_dir="data/cache/train", n_clips=1000,
                             allowed=np.array([1, 3, 5], dtype=np.int64))
    assert got.size == 0


def test_an_empty_index_file_is_not_an_error(tmp_path):
    p = write_indices(tmp_path, [], n_clips=1000)
    assert load_hard_negatives(p, cache_dir="data/cache/train",
                               n_clips=1000).size == 0


def test_duplicate_indices_are_preserved(tmp_path):
    """Oversampling is the point, so de-duplicating here would quietly undo the
    repeat count the caller asked for."""
    p = write_indices(tmp_path, [7, 7, 8], n_clips=1000)
    got = load_hard_negatives(p, cache_dir="data/cache/train", n_clips=1000)
    assert got.tolist() == [7, 7, 8]


# --------------------------------------------------------------------------- #
# the config side
# --------------------------------------------------------------------------- #
def test_hard_negatives_are_off_by_default():
    from training.train import TrainConfig

    assert TrainConfig().hard_negative_file is None


def test_the_e10_config_points_at_a_train_mined_path():
    """E10 is the config that consumes the indices. If its path ever points at a
    test-mined directory the guard would fire at run time -- better to pin it
    here, where it costs nothing to notice."""
    from pathlib import Path

    from training.train import TrainConfig

    root = Path(__file__).resolve().parents[1]
    cfg = TrainConfig.load(root / "configs" / "e10_hard_negatives.yaml")
    assert cfg.hard_negative_file
    assert "error_analysis_train" in cfg.hard_negative_file
    assert "test" not in Path(cfg.hard_negative_file).parts
    assert cfg.hard_negative_repeat >= 1
    # E10 is a candidate like any other, so it must not score the held-out set.
    assert cfg.test_cache_dir is None
