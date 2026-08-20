"""Download the Smart Turn corpus into a local cache. Day 1 / Day 2.

    # smoke test the whole path on a few hundred clips (minutes, not hours)
    python scripts/prepare_data.py --split test --max-rows 300

    # the working subset: English + the three Indic languages, capped
    python scripts/prepare_data.py --split train --languages eng hin ben mar \
        --max-rows 40000
    python scripts/prepare_data.py --split test  --languages eng hin ben mar

    # everything (41.4 GB of source audio; ~35 GB of cache)
    python scripts/prepare_data.py --split train --all-languages

Why the default is a subset rather than the full corpus: the published train set
is 270,946 rows / 41.4 GB. A free Colab session that spends four hours
downloading and then disconnects has produced nothing. Capping rows and
filtering language gets a complete, honest pipeline running first; the cap is a
flag, so scaling up later is one argument, and the report states which cap
produced which row.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset import (  # noqa: E402
    TEST_DATASET,
    TRAIN_DATASET,
    WaveCache,
    build_cache,
    describe_corpus_sizes,
    estimate_cache_gb,
    sanity_check,
)

# The Indic languages actually present in the corpus. Established by reading the
# dataset card rather than assumed — the plan expected none to be there.
INDIC = ("hin", "ben", "mar")
DEFAULT_LANGUAGES = ("eng",) + INDIC


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=("train", "test"), required=True)
    ap.add_argument("--out", default=None, help="cache dir (default data/cache/<split>)")
    ap.add_argument("--languages", nargs="*", default=None,
                    help=f"language codes to keep (default: {' '.join(DEFAULT_LANGUAGES)})")
    ap.add_argument("--all-languages", action="store_true", help="keep all 23")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--human-only", action="store_true",
                    help="keep only synthetic=False rows (the honest headline slice)")
    ap.add_argument("--synthetic-only", action="store_true")
    ap.add_argument("--no-streaming", action="store_true",
                    help="download fully before reading — faster per row, needs the disk")
    ap.add_argument("--info", action="store_true", help="print corpus sizes and exit")
    args = ap.parse_args(argv)

    if args.info:
        print(describe_corpus_sizes())
        return 0

    if args.human_only and args.synthetic_only:
        ap.error("--human-only and --synthetic-only are mutually exclusive")

    hf = TRAIN_DATASET if args.split == "train" else TEST_DATASET
    out = Path(args.out or f"data/cache/{args.split}")
    languages = None if args.all_languages else tuple(args.languages or DEFAULT_LANGUAGES)
    include_synthetic = False if args.human_only else (True if args.synthetic_only else None)

    print(f"\n  source     : {hf}")
    print(f"  cache dir  : {out}")
    print(f"  languages  : {'all' if languages is None else ' '.join(languages)}")
    print(f"  max rows   : {args.max_rows or 'no cap'}")
    print(f"  synthetic  : {'both' if include_synthetic is None else include_synthetic}")
    print(f"  streaming  : {not args.no_streaming}")
    if args.max_rows:
        print(f"  est. cache : ~{estimate_cache_gb(args.max_rows):.2f} GB")
    print()

    stats = build_cache(
        out_dir=out,
        hf_dataset=hf,
        split="train",  # both published datasets expose a single "train" split
        languages=languages,
        max_rows=args.max_rows,
        max_seconds=args.max_seconds,
        streaming=not args.no_streaming,
        include_synthetic=include_synthetic,
    )
    print(f"\n  {stats}\n")
    print(sanity_check(WaveCache(out)))
    print(f"\n  cache written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
