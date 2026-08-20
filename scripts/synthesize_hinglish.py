"""Day 8 — render the Hinglish stress set with Sarvam Bulbul TTS.

    python scripts/synthesize_hinglish.py                 # all phrases x all speakers
    python scripts/synthesize_hinglish.py --speakers anushka --limit 8
    python scripts/synthesize_hinglish.py --pause-variants  # + inserted pauses

Requires ``SARVAM_API_KEY`` in ``.env``. This costs API credits — roughly one
short TTS call per clip — so ``--limit`` exists and ``--dry-run`` lists what
would be generated without calling anything.

Carried over from the voice-agent work's ``tests/language_matrix.py``: Bulbul is
**script-driven**, not language-driven, so Latin-script Hinglish and Devanagari
both voice correctly under ``hi-IN``. That is what makes a code-switched clip set
possible without a separate model per language.

**The honesty requirement.** TTS clips are clean and evenly paced; real hesitation
is neither. Every number computed on this set is an *upper bound* and the manifest
records ``synthetic: true`` on every row so that no downstream script can quietly
average these into a headline figure. ``--pause-variants`` narrows the gap a
little by splicing real silence into the middle of an utterance, which is the one
prosodic feature of hesitation that can be faked convincingly.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import SAMPLE_RATE  # noqa: E402
from src.audio import decode_bytes, resample  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "data" / "hinglish"))
from phrases import (  # noqa: E402
    ALL_PHRASES,
    BULBUL_V3_SPEAKERS,
    SPEAKERS,
    Phrase,
    summary,
)

BULBUL_MODEL = "bulbul:v3"

# Pause lengths spliced into the middle of an utterance. 250 and 400 ms sit on
# either side of the 300-350 ms region where a thinking pause and a turn-final
# pause become hard to tell apart — which is exactly the region a detector has
# to get right.
PAUSE_VARIANTS_MS = (250, 400, 700)


def load_key() -> str:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    key = os.environ.get("SARVAM_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "SARVAM_API_KEY is not set. Copy .env.example to .env and fill it in.\n"
            "Get a key at https://dashboard.sarvam.ai"
        )
    return key


def synth(client, text: str, speaker: str, language_code: str) -> tuple[np.ndarray, int]:
    r = client.text_to_speech.convert(
        text=text, language_code=language_code, speaker=speaker, model=BULBUL_MODEL
    )
    return decode_bytes(base64.b64decode(r.audios[0]))


def splice_pause(wave: np.ndarray, pause_ms: int, at_fraction: float = 0.55) -> np.ndarray:
    """Insert silence mid-utterance to imitate a thinking pause.

    Inserted rather than appended: appending silence to a complete utterance
    changes nothing a detector should care about, whereas a mid-utterance gap is
    the actual hard case. Room tone at a realistic level rather than digital
    zero, because a perfectly silent gap is a giveaway no real recording has.
    """
    n = wave.size
    cut = int(n * at_fraction)
    rng = np.random.default_rng(pause_ms)
    floor = max(float(np.percentile(np.abs(wave), 5)), 1e-4)
    gap = (rng.normal(0, floor * 0.6, int(pause_ms * SAMPLE_RATE / 1000))).astype(np.float32)
    return np.concatenate([wave[:cut], gap, wave[cut:]]).astype(np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/hinglish/clips")
    ap.add_argument("--speakers", nargs="*", default=list(SPEAKERS))
    ap.add_argument("--limit", type=int, default=None, help="cap phrases (saves credits)")
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--pause-variants", action="store_true",
                    help="also emit mid-utterance-pause variants of complete phrases")
    ap.add_argument("--dry-run", action="store_true", help="list, do not call the API")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    # Fail before spending any credits: an invalid speaker name produces a 400
    # on every single call, which is a slow and expensive way to find a typo.
    unknown = [s for s in args.speakers if s not in BULBUL_V3_SPEAKERS]
    if unknown:
        raise SystemExit(
            f"unknown Bulbul v3 speaker(s): {unknown}\n"
            f"valid names: {', '.join(BULBUL_V3_SPEAKERS)}"
        )

    phrases: list[Phrase] = list(ALL_PHRASES)
    if args.categories:
        phrases = [p for p in phrases if p.category in set(args.categories)]
    if args.limit:
        phrases = phrases[: args.limit]

    out = Path(args.out)
    n_planned = len(phrases) * len(args.speakers)
    if args.pause_variants:
        n_planned += sum(1 for p in phrases if p.endpoint) * len(args.speakers) * len(PAUSE_VARIANTS_MS)

    print(f"\n  {summary()}\n")
    print(f"  selected  : {len(phrases)} phrases x {len(args.speakers)} speakers")
    print(f"  clips      : {n_planned} (~{n_planned} TTS calls)")
    print(f"  out        : {out}")

    if args.dry_run:
        print("\n  --dry-run: nothing will be generated\n")
        for p in phrases[:20]:
            print(f"    [{'END ' if p.endpoint else 'OPEN'}] {p.category:<18s} {p.text}")
        if len(phrases) > 20:
            print(f"    ... and {len(phrases) - 20} more")
        return 0

    key = load_key()
    from sarvamai import SarvamAI

    import soundfile as sf

    client = SarvamAI(api_subscription_key=key)
    out.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    manifest_path = out / "manifest.jsonl"
    if manifest_path.exists() and not args.overwrite:
        manifest = [
            json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()
        ]
        print(f"  resuming: {len(manifest)} clips already present")
    done = {(m["text"], m["speaker"], m.get("pause_ms", 0)) for m in manifest}

    errors = 0
    for pi, phrase in enumerate(phrases):
        for speaker in args.speakers:
            variants: list[int] = [0]
            if args.pause_variants and phrase.endpoint:
                variants += list(PAUSE_VARIANTS_MS)

            base_wave = None
            for pause_ms in variants:
                key_t = (phrase.text, speaker, pause_ms)
                if key_t in done:
                    continue
                try:
                    if base_wave is None:
                        wave, sr = synth(client, phrase.text, speaker, phrase.language_code)
                        base_wave = resample(wave, sr, SAMPLE_RATE)
                        time.sleep(0.15)  # be polite to the API
                    wave = base_wave if pause_ms == 0 else splice_pause(base_wave, pause_ms)
                except Exception as exc:
                    errors += 1
                    print(f"    FAILED {speaker}/{phrase.text[:34]!r}: {type(exc).__name__}: {exc}")
                    continue

                stem = f"{pi:03d}_{phrase.category}_{speaker}"
                if pause_ms:
                    stem += f"_pause{pause_ms}"
                path = out / f"{stem}.wav"
                sf.write(path, wave, SAMPLE_RATE, subtype="PCM_16")

                row = {
                    "file": path.name,
                    "text": phrase.text,
                    "speaker": speaker,
                    "language_code": phrase.language_code,
                    "category": phrase.category,
                    "endpoint_bool": phrase.endpoint,
                    "pause_ms": pause_ms,
                    "note": phrase.note,
                    # Never let a downstream script forget what this audio is.
                    "synthetic": True,
                    "duration_s": round(wave.size / SAMPLE_RATE, 3),
                }
                # A spliced pause turns a complete utterance into one that sounds
                # hesitant partway through. The utterance still *ends* complete,
                # so the label stays True — but it is flagged so the report can
                # break these out, since they test a different thing.
                if pause_ms:
                    row["variant"] = "mid_utterance_pause"
                manifest.append(row)
                print(f"    {len(manifest):>4d}  {stem}  ({wave.size / SAMPLE_RATE:.2f}s)")

        with manifest_path.open("w", encoding="utf-8") as fh:
            for m in manifest:
                fh.write(json.dumps(m, ensure_ascii=False) + "\n")

    pos = sum(1 for m in manifest if m["endpoint_bool"])
    print(f"\n  {len(manifest)} clips written to {out}")
    print(f"  {pos} endpoint / {len(manifest) - pos} not-endpoint")
    if errors:
        print(f"  {errors} synthesis failures (see above)")
    print(f"  manifest -> {manifest_path}")
    print(
        "\n  Reminder for the report: these are TTS clips. Clean, evenly paced,\n"
        "  and an upper bound on real-world performance. State that next to the\n"
        "  numbers rather than after someone asks."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
