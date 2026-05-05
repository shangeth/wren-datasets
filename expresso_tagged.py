"""
Derive shangeth/expresso-mimi-codes-tagged from shangeth/expresso-mimi-codes.

The fine-tune trainer expects a canonical schema (id, text, speaker_id, codes,
n_frames, k_codebooks). Our published mimi-codes dataset has extra columns
(style, substyle, corpus, other_speaker_id, other_style, source_file_id, channel,
start_s, end_s) and lives in two configs (read, conversational).

This script:
  1. Loads BOTH configs of shangeth/expresso-mimi-codes (all 3 splits)
  2. Drops 5 disallowed styles (animal, animaldir, child, childdir, nonverbal)
  3. Prepends `<style>` tag to text for 19 tagged styles; leaves default untagged
  4. Concats read + conversational per split
  5. Strips to canonical schema
  6. Pushes to shangeth/expresso-mimi-codes-tagged (single config, 3 splits)

Tags (19): happy sad angry fearful disgusted awe desire calm sympathetic
           laughing enunciated whisper fast projected
           confused sarcastic narration
           bored sleepy

Usage:
  export HF_TOKEN=hf_...
  python expresso_tagged.py --src_repo shangeth/expresso-mimi-codes \\
                             --dst_repo shangeth/expresso-mimi-codes-tagged --private
"""

import argparse
import logging
from typing import Iterator

from datasets import (
    Dataset, Features, Sequence, Value,
    load_dataset, concatenate_datasets,
)
from huggingface_hub import HfApi


logger = logging.getLogger(__name__)

DROP_STYLES = {"animal", "animaldir", "child", "childdir", "nonverbal"}
TAG_STYLES  = {
    "happy", "sad", "angry", "fearful", "disgusted",
    "awe", "desire", "calm", "sympathetic",
    "laughing", "enunciated", "whisper", "fast", "projected",
    "confused", "sarcastic", "narration",
    "bored", "sleepy",
}
# Untagged "normal": default. Anything else still in the data → defensively keep.

CANONICAL_FEATURES = Features({
    "id":          Value("string"),
    "text":        Value("string"),
    "speaker_id":  Value("string"),     # cast int → string for trainer compatibility
    "codes":       Sequence(Sequence(Value("int16"))),
    "n_frames":    Value("int32"),
    "k_codebooks": Value("int32"),
})


def _format_text(style: str, text: str) -> str:
    if style in TAG_STYLES:
        return f"<{style}> {text}"
    return text   # default (and any unknown style fallback) → no tag


def _build_split_rows(src_repo: str, split: str, token: str = None) -> Iterator[dict]:
    n_dropped = 0
    for cfg in ("read", "conversational"):
        ds = load_dataset(src_repo, cfg, split=split, token=token)
        for r in ds:
            style = r.get("style", "default")
            if style in DROP_STYLES:
                n_dropped += 1
                continue
            yield {
                "id":          r["id"],
                "text":        _format_text(style, r["text"] or ""),
                "speaker_id":  str(r["speaker_id"]),
                "codes":       r["codes"],
                "n_frames":    int(r["n_frames"]),
                "k_codebooks": int(r["k_codebooks"]),
            }
    if n_dropped:
        logger.info(f"  dropped {n_dropped} rows from disallowed styles in split={split!r}")


def _build_split(src_repo: str, split: str, token: str = None) -> Dataset:
    return Dataset.from_generator(
        lambda: _build_split_rows(src_repo, split, token),
        features = CANONICAL_FEATURES,
    )


def _upload_card(dst_repo: str, token: str = None) -> None:
    from pathlib import Path
    card = Path(__file__).resolve().parent / "cards" / "expresso_tagged.md"
    if not card.exists():
        return
    HfApi(token=token).upload_file(
        path_or_fileobj = str(card),
        path_in_repo    = "README.md",
        repo_id         = dst_repo,
        repo_type       = "dataset",
        commit_message  = "Update dataset card",
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_repo", default="shangeth/expresso-mimi-codes")
    parser.add_argument("--dst_repo", default="shangeth/expresso-mimi-codes-tagged")
    parser.add_argument("--splits",   default="train,dev,test")
    parser.add_argument("--private",  action="store_true")
    parser.add_argument("--token",    default=None)
    parser.add_argument("--skip_card", action="store_true")
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    print(f"src: {args.src_repo}\ndst: {args.dst_repo}\nsplits: {splits}")
    print(f"drop styles: {sorted(DROP_STYLES)}")
    print(f"tag styles : {sorted(TAG_STYLES)}")

    for split in splits:
        print(f"\n=== {split} ===")
        ds = _build_split(args.src_repo, split, args.token)
        print(f"  built {len(ds):,} rows")
        ds.push_to_hub(
            args.dst_repo,
            split          = split,
            private        = args.private,
            token          = args.token,
            commit_message = f"Build {split} from {args.src_repo} (read+conv merged, tagged, k=32 codes)",
        )
        print(f"  pushed → {args.dst_repo} ({split})")

    if not args.skip_card:
        _upload_card(args.dst_repo, args.token)
        print("Uploaded dataset card")

    print(f"\nDone: https://huggingface.co/datasets/{args.dst_repo}")


if __name__ == "__main__":
    main()
