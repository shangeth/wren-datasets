"""
Extract Mimi codes for LibriSpeech splits and publish them to the Hugging Face Hub.

Three phases (each skippable):
  1. Download each requested LibriSpeech split (torchaudio handles the tar.gz)
  2. Encode every utterance with Mimi → cache as .pt files (idempotent)
  3. Build an Arrow dataset and push_to_hub per split (preserves splits already on HF)

Transcripts are lowercased (raw LibriSpeech is ALL UPPER).

Usage:
  export HF_TOKEN=hf_...

  # Extract + push specific splits
  python librispeech.py \
    --splits dev-clean,test-clean \
    --repo_id shangeth/librispeech-mimi-codes --private

  # Push already-cached splits without re-encoding
  python librispeech.py \
    --splits train-clean-100,train-clean-360 \
    --skip_extract \
    --repo_id shangeth/librispeech-mimi-codes --private

  # Extract big splits with --cleanup_audio to drop .flac files per split
  python librispeech.py \
    --splits train-other-500 --cleanup_audio \
    --repo_id shangeth/librispeech-mimi-codes --private
"""

import argparse
import logging
from pathlib import Path
from typing import Iterator, Optional

import torch
from datasets import Dataset, Features, Sequence, Value
from tqdm import tqdm

from mimi import MimiCodec, to_int16


logger = logging.getLogger(__name__)


ALL_SPLITS = [
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
]

APPROX_SIZE_GB = {
    "train-clean-100":  6,
    "train-clean-360": 23,
    "train-other-500": 30,
    "dev-clean":        0.4,
    "dev-other":        0.4,
    "test-clean":       0.4,
    "test-other":       0.4,
}

FEATURES = Features({
    "id":          Value("string"),
    "text":        Value("string"),
    "speaker_id":  Value("int32"),
    "codes":       Sequence(Sequence(Value("int16"))),
    "n_frames":    Value("int32"),
    "k_codebooks": Value("int32"),
})


# -------- Mimi encoding (downloads + extracts each split on demand) --------

def extract_split_codes(
    split_name:       str,
    root:             Path,
    cache_dir:        Path,
    k_codebooks:      int,
    mimi_model_name:  str,
    device:           str,
) -> None:
    """Download (if needed) + Mimi-encode one LibriSpeech split. Idempotent."""
    import torchaudio

    cache_dir.mkdir(parents=True, exist_ok=True)
    codec = MimiCodec(model_name=mimi_model_name, device=device, k_codebooks=k_codebooks)

    logger.info(f"Loading LibriSpeech split: {split_name}")
    ds = torchaudio.datasets.LIBRISPEECH(root=str(root), url=split_name, download=True)

    skipped, errors = 0, 0
    for idx in tqdm(range(len(ds)), desc=f"Mimi encode {split_name}"):
        waveform, sample_rate, _text, speaker_id, chapter_id, utt_id = ds[idx]
        key       = f"{speaker_id}-{chapter_id}-{utt_id:04d}"
        out_path  = cache_dir / f"{key}.pt"
        if out_path.exists():
            skipped += 1
            continue
        try:
            codes = codec.encode(waveform, sample_rate)
            torch.save(codes, out_path)
        except Exception as e:
            logger.warning(f"Failed to encode {key}: {e}")
            errors += 1

    total     = len(ds)
    extracted = total - skipped - errors
    logger.info(f"{split_name}: encoded={extracted}  skipped={skipped}  errors={errors}  total={total}")


# -------- Build Arrow dataset from cached transcripts + codes --------

def _iter_split_rows(
    split_root: Path,
    cache_dir:  Path,
    split_name: str,
) -> Iterator[dict]:
    """Walk .trans.txt files (no audio loading) and emit rows."""
    trans_files = sorted(split_root.rglob("*.trans.txt"))
    missing = 0
    n = 0
    for trans_file in tqdm(trans_files, desc=split_name):
        with open(trans_file, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                key, _, text = line.partition(" ")
                if not text:
                    continue
                speaker_id = int(key.split("-")[0])
                cache_path = cache_dir / f"{key}.pt"
                if not cache_path.exists():
                    missing += 1
                    continue
                codes = to_int16(torch.load(cache_path, map_location="cpu", weights_only=True))
                yield {
                    "id":          key,
                    "text":        text.lower(),
                    "speaker_id":  speaker_id,
                    "codes":       codes.tolist(),
                    "n_frames":    int(codes.shape[1]),
                    "k_codebooks": int(codes.shape[0]),
                }
                n += 1
    if missing:
        print(f"  {split_name}: missing codes for {missing} utterances (skipped; yielded {n})")


def _cleanup_split_audio(split_root: Path):
    """Delete .flac files but keep .trans.txt transcripts (needed for build)."""
    if not split_root.exists():
        return
    n = 0
    for flac in split_root.rglob("*.flac"):
        flac.unlink()
        n += 1
    print(f"  Removed {n} .flac files under {split_root}")


def _push_split(
    split:     str,
    root:      Path,
    cache_dir: Path,
    repo_id:   str,
    private:   bool,
    token:     Optional[str],
):
    split_root = root / "LibriSpeech" / split
    if not split_root.exists():
        print(f"  Skip push ({split}): {split_root} missing")
        return

    ds = Dataset.from_generator(
        lambda s=split, sr=split_root: _iter_split_rows(sr, cache_dir, s),
        features=FEATURES,
    )
    if len(ds) == 0:
        print(f"  Skip push ({split}): no rows built")
        return

    hf_split = split.replace("-", "_")
    print(f"  Pushing {hf_split} ({len(ds):,} rows) → {repo_id}")
    ds.push_to_hub(
        repo_id,
        split          = hf_split,
        private        = private,
        token          = token,
        commit_message = f"Add {hf_split}",
    )


def _upload_card(repo_id: str, token: Optional[str]):
    card = Path(__file__).resolve().parent / "cards" / "librispeech.md"
    if not card.exists():
        return
    from huggingface_hub import HfApi
    HfApi(token=token).upload_file(
        path_or_fileobj = str(card),
        path_in_repo    = "README.md",
        repo_id         = repo_id,
        repo_type       = "dataset",
        commit_message  = "Update dataset card",
    )
    print("Uploaded dataset card")


# -------- Driver --------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id",          default="shangeth/librispeech-mimi-codes")
    parser.add_argument("--splits", type=lambda s: s.split(","), default=None,
                        help=f"Comma-sep list. Default: all ({','.join(ALL_SPLITS)})")
    parser.add_argument("--librispeech_root", default="data")
    parser.add_argument("--cache_dir",        default="librispeech_mimi_cache")
    parser.add_argument("--k_codebooks",      type=int, default=8)
    parser.add_argument("--mimi_model_name",  default="kyutai/mimi")
    parser.add_argument("--device",           default="cuda")
    parser.add_argument("--private",          action="store_true")
    parser.add_argument("--token",            default=None)
    parser.add_argument("--skip_extract",     action="store_true")
    parser.add_argument("--skip_push",        action="store_true")
    parser.add_argument("--skip_card",        action="store_true")
    parser.add_argument("--cleanup_audio",    action="store_true",
                        help="Delete .flac files after each split's codes are extracted.")
    args = parser.parse_args()

    splits    = args.splits or ALL_SPLITS
    root      = Path(args.librispeech_root).resolve()
    cache_dir = Path(args.cache_dir).resolve()

    total_gb = sum(APPROX_SIZE_GB.get(s, 1) for s in splits)
    print(f"Plan: {len(splits)} split(s), ~{total_gb:.1f} GB of audio download")
    for s in splits:
        print(f"  - {s}  (~{APPROX_SIZE_GB.get(s, '?')} GB)")
    if not args.cleanup_audio and total_gb > 10 and not args.skip_extract:
        print("  (Consider --cleanup_audio to drop .flac per split.)")
    print()

    for split in splits:
        print(f"=== {split} ===")

        if not args.skip_extract:
            extract_split_codes(
                split_name      = split,
                root            = root,
                cache_dir       = cache_dir,
                k_codebooks     = args.k_codebooks,
                mimi_model_name = args.mimi_model_name,
                device          = args.device,
            )

        if not args.skip_push:
            _push_split(split, root, cache_dir, args.repo_id, args.private, args.token)

        if args.cleanup_audio and not args.skip_extract:
            _cleanup_split_audio(root / "LibriSpeech" / split)
            for tar in root.glob(f"{split}.tar.gz"):
                tar.unlink()
                print(f"  Removed {tar}")

    if not args.skip_push and not args.skip_card:
        print()
        _upload_card(args.repo_id, args.token)

    print("\nAll done.")


if __name__ == "__main__":
    main()
