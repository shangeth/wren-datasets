"""
Extract Mimi codes for Expresso (conversational) and publish to HuggingFace.

Expresso: 4 speakers, ~40h, 26 expressive styles (happy, sad, animal, whisper, etc.)
recorded in conversational pairs. Directly relevant to Wren's speech disentanglement
research — same speaker/same text across different styles.

Source: nytopop/expresso-conversational on HuggingFace
License: CC-BY-NC-4.0 (non-commercial use only)

Key columns kept:
  style             — expressive style of this utterance (the core research signal)
  other_speaker_id  — conversation partner's speaker ID
  other_style       — conversation partner's expressive style

Usage:
  export HF_TOKEN=hf_...

  python expresso.py --repo_id shangeth/expresso-mimi-codes --private
  python expresso.py --repo_id shangeth/expresso-mimi-codes --skip_extract --private
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
HF_SRC = "nytopop/expresso-conversational"

FEATURES = Features({
    "id":               Value("string"),
    "text":             Value("string"),
    "speaker_id":       Value("int32"),
    "style":            Value("string"),
    "other_speaker_id": Value("int32"),
    "other_style":      Value("string"),
    "codes":            Sequence(Sequence(Value("int16"))),
    "n_frames":         Value("int32"),
    "k_codebooks":      Value("int32"),
})


def _speaker_int(s: str) -> int:
    return int(str(s).lstrip("ex").lstrip("0") or "0")


def _hf_size(token: Optional[str]) -> Optional[int]:
    try:
        from datasets import load_dataset_builder
        b    = load_dataset_builder(HF_SRC, token=token)
        info = b.info.splits.get("train")
        return info.num_examples if info else None
    except Exception:
        return None


def extract_codes(
    cache_dir:       Path,
    k_codebooks:     int,
    mimi_model_name: str,
    device:          str,
    token:           Optional[str],
) -> None:
    from datasets import load_dataset
    ds    = load_dataset(HF_SRC, split="train", streaming=True, token=token)
    total = _hf_size(token)
    cache_dir.mkdir(parents=True, exist_ok=True)
    codec = MimiCodec(model_name=mimi_model_name, device=device, k_codebooks=k_codebooks)

    skipped = errors = 0
    for ex in tqdm(ds, desc="Encode Expresso", total=total):
        utt_id   = ex["id"]
        out_path = cache_dir / f"{utt_id}.pt"
        if out_path.exists():
            skipped += 1
            continue
        try:
            audio = ex["audio"]
            wav   = torch.tensor(audio["array"]).unsqueeze(0).float()
            codes = codec.encode(wav, audio["sampling_rate"])
            torch.save(codes, out_path)
        except Exception as e:
            logger.warning(f"Failed {utt_id}: {e}")
            errors += 1

    logger.info(f"Expresso: skipped={skipped}  errors={errors}")


def _iter_rows(cache_dir: Path, token: Optional[str]) -> Iterator[dict]:
    from datasets import load_dataset
    ds      = load_dataset(HF_SRC, split="train", streaming=True, token=token)
    total   = _hf_size(token)
    missing = 0

    for ex in tqdm(ds, desc="Build Expresso", total=total):
        utt_id     = ex["id"]
        cache_path = cache_dir / f"{utt_id}.pt"
        if not cache_path.exists():
            missing += 1
            continue
        codes = to_int16(torch.load(cache_path, map_location="cpu", weights_only=True))
        yield {
            "id":               utt_id,
            "text":             ex.get("text", ""),
            "speaker_id":       _speaker_int(ex.get("speaker_id", "0")),
            "style":            ex.get("style", ""),
            "other_speaker_id": _speaker_int(ex.get("other_speaker_id", "0")),
            "other_style":      ex.get("other_style", ""),
            "codes":            codes.tolist(),
            "n_frames":         int(codes.shape[1]),
            "k_codebooks":      int(codes.shape[0]),
        }
    if missing:
        print(f"  Skipped {missing} utterances (missing cached codes)")


def _upload_card(repo_id: str, token: Optional[str]):
    card = Path(__file__).resolve().parent / "cards" / "expresso.md"
    if not card.exists():
        return
    from huggingface_hub import HfApi
    HfApi(token=token).upload_file(
        path_or_fileobj=str(card), path_in_repo="README.md",
        repo_id=repo_id, repo_type="dataset", commit_message="Update dataset card",
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id",         default="shangeth/expresso-mimi-codes")
    parser.add_argument("--cache_dir",       default="expresso_mimi_cache")
    parser.add_argument("--k_codebooks",     type=int, default=8)
    parser.add_argument("--mimi_model_name", default="kyutai/mimi")
    parser.add_argument("--device",          default="cuda")
    parser.add_argument("--private",         action="store_true")
    parser.add_argument("--token",           default=None)
    parser.add_argument("--skip_extract",    action="store_true")
    parser.add_argument("--skip_push",       action="store_true")
    parser.add_argument("--skip_card",       action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    print(f"Source:    {HF_SRC}")
    print(f"Cache dir: {cache_dir}")
    print("License:   CC-BY-NC-4.0 (non-commercial only)")

    if not args.skip_extract:
        extract_codes(cache_dir, args.k_codebooks, args.mimi_model_name, args.device, args.token)

    if not args.skip_push:
        ds = Dataset.from_generator(
            lambda: _iter_rows(cache_dir, args.token), features=FEATURES
        )
        print(f"Pushing {len(ds):,} rows → {args.repo_id}")
        ds.push_to_hub(
            args.repo_id, private=args.private, token=args.token,
            commit_message="Upload Expresso Mimi codes",
        )
        if not args.skip_card:
            _upload_card(args.repo_id, args.token)

    print("\nAll done.")


if __name__ == "__main__":
    main()
