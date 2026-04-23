"""
Extract Mimi codes for LJSpeech and publish to the Hugging Face Hub.

Three phases (each skippable via --skip_*):
  1. Download LJSpeech archive if missing (~2.6 GB, idempotent)
  2. Encode each utterance with Mimi → cache as .pt files (idempotent)
  3. Build an Arrow dataset and push to the HF Hub

Usage:
  export HF_TOKEN=hf_...

  # Extract + push (downloads LJSpeech if needed)
  python ljspeech.py --repo_id shangeth/ljspeech-mimi-codes --private

  # Push already-cached codes without re-extracting
  python ljspeech.py --repo_id shangeth/ljspeech-mimi-codes --skip_extract --private

  # Dry-run (build dataset locally, no upload)
  python ljspeech.py --repo_id shangeth/ljspeech-mimi-codes --local_dir ./staged_lj
"""

import argparse
import logging
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterator, Optional

import torch
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from tqdm import tqdm

from mimi import MimiCodec, to_int16


logger = logging.getLogger(__name__)

LJSPEECH_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"

FEATURES = Features({
    "id":          Value("string"),
    "text":        Value("string"),
    "codes":       Sequence(Sequence(Value("int16"))),
    "n_frames":    Value("int32"),
    "k_codebooks": Value("int32"),
})


# -------- Download + extract raw corpus --------

def download_and_extract(root: Path) -> Path:
    """Return the path to LJSpeech-1.1/, downloading + extracting if needed."""
    root.mkdir(parents=True, exist_ok=True)
    corpus_dir = root / "LJSpeech-1.1"
    metadata   = corpus_dir / "metadata.csv"
    wavs_dir   = corpus_dir / "wavs"

    if metadata.exists() and wavs_dir.exists():
        n_wavs = sum(1 for _ in wavs_dir.glob("*.wav"))
        n_rows = sum(1 for _ in open(metadata, encoding="utf-8"))
        if n_wavs >= n_rows:
            return corpus_dir

    archive = root / "LJSpeech-1.1.tar.bz2"
    if not archive.exists():
        logger.info(f"Downloading LJSpeech (~2.6 GB) → {archive}")
        with tqdm(unit="B", unit_scale=True, unit_divisor=1024, desc="LJSpeech") as t:
            def _hook(count, block, total):
                if t.total is None:
                    t.total = total
                t.update(block)
            urllib.request.urlretrieve(LJSPEECH_URL, archive, reporthook=_hook)

    logger.info(f"Extracting {archive}")
    with tarfile.open(archive, "r|bz2") as tar:
        for member in tqdm(tar, desc="Extract", unit="files", total=13101):
            tar.extract(member, path=root)
    return corpus_dir


# -------- Mimi encoding (populates cache dir) --------

def extract_codes(
    corpus_dir:       Path,
    cache_dir:        Path,
    k_codebooks:      int,
    mimi_model_name:  str,
    device:           str,
) -> None:
    """Encode every LJSpeech utterance with Mimi; skip files already cached."""
    import torchaudio

    cache_dir.mkdir(parents=True, exist_ok=True)
    codec = MimiCodec(model_name=mimi_model_name, device=device, k_codebooks=k_codebooks)
    ds    = torchaudio.datasets.LJSPEECH(root=str(corpus_dir.parent), download=False)

    # Read utterance IDs from metadata.csv (avoid private torchaudio attrs)
    import csv
    with open(corpus_dir / "metadata.csv", encoding="utf-8") as f:
        ids = [row[0] for row in csv.reader(f, delimiter="|")]

    skipped, errors = 0, 0
    for idx in tqdm(range(len(ids)), desc="Mimi encode"):
        utt_id   = ids[idx]
        out_path = cache_dir / f"{utt_id}.pt"
        if out_path.exists():
            skipped += 1
            continue
        waveform, sample_rate, _raw, _norm = ds[idx]
        try:
            codes = codec.encode(waveform, sample_rate)  # [k, n_frames]
            torch.save(codes, out_path)
        except Exception as e:
            logger.warning(f"Failed to encode {utt_id}: {e}")
            errors += 1

    total     = len(ids)
    extracted = total - skipped - errors
    logger.info(f"Encoded: {extracted}  Skipped: {skipped}  Errors: {errors}  Total: {total}")


# -------- Build Arrow dataset from cache --------

def _iter_rows(cache_dir: Path, metadata: Path) -> Iterator[dict]:
    with open(metadata, encoding="utf-8") as f:
        rows = [line.rstrip("\n").split("|") for line in f if line.strip()]

    missing = 0
    for parts in tqdm(rows, desc="Building Arrow"):
        if len(parts) < 2:
            continue
        utt_id     = parts[0]
        # metadata.csv: id|raw|normalized — prefer normalized, preserve mixed case
        text       = parts[2] if len(parts) >= 3 and parts[2] else parts[1]
        cache_path = cache_dir / f"{utt_id}.pt"
        if not cache_path.exists():
            missing += 1
            continue
        codes = to_int16(torch.load(cache_path, map_location="cpu", weights_only=True))
        yield {
            "id":          utt_id,
            "text":        text,
            "codes":       codes.tolist(),
            "n_frames":    int(codes.shape[1]),
            "k_codebooks": int(codes.shape[0]),
        }
    if missing:
        print(f"  Skipped {missing} utterances without cached codes")


def build(cache_dir: Path, metadata: Path) -> DatasetDict:
    if not metadata.exists():
        raise FileNotFoundError(f"{metadata} — LJSpeech not extracted?")
    if not cache_dir.exists():
        raise FileNotFoundError(f"{cache_dir} — Mimi codes not extracted?")
    ds = Dataset.from_generator(
        lambda: _iter_rows(cache_dir, metadata),
        features=FEATURES,
    )
    return DatasetDict({"train": ds})


# -------- Driver --------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id",          default="shangeth/ljspeech-mimi-codes")
    parser.add_argument("--ljspeech_root",    default="data",
                        help="Directory containing (or to populate with) LJSpeech-1.1/")
    parser.add_argument("--cache_dir",        default="mimi_cache")
    parser.add_argument("--k_codebooks",      type=int, default=8)
    parser.add_argument("--mimi_model_name",  default="kyutai/mimi")
    parser.add_argument("--device",           default="cuda")
    parser.add_argument("--private",          action="store_true")
    parser.add_argument("--token",            default=None, help="HF token; falls back to HF_TOKEN env")
    parser.add_argument("--skip_extract",     action="store_true",
                        help="Skip download + Mimi encoding (codes must already be cached)")
    parser.add_argument("--skip_push",        action="store_true",
                        help="Only extract, don't push")
    parser.add_argument("--skip_card",        action="store_true")
    parser.add_argument("--local_dir",        default=None,
                        help="Save the built dataset locally instead of uploading")
    parser.add_argument("--commit_message",   default="Upload LJSpeech Mimi codes")
    args = parser.parse_args()

    root       = Path(args.ljspeech_root).resolve()
    cache_dir  = Path(args.cache_dir).resolve()
    corpus_dir = root / "LJSpeech-1.1"

    if not args.skip_extract:
        corpus_dir = download_and_extract(root)
        extract_codes(
            corpus_dir      = corpus_dir,
            cache_dir       = cache_dir,
            k_codebooks     = args.k_codebooks,
            mimi_model_name = args.mimi_model_name,
            device          = args.device,
        )

    if args.skip_push:
        print("Done (extract only).")
        return

    dd = build(cache_dir=cache_dir, metadata=corpus_dir / "metadata.csv")
    print(f"Built: {len(dd['train'])} rows")

    card = Path(__file__).resolve().parent / "cards" / "ljspeech.md"

    if args.local_dir:
        out = Path(args.local_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        dd.save_to_disk(str(out))
        if card.exists():
            (out / "README.md").write_text(card.read_text())
        print(f"Dry run: saved to {out}")
        return

    print(f"Pushing → {args.repo_id}")
    dd.push_to_hub(
        args.repo_id,
        private        = args.private,
        token          = args.token,
        commit_message = args.commit_message,
    )

    if not args.skip_card and card.exists():
        from huggingface_hub import HfApi
        HfApi(token=args.token).upload_file(
            path_or_fileobj = str(card),
            path_in_repo    = "README.md",
            repo_id         = args.repo_id,
            repo_type       = "dataset",
            commit_message  = "Update dataset card",
        )
        print("Uploaded dataset card")

    print(f"Done: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
