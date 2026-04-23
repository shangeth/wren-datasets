"""
Extract Mimi codes for LibriTTS-R and publish to the Hugging Face Hub.

LibriTTS-R is LibriTTS with speech restoration applied — same underlying LibriVox
audiobooks, sentence-level segmentation, punctuation preserved, 24 kHz native
(no resampling needed for Mimi). Strictly better than LibriSpeech for TTS.

Two download modes:
  --source_hf_dataset  Stream audio from an HF mirror (recommended, avoids SSL issues)
  (default)            Download tarballs directly from OpenSLR 141

HF split name conversion: "train-clean-100" → "train.clean.100" (hyphens → dots),
which matches the naming used by HF mirrors such as mythicinfinity/libritts_r.

Usage:
  export HF_TOKEN=hf_...

  # Recommended — stream from HF mirror:
  python libritts_r.py \
    --source_hf_dataset mythicinfinity/libritts_r \
    --splits dev-clean,test-clean \
    --repo_id shangeth/libritts-r-mimi-codes --private

  # All splits via HF mirror:
  python libritts_r.py \
    --source_hf_dataset mythicinfinity/libritts_r \
    --repo_id shangeth/libritts-r-mimi-codes --private

  # Push already-cached splits without re-encoding:
  python libritts_r.py \
    --source_hf_dataset mythicinfinity/libritts_r \
    --splits train-clean-100,train-clean-360 \
    --skip_extract \
    --repo_id shangeth/libritts-r-mimi-codes --private

  # Fallback: direct OpenSLR download (may have SSL issues on some machines):
  python libritts_r.py \
    --splits dev-clean --cleanup_audio \
    --repo_id shangeth/libritts-r-mimi-codes --private
"""

import argparse
import logging
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterator, Optional

import torch
import torchaudio
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
    "train-clean-100":   6,
    "train-clean-360":  23,
    "train-other-500":  30,
    "dev-clean":         0.4,
    "dev-other":         0.4,
    "test-clean":        0.4,
    "test-other":        0.4,
}

OPENSLR_BASE = "https://us.openslr.org/resources/141"

# HF mirrors (e.g. mythicinfinity/libritts_r) use configs + sub-splits.
# "dev.clean"       → config="clean", split="dev"
# "train.clean.100" → config="clean", split="train.100"
# "train.other.500" → config="other", split="train.500"
def _parse_hf_config_split(split_name: str, force_config: Optional[str] = None):
    """Return (config, full_split_name) for load_dataset().

    If force_config is set (e.g. 'all'), use it directly — no auto-detection.
    Otherwise, infer from the split name: 'dev.clean' → config='clean', etc.

    Examples:
      force_config='all', 'dev.clean'       → ('all',   'dev.clean')
      auto,               'dev.clean'       → ('clean', 'dev.clean')
      auto,               'train.other.500' → ('other', 'train.other.500')
    """
    dot_name = split_name.replace("-", ".")
    if force_config:
        return force_config, dot_name
    parts = dot_name.split(".")
    for qual in ("clean", "other"):
        if qual in parts:
            return qual, dot_name
    return None, dot_name


def _to_hf_split(split_name: str) -> str:
    """Hyphen-to-dot conversion for simple cases (kept for backwards compat)."""
    return split_name.replace("-", ".")

FEATURES = Features({
    "id":          Value("string"),
    "text":        Value("string"),
    "speaker_id":  Value("int32"),
    "codes":       Sequence(Sequence(Value("int16"))),
    "n_frames":    Value("int32"),
    "k_codebooks": Value("int32"),
})


# -------- Download + extract --------

def download_and_extract(split: str, root: Path) -> Path:
    """Download split tar.gz from OpenSLR 141 and extract. Idempotent."""
    split_root = root / "LibriTTS_R" / split
    if split_root.exists() and any(split_root.rglob("*.wav")):
        logger.info(f"{split}: already extracted at {split_root}")
        return split_root

    archive = root / f"libritts_r_{split}.tar.gz"
    url     = f"{OPENSLR_BASE}/{split}.tar.gz"

    if not archive.exists():
        logger.info(f"Downloading LibriTTS-R/{split} from {url}")
        with tqdm(unit="B", unit_scale=True, unit_divisor=1024, desc=split) as t:
            def _hook(count, block, total):
                if t.total is None:
                    t.total = total
                t.update(block)
            urllib.request.urlretrieve(url, archive, reporthook=_hook)

    logger.info(f"Extracting {archive} ...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=root)

    return split_root


# -------- HF-mirror extraction --------

def _hf_split_size(
    source_dataset: str,
    config:         Optional[str],
    hf_split:       str,
    token:          Optional[str],
) -> Optional[int]:
    """Query the number of examples in an HF split without downloading data."""
    try:
        from datasets import load_dataset_builder
        builder = load_dataset_builder(source_dataset, config, token=token)
        info    = builder.info.splits.get(hf_split)
        return info.num_examples if info else None
    except Exception:
        return None


def extract_codes_from_hf_split(
    split:            str,
    source_dataset:   str,
    cache_dir:        Path,
    k_codebooks:      int,
    mimi_model_name:  str,
    device:           str,
    token:            Optional[str],
    hf_config:        Optional[str] = None,
) -> None:
    """Stream audio from an HF mirror dataset and encode with Mimi."""
    from datasets import load_dataset
    config, hf_split = _parse_hf_config_split(split, force_config=hf_config)
    logger.info(f"Streaming {source_dataset}  config={config}  split={hf_split}")
    load_kw = dict(split=hf_split, streaming=True, token=token)
    ds    = (load_dataset(source_dataset, config, **load_kw) if config
             else load_dataset(source_dataset, **load_kw))
    total = _hf_split_size(source_dataset, config, hf_split, token)
    cache_dir.mkdir(parents=True, exist_ok=True)
    codec = MimiCodec(model_name=mimi_model_name, device=device, k_codebooks=k_codebooks)

    skipped = errors = 0
    for ex in tqdm(ds, desc=f"Encode {split}", total=total):
        utt_id   = str(ex.get("id") or ex.get("file_id") or ex.get("utterance_id", ""))
        out_path = cache_dir / f"{utt_id}.pt"
        if out_path.exists():
            skipped += 1
            continue
        try:
            audio = ex["audio"]
            wav   = torch.tensor(audio["array"]).unsqueeze(0).float()
            sr    = audio["sampling_rate"]
            codes = codec.encode(wav, sr)
            torch.save(codes, out_path)
        except Exception as e:
            logger.warning(f"Failed to encode {utt_id}: {e}")
            errors += 1

    logger.info(f"{split}: skipped={skipped}  errors={errors}")


def _iter_rows_from_hf(
    split:          str,
    source_dataset: str,
    cache_dir:      Path,
    token:          Optional[str],
    hf_config:      Optional[str] = None,
) -> Iterator[dict]:
    """Re-stream metadata from HF; match against cached .pt files."""
    from datasets import load_dataset
    config, hf_split = _parse_hf_config_split(split, force_config=hf_config)
    load_kw = dict(split=hf_split, streaming=True, token=token)
    ds    = (load_dataset(source_dataset, config, **load_kw) if config
             else load_dataset(source_dataset, **load_kw))
    total   = _hf_split_size(source_dataset, config, hf_split, token)
    missing = 0

    for ex in tqdm(ds, desc=f"Build {split}", total=total):
        utt_id     = str(ex.get("id") or ex.get("file_id") or ex.get("utterance_id", ""))
        cache_path = cache_dir / f"{utt_id}.pt"
        if not cache_path.exists():
            missing += 1
            continue

        text       = ex.get("text_normalized") or ex.get("text", "")
        speaker_id = ex.get("speaker_id", 0)
        codes      = to_int16(torch.load(cache_path, map_location="cpu", weights_only=True))
        yield {
            "id":          utt_id,
            "text":        text,
            "speaker_id":  int(speaker_id),
            "codes":       codes.tolist(),
            "n_frames":    int(codes.shape[1]),
            "k_codebooks": int(codes.shape[0]),
        }

    if missing:
        print(f"  {split}: skipped {missing} utterances (missing cached codes)")


# -------- Local Mimi encoding (OpenSLR download) --------

def extract_split_codes(
    split:            str,
    root:             Path,
    cache_dir:        Path,
    k_codebooks:      int,
    mimi_model_name:  str,
    device:           str,
) -> None:
    """Encode every .wav in the split; skip already-cached files."""
    split_root = root / "LibriTTS_R" / split
    if not split_root.exists():
        raise FileNotFoundError(f"{split_root} — run without --skip_extract first")

    cache_dir.mkdir(parents=True, exist_ok=True)
    codec   = MimiCodec(model_name=mimi_model_name, device=device, k_codebooks=k_codebooks)
    wavs    = sorted(split_root.rglob("*.wav"))
    skipped = errors = 0

    for wav_path in tqdm(wavs, desc=f"Encode {split}"):
        utt_id    = wav_path.stem                     # e.g. 84_121123_000003_000000
        out_path  = cache_dir / f"{utt_id}.pt"
        if out_path.exists():
            skipped += 1
            continue
        try:
            wav, sr = torchaudio.load(wav_path)       # already 24 kHz, no resample needed
            codes   = codec.encode(wav, sr)
            torch.save(codes, out_path)
        except Exception as e:
            logger.warning(f"Failed to encode {utt_id}: {e}")
            errors += 1

    total     = len(wavs)
    extracted = total - skipped - errors
    logger.info(f"{split}: encoded={extracted}  skipped={skipped}  errors={errors}  total={total}")


# -------- Build Arrow dataset --------

def _iter_split_rows(
    split_root: Path,
    cache_dir:  Path,
    split_name: str,
) -> Iterator[dict]:
    """
    Walk per-utterance .normalized.txt files (same directory as each .wav).
    Text is mixed-case with punctuation — no lowercasing.
    """
    wav_files = sorted(split_root.rglob("*.wav"))
    missing   = 0
    n         = 0

    for wav_path in tqdm(wav_files, desc=split_name):
        utt_id     = wav_path.stem
        txt_path   = wav_path.with_suffix(".normalized.txt")
        cache_path = cache_dir / f"{utt_id}.pt"

        if not txt_path.exists() or not cache_path.exists():
            missing += 1
            continue

        text       = txt_path.read_text(encoding="utf-8").strip()
        speaker_id = int(utt_id.split("_")[0])
        codes      = to_int16(torch.load(cache_path, map_location="cpu", weights_only=True))

        yield {
            "id":          utt_id,
            "text":        text,
            "speaker_id":  speaker_id,
            "codes":       codes.tolist(),
            "n_frames":    int(codes.shape[1]),
            "k_codebooks": int(codes.shape[0]),
        }
        n += 1

    if missing:
        print(f"  {split_name}: skipped {missing} utterances (missing .txt or cached codes)")


def _cleanup_split_audio(split_root: Path):
    """Delete .wav files after extraction; keep .normalized.txt for push."""
    if not split_root.exists():
        return
    n = 0
    for wav in split_root.rglob("*.wav"):
        wav.unlink()
        n += 1
    print(f"  Removed {n} .wav files under {split_root}")


def _push_split(
    split:          str,
    root:           Path,
    cache_dir:      Path,
    repo_id:        str,
    private:        bool,
    token:          Optional[str],
    source_hf:      Optional[str] = None,
    hf_config:      Optional[str] = None,
):
    if source_hf:
        gen = lambda s=split: _iter_rows_from_hf(s, source_hf, cache_dir, token, hf_config=hf_config)
    else:
        split_root = root / "LibriTTS_R" / split
        if not split_root.exists():
            print(f"  Skip push ({split}): {split_root} missing")
            return
        gen = lambda s=split, sr=split_root: _iter_split_rows(sr, cache_dir, s)

    ds = Dataset.from_generator(gen, features=FEATURES)
    if len(ds) == 0:
        print(f"  Skip push ({split}): no rows built")
        return

    hf_split = split.replace("-", "_").replace(".", "_")
    print(f"  Pushing {hf_split} ({len(ds):,} rows) → {repo_id}")
    ds.push_to_hub(
        repo_id,
        split          = hf_split,
        private        = private,
        token          = token,
        commit_message = f"Add {hf_split}",
    )


def _upload_card(repo_id: str, token: Optional[str]):
    card = Path(__file__).resolve().parent / "cards" / "libritts_r.md"
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


# -------- Driver --------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id",             default="shangeth/libritts-r-mimi-codes")
    parser.add_argument("--splits", type=lambda s: s.split(","), default=None,
                        help=f"Default: all ({','.join(ALL_SPLITS)})")
    parser.add_argument("--source_hf_dataset",   default=None,
                        help="Stream audio from this HF dataset instead of OpenSLR. Recommended.")
    parser.add_argument("--hf_config",           default=None,
                        help="Force a specific HF dataset config (e.g. 'all'). "
                             "If omitted, config is auto-detected from split name.")
    parser.add_argument("--root",                default="data",
                        help="Directory where LibriTTS_R/ is extracted (local mode only)")
    parser.add_argument("--cache_dir",           default="libritts_r_mimi_cache")
    parser.add_argument("--k_codebooks",         type=int, default=8)
    parser.add_argument("--mimi_model_name",     default="kyutai/mimi")
    parser.add_argument("--device",              default="cuda")
    parser.add_argument("--private",             action="store_true")
    parser.add_argument("--token",               default=None)
    parser.add_argument("--skip_extract",        action="store_true")
    parser.add_argument("--skip_push",           action="store_true")
    parser.add_argument("--skip_card",           action="store_true")
    parser.add_argument("--cleanup_audio",       action="store_true",
                        help="(local mode) Delete .wav files after encoding each split.")
    args = parser.parse_args()

    splits    = args.splits or ALL_SPLITS
    root      = Path(args.root).resolve()
    cache_dir = Path(args.cache_dir).resolve()

    src = args.source_hf_dataset
    if src:
        print(f"Source: HF dataset {src}")
        print(f"Split name conversion: 'train-clean-100' → '{_to_hf_split('train-clean-100')}'")
    else:
        total_gb = sum(APPROX_SIZE_GB.get(s, 1) for s in splits)
        print(f"Source: OpenSLR 141, ~{total_gb:.1f} GB download")
    print()

    for split in splits:
        print(f"=== {split} ===")

        if not args.skip_extract:
            if src:
                extract_codes_from_hf_split(
                    split           = split,
                    source_dataset  = src,
                    cache_dir       = cache_dir,
                    k_codebooks     = args.k_codebooks,
                    mimi_model_name = args.mimi_model_name,
                    device          = args.device,
                    token           = args.token,
                    hf_config       = args.hf_config,
                )
            else:
                download_and_extract(split, root)
                extract_split_codes(
                    split           = split,
                    root            = root,
                    cache_dir       = cache_dir,
                    k_codebooks     = args.k_codebooks,
                    mimi_model_name = args.mimi_model_name,
                    device          = args.device,
                )

        if not args.skip_push:
            _push_split(split, root, cache_dir, args.repo_id, args.private, args.token,
                        source_hf=src, hf_config=args.hf_config)

        if args.cleanup_audio and not args.skip_extract and not src:
            _cleanup_split_audio(root / "LibriTTS_R" / split)
            for tar in root.glob(f"libritts_r_{split}.tar.gz"):
                tar.unlink()
                print(f"  Removed {tar.name}")

    if not args.skip_push and not args.skip_card:
        _upload_card(args.repo_id, args.token)

    print("\nAll done.")


if __name__ == "__main__":
    main()
