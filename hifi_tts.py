"""
Extract Mimi codes for HiFi-TTS and publish to the Hugging Face Hub.

HiFi-TTS is professionally narrated audiobook data — NOT LibriVox.
Studio-quality recordings (SNR >32 dB), 10 speakers, ~292 h total.
Original audio is 44.1 kHz; MimiCodec resamples to 24 kHz internally.

HF source: MikhailT/hifi-tts
  Configs:    clean | other | all
  all splits: train.clean  train.other  dev.clean  dev.other  test.clean  test.other
  Columns:    speaker  file  text_normalized  audio (44100 Hz)

Usage:
  export HF_TOKEN=hf_...

  # All splits via HF (recommended)
  python hifi_tts.py \
    --source_hf_dataset MikhailT/hifi-tts \
    --hf_config all \
    --repo_id shangeth/hifi-tts-mimi-codes --private

  # Specific splits
  python hifi_tts.py \
    --source_hf_dataset MikhailT/hifi-tts \
    --hf_config all \
    --splits dev.clean,test.clean \
    --repo_id shangeth/hifi-tts-mimi-codes --private

  # Push already-cached splits
  python hifi_tts.py \
    --source_hf_dataset MikhailT/hifi-tts \
    --hf_config all \
    --skip_extract \
    --repo_id shangeth/hifi-tts-mimi-codes --private
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
    "train.clean",
    "train.other",
    "dev.clean",
    "dev.other",
    "test.clean",
    "test.other",
]

FEATURES = Features({
    "id":          Value("string"),
    "text":        Value("string"),
    "speaker_id":  Value("int32"),
    "codes":       Sequence(Sequence(Value("int16"))),
    "n_frames":    Value("int32"),
    "k_codebooks": Value("int32"),
})


# -------- Helpers --------

def _hf_split_size(
    source_dataset: str,
    config:         Optional[str],
    hf_split:       str,
    token:          Optional[str],
) -> Optional[int]:
    try:
        from datasets import load_dataset_builder
        builder = load_dataset_builder(source_dataset, config, token=token)
        info    = builder.info.splits.get(hf_split)
        return info.num_examples if info else None
    except Exception:
        return None


def _load_hf(source_dataset, config, split, token):
    from datasets import load_dataset
    load_kw = dict(split=split, streaming=True, token=token)
    return (load_dataset(source_dataset, config, **load_kw) if config
            else load_dataset(source_dataset, **load_kw))


def _utt_id(ex: dict, fallback: int) -> str:
    """MikhailT/hifi-tts has a 'file' column; use its stem as the utterance ID."""
    f = ex.get("file") or ex.get("id") or ex.get("file_id") or str(fallback)
    return Path(f).stem


def _text(ex: dict) -> str:
    return ex.get("text_normalized") or ex.get("text") or ""


def _speaker(ex: dict) -> int:
    return int(ex.get("speaker") or ex.get("speaker_id") or 0)


# -------- Encoding --------

def extract_codes_from_hf_split(
    split:            str,
    source_dataset:   str,
    hf_config:        Optional[str],
    cache_dir:        Path,
    k_codebooks:      int,
    mimi_model_name:  str,
    device:           str,
    token:            Optional[str],
) -> None:
    logger.info(f"Streaming {source_dataset}  config={hf_config}  split={split}")
    ds    = _load_hf(source_dataset, hf_config, split, token)
    total = _hf_split_size(source_dataset, hf_config, split, token)
    cache_dir.mkdir(parents=True, exist_ok=True)
    codec = MimiCodec(model_name=mimi_model_name, device=device, k_codebooks=k_codebooks)

    skipped = errors = 0
    for i, ex in enumerate(tqdm(ds, desc=f"Encode {split}", total=total)):
        utt_id   = _utt_id(ex, i)
        out_path = cache_dir / f"{utt_id}.pt"
        if out_path.exists():
            skipped += 1
            continue
        try:
            audio = ex["audio"]
            wav   = torch.tensor(audio["array"]).unsqueeze(0).float()
            sr    = audio["sampling_rate"]
            codes = codec.encode(wav, sr)    # MimiCodec resamples 44100→24000
            torch.save(codes, out_path)
        except Exception as e:
            logger.warning(f"Failed to encode {utt_id}: {e}")
            errors += 1

    logger.info(f"{split}: skipped={skipped}  errors={errors}")


# -------- Build Arrow dataset --------

def _iter_rows_from_hf(
    split:          str,
    source_dataset: str,
    hf_config:      Optional[str],
    cache_dir:      Path,
    token:          Optional[str],
) -> Iterator[dict]:
    ds      = _load_hf(source_dataset, hf_config, split, token)
    total   = _hf_split_size(source_dataset, hf_config, split, token)
    missing = 0

    for i, ex in enumerate(tqdm(ds, desc=f"Build {split}", total=total)):
        utt_id     = _utt_id(ex, i)
        cache_path = cache_dir / f"{utt_id}.pt"
        if not cache_path.exists():
            missing += 1
            continue
        codes = to_int16(torch.load(cache_path, map_location="cpu", weights_only=True))
        yield {
            "id":          utt_id,
            "text":        _text(ex),
            "speaker_id":  _speaker(ex),
            "codes":       codes.tolist(),
            "n_frames":    int(codes.shape[1]),
            "k_codebooks": int(codes.shape[0]),
        }
    if missing:
        print(f"  {split}: skipped {missing} utterances (missing cached codes)")


def _push_split(
    split:          str,
    source_dataset: str,
    hf_config:      Optional[str],
    cache_dir:      Path,
    repo_id:        str,
    private:        bool,
    token:          Optional[str],
):
    ds = Dataset.from_generator(
        lambda: _iter_rows_from_hf(split, source_dataset, hf_config, cache_dir, token),
        features=FEATURES,
    )
    if len(ds) == 0:
        print(f"  Skip push ({split}): no rows built")
        return

    hf_split = split.replace(".", "_")
    print(f"  Pushing {hf_split} ({len(ds):,} rows) → {repo_id}")
    ds.push_to_hub(
        repo_id,
        split          = hf_split,
        private        = private,
        token          = token,
        commit_message = f"Add {hf_split}",
    )


def _upload_card(repo_id: str, token: Optional[str]):
    card = Path(__file__).resolve().parent / "cards" / "hifi_tts.md"
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
    parser.add_argument("--repo_id",            default="shangeth/hifi-tts-mimi-codes")
    parser.add_argument("--source_hf_dataset",  default="MikhailT/hifi-tts")
    parser.add_argument("--hf_config",          default="all",
                        help="HF dataset config (default: 'all' for all speakers/splits)")
    parser.add_argument("--splits", type=lambda s: s.split(","), default=None,
                        help=f"Default: all ({','.join(ALL_SPLITS)})")
    parser.add_argument("--cache_dir",          default="hifi_tts_mimi_cache")
    parser.add_argument("--k_codebooks",        type=int, default=8)
    parser.add_argument("--mimi_model_name",    default="kyutai/mimi")
    parser.add_argument("--device",             default="cuda")
    parser.add_argument("--private",            action="store_true")
    parser.add_argument("--token",              default=None)
    parser.add_argument("--skip_extract",       action="store_true")
    parser.add_argument("--skip_push",          action="store_true")
    parser.add_argument("--skip_card",          action="store_true")
    args = parser.parse_args()

    splits    = args.splits or ALL_SPLITS
    cache_dir = Path(args.cache_dir).resolve()

    print(f"Source:    {args.source_hf_dataset}  config={args.hf_config}")
    print(f"Splits:    {splits}")
    print(f"Cache dir: {cache_dir}")
    print()

    for split in splits:
        print(f"=== {split} ===")

        if not args.skip_extract:
            extract_codes_from_hf_split(
                split           = split,
                source_dataset  = args.source_hf_dataset,
                hf_config       = args.hf_config,
                cache_dir       = cache_dir,
                k_codebooks     = args.k_codebooks,
                mimi_model_name = args.mimi_model_name,
                device          = args.device,
                token           = args.token,
            )

        if not args.skip_push:
            _push_split(
                split          = split,
                source_dataset = args.source_hf_dataset,
                hf_config      = args.hf_config,
                cache_dir      = cache_dir,
                repo_id        = args.repo_id,
                private        = args.private,
                token          = args.token,
            )

    if not args.skip_push and not args.skip_card:
        _upload_card(args.repo_id, args.token)

    print("\nAll done.")


if __name__ == "__main__":
    main()
