"""
Extract Mimi codes for Multilingual LibriSpeech (MLS) and publish to the Hub.

MLS is the multilingual extension of LibriSpeech — read speech from LibriVox in
8 languages (no English here; that's covered by librispeech-mimi-codes). Each
language is a separate HF *config*; each config has 5 splits:

  train       full training data
  dev         development
  test        test
  9_hours     low-resource subset (~9h training)
  1_hours     low-resource subset (~1h training)

Source HF dataset:  facebook/multilingual_librispeech
Audio:              48 kHz opus → MimiCodec resamples to 24 kHz internally
Configs (langs):    dutch, french, german, italian, polish, portuguese, spanish

Disk-tight strategy: process one (language, split) at a time. Encode → push to
the Hub under config={lang}, split={split} → optionally drop the .pt cache for
that split → move to the next.

Cache layout (per (language, split)):
  <cache_dir>/<lang>/<split>/<id>.pt        Mimi codes
  <cache_dir>/<lang>/<split>.jsonl          metadata manifest written during
                                            extraction so the push phase does
                                            not have to re-stream audio.

Usage:
  export HF_TOKEN=hf_...

  # Everything (7 langs × 5 splits) — drops cache after each push:
  python mls.py --cleanup_cache --repo_id shangeth/mls-mimi-codes --private

  # One language, just dev/test:
  python mls.py \
    --languages dutch \
    --splits dev,test \
    --repo_id shangeth/mls-mimi-codes --private

  # Push already-cached splits (no re-encoding):
  python mls.py --languages german --splits train --skip_extract \
    --repo_id shangeth/mls-mimi-codes --private
"""

import argparse
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Iterator, Optional

import torch
from datasets import Dataset, Features, Sequence, Value
from tqdm import tqdm

from mimi import MimiCodec, to_int16


logger = logging.getLogger(__name__)


ALL_LANGUAGES = [
    "dutch",
    "french",
    "german",
    "italian",
    "polish",
    "portuguese",
    "spanish",
]

ALL_SPLITS = ["train", "dev", "test", "9_hours", "1_hours"]

SOURCE_DATASET = "facebook/multilingual_librispeech"

FEATURES = Features({
    "id":          Value("string"),
    "text":        Value("string"),
    "speaker_id":  Value("int32"),
    "chapter_id":  Value("int32"),
    "codes":       Sequence(Sequence(Value("int16"))),
    "n_frames":    Value("int32"),
    "k_codebooks": Value("int32"),
})


# -------- Helpers --------

def _hf_split_size(language: str, split: str, token: Optional[str]) -> Optional[int]:
    try:
        from datasets import load_dataset_builder
        builder = load_dataset_builder(SOURCE_DATASET, language, token=token)
        info    = builder.info.splits.get(split)
        return info.num_examples if info else None
    except Exception:
        return None


def _load_hf_stream(language: str, split: str, token: Optional[str]):
    from datasets import load_dataset
    return load_dataset(SOURCE_DATASET, language, split=split, streaming=True, token=token)


def _hf_split_name(split: str) -> str:
    """HF split names disallow hyphens; MLS splits are already underscored."""
    return split.replace("-", "_").replace(".", "_")


def _safe_int(v) -> int:
    """MLS stores speaker_id / chapter_id as strings."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _split_paths(cache_dir: Path, language: str, split: str) -> tuple[Path, Path]:
    """Return (codes_dir, manifest_path) for one (language, split)."""
    codes_dir = cache_dir / language / split
    manifest  = cache_dir / language / f"{split}.jsonl"
    return codes_dir, manifest


# -------- Encoding --------

def extract_split(
    language:         str,
    split:            str,
    cache_dir:        Path,
    k_codebooks:      int,
    mimi_model_name:  str,
    device:           str,
    token:            Optional[str],
) -> None:
    """Stream MLS (language, split), Mimi-encode, write codes + JSONL manifest."""
    codes_dir, manifest_path = _split_paths(cache_dir, language, split)
    codes_dir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["id"])
                except Exception:
                    pass

    logger.info(f"Streaming {SOURCE_DATASET}  config={language}  split={split}")
    ds    = _load_hf_stream(language, split, token)
    total = _hf_split_size(language, split, token)
    codec = MimiCodec(model_name=mimi_model_name, device=device, k_codebooks=k_codebooks)

    skipped = errors = encoded = 0
    with manifest_path.open("a", encoding="utf-8") as mf:
        for ex in tqdm(ds, desc=f"Encode {language}/{split}", total=total):
            utt_id   = str(ex.get("id") or ex.get("file") or "")
            if not utt_id:
                errors += 1
                continue
            out_path = codes_dir / f"{utt_id}.pt"

            if utt_id in seen and out_path.exists():
                skipped += 1
                continue

            try:
                audio = ex["audio"]
                wav   = torch.tensor(audio["array"]).unsqueeze(0).float()
                sr    = int(audio["sampling_rate"])
                codes = codec.encode(wav, sr)
                torch.save(codes, out_path)
            except Exception as e:
                logger.warning(f"Failed to encode {utt_id}: {e}")
                errors += 1
                continue

            mf.write(json.dumps({
                "id":         utt_id,
                "text":       ex.get("transcript", "") or "",
                "speaker_id": _safe_int(ex.get("speaker_id")),
                "chapter_id": _safe_int(ex.get("chapter_id")),
            }, ensure_ascii=False) + "\n")
            mf.flush()
            encoded += 1

    logger.info(f"{language}/{split}: encoded={encoded}  skipped={skipped}  errors={errors}")


# -------- Build Arrow dataset --------

def _iter_split_rows(
    language:  str,
    split:     str,
    cache_dir: Path,
) -> Iterator[dict]:
    """Read manifest + cached codes; no audio access required."""
    codes_dir, manifest_path = _split_paths(cache_dir, language, split)
    if not manifest_path.exists():
        return

    seen: set[str] = set()
    missing = 0

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Build {language}/{split}"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            utt_id = rec["id"]
            if utt_id in seen:
                continue          # de-dupe in case of crash/append duplication
            seen.add(utt_id)

            cache_path = codes_dir / f"{utt_id}.pt"
            if not cache_path.exists():
                missing += 1
                continue

            codes = to_int16(torch.load(cache_path, map_location="cpu", weights_only=True))
            yield {
                "id":          utt_id,
                "text":        rec.get("text", ""),
                "speaker_id":  int(rec.get("speaker_id", 0)),
                "chapter_id":  int(rec.get("chapter_id", 0)),
                "codes":       codes.tolist(),
                "n_frames":    int(codes.shape[1]),
                "k_codebooks": int(codes.shape[0]),
            }

    if missing:
        print(f"  {language}/{split}: skipped {missing} utterances (missing cached codes)")


def push_split(
    language:  str,
    split:     str,
    cache_dir: Path,
    repo_id:   str,
    private:   bool,
    token:     Optional[str],
) -> bool:
    _, manifest_path = _split_paths(cache_dir, language, split)
    if not manifest_path.exists():
        print(f"  Skip push ({language}/{split}): no manifest at {manifest_path}")
        return False

    ds = Dataset.from_generator(
        lambda l=language, s=split: _iter_split_rows(l, s, cache_dir),
        features=FEATURES,
    )
    if len(ds) == 0:
        print(f"  Skip push ({language}/{split}): no rows built")
        return False

    hf_split = _hf_split_name(split)
    print(f"  Pushing {language}/{hf_split} ({len(ds):,} rows) → {repo_id}")
    ds.push_to_hub(
        repo_id,
        config_name    = language,
        split          = hf_split,
        private        = private,
        token          = token,
        commit_message = f"Add {language}/{hf_split}",
    )
    return True


def cleanup_split_cache(language: str, split: str, cache_dir: Path) -> None:
    codes_dir, manifest_path = _split_paths(cache_dir, language, split)
    if codes_dir.exists():
        shutil.rmtree(codes_dir)
        print(f"  Removed {codes_dir}")
    if manifest_path.exists():
        manifest_path.unlink()
        print(f"  Removed {manifest_path}")


def _build_configs_block(api, repo_id: str) -> Optional[str]:
    """Discover configs/splits from the repo's parquet layout and emit the YAML
    block that HF needs in README.md frontmatter. Returns None if no parquets."""
    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")
    except Exception as e:
        logger.warning(f"Could not list repo files: {e}")
        return None

    configs: dict[str, set[str]] = {}
    for f in files:
        if not f.endswith(".parquet"):
            continue
        parts = f.split("/")
        if len(parts) != 2:
            continue
        cfg, fname = parts
        # fname is like "train-00000-of-00004.parquet" or "dev-00000-of-00001.parquet"
        # OR for multi-token splits like "9_hours-00000-of-00001.parquet"
        m = re.match(r"^(.+?)-\d{5}-of-\d{5}\.parquet$", fname)
        if not m:
            continue
        configs.setdefault(cfg, set()).add(m.group(1))

    if not configs:
        return None

    # Stable order: matches ALL_SPLITS, unknown splits sorted at the end
    order = {s: i for i, s in enumerate(ALL_SPLITS)}
    lines = ["configs:"]
    for cfg in sorted(configs):
        lines.append(f"- config_name: {cfg}")
        lines.append("  data_files:")
        for split in sorted(configs[cfg], key=lambda s: (order.get(s, 999), s)):
            lines.append(f"  - split: {split}")
            lines.append(f"    path: {cfg}/{split}-*")

    # Merged "all" config — globs across language directories. No extra files;
    # same parquets are served under both per-language and "all" configs.
    if len(configs) >= 2:
        all_splits: set[str] = set()
        for v in configs.values():
            all_splits |= v
        lines.append("- config_name: all")
        lines.append("  data_files:")
        for split in sorted(all_splits, key=lambda s: (order.get(s, 999), s)):
            lines.append(f"  - split: {split}")
            lines.append(f'    path: "*/{split}-*"')

    return "\n".join(lines)


def _upload_card(repo_id: str, token: Optional[str]) -> None:
    """Upload cards/mls.md as README.md, injecting a `configs:` YAML block
    derived from the actual parquet layout in the repo. Without this block,
    HF treats the repo as a single default config and the per-language
    organization is invisible."""
    card_path = Path(__file__).resolve().parent / "cards" / "mls.md"
    if not card_path.exists():
        return

    from huggingface_hub import HfApi
    api = HfApi(token=token)

    raw   = card_path.read_text(encoding="utf-8")
    block = _build_configs_block(api, repo_id)

    if block:
        m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.DOTALL)
        if m:
            front, body = m.group(1).rstrip(), m.group(2)
            new = f"---\n{front}\n{block}\n---\n{body}"
        else:
            new = f"---\n{block}\n---\n\n{raw}"
        api.upload_file(
            path_or_fileobj = new.encode("utf-8"),
            path_in_repo    = "README.md",
            repo_id         = repo_id,
            repo_type       = "dataset",
            commit_message  = "Update dataset card (with configs block)",
        )
        n_cfgs = block.count("config_name:")
        print(f"Uploaded dataset card (with {n_cfgs} configs detected)")
    else:
        api.upload_file(
            path_or_fileobj = str(card_path),
            path_in_repo    = "README.md",
            repo_id         = repo_id,
            repo_type       = "dataset",
            commit_message  = "Update dataset card",
        )
        print("Uploaded dataset card (no parquet files found yet)")


# -------- Driver --------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id",         default="shangeth/mls-mimi-codes")
    parser.add_argument("--languages", type=lambda s: s.split(","), default=None,
                        help=f"Comma-sep list. Default: all ({','.join(ALL_LANGUAGES)})")
    parser.add_argument("--splits",    type=lambda s: s.split(","), default=None,
                        help=f"Comma-sep list. Default: all ({','.join(ALL_SPLITS)})")
    parser.add_argument("--cache_dir",       default="mls_mimi_cache")
    parser.add_argument("--k_codebooks",     type=int, default=8)
    parser.add_argument("--mimi_model_name", default="kyutai/mimi")
    parser.add_argument("--device",          default="cuda")
    parser.add_argument("--private",         action="store_true")
    parser.add_argument("--token",           default=None)
    parser.add_argument("--skip_extract",    action="store_true")
    parser.add_argument("--skip_push",       action="store_true")
    parser.add_argument("--skip_card",       action="store_true")
    parser.add_argument("--cleanup_cache",   action="store_true",
                        help="Delete codes + manifest after each successful push.")
    parser.add_argument("--card_only",       action="store_true",
                        help="Just upload the dataset card (with configs block) and exit. "
                             "Use this to fix a repo whose README is missing the configs block.")
    args = parser.parse_args()

    if args.card_only:
        _upload_card(args.repo_id, args.token)
        return

    languages = args.languages or ALL_LANGUAGES
    splits    = args.splits    or ALL_SPLITS
    cache_dir = Path(args.cache_dir).resolve()

    bad_langs = [l for l in languages if l not in ALL_LANGUAGES]
    if bad_langs:
        parser.error(f"Unknown languages: {bad_langs}. Valid: {ALL_LANGUAGES}")
    bad_splits = [s for s in splits if s not in ALL_SPLITS]
    if bad_splits:
        parser.error(f"Unknown splits: {bad_splits}. Valid: {ALL_SPLITS}")

    print(f"Source:    {SOURCE_DATASET}")
    print(f"Languages: {languages}")
    print(f"Splits:    {splits}")
    print(f"Cache dir: {cache_dir}")
    print(f"Cleanup:   {args.cleanup_cache}")
    print()

    for language in languages:
        for split in splits:
            print(f"=== {language} / {split} ===")

            if not args.skip_extract:
                extract_split(
                    language        = language,
                    split           = split,
                    cache_dir       = cache_dir,
                    k_codebooks     = args.k_codebooks,
                    mimi_model_name = args.mimi_model_name,
                    device          = args.device,
                    token           = args.token,
                )

            pushed = True
            if not args.skip_push:
                pushed = push_split(
                    language  = language,
                    split     = split,
                    cache_dir = cache_dir,
                    repo_id   = args.repo_id,
                    private   = args.private,
                    token     = args.token,
                )

            if args.cleanup_cache and pushed and not args.skip_push:
                cleanup_split_cache(language, split, cache_dir)

    if not args.skip_push and not args.skip_card:
        print()
        _upload_card(args.repo_id, args.token)

    print("\nAll done.")


if __name__ == "__main__":
    main()
