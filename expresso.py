"""
Extract Mimi codes for Expresso (read + conversational) and push to HuggingFace.

Sources from the local extracted tar (data/expresso/) — no dependency on the audio
dataset being on HF. Uses helpers from expresso_audio and expresso_conversational
to find audio segments, then encodes each with Kyutai Mimi.

Two configs in shangeth/expresso-mimi-codes:
  - read           — sourced from splits/{train,dev,test}.txt + read_transcriptions.txt
  - conversational — sourced from conv_manifest.json + conv_asr_cache.jsonl

Each row carries Mimi codes for **all 32 codebooks** by default (the full Mimi output;
slice `codes[:k]` for fewer downstream). At Mimi's 12.5 fps × int16, a 5 s utterance
takes ~4 KB. Total cache for 40+ h of audio is ~120 MB.

Phases (each --skip_*-able, idempotent):
  1. EXTRACT — Mimi-encode each segment → cache .pt
  2. PUSH    — build HF Dataset per (config, split), push to HF Hub

Usage:
  export HF_TOKEN=hf_...

  # Both configs, all splits, 32 codebooks:
  python expresso.py --repo_id shangeth/expresso-mimi-codes --private

  # Re-push only (codes already cached):
  python expresso.py --skip_extract --private

  # Just read config:
  python expresso.py --configs read --private

License: CC-BY-NC-4.0 (non-commercial only)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import torch
from datasets import Dataset, Features, Sequence, Value
from huggingface_hub import HfApi
from tqdm import tqdm

from mimi import MimiCodec, to_int16
from expresso_audio import (
    DEFAULT_DATA_ROOT,
    _parse_split_line, _is_read_id, _find_read_wav,
    _read_transcripts, _parse_substyle, _row_id, _spk_int,
)
from expresso_conversational import (
    DEFAULT_MANIFEST, DEFAULT_CACHE_PATH,
    _find_conv_wav, _load_cache as _load_asr_cache,
)


logger = logging.getLogger(__name__)

DEFAULT_REPO         = "shangeth/expresso-mimi-codes"
DEFAULT_K_CODEBOOKS  = 32
DEFAULT_CACHE_ROOT   = "expresso_mimi_cache"


READ_FEATURES = Features({
    "id":          Value("string"),
    "text":        Value("string"),
    "speaker_id":  Value("int32"),
    "style":       Value("string"),
    "substyle":    Value("string"),
    "corpus":      Value("string"),
    "start_s":     Value("float32"),
    "end_s":       Value("float32"),
    "codes":       Sequence(Sequence(Value("int16"))),
    "n_frames":    Value("int32"),
    "k_codebooks": Value("int32"),
})

CONV_FEATURES = Features({
    "id":               Value("string"),
    "text":             Value("string"),
    "speaker_id":       Value("int32"),
    "style":            Value("string"),
    "other_speaker_id": Value("int32"),
    "other_style":      Value("string"),
    "source_file_id":   Value("string"),
    "channel":          Value("int32"),
    "start_s":          Value("float32"),
    "end_s":            Value("float32"),
    "codes":            Sequence(Sequence(Value("int16"))),
    "n_frames":         Value("int32"),
    "k_codebooks":      Value("int32"),
})


# -------- Read manifest (derived from splits + filesystem) --------

def _iter_read_segments(data_root: Path, splits: List[str]) -> Iterator[dict]:
    """Yield read segment dicts (no audio loaded) per split."""
    transcripts = _read_transcripts(data_root / "read_transcriptions.txt")
    for split in splits:
        with open(data_root / "splits" / f"{split}.txt", encoding="utf-8") as f:
            entries = [_parse_split_line(L) for L in f]
        entries = [e for e in entries if e and _is_read_id(e[0])]

        for file_id, t_start, t_end in entries:
            # Longform reads → full file in train only (matches expresso_audio.py)
            is_longform = _parse_substyle(file_id).endswith("_longform")
            if is_longform:
                if split != "train":
                    continue
                t_start, t_end = None, None

            wav = _find_read_wav(data_root, file_id)
            if wav is None:
                continue
            rel = wav.relative_to(data_root / "audio_48khz" / "read").parts
            speaker, style, corpus = rel[0], rel[1], rel[2]
            text = transcripts.get(file_id, "") if (t_start is None and t_end is None) else ""

            yield {
                "split":      split,
                "id":         _row_id(file_id, t_start, t_end),
                "wav":        str(wav),
                "start_s":    t_start,
                "end_s":      t_end,
                "text":       text,
                "speaker_id": _spk_int(speaker),
                "style":      style,
                "substyle":   _parse_substyle(file_id),
                "corpus":     corpus,
            }


def _iter_conv_segments(manifest: dict, asr_cache: dict, splits: List[str]) -> Iterator[dict]:
    """Yield conv segment dicts annotated with split + ASR text."""
    for split in splits:
        for seg in manifest.get(split, []):
            seg2 = dict(seg)
            seg2["split"] = split
            seg2["text"]  = asr_cache.get(seg["id"], "")
            yield seg2


# -------- Encoding --------

def _encode_segment_audio(wav_path: Path, start_s, end_s, channel: Optional[int] = None) -> Tuple[torch.Tensor, int]:
    """Read (and slice + pick channel if needed) a wav → float32 mono tensor + sample rate."""
    import soundfile as sf
    info = sf.info(str(wav_path))
    sr = info.samplerate
    if start_s is None and end_s is None:
        data, _ = sf.read(str(wav_path), dtype="float32", always_2d=channel is not None)
    else:
        s = float(start_s) if start_s is not None else 0.0
        e = float(end_s)   if end_s   is not None else info.frames / sr
        offset = int(round(s * sr))
        n_fr   = int(round((e - s) * sr))
        data, _ = sf.read(str(wav_path), start=offset, frames=n_fr,
                          dtype="float32", always_2d=channel is not None)
    if channel is not None:
        ch_idx = channel - 1
        data = data[:, ch_idx] if data.shape[1] > ch_idx else data[:, 0]
    return torch.tensor(data).float(), sr


def extract_codes(
    segments:  List[dict],
    cache_dir: Path,
    codec:     MimiCodec,
    desc:      str = "encode",
    is_conv:   bool = False,
    data_root: Optional[Path] = None,
) -> None:
    """Mimi-encode each segment (skip if cache .pt exists). Saves int16 [k, n_frames]."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    skipped = errors = 0
    for seg in tqdm(segments, desc=desc):
        cache_path = cache_dir / f"{seg['id']}.pt"
        if cache_path.exists():
            skipped += 1; continue
        try:
            if is_conv:
                wav = _find_conv_wav(data_root, seg["source_file_id"])
                wav_t, sr = _encode_segment_audio(wav, seg["start_s"], seg["end_s"], channel=seg["channel"])
            else:
                wav_t, sr = _encode_segment_audio(Path(seg["wav"]), seg["start_s"], seg["end_s"])
            codes = codec.encode(wav_t, sr)         # LongTensor [k, n_frames]
            torch.save(to_int16(codes), cache_path)  # int16 [k, n_frames]
        except Exception as ex:
            logger.warning(f"  fail {seg['id']}: {ex}")
            errors += 1
    print(f"  {desc}: skipped={skipped} errors={errors}")


# -------- Build HF dataset rows from cache --------

def _read_codes(cache_path: Path) -> torch.Tensor:
    return torch.load(cache_path, map_location="cpu", weights_only=True)


def _build_read_rows(segments: List[dict], cache_dir: Path) -> Iterator[dict]:
    missing = 0
    for seg in tqdm(segments, desc="build read"):
        cache_path = cache_dir / f"{seg['id']}.pt"
        if not cache_path.exists():
            missing += 1; continue
        codes = _read_codes(cache_path)
        yield {
            "id":          seg["id"],
            "text":        seg["text"],
            "speaker_id":  seg["speaker_id"],
            "style":       seg["style"],
            "substyle":    seg["substyle"],
            "corpus":      seg["corpus"],
            "start_s":     float(seg["start_s"]) if seg["start_s"] is not None else None,
            "end_s":       float(seg["end_s"])   if seg["end_s"]   is not None else None,
            "codes":       codes.tolist(),
            "n_frames":    int(codes.shape[1]),
            "k_codebooks": int(codes.shape[0]),
        }
    if missing:
        print(f"  missing {missing} cached codes")


def _build_conv_rows(segments: List[dict], cache_dir: Path) -> Iterator[dict]:
    missing = 0
    for seg in tqdm(segments, desc="build conv"):
        cache_path = cache_dir / f"{seg['id']}.pt"
        if not cache_path.exists():
            missing += 1; continue
        codes = _read_codes(cache_path)
        yield {
            "id":               seg["id"],
            "text":             seg["text"],
            "speaker_id":       seg["speaker_id"],
            "style":            seg["style"],
            "other_speaker_id": seg["other_speaker_id"],
            "other_style":      seg["other_style"],
            "source_file_id":   seg["source_file_id"],
            "channel":          seg["channel"],
            "start_s":          float(seg["start_s"]),
            "end_s":            float(seg["end_s"]),
            "codes":            codes.tolist(),
            "n_frames":         int(codes.shape[1]),
            "k_codebooks":      int(codes.shape[0]),
        }
    if missing:
        print(f"  missing {missing} cached codes")


# -------- Driver --------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id",         default=DEFAULT_REPO)
    parser.add_argument("--data_root",       default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest_path",   default=DEFAULT_MANIFEST)
    parser.add_argument("--asr_cache_path",  default=DEFAULT_CACHE_PATH)
    parser.add_argument("--cache_root",      default=DEFAULT_CACHE_ROOT,
                        help="Per-k cache subdir is created automatically (e.g. ./cache_k32/)")
    parser.add_argument("--configs",         default="read,conversational",
                        help="Comma-sep: read, conversational")
    parser.add_argument("--splits",          default="train,dev,test",
                        help="Comma-sep subset of train/dev/test (push order)")
    parser.add_argument("--k_codebooks",     type=int, default=DEFAULT_K_CODEBOOKS)
    parser.add_argument("--mimi_model_name", default="kyutai/mimi")
    parser.add_argument("--device",          default="cuda")
    parser.add_argument("--private",         action="store_true")
    parser.add_argument("--token",           default=None)
    parser.add_argument("--skip_extract",    action="store_true")
    parser.add_argument("--skip_push",       action="store_true")
    parser.add_argument("--skip_card",       action="store_true")
    args = parser.parse_args()

    data_root  = Path(args.data_root).resolve()
    cache_root = Path(args.cache_root).resolve() / f"k{args.k_codebooks}"
    cache_root.mkdir(parents=True, exist_ok=True)
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    splits  = [s.strip() for s in args.splits.split(",")  if s.strip()]

    # Lazy-load codec only if extracting
    codec = None
    if not args.skip_extract:
        codec = MimiCodec(
            model_name  = args.mimi_model_name,
            device      = args.device,
            k_codebooks = args.k_codebooks,
        )

    # ---- READ ----
    if "read" in configs:
        read_segs = list(_iter_read_segments(data_root, splits))
        print(f"\n=== read: {len(read_segs):,} segments across {splits} ===")
        read_cache_dir = cache_root / "read"
        if not args.skip_extract:
            extract_codes(read_segs, read_cache_dir, codec, desc="extract read")
        if not args.skip_push:
            for split in splits:
                segs_split = [s for s in read_segs if s["split"] == split]
                print(f"\n--- read/{split} ({len(segs_split):,} rows) ---")
                ds = Dataset.from_generator(
                    lambda gs=segs_split: _build_read_rows(gs, read_cache_dir),
                    features = READ_FEATURES,
                )
                print(f"  built {len(ds):,} rows")
                ds.push_to_hub(
                    args.repo_id,
                    config_name    = "read",
                    split          = split,
                    private        = args.private,
                    token          = args.token,
                    commit_message = f"Upload Mimi codes (k={args.k_codebooks}) read/{split}",
                )

    # ---- CONVERSATIONAL ----
    if "conversational" in configs:
        manifest_path = Path(args.manifest_path).resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"{manifest_path} — run expresso_conversational.py first")
        manifest = json.loads(manifest_path.read_text())
        asr_cache = _load_asr_cache(Path(args.asr_cache_path).resolve())
        conv_segs = list(_iter_conv_segments(manifest, asr_cache, splits))
        print(f"\n=== conversational: {len(conv_segs):,} segments across {splits} ===")
        conv_cache_dir = cache_root / "conv"
        if not args.skip_extract:
            extract_codes(conv_segs, conv_cache_dir, codec, desc="extract conv",
                          is_conv=True, data_root=data_root)
        if not args.skip_push:
            for split in splits:
                segs_split = [s for s in conv_segs if s["split"] == split]
                print(f"\n--- conversational/{split} ({len(segs_split):,} rows) ---")
                ds = Dataset.from_generator(
                    lambda gs=segs_split: _build_conv_rows(gs, conv_cache_dir),
                    features = CONV_FEATURES,
                )
                print(f"  built {len(ds):,} rows")
                ds.push_to_hub(
                    args.repo_id,
                    config_name    = "conversational",
                    split          = split,
                    private        = args.private,
                    token          = args.token,
                    commit_message = f"Upload Mimi codes (k={args.k_codebooks}) conversational/{split}",
                )

    # ---- Card ----
    if not args.skip_push and not args.skip_card:
        card = Path(__file__).resolve().parent / "cards" / "expresso.md"
        if card.exists():
            HfApi(token=args.token).upload_file(
                path_or_fileobj = str(card),
                path_in_repo    = "README.md",
                repo_id         = args.repo_id,
                repo_type       = "dataset",
                commit_message  = "Update dataset card",
            )
            print("Uploaded dataset card")

    print(f"\nDone: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
