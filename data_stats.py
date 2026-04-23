"""
Print text-token and audio-frame length statistics for LJSpeech or LibriSpeech.

Usage:
  python data_stats.py                          # LJSpeech (default config)
  python data_stats.py --dataset librispeech
  python data_stats.py --dataset librispeech --librispeech_cache_dir librispeech_mimi_cache
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from config import Config


def percentile(sorted_vals, p):
    idx = int(len(sorted_vals) * p / 100)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def print_stats(name, values):
    s = sorted(values)
    n = len(s)
    mean = sum(s) / n
    print(f"\n  {name}  (n={n:,})")
    print(f"    min={s[0]}  max={s[-1]}  mean={mean:.1f}")
    print(f"    p50={percentile(s,50)}  p75={percentile(s,75)}  "
          f"p90={percentile(s,90)}  p95={percentile(s,95)}  p99={percentile(s,99)}")


def ljspeech_stats(cfg, tokenizer):
    metadata = Path(cfg.ljspeech_root) / "LJSpeech-1.1" / "metadata.csv"
    cache    = Path(cfg.mimi_cache_dir)

    text_lens, audio_frames = [], []
    missing = 0
    with open(metadata, encoding="utf-8") as f:
        for line in f:
            # Manual split — csv.reader mis-parses 16 LJSpeech rows that start with '"'.
            parts = line.rstrip("\n").split("|", 2)
            if len(parts) < 2:
                continue
            rid  = parts[0]
            text = (parts[2] if len(parts) > 2 and parts[2].strip() else parts[1]).lower()
            pt   = cache / f"{rid}.pt"
            if not pt.exists():
                missing += 1
                continue
            text_lens.append(len(tokenizer.encode(text, add_special_tokens=False)))
            codes = torch.load(pt, map_location="cpu")
            audio_frames.append(codes.shape[1])

    print(f"\nLJSpeech  (cache: {cache})  missing={missing}")
    print_stats("text tokens", text_lens)
    print_stats("audio frames (1 frame = 80ms at 12.5fps)", audio_frames)
    _threshold_table(text_lens, audio_frames, cfg)


def librispeech_stats(cfg, tokenizer):
    cache = Path(cfg.librispeech_cache_dir)

    text_lens, audio_frames = [], []
    missing = 0
    for split_name in cfg.librispeech_splits:
        split_dir = Path(cfg.librispeech_root) / "LibriSpeech" / split_name
        if not split_dir.exists():
            print(f"  WARNING: split not found: {split_dir}")
            continue
        for trans_file in sorted(split_dir.rglob("*.trans.txt")):
            with open(trans_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(" ", 1)
                    if len(parts) < 2:
                        continue
                    utt_id, text = parts
                    text = text.lower()
                    pt = cache / f"{utt_id}.pt"
                    if not pt.exists():
                        missing += 1
                        continue
                    text_lens.append(len(tokenizer.encode(text, add_special_tokens=False)))
                    codes = torch.load(pt, map_location="cpu")
                    audio_frames.append(codes.shape[1])

    print(f"\nLibriSpeech {cfg.librispeech_splits}  (cache: {cache})  missing={missing}")
    print_stats("text tokens", text_lens)
    print_stats("audio frames (1 frame = 80ms at 12.5fps)", audio_frames)
    _threshold_table(text_lens, audio_frames, cfg)


def _threshold_table(text_lens, audio_frames, cfg):
    print(f"\n  Coverage at current limits  "
          f"(max_text_tokens={cfg.max_text_tokens}, max_audio_frames={cfg.max_audio_frames}):")
    n = len(text_lens)
    text_ok  = sum(t <= cfg.max_text_tokens  for t in text_lens)
    audio_ok = sum(a <= cfg.max_audio_frames for a in audio_frames)
    both_ok  = sum(t <= cfg.max_text_tokens and a <= cfg.max_audio_frames
                   for t, a in zip(text_lens, audio_frames))
    print(f"    text  fits: {text_ok:,}/{n:,}  ({100*text_ok/n:.1f}%)")
    print(f"    audio fits: {audio_ok:,}/{n:,}  ({100*audio_ok/n:.1f}%)")
    print(f"    both  fit:  {both_ok:,}/{n:,}  ({100*both_ok/n:.1f}%)")

    print("\n  Retention at various max_audio_frames thresholds:")
    for thresh in [75, 100, 125, 150, 200, 250, 300]:
        kept = sum(a <= thresh for a in audio_frames)
        print(f"    {thresh:4d} frames ({thresh/12.5:.0f}s): {kept:,}/{n:,}  ({100*kept/n:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, help="Override config dataset")
    parser.add_argument("--librispeech_cache_dir", default=None)
    parser.add_argument("--librispeech_splits",    default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = Config.load(args.config) if args.config else Config()
    if args.dataset:
        cfg.dataset = args.dataset
    if args.librispeech_cache_dir:
        cfg.librispeech_cache_dir = args.librispeech_cache_dir
    if args.librispeech_splits:
        cfg.librispeech_splits = args.librispeech_splits.split(",")

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            "<|audio_sep|>", "<|audio_eos|>",
            "<|audio_start|>", "<|audio_end|>",
        ]
    })

    if cfg.dataset == "librispeech":
        librispeech_stats(cfg, tokenizer)
    else:
        ljspeech_stats(cfg, tokenizer)


if __name__ == "__main__":
    main()
