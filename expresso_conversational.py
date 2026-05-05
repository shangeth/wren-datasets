"""
Build and push the Expresso `conversational` config to shangeth/expresso.

Pipeline (each phase idempotent, skippable via --skip_*):
  1. MANIFEST   — parse splits/*.txt + VAD_segments.txt; intersect VAD turns with split
                  time windows; emit per-split lists of (file, channel, start, end, ...)
  2. ASR        — for each segment, slice mono channel from stereo source, resample
                  48k → 16k, transcribe with Whisper-Large-V3-Turbo. Cache as JSONL —
                  re-runs skip already-transcribed segments.
  3. PUSH       — build per-split HF Dataset (Audio @ 48k mono, transcribed) and push
                  each split as `conversational` config of shangeth/expresso.

Schema:
  id                e.g. ex01-ex02_default_001__ch1_23.88-28.14
  audio             Audio @ 48 kHz mono (the VAD turn, clipped to split window)
  text              Whisper Turbo transcript (mixed case + punctuation)
  speaker_id        this channel's speaker (1-4)
  style             this channel's expressive style (e.g. happy, whisper, animal)
  other_speaker_id  partner's speaker id
  other_style       partner's expressive style
  source_file_id    e.g. ex01-ex02_default_001
  channel           1 or 2
  start_s, end_s    clip times within source file (post VAD ∩ split intersection)

Splits (from official splits/*.txt): train / dev / test.

Usage:
  python expresso_conversational.py --repo_id shangeth/expresso --private
  python expresso_conversational.py --skip_asr --skip_push    # build manifest only
  python expresso_conversational.py --skip_manifest           # ASR + push only
"""

import argparse
import gc
import json
import logging
import re
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import torch
import torchaudio.transforms as TT
from datasets import Dataset, Features, Value, Audio
from huggingface_hub import HfApi
from tqdm import tqdm

from expresso_audio import DEFAULT_DATA_ROOT, _parse_split_line


logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = "conv_asr_cache.jsonl"
DEFAULT_MANIFEST   = "conv_manifest.json"
WHISPER_MODEL      = "openai/whisper-large-v3-turbo"
MIN_TURN_S         = 0.3   # drop sub-300ms VAD turns
MAX_TURN_S         = 28.0  # split anything longer (Whisper context is 30s)

FEATURES = Features({
    "id":                Value("string"),
    "audio":             Audio(sampling_rate=48000),
    "text":              Value("string"),
    "speaker_id":        Value("int32"),
    "style":             Value("string"),
    "other_speaker_id":  Value("int32"),
    "other_style":       Value("string"),
    "source_file_id":    Value("string"),
    "channel":           Value("int32"),
    "start_s":           Value("float32"),
    "end_s":             Value("float32"),
})


# -------- Parsing helpers --------

def _spk_int(s: str) -> int:
    """ex01 → 1"""
    return int(s.lstrip("ex"))


def _is_conv_id(file_id: str) -> bool:
    """Conversational IDs start with `exA-exB_…` (hyphen in speaker pair)."""
    return len(file_id) >= 5 and file_id[4] == "-"


def _parse_conv_id(file_id: str) -> Tuple[str, str, str, str]:
    """ex03-ex02_animal-animaldir_005 → (ex03, ex02, animal, animaldir)
       ex01-ex02_default_001          → (ex01, ex02, default, default)"""
    parts = file_id.split("_")
    spk1, spk2 = parts[0].split("-")
    style_part = "_".join(parts[1:-1])  # styles never contain '_' in this corpus
    if "-" in style_part:
        s1, s2 = style_part.split("-", 1)
    else:
        s1 = s2 = style_part
    return spk1, spk2, s1, s2


def _find_conv_wav(data_root: Path, file_id: str) -> Optional[Path]:
    """audio_48khz/conversational/{spk_pair}/{styles}/{file_id}.wav"""
    spk_pair = file_id.split("_", 1)[0]
    base = data_root / "audio_48khz" / "conversational" / spk_pair
    if not base.exists():
        return None
    matches = list(base.glob(f"*/{file_id}.wav"))
    return matches[0] if matches else None


_VAD_RX = re.compile(r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)")


def parse_vad(vad_path: Path) -> dict:
    """Returns {(file_id, channel): [(start, end), ...]} for conversational entries."""
    out: dict = {}
    with open(vad_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#") or "\t" not in line:
                continue
            key, segs = line.split("\t", 1)
            if "/" not in key:
                continue  # only conv has /channelN
            fid, ch_str = key.rsplit("/", 1)
            if not ch_str.startswith("channel"):
                continue
            ch = int(ch_str[len("channel"):])
            turns = [(float(s), float(e)) for s, e in _VAD_RX.findall(segs)]
            out[(fid, ch)] = turns
    return out


# -------- Phase 1: Manifest --------

def build_manifest(
    data_root:    Path,
    min_turn_s:   float = MIN_TURN_S,
    max_turn_s:   float = MAX_TURN_S,
) -> dict:
    """Parse splits + VAD, intersect, return {split: [seg dict, ...]}."""
    import soundfile as sf

    vad = parse_vad(data_root / "VAD_segments.txt")
    print(f"Parsed VAD: {len(vad):,} (file, channel) entries")

    file_dur_cache: dict = {}
    def _file_dur(fid: str) -> Optional[float]:
        if fid in file_dur_cache:
            return file_dur_cache[fid]
        wav = _find_conv_wav(data_root, fid)
        if wav is None:
            file_dur_cache[fid] = None
            return None
        info = sf.info(str(wav))
        d = info.frames / info.samplerate
        file_dur_cache[fid] = d
        return d

    manifest: dict = {"train": [], "dev": [], "test": []}
    for split in manifest:
        with open(data_root / "splits" / f"{split}.txt", encoding="utf-8") as f:
            entries = [_parse_split_line(L) for L in f]
        entries = [e for e in entries if e and _is_conv_id(e[0])]

        n_drop_short = n_drop_split = 0
        for file_id, t_start, t_end in tqdm(entries, desc=f"manifest/{split}"):
            dur = _file_dur(file_id)
            if dur is None:
                continue
            t_lo = t_start if t_start is not None else 0.0
            t_hi = t_end   if t_end   is not None else dur

            spk1, spk2, style1, style2 = _parse_conv_id(file_id)
            for ch in (1, 2):
                self_spk    = spk1 if ch == 1 else spk2
                other_spk   = spk2 if ch == 1 else spk1
                self_style  = style1 if ch == 1 else style2
                other_style = style2 if ch == 1 else style1
                for vs, ve in vad.get((file_id, ch), []):
                    s = max(t_lo, vs)
                    e = min(t_hi, ve)
                    if e - s < min_turn_s:
                        n_drop_short += 1
                        continue
                    # Split overly long turns into ≤max_turn_s pieces (rare).
                    # Re-check min_turn_s after splitting — last chunk may be tiny.
                    cuts = []
                    cur = s
                    while e - cur > max_turn_s:
                        cuts.append((cur, cur + max_turn_s))
                        cur += max_turn_s
                    cuts.append((cur, e))
                    cuts = [(cs, ce) for cs, ce in cuts if ce - cs >= min_turn_s]
                    for cs, ce in cuts:
                        manifest[split].append({
                            "id":               f"{file_id}__ch{ch}_{cs:.2f}-{ce:.2f}",
                            "source_file_id":   file_id,
                            "channel":          ch,
                            "start_s":          float(cs),
                            "end_s":            float(ce),
                            "speaker_id":       _spk_int(self_spk),
                            "style":            self_style,
                            "other_speaker_id": _spk_int(other_spk),
                            "other_style":      other_style,
                        })
        print(f"  {split}: {len(manifest[split]):,} segments (dropped {n_drop_short} short)")
    return manifest


# -------- Phase 2: ASR --------

def _load_cache(path: Path) -> dict:
    cache = {}
    if not path.exists():
        return cache
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                d = json.loads(line)
                cache[d["id"]] = d["text"]
            except Exception:
                continue
    return cache


def transcribe(
    data_root:   Path,
    manifest:    dict,
    cache_path:  Path,
    model_name:  str = WHISPER_MODEL,
    device:      str = "cuda",
    batch_size:  int = 8,
) -> dict:
    """Run Whisper Turbo on every segment not yet in cache. Append results to JSONL.
       Returns the full {id: text} cache."""
    import soundfile as sf
    from transformers import pipeline

    cache = _load_cache(cache_path)

    pending: list = []
    for split, segs in manifest.items():
        for s in segs:
            if s["id"] not in cache:
                pending.append(s)
    print(f"\nASR cache: {len(cache):,} done, {len(pending):,} pending")
    if not pending:
        return cache

    print(f"Loading {model_name} ...")
    pipe = pipeline(
        "automatic-speech-recognition",
        model       = model_name,
        device      = 0 if device == "cuda" else -1,
        torch_dtype = torch.float16,
    )
    gen_kwargs = {
        "language":                 "en",
        "task":                     "transcribe",
        "no_repeat_ngram_size":     4,
        "repetition_penalty":       1.2,
        "condition_on_prev_tokens": False,
    }
    resamplers: dict = {}

    def _read_segment(seg: dict):
        wav = _find_conv_wav(data_root, seg["source_file_id"])
        info = sf.info(str(wav))
        sr = info.samplerate
        offset = int(round(seg["start_s"] * sr))
        n_fr   = int(round((seg["end_s"] - seg["start_s"]) * sr))
        data, _ = sf.read(str(wav), start=offset, frames=n_fr,
                          dtype="float32", always_2d=True)
        ch_idx = seg["channel"] - 1
        arr = data[:, ch_idx] if data.shape[1] > ch_idx else data[:, 0]
        if sr == 16000:
            return arr
        if sr not in resamplers:
            resamplers[sr] = TT.Resample(sr, 16000)
        return resamplers[sr](torch.tensor(arr, dtype=torch.float32)).numpy()

    t0 = time.time()
    with open(cache_path, "a", encoding="utf-8") as fout, tqdm(total=len(pending), desc="ASR") as pbar:
        for i in range(0, len(pending), batch_size):
            batch_segs = pending[i : i + batch_size]
            try:
                inputs = [
                    {"raw": _read_segment(s), "sampling_rate": 16000}
                    for s in batch_segs
                ]
                results = pipe(
                    inputs,
                    batch_size        = len(inputs),
                    generate_kwargs   = gen_kwargs,
                    return_timestamps = False,
                )
            except Exception as ex:
                logger.warning(f"Batch {i}..{i+len(batch_segs)} failed: {ex} — falling back to single")
                results = []
                for s in batch_segs:
                    try:
                        r = pipe(
                            {"raw": _read_segment(s), "sampling_rate": 16000},
                            generate_kwargs=gen_kwargs, return_timestamps=False,
                        )
                        results.append(r)
                    except Exception as e2:
                        logger.warning(f"  single fail {s['id']}: {e2}")
                        results.append({"text": ""})
            for seg, res in zip(batch_segs, results):
                text = (res.get("text") or "").strip()
                cache[seg["id"]] = text
                fout.write(json.dumps({"id": seg["id"], "text": text}, ensure_ascii=False) + "\n")
            fout.flush()
            pbar.update(len(batch_segs))

    print(f"ASR done in {(time.time()-t0)/60:.1f} min")
    del pipe
    gc.collect(); torch.cuda.empty_cache()
    return cache


# -------- Phase 3: Build + push --------

def _iter_split_rows(
    data_root: Path,
    segs:      List[dict],
    cache:     dict,
) -> Iterator[dict]:
    import soundfile as sf

    import io
    missing = 0
    for seg in tqdm(segs, desc="build"):
        wav = _find_conv_wav(data_root, seg["source_file_id"])
        if wav is None:
            missing += 1; continue
        info = sf.info(str(wav))
        sr = info.samplerate
        offset = int(round(seg["start_s"] * sr))
        n_fr   = int(round((seg["end_s"] - seg["start_s"]) * sr))
        # Read as int16 to halve arrow-cache footprint (studio-quality recording,
        # int16 noise floor is well below recording noise — no audible loss).
        data, _ = sf.read(str(wav), start=offset, frames=n_fr,
                          dtype="int16", always_2d=True)
        ch_idx = seg["channel"] - 1
        arr = data[:, ch_idx] if data.shape[1] > ch_idx else data[:, 0]
        # Pre-encode to PCM_16 wav bytes so HF stores the compact representation.
        buf = io.BytesIO()
        sf.write(buf, arr, sr, format="wav", subtype="PCM_16")

        yield {
            "id":               seg["id"],
            "audio":            {"path": seg["id"] + ".wav", "bytes": buf.getvalue()},
            "text":             cache.get(seg["id"], ""),
            "speaker_id":       seg["speaker_id"],
            "style":            seg["style"],
            "other_speaker_id": seg["other_speaker_id"],
            "other_style":      seg["other_style"],
            "source_file_id":   seg["source_file_id"],
            "channel":          seg["channel"],
            "start_s":          float(seg["start_s"]),
            "end_s":            float(seg["end_s"]),
        }
    if missing:
        print(f"  Skipped {missing} segments (missing source wav)")


def push_split(
    data_root:    Path,
    repo_id:      str,
    split_name:   str,
    segs:         List[dict],
    cache:        dict,
    private:      bool,
    token:        Optional[str],
    local_dir:    Optional[Path] = None,
) -> None:
    print(f"\n=== conversational/{split_name} ({len(segs):,} segments) ===")
    ds = Dataset.from_generator(
        lambda: _iter_split_rows(data_root, segs, cache),
        features = FEATURES,
    )
    print(f"  built: {len(ds):,} rows")

    if local_dir:
        out = local_dir / "conversational" / split_name
        out.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(out))
        print(f"  saved → {out}")
        return

    ds.push_to_hub(
        repo_id,
        config_name    = "conversational",
        split          = split_name,
        private        = private,
        token          = token,
        commit_message = f"Upload Expresso conversational/{split_name}",
    )
    print(f"  pushed → {repo_id} (conversational/{split_name})")


def upload_card(repo_id: str, token: Optional[str]) -> None:
    card = Path(__file__).resolve().parent / "cards" / "expresso_audio.md"
    if not card.exists():
        return
    HfApi(token=token).upload_file(
        path_or_fileobj = str(card),
        path_in_repo    = "README.md",
        repo_id         = repo_id,
        repo_type       = "dataset",
        commit_message  = "Update dataset card with conversational config",
    )


# -------- Driver --------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id",        default="shangeth/expresso")
    parser.add_argument("--data_root",      default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest_path",  default=DEFAULT_MANIFEST)
    parser.add_argument("--cache_path",     default=DEFAULT_CACHE_PATH)
    parser.add_argument("--whisper_model",  default=WHISPER_MODEL)
    parser.add_argument("--splits",         default="dev,test,train",
                        help="Order matters — push smallest first to validate cheaply")
    parser.add_argument("--min_turn_s",     type=float, default=MIN_TURN_S)
    parser.add_argument("--max_turn_s",     type=float, default=MAX_TURN_S)
    parser.add_argument("--private",        action="store_true")
    parser.add_argument("--token",          default=None)
    parser.add_argument("--skip_manifest",  action="store_true")
    parser.add_argument("--skip_asr",       action="store_true")
    parser.add_argument("--skip_push",      action="store_true")
    parser.add_argument("--skip_card",      action="store_true")
    parser.add_argument("--local_dir",      default=None)
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--asr_batch_size", type=int, default=8,
                        help="Whisper batch size during ASR (8 fits comfortably on a 40GB A100)")
    args = parser.parse_args()

    data_root     = Path(args.data_root).resolve()
    manifest_path = Path(args.manifest_path).resolve()
    cache_path    = Path(args.cache_path).resolve()
    if not (data_root / "VAD_segments.txt").exists():
        raise FileNotFoundError(f"{data_root}: VAD_segments.txt missing — extract expresso.tar?")

    # Phase 1
    if args.skip_manifest and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for split in manifest:
            print(f"  loaded {split}: {len(manifest[split]):,}")
    else:
        manifest = build_manifest(data_root, args.min_turn_s, args.max_turn_s)
        manifest_path.write_text(json.dumps(manifest))
        print(f"\nWrote manifest → {manifest_path}")

    requested = [s.strip() for s in args.splits.split(",") if s.strip()]

    # Phase 2: only ASR the splits we'll actually push
    if not args.skip_asr:
        manifest_for_asr = {k: v for k, v in manifest.items() if k in requested}
        cache = transcribe(data_root, manifest_for_asr, cache_path, args.whisper_model,
                           args.device, args.asr_batch_size)
    else:
        cache = _load_cache(cache_path)
        print(f"Loaded ASR cache: {len(cache):,} entries")

    # Phase 3
    if args.skip_push:
        print("Done (no push).")
        return
    local_dir = Path(args.local_dir).resolve() if args.local_dir else None
    for split in requested:
        if split not in manifest:
            print(f"Skipping unknown split: {split}"); continue
        push_split(
            data_root, args.repo_id, split, manifest[split], cache,
            args.private, args.token, local_dir,
        )

    if local_dir is None and not args.skip_card:
        upload_card(args.repo_id, args.token)
        print("Uploaded dataset card")
        print(f"\nDone: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
