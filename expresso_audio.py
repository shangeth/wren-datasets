"""
Build and push the Expresso `read` config (audio + text) to the Hugging Face Hub.

Source: official tar at
  https://dl.fbaipublicfiles.com/textless_nlp/expresso/data/expresso.tar  (~36 GB)

Expected layout (after extraction to --data_root):
  data/expresso/
    README.txt, LICENSE.txt
    read_transcriptions.txt
    VAD_segments.txt
    splits/{train,dev,test}.txt
    audio_48khz/read/{speaker}/{style}/{corpus}/{file_id}.wav

Schema (read config):
  id          string  e.g. ex01_confused_00001  (longform chunks: ex01_default_longform_00001__16.49-32.98)
  audio       Audio @ 48000 Hz, mono
  text        string  human transcription (full-file for longform — see card)
  speaker_id  int32   1..4
  style       string  e.g. default, confused, happy, narration, …
  substyle    string  e.g. default, default_emphasis, default_essentials, default_longform, narration_longform
  corpus      string  base | longform
  start_s     float32 null for full-file rows; chunk start for longform
  end_s       float32 null for full-file rows; chunk end   for longform

Splits (from splits/*.txt):
  - read base files (no time range)         → full-file rows
  - read longform files (with time range)   → sliced to chunk

Singing is intentionally excluded (not in official splits, only 12 wavs).
Conversational is in a separate config (handled in a future script).

Usage:
  export HF_TOKEN=hf_...
  python expresso_audio.py --repo_id shangeth/expresso --private
  python expresso_audio.py --repo_id shangeth/expresso --local_dir ./staged   # dry run
"""

import argparse
import logging
from pathlib import Path
from typing import Iterator, Optional, Tuple

from datasets import Dataset, Features, Value, Audio
from huggingface_hub import HfApi
from tqdm import tqdm


logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = "data/expresso"
EXPRESSO_TAR_URL  = "https://dl.fbaipublicfiles.com/textless_nlp/expresso/data/expresso.tar"

FEATURES = Features({
    "id":         Value("string"),
    "audio":      Audio(sampling_rate=48000),
    "text":       Value("string"),
    "speaker_id": Value("int32"),
    "style":      Value("string"),
    "substyle":   Value("string"),
    "corpus":     Value("string"),
    "start_s":    Value("float32"),
    "end_s":      Value("float32"),
})


# -------- Parsing helpers --------

def _spk_int(s: str) -> int:
    """ex01 → 1"""
    return int(s.lstrip("ex"))


def _parse_substyle(file_id: str) -> str:
    """
    ex01_confused_00001          → 'confused'
    ex01_default_emphasis_00010  → 'default_emphasis'
    ex01_default_longform_00001  → 'default_longform'
    """
    parts = file_id.split("_")
    return "_".join(parts[1:-1])


def _parse_split_line(line: str) -> Optional[Tuple[str, Optional[float], Optional[float]]]:
    """
    Returns (file_id, start_s, end_s) — start/end may be None.
    Skips comments / blanks. Tolerates both `(60.0s,)` and `(,60.0s)` and `(0,60s)`.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    parts = line.split("\t")
    file_id = parts[0]
    start_s, end_s = None, None
    if len(parts) > 1 and parts[1]:
        rng = parts[1].strip()
        if rng.startswith("(") and rng.endswith(")"):
            rng = rng[1:-1]
        s, e = (rng.split(",", 1) + [""])[:2]
        s = s.strip().rstrip("s").strip()
        e = e.strip().rstrip("s").strip()
        start_s = float(s) if s else None
        end_s   = float(e) if e else None
    return file_id, start_s, end_s


def _read_transcripts(path: Path) -> dict:
    out = {}
    if not path.exists():
        logger.warning(f"transcripts file missing: {path}")
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            fid, text = line.split("\t", 1)
            out[fid] = text
    return out


def _is_read_id(file_id: str) -> bool:
    """Read IDs start with `ex0X_` (underscore). Conv IDs start with `ex0X-` (hyphen)."""
    return len(file_id) >= 5 and file_id[4] == "_"


def _find_read_wav(data_root: Path, file_id: str) -> Optional[Path]:
    """audio_48khz/read/{speaker}/{*style}/{*corpus}/{file_id}.wav"""
    speaker = file_id.split("_", 1)[0]
    speaker_dir = data_root / "audio_48khz" / "read" / speaker
    if not speaker_dir.exists():
        return None
    matches = list(speaker_dir.glob(f"*/*/{file_id}.wav"))
    return matches[0] if matches else None


# -------- Row generation --------

def _row_id(file_id: str, start_s: Optional[float], end_s: Optional[float]) -> str:
    if start_s is None and end_s is None:
        return file_id
    s = f"{(start_s or 0.0):.2f}".rstrip("0").rstrip(".")
    e = f"{end_s:.2f}".rstrip("0").rstrip(".") if end_s is not None else "end"
    return f"{file_id}__{s}-{e}"


def _build_split(
    data_root:   Path,
    split_file:  Path,
    transcripts: dict,
) -> Iterator[dict]:
    import soundfile as sf

    with open(split_file, encoding="utf-8") as f:
        raw_entries = [_parse_split_line(L) for L in f]
    entries = [e for e in raw_entries if e and _is_read_id(e[0])]

    skipped = 0
    for file_id, start_s, end_s in tqdm(entries, desc=f"read/{split_file.stem}"):
        # Longform reads: FAIR's official splits slice them into 3 chunks across train/dev/test
        # for resynthesis benchmarking. For TTS/ASR, chunk text doesn't align — so we put each
        # full longform file (audio + full transcript) in train only, and skip dev/test entries.
        is_longform = _parse_substyle(file_id).endswith("_longform")
        if is_longform:
            if split_file.stem != "train":
                continue
            start_s, end_s = None, None  # use the full file in train

        wav = _find_read_wav(data_root, file_id)
        if wav is None:
            logger.warning(f"missing wav for {file_id}")
            skipped += 1
            continue

        rel_parts = wav.relative_to(data_root / "audio_48khz" / "read").parts
        # rel_parts: (speaker, style, corpus, basename.wav)
        speaker, style, corpus = rel_parts[0], rel_parts[1], rel_parts[2]

        info = sf.info(str(wav))
        sr   = info.samplerate
        if start_s is None and end_s is None:
            data, _ = sf.read(str(wav), dtype="float32", always_2d=False)
        else:
            dur    = info.frames / sr
            s      = float(start_s) if start_s is not None else 0.0
            e      = float(end_s)   if end_s   is not None else dur
            offset = int(round(s * sr))
            n_fr   = max(0, int(round((e - s) * sr)))
            data, _ = sf.read(
                str(wav), start=offset, frames=n_fr,
                dtype="float32", always_2d=False,
            )
        audio_entry = {"array": data, "sampling_rate": sr}

        yield {
            "id":         _row_id(file_id, start_s, end_s),
            "audio":      audio_entry,
            "text":       transcripts.get(file_id, ""),
            "speaker_id": _spk_int(speaker),
            "style":      style,
            "substyle":   _parse_substyle(file_id),
            "corpus":     corpus,
            "start_s":    float(start_s) if start_s is not None else None,
            "end_s":      float(end_s)   if end_s   is not None else None,
        }

    if skipped:
        print(f"  Skipped {skipped} entries from {split_file.name}")


# -------- Sidecar uploads --------

def _upload_sidecars(data_root: Path, repo_id: str, token: Optional[str]) -> None:
    api = HfApi(token=token)
    files = [
        ("README.txt",                "original_metadata/README.txt"),
        ("LICENSE.txt",               "original_metadata/LICENSE.txt"),
        ("read_transcriptions.txt",   "original_metadata/read_transcriptions.txt"),
        ("VAD_segments.txt",          "original_metadata/VAD_segments.txt"),
        ("splits/train.txt",          "original_metadata/splits/train.txt"),
        ("splits/dev.txt",            "original_metadata/splits/dev.txt"),
        ("splits/test.txt",           "original_metadata/splits/test.txt"),
        ("splits/README",             "original_metadata/splits/README"),
    ]
    for src_rel, dst in files:
        src = data_root / src_rel
        if not src.exists():
            logger.warning(f"sidecar missing: {src}")
            continue
        api.upload_file(
            path_or_fileobj = str(src),
            path_in_repo    = dst,
            repo_id         = repo_id,
            repo_type       = "dataset",
            commit_message  = f"Upload original metadata: {dst}",
        )


def _upload_card(repo_id: str, token: Optional[str]) -> None:
    card = Path(__file__).resolve().parent / "cards" / "expresso_audio.md"
    if not card.exists():
        return
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
    parser.add_argument("--repo_id",        default="shangeth/expresso")
    parser.add_argument("--data_root",      default=DEFAULT_DATA_ROOT,
                        help="Path to extracted expresso/ directory")
    parser.add_argument("--config_name",    default="read")
    parser.add_argument("--private",        action="store_true")
    parser.add_argument("--token",          default=None)
    parser.add_argument("--splits",         default="train,dev,test",
                        help="Comma-separated subset of {train,dev,test}")
    parser.add_argument("--skip_push",      action="store_true",
                        help="Build datasets locally without uploading")
    parser.add_argument("--skip_card",      action="store_true")
    parser.add_argument("--skip_sidecars",  action="store_true",
                        help="Don't upload original metadata files (transcripts, VAD, splits)")
    parser.add_argument("--local_dir",      default=None,
                        help="Save built datasets to this directory instead of pushing")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    if not (data_root / "splits").exists() or not (data_root / "audio_48khz" / "read").exists():
        raise FileNotFoundError(
            f"{data_root} doesn't look like an extracted expresso/ — "
            f"missing splits/ or audio_48khz/read/.\n"
            f"Get the tar from {EXPRESSO_TAR_URL} (~36 GB) and extract."
        )

    transcripts = _read_transcripts(data_root / "read_transcriptions.txt")
    print(f"Loaded {len(transcripts):,} read transcripts")

    requested = [s.strip() for s in args.splits.split(",") if s.strip()]
    splits = {s: data_root / "splits" / f"{s}.txt" for s in requested}

    for split_name, split_file in splits.items():
        if not split_file.exists():
            raise FileNotFoundError(split_file)
        print(f"\n=== {args.config_name}/{split_name} ===")
        ds = Dataset.from_generator(
            lambda f=split_file: _build_split(data_root, f, transcripts),
            features=FEATURES,
        )
        print(f"  {len(ds):,} rows")

        if args.local_dir:
            out = Path(args.local_dir).resolve() / args.config_name / split_name
            out.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(out))
            print(f"  saved → {out}")
            continue

        if args.skip_push:
            continue

        ds.push_to_hub(
            args.repo_id,
            config_name    = args.config_name,
            split          = split_name,
            private        = args.private,
            token          = args.token,
            commit_message = f"Upload Expresso {args.config_name}/{split_name}",
        )
        print(f"  pushed → {args.repo_id} ({args.config_name}/{split_name})")

    if args.skip_push or args.local_dir:
        print("\nDone (no upload).")
        return

    if not args.skip_sidecars:
        print("\nUploading original metadata sidecars …")
        _upload_sidecars(data_root, args.repo_id, args.token)

    if not args.skip_card:
        _upload_card(args.repo_id, args.token)
        print("Uploaded dataset card")

    print(f"\nDone: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
