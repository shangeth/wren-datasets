"""
Benchmark ASR models on Expresso `read` speech, per expressive style.

We use the human-transcribed read split (which we built and pushed) as ground truth,
sample N utterances per style, and compute WER per (model, style). This tells us
which ASR is most robust on Expresso's expressive styles before we use it to
transcribe conversational (which has no ground truth).

Default models (transformers, no extra deps):
  - openai/whisper-large-v3-turbo
  - openai/whisper-large-v3

Optional (requires `pip install nemo_toolkit[asr]`):
  - nvidia/parakeet-tdt-1.1b
  - nvidia/parakeet-tdt-0.6b-v2

WER is computed with Whisper's EnglishTextNormalizer applied identically to refs and
predictions (lowercase, expand abbreviations, drop punctuation, normalize numbers).

Usage:
  # Download data first if not already extracted to data/expresso/
  python asr_benchmark.py --n_per_style 30
  python asr_benchmark.py --n_per_style 30 --models whisper-turbo,whisper-v3
  python asr_benchmark.py --n_per_style 30 --models whisper-turbo,parakeet-1.1b
"""

import argparse
import gc
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm

from expresso_audio import (
    DEFAULT_DATA_ROOT, FEATURES,
    _build_split, _read_transcripts,
)


# Model registry: shortname → (kind, hf_id)
MODELS = {
    "whisper-turbo":  ("whisper",  "openai/whisper-large-v3-turbo"),
    "whisper-v3":     ("whisper",  "openai/whisper-large-v3"),
    "parakeet-1.1b":  ("parakeet", "nvidia/parakeet-tdt-1.1b"),
    "parakeet-0.6b":  ("parakeet", "nvidia/parakeet-tdt-0.6b-v2"),
}


def _normalize(text: str):
    """Whisper's English normalizer if available, else basic."""
    try:
        from whisper.normalizers import EnglishTextNormalizer
        if not hasattr(_normalize, "_n"):
            _normalize._n = EnglishTextNormalizer()
        return _normalize._n(text)
    except Exception:
        import re
        text = text.lower()
        text = re.sub(r"[^\w\s']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


def sample_dev_utterances(data_root: Path, n_per_style: int, seed: int) -> list:
    """Sample N base read utterances per style from the dev split."""
    transcripts = _read_transcripts(data_root / "read_transcriptions.txt")
    rows = list(_build_split(data_root, data_root / "splits" / "dev.txt", transcripts))
    rows = [r for r in rows if r["corpus"] == "base" and r["text"]]

    by_style = defaultdict(list)
    for r in rows:
        by_style[r["style"]].append(r)

    rng = random.Random(seed)
    picked = []
    for style, ex in sorted(by_style.items()):
        rng.shuffle(ex)
        chosen = ex[:n_per_style]
        picked.extend(chosen)
        print(f"  {style:<12} {len(chosen):>3} / {len(ex):>3} available")
    return picked


def run_whisper(hf_id: str, samples: list) -> List[str]:
    from transformers import pipeline
    print(f"\nLoading {hf_id} ...")
    pipe = pipeline(
        "automatic-speech-recognition",
        model       = hf_id,
        device      = 0 if torch.cuda.is_available() else -1,
        torch_dtype = torch.float16,
    )
    # Anti-hallucination: prevent repetition loops + don't condition on prev (no compounding errors)
    gen_kwargs = {
        "language":              "en",
        "task":                  "transcribe",
        "no_repeat_ngram_size":  4,
        "repetition_penalty":    1.2,
        "condition_on_prev_tokens": False,
    }
    import torchaudio.transforms as TT
    resamplers = {}

    def _to_16k(arr, src_sr):
        if src_sr == 16000:
            return arr
        if src_sr not in resamplers:
            resamplers[src_sr] = TT.Resample(src_sr, 16000)
        return resamplers[src_sr](torch.tensor(arr, dtype=torch.float32)).numpy()

    preds = []
    t0 = time.time()
    for s in tqdm(samples, desc=hf_id.split("/")[-1]):
        audio = s["audio"]
        arr16 = _to_16k(audio["array"], audio["sampling_rate"])
        result = pipe(
            {"raw": arr16, "sampling_rate": 16000},
            generate_kwargs = gen_kwargs,
            return_timestamps = False,
        )
        preds.append(result["text"])
    print(f"  Done in {time.time() - t0:.1f}s")
    del pipe
    gc.collect(); torch.cuda.empty_cache()
    return preds


def run_parakeet(hf_id: str, samples: list) -> Optional[List[str]]:
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        print(f"\nSkipping {hf_id}: nemo_toolkit not installed (pip install 'nemo_toolkit[asr]')")
        return None
    import tempfile, soundfile as sf
    import torchaudio.transforms as TT

    print(f"\nLoading {hf_id} ...")
    asr = nemo_asr.models.ASRModel.from_pretrained(hf_id).to("cuda").eval()

    preds = []
    resamplers = {}
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        wav_paths = []
        for i, s in enumerate(samples):
            arr = torch.tensor(s["audio"]["array"], dtype=torch.float32)
            sr  = s["audio"]["sampling_rate"]
            if sr != 16000:
                if sr not in resamplers:
                    resamplers[sr] = TT.Resample(sr, 16000)
                arr = resamplers[sr](arr)
            path = f"{tmp}/{i}.wav"
            sf.write(path, arr.numpy(), 16000)
            wav_paths.append(path)
        out = asr.transcribe(wav_paths, batch_size=16)
        if out and not isinstance(out[0], str):
            out = [getattr(o, "text", str(o)) for o in out]
        preds = list(out)
    print(f"  Done in {time.time() - t0:.1f}s")
    del asr
    gc.collect(); torch.cuda.empty_cache()
    return preds


def compute_wer(samples: list, preds: List[str]) -> tuple:
    import jiwer
    by_style = defaultdict(lambda: {"refs": [], "hyps": []})
    for s, p in zip(samples, preds):
        ref, hyp = _normalize(s["text"]), _normalize(p or "")
        if not ref:
            continue
        by_style[s["style"]]["refs"].append(ref)
        by_style[s["style"]]["hyps"].append(hyp)

    per_style, all_refs, all_hyps = {}, [], []
    for style, d in sorted(by_style.items()):
        per_style[style] = (jiwer.wer(d["refs"], d["hyps"]) * 100, len(d["refs"]))
        all_refs.extend(d["refs"]); all_hyps.extend(d["hyps"])
    overall = jiwer.wer(all_refs, all_hyps) * 100
    return per_style, overall


def print_table(results: dict, samples: list):
    if not results:
        print("\nNo models ran.")
        return
    styles = sorted({st for r in results.values() for st in r[0]})
    width  = max(16, max(len(m) for m in results) + 2)
    header = f"{'style':<12}  " + "  ".join(f"{m:>{width}}" for m in results)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for st in styles:
        row = f"{st:<12}  "
        for m, (per_style, _) in results.items():
            wer, n = per_style.get(st, (None, 0))
            cell = f"{wer:5.2f}% (n={n})" if wer is not None else "--"
            row += f"  {cell:>{width}}"
        print(row)
    print("-" * len(header))
    overall_row = f"{'OVERALL':<12}  "
    for m, (_, overall) in results.items():
        overall_row += f"  {overall:>{width-1}.2f}%"
    print(overall_row)
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   default=DEFAULT_DATA_ROOT)
    parser.add_argument("--n_per_style", type=int, default=30)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--models",      default="whisper-turbo,whisper-v3",
                        help="Comma-separated. Available: " + ", ".join(MODELS))
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    if not (data_root / "splits" / "dev.txt").exists():
        raise FileNotFoundError(f"{data_root} not extracted")

    print("Sampling utterances per style …")
    samples = sample_dev_utterances(data_root, args.n_per_style, args.seed)
    print(f"\nTotal: {len(samples)} samples\n")

    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    results = {}
    for m in requested:
        if m not in MODELS:
            print(f"Unknown model: {m}"); continue
        kind, hf_id = MODELS[m]
        runner = {"whisper": run_whisper, "parakeet": run_parakeet}[kind]
        preds = runner(hf_id, samples)
        if preds is None:
            continue
        results[m] = compute_wer(samples, preds)

    print_table(results, samples)

    if results:
        # Show 3 worst-WER examples for the first model — useful for spot-checking
        first_model = list(results.keys())[0]
        kind, hf_id = MODELS[first_model]
        # Re-run prediction for spot check is wasteful; skip here
        print(f"\nWinner = lowest OVERALL: {min(results, key=lambda m: results[m][1])}")


if __name__ == "__main__":
    main()
