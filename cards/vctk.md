---
license: cc-by-4.0
language:
- en
task_categories:
- text-to-speech
tags:
- mimi
- neural-codec
- speech-synthesis
- vctk
- audio-tokens
- accents
pretty_name: VCTK Mimi Codes (mic1)
size_categories:
- 10K<n<100K
---

# VCTK — Mimi Codes (mic1)

Pre-extracted [Kyutai Mimi](https://huggingface.co/kyutai/mimi) tokens for the
[VCTK Corpus](https://datashare.ed.ac.uk/handle/10283/2950) — 109 speakers across
11 British, Scottish, and American accents. ~44h of read speech.

**Only mic1 recordings are included.** Each utterance was recorded with two
microphones; mic1 (close microphone) gives a cleaner signal. Mic2 duplicates are
excluded. Utterance IDs end in `_mic1` (e.g. `p225_001_mic1`).

## Schema

| Column | Type | Notes |
|---|---|---|
| `id` | string | e.g. `p225_001_mic1` |
| `text` | string | read sentence, mixed-case with punctuation |
| `speaker_id` | int32 | numeric speaker ID (225 for p225) |
| `accent` | string | e.g. `English`, `Scottish`, `American` |
| `codes` | `int16[k=8][n_frames]` | Mimi codebook indices @ 12.5 fps |
| `n_frames` | int32 | |
| `k_codebooks` | int32 | 8 |

## Extraction details

- **Source:** [`sanchit-gandhi/vctk`](https://huggingface.co/datasets/sanchit-gandhi/vctk)
- **Codec:** [`kyutai/mimi`](https://huggingface.co/kyutai/mimi) @ 24 kHz, 12.5 fps
- **Resampling:** 48 kHz → 24 kHz
- **Filter:** `file` column stem must end with `_mic1`

## Usage

```python
from datasets import load_dataset
import torch

ds = load_dataset("shangeth/vctk-mimi-codes", split="train")
ex = ds[0]
codes = torch.tensor(ex["codes"], dtype=torch.long)  # [8, n_frames]
print(ex["id"], ex["accent"], "→", ex["text"])
```

## Citation

```bibtex
@misc{wren2026,
  title  = {Wren: A Family of Small Open-Weight Models for Unified Speech-Text Modelling},
  author = {Shangeth Rajaa},
  year   = {2026},
  url    = {https://github.com/shangeth/wren}
}

@inproceedings{veaux2017cstr,
  title     = {CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit},
  author    = {Veaux, Christophe and Yamagishi, Junichi and MacDonald, Kirsten},
  year      = {2017}
}
```

## License

CC-BY-4.0.
