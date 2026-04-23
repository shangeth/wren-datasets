---
license: apache-2.0
language:
- en
task_categories:
- text-to-speech
tags:
- mimi
- neural-codec
- speech-synthesis
- jenny
- audio-tokens
- single-speaker
pretty_name: Jenny TTS Mimi Codes
size_categories:
- 10K<n<100K
---

# Jenny TTS — Mimi Codes

Pre-extracted [Kyutai Mimi](https://huggingface.co/kyutai/mimi) tokens for the
[Jenny TTS Dataset](https://huggingface.co/datasets/reach-vb/jenny_tts_dataset) —
a single female speaker, ~30h, clean studio-quality recordings. Apache-2.0 license.

## Schema

| Column | Type | Notes |
|---|---|---|
| `id` | string | e.g. `jenny_0` |
| `text` | string | mixed-case with punctuation |
| `codes` | `int16[k=8][n_frames]` | Mimi codebook indices @ 12.5 fps |
| `n_frames` | int32 | |
| `k_codebooks` | int32 | 8 |

No `speaker_id` column — single speaker dataset.

## Extraction details

- **Source:** [`reach-vb/jenny_tts_dataset`](https://huggingface.co/datasets/reach-vb/jenny_tts_dataset)
- **Codec:** [`kyutai/mimi`](https://huggingface.co/kyutai/mimi) @ 24 kHz, 12.5 fps
- **Resampling:** 48 kHz → 24 kHz
- **Text:** `transcription` column (mixed-case, not the lowercased `transcription_normalised`)

## Usage

```python
from datasets import load_dataset
import torch

ds = load_dataset("shangeth/jenny-mimi-codes", split="train")
ex = ds[0]
codes = torch.tensor(ex["codes"], dtype=torch.long)  # [8, n_frames]
print(ex["text"])
```

## Citation

```bibtex
@misc{wren2026,
  title  = {Wren: A Family of Small Open-Weight Models for Unified Speech-Text Modelling},
  author = {Shangeth Rajaa},
  year   = {2026},
  url    = {https://github.com/shangeth/wren}
}
```

## License

Apache-2.0.
