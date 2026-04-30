---
license: cc-by-nc-4.0
language:
- en
task_categories:
- text-to-speech
tags:
- mimi
- neural-codec
- speech-synthesis
- expresso
- audio-tokens
- expressive-speech
- style-transfer
- disentanglement
pretty_name: Expresso (Conversational) Mimi Codes
size_categories:
- 10K<n<100K
---

# Expresso (Conversational) — Mimi Codes

Pre-extracted [Kyutai Mimi](https://huggingface.co/kyutai/mimi) tokens for
[Expresso](https://huggingface.co/datasets/nytopop/expresso-conversational) — 4 speakers
performing ~40h of conversational speech across 26 expressive styles.

> ⚠️ **License: CC-BY-NC-4.0** — non-commercial use only.

## Why Expresso for Wren?

Expresso is the most directly relevant dataset for **speech disentanglement research** —
the same speakers utter similar content across different styles (happy, sad, confused,
whispered, animal, etc.). This makes it ideal for studying what a model learns about:
- **Style** vs **content** vs **speaker identity**
- How expressive variation is encoded in Mimi's codebook hierarchy
- Controllable style transfer without changing speaker identity

## Schema

| Column | Type | Notes |
|---|---|---|
| `id` | string | e.g. `ex04-ex01_animal-animaldir_007_312480_1378320` |
| `text` | string | mixed-case with punctuation |
| `speaker_id` | int32 | 1–4 (from ex01–ex04) |
| `style` | string | expressive style of this utterance (e.g. `happy`, `sad`, `whisper`) |
| `other_speaker_id` | int32 | conversation partner's speaker ID |
| `other_style` | string | conversation partner's expressive style |
| `codes` | `int16[k=8][n_frames]` | Mimi codebook indices @ 12.5 fps |
| `n_frames` | int32 | |
| `k_codebooks` | int32 | 8 |

## Extraction details

- **Source:** [`nytopop/expresso-conversational`](https://huggingface.co/datasets/nytopop/expresso-conversational)
- **Codec:** [`kyutai/mimi`](https://huggingface.co/kyutai/mimi) @ 24 kHz, 12.5 fps
- **Resampling:** 48 kHz → 24 kHz

## Usage

```python
from datasets import load_dataset
import torch

ds = load_dataset("shangeth/expresso-mimi-codes", split="train")

# Filter by style
happy = ds.filter(lambda x: x["style"] == "happy")

# Style diversity per speaker
from collections import Counter
styles = Counter(ds["style"])
print(styles.most_common(5))

ex    = ds[0]
codes = torch.tensor(ex["codes"], dtype=torch.long)  # [8, n_frames]
print(f"Speaker {ex['speaker_id']} | style={ex['style']} | {ex['text'][:60]}")
```

## Links

- **Dataset extraction code:** [github.com/shangeth/wren-datasets](https://github.com/shangeth/wren-datasets)
- **Wren research project:** [github.com/shangeth/wren](https://github.com/shangeth/wren)
- **TTS models trained on these codes:** [github.com/shangeth/wren-tts](https://github.com/shangeth/wren-tts)

## Citation

```bibtex
@misc{wren2026,
  title  = {Wren: A Family of Small Open-Weight Models for Unified Speech-Text Modelling},
  author = {Shangeth Rajaa},
  year   = {2026},
  url    = {https://github.com/shangeth/wren}
}

@inproceedings{nguyen2023expresso,
  title     = {Expresso: A Benchmark and Analysis of Discrete Expressive Speech Resynthesis},
  author    = {Nguyen, Tu Anh and others},
  booktitle = {Interspeech},
  year      = {2023}
}
```

## License

**CC-BY-NC-4.0** — non-commercial use only. See [original dataset](https://huggingface.co/datasets/nytopop/expresso-conversational) for details.
