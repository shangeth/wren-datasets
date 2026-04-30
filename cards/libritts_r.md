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
- libritts
- audio-tokens
pretty_name: LibriTTS-R Mimi Codes
size_categories:
- 100K<n<1M
---

# LibriTTS-R — Mimi Codes

Pre-extracted [Kyutai Mimi](https://huggingface.co/kyutai/mimi) neural-codec tokens
for [LibriTTS-R](https://www.openslr.org/141/) — a speech-restored version of
LibriTTS built specifically for TTS research.

**Why LibriTTS-R instead of LibriSpeech?**

| | LibriSpeech | LibriTTS-R |
|---|---|---|
| Purpose | ASR | TTS |
| Sample rate | 16 kHz | **24 kHz** (Mimi-native, no resampling) |
| Segmentation | Arbitrary chunks | **Sentence-level** |
| Punctuation | Stripped (ALL CAPS) | **Preserved** |
| Audio quality | Raw amateur | **Speech restoration applied** |

No resampling is needed — 24 kHz matches Mimi exactly.

## Schema

| Column | Type | Notes |
|---|---|---|
| `id` | string | e.g. `84_121123_000003_000000` |
| `text` | string | normalized text, **mixed-case with punctuation preserved** |
| `speaker_id` | int32 | LibriTTS speaker ID |
| `codes` | `int16[k=8][n_frames]` | Mimi codebook indices @ 12.5 fps |
| `n_frames` | int32 | |
| `k_codebooks` | int32 | 8 |

## Extraction details

- **Codec:** [`kyutai/mimi`](https://huggingface.co/kyutai/mimi) @ 24 kHz, 12.5 fps
- **Codebooks:** all 8 extracted. Slice `codes[:k]` for fewer.
- **Source:** [OpenSLR 141](https://www.openslr.org/141/)

## Splits

| HF Split | Source | ~Rows |
|---|---|---|
| `train_clean_100` | train-clean-100 | ~33.2k |
| `train_clean_360` | train-clean-360 | ~116k |
| `train_other_500` | train-other-500 | ~205k |
| `dev_clean` | dev-clean | ~2.7k |
| `dev_other` | dev-other | ~2.9k |
| `test_clean` | test-clean | ~2.6k |
| `test_other` | test-other | ~2.9k |

## Usage

```python
from datasets import load_dataset
import torch

ds = load_dataset("shangeth/libritts-r-mimi-codes", split="train_clean_100")
ex = ds[0]
codes = torch.tensor(ex["codes"], dtype=torch.long)  # [8, n_frames]
print(ex["text"])  # "He hoped there would be stew for dinner, turnips and carrots."
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

@inproceedings{koizumi2023libritts,
  title     = {LibriTTS-R: A Restored Multi-Speaker Text-to-Speech Corpus},
  author    = {Koizumi, Yuma and others},
  booktitle = {Interspeech},
  year      = {2023}
}
```

## License

CC-BY-4.0 (inherited from LibriTTS-R).
