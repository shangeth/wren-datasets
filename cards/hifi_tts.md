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
- hifi-tts
- audio-tokens
pretty_name: HiFi-TTS Mimi Codes
size_categories:
- 100K<n<1M
---

# HiFi-TTS — Mimi Codes

Pre-extracted [Kyutai Mimi](https://huggingface.co/kyutai/mimi) neural-codec tokens
for [HiFi-TTS](https://www.openslr.org/109/) — professionally narrated audiobooks
recorded in studio conditions.

**Why HiFi-TTS?** Unlike LibriVox-based datasets (LibriSpeech, LibriTTS-R) which
use amateur home recordings, HiFi-TTS features professional voice actors with studio
microphones: SNR >32 dB, flat frequency response, no background noise. The quality
ceiling is fundamentally higher.

| | LibriTTS-R | HiFi-TTS |
|---|---|---|
| Source | LibriVox (amateur) | Professional audiobooks |
| Speakers | ~2,456 | **10** |
| Hours | ~585h | ~292h |
| Audio SNR | ~15–20 dB | **>32 dB** |
| Sample rate | 24 kHz | 44.1 kHz → resampled to 24 kHz for Mimi |

## Schema

| Column | Type | Notes |
|---|---|---|
| `id` | string | utterance ID from source manifest |
| `text` | string | normalized text, mixed-case with punctuation |
| `speaker_id` | int32 | speaker ID (10 speakers) |
| `codes` | `int16[k=8][n_frames]` | Mimi codebook indices @ 12.5 fps |
| `n_frames` | int32 | |
| `k_codebooks` | int32 | 8 |

## Extraction details

- **Codec:** [`kyutai/mimi`](https://huggingface.co/kyutai/mimi) @ 24 kHz, 12.5 fps
- **Resampling:** 44.1 kHz → 24 kHz applied at extraction time
- **Codebooks:** all 8 extracted
- **Source:** [OpenSLR 109](https://www.openslr.org/109/)

## Usage

```python
from datasets import load_dataset
import torch

ds = load_dataset("shangeth/hifi-tts-mimi-codes", split="train")
ex = ds[0]
codes = torch.tensor(ex["codes"], dtype=torch.long)  # [8, n_frames]
print(ex["text"], "| speaker:", ex["speaker_id"])
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

@inproceedings{bakhturina2021hi,
  title     = {Hi-Fi Multi-Speaker English TTS Dataset},
  author    = {Bakhturina, Evelina and others},
  booktitle = {Interspeech},
  year      = {2021}
}
```

## License

CC-BY-4.0 (inherited from HiFi-TTS).
