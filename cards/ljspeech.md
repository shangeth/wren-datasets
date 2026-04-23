---
license: cc0-1.0
language:
- en
task_categories:
- text-to-speech
- automatic-speech-recognition
tags:
- mimi
- neural-codec
- speech-synthesis
- speech-recognition
- ljspeech
- audio-tokens
pretty_name: LJSpeech Mimi Codes
size_categories:
- 10K<n<100K
---

# LJSpeech — Mimi Codes

Pre-extracted [Kyutai Mimi](https://huggingface.co/kyutai/mimi) neural-codec tokens for the
[LJSpeech](https://keithito.com/LJ-Speech-Dataset/) corpus — 13,100 English utterances
from a single female speaker reading public-domain audiobook passages (~24 hours).

**This dataset contains codes only, not audio.** For waveforms, go to the original LJSpeech
release; these codes are designed to be loaded alongside it for training Mimi-based speech
models without paying the ~1 hour of GPU extraction cost.

## Schema

One row per utterance:

| Column | Type | Notes |
|---|---|---|
| `id` | string | e.g. `LJ001-0001` |
| `text` | string | normalized transcript, original mixed-case preserved |
| `codes` | `int16[k=8][n_frames]` | Mimi codebook indices @ 12.5 fps |
| `n_frames` | int32 | = `codes.shape[1]` |
| `k_codebooks` | int32 | = 8 |

## Extraction details

- **Codec:** [`kyutai/mimi`](https://huggingface.co/kyutai/mimi) @ 24 kHz, 12.5 fps
- **Codebooks:** all 8 extracted. Slice `codes[:k]` if you want fewer (Mimi's codebooks are
  ordered by importance; the first few capture most of the signal).
- **Codebook size:** 2048 per codebook → values stored as `int16`
- **Transcripts:** the `normalized` column from `metadata.csv` (punctuation preserved,
  expanded numerics/abbreviations). Original mixed-case is kept — apply `.lower()` at
  load time if your model expects lowercase (e.g. to reproduce Wren-TTS training).

## Usage

```python
from datasets import load_dataset
import torch

ds = load_dataset("shangeth/ljspeech-mimi-codes", split="train")

ex    = ds[0]
codes = torch.tensor(ex["codes"], dtype=torch.long)   # [8, n_frames]
print(ex["id"], "→", ex["text"][:60])
print("codes:", codes.shape, "duration:", codes.shape[1] / 12.5, "s")

# Use only the first 3 codebooks (e.g. for a smaller model):
codes_3 = codes[:3]
```

Decode back to audio with the Mimi decoder:

```python
from transformers import MimiModel
mimi = MimiModel.from_pretrained("kyutai/mimi").cuda().eval()
with torch.no_grad():
    wav = mimi.decode(codes.unsqueeze(0).cuda()).audio_values[0].cpu()
# wav is [1, T] @ 24 kHz
```

## Splits

| Split | Rows |
|---|---|
| `train` | ~13,100 |

LJSpeech has no canonical train/val/test split — partition as your task requires.

## License

The underlying LJSpeech corpus is in the **public domain (CC0)**. The derived Mimi codes
inherit this license. You can use, redistribute, and modify without attribution, though
citing the original corpus is encouraged.

## Citation

```bibtex
@misc{ito2017lj,
  title  = {The LJ Speech Dataset},
  author = {Keith Ito and Linda Johnson},
  year   = {2017},
  url    = {https://keithito.com/LJ-Speech-Dataset/}
}

@article{defossez2024moshi,
  title   = {Moshi: a speech-text foundation model for real-time dialogue},
  author  = {D{\'e}fossez, Alexandre and others},
  year    = {2024}
}
```

## Related

Used to train the [Wren](https://huggingface.co/shangeth/Wren-TTS-360M-v1) series of
small speech LLMs.
