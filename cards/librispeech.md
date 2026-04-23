---
license: cc-by-4.0
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
- librispeech
- audio-tokens
pretty_name: LibriSpeech Mimi Codes
size_categories:
- 100K<n<1M
---

# LibriSpeech — Mimi Codes

Pre-extracted [Kyutai Mimi](https://huggingface.co/kyutai/mimi) neural-codec tokens for the
[LibriSpeech](https://www.openslr.org/12) corpus — multi-speaker English audiobook readings
from the LibriVox project.

**This dataset contains codes only, not audio.** For waveforms, use any of the LibriSpeech
mirrors (e.g. [openslr/librispeech_asr](https://huggingface.co/datasets/openslr/librispeech_asr));
these codes let you skip the ~hours of GPU extraction needed to train Mimi-based speech models.

## Schema

One row per utterance:

| Column | Type | Notes |
|---|---|---|
| `id` | string | `{speaker_id}-{chapter_id}-{utterance_id:04d}`, e.g. `103-1240-0000` |
| `text` | string | lowercased transcript |
| `speaker_id` | int32 | LibriSpeech speaker ID |
| `codes` | `int16[k=8][n_frames]` | Mimi codebook indices @ 12.5 fps |
| `n_frames` | int32 | = `codes.shape[1]` |
| `k_codebooks` | int32 | = 8 |

## Extraction details

- **Codec:** [`kyutai/mimi`](https://huggingface.co/kyutai/mimi) @ 24 kHz, 12.5 fps
- **Codebooks:** all 8 extracted. Slice `codes[:k]` for fewer (Mimi's codebooks are ordered
  by importance; the first few capture most of the signal).
- **Codebook size:** 2048 per codebook → values stored as `int16`
- **Transcripts:** sourced from LibriSpeech's `.trans.txt` files, **lowercased** (the raw
  release is ALL-UPPER)

## Splits

Each standard LibriSpeech split is a separate HF split (hyphens replaced with underscores):

| HF Split | Upstream | Approx. rows | Notes |
|---|---|---|---|
| `train_clean_100` | `train-clean-100` | ~28.5k | clean read speech, ~100 h |
| `train_clean_360` | `train-clean-360` | ~104.0k | clean read speech, ~360 h |
| `train_other_500` | `train-other-500` | ~148.7k | noisier/accented, ~500 h |
| `dev_clean` | `dev-clean` | ~2.7k | dev set, clean |
| `dev_other` | `dev-other` | ~2.9k | dev set, noisier |
| `test_clean` | `test-clean` | ~2.6k | test set, clean |
| `test_other` | `test-other` | ~2.9k | test set, noisier |

Splits are added incrementally — consult the "Files" tab or `load_dataset(...).splits` for
the exact subset currently available.

## Usage

```python
from datasets import load_dataset
import torch

ds = load_dataset("shangeth/librispeech-mimi-codes", split="train_clean_100")

ex    = ds[0]
codes = torch.tensor(ex["codes"], dtype=torch.long)   # [8, n_frames]
print(f"{ex['id']} (speaker {ex['speaker_id']}) → {ex['text'][:60]}")
print("codes:", codes.shape, "duration:", codes.shape[1] / 12.5, "s")

# Use only the first 3 codebooks:
codes_3 = codes[:3]
```

Streaming (no full download):

```python
ds = load_dataset("shangeth/librispeech-mimi-codes", split="train_clean_360", streaming=True)
for ex in ds.take(10):
    print(ex["id"], len(ex["codes"]), "codebooks")
```

Decode to audio with the Mimi decoder:

```python
from transformers import MimiModel
mimi = MimiModel.from_pretrained("kyutai/mimi").cuda().eval()
with torch.no_grad():
    wav = mimi.decode(codes.unsqueeze(0).cuda()).audio_values[0].cpu()
# wav is [1, T] @ 24 kHz
```

## License & Attribution

LibriSpeech is released under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
The derived Mimi codes inherit this license — **attribution is required**. Please cite
both the original corpus and this dataset when redistributing.

## Citations

```bibtex
@inproceedings{panayotov2015librispeech,
  title     = {Librispeech: an ASR corpus based on public domain audio books},
  author    = {Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle = {ICASSP},
  year      = {2015}
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
