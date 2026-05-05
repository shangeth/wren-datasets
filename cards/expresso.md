---
license: cc-by-nc-4.0
language:
- en
task_categories:
- text-to-speech
- automatic-speech-recognition
tags:
- mimi
- neural-codec
- speech-synthesis
- expresso
- audio-tokens
- expressive-speech
- style-transfer
- disentanglement
pretty_name: Expresso — Mimi Codes (k=32)
size_categories:
- 10K<n<100K
configs:
- config_name: read
  data_files:
  - split: train
    path: read/train-*
  - split: dev
    path: read/dev-*
  - split: test
    path: read/test-*
- config_name: conversational
  data_files:
  - split: train
    path: conversational/train-*
  - split: dev
    path: conversational/dev-*
  - split: test
    path: conversational/test-*
---

# Expresso — Mimi Codes (k = 32)

Pre-extracted [Kyutai Mimi](https://huggingface.co/kyutai/mimi) tokens (all **32 codebooks**) for both the read and conversational subsets of [Expresso](https://huggingface.co/datasets/shangeth/expresso). Source audio + transcripts live in [`shangeth/expresso`](https://huggingface.co/datasets/shangeth/expresso); this dataset publishes the discrete-token version for training Mimi-based speech models without re-extracting.

> ⚠️ **License: CC-BY-NC-4.0** — non-commercial use only.

## Why Expresso for Wren?

Expresso is the most directly relevant dataset for **speech disentanglement research** — the same speakers utter similar content across different expressive styles (happy, sad, confused, whispered, animal, etc.). Useful for studying:
- **Style** vs **content** vs **speaker identity**
- How expressive variation is encoded across Mimi's 32 codebooks (semantic + acoustic hierarchy)
- Controllable style transfer at fixed speaker identity

## Configs

- **`read`** — ~11.6k mono utterances with **human transcripts**.
- **`conversational`** — ~15.9k mono per-utterance turns from stereo dialogues, transcribed with **Whisper Large V3 Turbo** (3.0% overall WER on a held-out human-transcribed set).

## Why 32 codebooks?

Most published speech-codec datasets use only the first 8 of Mimi's 32 codebooks (1 semantic + 7 acoustic), which is enough for the original [Moshi](https://arxiv.org/abs/2410.00037) recipe. We publish all 32 so you can:
- Train models on more codebooks for higher resynthesis fidelity
- Study which codebooks carry which content (style/timbre/prosody)
- Slice `codes[:k]` at load time to use any prefix

`k_codebooks` is stored per row so the schema works for both `k=32` and any subset you slice.

## Schemas

### `read`

| Column | Type | Notes |
|---|---|---|
| `id` | string | e.g. `ex01_confused_00001`; longform: `ex01_default_longform_00001` (full file in `train`) |
| `text` | string | human transcription (mixed case + punctuation); empty for chunked rows |
| `speaker_id` | int32 | 1–4 |
| `style` | string | `default`, `confused`, `enunciated`, `happy`, `laughing`, `narration`, `sad`, `whisper` |
| `substyle` | string | finer label (e.g. `default_emphasis`, `default_essentials`, `default_longform`) |
| `corpus` | string | `base` or `longform` |
| `start_s` / `end_s` | float32 | null for full-file rows |
| `codes` | `int16[32][n_frames]` | Mimi codebook indices @ 12.5 fps |
| `n_frames` | int32 | |
| `k_codebooks` | int32 | 32 |

### `conversational`

| Column | Type | Notes |
|---|---|---|
| `id` | string | e.g. `ex01-ex02_default_001__ch1_23.88-28.14` |
| `text` | string | Whisper Large V3 Turbo transcript |
| `speaker_id` | int32 | this channel's speaker (1–4) |
| `style` | string | this channel's style |
| `other_speaker_id` | int32 | partner's speaker id |
| `other_style` | string | partner's style |
| `source_file_id` | string | the parent stereo file |
| `channel` | int32 | 1 or 2 |
| `start_s` / `end_s` | float32 | turn boundaries within source file |
| `codes` | `int16[32][n_frames]` | Mimi codebook indices @ 12.5 fps |
| `n_frames` | int32 | |
| `k_codebooks` | int32 | 32 |

## Extraction details

- **Source audio**: [`shangeth/expresso`](https://huggingface.co/datasets/shangeth/expresso) (the official Expresso tar, segmented and built into HF format)
- **Codec**: [`kyutai/mimi`](https://huggingface.co/kyutai/mimi) @ 24 kHz, 12.5 fps, codebook size 2048 (fits int16)
- **Resampling**: 48 kHz mono → 24 kHz before encoding
- **Conversational text**: machine-transcribed (Whisper Large V3 Turbo with anti-hallucination decoding)

## Usage

```python
from datasets import load_dataset
import torch

# Pick a config
read = load_dataset("shangeth/expresso-mimi-codes", "read",          split="train")
conv = load_dataset("shangeth/expresso-mimi-codes", "conversational", split="train")

ex = read[0]
codes = torch.tensor(ex["codes"], dtype=torch.long)  # [32, n_frames]
print(f"speaker={ex['speaker_id']} style={ex['style']} | {ex['text'][:60]}")
print(f"codes shape: {codes.shape}  ({codes.shape[1]/12.5:.2f}s @ 12.5 fps)")

# Use only the first 8 codebooks (Moshi-style)
codes8 = codes[:8]

# Decode back to 24 kHz audio
from transformers import MimiModel
mimi = MimiModel.from_pretrained("kyutai/mimi").cuda().eval()
with torch.no_grad():
    wav = mimi.decode(codes.unsqueeze(0).cuda()).audio_values[0].cpu()  # [1, T] @ 24 kHz
```

## Links

- **Audio dataset**: [`shangeth/expresso`](https://huggingface.co/datasets/shangeth/expresso)
- **Extraction code**: [github.com/shangeth/wren-datasets](https://github.com/shangeth/wren-datasets)
- **Wren research**: [github.com/shangeth/wren](https://github.com/shangeth/wren)

## Citation

```bibtex
@inproceedings{nguyen2023expresso,
  title     = {Expresso: A Benchmark and Analysis of Discrete Expressive Speech Resynthesis},
  author    = {Nguyen, Tu Anh and Hsu, Wei-Ning and D'Avirro, Antony and Shi, Bowen and
               Gat, Itai and Fazel-Zarani, Maryam and Remez, Tal and Copet, Jade and
               Synnaeve, Gabriel and Hassid, Michael and Kreuk, Felix and Adi, Yossi and Dupoux, Emmanuel},
  booktitle = {Interspeech},
  year      = {2023}
}

@misc{wren2026,
  title  = {Wren: A Family of Small Open-Weight Models for Unified Speech-Text Modelling},
  author = {Shangeth Rajaa},
  year   = {2026},
  url    = {https://github.com/shangeth/wren}
}
```

## License

**CC-BY-NC-4.0** — non-commercial use only. See [`shangeth/expresso`](https://huggingface.co/datasets/shangeth/expresso) `original_metadata/LICENSE.txt`.
