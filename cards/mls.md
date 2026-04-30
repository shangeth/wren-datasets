---
license: cc-by-4.0
language:
- nl
- fr
- de
- it
- pl
- pt
- es
task_categories:
- text-to-speech
- automatic-speech-recognition
tags:
- mimi
- neural-codec
- multilingual
- mls
- multilingual-librispeech
- audio-tokens
pretty_name: Multilingual LibriSpeech Mimi Codes
size_categories:
- 1M<n<10M
---

# Multilingual LibriSpeech (MLS) — Mimi Codes

Pre-extracted [Kyutai Mimi](https://huggingface.co/kyutai/mimi) neural-codec tokens
for [Multilingual LibriSpeech](https://huggingface.co/datasets/facebook/multilingual_librispeech) —
LibriVox audiobooks in 7 non-English languages.

English is intentionally excluded. For English Mimi codes, use:

- [shangeth/librispeech-mimi-codes](https://huggingface.co/datasets/shangeth/librispeech-mimi-codes) — LibriSpeech (~280k rows, 7 splits)
- [shangeth/libritts-r-mimi-codes](https://huggingface.co/datasets/shangeth/libritts-r-mimi-codes) — LibriTTS-R (~360k rows, 7 splits, 24 kHz native)
- [shangeth/vctk-mimi-codes](https://huggingface.co/datasets/shangeth/vctk-mimi-codes) — VCTK (~44k rows, 110 speakers w/ accents)
- [shangeth/jenny-mimi-codes](https://huggingface.co/datasets/shangeth/jenny-mimi-codes) — Jenny TTS (~21k rows, single speaker)
- [shangeth/ljspeech-mimi-codes](https://huggingface.co/datasets/shangeth/ljspeech-mimi-codes) — LJSpeech (~13k rows, single speaker)

## Configs (languages)

One HF dataset config per language:

| Config | Language | ISO | Approx hours (train) |
|---|---|---|---|
| `dutch`      | Dutch       | nl | ~1.5k |
| `french`     | French      | fr | ~1.1k |
| `german`     | German      | de | ~3.3k |
| `italian`    | Italian     | it | ~250  |
| `polish`     | Polish      | pl | ~100  |
| `portuguese` | Portuguese  | pt | ~160  |
| `spanish`    | Spanish     | es | ~920  |

## Splits (per config)

| Split | Description |
|---|---|
| `train`   | full training set |
| `dev`     | development |
| `test`    | test |
| `9_hours` | low-resource ~9h training subset |
| `1_hours` | low-resource ~1h training subset |

### Merged `all` config

A virtual `all` config aliases the per-language parquets via glob — no extra
storage, just a YAML entry. Use it to train a single multilingual model:

```python
ds = load_dataset("shangeth/mls-mimi-codes", "all", split="train")
```

Caveat: there is no `language` column, and `speaker_id` is per-language —
so speaker IDs may collide across languages (e.g. German speaker `12345`
is unrelated to French speaker `12345`). If you need clean per-language
speaker bookkeeping, load each language config separately.

## Schema

| Column | Type | Notes |
|---|---|---|
| `id` | string | utterance ID, format `{speaker}_{chapter}_{segment}` |
| `text` | string | transcript, mixed-case as-is from MLS |
| `speaker_id` | int32 | speaker ID (parsed from MLS string) |
| `chapter_id` | int32 | chapter ID |
| `codes` | `int16[k=8][n_frames]` | Mimi codebook indices @ 12.5 fps |
| `n_frames` | int32 | |
| `k_codebooks` | int32 | 8 |

## Extraction details

- **Codec:** [`kyutai/mimi`](https://huggingface.co/kyutai/mimi) @ 24 kHz, 12.5 fps
- **Resampling:** MLS audio is 48 kHz opus → resampled to 24 kHz at extraction
- **Codebooks:** all 8 extracted; slice `codes[:k]` for fewer
- **Source:** [`facebook/multilingual_librispeech`](https://huggingface.co/datasets/facebook/multilingual_librispeech)

## Usage

```python
from datasets import load_dataset
import torch

ds = load_dataset("shangeth/mls-mimi-codes", "german", split="dev")
ex = ds[0]
codes = torch.tensor(ex["codes"], dtype=torch.long)  # [8, n_frames]
print(ex["id"], "| speaker:", ex["speaker_id"], "|", ex["text"][:60])

# Decode back to 24 kHz audio
from transformers import MimiModel
mimi = MimiModel.from_pretrained("kyutai/mimi").cuda().eval()
with torch.no_grad():
    wav = mimi.decode(codes.unsqueeze(0).cuda()).audio_values[0].cpu()
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

@inproceedings{pratap2020mls,
  title     = {MLS: A Large-Scale Multilingual Dataset for Speech Research},
  author    = {Pratap, Vineel and Xu, Qiantong and Sriram, Anuroop and Synnaeve, Gabriel and Collobert, Ronan},
  booktitle = {Interspeech},
  year      = {2020}
}
```

## License

CC-BY-4.0 (inherited from MLS / LibriVox).
