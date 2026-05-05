---
license: cc-by-nc-4.0
language:
- en
task_categories:
- text-to-speech
tags:
- mimi
- neural-codec
- expresso
- expressive-speech
- style-tags
- fine-tuning
pretty_name: Expresso — Tagged Mimi Codes (k=32)
size_categories:
- 10K<n<100K
---

# Expresso — Tagged Mimi Codes (k=32)

Training-ready derivative of [`shangeth/expresso-mimi-codes`](https://huggingface.co/datasets/shangeth/expresso-mimi-codes), built specifically for **style-conditioned TTS fine-tuning**.

> ⚠️ **License: CC-BY-NC-4.0** — non-commercial use only.

## What this differs from `expresso-mimi-codes`

This dataset:

1. **Merges** `read` + `conversational` configs into a **single flat dataset** per split (matches the canonical schema other `*-mimi-codes` datasets use).
2. **Drops 5 styles** whose ASR transcripts are not reliably aligned to the audio (the speaker is doing voice acting / non-verbal sounds): `animal`, `animaldir`, `child`, `childdir`, `nonverbal`.
3. **Prepends a style tag** to the text for 19 tagged styles. `default` rows are left untagged. The model learns "no tag = default voice, `<style>` = stylized delivery".

## Schema

| Column | Type | Notes |
|---|---|---|
| `id` | string | unchanged |
| `text` | string | tagged: `<style> {text}` for the 19 tags; bare for `default` |
| `speaker_id` | string | cast from int (1–4) |
| `codes` | `int16[32][n_frames]` | all 32 Mimi codebooks @ 12.5 fps; slice `codes[:k]` for fewer |
| `n_frames` | int32 | |
| `k_codebooks` | int32 | 32 |

## The 19 tags

```
EMOTIONAL    <happy> <sad> <angry> <fearful> <disgusted>
             <awe> <desire> <calm> <sympathetic>
DELIVERY     <laughing> <enunciated> <whisper> <fast> <projected>
PERFORMANCE  <confused> <sarcastic> <narration>
STATE        <bored> <sleepy>
```

Plus untagged `default` (~250 min / 4.2 h) as the no-tag baseline.

## Splits

| split | rows | hours |
|---|---|---|
| train | ~26k | ~37 |
| dev | ~1.1k | ~1.4 |
| test | ~1.1k | ~1.4 |

## Usage

```python
from datasets import load_dataset
import torch

ds = load_dataset("shangeth/expresso-mimi-codes-tagged", split="train")
ex = ds[0]
print(ex["text"])              # e.g. "<happy> Hello, how are you?"
print(ex["speaker_id"])        # "1" .. "4"
codes = torch.tensor(ex["codes"], dtype=torch.long)  # [32, n_frames]

# Use only first 8 codebooks (Moshi-style)
codes8 = codes[:8]
```

## Reproducing this dataset

```bash
python expresso_tagged.py \\
  --src_repo shangeth/expresso-mimi-codes \\
  --dst_repo shangeth/expresso-mimi-codes-tagged --private
```

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

**CC-BY-NC-4.0** — non-commercial use only.
