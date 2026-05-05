---
license: cc-by-nc-4.0
language:
- en
task_categories:
- text-to-speech
- automatic-speech-recognition
tags:
- expressive-speech
- expresso
- emotional-speech
- style-transfer
- multi-speaker
pretty_name: Expresso (audio + text)
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

# Expresso — audio + text

A faithful re-publication of the official [Expresso](https://speechbot.github.io/expresso/)
dataset (Nguyen et al., Interspeech 2023) as a loadable HuggingFace audio dataset, sourced
directly from FAIR's official tar.

> ⚠️ **License: CC-BY-NC-4.0** — non-commercial use only.

## Configs

- **`read`** — 11.6k mono read-speech utterances with **human transcripts**.
- **`conversational`** — ~15.9k mono per-utterance turns derived from the stereo conversational dialogues, transcribed with **Whisper Large V3 Turbo**.

## `read` config

11.6k mono utterances at **48 kHz / 24-bit**, fully transcribed by humans.

| | train | dev | test |
|---|---|---|---|
| rows | 10,388 | 628 | 588 |

### Schema

| Column | Type | Notes |
|---|---|---|
| `id` | string | e.g. `ex01_confused_00001`; longform chunks: `ex01_default_longform_00001__0-16.49` |
| `audio` | Audio @ 48 kHz mono | |
| `text` | string | human-written transcription (mixed case, with punctuation) |
| `speaker_id` | int32 | 1–4 |
| `style` | string | one of: `default`, `confused`, `enunciated`, `happy`, `laughing`, `narration`, `sad`, `whisper` |
| `substyle` | string | finer-grained label, e.g. `default_emphasis`, `default_essentials`, `default_longform`, `narration_longform` |
| `corpus` | string | `base` (short utterances) or `longform` (multi-minute readings) |
| `start_s` | float32 | null for full-file rows; chunk start for longform |
| `end_s` | float32 | null for full-file rows; chunk end   for longform |

### Splits

We follow the official Expresso train/dev/test splits, with **one TTS-oriented deviation**:

- **base read** (~11,600 utterances): full-file rows, no slicing — official splits applied as-is.
- **longform read** (8 source files: `default_longform`, `narration_longform` × 4 speakers): kept as **full files in `train` only**. The official Expresso splits slice each longform file into 3 non-overlapping chunks (60 s for dev/test, the rest for train) for resynthesis benchmarking. Those chunks don't align with the full-file transcripts, so for TTS/ASR we keep the longform audio + transcript intact and place the full files in `train` only. If you need the official chunked benchmark, see `original_metadata/splits/`.
- **singing** is intentionally **excluded** (only 12 wavs total, not in official splits).

All rows have aligned `(audio, text)` pairs.

### Style coverage per speaker

All 4 speakers have all 8 styles, with these caveats:
- `narration` is **longform-only** for all speakers (1 file each).
- `default` includes the substyles `default`, `default_emphasis`, `default_essentials`, `default_longform`.

---

## `conversational` config

~15.9k per-utterance mono turns derived from the official 339 stereo dialog files. Each row is **one speaker's turn** at a known time range within the source file, transcribed by Whisper.

| | train | dev | test |
|---|---|---|---|
| rows | ~14.8k | ~520 | ~515 |
| audio | ~29 h | ~50 min | ~51 min |

### Schema

| Column | Type | Notes |
|---|---|---|
| `id` | string | e.g. `ex01-ex02_default_001__ch1_23.88-28.14` |
| `audio` | Audio @ 48 kHz mono | the VAD-extracted turn from one channel |
| `text` | string | Whisper Large V3 Turbo transcript (mixed case + punctuation) |
| `speaker_id` | int32 | this channel's speaker (1–4) |
| `style` | string | this channel's expressive style |
| `other_speaker_id` | int32 | partner's speaker id |
| `other_style` | string | partner's expressive style |
| `source_file_id` | string | e.g. `ex01-ex02_default_001` (the stereo source) |
| `channel` | int32 | 1 or 2 |
| `start_s` | float32 | turn start within source file (after VAD ∩ split clip) |
| `end_s` | float32 | turn end |

### How it was built

1. **Parse** the official `splits/{train,dev,test}.txt` time-window assignments per source file.
2. **Intersect** each split window with `VAD_segments.txt` (per-channel pyannote turns) — turns straddling the dev/test boundary are **clipped to the split window** so dev/test never leak into train.
3. **Slice** the stereo source file → mono channel → 48 kHz mono turn.
4. **Transcribe** with `openai/whisper-large-v3-turbo`, with anti-hallucination decoding (`no_repeat_ngram_size=4`, `repetition_penalty=1.2`, `condition_on_prev_tokens=False`) and pre-resampled to 16 kHz.

### Turn filtering

- **Min duration**: 0.3 s. Sub-300ms VAD turns (mostly backchannels and clicks) are dropped.
- **Max duration**: 28 s. Long turns are split into ≤28 s pieces (Whisper's context is 30 s).

### Style coverage

26 styles total in the conversational subset, including styles **not present in `read`**: `angry`, `animal`, `awe`, `bored`, `calm`, `desire`, `disgusted`, `fast`, `fearful`, `nonverbal`, `projected`, `sarcastic`, `sleepy`, `sympathetic`, plus mixed pairs like `animal-animaldir` and `child-childdir` (where the two channels carry different styles — one row's `style` and `other_style` will differ).

### ASR quality (validated against `read` ground truth)

We benchmarked Whisper Large V3 Turbo on 210 human-transcribed read utterances spanning all 7 transcribed read styles. Per-style WER:

| confused | default | sad | happy | enunciated | laughing | whisper | **overall** |
|---|---|---|---|---|---|---|---|
| 0.96% | 1.67% | 2.00% | 2.76% | 3.18% | 4.98% | 5.31% | **3.00%** |

ASR errors are highest on `whisper` and `laughing` styles (the toughest acoustic conditions), but still under 6% WER. Conversational rows are expected to track the same per-style quality.

### Caveats

- Transcripts are **machine-generated** — expect a small error rate, especially on whispered/laughing/animal-style turns.
- Mixed-style pairs (`animal-animaldir`, `child-childdir`, `sad-sympathetic` and reversals) — speakers in the two channels carry different styles. Ground-truth styles are encoded per-row in `style` (this channel) and `other_style` (partner).

---

## Sidecar files

The original FAIR metadata is uploaded under `original_metadata/`:
- `original_metadata/README.txt`, `LICENSE.txt` — official Expresso documentation
- `original_metadata/read_transcriptions.txt` — per-file transcripts (tab-separated)
- `original_metadata/VAD_segments.txt` — per-channel VAD timings for the conversational subset (used to derive the `conversational` config)
- `original_metadata/splits/{train,dev,test}.txt`, `splits/README` — official split definitions

## Quick start

```python
from datasets import load_dataset

# Pick a config — there is no default
read = load_dataset("shangeth/expresso", "read",          split="train")
conv = load_dataset("shangeth/expresso", "conversational", split="train")

ex = read[0]
print(ex["id"], "|", ex["style"], "|", ex["text"])
print(ex["audio"]["array"].shape, "@", ex["audio"]["sampling_rate"], "Hz")

# Filter conv to mixed-style pairs (cross-style modeling)
mixed = conv.filter(lambda x: x["style"] != x["other_style"])
print(f"{len(mixed)} cross-style turns")

# Per-style coverage
from collections import Counter
print(Counter(conv["style"]).most_common(10))
```

## Reproducing this dataset

```bash
# Download the official Expresso tar (~36 GB) and extract:
mkdir -p data && cd data
curl -L https://dl.fbaipublicfiles.com/textless_nlp/expresso/data/expresso.tar | tar -xf -
cd ..

# Build + push:
python expresso_audio.py        --repo_id shangeth/expresso --private  # read config
python expresso_conversational.py --repo_id shangeth/expresso --private  # conversational config
```

See [github.com/shangeth/wren-datasets](https://github.com/shangeth/wren-datasets) for the full extraction code.

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

**CC-BY-NC-4.0** — non-commercial use only. See `original_metadata/LICENSE.txt`.
