# Wren-Datasets

Tooling for extracting [Kyutai Mimi](https://huggingface.co/kyutai/mimi) neural-codec
tokens from public speech corpora and publishing them as HuggingFace datasets.

The published datasets underpin the [Wren](https://github.com/shangeth/wren-tts)
series of small speech LLMs — but are useful to anyone training Mimi-based speech
models who doesn't want to burn GPU hours re-extracting codes.

## Published datasets

| Dataset | Source | Rows | Splits | License |
|---|---|---|---|---|
| [shangeth/ljspeech-mimi-codes](https://huggingface.co/datasets/shangeth/ljspeech-mimi-codes) | LJSpeech | ~13k | `train` | CC0 |
| [shangeth/librispeech-mimi-codes](https://huggingface.co/datasets/shangeth/librispeech-mimi-codes) | LibriSpeech | ~280k | 7 splits | CC-BY-4.0 |
| [shangeth/libritts-r-mimi-codes](https://huggingface.co/datasets/shangeth/libritts-r-mimi-codes) | LibriTTS-R | ~360k | 7 splits | CC-BY-4.0 |
| [shangeth/hifi-tts-mimi-codes](https://huggingface.co/datasets/shangeth/hifi-tts-mimi-codes) | HiFi-TTS | ~290k | 6 splits | CC-BY-4.0 |
| [shangeth/vctk-mimi-codes](https://huggingface.co/datasets/shangeth/vctk-mimi-codes) | VCTK | ~44k | `train` | CC-BY-4.0 |
| [shangeth/jenny-mimi-codes](https://huggingface.co/datasets/shangeth/jenny-mimi-codes) | Jenny TTS | ~21k | `train` | Apache-2.0 |
| [shangeth/expresso-mimi-codes](https://huggingface.co/datasets/shangeth/expresso-mimi-codes) | Expresso (conversational) | ~40k | `train` | **CC-BY-NC-4.0** |

Each row: `id`, `text`, `codes` (`int16[k=8][n_frames]` @ 12.5 fps), `n_frames`,
`k_codebooks`. Speaker datasets also include `speaker_id`. VCTK also includes `accent`.
Codes are all 8 codebooks — slice `codes[:k]` for fewer.

## Quick start (loading)

```python
from datasets import load_dataset
import torch

ds    = load_dataset("shangeth/librispeech-mimi-codes", split="train_clean_100")
ex    = ds[0]
codes = torch.tensor(ex["codes"], dtype=torch.long)   # [8, n_frames]
print(ex["id"], "→", ex["text"][:60])

# Decode back to 24 kHz audio
from transformers import MimiModel
mimi = MimiModel.from_pretrained("kyutai/mimi").cuda().eval()
with torch.no_grad():
    wav = mimi.decode(codes.unsqueeze(0).cuda()).audio_values[0].cpu()
```

---

## Reproducing the extraction

```bash
git clone https://github.com/shangeth/wren-datasets
cd wren-datasets
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export HF_TOKEN=hf_...
```

### LJSpeech  (~13k rows, ~30 min GPU)

```bash
python ljspeech.py --repo_id shangeth/ljspeech-mimi-codes --private
```

**Splits:** `train`

---

### LibriSpeech  (~280k rows total, ~6h GPU)

```bash
python librispeech.py \
  --splits train-clean-100,train-clean-360,train-other-500,dev-clean,dev-other,test-clean,test-other \
  --repo_id shangeth/librispeech-mimi-codes --private
```

**Splits (HF names):** `train_clean_100`, `train_clean_360`, `train_other_500`, `dev_clean`, `dev_other`, `test_clean`, `test_other`

> `train-other-500` is ~30 GB — use `--cleanup_audio` to delete `.flac` after encoding:
> ```bash
> python librispeech.py --splits train-other-500 --cleanup_audio --repo_id shangeth/librispeech-mimi-codes --private
> ```

---

### LibriTTS-R  (~360k rows total, ~8h GPU)

Streams audio directly from HuggingFace — no manual download needed.

```bash
python libritts_r.py \
  --source_hf_dataset mythicinfinity/libritts_r \
  --hf_config all \
  --splits dev.clean,dev.other,test.clean,test.other,train.clean.100,train.clean.360,train.other.500 \
  --repo_id shangeth/libritts-r-mimi-codes --private
```

**Splits (HF names):** `dev_clean`, `dev_other`, `test_clean`, `test_other`, `train_clean_100`, `train_clean_360`, `train_other_500`

> For large train splits, run separately with `--cleanup_audio` if disk is tight.

---

### HiFi-TTS  (~290k rows total, ~6h GPU)

```bash
python hifi_tts.py \
  --splits train.clean,train.other,dev.clean,dev.other,test.clean,test.other \
  --repo_id shangeth/hifi-tts-mimi-codes --private
```

**Splits (HF names):** `train_clean`, `train_other`, `dev_clean`, `dev_other`, `test_clean`, `test_other`

---

### VCTK  (~44k rows, mic1 only, ~1h GPU)

```bash
python vctk.py --repo_id shangeth/vctk-mimi-codes --private
```

**Splits:** `train`  
Mic2 duplicates are automatically skipped (filtered by `_mic1` suffix in filename).

---

### Jenny TTS  (~21k rows, ~30 min GPU)

```bash
python jenny.py --repo_id shangeth/jenny-mimi-codes --private
```

**Splits:** `train`

---

### Expresso  (~40k rows, ~45 min GPU) ⚠️ CC-BY-NC-4.0

4 speakers × 26 expressive styles in conversational pairs. The `style`,
`other_speaker_id`, and `other_style` columns are preserved — key for disentanglement
research.

```bash
python expresso.py --repo_id shangeth/expresso-mimi-codes --private
```

**Splits:** `train`

---

## Common flags

| Flag | Effect |
|---|---|
| `--skip_extract` | Skip download + Mimi encoding (codes already in `--cache_dir`) |
| `--skip_push` | Extract only, don't upload |
| `--cleanup_audio` | Delete audio files after encoding each split (saves disk) |
| `--k_codebooks N` | Number of Mimi codebooks to extract (default 8) |
| `--private` | Create/update the HF repo as private |

All scripts are **idempotent** — already-cached `.pt` files are skipped on re-run.
LibriSpeech and LibriTTS-R push per-split so existing splits on the Hub stay untouched.

---

## Design notes

- **int16:** Mimi codebook_size is 2048 → fits in int16, halves storage vs int32.
- **All 8 codebooks:** Users slice `codes[:k]` for fewer; can't reconstruct extra codebooks later.
- **Codes not audio:** Raw corpora are already on HF. The novel artifact is the Mimi codes.
- **Casing:** All datasets preserve text casing as-is from their source (LJSpeech mixed-case, LibriSpeech pre-lowercased, LibriTTS-R/HiFi-TTS/VCTK/Jenny naturally cased).
- **Split names underscored:** HF disallows hyphens in split names → `train-clean-100` becomes `train_clean_100`.

## Repository layout

```
.
├── mimi.py              MimiCodec wrapper + int16 conversion helper
├── ljspeech.py          LJSpeech — download + extract + push
├── librispeech.py       LibriSpeech — download + extract + push (per split)
├── libritts_r.py        LibriTTS-R — stream from HF + encode + push (per split)
├── hifi_tts.py          HiFi-TTS — stream from HF + encode + push (per split)
├── vctk.py              VCTK — stream from HF + encode + push (mic1 only)
├── jenny.py             Jenny TTS — stream from HF + encode + push
├── expresso.py          Expresso conversational — stream from HF + encode + push
├── data_stats.py        Quick stats over a local cache
└── cards/               Dataset cards (uploaded as README.md to each HF dataset repo)
    ├── ljspeech.md
    ├── librispeech.md
    ├── libritts_r.md
    ├── hifi_tts.md
    ├── vctk.md
    ├── jenny.md
    └── expresso.md
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

Please also cite the upstream corpus for whichever datasets you use (see each dataset card for the relevant BibTeX).

## Related

- [wren-tts](https://github.com/shangeth/wren-tts) — TTS model trained on these datasets.

## License

Apache-2.0 (tooling). Published datasets inherit their upstream corpus license —
see each dataset card on the Hub for details.
