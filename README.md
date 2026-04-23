# Wren-Datasets

Tooling for extracting [Kyutai Mimi](https://huggingface.co/kyutai/mimi) neural-codec
tokens from public speech corpora and publishing them as HuggingFace datasets.

The published datasets underpin the [Wren](https://github.com/shangeth/wren-tts)
series of small speech LLMs — but are useful to anyone training Mimi-based speech
models who doesn't want to burn an hour of GPU re-extracting codes.

## Published datasets

| Dataset | Source | Rows | License |
|---|---|---|---|
| [shangeth/ljspeech-mimi-codes](https://huggingface.co/datasets/shangeth/ljspeech-mimi-codes) | LJSpeech (single speaker, 24 h) | ~13k | CC0 |
| [shangeth/librispeech-mimi-codes](https://huggingface.co/datasets/shangeth/librispeech-mimi-codes) | LibriSpeech standard splits | ~280k | CC-BY-4.0 |

Each row: `id`, `text`, `codes` (`int16[k=8][n_frames]` @ 12.5 fps), `n_frames`, `k_codebooks`, and `speaker_id` (LibriSpeech only). Codes are the full 8 codebooks — users slice `codes[:k]` for fewer.

## Using the published datasets

```python
from datasets import load_dataset
import torch

ds    = load_dataset("shangeth/librispeech-mimi-codes", split="train_clean_100")
ex    = ds[0]
codes = torch.tensor(ex["codes"], dtype=torch.long)       # [8, n_frames]
print(ex["id"], "→", ex["text"][:60])

# Decode back to 24 kHz audio
from transformers import MimiModel
mimi = MimiModel.from_pretrained("kyutai/mimi").cuda().eval()
with torch.no_grad():
    wav = mimi.decode(codes.unsqueeze(0).cuda()).audio_values[0].cpu()
```

## Reproducing the extraction

```bash
git clone https://github.com/shangeth/wren-datasets
cd wren-datasets
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login

export HF_TOKEN=hf_...

# LJSpeech — downloads ~2.6 GB, ~30 min GPU
python ljspeech.py --repo_id shangeth/ljspeech-mimi-codes --private

# LibriSpeech — choose splits. See --help for flags.
python librispeech.py --splits dev-clean,test-clean --private
python librispeech.py --splits train-clean-100,train-clean-360 --private
python librispeech.py --splits train-other-500 --cleanup_audio --private   # ~30 GB download
```

Both scripts are idempotent: already-cached `.pt` codes are skipped, and LibriSpeech
pushes per-split (so other splits already on the Hub stay untouched).

Common flags:

| Flag | Effect |
|---|---|
| `--skip_extract` | Skip download + Mimi encoding (codes must already be in `--cache_dir`) |
| `--skip_push` | Extract only, don't upload |
| `--local_dir DIR` | Save Arrow dataset locally; skip upload (dry run) |
| `--cleanup_audio` | (LibriSpeech) delete `.flac` + tar after each split is extracted |
| `--k_codebooks N` | Number of Mimi codebooks to extract (default 8) |
| `--private` | Create/update the HF repo as private |

## Design notes

- **Why int16?** Mimi's codebook_size is 2048 → fits trivially in `int16` (max 32767), halves storage vs `int32`.
- **Why all 8 codebooks?** Users can slice `codes[:k]` for smaller/faster models, but can't magically add codebooks back if we only publish 3. Publishing the full tensor is strictly more useful.
- **Why publish codes instead of audio?** The raw corpora are already on HF (`openslr/librispeech_asr`, `keithito/lj_speech`). Re-hosting audio is redundant; the novel (expensive-to-produce) artifact is the Mimi codes.
- **LJSpeech casing preserved, LibriSpeech lowercased.** LJSpeech's `metadata.csv` is already mixed-case; LibriSpeech's `.trans.txt` is ALL UPPER. We lowercase the latter since uppercase in ASR corpora is a transcription convention, not a stylistic choice.
- **Per-split `push_to_hub(split=X)`** (LibriSpeech) rather than `DatasetDict.push_to_hub` — lets you add new splits incrementally without clobbering existing ones on the Hub.
- **Split names are underscored.** HF disallows hyphens in split names (clash with shard filename pattern), so `train-clean-100` on disk becomes `train_clean_100` on the Hub.

## Repository layout

```
.
├── mimi.py              MimiCodec wrapper + int16 conversion helper
├── ljspeech.py          Download + extract + push LJSpeech
├── librispeech.py       Download + extract + push LibriSpeech (per split)
├── data_stats.py        Quick stats over a local cache
└── cards/
    ├── ljspeech.md      Uploaded as README.md to the LJSpeech dataset repo
    └── librispeech.md   Uploaded as README.md to the LibriSpeech dataset repo
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

Please also cite the upstream corpora (LJSpeech, LibriSpeech) per their respective licenses.

## Related

- [wren-tts](https://github.com/shangeth/wren-tts) — TTS model trained on these datasets.

## License

Apache-2.0 (tooling). See [LICENSE](LICENSE). The *published datasets* inherit the
license of their upstream corpus (CC0 for LJSpeech, CC-BY-4.0 for LibriSpeech) —
see each dataset card on the Hub.
