# Reproducing MedTsLLM decoder-only classification (PTB-XL)

The public `flixpar/med-ts-llm` repo is the **MLHC-2024** version (the paper's prior
work [9]). It does **not** ship the classification task, the PTB-XL loader, or the
classification config that the IEEE/JBHI extended paper added. These files add them on
top of the existing framework, following its conventions.

## Files

New:
- `tasks/classification.py` — sequence-level classification trainer (CE loss; accuracy /
  macro-F1 / macro-precision / macro-recall, as in Table VIII).
- `datasets/ptbxl.py` — PTB-XL loader + a self-contained `ClassificationDataset` base
  (stores recordings as `[N, T, F]` with one label per recording).
- `configs/datasets/ptbxl.toml` — the decoder-only config (Table III hyperparameters).

Edited (see `medtsllm_classification.patch`):
- `models/medtsllm.py` — adds `"classification"` to `supported_tasks`, a sequence-level
  K-class output head, eval-time softmax, and a task description.
- `tasks/__init__.py` — registers `ClassificationTask`.
- `datasets/__init__.py` — registers the `PTB-XL` dataset.

Apply the patch from the repo root:
```bash
git apply medtsllm_classification.patch
# then copy tasks/classification.py, datasets/ptbxl.py, configs/datasets/ptbxl.toml into place
```

## Data layout

Download PTB-XL (v1.0.3) from PhysioNet and place it at `data/ptbxl/`:
```
data/ptbxl/
  ptbxl_database.csv
  scp_statements.csv
  records100/...        # 100 Hz waveforms (referenced by filename_lr)
```
The loader caches a processed `cache_100hz_{split}.npz` per split on first run.

It uses the predefined `strat_fold` split (folds 1–8 train, 9 val, 10 test, patient-level
separated) and maps each record to the dominant of the five diagnostic superclasses
(NORM, MI, STTC, CD, HYP), giving a single-label 5-class problem with softmax + CE, as the
paper describes.

## Run

```bash
python3 train.py configs/datasets/ptbxl.toml
```

`Llama-2-7b-hf` is gated — request access on HuggingFace and `huggingface-cli login`
first. The decoder-only variant in the paper is Llama-2-7B; the encoder-decoder variant
is Flan-T5-XL (swap `[models.medtsllm.llm].llm = "google/flan-t5-xl"` for that).

## Hyperparameters (Table III, Class.-PTB-XL)

history_len 512 · patch_len 32 · stride 16 · d_model 32 · d_ff 128 · lr 1e-4 · CE ·
batch 16. Covariate mode is `concat` (the 12 leads), consistent with the ablation
(Table IX) where concat/interleave perform best.

## Expected ballpark (Table VIII)

Decoder-only: ~66.85% accuracy, ~51.07 F1. Encoder-decoder: ~70.84% / ~60.00.

## Notes / deviations to be aware of

- PTB-XL records are 1000 samples (100 Hz × 10 s); with `history_len = 512` the loader
  center-crops each record to 512 samples. If you prefer to use the full record, set
  `history_len`/`pred_len` to 1000 (n_patches scales accordingly) — the head adapts.
- PTB-XL is natively multi-label; this loader reduces to single-label by taking the
  superclass with the highest summed likelihood and dropping records with no diagnostic
  superclass. That matches the single-label softmax/CE setup in the paper, but exact
  counts depend on this aggregation choice.
- Classification is wired for the batch-preserving covariate modes (concat, interleave,
  weighted-average, add). `independent`/`merge-end` raise `NotImplementedError`.
