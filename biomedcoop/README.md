# BiomedCoOp (standalone)

Faithful, Dassl-free reimplementation of **BiomedCoOp** (Koleilat et al., CVPR 2025)
for few-shot biomedical **image** classification by prompt-tuning **BiomedCLIP**.
This is its own module — it is unrelated to time-series models (different modality,
backbone, and task) and does not plug into a time-series trainer.

Official reference (recommended for full benchmarks): https://github.com/HealthX-Lab/BiomedCoOp

## How the code maps to the paper
| Paper | Code |
|-------|------|
| Learnable context (CoOp), init "a photo of a" | `model.PromptLearner` |
| LLM prompt query Q + ensemble (Eq. 3) | `prompt_ensemble.build_query`, `train.build_prompt_bank` |
| MAD outlier pruning (Eqs. 5-7) | `losses.mad_outlier_mask` |
| Selective ensemble Ps | `model.BiomedCoOp.forward` |
| SCCM, MSE(Tp, Pg) (Eq. 9) | `losses.sccm_loss` |
| KDSP, KL(teacher || student) (Eq. 10) | `losses.kdsp_loss` |
| L = CE + λ1·SCCM + λ2·KDSP (Eq. 11) | `model.BiomedCoOp.forward` + `train.py` |

Defaults (`config.yaml`): N_CTX=4, N=50 prompts, λ1=0.25, λ2=3.0, ζs(TAU)=1.25,
SGD lr 2.5e-3, cosine, 100 epochs (50 for base-to-novel), batch 4, 224px.

## Install & run
```bash
pip install torch open_clip_torch pyyaml pillow      # BiomedCLIP downloads from HuggingFace
python3 selftest.py                                   # offline math check (no backbone)
python3 train.py --root /data/BTMRI --modality MRI \
                 --prompts prompts/btmri_example.json --kshot 16
```
`--root` is an ImageFolder (`root/<class>/*.png`). Omit `--prompts` to use the
offline template fallback (runs, but use real GPT-4 prompts for paper-level results).

## Caveats
- BiomedCLIP's text tower is **PubMedBERT**, not CLIP's transformer; context is
  inserted into BERT word embeddings (ctx_dim=768). Only `prompt_learner.ctx` trains.
- The novel losses were validated against a numpy mirror (`selftest.py`). The
  BiomedCLIP forward path needs network access to HuggingFace to run.
