"""
BiomedCoOp losses (Koleilat et al., CVPR 2025), faithful to the official
HealthX-Lab/BiomedCoOp BiomedCLIP trainer.

  L = L_CE + lambda1 * L_SCCM + lambda2 * L_KDSP     (Eq. 11)

  L_SCCM : MSE(Tp, Pg)                               (Eq. 9)   full LLM ensemble
  L_KDSP : KL(teacher || student)                    (Eq. 10)  selective ensemble (teacher)
  MAD    : modified z-score outlier pruning          (Eqs. 5-7)

The numerics here were verified against a numpy mirror of the reference ops.
"""

import torch
import torch.nn.functional as F


def mad_outlier_mask(scores: torch.Tensor, tau: float) -> torch.Tensor:
    """Median-Absolute-Deviation pruning over per-prompt scores S (shape [N]).

    Eqs. 5-7:  Ms = median(S);  D = median(|S - Ms|);  z = (S - Ms) / D
    The official code then re-standardises z and keeps |(z - mean z)/std z| <= tau.
    Returns a boolean keep-mask of shape [N].
    """
    s_bar = torch.median(scores)
    d_bar = torch.median(torch.abs(scores - s_bar))
    z = (scores - s_bar) / (d_bar + 1e-8)
    return torch.abs((z - z.mean()) / (z.std() + 1e-8)) <= tau


def prompt_scores(image_features, prompt_bank, logit_scale):
    """Score each LLM prompt by mean over the batch of its max class logit (Eq. 4 style).

    image_features : [B, D]   (L2-normalised)
    prompt_bank    : [C, N, D] (L2-normalised) frozen LLM text embeddings (Tg)
    returns        : [N] scalar score per prompt
    """
    N = prompt_bank.shape[1]
    scores = []
    for i in range(N):
        logits_i = logit_scale * image_features @ prompt_bank[:, i, :].t()   # [B, C]
        scores.append(logits_i.max(dim=1).values.mean())
    return torch.stack(scores)


def sccm_loss(text_features, ensemble_mean):
    """Eq. 9 - pull learnable prompt embeddings toward the full LLM ensemble mean Pg.
    Both inputs are L2-normalised [C, D]."""
    return F.mse_loss(text_features, ensemble_mean)


def kdsp_loss(student_logits, teacher_logits):
    """Eq. 10 - KL(teacher || student); teacher comes from the selective ensemble Ps.
    Matches the reference: log_target KL, summed, normalised by element count."""
    return F.kl_div(
        F.log_softmax(student_logits, dim=1),
        F.log_softmax(teacher_logits, dim=1),
        reduction="sum",
        log_target=True,
    ) / student_logits.numel()
