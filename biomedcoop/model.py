"""
BiomedCoOp model — CoOp-style learnable context on the BiomedCLIP backbone,
with SCCM (semantic consistency) and KDSP (selective knowledge distillation).

Faithful to HealthX-Lab/BiomedCoOp (trainers/BiomedCoOp/biomedcoop_biomedclip.py),
rewritten to be Dassl-free and importable as a plain nn.Module.

Backbone note: BiomedCLIP's text tower is PubMedBERT (a HF BERT), not CLIP's
GPT-style text transformer. Context is therefore inserted into BERT word
embeddings, position 0 is the prefix ([CLS]) and the suffix starts after the
context tokens. `encode_text(embeds, inputs_embeds=True, y=tokenized)` is the
open_clip BiomedCLIP entry point that accepts precomputed embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import mad_outlier_mask, prompt_scores, sccm_loss, kdsp_loss

BIOMEDCLIP_HF = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
CTX_DIM = 768  # PubMedBERT hidden size


class TextEncoder(nn.Module):
    def __init__(self, biomedclip_model):
        super().__init__()
        self.model = biomedclip_model
        self.dtype = biomedclip_model.text.transformer.dtype

    def forward(self, prompt_embeds, tokenized_prompts):
        # inputs_embeds=True -> text tower consumes precomputed embeddings;
        # tokenized_prompts (y) supplies attention mask / pooling positions.
        return self.model.encode_text(prompt_embeds, True, tokenized_prompts)


class PromptLearner(nn.Module):
    def __init__(self, classnames, biomedclip_model, tokenizer, prompt_bank,
                 n_ctx=4, ctx_init="a photo of a", class_token_position="end"):
        super().__init__()
        dtype = biomedclip_model.text.transformer.dtype
        word_emb = biomedclip_model.text.transformer.embeddings.word_embeddings
        n_cls = len(classnames)
        self.tokenizer = tokenizer

        # --- context init (paper: embedding of "a photo of a", which is 4 tokens) ---
        if ctx_init and n_ctx == 4:
            prompt = tokenizer(ctx_init.replace("_", " "))
            with torch.no_grad():
                emb = word_emb(prompt).type(dtype)
            ctx_vectors = emb[0, 1:1 + n_ctx, :].clone()
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, CTX_DIM, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        self.ctx = nn.Parameter(ctx_vectors)  # <-- the ONLY trainable tensor

        classnames = [c.replace("_", " ") for c in classnames]
        self.name_lens = [len(tokenizer(c)) for c in classnames]
        prompts = [f"{prompt_prefix} {c}." for c in classnames]
        tokenized = torch.cat([tokenizer(p) for p in prompts])  # [C, L]

        with torch.no_grad():
            embedding = word_emb(tokenized).type(dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :])           # [CLS]
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])   # class + [SEP]

        # frozen LLM prompt bank Tg : [C, N, D] (already L2-normalised upstream)
        self.register_buffer("prompt_bank", prompt_bank)

        self.n_cls, self.n_ctx = n_cls, n_ctx
        self.tokenized_prompts = tokenized
        self.class_token_position = class_token_position

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix, suffix = self.token_prefix, self.token_suffix
        if self.class_token_position == "end":
            return torch.cat([prefix, ctx, suffix], dim=1)
        raise NotImplementedError("only class_token_position='end' is implemented")


class BiomedCoOp(nn.Module):
    def __init__(self, classnames, biomedclip_model, tokenizer, prompt_bank,
                 n_ctx=4, ctx_init="a photo of a",
                 sccm_lambda=0.25, kdsp_lambda=3.0, tau=1.25):
        super().__init__()
        self.prompt_learner = PromptLearner(
            classnames, biomedclip_model, tokenizer, prompt_bank, n_ctx, ctx_init)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = biomedclip_model.visual
        self.text_encoder = TextEncoder(biomedclip_model)
        self.logit_scale = biomedclip_model.logit_scale
        self.dtype = biomedclip_model.text.transformer.dtype
        self.sccm_lambda, self.kdsp_lambda, self.tau = sccm_lambda, kdsp_lambda, tau

    def forward(self, image, label=None):
        logit_scale = self.logit_scale.exp()
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        logits = logit_scale * image_features @ text_features.t()

        if not self.prompt_learner.training or label is None:
            return logits

        bank = self.prompt_learner.prompt_bank                      # Tg [C, N, D]
        Pg = F.normalize(bank.mean(dim=1), dim=-1)                  # full ensemble (SCCM target)

        with torch.no_grad():                                      # MAD prune -> selective Ps
            scores = prompt_scores(image_features, bank, logit_scale)
            mask = mad_outlier_mask(scores, self.tau)
            if mask.sum() < 2:                                      # safety: never drop everything
                mask = torch.ones_like(mask)
            Ps = F.normalize(bank[:, mask, :].mean(dim=1), dim=-1)
            teacher_logits = logit_scale * image_features @ Ps.t()  # frozen zero-shot teacher

        loss_ce = F.cross_entropy(logits, label)
        loss_sccm = self.sccm_lambda * sccm_loss(text_features, Pg)
        loss_kdsp = self.kdsp_lambda * kdsp_loss(logits, teacher_logits)
        return logits, loss_ce, loss_sccm, loss_kdsp


def freeze_all_but_context(model: nn.Module):
    """Train only prompt_learner.ctx; everything else frozen (CoOp regime)."""
    for name, p in model.named_parameters():
        p.requires_grad_(name == "prompt_learner.ctx")
    return [p for p in model.parameters() if p.requires_grad]
