"""
BiomedCoOp training entrypoint (Dassl-free).

  python3 train.py --config config.yaml --root /path/to/dataset --modality MRI \
                   --prompts prompts/btmri.json --kshot 16

Steps: load BiomedCLIP -> build LLM prompt bank Tg -> build BiomedCoOp ->
freeze all but ctx -> SGD + cosine -> few-shot train -> evaluate.
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from open_clip import create_model_from_pretrained, get_tokenizer  # open_clip (BiomedCLIP fork)
from model import BiomedCoOp, freeze_all_but_context, BIOMEDCLIP_HF
from data import FewShotImages, list_classes
from prompt_ensemble import load_cache, offline_templates


def build_prompt_bank(classnames, prompts_by_class, biomedclip, tokenizer, device):
    """Encode N LLM descriptions per class -> normalised bank Tg [C, N, D]."""
    n = min(len(prompts_by_class[c]) for c in classnames)
    per_prompt = []
    with torch.no_grad():
        for i in range(n):
            toks = torch.cat([tokenizer(prompts_by_class[c][i]) for c in classnames]).to(device)
            feats = biomedclip.encode_text(toks)               # [C, D]
            per_prompt.append(torch.nn.functional.normalize(feats, dim=-1).unsqueeze(1))
    return torch.cat(per_prompt, dim=1)                        # [C, N, D]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--root", required=True, help="ImageFolder root: root/<class>/*.png")
    ap.add_argument("--modality", required=True, help="e.g. MRI, Ultrasound, X-Ray")
    ap.add_argument("--prompts", default=None, help="JSON cache of GPT-4 prompts per class")
    ap.add_argument("--kshot", type=int, default=16)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    biomedclip, preprocess = create_model_from_pretrained(BIOMEDCLIP_HF)
    biomedclip = biomedclip.float().eval().to(device)
    tokenizer = get_tokenizer(BIOMEDCLIP_HF)

    classnames = list_classes(args.root)
    prompts = (load_cache(args.prompts) if args.prompts
               else offline_templates(classnames, args.modality, cfg["n_prompts"]))
    bank = build_prompt_bank(classnames, prompts, biomedclip, tokenizer, device)

    model = BiomedCoOp(
        classnames, biomedclip, tokenizer, bank,
        n_ctx=cfg["n_ctx"], ctx_init=cfg["ctx_init"],
        sccm_lambda=cfg["sccm_lambda"], kdsp_lambda=cfg["kdsp_lambda"], tau=cfg["tau"],
    ).to(device)
    params = freeze_all_but_context(model)
    print("trainable tensors:", [n for n, p in model.named_parameters() if p.requires_grad])

    opt = torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    tr = DataLoader(FewShotImages(args.root, classnames, preprocess, args.kshot, "train"),
                    batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    te = DataLoader(FewShotImages(args.root, classnames, preprocess, args.kshot, "test"),
                    batch_size=100, shuffle=False, num_workers=cfg["num_workers"])

    for epoch in range(cfg["epochs"]):
        model.prompt_learner.train()
        for batch in tr:
            img, label = batch["img"].to(device), batch["label"].to(device)
            logits, l_ce, l_sccm, l_kdsp = model(img, label)
            loss = l_ce + l_sccm + l_kdsp
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        print(f"epoch {epoch+1:3d} | loss {loss.item():.4f} "
              f"(ce {l_ce.item():.3f} sccm {l_sccm.item():.3f} kdsp {l_kdsp.item():.3f})")

    model.prompt_learner.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in te:
            img, label = batch["img"].to(device), batch["label"].to(device)
            pred = model(img).argmax(1)
            correct += (pred == label).sum().item(); total += label.numel()
    print(f"\nTest accuracy: {100*correct/max(total,1):.2f}%  ({total} query images)")


if __name__ == "__main__":
    main()
