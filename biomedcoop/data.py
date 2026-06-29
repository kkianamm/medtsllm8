"""
Minimal few-shot biomedical image dataset for BiomedCoOp.

Expects an ImageFolder-style root:
    root/<class_name>/*.png|jpg|...
K-shot sampling draws K images per class for the support (train) set; the rest
form the query (test) set. For base-to-novel, pass the class subset you want.
Uses the BiomedCLIP preprocess returned by open_clip.
"""

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_classes(root):
    return sorted(d.name for d in Path(root).iterdir() if d.is_dir())


class FewShotImages(Dataset):
    def __init__(self, root, classnames, preprocess, k_shot=None,
                 split="train", seed=0):
        self.preprocess = preprocess
        self.classnames = classnames
        self.cls_to_idx = {c: i for i, c in enumerate(classnames)}
        rng = random.Random(seed)
        self.items = []
        for c in classnames:
            files = sorted(p for p in (Path(root) / c).iterdir()
                           if p.suffix.lower() in IMG_EXT)
            rng.shuffle(files)
            if k_shot is None:
                chosen = files
            elif split == "train":
                chosen = files[:k_shot]
            else:  # test = everything not used for the support set
                chosen = files[k_shot:]
            self.items += [(f, self.cls_to_idx[c]) for f in chosen]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = self.preprocess(Image.open(path).convert("RGB"))
        return {"img": img, "label": torch.tensor(label)}
