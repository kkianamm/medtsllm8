"""
LLM prompt ensembling for BiomedCoOp (Sec. 3.2).

The paper queries GPT-4 once, offline, per dataset with:

    Q = "Give {N} textual descriptions of visual discriminative features
         for distinct medical cases of {CLASS} found in {MODALITY}."

and caches N descriptions per class. At train time we only LOAD that cache
(a JSON of {classname: [desc, ...]}) and encode it with the frozen text tower
to build the prompt bank Tg in [C, N, D].

This module does NOT call any paid API by default. `build_query` reproduces the
exact prompt; `generate_with_llm` is an optional hook you can wire to your own
client. For pipeline testing without any network, `offline_templates` fabricates
plausible-looking descriptions so the rest of the code runs end to end.
"""

import json
from pathlib import Path

QUERY_TEMPLATE = (
    "Give {n} textual descriptions of visual discriminative features "
    "for distinct medical cases of {cls} found in {modality}."
)


def build_query(classname: str, modality: str, n: int) -> str:
    return QUERY_TEMPLATE.format(n=n, cls=classname, modality=modality)


def load_cache(path) -> dict:
    """Load {classname: [description, ...]} produced offline by GPT-4."""
    data = json.loads(Path(path).read_text())
    assert all(isinstance(v, list) and v for v in data.values()), \
        "each class must map to a non-empty list of descriptions"
    return data


def save_cache(prompts: dict, path):
    Path(path).write_text(json.dumps(prompts, indent=2))


def generate_with_llm(classnames, modality, n, client_fn):
    """Optional: build the cache by calling your own LLM.

    client_fn(prompt_str) -> str must return the model's raw text; we split it
    into <= n non-empty lines per class. Wire this to whatever provider you use.
    """
    out = {}
    for c in classnames:
        text = client_fn(build_query(c, modality, n))
        lines = [ln.strip(" -\t") for ln in text.splitlines() if ln.strip()]
        out[c] = lines[:n] if lines else offline_templates([c], modality, n)[c]
    return out


def offline_templates(classnames, modality, n) -> dict:
    """No-network fallback so the pipeline is runnable without GPT-4 access.
    These are generic stand-ins, NOT a substitute for real LLM prompts."""
    bases = [
        "In {m}, a case of {c} typically shows characteristic morphology and texture.",
        "{c} on {m} presents with distinct margins and signal/intensity patterns.",
        "Discriminative features of {c} in {m} include shape, density, and contrast cues.",
        "A {m} image of {c} reveals region-specific changes versus surrounding tissue.",
        "Typical {c} findings on {m} differ in boundary sharpness and internal structure.",
    ]
    return {
        c: [bases[i % len(bases)].format(c=c, m=modality) for i in range(n)]
        for c in classnames
    }
