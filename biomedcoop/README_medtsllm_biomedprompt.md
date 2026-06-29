# MedTsLLM + BiomedCoOp prompt strategy (class-description prompting)

This keeps **MedTsLLM** as the classification model (patch reprogramming, frozen
LLM, softmax head) and adds only the **prompt strategy** from BiomedCoOp: the
LLM-generated per-class ECG feature descriptions are injected into MedTsLLM's text
prompt, giving the frozen LLM explicit domain knowledge about what separates the
five diagnostic classes. No SCCM/KDSP, no cosine head — just richer prompts.

## What changed
- `models/medtsllm.py` (replace): `__init__` optionally loads your class
  descriptions; a new `build_class_prompt()` formats them; `build_prompt()` inserts
  a "Diagnostic categories and their characteristic ECG features: (1) Normal ECG:
  ...; (2) Myocardial Infarction: ...; ..." block after the dataset description.
- `configs/datasets/ptbxl-medtsllm-biomedprompt.toml` (new): turns it on via the
  `prompting` block.
- Reuses `prompts/ptbxl_prompts.json` and `load_class_prompts` from
  `models/biomedcoop_ts.py` (so that file must be present, but the BiomedCoOp model
  itself is not used here).

## How the prompt strategy is applied
- `class_prompts_per_class` controls how many descriptions per class go into the
  prompt (default 3 — keeps the prompt short). Increase for richer context at the
  cost of sequence length / memory / speed.
- `class_prompts_sample = true` resamples which descriptions are used **each training
  step**, so the model sees varied phrasings of each class (a light augmentation /
  ensemble effect). At eval it deterministically uses the first-k for stable scores.
- `class_codes` order MUST match the dataset labels (NORM=0, MI=1, STTC=2, CD=3, HYP=4).

## Run
```bash
# ensure your 50 prompts/class are in prompts/ptbxl_prompts.json
python3 train.py configs/datasets/ptbxl-medtsllm-biomedprompt.toml
```

## Tuning for accuracy
- Use your full 50 descriptions/class in the JSON; raise `class_prompts_per_class`
  (e.g. 5-8) if the LLM context and memory allow.
- The paper's strongest single signal was patient-specific info — keep `clip = true`
  so demographics are also in the prompt alongside the class descriptions.
- For best classification, switch `llm` to `google/flan-t5-xl` (encoder-decoder), which
  the MedTsLLM paper reports as best for classification.

## Notes
- Injecting class descriptions lengthens the prompt, so each step processes more
  tokens — expect somewhat slower epochs and higher memory than the no-class-prompt
  run. With `load_in_4bit = true` it still fits 40 GB.
- Validated here: the edits compile and the prompt builder produces well-formed,
  label-ordered text (sampled at train, deterministic at eval). Not yet run against
  live PTB-XL + the LLM in this environment.
