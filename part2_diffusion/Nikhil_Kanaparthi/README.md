# Part 2 — Diffusion LoRA fine-tuning (Nikhil Kanaparthi)

Stable Diffusion 1.5 fine-tuned on Naruto BLIP Captions with two LoRA runs that share
data, steps (1000), effective batch size (4), learning rate (1e-4) and seed (266);
only the LoRA rank changes (4 vs 16).

## How to run
```bash
pip install -r ../../requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/p2_finetune.ipynb   # from repo root
```
The notebook clones diffusers v0.31.0 into `part2_diffusion/diffusers/` (git-ignored)
and writes adapters to `part2_diffusion/lora_rank4/` and `lora_rank16/` (git-ignored, push to Hugging Face).

## Folder contents
```text
config.json              training + sampler settings
metrics_report.csv/json  per-run table (params, LR, steps, batch, time, memory)
lora_rank4/, lora_rank16/run_metrics.json
prompts.json             the 20 fixed prompts (written before training ended)
MODEL_CARD.md            model card (upload as README.md of the rank-4 HF repo)
generations/{base,rank4,rank16}/  metadata.csv (model, prompt, seed, path for all 40 images) + sample images; full set goes to HF
lora_rank4_smoke/        20-step smoke-test run metrics
evaluation/              human_audit.csv and memorization_check.csv worksheets (to fill)
raw_logs/                notebook outputs incl. accelerate training log
src/finetune_lora.py     code extracted from the notebook
results.md, failure_analysis.md
```
