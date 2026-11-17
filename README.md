# DATA266 Lab 2 — Pair 45

Multimodal RAG, diffusion fine-tuning, and a multimodal multi-agent system.
Each part has its own folder with one sub-folder per team member, using the same
layout as Lab 1 (config, metrics, logs, source, results and failure analysis).

## Repository Structure

```text
DATA266-Lab2-Pair45/
├── notebooks/
│   ├── p1_rag.ipynb            # Part 1 (executed, outputs saved)
│   ├── p2_finetune.ipynb       # Part 2 training + generation
│   ├── p2_evaluate.ipynb       # Part 2 evaluation (KID, CLIP, LPIPS, grids, memorization, human eval, HF upload)
│   └── p3_agents.ipynb         # Part 3
├── part1_rag/
│   └── Nikhil_Kanaparthi/
├── part2_diffusion/
│   └── Nikhil_Kanaparthi/
├── part3_agents/
│   └── Nikhil_Kanaparthi/
├── reproducibility/manifests/
├── requirements.txt
├── gpu_specs.txt
├── links.txt                   # GitHub tag commits, Hugging Face, Kaggle links
├── CONTRIBUTIONS.md
├── AI_USAGE.md
└── verify_package.py
```

Each member folder contains:

```text
README.md            how to run that member's pipeline
config.json          settings used
metrics_report.csv   results table (baseline vs final)
metrics_report.json
run_manifest.json    environment + config snapshot
results.md           write-up of the results
failure_analysis.md  known failures and open items
raw_logs/            notebook outputs captured verbatim
src/                 code extracted from the notebook
outputs/ ...         figures, generations, traces
```

## Part Summary

| Part | Description | Notebook | Main folder |
|---|---|---|---|
| Part 1 | Multimodal RAG over DOC1–DOC3 (Kaggle) | `notebooks/p1_rag.ipynb` | `part1_rag/Nikhil_Kanaparthi/` |
| Part 2 | SD 1.5 LoRA fine-tuning on Naruto BLIP captions (rank 4 vs 16) | `notebooks/p2_finetune.ipynb` | `part2_diffusion/Nikhil_Kanaparthi/` |
| Part 3 | Five-agent image understanding/generation system + Gradio | `notebooks/p3_agents.ipynb` | `part3_agents/Nikhil_Kanaparthi/` |

## Setup

```bash
git clone https://github.com/Nikikanaparthi21/DATA266-Lab2-Pair45.git
cd DATA266-Lab2-Pair45
pip install -r requirements.txt
python verify_package.py        # checks data/part1 files and notebooks are present
```

Every notebook's first cell prints the git commit, Python/torch versions, GPU and seed (266).

## Part 1 — Multimodal RAG

1. Create `data/part1/` (git-ignored) and copy `Document.pdf`, `Document2_Supplement_A_Outturns_2011-2013.pdf`,
   `Document3_Briefing_B_Labour_Prices_Policy.pdf`, `kb_manifest.json`, `Questions.csv`
   and `validate_submission.py` into `data/part1/`.
2. Run `notebooks/p1_rag.ipynb` top to bottom (Restart & Run All). Set `QUESTIONS_PATH`
   to use a different questions file.
3. Validate: `python data/part1/validate_submission.py part1_rag/outputs/submission.csv data/part1/Questions.csv`

Stack: PyMuPDF, `BAAI/bge-small-en-v1.5`, `rank_bm25`, ChromaDB.

## Part 2 — Diffusion Fine-Tuning

Run `notebooks/p2_finetune.ipynb`. It downloads `lambdalabs/naruto-blip-captions`,
cleans and splits it (seed 266, 90/10), clones diffusers v0.31.0, trains LoRA rank 4 and
rank 16 (1000 steps each, same data/LR/batch), and generates 20 prompts × 2 seeds for
base, rank 4 and rank 16. Adapters and generated images go to Hugging Face (see `links.txt`).

## Part 3 — Multi-Agent System

Run `notebooks/p3_agents.ipynb` after Part 2 (it loads the rank-4 adapter). It runs the
15 scenarios and 10 red-team cases, writes a JSON trace per request, and defines the Gradio
app (`demo.launch()` in the last cell).

## Hosted APIs

None. All models run locally: `runwayml/stable-diffusion-v1-5`, `BAAI/bge-small-en-v1.5`.

## Model weights and data

Weights (`*.safetensors`, `*.bin`, `*.pt`, `*.pkl`), the cloned diffusers repo and the
Naruto training images are git-ignored. Weights and generated images are published on
Hugging Face (links in `links.txt`).

## Tags

Graders use `p1-v1.0`, `p2-v1.0`, `p3-v1.0`.

## Team

Pair 45 — Nikhil Kanaparthi, Samruddhi Chitnis. See `CONTRIBUTIONS.md` and `AI_USAGE.md`.
