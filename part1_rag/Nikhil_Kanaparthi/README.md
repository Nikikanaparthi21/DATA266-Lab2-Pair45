# Part 1 — Multimodal RAG (Nikhil Kanaparthi)

Hybrid retrieval over the three read-only economics PDFs (DOC1–DOC3) with
conversation-local memory and a Kaggle-format submission writer.

## How to run

```bash
pip install -r ../../requirements.txt
# put Document.pdf, Document2_*.pdf, Document3_*.pdf, kb_manifest.json,
# Questions.csv, validate_submission.py in data/part1/
jupyter nbconvert --to notebook --execute --inplace notebooks/p1_rag.ipynb   # from repo root
python data/part1/validate_submission.py part1_rag/outputs/submission.csv data/part1/Questions.csv
```

## Folder contents

```text
config.json            retrieval / chunking settings
metrics_report.csv     results table (baseline vs final) — see status below
metrics_report.json    machine-readable version
run_manifest.json      environment + config used
results.md             results write-up
failure_analysis.md    what is missing / known weaknesses
raw_logs/              notebook outputs captured verbatim
src/rag_pipeline.py    code extracted from the notebook
outputs/figures/       rendered pages containing figures/tables
```
