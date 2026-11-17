# Part 3 — Multimodal multi-agent system (Nikhil Kanaparthi)

Planner → Vision Understanding → Prompt Engineering → Image Generation → Critique,
sharing one typed `AgentState` dataclass, with at most 2 revisions and a JSON trace per request.
Domain-style generation uses the Part 2 rank-4 LoRA on SD 1.5.

## How to run
```bash
pip install -r ../../requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/p3_agents.ipynb   # from repo root; needs part2_diffusion/lora_rank4
```
Uncomment `demo.launch()` in the last code cell for the Gradio app.

## Folder contents
```text
scenarios.json, red_team_cases.json   15 scenarios + 10 red-team cases (written before running)
part3_metrics.json                    counts printed by the notebook
traces/                               per-request JSON traces
generated_images/                     images produced by the generation agent
config.json, run_manifest.json, metrics_report.csv
raw_logs/, src/agents.py, results.md, failure_analysis.md
```
