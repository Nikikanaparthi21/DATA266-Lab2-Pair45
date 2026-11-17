# Part 2 Results — Nikhil Kanaparthi

Numbers are copied from executed outputs (`raw_logs/p2_notebook_outputs.log`).

## Data (2.1)
- Naruto BLIP Captions: 1221 images loaded, 0 corrupted and 0 exact duplicates removed (SHA-256 of PNG bytes).
- Split with seed 266: **1098 train / 123 test**, resized to 512×512.
- Limitations/biases (to expand in report): single anime style and narrow character set; BLIP captions are short and generic.

## Training runs (2.2)
| Run | LoRA rank | LR | Steps | Eff. batch | Train time (s) | GPU-hours | Peak GPU mem |
|---|---|---|---|---|---|---|---|
| Run 1 | 4 | 1e-4 | 1000 | 4 | 2099.3 | 0.583 | not captured* |
| Run 2 | 16 | 1e-4 | 1000 | 4 | 2029.9 | 0.564 | not captured* |

\*Logged as 0.0 GB because training ran in a subprocess (see failure_analysis.md).
Trainable parameter counts were not printed.

## Generation (2.3)
20 fixed prompts (10 in-domain from test captions, 5 out-of-domain, 5 compositional),
seeds 0 and 1, 30 steps, guidance 7.5 → 40 images per model for base, rank4 and rank16.
Prompts: `prompts.json`; per-image model/prompt/seed: `generations/*/metadata.csv`.
Smoke test before the final runs: rank 4, 20 steps, 136.2 s (`lora_rank4_smoke/run_metrics.json`).
Safety checker replaced images with black frames: rank4 = 2, rank16 = 4, base = 0.

## Evaluation (2.4)
| Measurement | Base | Rank 4 | Rank 16 |
|---|---|---|---|
| KID (in-domain vs test) | not run | not run | not run |
| CLIP score | not run | not run | not run |
| LPIPS diversity | not run | not run | not run |
| Human win rate (final vs base, 25 pairs, ≥3 raters) | — | not run | not run |
| GPU-hours per run | — | 0.583 | 0.564 |

Memorization check (10 in-domain images vs nearest training image by CLIP): template only.
Part 3 uses the **rank-4** adapter.
