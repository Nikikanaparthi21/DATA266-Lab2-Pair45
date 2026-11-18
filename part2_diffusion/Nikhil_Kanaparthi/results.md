# Part 2 Results — Nikhil Kanaparthi

All numbers come from executed notebooks: training in `notebooks/p2_finetune.ipynb` (RTX 5090),
restore + regeneration in `notebooks/p2_restore_generate.ipynb` and evaluation in
`notebooks/p2_evaluate.ipynb` (RTX 4090, same adapters, prompts, seeds and sampler). Output files are in
`part2_diffusion/evaluation/`.

## Data (2.1)
- Naruto BLIP Captions: 1221 images loaded, 0 corrupted and 0 exact duplicates removed (SHA-256 of PNG bytes).
- Split with seed 266: **1098 train / 123 test**, resized to 512×512. The restore notebook rebuilt the identical split
  (the 10 in-domain prompts equal the first 10 test captions).
- Limitations/biases: one anime style and a narrow set of characters; BLIP captions are short and generic
  ("a man with black hair and green eyes"), so prompts and captions carry little detail; the images are copyrighted
  fan/anime art, so outputs are for non-commercial research only.

## Training runs (2.2)
| Run | LoRA rank | Trainable params | LR | Steps | Batch | Train time (s) | GPU-hours |
|---|---|---|---|---|---|---|---|
| Run 1 (final) | 4 | 797,184 | 1e-4 | 1000 | 1 × 4 grad-accum | 2099.3 | 0.583 |
| Run 2 | 16 | 3,188,736 | 1e-4 | 1000 | 1 × 4 grad-accum | 2029.9 | 0.564 |

Same data, seed, resolution (512), sampler and step budget; only the rank changes. Peak GPU memory was logged as
0.0 GB because training ran in a subprocess (see failure_analysis.md).

## Generation (2.3)
20 fixed prompts (`prompts.json`: 10 in-domain from test captions, 5 out-of-domain, 5 compositional) × seeds 0 and 1,
30 steps, guidance 7.5 → 40 images per model for base, rank 4 and rank 16 (120 images). Per-image metadata:
`generations/*/metadata.csv`. Dataset: https://huggingface.co/datasets/NikhilKanaparthi/data266-lab2-team45-generations

## Results table (C2)
| Measurement | Base SD 1.5 | Rank 4 (final) | Rank 16 |
|---|---|---|---|
| **KID ↓** (20 in-domain images vs 123 test images, subset 20) | 0.172 ± 0.013 | **0.036 ± 0.007** | 0.051 ± 0.007 |
| **CLIP score ↑** (all 40) | **35.11** | 32.49 | 30.48 |
| CLIP — in-domain / out-of-domain / compositional | 35.00 / 34.18 / 36.26 | 30.86 / 32.88 / 35.37 | 28.48 / 31.41 / 33.55 |
| LPIPS diversity (seed 0 vs seed 1, mean of 20 prompts) | 0.648 | 0.694 | 0.751 |
| Images blacked out by the safety checker (of 40) | 0 | 1 | 4 |
| GPU-hours (training) | — | 0.583 | 0.564 |
| **Human win rate vs base, style** (25 blind pairs, 3 raters) | — | **0.84** (0.92 / 0.72 / 0.88) | — |
| **Human win rate vs base, prompt match** | — | **0.64** (0.64 / 0.60 / 0.68) | — |

Raters: Nikhil Kanaparthi, Samruddhi Chitnis (members) and Abhijith Barre (outside the team).
`evaluation/human_eval_results.csv`; blind key `evaluation/human_eval/key_DO_NOT_SHOW_RATERS.csv`.

## Interpretation
- Rank 4 moves generations about 5× closer to the Naruto test distribution than the base model (KID 0.172 → 0.036),
  and beats rank 16 on KID, CLIP score and safety-checker blackouts, so it is the final model and the one used in Part 3.
- The cost is prompt adherence: CLIP score drops most on in-domain prompts (35.0 → 30.9) because the adapter pulls
  every image toward the anime style and the generic BLIP captions it was trained on. Humans still preferred rank 4
  for prompt match 64% of the time, so CLIP (which favours photo-like images) partly penalises the style change itself.
- Rank 16 has 4× the trainable parameters, no better quality, more blackouts and higher seed-to-seed LPIPS
  (partly from black images); extra capacity did not help on 1098 images and 1000 steps.

## Memorization (C4)
Nearest training image by CLIP image similarity for the 10 in-domain rank-4 images (seed 0): max 0.794,
min 0.384, all below the ~0.9 level of near-copies. Top-3 pairs: `evaluation/memorization_top3.png`;
table: `evaluation/memorization_check.csv`.
