# Part 2 — Failure analysis and open items

1. Peak GPU memory reads 0.0 GB for both runs: `torch.cuda.max_memory_allocated()` was called in the
   notebook process, but training ran in a child process. Re-measure with `nvidia-smi` polling or
   inside the training script.
2. The safety checker blacked out 2 rank-4 and 4 rank-16 generations and none for the
   base model — a candidate failure mode (fine-tuned outputs trip the NSFW classifier more often).
3. KID, CLIP score, LPIPS diversity, the 25-pair blind human evaluation and the memorization check are
   still templates (`human_audit.csv`, `memorization_check.csv` contain TO_FILL).
4. Side-by-side grids for 8 prompts and the Hugging Face image dataset/model card are not yet produced.
5. Training used an RTX 5090 (32 GB); the lab requires everything to run on one 16 GB GPU — confirm
   the same settings fit a T4 (they should: batch 1, fp16, gradient checkpointing).
6. Only rank-16 training output is visible in the notebook (the rank-4 call was replaced by the rank-16 call
   in the same cell); rank-4 metrics come from the `run1` dict printed in cell 13.
