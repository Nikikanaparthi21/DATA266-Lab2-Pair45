# Part 2 — Failure analysis

1. **Safety-checker blackouts after fine-tuning.** The base model produced no black images; rank 4 produced 1 and
   rank 16 produced 4 (prompts 0, 2, 4, 5 — all in-domain). Fine-tuning on anime art pushes some outputs toward
   what the NSFW classifier flags, and the higher-capacity adapter does so more often. These black images also
   lower CLIP score and raise LPIPS for rank 16.
2. **Loss of prompt adherence on in-domain prompts.** CLIP score falls from 35.0 (base) to 30.9 (rank 4) and 28.5
   (rank 16) on in-domain prompts: the adapter overrides details such as clothing colour or setting with the dominant
   training style. Out-of-domain and compositional prompts drop less (rank 4: 32.9 and 35.4).
3. **Peak GPU memory not captured.** `torch.cuda.max_memory_allocated()` ran in the notebook process while training ran
   in an `accelerate` subprocess, so both runs logged 0.0 GB. Batch size 1 with gradient accumulation 4, fp16 and
   gradient checkpointing keeps SD 1.5 LoRA training within 16 GB.
4. **Hardware differs from the 16 GB requirement.** Training used an RTX 5090 (32 GB) and evaluation an RTX 4090 (24 GB);
   the settings above are 16 GB-compatible but were not re-run on a 16 GB card.
5. **Regeneration on a second machine.** The evaluation images were regenerated from the same adapters, prompts, seeds
   and sampler on a new GPU session (torch 2.1.2 vs 2.7.0), so individual images can differ slightly from the first run
   (e.g. rank 4 blacked out 2 images in the first run and 1 in the regenerated set).
6. **KID on a small sample.** KID uses 20 in-domain generations per model (subset size 20, 100 subsets), so the
   standard deviation (~0.007) should be read with the mean.
