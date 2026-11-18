---
license: creativeml-openrail-m
base_model: stable-diffusion-v1-5/stable-diffusion-v1-5
datasets: [lambda/naruto-blip-captions]
tags: [stable-diffusion, lora, text-to-image, diffusers, data266]
---
# DATA 266 Lab 2 Part 2 - run1_lora_r4

**Base model:** stable-diffusion-v1-5/stable-diffusion-v1-5 (512x512). **Method:** LoRA rank 4 (alpha 4) on attn q/k/v/out.

**Data:** lambda/naruto-blip-captions; 1099 training / 122 test images after removing 0 corrupted and
0 duplicate images; 90/10 split with seed 266; center-crop + resize to 512.

**Training:** 1500 steps, learning rate 0.0001, batch 4 x 1 accum = 4, optimizer AdamW,
fp16 autocast, fp32 LoRA weights, 797,184 trainable parameters, peak GPU memory 2.87 GB,
32.9 min on Tesla T4 (0.548 GPU-hours).

**Results** (20 prompts x seeds 0/1, DPM-Solver 30 steps, CFG 7.5):
KID (in-domain vs test) 0.0533 +/- 0.0246 (base: 0.1194);
CLIP score 30.4 (base: 30.7); LPIPS diversity 0.725
(base: 0.6978).

**Limitations:** small dataset of one art style; BLIP captions are short and sometimes wrong; out-of-domain prompts
are pulled toward the training style; compositional prompts (counts, left/right, colours per object) are often wrong.

**License / use:** the training images show copyrighted characters - this adapter is for **non-commercial research
and education only**. Outputs pass through the Stable Diffusion safety checker.

**Usage:**
```python
pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
pipe.load_lora_weights("<this repo>")
```
