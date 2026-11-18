---
license: creativeml-openrail-m
base_model: runwayml/stable-diffusion-v1-5
tags: [stable-diffusion, lora, text-to-image, diffusers]
datasets: [lambdalabs/naruto-blip-captions]
---
# Naruto-style LoRA (rank 4) for Stable Diffusion 1.5 — DATA 266 Lab 2, Team 45

**Use:** LoRA adapter that steers Stable Diffusion 1.5 toward the Naruto anime style. Course project, **non-commercial
research and education only**.

```python
import torch
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
pipe.load_lora_weights("NikhilKanaparthi/data266-lab2-team45-naruto-lora-rank4")
image = pipe("a girl with pink hair and green eyes", num_inference_steps=30, guidance_scale=7.5).images[0]
```

## Training
- Data: `lambdalabs/naruto-blip-captions`, 1221 images → 1098 train / 123 test (seed 266), 512×512, BLIP captions.
- diffusers v0.31.0 `train_text_to_image_lora.py`; rank 4 (797,184 trainable parameters); LR 1e-4; 1000 steps;
  batch 1 × gradient accumulation 4; fp16; gradient checkpointing; seed 266; 0.583 GPU-hours (RTX 5090).

## Evaluation (40 images per model: 20 prompts × 2 seeds, 30 steps, guidance 7.5)
| | Base SD 1.5 | This adapter (rank 4) | Rank 16 |
|---|---|---|---|
| KID vs test images ↓ | 0.172 | **0.036** | 0.051 |
| CLIP score ↑ | **35.11** | 32.49 | 30.48 |
| LPIPS diversity | 0.648 | 0.694 | 0.751 |
| Human win rate vs base (style / prompt match, 25 blind pairs, 3 raters) | — | **0.84 / 0.64** | — |

## Limitations and risks
- Lower prompt adherence than the base model, especially for detailed in-domain prompts.
- The safety checker blacked out 1 of 40 images (0 for base): the style can trigger the NSFW classifier.
- Trained on copyrighted anime artwork: outputs may imitate a recognisable commercial style and characters;
  do not use commercially or to pass off work as official. Nearest-training-image CLIP similarity ≤ 0.79 on 10 checks
  (no near-copies found), but memorization of training images cannot be ruled out.
- Inherits the biases and license (CreativeML OpenRAIL-M) of Stable Diffusion 1.5.

Generated images: https://huggingface.co/datasets/NikhilKanaparthi/data266-lab2-team45-generations
