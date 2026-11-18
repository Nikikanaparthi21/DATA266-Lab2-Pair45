# Part 2 Evaluation & Human Evaluation Results

## 1. System Overview & Hyperparameters
* **Base Model**: `stable-diffusion-v1-5/stable-diffusion-v1-5` (512×512)
* **Dataset**: `lambdalabs/naruto-blip-captions` (90% train / 10% test, split seed: 266)
* **Run 1**: LoRA rank $r=4$ ($\alpha=4$), 1,500 steps, batch size 1, learning rate $1 \times 10^{-4}$
* **Run 2**: LoRA rank $r=32$ ($\alpha=32$), 1,500 steps, batch size 1, learning rate $1 \times 10^{-4}$
* **Selected Final Model**: `run1_lora_r4` (lower KID, higher diversity, less overfitting)

---

## 2. Evaluation Results Table (Task 2.4 & C2)

| Model | KID (in-domain) $\downarrow$ | CLIP Score (all) $\uparrow$ | LPIPS Diversity $\uparrow$ | Human Win Rate (Domain Style) $\uparrow$ | Human Win Rate (Prompt Match) $\uparrow$ | Train GPU Hours | Peak GPU Memory | Latency / Image | NSFW Flags |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base SD 1.5** | 0.1194 $\pm$ 0.027 | 30.70 | 0.6978 | — (reference) | — (reference) | 0.00 | — | 5.81s | 0 |
| **Run 1 (LoRA r=4)** *(Final)* | **0.0533 $\pm$ 0.025** | **30.40** | **0.7250** | **81.3%** | **70.7%** | **0.548 h** | **2.87 GB** | **6.04s** | 2 |
| **Run 2 (LoRA r=32)** | 0.1452 $\pm$ 0.072 | 29.02 | 0.6221 | — | — | 0.548 h | 2.94 GB | 6.04s | 9 |

---

## 3. Human Evaluation Breakdown (Task 2.4)
* **Design**: 25 blinded, randomized image pairs (`run1_lora_r4` vs. `base`) evaluated by 3 raters (`member1`, `member2`, and `outside_rater`).
* **Overall Win Rates for `run1_lora_r4`**:
  * **Domain Style Win Rate**: **81.3%**
  * **Prompt Match Win Rate**: **70.7%**

### Per-Rater Breakdown:
| Rater | Domain Style Win Rate | Prompt Match Win Rate |
| :--- | :---: | :---: |
| **member1** | 84.0% | 68.0% |
| **member2** | 88.0% | 64.0% |
| **outside_rater** | 72.0% | 80.0% |

---

## 4. Key Failure Modes & Model Choice
1. **Model Choice**: `run1_lora_r4` strongly outperforms `run2_lora_r32`. LoRA $r=32$ suffered from severe overfitting on the small dataset, leading to degraded KID (0.1452 vs 0.0533), lower diversity (LPIPS 0.6221 vs 0.7250), and higher NSFW false positives (9 vs 2).
2. **Failure Mode 1 (Style Bleeding / Out-of-Domain Leakage)**: On non-anime prompts (e.g., realistic landscapes), `run1_lora_r4` occasionally injects stylized anime outlines and textures.
3. **Failure Mode 2 (Compositional Binding)**: On prompts with multiple colored attributes (e.g., *"two characters, one in red and one in blue"*), both models occasionally bleed colors across character boundaries.

