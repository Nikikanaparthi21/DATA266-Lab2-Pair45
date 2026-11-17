# Running Lab 2 on an HPC-lab GPU machine (Docker)

Follows the ADS lab Docker guide: work only inside a folder you create on the Desktop, no USB drives,
save everything to GitHub / Hugging Face before logging out (sessions are deleted).

## 1. Start the container
1. Log in (`.\<username>`), open **Docker Desktop** → Accept → Continue without signing in.
2. File Explorer → `C:\DockerImages` → double-click the **pytorch** image.
3. Create `C:\Users\<you>\Desktop\nikhil` (your name only).
4. **Command Prompt** (not PowerShell):
   ```
   docker run --gpus all -p 8888:8888 -v "C:\Users\<you>\Desktop\nikhil":/app pytorch:latest
   ```
   Open the printed `http://127.0.0.1:8888/?token=...` link.

## 2. Get the code (Jupyter → New → Terminal)
```bash
cd /app
git clone https://github.com/Nikikanaparthi21/DATA266-Lab2-Pair45.git
cd DATA266-Lab2-Pair45
nvidia-smi
```
Clone (don't download the zip) so cell 1 of each notebook prints the commit hash.
For a private repo, use a GitHub personal access token as the password.

## 3. Part 2
1. Run `notebooks/p2_restore_generate.ipynb` (Restart & Run All): rebuilds the seed-266 split,
   loads the adapters, regenerates 40 images per model. Set `HF_USER` first, or copy the two
   `pytorch_lora_weights.safetensors` files into `part2_diffusion/lora_rank4/` and `lora_rank16/`.
2. Run `notebooks/p2_evaluate.ipynb` (Restart & Run All).

## 4. Part 1 data
Download from Canvas (Files → Labs → Lab 2 → DATA266_Lab2_Part1) or the Kaggle Data tab into
`data/part1/`: the three PDFs, `kb_manifest.json`, `Questions.csv`, `sample_submission.csv`,
`validate_submission.py` (git-ignored).

## 5. Before you leave
```bash
git add -A && git commit -m "Lab run results" && git push
```
Upload weights/images to Hugging Face, then delete the Desktop folder and lock the machine (Win+L).
