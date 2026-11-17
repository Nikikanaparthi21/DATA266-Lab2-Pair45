"""Part 2 - SD 1.5 LoRA fine-tuning on Naruto BLIP captions (Nikhil Kanaparthi).

Extracted from the executed notebook so the code can be reviewed
without opening Jupyter. The notebook remains the source of truth."""

# ---- notebook cell 1 ----
import sys, subprocess, random, time, json, hashlib, io
from pathlib import Path
import numpy as np
import pandas as pd
import torch

SEED = 266
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
try:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
except Exception:
    commit = "not-in-git-yet"
print({"commit": commit, "python": sys.version.split()[0],
       "torch": torch.__version__,
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
       "seed": SEED})

# ---- notebook cell 2 ----
PROJECT_ROOT = Path.cwd()
OUT = PROJECT_ROOT / "part2_diffusion"
DATA = OUT / "data"
CLEAN = DATA / "clean"
OUT.mkdir(exist_ok=True); CLEAN.mkdir(parents=True, exist_ok=True)
# Start with True, run the smoke test, then switch to False for the two final runs.
SMOKE_TEST = False
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
MAX_TRAIN_STEPS = 1000

# ---- notebook cell 3 ----
# Optional Colab-only installation.
# %pip install -q datasets diffusers transformers accelerate peft safetensors
from datasets import load_dataset
from PIL import Image

# ---- notebook cell 4 ----
dataset = load_dataset("lambdalabs/naruto-blip-captions", split="train")
print(dataset)
print(dataset.column_names)

# ---- notebook cell 5 ----
seen = set(); kept = []; removed_corrupt = 0; removed_duplicate = 0
for row in dataset:
    try:
        image = row["image"].convert("RGB")
        blob = io.BytesIO(); image.save(blob, format="PNG")
        digest = hashlib.sha256(blob.getvalue()).hexdigest()
        if digest in seen:
            removed_duplicate += 1
            continue
        seen.add(digest)
        caption = str(row.get("text", row.get("caption", ""))).strip()
        kept.append((image, caption))
    except Exception:
        removed_corrupt += 1
print({"kept": len(kept), "removed_corrupt": removed_corrupt,
       "removed_duplicate": removed_duplicate})

# ---- notebook cell 6 ----
rng = np.random.default_rng(SEED)
order = rng.permutation(len(kept))
split = int(0.9 * len(order))
train_ids, test_ids = order[:split], order[split:]
train_dir, test_dir = CLEAN / "train", CLEAN / "test"
train_dir.mkdir(exist_ok=True); test_dir.mkdir(exist_ok=True)

def save_split(indices, folder):
    rows = []
    for j, idx in enumerate(indices):
        image, caption = kept[int(idx)]
        name = f"image_{j:05d}.png"
        image.resize((512, 512)).save(folder / name)
        rows.append({"file_name": name, "text": caption})
    (folder / "metadata.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows))
    return rows

train_rows = save_split(train_ids, train_dir)
test_rows = save_split(test_ids, test_dir)
print({"train": len(train_rows), "test": len(test_rows)})

# ---- notebook cell 7 ----
# These prompts are fixed before tuning.
prompts = ([row["text"] for row in test_rows[:10]] +
           ["a realistic city skyline at sunset", "a bowl of fruit on a wooden table",
            "a small robot in a library", "a snowy mountain landscape",
            "a watercolor ocean scene",
            "two characters, one in red and one in blue",
            "a character holding a lantern beside a river",
            "three characters standing in a forest",
            "a character riding a bicycle under cherry blossoms",
            "a character and a dragon in a village square"])
assert len(prompts) == 20
(OUT / "prompts.json").write_text(json.dumps(
    [{"prompt_id": i, "prompt": p} for i, p in enumerate(prompts)], indent=2))
print("Saved 20 fixed prompts.")

# ---- notebook cell 8 ----
import shutil
import subprocess
from pathlib import Path

DIFFUSERS_DIR = OUT / "diffusers"

# Remove the incompatible latest clone
if DIFFUSERS_DIR.exists():
    shutil.rmtree(DIFFUSERS_DIR)

# Clone the version matching the installed package
subprocess.run([
    "git", "clone",
    "--depth", "1",
    "--branch", "v0.31.0",
    "https://github.com/huggingface/diffusers.git",
    str(DIFFUSERS_DIR)
], check=True)

TRAIN_SCRIPT = (
    DIFFUSERS_DIR
    / "examples"
    / "text_to_image"
    / "train_text_to_image_lora.py"
)

print("Using:", TRAIN_SCRIPT)

# ---- notebook cell 10 ----
DIFFUSERS_DIR = OUT / "diffusers"
if not DIFFUSERS_DIR.exists():
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/huggingface/diffusers.git",
                    str(DIFFUSERS_DIR)], check=True)
TRAIN_SCRIPT = DIFFUSERS_DIR / "examples" / "text_to_image" / "train_text_to_image_lora.py"

def train_lora(rank, output_name):
    output_dir = OUT / output_name
    command = [
        sys.executable, str(TRAIN_SCRIPT),
        "--pretrained_model_name_or_path", BASE_MODEL,
        "--train_data_dir", str(train_dir),
        "--resolution", "512",
        "--train_batch_size", "1",
        "--gradient_accumulation_steps", "4",
        "--gradient_checkpointing",
        "--mixed_precision", "fp16",
        "--learning_rate", "1e-4",
        "--max_train_steps", str(MAX_TRAIN_STEPS),
        "--checkpointing_steps", "250",
        "--rank", str(rank),
        "--output_dir", str(output_dir),
        "--seed", str(SEED),
        "--report_to", "tensorboard"
    ]
    print("Running:", " ".join(command))
    start = time.time()
    result = subprocess.run(command, text=True)
    metrics = {"rank": rank, "steps": MAX_TRAIN_STEPS,
               "seconds": time.time() - start,
               "peak_gpu_memory_gb": (torch.cuda.max_memory_allocated() / 1e9
                                      if torch.cuda.is_available() else None)}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_metrics.json").write_text(json.dumps(metrics, indent=2))
    if result.returncode:
        raise RuntimeError("LoRA training failed")
    return metrics

# Smoke test:
# train_lora(4, "lora_rank4_smoke")
# Final runs:
run2 = train_lora(16, "lora_rank16")
run2

# ---- notebook cell 11 ----
# Generate two seeds for each of 20 prompts for base, rank4, and rank16.
# Uncomment only after the adapters finish.
from diffusers import StableDiffusionPipeline

def generate_images(adapter_dir=None, name="base"):
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16).to("cuda")
    if adapter_dir:
        pipe.load_lora_weights(str(adapter_dir))
    output_dir = OUT / "generations" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in json.loads((OUT / "prompts.json").read_text()):
        for seed in (0, 1):
            generator = torch.Generator(device="cuda").manual_seed(seed)
            image = pipe(item["prompt"], num_inference_steps=30,
                         guidance_scale=7.5, generator=generator).images[0]
            path = output_dir / f"prompt_{item['prompt_id']:02d}_seed_{seed}.png"
            image.save(path)
            rows.append({"model": name, "prompt_id": item["prompt_id"],
                         "prompt": item["prompt"], "seed": seed, "path": str(path)})
    pd.DataFrame(rows).to_csv(output_dir / "metadata.csv", index=False)

# generate_images(None, "base")
# generate_images(OUT / "lora_rank4", "rank4")
# generate_images(OUT / "lora_rank16", "rank16")

# ---- notebook cell 12 ----
audit_path = OUT / "human_audit.csv"
if not audit_path.exists():
    pd.DataFrame({
        "pair_id": range(25),
        "prompt_id": [i % 20 for i in range(25)],
        "left_model": ["final"] * 25, "right_model": ["base"] * 25,
        "rater": ["TO_FILL"] * 25, "style_choice": ["TO_FILL"] * 25,
        "prompt_match_choice": ["TO_FILL"] * 25,
    }).to_csv(audit_path, index=False)
memory_path = OUT / "memorization_check.csv"
if not memory_path.exists():
    pd.DataFrame({"test_image": ["TO_FILL"] * 10,
                  "closest_train_image": ["TO_FILL"] * 10,
                  "clip_similarity": [np.nan] * 10}).to_csv(memory_path, index=False)
print("Complete human_audit.csv and memorization_check.csv after generation.")

# ---- notebook cell 13 ----
print(run1)
print(run2)

print((OUT / "lora_rank4").exists())
print((OUT / "lora_rank16").exists())

# ---- notebook cell 14 ----
generate_images(None, "base")

# ---- notebook cell 15 ----
generate_images(OUT / "lora_rank4", "rank4")

# ---- notebook cell 16 ----
generate_images(OUT / "lora_rank16", "rank16")

# ---- notebook cell 17 ----
from pathlib import Path

for name in ["base", "rank4", "rank16"]:
    folder = OUT / "generations" / name
    images = list(folder.glob("*.png"))
    print(name, len(images), "images")
