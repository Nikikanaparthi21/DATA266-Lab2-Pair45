"""Fast pre-run verification for the DATA266 Lab 2 package."""
from pathlib import Path
import json, csv, hashlib

ROOT = Path(__file__).resolve().parent
required = [
    "data/part1/Document.pdf",
    "data/part1/Document2_Supplement_A_Outturns_2011-2013.pdf",
    "data/part1/Document3_Briefing_B_Labour_Prices_Policy.pdf",
    "data/part1/kb_manifest.json",
    "data/part1/Questions.csv",
    "data/part1/validate_submission.py",
    "notebooks/p1_rag.ipynb",
    "notebooks/p2_finetune.ipynb",
    "notebooks/p3_agents.ipynb",
]
missing = [p for p in required if not (ROOT / p).is_file()]
if missing:
    raise SystemExit("Missing required files:\n" + "\n".join(missing))

manifest = json.loads((ROOT / "data/part1/kb_manifest.json").read_text())
assert manifest["policy"]["base_is_read_only"] is True
assert len(manifest["documents"]) == 3
with (ROOT / "data/part1/Questions.csv").open(newline="") as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 40 and [r["id"] for r in rows] == [str(i) for i in range(1, 41)]
for p in required:
    digest = hashlib.sha256((ROOT / p).read_bytes()).hexdigest()[:12]
    print(f"OK  {p}  sha256[:12]={digest}")
print("Package verification passed. Next: run p1, p2 smoke test, then p3.")
