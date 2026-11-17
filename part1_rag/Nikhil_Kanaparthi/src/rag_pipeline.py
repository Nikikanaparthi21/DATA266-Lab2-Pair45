"""Part 1 - Multimodal RAG pipeline (Nikhil Kanaparthi).

Extracted from the executed notebook so the code can be reviewed
without opening Jupyter. The notebook remains the source of truth."""

# ---- notebook cell 1 ----
import sys, subprocess, random, time, json, re
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
# In Colab, change PROJECT_ROOT to your Drive/repository path.
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data" / "part1"
OUT = PROJECT_ROOT / "part1_rag" / "outputs"
FIGURES = OUT / "figures"
CHROMA_DIR = OUT / "chroma"
OUT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
print("Part 1 data directory:", DATA_DIR)

# ---- notebook cell 3 ----
# Optional Colab-only installation. Use the approved Docker environment on the
# lab machine instead of installing unauthorized packages there.
# %pip install -q PyMuPDF sentence-transformers rank-bm25 chromadb
import fitz
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ---- notebook cell 4 ----
DOCS = {
    "DOC1": DATA_DIR / "Document.pdf",
    "DOC2": DATA_DIR / "Document2_Supplement_A_Outturns_2011-2013.pdf",
    "DOC3": DATA_DIR / "Document3_Briefing_B_Labour_Prices_Policy.pdf",
}
manifest_path = DATA_DIR / "kb_manifest.json"
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

def trust(doc_id):
    # kb_manifest.json stores documents as a list; keep the base KB read-only.
    docs = manifest.get("documents", [])
    item = next((d for d in docs if d.get("doc_id") == doc_id), {}) if isinstance(docs, list) else docs.get(doc_id, {})
    return item.get("trust_level", item.get("trust", "unknown")) if isinstance(item, dict) else "unknown"

def chunks(text, size=850, overlap=120):
    words = text.split()
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        piece = " ".join(words[start:start + size]).strip()
        if piece:
            yield piece
        if start + size >= len(words):
            break

records = []
for doc_id, path in DOCS.items():
    if not path.exists():
        print("Missing:", path)
        continue
    pdf = fitz.open(path)
    for page_number, page in enumerate(pdf, start=1):
        text = page.get_text("text")
        for i, piece in enumerate(chunks(text)):
            records.append({"record_id": f"{doc_id}:p{page_number}:t{i}",
                            "doc_id": doc_id, "page": page_number,
                            "figure_id": "NONE", "trust_level": trust(doc_id),
                            "record_type": "text", "text": piece})
        for block_idx, block in enumerate(page.get_text("blocks")):
            block_text = block[4].strip()
            match = re.search(r"\b(Figure|Table)\s+[A-Z]?[- ]?\d+(?:[- ]\d+)?[A-Za-z]?", block_text, re.I)
            if match:
                label = re.sub(r"\s+", " ", match.group(0))
                # Include the block index: a page can contain repeated labels
                # or several extracted text blocks for one figure/table.
                records.append({"record_id": f"{doc_id}:p{page_number}:b{block_idx}",
                                "doc_id": doc_id, "page": page_number,
                                "figure_id": label, "trust_level": trust(doc_id),
                                "record_type": "figure_or_table", "text": block_text})
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        pix.save(FIGURES / f"{doc_id}_page_{page_number:03d}.png")
records_df = pd.DataFrame(records)
records_df.to_json(OUT / "ingested_records.jsonl", orient="records", lines=True)
print("Indexed records:", len(records_df))

# ---- notebook cell 5 ----
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

embedding_fn = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5")
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_or_create_collection("lab2_knowledge",
                                             embedding_function=embedding_fn)
if collection.count():
    # Rebuild deterministically if the notebook is rerun.
    client.delete_collection("lab2_knowledge")
    collection = client.get_or_create_collection("lab2_knowledge",
                                                 embedding_function=embedding_fn)
collection.add(
    ids=records_df.record_id.tolist(),
    documents=records_df.text.tolist(),
    metadatas=records_df[["doc_id", "page", "figure_id", "trust_level", "record_type"]].to_dict("records"),
)
bm25 = BM25Okapi([str(x).lower().split() for x in records_df.text])
print("Chroma records:", collection.count())

# ---- notebook cell 6 ----
def retrieve(question, k=5):
    exact = re.findall(r"(?:Figure|Table)\s+[A-Z]?[- ]?\d+[A-Za-z]?", question, flags=re.I)
    dense = collection.query(query_texts=[question], n_results=max(k, 10))
    dense_ids = dense["ids"][0]
    dense_rows = records_df.set_index("record_id").loc[dense_ids].reset_index()
    scores = bm25.get_scores(question.lower().split())
    bm25_rows = records_df.iloc[np.argsort(scores)[::-1][:max(k, 10)]]
    candidates = pd.concat([dense_rows, bm25_rows]).drop_duplicates("record_id")
    if exact:
        exact_rows = records_df[records_df.figure_id.str.lower().isin([x.lower() for x in exact])]
        candidates = pd.concat([exact_rows, candidates]).drop_duplicates("record_id")
    return candidates.head(k).to_dict("records")

print(retrieve("What does the document report about inflation?", 3))

# ---- notebook cell 7 ----
conversation_memory = {}

def answer_question(question, conversation_id="demo"):
    hits = retrieve(question, k=5)
    if not hits:
        result = {"answer": "The documents do not support an answer.",
                  "status": "not_found", "citations": ["NONE"],
                  "figure_references": ["NONE"]}
    else:
        result = {
            "answer": " ".join(hits[0]["text"].split())[:1200],
            "status": "answered",
            "citations": [f"{h['doc_id']}:p{h['page']}" for h in hits[:2]],
            "figure_references": [h["figure_id"] for h in hits if h["figure_id"] != "NONE"] or ["NONE"],
        }
    conversation_memory.setdefault(conversation_id, []).append(
        {"question": question, "result": result})
    return result

print(json.dumps(answer_question("What is the reported value?"), indent=2))

# ---- notebook cell 8 ----
golden_path = OUT / "golden_set.csv"
if not golden_path.exists():
    categories = (["fact_table"] * 5 + ["figure"] * 6 + ["two_sources"] * 3 +
                  ["multi_turn_override"] * 5 + ["conflict_revision"] * 3 + ["safety"] * 3)
    pd.DataFrame({
        "question_id": [f"G{i:02d}" for i in range(1, 26)],
        "category": categories,
        "question": ["TO_FILL"] * 25,
        "gold_answer": ["TO_FILL"] * 25,
        "gold_status": ["TO_FILL"] * 25,
        "gold_citations": ["TO_FILL"] * 25,
        "gold_figures": ["NONE"] * 25,
    }).to_csv(golden_path, index=False)
print("Replace every TO_FILL value using original questions written from the PDFs.")

# ---- notebook cell 9 ----
questions_path = DATA_DIR / "Questions.csv"
if questions_path.exists():
    q = pd.read_csv(questions_path, dtype=str, keep_default_na=False)
    rows = []
    for conv_id, turns in q.groupby("conversation_id", sort=False):
        conversation_memory.pop(conv_id, None)
        for _, row in turns.sort_values("turn", key=lambda s: s.astype(int)).iterrows():
            result = answer_question(row["question"], conv_id)
            rows.append({"id": row["id"], "conversation_id": conv_id,
                         "question_id": row["question_id"],
                         "answer": " ".join(result["answer"].split()),
                         "figure_references": ",".join(result["figure_references"]),
                         "citations": ",".join(result["citations"]),
                         "status": result["status"]})
    pd.DataFrame(rows).set_index("id").loc[q.id].reset_index().to_csv(OUT / "submission.csv", index=False)
    print("Wrote", OUT / "submission.csv")
