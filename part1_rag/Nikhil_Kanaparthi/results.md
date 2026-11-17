# Part 1 Results — Nikhil Kanaparthi

All numbers below are copied from executed notebook outputs (`raw_logs/p1_notebook_outputs.log`).

## Run information
GPU NVIDIA GeForce RTX 5090, Python 3.10.13, torch 2.7.0+cu128, seed 266.

## What ran
| Step | Status | Evidence |
|---|---|---|
| Text extraction + chunking (doc_id, page) | Done — 128 records | cell 4 |
| ChromaDB index with doc_id/page/figure_id/trust_level | Done — 128 records | cell 5 |
| Hybrid retrieval (bge-small dense + BM25, exact figure-label match) | Done | cell 6 |
| Conversation memory keyed by conversation_id | Done (stores history; not yet used for follow-ups) | cell 7 |
| Answer generation with answering rules (conflict / not_found / false_premise / session_override) | **Not implemented** — returns top chunk text, status always `answered` | cell 7 |
| Figure crops + VLM descriptions or CLIP embeddings | **Not done** — figure records are text blocks near captions | cell 4 |
| submission.csv for Questions.csv | Written on GPU machine | cell 9 |
| Golden set (25 questions) | **Template only (TO_FILL)** | cell 8 |

## Results table (C2)
| Measurement | Baseline (dense only) | Final (hybrid) |
|---|---|---|
| Answer correctness /25 | not run | not run |
| Recall@5 / citation accuracy | not run | not run |
| Human groundedness (avg 1–5, agreement ±1) | not run | not run |
| Median sec / question | not run | not run |

## Kaggle
Public score / rank: _record after submitting_.
