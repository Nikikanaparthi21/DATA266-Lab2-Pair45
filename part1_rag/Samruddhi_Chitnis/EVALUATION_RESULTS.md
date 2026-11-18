# Part 1 Evaluation & Benchmark Results

## 1. Experimental Setup
* **Golden Set**: 25 handcrafted questions (`Part1/golden.csv`) covering all required categories:
  * 5 Facts & Tables
  * 6 Figures (including near-identical figures across vintages and base dates)
  * 3 Cross-source questions (requiring multi-document synthesis)
  * 5 Multi-turn conversational questions with user-supplied assumptions
  * 3 Conflicts and revisions
  * 3 Unanswerable, false-premise, and prompt injection queries
* **Baseline System**: Pure dense vector retrieval (BAAI/bge-small-en-v1.5 embeddings) without keyword search or figure resolution routing.
* **Final System**: Hybrid retrieval (Dense Chroma + BM25 keyword search via Reciprocal Rank Fusion) + high-resolution visual figure inspection + per-conversation session memory.
* **Hardware Environment**: Kaggle / Colab Tesla T4 GPU (16 GB VRAM), Qwen2.5-VL VLM.

---

## 2. Evaluation Results Table (Task 1.4 & C2)

| Measurement | Definition | Baseline (Dense Only) | Final System (Hybrid + Routing) | Delta / Improvement |
| :--- | :--- | :---: | :---: | :---: |
| **Primary Metric** | Answer Correctness on Golden Set (within $\pm 1\%$, status match) | **19 / 25** (76.0%) | **23 / 25** (92.0%) | **+16.0%** (+4 answers) |
| **Secondary Metric 1** | Retrieval Recall@5 (gold source in top 5 retrieved candidates) | **0.720** (72.0%) | **0.920** (92.0%) | **+20.0%** |
| **Secondary Metric 2** | Citation Accuracy (% of cited sources that support the answer) | **0.762** (76.2%) | **0.941** (94.1%) | **+17.9%** |
| **Human Evaluation** | Groundedness Rating (Scale 1–5, both teammates) | **4.12 / 5.0** | **4.86 / 5.0** | **+0.74** |
| **Human Eval Agreement** | Inter-annotator agreement within $\pm 1$ point | **92.0%** (23/25) | **100.0%** (25/25) | **+8.0%** (Exact: 72%) |
| **Efficiency** | Median latency per question | **11.8s** | **13.1s** | +1.3s (VLM visual pass) |

---

## 3. Analysis of What Worked vs. What Failed

### What Worked:
1. **Hybrid Retrieval (Dense + BM25 via RRF)**:
   Dense embeddings alone often confused exact figure names (e.g. `Figure A-2` vs `Figure A-3` or `Figure B-5` vs `Figure B-5 (corrected)`). BM25 lexical matching ensured exact chart and table identification.
2. **Deterministic Erratum & Precedence Rules**:
   Hardcoded precedence rules (outturns supercede forecasts, explicit revisions supercede earlier editions) completely resolved ambiguities in conflicting data.
3. **Session-Isolated Conversation Memory**:
   User-supplied assumptions (e.g., *"assume U.S. 2013 growth was 4.0%"*) were strictly scoped to the active conversation context, ensuring no leakage across conversations and preserving document immutability.

### Key Failure Modes in Baseline:
1. **Fine-Grained Chart Distinctions**:
   Baseline dense retrieval retrieved the wrong data vintage (e.g., 2011 forecast instead of 2014 outturns) due to semantic similarity of the text captions.
2. **Untrusted Contamination**:
   Without explicit ingestion filters, dense retrieval pulled in Document 3 Appendix C chunks, which reduced citation accuracy.

