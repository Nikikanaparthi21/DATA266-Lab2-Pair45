# Contributions — Pair_Programming_Team_45

Members: Nikhil Kanaparthi (019096441), Samruddhi Chitnis (018452122)

| Part | Nikhil Kanaparthi | Samruddhi Chitnis |
|---|---|---|
| Part 1 — Multimodal RAG | Implemented the RAG pipeline (ingestion, figure transcription, hybrid retrieval, citations, answering rules, conversation memory) and the Kaggle submission (`part1_rag/Nikhil_Kanaparthi/`) | Golden test suite (25 questions), human evaluation (mean 4.86/5.0), safety & red-teaming analysis (`part1_rag/Samruddhi_Chitnis/`) |
| Part 2 — Diffusion fine-tuning | Configured and ran the LoRA rank-4 and rank-16 experiments, generated the comparison images, ran the evaluation notebook (`part2_diffusion/Nikhil_Kanaparthi/`) | Blind human evaluation (3 raters, 81.3% style / 70.7% text win rate), memorization audit, model card (`part2_diffusion/Samruddhi_Chitnis/`) |
| Part 3 — Multi-agent system | Implemented and ran the planner, vision, prompt, generation and critique agents and the safety evaluation (`part3_agents/Nikhil_Kanaparthi/`) | Multi-agent critique loops, critic agreement validation (88.2%), 10 adversarial defense cases (100% defense), system card (`part3_agents/Samruddhi_Chitnis/`) |
| Report and evaluation | Comparative analysis, LaTeX compilation, figure grids | Full IEEE-format report, architecture diagrams, benchmark tables, safety & system cards (`report/REPORT_SAMRUDDHI.md`) |

Human evaluations: both members rated independently (Part 1 groundedness, Part 2 blind pairs with one outside rater,
Part 3 pass/fail with and without critique).

## Meetings and knowledge transfer

All design discussions and knowledge-transfer sessions were held **in person**; they were not recorded, so no
recording links exist.

| Date | Topic | Format |
|---|---|---|
| 2026-10-20 | Part 1 PDF ingestion and document review | In person (not recorded) |
| 2026-10-24 | Part 1 hybrid retrieval and golden-set planning | In person (not recorded) |
| 2026-10-28 | Part 2 dataset cleaning and LoRA setup | In person (not recorded) |
| 2026-11-02 | Part 2 rank-4 / rank-16 training and image generation | In person (not recorded) |
| 2026-11-06 | Part 3 agent scenarios and safety tests | In person (not recorded) |
| 2026-11-10 | Human evaluation, pass/fail ratings and metrics review | In person (not recorded) |
| 2026-11-15 | Final report, GitHub organization and submission checklist | In person (not recorded) |
