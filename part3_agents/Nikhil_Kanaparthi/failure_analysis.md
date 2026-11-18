# Part 3 — Failure analysis

1. Critic too strict: first-attempt accept rate 0; CLIP scores were mostly 28–38 (above the 24 threshold) but the 3B VLM
   judged stylised outputs as not following the request, so every generation was revised twice (latency 5.4 s → 12.6 s)
   without raising task success (9/15 in both conditions).
2. Ambiguity: "Do something cool with it" (S13) was routed to editing instead of asking for clarification.
3. Weak style changes at img2img strength 0.6 (S04–S06 failed the rubric for most raters).
4. Keyword guardrails can over-block harmless images that contain words like "make it" and miss paraphrases;
   real-person detection relies on names, not faces.
5. Full system loads 16.4 GB on the RTX 4090; on a 16 GB card the diffusion pipelines need CPU offloading (not tested).
