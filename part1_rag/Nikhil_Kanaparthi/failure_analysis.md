# Part 1 — Failure analysis and open items

1. The answer is the raw top-ranked chunk truncated to 1200 characters, so it breaks the
   120-word limit and never returns `conflict`, `not_found`, `false_premise` or `session_override`.
2. Appendix C of DOC3 (trust: low) is indexed like any other page; nothing filters it out,
   so injected instructions and false claims can be retrieved as evidence.
3. Figure records are text blocks near a "Figure"/"Table" label, not image crops, so
   near-identical charts (e.g. Figure B-5 vs Figure B-5 (corrected)) are not distinguished visually.
4. Follow-up questions ("And in 2013?") do not read earlier turns of the conversation.
5. Golden set, baseline-vs-final evaluation, human groundedness ratings and timing are not computed.
6. The first notebook cell printed an empty commit hash because the notebook was run outside the git checkout.
