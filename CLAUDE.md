# Working agreement for agents in this repo

You run this experiment autonomously from here to the final report. Do not stop between steps
for human review. The human reads the results on GitHub.

## Loop for each remaining step
1. `git pull`. Read `TASK.md`, then `REVIEW.md` (latest section wins where they conflict).
2. Pilot on ~12 sequences, sanity-check, then do the full run.
3. **Independent review.** Spawn a subagent with fresh context (it must not see your reasoning).
   Give it only: TASK.md, REVIEW.md, the step's code, and the step's results/STEPn.md + CSVs. Ask it to
   audit against the checklist below and report concrete problems with file/line references.
   Fix every real problem it finds, rerun what is affected, and record what changed in
   `results/STEPn.md` under "Review fixes". If you disagree with a finding, say why there.
4. Commit and push (`git pull --rebase` first). Then continue to the next step without waiting.

## Reviewer checklist
- Leakage: anything tuned or selected (heads, λ, mixing weights) uses the selection split only;
  every reported number comes from the eval split.
- Lead time: each predictor only uses information that would be available at the claimed time.
  In particular, no predictor for token t at layer l may use token t's own layer-l (or later) routing,
  and "previous layer" predictors may use nothing from layer l's attention.
- Baselines are implemented as specified and are not weakened (e.g. recency λ actually tuned).
- Suspicious numbers: recall near 1.0, identical values across layers or predictors, a predictor
  beating an oracle, CIs that are implausibly tight or wide.
- Code matches the written description in STEPn.md.

## When to stop and ask instead (push first, then end with a short note in results/BLOCKED.md)
- A result that contradicts the hypothesis setup itself (e.g. hooks or routing turn out wrong).
- Anything that needs >4 GPU-hours, a different model, or a TASK.md scope change.
- The reviewer finds a problem you cannot fix.

## Done
`results/REPORT.md` written per TASK.md, reviewed by a fresh subagent, pushed. Final commit message
starts with `Final report:`.

## Git
Commit as the existing author identity on the VM; email must be the GitHub noreply address or pushes
are rejected. Never commit weights, datasets, caches or raw dumps (see .gitignore). Keep files < 50 MB.
