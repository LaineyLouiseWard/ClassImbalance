# Plan — the final catch-all audit

**Status: APPROVED 2026-07-26. Not yet run.** Do not execute until the diff/design audit currently in
flight has reported and its findings are fixed.

**Decisions taken at approval:**

- **Orphan deletion approved in principle** (Rule 2, Phase 2): remove things once they are
  *definitely* no longer needed, to avoid confusion later. Phase 2 still proposes the list rather
  than executing it, and anything genuinely uncertain is ARCHIVE rather than DELETE — the point is a
  smaller surface, not a bolder one. Git history is the safety net for anything deleted in error.
- **Audit A (label quality) approved**, to run *after* this catch-all rather than in parallel with it.
- Audits B (figure-code smoke test) and C (reproducibility) stand as proposed: B before stage E,
  C before submission.

---

## 1. Why every audit finds new things, with the evidence

Seven audits have run. Each found real defects. That is usually read as "the code is bad", and it is
not the right reading — the design has survived every one. The correct reading is that **no audit has
ever known its own denominator**, so none could distinguish "we found everything" from "we stopped
looking".

Measured today:

| | count |
|---|---|
| Python files (excluding `_archive/`) | **120** |
| of those, in `scripts/` | 62 |
| `scripts/` files with a `--self-test` | **3 of 62** |
| scripts invoked by `RUNBOOK.sh` | 22 |
| scripts invoked by `build_all_figures.py` | 13 |
| **scripts invoked by nothing at all** | **43 of 62** |
| shell / slurm files | 29 |
| tracked artefacts | 10 |

Four mechanisms follow from that table, and each one has produced a real defect already:

1. **The surface was never enumerated, so every audit sampled it.** Findings were a function of where
   a reviewer happened to look. Two reviewers looking at the same repo found disjoint sets.
2. **43 uninvoked scripts are an unexamined surface that is still importable and still runnable.**
   `boundary_exposure.py` — which read the withdrawn campaign's leaked IoU from a hard-coded path —
   is one of those 43. It was a landmine precisely because nothing called it, so nothing exercised it.
3. **Auditing manufactures the material for the next audit.** Audit 6 fixed 42 findings and
   introduced at least 2. Audit 7 changed 28 files. Fixes are new, unreviewed code, and the treadmill
   is structural, not a failure of care.
4. **A defect class was fixed in the instances someone listed, not in all instances.** The
   mean-latitude block bug was corrected in 2 of 3 implementations because nobody enumerated the
   implementations. The provenance guard had one blind spot copied into 3 places.

**So this audit must differ in kind, not in effort.** It must partition the surface rather than
search it, report coverage rather than findings alone, and carry a stopping rule that can actually be
satisfied.

## 2. The three rules that make it a catch-all

**Rule 1 — enumerate, then check.** Phase 1 produces a machine-readable inventory of everything that
can reach a reported number. Every later phase consumes that inventory and marks each item. The final
report states *X of Y examined*, and any unexamined item is named. A finding is then a property of a
known population, not an anecdote.

**Rule 2 — shrink the surface before checking it.** 43 uninvoked scripts do not all need auditing;
most need deleting. **You cannot ship a defect in a file that does not exist.** Triage each orphan to
KEEP (with the caller named), ARCHIVE, or DELETE. This is the single highest-leverage step and it
serves "robust, not convoluted" directly — expect the audited surface to fall by half before a single
check runs.

**Rule 3 — the audit must not leave unreviewed code behind.** Findings get fixed, and then the fixes
are re-checked against the same inventory in a final pass. Without this the treadmill continues and
audit 9 becomes necessary the moment audit 8 finishes.

## 3. Scope

**In:** every file that can reach a number the paper reports, or that gates the campaign —
`scripts/`, `evaluation/`, `geoseg/`, `train/`, `config/`, `RUNBOOK.sh`, `run_campaign.sh`,
`sonic/campaign/`, the 10 tracked artefacts, and the tracked docs that state a number.

**Out:** `manuscript/` (being rewritten), `_archive/`, `notes/` (working notes, gitignored),
`README.md` / `RUNBOOK.md` / `docs/DESIGN_NOTES.md` (stale by decision), the settled decisions in the
out-of-scope list of `docs/audit/BRIEF_FINAL_AUDIT.md`, and anything requiring trained weights.

## 4. The phases

**Phase 0 — freeze.** *(inline)* Commit everything first. The audit runs against a fixed commit, and every
finding cites that commit. Auditing a dirty tree is how audit 7 ended up reviewing a diff that had
moved underneath it.

**Phase 1 — inventory (no findings).** *(inline)* Build `artifacts/audit_inventory.json`: for every file, its
callers, whether it is in the shipped pipeline, whether it has a self-test, what it reads, what it
writes, and whether any of what it writes has a consumer. This is mechanical and its output is the
denominator every later phase reports against.

**Phase 2 — orphan triage.** *(inline)* Every uninvoked script → KEEP / ARCHIVE / DELETE with a one-line reason.
Deletions are proposed, not executed; you approve the list.

**Phase 3 — producer/consumer closure.** *(inline)* Trace every file the 40 runs will produce and every file any
analysis reads. Flag: outputs nothing consumes, inputs nothing produces, and any writer/reader path
disagreement. **This is the C5/C7 defect class generalised into a mechanical check**, and it is the
one that would have caught the softmax path mismatch without anyone being clever.

**Phase 4 — parallel checks over the enumerated surface** *(agents — four lenses)*, each reporting
coverage against Phase 1:
- *Gates* — every guard in the shipped pipeline, each constructed a known-bad input and observed to
  fail, and a good input observed to pass. Any guard that cannot be made to fail is reported as such.
- *Constants and duplicates* — every hard-coded number and every duplicated implementation. Exhaustive
  over the inventory, not by grep intuition. This is the check that would have found the third block
  function.
- *Soft defaults and silent fallbacks* — every `os.environ.get(..., default)`, every `.get()` with a
  fallback on a load-bearing value, every bare `except`, every path that can write an output from
  zero input rows.
- *Numbers in tracked docs* — every falsifiable claim in `docs/*.md` and `CLAUDE.md`, both directions:
  is the doc true of the code, and does the code do something material the docs do not mention.

**Phase 5 — adversarial verification.** *(agents)* Every finding independently attacked by a reviewer instructed
to refute it and defaulting to refuted. Only survivors are reported.

**Phase 6 — fix, then re-check the fixes.** *(inline)* Fixes applied, then a narrow pass over only the changed
lines against the Phase 1 inventory. This is Rule 3 and it is what ends the treadmill.

## 4a. Execution model — what runs inline and what gets agents

**Agreed 2026-07-26 after measuring the cost of the diff/design run.** The rule is about what the
work actually *is*, not about how important it is:

| do INLINE | give it AGENTS |
|---|---|
| tracing, grepping, enumerating | independent re-derivation of a number |
| running gates and self-tests | adversarial refutation of a finding |
| checking imports and call sites | design judgement needing distinct lenses |
| building the inventory | anything where one viewpoint is a weakness |

The test: **if the answer is determined by the repository rather than by judgement, it does not need
an agent.** Confirming the block bootstrap was fully removed — 28 pipeline scripts and 11 figure
scripts imported, artefacts checked, dead references grepped — took three tool calls inline and would
have taken a dozen agents to do worse, because agents would have re-derived the enumeration each time
and disagreed at the margins.

Applied here: **Phases 0–3 and 6 run inline.** Agents are spent only on Phase 4's four checking
lenses and Phase 5's verification, which is where independence is the point. That is roughly a
quarter of the agent count of the diff/design run for strictly more coverage, because the expensive
part of that run was re-deriving an enumeration that Phase 1 will have already committed to a file.

**The same split applies to audits A, B and C in §7.** Audit A's inventory work (which masks were
inspected, what the records say) is inline; only the judgement of whether the evidence supports the
88% claim needs an independent reader. Audit B is entirely inline — it is a smoke test.

## 5. The stopping rule

The audit is complete when **all five** hold, and the report must state each explicitly:

1. Every item in the Phase 1 inventory has a verdict; unexamined items are named and counted.
2. Every gate in the shipped pipeline has been *observed to fail* on a constructed bad input.
3. Every campaign output has a named consumer; every analysis input has a named producer.
4. Every number in the tracked docs has been checked against the code in both directions.
5. Phase 6's re-check of the fixes returns nothing new.

If any is unmet, the report says so rather than implying completion. **"We stopped looking" and "there
is nothing left" must be distinguishable in the output** — that is the property every previous audit
lacked.

## 6. What this audit will NOT do

- Re-litigate settled decisions (D12, D14, D18, D19, the 5-of-10 phases, the Test B unit — settled in
  `AUDIT_7_FINDINGS.md` §5a and **not reopening**).
- Propose new statistical machinery. The brief is robust, not convoluted; a recommendation that
  lengthens the methods section without changing a conclusion is a bad recommendation.
- Invent thresholds.
- Touch the manuscript.

## 7. Other audits I would suggest — separately, not folded into this one

**A. The label-quality audit — my strongest recommendation, and nobody has done it.** *(mostly
inline; agents only for the judgement)*
The paper's central claim is that a **label-quality ceiling** binds performance. The evidence for the
premise is that expert inspection found ~88% of inspected masks contain labelling errors — a figure
that lives in a note. Every audit so far has checked the *code that measures* label quality; none has
checked the *evidence for the claim itself*: how many masks were inspected, by what protocol, what
counts as an error, and whether 88% is a rate over tiles, over pixels, or over inspected items. A
reviewer will ask, and it is the load-bearing premise of the contribution. It needs no GPU and can run
in parallel with the campaign.

**B. Figure-code smoke test — before stage E, not after.** *(entirely inline)*
Thirteen figures, all built from the withdrawn campaign, none ever run against the current split. If
the figure code breaks or silently mis-plots on the new data, you find out after 800 GPU-hours. It can
be smoke-tested now with synthetic inputs of the right shape. Cheap, and it targets the same
"produces output at exit 0 from wrong input" class as the trimap defect.

**C. Reproducibility audit — post-campaign, before submission.** *(mostly inline)*
Can a stranger with the same data reproduce this? Environment pinning, data provenance, what is on
Zenodo. Not urgent now; genuinely needed before submission.

I would run **A** in parallel with the campaign, **B** before stage E, and **C** after results exist.

## 8. Sequencing

1. Diff/design audit (in flight) reports → fix its findings
2. **Commit** — Phase 0 freeze
3. Review and approve this plan
4. Run the catch-all
5. Fix, re-check (Phase 6)
6. Commit, push, stage to Sonic
7. **Audit A (label quality)** — after the catch-all, before or during the campaign
8. Audit B (figure-code smoke test) before stage E; Audit C (reproducibility) before submission
