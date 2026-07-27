# Plan — the design interrogation, conducted in chat

**Status: PLAN. Not yet run.** To be conducted interactively after compaction, before staging to
Sonic. Scope is the **paper-facing design and statistics**, not gates, paths or plumbing.

---

## 1. Why this is a conversation and not another review

Four audit workflows ran on 2026-07-26/27: 87 agents, ~9.4M tokens. They were worth it — they found a
third block implementation that made the per-class Test B support table wrong for four of five
classes, and an ensemble-only accumulator that left the headline boundary rates with no uncertainty
at all.

But look at where the errors in *my own* reasoning were caught:

| error | caught by |
|---|---|
| compared the buffer to 950 m (ireland2's range) instead of 750 m (the inland site's) | **you, pushing back** |
| described the inland site as a rectangle when 20% of its bounding box is empty | **you, pushing back** |
| carried the withdrawn `+1.66` into a "correction" | adversarial verifier |
| missed a broken import because my test set `PYTHONPATH` and the real invocation does not | diff review |

**Zero were caught by one-shot review.** A reviewer that reads, reports and stops cannot catch a wrong
premise, because it never has to defend the premise against a second question. What caught them was
*"is that actually true?"* asked twice.

So this audit has a different mechanic: **every element is interrogated until it bottoms out**, either
in a measurement someone can reproduce, or in an explicit admission that it is a judgement call. No
element is settled because it sounded reasonable the first time.

## 2. Scope

**In — anything paper-facing that is new or changed since the old manuscript**, i.e. everything a
referee reads in §2 and §3 and everything that produces a number in a table or figure.

**Out** — gates, provenance checks, path handling, file layout, the campaign launcher, anything that
cannot reach a reported number. Eight audits have covered those and the last one found only
directory-path defects. Also out: `manuscript/` prose (not being edited), and the settled decisions in
`docs/DO_NOT_ADD.md`.

## 3. The elements, in the order I would take them

Ordered so that later ones depend on earlier ones being settled. Each is new or materially changed
since the old manuscript.

**The data and the split**
1. Why 50% stride chipping exists at all, and what it forces on everything downstream
2. The single-axis three-strip cut — why one axis, why that axis, why not a block grid
3. The buffer widths (256 m, realised 768 m) against the measured inland range (750 m)
4. Dropping 413 tiles, 21.2% of the site — what it buys, and whether it was worth it
5. Two upland sites held out whole; two estimands never pooled
6. Training on 1,072 tiles when the epoch budget was set for 1,706

**Admissibility**
7. Class support by grid cells rather than class share — and what a "cell" is now that it is not a
   block
8. The floor of 5/8, and why it never applied to Test B

**The factorial**
9. What main effect A actually estimates, given the 2.00x step confound and the double selection
10. What the interaction term means when it is collinear with (extra pass x sampler)
11. Whether "modest gains" is a conclusion this design can carry

**The boundary claim**
12. The 8 m band — an a-priori number that cannot be cited to anyone
13. The two statistics currently both called rho (8 m vs 1.5 m partitions) — **open**
14. Ground-truth-only banding and its known asymmetry
15. Excluding boundary-free tiles
16. The second arm: what it establishes and what it cannot — **open**
17. The trimap exclusion curve as primary evidence

**The statistics**
18. Uncertainty over training runs, not over ground — and why no spatial interval anywhere
19. Paired per-seed contrasts; what pairing buys and what it assumes
20. What Test B can support, and which verb is defensible

## 4. The protocol, per element

I answer four questions. You push on any of them. I verify live — running code, reading the actual
file, re-deriving the number — rather than asserting. We do not move on until it is settled or
explicitly parked.

1. **What is it**, in one sentence a non-specialist follows.
2. **What did we do instead of what?** Name the alternative that was rejected and why. If no
   alternative was ever considered, that is the finding.
3. **What backs it?** A measurement someone can reproduce, or an explicit "this is a judgement call".
   Anything that bottoms out in "it seemed reasonable" is not settled.
4. **What would falsify it, and what does it cost?** If nothing could falsify it, say so — that is
   itself reportable.

**Rules for me, because these are the ways I got it wrong today:**

- Read the file before answering. Never from memory, never from a summary.
- Quote the actual line, then reason from the quote.
- When I give a number, say where it came from and re-derive it if it is load-bearing.
- If I compare two numbers, check they are the same *kind* of number and from the same *site* — that
  is exactly the 950-vs-750 error.
- Say "I don't know" rather than reconstructing a plausible answer.

**Rules for you, if useful:** the two things that worked best today were *"is that actually true?"* and
*"did you consider X?"* — asked about something I had just stated confidently.

## 5. Where agents are used, and where they are not

**Not for the dialogue.** The back-and-forth happens here, in chat, because that is the mechanic that
caught things.

**Agents only for independent re-derivation**, spawned mid-conversation when a specific number is
load-bearing and I should not be the only one computing it — the same use as the geometry
re-measurement that found the Test A over-count. One narrow question, one agent, result folded back
into the conversation.

## 6. What "settled" means, and when we stop

An element is settled when all four questions are answered, the answers are consistent with the code
as it actually is, and **you** are satisfied — not when I run out of things to say.

The audit is complete when every one of the twenty elements is either settled or explicitly parked
with a reason. **Parked is a legitimate outcome** and must be visible in the output: a design property
you have decided to state rather than fix is finished business, and re-auditing it later is the waste
this whole exercise exists to stop.

Output: each element gets one line in `docs/CORRECTIONS.md` (if it needs changing) or
`docs/METHODS_STATED_LIMITATIONS.md` (if it needs stating). Nothing else. No new documents.

## 7. Two things already known to be open before we start

- **Element 13** — two statistics both called rho. `boundary_rate_ratio.py` uses an 8 m near band;
  `boundary_trimap_iou.py:243` uses `BND_MAX_M, INT_MIN_M = 1.5, 8.0`, so the 1.5–8 m annulus is in
  neither set and 71.75% of the 8 m band sits inside it. Fix is a rename, not machinery.
- **Element 16** — the second arm is stated in `CLAUDE.md` and `METHODOLOGICAL_CHOICES.md` §E5 without
  the two qualifiers the registered version carried: the scale ("in relative terms") and the falsifier
  ("if both fall proportionally, the concentration is a property of model quality"). It also has no
  across-cell implementation, so as it stands it would be judged by eye.
