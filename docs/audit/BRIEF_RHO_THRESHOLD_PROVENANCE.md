# SUPERSEDED 2026-07-26 — do not run this brief

The threshold decision was settled directly: there is no pre-registered threshold. See D18 in `docs/audit/DECISIONS_REBUILD_2026-07.md`. Retained for the argument on both sides and for the measured interval evidence at the end.

---

# Brief — should this paper pre-register a threshold at all, and if so, where does it come from?

Paste below the rule into a **fresh chat**. This is a decision brief, not an implementation task. It
must be settled before the campaign runs, because after that any change becomes post-hoc.

**You are deciding. The previous chat took a position and withdrew the pre-registration; that decision
is explicitly reopened here and you may reverse it. Its reasoning is given so you can attack it, not so
you can ratify it.**

---

<role>
You are a remote-sensing methodologist deciding how a paper should evidence its central claim. You care
about what a reviewer will accept, what can be written honestly in a methods section, and whether the
apparatus proposed is proportionate to the claim. You are willing to say "the simpler version is
better" and equally willing to say "the author has talked themselves out of something they need".
</role>

<the_paper>
`label-quality-ceiling`, for MDPI *Remote Sensing*, special issue on Data Curation for AI.

**The claim.** On a fixed FT-UNetFormer, two off-the-shelf data-curation interventions (cross-dataset
transfer from OpenEarthMap, and a class-balanced sampler) give only modest gains, and the residual
error is concentrated at class boundaries — i.e. it is limited by where the label boundaries were
drawn, not by model capacity or class imbalance.

**Why this is delicate.** The first campaign was withdrawn. The tiles are chipped on a 50% stride, and
the original split assigned them at random, so ~93% of each held-out tile's ground was also in
training, with byte-identical imagery and labels. Reported accuracy was ~98% mIoU. The evaluation has
been rebuilt on a spatially blocked, buffered split; nothing has been trained on it yet, so no
corrected result exists.

**What the ORIGINAL manuscript already reports for the boundary claim** (verified in `manuscript/main.tex`):
- a trimap exclusion curve — exclude a band of width b around every ground-truth boundary, recompute
  macro foreground IoU, widen b. Reported as 90.1% rising to 98.0% at a 4 m exclusion.
- boundary-band versus deep-interior error rates, per class, with the deep interior beyond 8 m.
- the share of foreground error falling within 8 m of a boundary: 92% (val) / 96% (test).
- the stratification uses ground-truth labels only, never the model's predictions, so it is not
  self-referential.

**What is new in the rebuild:** the blocked split; a second held-out estimand (two upland sites held out
whole, which the original mixed into its random split); per-class support reported as pixels, tiles and
independent 950 m blocks; bootstrap intervals resampling spatial blocks rather than tiles.
</the_paper>

<the_question>
A pre-registration was written during the rebuild, dated before any training. It registered:

    rho = (foreground error rate within 8 m of a GT boundary) / (rate beyond 8 m)
    threshold: rho >= 4.0 on both test sets, judged on the LOWER bound of a block bootstrap CI
               dead below 2.0; weak in between

It has since been withdrawn by the previous chat. **Your job is to decide whether that withdrawal was
right**, and to settle what the paper does instead.

Three sub-questions, in order:

1. **Should this paper pre-register a threshold at all?**
2. **If yes: which statistic, which threshold, and where does the threshold's justification come from?**
3. **If no: what exactly is reported instead, and how is the claim kept falsifiable rather than
   interpreted into existence?**
</the_question>

<the_case_for_withdrawing>
The previous chat's reasoning, recorded as D16 in `docs/audit/DECISIONS_REBUILD_2026-07.md`. Attack it.

- Pre-registration is **not standard** in remote sensing or computer vision. It is standard in clinical
  trials and psychology. It was adopted here as credibility armour after the leak, not because the
  field expects it — so it is optional, and optional things must earn their cost.
- **The cost compounded.** rho needed a threshold; the threshold needed a provenance; the only
  calibration data came from the withdrawn campaign. Each layer added a number that could be argued
  with, sitting on top of a curve that already showed the effect directly.
- **The statistic was re-specified twice in one day**, both times because it turned out to be
  landscape-dependent:
  - v0, error share >= 65%: has a mechanical floor equal to the share of foreground *area* within 8 m —
    37.8% inland versus 26.5% upland — so one number asks a different question of each test set.
  - v1, that share divided by the area share: this is exactly `lift`, whose ceiling is 1/area_share, so
    a common threshold sits at different points in each attainable range. It also rises as a model
    improves, because a better model removes interior error first — so it partly tests how well the
    model was trained, which is the alternative hypothesis the claim exists to exclude.
  - v2, rho: a ratio of two rates over disjoint pixel sets, so no area term enters.
- **The plot is better evidence than the number.** The trimap curve shows the whole shape; the reader
  is not asked to accept one summary and one self-chosen bar.
- **The author's objection, which prompted the withdrawal:** "why can't we just look at our data and get
  a reasonable number — this is you introducing too much complex machinery." That objection is on the
  record and should be weighed, not dismissed.
</the_case_for_withdrawing>

<the_case_against_withdrawing>
Put as strongly as it deserves, because nobody has yet argued this side properly.

- **The history is exactly why a pre-registration has value here.** The first campaign reported ~98%
  mIoU and was withdrawn for leakage. The second set of numbers will be met with the reasonable
  suspicion that they were tuned until they looked right. A timestamped commitment made before training
  is the only clean answer to that, and this project has one.
- **Without a threshold, the claim is adjudicated by interpretation.** Two readers can look at
  90.1% -> 98.0% and disagree about whether that is "dominated by" boundary ambiguity. The manuscript
  currently asserts "every class collapses to a near-zero interior rate" — a claim a previous audit
  found does not survive leakage correction. That is what unbounded interpretation produces.
- **Withdrawing a timestamped commitment is itself a move a reviewer can question**, even when done
  before results exist. The document is committed and dated in the repository.
- **The threshold's provenance may be less damaging than assumed.** Calibrating on pilot data is normal
  practice. The pilot estimates used were restricted to pixels no training tile covered — the
  non-leaking subset — giving rho = 3.25 (baseline model), 4.77 (full model, val), 12.02 (full model,
  test). A bar at 4.0 sits below the weakest full-model estimate and above the baseline's, so the
  baseline fails and the full model is at genuine risk. That is a binding registration, not a
  decorative one.
</the_case_against_withdrawing>

<what_you_must_check_yourself>
Do not take any number above on trust. Verify what you rely on.

1. **Is there a literature-derived reference value for boundary error concentration?** Published trimap
   work reports error rate against distance to boundary; a rate ratio can be computed from such a curve
   or table. If one exists, an externally-grounded threshold beats a pilot-calibrated one. Start with
   Kohli, Ladicky & Torr 2009 (IJCV 82(3), trimap); Csurka, Larlus & Perronnin 2013 (BMVC, trimap
   accuracy and the accuracy-vs-bandwidth curve); Cheng et al. 2021 (CVPR, Boundary IoU). Then widen to
   any segmentation or land-cover paper reporting accuracy stratified by distance to boundary. Say which
   domain each number comes from — natural-image object boundaries are not field edges at 0.5 m.
   **If none exists, say so plainly.** Do not invent one.
2. **How wide would the interval be?** The threshold is judged on the lower bound of a block bootstrap.
   Test A has 16 independent 950 m blocks, Test B has 12. Characterise how large rho must be for the
   lower bound to clear 4.0 at those block counts — you cannot compute rho without trained models, but
   you can simulate from the real block structure via `scripts/analysis/utils.py:spatial_blocks` on
   `data/split_f1`. If the required rho is implausibly high, the registration is unpassable as written
   and that fact alone may settle question 1.
3. **What does the trimap curve alone actually establish?** Read
   `scripts/analysis/boundary_trimap_iou.py` and the manuscript's Figure 8 discussion. Is the curve
   self-sufficient evidence, or does it need a summary statistic to be checkable?

Check `~/Documents/Github/papers-md/` for markdown conversions first. If a paper is not there, use the
Zotero MCP but run `zotero_switch_library(library_id="6343594", library_type="group")` first — research
papers are in the group library, not the personal one. Cite nothing you have not opened.
</what_you_must_check_yourself>

<constraints>
- Recommend only. Change no code and no document; the author will apply your decision.
- Do NOT recommend calibrating any threshold from the corrected campaign's results. That is the
  post-hoc trap this exists to avoid. Anything you propose must be fixable before training.
- Do not recommend new experiments or new data collection. There is a deadline.
- Do not invent a citation or a number. "Not verified" is an acceptable answer.
- Weigh proportionality explicitly. This project's recurring failure has been machinery accreting around
  a decision instead of the decision being questioned — a threshold, then its provenance, then a search
  to fix the provenance. If your recommendation adds apparatus, justify why the simpler version is
  insufficient.
- If your honest conclusion is that the previous chat was right to withdraw, say so. If it is that the
  withdrawal was a mistake, say that just as plainly.
</constraints>

<output_format>
1. **Decision on question 1:** pre-register, or do not. One line, then the reasoning in under 200 words.
2. **If pre-registering:** the statistic, the threshold, and the provenance — with the methods sentence
   ready to paste. State whether the baseline still fails it and the full model is still at risk.
3. **If not:** exactly what is reported instead, and how a reader can tell the claim from a plausible
   story. Write that methods sentence too.
4. **Literature-derived rho values found**, as a table: source (author, year, domain) | quoted numbers |
   rho computed | fair bar here y/n. Or "none found".
5. **Interval-width finding** from item 2, with the numbers.
6. **What the trimap curve establishes on its own.**
7. **Options considered and rejected**, with reasons.
8. **Anything you could not verify.**
</output_format>

---

<supplied_evidence>
**Item 2 of `<what_you_must_check_yourself>` has already been measured. Do not re-derive it; use these
numbers, or attack them.** Measured 2026-07-26 on `data/split_f1`, before any training. Two
implementations sharing no code — one using `utils.spatial_blocks`, one written from scratch against
the rasters — agree on the geometry to seven digits.

**The block structure.** Test A: 294 tiles, 16 blocks at 950 m, Kish n_eff **9.85**. Test B: 172
scorable tiles after the registered boundary-free exclusion, 12 blocks, n_eff **7.36** — *corrected
2026-07-26 to 14 blocks and n_eff 7.27; the 12 came from a block function that scaled 950 m by one
mean latitude across two sites 50 km apart. See METHODS §6. The simulation below used 12, so it is
very slightly pessimistic; the conclusion is unchanged and the brief is superseded anyway.* Test A's band
area share 0.377737, Test B's 0.264796 — both reproduce the committed denominators exactly. Six of
Test A's sixteen blocks hold 74% of its tiles; Test B's per-block band share spans 0.055 to 0.539.

**How large rho must be for the lower bound to clear 4.0 with probability 0.80.** Simulated from the
real per-block band and interior pixel counts, with log-normal block random effects on both the
interior rate and the rate ratio (sigma_rho below; sigma_interior 0.35, interior rate 0.05):

| sigma_rho | Test A percentile | Test A BCa | Test A jackknife-t | Test B percentile | Test B BCa | Test B jackknife-t |
|---|---|---|---|---|---|---|
| 0.00 | 4.25 | 4.50 | 4.50 | 5.25 | 5.50 | 6.00 |
| 0.20 | 4.75 | 4.75 | 5.00 | 5.75 | 5.75 | 6.75 |
| 0.35 | 5.25 | 5.25 | 5.50 | 6.25 | 6.50 | 7.50 |
| 0.50 | 5.75 | 5.75 | 6.00 | 7.00 | 7.25 | 8.50 |

**The registration is nowhere an 80%-power test at rho = 4.** Even with zero between-block
heterogeneity the true ratio must reach about 4.25 on Test A and 5.25 on Test B. At the heterogeneity
the audit assumed, 5.25 and 6.25. Test B needs roughly 1.0 to 1.4 more than Test A throughout, purely
because it has fewer and more unequal blocks — nothing to do with label quality. The only non-leaking
prior estimates are 3.25 (baseline, validation) and 4.77 (full model, validation).

**Three claims adjudicated, because two of them are wrong.**

- *"A properly-covering interval will be wider, so rho may fail where it previously passed."*
  **Refuted.** BCa's median width is 1.012x the percentile's and its *lower* bound is HIGHER in 63 of
  64 simulated cells (median +0.030). Switching to BCa flips a verdict 2–9% of the time, and more
  often from fail to pass than the reverse.
- *"The registered percentile interval under-covers."* **Confirmed.** Coverage 0.86–0.93 against a
  nominal 0.95, and 0.92–0.93 even with no heterogeneity, which is six Monte-Carlo standard errors
  low. The miss is asymmetric and on the side the gate reads: the interval lies entirely above the
  truth about 7% of the time on Test A against a nominal 2.5%. BCa is worse, not better — 0.74–0.86,
  because with twelve blocks its acceleration is estimated from twelve jackknife points. Only a
  delete-one-block jackknife t-interval on log(rho) covers near nominal, and it is 20–45% wider.
- *"A percentile interval on log(rho) is bit-identical."* **Refuted**, but only at 5e-8 — quantile
  interpolation does not commute with a convex transform. Numerically irrelevant; the word
  "bit-identical" should simply not be written down.

**What this does and does not settle.** It does not decide question 1. It does establish that a bar of
4.0 judged on a lower bound is not a bar of 4.0: its operating point is ~5.2 on Test A and ~6.3 on
Test B. If the registration is kept, that operating characteristic has to be registered with it, and
the interval has to be the jackknife rather than the percentile or BCa — which is two more pieces of
apparatus defending the first one. Weigh that against the proportionality constraint above.

Reproduce: `PYTHONPATH=. python scripts/analysis/block_phase_sweep.py` for the block structure and
n_eff; the simulation is in the session scratch, not the repo, because it answers a question that may
be withdrawn.
</supplied_evidence>
