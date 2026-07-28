# Plan — manuscript surgery

> **ALL FOUR GATES ARE GREEN as of 2026-07-28.** Ten seeds completed; metrics aggregated on the
> non-overlapping subset; class-pair share and symmetry restated (§7 fired — see below); boundary
> sweep restated per seed, four cells, both widths, plus the registered across-cell arm.
>
> **But the plan below is now partly obsolete, because the numbers moved the story.** Read
> `docs/NARRATIVE_FINAL.md` first — it replaces the arc this plan was written to serve. In
> particular R1's upper-bound framing is gone, R2's symmetry claim is withdrawn, and the paper's
> centre of gravity is now the per-class factorial decomposition, which this plan does not contain.
> §1 (strike the forbidden sentences), §5 (sentences owed to Methods and Limitations) and §6
> (language discipline) are unaffected and still correct.
>
> **§7 outcome:** the symmetry placeholder FIRED — the confusion is imbalance-shaped, not symmetric,
> and the absolute-count version of the claim was wrong. The registered across-cell arm PASSED in
> 10 of 10 seeds. Boundary ratio held its shape (3.85 at 1 m, 2.28 at 8 m). Transfer's +2.6 pp did
> not survive.

Line numbers are as of 2026-07-28 and will drift. Re-grep after the first edit. Class is `mdpi`
`remotesensing,article,submit`, single column.

---

## 0. The paper in plain words

**The partner's model stays as it is. We asked whether its errors could be fixed by tackling the class
imbalance in their data. They cannot. The errors come from two things: mistakes pile up at the edges
between land types, and the model cannot reliably tell the two grassland types apart.**

Everything below serves that. If an edit does not, it is not needed. Full version and its two
corrections: `CORRECTIONS_PAPER_PT4.md`.

## 1. Strike the forbidden sentences (30 min, first)

On `DO_NOT_ADD.md`'s list. Fastest possible rejection.

| line | strike |
|---|---|
| 52 abstract | **THE ABSTRACT IS AUTHOR-FROZEN** (`PROSE_PASS_2026-07-27.md`). Lift the freeze deliberately before any of the three abstract edits below |
| 52 abstract | "label quality, rather than class imbalance or model capacity, as the dominant remaining constraint, and shows where scarce annotation effort is best spent" |
| 52 abstract | "cross-dataset transfer" as the name of factor A → *OpenEarthMap pre-training* |
| 52 abstract | "applies directly when fine-tuning pre-trained Earth-observation foundation models" — contradicted by the paper's own limitation |
| 64 highlight | "indicating a label-quality ceiling rather than a limit of model capacity" |
| 68 highlight | "annotation effort is best spent there" — must be conditioned |
| 126 contribution 3 | same forbidden sentence as the abstract |
| 459 | "the consistency of that supervision, not the capacity of the model, becomes the binding constraint" |

**Do not touch line 471** — *"such models blur edges even on clean labels… convergent evidence rather
than proof."* The most protective sentence in the paper.

## 2. Contribution 1 and the title

Contribution 1 introduces a proprietary dataset a reader cannot obtain. Replace with the measurement
the benchmarks discard: error rate as a continuous function of distance to the nearest contour. ISPRS
2D Semantic Labeling erodes a 3-pixel disc before scoring, names uncertain border definitions as the
reason, and offers no measurement for the width.

**Hedge it** — "we are not aware of", never "nobody has". Volpi & Tuia did compare eroded against
non-eroded ground truth. What is absent is the ratio stated as such, and the curve.

**Title:** drop "Diagnosing". The paper declines to diagnose.

## 3. Cuts

**One only.** `frequency_vs_difficulty` (331–338) and the rarity claim at 126, 336, 392, 501 — the
claim is refuted, Spearman train-share against IoU is +0.70. Not a length cut.

No layout work. No widening, no combining floats.

*Reserve, only if the page limit actually bites:* §3.5 and its three figures (~2 pp), the
confident-learning appendix (~1 pp), `mitigation_axes`, the TTA row.

## 4. Additions

| | content | slot |
|---|---|---|
| R1 | Factorial on **Test A**, not validation. Per-class table, paired per-seed contrasts, factor A on both test sets. State the upper-bound framing | replaces §3.2+§3.3 merged |
| R2 | **The class pair** — ~half of foreground error, near-symmetric. Read the confusion figure on absolute volumes and net flow, never row-normalised | slot vacated by the cut |
| R3 | Boundary concentration as a **width sweep**, per seed not ensemble | rewrites §3.4 |
| R4 | **The across-cell arm**, in relative terms, with the falsifier | new, after R3 |
| R5 | **Is it the input?** NIR control, spectral probe narrowed, terrain as geographic shortcut | moved up from 461–465 |
| D3 | Two kinds of boundary - real edges, and transitions with no line on the ground - and why the confusion is symmetric. **Labelled interpretation.** Plain words, not "gradational" | Discussion |
| D4 | The disjunction, and the conditioned recommendation | Discussion |

## 5. Sentences owed to §2 and Limitations

Evidence exists, prose missing. One or two sentences each.

Per-site normalisation · non-overlapping scored subset (cite Cira et al. 2024) · ground area not tile
counts, Test B 50.7% labelled · tiled inference as a third rival, rho biased downward · NDWI reframed
as missing-SWIR (McFeeters 1996 vs Gao 1996) · multivariate separability considered, not computed ·
texture and multi-temporal explicitly untested · **single annotator, single protocol** · Test B is not
a sample of Irish uplands, delete "generalisation".

## 6. Language

**`/prose-pass` cannot write the new material** — it restructures existing sentences with the claim
skeleton fixed. It can only check a draft.

**A full prose pass is already done** — `notes/PROSE_PASS_2026-07-27.md`, all six sections, register
clean, worklist exhausted. New material must not undo it. Read that file before drafting.

**Calibrate first.** Run it over `main.tex` and read the voice detection, the protected-term ledger and
`high_premium_leave_alone` before writing anything. Those sentences are the shape to match.

    python ~/.claude/skills/prose-pass/tools/prose_metrics.py manuscript/main.tex --mode paper --strictness 2 --worklist
    python ~/.claude/skills/prose-pass/tools/prose_metrics.py <new-text> --register

**Conventions:** "we" throughout · passive where standard · technical vocabulary exact · em dashes
sparse · one colon per paragraph · never call the reference data "truth" ·
`Research_Communication/STYLE_WARNINGS.md`.

**Check cumulative load, not per-sentence scores.** The completed pass found that splitting a 118-word
sentence into 36+43+43 scored as a win on every per-sentence metric and was a loss in practice: one
monster traded for a run of long sentences. Back-to-back runs of >35 words went 11 -> 4 across the
paper. Do not put a new run back in. `--register` cannot see this.

**Then read the section continuously.** The same pass records the lesson three times: every continuous
read found a defect the flagged list could not, including six surviving "transfer" references for
factor A sitting in paragraphs beside renamed ones.

**The known failure:** casual idiom, invisible to the writer at the time. Two real examples from
2026-07-28, both caught only in review: "doing the damage", "holding the numbers down". Check for it
rather than trusting the draft. And no sentence more sophisticated than the argument it carries.

## 6b. Two open FLAGs awaiting an author decision

Both in §4.4, from the completed prose pass. Neither can be fixed without changing a claim.

- *"…spectral indices and terrain, **do not**."* Elliptical - do not *what*? The antecedent is two
  clauses earlier. Three rewrites tried, all rejected on FK-t or on the claim skeleton.
- *"the features distinguishing **the minority class**"* - singular, where two minority classes are
  defined. Mild, disambiguated by the next sentence.

## 7. Placeholders that could change the plan

- **Symmetry margin — the exposed one.** Five seeds give 26.4% against 22.6%. If ten seeds make it
  clearly lopsided toward the majority class, **the imbalance account is back and R2/D3 fail.** Check
  this first.
- Class-pair share ~49%. Below about 40% the "half the error" framing weakens.
- Boundary ratio, one seed: ~4x at 1–2 px, ~2x at 8 m. Shape matters more than level.
- Factorial: transfer +2.6 pp, sampler ~0. A materially positive sampler needs rewording.
- **The across-cell arm has never been run.** If the near-boundary rate does not stay flat relative to
  the interior rate, the registered falsifier fires and the label-ceiling reading is unsupported by
  that evidence. Plan for that outcome existing.

## 8. Order

1. §1, §2 and everything in §5 — no campaign dependency, start any time
2. The cut in §3 — frees the slot
3. R1–R3 — needs ten-seed numbers
4. R4–R5, D3–D4 — R4 needs the arm computed

**Last, and in one pass together: abstract, contributions, conclusions, cover letter.** They are the
compressed statement of everything else, so drafting them before the body is settled means drafting
them twice. They are also where the forbidden sentences cluster and where a reviewer looks first, so
they carry the most risk per word. Lift the abstract freeze at this point, not earlier.
