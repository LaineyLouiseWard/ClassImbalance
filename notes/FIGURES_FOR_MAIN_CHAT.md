# Three figures, ready to place — handover to the manuscript chat

Written 2026-07-29 by the figures chat. Nothing here touches `manuscript_v2/main.tex` or
`Bibliography.bib`. PDFs are already copied into `manuscript_v2/Figures/`.

Every factual claim below was reached by **two independent routes** before being written down; where
only one route exists, the line says so.

---

## Where the files are

| figure | PDF (already in the bundle) | PNG to look at | printed size |
|---|---|---|---|
| pair error by class pair | `manuscript_v2/Figures/pair_error_confusion.pdf` | `figures/pair_error_confusion.png` | 6.61 × 3.37 in |
| the two grasslands on the ground | `manuscript_v2/Figures/two_grasslands_qualitative.pdf` | `figures/two_grasslands_qualitative.png` | 7.07 × 8.07 in |
| do the classes meet | `manuscript_v2/Figures/class_seam.pdf` | `figures/class_seam.png` | 6.45 × 2.88 in |

**All three are built for the wide layout**, not `\textwidth`. `\textwidth` is 394.36 pt = 13.90 cm
and `\extralength` is 4.61 cm, so wrap each in the pattern `main.tex` already uses at :354:

```latex
\begin{adjustwidth}{-\extralength}{0cm}
  \includegraphics[width=\linewidth]{Figures/pair_error_confusion.pdf}
\end{adjustwidth}
```

Built at 7.28 in, one point in the figure is one point on the page. Dropping them into a plain
`\textwidth` figure shrinks every label by a quarter.

---

## What replaces what

- `confusion_matrices.pdf` is **gone**. The script was renamed `scripts/figures/confusion_matrices.py`
  → `pair_error_confusion.py` and cut from three panels (baseline / full / delta) to one factorial
  cell. Any `\includegraphics{Figures/confusion_matrices.pdf}` must be repointed.
- `class_distributions`, `workflow_pipeline` and `oem_mapping` are out of the build, per the brief.
- `two_grasslands_qualitative` and `class_seam` are new.
- `docs/FIGURES.md` and `scripts/figures/build_all_figures.py` are updated to match.

---

## Figure 1 — `pair_error_confusion`

**What it is for.** The sentence at `NARRATIVE_LITERATURE_FINAL.md` :78 — "Grassland and semi-natural
grassland confused with each other account for 46.7% of foreground error — more than twice what their
share of the ground would give it."

Panel (a) is the 5 × 5 foreground confusion on **absolute volume**: each off-diagonal cell is that
directed confusion as a share of all foreground error, so the panel sums to 100%. It is deliberately
**not** row-normalised — row normalisation turns it into per-class recall and hides that one pair
holds half the volume.

Panel (b) is the second half of the claim, and it is not redundant with (a). (a) shows the pair is
the *largest*; the obvious objection is "the two biggest classes confuse most". (b) answers it:
Grassland–Semi-natural rises from 22.2% expected-from-area to 46.7% observed, while Forest–Grassland
*falls* from 50.3% to 30.1%. That reversal is the 2.1× the paper quotes.

**Caption must carry:** baseline cell, ten seeds, 90 Test A chips; that (a) is a share of all
foreground error and not row-normalised; and Forest–Grassland at 30.1%, because
`NARRATIVE_LITERATURE_FINAL.md` :80 forbids writing that no other pair is close.

Numbers on the figure: 46.7, 2.1×, 30.1, 50.3, 22.2 and every matrix cell. All ledgered.

---

## Figure 2 — `two_grasslands_qualitative`

**What it is for.** That the pair's error sits **inside** large areas rather than along their edges,
and that the two classes are a distinction of management rather than appearance
(`NARRATIVE_LITERATURE_FINAL.md` :27–31).

Four columns, four rows (image / reference / prediction / pair error). The error panels show whole
parcels flipped, not rims.

**The selection rule, which must be in the caption or a reviewer assumes the prettiest chips were
chosen:**

> Of the 90 Test A scoring chips, the twelve that are fully labelled and hold at least one hectare
> of both Grassland and Semi-natural. Those twelve ranked by pair-error rate averaged over the ten
> seeds, and the 0th, 33rd, 67th and 100th percentiles shown — minimum, maximum and two interior
> points. Each column is displayed at the seed whose rate on that chip is closest to that chip's own
> ten-seed mean.
>
> **CAPTION CHANGE 2026-07-29.** The panel titles now carry the ten-seed mean rate alone —
> "(a) 1.7%" — and no longer print the displayed seed. If the caption says the seed is on the
> panel, that is now wrong. The four seeds, in column order, are **50, 48, 42, 51**; the script
> prints them on every run. The number is the pair-error rate: (grassland-called-semi-natural +
> semi-natural-called-grassland) pixels over grassland + semi-natural reference pixels.
> The caption must also carry the faded-reference note, which came off the legend.

**Why a ladder and not one chip — this is the part worth knowing.** The first version showed the
median chip plus the largest-volume chip. An independent checker, given only the per-chip statistics
and told to argue the choice was cherry-picked, broke it: a chip's pair-error rate varies by up to
three orders of magnitude across the ten seeds, so "the median chip" is not a stable object, and five
individually innocuous perturbations of the rule — a minimum class area, seed median instead of seed
mean, 95% instead of 100% labelled, per-seed instead of pooled ranking, rate instead of volume — each
moved the answer to a chip on which the figure would have looked worse. Showing the range removes the
choice. **Do not quietly reduce this to one panel.**

**Never call these patches fields** (`NARRATIVE_LITERATURE_FINAL.md` :188). Almost all the grassland
in the scored chips is cut by a chip edge, so every size here is part of a field.

---

## Figure 3 — `class_seam`

**What it is for.** The lead argument, `NARRATIVE_LITERATURE_FINAL.md` :97–102 — "the two grassland
classes barely touch each other … so most of the error between them cannot be a seam problem."

Panel (a): for each row class, the share of its ground lying within 8 m of each column class, over
the 90 chips. Read along the Grassland row — 21.4% within 8 m of forest, **0.6%** within 8 m of
semi-natural. The two cells of the pair are outlined. Panels (b) and (c) show one chip twice so a
reader sees what those two numbers look like: a dense lattice along every hedge, and nothing.

This is arithmetic on the two **reference masks**. No model output enters it, which is why it is a
bound rather than a failed test.

**Chip selection for (b)/(c), also for the caption:** of the chips that are fully labelled and
contain grassland, semi-natural and forest, the one minimising the larger of the two absolute log
deviations from the pooled values (0.60% and 21.44%) — `biodiversity_1594`, at 0.50% and 23.75%.
Choosing on the semi-natural number alone returns a chip whose forest seam is twice the pooled value,
which would overstate the contrast.

---

## One number you should consider putting in the text, currently only in a ledger row

**54 of the 90 Test A chips contain no semi-natural grassland at all, and they carry 40.3% of the
pair's whole error volume.** On such a chip there is no seam with semi-natural anywhere, so none of
that error can be a seam drawn in the wrong place. It is a blunter statement of the same argument
than the adjacency matrix, and it does not depend on any choice of distance.

Confirmed twice: computed by an independent checker from the ten per-seed files, and recomputed here
straight from the reference masks. The pair-error total it divides into, 12,899,256 px, matches the
existing `pair error pixels` ledger row exactly.

Ledger rows added: `Test A chips with no semi-natural at all`, `pair error on chips with no
semi-natural, %`, `pair error px, summed per chip (10 seeds)`.

---

## Two traps that cost time here — do not repeat them

**1. Two different "8 m" quantities, differing by ~24 points.**
`scripts/analysis/pair_error_geometry.py` measures each error pixel's distance to the nearest
reference boundary **of any kind**, with a tile-edge guard — that is where 75.9% / 68.9% "beyond
8 m" come from. `scripts/analysis/pair_error_by_tile.py` (new) measures distance to the nearest
pixel **of the class it was confused with**, unguarded. They answer different questions. A checker
flagged them as an inconsistency; they are not, but never quote one for the other, and never put
them in one sentence without saying which is which.
Confirmed twice: the discrepancy was raised independently, and resolved against the docstring of
`pair_error_geometry.py` :17–25, which states the any-boundary convention and the guard in its own
words.

**2. Seed 44 is not the median seed, and two scripts say it is.**
`scripts/figures/ablation_qualitative.py` :14 and :274 call seed 44 "the median seed used for the
paper figures". On Test A, `stage1_baseline`, foreground mIoU, seed 44 ranks **4th of ten** (58.14%);
the two middle seeds are 47 (60.13%) and 43 (60.36%). The claim is almost certainly left over from
the withdrawn campaign. **Single route so far** — computed once from
`analysis/metrics/test/seed*_stage1_baseline.json`. It does not affect the three new figures, which
state their seed rule explicitly, but `ablation_qualitative`'s caption should not repeat "median
seed" without someone checking it.

---

## Known imperfections — decide whether they matter, do not assume they are fixed

A checker was given the three PNGs and the draft captions and **nothing else**, and asked what each
figure shows and what is wrong with it. **Single route — one agent, not confirmed by a second.** It
read all three findings correctly from the images, which is the test that mattered. What it flagged:

- **Fig 1.** No colourbar on (a), so the blue shading has no key — it is emphasis, and the caption
  should say the numbers are the data. The grey diagonal is never explained; say "correct pixels,
  excluded" in the caption.
- **Fig 2.** The pair-error row shows the reference classes faded behind the two error colours, and
  the legend does not say so. Add it to the caption. Settlement blue and "semi-natural called
  grassland" navy are close in hue and both appear in columns (c) and (d).
- **Fig 2.** The two numbers in each column header (ten-seed mean, then the displayed seed) are not
  defined on the figure. The caption must define them.
- **Fig 3.** Panel (c) reads as a failed render rather than as a result — the emptiness *is* the
  finding, and the caption has to say so in its first clause or a reader thinks the overlay is off.
- **Fig 3.** Panel (a)'s purple scale saturates: 70.3 and 69.0 are both at the dark end and
  everything below ~25 sits in a near-white band. Same remedy as Fig 1 — the numbers are the data.
- **Fig 3.** The matrix invites reading *across* rows (70.3 against 21.4), which is mostly an
  object-size effect. `NARRATIVE_LITERATURE_FINAL.md` :120 forbids exactly that comparison, so the
  caption must warn against it.
- **Fig 3.** The chip's numbers (23.7%, 0.50%) do not equal the pooled numbers (21.4%, 0.6%), and
  nothing says that is expected. One clause fixes it.

None of these change a claim. They are caption work, which is yours.

## Which of the OLD manuscript's floats to bring across

The question: `manuscript/main.tex` (the superseded draft) carries thirteen figures and four tables.
Which of them need to exist in the rebuild?

**Answer: two. `oem_mapping` whole, and two of `boundary_limited_error`'s three panels.** Everything
else is already in v2, superseded by the three new figures, or was cut for a reason that still holds.

The bar I applied: a float earns its place only if it carries something the prose in
`manuscript_v2/main.tex` cannot. The Results text is already dense with numbers — every headline
figure is in the prose — so "illustrates the section" is not a reason. The float has to do work
sentences cannot: hold a many-to-many correspondence, or show a shape.

---

## Bring across

### 1. `oem_mapping` — the strongest case of anything in the old paper

v2 :210 states it in prose: *"The grounded mapping from OpenEarthMap reaches only three of the five
classes: its agricultural, rangeland and bareland classes all resolve to grassland, so neither
cropland nor semi-natural grassland receives a transferred label."*

That sentence asks the reader to hold nine source classes, six target classes and an argmax over a
confusion matrix in their head. It is a many-to-few correspondence, which is precisely what a
diagram does and prose does not. It is also the paper's only real methodological artefact — the
mapping is grounded on the teacher's empirical confusion rather than hand-written by name — and a
reader cannot check the grounding from a sentence.

And it is the paper's best answer to *why* the null result is not a surprise. The narrative makes
that point explicitly (:54–57): one property of pre-training is readable before a single GPU hour is
spent. A figure that lets a practitioner check that in their own class mapping is the most
transferable thing in the Methods.

**Verified not stale.** `oem_mapping.tex` hard-codes its percentages; I regenerated them from
`artifacts/teacher_oem_gt_confusion_f1.npz` via `_gen_mapping_values.py` and they match exactly
(98.4, 55.7, 56.7, 57.9, 52.2, 79.9, 81.8, 88.3). It was corrected on 2026-07-28 — the Water arrow
was removed because drawing it shows a mapping the model does not use. This one is safe to lift.

### 2. `boundary_limited_error`, panels (b) and (c) only — never panel (a)

Two sentences in v2 are load-bearing and currently unsupported by any figure:

- :247 — *"This is not because the map has unusually clean edges. Across all classes a pixel within
  one metre of a class boundary is misclassified about 3.7 times as often as one further away… The
  map follows the usual boundary pattern; the grassland pair is the exception to it."*
- :249 — *"Beyond thirty-two metres from any boundary, forest is wrong 0.7% of the time and
  grassland 5.1%, but semi-natural grassland is still wrong 27.0%."*

The first is the paper's pre-emption of the obvious referee objection — *your model just has clean
edges, so of course the error is not at boundaries*. If that pre-emption fails, the whole seam
argument fails with it. Panel (b), the error rate against distance to the nearest boundary on a log
axis, is the display that settles it: the reader sees a normal decay, then reads the grassland pair
as the exception. Panel (c) carries the second sentence, the per-class deep-interior contrast.

**Panel (a) is now gone — I made this edit, 2026-07-29.** `boundary_limited_error.py` renders two
panels, relettered (a) and (b), at 7.28 in for the wide layout. `panel_recovery` is left in the file
unused, because the JSON block still feeds it and a supplementary table may want it. This was a
correctness fix, not a preference: Trimap IoU recovery *is* the withdrawn label-ceiling claim — the figure's own
title is "The residual error is boundary-limited". Re-title the whole float; the current one asserts
a conclusion the paper has withdrawn.

**Two cautions the caption must carry.** These numbers are computed over all 294 overlapping Test A
tiles, whereas `pair_error_confusion` and `class_seam` are on the 90-chip non-overlapping subset —
`NARRATIVE_LITERATURE_FINAL.md` :161 forbids putting the two populations in one sentence without
saying which is which. And the figure calls everything beyond eight metres "interior" while the
prose reserves "deep interior" for thirty-two (:166–169); one of the two has to change or they will
read as contradicting each other for the same class.

Data is live (`analysis/label_ceiling/test/`), so it rebuilds without a cluster run.

**One decision left, and it is yours.** The script's default cell is `stage3_clsbal`
(:254), the shipped model — but `pair_error_confusion`, `class_seam` and
`two_grasslands_qualitative` are all `stage1_baseline`, and so are the narrative's per-seed interior
numbers (`analysis/interior/interior_test_stage1_baseline.json`). I did **not** change the default,
because either is defensible — the narrative says the geometry barely moves across the four
configurations (:129–130) — but a paper that mixes cells across figures without saying so is a trap
a referee will find. Either rebuild it with `--cell stage1_baseline` for consistency, or state in the
caption that this one float is the shipped model and the others are the baseline.

The float still needs re-titling. Its current caption opens "The residual error is boundary-limited",
which is the withdrawn claim.

---

## Leave out, with the reason each fails the bar

| from the original | verdict | why |
|---|---|---|
| `workflow_pipeline` | cut | a 2×2 crossed over two binary factors is one sentence plus the table v2 already has. A flowchart of a design this simple is padding. |
| `class_distributions` | cut | four bar panels for numbers the prose gives as single figures. The one thing it carried that still matters — cropland absent from OEM — is `oem_mapping`'s job, so it duplicates a figure we are keeping. |
| `mitigation_axes` | cut | draws a data-versus-model split the fixed architecture cannot test. Schematic, "placements are schematic, not measured" in its own caption. |
| `ablation_qualitative` | cut | four cells × four chips of near-identical predictions, inviting the reader to compare cells the design cannot separate. Also states seed 44 is "the median seed", which is wrong — see `FIGURES_FOR_MAIN_CHAT.md`. |
| `confusion_matrices` | superseded | replaced by `pair_error_confusion`, one cell, absolute volume. |
| `frequency_vs_difficulty` | cut | five classes cannot support the claim in either direction. |
| `reliability_ece`, `uncertainty_quality`, `uncertainty_overlay` | cut | the residual-uncertainty section is gone. `uncertainty_overlay`'s default tiles are now in train and test. |
| `confident_learning_overlay` + its table | cut | headline was a retracted statistic, and its noise assumption is violated by the very spatial structure the paper claims. |
| Table: per-class IoU, four cells, 219 val tiles | superseded | the withdrawn split. v2's `tab:factorial` replaces it on Test A. |
| Table: factor effects | superseded | merged into v2's `tab:factorial`. |
| Table: held-out test set, 218 tiles | cut | withdrawn split, and the narrative gives Test B one hedged sentence with no unrowed number. |

---

## The resulting set — six figures

1. `study_area` — in v2 already
2. `oem_mapping` — **lift from the original**
3. `pair_error_confusion` — new
4. `class_seam` — new
5. `two_grasslands_qualitative` — new
6. `boundary_limited_error`, panels (b) and (c), re-titled — **lift from the original**

Plus `tab:factorial`, already in v2. Two figures in Methods, four in Results, one table. For a case
study that wants to be short, that is the right shape: the Methods figure explains why an
intervention could not reach two classes, and the four Results figures are the argument in order —
which pair, whether they touch, what it looks like, and that the map is otherwise normal.

---

## If you can only fit four

Drop `two_grasslands_qualitative` and `boundary_limited_error`. Keep study area, `oem_mapping`,
`pair_error_confusion`, `class_seam`. Every claim still has a figure. What you lose is the part that
makes the argument land on a reader who does not read matrices, and the pre-emption of the
clean-edges objection — so if a referee raises that objection, panel (b) is the answer and it should
go straight into the response.

---

## One thing that is not a figure

The graphical abstract. MDPI asks for one, it is built separately
(`scripts/figures/graphical_abstract_panels.py` + `graphical_abstract_tikz.tex`), and its panels
come from the withdrawn campaign. It is the one image a reader sees before the abstract. Somebody
should check it before submission; I have not.

---

## Rebuild

```bash
python scripts/figures/build_all_figures.py           # all figures, syncs to manuscript_v2/Figures/
python scripts/figures/pair_error_confusion.py        # or one at a time
PYTHONPATH=. python scripts/analysis/verify_narrative_numbers.py    # 84 numbers, all reproduce
```

Predictions for the panels are staged under `analysis/panel_root/seed<N>/...` (four chips, four
seeds, ~12 MB); per-chip statistics for all ten seeds are in `analysis/tile_stats/`. Both were
pulled from Sonic and are gitignored, so the figure scripts fail loudly rather than silently if they
are missing.
