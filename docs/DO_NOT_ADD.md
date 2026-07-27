# DO NOT ADD — sentences and sources this design forbids

**Read this before writing any prose, and before commissioning any review.** Everything here has
already been established and re-established, in some cases three times. It exists so that the next
chat — which will not have the context — does not audit it a fourth time.

Two kinds of entry: **sentences the design cannot support** (§1), and **sources that do not exist or
do not say what they are cited for** (§2). Neither list is about the code.

Companion documents: `docs/CORRECTIONS.md` (things currently in the manuscript that must change),
`docs/METHODOLOGICAL_CHOICES.md` (why the design is what it is),
`docs/audit/DECISIONS_REBUILD_2026-07.md` (D1–D19, settled with reasoning).

---

## 1. Sentences that must never be written

| Do not write | Because |
|---|---|
| "Label quality, not model capacity, is the binding constraint" | The architecture is held constant in every contrast the design computes, so an architecture-induced boundary limit is unfalsifiable here. The claim is a reading of convergent evidence, not a measured result — say so. |
| "Annotation effort is best spent at class boundaries" *(unconditioned)* | If the boundary residual is mixed pixels at 0.5 m, re-annotation recovers nothing. Where a fresh pass would change most labels is not where accuracy can be recovered. |
| "The model generalises to unsurveyed terrain" | Test B is n = 2 purposively chosen sites, 5.16 km², where semi-natural is 60% of foreground against 4% in training. No form of "generalises" in the statistical sense is available. |
| "Cross-dataset transfer improves X by Y pp" *(as the name of main effect A)* | The transfer arm receives exactly 2.00x the Biodiversity gradient steps. Factor A is the pre-train-then-finetune **procedure**; naming it "transfer" attributes the effect to a mechanism the design cannot isolate. |
| "92% / 96% of foreground error falls within 8 m" | Leakage-inflated, and the *share* form was registered and retracted twice for landscape-dependence (D9 v0/v1). Both numbers are void. |
| "The class-balanced sampler has no effect" / "is redundant" | Claiming "no effect" needs a bar for what counts as nothing, which D17 forbids. The honest form is practical: report the number and say it is too small to justify the added pipeline complexity. |
| "Approximately 88% of masks contain labelling errors" *(as a measured bound)* | No recorded protocol. It may be cited as **motivation** (`PAPER_PURPOSE.md:48` already says so) and never as a measured inter-annotator ceiling — this project has never measured one. See §2 for why this number is especially dangerous. |
| "Every class collapses to a near-zero interior rate" | Stated at an undeclared 1.5 m / 8 m partition, and the previous campaign's version was flagged internally as unsupportable (D16a). |
| Any "95% CI" on a Test B level | Both test sets are complete enumerations; there is no sample to resample. Uncertainty is per-seed and paired (METHODS §7). Contrasts carry intervals; levels do not. |
| "N independent 950 m blocks" | The grid counts **cells touched**, not independent parcels: Test A touches 16 cells on 7.52 cells' worth of ground, Test B 14 on 5.72. Write "grid cells containing the class". |

## 2. Sources that must never be cited

### ⛔ Ortiz et al. 2025 (TGRS) — FABRICATED. It does not exist.

Two targeted web searches negative; not cited by Csurka or Cheng. Recorded in
`notes/POST_COMPACT_CHECKS.md` and `notes/BOUNDARY_ADDITIONS_DRAFT.md:86`. Removed from the code and
the manuscript.

**But roughly forty references survive in `notes/`**, several presenting it as read in full with
specific numbers — BS_γβ recovery, "+11.7pp Dice at 1px", Fig 3 / Fig 7 / Table I, a 2 km GOES pixel
comparison. Warning banners were added to the four worst offenders on 2026-07-27
(`NARRATIVE_STATE_2026-06-26.md`, `PLOT_PLAN_2026-06-26.md`, `NEW_CITATIONS_STEP4.md`,
`CLAIMS_CITATION_AUDIT/00_PLAN.md`). **The rest of those notes are unreliable on this point.**

**The specific trap, which has already caught one external reviewer:**
`NARRATIVE_STATE_2026-06-26.md:230` reads *"Ortiz's 88% = up to 8 annotators/scene"*. There are **two
different 88s** in this project:

| the number | what it is | may it be used? |
|---|---|---|
| ~88% of inspected masks contain labelling errors | a domain expert's audit (`PAPER_PURPOSE.md:14`) | **yes, as motivation only** |
| "Ortiz's 88% inter-annotator bound" | **from the fabricated paper** | **never** |
| ~87.57% / "~20% improvement" | a collaborator's mIoU, from a self-agreement loop where 188/215 masks were rewritten by the models under test | **never** (`PAPER_PURPOSE.md:48`) |

An inter-annotator ceiling would be a *measured* label ceiling. This project has never measured one,
and saying it has would be the single most damaging error available.

### Kohli, Ladický & Torr 2009 — real, but not a citation for the 8 m band

They **sweep** the band width and report a curve; they never choose one. Their only numbers are
*"an 8 pixel band"* and *"an evaluation band width of 16 pixels"*, inside a figure caption, in
**pixels on 320×213 MSRC images**. Ours is 8 **metres**. The numeric coincidence is a coincidence.
"Trimap" appears three times in the paper, all in one caption, and is never defined in prose — so it
cannot be attributed to Kohli as a named metric either.

Kohli **is** the right citation for two things: that the deliverable is a curve, and that the
dataset's own ground truth failed at boundaries (*"quite rough… a significant number of pixels…
unlabelled… generally occur at object boundaries"*).

### Cheng et al. 2021 (Boundary IoU) — real, but supports the opposite of what it looks like

They describe ground-truth-only banding precisely **and as a defect Boundary IoU exists to remove**:
*"not symmetric and favors predictions whose masks are larger than the corresponding ground truth
masks"*. That asymmetry applies here and must be stated as a limitation, not cited as support.

Their width rule — *"the annotation consistency sets the lower bound on d"* — **cannot be applied
here**, because this dataset has a single annotation pass.

**Currently converted from arXiv, not the CVPR proceedings.** Any quotation must be checked against
the proceedings before submission.

### Volpi & Tuia 2017 — real, and the derived ratio is right, but it is not a benchmark

Their 1.24–1.33 (Vaihingen) and 1.14–1.19 (Potsdam) reproduce exactly from Tables I and III — I
recomputed them. But their band is **3 px at 9 cm and 5 cm = 0.27 m and 0.15 m**. Ours is **8 m**,
30–53x wider. It is not a same-scale comparison for rho and must not be presented as one.

Also note their own inference — *"boundaries are often blurred within the 3 pixel erosion radius"* —
makes exactly the leap this paper is trying to avoid: it reads "error concentrates at boundaries" as
"boundaries are blurred". Cite the observation, not the interpretation.

### Csurka et al. 2013 — real, and the safest of the four

Correct citation for **why a curve rather than a single width**. Their use of a validation set is a
PASCAL-server submission-quota constraint, **not** a methodological precedent for evaluating boundary
metrics on validation — do not cite it as one. The footnotes carrying that reason are **dropped by the
markdown conversion**; use the PDF.

---

## 3. Where the sources actually are

All four boundary papers are now converted, with DOIs and Zotero keys recorded in
`references_md/SOURCES_BOUNDARY_LITERATURE.md`. Three of them were absent from every library on this
machine until 2026-07-27, which is why §10 and §11 of METHODS were unverifiable by anyone. They are
verifiable now. **Do not re-derive those readings — check them against the conversions.**
