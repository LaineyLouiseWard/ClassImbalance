# Claim-support audit — `manuscript_v2/main.tex`, 2026-07-29

Does each citation support the claim it is attached to? (Different question from
`docs/BIB_AUDIT_2026-07-29.md`, which only asked whether the entries describe real papers correctly.)

**25 citation sites, 22 distinct keys, 28 key-checks.** Three agents plus a dedicated `DO_NOT_ADD`
compliance lane. Sources read from the repo's own `references_md/` conversions.

**16 SUPPORTS · 10 OVERREACH · 0 CONTRADICTS · 2 SOURCE UNAVAILABLE · 1 DO_NOT_ADD VIOLATION.**

Findings are anchored on **quoted manuscript text, not line numbers** — `main.tex` grew from 52,814
to 55,602 bytes during the audit and every line number shifted.

## Scope — three limits, all real

- **The draft is a third written.** Abstract, Introduction, Highlights and Conclusions are
  placeholders. The hazards that live there — the ~88% mask-error motivation, the ceiling framing,
  the boundary prior — are **unwritten, not cleared**. Re-run after steps 4 and 5.
- **22 of 58 bibliography entries are cited.** The citation load will roughly double.
- **Four findings I verified myself** against the sources (marked ✓). The rest are single-agent and
  worth a second look before acting.

---

## 1. `DO_NOT_ADD` violation — fix before anything else

**✓ Volpi presented at our scale.** Found independently by two agents; I confirmed it.

> "Aerial land-cover benchmarks conventionally erode **this band** away before scoring
> \cite{volpiDenseSemanticLabeling2017}; we measure what that removes."

"this band" refers back to the **8 m** band defined two sentences earlier. Volpi erode a **3 pixel
circular structuring element** — *"eroding the edges of each class in the ground truth with a 3 pixel
circular structuring element, so that evaluation is tolerant to small errors on object edges"* —
which at Vaihingen's 9 cm and Potsdam's 5 cm is **0.27 m and 0.15 m**. Ours is 30–53× wider.

`DO_NOT_ADD.md`: *"It is not a same-scale comparison for rho and must not be presented as one."*

Note the compliance grep came back clean on `Vaihingen`, `Potsdam`, `0.27`, `0.15` — the violation is
carried by the pronoun "this", not by any number. **Grep clears numbers, not referents.**

"conventionally" also generalises from one paper on one ISPRS contest, and the word Volpi attach to
"standard" is the *non*-eroded evaluation.

The manuscript correctly does **not** repeat Volpi's interpretive leap ("the boundaries are often
blurred within the 3 pixel erosion radius") — `blurred` has zero hits.

---

## 2. Substantive — a reviewer would catch these

**✓ Kang's `q` means the opposite of ours.** The manuscript writes "$q=1.0$ gives full
inverse-frequency weighting" while citing Kang. Kang, §3:

> "**Instance-balanced sampling.** …the probability $p_j^{IB}$ is given by Equation 1 with **q = 1**,
> i.e., a data point from class $j$ will be sampled **proportionally to the cardinality $n_j$**."

In Kang, q = 1 is *no rebalancing at all*. Our code computes the reciprocal —
`build_clsbal_sampler.py:80`, `inv = f[c] ** (-args.q)` — so our q = 1.0 is Kang's **q = 0**. The
sentence is true of our parameterisation and false as a reading of the paper it cites. A reviewer who
knows Kang reads q = 1.0 as "they did no rebalancing".

Two related points: Kang's headline finding is that *"instance-balanced sampling learns the best and
most generalizable representations"*, with rebalancing belonging at the **classifier** stage — not an
endorsement of rebalancing one end-to-end run. And he credits class-balanced sampling to *"(Shen
et al., 2016; Mahajan et al., 2018)"*, so he is not its origin.

**Fix:** state the exponent in our own terms without borrowing his symbol, or say explicitly "q in
the inverse-frequency sense, equivalent to Kang's q = 0".

---

**✓ Csurka did not sweep — they chose, and argued against sweeping.** The manuscript groups them with
Kohli: "the nearest precedents sweep the width and report a curve **rather than choosing one**
\cite{kohli…,csurka…}". Csurka:

> "The idea is to define a narrow band around each contour and to compute pixel accuracies in the
> given band (**r = 5 in our experiments**)."

> "To overcome this limitation, **[7] proposes** to plot the accuracy as a function of the bandwidth
> (r). **This however makes the cost of the evaluation much higher. Furthermore, having multiple
> accuracy measures makes it harder to pick the best model.**"

So they attribute the curve to Kohli ([7]), argue against it on evaluation cost and model selection,
and pick r = 5 themselves. They do plot trimap accuracy varying r in Fig. 1, but "rather than
choosing one" is false of them.

**This is subtle enough that `DO_NOT_ADD`'s own shorthand invited it.** That ruling reads *"Correct
citation for why a curve rather than a single width"* — true in the sense that Csurka articulate why
a single width is awkward (too narrow ignores context, too large converges to JI), but their own
conclusion runs the other way. **Worth amending the ruling** so the next chat is not led into the
same sentence. `DO_NOT_ADD` also warns their footnotes are dropped by the conversion — the footnote
marker on "pick the best model" is one of them, so confirm against the PDF before rewriting.

**Fix:** cite Kohli alone for sweep-and-report-a-curve; cite Csurka for why the single width is
awkward, and acknowledge they chose one.

---

**✓ The curve is named as primary evidence but not delivered.**

> "The curve over band widths is therefore the primary evidence."

`boundary_limited_error` appears **0 times** in `main.tex`, there is no such PDF in
`manuscript_v2/Figures/`, and `TODO.md` still lists that panel as pending. §3.3 gives three points
(3.7× at 1 m, 2.1× at 8 m, rates beyond 32 m), not a curve. Either land the figure or soften the
sentence.

---

**Wang 2022 does not state the pre-training recipe.** *(agent-only)* "Following the original design,
the encoder is … initialised from an ImageNet-1K Swin-Base backbone further pre-trained … on ADE20K
with a UPerNet head" attributes a recipe to the UNetFormer paper that it never gives — its only
pre-training sentence concerns ResNet18, and its Implementation Details say nothing about backbone
weights. Flagged by two agents. **Fix:** attribute the recipe to where it actually comes from, or
state it as our own choice.

**Milletari's Dice is not the shipped Dice.** *(agent-only)* Attribution of "Dice loss" is right, but
the shipped loss is linear-denominator Sørensen–Dice (`geoseg/losses/functional.py:196,200`), not
Milletari's squared $\sum p^2 + \sum g^2$; it carries a smoothing constant he never uses, is
multi-class where he says the loss "is indicated for binary segmentation tasks", and is paired with
cross-entropy and an oversampler when his selling point was removing the need for re-weighting.

**Roberts states a minimum, not a bar.** *(agent-only)* He says the block size should be "at least"
the autocorrelation range and that "larger blocks … may be required"; the manuscript reads this as a
pass/fail threshold the split "clears". The 768 m train–val gap is 1.02× the 750 m range and is not
adjudicated either way. Given `CLAUDE.md` already records that train–val at 256 m sits *inside* the
range and is where every checkpoint is selected, this one is worth getting exactly right.

---

## 3. Smaller, still worth fixing

| Claim as used | Source says | Fix |
|---|---|---|
| OpenEarthMap is "RGB satellite imagery" (twice) | "5000 aerial and satellite images" from "satellite, aircraft, and UAV" | Say "aerial and satellite". Load-bearing — the platform mismatch against Pléiades is part of the domain gap the next clause invokes. The added "dates" diversity axis is not claimed by the paper either |
| Yuan cited for "without alignment, naïve transfer can degrade the target" | Scene classification, one label per image; never studies label-space or taxonomy mismatch; attributes degradation to excessive/noisy pre-training and domain-gap width | Cite something that studies taxonomy mismatch, or drop the causal attribution |
| "the Montgomery halving convention" in a figure caption | No `\cite`, no chapter, no page | Add the citation and a chapter/page, or name the convention without the eponym |
| Taxonomy reconciliation is done "conventionally" (Lambert) | One paper, which frames manual reconciliation as *its own contribution* against a naive name-matched norm | Soften "conventionally" |

**Not a citation problem but visible to a reviewer:** the Zenodo deposit title still reads *"Code for
'Diagnosing a Label-Quality Ceiling…'"* — the claim withdrawn on 2026-07-29. The deposit is cited
correctly as a pure availability pointer with no circular use, but the title contradicts the paper.

---

## 4. Clean — checked and correct

- **Kohli complies exactly.** Cited only for sweep-and-report-a-curve. No 8 m attribution; "trimap"
  appears nowhere in `main.tex`. The manuscript states the 8 m band is "an a-priori choice, not one
  borrowed from the literature".
- **Cheng complies, twice.** Cited in the limitation direction for the ground-truth-only asymmetry,
  and his annotation-consistency width rule is explicitly declined for want of repeated annotations.
- **Reina** is held to its single permitted use (tiled inference degrades boundaries independently of
  label quality).
- **Saadeldin** is used only for the confusion pair and 55.6%, never for "not more labels".
- **Loshchilov** — $T_0$/$T_{mult}$ match the 15/45 schedule. **Liu** carries only the Swin encoder,
  with ADE20K/UPerNet correctly sent to Xiao and Zhou. **Kattenborn**'s "up to 28%" hedge survives
  intact.
- **All §1 forbidden sentences absent.** Zero hits on `Ortiz`, `11.7`, `GOES`, `annotator`, `trimap`,
  `8 pixel`, `Vaihingen`, `Potsdam`, `0.27`, `0.15`, `blurred`, `92\%`, `96\%`, `75.3`, `15.0`,
  `72.4`, `1.27`, `2.84`, `pre-regist`, `binding constraint`, `no effect`, `redundant`, `near-zero`,
  `independent block`, `annotation effort`. `Ortiz` is also absent from `Bibliography.bib`,
  `Bibliography_additions.bib` and all 22 bibitems of `main.bbl`.
- **"The second prediction we report"** is used with the falsifier attached, as required.

## 5. Unchecked

- **Deng 2009** and **Montgomery 2017** — no local conversion. Montgomery's in-text use is generic
  enough to be safe; the caption issue above is the real flag.
- A **stale arXiv Cheng conversion** sits alongside the proceedings one in `references_md/`. Delete
  the arXiv copy so no future check quotes the wrong version — `DO_NOT_ADD` already warns quotations
  must come from the proceedings.
