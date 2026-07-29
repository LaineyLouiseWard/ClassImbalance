# Claim-support audit — `manuscript_v2/main.tex`, 2026-07-29

Does each citation support the claim it is attached to? (Different question from
`docs/BIB_AUDIT_2026-07-29.md`, which only asked whether the entries describe real papers correctly.)

**25 citation sites, 22 distinct keys, 28 key-checks.** Three agents plus a dedicated `DO_NOT_ADD`
compliance lane. Sources read from the repo's own `references_md/` conversions.

**18 SUPPORTS · 8 OVERREACH · 0 CONTRADICTS · 1 SOURCE UNAVAILABLE · 1 DO_NOT_ADD VIOLATION**
*(post-verification; two first-pass OVERREACH findings were withdrawn — see §6)*

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

**✓ Yuan does not study taxonomy mismatch.** The sentence blames the absence of taxonomy alignment
for transfer degradation and cites a *scene-classification* letter (one label per image) that never
studies label spaces, and that attributes degradation to **excessive and noisy pre-training** and to
domain-gap width. Anyone who opens the reference sees the mismatch at once. **This is the one a
reviewer is most likely to catch.** **Fix:** cite something that studies label-space or taxonomy
mismatch, or drop the causal attribution and state the alignment step as our own design choice.

**Wang 2022 — the recipe is true, only the attribution mispoints.** *(downgraded on verification)*
"Following the original design, the encoder is … initialised from an ImageNet-1K Swin-Base backbone
further pre-trained … on ADE20K with a UPerNet head." The verifier **loaded
`pretrain_weights/stseg_base.pth`** and confirmed it is Swin-Base + UPerNet on ADE20K — 150 classes,
ADE20K class names in `meta`, `psp_modules`/FPN decode head — and that
`weight_path='pretrain_weights/stseg_base.pth'` is the upstream GeoSeg default. So the recipe is
correct and verifiable; the only fault is "Following the original design", which points the reader at
a paper that says nothing about backbone weights. **Fix:** three words. Drop the attribution phrase
and state the provenance directly.

---

## 3. Smaller, still worth fixing

| Claim as used | Source says | Fix |
|---|---|---|
| OpenEarthMap is "RGB satellite imagery" (**once**, not twice) | "5000 aerial and satellite images" from "satellite, aircraft, and UAV" | Say "aerial and satellite". One word. *(The first pass called this load-bearing because the next clause invoked a platform mismatch; on re-reading, that clause is about class granularity, so the rationale was wrong even though the error is real.)* |
| "the Montgomery halving convention" in a figure caption | The key **is** cited two paragraphs earlier and the halving arithmetic is correct, but the phrase is a **coinage** — his Ch. 6 attaches "one-half" to interactions and regression coefficients, not to main effects | Rename it, or attach a chapter reference. Minor |
| Taxonomy reconciliation is done "conventionally" (Lambert) | One paper, which frames manual reconciliation as *its own contribution* against a naive name-matched norm | Soften "conventionally". One adverb |

### Two first-pass findings withdrawn on verification

**Milletari — pedantry, no change needed.** The code facts all check out (linear denominator at
`geoseg/losses/functional.py:196,200`, `mode='multiclass'`, `smooth=0.05`, joined to cross-entropy),
but a bare "Dice loss \cite{Milletari}" is the universal *family* citation and asserts no equivalence
of algebraic form. No competent reviewer reads it as a claim to have implemented his exact squared
denominator.

**Roberts — mostly pedantry, and the first pass misread the split.** The 768 m gap is **val–test**,
not train–val; train–val is **256 m** (`CLAUDE.md`: `train | 256 m | val | 768 m | test`). The
manuscript also already states the concession the first pass said was missing. The "substantially
larger" quote attributed to Roberts **does not exist in the paper**. Withdrawn.

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

## 5. Unchecked, and where the sources actually live

- **Deng 2009** — no local conversion. Its use (ImageNet provenance) is generic enough to be safe.
- A **stale arXiv Cheng conversion** sits alongside the proceedings one in `references_md/`. Delete
  the arXiv copy so no future check quotes the wrong version — `DO_NOT_ADD` already warns quotations
  must come from the proceedings.
- **Conversions are split across two directories** and this cost time twice. Most live in the repo's
  own `references_md/`; some (Roberts, Maxwell, Krawczyk, Saadeldin, Reina) live in
  `~/Documents/Github/papers-md/`. Search both. **Montgomery does have a conversion** —
  `references_md/montgomery-2012-ch6-2k-factorial-design.md` — contrary to what the first pass said.

---

## 6. Verification record

Every finding above was produced by one agent and then attacked by a second on a different evidence
path. That second pass changed the result in five places, so the counts in the header are the
post-verification ones:

| Finding | First pass | After verification |
|---|---|---|
| Milletari Dice form | OVERREACH | **Withdrawn — pedantry** |
| Roberts minimum-vs-bar | OVERREACH | **Withdrawn — and the first pass misread the split** |
| Wang 2022 recipe | "attributes a recipe the paper never gives" | **Recipe is true** (checkpoint inspected); only the attribution phrase is wrong |
| Xia / OpenEarthMap | wrong in two places, load-bearing | Wrong in **one** place; stated rationale was wrong |
| Montgomery | "no local conversion" | Conversion exists; key already cited; the coinage is the real issue |

Four findings were verified by me directly against the primary sources rather than delegated: the
Volpi band conflation, the Kang `q` inversion, the Csurka r = 5 misattribution, and the missing
boundary curve. Those four are the ones to act on first.
