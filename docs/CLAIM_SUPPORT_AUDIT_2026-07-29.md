# Citation and claim audit — `manuscript_v2`, 2026-07-29

## START HERE — eight blockers

Covers the complete draft plus the cover letter. All eight verified against the source or the
manuscript. Full detail and suggested rewordings in **§7**; §1–§6 are an earlier Methods/Results-only
pass kept for reasoning, and where §7 disagrees, §7 wins.

| # | Where | What is wrong |
|---|---|---|
| 1 | **Results §3.3** | Contradicts itself. Reports semi-natural wrong **27.0%** beyond 32 m, then says "Both grassland classes concentrate error near boundaries; only forest and grassland clear it deep inside." Disagrees with itself and the numbers beside it. **The paper's central finding — fix first.** |
| 2 | **Contributions** | "Because the error is not concentrated at boundaries" — unqualified. §3.3 says a pixel within 1 m is misclassified **3.7×** as often; only the grassland pair is the exception. Add the qualifier. |
| 3 | **Cover letter** | Promises foundation models. "foundation" = 2× in `cover_letter.tex`, **0× in `main.tex`**, and the scope note rules out the angle. Cut. |
| 4 | **Introduction** | The claim attached to Volpi is false — "without measuring what the band holds". He measured it and reports it (`volpi…md:210`). |
| 5 | **Methods + cover letter** | 8 m same-scale conflation survives: "erode **this band** away" / "erode **this** boundary band". Volpi's band is 0.15–0.27 m. The Introduction already says "**a** band" — match it. |
| 6 | **Methods** | "q=1.0 gives full inverse-frequency weighting" cites Kang, whose q=1 is instance-balanced, i.e. **no rebalancing**. Our code computes the reciprocal, so ours is his q=0. The sampler is right; the sentence is not. |
| 7 | **Methods** | Csurka cited as a precedent who swept the band width. They chose r=5 and argued *against* sweeping, attributing the curve to Kohli. Cite Kohli alone. |
| 8 | **Four sites** | The band-width curve is claimed in Methods, Introduction and Contributions and does not exist. `boundary_limited_error`: zero hits, no PDF in `Figures/`, still open in `TODO.md`. |

**Clean:** no fabricated reference; one benign "88"; no revival of the withdrawn ceiling framing in
the manuscript; every §1 forbidden sentence absent; every interval on a contrast, never a Test B
level; numbers reconcile across sections; abstract is 302 words with the seven narrative moves in
order and move 5 at sentence 5.

**Caveat:** the medium-priority rows in §7 are single-agent leads with no adversarial pass. The eight
above are verified.

**`docs/BIB_AUDIT_2026-07-29.md` needs no action** — its 25 fixes are already applied and
`Bibliography.bib` renders with 0 warnings.

---

## Earlier pass — Methods and Results only

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

---

# §7. Full-draft pass — the handoff list

Run once the abstract, Introduction, Contributions, Discussion, Conclusions and cover letter had all
landed. Four agents (abstract+intro, discussion+conclusions, cover letter, whole-document compliance).
**✓ = I verified it myself against the source or the manuscript; everything else is single-agent.**

**Clean, and checked:** no `Ortiz` trace anywhere; exactly one "88" in the file ("88\% of it carrying
a label", Test A coverage — neither the expert mask-error audit nor the fabricated inter-annotator
bound); no revival of the withdrawn ceiling framing in the manuscript; every §1 forbidden sentence
absent; no "pre-registered"; no 75.3 / 15.0 / 1.27 / 2.84; every 95% CI sits on a contrast, never a
Test B level; cross-document numbers reconcile; abstract is 302 words with the seven narrative moves
in order and move 5 at sentence 5; Krawczyk's taxonomy citation is exactly compliant; Maxwell's
boundary-expectation use clears all three traps; Saadeldin and Reina are both held to their permitted
uses; the COI is declared correctly.

## Blockers

**1. ✓ Results §3.3 contradicts itself, in the paper's central finding.**
> "Beyond thirty-two metres from any boundary, forest is wrong 0.7\% of the time and grassland 5.1\%,
> but **semi-natural grassland is still wrong 27.0\%** of the time… **Both grassland classes
> concentrate error near boundaries; only forest and grassland clear it deep inside.**"

If semi-natural is wrong 27% deep inside, its error is *not* concentrated at boundaries — and the
second clause excludes semi-natural anyway. The sentence disagrees with itself and with the numbers
beside it. *(A downstream agent read this as the Discussion contradicting Results; it does not — the
Discussion faithfully restates the summary line. The fault is here.)*

**2. ✓ Contributions asserts the opposite of the map-wide result.** "Because the error is not
concentrated at boundaries," — unqualified. §3.3 says a pixel within one metre is misclassified
**3.7 times** as often, and "the map follows the usual boundary pattern; the grassland pair is the
exception to it". The qualifier is what makes the claim true. Add it.

**3. ✓ The cover letter promises foundation models; the paper never mentions them.** "foundation"
appears **2× in `cover_letter.tex`, 0× in `main.tex`**. The letter tells the Guest Editors the
measurements "should apply to other architectures, including pre-trained Earth-observation foundation
models" — two of the three measurements need predicted label maps, so they are not model-free, and
the repo's own scope note says not to argue this angle. Cut it.

**4. ✓ The one claim attached to Volpi is false.** The Introduction says benchmarks erode a band
"**without measuring what the band holds**". Volpi measured exactly that — he runs four evaluation
strategies and reports both: *"By evaluating on eroded boundary ground truths, we observe a similar
behavior, but with significantly higher accuracies"* (`volpi…md:210`). Also "aerial land-cover
benchmarks" plural describes what is really one paper's choice.

**5. ✓ The 8 m same-scale conflation survives in two places.** The Introduction fix ("**a** band")
holds, but Methods still reads "conventionally erode **this band** away before scoring", and the
cover letter "erode **this** boundary band". Volpi's band is 0.15–0.27 m against our 8 m.
`DO_NOT_ADD.md` bans this by name. Make both match the Introduction.

**6. ✓ Kang's `q` still inverted** and **7. ✓ Csurka still cited as a sweeper** — both unchanged from
§2 above.

**8. ✓ The curve is claimed at four sites and does not exist.** Methods now also says the ratio is
reported "beside the curve over band widths", and the claim has spread to the Introduction and
Contributions ("as a function of distance to the nearest class boundary"). `boundary_limited_error`
has zero hits, no such PDF is in `manuscript_v2/Figures/`, and `TODO.md` still lists the panel.
Three discrete points is not a curve.

## High

| Where | Problem | Fix |
|---|---|---|
| Discussion | "Tracing recovers error where a boundary exists on the ground and the error there is one of precision" — the **unconditioned** annotation-at-boundaries claim `DO_NOT_ADD.md:21` forbids, and it contradicts §4.2's own list of blur and 0.5 m mixed pixels | Condition it, or cut |
| Discussion | "the class the model **cannot isolate**" quietly picks model failure out of the three explanations the paper declares open | Keep all three open |
| Cover letter | "Neither produces a change the design can resolve" without the ~3 pp bound reads as the forbidden "no effect" | State the bound |
| Cover letter | "all reported results are averaged over ten independently seeded runs" — false for the adjacency bound, which uses reference masks only, no model, no seeds | Qualify |
| Bibliography + letter | Zenodo deposit still titled *Code for "Diagnosing a Label-Quality Ceiling…"* — prints in the reference list via `Bibliography.bib:587`. The one place a reader is told the paper diagnoses a ceiling it withdrew | Retitle the deposit |

## Medium

- **Maxwell, first use:** invents the word "positional", which never appears in his paper, and leans
  on him for a thematic/positional separation he explicitly argues against (*"these types of
  accuracies are not necessarily separable"*).
- **Krawczyk:** the sentence calls pre-training a "data-level move"; his definition covers methods
  that "balance distributions", which pre-training does not.
- **Abstract:** opens by generalising one case study to all operational farmland maps; derives the
  four-fifths bound from only one of the two adjacency figures it needs; gives the hectare-patch level
  without its quarter-to-three-quarters seed range; omits the 3.7× the narrative insists be stated.
- **Highlight 2** says the interventions failed to shift "that error" (the class pair) when the null
  is on foreground mIoU.
- **Registration offset** is missing from an otherwise-correct rival-cause list (zero hits).
- The **"~five effective areas, one dominant"** caveat on the semi-natural half of the 46.7% appears
  nowhere, though the narrative requires it.
- **Conclusions** attribute the Irish-grassland confusion result to the literature with no `\cite`.
- **Introduction** paraphrases Saadeldin's "fertilised… low number of grass species" and "rough
  grazing" without citing him at that site.
- **Cover letter housekeeping:** omits that model weights are also undistributable; drops the "K.D.'s
  contributions were made in their capacity as a co-author" clause; opens with an unsupported
  prevalence claim that edges into method novelty.
- Carried from §2–§3: **Yuan** cited for taxonomy-mismatch degradation; **OpenEarthMap** called
  satellite imagery; **"Following the original design"** attributing a correct recipe to a paper that
  does not state it.

## Not done

No adversarial second pass on the §7 single-agent findings — the earlier one overturned five of
eleven, so treat the unmarked rows as leads, not verdicts. The eight blockers are all ✓-verified.
