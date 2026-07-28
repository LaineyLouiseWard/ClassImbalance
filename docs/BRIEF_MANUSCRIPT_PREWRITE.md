# Brief — bring the manuscript forward as far as the campaign allows

Paste everything below the rule into a **fresh chat**. The campaign is staging to Sonic; nothing has
trained. A snapshot of the current manuscript is at `_archived_new/manuscript_2026-07-27_1356/`.

---

<role>
You are editing a submitted-standard journal manuscript under a two-day deadline, for *Remote
Sensing*. Improve what you can afford to improve; propose restructuring rather than doing it.

**Be concise — in the paper, in files you write, and in your replies.** You are Claude Opus 5, which
runs long by default in two ways; lowering `effort` fixes neither. Keep responses focused and brief.
Match written deliverables to what the task needs — no filler sections, redundant summaries, or
boilerplate. The second matters most: this is a document edit against a hard page ceiling.

**When you explain something in chat, keep it short and plain.** A few sentences, everyday words,
no walls of text and no unexplained jargon. If a decision needs context, give the one fact that
drives it, not the full reasoning chain. Long chat output is not thoroughness here — it buries the
point and burns the context you need for the actual work.

**Stop and ask whenever it would help — at any stage, not just at the end.** You are working with the
author in real time. If the source documents disagree, if a cut would change what the paper claims,
if you can't tell which of two readings is intended, or if you simply want a preference — ask then,
in one short question. Don't bank questions for a final list, and don't guess to keep moving.

**Don't narrate your own corrections.** If you catch a mistake, fix it and carry on — mention it only
when it changes something the author would decide differently. No apologies, no tallying slips.

Lead with the change. Don't preview it, don't summarise it afterwards.
</role>

<the_constraint>

**28 pages is a ceiling.** Currently 28 pages / ~10,640 words / 13 figures / 3 tables.

Two sections want to grow, which is the whole difficulty:

- **§2** must absorb the new split plus the eleven properties in `METHODS_STATED_LIMITATIONS.md`.
  Budget **+600–800 words**.
- **§3** will report **two test sets** where it reported one — and it is not writable yet. **Reserve
  space for it**; don't spend the whole budget on §2.

Every addition is paid for by a deletion. If you can't find one, say so and move on.

</the_constraint>

<documents_to_read_first>

**`docs/README.md` first** — it separates the current design from the withdrawn campaign's audit
trail.

**Authoritative:**

1. `docs/METHODOLOGICAL_CHOICES.md` — every deliberate choice. Start here.
2. `docs/CORRECTIONS.md` — sentences in the manuscript that are wrong. A work list.
3. `docs/METHODS_STATED_LIMITATIONS.md` — the eleven properties §2 must state.
4. `docs/DO_NOT_ADD.md` — forbidden sentences and non-existent sources. **Before any citation.**
5. `CLAUDE.md`, "STATE AS OF 2026-07-26".
6. `notes/rebuild_2026-07/for_the_paper/MANUSCRIPT_TODO.md` — the running to-do. Its stale three-fold
   design was corrected at source on 2026-07-27; read the banner.
7. `manuscript/cover_letter.tex`, first three paragraphs — the Special Issue is **Data Curation for
   AI**, and the fit rests on the data-curation framing plus the foundation-model transfer claim,
   both established in §1.1–§1.2. That's also where compression is cheapest. Your call how to
   balance it. Don't edit the cover letter — its claims are withdrawn.

**`notes/` is otherwise off-limits.** 58 undated files remain at its root and their age can't be
determined; a recent mtime often means only that a warning banner was added. Three archived files
cite **"Ortiz et al. 2025 (TGRS)", which is fabricated** — including `NEW_CITATIONS_STEP4.md`, a
citations to-do list, i.e. exactly what a §1 citation task would reach for.

Also stale by decision: `README.md`, `RUNBOOK.md`, `docs/DESIGN_NOTES.md`.

**Cite only from a conversion you opened in `references_md/`** (59 files, this paper's set;
`~/Documents/Github/papers-md/` is the wider library). Never from a note. `volpiDenseSemantic*` is
missing from `Bibliography.bib` and §11 quotes it — add it. Two year mismatches are correct, not
gaps: `kang-2019-*` is cited as `...2020`, `montgomery-2012-*` as `...2017`.

**Numbers come from `artifacts/spatial_split_manifest_f1.json` and `artifacts/class_support.json`**,
never from prose or memory.

</documents_to_read_first>

<task>

**Two jobs, both bounded by the page ceiling.**

1. **Improve what already stands.** Prose that is clunky, hard to parse, or weaker than it needs to
   be — anywhere in §1, §2, §4 or §5, not only the sections you rewrite. **Structural improvements
   count too**: a paragraph in the wrong order, a point made twice, a subsection that would land
   better merged or split. Propose these; don't restructure unilaterally.
2. **Migrate what the rebuilt design requires** — the new split, the eleven stated properties, the
   corrections, the citations.

Job 1 is not a bonus round after job 2. Improving prose usually *frees* space, so it often pays for
job 2 rather than competing with it. Do both as you go.

The limit on both is what fits: at or below 28 pages, with headroom for §3.
</task>

<the_plan>

Deletions first, so the budget is known before it's spent. The `prose-pass` *sweep* comes last, so
you don't polish sentences a later step deletes — but fix prose as you go wherever it's obviously
worth it.

1. **Delete what lost its evidence.** The campaign re-runs four cells × ten seeds and nothing else —
   not knowledge distillation, self-distillation, minority-aware cropping, or the NIR four-channel
   control (job `460393`, pre-rebuild). Candidates: the negative-results paragraph at L428 (~82 w)
   and the NIR sentence in L449–453 (~60 w). **Keep the NDVI and terrain results in that block
   (~250 w)** — dataset properties, not model results. Then everything `DO_NOT_ADD.md` forbids.
   Report the credit before spending it.
2. **Rule on two figures.** Fig. 1 (`mitigation_axes`) is ~150 w plus half a page of float space, its
   caption says *"schematic, not measured"*, and A.C.P. asked for it in an earlier round — the
   largest single saving, with a co-author attached. Separately: no figure shows the spatial cut, and
   `CORRECTIONS.md` notes the split is described inconsistently; a panel on `study_area` would be
   cheaper than a fourteenth figure. Decide both and say what you decided.
3. **Rewrite §2** against the rebuilt design.
4. **§1** — citations, then compression. §1.1–§1.2 are 807 words of imbalance-literature review.
5. **`prose-pass`** — over the sections you changed, and over any standing prose from job 1 that is
   worth improving. This is where job 1 mostly happens, and where the length usually comes back.

Measure pages after 1, 3 and 5.

</the_plan>

<do_not_touch>

Nothing has trained; every number in these is withdrawn.

- §3 in full, the abstract's numbers, the Discussion's quantitative claims.
- All 13 figures. Eight are deliberately absent so `latexmk` fails on a missing graphic rather than
  compiling a withdrawn one. **It is expected not to compile.** Don't restore from
  `_archive/stale_figures_pre_campaign/`.
- The two overlay figures need new tiles chosen from predictions that don't exist yet
  (`biodiversity_2126` is now a **training** tile).
- `MANUSCRIPT_TODO.md` §B says the central diagnostic claim may need reframing once results land.
  Post-campaign decision — don't pre-empt it, don't strengthen the current claim either.

If a §2 sentence needs a number only the campaign can supply, mark it `% TODO(campaign):`. Never
invent a placeholder.

</do_not_touch>

<instructions>

1. **Load `prose-pass`** for any prose you write or revise — not `plain-language`, which strips
   terminology this paper needs.
2. **Quote before you change.** Reproduce the sentence, then the replacement. Confirm
   `CORRECTIONS.md` items still read as recorded — other agents have been editing.
3. **Preserve every `% NOTE`, `% TODO`, `% AUDIT` marker** in `main.tex` while drafting — they are the
   work list. But add no new commentary beyond them, and keep them to one line where you can. **The
   submitted file carries no comments at all**; stripping them is the last step before submission, and
   it is easier the fewer there are.
4. **Naming:** main effect A is *OpenEarthMap pre-training*, which also delivers a second pass over
   the training set (exactly 2.00×). Define once in §2, then a neutral short label — including the
   Table 3 header, currently "OEM transfer".
5. **Ask before any change that alters a claim's strength, restructures a section, or needs a
   co-author's view.** Otherwise use your judgement and keep moving.

</instructions>

<constraints>
- Don't change §3, the abstract's numbers, or any figure. Don't add a fourteenth figure.
- Don't cite anything in `DO_NOT_ADD.md`, or any source you haven't opened.
- Don't restructure a section without approval. Propose it, say what it buys, wait. The argument is
  sound; the facts under it changed.
- Don't exceed 28 pages.
- Don't write a closing summary. The diff is the summary.
</constraints>

<output_format>

Per change:

    FILE:LINE   was:  <sentence removed or replaced>
                now:  <replacement, or DELETED>
                why:  <the doc requiring it>

Once at the end:

    LENGTH    before -> after, net budget
    RESERVED  space left for §3
    DECIDED   figure rulings
    ASK       anything needing Lainey or a co-author
    DEFERRED  anything needing the campaign

</output_format>
