# Brief — find a defensible provenance for the rho threshold

Paste below the rule into a **fresh chat**. This is a literature and reasoning task, not an
implementation task. It is short but it must be right, and it must finish before the campaign launches.

---

<role>
You are a remote-sensing methodologist deciding how a pre-registered threshold should be justified in a
paper. You care about whether a sentence can be written honestly in a methods section and survive a
reviewer. You do not write code beyond what is needed to extract a number from a published figure or
table.
</role>

<the_problem>
A paper registers this statistic:

    rho = (foreground error rate within 8 m of a ground-truth class boundary)
          / (foreground error rate beyond 8 m)

It is a rate ratio over two disjoint pixel sets, so unlike an error *share* it does not depend on how
much of the landscape lies near a boundary. The claim it adjudicates is that residual segmentation
error is concentrated at class boundaries rather than limited by model capacity or class imbalance.

The registered threshold is **rho >= 4.0 on both test sets**, judged on the lower bound of a block
bootstrap CI; dead below 2.0; weak in between.

**The problem is the provenance of 4.0.** It was set from preliminary estimates on a campaign that was
later withdrawn because of train/test leakage: rho = 3.25 (baseline model, validation), 4.77 (full
model, validation), 12.02 (full model, test). Those were restricted to pixels no training tile covered
— the non-leaking subset — but the models that produced them were still trained with leakage present.

The methods sentence would currently have to read: *"the threshold was set from preliminary estimates
on a superseded, leaking campaign."* That is honest but weak, and the author does not want to write it.

Nothing has been trained on the corrected split, so the threshold can still be changed. After the
campaign runs it cannot, and any change becomes post-hoc.
</the_problem>

<task>
Find the most defensible available provenance for this threshold, and write the methods sentence that
goes with it.
</task>

<instructions>
Quote any source before drawing on it. Do not cite anything you have not opened. Check
`~/Documents/Github/papers-md/` first for markdown conversions; if a paper is not there, use the Zotero
MCP but run `zotero_switch_library(library_id="6343594", library_type="group")` first, because research
papers are in the group library. Web search is permitted.

1. **Look for a literature-derived reference value.** Published boundary-error analyses report error
   rate as a function of distance to a ground-truth boundary — this is what trimap evaluation is. From
   such a curve or table, a rate ratio equivalent to rho can be computed directly. Search for it and, if
   you find one, compute rho from the published numbers and show your working.
   Start with: **Kohli, Ladicky & Torr 2009** (trimap, IJCV 82(3)); **Csurka, Larlus & Perronnin 2013**
   (BMVC, trimap accuracy TO/TJ and the accuracy-vs-bandwidth curve); **Cheng et al. 2021** (Boundary
   IoU, CVPR). Then widen: any semantic-segmentation or land-cover paper reporting error or accuracy
   stratified by distance to boundary. Remote sensing, medical imaging and natural-image segmentation
   are all admissible — say which domain each number comes from, since boundary sharpness differs.

2. **If a reference value exists**, assess whether it is a fair bar for this study. The dataset is
   0.5 m aerial imagery of Irish rural land cover, six classes, boundaries that are real field edges
   rather than object silhouettes. A ratio drawn from natural-image object segmentation may not
   transfer. Say so if it does not.

3. **If no reference value exists**, say that plainly, then assess the alternatives:
   - **A priori on principle.** Can a specific number be argued from what "dominated by boundary
     ambiguity" should mean, without appeal to any measurement? If so, which number, and on what
     reasoning? If the honest answer is that any specific value is arbitrary, say that.
   - **Pilot-data provenance, stated openly.** Registering a threshold from preliminary work is normal
     practice. Assess how damaging the specific provenance is here — the pilot leaked, though the
     estimates used were restricted to unseen pixels. Would a reviewer accept it? What is the strongest
     version of that sentence?
   - **Reframe so no threshold is needed.** Could the claim be adjudicated by a comparison instead of an
     absolute bar — for example rho in the full model against rho in the baseline, both from the same
     campaign, which needs no external calibration? State what that would and would not establish, and
     whether it weakens the claim.
   - **Anything else you think of.** These four are not exhaustive.

4. **Recommend one**, and write the actual methods sentence, ready to paste. Two or three sentences at
   most, in the paper's voice, with citations if any.

5. **Say what the threshold should be** under your recommendation. If it is not 4.0, say what and why.
   Note that 4.0 currently sits below the weakest full-model pilot estimate (4.77) and above the
   baseline model's (3.25), so as registered the baseline fails and the full model is at genuine risk —
   any replacement should be checked for whether it preserves that property, which is what makes the
   pre-registration binding rather than decorative.
</instructions>

<constraints>
- Review and recommend only. Change no code and no registered document; the author will apply it.
- Do not invent a citation or a number. If you cannot open a source, say "not verified".
- Do not recommend re-calibrating the threshold from the corrected campaign's results — that is the
  post-hoc trap this exists to avoid.
- Do not recommend collecting more data or running more experiments. There is a deadline.
- If your honest conclusion is that the current provenance is the best available, say so. That is a
  useful answer.
</constraints>

<output_format>
1. **Recommendation** in one line: where the threshold's justification should come from.
2. **Literature-derived rho values found**, as a table: source (author, year, domain) | the quoted
   numbers | rho computed from them | is it a fair bar here.
3. **The methods sentence**, ready to paste.
4. **The threshold** you recommend, and whether the baseline still fails it and the full model is still
   at risk.
5. **Options considered and rejected**, with the reason.
6. **Anything you could not verify.**
</output_format>
