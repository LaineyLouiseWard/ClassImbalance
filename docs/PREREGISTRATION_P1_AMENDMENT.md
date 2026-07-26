# Pre-registration amendment — §P1, dated 2026-07-26

**Status: made BEFORE any model was trained on the current split.** No accuracy, error share or
boundary statistic from the spatially blocked design exists at the time of writing. The original text
is preserved verbatim below so the change is auditable.

## Original §P1, verbatim

> The paper's claim is that residual error is **dominated** by boundary ambiguity. The statistic for
> that is the **share of total error falling within 8 m of a boundary** — not the raw boundary and
> interior rates, which are vulnerable to floor and ceiling effects.
>
> **Pre-registered:** the label-ceiling claim holds if the near-boundary error share stays above
> **65% on both test sets**. If it falls to ~40%, the claim is dead regardless of how large the ratio
> between boundary and interior rates looks.

## Why it is being amended

The near-boundary error share has a mechanical floor equal to the share of foreground **area** within
8 m of a boundary. That is a property of how finely parcelled the landscape is, not of the model. A
single absolute threshold therefore asks a different question of each test set. Measured on the
current split with per-site pixel sizes:

| test set | foreground area within 8 m | enrichment 65% implies |
|---|---|---|
| Test A — inland test strip, 294 tiles | 38.0% | 1.71x |
| Test B — upland external sites, 191 tiles | 22.2% | **2.93x** |

Test B is 60.3% semi-natural: a coarse-grained, largely single-class landscape with structurally fewer
class contacts, so a smaller fraction of its area lies near any boundary.

Against that, the largest enrichment ever observed in this project on genuinely unseen pixels — the
baseline §P1 itself names — is 2.698x:

| measurement (pixels no training tile covered) | error share | area share | enrichment |
|---|---|---|---|
| full model, validation | 71.6% | 34.6% | 2.069x |
| full model, test | 84.6% | 31.4% | 2.698x |
| baseline model, validation | 63.3% | 34.6% | 1.827x |

So as written, §P1 sets a threshold that is **comfortably clearable on Test A (1.71x) and has never
once been reached on what Test B would require (2.93x)**. It would retire the paper's headline on the
geometry of the uplands rather than on the behaviour of the model, and a failure would be
uninterpretable — which is the opposite of what a pre-registration is for.

This was not noticed when §P1 was written because the area denominators had never been measured.

## Amended §P1

> The paper's claim is that residual error is **concentrated** at class boundaries beyond what the
> distribution of ground area alone predicts. The statistic is the **boundary enrichment ratio**:
>
>     E  =  (share of total error within 8 m of a GT boundary)
>           -------------------------------------------------
>           (share of foreground AREA within 8 m of a GT boundary)
>
> E = 1.0 is chance: error is spread exactly as area is. E is scale-free, so it is comparable across
> landscapes of different granularity, which the raw share is not.
>
> **Registered denominators, measured before training and fixed:** Test A 38.0%, Test B 22.2%.
> Computed by `scripts/analysis/report_class_support.py`-style per-site distance transforms with
> `sampling=(gsd_y, gsd_x)`; inland 0.500 x 0.500 m, ireland1/2 0.515 x 0.641 / 0.634 m.
>
> **Pre-registered:** the boundary-concentration claim holds if **E >= 2.0 on both test sets** — error
> at least twice as dense near boundaries as area alone predicts. If E falls below 1.5 on either, the
> claim is dead. Between 1.5 and 2.0 it is reported as weak and the paper's framing drops from
> "dominated by" to "concentrated at".
>
> Error is measured on the ten-seed ensemble argmax, foreground pixels only (`ignore_index=0`),
> distances per-site in metres.

## Why 2.0, and the honest statement of how close it sits

2.0 is chosen as the round value expressing "twice chance". It is a real risk, not a rubber stamp:
the weakest relevant measurement in the table above is 2.069x, so a drop of only 3.5% from the current
best estimate fails it. The estimates also come from a leaky split, and although they are restricted
to unseen pixels, nothing guarantees they transfer.

Recorded explicitly so nobody can later claim the bar was set where it was known to be safe: **at the
time of registration, the project's best estimate of E on clean data is 2.07-2.70, against a threshold
of 2.0.**

## What is NOT changed

- The 8 m band, unchanged.
- The unseen-pixel baseline, unchanged — it remains the reference, not the all-pixel figure.
- The requirement that BOTH test sets clear the threshold, unchanged. Test B is not exempted.
- Two test sets reported separately, never pooled. Pooling hides a sign reversal: ireland1 is *more*
  boundary-dense than the training distribution and behaves like Test A, while ireland2 dominates the
  pooled Test B figure.

## Consequence for the manuscript

The manuscript currently quotes **92% / 96%** for the near-boundary error share. Those are
leakage-inflated all-pixel figures. The unseen-pixel values are **71.6% / 84.6%**, which is what §P1
was always about. Both the old and new numbers must be replaced by E once the campaign has run, and
the inflated pair must not appear anywhere.
