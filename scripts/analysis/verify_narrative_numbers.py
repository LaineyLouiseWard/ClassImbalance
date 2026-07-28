#!/usr/bin/env python3
"""
Every number the narrative quotes, checked against the committed artifact that produced it.

WHY THIS EXISTS. Over one working session the reported numbers came from a mix of committed scripts,
SSH one-liners and throwaway Python. Several were right and none was reproducible, so a later reader
could not tell which were which. This is the ledger: one row per quoted number, naming the artifact,
the accessor, and the command that regenerates it. If a row cannot be resolved, the number does not
belong in the paper.

Run it before quoting anything, and after any recomputation:

    PYTHONPATH=. python scripts/analysis/verify_narrative_numbers.py

Exit 0 means every quoted number reproduces. Exit 1 names the ones that do not.

ADDING A NUMBER. Put it in NUMBERS with the artifact it comes from and the command that makes that
artifact. A number with no row here is a number nobody can defend, and reviewers ask.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        # `artifacts` and `scripts`, NOT `data`: data/ is gitignored, so keying on it made every
        # one of these scripts raise "repo root not found" in a fresh clone -- including the ledger
        # a reviewer would run to check the paper's numbers.
        if (parent / "artifacts").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError("repo root not found")


ROOT = find_repo_root()
SEEDS = list(range(42, 52))


def load(rel: str):
    p = ROOT / rel
    if not p.exists():
        raise FileNotFoundError(rel)
    return np.load(p) if p.suffix == ".npy" else json.loads(p.read_text())


# --- accessors, one per shape of artifact -----------------------------------

def pair_field(field, pair="Grassland-Seminatural"):
    def f():
        d = load("artifacts/narrative_numbers_test.json")
        return next(r[field] for r in d["pair_ratios"] if r["pair"] == pair)
    return f


def conf_summary(cls, field):
    def f():
        return load("artifacts/narrative_numbers_test.json")["confusion_summary"][cls][field]
    return f


def pair_rate(key):
    def f():
        return load("artifacts/narrative_numbers_test.json")["confusion_summary"]["pair_rates"][key]
    return f


def sampler(cls, field):
    def f():
        return load("artifacts/narrative_numbers_test.json")["sampler_effect"][cls][field]
    return f


def rho_mean(tag, band, cell="stage1_baseline"):
    def f():
        v = [load(f"analysis/{tag}/rho_test_{cell}_seed{s}_band{band}.json")["pooled"]["rho"]
             for s in SEEDS]
        return float(np.mean(v))
    return f


def seed_cv(tag, band, which, rate):
    def f():
        d = load(f"artifacts/seedcontrol_{tag}_band{band}.json")["seed_control"]
        return d[which][f"{rate}_cv_pct"]
    return f


def interior(cls, split="test", cell="stage1_baseline", field="interior_rate"):
    def f():
        d = load(f"analysis/interior/interior_{split}_{cell}.json")["per_seed"]
        return 100 * float(np.mean([d[str(s)][cls][field] for s in SEEDS]))
    return f


def interior_n(cls, split="test", cell="stage1_baseline"):
    def f():
        return load(f"analysis/interior/interior_{split}_{cell}.json")["per_seed"]["42"][cls]["interior_n"]
    return f


def geom(pair, field, cell="stage1_baseline"):
    def f():
        rows = load(f"analysis/pair_geometry/pair_geometry_test_{cell}.json")["pairs"][pair]
        return float(np.mean([r[field] for r in rows]))
    return f


def adjacency(field, a="Grassland", b="Seminatural"):
    def f():
        d = load("artifacts/class_adjacency_test.json")
        return next(r[field] for r in d["pairs"] if r["a"] == a and r["b"] == b)
    return f


def contrast(effect, cls, field):
    def f():
        return load("artifacts/narrative_numbers_test.json")["per_class_contrasts"][effect][cls][field]
    return f


def support(split, cls):
    def f():
        return load("artifacts/class_support.json")[0]["splits"][split][cls]["pixels"]
    return f


def rho_rate(tag, band, which, cell="stage1_baseline"):
    def f():
        v = [load(f"analysis/{tag}/rho_test_{cell}_seed{s}_band{band}.json")["pooled"][which]
             for s in SEEDS]
        return 100 * float(np.mean(v))
    return f


def rho_cv_all_runs(tag, band, which):
    """CV of the rate over all forty runs -- four cells x ten seeds -- which is what the stability
    claim in the narrative is about, not the four-cell spread the registered arm uses."""
    cells = ["stage1_baseline", "stage2b_oem_finetune", "stage_sampler_only", "stage3_clsbal"]
    def f():
        v = [load(f"analysis/{tag}/rho_test_{c}_seed{s}_band{band}.json")["pooled"][which]
             for c in cells for s in SEEDS]
        a = np.array(v)
        return 100 * float(a.std(ddof=1) / a.mean())
    return f


def conf_total(field):
    def f():
        return load("analysis/confusion/confusion_test_stage1_baseline.json")[field]
    return f


# --- the ledger -------------------------------------------------------------
# quoted value, tolerance, accessor, and the command that regenerates the artifact.

MK_NN = ("PYTHONPATH=. python scripts/analysis/narrative_numbers.py --confusion-dir "
         "analysis/confusion --metrics-dir analysis/metrics/test --split test "
         "--out artifacts/narrative_numbers_test.json")
MK_CONF = ("PYTHONPATH=. python scripts/analysis/pooled_confusion.py --softmax-root <cluster> "
           "--split test --cell <cell> --out-dir analysis/confusion")
MK_RHO = ("PYTHONPATH=. python scripts/analysis/boundary_rate_ratio.py --split-root <dedup|split_f1> "
          "--split test --softmax-root <cluster> --cell <cell> --seeds <seed> --band-m <band>")
MK_SC = ("PYTHONPATH=. python scripts/analysis/narrative_numbers.py --rho-dir analysis/rho_dedup "
         "--split test --band-m <band> --out artifacts/seedcontrol_dedup_band<band>.json")
MK_INT = ("PYTHONPATH=. python scripts/analysis/interior_error_by_class.py --softmax-root <cluster> "
          "--split test --cell <cell> --out-dir analysis/interior")
MK_GEOM = ("PYTHONPATH=. python scripts/analysis/pair_error_geometry.py --softmax-root <cluster> "
           "--split test --cell <cell> --out-dir analysis/pair_geometry")
MK_ADJ = ("PYTHONPATH=. python scripts/analysis/class_adjacency.py --split test --confusion "
          "analysis/confusion/confusion_test_stage1_baseline.npy")
MK_SUP = "PYTHONPATH=. python scripts/analysis/report_class_support.py"

# Human names for the groups in docs/NUMBERS.md, so the headings are readable.
CMD_NAMES = {
    MK_NN: "Class-pair ratios, confusion summary, sampler effect, per-class contrasts",
    MK_CONF: "Pooled confusion matrices",
    MK_RHO: "rho and the near/interior error rates",
    MK_SC: "The seed-only control",
    MK_INT: "Interior error rate by class",
    MK_GEOM: "Pair error geometry: distance and component size",
    MK_ADJ: "Class adjacency and the contact null",
    MK_SUP: "Class support",
}

NUMBERS = [
    # (label, quoted, tol, accessor, regenerating command)
    ("pair share of foreground error, %", 46.68, 0.02,
     lambda: 100 * pair_field("share_of_error")(), MK_NN),
    ("pair co-area ratio", 2.102, 0.005, pair_field("ratio"), MK_NN),
    ("pair error pixels", 12899256, 1, pair_field("pixels"), MK_NN),
    ("foreground errors, Test A baseline", 27631446, 1, conf_total("foreground_errors"), MK_CONF),
    ("foreground pixels scored (10 seeds)", 208096710, 1, conf_total("foreground_px"), MK_CONF),

    ("semi-natural predicted vs reference area, %", 11.4, 0.1,
     conf_summary("Seminatural", "predicted_over_reference_pct"), MK_NN),
    ("semi-natural recall, %", 56.03, 0.02,
     lambda: 100 * conf_summary("Seminatural", "recall")(), MK_NN),
    ("semi-natural precision, %", 50.28, 0.02,
     lambda: 100 * conf_summary("Seminatural", "precision")(), MK_NN),
    ("semi-natural called grassland, %", 39.15, 0.02,
     lambda: 100 * pair_rate("Seminatural->Grassland")(), MK_NN),
    ("grassland called semi-natural, %", 4.82, 0.02,
     lambda: 100 * pair_rate("Grassland->Seminatural")(), MK_NN),
    ("sampler: extra predicted semi-natural px", 1785429, 1,
     sampler("Seminatural", "extra_predicted"), MK_NN),
    ("sampler: share of extra that was correct, %", 3.0, 0.1,
     lambda: 100 * sampler("Seminatural", "share_of_extra_that_was_correct")(), MK_NN),
    ("sampler: semi-natural recall after, %", 56.40, 0.02,
     lambda: 100 * sampler("Seminatural", "recall_full")(), MK_NN),

    ("rho, dedup subset, 8 m", 2.137, 0.002, rho_mean("rho_dedup", "8.0"), MK_RHO),
    ("rho, dedup subset, 1 m", 3.692, 0.002, rho_mean("rho_dedup", "1.0"), MK_RHO),
    ("seed control 8 m: near CV across cells, %", 3.76, 0.02,
     seed_cv("dedup", "8.0", "across_cells_within_seed", "near"), MK_SC),
    ("seed control 8 m: near CV across seeds, %", 3.70, 0.02,
     seed_cv("dedup", "8.0", "across_seeds_within_cell", "near"), MK_SC),
    ("seed control 8 m: interior CV across cells, %", 13.89, 0.02,
     seed_cv("dedup", "8.0", "across_cells_within_seed", "interior"), MK_SC),
    ("seed control 8 m: interior CV across seeds, %", 14.21, 0.02,
     seed_cv("dedup", "8.0", "across_seeds_within_cell", "interior"), MK_SC),

    ("interior error, semi-natural, Test A, %", 26.98, 0.02, interior("Seminatural"), MK_INT),
    ("interior error, cropland, Test A, %", 76.97, 0.02, interior("Cropland"), MK_INT),
    ("interior error, grassland, Test A, %", 5.15, 0.02, interior("Grassland"), MK_INT),
    ("interior error, forest, Test A, %", 0.75, 0.02, interior("Forest"), MK_INT),
    ("interior error, semi-natural, Test B, %", 38.73, 0.02,
     interior("Seminatural", split="external_test"), MK_INT),
    ("interior support, semi-natural, Test A, px", 1704635, 1, interior_n("Seminatural"), MK_INT),
    ("interior support, settlement, Test A, px", 2907, 1, interior_n("Settlement"), MK_INT),

    ("S->G guarded beyond 8 m, %", 68.5, 0.1,
     geom("Seminatural->Grassland", "guarded_beyond8_pct"), MK_GEOM),
    ("G->S guarded beyond 8 m, %", 75.3, 0.1,
     geom("Grassland->Seminatural", "guarded_beyond8_pct"), MK_GEOM),
    ("F->G guarded beyond 8 m, %", 4.8, 0.1,
     geom("Forest->Grassland", "guarded_beyond8_pct"), MK_GEOM),
    ("S->G mass in components >1 ha, %", 56.2, 0.1,
     lambda: 100 * geom("Seminatural->Grassland", "share_mass_in_components_over_10000m2")(), MK_GEOM),
    ("G->S mass in components >1 ha, %", 53.8, 0.1,
     lambda: 100 * geom("Grassland->Seminatural", "share_mass_in_components_over_10000m2")(), MK_GEOM),
    ("F->G mass in components >0.1 ha, %", 10.4, 0.1,
     lambda: 100 * geom("Forest->Grassland", "share_mass_in_components_over_1000m2")(), MK_GEOM),
    ("F->G median component, m2", 1.0, 0.1,
     geom("Forest->Grassland", "median_component_m2"), MK_GEOM),

    # rho on ALL 294 tiles -- the narrative quotes these beside the deduplicated pair, so both need
    # rows or a reader cannot tell which population a figure belongs to.
    ("rho, all tiles, 8 m", 2.282, 0.003, rho_mean("rho_full", "8.0"), MK_RHO),
    ("rho, all tiles, 1 m", 3.848, 0.003, rho_mean("rho_full", "1.0"), MK_RHO),
    ("near-boundary rate, all tiles, 8 m, %", 19.63, 0.02,
     rho_rate("rho_full", "8.0", "rate_near"), MK_RHO),
    ("near-boundary rate, all tiles, 1 m, %", 41.18, 0.02,
     rho_rate("rho_full", "1.0", "rate_near"), MK_RHO),
    ("near-boundary rate CV over 40 runs, 8 m, %", 3.5, 0.3,
     rho_cv_all_runs("rho_full", "8.0", "rate_near"), MK_RHO),
    ("near-boundary rate CV over 40 runs, 1 m, %", 1.9, 0.3,
     rho_cv_all_runs("rho_full", "1.0", "rate_near"), MK_RHO),

    # Adjacency bounds, superseded as a headline by the direct distance figures but still quoted.
    ("S->G bound: share not at a shared edge, %", 84.4, 0.1,
     lambda: 100 * adjacency("min_share_far_from_b", a="Seminatural", b="Grassland")(), MK_ADJ),
    ("G->S bound: share not at a shared edge, %", 87.6, 0.1,
     lambda: 100 * adjacency("min_share_far_from_b", a="Grassland", b="Seminatural")(), MK_ADJ),
    ("C->G bound: share not at a shared edge, %", 77.6, 0.1,
     lambda: 100 * adjacency("min_share_far_from_b", a="Cropland", b="Grassland")(), MK_ADJ),
    ("contact-null ratio, grassland pair", 29.28, 0.05, adjacency("contact_null_ratio"), MK_ADJ),
    ("total foreground contacts", 412641, 1,
     lambda: load("artifacts/class_adjacency_test.json")["total_fg_contacts"], MK_ADJ),

    # Unguarded distance shares, quoted beside the guarded ones to show the size of the artefact.
    ("S->G beyond 8 m, unguarded, %", 68.4, 0.1,
     geom("Seminatural->Grassland", "beyond8_pct"), MK_GEOM),
    ("G->S beyond 8 m, unguarded, %", 76.2, 0.1,
     geom("Grassland->Seminatural", "beyond8_pct"), MK_GEOM),
    ("S->G mass in components >0.1 ha, %", 90.6, 0.1,
     lambda: 100 * geom("Seminatural->Grassland", "share_mass_in_components_over_1000m2")(), MK_GEOM),
    ("G->S mass in components >0.1 ha, %", 92.9, 0.1,
     lambda: 100 * geom("Grassland->Seminatural", "share_mass_in_components_over_1000m2")(), MK_GEOM),
    ("S->G error px per seed", 570425, 200, geom("Seminatural->Grassland", "n"), MK_GEOM),
    ("G->S error px per seed", 719500, 200, geom("Grassland->Seminatural", "n"), MK_GEOM),

    # Test B. Thin by design -- the narrative leads on Test A -- but every figure it does quote.
    ("interior error, forest, Test B, %", 23.48, 0.02,
     interior("Forest", split="external_test"), MK_INT),
    ("interior error, grassland, Test B, %", 2.28, 0.02,
     interior("Grassland", split="external_test"), MK_INT),
    ("interior error, cropland, Test B, %", 6.38, 0.02,
     interior("Cropland", split="external_test"), MK_INT),
    ("interior support, semi-natural, Test B, px", 10090826, 1,
     interior_n("Seminatural", split="external_test"), MK_INT),
    ("interior support, settlement, Test B, px", 855, 1,
     interior_n("Settlement", split="external_test"), MK_INT),

    ("grassland pair contact share, %", 1.594, 0.005,
     lambda: 100 * adjacency("contact_share")(), MK_ADJ),
    ("semi-natural within 8 m of grassland, %", 6.13, 0.02,
     lambda: 100 * adjacency("share_of_a_within_8m_of_b", a="Seminatural", b="Grassland")(), MK_ADJ),

    ("OEM pre-training, mIoU, pp", -0.37, 0.02,
     contrast("oem_pretraining", "foreground_mIoU", "mean"), MK_NN),
    ("sampler, mIoU, pp", 0.19, 0.02,
     contrast("class_balanced_sampler", "foreground_mIoU", "mean"), MK_NN),
    ("OEM pre-training, mIoU, CI low", -3.67, 0.02,
     contrast("oem_pretraining", "foreground_mIoU", "ci_low"), MK_NN),
    ("OEM pre-training, mIoU, CI high", 2.94, 0.02,
     contrast("oem_pretraining", "foreground_mIoU", "ci_high"), MK_NN),

    ("train grassland pixels", 193670068, 1, support("train", "Grassland"), MK_SUP),
    ("train semi-natural pixels", 11780653, 1, support("train", "Seminatural"), MK_SUP),
    ("Test A foreground, all 294 tiles", 69558747, 1,
     lambda: sum(support("test", c)() for c in
                 ("Forest", "Grassland", "Cropland", "Settlement", "Seminatural")), MK_SUP),
]


def emit_markdown() -> int:
    """Write docs/NUMBERS.md from the ledger, so the map cannot drift from the checker.

    The manuscript quotes these; this says for each one which committed file holds it and which
    command rebuilds that file. Regenerate with `--markdown` after adding a row.
    """
    by_cmd: dict = {}
    for label, quoted, tol, fn, cmd in NUMBERS:
        try:
            got = f"{float(fn()):.6g}"
        except Exception:                                # noqa: BLE001
            got = "UNRESOLVED"
        by_cmd.setdefault(cmd, []).append((label, quoted, got))

    out = ["# Where the paper's numbers come from", "",
           "**Generated — do not hand-edit.** Rebuild with:", "",
           "    PYTHONPATH=. python scripts/analysis/verify_narrative_numbers.py --markdown", "",
           "Every number the write-up quotes has a row here, the committed artifact that holds it, and",
           "the command that regenerates that artifact. To check them all at once, run the same script",
           "with no arguments; it exits non-zero if any has drifted.", "",
           f"{len(NUMBERS)} numbers, grouped by the command that produces them.", ""]
    for i, (cmd, rows) in enumerate(sorted(by_cmd.items()), 1):
        out += [f"## {i}. {CMD_NAMES.get(cmd, 'Other')}", "",
                "```", cmd, "```", "",
                "| number | value |", "|---|---|"]
        for label, quoted, got in rows:
            out.append(f"| {label} | {got} |")
        out.append("")
    out += ["## Not in this ledger, and why", "",
            "- **Test B** is covered only where the narrative quotes it. If more Test B figures enter",
            "  the paper, they need rows.",
            "- **Per-tile illustrations** (individual parcels in the qualitative panel) come from softmax",
            "  dumps that are not committed, being ~50 MB. They are illustrative; the component",
            "  statistics carry the claim and those are here.", ""]
    (ROOT / "docs/NUMBERS.md").write_text("\n".join(out))
    print(f"wrote docs/NUMBERS.md ({len(NUMBERS)} numbers)")
    return 0


def main() -> int:
    if "--markdown" in sys.argv:
        return emit_markdown()
    ok = True
    missing = []
    print(f"{'number':52s}{'quoted':>14s}{'found':>14s}   status")
    for label, quoted, tol, fn, cmd in NUMBERS:
        try:
            got = float(fn())
        except FileNotFoundError as e:
            print(f"{label:52s}{quoted:>14}{'MISSING':>14}   artifact absent: {e}")
            missing.append((label, cmd))
            ok = False
            continue
        except Exception as e:                       # noqa: BLE001 - report, do not mask
            print(f"{label:52s}{quoted:>14}{'ERROR':>14}   {type(e).__name__}: {e}")
            missing.append((label, cmd))
            ok = False
            continue
        good = abs(got - quoted) <= tol
        ok &= good
        print(f"{label:52s}{quoted:>14.4g}{got:>14.4g}   {'ok' if good else 'MISMATCH'}")

    if missing:
        print("\nTo regenerate the missing artifacts:")
        for cmd in dict.fromkeys(c for _, c in missing):
            print(f"  {cmd}")

    print(f"\n{len(NUMBERS)} numbers checked. "
          + ("ALL REPRODUCE" if ok else "SOME DO NOT REPRODUCE -- do not quote them"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
