import json
from pathlib import Path
import numpy as np


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Aggregate metrics.json files from evaluation_results/")
    ap.add_argument("--eval-root", type=str, default="evaluation_results", help="Folder created by model_metrics.py")
    ap.add_argument("--out-file", type=str, default="evaluation_results/metrics_summary.txt", help="Output summary text")
    args = ap.parse_args()

    eval_root = Path(args.eval_root)
    metrics_files = sorted(eval_root.rglob("metrics.json"))
    if not metrics_files:
        raise FileNotFoundError(f"No metrics.json files found under {eval_root}. Run model_metrics.py first.")

    rows = []
    for mf in metrics_files:
        with open(mf, "r", encoding="utf-8") as f:
            m = json.load(f)
        rows.append(
            {
                "run_dir": str(mf.parent),
                "checkpoint": m.get("checkpoint", ""),
                "model": m.get("model_name", ""),
                "OA": float(m.get("OA", np.nan)),
                "mIoU_excluding_bg": float(m.get("mIoU_excluding_bg", np.nan)),
                "mF1_excluding_bg": float(m.get("mF1_excluding_bg", np.nan)),
            }
        )

    def write_block(fh, key):
        fh.write(f"\n{key} (sorted desc)\n")
        fh.write("=" * 70 + "\n")
        sorted_rows = sorted(rows, key=lambda r: (r[key] if r[key] == r[key] else -1), reverse=True)
        for r in sorted_rows:
            fh.write(f"{r[key]:.4f}  | {r['model']:<10} | {Path(r['checkpoint']).name}\n")

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(f"Found {len(rows)} evaluated checkpoints under: {eval_root}\n")
        write_block(fh, "mIoU_excluding_bg")
        write_block(fh, "mF1_excluding_bg")
        write_block(fh, "OA")

        fh.write("\n\nFull listing:\n")
        fh.write("=" * 70 + "\n")
        for r in rows:
            fh.write(
                f"{r['mIoU_excluding_bg']:.4f} mIoU | {r['mF1_excluding_bg']:.4f} mF1 | {r['OA']:.4f} OA"
                f" | {r['model']} | {Path(r['checkpoint']).name}\n"
            )

    print(f"✓ Wrote summary: {out_path}")


if __name__ == "__main__":
    main()
