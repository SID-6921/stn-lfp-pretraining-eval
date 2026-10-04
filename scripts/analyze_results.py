"""
Reproduces every number reported in the paper (Tables 1-3): paired Wilcoxon
signed-rank tests, paired effect sizes (Cohen's d_z), and the control-arms
confound check, computed from the result JSONs in results/.

Usage:
    python analyze_results.py [--results-dir results]
"""
import argparse
import json
import os

import numpy as np
import polars as pl
from scipy.stats import wilcoxon

pl.Config.set_tbl_formatting("ASCII_FULL")  # avoid box-drawing chars that crash non-UTF-8 consoles


def load(results_dir: str, variant: str) -> pl.DataFrame:
    # Prefer the multiseed file when one exists: it is the full seed set (e.g. 0, 1, 2),
    # not an addition to "_all" -- loading both would double-count whichever seed overlaps.
    for suffix in ["_multiseed_all", "_all", ""]:
        path = os.path.join(results_dir, f"{variant}{suffix}.json")
        if os.path.exists(path):
            with open(path) as f:
                return pl.DataFrame(json.load(f))
    raise FileNotFoundError(f"No result file found for variant {variant!r} in {results_dir}")


def cohens_dz(diff: np.ndarray) -> float:
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float("nan")


def table2_scratch_vs_pretrained(results_dir: str) -> None:
    print("=" * 70)
    print("TABLE 2: scratch vs. pretrained-finetuned, by objective/framing")
    print("=" * 70)
    for variant in ["fewshot", "fewshot_spectral", "refchan3"]:
        df = load(results_dir, variant)
        agg = df.group_by(["held_out", "calib_seconds"]).agg(
            pl.col("scratch").mean(), pl.col("pretrained_finetuned").mean()
        )
        print(f"\n--- {variant} (n_subjects={agg['held_out'].n_unique()}) ---")
        for calib in sorted(agg["calib_seconds"].unique().to_list()):
            sub = agg.filter(pl.col("calib_seconds") == calib)
            scratch = sub["scratch"].to_numpy()
            pretrained = sub["pretrained_finetuned"].to_numpy()
            diff = pretrained - scratch
            try:
                _, p = wilcoxon(pretrained, scratch)
            except ValueError:
                p = float("nan")
            n_pos = int((diff > 0).sum())
            print(
                f"  calib={calib:>4}s  mean_lift={diff.mean():+.4f}  sd={diff.std(ddof=1):.4f}  "
                f"d_z={cohens_dz(diff):+.3f}  n_improved={n_pos}/{len(diff)}  wilcoxon p={p:.4f}"
            )


def table1_control_arms(results_dir: str) -> None:
    print("\n" + "=" * 70)
    print("TABLE 1: control arms (is the gain from real cross-patient structure,")
    print("or just from pretraining on ANY data with the same power spectrum?)")
    print("=" * 70)
    multiseed_path = os.path.join(results_dir, "control_arms_multiseed_all.json")
    if os.path.exists(multiseed_path):
        with open(multiseed_path) as f:
            ca = pl.DataFrame(json.load(f))
    else:
        ca = load(results_dir, "control_arms_all_seed0")

    arms = ["A_scratch", "C_pretrained_17real", "D_pretrained_1real", "E_pretrained_17surrogate"]
    agg = ca.group_by(["held_out", "calib_seconds"]).agg([pl.col(a).mean() for a in arms])
    for calib in sorted(agg["calib_seconds"].unique().to_list()):
        sub = agg.filter(pl.col("calib_seconds") == calib)
        print(f"\n  calib={calib}s (n={sub.height} held-out subjects)")
        for a in arms:
            vals = sub[a].to_numpy()
            print(f"    {a:<28} mean={vals.mean():.4f}  sd={vals.std(ddof=1):.4f}")

        c, e, d = sub["C_pretrained_17real"].to_numpy(), sub["E_pretrained_17surrogate"].to_numpy(), sub["D_pretrained_1real"].to_numpy()
        diff_ce = c - e
        _, p_ce = wilcoxon(c, e) if diff_ce.std(ddof=1) > 0 else (None, float("nan"))
        print(f"    C vs E (real-structure test): mean_diff={diff_ce.mean():+.4f}  d_z={cohens_dz(diff_ce):+.3f}  wilcoxon p={p_ce:.4f}")
        diff_cd = c - d
        _, p_cd = wilcoxon(c, d) if diff_cd.std(ddof=1) > 0 else (None, float("nan"))
        print(f"    C vs D (breadth test):        mean_diff={diff_cd.mean():+.4f}  d_z={cohens_dz(diff_cd):+.3f}  wilcoxon p={p_cd:.4f}")


def table3_bandpower_baseline(results_dir: str) -> None:
    print("\n" + "=" * 70)
    print("TABLE 3: naive bandpower baseline vs. learned decoders (refchan3)")
    print("=" * 70)
    bp = load(results_dir, "baseline_bandpower")
    rc = load(results_dir, "refchan3")
    bp_agg = bp.group_by("calib_seconds").agg(pl.col("bandpower_baseline").mean())
    rc_agg = rc.group_by("calib_seconds").agg(
        pl.col("scratch").mean().alias("scratch_mean"),
        pl.col("pretrained_finetuned").mean().alias("pretrained_mean"),
    )
    print(bp_agg.join(rc_agg, on="calib_seconds").sort("calib_seconds"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    args = ap.parse_args()

    table1_control_arms(args.results_dir)
    table2_scratch_vs_pretrained(args.results_dir)
    table3_bandpower_baseline(args.results_dir)


if __name__ == "__main__":
    main()
