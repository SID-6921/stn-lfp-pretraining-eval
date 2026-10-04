"""
Non-learned baseline requested by peer review (Reviewer 2 / Devil's Advocate):
does a simple, non-deep-learning classifier on raw beta-band power already
match the pretrained/from-scratch arms? If so, the pretrained-representation
machinery isn't earning its complexity; if not, it supports there being real
value in the learned representation beyond a hand-crafted spectral feature.

For each held-out subject and calibration budget: compute each window's
mean log beta-band (12-30 Hz) power per channel (via the SAME bandpass +
Hilbert procedure used for the spectral pretraining target in eval_harness.py,
so this is the most direct apples-to-apples non-learned comparison), average
across real (non-padded) channels, and fit a single-feature logistic
regression on the calibration windows, evaluated on the test windows.

Usage: python baseline_bandpower.py results/ --out-json baseline_bandpower.json
"""
import argparse
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from dataset import load_all_runs, subjects_in
from eval_harness import beta_power_envelope, LFP_SFREQ
from few_shot_eval import CALIB_BUDGETS_SECONDS, chronological_split


def mean_beta_power_feature(windows: np.ndarray) -> np.ndarray:
    """windows: (n_windows, n_channels, win_len). Returns (n_windows,) mean
    log beta-band power per window, averaged across all real channels present
    in this subject's own data. Each window is first per-channel z-scored
    (mean 0, unit variance over time) BEFORE computing beta power -- matching
    exactly what the deep-learning arms see (LFPWindowDataset applies the same
    per-window normalization). Computing power on raw unnormalized amplitude
    instead measured absolute electrode/session scale rather than relative
    spectral content, and produced a degenerate, non-discriminative feature
    (verified: AUROC exactly 0.500 at every budget until this fix)."""
    n_windows, n_channels, win_len = windows.shape
    mu = windows.mean(axis=2, keepdims=True)
    sigma = windows.std(axis=2, keepdims=True) + 1e-6
    windows_norm = (windows - mu) / sigma
    per_channel_power = np.zeros((n_windows, n_channels), dtype=np.float32)
    for ch in range(n_channels):
        envelope = beta_power_envelope(windows_norm[:, ch, :], LFP_SFREQ)  # (n_windows, win_len)
        per_channel_power[:, ch] = envelope.mean(axis=1)
    return per_channel_power.mean(axis=1)


def run_subject_baseline(held_out: str, runs: list[dict]) -> list[dict]:
    held_runs = [r for r in runs if r["subject"] == held_out]
    assert len(held_runs) == 2, f"Expected 2 runs for {held_out}, got {len(held_runs)}"

    results = []
    for calib_seconds in CALIB_BUDGETS_SECONDS:
        calib_feat, calib_labels, test_feat, test_labels = [], [], [], []
        for run in held_runs:
            (cw, cl), (tw, tl) = chronological_split(run, calib_seconds)
            calib_feat.append(mean_beta_power_feature(cw)); calib_labels.append(cl)
            test_feat.append(mean_beta_power_feature(tw)); test_labels.append(tl)
        calib_feat = np.concatenate(calib_feat).reshape(-1, 1)
        calib_labels = np.concatenate(calib_labels)
        test_feat = np.concatenate(test_feat).reshape(-1, 1)
        test_labels = np.concatenate(test_labels)

        if len(set(calib_labels.tolist())) < 2:
            auroc = float("nan")  # calibration set doesn't include both classes at this budget
        else:
            clf = LogisticRegression()
            clf.fit(calib_feat, calib_labels)
            probs = clf.predict_proba(test_feat)[:, 1]
            auroc = roc_auc_score(test_labels, probs) if len(set(test_labels.tolist())) > 1 else float("nan")

        print(f"  {held_out} calib={calib_seconds}s: band-power-logreg AUROC={auroc:.3f}")
        results.append({"held_out": held_out, "calib_seconds": calib_seconds, "bandpower_baseline": auroc})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    runs = load_all_runs(args.results_dir)
    all_subjects = subjects_in(runs)
    print(f"{len(all_subjects)} subjects")

    all_results = []
    for held_out in all_subjects:
        all_results.extend(run_subject_baseline(held_out, runs))

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Wrote {len(all_results)} rows to {args.out_json}")

    print("\n=== Band-power logistic regression baseline: mean AUROC by budget ===")
    for budget in CALIB_BUDGETS_SECONDS:
        subset = [r for r in all_results if r["calib_seconds"] == budget]
        vals = [r["bandpower_baseline"] for r in subset if not np.isnan(r["bandpower_baseline"])]
        print(f"calib={budget:4d}s (n={len(vals)}): mean AUROC={np.mean(vals):.3f}")


if __name__ == "__main__":
    main()
