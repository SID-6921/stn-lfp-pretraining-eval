"""
Few-shot calibration framing: the clinically realistic question is not "train
on 17 patients' LFP, test on an unseen 18th with zero calibration" (the
leave-one-subject-out setup in eval_harness.py) -- it's "a newly implanted
patient has only a few minutes of their OWN labeled data; does starting from
a cross-patient-pretrained representation let you calibrate a usable decoder
faster than starting from scratch with that same tiny amount of data?"

For each held-out subject:
  1. Pretrain a backbone on the OTHER 17 subjects (same self-supervised
     objective as eval_harness.py) -- computed once per subject, reused
     across all calibration budgets tested on that subject.
  2. Take the FIRST `calib_seconds` of each of the held-out subject's own
     MedOff/MedOn runs as the "calibration set" (chronological, not
     shuffled -- this is what a clinician would actually have at calibration
     time, not a random sample from across the whole recording).
  3. Take the REMAINDER of that subject's recordings as the test set.
  4. Train two arms on the calibration set ONLY, evaluate both on the test set:
       A' from scratch (random init, no other-patient data at all)
       C' pretrained backbone (from step 1), fine-tuned on the calibration set
  5. Repeat for several calibration budgets to see whether any pretraining
     advantage (if one exists) is largest exactly where it would matter
     clinically -- the smallest calibration budgets.

Usage:
    python few_shot_eval.py results/ --shard-index 0 --num-shards 3 --out-json fewshot_shard_0.json
"""
import argparse
import copy
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from dataset import load_all_runs, subjects_in, window_run, LABEL_MAP
from eval_harness import ConvBackbone, ClsHead, pretrain_backbone, train_classifier, evaluate, DEVICE, LFPWindowDataset

CALIB_BUDGETS_SECONDS = [15, 30, 60, 120]
WINDOW_SECONDS = 2.0


def chronological_split(run: dict, calib_seconds: float):
    """Split one run's windows into (calibration, test) preserving time order --
    calibration is the FIRST calib_seconds of the recording, not a random sample,
    since that's what would actually be available at calibration time."""
    windows, labels = window_run(run, WINDOW_SECONDS)
    n_calib_windows = int(calib_seconds / WINDOW_SECONDS)
    assert n_calib_windows > 0, f"calib_seconds={calib_seconds} rounds to zero windows at {WINDOW_SECONDS}s/window"
    assert n_calib_windows < windows.shape[0], (
        f"calib_seconds={calib_seconds} ({n_calib_windows} windows) leaves no test windows "
        f"from a run with only {windows.shape[0]} windows"
    )
    return (windows[:n_calib_windows], labels[:n_calib_windows]), (windows[n_calib_windows:], labels[n_calib_windows:])


def pad_to(windows: np.ndarray, n_channels_fixed: int) -> tuple[np.ndarray, np.ndarray]:
    """Zero-pad a (n_windows, n_channels, win_len) array up to n_channels_fixed channels,
    returning the padded array and a (n_windows, n_channels_fixed) mask marking real channels --
    needed because even runs from the SAME subject can have different channel counts
    (a contact marked bad in one acquisition but not another)."""
    n_windows, n_ch, win_len = windows.shape
    assert n_ch <= n_channels_fixed, f"{n_ch} channels > n_channels_fixed={n_channels_fixed}"
    padded = np.zeros((n_windows, n_channels_fixed, win_len), dtype=np.float32)
    padded[:, :n_ch, :] = windows
    mask = np.zeros((n_windows, n_channels_fixed), dtype=np.float32)
    mask[:, :n_ch] = 1.0
    return padded, mask


def make_loader(windows: np.ndarray, mask: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool):
    mu = windows.mean(axis=2, keepdims=True)
    sigma = windows.std(axis=2, keepdims=True) + 1e-6
    windows_norm = (windows - mu) * mask[:, :, None] / sigma  # keep padded rows at exactly 0
    ds = TensorDataset(
        torch.from_numpy(windows_norm.astype(np.float32)),
        torch.from_numpy(mask),
        torch.from_numpy(labels),
    )
    return DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=shuffle)


def run_subject(held_out: str, runs: list[dict], objective: str = "raw", ref_channel: int = 0) -> list[dict]:
    train_runs = [r for r in runs if r["subject"] != held_out]
    held_runs = [r for r in runs if r["subject"] == held_out]
    assert len(held_runs) == 2, f"Expected exactly 2 runs (MedOff/MedOn) for {held_out}, got {len(held_runs)}"
    # runs from the SAME subject can still differ in channel count (a contact marked bad in
    # one acquisition but not the other) -- pad to this subject's own max, not just run[0]'s count
    n_channels = max(r["data"].shape[0] for r in held_runs)

    # Pretrain once on the other 17 subjects, reused across all calibration budgets for this held-out subject
    pretrain_n_channels = max(r["data"].shape[0] for r in train_runs)
    pretrain_ds = LFPWindowDataset(train_runs, pretrain_n_channels)
    pretrain_loader = DataLoader(pretrain_ds, batch_size=32, shuffle=True)
    backbone_pretrained_base = ConvBackbone().to(DEVICE)
    backbone_pretrained_base = pretrain_backbone(backbone_pretrained_base, pretrain_loader,
                                                  objective=objective, ref_channel=ref_channel)

    results = []
    for calib_seconds in CALIB_BUDGETS_SECONDS:
        calib_windows, calib_masks, calib_labels = [], [], []
        test_windows, test_masks, test_labels = [], [], []
        for run in held_runs:
            (cw, cl), (tw, tl) = chronological_split(run, calib_seconds)
            cw_p, cm = pad_to(cw, n_channels)
            tw_p, tm = pad_to(tw, n_channels)
            calib_windows.append(cw_p); calib_masks.append(cm); calib_labels.append(cl)
            test_windows.append(tw_p); test_masks.append(tm); test_labels.append(tl)
        calib_windows = np.concatenate(calib_windows); calib_masks = np.concatenate(calib_masks)
        calib_labels = np.concatenate(calib_labels)
        test_windows = np.concatenate(test_windows); test_masks = np.concatenate(test_masks)
        test_labels = np.concatenate(test_labels)

        calib_loader = make_loader(calib_windows, calib_masks, calib_labels, batch_size=16, shuffle=True)
        test_loader = make_loader(test_windows, test_masks, test_labels, batch_size=32, shuffle=False)

        # A': from scratch on calibration data only
        backbone_scratch = ConvBackbone().to(DEVICE)
        head_scratch = ClsHead(backbone_scratch.emb_dim).to(DEVICE)
        backbone_scratch, head_scratch = train_classifier(backbone_scratch, head_scratch, calib_loader,
                                                            freeze_backbone=False, epochs=60)
        auroc_scratch = evaluate(backbone_scratch, head_scratch, test_loader)

        # C': pretrained backbone, fine-tuned on the SAME calibration data only
        backbone_finetuned = copy.deepcopy(backbone_pretrained_base)
        head_finetuned = ClsHead(backbone_finetuned.emb_dim).to(DEVICE)
        backbone_finetuned, head_finetuned = train_classifier(backbone_finetuned, head_finetuned, calib_loader,
                                                                freeze_backbone=False, epochs=60)
        auroc_finetuned = evaluate(backbone_finetuned, head_finetuned, test_loader)

        print(f"  {held_out} calib={calib_seconds}s (n_calib_windows={len(calib_labels)}, "
              f"n_test_windows={len(test_labels)}): scratch={auroc_scratch:.3f} pretrained_ft={auroc_finetuned:.3f}")
        results.append({
            "held_out": held_out, "calib_seconds": calib_seconds,
            "n_calib_windows": len(calib_labels), "n_test_windows": len(test_labels),
            "scratch": auroc_scratch, "pretrained_finetuned": auroc_finetuned,
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--objective", choices=["raw", "spectral"], default="raw",
                     help="Pretraining SSL objective: raw-waveform masked reconstruction (original) "
                          "or beta-band power envelope reconstruction (targets the actual PD biomarker)")
    ap.add_argument("--seed", type=int, default=0,
                     help="Controls model init, training shuffling, and the random masked-chunk position "
                          "during pretraining -- run with multiple seeds to check the result isn't a lucky draw")
    ap.add_argument("--ref-channel", type=int, default=0,
                     help="Which channel the pretraining reconstruction target is computed from. "
                          "Default 0 (each subject's first right-hemisphere bipolar contact); use a "
                          "different index to check sensitivity to this choice (peer review item 5). "
                          "Every subject has >=5 real channels, so indices 0-4 are always valid.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    runs = load_all_runs(args.results_dir)
    all_subjects = subjects_in(runs)
    shard_subjects = all_subjects[args.shard_index::args.num_shards]
    print(f"{len(all_subjects)} subjects total, shard {args.shard_index}/{args.num_shards} "
          f"handles {len(shard_subjects)}: {shard_subjects}, device={DEVICE}, "
          f"objective={args.objective}, seed={args.seed}, ref_channel={args.ref_channel}")

    all_results = []
    for held_out in shard_subjects:
        print(f"\n=== Subject {held_out} ===")
        for r in run_subject(held_out, runs, objective=args.objective, ref_channel=args.ref_channel):
            r["seed"] = args.seed
            all_results.append(r)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nWrote {len(all_results)} (subject, calib_budget) results to {args.out_json}")

    if args.num_shards == 1:
        print_fewshot_summary(all_results)


def print_fewshot_summary(results: list[dict]):
    print("\n=== Few-shot calibration summary (mean AUROC by calibration budget) ===")
    for calib_seconds in CALIB_BUDGETS_SECONDS:
        subset = [r for r in results if r["calib_seconds"] == calib_seconds]
        if not subset:
            continue
        scratch_vals = [r["scratch"] for r in subset if not np.isnan(r["scratch"])]
        ft_vals = [r["pretrained_finetuned"] for r in subset if not np.isnan(r["pretrained_finetuned"])]
        print(f"calib={calib_seconds:4d}s (n={len(subset)} subjects): "
              f"from-scratch={np.mean(scratch_vals):.3f}  pretrained-finetuned={np.mean(ft_vals):.3f}  "
              f"delta={np.mean(ft_vals) - np.mean(scratch_vals):+.3f}")
    print("\nReading guide: if pretraining helps where it clinically matters, the delta column should "
          "be most positive at the SMALLEST calibration budgets and shrink as calibration data grows -- "
          "that's the pattern that would make a real clinical calibration-time argument.")


if __name__ == "__main__":
    main()
