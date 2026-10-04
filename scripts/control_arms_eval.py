"""
Peer-review-requested CRITICAL control (Devil's Advocate): does the spectral
objective's few-shot benefit require genuine CROSS-PATIENT information, or
does merely training toward a beta-band reconstruction target induce a
useful frequency-selective inductive bias regardless of whose data (or even
how structured the data) it's computed on?

Two new control arms, compared against the existing Arm A (from scratch) and
Arm C (pretrained on all 17 other subjects, real data, spectral objective):

  Arm D: pretrained on a SINGLE other subject's real data (not 17) --
         isolates whether the breadth of cross-patient data matters, or
         whether even minimal real cross-patient exposure is sufficient.
  Arm E: pretrained on PHASE-RANDOMIZED SURROGATE data from the same 17
         subjects' pretraining pool -- preserves each channel's power
         spectrum (so the beta-band content a bandpass filter would see is
         statistically similar) while destroying genuine temporal structure
         and any real disease-state information. If Arm E matches Arm C,
         the benefit is attributable to the objective's inductive bias, not
         real cross-patient transfer.

Usage:
    python control_arms_eval.py results/ --shard-index 0 --num-shards 3 --seed 0 --out-json control_shard_0_seed0.json
"""
import argparse
import copy
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import load_all_runs, subjects_in, window_run, LABEL_MAP
from eval_harness import ConvBackbone, ClsHead, pretrain_backbone, train_classifier, evaluate, DEVICE, LFPWindowDataset
from few_shot_eval import CALIB_BUDGETS_SECONDS, chronological_split, pad_to, make_loader


def phase_randomize(data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """data: (n_channels, n_samples). Returns a surrogate with the same per-channel
    power spectrum (|FFT| preserved) but randomized phase, destroying temporal
    structure and genuine cross-patient disease information while keeping the
    frequency content (including beta-band power) statistically similar."""
    n_channels, n_samples = data.shape
    out = np.zeros_like(data)
    for ch in range(n_channels):
        spectrum = np.fft.rfft(data[ch])
        magnitude = np.abs(spectrum)
        random_phase = rng.uniform(0, 2 * np.pi, size=magnitude.shape)
        random_phase[0] = 0.0  # keep DC term real
        if n_samples % 2 == 0:
            random_phase[-1] = 0.0  # keep Nyquist term real for even-length signals
        surrogate_spectrum = magnitude * np.exp(1j * random_phase)
        out[ch] = np.fft.irfft(surrogate_spectrum, n=n_samples)
    return out.astype(np.float32)


def make_surrogate_runs(runs: list[dict], seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    surrogate = []
    for run in runs:
        surrogate.append({**run, "data": phase_randomize(run["data"], rng)})
    return surrogate


def pretrain_on_runs(pretrain_runs: list[dict], objective: str = "spectral") -> ConvBackbone:
    n_channels = max(r["data"].shape[0] for r in pretrain_runs)
    ds = LFPWindowDataset(pretrain_runs, n_channels)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    backbone = ConvBackbone().to(DEVICE)
    return pretrain_backbone(backbone, loader, objective=objective)


def finetune_and_eval(backbone_base: ConvBackbone, held_runs: list[dict], n_channels: int, calib_seconds: int):
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

    backbone = copy.deepcopy(backbone_base)
    head = ClsHead(backbone.emb_dim).to(DEVICE)
    backbone, head = train_classifier(backbone, head, calib_loader, freeze_backbone=False, epochs=60)
    return evaluate(backbone, head, test_loader)


def run_subject(held_out: str, runs: list[dict], seed: int) -> list[dict]:
    all_subjects = subjects_in(runs)
    train_runs = [r for r in runs if r["subject"] != held_out]
    held_runs = [r for r in runs if r["subject"] == held_out]
    assert len(held_runs) == 2, f"Expected 2 runs for {held_out}, got {len(held_runs)}"
    n_channels = max(r["data"].shape[0] for r in held_runs)

    # Arm D source: the next subject alphabetically after held_out, circularly --
    # a single, deterministic other-subject choice (not cherry-picked per budget)
    sorted_subjects = sorted(all_subjects)
    idx = sorted_subjects.index(held_out)
    single_source_subject = sorted_subjects[(idx + 1) % len(sorted_subjects)]
    if single_source_subject == held_out:
        single_source_subject = sorted_subjects[(idx + 2) % len(sorted_subjects)]
    single_source_runs = [r for r in train_runs if r["subject"] == single_source_subject]
    assert len(single_source_runs) == 2, f"Expected 2 runs for single-source {single_source_subject}"

    backbone_scratch_init = ConvBackbone().to(DEVICE)  # untrained; Arm A trains from this fresh init per budget
    backbone_C = pretrain_on_runs(train_runs, objective="spectral")  # all 17 other subjects, real data
    backbone_D = pretrain_on_runs(single_source_runs, objective="spectral")  # 1 other subject, real data
    surrogate_runs = make_surrogate_runs(train_runs, seed=seed)
    backbone_E = pretrain_on_runs(surrogate_runs, objective="spectral")  # 17 subjects, phase-randomized

    results = []
    for calib_seconds in CALIB_BUDGETS_SECONDS:
        auroc_A = finetune_and_eval(ConvBackbone().to(DEVICE), held_runs, n_channels, calib_seconds)
        auroc_C = finetune_and_eval(backbone_C, held_runs, n_channels, calib_seconds)
        auroc_D = finetune_and_eval(backbone_D, held_runs, n_channels, calib_seconds)
        auroc_E = finetune_and_eval(backbone_E, held_runs, n_channels, calib_seconds)
        print(f"  {held_out} calib={calib_seconds}s: A(scratch)={auroc_A:.3f} "
              f"C(17-real)={auroc_C:.3f} D(1-real)={auroc_D:.3f} E(17-surrogate)={auroc_E:.3f}")
        results.append({
            "held_out": held_out, "calib_seconds": calib_seconds, "seed": seed,
            "single_source_subject": single_source_subject,
            "A_scratch": auroc_A, "C_pretrained_17real": auroc_C,
            "D_pretrained_1real": auroc_D, "E_pretrained_17surrogate": auroc_E,
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    runs = load_all_runs(args.results_dir)
    all_subjects = subjects_in(runs)
    shard_subjects = all_subjects[args.shard_index::args.num_shards]
    print(f"{len(all_subjects)} subjects total, shard {args.shard_index}/{args.num_shards} "
          f"handles {len(shard_subjects)}: {shard_subjects}, device={DEVICE}, seed={args.seed}")

    all_results = []
    for held_out in shard_subjects:
        print(f"\n=== Subject {held_out} ===")
        all_results.extend(run_subject(held_out, runs, args.seed))

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nWrote {len(all_results)} rows to {args.out_json}")


if __name__ == "__main__":
    main()
