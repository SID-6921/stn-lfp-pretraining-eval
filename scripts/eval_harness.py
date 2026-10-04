"""
Leave-one-subject-out comparison: does pretraining a representation across
OTHER patients' LFP improve cross-patient transfer of a medication-state
(symptom-biomarker proxy) decoder, versus training from scratch on the
training subjects alone? This is the actual question Idea 3 asks.

Both arms see exactly the same training subjects and the same held-out test
subject per fold -- the only difference is whether the backbone was
pretrained (self-supervised, masked-timestep reconstruction) before the
classification head is trained, or trained end-to-end from random init.

Usage:
    python eval_harness.py results/
"""
import argparse
import copy
import json
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt, hilbert
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from dataset import load_all_runs, subjects_in, LFPWindowDataset

BETA_BAND_HZ = (12.0, 30.0)  # the PD STN-LFP biomarker band adaptive DBS systems actually threshold on
SPECTRAL_TARGET_POINTS = 8   # coarse temporal profile of beta power, not full-resolution reconstruction
LFP_SFREQ = 250.0            # fixed resample rate used by extract_lfp.py for every saved run

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvBackbone(nn.Module):
    """Small per-channel 1D-CNN encoder + masked mean-pool across channels.
    Channel-count-agnostic by construction (via the mask), so it can be
    pretrained on subjects with 5 contacts and evaluated on subjects with 6."""

    def __init__(self, emb_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=25, stride=2, padding=12), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=11, stride=2, padding=5), nn.ReLU(),
            nn.Conv1d(32, emb_dim, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.emb_dim = emb_dim

    def forward(self, x, mask):
        # x: (B, C, T), mask: (B, C)
        B, C, T = x.shape
        x = x.reshape(B * C, 1, T)
        feat = self.conv(x).reshape(B, C, self.emb_dim)  # (B, C, emb_dim)
        mask_exp = mask.unsqueeze(-1)  # (B, C, 1)
        summed = (feat * mask_exp).sum(dim=1)
        counts = mask_exp.sum(dim=1).clamp(min=1.0)
        return summed / counts  # masked mean over real (non-padded) channels


class ReconHead(nn.Module):
    """Self-supervised pretraining head: reconstruct a randomly masked chunk
    of each channel's waveform from the pooled embedding (forces the
    embedding to carry real signal content, not just channel count)."""

    def __init__(self, emb_dim: int, win_len: int):
        super().__init__()
        self.proj = nn.Linear(emb_dim, win_len)

    def forward(self, pooled_emb):
        return self.proj(pooled_emb)


class SpectralReconHead(nn.Module):
    """Self-supervised pretraining head: predict a coarse beta-band power
    envelope profile (SPECTRAL_TARGET_POINTS values) of the masked chunk,
    instead of raw waveform samples. Unlike ReconHead, this directly targets
    the oscillatory feature (beta-band power) that is the actual clinical
    biomarker for adaptive DBS, which raw-waveform reconstruction can satisfy
    without ever representing."""

    def __init__(self, emb_dim: int, n_target: int = SPECTRAL_TARGET_POINTS):
        super().__init__()
        self.proj = nn.Linear(emb_dim, n_target)

    def forward(self, pooled_emb):
        return self.proj(pooled_emb)


def beta_power_envelope(x_np: np.ndarray, sfreq: float) -> np.ndarray:
    """x_np: (B, T) channel-0 waveform (numpy, CPU). Returns (B, T) beta-band
    power envelope via bandpass filter + Hilbert transform, log-scaled since
    power has heavy-tailed dynamic range. Filtering the FULL window (not just
    the masked chunk) avoids edge artifacts at the chunk boundary."""
    nyq = sfreq / 2.0
    b, a = butter(4, [BETA_BAND_HZ[0] / nyq, BETA_BAND_HZ[1] / nyq], btype="band")
    filtered = filtfilt(b, a, x_np, axis=-1)
    envelope = np.abs(hilbert(filtered, axis=-1))
    return np.log1p(envelope)


def downsample_mean(x: np.ndarray, n_points: int) -> np.ndarray:
    """(B, T) -> (B, n_points) by averaging T//n_points-sized chunks."""
    B, T = x.shape
    chunk = T // n_points
    trimmed = x[:, : chunk * n_points]
    return trimmed.reshape(B, n_points, chunk).mean(axis=2)


class ClsHead(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.fc = nn.Linear(emb_dim, 1)

    def forward(self, pooled_emb):
        return self.fc(pooled_emb).squeeze(-1)


def pretrain_backbone(backbone: ConvBackbone, loader: DataLoader, epochs: int = 15,
                       objective: str = "raw", ref_channel: int = 0) -> ConvBackbone:
    """objective="raw": reconstruct the masked chunk's raw waveform (original recipe).
    objective="spectral": reconstruct the masked chunk's beta-band power envelope instead --
    directly targets the oscillatory feature that's the actual clinical biomarker, which raw
    reconstruction can satisfy via generic waveform morphology without ever representing.
    ref_channel selects which channel the reconstruction target is computed from (default 0,
    i.e. each subject's first right-hemisphere bipolar contact by construction of the montage
    mapping in extract_lfp.py) -- parameterized for the channel-choice sensitivity check in
    idea3_peer_review_round2.md item 5, not varied in the main results."""
    assert objective in ("raw", "spectral"), objective
    win_len = next(iter(loader))[0].shape[-1]
    # ref_channel must be a REAL channel for every window in the pool, never a zero-padded slot --
    # LFPWindowDataset pads channel counts up to the batch-wide max, so an index beyond a given
    # subject's own channel count would silently make the reconstruction target all zeros.
    # Checked against the full dataset's mask array, not just one sampled batch.
    assert bool(loader.dataset.masks[:, ref_channel].all()), (
        f"ref_channel={ref_channel} is zero-padded (not a real channel) for at least one window in "
        f"this pretraining pool -- pick an index within every subject's real channel count"
    )
    if objective == "raw":
        head = ReconHead(backbone.emb_dim, win_len).to(DEVICE)
    else:
        head = SpectralReconHead(backbone.emb_dim, SPECTRAL_TARGET_POINTS).to(DEVICE)
    opt = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=1e-3)
    backbone.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_seen = 0
        for x, mask, _ in loader:
            x, mask = x.to(DEVICE), mask.to(DEVICE)
            # mask out a random 25% time chunk as the reconstruction target (per-window, not per-channel,
            # so the task genuinely requires cross-channel context rather than copying the input)
            B, C, T = x.shape
            chunk_len = T // 4
            start = np.random.randint(0, T - chunk_len)
            x_masked = x.clone()
            x_masked[:, :, start:start + chunk_len] = 0.0

            if objective == "raw":
                target = x[:, ref_channel, start:start + chunk_len]
            else:
                # compute beta-band power envelope from the UNMASKED reference-channel waveform
                # (filtering needs the full window to avoid edge artifacts at the chunk boundary),
                # then slice out the masked region and downsample to a coarse temporal profile
                ch0_np = x[:, ref_channel, :].detach().cpu().numpy()
                envelope = beta_power_envelope(ch0_np, LFP_SFREQ)
                envelope_chunk = envelope[:, start:start + chunk_len]
                target_np = downsample_mean(envelope_chunk, SPECTRAL_TARGET_POINTS)
                target = torch.from_numpy(target_np.astype(np.float32)).to(DEVICE)

            pooled = backbone(x_masked, mask)
            pred = head(pooled)
            if objective == "raw":
                pred = pred[:, start:start + chunk_len]
            loss = nn.functional.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * B
            n_seen += B
        print(f"  pretrain epoch {epoch+1}/{epochs}  {objective}_mse={total_loss / n_seen:.4f}")
    return backbone


def train_classifier(backbone: ConvBackbone, head: ClsHead, loader: DataLoader,
                      freeze_backbone: bool, epochs: int = 20) -> tuple[ConvBackbone, ClsHead]:
    params = list(head.parameters()) if freeze_backbone else list(backbone.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    backbone.train(mode=not freeze_backbone)
    head.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x, mask, y in loader:
            x, mask, y = x.to(DEVICE), mask.to(DEVICE), y.float().to(DEVICE)
            with torch.set_grad_enabled(not freeze_backbone):
                pooled = backbone(x, mask)
            if freeze_backbone:
                pooled = pooled.detach()
            logits = head(pooled)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.shape[0]
    return backbone, head


@torch.no_grad()
def evaluate(backbone: ConvBackbone, head: ClsHead, loader: DataLoader) -> float:
    backbone.eval()
    head.eval()
    all_probs, all_labels = [], []
    for x, mask, y in loader:
        x, mask = x.to(DEVICE), mask.to(DEVICE)
        pooled = backbone(x, mask)
        probs = torch.sigmoid(head(pooled)).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    if len(set(all_labels.tolist())) < 2:
        return float("nan")  # degenerate fold (held-out subject has only one label) -- can't compute AUROC
    return roc_auc_score(all_labels, all_probs)


def run_fold(train_runs, test_runs, n_channels_fixed: int, batch_size: int = 32):
    train_ds = LFPWindowDataset(train_runs, n_channels_fixed)
    test_ds = LFPWindowDataset(test_runs, n_channels_fixed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Arm A: from scratch
    backbone_scratch = ConvBackbone().to(DEVICE)
    head_scratch = ClsHead(backbone_scratch.emb_dim).to(DEVICE)
    backbone_scratch, head_scratch = train_classifier(backbone_scratch, head_scratch, train_loader, freeze_backbone=False)
    auroc_scratch = evaluate(backbone_scratch, head_scratch, test_loader)

    # Pretrain backbone once (self-supervised, on the SAME train subjects only -- no leakage
    # from the held-out test subject); Arms B and C both start from this same initialization
    # so any difference between them isolates the effect of freezing vs. fine-tuning, not a
    # difference in what was pretrained.
    backbone_pretrained_base = ConvBackbone().to(DEVICE)
    backbone_pretrained_base = pretrain_backbone(backbone_pretrained_base, train_loader)

    # Arm B: frozen backbone + linear probe only
    backbone_frozen = copy.deepcopy(backbone_pretrained_base)
    head_frozen = ClsHead(backbone_frozen.emb_dim).to(DEVICE)
    backbone_frozen, head_frozen = train_classifier(
        backbone_frozen, head_frozen, train_loader, freeze_backbone=True
    )
    auroc_frozen_probe = evaluate(backbone_frozen, head_frozen, test_loader)

    # Arm C: same pretrained initialization, but fully fine-tuned -- isolates whether
    # pretraining helps once both arms have equal capacity to adapt (fixes the Arm A vs. B confound)
    backbone_finetuned = copy.deepcopy(backbone_pretrained_base)
    head_finetuned = ClsHead(backbone_finetuned.emb_dim).to(DEVICE)
    backbone_finetuned, head_finetuned = train_classifier(
        backbone_finetuned, head_finetuned, train_loader, freeze_backbone=False
    )
    auroc_pretrained_finetuned = evaluate(backbone_finetuned, head_finetuned, test_loader)

    return auroc_scratch, auroc_frozen_probe, auroc_pretrained_finetuned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--shard-index", type=int, default=0,
                     help="Which shard of subjects this process handles (for splitting folds across GPUs)")
    ap.add_argument("--num-shards", type=int, default=1,
                     help="Total number of shards -- folds are independent per held-out subject, "
                          "so running N shards on N GPUs in parallel is a straight speedup, not an approximation")
    ap.add_argument("--out-json", default=None,
                     help="Where to write this shard's per-fold results as JSON (for later aggregation)")
    args = ap.parse_args()

    runs = load_all_runs(args.results_dir)
    all_subjects = subjects_in(runs)
    n_channels_fixed = max(r["data"].shape[0] for r in runs)
    shard_subjects = all_subjects[args.shard_index::args.num_shards]
    print(f"{len(all_subjects)} subjects total, this shard ({args.shard_index}/{args.num_shards}) "
          f"handles {len(shard_subjects)}: {shard_subjects}, device={DEVICE}")

    results = []
    for held_out in shard_subjects:
        train_runs = [r for r in runs if r["subject"] != held_out]
        test_runs = [r for r in runs if r["subject"] == held_out]
        print(f"\n=== Fold: held-out subject {held_out} ({len(train_runs)} train runs, {len(test_runs)} test runs) ===")
        auroc_scratch, auroc_frozen, auroc_finetuned = run_fold(train_runs, test_runs, n_channels_fixed)
        print(f"  A from-scratch (full fine-tune)        AUROC={auroc_scratch:.3f}")
        print(f"  B pretrained + frozen linear probe       AUROC={auroc_frozen:.3f}")
        print(f"  C pretrained + full fine-tune            AUROC={auroc_finetuned:.3f}")
        results.append({"held_out": held_out, "scratch": auroc_scratch,
                         "frozen": auroc_frozen, "finetuned": auroc_finetuned})

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {len(results)} fold results to {args.out_json}")

    if args.num_shards == 1:
        print_summary(results)


def print_summary(results):
    print("\n=== Summary across folds ===")
    for key, label in [("scratch", "A from-scratch (full fine-tune)"),
                        ("frozen", "B pretrained + frozen linear probe"),
                        ("finetuned", "C pretrained + full fine-tune")]:
        vals = [r[key] for r in results if not np.isnan(r[key])]
        print(f"{label}: mean AUROC {np.mean(vals):.3f} (n={len(vals)} folds)")
    print("\nReading guide: A vs C isolates whether pretraining helps once both arms can fully "
          "adapt (equal capacity) -- this is the clean test of Idea 3's hypothesis. A vs B conflates "
          "pretraining with the frozen-probe capacity constraint and should not be used alone to "
          "judge whether pretraining helps.")


if __name__ == "__main__":
    main()
