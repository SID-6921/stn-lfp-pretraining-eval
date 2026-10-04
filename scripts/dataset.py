"""
Windowing + labeling for the extracted STN-LFP .npz files (see extract_lfp.py).

Design note on channel vocabulary (important, read before extending):
LaBraM/BIOT's channel-position embeddings are built for scalp 10-20 EEG
names; our bipolar DBS-lead channels ("LFP-right-01" etc.) have no entry in
either model's vocabulary, so bolting one of those backbones directly onto
this data would require an unjustified made-up channel mapping. Instead this
harness pretrains its own small representation across the pooled patient
corpus (self-supervised, channel-count-agnostic via per-channel processing)
and compares it against a from-scratch-per-fold baseline -- the actual
question Idea 3 asks (does ANY cross-patient pretraining help transfer here)
without smuggling in a false scalp-EEG transfer claim.
"""
import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset

WINDOW_SECONDS = 2.0
LABEL_MAP = {"MedOff": 1, "MedOn": 0}  # MedOff = higher-symptom proxy state, the one adaptive stimulation should detect


def load_run(npz_path: str) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    return {
        "data": d["data"],  # (n_channels, n_samples)
        "sfreq": float(d["sfreq"]),
        "subject": str(d["subject"]),
        "med_state": str(d["med_state"]),
        "ch_names": list(d["ch_names"]),
    }


def window_run(run: dict, window_seconds: float = WINDOW_SECONDS) -> tuple[np.ndarray, np.ndarray]:
    """Split one run's continuous signal into non-overlapping fixed-length windows.
    Returns (windows, labels) where windows has shape (n_windows, n_channels, win_len)."""
    win_len = int(window_seconds * run["sfreq"])
    data = run["data"]
    n_channels, n_samples = data.shape
    n_windows = n_samples // win_len
    assert n_windows > 0, f"Run too short for a single {window_seconds}s window: {n_samples} samples @ {run['sfreq']}Hz"
    trimmed = data[:, : n_windows * win_len]
    windows = trimmed.reshape(n_channels, n_windows, win_len).transpose(1, 0, 2)
    label = LABEL_MAP[run["med_state"]]
    labels = np.full(n_windows, label, dtype=np.int64)
    return windows, labels


def load_all_runs(results_dir: str) -> list[dict]:
    paths = sorted(glob.glob(os.path.join(results_dir, "*.npz")))
    assert len(paths) > 0, f"No extracted .npz runs found in {results_dir} — run extract_lfp.py first"
    return [load_run(p) for p in paths]


def subjects_in(runs: list[dict]) -> list[str]:
    return sorted(set(r["subject"] for r in runs))


class LFPWindowDataset(Dataset):
    """Windows from a given subset of runs (e.g. all-but-one subject for training,
    the held-out subject for testing), channel count normalized by zero-padding
    or truncating to `n_channels_fixed` since subjects have different contact
    counts (5 vs 6 in the pilot) -- padding is masked out, never silently
    averaged in, so the model can tell real channels from padding."""

    def __init__(self, runs: list[dict], n_channels_fixed: int, window_seconds: float = WINDOW_SECONDS):
        all_windows, all_labels, all_masks = [], [], []
        for run in runs:
            windows, labels = window_run(run, window_seconds)
            n_ch = windows.shape[1]
            assert n_ch <= n_channels_fixed, (
                f"Run for subject {run['subject']} has {n_ch} channels > n_channels_fixed={n_channels_fixed}; "
                "raise n_channels_fixed to at least the max channel count across all runs."
            )
            padded = np.zeros((windows.shape[0], n_channels_fixed, windows.shape[2]), dtype=np.float32)
            padded[:, :n_ch, :] = windows
            mask = np.zeros((windows.shape[0], n_channels_fixed), dtype=np.float32)
            mask[:, :n_ch] = 1.0
            all_windows.append(padded)
            all_labels.append(labels)
            all_masks.append(mask)
        self.windows = np.concatenate(all_windows, axis=0)
        self.labels = np.concatenate(all_labels, axis=0)
        self.masks = np.concatenate(all_masks, axis=0)
        # per-window z-score normalization (robust to amplitude differences across subjects/electrodes)
        mu = self.windows.mean(axis=2, keepdims=True)
        sigma = self.windows.std(axis=2, keepdims=True) + 1e-6
        self.windows = (self.windows - mu) * self.masks[:, :, None] / sigma + 0  # keep padded rows at 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.windows[idx]),
            torch.from_numpy(self.masks[idx]),
            torch.tensor(self.labels[idx]),
        )


if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    runs = load_all_runs(results_dir)
    print(f"Loaded {len(runs)} runs across subjects: {subjects_in(runs)}")
    max_ch = max(r["data"].shape[0] for r in runs)
    print(f"Max channel count across runs: {max_ch}")
    ds = LFPWindowDataset(runs, n_channels_fixed=max_ch)
    print(f"Total windows: {len(ds)}, label distribution: "
          f"MedOff={int((ds.labels == 1).sum())}, MedOn={int((ds.labels == 0).sum())}")
    w, m, y = ds[0]
    print(f"Sample window shape: {w.shape}, mask shape: {m.shape}, label: {y.item()}")
