"""
Extract bipolar STN-LFP channels (discard the much larger MEG channels) from
ds004998 BIDS recordings, downsample, and save a compact per-run .npz with
the signal + medication-state label.

Channel identification follows the dataset's own validated preprocessing
recipe (Fayed-Rsl/RHM_preprocessing, linked from the ds004998 README): raw
"EEG00N" channels are LFP contacts in disguise (digitized through the MEG
system's auxiliary amplifier) and must be renamed via each subject's
montage.tsv before they're identifiable — there is no generic MNE channel
type or name pattern that reveals this; it must be read from that file.

Usage:
    python extract_lfp.py <bids_root> <subject> <task> <acquisition> <run> <out_npz>
    e.g. python extract_lfp.py ~/work/.../pd_stn_lfp/raw 0cGdk9 MoveL MedOff 1 out.npz
"""
import argparse
import pandas as pd
import numpy as np
import mne
from mne_bids import BIDSPath, read_raw_bids

LFP_TARGET_SFREQ = 250  # STN-LFP control signals (beta band 12-32 Hz) need far less than MEG's native rate


def load_montage_mapping(bids_root: str, session: str, subject: str, raw_ch_names: list[str]) -> dict:
    montage_path = BIDSPath(root=bids_root, session=session, subject=subject,
                             datatype="montage", extension=".tsv", check=False)
    matches = montage_path.match()
    assert len(matches) == 1, (
        f"Expected exactly one montage.tsv for subject {subject}, found {len(matches)}: {matches}"
    )
    df = pd.read_csv(matches[0], sep="\t")
    mapping = {row["right_contacts_old"]: row["right_contacts_new"] for _, row in df.iterrows()}
    mapping.update({row["left_contacts_old"]: row["left_contacts_new"] for _, row in df.iterrows()})
    # drop entries for channels marked bad / absent in this particular run
    mapping = {k: v for k, v in mapping.items() if k in raw_ch_names}
    assert len(mapping) > 0, (
        f"None of the montage.tsv old-contact names {list(mapping.keys())[:4]}... "
        f"were found in this run's raw channels {raw_ch_names[:10]}... — "
        "montage file and recording may be mismatched."
    )
    return mapping


def bipolar_reference(raw: mne.io.Raw, side: str) -> mne.io.Raw:
    """Re-reference consecutive LFP contacts on one side into a bipolar
    derivation, matching the contact scheme actual adaptive-DBS systems use
    (e.g. Oehrn et al. 2024) rather than raw monopolar contacts."""
    chans = sorted([c for c in raw.ch_names if f"LFP-{side}-" in c])
    if len(chans) < 2:
        return raw, []
    bipolar_names = [f"{chans[i]}{chans[i+1][-1]}" for i in range(len(chans) - 1)]
    raw = mne.set_bipolar_reference(raw, anode=chans[:-1], cathode=chans[1:],
                                     ch_name=bipolar_names, drop_refs=True, verbose="ERROR")
    return raw, bipolar_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bids_root")
    ap.add_argument("subject")
    ap.add_argument("task")
    ap.add_argument("acquisition", choices=["MedOff", "MedOn"])
    ap.add_argument("run")
    ap.add_argument("out_npz")
    ap.add_argument("--session", default="PeriOp")
    args = ap.parse_args()

    bids_path = BIDSPath(root=args.bids_root, session=args.session, subject=args.subject,
                          datatype="meg", task=args.task, acquisition=args.acquisition,
                          run=args.run, extension=".fif")
    try:
        raw = read_raw_bids(bids_path, verbose="ERROR")
    except FileNotFoundError:
        # some recordings are split across multiple .fif parts (BIDS "split" entity);
        # mne_bids needs the first split named explicitly, then follows the internal
        # FIF link to the remaining parts itself
        raw = read_raw_bids(bids_path.copy().update(split="01"), verbose="ERROR")
    raw.pick_types(meg=False, eeg=True, emg=False, eog=False, stim=False, exclude="bads")

    mapping = load_montage_mapping(args.bids_root, args.session, args.subject, raw.ch_names)
    raw.rename_channels(mapping)

    lfp_names_before = [c for c in raw.ch_names if c.startswith("LFP-")]
    assert len(lfp_names_before) > 0, f"No LFP-renamed channels found after applying montage mapping: {raw.ch_names}"

    raw.load_data()
    raw, right_bi = bipolar_reference(raw, "right")
    raw, left_bi = bipolar_reference(raw, "left")
    bipolar_names = right_bi + left_bi
    assert len(bipolar_names) > 0, "Bipolar referencing produced zero channels — check montage/contact count"

    raw.pick(bipolar_names)
    raw.load_data()
    orig_sfreq = raw.info["sfreq"]
    raw.filter(l_freq=1.0, h_freq=100.0, method="fir", verbose="ERROR")  # match dataset's published preprocessing
    raw.resample(LFP_TARGET_SFREQ)

    data = raw.get_data()
    assert data.shape[0] == len(bipolar_names), f"Shape mismatch: {data.shape} vs {len(bipolar_names)} channels"

    np.savez_compressed(
        args.out_npz,
        data=data.astype(np.float32),
        ch_names=np.array(bipolar_names),
        sfreq=LFP_TARGET_SFREQ,
        orig_sfreq=orig_sfreq,
        subject=args.subject,
        task=args.task,
        med_state=args.acquisition,
    )
    print(f"Saved {args.out_npz}: {data.shape} @ {LFP_TARGET_SFREQ}Hz (orig {orig_sfreq}Hz), "
          f"channels={bipolar_names}, med_state={args.acquisition}")


if __name__ == "__main__":
    main()
