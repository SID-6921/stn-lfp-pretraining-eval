"""
Discover and download one subject's best-available movement-task recording
(both medication states) from ds004998's public S3 mirror, in proper BIDS
layout, then extract LFP via extract_lfp.py and delete the raw MEG+LFP files
to save disk (each subject's raw files are ~1.5-7GB; the extracted LFP-only
.npz is a few MB).

Subjects use inconsistent task naming (MoveL vs MoveR depending on which
side was tested) and some recordings are split across multiple .fif parts
(a BIDS/FIF convention for large files) -- this script discovers what's
actually there via the S3 listing API rather than assuming a fixed filename,
then downloads whatever task name and split count that subject actually has.

Usage:
    python fetch_subject.py sub-XXXXXX
"""
import argparse
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from urllib.request import urlopen

S3_BASE = "https://s3.amazonaws.com/openneuro.org"
BIDS_ROOT = os.path.expanduser("~/work/neurotech_eeg_eval/data/pd_stn_lfp/raw")
RESULTS_DIR = os.path.expanduser("~/work/neurotech_eeg_eval/results")
SCRIPT_DIR = os.path.expanduser("~/work/neurotech_eeg_eval/scripts")
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def s3_list(prefix: str) -> list[str]:
    url = f"{S3_BASE}/?list-type=2&prefix={prefix}&max-keys=200"
    with urlopen(url, timeout=20) as resp:
        xml_bytes = resp.read()
    root = ET.fromstring(xml_bytes)
    return [el.text for el in root.findall(".//s3:Key", NS)]


def find_best_task(subject: str) -> tuple[str, dict]:
    """Return (task_name, {med_state: [filenames in split order]}) for the
    first movement task where BOTH MedOff and MedOn recordings exist."""
    prefix = f"ds004998/{subject}/ses-PeriOp/meg/"
    keys = s3_list(prefix)
    fif_keys = [k for k in keys if k.endswith("_meg.fif")]

    pattern = re.compile(
        r"task-(?P<task>\w+?)_acq-(?P<acq>MedOff|MedOn)_run-(?P<run>\d+)"
        r"(?:_split-(?P<split>\d+))?_meg\.fif$"
    )
    by_task_acq = {}
    for key in fif_keys:
        fname = key.split("/")[-1]
        m = pattern.search(fname)
        if not m:
            continue
        task, acq, run, split = m.group("task"), m.group("acq"), m.group("run"), m.group("split")
        by_task_acq.setdefault((task, acq), {}).setdefault(run, []).append((split or "00", fname))

    for task in ["MoveL", "MoveR", "HoldL", "HoldR"]:
        has_off = (task, "MedOff") in by_task_acq
        has_on = (task, "MedOn") in by_task_acq
        if has_off and has_on:
            result = {}
            for acq in ["MedOff", "MedOn"]:
                runs = by_task_acq[(task, acq)]
                best_run = sorted(runs.keys())[0]  # prefer run-1
                files_sorted = [f for _, f in sorted(runs[best_run])]
                result[acq] = {"run": best_run, "files": files_sorted}
            return task, result

    raise RuntimeError(f"No movement task with both MedOff and MedOn found for {subject}. "
                        f"Available (task, acq) pairs: {sorted(by_task_acq.keys())}")


def download(url: str, dest: str):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    subprocess.run(["curl", "-sL", "--max-time", "600", "-o", dest, url], check=True)
    with open(dest, "rb") as f:
        head = f.read(64)
    assert not head.startswith(b"<?xml"), (
        f"Download returned an S3 error body instead of file content: {url} -> {dest}\n{head}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject")  # e.g. sub-FYbcap
    args = ap.parse_args()
    sub = args.subject
    sub_id = sub.replace("sub-", "")

    task, acq_info = find_best_task(sub)
    print(f"{sub}: using task={task}, acq_runs={ {k: v['run'] for k, v in acq_info.items()} }")

    meg_dir = f"{BIDS_ROOT}/{sub}/ses-PeriOp/meg"
    montage_dir = f"{BIDS_ROOT}/{sub}/ses-PeriOp/montage"
    for acq, info in acq_info.items():
        for fif_fname in info["files"]:
            # Sidecar naming is inconsistent across this dataset's recordings: events.tsv is
            # per-split when the recording is split (e.g. "..._split-01_events.tsv"), but
            # channels.tsv is always run-level with no split suffix even on split recordings.
            # Downloading a guessed name silently saves an S3 XML error body as if it were the
            # tsv (caught downstream only by the assert in download()) -- so derive each sidecar
            # name according to its own convention rather than reusing one stem for both.
            stem_split = fif_fname[: -len("_meg.fif")]
            stem_run = re.sub(r"_split-\d+$", "", stem_split)
            download(f"{S3_BASE}/ds004998/{sub}/ses-PeriOp/meg/{fif_fname}", f"{meg_dir}/{fif_fname}")
            # events.tsv exists only once per run -- on split-01 (or the bare run file when
            # there's no split at all), never on continuation parts (split-02+)
            split_match = re.search(r"_split-(\d+)$", stem_split)
            is_first_part = split_match is None or split_match.group(1) == "01"
            if is_first_part:
                events_fname = f"{stem_split}_events.tsv"
                download(f"{S3_BASE}/ds004998/{sub}/ses-PeriOp/meg/{events_fname}", f"{meg_dir}/{events_fname}")
            channels_fname = f"{stem_run}_channels.tsv"
            channels_dest = f"{meg_dir}/{channels_fname}"
            if not os.path.exists(channels_dest):
                download(f"{S3_BASE}/ds004998/{sub}/ses-PeriOp/meg/{channels_fname}", channels_dest)

    download(f"{S3_BASE}/ds004998/{sub}/ses-PeriOp/montage/{sub}_ses-PeriOp_montage.tsv",
              f"{montage_dir}/{sub}_ses-PeriOp_montage.tsv")

    # extract LFP for both med states, then delete the raw MEG+LFP files to save disk
    for acq in acq_info:
        run = acq_info[acq]["run"]
        out_npz = f"{RESULTS_DIR}/{sub}_{task}_{acq}.npz"
        cmd = [sys.executable, f"{SCRIPT_DIR}/extract_lfp.py", BIDS_ROOT, sub_id, task, acq, run, out_npz]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    subprocess.run(["rm", "-rf", f"{BIDS_ROOT}/{sub}"], check=True)
    print(f"{sub}: done, raw files removed, extracted npz kept in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
