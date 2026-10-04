# STN-LFP pretraining: a surrogate-data control

Does pretraining a decoder across other Parkinson's patients' subthalamic nucleus
local field potentials (STN-LFP) actually transfer genuine cross-patient
physiology, or does it just look like it does?

This repo is the code and results behind that question: a leave-one-subject-out
evaluation of self-supervised pretraining for short-calibration medication-state
(on/off levodopa) decoding, plus a four-arm control design built specifically to
test whether an apparent pretraining benefit survives destroying the pretraining
data's real temporal structure while leaving its power spectrum untouched.

**Full writeup**: see the accompanying paper for the complete methods, results,
and discussion. This README covers the code.

## Headline result

A short-calibration decoding benefit from pretraining does appear under some
signal framings, but pretraining on 17 real patients performs statistically
indistinguishably from pretraining on 17 phase-randomized surrogate (fake)
patients, at every calibration budget tested. The benefit is real; attributing
it to learned cross-patient physiology is not supported by this control. The
one benefit that doesn't depend on resolving that question: any learned
decoder, pretrained or not, clearly beats a naive bandpower-threshold baseline.

## Data

STN-LFP + MEG recordings come from [OpenNeuro ds004998](https://openneuro.org/datasets/ds004998)
(20 Parkinson's patients, simultaneous MEG and deep-brain recordings, on and
off medication, during rest/hold/move tasks), described in Rassoulou et al.,
*Scientific Data* 11:889 (2024), [doi:10.1038/s41597-024-03768-1](https://doi.org/10.1038/s41597-024-03768-1).
The dataset is CC0-licensed (public domain). This repo does not redistribute
the raw or extracted signal; `scripts/fetch_subject.py` downloads and extracts
it from the public OpenNeuro S3 mirror on demand.

## Repo layout

```
scripts/
  fetch_subject.py       # download one subject from ds004998's public S3 mirror, extract LFP
  extract_lfp.py          # BIDS -> compact per-run .npz (channels renamed via montage.tsv)
  dataset.py              # windowing + MedOn/MedOff labeling
  eval_harness.py         # ConvBackbone encoder, raw/spectral pretraining objectives, LOSO eval
  few_shot_eval.py        # scratch-vs-pretrained comparison across calibration budgets
  control_arms_eval.py    # the 4-arm confound check (real / 1-real / surrogate pretraining)
  baseline_bandpower.py   # naive beta-band-power threshold baseline
  aggregate_shards.py     # merge sharded result JSONs
  analyze_results.py      # reproduces every number in the paper's Tables 1-3

results/
  *.json                  # aggregate per-subject, per-calibration-budget results (no raw signal)
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python >= 3.10. A CUDA GPU is strongly recommended for `eval_harness.py`
and `control_arms_eval.py` (each leave-one-subject-out fold trains several small
models); `analyze_results.py` needs no GPU and runs on the committed result JSONs.

## Reproducing the paper's tables

The `results/` JSONs are already here — no data download needed for this step:

```bash
python scripts/analyze_results.py
```

Reproduces Tables 1-3: the control-arms confound check, the scratch-vs-pretrained
comparison by objective/framing (with paired Wilcoxon tests and Cohen's d_z effect
sizes), and the bandpower-baseline comparison.

## Reproducing the results from raw data

```bash
# 1. Fetch and extract one subject (repeat for all subjects you want)
python scripts/fetch_subject.py sub-0cGdk9

# 2. Run the leave-one-subject-out scratch-vs-pretrained comparison
python scripts/few_shot_eval.py results/ --objective spectral --ref-channel 0 --out-json fewshot_spectral.json

# 3. Run the control-arms confound check
python scripts/control_arms_eval.py results/ --seed 0 --out-json control_arms_seed0.json

# 4. Run the naive baseline
python scripts/baseline_bandpower.py results/

# 5. Merge shards and analyze
python scripts/aggregate_shards.py "control_arms_seed*.json"
python scripts/analyze_results.py
```

## License

Code: MIT (see `LICENSE`). Data: CC0, via OpenNeuro ds004998 (not redistributed here).
