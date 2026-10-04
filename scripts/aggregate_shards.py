"""Merge per-GPU-shard fold results (from eval_harness.py --out-json) and print the combined summary."""
import argparse
import glob
import json
import numpy as np

from eval_harness import print_summary

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("shard_json_glob")
    args = ap.parse_args()

    results = []
    for path in sorted(glob.glob(args.shard_json_glob)):
        with open(path) as f:
            results.extend(json.load(f))

    seen = [r["held_out"] for r in results]
    assert len(seen) == len(set(seen)), f"Duplicate held-out subjects across shards: {seen}"
    print(f"Aggregated {len(results)} folds from shards matching {args.shard_json_glob}")
    print_summary(results)
