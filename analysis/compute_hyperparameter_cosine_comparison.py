#!/usr/bin/env python3
"""
Human-vs-AI cosine similarity (to the human ground-truth reply) by decoding
variant, for the Bluesky decoding-hyperparameter ablation (Supplementary
Section 13, R1.2 / Concern 2). Feeds plot_hyperparameter_cosine_comparison.py.

For each model, extracts the cosine similarity already precomputed per
candidate in each *_optimal_response.json (all_valid_responses[i].cosine_similarity),
matched back to whichever candidate was actually selected under each of the
three selection methods (random / ML-optimal / cosine-optimal), for three
decoding variants:
  - baseline: default decoding (T=0.8, top_p=0.9, top_k=50), from the
    reference-configuration run in results_PNAS_revision/
  - medium / looser: from the ablation run in results_hyperparameters/
    (see REVIEWER_EXPERIMENTS.md, Experiment A)

Mistral-7B-Instruct-v0.2 has no "medium" variant (not part of the ablation
design), so that combination is simply absent from the output.

Usage:
    python analysis/compute_hyperparameter_cosine_comparison.py \
        --baseline-dir results_PNAS_revision --hyperparam-dir results_hyperparameters \
        --output-dir results_hyperparameters
"""

import os
import argparse
import json
from pathlib import Path

import pandas as pd

# (model, org, config_suffix_within_hyperparam_dir) -- config_suffix is None for baseline
MODELS = {
    "Llama-3.1-8B": "meta-llama",
    "Mistral-7B-Instruct-v0.2": "mistralai",
}
DECODING_VARIANTS = ["baseline", "medium", "looser"]


def response_file(base_dir, org, model, variant):
    tag = "" if variant == "baseline" else f"{variant}__"
    fname = f"{model}__noft__ctx1__style10__{tag}optimal_response.json"
    return Path(base_dir) / "bluesky" / org / fname


def extract_similarities(path, model, variant):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        candidates = entry.get("all_valid_responses", [])
        if not candidates:
            continue

        selections = {
            "random": entry.get("response", "").strip().lower(),
            "ML-optimal": entry.get("ML_best_response", "").strip().lower(),
            "cosine-optimal": entry.get("cosine_best_response", "").strip().lower(),
        }
        found = {k: False for k in selections}

        for cand in candidates:
            text = cand.get("response", "").strip().lower()
            sim = cand.get("cosine_similarity", None)
            if sim is None:
                continue
            for method, target in selections.items():
                if not found[method] and text == target:
                    rows.append({
                        "model": model, "decoding_variant": variant,
                        "selection_method": method, "cosine_similarity": sim,
                    })
                    found[method] = True
            if all(found.values()):
                break

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Compute human-vs-AI cosine similarity by decoding variant (Bluesky ablation).")
    parser.add_argument("--baseline-dir", default="results_PNAS_revision",
                         help="Results folder containing the default-decoding reference run")
    parser.add_argument("--hyperparam-dir", default="results_hyperparameters",
                         help="Results folder containing the medium/looser decoding ablation run")
    parser.add_argument("--output-dir", default=None,
                         help="Where to save hyperparameter_cosine_comparison.csv (default: --hyperparam-dir)")
    args = parser.parse_args()

    output_dir = args.output_dir or args.hyperparam_dir
    os.makedirs(output_dir, exist_ok=True)

    all_rows = []
    missing = []

    for model, org in MODELS.items():
        for variant in DECODING_VARIANTS:
            base_dir = args.baseline_dir if variant == "baseline" else args.hyperparam_dir
            path = response_file(base_dir, org, model, variant)
            if not path.exists():
                missing.append((model, variant, str(path)))
                continue
            all_rows.extend(extract_similarities(path, model, variant))

    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(output_dir, "hyperparameter_cosine_comparison.csv")
    df.to_csv(out_csv, index=False)

    print(f"Total rows: {len(df)}")
    print(f"Missing model/variant combinations: {len(missing)}")
    for m in missing:
        print("  MISSING:", m)

    print("\nMedian cosine similarity by model / decoding variant / selection method:")
    print(df.groupby(["model", "decoding_variant", "selection_method"])["cosine_similarity"]
          .median().round(3).to_string())

    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
