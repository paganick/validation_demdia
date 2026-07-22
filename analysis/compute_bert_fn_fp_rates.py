#!/usr/bin/env python3
"""
BERT classifier accuracy, false-negative rate (AI misclassified as human), and
false-positive rate (human misclassified as AI), for the reference configuration
and the best-performing configuration (with its three post-generation selection
variants), by model and platform. Feeds plot_bert_fn_fp.py.

Rates are averaged across all seed runs in trainer_results.json, matching the
convention already used by best_configurations_random_accuracy.csv (a single
bert_report.json only reflects one seed).

Usage:
    python analysis/compute_bert_fn_fp_rates.py results_PNAS_revision/ \
        --output-dir results_PNAS_revision/configuration_optimization_figures
"""

import os
import argparse
import json
import pandas as pd
import numpy as np

MODEL_ORDER = [
    "Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Llama-3.1-70B",
    "Mistral-7B-v0.1", "Mistral-7B-Instruct-v0.2", "gemma-3-4b-it",
    "Qwen2.5-7B-Instruct", "DeepSeek-R1-Distill-Llama-8B", "Apertus-8B-2509",
]
PLATFORMS = ["reddit", "twitter", "bluesky"]


def build_org_by_model(base):
    org_by_model = {}
    for platform in PLATFORMS:
        platform_dir = os.path.join(base, platform)
        if not os.path.isdir(platform_dir):
            continue
        for org in os.listdir(platform_dir):
            org_path = os.path.join(platform_dir, org)
            if not os.path.isdir(org_path):
                continue
            for fname in os.listdir(org_path):
                if fname.endswith("optimal_response.json") and "__" in fname:
                    model = fname.split("__")[0]
                    org_by_model[model] = org
    return org_by_model


def config_suffix(ft, ctx, style, persona):
    s = f"{'ft' if ft else 'noft'}__ctx{ctx}__style{style}__"
    if not persona:
        s += "no_persona__"
    return s


def _cm_metrics(cm):
    cm = np.array(cm)
    tn, fp_c, fn_c, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    total = tn + fp_c + fn_c + tp
    acc = (tn + tp) / total if total > 0 else np.nan
    fn = fp_c / (tn + fp_c) if (tn + fp_c) > 0 else np.nan
    fp = fn_c / (fn_c + tp) if (fn_c + tp) > 0 else np.nan
    return acc, fn, fp


def find_report(base, org_by_model, platform, model, suffix, validation_type):
    org = org_by_model.get(model)
    if org is None:
        return None, f"no org found for model {model}"
    path_base = os.path.join(base, platform, org, f"{model}__{suffix}_{validation_type}_validation_data")
    trainer_path = f"{path_base}_trainer_results.json"
    if os.path.exists(trainer_path):
        with open(trainer_path) as f:
            runs = json.load(f)
        if isinstance(runs, dict):
            runs = [runs]
        metrics = [_cm_metrics(r["confusion_matrix"]) for r in runs if "confusion_matrix" in r]
        if metrics:
            arr = np.array(metrics)
            return tuple(np.nanmean(arr, axis=0)), None
    cm_path = f"{path_base}_confusion_matrix.csv"
    if os.path.exists(cm_path):
        cm = pd.read_csv(cm_path, header=None).values
        return _cm_metrics(cm), None
    json_path = f"{path_base}_bert_report.json"
    if os.path.exists(json_path):
        with open(json_path) as f:
            d = json.load(f)
        return (d["accuracy"], 1 - d["0"]["recall"], 1 - d["1"]["recall"]), None
    return None, f"missing: {trainer_path}"


def main():
    parser = argparse.ArgumentParser(description="Compute BERT FN/FP rates by model, platform, and condition.")
    parser.add_argument("results_folder", help="Base results folder (e.g. results_PNAS_revision/)")
    parser.add_argument("--output-dir", default=None,
                         help="Where to save bert_fn_fp_rates.csv (default: <results_folder>/configuration_optimization_figures)")
    args = parser.parse_args()

    base = args.results_folder
    output_dir = args.output_dir or os.path.join(base, "configuration_optimization_figures")
    os.makedirs(output_dir, exist_ok=True)

    best_csv = os.path.join(output_dir, "best_configurations_random_accuracy.csv")
    best_df = pd.read_csv(best_csv)
    org_by_model = build_org_by_model(base)

    rows = []
    missing = []

    for _, r in best_df.iterrows():
        model, platform = r["model"], r["dataset"]
        if model not in MODEL_ORDER:
            continue

        ref_suffix = config_suffix(False, 0, 0, True)
        res, err = find_report(base, org_by_model, platform, model, ref_suffix, "random")
        if err:
            missing.append((platform, model, "Reference", err))
        else:
            rows.append({"platform": platform, "model": model, "condition": "Reference (BL+PE)",
                         "config_label": "BL + PE", "accuracy": res[0], "fn_rate": res[1], "fp_rate": res[2]})

        best_suffix = config_suffix(bool(r["has_finetuning"]), int(r["has_context"]), int(r["has_style"]), bool(r["has_persona"]))
        for cond_name, vtype in [("Best config", "random"),
                                  ("Best config + Cosine-optimal", "cosine"),
                                  ("Best config + ML-optimal", "ml")]:
            res, err = find_report(base, org_by_model, platform, model, best_suffix, vtype)
            if err:
                missing.append((platform, model, cond_name, err))
            else:
                rows.append({"platform": platform, "model": model, "condition": cond_name,
                             "config_label": r["best_config_short"], "accuracy": res[0],
                             "fn_rate": res[1], "fp_rate": res[2]})

    df = pd.DataFrame(rows)
    out_csv = os.path.join(output_dir, "bert_fn_fp_rates.csv")
    df.to_csv(out_csv, index=False)

    print(f"Total rows: {len(df)}")
    print(f"Missing cells: {len(missing)}")
    for m in missing:
        print("  MISSING:", m)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
