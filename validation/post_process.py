"""
Aggregate validation results from multiple experiments into a summary table.

This script walks through a directory tree containing validation results from
multiple model configurations and aggregates them into a single CSV summary file.

It collects three types of metrics:
    1. Significant features: Count of Empath features that differ significantly
       between human and AI text (from *_significant_features.csv)
    2. Confusion matrices: Classification results from BERT validation
       (from *_confusion_matrix.csv)
    3. Stylistic metrics: Average word count, links, emojis, mentions
       (from *_aggregate_stylistic_metrics.csv)

Filename format convention:
    {model}__{ft|noft}__{ctx0|ctx1}__style{N}__{'OPPU'|'noOPPU'}__*

Where:
    - model: llama3.1_8b, mistral_7b, etc.
    - ft/noft: Fine-tuned or not
    - ctx0/ctx1: Context retrieval disabled/enabled
    - styleN: Number of style examples (0, 1, 3, 5)
    - OPPU/noOPPU: Personalized model or not

Usage:
    python post_process.py /path/to/results/

Output:
    summary_metrics.csv with columns:
        - model, finetuning, context, style, OPPU (configuration)
        - significant_feature_count (Empath analysis)
        - 0-0, 1-0, 0-1, 1-1 (confusion matrix cells)
        - avg_words, avg_links, avg_emojis, avg_mentions (stylistic metrics)
        - top_10_emojis (most common emojis)
"""

import os
import csv
import argparse
import pandas as pd

def parse_filename(filename):
    """
    Extract model configuration from standardized filename.

    Parses filenames following the convention:
    {model}__{ft|noft}__{ctx0|ctx1}__style{N}__{OPPU|noOPPU}__*

    Args:
        filename: Filename to parse (can be full path, will extract basename)

    Returns:
        tuple: (model, ft, context, style, oppu)
            - model (str): Model name (e.g., "llama3.1_8b")
            - ft (int): 1 if fine-tuned, 0 otherwise
            - context (int): 1 if context retrieval enabled, 0 otherwise
            - style (int): Number of style examples (0, 1, 3, 5)
            - oppu (int): 1 if OPPU personalization enabled, 0 otherwise
    """
    base = os.path.basename(filename)
    parts = base.split("__")
    model = parts[0]
    ft = 1 if parts[1] == "ft" else 0
    context = 1 if parts[2] == "ctx1" else 0
    style = int(parts[3].replace("style", ""))
    oppu = 1 if parts[4].startswith("OPPU") else 0
    return model, ft, context, style, oppu

def count_significant_features(filepath):
    """
    Count number of significant Empath features from CSV.

    Reads a CSV with Empath feature analysis results and counts the
    number of features that showed significant differences (number of
    data rows excluding header).

    Args:
        filepath: Path to *_significant_features.csv file

    Returns:
        int: Number of significant features, 0 if file cannot be read
    """
    try:
        with open(filepath, 'r') as f:
            return sum(1 for line in f) - 1  # subtract header
    except Exception as e:
        print(f"Failed reading {filepath}: {e}")
        return 0

def read_confusion_matrix(filepath):
    """
    Parse BERT validation confusion matrix from CSV.

    Reads confusion matrix with format:
        Row 1 (Actual=0): [true_negatives, false_positives]
        Row 2 (Actual=1): [false_negatives, true_positives]

    Args:
        filepath: Path to *_confusion_matrix.csv file

    Returns:
        dict: Confusion matrix cells with keys:
            - '0-0': True negatives (AI correctly classified as AI)
            - '1-0': False positives (AI misclassified as human)
            - '0-1': False negatives (Human misclassified as AI)
            - '1-1': True positives (Human correctly classified as human)
        Returns None values if file cannot be read.
    """
    try:
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            row1 = next(reader)
            row2 = next(reader)
            return {
                '0-0': int(row1[0]),
                '1-0': int(row1[1]),
                '0-1': int(row2[0]),
                '1-1': int(row2[1]),
            }
    except Exception as e:
        print(f"Failed reading confusion matrix from {filepath}: {e}")
        return {'0-0': None, '1-0': None, '0-1': None, '1-1': None}

def read_aggregate_stylistic_metrics(filepath):
    """
    Extract aggregate stylistic metrics from CSV.

    Reads summary statistics about text style including average word count,
    links, emojis, mentions, and most common emojis.

    Args:
        filepath: Path to *_aggregate_stylistic_metrics.csv file

    Returns:
        dict: Stylistic metrics with keys:
            - avg_words: Mean word count
            - avg_links: Mean number of links
            - avg_emojis: Mean number of emojis
            - avg_mentions: Mean number of @mentions
            - top_10_emojis: Most frequently used emojis
        Returns None values if file cannot be read.
    """
    try:
        df = pd.read_csv(filepath)
        # Extract relevant metrics from the DataFrame
        return {
            "avg_words": df["avg_words"].iloc[0],
            "avg_links": df["avg_links"].iloc[0],
            "avg_emojis": df["avg_emojis"].iloc[0],
            "avg_mentions": df["avg_mentions"].iloc[0],
            "top_10_emojis": df["top_10_emojis"].iloc[0]  # It's assumed to be a list or string
        }
    except Exception as e:
        print(f"Failed reading aggregate stylistic metrics from {filepath}: {e}")
        return {"avg_words": None, "avg_links": None, "avg_emojis": None, "avg_mentions": None, "top_10_emojis": None}

def main():
    """
    Aggregate validation results from all experiments into summary CSV.

    Walks through the directory tree starting from root_folder, finds all
    validation result files, parses their configurations and metrics, and
    creates a unified summary table saved to summary_metrics.csv.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("root_folder", type=str, help="Path to the root folder")
    args = parser.parse_args()

    records = {}

    for subdir, _, files in os.walk(args.root_folder):
        for file in files:
            full_path = os.path.join(subdir, file)
            if file.endswith("_significant_features.csv"):
                model, ft, context, style, oppu = parse_filename(file)
                key = (model, ft, context, style, oppu)

                count = count_significant_features(full_path)
                if key not in records:
                    records[key] = {}
                records[key]['significant_feature_count'] = count

            elif file.endswith("_confusion_matrix.csv"):
                model, ft, context, style, oppu = parse_filename(file)
                key = (model, ft, context, style, oppu)

                confusion = read_confusion_matrix(full_path)
                if key not in records:
                    records[key] = {}
                records[key].update(confusion)

            elif file.endswith("aggregate_stylistic_metrics.csv"):
                model, ft, context, style, oppu = parse_filename(file)
                key = (model, ft, context, style, oppu)

                stylistic_metrics = read_aggregate_stylistic_metrics(full_path)
                if key not in records:
                    records[key] = {}
                records[key].update(stylistic_metrics)

    # Convert to DataFrame
    rows = []
    for key, metrics in records.items():
        model, ft, context, style, oppu = key
        row = {
            "model": model,
            "finetuning": ft,
            "context": context,
            "style": style,
            "OPPU": oppu,
            "significant_feature_count": metrics.get("significant_feature_count", 0),
            "0-0": metrics.get("0-0"),
            "1-0": metrics.get("1-0"),
            "0-1": metrics.get("0-1"),
            "1-1": metrics.get("1-1"),
            "avg_words": metrics.get("avg_words"),
            "avg_links": metrics.get("avg_links"),
            "avg_emojis": metrics.get("avg_emojis"),
            "avg_mentions": metrics.get("avg_mentions"),
            "top_10_emojis": metrics.get("top_10_emojis")
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    output_path = os.path.join(args.root_folder, "summary_metrics.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved summary to {output_path}")

if __name__ == "__main__":
    main()
