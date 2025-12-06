#!/usr/bin/env python3
"""
Compare plotting data between old and consolidated formats.

This script extracts the underlying data from both plotting approaches
and compares them to verify correctness of the refactoring.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
import sys

# Import utilities
from plotting_utils import (
    parse_filename, normalize_dataset_name,
    filter_for_baseline_persona
)


def extract_best_model_performance_old(DATASETS) -> pd.DataFrame:
    """Extract data for best model performance using OLD format."""
    print("\n=== Extracting Best Model Performance (OLD FORMAT) ===")

    all_results = []

    for dataset in DATASETS:
        if not os.path.exists(dataset):
            print(f"Warning: Dataset {dataset} does not exist, skipping...")
            continue

        print(f"Processing dataset: {dataset}")

        # Find all trainer_results.json files
        json_files = list(Path(dataset).rglob("*trainer_results.json"))
        json_files = [f for f in json_files if filter_for_baseline_persona(str(f))]
        print(f"  Found {len(json_files)} matching files")

        for filepath in json_files:
            try:
                # Parse filename
                model, ft, context, style, oppu, persona = parse_filename(str(filepath))
                if model is None or "random_validation" not in str(filepath):
                    continue

                # Load JSON
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Extract confusion matrices
                confusion_matrices = []
                if isinstance(data, list):
                    confusion_matrices = [item['confusion_matrix'] for item in data if 'confusion_matrix' in item]
                elif 'confusion_matrix' in data:
                    confusion_matrices = [data['confusion_matrix']]

                if len(confusion_matrices) < 2:
                    continue

                # Calculate accuracies
                accuracies = []
                for cm in confusion_matrices:
                    cm_array = np.array(cm)
                    total = cm_array.sum()
                    correct = cm_array[0,0] + cm_array[1,1]
                    accuracies.append(correct / total if total > 0 else 0)

                all_results.append({
                    "model": model,
                    "dataset": normalize_dataset_name(dataset),
                    "accuracy_mean": np.mean(accuracies),
                    "accuracy_std": np.std(accuracies, ddof=1),
                    "n_runs": len(accuracies)
                })

            except Exception as e:
                print(f"  Error processing {filepath.name}: {e}")
                continue

    df = pd.DataFrame(all_results)
    print(f"\nExtracted {len(df)} records from OLD format")
    return df


def extract_best_model_performance_consolidated(DATASETS) -> pd.DataFrame:
    """Extract data for best model performance using CONSOLIDATED format."""
    print("\n=== Extracting Best Model Performance (CONSOLIDATED FORMAT) ===")

    all_results = []

    for dataset in DATASETS:
        # Transform path to consolidated
        consolidated_path = dataset.replace('results/results_', 'postprocessing_consolidated/')

        if not os.path.exists(consolidated_path):
            print(f"Warning: {consolidated_path} does not exist, skipping...")
            continue

        print(f"Processing dataset: {dataset}")
        dataset_name = normalize_dataset_name(dataset)

        # Find all validation_results.json files
        all_json_files = list(Path(consolidated_path).rglob("validation_results.json"))

        # Filter for baseline+persona
        json_files = []
        for f in all_json_files:
            config_dir = f.parent.name
            if '__noft__ctx0__style0__no_OPPU' in config_dir and '__no_persona' not in config_dir:
                json_files.append(f)

        print(f"  Found {len(json_files)} matching files")

        for filepath in json_files:
            try:
                # Parse config from directory name
                config_dir = filepath.parent.name
                model, ft, context, style, oppu, persona = parse_filename(config_dir)
                if model is None:
                    continue

                # Load validation results
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Extract accuracies from random validation
                if 'random' not in data.get('results', {}):
                    continue

                random_results = data['results']['random']
                trainer_results = random_results.get('trainer_results', [])

                if len(trainer_results) < 2:
                    continue

                # Extract accuracies
                accuracies = [run.get('accuracy', 0) for run in trainer_results
                             if 'accuracy' in run]

                if not accuracies:
                    continue

                all_results.append({
                    "model": model,
                    "dataset": dataset_name,
                    "accuracy_mean": np.mean(accuracies),
                    "accuracy_std": np.std(accuracies, ddof=1),
                    "n_runs": len(accuracies)
                })

            except Exception as e:
                print(f"  Error processing {filepath.name}: {e}")
                continue

    df = pd.DataFrame(all_results)
    print(f"\nExtracted {len(df)} records from CONSOLIDATED format")
    return df


def compare_dataframes(df_old: pd.DataFrame, df_new: pd.DataFrame,
                       name: str, tolerance: float = 1e-6) -> Tuple[bool, str]:
    """Compare two dataframes and return whether they match."""
    print(f"\n=== Comparing {name} ===")

    if df_old.empty and df_new.empty:
        print("✅ Both dataframes are empty - OK")
        return True, "Both empty"

    if df_old.empty:
        print("❌ Old dataframe is empty but new is not")
        return False, "Old dataframe empty"

    if df_new.empty:
        print("❌ New dataframe is empty but old is not")
        return False, "New dataframe empty"

    # Check number of rows
    if len(df_old) != len(df_new):
        msg = f"❌ Different number of rows: old={len(df_old)}, new={len(df_new)}"
        print(msg)
        return False, msg

    print(f"✅ Same number of rows: {len(df_old)}")

    # Sort both by all common columns for proper comparison
    # This ensures rows are aligned correctly even if the order differs
    common_cols = list(set(df_old.columns) & set(df_new.columns))
    if common_cols:
        df_old = df_old.sort_values(common_cols).reset_index(drop=True)
        df_new = df_new.sort_values(common_cols).reset_index(drop=True)

    # Compare columns
    old_cols = set(df_old.columns)
    new_cols = set(df_new.columns)

    if old_cols != new_cols:
        print(f"⚠️  Different columns:")
        print(f"   Old only: {old_cols - new_cols}")
        print(f"   New only: {new_cols - old_cols}")
    else:
        print(f"✅ Same columns: {list(df_old.columns)}")

    # Compare values for common columns
    common_cols = old_cols & new_cols
    differences = []

    for col in common_cols:
        if df_old[col].dtype in [np.float64, np.float32]:
            # Numeric comparison with tolerance
            if not np.allclose(df_old[col], df_new[col], rtol=tolerance, atol=tolerance, equal_nan=True):
                max_diff = np.abs(df_old[col] - df_new[col]).max()
                differences.append(f"{col}: max diff = {max_diff}")
        else:
            # Exact comparison for non-numeric
            if not (df_old[col] == df_new[col]).all():
                n_diff = (df_old[col] != df_new[col]).sum()
                differences.append(f"{col}: {n_diff} values differ")

    if differences:
        print(f"❌ Differences found:")
        for diff in differences:
            print(f"   - {diff}")
        return False, "; ".join(differences)
    else:
        print(f"✅ All values match (tolerance={tolerance})")
        return True, "All match"


def extract_cosine_similarities_old(DATASETS) -> pd.DataFrame:
    """Extract cosine similarity data using OLD format."""
    print("\n=== Extracting Cosine Similarities (OLD FORMAT) ===")

    rows = []

    for dataset in DATASETS:
        if not os.path.exists(dataset):
            print(f"Warning: Dataset {dataset} does not exist, skipping...")
            continue

        print(f"Processing dataset: {dataset}")

        # Find all optimal_response.json files
        optimal_files = []
        for dirpath, _, filenames in os.walk(dataset):
            for fname in filenames:
                if fname.endswith("__optimal_response.json"):
                    optimal_files.append(Path(dirpath) / fname)

        # Filter for baseline+persona only (noft, ctx0, style0, no_OPPU, with persona)
        filtered_files = [f for f in optimal_files
                         if '__noft__ctx0__style0__no_OPPU' in str(f)
                         and '__no_persona' not in str(f)]
        print(f"  Found {len(filtered_files)} matching files")

        for filepath in filtered_files:
            try:
                # Parse filename
                model, ft, context, style, oppu, persona = parse_filename(
                    filepath.stem.replace("__optimal_response", "")
                )
                if model is None:
                    continue

                # Load JSON
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Extract similarities
                for entry in data:
                    all_valid = entry.get("all_valid_responses", [])

                    random_resp = entry.get("response", "").strip().lower()
                    ml_resp = entry.get("ML_best_response", "").strip().lower()
                    cosine_resp = entry.get("cosine_best_response", "").strip().lower()

                    # Find indices of selected responses (first match only to avoid duplicates)
                    random_idx = None
                    ml_idx = None
                    cosine_idx = None

                    for idx, valid_resp in enumerate(all_valid):
                        resp_text = valid_resp.get("response", "").strip().lower()

                        if random_idx is None and resp_text == random_resp:
                            random_idx = idx
                        if ml_idx is None and resp_text == ml_resp:
                            ml_idx = idx
                        if cosine_idx is None and resp_text == cosine_resp:
                            cosine_idx = idx

                    # Now add only the selected candidates using indices
                    for idx, valid_resp in enumerate(all_valid):
                        similarity = valid_resp.get("cosine_similarity")

                        if similarity is None:
                            continue

                        # Add based on index match
                        if idx == random_idx:
                            rows.append({
                                "model": model,
                                "dataset": normalize_dataset_name(dataset),
                                "method": "random",
                                "similarity": similarity
                            })

                        if idx == ml_idx:
                            rows.append({
                                "model": model,
                                "dataset": normalize_dataset_name(dataset),
                                "method": "ml",
                                "similarity": similarity
                            })

                        if idx == cosine_idx:
                            rows.append({
                                "model": model,
                                "dataset": normalize_dataset_name(dataset),
                                "method": "cosine",
                                "similarity": similarity
                            })

            except Exception as e:
                print(f"  Error processing {filepath.name}: {e}")
                continue

    df = pd.DataFrame(rows)
    print(f"\nExtracted {len(df)} cosine similarity records from OLD format")
    if len(df) > 0:
        print(f"  By method: {df.groupby('method').size().to_dict()}")
    return df


def extract_cosine_similarities_consolidated(DATASETS) -> pd.DataFrame:
    """Extract cosine similarity data using CONSOLIDATED format."""
    print("\n=== Extracting Cosine Similarities (CONSOLIDATED FORMAT) ===")

    rows = []

    for dataset in DATASETS:
        # Transform path to consolidated
        consolidated_path = dataset.replace('results/results_', 'results_consolidated/')

        if not os.path.exists(consolidated_path):
            print(f"Warning: {consolidated_path} does not exist, skipping...")
            continue

        print(f"Processing dataset: {dataset}")
        dataset_name = normalize_dataset_name(dataset)

        # Find response files in consolidated format
        response_dir = Path(consolidated_path) / "responses"
        if not response_dir.exists():
            continue

        response_files = list(response_dir.glob("*.json"))
        # Filter for baseline+persona only
        filtered_files = [f for f in response_files if '__noft__ctx0__style0__no_OPPU' in str(f)
                         and '__no_persona' not in str(f)]

        print(f"  Found {len(filtered_files)} matching files")

        for filepath in filtered_files:
            try:
                # Parse filename
                model, ft, context, style, oppu, persona = parse_filename(filepath.stem)
                if model is None:
                    continue

                # Load responses
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Extract similarities
                for response_entry in data.get('responses', []):
                    candidates = response_entry.get('candidates', [])
                    selected_indices = response_entry.get('selected_indices', {})

                    if not candidates:
                        continue

                    # Get selected response indices for each method
                    random_idx = selected_indices.get('random')
                    ml_idx = selected_indices.get('ml')
                    cosine_idx = selected_indices.get('cosine')

                    # Extract similarities for selected responses
                    for idx, candidate in enumerate(candidates):
                        if not isinstance(candidate, dict):
                            continue

                        similarity = candidate.get('cosine_similarity')
                        if similarity is None:
                            continue

                        # Add based on selection
                        if idx == random_idx:
                            rows.append({
                                "model": model,
                                "dataset": dataset_name,
                                "method": "random",
                                "similarity": similarity
                            })

                        if idx == ml_idx:
                            rows.append({
                                "model": model,
                                "dataset": dataset_name,
                                "method": "ml",
                                "similarity": similarity
                            })

                        if idx == cosine_idx:
                            rows.append({
                                "model": model,
                                "dataset": dataset_name,
                                "method": "cosine",
                                "similarity": similarity
                            })

            except Exception as e:
                print(f"  Error processing {filepath.name}: {e}")
                continue

    df = pd.DataFrame(rows)
    print(f"\nExtracted {len(df)} cosine similarity records from CONSOLIDATED format")
    if len(df) > 0:
        print(f"  By method: {df.groupby('method').size().to_dict()}")
    return df


def save_comparison_data(df_old: pd.DataFrame, df_new: pd.DataFrame,
                         name: str, output_dir: str):
    """Save dataframes for manual inspection."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    df_old.to_csv(output_path / f"{name}_old.csv", index=False)
    df_new.to_csv(output_path / f"{name}_new.csv", index=False)

    # Save as JSON for easier diffing
    df_old.to_json(output_path / f"{name}_old.json", orient='records', indent=2)
    df_new.to_json(output_path / f"{name}_new.json", orient='records', indent=2)

    print(f"  Saved comparison data to {output_path}/")


def main():
    """Main comparison function."""
    print("=" * 80)
    print("PLOTTING DATA COMPARISON - OLD vs CONSOLIDATED FORMAT")
    print("=" * 80)

    # Define datasets
    DATASETS = [
        'results/results_bluesky',
        'results/results_twitter',
        'results/results_reddit'
    ]

    output_dir = "comparison_output"

    # Check if datasets exist
    print("\nChecking dataset availability:")
    for ds in DATASETS:
        old_exists = os.path.exists(ds)
        new_exists = os.path.exists(ds.replace('results/results_', 'postprocessing_consolidated/'))
        print(f"  {ds}:")
        print(f"    Old format: {old_exists}")
        print(f"    New format: {new_exists}")

    # Test 1: Best Model Performance
    print("\n" + "=" * 80)
    print("TEST 1: Best Model Performance Data")
    print("=" * 80)

    df_perf_old = extract_best_model_performance_old(DATASETS)
    df_perf_new = extract_best_model_performance_consolidated(DATASETS)

    save_comparison_data(df_perf_old, df_perf_new, "best_model_performance", output_dir)
    match_perf, msg_perf = compare_dataframes(df_perf_old, df_perf_new, "Best Model Performance")

    # Test 2: Cosine Similarity Data
    print("\n" + "=" * 80)
    print("TEST 2: Cosine Similarity Data")
    print("=" * 80)

    df_cosine_old = extract_cosine_similarities_old(DATASETS)
    df_cosine_new = extract_cosine_similarities_consolidated(DATASETS)

    save_comparison_data(df_cosine_old, df_cosine_new, "cosine_similarities", output_dir)
    match_cosine, msg_cosine = compare_dataframes(df_cosine_old, df_cosine_new, "Cosine Similarities")

    # Summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    results = [
        ("Best Model Performance", match_perf, msg_perf),
        ("Cosine Similarities", match_cosine, msg_cosine)
    ]

    all_pass = True
    for test_name, passed, message in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            print(f"         {message}")
            all_pass = False

    print("\n" + "=" * 80)
    if all_pass:
        print("✅ ALL TESTS PASSED - Refactoring is correct!")
        print("=" * 80)
        return 0
    else:
        print("❌ SOME TESTS FAILED - Review differences")
        print(f"   Check {output_dir}/ for detailed comparison files")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
