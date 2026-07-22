"""
Shared helpers for the configuration-optimization ("best config") figure
scripts: generate_config_main_figures.py and generate_config_SI_figures.py.
"""

import os
import json
from pathlib import Path

import pandas as pd
import numpy as np

from simulation.src.plotting_utils import (
    parse_filename, make_label, normalize_dataset_name,
    parse_confusion_matrix, calculate_metrics, find_confusion_matrix_fields,
    get_confusion_matrix_value,
)

HUMAN_COLOR = '#888888'
HUMAN_LABEL = 'Human'


def _save_stats(df, groupby_cols, value_cols, output_path):
    """Save summary statistics (count, mean, std, median, Q1, Q3) for a figure's data."""
    if isinstance(value_cols, str):
        value_cols = [value_cols]
    agg = df.groupby(groupby_cols)[value_cols].describe(percentiles=[.25, .5, .75])
    agg.columns = ['_'.join(c).strip('_') for c in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(output_path, index=False)
    print(f"  Stats saved to: {output_path}")


def format_feature_name(feature):
    """Format feature names for display."""
    return feature.replace('_', ' ').title()


# === Data Loading ===
def load_configuration_data(folder_paths):
    """
    Load configuration optimization data from trainer_results.json files.
    Returns DataFrame with accuracy statistics for each configuration.
    """
    results = []

    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} does not exist, skipping...")
            continue

        dataset_name = normalize_dataset_name(folder_path)
        print(f"\nProcessing {dataset_name}...")

        root_path = Path(folder_path)
        json_files = list(root_path.rglob("*trainer_results.json"))

        for filepath in json_files:
            try:
                model, ft, context, style, persona = parse_filename(str(filepath))

                if model is None:
                    continue

                if "cosine_validation" in str(filepath):
                    response_type = "cosine_optimal"
                elif "ml_validation" in str(filepath):
                    response_type = "ML_optimal"
                elif "random_validation" in str(filepath):
                    response_type = "random"
                else:
                    continue

                with open(filepath, 'r') as f:
                    data = json.load(f)

                cm_fields = find_confusion_matrix_fields(data)

                if len(cm_fields) == 0:
                    continue

                accuracies = []

                for field in cm_fields:
                    cm_data = get_confusion_matrix_value(data, field)
                    cm = parse_confusion_matrix(cm_data)
                    metrics = calculate_metrics(cm)

                    if metrics['accuracy'] is not None:
                        accuracies.append(metrics['accuracy'])

                if len(accuracies) > 1:
                    results.append({
                        "model": model,
                        "dataset": dataset_name,
                        "style": style,
                        "context": context,
                        "ft": ft,
                        "persona": persona,
                        "accuracy_mean": np.mean(accuracies),
                        "accuracy_std": np.std(accuracies, ddof=1),
                        "label": make_label(model, ft, context, style, persona),
                        "short_label": make_label(model, ft, context, style, persona, False),
                        "response_type": response_type,
                        "n_measurements": len(accuracies)
                    })

            except Exception as e:
                continue

    return pd.DataFrame(results)


# === Cosine Similarity Loading ===
def load_cosine_similarities_by_response_type(folder_paths):
    """
    Load cosine similarities for ALL response types from optimal_response.json files.
    Returns dictionary with DataFrames for each response type.
    """
    print(f"\n=== Loading Cosine Similarities ===")

    rows_random = []
    rows_ml = []
    rows_cosine = []

    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            continue

        dataset_name = normalize_dataset_name(folder_path)
        print(f"\nProcessing {dataset_name}...")

        root_path = Path(folder_path)
        files_found = 0

        for dirpath, _, filenames in os.walk(root_path):
            for fname in filenames:
                if fname.endswith("_optimal_response.json"):
                    file_path = Path(dirpath) / fname

                    try:
                        model, ft, context, style, persona = parse_filename(fname)

                        if model is None:
                            continue

                        config_parts = ["BL"]
                        if persona == 1:
                            config_parts.append("PE")
                        if style == 1:
                            config_parts.append("SE")
                        if context == 1:
                            config_parts.append("CR")
                        if ft == 1:
                            config_parts.append("FT")
                        config_label = " + ".join(config_parts)

                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)

                            for entry in data:
                                all_valid_responses = entry.get("all_valid_responses", [])

                                if not all_valid_responses:
                                    continue

                                random_response = entry.get("response", "").strip().lower()
                                ml_response = entry.get("ML_best_response", "").strip().lower()
                                cosine_response = entry.get("cosine_best_response", "").strip().lower()

                                found_random = False
                                found_ml = False
                                found_cosine = False

                                for valid_resp in all_valid_responses:
                                    resp_text = valid_resp.get("response", "").strip().lower()
                                    similarity = valid_resp.get("cosine_similarity", None)

                                    if similarity is None:
                                        continue

                                    base_data = {
                                        "dataset": dataset_name,
                                        "model": model,
                                        "persona": persona,
                                        "style": style,
                                        "context": context,
                                        "ft": ft,
                                        "config_label": config_label,
                                        "similarity": similarity
                                    }

                                    if not found_random and resp_text == random_response:
                                        rows_random.append(base_data.copy())
                                        found_random = True

                                    if not found_ml and resp_text == ml_response:
                                        rows_ml.append(base_data.copy())
                                        found_ml = True

                                    if not found_cosine and resp_text == cosine_response:
                                        rows_cosine.append(base_data.copy())
                                        found_cosine = True

                                    if found_random and found_ml and found_cosine:
                                        break

                        files_found += 1

                    except Exception as e:
                        continue

        print(f"  Found {files_found} optimal_response.json files")

    df_random = pd.DataFrame(rows_random)
    df_ml = pd.DataFrame(rows_ml)
    df_cosine = pd.DataFrame(rows_cosine)

    print(f"\nSimilarities loaded:")
    print(f"  Random: {len(df_random)}")
    print(f"  ML Optimal: {len(df_ml)}")
    print(f"  Cosine Optimal: {len(df_cosine)}")

    return {
        'random': df_random,
        'ML_optimal': df_ml,
        'cosine_optimal': df_cosine
    }


# === Best Configuration Saving ===
def save_best_configurations(df, output_dir, response_type="random", metric="accuracy"):
    """
    Save detailed information about the best configuration for each model-dataset combination.
    """
    print(f"\n=== Saving Best Configurations for {response_type} ===")

    subset = df[df["response_type"] == response_type].copy()

    if subset.empty:
        print(f"No data available for {response_type}")
        return None

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    best_configs = []

    for (model, dataset), group in subset.groupby(['model', 'dataset']):
        best_idx = group[mean_col].idxmin()
        best_row = group.loc[best_idx]

        best_acc = best_row[mean_col]
        best_std = best_row[std_col] if pd.notna(best_row[std_col]) else 0

        sota_group = group[(group['persona'] == 1) &
                           (group['style'] == 0) &
                           (group['context'] == 0) &
                           (group['ft'] == 0)
                           ]

        if not sota_group.empty:
            sota_row = sota_group.iloc[0]
            sota_acc = sota_row[mean_col]
            sota_std = sota_row[std_col] if pd.notna(sota_row[std_col]) else 0
            improvement_over_sota = sota_acc - best_acc
        else:
            sota_acc = None
            sota_std = None
            improvement_over_sota = None

        best_configs.append({
            'model': model,
            'dataset': dataset,
            'best_accuracy': best_acc,
            'best_std': best_std,
            'best_config_short': best_row['short_label'],
            'best_config_full': best_row['label'],
            'has_persona': best_row['persona'],
            'has_style': best_row['style'],
            'has_context': best_row['context'],
            'has_finetuning': best_row['ft'],
            'sota_accuracy': sota_acc,
            'sota_std': sota_std,
            'improvement_over_sota': improvement_over_sota,
            'n_measurements': best_row['n_measurements']
        })

        print(f"  {model} | {dataset}: {best_row['short_label']} "
              f"(acc={best_acc:.4f}±{best_std:.4f})")

    df_best_configs = pd.DataFrame(best_configs)
    df_best_configs = df_best_configs.sort_values(['model', 'dataset'])

    output_path = os.path.join(output_dir, f"best_configurations_{response_type}_{metric}.csv")
    df_best_configs.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")

    return df_best_configs


# === Feature Importance Loading ===
def load_feature_importance_data(folder_paths, df_best_configs, response_type='random'):
    """
    Load feature importance data for optimal configurations only.
    Uses the best_configurations dataframe to filter.
    """
    print(f"\n=== Loading Feature Importance Data ===")

    all_data = []
    processed_files = []

    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            continue

        dataset_name = normalize_dataset_name(folder_path)
        print(f"\nProcessing {dataset_name}...")

        dataset_configs = df_best_configs[df_best_configs['dataset'] == dataset_name]

        root_path = Path(folder_path)

        for _, config_row in dataset_configs.iterrows():
            model = config_row['model']

            expected_config = {
                'ft': int(config_row['has_finetuning']),
                'context': int(config_row['has_context']),
                'style': int(config_row['has_style']),
                'persona': int(config_row['has_persona'])
            }

            for root, _, files in os.walk(root_path):
                for file in files:
                    if not file.endswith("feature_correlation_stats.csv"):
                        continue

                    if response_type == "random" and "random_validation" not in file:
                        continue

                    filepath = os.path.join(root, file)

                    try:
                        file_model, file_ft, file_context, file_style, file_persona = parse_filename(file)

                        if (file_model == model and
                            file_ft == expected_config['ft'] and
                            file_context == expected_config['context'] and
                            file_style == expected_config['style'] and
                            file_persona == expected_config['persona']):

                            df = pd.read_csv(filepath)

                            df['dataset'] = dataset_name
                            df['model'] = model
                            df['ft'] = file_ft
                            df['context'] = file_context
                            df['style'] = file_style
                            df['persona'] = file_persona
                            df['config_label'] = make_label(model, file_ft, file_context,
                                                            file_style, file_persona)

                            all_data.append(df)
                            processed_files.append((dataset_name, model, config_row['best_config_short']))
                            print(f"  Loaded: {model} - {config_row['best_config_short']}")

                    except Exception as e:
                        continue

    if not all_data:
        print("  No feature importance files found!")
        return None

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nLoaded {len(combined_df)} records from {len(all_data)} files")
    print(f"Processed {len(processed_files)} optimal configurations")

    return combined_df
